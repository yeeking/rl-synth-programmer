# Developer Onboarding: RL Synth Programmer

This project trains a reinforcement-learning agent to program VST3 instrument presets. The agent edits normalized synth parameters, renders the plugin audio, embeds that audio, and receives reward when the rendered sound moves closer to a target preset sound.

## Repository Shape

- `src/rl_synth_programmer/config.py` contains dataclass configuration for the host, environment, reward, curriculum, DQN, and full experiment.
- `src/rl_synth_programmer/host.py` wraps `pedalboard` VST3 loading, parameter discovery, preset state capture/restore, parameter setting, and MIDI note rendering.
- `src/rl_synth_programmer/curriculum.py` defines `TargetSpec` and `TargetPool`, which provide target presets to training episodes.
- `src/rl_synth_programmer/env.py` implements the Gym-style single-environment rollout.
- `src/rl_synth_programmer/reward.py` wraps CLAP embeddings and distance-based reward.
- `src/rl_synth_programmer/agent.py` implements random policy, replay buffer, and DQN.
- `src/rl_synth_programmer/training.py` contains single-env and batched DQN training plus evaluation.
- `src/rl_synth_programmer/parallel_rollout.py` contains multiprocessing render workers and the batched rollout coordinator.
- `src/rl_synth_programmer/offline_dataset.py` generates all-actions offline reward datasets from target/start preset pairs.
- `src/rl_synth_programmer/networks.py` builds MLP, residual MLP, CNN, GRU, and LSTM action-value networks.
- `src/rl_synth_programmer/architecture_sweep.py` trains and ranks supervised action-value or action-conditioned reward predictors.
- `src/rl_synth_programmer/hyperparameter_search.py` discovers action datasets and runs the default CNN/RNN-heavy search across them.
- `src/rl_synth_programmer/smoke.py` contains end-to-end smoke workflows and artifact writers.
- `src/rl_synth_programmer/cli.py` is the `rl-synth` command-line entrypoint.
- `tests/` uses small fakes for the host, environment, render pool, and agent so most tests avoid real VST, Torch, or CLAP work.

Generated output normally lives under `artifacts/`. Local CLAP/GPT model files are cached under `clap-weights/`. These are runtime assets, not source architecture.

## Runtime Workflow

The usual workflow is:

1. Inspect a VST3 plugin with `rl-synth inspect-plugin`.
2. Generate a target set with `rl-synth generate-target-set`.
3. Generate an offline all-actions dataset with `rl-synth generate-action-dataset`.
4. Compare supervised architectures with `rl-synth compare-architectures` or `rl-synth search-feature-change-models`.
5. Train with `rl-synth train-dqn`.
6. Evaluate with `rl-synth evaluate`.

`generate-target-set` discovers program/preset states through `SynthHost.enumerate_program_states()`. Each captured target stores:

- a binary preset state in `targets/states/`
- rendered target audio in `targets/audio/`
- metadata in `targets/manifest.json` and `targets/manifest.csv`

Training loads that manifest through `TargetPool`, not by scanning the filesystem directly.

`generate-action-dataset` then uses the manifest to create `<run-folder>/action_dataset/dataset.npz`. It first probes a small global sample of start states to calibrate per-parameter action steps by embedding-distance sensitivity, unless `--no-action-step-calibration` is used. Each row stores the flattened observation and the immediate reward for every discrete action. It does not currently store per-action next embeddings; action-conditioned feature-change searches therefore use immediate reward/distance improvement as the available feature-change proxy.

## Core Data Model

### Parameters

`ParameterSpec` in `host.py` is the source of truth for synth parameters. It stores stable ID, display name, index, raw default, raw min/max, and flags for automatable/meta parameters.

The environment uses normalized parameter values in `[0, 1]`. `ParameterSpec.denormalize()` converts an action-updated normalized value back to the raw plugin range before assigning `plugin.parameters[stable_id].raw_value`.

Parameter filtering happens in `SynthHost.filter_parameters()`:

- allowlist/denylist are applied first
- non-automatable and meta parameters are skipped
- program/preset/bypass/MIDI-CC-like controls are skipped

This filtered list determines both action space size and the parameter snapshot saved in target manifests.

### Targets

`TargetSpec` represents one sound target. It can be synthetic random parameters or a manifest-backed preset:

- `parameters`: normalized parameter snapshot
- `embedding`: lazily populated target embedding
- `audio`: lazily populated target audio
- `preset_state_path`: optional binary preset state path
- `audio_path`: optional rendered target WAV path
- `split`: `train`, `val`, or `test`

`TargetPool.maybe_advance()` rotates over training targets with a configurable dwell count. Current switching is `uniform_rotation`.

## Environment Mechanics

`SynthProgrammingEnv` is the single-process Gym-style environment.

On `reset()`:

1. `TargetPool.maybe_advance()` selects the current training target.
2. The target embedding is computed if needed.
3. The initial state is sampled.
4. If manifest presets are available, the start state prefers another preset from the same split, falling back to any other preset.
5. Otherwise the start state is random normalized parameters.
6. If the reset starts too close to the target, a small deterministic parameter nudge avoids a zero-distance episode.

On `step(action)`:

1. The discrete action is decoded into `(parameter_id, signed_delta)`. Offline datasets may use calibrated per-parameter deltas; online RL uses the global action step unless given calibrated steps explicitly.
2. The selected normalized parameter is clipped into `[0, 1]`.
3. The host renders a MIDI note.
4. Audio is embedded.
5. Distance to target embedding is recomputed.
6. Reward is either random or `previous_distance - new_distance`.
7. The episode terminates on `success_threshold` and truncates at `max_episode_steps`.

The observation is:

```text
[target_embedding, current_embedding, target_embedding - current_embedding, current_normalized_params]
```

The observation size is only known after the first embedding is available, so `reset()` updates the Gym `observation_space` shape.

## Reward and Embeddings

`RewardConfig.mode` controls reward behavior:

- `random`: no embedder; reward is random and mainly useful for plumbing checks.
- `clap`: `CLAPEmbedder` embeds rendered audio through `msclap`.

`SimilarityRewardModel` supports:

- `cosine`: `1 - cosine_similarity`
- `l2`: Euclidean distance

Reward is based on improvement, not absolute similarity:

```text
reward = previous_distance - new_distance
```

So positive reward means the rendered sound moved closer to the target embedding.

The CLAP wrapper also supports local/offline model paths used by smoke tests through `clap-weights/CLAP_weights_2023.pth` and `clap-weights/gpt2/`.

## DQN Agent

`DQNAgent` in `agent.py` contains:

- MLP online network
- MLP target network
- Adam optimizer
- replay buffer
- epsilon-greedy action selection
- step-based epsilon decay
- periodic target-network sync

`ReplayTransition` stores observation, action, reward, next observation, done flag, and target ID. The training loop owns episode bookkeeping and TensorBoard logging; the agent only owns learning state.

## Training Paths

There are two training implementations.

### Single Environment

`train_dqn()` uses one `SynthProgrammingEnv`. It is easier to reason about and remains the default when `--num-workers 1`.

It performs:

1. environment reset
2. DQN action
3. environment step
4. replay insert
5. one `agent.train_step()`
6. scalar/text logging
7. episode reset when terminated/truncated

### Batched Parallel Rollout

`train_dqn_batched()` activates when `--num-workers > 1`.

It splits work into:

- `ParallelRenderPool`: multiprocessing pool where each process owns a loaded VST host
- `BatchedRolloutCoordinator`: pure coordination of target selection, action decoding, observations, distances, rewards, and slot state
- one shared CLAP embedder in the parent process
- one shared DQN learner in the parent process

The design keeps VST rendering parallel while avoiding one CLAP model per worker. Each tick:

1. active slots choose actions
2. render requests are built
3. workers render audio in parallel
4. parent batches CLAP embeddings
5. coordinator applies rewards and new observations
6. transitions enter replay
7. learner performs `updates_per_tick` optimizer updates
8. completed slots are reset

Target embeddings are precomputed at startup by `_prime_target_embeddings()` so each step only embeds current audio.

## Offline Architecture Search

There are two supervised search modes.

`compare-architectures` reads one `action_dataset/dataset.npz` and a JSON config. By default, it predicts the full per-action reward vector from each stored observation:

```text
observation -> action_rewards
```

If the config contains `"target": "action_reward_as_feature_change_proxy"`, it expands the dataset into action-conditioned examples:

```text
[observation, parameter_index_normalized, signed_delta] -> immediate_reward
```

This mode preserves grouped train/val/test splits by original source row to avoid leaking actions from the same state across splits. Set `"cv_folds": 2` or higher in the sweep config, or pass `--cv-folds` to `search-feature-change-models` when using the generated config, to run grouped cross-validation. Use `3` or higher when you want distinct train/validation/test fold roles. By default the generated hypersearch config excludes rows with failed action renders and leaves cross-validation disabled.

Architecture training is Lightning-backed. The repo still owns the search loop and artifact schema, while Lightning owns per-trial train/validation/test execution, deterministic seeding, progress, and TensorBoard logging. With `--tensorboard`, per-epoch metrics and TensorBoard HParams summaries are written below each architecture directory; point TensorBoard at the sweep root to compare trials.

`search-feature-change-models` wraps this for public experiments. It discovers all `*/action_dataset/dataset.npz` files under `artifacts/`, writes a generated CNN/RNN-heavy `search_config.json`, runs each dataset, and emits:

```text
combined_leaderboard.md
combined_leaderboard.csv
combined_leaderboard.json
```

The latest local initial run used:

```bash
.venv/bin/rl-synth search-feature-change-models \
  --artifacts-root artifacts \
  --out-dir artifacts/architecture_search/feature_change_action_conditioned_initial \
  --epochs 5 \
  --no-progress
```

Headline results from that run:

- `dexed_real`: `cnn-small-s7`, validation regret `0.159846`, validation MSE `0.0102375`
- `ultra_real`: `lstm-small-s7`, validation regret `0.07357`, validation MSE `0.000388206`

For a longer GPU-server rerun, increase `--epochs`, optionally remove or raise `max_expanded_rows` in the generated config, consider `--cv-folds 3`, and enable `--tensorboard`.

## CLI and Artifacts

The CLI always resolves user-facing run folders under `artifacts/`. Passing either `my_run` or `artifacts/my_run` resolves to `artifacts/my_run`.

Expected run layout:

```text
artifacts/<run>/
  targets/
    manifest.json
    manifest.csv
    states/
    audio/
  train_dqn/
    dqn_latest.pt
    tensorboard/
  smoke_random_env/
  smoke_train_clap/
  smoke_evaluate/
  action_dataset/
    dataset.npz
    metadata.json
    summary.json
    shards/
```

The CLI helper functions `_find_manifest()`, `_find_train_checkpoint()`, and `_find_smoke_checkpoint()` enforce the expected workflow and produce user-facing error messages.

## Tests

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run all tests:

```bash
pytest
```

Most tests patch heavy dependencies with fakes. The code paths that need real VST hosting, CLAP, Torch, and local model files are exercised by smoke commands rather than the default unit tests.

## Common Change Points

- To change action semantics, edit `_decode_action()` in `env.py` and `decode_action()` in `parallel_rollout.py`. Keep them behaviorally identical.
- To change observation contents, edit `_flatten_observation()` in `env.py` and `flatten_observation()` in `parallel_rollout.py`.
- To change reward math, edit `SimilarityRewardModel` or the reward branch in both environment/coordinator step application.
- To change CNN/RNN supervised model shapes, edit `networks.py` and update architecture validation in `architecture_sweep.py`.
- To change the default multi-dataset hyperparameter search, edit `default_feature_change_search_config()` in `hyperparameter_search.py`.
- To change target scheduling, edit `TargetPool`.
- To add a CLI option, update `_base_parser()`, `_cmd_*`, and `_experiment_config()` if it affects runtime config.
- To add new training metrics, update both `train_dqn()` and `train_dqn_batched()` when the metric applies to both paths.

## Practical Pitfalls

- The real plugin path must point to an existing `.vst3` instrument.
- Manifest-backed reset starts from another preset when possible. This is intentional; starting from full random vectors can be less useful for preset-to-preset programming.
- Target embedding computation may temporarily restore a preset state on the host, then restore the previous host state.
- The batched path uses `spawn` multiprocessing, so worker functions and request/result objects must stay pickle-friendly.
- Keep CLAP in the parent process for batched training unless there is a strong reason to pay for one model per render worker.
- The CLI validates common user-input mistakes before dispatch. Lower-level modules still use `assert` for internal invariants and defensive checks; if the package is ever run with Python optimization (`-O`), those checks disappear.
