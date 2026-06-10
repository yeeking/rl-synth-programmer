# RL Synth Programmer

Tools for programming VST3 instrument presets with offline reward datasets, supervised action-value training, and a DQN reinforcement-learning agent.

The project is built around a preset-to-preset workflow:

1. Capture a synth's built-in presets as target sounds.
2. Generate an offline all-actions dataset from target/start preset pairs.
3. Train and compare supervised action-value models on that dataset.
4. Train or evaluate the online DQN agent against the same target manifest.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[runtime,ml,dev]
```

You need a local VST3 instrument for real runs:

```bash
export SYNTH_PLUGIN="/path/to/Instrument.vst3"
export RUN_FOLDER="artifacts/my_synth_run"
```

CLAP weights and text-model files are cached under `clap-weights/`. The code checks that local cache first and downloads missing assets from Hugging Face when CLAP is used. For offline-only smoke runs, make sure the cache already contains `CLAP_weights_2023.pth` and the required text model directory, such as `clap-weights/gpt2/`.

## Main Workflow

Generate a target manifest from the synth's built-in presets:

```bash
rl-synth generate-target-set \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --subset-limit 32
```

This writes:

```text
artifacts/my_synth_run/
  targets/
    manifest.json
    manifest.csv
    states/
    audio/
```

Generate a reusable offline action dataset. Each row stores the current observation plus the immediate CLAP reward for every available parameter-tweak action:

```bash
rl-synth generate-action-dataset \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --max-states 1024 \
  --moves-per-start 8 \
  --num-workers 4 \
  --clap-batch-size 8 \
  --render-timeout-seconds 300 \
  --max-state-seconds 90 \
  --shard-size 16 \
  --yes
```

Preview cost before writing `dataset.npz`:

```bash
rl-synth generate-action-dataset \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --estimate-only
```

Compare supervised action-value architectures:

```bash
rl-synth compare-architectures \
  --dataset "$RUN_FOLDER/action_dataset/dataset.npz" \
  --config "$RUN_FOLDER/sweep.json" \
  --out-dir "$RUN_FOLDER/architecture_sweep"
```

Run the default CNN/RNN-focused action-conditioned search across every discovered action dataset:

```bash
rl-synth search-feature-change-models \
  --artifacts-root artifacts \
  --out-dir artifacts/architecture_search/feature_change \
  --epochs 5
```

Train the online DQN agent:

```bash
rl-synth train-dqn \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --reward-mode clap \
  --steps 20000 \
  --epsilon-decay-steps 50000 \
  --max-episode-steps 48
```

Evaluate the latest DQN checkpoint:

```bash
rl-synth evaluate \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --episodes 16
```

## Command Reference

```bash
rl-synth inspect-plugin --plugin "$SYNTH_PLUGIN" --run-folder artifacts/inspect
rl-synth render --plugin "$SYNTH_PLUGIN" --note 60 --duration 1.0
rl-synth generate-target-set --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --subset-limit 12
rl-synth generate-action-dataset --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --max-states 256
rl-synth compare-architectures --dataset "$RUN_FOLDER/action_dataset/dataset.npz" --config "$RUN_FOLDER/sweep.json" --out-dir "$RUN_FOLDER/architecture_sweep"
rl-synth random-agent --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --episodes 4
rl-synth train-dqn --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --reward-mode clap --steps 2000
rl-synth evaluate --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --episodes 8
rl-synth search-feature-change-models --artifacts-root artifacts --out-dir artifacts/architecture_search/feature_change --epochs 5
rl-synth full-smoke --plugin "$SYNTH_PLUGIN" --run-folder artifacts/full_smoke
```

`--run-folder` is the artifact root. Passing `my_run` or `artifacts/my_run` both resolve to `artifacts/my_run`; absolute paths are converted to a folder under `artifacts/` using the final path name.

## Offline Dataset Details

`generate-action-dataset` samples target/start preset pairs round-robin. It renders move 0 for each pair first, then move 1 for each pair, up to `--moves-per-start` or `--max-states`. For every sampled state, it evaluates every discrete action and stores:

- `observations`: flattened `[target_embedding, current_embedding, delta_embedding, params]`
- `action_rewards`: immediate reward for each action
- `current_distances`, target/start indices, move indices, best actions, and render diagnostics

Long runs write recoverable shards under `<run-folder>/action_dataset/shards/` before merging the final `<run-folder>/action_dataset/dataset.npz`. Metadata and summary files are written next to the dataset.

Useful reliability options:

- `--render-timeout-seconds`: timeout for one render chunk
- `--skip-failed-actions` / `--no-skip-failed-actions`: continue with a large negative reward for timed-out actions or abort
- `--max-state-seconds`: skip pathological slow states
- `--reload-workers-every-renders`: periodically reload plugin worker processes
- `--preset-render-slowdown-threshold`: detect sudden per-action render slowdowns
- `--reload-workers-on-render-slowdown` / `--no-reload-workers-on-render-slowdown`: reload workers or assert on slowdown

## Architecture Sweep

Example `sweep.json`:

```json
{
  "split": {"train": 0.8, "val": 0.1, "test": 0.1},
  "architectures": [
    {
      "name": "mlp-512-256",
      "type": "mlp",
      "hidden_sizes": [512, 256],
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 20,
      "seed": 7
    },
    {
      "name": "cnn-small",
      "type": "cnn1d",
      "channels": [32, 64],
      "kernel_sizes": [5, 3],
      "embedding_hidden_size": 128,
      "param_hidden_sizes": [64],
      "head_hidden_sizes": [128],
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 20,
      "seed": 8
    }
  ]
}
```

Supported architecture types are `mlp`, `residual_mlp`, `cnn1d`, `hybrid_cnn_mlp`, `rnn`, `gru`, and `lstm`. The sweep writes per-architecture checkpoints and metrics plus `leaderboard.json` and `leaderboard.csv`.

`search-feature-change-models` generates a compact CNN/RNN-heavy config and runs it across all discovered `*/action_dataset/dataset.npz` files. It uses the stored immediate reward as a feature-change proxy because the current dataset format stores all-action rewards but not per-action next embeddings. The action-conditioned input is:

```text
[target_embedding, current_embedding, target-current delta, params, parameter_index_normalized, signed_delta]
```

The combined results are written to:

```text
<out-dir>/combined_leaderboard.md
<out-dir>/combined_leaderboard.csv
<out-dir>/combined_leaderboard.json
```

For a longer GPU-server rerun, increase `--epochs`, remove or raise `max_expanded_rows` in the generated `search_config.json`, and optionally enable `--tensorboard`.

### Latest Initial Search

The latest local action-conditioned search was run with:

```bash
.venv/bin/rl-synth search-feature-change-models \
  --artifacts-root artifacts \
  --out-dir artifacts/architecture_search/feature_change_action_conditioned_initial \
  --epochs 5 \
  --no-progress
```

It discovered:

- `artifacts/dexed_real/action_dataset/dataset.npz`
- `artifacts/ultra_real/action_dataset/dataset.npz`

The combined result table is in:

```text
artifacts/architecture_search/feature_change_action_conditioned_initial/combined_leaderboard.md
artifacts/architecture_search/feature_change_action_conditioned_initial/combined_leaderboard.csv
artifacts/architecture_search/feature_change_action_conditioned_initial/combined_leaderboard.json
```

Initial best-by-dataset results:

| dataset | best model | type | val regret | val MSE | test regret | test MSE |
| --- | --- | --- | --- | --- | --- | --- |
| `dexed_real` | `cnn-small-s7` | `cnn1d` | `0.159846` | `0.0102375` | `0.123465` | `0.0104374` |
| `ultra_real` | `lstm-small-s7` | `lstm` | `0.07357` | `0.000388206` | `0.0684737` | `0.000479126` |

Interpret these as short CPU sanity results, not final model selection. The current artifact dataset format stores per-action rewards, not per-action next embeddings, so `search-feature-change-models` predicts immediate reward/distance improvement as the available feature-change proxy.

## DQN Agent Path

`train-dqn` can run either a classic single environment or batched parallel rollout:

```bash
rl-synth train-dqn \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder "$RUN_FOLDER" \
  --reward-mode clap \
  --steps 2000 \
  --num-workers 4 \
  --updates-per-tick 1 \
  --clap-batch-size 8
```

The batched path activates automatically when `--num-workers > 1`. It uses multiple synth-render worker processes and one shared CLAP embedder in the parent process.

Current training behavior:

- Manifest-backed episodes start from another preset in the target set when possible.
- Rewards are improvement based: `previous_distance - new_distance`.
- `--epsilon-decay-steps` controls step-based epsilon decay.
- `--max-episode-steps` controls episode truncation.

## Smoke Checks

The full real-plugin smoke workflow exercises plugin inspection, target generation, CLAP embedding, random rollout, DQN training, and evaluation:

```bash
rl-synth full-smoke \
  --plugin "$SYNTH_PLUGIN" \
  --run-folder artifacts/full_smoke \
  --subset-limit 12 \
  --random-episodes 6 \
  --train-steps 128 \
  --eval-episodes 4
```

Smaller smoke phases are also available:

```bash
rl-synth smoke-random-env --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --episodes 4
rl-synth smoke-train-clap --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --steps 128
rl-synth smoke-evaluate --plugin "$SYNTH_PLUGIN" --run-folder "$RUN_FOLDER" --episodes 4
```

## TensorBoard

Training and evaluation can write TensorBoard logs. By default:

- `train-dqn` writes to `<run-folder>/train_dqn/tensorboard`
- `smoke-train-clap` and `smoke-evaluate` write to `<run-folder>/smoke_train_clap/tensorboard`
- `full-smoke` writes to `<run-folder>/tensorboard`

Example:

```bash
tensorboard --logdir "$RUN_FOLDER/train_dqn/tensorboard"
```

## Development

Run the fast test suite:

```bash
pytest
```

Most tests use fakes and avoid real VST, Torch, and CLAP execution. Use the smoke commands for end-to-end validation with a real plugin.

Generated artifacts, model caches, Python bytecode, build metadata, and local virtual environments should stay out of version control. The source tree should focus on `src/`, `tests/`, project metadata, and docs.
