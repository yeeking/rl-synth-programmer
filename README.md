# RL Synth Programmer

VST3 synth hosting, preset-derived target generation, Gym environment wrapping, and DQN training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[runtime,ml]
```

## Example Commands

Full lightweight smoke run:

```bash
rl-synth full-smoke \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_smoke_test" \
  --subset-limit 12 \
  --random-episodes 8 \
  --train-steps 1000 \
  --eval-episodes 8
```

Basic training workflow:

```bash
rl-synth generate-target-set \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --subset-limit 12

rl-synth generate-target-set \
  --plugin "/Users/matthewyk/Library/Audio/Plug-Ins/VST3/Dexed.vst3" \
  --run-folder "artifacts/Dexed_real" \
  --subset-limit 12

rl-synth train-dqn \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --reward-mode clap \
  --steps 20000 \
  --epsilon-decay-steps 50000 \
  --max-episode-steps 48

rl-synth evaluate \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --episodes 16
```

Current training behavior:

- manifest-backed episodes start from another preset in the target set when possible, rather than from a full-random parameter vector
- `--epsilon-decay-steps` controls epsilon decay over action steps
- `--num-workers > 1` enables batched parallel rollout

Verified 4-worker batched smoke:

```bash
rl-synth train-dqn \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/runfolder_smoke" \
  --reward-mode clap \
  --steps 8 \
  --num-workers 4 \
  --clap-batch-size 4
```

## Main Commands

```bash
rl-synth inspect-plugin --plugin /path/to/synth.vst3 --run-folder "artifacts/inspect"
rl-synth generate-target-set --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run" --subset-limit 12
rl-synth random-agent --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run"
rl-synth train-dqn --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run" --reward-mode clap --steps 2000
rl-synth evaluate --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run" --episodes 8
rl-synth smoke-random-env --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run"
rl-synth smoke-train-clap --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run" --steps 128
rl-synth smoke-evaluate --plugin /path/to/synth.vst3 --run-folder "artifacts/my_run" --episodes 4
rl-synth full-smoke --plugin /path/to/synth.vst3 --run-folder "artifacts/full_smoke"
```

`--run-folder` is the user-facing artifact root. The CLI creates it if needed for write commands and auto-discovers internal files like manifests and checkpoints beneath it.

Internal layout under a run folder is:

- `targets/` for generated preset targets and `manifest.json`
- `train_dqn/` for the main training checkpoint and TensorBoard logs
- `smoke_*` folders for smoke-run outputs

Console progress bars and stage logs are enabled by default for target generation, training, and evaluation. Use `--no-progress` to reduce live terminal output.

## Offline Action Dataset and Architecture Sweep

Generate a reusable supervised dataset where each row stores the current observation and the immediate reward for every available action:

```bash
rl-synth generate-action-dataset \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --max-states 256 \
  --moves-per-start 4 \
  --num-workers 4 \
  --clap-batch-size 8 \
  --render-timeout-seconds 300 \
  --max-state-seconds 90 \
  --shard-size 16 \
  --reload-workers-every-pair \
  --yes

rl-synth generate-action-dataset \
  --plugin "/Users/matthewyk/Library/Audio/Plug-Ins/VST3/Dexed.vst3" \
  --run-folder "artifacts/Dexed_real" \
  --max-states 256 \
  --moves-per-start 4 \
  --num-workers 12 \
  --clap-batch-size 8 \
  --render-timeout-seconds 300 \
  --max-state-seconds 90 \
  --shard-size 16 \
  --reload-workers-every-pair \
  --yes
```

Long dataset runs write recoverable shards under `<run-folder>/action_dataset/shards/` before the final merged `dataset.npz`.
If a render chunk times out, `--skip-failed-actions` is enabled by default and assigns those actions a large negative reward so the run can continue.
Use `--no-skip-failed-actions` to abort instead.
Use `--max-state-seconds` to skip pathological slow start presets and continue with the next target/start pair.
Render workers are reloaded after each target/start pair by default; use `--no-reload-workers-every-pair` to keep plugin instances alive for speed.

Preview estimated renders, runtime, and dataset size without writing `dataset.npz`:

```bash
rl-synth generate-action-dataset \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --estimate-only
```

```bash
rl-synth generate-action-dataset \
  --plugin "/Users/matthewyk/Library/Audio/Plug-Ins/VST3/Dexed.vst3" \
  --run-folder "artifacts/Dexed_real" \
  --estimate-only
```

Compare network architectures against the generated dataset:

```bash
rl-synth compare-architectures \
  --dataset "artifacts/kr106_real/action_dataset/dataset.npz" \
  --config "artifacts/kr106_real/sweep.json" \
  --out-dir "artifacts/kr106_real/architecture_sweep"
```

Example sweep config:

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

## Parallel Batched Training

`train-dqn` can run multiple synth-render workers in parallel while batching CLAP embeddings through one shared model instance. The classic single-env path remains the default.

Example:

```bash
rl-synth train-dqn \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_parallel" \
  --reward-mode clap \
  --steps 2000 \
  --num-workers 4 \
  --updates-per-tick 1 \
  --clap-batch-size 8
```

The batched path activates automatically when `--num-workers > 1`.

Useful parallel options:

- `--num-workers`: number of synth-render worker processes and active episode slots
- `--updates-per-tick`: learner updates after each rollout batch
- `--clap-batch-size`: number of audio buffers embedded together by CLAP; if omitted it defaults to `--num-workers`

Useful exploration option:

- `--epsilon-decay-steps`: number of action steps over which epsilon decays; the current scheduler is step-based, not episode-based
- `--max-episode-steps`: maximum number of actions per episode before truncation

Recent KR-106 throughput check after switching reset starts from random parameters to other presets:

- `--num-workers 1`: about `1.31` steps/s
- `--num-workers 4`: about `1.82` steps/s
- `--num-workers 8`: about `1.90` steps/s

## TensorBoard

Training and evaluation can write TensorBoard logs. By default:

- `train-dqn` writes to `<run-folder>/train_dqn/tensorboard`
- `smoke-train-clap` and `smoke-evaluate` write to `<run-folder>/smoke_train_clap/tensorboard`
- `full-smoke` writes to `<run-folder>/tensorboard`

Example:

```bash
tensorboard --logdir artifacts/kr106_real/train_dqn/tensorboard
```

## Example commands from myk

Generate a large dataset on Dexed

```
## this generates the target sounds from presets which are used as 'from->to'
## positions in training
## subset-limit dictates how many presets we render, which for Dexed
## is the 32 that are in the bank 
rl-synth generate-target-set \
  --plugin "/Users/matthewyk/Library/Audio/Plug-Ins/VST3/Dexed.vst3" \
  --run-folder "artifacts/Dexed_large" \
  --subset-limit 32 

## now we have our target dataset, we render out 
## a pre-training dataset which contains 
## pre-computed rewards for parameter tweaks (actions) made
## as we move around in the latent space 
rl-synth generate-action-dataset \
  --plugin "/Users/matthewyk/Library/Audio/Plug-Ins/VST3/Dexed.vst3" \
  --run-folder "artifacts/Dexed_large" \
  --max-states 1024 \
  --moves-per-start 128 \
  --num-workers 12 \
  --clap-batch-size 8 \
  --render-timeout-seconds 120 \
  --shard-size 8 \
  --yes

```
