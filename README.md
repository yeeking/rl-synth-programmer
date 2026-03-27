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
