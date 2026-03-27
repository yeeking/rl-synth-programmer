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
  --steps 20000

rl-synth evaluate \
  --plugin "/home/matthew/.vst3/Ultramaster KR-106.vst3" \
  --run-folder "artifacts/kr106_real" \
  --episodes 16
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

## TensorBoard

Training and evaluation can write TensorBoard logs. By default:

- `train-dqn` writes to `<run-folder>/train_dqn/tensorboard`
- `smoke-train-clap` and `smoke-evaluate` write to `<run-folder>/smoke_train_clap/tensorboard`
- `full-smoke` writes to `<run-folder>/tensorboard`

Example:

```bash
tensorboard --logdir artifacts/kr106_real/train_dqn/tensorboard
```
