rl-synth generate-action-dataset \
  --plugin "./plugins/Ultramaster KR-106.vst3" \
  --run-folder ./artefacts/larger-data/ultra \
  --rows-to-generate 1048576 \
  --moves-per-cycle 20 \
  --num-workers 20 \
  --clap-batch-size 20 \
  --clap-device auto \
  --action-step-calibration \
  --render-timeout-seconds 300 \
  --max-state-seconds 90 \
  --shard-size 16 \
  --confirm-large-run


rl-synth generate-action-dataset \
  --plugin "./plugins/Dexed.vst3" \
  --run-folder ./artefacts/larger-data/dexed \
  --rows-to-generate 1048576 \
  --moves-per-cycle 20 \
  --num-workers 20 \
  --clap-batch-size 20 \
  --clap-device auto \
  --action-step-calibration \
  --render-timeout-seconds 300 \
  --max-state-seconds 90 \
  --shard-size 16 \
  --confirm-large-run


SEARCH_DIR=artifacts/artefacts/larger-data/search
SWEEP_CONFIG="$SEARCH_DIR/full_sweep_config.json"
mkdir -p "$SEARCH_DIR"

${PYTHON:-python3} - "$SWEEP_CONFIG" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])

common = {
    "epochs": 12,
    "num_workers": 20,
    "pin_memory": True,
    "accelerator": "auto",
    "devices": "auto",
}

mlp_options = [
    ("medium", [512, 256], 0.001, 0.00001, 512, 7, 0.05),
    ("large", [1024, 512, 256], 0.0007, 0.00001, 512, 7, 0.08),
    ("xl", [2048, 1024, 512], 0.0003, 0.00003, 1024, 11, 0.10),
]

residual_options = [
    ("512x4", 512, 4, 0.0007, 0.00003, 512, 7, 0.05),
    ("1024x6", 1024, 6, 0.0003, 0.00003, 1024, 11, 0.08),
]

cnn_options = [
    ("small", [32, 64], [7, 5], 128, [64], [128], 0.001, 0.00001, 512, 7, 0.05),
    ("medium", [64, 96, 128], [9, 7, 5], 256, [128], [256, 128], 0.0007, 0.00001, 512, 7, 0.08),
    ("wide", [96, 128, 192], [11, 7, 5], 384, [192, 96], [384, 192], 0.0003, 0.00003, 1024, 11, 0.10),
]

hybrid_options = [
    ("medium", [64, 96], [9, 5], 256, [128], [256, 128], 0.0007, 0.00001, 512, 7, 0.08),
    ("wide", [96, 128, 192], [11, 7, 5], 384, [192, 96], [384, 192], 0.0003, 0.00003, 1024, 11, 0.10),
]

recurrent_options = [
    ("128x1", 128, 1, False, [64], [128], 0.001, 0.00001, 512, 7, 0.05),
    ("256x2", 256, 2, False, [128], [256, 128], 0.0007, 0.00003, 512, 11, 0.08),
    ("bidir-192x1", 192, 1, True, [128], [256, 128], 0.0005, 0.00003, 512, 13, 0.08),
]

architectures = []

for size, hidden_sizes, lr, weight_decay, batch_size, seed, dropout in mlp_options:
    architectures.append(
        {
            **common,
            "name": f"mlp-{size}-lr{lr:g}-s{seed}",
            "type": "mlp",
            "hidden_sizes": hidden_sizes,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "dropout": dropout,
        }
    )

for size, width, blocks, lr, weight_decay, batch_size, seed, dropout in residual_options:
    architectures.append(
        {
            **common,
            "name": f"resmlp-{size}-s{seed}",
            "type": "residual_mlp",
            "width": width,
            "blocks": blocks,
            "layer_norm": True,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "dropout": dropout,
        }
    )

for size, channels, kernels, embedding_size, param_hidden, head_hidden, lr, weight_decay, batch_size, seed, dropout in cnn_options:
    architectures.append(
        {
            **common,
            "name": f"cnn-{size}-s{seed}",
            "type": "cnn1d",
            "channels": channels,
            "kernel_sizes": kernels,
            "embedding_hidden_size": embedding_size,
            "param_hidden_sizes": param_hidden,
            "head_hidden_sizes": head_hidden,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "dropout": dropout,
        }
    )

for size, channels, kernels, embedding_size, param_hidden, fusion_hidden, lr, weight_decay, batch_size, seed, dropout in hybrid_options:
    architectures.append(
        {
            **common,
            "name": f"hybrid-cnn-mlp-{size}-s{seed}",
            "type": "hybrid_cnn_mlp",
            "channels": channels,
            "kernel_sizes": kernels,
            "fusion_embedding_size": embedding_size,
            "param_hidden_sizes": param_hidden,
            "fusion_hidden_sizes": fusion_hidden,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "dropout": dropout,
        }
    )

for cell in ("gru", "lstm"):
    for size, hidden_size, layers, bidirectional, param_hidden, head_hidden, lr, weight_decay, batch_size, seed, dropout in recurrent_options:
        architectures.append(
            {
                **common,
                "name": f"{cell}-{size}-s{seed}",
                "type": cell,
                "hidden_size": hidden_size,
                "layers": layers,
                "bidirectional": bidirectional,
                "param_hidden_sizes": param_hidden,
                "head_hidden_sizes": head_hidden,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "seed": seed,
                "dropout": dropout,
            }
        )

config = {
    "seed": 7,
    "cv_folds": 1,
    "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    "target": "action_reward_as_feature_change_proxy",
    "exclude_failed_rows": True,
    "max_expanded_rows": 1000000,
    "architectures": architectures,
}

config_path.write_text(json.dumps(config, indent=2) + "\n")
print(f"Wrote {config_path} with {len(architectures)} architecture specs.")
PY

rl-synth search-feature-change-models \
  --dataset artifacts/artefacts/larger-data/ultra/action_dataset/dataset.npz \
  --dataset artifacts/artefacts/larger-data/dexed/action_dataset/dataset.npz \
  --config "$SWEEP_CONFIG" \
  --out-dir "$SEARCH_DIR/full_hparam_sweep" \
  --tensorboard \
  --no-progress
