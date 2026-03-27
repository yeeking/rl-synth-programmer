from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .config import CurriculumConfig
from .host import ParameterSpec


@dataclass(slots=True)
class TargetSpec:
    target_id: str
    split: str
    parameters: dict[str, float]
    embedding: np.ndarray | None = None
    audio: np.ndarray | None = None
    label: str | None = None
    preset_state_path: str | None = None
    audio_path: str | None = None
    embedding_path: str | None = None
    state_hash: str | None = None


class TargetPool:
    def __init__(self, config: CurriculumConfig, parameter_specs: list[ParameterSpec]):
        self.config = config
        self.parameter_specs = parameter_specs
        self._rng = np.random.default_rng(config.seed)
        self._targets = self._load_targets()
        self._active_index = -1
        self._episodes_on_current = 0

    def _load_targets(self) -> list[TargetSpec]:
        if self.config.manifest_path is not None:
            return self._load_manifest(self.config.manifest_path)
        assert self.config.pool_size == config_total(self.config), (
            "Pool size must equal train + val + test sizes."
        )
        return self._build_targets()

    def _build_targets(self) -> list[TargetSpec]:
        targets: list[TargetSpec] = []
        splits = (
            ["train"] * self.config.train_size
            + ["val"] * self.config.val_size
            + ["test"] * self.config.test_size
        )
        for index, split in enumerate(splits):
            params = {spec.stable_id: float(self._rng.uniform(0.0, 1.0)) for spec in self.parameter_specs}
            targets.append(TargetSpec(target_id=f"{split}-{index:03d}", split=split, parameters=params))
        return targets

    def _load_manifest(self, manifest_path: Path) -> list[TargetSpec]:
        payload = json.loads(Path(manifest_path).read_text())
        targets: list[TargetSpec] = []
        records = payload["targets"]
        if self.config.subset_limit is not None:
            records = records[: self.config.subset_limit]
        for record in records:
            parameter_snapshot = record.get("parameter_snapshot", record.get("parameters", {}))
            targets.append(
                TargetSpec(
                    target_id=str(record["target_id"]),
                    split=str(record["split"]),
                    parameters={str(k): float(v) for k, v in parameter_snapshot.items()},
                    label=record.get("label"),
                    preset_state_path=record.get("preset_state_path"),
                    audio_path=record.get("audio_path"),
                    embedding_path=record.get("embedding_path"),
                    state_hash=record.get("state_hash"),
                )
            )
        assert targets, f"Manifest {manifest_path} did not contain any targets."
        return targets

    def targets_for_split(self, split: str) -> list[TargetSpec]:
        return [target for target in self._targets if target.split == split]

    def _next_train_index(self) -> int:
        train_targets = self.targets_for_split("train")
        assert train_targets, "Target pool contains no training targets."
        if self.config.switching_mode == "uniform_rotation":
            if self._active_index < 0:
                next_target = train_targets[0]
            else:
                current_target = self._targets[self._active_index]
                position = train_targets.index(current_target)
                next_target = train_targets[(position + 1) % len(train_targets)]
            return self._targets.index(next_target)
        raise ValueError(f"Unsupported switching mode: {self.config.switching_mode}")

    def current_target(self) -> TargetSpec | None:
        if self._active_index < 0:
            return None
        return self._targets[self._active_index]

    def activate_next_target(self) -> TargetSpec:
        self._active_index = self._next_train_index()
        self._episodes_on_current = 0
        return self._targets[self._active_index]

    def maybe_advance(self) -> TargetSpec:
        if self._active_index < 0:
            return self.activate_next_target()
        self._episodes_on_current += 1
        if self._episodes_on_current >= self.config.dwell_episodes:
            return self.activate_next_target()
        return self._targets[self._active_index]


def config_total(config: CurriculumConfig) -> int:
    return config.train_size + config.val_size + config.test_size
