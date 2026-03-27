from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from .config import SynthHostConfig
from .optional_deps import require_dependency


@dataclass(slots=True)
class ParameterSpec:
    stable_id: str
    display_name: str
    index: int
    default_value: float
    minimum: float = 0.0
    maximum: float = 1.0
    automatable: bool = True
    is_meta: bool = False

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.minimum, self.maximum))

    def normalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        if span <= 0:
            return 0.0
        return float((value - self.minimum) / span)

    def denormalize(self, value: float) -> float:
        return self.minimum + float(np.clip(value, 0.0, 1.0)) * (self.maximum - self.minimum)


class SynthHost:
    """Thin VST3 instrument host around pedalboard."""

    RESERVED_PARAMETER_TOKENS = ("bypass", "program", "preset", "midi cc")

    def __init__(self, config: SynthHostConfig):
        self.config = config
        self._plugin: Any | None = None
        self._parameter_specs: list[ParameterSpec] | None = None

    @property
    def plugin(self) -> Any:
        if self._plugin is None:
            self.load()
        assert self._plugin is not None
        return self._plugin

    def load(self) -> Any:
        pedalboard = require_dependency("pedalboard", "runtime")
        plugin = pedalboard.load_plugin(str(self.config.plugin_path))
        assert getattr(plugin, "is_instrument", False), (
            f"Plugin '{self.config.plugin_path}' is not an instrument plugin."
        )
        self._plugin = plugin
        self._parameter_specs = self._build_parameter_specs()
        return plugin

    def inspect_plugin(self) -> dict[str, Any]:
        plugin = self.plugin
        parameters = self.list_parameters()
        return {
            "path": str(self.config.plugin_path),
            "name": str(getattr(plugin, "name", self.config.plugin_path.stem)),
            "version": str(getattr(plugin, "version", "unknown")),
            "identifier": str(getattr(plugin, "identifier", "unknown")),
            "is_instrument": bool(getattr(plugin, "is_instrument", False)),
            "parameter_count": len(parameters),
            "program_controls": self.list_program_controls(),
            "supports_preset_data": hasattr(plugin, "preset_data"),
            "supports_raw_state": hasattr(plugin, "raw_state"),
        }

    def _build_parameter_specs(self) -> list[ParameterSpec]:
        plugin = self.plugin
        specs: list[ParameterSpec] = []
        for index, (python_name, parameter) in enumerate(plugin.parameters.items()):
            default_value = self._safe_float(getattr(parameter, "raw_value", getattr(parameter, "value", 0.0)), 0.0)
            minimum = self._safe_float(getattr(parameter, "min_value", 0.0), 0.0)
            maximum = self._safe_float(getattr(parameter, "max_value", 1.0), 1.0)
            automatable = bool(getattr(parameter, "is_automatable", True))
            is_meta = bool(getattr(parameter, "is_meta_parameter", False))
            if maximum < minimum:
                minimum, maximum = maximum, minimum
            if maximum == minimum:
                maximum = minimum + 1.0
            specs.append(
                ParameterSpec(
                    stable_id=python_name,
                    display_name=str(getattr(parameter, "name", python_name)),
                    index=index,
                    default_value=default_value,
                    minimum=minimum,
                    maximum=maximum,
                    automatable=automatable,
                    is_meta=is_meta,
                )
            )
        return specs

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def list_parameters(self) -> list[ParameterSpec]:
        if self._parameter_specs is None:
            self.load()
        assert self._parameter_specs is not None
        return list(self._parameter_specs)

    def list_program_controls(self) -> dict[str, Any] | None:
        for spec in self.list_parameters():
            lowered = f"{spec.stable_id} {spec.display_name}".lower()
            if "program" in lowered or "preset" in lowered:
                parameter = self.plugin.parameters[spec.stable_id]
                valid_values = getattr(parameter, "valid_values", None)
                return {
                    "stable_id": spec.stable_id,
                    "display_name": spec.display_name,
                    "minimum": spec.minimum,
                    "maximum": spec.maximum,
                    "default_value": spec.default_value,
                    "valid_values": list(valid_values) if valid_values else None,
                }
        return None

    def filter_parameters(
        self,
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
    ) -> list[ParameterSpec]:
        allowset = set(allowlist or [])
        denyset = set(denylist or [])
        filtered: list[ParameterSpec] = []
        for spec in self.list_parameters():
            stable_lower = spec.stable_id.lower()
            display_lower = spec.display_name.lower()
            if allowset and spec.stable_id not in allowset:
                continue
            if spec.stable_id in denyset:
                continue
            if not spec.automatable or spec.is_meta:
                continue
            if any(token in stable_lower or token in display_lower for token in self.RESERVED_PARAMETER_TOKENS):
                continue
            filtered.append(spec)
        assert filtered, "Parameter filtering produced an empty parameter set."
        return filtered

    def get_normalized_defaults(self, parameter_specs: list[ParameterSpec]) -> dict[str, float]:
        return {spec.stable_id: spec.normalize(spec.default_value) for spec in parameter_specs}

    def set_parameters(self, normalized_values: dict[str, float], parameter_specs: list[ParameterSpec] | None = None) -> None:
        specs = parameter_specs or self.list_parameters()
        spec_by_id = {spec.stable_id: spec for spec in specs}
        plugin = self.plugin
        for stable_id, normalized_value in normalized_values.items():
            assert stable_id in spec_by_id, f"Unknown parameter '{stable_id}'."
            spec = spec_by_id[stable_id]
            raw_value = spec.denormalize(normalized_value)
            plugin.parameters[stable_id].raw_value = raw_value

    def capture_preset_state(self) -> bytes:
        plugin = self.plugin
        if hasattr(plugin, "preset_data"):
            return bytes(plugin.preset_data)
        if hasattr(plugin, "raw_state"):
            return bytes(plugin.raw_state)
        raise RuntimeError("Plugin does not expose preset_data or raw_state.")

    def restore_preset_state(self, state: bytes) -> None:
        plugin = self.plugin
        if hasattr(plugin, "preset_data"):
            plugin.preset_data = state
            return
        if hasattr(plugin, "raw_state"):
            plugin.raw_state = state
            return
        raise RuntimeError("Plugin does not expose preset_data or raw_state.")

    def select_program(self, index: int | str) -> None:
        control = self.list_program_controls()
        assert control is not None, "Plugin does not expose a program-like control."
        spec = next(spec for spec in self.list_parameters() if spec.stable_id == control["stable_id"])
        valid_values = control.get("valid_values")
        if valid_values:
            if isinstance(index, int):
                assert 0 <= index < len(valid_values), f"Program index {index} out of range."
                setattr(self.plugin, spec.stable_id, valid_values[index])
                return
            setattr(self.plugin, spec.stable_id, index)
            return
        raw_index = float(np.clip(index, int(spec.minimum), int(spec.maximum)))
        setattr(self.plugin, spec.stable_id, raw_index)

    def current_parameter_snapshot(self, parameter_specs: list[ParameterSpec] | None = None) -> dict[str, float]:
        specs = parameter_specs or self.list_parameters()
        snapshot: dict[str, float] = {}
        for spec in specs:
            current_raw = self._safe_float(getattr(self.plugin.parameters[spec.stable_id], "raw_value", None), spec.default_value)
            snapshot[spec.stable_id] = spec.normalize(current_raw)
        return snapshot

    def enumerate_program_states(self, max_programs: int | None = None) -> list[dict[str, Any]]:
        control = self.list_program_controls()
        if control is None:
            state = self.capture_preset_state()
            return [
                {
                    "target_id": f"state-{sha1(state).hexdigest()[:12]}",
                    "label": "default_state",
                    "program_index": None,
                    "state_bytes": state,
                    "state_hash": sha1(state).hexdigest(),
                }
            ]
        spec = next(spec for spec in self.list_parameters() if spec.stable_id == control["stable_id"])
        valid_values = control.get("valid_values")
        states: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        probe_values = valid_values if valid_values else self._program_probe_values(spec)
        for ordinal, raw_value in enumerate(probe_values):
            if valid_values:
                self.select_program(ordinal)
            else:
                setattr(self.plugin, spec.stable_id, raw_value)
            state = self.capture_preset_state()
            state_hash = sha1(state).hexdigest()
            if state_hash in seen_hashes:
                continue
            seen_hashes.add(state_hash)
            states.append(
                {
                    "target_id": f"program-{ordinal:03d}-{state_hash[:8]}",
                    "label": str(raw_value) if valid_values else f"program_{ordinal:03d}",
                    "program_index": ordinal,
                    "program_raw_value": raw_value,
                    "state_bytes": state,
                    "state_hash": state_hash,
                }
            )
            if max_programs is not None and len(states) >= max_programs:
                break
        return states

    def _program_probe_values(self, spec: ParameterSpec) -> list[float]:
        span = spec.maximum - spec.minimum
        if span <= 1.01:
            return np.linspace(spec.minimum, spec.maximum, num=128, dtype=np.float32).tolist()
        count = int(round(span)) + 1
        return [float(value) for value in np.linspace(spec.minimum, spec.maximum, num=count)]

    def render_note(
        self,
        parameter_values: dict[str, float] | None = None,
        note: int | None = None,
        duration: float | None = None,
        velocity: int | None = None,
    ) -> np.ndarray:
        midi_mod = require_dependency("mido", "runtime")
        note_value = self.config.note if note is None else note
        duration_value = self.config.render_duration if duration is None else duration
        velocity_value = self.config.velocity if velocity is None else velocity
        assert duration_value > 0, "Render duration must be positive."
        if parameter_values:
            self.set_parameters(parameter_values)
        midi_messages = [
            midi_mod.Message("note_on", note=note_value, velocity=velocity_value, time=0),
            midi_mod.Message("note_off", note=note_value, velocity=0, time=duration_value),
        ]
        total_duration = duration_value + self.config.tail_duration + self.config.warmup_duration
        rendered = self.plugin(
            midi_messages,
            duration=total_duration,
            sample_rate=self.config.sample_rate,
            reset=True,
        )
        audio = np.asarray(rendered, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        assert audio.ndim == 1, f"Expected mono audio, got shape {audio.shape}."
        return audio

    def ensure_plugin_path(self) -> Path:
        path = Path(self.config.plugin_path)
        assert path.exists(), f"Plugin path does not exist: {path}"
        assert path.suffix.lower() == ".vst3", f"Expected a .vst3 plugin path, got: {path}"
        return path
