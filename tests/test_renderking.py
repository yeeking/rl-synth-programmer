from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rl_synth_programmer._renderking") is None,
    reason="RenderKing native extension is not built",
)


def _plugins() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        root / "plugins" / "Dexed.vst3",
        root / "plugins" / "Ultramaster KR-106.vst3",
    ]


@pytest.mark.parametrize("plugin_path", _plugins())
def test_renderking_loads_plugin_and_renders_audio(plugin_path: Path) -> None:
    from rl_synth_programmer import _renderking

    host = _renderking.Host(str(plugin_path), sample_rate=44_100, render_duration=0.25)
    metadata = host.inspect_plugin()
    assert metadata["is_instrument"]
    parameters = host.list_parameters()
    assert parameters
    audio = np.asarray(host.render_note(), dtype=np.float32)
    assert audio.ndim == 1
    assert audio.size > 0
    assert np.all(np.isfinite(audio))


@pytest.mark.parametrize("plugin_path", _plugins())
def test_renderking_state_round_trip(plugin_path: Path) -> None:
    from rl_synth_programmer import _renderking

    host = _renderking.Host(str(plugin_path), sample_rate=44_100, render_duration=0.1)
    state = host.capture_preset_state()
    before = host.current_parameter_snapshot()
    host.restore_preset_state(state)
    after = host.current_parameter_snapshot()
    assert before.keys() == after.keys()
