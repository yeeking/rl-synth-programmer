from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rl_synth_programmer.config import SynthHostConfig
from rl_synth_programmer.host import make_synth_host


def _run(plugin: Path, backend: str, renders: int) -> dict[str, float | str]:
    host = make_synth_host(SynthHostConfig(plugin_path=plugin, host_backend=backend))
    host.load()
    specs = host.filter_parameters()
    params = host.current_parameter_snapshot(specs)
    started = time.perf_counter()
    total_samples = 0
    for index in range(renders):
        if specs:
            spec = specs[index % len(specs)]
            params[spec.stable_id] = float(np.clip(params[spec.stable_id] + 0.01, 0.0, 1.0))
        audio = host.render_note(params)
        total_samples += int(audio.shape[0])
    elapsed = time.perf_counter() - started
    return {
        "backend": backend,
        "plugin": str(plugin),
        "renders": renders,
        "seconds": elapsed,
        "renders_per_second": renders / max(elapsed, 1e-9),
        "total_samples": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pedalboard and RenderKing render throughput.")
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--renders", type=int, default=32)
    args = parser.parse_args()
    for backend in ("pedalboard", "renderking"):
        print(_run(args.plugin, backend, args.renders))


if __name__ == "__main__":
    main()
