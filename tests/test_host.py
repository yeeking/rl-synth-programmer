from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl_synth_programmer.config import SynthHostConfig
from rl_synth_programmer.host import ParameterSpec, RenderKingSynthHost, SynthHost, make_synth_host


class HostFilterTests(unittest.TestCase):
    def test_filter_parameters_drops_reserved_and_meta(self):
        host = SynthHost(SynthHostConfig(plugin_path=Path("dummy.vst3")))
        host._parameter_specs = [
            ParameterSpec("cutoff", "Cutoff", 0, 0.5),
            ParameterSpec("program", "Program", 1, 0.0),
            ParameterSpec("bypass", "Bypass", 2, 0.0),
            ParameterSpec("macro", "Macro", 3, 0.5, automatable=False),
            ParameterSpec("meta", "Meta", 4, 0.2, is_meta=True),
        ]
        filtered = host.filter_parameters()
        self.assertEqual([spec.stable_id for spec in filtered], ["cutoff"])

    def test_normalized_defaults(self):
        host = SynthHost(SynthHostConfig(plugin_path=Path("dummy.vst3")))
        specs = [ParameterSpec("gain", "Gain", 0, 5.0, minimum=0.0, maximum=10.0)]
        defaults = host.get_normalized_defaults(specs)
        self.assertAlmostEqual(defaults["gain"], 0.5)

    def test_host_factory_defaults_to_pedalboard_backend(self):
        host = make_synth_host(SynthHostConfig(plugin_path=Path("dummy.vst3")))
        self.assertIsInstance(host, SynthHost)

    def test_host_factory_selects_renderking_backend(self):
        host = make_synth_host(SynthHostConfig(plugin_path=Path("dummy.vst3"), host_backend="renderking"))
        self.assertIsInstance(host, RenderKingSynthHost)

    def test_renderking_backend_reports_missing_native_module_helpfully(self):
        host = RenderKingSynthHost(SynthHostConfig(plugin_path=Path("dummy.vst3"), host_backend="renderking"))
        try:
            host._module()
        except RuntimeError as exc:
            self.assertIn("not importable", str(exc))
        else:
            self.skipTest("RenderKing native module is available in this environment.")


if __name__ == "__main__":
    unittest.main()
