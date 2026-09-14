import unittest
from unittest.mock import patch
import io
import json
import os
import tempfile
from contextlib import redirect_stdout
import wsl_gpu_bridge as b


class T(unittest.TestCase):
    def test_parse_smi(self):
        self.assertEqual(b.parse_smi("NVIDIA GeForce RTX 5090, 32607 MiB, 572.83")["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(b.parse_smi(""), {})

    def test_fp8_matrix(self):
        self.assertTrue(b.fp8_usable(120, (12, 8), True, True)[0])
        self.assertFalse(b.fp8_usable(120, (12, 8), False, True)[0])
        self.assertFalse(b.fp8_usable(89, (12, 8), True, False)[0])
        self.assertEqual(b.fallback("fp8", False), "bf16")

    @patch("wsl_gpu_bridge.sh")
    def test_collect_mocked_smi(self, mock_sh):
        mock_sh.side_effect = ["NVIDIA GeForce RTX 5090, 32607 MiB, 572.83", "6.6.87-microsoft-standard-WSL2", "CUDA Version: 12.8"]
        with tempfile.TemporaryDirectory() as d:
            with patch.object(b, "CACHE_PATH", os.path.join(d, "c.json")):
                with patch("os.path.exists", return_value=True):
                    info = b.collect()
        self.assertTrue(info["wsl"] and info["dxg"] and info["fp8_ok"])
        self.assertEqual(info["sm"], 120)

    def test_bench_order(self):
        self.assertGreater(b.sim_toks(7, 16, "int4"), b.sim_toks(7, 16, "bf16"))

    def test_real_bench_fallback(self):
        r = b.real_bench()
        self.assertTrue(r is None or isinstance(r, dict))

    def test_arch_vendor(self):
        self.assertEqual(b.arch_of("AMD Instinct MI300X"), ("amd", "gfx942"))
        self.assertEqual(b.arch_of("Intel Data Center GPU Max 1550"), ("intel", "xpu"))
        self.assertEqual(b.arch_of("NVIDIA GeForce RTX 4090"), ("nvidia", "sm_89"))
        self.assertEqual(b.arch_of(""), ("cpu", ""))
        self.assertTrue(b.fp8_usable(0, (0, 0), False, False, "amd", "gfx942")[0])
        self.assertFalse(b.fp8_usable(0, (0, 0), False, False, "intel", "xpu")[0])

    @patch("wsl_gpu_bridge.sh", return_value="x not found")
    def test_detect_gpu_none(self, _mock):
        self.assertEqual(b.detect_gpu(), ("", ""))

    @patch("wsl_gpu_bridge.sh")
    def test_collect_cache(self, mock_sh):
        mock_sh.side_effect = ["NVIDIA GeForce RTX 5090, 32607 MiB, 572.83", "6.6.87-microsoft-standard-WSL2", "CUDA Version: 12.8"]
        with tempfile.TemporaryDirectory() as d:
            with patch.object(b, "CACHE_PATH", os.path.join(d, "c.json")):
                with patch("os.path.exists", return_value=True):
                    info1 = b.collect()
                    info2 = b.collect()
        self.assertEqual(info1, info2)
        self.assertEqual(mock_sh.call_count, 3)  # 2e appel servi par le cache

    def test_recommend_json(self):
        fake = {"gpu": "X", "mem_mb": 16384, "fp8_ok": True, "fp8_why": "ok"}
        with patch.object(b, "collect", return_value=fake):
            buf = io.StringIO()
            with redirect_stdout(buf):
                b.main(["recommend", "--dtype", "fp8", "--params", "14", "--json"])
        out = json.loads(buf.getvalue())
        self.assertEqual(out["dtype_effective"], "fp8")


if __name__ == "__main__":
    unittest.main()
