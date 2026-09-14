import unittest
from unittest.mock import patch
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
        with patch("os.path.exists", return_value=True):
            info = b.collect()
        self.assertTrue(info["wsl"] and info["dxg"] and info["fp8_ok"])
        self.assertEqual(info["sm"], 120)

    def test_bench_order(self):
        self.assertGreater(b.sim_toks(7, 16, "int4"), b.sim_toks(7, 16, "bf16"))


if __name__ == "__main__":
    unittest.main()
