# WSL GPU Bridge

Détecte si FP8 Blackwell `sm_120` (RTX 5090) est réellement exposé sous WSL2 via `dxgkrnl`. Fallback auto si émulé.

Problème : `torch.cuda.get_device_capability()` retourne `(12,0)` mais FP8 tombe en émulé FP16 → 44 tok/s au lieu de ~140 (vLLM Qwen3-14B FP8).

## Quickstart

```bash
python3 wsl_gpu_bridge.py detect
python3 wsl_gpu_bridge.py check-fp8
python3 wsl_gpu_bridge.py bench --params 14 --dtype fp8
python3 wsl_gpu_bridge.py bench --params 14 --dtype fp8 --real  # mesure torch si CUDA dispo
python3 wsl_gpu_bridge.py recommend --dtype fp8 --params 14
python3 wsl_gpu_bridge.py recommend --dtype fp8 --params 14 --json
```

## GPUs

| GPU | SM | FP8 WSL2 | Fallback |
|---|---|---|---|
| RTX 5090 / Blackwell | sm_120 | oui si CUDA≥12.8 + /dev/dxg | BF16 |
| RTX 4090 / Ada | sm_89 | non (pas de FP8 natif) | BF16 |
| H100 / Hopper | sm_90 | oui si CUDA≥12.8 | BF16 |
| Autre / CPU | — | non | INT4 CPU (llama.cpp) |

Règle : `FP8 → BF16 → INT4`. Jamais de crash, toujours un mode runnable.

## Exemple

```json
// detect
{"gpu": "NVIDIA GeForce RTX 5090", "sm": 120, "cuda": "12.8", "fp8_ok": true, "wsl": true, "dxg": true}
```

## Roadmap
- [x] flag `--json` (`recommend --json`), cache detect (TTL 60s, `--no-cache` pour bypass)
- [x] bench réel torch (`bench --real` → `real:{matmul_ms,tflops}`, `real:null` + `real_why` sinon)
- [x] support ROCm/Intel (`detect_gpu()` multi-backend, `arch_of()` → vendor/arch, FP8 gfx942/gfx950 via ROCm)
