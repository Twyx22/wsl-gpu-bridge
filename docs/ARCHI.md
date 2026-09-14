# Archi WSL GPU Bridge

```
nvidia-smi / /dev/dxg / uname -r -> detect() -> check_fp8(sm,cuda,dxg) -> recommend() -> run
                                        |               |                      |
                                   parse_smi()    fp8_usable()           fallback()
```

1. `detect` : parse `nvidia-smi`, `uname -r` (WSL2 ?), `/dev/dxg` (dxgkrnl ?).
2. `check-fp8` : `sm==120` + `CUDA>=12.8` + `dxg` présent si WSL2.
3. `recommend` : dtype effectif + tok/s simulés.

Limites : dxgkrnl n'expose pas toujours FP8 natif ni les perf counters, version CUDA WSL < native.
Contournement : ne pas croire le seul driver, tester alloc réelle via `torch.cuda` quand dispo, sinon fallback BF16/INT4.
