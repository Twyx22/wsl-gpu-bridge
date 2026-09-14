#!/usr/bin/env python3
"""WSL GPU Bridge MVP (stdlib only): detect GPU WSL2, check FP8 sm_120 via dxgkrnl, bench+reco."""
import argparse
import json
import os
import re
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def parse_smi(s):
    if not s or "not found" in s.lower() or "failed" in s.lower():
        return {}
    p = [x.strip() for x in s.splitlines()[0].split(",")]
    m = re.search(r"(\d+)", p[1] if len(p) > 1 else "")
    return {"gpu": p[0], "mem_mb": int(m.group(1)) if m else 0, "driver": p[2] if len(p) > 2 else ""}


def is_wsl(kernel):
    return "microsoft-standard-wsl2" in (kernel or "").lower()


def cuda_ver(s):
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def sm_from_name(name):
    n = (name or "").upper()
    if re.search(r"50[6789]0|5060|B200|GB200|BLACKWELL", n):
        return 120
    if re.search(r"H100|H200|HOPPER", n):
        return 90
    if re.search(r"40[6789]0|ADA", n):
        return 89
    return 0


def torch_state():
    try:
        import torch
        return {"avail": bool(torch.cuda.is_available()), "devs": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    except Exception:
        return {"avail": None, "devs": 0}


def fp8_usable(sm, cuda, dxg, wsl):
    # ponytail: FP8 reel = sm_120 + CUDA>=12.8 + dxgkrnl expose sous WSL2
    if sm != 120:
        return (False, f"sm_{sm} != sm_120: pas de FP8 Blackwell")
    if cuda < (12, 8):
        return (False, f"CUDA {cuda[0]}.{cuda[1]} < 12.8 requis pour sm_120")
    if wsl and not dxg:
        return (False, "WSL2 sans /dev/dxg: dxgkrnl n'expose pas le GPU")
    return (True, "FP8 sm_120 expose via dxgkrnl")


def fallback(req, ok):
    if ok:
        return req.lower()
    return {"fp8": "bf16", "fp16": "bf16", "int8": "int4"}.get(req.lower(), "bf16")


def sim_toks(params_b=7.0, mem_gb=16.0, dtype="bf16"):
    f = {"fp8": 1.0, "bf16": 0.62, "fp16": 0.62, "int4": 1.7}.get(dtype.lower(), 0.62)
    return round(2000.0 / max(params_b, 0.1) * f * min(mem_gb, 80) / 16, 1)


def collect(smi_txt=None, kernel=None, dxg=None, cuda_txt=None):
    if smi_txt is None:
        smi_txt = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if kernel is None:
        kernel = sh("uname -r")
    if dxg is None:
        dxg = os.path.exists("/dev/dxg")
    if cuda_txt is None:
        cuda_txt = sh("nvidia-smi")
    wsl = is_wsl(kernel)
    g = parse_smi(smi_txt)
    sm = sm_from_name(g.get("gpu", ""))
    cuda = cuda_ver(cuda_txt)
    ok, why = fp8_usable(sm, cuda, dxg, wsl)
    t = torch_state()
    return {"wsl": wsl, "kernel": kernel.strip(), "dxg": dxg, "cuda": f"{cuda[0]}.{cuda[1]}",
            "sm": sm, "fp8_ok": ok, "fp8_why": why, "torch": t, **g}


def recommend(info, dtype="fp8", params_b=7.0):
    if not info.get("gpu"):
        return "CPU seul: petit modele GGUF int4, Ollama/llama.cpp."
    eff = fallback(dtype, info["fp8_ok"])
    tps = sim_toks(params_b, info.get("mem_mb", 0) / 1024, eff)
    if info["fp8_ok"]:
        return f"Utilise {eff.upper()} ({tps} tok/s simules, {params_b}B). vLLM/TensorRT-LLM ok."
    if info.get("mem_mb", 0) < 12000:
        return f"Fallback {eff.upper()}->INT4 recommande ({tps} tok/s simules). Raison: {info['fp8_why']}"
    return f"Fallback auto {dtype.upper()}->{eff.upper()} ({tps} tok/s simules). Raison: {info['fp8_why']}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsl_gpu_bridge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    sub.add_parser("check-fp8")
    b = sub.add_parser("bench")
    b.add_argument("--params", type=float, default=7.0)
    b.add_argument("--dtype", default="bf16")
    r = sub.add_parser("recommend")
    r.add_argument("--dtype", default="fp8")
    r.add_argument("--params", type=float, default=7.0)
    a = ap.parse_args(argv)
    info = collect()
    if a.cmd == "detect":
        print(json.dumps(info, indent=2))
    elif a.cmd == "check-fp8":
        print(json.dumps({"fp8_ok": info["fp8_ok"], "why": info["fp8_why"], "fallback": fallback("fp8", info["fp8_ok"])}))
    elif a.cmd == "bench":
        mem = info.get("mem_mb", 16384) / 1024
        print(json.dumps({"dtype": a.dtype, "tok_s": sim_toks(a.params, mem, a.dtype)}))
    elif a.cmd == "recommend":
        print(recommend(info, a.dtype, a.params))


def _selfcheck():
    assert parse_smi("NVIDIA GeForce RTX 5090, 32607 MiB, 572.83")["mem_mb"] == 32607
    assert parse_smi("") == {}
    assert is_wsl("6.6.87-microsoft-standard-WSL2") and not is_wsl("6.8-generic")
    assert sm_from_name("NVIDIA GeForce RTX 5090") == 120 and sm_from_name("RTX 4090") == 89
    assert fp8_usable(120, (12, 8), True, True)[0] and not fp8_usable(120, (12, 8), False, True)[0]
    assert not fp8_usable(89, (12, 8), True, True)[0] and not fp8_usable(120, (12, 4), True, False)[0]
    assert fallback("fp8", False) == "bf16" and fallback("fp8", True) == "fp8"
    assert sim_toks(7, 16, "int4") > sim_toks(7, 16, "bf16") > 0
    assert "CPU" in recommend({})


if __name__ == "__main__":
    _selfcheck()
    print("self-check ok", file=sys.stderr)
    main()
