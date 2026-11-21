#!/usr/bin/env python3
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

def short_model(name):
    if not isinstance(name, str):
        return str(name)
    low = name.lower()
    if "gemma" in low or "12b" in low:
        return "Gemma4-12B"
    if "qwen2.5-0.5b" in low or "0.5b" in low:
        return "Qwen2.5-0.5B"
    if "qwen2.5-3b" in low or "3b" in low:
        return "Qwen2.5-3B"
    if "qwen2.5-7b" in low or "7b" in low:
        return "Qwen2.5-7B"
    if "llama-3.1-8b" in low or "llama3.1:8b" in low or "8b" in low:
        return "Llama3.1-8B"
    return name.split("/")[-1][:24]

QUANT_FALLBACK = {
    ("vllm", "Qwen2.5-0.5B"): "FP16",
    ("vllm", "Qwen2.5-3B"): "FP16",
    ("llama.cpp", "Qwen2.5-7B"): "Q4_K_M",
    ("llama.cpp", "Gemma4-12B"): "Q4_K_M",
    ("ollama", "Qwen2.5-0.5B"): "Q4_K_M",
    ("ollama", "Qwen2.5-7B"): "Q4_K_M",
    ("ollama", "Llama3.1-8B"): "Q4_K_M",
    ("ollama", "Gemma4-12B"): "Q4_K_M",
}

def keep_record(r):
    if r.get("timestamp") is None:
        return False
    if r.get("type") == "summary" and r.get("max_tokens") != 64:
        return False
    if "Failed to establish a new connection" in (r.get("error") or ""):
        return False
    return True

PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

cfg = json.loads((ROOT / "config" / "sustainability.json").read_text())
max_p95 = cfg["max_p95_latency_s"]

rows = []
for path in glob.glob(str(RAW / "*.jsonl")):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keep_record(r):
                rows.append(r)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No data found in results/raw")

# context data
ctx = df[df.get("target_context_tokens").notna()] if "target_context_tokens" in df else pd.DataFrame()
# concurrency summaries
conc = df[(df.get("type") == "summary")] if "type" in df else pd.DataFrame()
if not conc.empty:
    conc = conc.copy()
    conc["quantization"] = [
        q if pd.notna(q) else QUANT_FALLBACK.get((e, short_model(m)), "unknown")
        for q, e, m in zip(conc["quantization"], conc["engine"], conc["model"])
    ]

# 1. VRAM vs context length
if not ctx.empty:
    g = ctx.groupby(["engine", "model", "target_context_tokens"])["peak_vram_mb"].mean().reset_index()
    plt.figure()
    for (eng, model), sub in g.groupby(["engine", "model"]):
        plt.plot(sub["target_context_tokens"], sub["peak_vram_mb"], marker="o", label=f"{eng} {short_model(model)}")
    plt.xlabel("Context tokens")
    plt.ylabel("Peak VRAM (MB)")
    plt.title("VRAM vs context length")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "01_vram_vs_context.png")
    plt.close()

# 2. VRAM vs concurrent requests
if not conc.empty:
    g = conc.groupby(["engine", "model", "concurrency"])["peak_vram_mb"].mean().reset_index()
    plt.figure()
    for (eng, model), sub in g.groupby(["engine", "model"]):
        plt.plot(sub["concurrency"], sub["peak_vram_mb"], marker="o", label=f"{eng} {short_model(model)}")
    plt.xlabel("Concurrency")
    plt.ylabel("Peak VRAM (MB)")
    plt.title("VRAM vs concurrent requests")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "02_vram_vs_concurrency.png")
    plt.close()

# 3. Aggregate throughput vs concurrency
if not conc.empty:
    g = conc.groupby(["engine", "model", "concurrency"])["aggregate_tokens_per_sec"].mean().reset_index()
    plt.figure()
    for (eng, model), sub in g.groupby(["engine", "model"]):
        plt.plot(sub["concurrency"], sub["aggregate_tokens_per_sec"], marker="o", label=f"{eng} {short_model(model)}")
    plt.xlabel("Concurrency")
    plt.ylabel("Aggregate tokens/sec")
    plt.title("Throughput vs concurrency")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "03_throughput_vs_concurrency.png")
    plt.close()

# 4. p95 latency vs concurrency
if not conc.empty and "latency_p95_s" in conc:
    g = conc.groupby(["engine", "model", "concurrency"])["latency_p95_s"].mean().reset_index()
    plt.figure()
    for (eng, model), sub in g.groupby(["engine", "model"]):
        plt.plot(sub["concurrency"], sub["latency_p95_s"], marker="o", label=f"{eng} {short_model(model)}")
    plt.xlabel("Concurrency")
    plt.ylabel("p95 latency (s)")
    plt.title("p95 latency vs concurrency")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "04_p95_vs_concurrency.png")
    plt.close()

# 5. Max sustainable concurrency
if not conc.empty:
    def sustainable(row):
        if cfg.get("require_zero_failures") and row.get("n_failed", 0) > 0:
            return False
        p95 = row.get("latency_p95_s")
        if p95 is None:
            return False
        return p95 <= max_p95
    conc = conc.copy()
    conc["sustainable"] = conc.apply(sustainable, axis=1)
    maxconc = conc[conc["sustainable"]].groupby(["engine", "model"])["concurrency"].max().reset_index()
    plt.figure()
    labels = [f"{r['engine']}\n{short_model(r['model'])}" for _, r in maxconc.iterrows()]
    plt.bar(labels, maxconc["concurrency"])
    plt.ylabel("Max sustainable concurrency")
    plt.title("Max sustainable concurrency by config")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_max_sustainable_concurrency.png")
    plt.close()
    maxconc.to_csv(PLOTS / "max_sustainable_concurrency.csv", index=False)

# 6. Throughput vs quantization (use max aggregate throughput per config)
if not conc.empty:
    best = conc.groupby(["engine", "model", "quantization"])["aggregate_tokens_per_sec"].max().reset_index() if "quantization" in conc else None
    if best is not None and not best.empty:
        plt.figure()
        labels = [f"{r['engine']}\n{short_model(r['model'])}\n{r['quantization']}" for _, r in best.iterrows()]
        plt.bar(labels, best["aggregate_tokens_per_sec"])
        plt.ylabel("Max aggregate tokens/sec")
        plt.title("Throughput vs quantization")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        plt.savefig(PLOTS / "06_throughput_vs_quantization.png")
        plt.close()

print("Plots written to", PLOTS)
print("Rows:", len(df))
