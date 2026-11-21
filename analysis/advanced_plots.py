#!/usr/bin/env python3
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "plots" / "advanced"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for p in glob.glob(str(RAW / "*.jsonl")):
    for line in open(p):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("timestamp") is None:
                continue
            if r.get("type") == "summary" and r.get("max_tokens") != 64:
                continue
            if "Failed to establish a new connection" in (r.get("error") or ""):
                continue
            rows.append(r)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no data")

def short_model(name):
    if not isinstance(name, str):
        return str(name)
    low = name.lower()
    if "gemma" in low or "12b" in low: return "Gemma12B"
    if "0.5b" in low: return "Qwen0.5B"
    if "3b" in low: return "Qwen3B"
    if "7b" in low: return "Qwen7B"
    if "8b" in low: return "Llama8B"
    return name.split("/")[-1][:18]

df["model_s"] = df["model"].map(short_model) if "model" in df else ""
df["label"] = df.get("engine", "") + " " + df["model_s"]

# 1. TTFT boxplot by concurrency
conc = df[df.get("type") == "request"] if "type" in df else pd.DataFrame()
if not conc.empty and "ttft_s" in conc:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=conc, x="concurrency", y="ttft_s", hue="label")
    plt.title("TTFT distribution by concurrency")
    plt.tight_layout(); plt.savefig(OUT/"10_ttft_boxplot_concurrency.png"); plt.close()

# 2. TPOT boxplot by context length
ctx = df[df.get("target_context_tokens").notna()] if "target_context_tokens" in df else pd.DataFrame()
if not ctx.empty and "tpot_s" in ctx:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=ctx, x="target_context_tokens", y="tpot_s", hue="label")
    plt.title("TPOT distribution by context length")
    plt.tight_layout(); plt.savefig(OUT/"11_tpot_boxplot_context.png"); plt.close()

summ = df[df.get("type") == "summary"] if "type" in df else pd.DataFrame()

# 3. Heatmap peak VRAM (engine x concurrency)
if not summ.empty:
    pivot = summ.groupby(["label","concurrency"])["peak_vram_mb"].mean().unstack()
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Peak VRAM heatmap (MB)")
    plt.tight_layout(); plt.savefig(OUT/"12_vram_heatmap_concurrency.png"); plt.close()

# 4. Scatter throughput vs p95 latency
if not summ.empty:
    plt.figure(figsize=(8,5))
    for lab, sub in summ.groupby("label"):
        plt.scatter(sub["aggregate_tokens_per_sec"], sub["latency_p95_s"], label=lab, alpha=0.7)
    plt.xlabel("Aggregate tokens/sec"); plt.ylabel("p95 latency (s)")
    plt.title("Throughput vs p95 latency")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT/"13_throughput_vs_p95.png"); plt.close()

# 5. Error rate by concurrency
if not summ.empty:
    summ2 = summ.copy()
    summ2["error_rate"] = summ2["n_failed"] / summ2["n_requests"]
    g = summ2.groupby(["label","concurrency"])["error_rate"].mean().reset_index()
    plt.figure(figsize=(8,5))
    for lab, sub in g.groupby("label"):
        plt.plot(sub["concurrency"], sub["error_rate"], marker="o", label=lab)
    plt.xlabel("Concurrency"); plt.ylabel("Error rate")
    plt.title("Error rate vs concurrency")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT/"14_error_rate_concurrency.png"); plt.close()

# 6. Per-request tokens/sec distribution
if not conc.empty and "tokens_per_sec" in conc:
    plt.figure(figsize=(8,5))
    sns.histplot(conc, x="tokens_per_sec", hue="label", kde=True, element="step")
    plt.title("Per-request tokens/sec distribution")
    plt.tight_layout(); plt.savefig(OUT/"15_tokens_per_sec_hist.png"); plt.close()

# 7. System RAM vs VRAM
if not summ.empty:
    plt.figure(figsize=(8,5))
    plt.scatter(summ["peak_vram_mb"], summ["ram_peak_mb"], c=summ["concurrency"], cmap="plasma", alpha=0.6)
    plt.xlabel("Peak VRAM (MB)"); plt.ylabel("Peak system RAM (MB)")
    plt.title("System RAM vs VRAM")
    plt.colorbar(label="concurrency")
    plt.tight_layout(); plt.savefig(OUT/"16_ram_vs_vram.png"); plt.close()

# 8. TTFT vs prompt tokens
if not ctx.empty:
    plt.figure(figsize=(8,5))
    for lab, sub in ctx.groupby("label"):
        plt.scatter(sub["actual_prompt_tokens"], sub["ttft_s"], label=lab, alpha=0.6)
    plt.xlabel("Prompt tokens"); plt.ylabel("TTFT (s)")
    plt.title("TTFT vs prompt tokens")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT/"17_ttft_vs_prompt_tokens.png"); plt.close()

# 9. Concurrency scaling efficiency
if not summ.empty:
    eff = summ.copy()
    eff["efficiency"] = eff["aggregate_tokens_per_sec"] / eff["concurrency"]
    g = eff.groupby(["label","concurrency"])["efficiency"].mean().reset_index()
    plt.figure(figsize=(8,5))
    for lab, sub in g.groupby("label"):
        plt.plot(sub["concurrency"], sub["efficiency"], marker="o", label=lab)
    plt.xlabel("Concurrency"); plt.ylabel("Tokens/sec per concurrent request")
    plt.title("Concurrency scaling efficiency")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT/"18_scaling_efficiency.png"); plt.close()

# 10. VRAM vs context heatmap
if not ctx.empty:
    pivot = ctx.groupby(["label","target_context_tokens"])["peak_vram_mb"].mean().unstack()
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="magma")
    plt.title("Peak VRAM heatmap by context (MB)")
    plt.tight_layout(); plt.savefig(OUT/"19_vram_heatmap_context.png"); plt.close()

# 11. Latency CDF
if not conc.empty and "total_s" in conc:
    plt.figure(figsize=(8,5))
    for lab, sub in conc.groupby("label"):
        vals = np.sort(sub["total_s"])
        cdf = np.arange(1, len(vals)+1)/len(vals)
        plt.plot(vals, cdf, label=lab)
    plt.xlabel("End-to-end latency (s)"); plt.ylabel("CDF")
    plt.title("Latency CDF")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT/"20_latency_cdf.png"); plt.close()

print("Advanced plots written to", OUT)
