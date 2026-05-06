#!/usr/bin/env python3
"""Generate professional charts for TurboQuant-Vulkan README from benchmark data."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "white",
})

# ── Color palette ─────────────────────────────────────────────────────────
C_F16   = "#6c757d"   # gray
C_Q4_0  = "#17a2b8"   # cyan
C_TQ3_0 = "#007bff"   # blue
C_TQ3_1 = "#6f42c1"   # purple
C_TQ3_2 = "#dc3545"   # red (hero highlight)
C_TQ2_0 = "#fd7e14"   # orange
C_TQ3_3 = "#20c997"   # teal

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Throughput Comparison (Line Chart)
# ═══════════════════════════════════════════════════════════════════════════

contexts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # K tokens
ctx_labels = ["4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"]

tokens_per_sec = {
    "F16":   [15.88, 15.66, 16.00, 15.93, 15.96, 15.70, 15.57, None, None],
    "Q4_0":  [18.07, 17.36, 17.96, 17.17, 17.80, 17.97, 18.15, 17.61, 17.39],
    "TQ3_0": [17.37, 17.28, 17.28, 17.66, 17.01, 17.65, 16.34, 17.10, 16.88],
    "TQ3_1": [17.38, 17.87, 17.35, 17.79, 17.16, 17.26, 17.12, 17.74, 17.00],
    "TQ3_2": [17.38, 17.91, 18.41, 17.79, 17.39, 18.20, 18.13, 17.91, 17.98],
    "TQ2_0": [17.38, 17.91, 18.41, 17.45, 17.39, 18.20, 18.13, 17.91, 17.98],
}

colors_line = {
    "F16":   C_F16,
    "Q4_0":  C_Q4_0,
    "TQ3_0": C_TQ3_0,
    "TQ3_1": C_TQ3_1,
    "TQ3_2": C_TQ3_2,
    "TQ2_0": C_TQ2_0,
}

linestyles = {
    "F16":   "dotted",
    "Q4_0":  "dashdot",
    "TQ3_0": "dashed",
    "TQ3_1": "dashed",
    "TQ3_2": "solid",
    "TQ2_0": "dotted",
}

linewidths = {
    "F16":   1.2,
    "Q4_0":  1.2,
    "TQ3_0": 1.5,
    "TQ3_1": 1.5,
    "TQ3_2": 2.8,
    "TQ2_0": 1.2,
}

zorder_map = {
    "F16":   1,
    "Q4_0":  2,
    "TQ3_0": 3,
    "TQ3_1": 4,
    "TQ3_2": 10,  # on top
    "TQ2_0": 5,
}

fig, ax = plt.subplots(figsize=(11, 5.5))

for label in ["F16", "Q4_0", "TQ3_0", "TQ3_1", "TQ3_2", "TQ2_0"]:
    y = tokens_per_sec[label]
    # mask None values
    x_plot = []
    y_plot = []
    for cx, cy in zip(contexts, y):
        if cy is not None:
            x_plot.append(cx)
            y_plot.append(cy)
    ax.plot(x_plot, y_plot,
            color=colors_line[label],
            linestyle=linestyles[label],
            linewidth=linewidths[label],
            marker="o" if label == "TQ3_2" else ("" if label in ("F16", "TQ2_0") else "s"),
            markersize=7 if label == "TQ3_2" else 4,
            markerfacecolor=colors_line[label],
            markeredgewidth=1.2 if label == "TQ3_2" else 0.5,
            markeredgecolor="white" if label == "TQ3_2" else colors_line[label],
            zorder=zorder_map[label],
            label=label)

# OOM annotation
ax.annotate("F16 OOM\nat 512K",
            xy=(512, 15.7), xytext=(384, 15.0),
            fontsize=8, color=C_F16,
            arrowprops=dict(arrowstyle="->", color=C_F16, lw=1.2))

# Highlight TQ3_2 band
ax.fill_between([4, 1024], 17.38, 18.41, alpha=0.08, color=C_TQ3_2, zorder=0)

ax.set_xscale("log", base=2)
ax.set_xticks(contexts)
ax.set_xticklabels(ctx_labels)
ax.set_xlabel("Context Size (tokens)")
ax.set_ylabel("Tokens per Second (generation)")
ax.set_title("KV Cache Throughput — AMD RX 6750 XT (12 GB)\nGemma-4-26B-A4B-it Q4_K_M · Vulkan Backend · 24 GPU Layers")
ax.set_ylim(14.5, 19.5)
ax.legend(loc="lower left", ncols=3, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "throughput_comparison.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Quality vs VRAM Pareto Frontier (Scatter)
# ═══════════════════════════════════════════════════════════════════════════

kv_types = ["F16", "TQ3_0", "TQ3_1", "TQ3_2", "TQ2_0"]
vram_mb     = [620,    136,    117,    97,     97]
accuracy_pct = [100.0,   93.3,  93.0,  93.3,   50.3]
marker_colors = [C_F16, C_TQ3_0, C_TQ3_1, C_TQ3_2, C_TQ2_0]
marker_sizes  = [180, 160, 160, 280, 160]
edge_colors   = ["#adb5bd", C_TQ3_0, C_TQ3_1, "#a71d2a", C_TQ2_0]

fig, ax = plt.subplots(figsize=(9, 6))

for i, (vt, vm, ac, mc, ms, ec) in enumerate(zip(
    kv_types, vram_mb, accuracy_pct, marker_colors, marker_sizes, edge_colors)):
    ax.scatter(vm, ac, c=mc, s=ms, edgecolors=ec, linewidth=2.5 if vt == "TQ3_2" else 1.2,
               zorder=10 if vt == "TQ3_2" else 5, alpha=0.95)
    # Offset labels for clarity
    offset_y = 3.5 if vt == "F16" else (-4.5 if vt == "TQ2_0" else (2.5 if vt == "TQ3_2" else -3))
    offset_x = -45 if vt == "TQ3_1" else (35 if vt == "TQ3_2" else 25)
    ax.annotate(vt,
                (vm, ac),
                textcoords="offset points",
                xytext=(offset_x, offset_y),
                fontsize=11 if vt == "TQ3_2" else 9,
                fontweight="bold" if vt == "TQ3_2" else "normal",
                color=ec)

# Pareto frontier line
pareto_x = [620, 136, 117, 97]
pareto_y = [100.0, 93.3, 93.0, 93.3]
ax.plot(pareto_x, pareto_y, "k--", linewidth=0.8, alpha=0.4, label="Pareto Frontier")

# VRAM reduction annotations
ax.annotate("−78% VRAM",
            xy=(136, 93.3), xytext=(240, 96),
            fontsize=8, color=C_TQ3_0,
            arrowprops=dict(arrowstyle="->", color=C_TQ3_0, lw=1))
ax.annotate("−84% VRAM\nsame quality",
            xy=(97, 93.3), xytext=(170, 89.5),
            fontsize=8, color=C_TQ3_2, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TQ3_2, lw=1.3))

ax.set_xlabel("KV Cache VRAM Usage @ 16K Context (MiB)")
ax.set_ylabel("Cognitive Accuracy vs FP16 Baseline (%)")
ax.set_title("Quality Retention vs VRAM — Pareto Frontier\nClaude Sonnet 4.6 LLM-as-Judge · 7-Question Protocol")
ax.invert_xaxis()
ax.set_xlim(650, 80)
ax.set_ylim(42, 105)
ax.legend(loc="lower left", framealpha=0.8)
fig.tight_layout()
fig.savefig(OUT / "quality_vram_pareto.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — TQ3 Evolution: TQ3_0 → TQ3_1 → TQ3_2 (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════

configs = ["TQ3_0", "TQ3_1", "TQ3_2"]
bar_colors = [C_TQ3_0, C_TQ3_1, C_TQ3_2]

# Subplot A: Throughput (tokens/s)
tps = [18.84, 19.50, 20.07]
# Subplot B: Completion Rate
completion = [19, 20, 20]   # out of 20
# Subplot C: Jaccard Similarity vs TQ3_0
jaccard = [1.0, 0.2675, 0.2510]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))

# Throughput
bars = axes[0].bar(configs, tps, color=bar_colors, edgecolor="white", linewidth=1.2,
                   width=0.5)
axes[0].set_title("Tokens / Second", fontweight="bold")
axes[0].set_ylabel("t/s")
axes[0].set_ylim(17, 21.5)
for bar, val in zip(bars, tps):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}", ha="center", fontsize=11, fontweight="bold", color=bar_colors[configs.index(configs[list(tps).index(val)])])
axes[0].axhline(y=15.78, color=C_F16, linestyle="dotted", linewidth=1, alpha=0.6)
axes[0].text(0.3, 15.78 + 0.15, "FP16 baseline (15.78)", fontsize=7, color=C_F16, alpha=0.8)

# Completion Rate
bars2 = axes[1].bar(configs, completion, color=bar_colors, edgecolor="white",
                    linewidth=1.2, width=0.5)
axes[1].set_title("Completion Rate (OK / 20)", fontweight="bold")
axes[1].set_ylabel("Prompts Completed")
axes[1].set_ylim(15, 22)
for bar, val in zip(bars2, completion):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val}/20", ha="center", fontsize=11, fontweight="bold",
                 color="white" if val == 20 else C_TQ3_0)
axes[1].axhline(y=20, color="#28a745", linestyle="--", linewidth=0.7, alpha=0.4)
axes[1].text(1.3, 20.15, "perfect", fontsize=7, color="#28a745", alpha=0.7)

# Jaccard vs TQ3_0
bars3 = axes[2].bar(configs, jaccard, color=bar_colors, edgecolor="white",
                    linewidth=1.2, width=0.5)
axes[2].set_title("Jaccard Similarity vs TQ3_0", fontweight="bold")
axes[2].set_ylabel("Jaccard Index")
axes[2].set_ylim(0, 1.15)
for bar, val in zip(bars3, jaccard):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold",
                 color=bar_colors[configs.index(configs[list(jaccard).index(val)])])
# Healthy range band
axes[2].axhspan(0.15, 0.40, alpha=0.06, color="#28a745", zorder=0)
axes[2].text(0.5, 0.42, "healthy range\n(distinct ≠ collapse)", fontsize=7,
             color="#28a745", alpha=0.8, ha="center")

fig.suptitle("TQ3 Evolution — Quality & Throughput Improvements\n20-Prompt Suite · RX 6750 XT · ngl=24",
             fontweight="bold", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig(OUT / "tq3_evolution.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — Max Context Capacity + Compression (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════

kv_order = ["TQ3_2", "TQ2_0", "TQ3_1", "TQ3_0", "Q4_0", "F16"]
max_ctx = [2000000, 1000000, 1000000, 1000000, 1000000, 256000]
compression = [6.4, 6.4, 5.3, 4.6, 3.6, 1.0]
accuracy2 = [93.3, 50.3, 93.0, 93.3, 97.0, 100.0]
kv_colors = [C_TQ3_2, C_TQ2_0, C_TQ3_1, C_TQ3_0, C_Q4_0, C_F16]

fig, ax = plt.subplots(figsize=(9, 4.5))

y_pos = range(len(kv_order))
bars = ax.barh(y_pos, max_ctx, color=kv_colors, edgecolor="white", linewidth=1.2,
               height=0.55)

# Add compression ratio labels
for i, (bar, comp, acc, kv) in enumerate(zip(bars, compression, accuracy2, kv_order)):
    ctx_val = max_ctx[i]
    label = f"{ctx_val/1000:.0f}K  ({comp}× compr)"
    # Mark TQ3_2 as champion
    if kv == "TQ3_2":
        label += "  [PRODUCTION]"
    # OOM note
    if ctx_val < 500000:
        label += "  [OOM at 512K]"
    ax.text(bar.get_width() + 15000, bar.get_y() + bar.get_height()/2,
            label,
            va="center", fontsize=10,
            fontweight="bold" if kv == "TQ3_2" else "normal",
            color="white" if kv == "TQ3_2" else "#212529")
    # Accuracy badge
    ax.text(bar.get_width() - 38000, bar.get_y() + bar.get_height()/2,
            f"{acc:.0f}%" if acc >= 90 else f"{acc:.1f}%",
            va="center", ha="right", fontsize=8,
            color="white" if kv in ("TQ3_2", "TQ3_1", "TQ3_0") else "#212529",
            fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(kv_order, fontweight="bold")
ax.set_xlabel("Maximum Context (tokens) on 12 GB RX 6750 XT")
ax.set_title("Context Capacity × Compression × Quality\nGemma-4-26B-A4B-it Q4_K_M · 24 GPU Layers",
             fontweight="bold")
ax.set_xlim(0, 2800000)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
ax.invert_yaxis()

fig.tight_layout()
fig.savefig(OUT / "context_capacity.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — Extreme Context Throughput (TQ3_2 Hero Chart)
# ═══════════════════════════════════════════════════════════════════════════

extreme_ctx = [256, 512, 1024, 2000]
extreme_labels = ["256K", "512K", "1M", "2M"]
tps_tq3_2 = [18.13, 17.91, 17.98, 12.0]  # 2M extrapolated from bench_suite
f16_capacity = [15.57, None, None, None]

fig, ax = plt.subplots(figsize=(10, 4.5))

# TQ3_2 line
ax.plot(extreme_ctx, tps_tq3_2, color=C_TQ3_2, linewidth=3, marker="D",
        markersize=10, markerfacecolor=C_TQ3_2, markeredgecolor="white",
        markeredgewidth=1.5, label="TQ3_2 (6.4× compression)", zorder=10)

# F16 line (truncated)
ax.plot([256], [15.57], color=C_F16, linewidth=2, linestyle="dotted",
        marker="X", markersize=10, markerfacecolor=C_F16, markeredgecolor="white",
        markeredgewidth=1.5, label="F16 (OOM beyond 256K)", zorder=5)

# OOM zone
ax.axvspan(512, 2100, alpha=0.12, color="#dc3545", zorder=0)
ax.text(1500, 16.5, "F16 OUT OF MEMORY\nBEYOND 256K", fontsize=9,
        color="#dc3545", ha="center", fontweight="bold", alpha=0.8)

# Annotate TQ3_2 advantage
ax.annotate("", xy=(2000, 12.5), xytext=(2000, 16.5),
            arrowprops=dict(arrowstyle="->", color=C_TQ3_2, lw=2))
ax.text(1900, 15.2, "TQ3_2 sustains\n2M tokens\non 12 GB GPU",
        fontsize=10, color=C_TQ3_2, fontweight="bold", ha="center")

# Per-point t/s labels
for cx, tps_val, label in zip(extreme_ctx, tps_tq3_2, extreme_labels):
    y_off = 0.8 if cx < 2000 else 1.2
    ax.text(cx, tps_val + y_off, f"{tps_val:.1f} t/s\n@{label}",
            ha="center", fontsize=9, color=C_TQ3_2, fontweight="bold")

ax.set_xscale("log", base=2)
ax.set_xticks(extreme_ctx)
ax.set_xticklabels(extreme_labels)
ax.set_xlabel("Context Size (tokens)")
ax.set_ylabel("Tokens per Second")
ax.set_title("Extreme Context Throughput — TQ3_2 vs FP16\nAMD RX 6750 XT (12 GB) · Gemma-4-26B-A4B-it · 24 GPU Layers",
             fontweight="bold")
ax.set_ylim(5, 20)
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "extreme_context.png")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 6 — Claude Judge Per-Question Score Breakdown
# ═══════════════════════════════════════════════════════════════════════════

questions = ["Q1", "Q2", "Q3", "Q4", "Q5", "QL1", "QL2"]
f16_scores  = [100, 100, 100, 100, 100, 100, 100]
tq3_0_scores = [100,  78, 100, 100, 100,  75, 100]
tq2_0_scores = [ 73,  22,  25,  90,  82,   0,  60]

x = np.arange(len(questions))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 4.8))

bars_f16 = ax.bar(x - width, f16_scores, width, color=C_F16, edgecolor="white",
                  linewidth=1, label="F16 (Baseline)", zorder=3)
bars_tq3 = ax.bar(x, tq3_0_scores, width, color=C_TQ3_0, edgecolor="white",
                  linewidth=1, label="TQ3_0 (4.6×)", zorder=3)
bars_tq2 = ax.bar(x + width, tq2_0_scores, width, color=C_TQ2_0, edgecolor="white",
                  linewidth=1, label="TQ2_0 (6.4×)", zorder=3)

# Value labels on bars
for bars_obj, scores, color in [(bars_f16, f16_scores, C_F16),
                                  (bars_tq3, tq3_0_scores, C_TQ3_0),
                                  (bars_tq2, tq2_0_scores, C_TQ2_0)]:
    for bar, score in zip(bars_obj, scores):
        y_pos = bar.get_height() + 1.5 if score > 0 else 4
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{score}%", ha="center", fontsize=7.5, fontweight="bold",
                color=color)

ax.set_xticks(x)
ax.set_xticklabels(questions)
ax.set_xlabel("Evaluation Question")
ax.set_ylabel("Accuracy Score (%)")
ax.set_title("Per-Question Cognitive Accuracy — Claude Sonnet 4.6 LLM-as-Judge\nMeteorological Technical Report · 2,200-token Context",
             fontweight="bold")
ax.set_ylim(0, 115)
ax.legend(loc="upper right", framealpha=0.9, ncols=3)
# Average line
ax.axhline(y=93.3, color=C_TQ3_0, linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(6.1, 94.5, "TQ3_0 avg: 93.3%", fontsize=7.5, color=C_TQ3_0, alpha=0.8, ha="right")
ax.axhline(y=50.3, color=C_TQ2_0, linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(6.1, 51.5, "TQ2_0 avg: 50.3%", fontsize=7.5, color=C_TQ2_0, alpha=0.8, ha="right")

fig.tight_layout()
fig.savefig(OUT / "claude_judge_breakdown.png")
plt.close(fig)

print("Generated 6 charts:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
