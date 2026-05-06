# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
bench_kv_lines.py
Gráfico de linhas: X = tipo de KV  |  Y = % relativa ao f16
3 linhas: Qualidade / VRAM / t/s
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Valores absolutos ─────────────────────────────────────────────────────────
# qualidade: média das 7 questões (avaliação manual)
# vram: MiB do KV cache (CTX 16384)
# tps:  média (std + long)

ABS = {
    "f16":   {"quality": 100.0, "vram": 620.0, "tps": (15.18 + 15.76) / 2},
    "tq3_0": {"quality":  93.3, "vram": 136.0, "tps": (16.55 + 15.76) / 2},
    "tq2_0": {"quality":  50.3, "vram":  97.0, "tps": (16.56 + 16.30) / 2},
}

KV   = ["f16", "tq3_0", "tq2_0"]
XLAB = ["f16\n(FP16 baseline)", "tq3_0\n(3-bit KV)", "tq2_0\n(2-bit KV)"]

# Normaliza tudo relativo ao f16
BASE = ABS["f16"]
REL  = {k: {m: v / BASE[m] * 100 for m, v in ABS[k].items()} for k in KV}

quality = [REL[k]["quality"] for k in KV]
vram    = [REL[k]["vram"]    for k in KV]
tps     = [REL[k]["tps"]     for k in KV]

xs = np.array([0, 1, 2])

# ── Cores ─────────────────────────────────────────────────────────────────────
C_QUAL = "#4fc3f7"   # azul — qualidade
C_VRAM = "#ef5350"   # vermelho — VRAM (cai drasticamente = bom)
C_TPS  = "#66bb6a"   # verde — velocidade (quase plana)

FIGBG  = "#1a1a1a"
PLOTBG = "#242424"

# ── Figura ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=FIGBG)
ax.set_facecolor(PLOTBG)
for sp in ax.spines.values():
    sp.set_color("#444")
ax.tick_params(colors="#bbb", labelsize=10.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.22, color="#555")
ax.axhline(100, color="#555", lw=0.9, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# ── Área sombreada sob a curva de qualidade ───────────────────────────────────
ax.fill_between(xs, quality, 0, color=C_QUAL, alpha=0.07, zorder=1)

# ── Linhas ────────────────────────────────────────────────────────────────────
lw = 2.6
ms = 9

ax.plot(xs, quality, color=C_QUAL, lw=lw, marker="o", ms=ms,
        label="Quality  (accuracy %)", zorder=4)
ax.plot(xs, vram,    color=C_VRAM, lw=lw, marker="s", ms=ms,
        label="VRAM  (% of f16)", zorder=4)
ax.plot(xs, tps,     color=C_TPS,  lw=lw, marker="^", ms=ms,
        label="Speed  (t/s, % of f16)", zorder=4)

# ── Anotações numéricas ───────────────────────────────────────────────────────
# Offsets (dx_pts, dy_pts, ha, va) per metric × KV point
# f16 (idx=0): all lines converge at ~100%  → spread them out
# tq3_0 (idx=1): quality (93%) and t/s (104%) are close → separate them
ANN = {
    "quality": [(-30, +10, "center", "bottom"),   # f16  → esquerda, acima
                (  0, -14, "center", "top"   ),   # tq3_0 → abaixo do ponto
                (  0, +12, "center", "bottom")],  # tq2_0 → acima
    "vram":    [(  0, +28, "center", "bottom"),   # f16  → centro, bem acima
                (  0, +10, "center", "bottom"),   # tq3_0 → acima (ponto baixo)
                (  0, +10, "center", "bottom")],  # tq2_0 → acima
    "tps":     [(+30, +10, "center", "bottom"),   # f16  → direita, acima
                ( +8, +12, "center", "bottom"),   # tq3_0 → acima do ponto alto
                ( +8, +12, "center", "bottom")],  # tq2_0 → acima
}

def ann_metric(ax, xs, ys, fmts, color, key):
    for i, (x, y) in enumerate(zip(xs, ys)):
        dx, dy, ha, va = ANN[key][i]
        ax.annotate(fmts[i],
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha, va=va,
                    fontsize=9.5, color=color,
                    fontweight="bold", zorder=5)

labels_q = [f"{ABS[k]['quality']:.1f}%" for k in KV]
labels_v = [f"{ABS[k]['vram']:.0f} MiB\n({REL[k]['vram']:.0f}%)" for k in KV]
labels_t = [f"{ABS[k]['tps']:.1f} t/s" for k in KV]

ann_metric(ax, xs, quality, labels_q, C_QUAL, "quality")
ann_metric(ax, xs, vram,    labels_v, C_VRAM, "vram")
ann_metric(ax, xs, tps,     labels_t, C_TPS,  "tps")

# ── Eixos ─────────────────────────────────────────────────────────────────────
ax.set_xticks(xs)
ax.set_xticklabels(XLAB, fontsize=11, color="#ddd")
ax.set_xlim(-0.35, 2.35)
ax.set_ylim(-8, 130)
ax.set_ylabel("% relative to f16 baseline", color="#bbb", fontsize=10.5)
ax.set_xlabel("KV cache compression type", color="#bbb", fontsize=10.5)

ax.set_title(
    "Quality  ·  VRAM  ·  Speed\n"
    "by KV cache compression type  —  Gemma-4 26B Q4_K_M  ·  RX 6750 XT",
    color="white", fontsize=12, fontweight="bold", pad=12, linespacing=1.5,
)

ax.legend(loc="upper right", framealpha=0.3, facecolor="#333",
          edgecolor="#555", labelcolor="#ccc", fontsize=9.5)

# Nota de rodapé
fig.text(
    0.5, 0.01,
    "Quality: avg of 7 questions (5 chained × 2 rounds + 2 long-context × 1 round)  ·  "
    "Manual evaluation by Claude Sonnet 4.6  ·  VRAM: KV cache CTX=16384",
    ha="center", va="bottom", fontsize=7, color="#555", style="italic",
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out = r"c:\Users\hm\Desktop\paia\bench_kv_lines.png"
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[OK] {out}")
