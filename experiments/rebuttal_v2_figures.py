#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
"""Generate the new v2 figures from rebuttal result JSONs (no GPU needed).
Outputs to figures/rebuttal/: fig_w5_btx.png, fig_w3_centralized.png, fig_w7_7b.png
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, COLORS, clean_axes  # noqa: E402

OUT = Path("figures/rebuttal")
OUT.mkdir(parents=True, exist_ok=True)
apply_style()


def mean(vals):
    return float(np.mean(vals))


# ── W5: BTX head-to-head ──────────────────────────────────────────────────────
runs = [json.load(open(p)) for p in sorted(glob.glob("results/rebuttal/w5_btx/w5_btx_seed*.json"))
        if "smoke" not in p]
if runs:
    g = {k: mean([r["improvement_vs_base_pct"][k] for r in runs]) for k in ["uniform", "kalavai", "btx"]}
    fig, ax = plt.subplots(figsize=(7, 4.8))
    labels = ["Uniform\n(no training)", "KALAVAI\n(frozen + router)", "BTX-style\n(joint fine-tune)"]
    colors = [COLORS["base"], COLORS["moe"], COLORS["monolithic"]]
    bars = ax.bar(range(3), [g["uniform"], g["kalavai"], g["btx"]], color=colors, width=0.6)
    for b, v in zip(bars, [g["uniform"], g["kalavai"], g["btx"]]):
        ax.annotate(f"+{v:.1f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=10, fontweight="medium")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("Fusion gain vs. base (%)")
    ax.set_title("KALAVAI recovers ~96% of BTX quality\nwithout joint training or data pooling")
    ax.text(0.5, -0.30, "Requires: data pooling / joint expert training —  "
            "Uniform: no  •  KALAVAI: no  •  BTX: YES",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444")
    clean_axes(ax)
    fig.savefig(OUT / "fig_w5_btx.png", bbox_inches="tight")
    print("saved fig_w5_btx.png", g)

# ── W3: compute-matched centralized (cross-lingual) ──────────────────────────
d = json.load(open("results/rebuttal/w3w4/w3w4_seed137.json"))
fig, ax = plt.subplots(figsize=(7, 4.8))
names = ["Base", "KALAVAI\nMoE", "Best single\nspecialist", "Centralized\nmonolithic"]
vals = [d["base_ew"], d["moe_indomain_router_ew"], d["best_spec_ew"], d["monolithic_ew"]]
colors = [COLORS["base"], COLORS["moe"], COLORS["science"], COLORS["monolithic"]]
bars = ax.bar(range(4), vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(names)
ax.set_ylabel("Equal-weight loss (lower = better)")
ax.set_ylim(1.8, 3.05)
ax.set_title("Honest: compute-matched centralized training beats\ncooperative fusion by ~13% (cross-lingual, equal compute)")
clean_axes(ax)
fig.savefig(OUT / "fig_w3_centralized.png", bbox_inches="tight")
print("saved fig_w3_centralized.png")

# ── W7: 7B — LM gains but no task win ────────────────────────────────────────
w7 = json.load(open("results/rebuttal/w7_qwen7b/w7_seed42.json"))
D = ["tamil", "yoruba", "welsh", "code"]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(D)); w = 0.38
base_cl = [w7["downstream"]["base"]["cloze_acc"][d_] for d_ in D]
moe_cl = [w7["downstream"]["moe"]["cloze_acc"][d_] for d_ in D]
axA.bar(x - w / 2, base_cl, w, label="Base (Qwen2.5-7B)", color=COLORS["base"])
axA.bar(x + w / 2, moe_cl, w, label="KALAVAI MoE", color=COLORS["moe"])
axA.set_xticks(x); axA.set_xticklabels([d_.capitalize() for d_ in D])
axA.set_ylabel("Cloze accuracy"); axA.set_title("A. Language modeling improves (7B)")
axA.legend(); clean_axes(axA)
bl = ["tamil", "yoruba"]
xb = np.arange(len(bl))
bb = [w7["downstream"]["base"]["belebele"][d_]["accuracy"] for d_ in bl]
bm = [w7["downstream"]["moe"]["belebele"][d_]["accuracy"] for d_ in bl]
axB.bar(xb - w / 2, bb, w, label="Base", color=COLORS["base"])
axB.bar(xb + w / 2, bm, w, label="KALAVAI MoE", color=COLORS["moe"])
axB.axhline(0.25, ls="--", c="gray", lw=1, label="chance")
axB.set_xticks(xb); axB.set_xticklabels([d_.capitalize() for d_ in bl])
axB.set_ylabel("Belebele MC-QA accuracy")
axB.set_title("B. But downstream TASK accuracy does not\n(MoE ≤ base — raw-text FT erodes QA)")
axB.legend(); clean_axes(axB)
fig.suptitle("REB-W7: 7B fusion improves LM but not task accuracy", y=1.02, fontsize=12)
fig.savefig(OUT / "fig_w7_7b.png", bbox_inches="tight")
print("saved fig_w7_7b.png")
