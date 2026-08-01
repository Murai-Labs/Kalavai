#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
"""
Figure for REB-W1: cross-lingual DOWNSTREAM evaluation.

Panel A: per-language cloze accuracy (base / own specialist / MoE) — MoE matches
         each specialist (router recovers the diagonal) and beats base.
Panel B: THE HEADLINE — average across languages: base / best SINGLE specialist /
         MoE. The MoE beats any individual specialist because each specialist is
         good at only its own language (+~11% rel).
Panel C: FLORES-200 perplexity (base vs MoE), log scale — magnitude of the effect.

Reads results/rebuttal/w1_crosslingual/w1_downstream_seed*.json (averages seeds).
Writes figures/rebuttal/w1_crosslingual_downstream.png.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, COLORS, clean_axes  # noqa: E402

RESULTS_GLOB = "results/rebuttal/w1_crosslingual/w1_downstream_seed*.json"
OUT = Path("figures/rebuttal/w1_crosslingual_downstream.png")
LANGS = ["tamil", "yoruba", "welsh", "code"]
FLORES_LANGS = ["tamil", "yoruba", "welsh"]


def load_seeds():
    runs = [json.load(open(p)) for p in sorted(glob.glob(RESULTS_GLOB))]
    if not runs:
        raise SystemExit(f"No results found at {RESULTS_GLOB}")
    return runs


def mean_over_seeds(runs, fn):
    vals = [fn(r) for r in runs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def cloze(run, model, lang):
    return run["metrics"][model]["cloze_acc"][lang]


def own_spec_cloze(run, lang):
    return run["metrics"][f"{lang}_spec"]["cloze_acc"][lang]


def avg_cloze(run, model):
    return sum(run["metrics"][model]["cloze_acc"][x] for x in LANGS) / len(LANGS)


def best_single_spec_avg(run):
    return max(sum(run["metrics"][f"{s}_spec"]["cloze_acc"][x] for x in LANGS) / len(LANGS)
               for s in LANGS)


def flores_ppl(run, model, lang):
    v = run["metrics"][model]["flores"][lang]
    return v["ppl"] if v else None


def main():
    runs = load_seeds()
    seeds = [r["seed"] for r in runs]
    apply_style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))

    # ── Panel A: per-language cloze (base / own spec / MoE) ───────────────────
    x = np.arange(len(LANGS)); w = 0.26
    base = [mean_over_seeds(runs, lambda r, la=la: cloze(r, "base", la)) for la in LANGS]
    ownspec = [mean_over_seeds(runs, lambda r, la=la: own_spec_cloze(r, la)) for la in LANGS]
    moe = [mean_over_seeds(runs, lambda r, la=la: cloze(r, "moe", la)) for la in LANGS]
    axA.bar(x - w, base, w, label="Base", color=COLORS["base"])
    axA.bar(x, ownspec, w, label="Own specialist", color=COLORS["science"])
    axA.bar(x + w, moe, w, label="KALAVAI MoE", color=COLORS["moe"])
    axA.set_xticks(x); axA.set_xticklabels([la.capitalize() for la in LANGS])
    axA.set_ylabel("Cloze accuracy")
    axA.set_title("A. Per language: MoE recovers each specialist")
    axA.legend(); clean_axes(axA)

    # ── Panel B: HEADLINE — average across languages ─────────────────────────
    base_avg = mean_over_seeds(runs, lambda r: avg_cloze(r, "base"))
    bss_avg = mean_over_seeds(runs, best_single_spec_avg)
    moe_avg = mean_over_seeds(runs, lambda r: avg_cloze(r, "moe"))
    bars = axB.bar([0, 1, 2], [base_avg, bss_avg, moe_avg],
                   color=[COLORS["base"], COLORS["science"], COLORS["moe"]], width=0.6)
    axB.set_xticks([0, 1, 2])
    axB.set_xticklabels(["Base", "Best single\nspecialist", "KALAVAI\nMoE"])
    axB.set_ylabel("Cloze accuracy (avg over languages)")
    axB.set_title("B. One model beats any individual specialist")
    for b, v in zip(bars, [base_avg, bss_avg, moe_avg]):
        axB.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    axB.annotate(f"+{(moe_avg-bss_avg)*100:.1f}pp ({(moe_avg/bss_avg-1)*100:+.0f}%)",
                 (2, moe_avg), textcoords="offset points", xytext=(0, 16),
                 ha="center", fontsize=10, color=COLORS["moe"], fontweight="bold")
    clean_axes(axB)

    # ── Panel C: FLORES perplexity (log) ─────────────────────────────────────
    xb = np.arange(len(FLORES_LANGS)); wb = 0.36
    fb = [mean_over_seeds(runs, lambda r, la=la: flores_ppl(r, "base", la)) for la in FLORES_LANGS]
    fm = [mean_over_seeds(runs, lambda r, la=la: flores_ppl(r, "moe", la)) for la in FLORES_LANGS]
    axC.bar(xb - wb / 2, fb, wb, label="Base", color=COLORS["base"])
    axC.bar(xb + wb / 2, fm, wb, label="KALAVAI MoE", color=COLORS["moe"])
    axC.set_yscale("log"); axC.set_xticks(xb)
    axC.set_xticklabels([la.capitalize() for la in FLORES_LANGS])
    axC.set_ylabel("FLORES-200 perplexity (log)")
    axC.set_title("C. Magnitude: perplexity collapse")
    for i, (b, m) in enumerate(zip(fb, fm)):
        if b and m:
            axC.annotate(f"{b/m:.1f}×", (i, m), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=9, color=COLORS["moe"])
    axC.legend(); clean_axes(axC)

    fig.suptitle(f"REB-W1 cross-lingual downstream (Pythia-410M, seeds {seeds})",
                 fontsize=12, y=1.03)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT} (seeds={seeds}) | MoE_avg={moe_avg:.4f} "
          f"best_single_spec_avg={bss_avg:.4f} delta={(moe_avg-bss_avg):+.4f}")


if __name__ == "__main__":
    main()
