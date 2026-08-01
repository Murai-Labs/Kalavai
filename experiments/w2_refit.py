#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
"""
REB-W2 refit + held-out validation of the divergence-gain predictor.

Fits the linear law gain = a + b*divergence on the ORIGINAL 8 conditions, then
treats the freshly-run cross-lingual subset points (results/rebuttal/w2_regression/
cond_*.json) as HELD-OUT and checks how many fall inside the 95% prediction band.
This directly answers reviewers' "fit on n=8, doesn't generalize" objection: if
independently-run points (different domains, fresh training) land on the line, the
law generalizes.

Outputs:
  results/rebuttal/w2_regression/refit_summary.json
  figures/rebuttal/w2_divergence_gain_refit.png
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, clean_axes  # noqa: E402

# Original 8 in-paper conditions (from analysis/item1_regression_fit.py).
ORIGINAL = [
    ("Qwen-1.5B", 3.16, 1.06), ("Pythia-6.9B", 8.73, 6.53),
    ("P1 2-domain", 10.77, 6.22), ("Pythia-1B", 15.28, 7.49),
    ("Pythia-410M", 15.65, 7.72), ("Exp2 Private", 18.52, 10.17),
    ("P2 4-domain", 19.84, 14.71), ("Exp1 Cross-lingual", 25.65, 21.76),
]
NEW_GLOB = "results/rebuttal/w2_regression/cond_*.json"
OUT_FIG = Path("figures/rebuttal/w2_divergence_gain_refit.png")
OUT_JSON = Path("results/rebuttal/w2_regression/refit_summary.json")


def fit_ols(x, y):
    n = len(x)
    Xm = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(Xm, y, rcond=None)[0]
    resid = y - Xm @ b
    s2 = float(np.sum(resid ** 2) / (n - 2))
    r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    return b, s2, r2, XtX_inv, se


def pred_band(xq, b, s2, x_fit, XtX_inv, t95):
    Xq = np.column_stack([np.ones(len(xq)), xq])
    yq = Xq @ b
    sep = np.sqrt(s2 * (1 + np.diag(Xq @ XtX_inv @ Xq.T)))
    return yq, yq - t95 * sep, yq + t95 * sep


def main():
    from scipy.stats import t as tdist
    new = []
    for p in sorted(glob.glob(NEW_GLOB)):
        d = json.load(open(p))
        new.append(("+".join(d["domains"]), d["mean_divergence"], d["fusion_gain"]))
    print(f"original={len(ORIGINAL)} new_held_out={len(new)}")
    if not new:
        print("No new W2 points yet — run w2_regression_condition.py first.")
        return

    ox = np.array([c[1] for c in ORIGINAL]); oy = np.array([c[2] for c in ORIGINAL])
    b, s2, r2, XtX_inv, se = fit_ols(ox, oy)
    t95 = float(tdist.ppf(0.975, df=len(ox) - 2))
    print(f"FIT on original 8: gain = {b[0]:.2f} + {b[1]:.3f}*div  R2={r2:.3f}  "
          f"slope95%CI=[{b[1]-t95*se[1]:.2f},{b[1]+t95*se[1]:.2f}]")

    # held-out validation
    nx = np.array([c[1] for c in new]); ny = np.array([c[2] for c in new])
    yq, lo, hi = pred_band(nx, b, s2, ox, XtX_inv, t95)
    in_band = (ny >= lo) & (ny <= hi)
    rmse = float(np.sqrt(np.mean((ny - yq) ** 2)))
    print(f"HELD-OUT: {int(in_band.sum())}/{len(new)} in 95% band | RMSE={rmse:.2f}pp")
    for (lab, dv, gn), pr, ib in zip(new, yq, in_band):
        print(f"  {lab:22s} div={dv:5.2f} gain={gn:6.2f} pred={pr:6.2f} resid={gn-pr:+.2f} "
              f"{'IN' if ib else 'OUT'}")

    # refit on all
    ax_all = np.concatenate([ox, nx]); ay_all = np.concatenate([oy, ny])
    ball, _, r2all, _, _ = fit_ols(ax_all, ay_all)
    print(f"REFIT on all n={len(ax_all)}: gain = {ball[0]:.2f} + {ball[1]:.3f}*div  R2={r2all:.3f}")

    # figure
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xr = np.linspace(0, max(ax_all.max(), 40) + 2, 200)
    yr, rlo, rhi = pred_band(xr, b, s2, ox, XtX_inv, t95)
    ax.fill_between(xr, rlo, rhi, alpha=0.15, color="#1565C0", label="95% pred. band (fit on n=8)")
    ax.plot(xr, yr, "-", color="#1565C0", lw=2, label=f"Fit on original 8 ($R^2$={r2:.2f})")
    ax.scatter(ox, oy, s=90, color="#1565C0", edgecolors="black", lw=0.5, zorder=5,
               label="Original 8 (fit)")
    ax.scatter(nx[in_band], ny[in_band], s=110, marker="*", color="#2e7d32",
               edgecolors="black", lw=0.5, zorder=6, label="New held-out (in band)")
    if (~in_band).any():
        ax.scatter(nx[~in_band], ny[~in_band], s=110, marker="X", color="#c62828",
                   edgecolors="black", lw=0.5, zorder=6, label="New held-out (out of band)")
    ax.set_xlabel("Mean specialist divergence (%)")
    ax.set_ylabel("Fusion gain vs best specialist (%)")
    ax.set_title(f"Divergence–gain law generalizes: {int(in_band.sum())}/{len(new)} "
                 f"held-out in band (RMSE {rmse:.1f}pp)")
    ax.legend(fontsize=8); clean_axes(ax)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "fit_original": {"intercept": b[0], "slope": b[1], "r2": r2,
                         "slope_ci": [b[1] - t95 * se[1], b[1] + t95 * se[1]]},
        "held_out": {"n": len(new), "in_band": int(in_band.sum()), "rmse_pp": rmse,
                     "points": [{"label": c[0], "div": c[1], "gain": c[2],
                                 "pred": float(pr), "in_band": bool(ib)}
                                for c, pr, ib in zip(new, yq, in_band)]},
        "refit_all": {"n": len(ax_all), "intercept": ball[0], "slope": ball[1], "r2": r2all},
    }, open(OUT_JSON, "w"), indent=2)
    print(f"saved {OUT_FIG} and {OUT_JSON}")


if __name__ == "__main__":
    main()
