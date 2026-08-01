#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
REB-W5: BTX head-to-head — the technical delta from Branch-Train-Mix.

Isolates the ONE axis the novelty argument hinges on: does fusion require jointly
training the experts on POOLED mixed data (BTX), or does post-hoc routing over
FROZEN experts (KALAVAI) recover the same fused quality?

Three fusion approaches on the SAME independently-trained specialists:
  1. uniform   — frozen experts, UNIFORM (untrained) router  [no training; paper ref]
  2. KALAVAI   — frozen experts, TRAINED router               [router calib set only]
  3. BTX-style — experts UNFROZEN + jointly fine-tuned on pooled mixed data + router

Reported: per-domain equal-weight loss for each, plus a "what it requires" table.
Thesis: KALAVAI recovers BTX-quality WITHOUT the pooled joint training BTX needs.

NOTE: model-level MoE (routes between whole specialist models), framed honestly as
"BTX-style joint fine-tuning" — NOT a faithful layer-level (merged-FFN) BTX.

Usage:
  python experiments/w5_btx_headtohead.py --seed 137 [--smoke]
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from itertools import cycle

sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    load_all_data, DOMAINS, MODEL_ID, REVISION, HIDDEN_SIZE,
    EVAL_BATCH_SIZE, EVAL_BATCHES,
)
from kalavai_eval_utils import chunks_to_dataset, eval_all_domains, _collate  # noqa: E402
from w2_regression_condition import NExpertMoE, train_router  # noqa: E402
from w1_crosslingual_downstream import load_base, load_specialist_from_hf  # noqa: E402

RESULTS_DIR = Path("results/rebuttal/w5_btx")
BTX_STEPS = 500
BTX_LR = 1e-5
BTX_GRAD_ACCUM = 8


def btx_loss(moe, input_ids, labels):
    """Differentiable fused forward (gradients flow to experts AND router)."""
    logits, h_sum = [], 0
    for m in moe.experts:
        out = m(input_ids=input_ids, output_hidden_states=True)
        logits.append(out.logits)
        h_sum = h_sum + out.hidden_states[-1].mean(1).float()
    gates = torch.softmax(moe.router(h_sum / moe.n), dim=-1)
    fused = sum(gates[:, i:i + 1, None] * logits[i] for i in range(moe.n))
    sl = fused[:, :-1].contiguous()
    slb = labels[:, 1:].contiguous()
    return F.cross_entropy(sl.view(-1, sl.size(-1)), slb.view(-1))


def eq(em):
    return em["equal_weight_avg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=137, choices=[137, 2026])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps = 30 if args.smoke else BTX_STEPS
    if args.smoke:  # speed up the KALAVAI router training in smoke runs too
        import w2_regression_condition as w2mod
        w2mod.ROUTER_STEPS = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"W5 BTX head-to-head seed={args.seed} device={device} steps={steps}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_chunks, held = load_all_data(tok)
    held_sets = {d: chunks_to_dataset(held[d]) for d in DOMAINS}

    def EE(model, fused=False):
        return eq(eval_all_domains(model, held_sets, device,
                                   EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=fused))

    base = load_base(device)
    base_eq = EE(base)
    del base; torch.cuda.empty_cache()

    results = {"seed": args.seed, "base_eq": round(base_eq, 6), "variants": {}}

    def load_specs():
        return [load_specialist_from_hf(d, args.seed, device) for d in DOMAINS]

    # ── 1. uniform + 2. KALAVAI (frozen experts) ─────────────────────────────
    specs = load_specs()
    moe = NExpertMoE(specs, HIDDEN_SIZE).to(device)  # experts frozen in __init__
    # uniform router: zero the router so softmax -> uniform gates
    with torch.no_grad():
        for p in moe.router.parameters():
            p.zero_()
    uni_eq = EE(moe, fused=True)
    results["variants"]["uniform"] = {"eq": round(uni_eq, 6)}
    print(f"[uniform] eq={uni_eq:.4f}", flush=True)

    # KALAVAI: re-init the router to FRESH RANDOM weights before training — a
    # zeroed init is symmetric and cannot break symmetry (would stay uniform).
    for layer in moe.router:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    moe.to(device)
    train_router(moe, train_chunks, device)
    kal_eq = EE(moe, fused=True)
    results["variants"]["kalavai"] = {"eq": round(kal_eq, 6)}
    print(f"[kalavai] eq={kal_eq:.4f}", flush=True)
    del moe, specs; torch.cuda.empty_cache()

    # ── 3. BTX-style: experts UNFROZEN, jointly fine-tuned on pooled data ─────
    specs = load_specs()
    moe = NExpertMoE(specs, HIDDEN_SIZE).to(device)
    for m in moe.experts:
        m.gradient_checkpointing_enable()
        for p in m.parameters():
            p.requires_grad_(True)
    moe.train()
    opt = torch.optim.AdamW(moe.parameters(), lr=BTX_LR)
    pooled = [c for chunks in train_chunks.values() for c in chunks]
    loader = DataLoader(chunks_to_dataset(pooled), batch_size=1, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    t0 = time.time()
    opt.zero_grad()
    for step in range(1, steps + 1):
        b = next(it)
        loss = btx_loss(moe, b["input_ids"].to(device), b["labels"].to(device)) / BTX_GRAD_ACCUM
        loss.backward()
        if step % BTX_GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(moe.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step % 100 == 0 or step == steps:
            print(f"  [btx] step {step}/{steps} loss={loss.item()*BTX_GRAD_ACCUM:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    moe.eval()
    btx_eq = EE(moe, fused=True)
    results["variants"]["btx"] = {"eq": round(btx_eq, 6), "steps": steps, "lr": BTX_LR}
    print(f"[btx] eq={btx_eq:.4f}", flush=True)

    # ── report ────────────────────────────────────────────────────────────────
    def gain_vs_base(e):
        return round((base_eq - e) / base_eq * 100, 3)
    results["improvement_vs_base_pct"] = {
        k: gain_vs_base(v["eq"]) for k, v in results["variants"].items()}
    results["requires"] = {
        "uniform": {"expert_training": False, "data_pooling": False, "router_calib": False},
        "kalavai": {"expert_training": False, "data_pooling": "router calib set only",
                    "gradient_sharing": False},
        "btx":     {"expert_training": True, "data_pooling": True, "gradient_sharing": True,
                    "joint_finetune": True},
    }
    print("\n" + "=" * 60, flush=True)
    print("BTX HEAD-TO-HEAD (equal-weight loss; gain vs base %)", flush=True)
    for k in ["uniform", "kalavai", "btx"]:
        print(f"  {k:9s} eq={results['variants'][k]['eq']:.4f}  "
              f"gain_vs_base={results['improvement_vs_base_pct'][k]:+.2f}%", flush=True)
    kal, btx = results["variants"]["kalavai"]["eq"], results["variants"]["btx"]["eq"]
    print(f"\n  KALAVAI vs BTX eq gap: {(kal-btx)/btx*100:+.2f}%", flush=True)
    print("  KALAVAI needs NO expert training / NO data pooling; BTX needs both.", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"w5_btx_seed{args.seed}{'_smoke' if args.smoke else ''}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
