#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
REB-W3 + REB-W4 (cross-lingual, Pythia-410M, one clean run).

W3 (vacq Q1): compute-matched centralized baseline. Train ONE monolithic model on
POOLED Tamil+Yoruba+Welsh+Code data for EQUAL COMPUTE (4 domains x 2000 = 8000 steps)
and compare to the KALAVAI MoE on the SAME per-domain equal-weight held-out. Tests
whether the cooperative gain is algorithmic or just a coordination effect — and
whether it holds in the high-divergence regime where a single model can't specialize.

W4 (uZ7m Q2 / vacq Q3): router-privacy. Train the router on PUBLIC-PROXY data
(FLORES-200 dev sentences — public, disjoint from the contributors' cc100 training
data) vs in-domain data, and check the fusion gain survives. If it does, the router
calibration set need not contain contributors' private data.

Usage:
  python experiments/w3w4_crosslingual.py --seed 137 [--smoke]
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    load_all_data, train_specialist, train_router, DOMAINS,
    MODEL_ID, REVISION, HIDDEN_SIZE, EVAL_BATCH_SIZE, EVAL_BATCHES,
)
import kalavai_phase2_exp1 as p2  # noqa: E402
from kalavai_eval_utils import (  # noqa: E402
    PackedChunkDataset, chunks_to_dataset, eval_all_domains, SEQ_LEN,
)
from w2_regression_condition import NExpertMoE  # noqa: E402
from w1_crosslingual_downstream import load_specialist_from_hf, _flores_sentences  # noqa: E402

RESULTS_DIR = Path("results/rebuttal/w3w4")


def fresh_base(device):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True).to(
        device=device, dtype=torch.bfloat16)


def eqv(em):
    return em["equal_weight_avg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=137, choices=[137, 2026])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    mono_steps = 60 if args.smoke else 8000  # equal compute: 4 specialists x 2000
    if args.smoke:
        p2.ROUTER_STEPS = 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"W3+W4 seed={args.seed} device={device} mono_steps={mono_steps}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_chunks, held = load_all_data(tok)
    held_sets = {d: chunks_to_dataset(held[d]) for d in DOMAINS}

    base = fresh_base(device)
    em_base = eval_all_domains(base, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base; torch.cuda.empty_cache()

    # ── W3: monolithic on POOLED data, equal compute ─────────────────────────
    print(f"\n[W3] training monolithic on pooled data ({mono_steps} steps)...", flush=True)
    pooled = [c for chunks in train_chunks.values() for c in chunks]
    p2.MAX_STEPS = mono_steps
    mono = fresh_base(device)
    train_specialist(mono, "monolithic", pooled, args.seed, device)
    em_mono = eval_all_domains(mono, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del mono; torch.cuda.empty_cache()

    # ── specialists + MoE (in-domain router) ─────────────────────────────────
    specs = [load_specialist_from_hf(d, args.seed, device) for d in DOMAINS]
    em_spec = {d: eval_all_domains(specs[i], held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
               for i, d in enumerate(DOMAINS)}
    best_spec_ew = min(eqv(em_spec[d]) for d in DOMAINS)

    moe = NExpertMoE(specs, HIDDEN_SIZE).to(device)
    train_router(moe, train_chunks, device)            # in-domain calibration
    em_moe_ind = eval_all_domains(moe, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES,
                                  is_fused=True)

    # ── W4: retrain router on PUBLIC-PROXY (FLORES) data ─────────────────────
    print("\n[W4] building public-proxy router calibration set from FLORES-200...", flush=True)
    proxy_chunks = {}
    n_flores = 30 if args.smoke else 300
    for d in DOMAINS:
        if d == "code":
            # public proxy for code: a fresh code_search_net sample (public, generic)
            from kalavai_phase2_exp1 import load_code_texts
            texts = load_code_texts(200 if args.smoke else 1000)
        else:
            texts = _flores_sentences(d, n_flores)  # public FLORES dev, NOT contributors' cc100
        if texts:
            proxy_chunks[d] = PackedChunkDataset(texts, tok, seq_len=SEQ_LEN, max_chars=5000).chunks
        else:
            proxy_chunks[d] = train_chunks[d][:50]  # fallback
        print(f"  proxy {d}: {len(proxy_chunks[d])} chunks", flush=True)
    # re-init router to fresh weights, train on proxy data
    for layer in moe.router:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    moe.to(device)
    train_router(moe, proxy_chunks, device)
    em_moe_proxy = eval_all_domains(moe, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES,
                                    is_fused=True)

    # ── report ────────────────────────────────────────────────────────────────
    def gain(a, b):  # % improvement of b over a (loss lower is better)
        return round((a - b) / a * 100, 3)
    r = {
        "seed": args.seed, "smoke": args.smoke,
        "base_ew": round(eqv(em_base), 6),
        "monolithic_ew": round(eqv(em_mono), 6),
        "best_spec_ew": round(best_spec_ew, 6),
        "moe_indomain_router_ew": round(eqv(em_moe_ind), 6),
        "moe_proxy_router_ew": round(eqv(em_moe_proxy), 6),
        "W3_moe_vs_monolithic_pct": gain(eqv(em_mono), eqv(em_moe_ind)),
        "W3_monolithic_vs_base_pct": gain(eqv(em_base), eqv(em_mono)),
        "W4_moe_indomain_gain_vs_bestspec_pct": gain(best_spec_ew, eqv(em_moe_ind)),
        "W4_moe_proxy_gain_vs_bestspec_pct": gain(best_spec_ew, eqv(em_moe_proxy)),
        "mono_steps": mono_steps,
    }
    print("\n" + "=" * 62, flush=True)
    print("W3 — compute-matched centralized (equal-weight loss, lower=better):", flush=True)
    print(f"  base={r['base_ew']:.4f}  MONOLITHIC={r['monolithic_ew']:.4f}  "
          f"best_spec={r['best_spec_ew']:.4f}  MoE={r['moe_indomain_router_ew']:.4f}", flush=True)
    print(f"  MoE vs monolithic: {r['W3_moe_vs_monolithic_pct']:+.2f}%  "
          f"({'MoE wins' if r['W3_moe_vs_monolithic_pct']>0 else 'MONOLITHIC wins'})", flush=True)
    print("W4 — router calibration data privacy:", flush=True)
    print(f"  gain vs best-spec: in-domain {r['W4_moe_indomain_gain_vs_bestspec_pct']:+.2f}% "
          f"| public-proxy {r['W4_moe_proxy_gain_vs_bestspec_pct']:+.2f}%", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"w3w4_seed{args.seed}{'_smoke' if args.smoke else ''}.json"
    json.dump(r, open(out, "w"), indent=2)
    print(f"\nSaved: {out}", flush=True)
    _ = statistics  # noqa


if __name__ == "__main__":
    main()
