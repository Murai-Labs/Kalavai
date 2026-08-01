#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
REB-W6 + REB-W1 seed 42: retrain the seed-42 cross-lingual specialists (which are
NOT on the HF Hub), apply a router re-initialization fix for the known seed-42
Yoruba->Tamil router collapse, and run the W1 downstream battery to produce the
3rd seed for REB-W1.

W6 fix: the paper documents that at seed 42 the router collapses Yoruba onto the
Tamil expert (both are tokenizer-OOD byte-fallback scripts). Here we detect that
collapse from the held-out routing distribution and re-initialize + retrain the
router (varying its init) until every language routes to its own expert, up to
--max-router-tries. We report how many tries were needed.

Usage:
  python experiments/w6_seed42_train_and_downstream.py --limit 100
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    train_specialist, train_router, load_all_data, eval_router_distribution,
    FourExpertMoE, DOMAINS, MODEL_ID, REVISION, HIDDEN_SIZE,
)
from kalavai_eval_utils import chunks_to_dataset  # noqa: E402
from w1_crosslingual_downstream import (  # noqa: E402
    evaluate_model, _flores_sentences, _belebele, load_base,
)
from transformers import AutoModelForCausalLM  # noqa: E402

SEED = 42
RESULTS_DIR = Path("results/rebuttal/w1_crosslingual")


def fresh_base(device):
    # load-then-cast (version-independent; no dtype/torch_dtype kwarg)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
    ).to(device=device, dtype=torch.bfloat16)
    return m


def check_routing(moe, held_out_sets, device):
    """Every language should route to its OWN expert (argmax gate == its index)."""
    dist = eval_router_distribution(moe, held_out_sets, device)
    per = {}
    for i, d in enumerate(DOMAINS):
        gates = dist[d]
        per[d] = (max(range(4), key=lambda j: gates[j]) == i)
    return all(per.values()), per, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-router-tries", type=int, default=5)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end validation (few steps, limit=5)")
    args = ap.parse_args()

    if args.smoke:
        import kalavai_phase2_exp1 as p2
        p2.MAX_STEPS = 40
        p2.ROUTER_STEPS = 60
        args.limit = 5
        args.max_router_tries = 2
        print("[smoke] MAX_STEPS=40 ROUTER_STEPS=60 limit=5 tries=2", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"REB-W6 seed-42 retrain + downstream | device={device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(project=os.environ.get("WANDB_PROJECT", "Kalavai"),
                        entity=os.environ.get("WANDB_ENTITY"),
                        name="w6-seed42-crosslingual", config=vars(args))
            wandb = _wandb
        except Exception as e:
            print(f"[warn] W&B disabled: {e}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_chunks, held_out_chunks = load_all_data(tokenizer)
    held_out_sets = {d: chunks_to_dataset(held_out_chunks[d]) for d in DOMAINS}
    flores = {d: _flores_sentences(d, args.limit) for d in DOMAINS}
    belebele = {d: _belebele(d, args.limit) for d in DOMAINS}
    for d in DOMAINS:
        flores.setdefault(d, [])
        belebele.setdefault(d, [])

    # ── Train the four seed-42 specialists ───────────────────────────────────
    specialists = {}
    for d in DOMAINS:
        t0 = time.time()
        print(f"\n[train] {d} specialist (seed {SEED}, 2000 steps)...", flush=True)
        spec = fresh_base(device)
        train_specialist(spec, d, train_chunks[d], SEED, device)
        specialists[d] = spec
        print(f"[train] {d} done in {time.time()-t0:.0f}s", flush=True)

    # ── Router with collapse-fix retry ───────────────────────────────────────
    tries, routing_ok, per, dist, moe = 0, False, {}, {}, None
    while tries < args.max_router_tries:
        tries += 1
        torch.manual_seed(1000 + tries)  # vary router initialization each attempt
        moe = FourExpertMoE([specialists[d] for d in DOMAINS], HIDDEN_SIZE).to(device)
        train_router(moe, train_chunks, device)
        routing_ok, per, dist = check_routing(moe, held_out_sets, device)
        print(f"[router] try {tries}/{args.max_router_tries}: ok={routing_ok} {per}", flush=True)
        if routing_ok:
            break

    # ── Downstream battery ───────────────────────────────────────────────────
    base = load_base(device)
    results = {
        "seed": SEED, "router_tries": tries, "routing_ok": routing_ok,
        "router_distribution": dist, "metrics": {},
    }
    results["metrics"]["base"] = evaluate_model(
        "base", base, tokenizer, device, held_out_chunks, flores, belebele, False, args.limit)
    for d in DOMAINS:
        results["metrics"][f"{d}_spec"] = evaluate_model(
            f"{d}_spec", specialists[d], tokenizer, device, held_out_chunks,
            flores, belebele, False, args.limit)
    results["metrics"]["moe"] = evaluate_model(
        "moe", moe, tokenizer, device, held_out_chunks, flores, belebele, True, args.limit)

    print("\n" + "=" * 60, flush=True)
    print(f"SEED-42 DOWNSTREAM (router tries={tries}, routing_ok={routing_ok})", flush=True)
    for d in DOMAINS:
        b = results["metrics"]["base"]["cloze_acc"][d]
        m = results["metrics"]["moe"]["cloze_acc"][d]
        s = results["metrics"][f"{d}_spec"]["cloze_acc"][d]
        print(f"  {d:8s} base={b:.4f} spec={s:.4f} MoE={m:.4f} (MoE-base={m-b:+.4f})", flush=True)

    if wandb is not None:
        try:
            flat = {"router_tries": tries, "routing_ok": int(routing_ok)}
            for mt, md in results["metrics"].items():
                for d in DOMAINS:
                    flat[f"{mt}/{d}/cloze_acc"] = md["cloze_acc"][d]
            wandb.log(flat); wandb.finish()
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "w1_downstream_seed42.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
