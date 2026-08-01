#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI Rebuttal W1 — Cross-Lingual DOWNSTREAM evaluation
=========================================================
Answers NeurIPS AC condition #2: "downstream benchmark accuracy beyond loss."

The paper reports only loss/perplexity. This script evaluates base / each
specialist / best-specialist / MoE-fused on DOWNSTREAM metrics in the
cross-lingual regime (Tamil/Yoruba/Welsh + Code), where the base model is
least competent and fusion gains are largest.

Metrics (all valid for a *monolingual causal LM* — no translation assumed):
  1. cloze_acc    — held-out next-token top-1 accuracy (robust primary)
  2. belebele_acc — multilingual multiple-choice QA accuracy (log-likelihood MC)
  3. flores_ppl   — perplexity / bits-per-byte on FLORES-200 dev (clean held-out corpus)
  4. contin_chrf  — self-continuation chrF (generate 2nd half of a native sentence)

NOTE ON chrF: this is *self-continuation* quality (generate the rest of a native
sentence), NOT machine-translation BLEU/chrF. Pythia specialists are LM
fine-tunes, not translators; MT metrics do not apply.

Checkpoints are pulled from the Hugging Face Hub (no local weights needed):
  mechramc/kalavai-cross-lingual-{tamil,yoruba,welsh,code}-specialist-seed{137,2026}

Usage:
  python experiments/w1_crosslingual_downstream.py --seed 137
  python experiments/w1_crosslingual_downstream.py --seed 137 --no-wandb --limit 200
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the exact architecture / router / data pipeline that produced the paper's
# cross-lingual numbers, so the downstream eval is on the same fused model.
sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    FourExpertMoE, train_router, load_all_data,
    DOMAINS, MODEL_ID, REVISION, HIDDEN_SIZE,
)
from kalavai_eval_utils import chunks_to_dataset, _collate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

HF_ORG = os.environ.get("KALAVAI_HF_ORG", "mechramc")

# FLORES-200 / Belebele language codes for our three non-English domains.
FLORES_CODE = {"tamil": "tam_Taml", "yoruba": "yor_Latn", "welsh": "cym_Latn"}
RESULTS_DIR = Path(os.environ.get("KALAVAI_RESULTS_DIR", "results/rebuttal/w1_crosslingual"))


# ============================================================================
# Model loading (from HF Hub)
# ============================================================================

def load_base(device):
    # Load then cast — avoids the from_pretrained dtype kwarg, which only exists
    # in transformers >=4.49; this path is version-independent (and no torch_dtype).
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
    ).to(device=device, dtype=torch.bfloat16)
    m.eval()
    return m


def load_specialist_from_hf(domain: str, seed: int, device):
    repo = f"{HF_ORG}/kalavai-cross-lingual-{domain}-specialist-seed{seed}"
    print(f"  [{domain}] loading {repo}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code=True,
    ).to(device=device, dtype=torch.bfloat16)
    m.eval()
    return m


# ============================================================================
# Metric 1 — cloze / next-token top-1 accuracy
# ============================================================================

@torch.no_grad()
def cloze_accuracy(model, dataset, device, is_fused=False, batch_size=4, max_batches=200):
    """Fraction of positions where argmax(next-token logit) == true next token."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        if is_fused:
            _, logits, _ = model(ids, labels=ids)
        else:
            logits = model(input_ids=ids).logits
        pred = logits[:, :-1].argmax(-1)
        gold = ids[:, 1:]
        correct += (pred == gold).sum().item()
        total += gold.numel()
    return correct / total if total else 0.0


# ============================================================================
# Metric 3 — FLORES-200 perplexity / bits-per-byte
# ============================================================================

def _flores_sentences(domain: str, limit: int):
    """Return held-out native sentences from FLORES-200 dev, or [] if unavailable."""
    code = FLORES_CODE.get(domain)
    if code is None:
        return []
    from datasets import load_dataset
    for repo, kw in [("facebook/flores", {"name": code}),
                     ("Muennighoff/flores200", {"name": code})]:
        try:
            ds = load_dataset(repo, split="dev", trust_remote_code=True, **kw)
            key = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
            return [r[key] for r in ds][:limit]
        except Exception as e:
            print(f"    FLORES load {repo}/{code} failed: {e}", flush=True)
    return []


@torch.no_grad()
def flores_perplexity(model, tokenizer, sentences, device, is_fused=False):
    """Mean per-token NLL (nats) and bits-per-byte over FLORES dev sentences."""
    if not sentences:
        return None
    tot_nll = tot_tok = tot_bytes = 0
    for s in sentences:
        ids = tokenizer(s, return_tensors="pt")["input_ids"].to(device)
        if ids.size(1) < 2:
            continue
        if is_fused:
            _, logits, _ = model(ids, labels=ids)
        else:
            logits = model(input_ids=ids).logits
        ll = F.log_softmax(logits[0, :-1].float(), dim=-1)
        gold = ids[0, 1:]
        nll = -ll.gather(1, gold.unsqueeze(1)).squeeze(1).sum().item()
        tot_nll += nll
        tot_tok += gold.numel()
        tot_bytes += len(s.encode("utf-8"))
    if not tot_tok:
        return None
    return {
        "ppl": round(math.exp(tot_nll / tot_tok), 3),
        "bits_per_byte": round(tot_nll / math.log(2) / max(tot_bytes, 1), 4),
        "n_sentences": len(sentences),
    }


# ============================================================================
# Metric 2 — Belebele multiple-choice QA accuracy (log-likelihood)
# ============================================================================

def _belebele(domain: str, limit: int):
    code = FLORES_CODE.get(domain)
    if code is None:
        return []
    from datasets import load_dataset
    try:
        ds = load_dataset("facebook/belebele", code, split="test", trust_remote_code=True)
        return list(ds)[:limit]
    except Exception as e:
        print(f"    Belebele {code} unavailable: {e}", flush=True)
        return []


@torch.no_grad()
def _score_continuation(model, tokenizer, ctx, cont, device, is_fused):
    ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"].to(device)
    cont_ids = tokenizer(
        cont, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if cont_ids.size(1) == 0:
        return float("-inf")
    full = torch.cat([ctx_ids, cont_ids], dim=1)
    if is_fused:
        _, logits, _ = model(full, labels=full)
    else:
        logits = model(input_ids=full).logits
    ll = F.log_softmax(logits[0].float(), dim=-1)
    start = ctx_ids.size(1)
    score = 0.0
    for i in range(cont_ids.size(1)):
        score += ll[start + i - 1, cont_ids[0, i]].item()
    return score / cont_ids.size(1)  # length-normalized


@torch.no_grad()
def belebele_accuracy(model, tokenizer, items, device, is_fused=False):
    if not items:
        return None
    correct = 0
    for it in items:
        ctx = f"{it['flores_passage']}\nQuestion: {it['question']}\nAnswer:"
        cands = [it["mc_answer1"], it["mc_answer2"], it["mc_answer3"], it["mc_answer4"]]
        scores = [_score_continuation(model, tokenizer, ctx, " " + c, device, is_fused)
                  for c in cands]
        if scores.index(max(scores)) == int(it["correct_answer_num"]) - 1:
            correct += 1
    return {"accuracy": round(correct / len(items), 4), "n": len(items), "chance": 0.25}


# ============================================================================
# Metric 4 — self-continuation chrF
# ============================================================================

@torch.no_grad()
def _greedy_generate(model, input_ids, max_new_tokens, is_fused):
    ids = input_ids
    for _ in range(max_new_tokens):
        if is_fused:
            _, logits, _ = model(ids, labels=ids)
        else:
            logits = model(input_ids=ids).logits
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


CHRF_MAX_NEW_TOKENS = 24  # cap generation length — token-by-token gen is the costly metric


@torch.no_grad()
def continuation_chrf(model, tokenizer, sentences, device, is_fused=False, n=30):
    try:
        from sacrebleu.metrics import CHRF
    except Exception as e:
        print(f"    sacrebleu unavailable, skipping chrF: {e}", flush=True)
        return None
    if not sentences:
        return None
    chrf = CHRF()
    hyps, refs = [], []
    for s in sentences[:n]:
        ids = tokenizer(s, return_tensors="pt")["input_ids"].to(device)
        if ids.size(1) < 6:
            continue
        half = ids.size(1) // 2
        prompt = ids[:, :half]
        n_new = min(ids.size(1) - half, CHRF_MAX_NEW_TOKENS)
        gold = tokenizer.decode(ids[0, half:half + n_new], skip_special_tokens=True)
        out = _greedy_generate(model, prompt, max_new_tokens=n_new, is_fused=is_fused)
        hyp = tokenizer.decode(out[0, half:], skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(gold)
    if not hyps:
        return None
    return {"chrf": round(chrf.corpus_score(hyps, [refs]).score, 3), "n": len(hyps)}


# ============================================================================
# Per-model downstream battery
# ============================================================================

def evaluate_model(tag, model, tokenizer, device, held_out, flores, belebele, is_fused, limit):
    print(f"\n=== downstream eval: {tag} ===", flush=True)
    out = {"cloze_acc": {}, "flores": {}, "belebele": {}, "contin_chrf": {}}
    for domain in DOMAINS:
        t0 = time.time()
        ds = chunks_to_dataset(held_out[domain])
        fl = flores.get(domain, [])
        bel = belebele.get(domain, [])
        out["cloze_acc"][domain] = round(cloze_accuracy(model, ds, device, is_fused), 4)
        out["flores"][domain] = flores_perplexity(model, tokenizer, fl, device, is_fused)
        out["belebele"][domain] = belebele_accuracy(model, tokenizer, bel, device, is_fused)
        out["contin_chrf"][domain] = continuation_chrf(
            model, tokenizer, fl, device, is_fused, n=min(limit, 30))
        print(f"    {domain:8s} cloze={out['cloze_acc'][domain]:.4f} "
              f"flores={out['flores'][domain]} belebele={out['belebele'][domain]} "
              f"chrf={out['contin_chrf'][domain]} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[137, 2026],
                    help="Which cross-lingual seed's specialists to fuse (HF has 137/2026).")
    ap.add_argument("--limit", type=int, default=100,
                    help="Sentence cap per FLORES/chrF/Belebele metric.")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"W1 cross-lingual downstream | seed={args.seed} | device={device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # W&B is optional telemetry — it must NEVER crash the compute job.
    # Auth comes from ~/.netrc (written by the node bootstrap), which bypasses the
    # SDK env-var key validator that rejects newer wandb_v1_ key formats.
    wandb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _wandb.init(project=os.environ.get("WANDB_PROJECT", "kalavai-rebuttal"),
                        entity=os.environ.get("WANDB_ENTITY"),
                        name=f"w1-crosslingual-downstream-seed{args.seed}",
                        config=vars(args))
            wandb = _wandb
        except Exception as e:
            print(f"[warn] W&B disabled (init failed): {e}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data: reuse the exact loader (train chunks for router, held-out for cloze).
    train_chunks, held_out_chunks = load_all_data(tokenizer)

    # Downstream corpora (native FLORES sentences + Belebele items).
    flores = {d: _flores_sentences(d, args.limit) for d in DOMAINS if d != "code"}
    belebele = {d: _belebele(d, args.limit) for d in DOMAINS if d != "code"}
    for d in DOMAINS:
        flores.setdefault(d, [])
        belebele.setdefault(d, [])

    # Load models.
    base = load_base(device)
    specialists = {d: load_specialist_from_hf(d, args.seed, device) for d in DOMAINS}

    results = {"seed": args.seed, "hf_org": HF_ORG, "metrics": {}}

    results["metrics"]["base"] = evaluate_model(
        "base", base, tokenizer, device, held_out_chunks, flores, belebele, False, args.limit)
    for d in DOMAINS:
        results["metrics"][f"{d}_spec"] = evaluate_model(
            f"{d}_spec", specialists[d], tokenizer, device, held_out_chunks,
            flores, belebele, False, args.limit)

    # MoE — build + train router (same 500-step recipe as the paper).
    print("\n[moe] building FourExpertMoE + training router...", flush=True)
    moe = FourExpertMoE([specialists[d] for d in DOMAINS], HIDDEN_SIZE).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()
    results["metrics"]["moe"] = evaluate_model(
        "moe", moe, tokenizer, device, held_out_chunks, flores, belebele, True, args.limit)

    # Headline: per-domain MoE vs best-specialist on each downstream metric.
    print("\n" + "=" * 60, flush=True)
    print("DOWNSTREAM SUMMARY (cloze accuracy — higher is better)", flush=True)
    for domain in DOMAINS:
        base_a = results["metrics"]["base"]["cloze_acc"][domain]
        moe_a = results["metrics"]["moe"]["cloze_acc"][domain]
        spec_a = results["metrics"][f"{domain}_spec"]["cloze_acc"][domain]
        print(f"  {domain:8s}  base={base_a:.4f}  spec={spec_a:.4f}  MoE={moe_a:.4f}  "
              f"(MoE-base={moe_a-base_a:+.4f})", flush=True)

    if wandb is not None:
        try:
            flat = {}
            for model_tag, md in results["metrics"].items():
                for domain in DOMAINS:
                    flat[f"{model_tag}/{domain}/cloze_acc"] = md["cloze_acc"][domain]
            wandb.log(flat)
            wandb.finish()
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"w1_downstream_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
