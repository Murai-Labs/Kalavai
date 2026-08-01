#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
REB-W7: 7B cross-lingual — the real-task-win gamble to close W1's "beyond loss" gap.

Trains 4 cross-lingual specialists (Tamil/Yoruba/Welsh/Code) on a CAPABLE 7B base
(Qwen2.5-7B), fuses them post-hoc via the KALAVAI router, and evaluates DOWNSTREAM
(Belebele MC-QA — the real task that was near-chance at 410M — plus cloze + FLORES).

Memory: a 7B full fine-tune won't fit an 80GB GPU with standard AdamW (~84GB), so we
use 8-bit AdamW (bitsandbytes) + gradient checkpointing (~40GB). Specialists train one
at a time; the 4-expert MoE (≈61GB weights) runs at batch 1 for routing/eval.

Usage:
  python experiments/w7_qwen7b_crosslingual.py --smoke     # tiny, validates memory+pipeline
  python experiments/w7_qwen7b_crosslingual.py
"""
import argparse
import json
import math
import os
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    load_cc100_texts, load_code_texts, DOMAINS,
    N_SAMPLES_LANGUAGE, N_SAMPLES_CODE, EVAL_BATCHES,
)
from kalavai_eval_utils import (  # noqa: E402
    PackedChunkDataset, chunks_to_dataset, _collate, SEQ_LEN, eval_all_domains,
)
from w2_regression_condition import NExpertMoE  # noqa: E402
from w1_crosslingual_downstream import (  # noqa: E402
    cloze_accuracy, flores_perplexity, belebele_accuracy, _flores_sentences, _belebele,
)

MODEL_ID = os.environ.get("W7_MODEL_ID", "Qwen/Qwen2.5-7B")
HIDDEN_SIZE = int(os.environ.get("W7_HIDDEN", "3584"))  # Qwen2.5-7B hidden dim
RESULTS_DIR = Path("results/rebuttal/w7_qwen7b")
LANG_LOADERS = {
    "tamil":  lambda n: load_cc100_texts("ta", n), "yoruba": lambda n: load_cc100_texts("yo", n),
    "welsh":  lambda n: load_cc100_texts("cy", n), "code":   lambda n: load_code_texts(n),
}


def load_base(device):
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(
        device=device, dtype=torch.bfloat16)
    m.eval()
    return m


def train_specialist_7b(model, domain, train_chunks, seed, device, max_steps, grad_accum=8):
    import bitsandbytes as bnb
    set_seed(seed)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    loader = DataLoader(chunks_to_dataset(train_chunks), batch_size=1, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],
                              lr=2e-5, weight_decay=0.1)
    step, accum, t0 = 0, 0, time.time()
    opt.zero_grad()
    for batch in cycle(loader):
        if step >= max_steps:
            break
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        (out.loss / grad_accum).backward()
        accum += 1
        if accum == grad_accum:
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); accum = 0; step += 1
            if step % 50 == 0 or step == max_steps:
                print(f"    [{domain}] step {step}/{max_steps} loss={out.loss.item():.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
    model.eval(); model.config.use_cache = True
    del opt; torch.cuda.empty_cache()


def train_router_7b(moe, train_chunks_by_domain, device, steps):
    allc = [c for chunks in train_chunks_by_domain.values() for c in chunks]
    loader = DataLoader(chunks_to_dataset(allc), batch_size=1, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    opt = torch.optim.AdamW(moe.router.parameters(), lr=1e-3)
    moe.train()
    it = cycle(loader)
    for step in range(1, steps + 1):
        b = next(it)
        loss, _, _ = moe(b["input_ids"].to(device), labels=b["labels"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    router {step}/{steps} loss={loss.item():.4f}", flush=True)
    moe.eval()


def downstream(tag, model, tok, device, held_chunks, flores, belebele, is_fused, limit):
    print(f"\n=== W7 downstream: {tag} ===", flush=True)
    out = {"cloze_acc": {}, "flores": {}, "belebele": {}}
    for d in DOMAINS:
        ds = chunks_to_dataset(held_chunks[d])
        out["cloze_acc"][d] = round(cloze_accuracy(model, ds, device, is_fused,
                                                   batch_size=1, max_batches=100), 4)
        out["flores"][d] = flores_perplexity(model, tok, flores.get(d, []), device, is_fused)
        out["belebele"][d] = belebele_accuracy(model, tok, belebele.get(d, []), device, is_fused)
        print(f"    {d:8s} cloze={out['cloze_acc'][d]} flores={out['flores'][d]} "
              f"belebele={out['belebele'][d]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    max_steps = 20 if args.smoke else 2000
    router_steps = 30 if args.smoke else 500
    limit = 5 if args.smoke else args.limit
    n_lang = 2000 if args.smoke else N_SAMPLES_LANGUAGE
    n_code = 300 if args.smoke else N_SAMPLES_CODE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"W7 {MODEL_ID} H={HIDDEN_SIZE} seed={args.seed} device={device} "
          f"steps={max_steps} smoke={args.smoke}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # data
    train_chunks, held = {}, {}
    for d in DOMAINS:
        texts = LANG_LOADERS[d](n_lang if d != "code" else n_code)
        ds = PackedChunkDataset(texts, tok, seq_len=SEQ_LEN, max_chars=5000)
        n = len(ds.chunks); a, b = int(n * 0.8), int(n * 0.9)
        train_chunks[d], held[d] = ds.chunks[:a], ds.chunks[b:]
        print(f"  {d:8s} total={n} train={len(train_chunks[d])} held={len(held[d])}", flush=True)
    held_sets = {d: chunks_to_dataset(held[d]) for d in DOMAINS}
    flores = {d: _flores_sentences(d, limit) for d in DOMAINS if d != "code"}
    belebele = {d: _belebele(d, limit) for d in DOMAINS if d != "code"}
    for d in DOMAINS:
        flores.setdefault(d, []); belebele.setdefault(d, [])

    # base eval
    base = load_base(device)
    em_base = eval_all_domains(base, held_sets, device, 1, EVAL_BATCHES)
    ds_base = downstream("base", base, tok, device, held, flores, belebele, False, limit)
    del base; torch.cuda.empty_cache()

    # specialists (one at a time, saved to disk to free VRAM).
    # RESUME: if a specialist checkpoint already exists (durable ckpt dir survives a
    # node death), reload+re-eval instead of retraining — makes retries cheap.
    ckpt_dir = Path(os.environ.get("W7_CKPT_DIR", "w7_ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hf_tok = os.environ.get("HF_TOKEN") or None
    hf_org = os.environ.get("HF_ORG", "mechramc")
    api = None
    if hf_tok:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_tok)

    def hf_repo(d):
        return f"{hf_org}/kalavai-w7-qwen7b-{d}-seed{args.seed}{'-smoke' if args.smoke else ''}"

    em_spec, ds_spec = {}, {}
    for d in DOMAINS:
        cd = ckpt_dir / d
        if (cd / "config.json").exists():
            print(f"\n[resume-local] {d} @7B — {cd}", flush=True)
            s = AutoModelForCausalLM.from_pretrained(cd, trust_remote_code=True).to(
                device=device, dtype=torch.bfloat16).eval()
        elif api is not None and api.repo_exists(hf_repo(d)):
            print(f"\n[resume-hf] {d} @7B from {hf_repo(d)} (survived node death)", flush=True)
            s = AutoModelForCausalLM.from_pretrained(hf_repo(d), token=hf_tok,
                    trust_remote_code=True).to(device=device, dtype=torch.bfloat16).eval()
            s.save_pretrained(cd)  # cache locally for the fusion step
        else:
            print(f"\n[train] {d} @7B (seed {args.seed}, {max_steps} steps, 8bit)", flush=True)
            s = load_base(device)
            train_specialist_7b(s, d, train_chunks[d], args.seed, device, max_steps)
            s.save_pretrained(cd)
            if api is not None:  # durable checkpoint — survives Lambda node death
                try:
                    s.push_to_hub(hf_repo(d), private=True, token=hf_tok)
                    print(f"[hf-push] {d} -> {hf_repo(d)}", flush=True)
                except Exception as e:
                    print(f"[warn] HF push failed for {d}: {e}", flush=True)
        em_spec[d] = eval_all_domains(s, held_sets, device, 1, EVAL_BATCHES)
        ds_spec[d] = downstream(f"{d}_spec", s, tok, device, held, flores, belebele, False, limit)
        del s; torch.cuda.empty_cache()

    # MoE: reload 4 experts (batch 1 to fit ~61GB weights on 80GB)
    specs = [AutoModelForCausalLM.from_pretrained(ckpt_dir / d, trust_remote_code=True).to(
        device=device, dtype=torch.bfloat16).eval() for d in DOMAINS]
    moe = NExpertMoE(specs, HIDDEN_SIZE).to(device)
    train_router_7b(moe, train_chunks, device, router_steps)
    em_moe = eval_all_domains(moe, held_sets, device, 1, EVAL_BATCHES, is_fused=True)
    ds_moe = downstream("moe", moe, tok, device, held, flores, belebele, True, limit)

    # metrics
    def eqv(x):
        return x["equal_weight_avg"]
    best_spec_eq = min(eqv(em_spec[d]) for d in DOMAINS)
    gain = (best_spec_eq - eqv(em_moe)) / best_spec_eq * 100
    results = {
        "model": MODEL_ID, "seed": args.seed, "smoke": args.smoke,
        "eq": {"base": eqv(em_base), "best_spec": best_spec_eq, "moe": eqv(em_moe)},
        "fusion_gain_pct": round(gain, 3),
        "downstream": {"base": ds_base, "moe": ds_moe,
                       **{f"{d}_spec": ds_spec[d] for d in DOMAINS}},
        "perplexity_moe_vs_base": {d: {"base": round(math.exp(em_base[d]), 1),
                                       "moe": round(math.exp(em_moe[d]), 1)} for d in DOMAINS},
    }
    print(f"\n=== W7 gain vs best specialist: {gain:+.2f}% ===", flush=True)
    print("Belebele (base vs MoE):", flush=True)
    for d in ["tamil", "yoruba"]:
        b = ds_base["belebele"][d]; m = ds_moe["belebele"][d]
        ba = b["accuracy"] if b else None
        ma = m["accuracy"] if m else None
        print(f"  {d}: base={ba} moe={ma}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"w7_seed{args.seed}{'_smoke' if args.smoke else ''}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
