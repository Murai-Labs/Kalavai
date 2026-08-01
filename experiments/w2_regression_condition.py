#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
REB-W2: grow the divergence-gain predictor from n=8 to ~25 with a held-out
validation set. This script runs ONE condition (an arbitrary subset of the
cross-lingual domains) and emits a single (mean_divergence, fusion_gain) point,
using the exact protocol behind the paper's regression (2000-step specialists,
freeze=0, 500-step router, per-domain equal-weight eval).

Different subsets span a wide divergence range (code ~0.4% .. yoruba ~45%),
giving many fresh points for held-out validation of gain = a + b*divergence.

Usage:
  python experiments/w2_regression_condition.py --domains yoruba,code --seed 42
  python experiments/w2_regression_condition.py --domains tamil,yoruba,welsh --seed 42
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from kalavai_phase2_exp1 import (  # noqa: E402
    load_cc100_texts, load_code_texts, train_specialist,
    MODEL_ID, REVISION, HIDDEN_SIZE, N_SAMPLES_LANGUAGE, N_SAMPLES_CODE,
    ROUTER_STEPS, ROUTER_LR, ROUTER_BATCH, EVAL_BATCH_SIZE, EVAL_BATCHES,
)
from kalavai_eval_utils import (  # noqa: E402
    PackedChunkDataset, chunks_to_dataset, eval_all_domains, _collate, SEQ_LEN,
)
from torch.utils.data import DataLoader  # noqa: E402
from itertools import cycle  # noqa: E402

RESULTS_DIR = Path("results/rebuttal/w2_regression")
LANG_LOADERS = {  # domain -> loader thunk
    "tamil":  lambda: load_cc100_texts("ta", N_SAMPLES_LANGUAGE),
    "yoruba": lambda: load_cc100_texts("yo", N_SAMPLES_LANGUAGE),
    "welsh":  lambda: load_cc100_texts("cy", N_SAMPLES_LANGUAGE),
    "code":   lambda: load_code_texts(N_SAMPLES_CODE),
}


class NExpertMoE(nn.Module):
    """Sequence-level MoE over N frozen specialists; only the router trains."""
    def __init__(self, specs, hidden_size):
        super().__init__()
        self.experts = nn.ModuleList(specs)
        for p in self.parameters():
            p.requires_grad_(False)
        self.n = len(specs)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False), nn.ReLU(),
            nn.Linear(256, self.n, bias=False),
        )

    def _run(self, model, ids):
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(1).float()

    def forward(self, input_ids, labels=None):
        logits, h_sum = [], 0
        for m in self.experts:
            lg, h = self._run(m, input_ids)
            logits.append(lg)
            h_sum = h_sum + h
        gates = torch.softmax(self.router(h_sum / self.n), dim=-1)
        fused = sum(gates[:, i:i + 1, None] * logits[i] for i in range(self.n))
        loss = None
        if labels is not None:
            sl = fused[:, :-1].contiguous()
            slb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), slb.view(-1))
        return loss, fused, gates


def fresh_base(device):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
    ).to(device=device, dtype=torch.bfloat16)


def train_router(moe, train_chunks_by_domain, device):
    allc = [c for chunks in train_chunks_by_domain.values() for c in chunks]
    loader = DataLoader(chunks_to_dataset(allc), batch_size=ROUTER_BATCH,
                        shuffle=True, drop_last=True, collate_fn=_collate)
    opt = torch.optim.AdamW(moe.router.parameters(), lr=ROUTER_LR)
    moe.train()
    it = cycle(loader)
    for step in range(1, ROUTER_STEPS + 1):
        b = next(it)
        loss, _, _ = moe(b["input_ids"].to(device), labels=b["labels"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    router {step}/{ROUTER_STEPS} loss={loss.item():.4f}", flush=True)
    moe.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", required=True, help="comma list from tamil,yoruba,welsh,code")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    domains = [d.strip() for d in args.domains.split(",")]
    assert all(d in LANG_LOADERS for d in domains) and len(domains) >= 2, f"bad domains {domains}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"W2 condition domains={domains} seed={args.seed} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── data ──────────────────────────────────────────────────────────────────
    train_chunks, held = {}, {}
    for d in domains:
        texts = LANG_LOADERS[d]()
        ds = PackedChunkDataset(texts, tok, seq_len=SEQ_LEN, max_chars=5000)
        n = len(ds.chunks); a, b = int(n * 0.8), int(n * 0.9)
        train_chunks[d], held[d] = ds.chunks[:a], ds.chunks[b:]
        print(f"  {d:8s} total={n} train={len(train_chunks[d])} held={len(held[d])}", flush=True)
    held_sets = {d: chunks_to_dataset(held[d]) for d in domains}

    # ── base + specialists ────────────────────────────────────────────────────
    base = fresh_base(device)
    em_base = eval_all_domains(base, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base; torch.cuda.empty_cache()

    specs, em_spec = {}, {}
    for d in domains:
        print(f"\n[train] {d} (seed {args.seed})", flush=True)
        s = fresh_base(device)
        train_specialist(s, d, train_chunks[d], args.seed, device)
        specs[d] = s
        em_spec[d] = eval_all_domains(s, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)

    # ── MoE (router re-init retry avoids collapse artifacts; matches W6) ───────
    max_tries, tries, routing_ok, moe = 5, 0, False, None
    while tries < max_tries:
        tries += 1
        torch.manual_seed(1000 + tries)  # vary router init each attempt
        moe = NExpertMoE([specs[d] for d in domains], HIDDEN_SIZE).to(device)
        train_router(moe, train_chunks, device)
        moe.eval()
        ok = {}
        with torch.no_grad():
            for i, d in enumerate(domains):
                ld = DataLoader(held_sets[d], batch_size=EVAL_BATCH_SIZE, shuffle=False,
                                drop_last=True, collate_fn=_collate)
                gsum = [0.0] * moe.n
                for bi, bb in enumerate(ld):
                    if bi >= 10:
                        break
                    _, _, g = moe(bb["input_ids"].to(device))
                    for j in range(moe.n):
                        gsum[j] += g[:, j].mean().item()
                ok[d] = (max(range(moe.n), key=lambda j: gsum[j]) == i)
        routing_ok = all(ok.values())
        print(f"[router] try {tries}/{max_tries}: ok={routing_ok} {ok}", flush=True)
        if routing_ok:
            break
    em_moe = eval_all_domains(moe, held_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)

    # ── divergence + gain ─────────────────────────────────────────────────────
    divs = []
    for d in domains:
        bd, sd = em_base[d], em_spec[d][d]
        divs.append((bd - sd) / bd * 100 if bd > 0 else 0.0)
    mean_div = statistics.mean(divs)
    best_spec_eq = min(em_spec[d]["equal_weight_avg"] for d in domains)
    moe_eq = em_moe["equal_weight_avg"]
    gain = (best_spec_eq - moe_eq) / best_spec_eq * 100

    point = {
        "domains": domains, "seed": args.seed,
        "mean_divergence": round(mean_div, 3), "fusion_gain": round(gain, 4),
        "per_domain_divergence": {d: round(divs[i], 3) for i, d in enumerate(domains)},
        "best_spec_eq": round(best_spec_eq, 6), "moe_eq": round(moe_eq, 6),
        "router_tries": tries, "routing_ok": routing_ok,
    }
    print(f"\n=== POINT: div={mean_div:.2f}%  gain={gain:+.2f}%  domains={domains} ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "-".join(domains) + f"_seed{args.seed}"
    json.dump(point, open(RESULTS_DIR / f"cond_{tag}.json", "w"), indent=2)
    print(f"Saved: {RESULTS_DIR / f'cond_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
