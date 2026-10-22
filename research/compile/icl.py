"""The in-context transformer (docs/compiled-reasoner.md, "Learners compared", 4), its training loop and checkpoints.

    [glyph(p1)+outcome(p1), ..., glyph(pk)+outcome(pk), glyph(query)+QUERY] -> outcome(query)

Each glyph is projected dim -> d_model as one soft token; the precedent's outcome is a learned embedding added to its
token (the query gets a third, "unknown" embedding); a learned embedding of the precedent's rank (nearest first, as
Space.query returns its receipts) is added too. 4 layers, 4 heads, d_model 128, pre-norm, k=16 precedents plus the
query. The variant `sim` also adds the kernel's similarity of each precedent to the query (a scalar Space already
returns with every receipt) through a learned linear map.

Training is once, across the pool of synthetic worlds: an episode picks a world, a query among its training cases, a
store (the world's other training cases, or a random subset of them of log-uniform size, so it also learns from a
near-empty store), retrieves the store's k nearest to the query by glyph sim (what Space.query ranks by), and asks
for the query's outcome under the rule (noisy worlds: the stored outcomes carry the noise, the target does not).
Which outcome is 0 and which is 1 is swapped at random per episode: outcomes carry no meaning across worlds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from worlds import FAMILIES, build_pool

K = 16
QUERY = 2  # the outcome slot of the query (and of padding)
CACHE = Path("/tmp/ada-compile-cache")


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class ICL(nn.Module):
    def __init__(self, dim: int, d_model: int = 128, layers: int = 4, heads: int = 4, k: int = K, use_sim: bool = False):
        super().__init__()
        self.dim, self.k, self.use_sim = dim, k, use_sim
        self.proj = nn.Linear(dim, d_model)
        self.outcome = nn.Embedding(3, d_model)
        self.rank = nn.Embedding(k + 1, d_model)
        self.sim = nn.Linear(1, d_model) if use_sim else None
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, glyphs: torch.Tensor, labels: torch.Tensor, sims: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """glyphs (B, k+1, dim) float +-1, the query last; labels (B, k+1) in {0, 1, QUERY}; sims (B, k+1) the glyph
        sim of each to the query; pad (B, k+1) True where there is no precedent. Returns logits (B, 2)."""
        B, T, _ = glyphs.shape
        x = self.proj(glyphs) + self.outcome(labels) + self.rank(torch.arange(T, device=glyphs.device))[None]
        if self.sim is not None:
            x = x + self.sim(sims[..., None])
        h = self.encoder(x, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, -1]))


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Sampler:
    """Episodes from a pool of worlds (worlds.build_pool), glyphs held on the device."""

    def __init__(self, pool: list[dict], dev: torch.device, seed: int, k: int = K, split: str = "train"):
        self.pool, self.k, self.dev = pool, k, dev
        self.rng = np.random.default_rng(seed)
        self.offsets = np.cumsum([0] + [len(p["glyphs"]) for p in pool])
        self.table = torch.from_numpy(np.concatenate([p["glyphs"] for p in pool])).to(dev)  # int8
        self.dim = self.table.shape[1]

    def episodes(self, worlds, queries, stores) -> dict[str, torch.Tensor]:
        """Given per episode a world, a query index and a store (indices into the world), retrieve and pack."""
        B, k = len(worlds), self.k
        idx = np.zeros((B, k + 1), dtype=np.int64)
        labels = np.full((B, k + 1), QUERY, dtype=np.int64)
        sims = np.zeros((B, k + 1), dtype=np.float32)
        pad = np.ones((B, k + 1), dtype=bool)
        target = np.zeros(B, dtype=np.int64)
        fam = np.zeros(B, dtype=np.int64)
        for b, (w, q, store) in enumerate(zip(worlds, queries, stores)):
            p = self.pool[w]
            row = p["sims"][q, store].astype(np.int32)
            n = min(k, len(store))
            if n:
                top = np.argpartition(-row, n - 1)[:n] if len(store) > n else np.arange(len(store))
                top = top[np.argsort(-row[top], kind="stable")]
                ctx = store[top]
                idx[b, :n] = self.offsets[w] + ctx
                labels[b, :n] = p["stored"][ctx]
                sims[b, :n] = row[top] / self.dim
                pad[b, :n] = False
            idx[b, k] = self.offsets[w] + q
            sims[b, k] = 1.0
            pad[b, k] = False
            target[b] = p["clean"][q]
            fam[b] = p["family"]
        # precedents at 0..n-1, nearest first (the rank embedding reads their rank), padding after, the query at k
        return self._pack(idx, labels, sims, pad, target, fam)

    def _pack(self, idx, labels, sims, pad, target, fam) -> dict[str, torch.Tensor]:
        dev = self.dev
        it = torch.from_numpy(idx).to(dev)
        return {
            "glyphs": self.table[it].float(),
            "labels": torch.from_numpy(labels).to(dev),
            "sims": torch.from_numpy(sims).to(dev),
            "pad": torch.from_numpy(pad).to(dev),
            "target": torch.from_numpy(target).to(dev),
            "family": fam,
        }

    def train_batch(self, B: int) -> dict[str, torch.Tensor]:
        rng = self.rng
        worlds, queries, stores = [], [], []
        for _ in range(B):
            w = int(rng.integers(len(self.pool)))
            tr = self.pool[w]["train"]
            q = int(rng.choice(tr))
            rest = tr[tr != q]
            if rng.random() < 0.5:
                store = rest
            else:
                m = int(np.exp(rng.uniform(0, np.log(len(rest)))))
                store = rng.choice(rest, size=max(1, m), replace=False)
            worlds.append(w), queries.append(q), stores.append(store)
        batch = self.episodes(worlds, queries, stores)
        swap = torch.from_numpy(rng.random(B) < 0.5).to(self.dev)
        lab = batch["labels"]
        batch["labels"] = torch.where(swap[:, None] & (lab != QUERY), 1 - lab, lab)
        batch["target"] = torch.where(swap, 1 - batch["target"], batch["target"])
        return batch

    def fixed_eval(self, per_world: int = 20, seed: int = 0) -> list[dict[str, torch.Tensor]]:
        """Validation episodes: each world's own test cases against its whole training store."""
        rng = np.random.default_rng(seed)
        worlds, queries, stores = [], [], []
        for w, p in enumerate(self.pool):
            for q in rng.permutation(p["test"])[:per_world]:
                worlds.append(w), queries.append(int(q)), stores.append(p["train"])
        out = []
        for s in range(0, len(worlds), 512):
            out.append(self.episodes(worlds[s:s + 512], queries[s:s + 512], stores[s:s + 512]))
        return out


@torch.no_grad()
def evaluate(model: ICL, batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    right = np.zeros(len(FAMILIES))
    seen = np.zeros(len(FAMILIES))
    for b in batches:
        pred = model(b["glyphs"], b["labels"], b["sims"], b["pad"]).argmax(-1)
        ok = (pred == b["target"]).cpu().numpy()
        np.add.at(right, b["family"], ok)
        np.add.at(seen, b["family"], 1)
    model.train()
    out = {f: float(right[i] / seen[i]) for i, f in enumerate(FAMILIES) if seen[i]}
    out["all"] = float(right.sum() / seen.sum())
    return out


def ckpt_path(dim: int, use_sim: bool) -> Path:
    return CACHE / f"icl-{dim}-{'sim' if use_sim else 'base'}.pt"


def train(dim: int, use_sim: bool, steps: int, batch: int = 256, lr: float = 1e-3, warmup: int = 500,
          pool_size: int = 3000, val_size: int = 120, seed: int = 0, log_every: int = 500) -> dict:
    torch.manual_seed(seed)
    dev = device()
    t0 = time.time()
    pool = build_pool("train", pool_size, dim, CACHE)
    val = build_pool("val", val_size, dim, CACHE)
    t_pool = time.time() - t0
    sampler = Sampler(pool, dev, seed=seed)
    vsampler = Sampler(val, dev, seed=seed + 1)
    vbatches = vsampler.fixed_eval()
    # the training worlds' own reserved test cases (never a query nor in a store during training): seen worlds
    seen_batches = Sampler(pool[:600], dev, seed=seed + 2).fixed_eval(per_world=10)
    model = ICL(dim, use_sim=use_sim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    log = []
    t1 = time.time()
    running, counted = 0.0, 0
    for step in range(1, steps + 1):
        b = sampler.train_batch(batch)
        logits = model(b["glyphs"], b["labels"], b["sims"], b["pad"])
        loss = nn.functional.cross_entropy(logits, b["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:  # a sampled training loss (reading it syncs the device)
            running += loss.item()
            counted += 1
        if step % log_every == 0 or step == steps:
            v = evaluate(model, vbatches)
            entry = {"step": step, "loss": running / max(1, counted), "val": v, "secs": time.time() - t1}
            running, counted = 0.0, 0
            log.append(entry)
            print(json.dumps({"dim": dim, "sim": use_sim, **entry}), flush=True)
    seen = evaluate(model, seen_batches)
    out = {"dim": dim, "use_sim": use_sim, "steps": steps, "batch": batch, "lr": lr, "params": params(model),
           "pool_worlds": pool_size, "pool_cases": int(sum(len(p["glyphs"]) for p in pool)),
           "pool_secs": t_pool, "train_secs": time.time() - t1, "log": log, "seen_worlds": seen,
           "device": str(dev)}
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": out}, ckpt_path(dim, use_sim))
    return out


def load(dim: int, use_sim: bool, dev: torch.device | None = None) -> tuple[ICL, dict]:
    blob = torch.load(ckpt_path(dim, use_sim), map_location="cpu", weights_only=False)
    model = ICL(dim, use_sim=use_sim)
    model.load_state_dict(blob["state"])
    model.eval()
    return model.to(dev or torch.device("cpu")), blob["meta"]


class IclLearner:
    """The trained transformer as a learner: it stores precedents in a kernel Space (record, no training) and at
    query time reads the Space's k nearest (Space.query's receipts) in context. No gradient step ever runs here."""

    def __init__(self, world, dim: int, use_sim: bool, model: ICL | None = None):
        from baselines import Knn

        self.w, self.dim = world, dim
        self.name = "icl_sim" if use_sim else "icl"
        self.model = model if model is not None else load(dim, use_sim)[0]
        self.store = Knn(world, dim)
        self.label_of: dict[int, int] = {}

    def add(self, idx, labels) -> None:
        self.store.add(idx, labels)
        self.label_of.update({int(i): int(y) for i, y in zip(idx, labels)})

    def refit(self) -> None:
        pass  # nothing to fit: the store is the knowledge

    @torch.no_grad()
    def predict(self, idx) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx)
        B, k = len(idx), K
        g = self.w.glyphs
        glyphs = np.zeros((B, k + 1, g.shape[1]), dtype=np.float32)
        labels = np.full((B, k + 1), QUERY, dtype=np.int64)
        sims = np.zeros((B, k + 1), dtype=np.float32)
        pad = np.ones((B, k + 1), dtype=bool)
        for b, i in enumerate(idx):
            ctx = self.store.receipts(int(i), k) if self.label_of else []
            n = len(ctx)
            q = g[i].astype(np.float32)
            if n:
                c = g[ctx].astype(np.float32)
                glyphs[b, :n] = c
                labels[b, :n] = [self.label_of[j] for j in ctx]
                sims[b, :n] = c @ q / len(q)
                pad[b, :n] = False
            glyphs[b, k] = q
            sims[b, k] = 1.0
            pad[b, k] = False
        logits = self.model(torch.from_numpy(glyphs), torch.from_numpy(labels), torch.from_numpy(sims),
                            torch.from_numpy(pad))
        p1 = torch.softmax(logits, -1)[:, 1].numpy()
        return (p1 > 0.5).astype(int), p1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool", type=int, default=3000)
    a = ap.parse_args()
    meta = train(a.dim, a.sim, a.steps, batch=a.batch, pool_size=a.pool)
    print(json.dumps({k: v for k, v in meta.items() if k != "log"}))
