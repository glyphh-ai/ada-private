"""Every test in docs/compiled-reasoner.md, written to research/compile/REPORT.md.

  accuracy     30 unseen worlds per family (200-2000 cases): held-out cases and held-out combinations, every learner
  online       6 worlds per family: an empty store, precedents added one at a time, accuracy at 1, 2, 4 ... 512
               (kernel learners and the transformer never train; logreg and the MLP are refit at every point, as a
               reference; ada_learn runs the kernel's learn() + calibrate() at every point, as Model.refresh does)
  flip         6 worlds per family of 1000 cases: 100 precedents under one rule, then the rule is redrawn (same
               family, agreeing on 25-75% of cases) and up to 300 precedents arrive under the new one; at each point,
               the share of test cases where the rules disagree that a learner answers by the new rule
  calibration  reliability of P(predicted outcome) on the held-out cases (10-point bins, ECE)
  latency      encode + retrieve + forward per query, batch 1, store of 1,000 precedents

at dim 4096 and 1024, for every learner in baselines.py and both transformer variants (icl.py). Per-world results are
cached under /tmp/ada-compile-cache, so a rerun only computes what is missing.

Usage: python run.py [stage ...]   (stages: acc online flip latency report; default all)
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/ada-compile-cache/results")
DIMS = (4096, 1024)
ICLS = ("icl", "icl_sim")
LEARNERS = ("knn", "ada", "ada_learn", "logreg", "mlp") + ICLS
N_ACC, N_ONLINE, N_FLIP = 30, 6, 6  # worlds per family
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PRE_FLIP = 100
POST = (0, 1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 300)
TEST_CAP_ONLINE = 100

_MODELS: dict = {}


def _learner(name: str, w, dim: int):
    if name in ICLS:
        import torch

        torch.set_num_threads(1)
        from icl import IclLearner, load

        key = (dim, name)
        if key not in _MODELS:
            _MODELS[key] = load(dim, name == "icl_sim")[0]
        return IclLearner(w, dim, name == "icl_sim", model=_MODELS[key])
    from baselines import make

    return make(name, w, dim)


def _cached(kind: str, dim: int, i: int, name: str, fn):
    path = CACHE / kind / f"{dim}-{i}-{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# ── accuracy and calibration ────────────────────────────────────────


def job_acc(args):
    dim, i, name = args

    def go():
        from worlds import FAMILIES, make_world, world_seed

        fam = FAMILIES[i % len(FAMILIES)]
        w = make_world(world_seed("eval", i), fam, dim=dim)
        l = _learner(name, w, dim)
        t = time.time()
        l.add(w.train, w.stored[w.train])
        l.refit()
        fit = time.time() - t
        out = {"world": i, "family": fam, "base": w.base, "n_train": int(len(w.train)), "fit_secs": fit}
        for split, idx in (("test", w.test), ("combo", w.combo)):
            if len(idx):
                t = time.time()
                pred, p1 = l.predict(idx)
                out[split] = {"correct": (pred == w.clean[idx]).astype(int).tolist(), "p1": np.round(p1, 5).tolist(),
                              "y": w.clean[idx].tolist(), "secs": time.time() - t}
        if hasattr(l, "insufficient"):
            out["insufficient"] = l.insufficient / max(1, l.asked)
        if name == "ada_learn":
            out["pairs"] = l.pairs()
            out["reads"] = list(w.reads)
        return out

    return _cached("acc", dim, i, name, go)


# ── online learning: an empty store, one precedent at a time ────────


def job_online(args):
    dim, i, name = args

    def go():
        from worlds import FAMILIES, make_world, world_seed

        fam = FAMILIES[i % len(FAMILIES)]
        w = make_world(world_seed("eval", 20000 + i), fam, dim=dim, n_range=(800, 2000))
        l = _learner(name, w, dim)
        test = w.test[:TEST_CAP_ONLINE]
        curve, have = {}, 0
        for n in CHECKPOINTS:
            if n > len(w.train):
                break
            new = w.train[have:n]
            l.add(new, w.stored[new])
            have = n
            l.refit()
            pred, _ = l.predict(test)
            curve[n] = float((pred == w.clean[test]).mean())
        return {"world": i, "family": fam, "curve": curve}

    return _cached("online", dim, i, name, go)


# ── flip: the rule changes halfway ─────────────────────────────────


def flip_world(dim: int, i: int):
    from worlds import FAMILIES, NOISE, make_world, world_seed

    fam = FAMILIES[i % len(FAMILIES)]
    w = make_world(world_seed("eval", 30000 + i), fam, dim=dim, n=1000)
    r = np.random.default_rng(world_seed("eval", 30000 + i) + 7)
    best = None
    for _ in range(100):
        rule2 = w.redraw(r)
        y2 = np.array([rule2(c) for c in w.data])
        agree = float((y2 == w.clean).mean())
        if 0.25 <= agree <= 0.75:
            best = y2
            break
        if best is None or abs(agree - 0.5) < abs(float((best == w.clean).mean()) - 0.5):
            best = y2
    y2 = best
    s2 = y2.copy()
    if fam == "noisy":
        flip = r.random(len(s2)) < NOISE
        s2[flip] = 1 - s2[flip]
    return fam, w, y2, s2


def job_flip(args):
    dim, i, name = args

    def go():
        fam, w, y2, s2 = flip_world(dim, i)
        l = _learner(name, w, dim)
        test = w.test
        differ = test[w.clean[test] != y2[test]]
        pre = w.train[:PRE_FLIP]
        l.add(pre, w.stored[pre])
        l.refit()
        pred, _ = l.predict(test)
        before = float((pred == w.clean[test]).mean())
        post = w.train[PRE_FLIP:]
        follow, acc_new, have = {}, {}, 0
        for n in POST:
            if n > len(post):
                break
            if n > have:
                new = post[have:n]
                l.add(new, s2[new])
                have = n
                l.refit()
            pd = dict(zip(test.tolist(), l.predict(test)[0].tolist()))
            follow[n] = float(np.mean([pd[j] == y2[j] for j in differ])) if len(differ) else float("nan")
            acc_new[n] = float(np.mean([pd[j] == y2[j] for j in test]))
        return {"world": i, "family": fam, "before": before, "follow": follow, "acc_new": acc_new,
                "agree": float((w.clean == y2).mean()), "n_differ": int(len(differ))}

    return _cached("flip", dim, i, name, go)


# ── latency ─────────────────────────────────────────────────────────


def latency(dim: int) -> dict:
    import torch

    from baselines import Ada, Knn
    from icl import K, load
    from worlds import kernel, make_world, world_seed

    torch.set_num_threads(4)
    w = make_world(world_seed("eval", 40000), "xor", dim=dim, n=1400)
    store = w.train[:1000]
    knn = Knn(w, dim)
    knn.add(store, w.stored[store])
    knn.space.k = K
    ada = Ada(w, dim, learn=True)
    ada.add(store, w.stored[store])
    ada.refit()
    spec = w.spec
    queries = w.test[:100]
    out: dict = {"store": len(store)}
    t_enc, t_ret, t_cpu, t_mps, t_ada, t_knn = [], [], [], [], [], []
    model_cpu = load(dim, False)[0]
    mps = torch.backends.mps.is_available()
    model_mps = load(dim, False, torch.device("mps"))[0] if mps else None
    for i in queries:
        d = w.data[i]
        t = time.perf_counter()
        s = spec.situation(d)
        g = kernel.encode(s, dim=dim)
        t_enc.append(time.perf_counter() - t)
        t = time.perf_counter()
        a = knn.space.query(s)  # encodes again inside, as a caller handing it a situation would
        ctx = [int(r[0]) for r in a.receipts]
        t_ret.append(time.perf_counter() - t)
        t_knn.append(t_enc[-1] + t_ret[-1])
        glyphs = np.zeros((1, K + 1, dim), dtype=np.float32)
        labels = np.full((1, K + 1), 2, dtype=np.int64)
        sims = np.zeros((1, K + 1), dtype=np.float32)
        pad = np.zeros((1, K + 1), dtype=bool)
        t = time.perf_counter()
        c = w.glyphs[ctx].astype(np.float32)
        glyphs[0, : len(ctx)] = c
        glyphs[0, K] = g
        labels[0, : len(ctx)] = w.stored[ctx]
        sims[0, : len(ctx)] = c @ g.astype(np.float32) / dim
        with torch.no_grad():
            model_cpu(torch.from_numpy(glyphs), torch.from_numpy(labels), torch.from_numpy(sims), torch.from_numpy(pad))
        t_cpu.append(time.perf_counter() - t)
        if model_mps is not None:
            t = time.perf_counter()
            dev = torch.device("mps")
            with torch.no_grad():
                y = model_mps(torch.from_numpy(glyphs).to(dev), torch.from_numpy(labels).to(dev),
                              torch.from_numpy(sims).to(dev), torch.from_numpy(pad).to(dev))
                y.cpu()
            t_mps.append(time.perf_counter() - t)
        t = time.perf_counter()
        ada.model.query(d)
        t_ada.append(time.perf_counter() - t)
    ms = lambda xs: round(1000 * float(np.median(xs[5:])), 3)  # noqa: E731  (the first few warm caches)
    out.update({"encode": ms(t_enc), "retrieve": ms(t_ret), "forward_cpu": ms(t_cpu),
                "forward_mps": ms(t_mps) if t_mps else None, "knn_total": ms(t_knn), "ada_learn_query": ms(t_ada)})
    out["icl_total_cpu"] = round(out["encode"] + out["retrieve"] + out["forward_cpu"], 3)
    if t_mps:
        out["icl_total_mps"] = round(out["encode"] + out["retrieve"] + out["forward_mps"], 3)
    return out


# ── stages ──────────────────────────────────────────────────────────


def _map(fn, jobs, procs):
    jobs = list(jobs)
    t = time.time()
    with get_context("spawn").Pool(procs) as ex:
        for n, _ in enumerate(ex.imap_unordered(fn, jobs, chunksize=1), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"  {fn.__name__}: {n}/{len(jobs)} ({time.time() - t:.0f}s)", flush=True)


def available(names) -> list[str]:
    from icl import ckpt_path

    return [n for n in names if n not in ICLS or all(ckpt_path(d, n == "icl_sim").exists() for d in DIMS)]


def run_stage(stage: str, learners, procs: int) -> None:
    from worlds import FAMILIES

    nf = len(FAMILIES)
    if stage == "acc":
        _map(job_acc, [(d, i, n) for d in DIMS for n in learners for i in range(N_ACC * nf)], procs)
    elif stage == "online":
        _map(job_online, [(d, i, n) for d in DIMS for n in learners for i in range(N_ONLINE * nf)], procs)
    elif stage == "flip":
        _map(job_flip, [(d, i, n) for d in DIMS for n in learners for i in range(N_FLIP * nf)], procs)
    elif stage == "latency":
        path = CACHE / "latency.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(d): latency(d) for d in DIMS}, indent=1))


# ── the report ──────────────────────────────────────────────────────


def _load(kind: str, dim: int, name: str, count: int) -> list[dict]:
    out = []
    for i in range(count):
        p = CACHE / kind / f"{dim}-{i}-{name}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def _mean_ci(xs) -> tuple[float, float]:
    xs = np.asarray([x for x in xs if x == x], dtype=float)
    if not len(xs):
        return float("nan"), float("nan")
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0


def acc_table(dim: int, learners) -> tuple[dict, dict]:
    """{(family, split, learner): (mean over worlds, 95% half-width)}, and the raw world lists."""
    from worlds import FAMILIES

    res, raw = {}, {}
    for name in learners:
        rows = _load("acc", dim, name, N_ACC * len(FAMILIES))
        raw[name] = rows
        for fam in FAMILIES + ("all",):
            for split in ("test", "combo"):
                xs = [np.mean(r[split]["correct"]) for r in rows if (fam == "all" or r["family"] == fam) and split in r]
                res[(fam, split, name)] = _mean_ci(xs)
    return res, raw


def reliability(rows: list[dict]) -> tuple[float, list[tuple[float, int, float, float]], float]:
    conf, ok = [], []
    for r in rows:
        if "test" not in r:
            continue
        p1, y = np.array(r["test"]["p1"]), np.array(r["test"]["y"])
        pred = (p1 > 0.5).astype(int)
        conf.extend(np.maximum(p1, 1 - p1).tolist())
        ok.extend((pred == y).astype(int).tolist())
    conf, ok = np.array(conf), np.array(ok)
    bins = []
    ece = 0.0
    edges = np.linspace(0.5, 1.0, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.sum():
            bins.append((lo, int(m.sum()), float(conf[m].mean()), float(ok[m].mean())))
            ece += m.sum() / len(conf) * abs(conf[m].mean() - ok[m].mean())
    return float(ece), bins, float(ok.mean()) if len(ok) else float("nan")


def first_follow(curves: list[dict], level: float = 0.5) -> tuple[float | None, dict]:
    """The mean follow curve over worlds, and where it first reaches `level` (interpolated linearly between the two
    checkpoints around the crossing, rounded to a whole precedent)."""
    pts = sorted({int(k) for c in curves for k in c})
    mean = {n: float(np.nanmean([c[str(n)] if str(n) in c else c.get(n, np.nan) for c in curves])) for n in pts}
    for a, b in zip([None] + pts[:-1], pts):
        if mean[b] >= level:
            if a is None or mean[b] == mean[a]:
                return b, mean
            return int(round(a + (b - a) * (level - mean[a]) / (mean[b] - mean[a]))), mean
    return None, mean


def fmt(m: float, ci: float | None = None) -> str:
    if m != m:
        return "–"
    return f"{100 * m:.1f}" + (f" ±{100 * ci:.1f}" if ci is not None and ci == ci else "")


def report(learners) -> str:
    from icl import ckpt_path, load
    from worlds import FAMILIES

    L = []
    w = L.append
    metas = {}
    for d in DIMS:
        for n in ICLS:
            if ckpt_path(d, n == "icl_sim").exists():
                metas[(d, n)] = load(d, n == "icl_sim")[1]
    w("# Compiled reasoner: results\n")
    w("The experiment in [docs/compiled-reasoner.md](../../docs/compiled-reasoner.md), run as written, plus one baseline "
      "the doc lacks: **Ada with learned weights and pairs** (`ada_learn`: `glyphh_spec.Model` on the world's spec, "
      "`learn()` then `calibrate()` on the world's training cases; the kernel's own learning since 3.5). Generated by "
      "`research/compile/run.py`; the section headed *Reading* at the end is written by hand.\n")
    w("**Learners.** `knn`: `Space` (k=15, min_sim 0.10), record then `query()`. `ada`: `Model` on the world's spec, "
      "nothing learned (exact weighted sim). `ada_learn`: the same after `learn()` (role and value weights, up to 5 "
      "learned pairs of roles) and `calibrate()`. `logreg`: logistic regression on the glyph, C by 3-fold CV inside the "
      "training cases. `mlp`: one hidden layer of 256, early stopping on a slice of the training cases. `icl`: the "
      "transformer as the doc specifies (glyph projected to d_model 128 as a soft token, outcome embedding added, plus a "
      "rank embedding; 4 layers, 4 heads; k=16 precedents from `Space.query` receipts + the query), trained once across "
      "worlds, never on an evaluation world. `icl_sim`: the same plus each precedent's kernel similarity to the query "
      "(a number every receipt already carries) as an input feature. A kernel learner whose query finds nothing above "
      "min_sim answers with its store's majority outcome, counted right or wrong.\n")
    w("**Worlds.** Every name fresh per world (layer, segments, roles, values, outcomes). 3-6 roles, vocabularies of "
      "3-6, a numeric 0..9 role for threshold×category (the spec's thermometer bins). Two-hop: a selector role says "
      "which of two nested segments' role `x` decides. Noisy: one of the other five with 10% of stored outcomes "
      "flipped; graded against the rule. Each world holds out two value-combinations of the two roles its rule reads "
      "(single: its role and another): `cases` = held-out cases of seen combinations, `combos` = cases of held-out "
      "combinations. Every evaluation world is unseen by the transformer (held-out worlds), so the transformer's "
      "every number here is on held-out worlds; its accuracy on the training worlds' own reserved cases is given "
      "below for the gap.\n")
    w("Numbers are % correct, mean over worlds with a 95% interval (1.96 × standard error over worlds). Seeds fixed; "
      "nothing was chosen on evaluation data: every variant run is reported.\n")

    # training
    w("## Training budget\n")
    w("| variant | dim | params | steps × batch | episodes | training time | device | pool |")
    w("|---|---|---|---|---|---|---|---|")
    for (d, n), m in sorted(metas.items()):
        w(f"| {n} | {d} | {m['params']:,} | {m['steps']:,} × {m['batch']} | {m['steps'] * m['batch']:,} | "
          f"{m['train_secs'] / 60:.1f} min | {m['device']} | {m['pool_worlds']:,} worlds, {m['pool_cases']:,} cases |")
    w("")
    w("Transformer, training worlds' reserved cases (seen worlds) vs. validation worlds (unseen), at the end of training:\n")
    w("| variant | dim | " + " | ".join(FAMILIES) + " | all |")
    w("|---|---|" + "---|" * (len(FAMILIES) + 1))
    for (d, n), m in sorted(metas.items()):
        v = m["log"][-1]["val"]
        w(f"| {n} seen worlds | {d} | " + " | ".join(fmt(m["seen_worlds"][f]) for f in FAMILIES) + f" | {fmt(m['seen_worlds']['all'])} |")
        w(f"| {n} unseen worlds | {d} | " + " | ".join(fmt(v[f]) for f in FAMILIES) + f" | {fmt(v['all'])} |")
    w("")

    accs = {}
    for d in DIMS:
        res, raw = acc_table(d, learners)
        accs[d] = (res, raw)
        w(f"## Accuracy, dim {d}\n")
        for split, title in (("test", "held-out cases"), ("combo", "held-out combinations")):
            w(f"**{title}** ({N_ACC} unseen worlds per family)\n")
            w("| family | " + " | ".join(learners) + " |")
            w("|---|" + "---|" * len(learners))
            for fam in FAMILIES + ("all",):
                cells = []
                best = max(res[(fam, split, n)][0] for n in learners if res[(fam, split, n)][0] == res[(fam, split, n)][0])
                for n in learners:
                    m, ci = res[(fam, split, n)]
                    c = fmt(m, ci)
                    cells.append(f"**{c}**" if m == best else c)
                w(f"| {fam} | " + " | ".join(cells) + " |")
            w("")
        ins = []
        for n in ("knn", "ada", "ada_learn"):
            if raw.get(n):
                ins.append(f"`{n}` {100 * np.mean([r.get('insufficient', 0) for r in raw[n]]):.1f}%")
        w("Queries with nothing above min_sim (answered with the majority outcome): " + ", ".join(ins) + ".\n")

    # learned pairs
    w("## Ada's learned pairs\n")
    w("Share of worlds where `learn()` kept a pair of roles that are both roles the rule reads, and the mean number of "
      "pairs kept:\n")
    w("| family | " + " | ".join(f"rule pair found, dim {d}" for d in DIMS) + " | pairs kept (4096) |")
    w("|---|" + "---|" * (len(DIMS) + 1))
    for fam in FAMILIES:
        cells = []
        for d in DIMS:
            rows = [r for r in accs[d][1].get("ada_learn", []) if r["family"] == fam]
            hit = [any(set(p) <= set(r["reads"]) for p in r["pairs"]) for r in rows]
            cells.append(fmt(np.mean(hit)) if hit else "–")
        rows = [r for r in accs[DIMS[0]][1].get("ada_learn", []) if r["family"] == fam]
        w(f"| {fam} | " + " | ".join(cells) + f" | {np.mean([len(r['pairs']) for r in rows]):.1f} |")
    w("")

    # online
    for d in DIMS:
        w(f"## Online learning, dim {d}\n")
        w(f"Accuracy on each world's held-out cases (up to {TEST_CAP_ONLINE}) after n precedents, mean over "
          f"{N_ONLINE * len(FAMILIES)} worlds ({N_ONLINE} per family). `knn`, `ada` and the transformers never train; "
          "`ada_learn` relearns (the kernel's `learn()`, a counting pass, no gradient) at every point; `logreg` and `mlp` "
          "are retrained from scratch at every point (a reference: they are not online learners).\n")
        curves = {n: _load("online", d, n, N_ONLINE * len(FAMILIES)) for n in learners}
        w("| n | " + " | ".join(learners) + " |")
        w("|---|" + "---|" * len(learners))
        for c in CHECKPOINTS:
            cells = []
            for n in learners:
                xs = [r["curve"].get(str(c)) for r in curves[n] if str(c) in r["curve"]]
                cells.append(fmt(np.mean(xs)) if xs else "–")
            w(f"| {c} | " + " | ".join(cells) + " |")
        w("")
        w("Per family at n = 32 and n = 256:\n")
        w("| family | n | " + " | ".join(learners) + " |")
        w("|---|---|" + "---|" * len(learners))
        for fam in FAMILIES:
            for c in (32, 256):
                cells = []
                for n in learners:
                    xs = [r["curve"].get(str(c)) for r in curves[n] if r["family"] == fam and str(c) in r["curve"]]
                    cells.append(fmt(np.mean(xs)) if xs else "–")
                w(f"| {fam} | {c} | " + " | ".join(cells) + " |")
        w("")

    # flip
    flips = {}
    for d in DIMS:
        w(f"## Flip, dim {d}\n")
        w(f"{PRE_FLIP} precedents under rule 1, then the rule is redrawn and precedents arrive under rule 2. *Follows "
          "rule 2* = share of test cases where the rules disagree that the learner answers by rule 2 (mean over "
          f"{N_FLIP * len(FAMILIES)} worlds); *precedents to follow* = where that mean first reaches 50% (interpolated "
          "between checkpoints). No learner sees timestamps, so every one must be outvoted by new precedents.\n")
        rows = {n: _load("flip", d, n, N_FLIP * len(FAMILIES)) for n in learners}
        w("| learner | before flip (rule 1 acc) | " + " | ".join(f"+{p}" for p in POST) + " | precedents to follow |")
        w("|---|---|" + "---|" * len(POST) + "---|")
        for n in learners:
            if not rows[n]:
                continue
            hit, mean = first_follow([r["follow"] for r in rows[n]])
            flips[(d, n)] = hit
            w(f"| {n} | {fmt(np.mean([r['before'] for r in rows[n]]))} | "
              + " | ".join(fmt(mean.get(p, float('nan'))) for p in POST) + f" | {hit if hit is not None else '> 300'} |")
        w("")
        w("Precedents to follow, per family:\n")
        w("| family | " + " | ".join(learners) + " |")
        w("|---|" + "---|" * len(learners))
        for fam in FAMILIES:
            cells = []
            for n in learners:
                hit, _ = first_follow([r["follow"] for r in rows[n] if r["family"] == fam])
                cells.append(str(hit) if hit is not None else "> 300")
            w(f"| {fam} | " + " | ".join(cells) + " |")
        w("")

    # calibration
    w("## Calibration\n")
    w("Reliability of P(predicted outcome) on held-out cases, pooled over every accuracy world: ECE, and per bin of "
      "confidence the number of answers / mean confidence / accuracy. Kernel learners' probabilities are "
      "`probabilities()` after `calibrate()` on their own training store; their accuracy here is of the probability's "
      "argmax, which can differ from `query().top`.\n")
    for d in DIMS:
        w(f"**dim {d}**\n")
        w("| learner | ECE | acc | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 |")
        w("|---|---|---|---|---|---|---|---|")
        for n in learners:
            ece, bins, acc = reliability(accs[d][1].get(n, []))
            by = {round(lo, 1): (c, mc, a) for lo, c, mc, a in bins}
            cells = [f"{by[b][0]} / {fmt(by[b][1])} / {fmt(by[b][2])}" if b in by else "–" for b in (0.5, 0.6, 0.7, 0.8, 0.9)]
            w(f"| {n} | {100 * ece:.1f} | {fmt(acc)} | " + " | ".join(cells) + " |")
        w("")

    # latency
    lat = json.loads((CACHE / "latency.json").read_text()) if (CACHE / "latency.json").exists() else {}
    if lat:
        w("## Latency\n")
        w("Median per query, batch 1, store of 1,000 precedents, Apple M2 Max (PyTorch CPU with 4 threads, and `mps`). "
          "`retrieve` is `Space.query` (which encodes the situation again itself); `encode` is `Spec.situation` + "
          "`encode`. The transformer's forward includes building its input from the receipts.\n")
        w("| dim | encode | retrieve | forward CPU | forward mps | transformer total CPU | transformer total mps | knn total | ada_learn query |")
        w("|---|---|---|---|---|---|---|---|---|")
        for d in DIMS:
            x = lat.get(str(d))
            if x:
                w(f"| {d} | {x['encode']} ms | {x['retrieve']} ms | {x['forward_cpu']} ms | {x['forward_mps']} ms | "
                  f"{x['icl_total_cpu']} ms | {x.get('icl_total_mps', '–')} ms | {x['knn_total']} ms | {x['ada_learn_query']} ms |")
        w("")

    # verdict
    w("## Go / no-go\n")
    w("The doc's four criteria, per transformer variant and dim. Thresholds fixed before reading results: *matches* = "
      "not more than 2 points below kNN on single-role and noisy worlds, in both held-out cases and held-out "
      "combinations; *clearly beats* = at least 5 points above, on both XOR and two-hop, in held-out cases (combinations "
      "shown beside); *tracks the flip* = precedents to follow (interpolated) at most 5 more than kNN's; *fast* = encode + retrieve + forward under 10 ms on CPU, batch 1. "
      "Criterion 2 is judged twice: against kNN and logistic regression (the doc's wording), and against the strongest "
      "baseline including `ada_learn` (the kernel as it ships).\n")
    w("| variant | dim | 1 matches kNN (single, noisy) | 2 beats kNN & logreg (XOR, two-hop) | 2' beats every baseline incl. ada_learn | 3 flip | 4 < 10 ms | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    verdicts = []
    for d in DIMS:
        res = accs[d][0]
        for n in ICLS:
            if n not in learners:
                continue
            g = lambda fam, split, who: res[(fam, split, who)][0]  # noqa: E731
            c1 = all(g(f, s, n) >= g(f, s, "knn") - 0.02 for f in ("single", "noisy") for s in ("test", "combo"))
            d1 = ", ".join(f"{f} {fmt(g(f, 'test', n))} vs {fmt(g(f, 'test', 'knn'))} (combos {fmt(g(f, 'combo', n))} "
                           f"vs {fmt(g(f, 'combo', 'knn'))})" for f in ("single", "noisy"))
            m2 = {f: max(g(f, "test", "knn"), g(f, "test", "logreg")) for f in ("xor", "twohop")}
            c2 = all(g(f, "test", n) >= m2[f] + 0.05 for f in m2)
            d2 = ", ".join(f"{f} {fmt(g(f, 'test', n))} vs {fmt(m2[f])} (combos {fmt(g(f, 'combo', n))} vs "
                           f"{fmt(max(g(f, 'combo', 'knn'), g(f, 'combo', 'logreg')))})" for f in m2)
            others = [b for b in learners if b not in ICLS]
            m2b = {f: max(g(f, "test", b) for b in others) for f in ("xor", "twohop")}
            c2b = all(g(f, "test", n) >= m2b[f] + 0.05 for f in m2b)
            d2b = ", ".join(f"{f} {fmt(g(f, 'test', n))} vs {fmt(m2b[f])}" for f in m2b)
            fk, fi = flips.get((d, "knn")), flips.get((d, n))
            c3 = fi is not None and (fk is None or fi <= fk + 5)
            d3 = f"{fi if fi is not None else '> 300'} vs kNN {fk if fk is not None else '> 300'}"
            x = lat.get(str(d), {})
            c4 = bool(x) and x["icl_total_cpu"] < 10
            d4 = f"{x.get('icl_total_cpu', '–')} ms"
            ok = c1 and c2 and c3 and c4
            verdicts.append((n, d, ok, c1, c2, c2b, c3, c4))
            yn = lambda c: "yes" if c else "**no**"  # noqa: E731
            w(f"| {n} | {d} | {yn(c1)}: {d1} | {yn(c2)}: {d2} | {yn(c2b)}: {d2b} | {yn(c3)}: {d3} | {yn(c4)}: {d4} | "
              f"{'GO' if ok and c2b else ('GO by the doc, not against ada_learn' if ok else 'NO-GO')} |")
    w("")
    return "\n".join(L)


NOTES = "<!-- notes: everything below this line is hand-written and kept when run.py rewrites the report -->"


def write_report(learners) -> None:
    path = HERE / "REPORT.md"
    kept = ""
    if path.exists():
        text = path.read_text()
        if NOTES in text:
            kept = text[text.index(NOTES):]
    body = report(learners)
    path.write_text(body + "\n" + (kept or NOTES + "\n"))
    results = {}
    for d in DIMS:
        res, _ = acc_table(d, learners)
        results[str(d)] = {f"{f}/{s}/{n}": [round(m, 4), round(ci, 4)] for (f, s, n), (m, ci) in res.items()}
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["acc", "online", "flip", "latency", "report"]
    procs = int(os.environ.get("PROCS", "10"))
    only = os.environ.get("LEARNERS")
    learners = available(only.split(",") if only else LEARNERS)
    print("learners:", learners, flush=True)
    for s in stages:
        t = time.time()
        if s == "report":
            write_report(available(LEARNERS))
        else:
            run_stage(s, learners, procs)
        print(f"{s}: {time.time() - t:.0f}s", flush=True)
