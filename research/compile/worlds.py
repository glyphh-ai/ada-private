"""Rule-worlds for the compiled-reasoner experiment (docs/compiled-reasoner.md).

A world is a handful of roles with small vocabularies (or a numeric 0..9), a hidden rule, and 200-2000 cases,
every case encoded with the kernel (glyphh_spec.Spec.situation, then glyphh_kernel.encode). Every name in a world,
its layer, segments, roles, values and outcomes, is drawn fresh (r_8f3a, v_19c2, o_77d0), so no symbol carries
meaning from one world to the next.

Families: single role, AND, XOR, threshold x category, two-hop (a selector role says which nested segment's role
decides), and noisy (one of the other five with 10% of the stored labels flipped; tests grade against the rule).

Each world holds out whole combinations of two roles' values (never in its training cases); its test cases are
split into `test` (seen combinations) and `combo` (held-out combinations).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import glyphh_kernel as kernel  # noqa: E402
from glyphh_spec import Spec  # noqa: E402

FAMILIES = ("single", "and", "xor", "threshold", "twohop", "noisy")
BASES = ("single", "and", "xor", "threshold", "twohop")
NOISE = 0.10
SEED = 42  # the kernel's symbol seed (one shared vocabulary; the names are what is fresh)
TEST_CAP = 200


class Names:
    """Fresh, distinct names within one world."""

    def __init__(self, rng: np.random.Generator):
        self.rng, self.used = rng, set()

    def __call__(self, prefix: str) -> str:
        while True:
            name = f"{prefix}_{int(self.rng.integers(0, 16**4)):04x}"
            if name not in self.used:
                self.used.add(name)
                return name


@dataclass
class World:
    family: str
    base: str  # the rule's family (for noisy: the family it adds noise to)
    spec_json: dict
    data: list[dict]  # each case in the spec's shape
    clean: np.ndarray  # int 0/1: the rule's outcome
    stored: np.ndarray  # int 0/1: the outcome recorded (clean, or with NOISE flipped for noisy worlds)
    outcomes: tuple[str, str]  # fresh outcome names, index 0 and 1
    held: np.ndarray  # bool: the case is in a held-out combination
    train: np.ndarray  # indices
    test: np.ndarray
    combo: np.ndarray
    glyphs: np.ndarray | None = None  # int8 (N, dim)
    situations: list = field(default_factory=list)
    rule: Callable[[dict], int] | None = None
    redraw: Callable[[np.random.Generator], Callable[[dict], int]] | None = None  # a new rule, same family/roles
    describe: str = ""
    reads: tuple[str, ...] = ()  # the role paths (layer.segment.role) the rule reads

    @property
    def spec(self) -> Spec:
        return Spec.from_json(self.spec_json)


def _half(rng: np.random.Generator, values: list[str]) -> set[str]:
    n = len(values) // 2 if rng.random() < 0.5 else (len(values) + 1) // 2
    return set(rng.choice(values, size=max(1, min(n, len(values) - 1)), replace=False).tolist())


def make_world(seed: int, family: str, dim: int = 4096, n: int | None = None, n_range=(200, 2000),
               encode: bool = True) -> World:
    rng = np.random.default_rng(seed)
    name = Names(rng)
    base = family if family != "noisy" else str(rng.choice(BASES))
    if n is None:
        lo, hi = n_range
        n = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    layer = name("l")
    vocab = lambda k: [name("v") for _ in range(k)]  # noqa: E731

    if base == "twohop":
        top, left, right = name("s"), name("s"), name("s")
        sel, x, y = name("r"), name("r"), name("r")
        dists = [name("r") for _ in range(int(rng.integers(1, 3)))]
        sel_v = vocab(2)
        x_v = vocab(int(rng.integers(3, 7)))
        y_v = vocab(int(rng.integers(3, 7)))
        d_v = {d: vocab(int(rng.integers(3, 7))) for d in dists}
        segments = [
            {"name": top, "roles": [{"name": sel}] + [{"name": d} for d in dists]},
            {"name": left, "roles": [{"name": x}, {"name": y}]},
            {"name": right, "roles": [{"name": x}, {"name": y}]},
        ]

        def draw_case() -> dict:
            return {layer: {
                top: {sel: str(rng.choice(sel_v)), **{d: str(rng.choice(d_v[d])) for d in dists}},
                left: {x: str(rng.choice(x_v)), y: str(rng.choice(y_v))},
                right: {x: str(rng.choice(x_v)), y: str(rng.choice(y_v))},
            }}

        def make_rule(r: np.random.Generator) -> tuple[Callable[[dict], int], str]:
            S = _half(r, x_v)
            pick = str(r.choice(sel_v))

            def rule(c: dict) -> int:
                seg = left if c[layer][top][sel] == pick else right
                return int(c[layer][seg][x] in S)

            return rule, f"x of ({left} if {sel}=={pick} else {right}) in {sorted(S)}"

        held_pairs = {(sel_v[0], str(rng.choice(x_v))), (sel_v[1], str(rng.choice(x_v)))}

        def is_held(c: dict) -> bool:
            s = c[layer][top][sel]
            seg = left if s == sel_v[0] else right
            return (s, c[layer][seg][x]) in held_pairs

        reads = [f"{layer}.{top}.{sel}", f"{layer}.{left}.{x}", f"{layer}.{right}.{x}"]

    else:
        seg = name("s")
        n_roles = int(rng.integers(3, 7))
        roles = [name("r") for _ in range(n_roles)]
        numeric = roles[1] if base == "threshold" else None
        vals = {r: vocab(int(rng.integers(3, 7))) for r in roles if r != numeric}
        role_specs = [
            {"name": r, "numeric_config": {"min_value": 0, "max_value": 9, "bin_width": 1}} if r == numeric else {"name": r}
            for r in roles
        ]
        segments = [{"name": seg, "roles": role_specs}]

        def draw_case() -> dict:
            return {layer: {seg: {r: (int(rng.integers(0, 10)) if r == numeric else str(rng.choice(vals[r]))) for r in roles}}}

        def make_rule(r: np.random.Generator) -> tuple[Callable[[dict], int], str]:
            a, b = (str(v) for v in r.choice([q for q in roles if q != numeric], size=2, replace=False))
            if base == "single":
                S = _half(r, vals[a])
                return (lambda c: int(c[layer][seg][a] in S)), f"{a} in {sorted(S)}"
            if base == "and":
                Sa, Sb = _half(r, vals[a]), _half(r, vals[b])
                return (lambda c: int(c[layer][seg][a] in Sa and c[layer][seg][b] in Sb)), f"{a} in {sorted(Sa)} and {b} in {sorted(Sb)}"
            if base == "xor":
                Sa, Sb = _half(r, vals[a]), _half(r, vals[b])
                return (lambda c: int((c[layer][seg][a] in Sa) != (c[layer][seg][b] in Sb))), f"({a} in {sorted(Sa)}) xor ({b} in {sorted(Sb)})"
            # threshold x category
            S = _half(r, vals[a])
            t = int(r.integers(3, 7))
            return (lambda c: int(c[layer][seg][a] in S and c[layer][seg][numeric] > t)), f"{a} in {sorted(S)} and {numeric} > {t}"

    rule, desc = make_rule(rng)

    if base != "twohop":
        # the held-out combinations: two values of the two roles the rule reads (single: its role and another).
        # Which roles does the rule read? Found by probing: the roles a change of value moves the outcome for.
        probe = [draw_case() for _ in range(200)]
        reads = []
        for r in roles:
            moved = 0
            for c in probe:
                c2 = {layer: {seg: dict(c[layer][seg])}}
                choices = list(range(10)) if r == numeric else vals[r]
                c2[layer][seg][r] = choices[int(rng.integers(0, len(choices)))]
                if r == numeric:
                    c2[layer][seg][r] = int(c2[layer][seg][r])
                moved += rule(c) != rule(c2)
            if moved:
                reads.append(r)
        p = reads[0] if reads else roles[0]
        q = reads[1] if len(reads) > 1 else next(r for r in roles if r != p)
        pv = list(range(10)) if p == numeric else vals[p]
        qv = list(range(10)) if q == numeric else vals[q]
        all_pairs = [(a, b) for a in pv for b in qv]
        pick = rng.choice(len(all_pairs), size=2, replace=False)
        held_pairs = {all_pairs[i] for i in pick}

        def is_held(c: dict) -> bool:
            return (c[layer][seg][p], c[layer][seg][q]) in held_pairs

        reads = [f"{layer}.{seg}.{r}" for r in reads]

    def redraw(r: np.random.Generator) -> Callable[[dict], int]:
        return make_rule(r)[0]

    outcomes = (name("o"), name("o"))
    data = [draw_case() for _ in range(n)]
    clean = np.array([rule(c) for c in data], dtype=np.int64)
    stored = clean.copy()
    if family == "noisy":
        flip = rng.random(n) < NOISE
        stored[flip] = 1 - stored[flip]
    held = np.array([is_held(c) for c in data], dtype=bool)
    free = np.flatnonzero(~held)
    rng.shuffle(free)
    n_test = max(1, int(round(0.2 * len(free))))
    test, train = np.sort(free[:n_test][:TEST_CAP]), np.sort(free[n_test:])
    combo = np.flatnonzero(held)
    if len(combo) > TEST_CAP:
        combo = np.sort(rng.choice(combo, size=TEST_CAP, replace=False))
    spec_json = {"name": name("m"), "dimension": dim, "seed": SEED, "layers": [{"name": layer, "segments": segments}]}
    w = World(family, base, spec_json, data, clean, stored, outcomes, held, train, test, combo,
              rule=rule, redraw=redraw, describe=desc, reads=tuple(reads))
    if encode:
        encode_world(w, dim)
    return w


def encode_world(w: World, dim: int) -> None:
    spec = w.spec
    w.situations = [spec.situation(c) for c in w.data]
    w.glyphs = np.stack([kernel.encode(s, seed=SEED, dim=dim) for s in w.situations]).astype(np.int8)


def world_seed(split: str, i: int) -> int:
    """Disjoint seed ranges: training worlds, validation worlds (watched during training), eval worlds."""
    return {"train": 1_000_000, "val": 2_000_000, "eval": 3_000_000}[split] + i


def family_of(split: str, i: int) -> str:
    return FAMILIES[i % len(FAMILIES)]


# ── the training pool: many small worlds, packed for the transformer ──


def _pool_world(args: tuple[str, int, int]) -> dict[str, Any]:
    split, i, dim = args
    w = make_world(world_seed(split, i), family_of(split, i), dim=dim, n_range=(200, 500))
    g = w.glyphs.astype(np.float32)
    sims = (g @ g.T).astype(np.int16)  # dot products: sim * dim, exact integers in [-dim, dim]
    return {"glyphs": w.glyphs, "clean": w.clean, "stored": w.stored, "train": w.train, "test": w.test,
            "combo": w.combo, "sims": sims, "family": FAMILIES.index(w.family)}


def build_pool(split: str, count: int, dim: int, cache: Path, procs: int = 10) -> list[dict[str, Any]]:
    """`count` worlds of 200-500 cases, encoded at `dim`, with each world's exact glyph-sim matrix; cached."""
    import pickle
    from multiprocessing import get_context

    path = cache / f"pool-{split}-{count}-{dim}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            pool = pickle.load(f)
        for p in pool:
            p["glyphs"] = np.unpackbits(p["glyphs"], axis=1)[:, :dim].astype(np.int8) * 2 - 1
        return pool
    with get_context("spawn").Pool(procs) as ex:
        pool = ex.map(_pool_world, [(split, i, dim) for i in range(count)], chunksize=8)
    cache.mkdir(parents=True, exist_ok=True)
    packed = [{**p, "glyphs": np.packbits(p["glyphs"] > 0, axis=1)} for p in pool]
    with open(path, "wb") as f:
        pickle.dump(packed, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pool


if __name__ == "__main__":
    import time

    for fam in FAMILIES:
        t = time.time()
        w = make_world(world_seed("eval", 0) + FAMILIES.index(fam), fam, dim=1024)
        print(f"{fam:10s} base={w.base:9s} n={len(w.data):5d} train={len(w.train):4d} test={len(w.test):3d} "
              f"combo={len(w.combo):3d} P(1)={w.clean.mean():.2f} {time.time()-t:.2f}s  {w.describe}")
