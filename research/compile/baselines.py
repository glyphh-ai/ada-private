"""The learners the in-context transformer is judged against (docs/compiled-reasoner.md, "Learners compared").

  knn        glyphh_kernel.Space: record the training cases, query() each test case (the doc's bar)
  ada        glyphh_spec.Model on the world's spec, as shipped, without learn(): exact weighted sim, no weights learned
  ada_learn  the same Model after learn(): learned role and value weights and learned pairs of roles (interactions),
             then calibrate(); the kernel's own learning since 3.5
  logreg     logistic regression on the glyph (C by 3-fold CV inside the training cases)
  mlp        a one-hidden-layer MLP on the glyph, trained per world (early stopping on a slice of the training cases)

Every learner takes cases by index into a World and the labels to store (0/1), and answers with the predicted
label and P(label 1). A kernel learner with nothing above min_sim ('insufficient') answers with its store's
majority outcome (counted as its answer, right or wrong); its probability then is the store's base rate.
"""

from __future__ import annotations

import warnings

import numpy as np

from worlds import SEED, World, kernel

from glyphh_spec import Model, SpecError  # noqa: E402  (worlds put the repo root on sys.path)


class Knn:
    name = "knn"

    def __init__(self, world: World, dim: int, k: int = 15, min_sim: float = 0.10):
        self.w = world
        self.space = kernel.Space(dim=dim, k=k, min_sim=min_sim, seed=SEED)
        self.labels: list[int] = []
        self.insufficient = 0
        self.asked = 0

    def add(self, idx, labels) -> None:
        o = self.w.outcomes
        self.space.record_many(
            [{"situation": self.w.situations[i], "outcome": o[int(y)], "rid": str(int(i)), "ts": float(len(self.labels) + j)}
             for j, (i, y) in enumerate(zip(idx, labels))]
        )
        self.labels.extend(int(y) for y in labels)

    def refit(self) -> None:
        if len(self.labels) >= 10:
            self.space.calibrate(sample=100)

    def _base(self) -> float:
        return float(np.mean(self.labels)) if self.labels else 0.5

    def predict(self, idx) -> tuple[np.ndarray, np.ndarray]:
        o = self.w.outcomes
        preds, p1 = [], []
        base = self._base()
        for i in idx:
            s = self.w.situations[i]
            a = self.space.query(s)
            self.asked += 1
            if a.top is None:
                self.insufficient += 1
                preds.append(int(base > 0.5))
            else:
                preds.append(o.index(a.top))
            p = self.space.probabilities(s) if len(self.labels) else {}
            p1.append(p.get(o[1], 0.0) if p else base)
        return np.array(preds), np.array(p1)

    def receipts(self, i: int, k: int) -> list[int]:
        """The k nearest stored cases to case i, nearest first (Space.query's receipts, as the doc retrieves)."""
        saved = self.space.k
        self.space.k = k
        try:
            return [int(r[0]) for r in self.space.query(self.w.situations[i]).receipts]
        finally:
            self.space.k = saved


class Ada:
    """glyphh_spec.Model on the world's spec. learn=True: learn() (role and value weights, pairs of roles) and
    calibrate() at every refit, as Model.refresh() does."""

    def __init__(self, world: World, dim: int, learn: bool):
        self.w, self.learn = world, learn
        self.name = "ada_learn" if learn else "ada"
        spec = dict(world.spec_json, dimension=dim)
        from glyphh_spec import Spec

        self.model = Model(Spec.from_json(spec))
        self.labels: list[int] = []
        self.learned: dict | None = None
        self.insufficient = 0
        self.asked = 0

    def add(self, idx, labels) -> None:
        o = self.w.outcomes
        self.model.record_many(
            [{"data": self.w.data[i], "outcome": o[int(y)], "ts": float(len(self.labels) + j)}
             for j, (i, y) in enumerate(zip(idx, labels))]
        )
        self.labels.extend(int(y) for y in labels)

    def refit(self) -> None:
        if not self.learn or len(self.labels) < 10 or len(set(self.labels)) < 2:
            if len(self.labels) >= 10:
                self.model.calibrate()
            return
        try:
            self.learned = self.model.learn()
            self.model.calibrate()
        except SpecError:
            pass

    def predict(self, idx) -> tuple[np.ndarray, np.ndarray]:
        o = self.w.outcomes
        base = float(np.mean(self.labels)) if self.labels else 0.5
        preds, p1 = [], []
        for i in idx:
            d = self.w.data[i]
            a = self.model.query(d)
            self.asked += 1
            if a.top is None:
                self.insufficient += 1
                preds.append(int(base > 0.5))
            else:
                preds.append(o.index(a.top))
            p = self.model.probabilities(d) if self.labels else {}
            p1.append(p.get(o[1], 0.0) if p else base)
        return np.array(preds), np.array(p1)

    def pairs(self) -> list[list[str]]:
        """The pairs of roles learn() kept (role paths)."""
        if not self.learned:
            return []
        return [p["roles"] for p in self.learned.get("~interactions", [])]


class _Sk:
    """A scikit-learn classifier on the glyph, refit from scratch at every refit()."""

    def __init__(self, world: World):
        self.w = world
        self.X: list[int] = []
        self.y: list[int] = []
        self.clf = None
        self.const: int | None = None

    def add(self, idx, labels) -> None:
        self.X.extend(int(i) for i in idx)
        self.y.extend(int(v) for v in labels)

    def refit(self) -> None:
        y = np.array(self.y)
        if len(set(self.y)) < 2:
            self.const = int(y[0]) if len(y) else 0
            return
        self.const = None
        X = self.w.glyphs[self.X].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf = self._make(len(y), np.bincount(y, minlength=2).min())
            self.clf.fit(X, y)

    def predict(self, idx) -> tuple[np.ndarray, np.ndarray]:
        if self.const is not None or self.clf is None:
            c = self.const or 0
            return np.full(len(idx), c), np.full(len(idx), float(c))
        X = self.w.glyphs[np.asarray(idx)].astype(np.float32)
        p1 = self.clf.predict_proba(X)[:, 1]
        return (p1 > 0.5).astype(int), p1


class LogReg(_Sk):
    name = "logreg"

    def __init__(self, world: World, cv: bool = True):
        super().__init__(world)
        self.cv = cv

    def _make(self, n: int, minority: int):
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        if self.cv and minority >= 3:
            return LogisticRegressionCV(Cs=list(np.logspace(-3, 1, 5)), cv=3, max_iter=1000)
        return LogisticRegression(C=0.1, max_iter=1000)


class Mlp(_Sk):
    name = "mlp"

    def _make(self, n: int, minority: int):
        from sklearn.neural_network import MLPClassifier

        early = n >= 60 and minority >= 5
        return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=early,
                             validation_fraction=0.15, n_iter_no_change=15, random_state=0)


def make(name: str, world: World, dim: int):
    if name == "knn":
        return Knn(world, dim)
    if name == "ada":
        return Ada(world, dim, learn=False)
    if name == "ada_learn":
        return Ada(world, dim, learn=True)
    if name == "logreg":
        return LogReg(world)
    if name == "mlp":
        return Mlp(world)
    raise ValueError(name)


BASELINES = ("knn", "ada", "ada_learn", "logreg", "mlp")


if __name__ == "__main__":
    import time

    from worlds import FAMILIES, make_world, world_seed

    for fam in FAMILIES:
        w = make_world(world_seed("eval", 100 + FAMILIES.index(fam)), fam, dim=1024)
        row = []
        for name in BASELINES:
            t = time.time()
            m = make(name, w, 1024)
            m.add(w.train, w.stored[w.train])
            m.refit()
            acc = (m.predict(w.test)[0] == w.clean[w.test]).mean()
            cmb = (m.predict(w.combo)[0] == w.clean[w.combo]).mean() if len(w.combo) else float("nan")
            row.append(f"{name}={acc:.2f}/{cmb:.2f} ({time.time()-t:.1f}s)")
        print(f"{fam:9s} n={len(w.train):4d}", "  ".join(row))
