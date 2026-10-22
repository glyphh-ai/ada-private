# Compiled reasoner

Status: proposal. Nothing built yet. The experiment below decides whether it gets built.

## The idea

Today Ada learns lazily. `Space` stores each precedent as a glyph (4096-d, ±1), and a query takes the nearest
precedents above `min_sim` and lets their outcomes vote. That's instant to learn (`record()` is one write, no
training run) and every answer cites its receipts. But similarity voting can't express some things:

- **Interactions.** "car AND amount > 5k → bad, either alone → fine" (XOR-shaped). Similarity adds up agreeing
  attributes, so it can't flip on a combination.
- **Multi-hop structure.** Unbinding a role from a bound glyph gets noisier with every hop.
- **Weighing conflicting neighbours by *why* they differ**, not just by how close they are.

The proposal is to train a small transformer that **reads precedents in context** and answers about a new case:

```
[glyph(p1), outcome(p1), glyph(p2), outcome(p2), ..., glyph(pk), outcome(pk), glyph(query)] -> outcome(query)
```

The precedents come from `Space.query(...).receipts`, the same top-k Ada already retrieves.

## Ada keeps learning: the weights hold no precedents

Retraining on every new precedent would break Ada's main property, which is that she learns as she goes. So the
model never memorises real precedents:

- **The store holds what Ada knows.** `record()` is unchanged: one write, no training.
- **The model holds only how to reason from precedents.** It's trained once, on synthetic rule-worlds with made-up
  symbols, so there's nothing real for it to memorise. At query time it reads whatever `Space` retrieves, including a
  precedent recorded a second ago.

It's like a judge who learned to read case law in school and reads the actual cases at each hearing.

Retraining is needed only when the encoding changes (seed, `dim`, lexicon; i.e. after `reencode_store`), because a
glyph means nothing to a model trained on a different encoding.

Optional, later: a periodic offline "consolidation" retrain that also sees the real store. Skipping it breaks
nothing.

### Limits

1. **It sees only what retrieval hands it** (`k=15` by default): better reasoning over the neighbours, not wider
   recall than kNN.
2. **It must trust the context over its priors.** Training worlds shuffle symbols and flip rules, so the only way to
   score is to read the context. The experiment tests this directly (the "flip" test below).
3. **It loses `explain()`.** Use the model to *propose* and the kernel to *justify*. When they disagree, that's a signal,
   not something to hide.

## The experiment

Put the harness in `research/compile/`. It needs `numpy` and `torch` (on a Mac, PyTorch's `mps` device uses the GPU).

### 1. Rule-worlds (synthetic, encoded with the kernel)

Each world has 3–6 roles with small vocabularies or numeric bins, a hidden rule, and 200–2000 cases encoded with
`glyphh_kernel.encode(...)`. Draw role and value names fresh for every world (e.g. `r_8f3a`, `v_19c2`) so no
symbol carries meaning across worlds.

Rule families, from easy to hard:

| Family | Example | What kNN does |
|---|---|---|
| single role | `colour == v1 → A` | fine |
| AND | `a == v1 and b == v2 → A` | mostly fine |
| XOR | `(a == v1) != (b == v2) → A` | fails |
| threshold × category | `kind == v1 and amount > 5 → A` | partial |
| two-hop | outcome depends on the value of a role nested inside another role | degrades |
| noisy | any of the above, with 10% labels flipped | the baseline should be robust |

Hold out whole **combinations** (e.g. never train on `a=v1, b=v2` together) as well as whole **worlds**.

### 2. Learners compared

1. **`Space` kNN.** Record the training cases, then `query()` each test case. This is the bar.
2. **Logistic regression on the glyph.** Bundling is near-linear, so this is a strong baseline.
3. **Small MLP on the glyph,** trained per world. This shows what interactions are available at all.
4. **In-context transformer,** trained once across worlds and tested on unseen worlds with no gradient steps.

Starting transformer shape: each glyph projected 4096→128 (or `dim=1024`, see below) as one "soft token"; the outcome
as a learned embedding added to its precedent's token; 4 layers, 4 heads, d_model 128 (~1.3M params); context of k=16
precedents plus the query.

### 3. Tests

- **Accuracy** on held-out cases, held-out combinations and held-out worlds, per rule family.
- **Online learning (the important one).** Start with an empty store, add precedents one at a time, never retrain,
  and plot accuracy against the number of precedents seen, for kNN vs. the transformer.
- **Flip.** Halfway through, change the world's rule. Count how many new precedents each learner needs before it
  follows the new rule. The transformer must not lag far behind kNN here, or it's overriding the context with priors.
- **Calibration.** Does its confidence mean what kNN's does? Compare reliability curves.
- **Latency.** Encode + retrieve + forward, per query.

### 4. Go / no-go

Build it only if the transformer, **with no retraining**:

- matches kNN on single-role and noisy worlds,
- clearly beats kNN *and* logistic regression on XOR and two-hop rules,
- tracks the flip test within a few precedents of kNN, and
- answers in under ~10 ms per query on a laptop.

If it doesn't beat logistic regression anywhere, the kernel's algebra is already doing the work, and the right
place to improve is the encoding, not a network.

## Cost, as measured

Measured in a cloud Linux container (4 CPU cores, no GPU, numpy only), not on a Mac:

| Step | Cost |
|---|---|
| `encode()` one case | 0.24 ms |
| Transformer forward, ~1.3M params, 33 tokens, batch 1 | 3 ms |
| Same, batch 64 | 110 ms |

- **Per query:** about 0.25 ms encode + the `Space` lookup + 3 ms forward. That's under 5 ms.
- **Training 10k steps at batch 64:** an estimated ~1 h in numpy on that box, ~10–20 min with PyTorch on CPU, and
  minutes on Apple Silicon `mps`. These are estimates from the forward pass; time a real run.
- **Smaller encoding:** the 4096→128 input projection is ~40% of the parameters and over half the compute. Encoding at
  `dim=1024` raises interference between random symbols from ~0.016 to ~0.03 and roughly halves training cost. Run the
  experiment at both.

## Order of work

1. `research/compile/worlds.py`: rule-world generator plus kernel encoding, with fresh symbols per world.
2. `research/compile/baselines.py`: `Space` kNN, logistic regression, MLP.
3. `research/compile/icl.py`: the in-context transformer, its training loop, and checkpoints.
4. `research/compile/run.py`: every test above, written to `research/compile/REPORT.md` with a results table and the
   go/no-go verdict.
