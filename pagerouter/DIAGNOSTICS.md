# Diagnostic fallback options

If `scripts/run_visual_diagnostics.py` shows high neighbor entropy (>60% of
`log2(14) ≈ 3.81 bits`), visual clusters do not reliably predict the best model.
Three fallback framings are available:

---

## Option A — Model-family routing

Route to a model **family** (pipeline / specialist VLM / frontier) rather than
an individual model. Families are coarser targets and more visually separable.
Trade-off: you recover less of the oracle gap because you still pick the wrong
model within the winning family.

**When to use:** neighbor entropy is moderate (30–60% of max) and family-level
oracle gap is meaningfully larger than best-single performance.

---

## Option B — Confidence-gated routing

Only trust the KNN vote when neighbor entropy is **below a threshold**; fall
back to the best-single model otherwise. Report as an abstention curve:
routing performance vs. coverage percentage.

Steps:
1. Run KNN routing on all pages.
2. For each page, record the neighbor entropy.
3. Sort pages by ascending entropy (most confident first).
4. Plot mean NED vs. fraction of pages routed (coverage curve).
5. Find the entropy threshold where routing beats best-single.

**When to use:** a meaningful subset of pages has low entropy even if the
overall mean is high — routing works well on easy cases.

---

## Option C — Niche-detection reframe

Instead of routing *to* the best model, detect pages where the best-single
model **uniquely fails** and route *away* from it. Reduces the problem to one
binary classifier per model: "does this page expose model X's blind spot?"

Steps:
1. For each model X, label pages where X is worst (bottom quartile NED) but
   at least one other model is in the top quartile — i.e., X fails and
   something better exists.
2. Train a binary CLIP-KNN or logistic classifier per model.
3. At inference: run all classifiers; if model X is predicted to fail, exclude
   it from the candidate set and route to the next-best model.

**When to use:** entropy is uniformly high but specific models have
visually-identifiable failure modes (e.g., a handwriting specialist that fails
on printed PDFs).
