# SFEB Leptos Pattern Retrieval Evaluator

This standalone Rust utility measures retrieval over the public Leptos 0.8
pattern corpus and its paraphrased query set.

## Implemented baselines

- **Keyword overlap** — fraction of unique query terms present in a pattern.
- **BM25** — term-frequency/inverse-document-frequency ranking with length
  normalization.
- **TF-IDF** — logarithmic term-frequency and cosine similarity.
- **Bipolar HDC** — deterministic BLAKE3-seeded 16,384D hypervectors for word,
  adjacent-word and character-trigram features; feature vectors are bundled
  into bipolar document/query representations and compared by cosine/dot
  similarity.
- **Hybrid RRF** — reciprocal-rank fusion over BM25, TF-IDF and bipolar HDC.

The HDC method is a simple, reproducible associative baseline. It is **not**
claimed to reproduce Symthaea's full learned or cognitive HDC memory. Its
purpose is to test whether a transparent high-dimensional associative encoder
adds retrieval value beyond strong sparse lexical baselines.

Dense embeddings are not implemented in this offline runner: they require a
separately pinned embedding model/runtime adapter. The query-set plan retains
that comparison arm, but it must be reported as unavailable until the adapter
is supplied.

## Run

From the repository root, with Rust available:

```bash
cargo test --manifest-path benchmarks/frontend/retrieval-eval/Cargo.toml

cargo run --release \
  --manifest-path benchmarks/frontend/retrieval-eval/Cargo.toml -- \
  benchmarks/frontend/knowledge/leptos-0.8/retrieval-queries-v0.json \
  benchmarks/frontend/knowledge/leptos-0.8/patterns-v0.json \
  16384
```

The default arguments use the checked-in public pilot set and the 16,384D HDC
configuration. The runner prints a JSON result to stdout; validate the output
against `result-v1.schema.json` and store it with the exact source commit,
Rust toolchain, lockfile hash, CPU details and command line.

## Reported metrics

For positive queries, each method reports recall@1/3, MRR, nDCG@3 and
precision@3. For hard negatives, it reports the distribution of top-ranked
scores. It also records per-query ranking details and p50/p95/mean scoring
latency.

No default abstention threshold is baked in. Thresholds must be calibrated on
a separate development split and then frozen before evaluating hidden queries.
Until calibrated, the output explicitly leaves abstention threshold and
abstention accuracy unset; do not interpret a low top score as a proven
abstention decision.

## Comparison protocol

- Keep corpus, query wording, hardware, Rust toolchain and HDC dimension fixed
  across methods.
- Run at least 5 repeated trials for latency; report warm/cold state separately.
- Keep public pilot queries separate from private holdout queries.
- Add a pinned dense embedding adapter before claiming full comparison.
- Report scores by pattern family and hard-negative class, not only one mean.
- Run ablations: words only, word+bigram, word+character-trigram, 4K/16K/64K
  dimensions, and HDC with/without sparse fusion.
- A method wins only if retrieval quality, negative abstention and latency/cost
  all improve under the same protocol.

## Current evidence status

The evaluator source and unit tests are authored but have not been compiled or
executed in this environment. No retrieval score, HDC uplift or SOTA claim is
made. The query set is public pilot data, not a blind holdout.
