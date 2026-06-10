# symthaea-eml-egraph

Experimental `egg` spike for Symthaea's EML work.

This crate is intentionally isolated from production code. Its purpose is to
answer one narrow question: can equality saturation simplify or canonicalize the
same symbolic forms Symthaea currently handles with bespoke rewrites, without
dragging `egg` into the main conjecture engine too early?

Current scope:

- source-level equivalence checks for a tiny symbolic language
- EML bridging rules such as `exp(x) <=> eml(x, 1)`
- identity rewrites like `pow(x, 2) <=> mul(x, x)`
- comparative micro-benchmarks against Symthaea's current canonicalization and
  EML compilation path
- offline equivalence-class collapse for supported Symthaea `Expr` candidates

Not in scope yet:

- domain-aware real verification
- constructive-vs-strict backend choice
- replacing the current EML verifier or conjecture ranking

The batch-collapse helpers are intended for offline candidate dedupe experiments,
for example on dynamic-grammar subtree pools before promotion analysis.

Run the spike tests with:

```bash
cargo test -p symthaea-eml-egraph
```

Run the micro-benchmarks with:

```bash
cargo bench -p symthaea-eml-egraph --bench eml_egraph_bench
```

Run the offline candidate-collapse examples with:

```bash
cargo run -p symthaea-eml-egraph --example candidate_collapse
cargo run -p symthaea-eml-egraph --example dynamic_grammar_collapse
cargo run -p symthaea-eml-egraph --example real_discovery_collapse
```
