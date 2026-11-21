# MACE HDC/HDC-LTC retrieval adapter v0.1

This package implements the first executable slice specified by [#13127](https://github.com/Luminous-Dynamics/symthaea/issues/13127) and [PR #13128](https://github.com/Luminous-Dynamics/symthaea/pull/13128).

The adapter reuses the repository's existing `symthaea-hdc-ltc` implementation. It does not reimplement ContinuousHV or LTC dynamics.

## Contract

- Build one bipolar role/value associative bucket per exact `(regime, objective, action)` key.
- Bind each stable event-ID code to the role key, then bundle using `ContinuousHV::bundle`.
- Retrieve by unbinding the requested role key and ranking the stored event-ID codebook by cosine similarity.
- Optional `HdcLtcUnifiedNeuron` evolution applies a declared elapsed horizon to the same bundled state using zero input.
- Return candidate event IDs, scores, ranks, abstention, profile digest, latency and a vector/ID payload-byte lower bound.
- Never return an `Experience`, source receipt, causal proof, trust status or action authorization.

A consumer must resolve every returned ID against the canonical evidence registry and run the unchanged MACE source, upstream-root, measurement, intervention, estimand, objective, outcome, trust and retraction predicates. Unknown, duplicate or unresolved identifiers fail closed.

## Arms included

- Exact-scan candidate ceiling.
- Static ContinuousHV key/value associative bundle.
- HDC-LTC zero-input temporal evolution.

Distractor, semantically confusable interference, irregular-gap and delayed-correction schedules remain follow-on arms from #13099. This package does not claim that HDC-LTC improves retrieval or decision utility.

## Run

```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-hdc-retrieval
```

Tests are authored but have not been executed in the current environment.
