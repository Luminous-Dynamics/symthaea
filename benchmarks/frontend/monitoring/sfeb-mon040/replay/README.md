# MON-040 — Replay Reproducibility Qualification

MON-040 distinguishes a single observed failure from a reproducible failure signature. It evaluates a preregistered set of repeated replay observations for one exact fault set, scenario, recovery boundary and code revision.

## Offline example

    cd benchmarks/frontend/monitoring/sfeb-mon040/replay
    go run ./cmd/qualify --input INPUT-example.json
    bash CHECKS.sh

The example contains 29/29 exact blocked-signature matches. Its one-sided 95% Clopper–Pearson lower confidence bound is about 0.9019, just above the preregistered 0.90 target. It is a synthetic known-answer fixture, not a live reliability study.

## Statistical rule

The replay count and threshold are fixed before evaluation. Missing runs cannot reduce the denominator. Every replay has a unique replay ID and seed, and carries a receipt digest recomputed over the complete observation.

The one-sided lower confidence bound is calculated from the exact binomial tail. A qualified reproducibility result requires:
- exactly the preregistered number of complete observations;
- no duplicate replay IDs or seeds;
- identical trial, fault-set, scenario, recovery boundary and code revision;
- valid full-record receipt digests;
- the exact target blocked verdict and failure-signature digest often enough that the lower confidence bound meets the reliability target.

An indeterminate replay outcome holds the conclusion. A blocked result with a different failure signature is a non-match.

## Important assumption

The exact binomial confidence interval assumes independent, identically distributed Bernoulli replay outcomes. Unique seeds alone do not prove independence. The experiment report must document common executor, host, storage, clock, identity, network and control-plane dependencies; repeated runs sharing one fault domain may not support the assumed independence. The confidence interval is conditional on the preregistered trial population and fault schedule, not a universal production failure probability.

## Trust and safety boundary

Replay receipts are digest-bound, not authenticated. MON-026 or another independent custody system must authenticate the live executor, receipts and their provenance. The verifier does not launch replay jobs or inject faults.

Offline and runtime-candidate outputs remain campaign-indeterminate. `production_use_permitted` is always false. Symthaea may propose schedules; independent MON-037/MON-038 verdicts and signatures remain the oracle.
