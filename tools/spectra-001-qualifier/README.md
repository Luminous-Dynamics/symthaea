# SPECTRA-001 T0 qualifier

This standalone Rust crate implements the first executable slice of SPECTRA-001:
a strict structural validator and exact descriptive rank scorer for synthetic,
blinded hidden-target trial records.

## What it checks

- distinct participant, operator, target-custodian, evaluator and judge roles;
- lowercase SHA-256 digest syntax for the declared protocol, target pool,
  target-selection receipt and response commitments;
- unique candidate identity and exactly one true-target candidate;
- a complete permutation of the candidate set from every blinded judge;
- strictly ordered uncertainty windows for apparatus lock, target selection,
  response sealing and target reveal, all in one declared clock domain;
- exact integer rank summaries and deterministic replay of known-answer fixtures.

## Deliberate claim ceiling

The scorer receives only the trial record. Synthetic world labels are used by
the fixture verifier but are not inputs to the scorer. In particular, a
synthetic behavioral signal and a synthetic RF-leakage signal can produce the
same rank score. A rank score cannot identify the causal mechanism.

A PASS means only that the input is structurally consistent with this v0.1
contract and its expected synthetic score. It does not authenticate the event
logger, prove that a digest matches external bytes, verify a public randomness
beacon, establish target-custodian independence, measure shielding, detect
physical leakage, estimate a p-value, establish statistical significance, or
provide evidence for ESP.

The event time windows must already have been mapped into one calibrated clock
domain by an external adapter. If uncertainty windows overlap or use different
clock domains, the trial is not evaluable.

## Run

From the repository root:

    cargo test --manifest-path tools/spectra-001-qualifier/Cargo.toml

## Next qualification stages

T1 must add a separately frozen inferential layer with participant/session/lab
clustering, randomization at the true assignment unit, a predeclared
smallest-effect-of-interest, an equivalence procedure, multiplicity control,
and a prospective power calculation. It must compare conventional frequentist
and Bayesian/hierarchical baselines under equal data and compute budgets.
No T1 inferential claim is made by this crate.
