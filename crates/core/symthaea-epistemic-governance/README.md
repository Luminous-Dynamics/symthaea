# Symthaea Epistemic Governance

This crate implements policy for Recursive Cognitive Architecture (RCA) evidence handling.

It is deliberately **not** another evidence store and **not** an action-authority layer:

- `symthaea-types` owns canonical cognitive proposal/evidence wire types;
- `symthaea-evidence-plane` owns research-harness mechanical integrity, task validation, and seed discipline;
- `symthaea-epistemic-governance` owns lineage, independence, currentness, defeaters, and experiment qualification policy.

The crate must not grant `ActionAuthority` or `SelfImprovementPromotion`.

## RCA-001.1 invariant

Multiple evidence objects do not imply multiple independent observations. Independence is computed only from a closed, acyclic ancestry graph. Missing ancestry fails closed instead of being interpreted as independent corroboration.

## RCA-001.2 invariant

Historically valid evidence is not automatically current evidence. Producers never declare a `current` boolean; currentness is assessed at the point of use against explicit time and source/model/environment generations. Dynamic evidence with no expiry or generation boundary is invalid.

Typed evidence relations preserve support, contradiction, weakening, defeaters, supersession, corroboration, and irrelevance without collapsing them into a single support score. A `Corroborates` relation does not establish independence; lineage remains authoritative for that question.

## RCA-001.3 invariant

Experiment success criteria are frozen before result-bearing execution. `RegisteredExperimentContractV1` cryptographically commits the hypothesis, baseline/candidate identities, development and held-out corpora, evaluator, metrics, minimum meaningful effect, confidence criterion, resource ceilings, falsification criteria, allowed interpretations, and the existing evidence-plane seed plan.

The contract uses BLAKE3 commitments. It deliberately does not treat `symthaea-evidence-plane::config_hash()` as an authority commitment because that helper is documented as a non-cryptographic identity fingerprint. Any post-registration mutation of committed fields causes integrity verification and deserialization to fail.
