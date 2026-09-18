# EKM-058 — Mutation evidence-seal cross-component validation

## Status

Draft implementation. GitHub Actions is the executable authority; no format, compile, Clippy, test, or runtime PASS is inferred from static review or queued jobs.

## Purpose

EKM-057 decodes retained EKM-028 mutation-time evidence seals as bounded untrusted data. The sidecar checksum proves envelope integrity only. EKM-058 independently checks that the supplied sidecar and the supplied restart-v2 history describe the same mutations, revision decisions, claims, and evidence records.

The central invariant is:

> A mutation-evidence sidecar is admissible for further restart review only when its record and capsule digests can be reproduced from the independent base mutation history and the sealed evidence payload.

## Validation performed

`BeliefMutationSealWireValidator`:

- re-runs the complete EKM-043 restart-v2 semantic validator first;
- requires the seal sidecar to use the expected version and encoding;
- bounds in-memory validation to one million seal records and one million aggregate sealed evidence records;
- requires the sidecar to bind the same EKM-030 mutation capture epoch as the base restart;
- requires one seal record per persisted mutation;
- rejects duplicate mutation or revision seals and non-monotonic mutation IDs;
- binds mutation ID, source revision receipt, claim ID, seal cycle, authorization/application chronology, and final capture cycle;
- requires every referenced revision to exist and to be eligible;
- requires the sealed claim to remain semantically identical to the independent final-ledger claim;
- requires each sealed evidence record to remain semantically identical to the independent final-ledger evidence record;
- requires sealed evidence IDs to be strictly ordered and claim-bound;
- requires every frozen revision-basis snapshot to be present in the retained seal census;
- re-derives the full EKM-056 mutation-binding digest from the independent base mutation receipt, including support transition, authorization identity, authority label, and chronology;
- re-derives each EKM-056 record digest;
- re-derives the EKM-056 capsule digest and compares it with the sidecar claim.

No EKM-056 capsule is constructed from wire data.

## Critical claim boundary

Successful EKM-058 validation proves **cross-component consistency**, not independent historical-census completeness.

An untrusted actor able to replace an unsigned sidecar could omit a historical non-basis evidence record and recompute the sidecar's record/capsule digests. The independent base restart history contains the revision basis and final ledger, but not enough information by itself to prove that no additional non-basis evidence existed at the historical seal cycle.

Therefore the validation report deliberately returns:

- `cross_component_consistent = true` on successful validation;
- `historical_census_completeness_independently_proven = false`;
- `capsule_construction_authorized = false`;
- `hydration_authorized = false`;
- `activation_authorized = false`.

The stronger completeness claim requires the **original EKM-056 capsule digest** to be committed by protected restart lineage rather than supplied only by the untrusted EKM-057 sidecar.

## Tests added

`tests/ekm058_mutation_evidence_seal_validation.rs` covers:

1. a fully valid empty restart-v2 epoch plus empty mutation-seal capsule validates as cross-component consistent while keeping historical completeness false;
2. a changed claimed seal-capsule digest is rejected after independent re-derivation;
3. a changed linked mutation capture epoch is rejected.

Future non-empty historical fixtures should additionally exercise exact mutation/evidence replay once EKM-059 binds the original seal-capsule digest into protected restart evidence.

## Authority boundary

EKM-058 does not:

- construct an EKM-056 capsule from decoded bytes;
- mutate evidence, revision history, or support state;
- hydrate writable restart state;
- export writable state;
- activate restart state;
- advance trust checkpoints or verifier continuity;
- modify legacy confidence;
- modify causal/world-model/action state;
- perform file/network I/O or key custody.

## Next boundary

EKM-059 should bind the original EKM-056 mutation-evidence-seal capsule digest into a protected restart statement/checkpoint lineage. Only after that protected binding exists can a later layer treat the validated historical census as the exact census committed by the source restart state and use it to improve EKM-055 historical firewall replay.
