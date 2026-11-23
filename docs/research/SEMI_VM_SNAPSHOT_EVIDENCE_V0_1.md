# SEMI-VM-SNAPSHOT-001 — frozen consumed-label manifests and K0/K1 requalification v0.1

**Issue:** [#13444](https://github.com/Luminous-Dynamics/symthaea/issues/13444)  
**Implementation:** `symthaea-semiconductor::benchmark::vm_snapshot_evidence`  
**Status:** Rust reference implementation with authored unit tests; exact-head validation pending.

## Objective

Record the exact physical metrology labels consumed by a VM training or evaluation snapshot, bind each row to the supporting K0 evidence, and re-evaluate the same fixed set under later K1 calibration events.

The unit of qualification is not just a label, and not just a model. It is the immutable set of qualified evidence rows actually consumed by a particular model/configuration/split/preprocessing snapshot.

This is the consumer-side complement to:
- #13382 — static measurement/calibration/uncertainty qualification;
- #13412 — bitemporal calibration change and K0/K1 requalification;
- #13441 — per-label K0/K1 eligibility guard;
- #13336 — selection/score provenance where an audit policy applies;
- #13269 — process-event label lifecycle and availability chronology;
- #13273 — canonical benchmark scorecard.

## Paradigm shift: a model is qualified against the exact evidence it consumed

The snapshot receipt binds the model artifact/configuration, split manifest, preprocessing digest, information cutoff and the full consumed-label identity set.

Each consumed row records:
- label ID and revision;
- subject, measurand and measurement-event ID;
- label source-artifact digest;
- measurement digest and original qualification-report digest;
- calibration certificate identity and digest;
- K0 calibration-frontier digest;
- K0-only eligibility projection digest and state;
- optional metrology-selection receipt digest, mandatory when the input says a selection receipt is required.

The builder rejects incomplete, non-eligible or mismatched inputs. It does not silently remove a label to make a snapshot pass.

## Historical K0 projection

The upstream #13441 eligibility report carries both K0 and K1 context, so its whole-report digest is not a suitable immutable historical fingerprint when new K1 events arrive.

This module computes a K0-only projection digest by:
- retaining the K0 eligibility state and reasons;
- retaining the measurement and original qualification report digests;
- retaining the K0 frontier input digest;
- clearing only the combined requalification-report digest and the outer report digest, because they also bind K1 material.

The K0 frontier digest itself is built upstream from the K0 cutoff and only events visible by that cutoff. Later events observed after K0 therefore do not rewrite the historical projection.

This is a deliberate two-receipt design:
- the snapshot manifest is the immutable historical K0 object;
- the K1 requalification report is a new current-status receipt that references the original manifest digest.

## Time and identity safeguards

- Label measurement and availability clocks must match the declared snapshot clock domain and basis.
- v0.1 requires offset-aware timestamps because the calibration evidence uses Unix-second bounds. Source-local naive clocks are rejected until an explicit clock mapping converts them into the same time basis.
- The label measurement time must fall within the measurement's declared calibration time bounds.
- Snapshot cutoff and per-label K0 calibration frontier must agree for offset-aware timestamps.
- Every consumed label must have been measured and available by the cutoff.
- Label IDs, measurement-event IDs and active subject/measurand pairs must be unique.
- A label and its explicitly superseded predecessor cannot both be counted in the same snapshot.
- A required selection receipt must be supplied and have a valid digest.
- Every consumed label must be eligible under the declared policy at K0.
- The stored historical manifest digest is checked against its actual contents before requalification.

The initial v0.1 contract deliberately rejects duplicate subject/measurand rows. A future repeated-measures design must explicitly declare the repeat group and its statistical role rather than weakening uniqueness globally.

## K1 current-status semantics

The evaluator replays the original K0 manifest and checks the same label/evidence packets at K1.

- `CurrentEligible`: every consumed label remains eligible.
- `RebuildRequired`: one or more consumed training labels are BLOCKED at K1.
- `Indeterminate`: at least one consumed label is UNKNOWN and none are BLOCKED.
- `HistoricalOnly`: an evaluation snapshot has a blocked label; its historical result remains reconstructible but is not a current eligible evaluation set.

Per-label eligibility and reasons remain visible; aggregate counts never replace the row-level record.

## Implementation boundary

The Rust module provides:
- typed snapshot input, purpose, entry, manifest, current-disposition and requalification report;
- canonical entry ordering and deterministic manifest/report digests;
- exact model/configuration/split/preprocessing digest binding;
- offset-aware cutoff and measurement/availability checks;
- reuse of #13441 for per-label K0 and K1 eligibility;
- K0-only projection digest and explicit K1 requalification;
- duplicate label/event/subject-measurand and superseded-predecessor rejection;
- explicit false source-authenticity, snapshot-mutation, model-retraining and equipment-control fields.

It does not:
- train or retrain a VM;
- edit, rewrite or delete historical manifests/model snapshots;
- silently remove, replace or deduplicate consumed labels;
- authenticate issuers or physical source systems;
- claim fab-wide event completeness;
- authorize wafer disposition, recipe changes, APC/R2R, maintenance or equipment control.

## Authored tests

The current unit tests cover:
1. fully qualified K0/K1 set;
2. post-K0 revocation preserving the historical manifest digest while requiring a rebuild;
3. unmaterialized relevant revocation returning INDETERMINATE;
4. post-cutoff label availability rejection;
5. post-cutoff measurement rejection;
6. duplicate label and event IDs;
7. duplicate active subject/measurand rows;
8. superseded predecessor/successor collision;
9. clock mismatch and malformed selection receipt;
10. required-but-missing selection receipt;
11. source-local naive clock rejection without mapping;
12. order-independent manifest digest;
13. changed model/preprocessing digest changes the manifest;
14. stored manifest tampering rejection;
15. evaluation snapshot with blocked row becomes historical-only;
16. actual stored-manifest digest validation before replay.

Tests are authored, not yet claimed as passing. Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow are required.

## Next benchmark expansion

After the initial Rust contract passes:
- freeze JSON challenge cases for K0/K1 lineage and manifest mutation;
- add an independent reference manifest/status oracle;
- add multi-label partial-blocked/partial-unknown and mixed-event-ordering cases;
- connect selection-propensity receipts from #13328/#13336;
- define explicit repeated-measure groups with independent measurement identity;
- add a read-only training consumer adapter that records the exact consumed IDs;
- add snapshot dependency closure so a later K1 event can locate every affected model/configuration version.

## Claim ceiling

A synthetic snapshot manifest demonstrates evidence-set replay under declared inputs. It does not validate model accuracy, make current predictions safe, prove source authenticity, establish fab-wide event completeness, qualify physical metrology or authorize process control.
