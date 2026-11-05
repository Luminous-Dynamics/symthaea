# RMI-Bench v1 — deterministic synthetic harness implementation

**Protocol:** crates/domains/symthaea-muse/RMI_BENCH_V1_PREREGISTRATION_2026-11-02.md  
**Implementation:** tools/mel-rmi-synthetic/  
**Date:** 2026-11-05

## Scope

This implementation advances only the preregistered Track A harness milestone. It uses a dependency-free Rust oracle and finite symbolic feature fixtures. It does not run Melothaea, a vendor model, MIDI hardware, a human performer, or a listener study.

## Causal separation

Each fixture contains four observations sharing the same task, session, random seed, initial partner feature, and frozen history identity:

1. control — untreated partner observation;
2. targeted — the target musical feature changes;
3. sham — target feature remains unchanged;
4. unrelated — a different feature changes while the target stays fixed.

The validator rejects cross-session/history/seed leakage, duplicate event IDs, unrecorded condition ordering, and changes to protected inputs. The paired contrast therefore has an explicit finite intervention boundary rather than assuming two conveniently similar samples are counterfactuals.

## Policy controls

The crate implements fixed-response, imitation-only, one-way-prompt and cross-session yoked controls. A rule-based positive oracle applies the task's declared response transformation. The evaluator reports target-action delta, sham delta, unrelated-feature delta, input sensitivity, target specificity and synthetic response-rule agreement separately. Policy calls execute in the recorded seeded condition order, while output records are restored to canonical condition slots for paired comparison.

The imitation control is intentionally important: copying the current partner can be input-sensitive but is not automatically considered musically complementary. The oracle relation is deliberately narrow and task-defined; it must not be presented as a human judgment or a general metric of musicality.

A second, controlled-partner path now runs all four matched agent-action conditions and reports targeted, sham and unrelated effects separately. It distinguishes a fixed partner from a responsive simulator. A joint synthetic record requires both partner-to-agent and agent-to-partner gates, including a declared reverse-direction response-rule match; input-sensitive but rule-inconsistent partner behavior fails. These results are controlled-simulator evidence, not proof of influence over musicians.

## Evidence boundary and next integration seam

The manifest binds task/session/seed, policy/version, code revision, condition order, and all input/output event identities. The contrast API and manifest independently reject output-event rebinding; the controlled-partner adapter validates input and output feature dimensions. The harness does not claim to cryptographically hash arbitrary symbolic or audio payloads. The next adapter must bind those artifacts to the verified Studio transaction and actual renderer/output receipts under MEL-013A.

The crate also represents unsupported modality metrics explicitly as NotApplicable; a PCM-only adapter cannot invent note-level MIDI evidence.

## Canonical runnable fixture suite

A frozen five-case suite now exercises motif, rhythm, tempo, role and texture dimensions. `canonical_suite` emits a CSV of fixed, imitation, one-way, yoked and rule-based policy results, plus the reverse-direction four-condition controlled-partner contrast. It includes an unrelated-feature shortcut baseline that changes the target action for the wrong reason. The dedicated workflow asserts: rule-based positive control 5/5; fixed, imitation, one-way, yoked and distractor-shortcut controls 0/5; and the rule-consistent responsive-partner reciprocal oracle 5/5. These are finite synthetic control counts—not SOTA results or estimates of performance with musicians.

## Authored tests

The crate authors 29 deterministic tests covering the positive oracle, fixed, imitation, one-way, yoked and distractor negative controls, sham and unrelated control integrity, intervention isolation, event identity, cross-session yoking, controlled partner influence, manifest validation, modality applicability, and exact replay equality. The exact-head workflow on `806df7d5651df2a7edd44b8d94a033589ebe1ef2` completed successfully: rustfmt passed, 29/29 tests passed, and the canonical CSV suite passed all positive/negative-control gates. Repository-wide CI/governance/benchmark/showroom checks were skipped, not passed.

## Qualification and claim ceiling

Passing the finite oracle suite qualifies only the harness's ability to distinguish the declared synthetic policies and reject malformed fixtures. It does not establish actual Melothaea contingency, reciprocal collaboration, musician-valued outcomes, low latency, or commercial competitiveness.

The next step after this harness passes is an adapter against MEL-013A exact artifact transactions, followed by a sealed symbolic task split and only then a separately approved human pilot.
