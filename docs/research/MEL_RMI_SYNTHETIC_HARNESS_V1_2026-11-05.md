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

The crate implements fixed-response, imitation-only, one-way-prompt and cross-session yoked controls. A rule-based positive oracle applies the task's declared response transformation. The evaluator reports target-action delta, sham delta, unrelated-feature delta, input sensitivity, target specificity and synthetic response-rule agreement separately. Policy calls execute in the recorded seeded condition order, while output records are restored to canonical condition slots for paired comparison. Each condition receives a fresh policy fork restored from the same frozen pre-intervention checkpoint and receives the same explicit common-random-number seed; the trait contract forbids carrying mutable action/memory state across arms. The stateful-counter test asserts fork seeds 41,41,41,41 as well as per-fork ordinals 1,1,1,1.

The imitation control is intentionally important: copying the current partner can be input-sensitive but is not automatically considered musically complementary. The oracle relation is deliberately narrow and task-defined; it must not be presented as a human judgment or a general metric of musicality.

A second, controlled-partner path now runs all four matched agent-action conditions and reports targeted, sham and unrelated effects separately. It distinguishes a fixed partner from a responsive simulator. A joint synthetic record requires both partner-to-agent and agent-to-partner gates, including a declared reverse-direction response-rule match; input-sensitive but rule-inconsistent partner behavior fails. These results are controlled-simulator evidence, not proof of influence over musicians.

## Evidence boundary and next integration seam

The manifest binds task/session/seed, policy/version, code revision, condition order, and all input/output event identities. The contrast API and manifest independently reject output-event rebinding; the controlled-partner adapter validates input and output feature dimensions.

An optional content-evidence layer now hashes the exact bytes for each of the four input and four output event artifact sets (Score JSON, MIDI, WAV or little-endian f32 PCM) and the exact serialized Studio transaction-receipt bytes. The `sha2` implementation is pinned to 0.10.9 and the standalone lockfile is committed. A separate verification call re-reads the supplied payloads and receipt bytes, recomputes each digest/length and rejects any mismatch, event substitution, empty artifact set, duplicate representation or attempted rebinding. This is content-integrity evidence—not signer authentication, consent verification, rights, authorship, provenance of the original capture, or proof of musical quality. The verifier identity/version fields are caller-supplied and therefore are not a cryptographic attestation of the verifier binary.

The next adapter must populate these payloads from Studio-owned artifact storage after a committed MEL-013A transaction and verify the transaction signature/actor/purpose under a trusted policy. Missing or unverifiable receipts must remain unqualified; a hash of an untrusted receipt is not a trusted transaction.

The crate also represents unsupported modality metrics explicitly as NotApplicable; a PCM-only adapter cannot invent note-level MIDI evidence.

## Canonical runnable fixture suite

A frozen 160-case stress suite now exercises motif, rhythm, tempo, role and texture dimensions (32 deterministic replicates per dimension). `canonical_suite` emits a CSV of fixed, imitation, one-way, yoked and rule-based policy results, plus the reverse-direction four-condition controlled-partner contrast. It includes an unrelated-feature shortcut baseline that changes the target action for the wrong reason. The dedicated workflow asserts: rule-based positive control 160/160; fixed, imitation, one-way, yoked and distractor-shortcut controls each 0/160; and the rule-consistent responsive-partner reciprocal oracle 160/160. A stateful-counter adversarial test confirms each condition begins from the same policy checkpoint (ordinals 1,1,1,1), so a stateful implementation cannot leak condition history into the paired contrast. These are finite synthetic control counts—not SOTA results or estimates of performance with musicians.

## Authored tests

The crate authors 40 deterministic tests covering the positive oracle, fixed, imitation, one-way, yoked and distractor negative controls, sham and unrelated control integrity, intervention isolation, event identity, cross-session yoking, controlled partner influence, manifest validation, modality applicability, and exact replay equality. The 31-test, 160-fixture revision passed its exact-head run. The 40-test SHA-256 artifact-binding revision also passed its exact-head workflow (`4543e572a387c56a525efe566f74fe3eff818424`): 40 passed, 0 failed, plus the canonical 160-case five-dimension sweep and all positive/negative-control gates. The repository-wide CI, governance, benchmark and showroom workflows were skipped, not passed.

## Qualification and claim ceiling

Passing the finite oracle suite qualifies only the harness's ability to distinguish the declared synthetic policies and reject malformed fixtures. It does not establish actual Melothaea contingency, reciprocal collaboration, musician-valued outcomes, low latency, or commercial competitiveness.

The next step after this harness passes is an adapter against MEL-013A exact artifact transactions, followed by a sealed symbolic task split and only then a separately approved human pilot.
