# CEB Task Admission and Oracle Qualification V0 — 2026-11-05

**Status:** Intake contract only. No pilot tasks have been admitted or executed.  
**Parent pilot:** [`CEB-RUST-PILOT-001`](./CEB_RUST_PILOT_PREREGISTRATION_V0_2026-11-05.md)  
**Machine-readable quota registry:** [`benchmarks/ceb/task-admission-registry.v0.json`](../../benchmarks/ceb/task-admission-registry.v0.json)  
**Registry schema:** [`benchmarks/ceb/task-admission-registry.schema.json`](../../benchmarks/ceb/task-admission-registry.schema.json)  
**Per-task schema:** [`benchmarks/ceb/task-admission.schema.json`](../../benchmarks/ceb/task-admission.schema.json)  
**Validator:** [`scripts/validate_ceb_task_admission.py`](../../scripts/validate_ceb_task_admission.py)

## 1. Why task admission is a separate engineering system

A 48-task preregistration is not a 48-task benchmark. CEB needs a reproducible process for deciding which source issues become tasks, whether the prompt is answerable, whether the oracle is sound, whether the task is contaminated, and whether the evidence can be independently reconstructed.

The initial registry contains **48 planned capacity slots and zero actual task records**. The six quotas are not fabricated tasks, benchmark IDs, admitted tasks, or hidden holdout data.

## 2. Research lessons from current benchmarks

- **Rust-SWE-bench:** 500 Rust repository tasks across 34 repositories. The paper's construction starts from issue-linked merged PRs with test-file changes, requires fail-to-pass behavior, and manually filters task quality and solution leakage. This is a strong starting point for development candidates, not proof that any particular task is suitable for CEB. Source: https://arxiv.org/abs/2602.22764
- **SWE-Bench Pro V2:** 642 tasks across 11 repositories after retiring invalid tasks and rewriting contradictory task statements. It uses blind expert solving, agent endpoint-only networking, pristine-image regrading and a two-sided reference-patch-pass / empty-patch-fail gate. Source: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- **Terminal-Bench 4.0:** calibration of task resource limits, removal of saturated or unresolvable tasks, explicit task/verifier maintenance, and semantic version changes that distinguish reruns from regrades and reuse. Source: https://www.tbench.ai/news/terminal-bench-4-0

These patterns should be combined, not copied uncritically. Public Rust-SWE-bench tasks and public fix commits may seed the **development partition** after review. They should not be represented as secret holdout tasks. A sealed holdout needs new or private task material with an actual custodian boundary.

## 3. Per-task admission contract

Every task manifest binds:

1. **Repository provenance and rights:** repository identity, exact 40-hex source commit, SPDX license expression, license-file path and content digest.
2. **Task lineage:** issue/PR identity, source URL, dates, fix commit and whether the solution is public, private, unpublished or unknown.
3. **Contamination review:** dated search protocol, source surfaces searched, named reviewer and digest of the resulting review record. A completed search is evidence of a process, not proof that leakage is impossible.
4. **Prompt quality:** immutable prompt/acceptance digests and a blind instruction-only solve review.
5. **Environment reproducibility:** Rust toolchain, Cargo.lock digest, target triple, features, environment image digest, resource-policy digest, network policy, replay manifest and performance workload digest where applicable.
6. **Acceptance contract:** issue-specific fail-to-pass checks, pre-existing pass-to-pass checks, critical invariants, security review and side-effect assessment.
7. **Oracle:** two distinct blinded human gold reviewers, explicit disagreement handling, a gold patch digest, allowance for semantically equivalent repairs, and both reference-patch and empty-patch controls.
8. **Custody:** development versus holdout partition, custody-domain identifier, custody manifest digest and proof that holdout materials were outside the agent workspace before freeze.
9. **Execution qualification:** exact isolation-profile and trust-policy digests, with the #10064 and #10369 gates enforced. The V0 validator does not permit an independent-regrade claim.

## 4. Fail-closed task states

- `proposed`: metadata and candidate task exist; no qualification implied.
- `static_reviewed`: rights, provenance, leakage review, prompt review and blinded oracle review are acceptable. This state does not imply execution.
- `execution_qualified`: isolation is qualified, the reference patch passes, the empty patch fails, and the execution policy is digest-bound.
- `admitted`: the task meets the static and execution contract and independent regrade/trust custody is qualified.
- `rejected`: the record remains auditable, with its reason; it is not silently deleted from admission statistics.

A public task/fix pair may be eligible for development with explicit exposure disclosure. It cannot be called a sealed holdout. An unknown fix-visibility result blocks holdout admission.

## 5. Two-sided oracle controls

A green reference patch is not sufficient. The empty patch must fail the issue-specific contract. Both controls must be run only after #10064 qualifies the execution boundary. Before then, their status remains `not_run_blocked`.

Two independent gold reviewers should evaluate the task statement and acceptance contract without seeing candidate output. If they disagree, a third reviewer resolves the discrepancy. An LLM may assist with triage but cannot be the sole oracle.

Alternative semantically equivalent repairs are allowed. The gold patch is a control and a reference, not the only textual patch that can receive credit.

## 6. Pilot task quotas (capacity, not admitted tasks)

| Fault family | Slots |
|---|---:|
| Rust types and traits | 8 |
| Multi-crate API / regression | 8 |
| Feature and target configuration | 8 |
| Issue reproduction / debugging | 8 |
| Security correctness | 8 |
| Performance correctness | 8 |
| **Total planned** | **48** |

Current registry: **0 actual tasks; 0 admitted**.

A public Rust-SWE-bench task can be considered for development after license and provenance review, but its visible issue/fix history and benchmark-publication status disqualify it from being described as a secret holdout. For sealed evaluation, prefer newly contributed or private tasks whose fixes and acceptance tests remain under independent custody until freeze.

## 7. Current execution boundary

No candidate-controlled code, build scripts, procedural macros, tests or attack payloads were run to create this contract. The validator uses synthetic in-memory fixtures only.

- #10064 remains the prerequisite for running candidate-controlled code.
- #10369 remains the prerequisite for independent regrade and signer-custody claims.

Passing the schema and self-tests validates manifest shape and a set of state invariants. It does not prove the license, source facts, secrecy, oracle quality, runner isolation or benchmark performance.
