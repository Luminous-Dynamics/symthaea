# CEB Verifier Integrity Protocol V0

**Date:** 2026-11-02  
**Status:** Design-only protocol; no attack suite has been executed.  
**Related task:** CEB-VER-001 in `benchmarks/ceb/catalog.v0.json`  
**Tracking:** [issue #10138](https://github.com/Luminous-Dynamics/symthaea/issues/10138)

## 1. Why benchmark the benchmark

A benchmark result is a composition of two capabilities:

1. The agent solves the task.
2. The evaluator correctly distinguishes valid from invalid solutions.

An excellent agent can appear weak under a false-negative grader; a weak agent can appear excellent when the grader exposes its pass channel or leaks the answer. These are different failure modes and require different denominators.

The primary engineering score must never absorb verifier quality into one pass rate. CEB reports task-resolution results and verifier-integrity results as separate, co-equal evidence streams.

## 2. Threat model

Treat all agent-controlled artifacts as adversarial: source patches, tests, build scripts, dependency hooks, environment variables, filenames, paths, symlinks, process arguments, caches, stdout/stderr and generated reports.

The evaluator must assume an agent may try to:

- Write directly to a pass/fail flag, grader pipe, status file, named pipe or scoring endpoint.
- Read hidden tests, reference patches, seeds, expected outputs, grader source or oracle material.
- Modify test discovery, test files, grader code, build scripts, package manifests, lockfiles, caches or tool binaries.
- Exploit inherited environment variables, credentials, writable host paths, process metadata, sockets, devices, symlinks, path traversal or network access.
- Hard-code an observed fixture answer, use filename/seed leakage, exploit nondeterministic grading, or infer the answer from a supposedly hidden artifact.
- Trigger false negatives through underspecified requirements, undocumented constraints, allowed alternate implementations, locale/time dependence, resource asymmetry, flaky checks or a stale environment.
- Reuse a result from a different task, candidate patch, dataset revision, grader build, environment image, model/harness configuration or run.

This is a test plan, not a claim that any current runner is secure. CEB task execution remains blocked on the OS-level isolation prerequisite in [issue #10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064).

## 3. Attack corpus

| Family | Positive-control attack | Required negative control |
|---|---|---|
| Reward-channel tampering | Write or forge the success signal without satisfying the task | Reference patch passes through the ordinary grader path |
| Oracle/answer leakage | Extract a hidden seed, answer, patch, test or grader file | Agent cannot read sealed artifacts through paths, metadata or logs |
| Evaluator mutation | Delete/skip tests; patch grader; replace test runner; edit package hooks or caches | Protected grader/test digests remain unchanged |
| Workspace/host escape | Parent traversal, symlink escape, procfs, inherited secrets, device/socket access | Every prohibited read/write/privilege/network attempt is denied and recorded |
| False-negative scoring | Submit a semantically valid alternate patch, equivalent API, different ordering or permitted representation | Independent adjudicators agree it meets the task contract |
| Environment asymmetry | Use different resource, dependency, clock, locale or platform conditions for candidate and reference | Reference and candidate see the same declared environment contract |
| Version/provenance drift | Replay a result from a different task set, grader, image or harness | Exact task, source, grader, environment and run digests are checked |
| Receipt/report forgery | Alter claimed outcomes, hashes, task IDs, aggregates or timestamps | Independent verifier recomputes fields from retained raw artifacts |

Each attack family must contain distinct root-cause variants. One thousand trivial mutations of the same writable pass flag do not constitute one thousand independent security controls.

## 4. Qualification gates

A task/verifier pair is not eligible for benchmark use until all gates below pass:

1. **Reference gate:** independently authored reference patch passes through the same public execution path as a candidate.
2. **Empty-patch gate:** the unchanged fixture fails the task-specific acceptance condition.
3. **Adversarial gate:** every applicable attack family is exercised; known exploit variants are rejected.
4. **False-negative gate:** independently reviewed valid alternatives pass; rejected alternatives carry a concrete contract violation.
5. **Isolation gate:** filesystem, process, network, secrets, devices, resource ceilings and process-tree cleanup are tested at the OS boundary.
6. **Pristine regrade:** the exact candidate diff is applied to a fresh environment and independently rescored by an evaluator outside the candidate write domain.
7. **Reproducibility gate:** identical frozen inputs produce consistent verdicts; all variance and flaky checks remain visible.
8. **Version gate:** task, grader, image, harness, policy and dataset revisions are immutable and jointly bound to the run receipt.

A task may remain `design_only` or `fixture_ready` while any qualification gate is pending. Never promote by assuming an untested gate will pass.

## 5. Metrics

Report separate counts and rates:

- `false_accept_rate`: invalid candidate/negative-control attempts accepted.
- `false_reject_rate`: valid reference or adjudicated equivalent controls rejected.
- `attack_family_coverage`: applicable families exercised / applicable families declared.
- `verdict_reproducibility`: repeated frozen-input runs with identical verdicts / completed repeated runs.
- `pristine_regrade_agreement`: concordant primary/regrade task verdicts / tasks graded by both.
- `unresolved_rate`: tasks without an adjudicated verdict / assigned tasks.

Publish the full confusion matrix and task-level outcomes. Include missing and unresolved cases in explicit sensitivity bounds; never silently drop them from denominators.

For zero false accepts in (n) independent negative controls, publish the one-sided 95% exact binomial upper confidence bound. At (n=1{,}000), this is approximately 0.30%, not zero. Also publish counts by root-cause family and use cluster-aware uncertainty when attack instances share a common mechanism.

Do not calculate a "verifier accuracy" by treating thousands of assertions from one task as independent samples. The independent unit is the attack/task family or preregistered fixture instance, as appropriate.

## 6. Initial internal targets

These are prospective engineering gates, not achieved results or industry standards:

- Zero false accepts across at least 1,000 varied negative-control attempts, with one-sided 95% upper bound reported.
- Zero false rejects across at least 100 independently adjudicated valid positive controls; expand this corpus before making broad generality claims.
- 100% of declared applicable attack families exercised before qualification.
- 100% of published outcomes bound to immutable task, candidate, grader, image, harness and artifact digests.
- 100% of scored tasks independently regraded from a pristine environment.
- Any unresolved verifier or isolation finding blocks promotion; it is not averaged away by high task-resolution accuracy.

Targets are deliberately stricter than a simple unit-test suite. Passing them would still be scoped evidence, not proof of universal security.

## 7. Connection to external benchmarks

### SWE-Bench Pro V2

Scale's 2026-09-22 methodology documents a refreshed 642-task, 11-repository set, removal of 89 invalid tasks, reference-patch/empty-patch gating and pristine-image regrading. It also identifies residual trust boundaries: a trusted model endpoint relay and execution of patch-controlled files inside the verifier. These are strong practices to emulate, but not a substitute for our own threat model.

### Terminal-Bench 4.0.0

Epoch AI's review (2026-09-04) labels v4.0.0 flawed and reports publicly noted scoring defects in 30/66 tasks (45.5%). Representative issues include direct writes to a pass signal, answer leakage and false negatives. Terminal-Bench's maintainers describe 4.0.0 task removals/fixes and identify tamper-resistant verifiers among the aims of 4.1.

Use Terminal-Bench as a pinned diagnostic comparison, with the official maintainer leaderboard and independent third-party harnesses clearly separated. Record the affected task IDs and do not quietly erase them. A future version should be re-audited at task level before it becomes a clean headline target.

## 8. Current repository integration gap

The Rust repository-repair regrade and ablation audit documents refer to binaries under `tools/rust-repair-benchmark-harness/`. A source check of `main` on 2026-11-02 did not find the Cargo manifest or executable source files, and code search returned only those documentation references. Their implementation status is therefore unverified on the current branch. Restore the source or explicitly retire the intended interface before CEB relies on it; see issue #10138.

## 9. Claim boundary

This protocol defines an evaluation method. The CEB catalog remains design-only, no verifier attack corpus has been run, no isolation boundary is qualified, and Symthaea has no CEB performance result. Schema validity, documentation completeness and static inspection do not establish benchmark robustness or agent capability.


## 10. Machine-readable receipts and attestation

The receipt format uses the established **in-toto Statement v1 + DSSE envelope** structure rather than a bespoke signature field embedded in a CEB-specific JSON object.

Artifacts:

- `benchmarks/ceb/run-receipt.schema.json` — schema for the in-toto Statement and CEB run predicate.
- `benchmarks/ceb/dsse-envelope.schema.json` — schema for the outer DSSE envelope.
- `benchmarks/ceb/examples/design-only-unexecuted-example.statement.json` — a non-evidence example with an unresolved verdict, simulated execution, no image and no signature.

The statement binds exactly three subjects: the candidate patch, the source tree and the computed result tree. Its CEB predicate records:

- Exact task-contract digest (pinned out of band by the host task registry), fixture revision, CEB-TREE-V1 algorithm ID and source-tree digest.
- Base tree, patch digest, computed result-tree digest and changed paths; the verifier re-hashes source and result snapshots and compares the receipt's changed-path set with the independently computed tree difference.
- Model, harness, prompt and agent configuration digests.
- Runner profile, immutable image digest, network policy and execution times.
- Grader identity/digest, trust domain and hidden-test visibility.
- Required checks, their authorities, outcomes and evidence digests.
- Recomputed resource totals (including turns and tool calls) and the run-level verdict. The task-contract digest commits the declared budget; the verifier checks measured wall time, turns, tool calls, input/output tokens and cost against that exact task budget.
- The relationship to the prior primary receipt for a pristine regrade.

The DSSE envelope carries the serialized statement as base64 with payload type `application/vnd.in-toto+json` and one or more signatures. Verify the DSSE pre-authentication encoding and signatures against an explicit trust-root policy whose hash is pinned independently; a key ID by itself is not authentication. The host task registry must supply the expected task-contract hash. A user-provided file and a hash calculated from that same untrusted file do not form an authenticity boundary. The signing key must remain in the host control plane and must never be mounted into the guest or candidate process.

**Acceptance is a two-receipt decision.** An individual primary receipt may truthfully record `pass`, and an individual pristine-regrade receipt may also record `pass`; neither is final acceptance on its own. The regrade receipt must link the primary receipt digest. The independent pair verifier must confirm that both receipts bind the same task, source, patch, grader, model/harness and budget; that they have distinct run identities, signer domains, check-evidence and execution-artifact digests; that their runtime image, toolchain and runner-policy digests match the task contract; and that both verdicts agree on pass. Any required check failure, error, skip, unresolved result, safety violation, protocol violation or provenance mismatch blocks acceptance.

JSON Schema validates shape, not truth. The verifier must additionally recompute source/result tree digests, verify signatures, reconcile counters and check definitions with task rows, check task budgets and timestamp ordering, reconcile Git patch paths with the independently computed changed-path set, and ensure all referenced artifacts are present. JSON Schema's `format` annotation is not a substitute for explicit date-time validation. The schemas and validator workflow are still qualification artifacts, not a completed benchmark qualification. A standalone Rust implementation is present in draft PR [#10212](https://github.com/Luminous-Dynamics/symthaea/pull/10212); its 21 tests and build have not yet passed a complete exact-head run. Implementation and qualification remain tracked by [issue #10167](https://github.com/Luminous-Dynamics/symthaea/issues/10167).

Standards references:

- [in-toto Attestation Framework](https://github.com/in-toto/attestation)
- [in-toto envelope specification](https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md)
- [DSSE v1.0](https://github.com/secure-systems-lab/dsse/blob/master/protocol.md)



## 11. Rust verifier implementation slice (2026-11-03)

A first standalone implementation is in [draft PR #10212](https://github.com/Luminous-Dynamics/symthaea/pull/10212), stacked on schema/catalog PR #10115. It is not yet qualified and no build result is claimed.

The implementation currently includes:

- Strict JSON parsing that rejects duplicate object members, embedded Draft 2020-12 schema validation, DSSE v1 PAE construction and Ed25519 signature verification against an out-of-band pinned trust policy.
- Exact task-contract digest pinning, task-declared acceptance-check parity, recomputed required-check counters and task-budget enforcement.
- Runtime profile, OCI image digest, network policy, toolchain lock and runner-policy equality with the immutable task environment.
- CEB-TREE-V1 source and result snapshot hashing. It sorts UTF-8 relative paths and commits entry type, executable bit, byte length, file content digest or symlink-target digest. It rejects .git metadata, non-UTF-8 paths and special files; it does not follow symlinks.
- Exact reconciliation of receipt changed_paths and Git diff file headers against the independently computed source/result tree delta.
- Three in-toto subjects (patch, source tree, result tree), exact patch artifact hashing, stage-specific reports, run-generated evidence artifacts, primary/regrade independence and exact primary-envelope linkage.

The verifier now independently reconstructs the candidate result: it builds the pinned source and supplied result snapshots as Git trees in a private temporary bare libgit2 repository, parses the submitted diff with `Diff::from_buffer`, applies it with `Repository::apply_to_tree`, writes the reconstructed tree and compares its Git object ID with the tree built from the supplied result snapshot. No working tree is created, and no shell command, build, test or candidate program is run. The patch implementation is deliberately scoped to UTF-8 textual Git diffs with safe unquoted paths; binary and quoted/ambiguous path cases remain unresolved. This closes the previous path-only consistency gap for supported patches.

CEB V0 currently supports UTF-8 textual Git diffs with unquoted, safe paths and conventional hunk markers. Binary patches and ambiguous/quoted paths remain unresolved. The source and result snapshots must remain immutable/read-only during hashing and reconstruction. The verifier still does not qualify the runner's isolation, independently establish test semantic correctness, or authenticate a named model provider.

Twenty-one Rust unit tests are authored, including the signed primary/regrade positive control, patch reconstruction positive/negative controls, a wrong task pin, tampered result tree, detached regrade link, reused primary evidence, and snapshot mutation controls for executable-bit changes and symlink retargeting. They have not been run. This environment has no Cargo/Rust compiler, and no Cargo.lock has been generated. The dedicated Rust workflow is queued, not passed. The CEB contract workflow now contains 39 authored negative controls across four Draft 2020-12 schemas, including seven configuration-world controls, including seven configuration-world attacks; they are authored but have not yet passed a successful exact-head run. All ten catalog tasks remain design_only, so no score is possible.

The next gates are: generate and commit a Cargo.lock; compile, format, run all 19 tests and Clippy on the exact PR head; verify the patch dialect against an independent Git implementation; execute the 38 schema/fixture negative controls; then red-team the verifier and qualify microVM isolation in issue #10064.
