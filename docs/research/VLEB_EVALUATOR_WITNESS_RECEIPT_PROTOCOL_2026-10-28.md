# VLEB evaluator witness-receipt protocol — 2026-10-28

## Decision

The claim-integrity gate in PR #8707 is a necessary boundary, but a hash-pinned artifact is not proof that the claimed verifier executed. The next unit of assurance is a witness receipt issued by the locked evaluator workflow, bound to one claim and one run attempt, then included in the attested evidence bundle.

This branch adds a strict receipt schema and a reference auditor for the protocol. It is deliberately not yet part of confirmatory score eligibility. Integration with the pinned run manifest and a real GitHub artifact attestation is a follow-up requirement; this code must not be represented as closing the whole assurance gap.

## Receipt contract

Each VLEB evaluator-witness-receipt-v1 records producer identity (immutable repository ID, repository name, workflow path, workflow commit SHA/ref, run ID, run attempt and event); evaluator task ID and atomic claim ID; exact source path/digest and line/column span excerpt digests; verifier kind, ID and SHA-256 identity; normalized argv (never a shell command string), working directory and names plus hashes of allowlisted environment values (not raw secrets); execution status, exit code, duration, stdout/stderr digests explicitly bound to re-hashed log artifacts, immutable runtime image identity, toolchain identity/digest, OS/architecture and observed network mode; input/output artifact inventories whose bytes are re-hashed from the extracted evidence bundle; and a three-way claim outcome: supports, contradicts or inconclusive.

The receipt intentionally does not carry the final evidence-bundle digest: embedding a receipt in a bundle while also embedding that bundle's digest in the receipt creates a self-referential hash cycle. Instead, the outer run manifest pins the bundle digest, Sigstore attestation binds that bundle to the evaluator workflow, and the receipt binds the exact run ID/attempt, task, claim, source, verifier and runtime. The auditor accepts expected values only through evaluator-supplied command-line pins, not by trusting receipt self-description.

## Trust and execution model

1. An evaluator-controlled workflow loads a task and claim from a protected, versioned task registry; the candidate cannot write the task registry, trusted verifier registry, or workflow.
2. The runner resolves a permitted verifier by out-of-band (kind, id, sha256) identity and runs it in a pinned image with network disabled or constrained by the locked task protocol.
3. The runner records exact argv, working directory, an allowlisted environment fingerprint, status/exit code/timing, log digests, toolchain identity, and hashed input/output artifacts. Secret environment values are never copied into the receipt.
4. The runner emits the receipt and packages it with the witness artifacts. The trusted workflow attests the final bundle using the declared provenance mechanism.
5. The outer manifest verifier validates attestation and bundle digest before extraction. The witness auditor then validates each receipt against out-of-band pins and re-hashes every referenced artifact.
6. Where possible, the evaluator reruns the pinned verifier in a fresh isolated environment and compares the canonical result. If rerun is not possible, an evaluator-authenticated execution receipt must be the explicit trust boundary; claimant-authored receipt JSON is never enough.
7. Confirmatory score eligibility stays false unless manifest, attestation, receipt audit and claim semantics all pass. Missing evidence stays in the original denominator and is counted unresolved.

## Security properties and limits

The reference auditor checks schema, out-of-band identity pins, run-attempt replay, task/claim/source/verifier/runtime/toolchain binding, receipt and schema digests, path traversal/symlink containment, and the SHA-256 of source/input/output files. A timeout/spawn/cancellation status may only yield inconclusive. Duplicate paths, malformed data, missing files and mismatched digests fail closed.

It does not, in isolation, prove who created the receipt or whether a malicious but trusted workflow honestly recorded the execution. Those assurances come from workflow identity and the outer attestation verification; workflow code itself must be pinned/reviewed. It also does not prove that the program's behavior is correct: the admitted verifier and task oracle must be independently reviewed, and runtime claims require actual execution evidence. Hash integrity is not semantic truth.

## Anti-replay matrix

The first test corpus rejects receipt SHA mismatch; schema SHA mismatch; old run ID; old run attempt; wrong task; wrong claim; wrong source digest; wrong workflow commit; wrong verifier identity/digest; wrong runtime image; wrong toolchain digest; mutated input/output bytes; traversal path; symlink path; absent artifact; unknown receipt fields; and non-completed executions claiming a decisive verdict.

## Acceptance gates before score integration

- Wire receipts into the evaluator-pinned claim inventory and confirmatory run-manifest verifier; no optional bypass for confirmatory runs.
- Validate a real GitHub/Sigstore-attested bundle produced by the protected workflow, not a synthetic fixture only.
- Have the evaluator itself produce at least one witness receipt under a locked run and independently rerun the pinned verifier.
- Test replay across run IDs, run attempts, tasks, claims, source revisions, workflow commits, runtime images and bundle digests.
- Add mutations for every required binding and demonstrate that the eligibility gate is killed by each mutation; keep missing and unresolved claims in the locked denominator.
- Report observed false-support with a confidence interval and sample size; never infer a true zero error rate from zero observed failures.

## Benchmark direction

VLEB should report ordinary task pass rate alongside assurance-adjusted semantic success, false-support, unresolved/coverage, independently reproduced witness rate, mutation kill rate by defect family, security-correctness rate, latency, and fully loaded cost per independently verified fix. No aggregate should hide unresolved tasks. Comparisons to external coding-agent benchmarks must use matched tasks, tools, budgets, network policies, attempts and pristine regrading; cross-benchmark leaderboard ranks are context, not proof of superiority.


## Log-digest binding hardening (follow-up)

The receipt auditor now requires stdout and stderr paths, requires each path to appear in the declared output inventory, and requires the inventory digest to equal the execution log digest. The normal artifact audit then re-hashes the actual files. This closes a gap where a receipt could carry a self-reported stdout/stderr hash unrelated to any checked bytes. Two negative controls cover digest mismatch and missing log inventory.

## Evaluator pin completeness hardening (follow-up)

The auditor now requires out-of-band an exact source-path pin plus canonical SHA-256 pins for the full source-span list, exact invocation object, and input-artifact inventory. This prevents a receipt from selecting a different but still hash-valid source span, changing arguments/working directory/environment fingerprints, or substituting another scenario input while reusing the same task and source digest. It also compares the producer's repository full name, workflow ref, and triggering event against evaluator-supplied expectations in addition to numeric repository ID, workflow path/SHA, run ID, and attempt.

Canonical pins use UTF-8 JSON serialized with sorted keys, no insignificant whitespace, non-ASCII preserved, and non-finite numbers forbidden. The protected task/claim registry must calculate these pins before the run; they must not be copied from the receipt being audited. New negative controls cover changed spans, invocation, input inventory, workflow ref/event, and repository name.


## Strict JSON parsing hardening (follow-up)

The receipt auditor now rejects duplicate JSON object keys and non-standard numeric constants such as NaN and Infinity before schema validation. Some parsers silently accept duplicate keys by retaining only the last value; rejecting them prevents different tools from assigning different meanings to the same attested bytes. An end-to-end negative control verifies duplicate-key receipts fail closed.

The source path itself is also compared to an evaluator-supplied pin; matching bytes and spans at an alternate path are not silently treated as the same claim anchor.


## Complete runtime identity pin (follow-up)

The runtime object now has its own evaluator-supplied canonical SHA-256 pin in addition to the explicit image and toolchain pins. This binds every declared runtime field—including toolchain identifier, OS, architecture, and network mode—in one comparison, so a receipt cannot change its claimed network isolation or execution architecture while leaving the image digest unchanged. Negative controls mutate network mode, architecture, and the external runtime pin.
