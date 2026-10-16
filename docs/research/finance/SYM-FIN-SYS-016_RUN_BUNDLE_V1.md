# SYM-FIN-SYS-016 — Content-addressed run bundles

**Status:** research prototype and contract proposal.  
**Claim ceiling:** deterministic synthetic research only.  
**Integration status:** separate from the existing v1 receipt validator; not yet wired into its CI workflow or the current artifact manifest.

## Problem and paradigm shift

A digest-shaped string proves nothing about the bytes it supposedly names. In the current v1 fixture, only the scenario-registry digest is recomputed from supplied repository bytes; the other 18 run-generated digest values are explicitly unverified because their payload bytes are absent.

The next step is not to mark those fields “verified” through better metadata. It is to make the run an actual, self-contained evidence bundle: preserve the receipt, exact run inputs, model/profile payloads, event payloads, final state, evaluator output, counterexamples and raw measurements as files, then recompute the digest and length of every file.

This prototype introduces:
- `SYM-FIN-SYS-016_run_bundle_manifest_v1.schema.json`: an explicit payload inventory.
- `scripts/validate_socioeconomic_run_bundle.py`: a standard-library-only verifier that hashes exact file bytes, enforces pointer coverage, rejects path traversal and symlinks, and enforces the research-only authority boundary.
- `scripts/test_validate_socioeconomic_run_bundle.py`: adversarial standard-library tests for byte tampering, manifest mismatch, missing bindings, malformed digests, duplicate JSON keys, duplicate artifact IDs, path traversal and authority escalation.

## Bundle contract

A bundle directory contains a `bundle-manifest.json`, one `receipt.json`, and the payload artifacts described by the manifest. Each payload entry declares:
- a stable unique artifact ID and a semantic role;
- a relative POSIX path;
- SHA-256 of the *exact bytes* and exact byte length;
- the JSON Pointers of receipt digest fields bound to that payload.

The receipt's `/run/artifact_manifest_digest` must equal SHA-256 of the exact manifest bytes. The receipt is not inventoried as a payload, and the manifest does not hash itself: this avoids a circular digest dependency. Every other receipt field whose key denotes a digest must be canonical SHA-256 and must be bound to exactly one artifact entry. A matching pointer is necessary but not sufficient for truth; byte integrity cannot authenticate the issuer or establish that a physical event occurred.

The verifier rejects:
- duplicate object keys and non-standard JSON constants;
- an invalid or missing manifest/receipt field;
- duplicate artifact IDs or paths;
- unsupported or malformed digest fields;
- byte-length and SHA-256 mismatches;
- missing, duplicate, extra or inconsistent receipt-pointer bindings;
- absolute paths, parent traversal, non-canonical path aliases, symlinks and files outside the bundle root;
- a receipt that enables real-world observation, authorization, reservation, settlement or production import.

Run-bundle verification complements rather than replaces the full receipt validator. The latter still owns the v1 schema and cross-field/domain semantics, such as the registered scenario, metric IDs/dimensions, event ordering and bridge topology.

## Reproduce the prototype

From the repository root:

```sh
python scripts/test_validate_socioeconomic_run_bundle.py
python scripts/validate_socioeconomic_run_bundle.py \
  --root results/run-0001 \
  --manifest bundle-manifest.json
```

These commands are instructions, not a claim that the current PR head has executed them. No test pass is claimed until output is observed against the exact head and the environment/toolchain is recorded.

## Proposed qualification targets

| Gate | Proposed target |
|---|---:|
| Supplied payload integrity | 100% of declared artifacts checked against exact SHA-256 and byte length |
| Receipt digest coverage | 100% of digest fields byte-bound; zero unbound or multiply bound fields |
| Byte tamper | 100% of one-byte payload mutations rejected |
| Manifest tamper | 100% of manifest-byte mutations rejected by receipt binding |
| Path safety | 100% of absolute, traversal, symlink and out-of-root paths rejected |
| Parser robustness | Duplicate keys, NaN/Infinity, malformed digests and invalid UTF-8 fail closed |
| Authority boundary | Zero receipts accepted if any authority flag is true, missing, or non-boolean |
| Deterministic validation | 100/100 repeat validations yield identical normalized reports |
| Performance | Proposed p95 ≤ 2 seconds for up to 100 artifacts / 250 MiB on a pinned runner; measure before ratifying |

Keep safety gates separate from performance, utilization, or economic outcome metrics. Preserve failing fixtures and mutation IDs; never silently remove a control to restore a green result.

## Next integration step

Once the prototype's unit tests are observed passing:
1. Define a versioned receipt-v2 relation to this manifest so the old source-artifact manifest and the run-payload bundle cannot be confused.
2. Add bundle artifact references for the seed, environment, shock schedule, model, regime and bridge profiles, initial/final/resource states, every event payload, evaluator and counterexample corpus.
3. Retain the v1 validator and negative controls unchanged; add distinct v2 controls and exact-byte fixture bytes.
4. Include the bundle verifier in the versioned source-artifact inventory, update every affected digest from actual bytes, and make exact-head CI emit a qualification receipt.
5. Store raw benchmark outputs and minimized counterexamples with the bundle rather than only reporting aggregate summaries.

## Claim boundary

A valid bundle proves that the supplied bytes match the declared digests and the receipt pointers are complete. It does **not** prove that the execution genuinely occurred, that a model is valid, that an issuer is authorized, that a verifier is independent, that physical evidence is true, or that any simulated economic mechanism is fair or effective. Those require separate controls and evidence.


## Research grounding and novelty boundary

The byte inventory and receipt checks should eventually fit a standard attestation structure rather than become a parallel, unauthenticated format. The [in-toto Attestation Framework](https://github.com/in-toto/attestation/tree/main/spec/v1) separates the subject/predicate statement, authentication envelope, and bundle grouping. [SLSA Build Provenance v1](https://slsa.dev/spec/v1.2/build-provenance) provides a standard description of where and how an artifact was produced, with verification against consumer expectations described in the [SLSA artifact-verification guidance](https://slsa.dev/spec/v1.2/verifying-artifacts).

This prototype's bundle verifier validates exact bytes and declared relationships, but does not verify signatures. The focused CI workflow now creates an exact-head execution report after both test suites run, checks that the checkout SHA matches the PR head SHA (recording GitHub's event SHA separately), validates both the report and a distinct custom attestation predicate against versioned JSON Schemas, uploads the report/predicate/raw test log, and generates a GitHub artifact attestation using a pinned `actions/attest` revision. The report includes source/test/schema/dependency/workflow hashes, per-suite unittest counts, and GitHub's workflow ref/SHA separately from the tested source revision and triggering event SHA. The writer fails closed if the checkout SHA, executed summaries, AST-discovered test counts, or versioned per-suite test inventory disagree. This distinguishes the tested tree's workflow file (hashed among the source inputs) from the workflow revision GitHub says actually executed. The signed predicate binds the exact report bytes and repeats key run fields that the workflow independently cross-checks.

To verify a downloaded report independently, use GitHub CLI with an explicit repository, predicate type, and signer workflow: `gh attestation verify qualification-report.json --repo Luminous-Dynamics/symthaea --predicate-type https://luminousdynamics.io/attestations/socioeconomic-run-bundle-validation/v1 --signer-workflow Luminous-Dynamics/symthaea/.github/workflows/socioeconomic-run-bundle.yml`. This verifies the associated signed attestation and the requested workflow identity; reviewers must still inspect the predicate and test output. The attestation is evidence that a specified workflow produced the report, not proof that a model is correct or a simulation is true. The future versioned design should keep these layers separate and reuse standard attestation envelopes rather than claiming the custom manifest alone proves execution provenance.


## Exact-head execution attestation follow-on

The focused workflow now has two test modules: the run-bundle verifier's adversarial controls and tests for the execution-record writer. The latter refuses to report success if GitHub's expected source revision differs from `git rev-parse HEAD`, the test log is empty, the per-suite `Ran N tests ... OK` summaries disagree with the source-discovered suite sizes, or a required source artifact is missing/symlinked/outside the workspace.

The job checks out `github.event.pull_request.head.sha` for PRs (or the event SHA for push/manual runs) and records the original event SHA separately. After schema validation, the report and raw test log are uploaded and the report receives a signed GitHub artifact attestation. On PR and push runs, signing is deliberately skipped. Signing is gated to an explicit `workflow_dispatch` on `main` and qualifies only the workflow event's own default-branch SHA. The workflow does not accept an arbitrary `source_revision` input for signing, because checking out candidate-controlled report/test-writer code and then signing its self-authored report would weaken the trust boundary. PR/push runs remain unsigned evidence reports; signing is for code already present on the trusted default branch. The workflow action and artifact uploader are pinned to immutable commit SHAs.

Proposed acceptance criteria for a signed run report:
- The attestation signature, certificate identity and signer workflow verify with `gh attestation verify`.
- The report's total and per-suite test counts agree with both AST-discovered test functions and the actual `Ran N tests ... OK` log summaries. The current versioned suite inventory is 14 run-bundle verifier tests plus 11 execution-report tests; changing it requires an intentional contract/test update.
- The custom predicate's source revision, event SHA, workflow ref/SHA, run metadata, result, count fields, non-authority boundary and report digest match the signed subject and qualification report.
- The signed subject digest matches the downloaded report bytes.
- The report source revision matches the inspected exact PR head, not merely a synthetic merge ref.
- Per-suite executed counts match the test log, authored AST counts and expected negative-control inventory.
- The raw test log, schema, test source, validator, dependency lock and workflow hashes in the report match their downloaded bytes.
- No report claims more than validator/test execution and artifact integrity; simulation truth, real-domain evidence and economic efficacy remain unproven.
