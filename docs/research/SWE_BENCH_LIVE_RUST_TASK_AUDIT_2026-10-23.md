# SWE-bench-Live Rust Task-Audit Extension (2026-10-23)

## Why this exists

The pinned upstream source revision is `62d02936a0726fbf4597b05bd07b09fb99eddae6`. In `evaluation/validation.py`, `compare()` builds:

- `PASS_TO_PASS = pre_pass & post_pass`
- `FAIL_TO_PASS = post_pass - pre_pass`

This derived-label logic creates two validity hazards. A test not present in baseline output but present and passing after the gold patch is counted as newly passing without evidence of a baseline failure. A baseline-passing test that regresses after the gold patch is removed from `PASS_TO_PASS`, because the intersection only retains tests that passed both before and after. When the validation output is reused as benchmark data, that can misstate which tests discriminate the fix and which regressions must be rejected.

Reference: [pinned upstream validation.py](https://github.com/microsoft/SWE-bench-Live/blob/62d02936a0726fbf4597b05bd07b09fb99eddae6/evaluation/validation.py).

## Independent rules

The Rust binary `rust-repair-manifest` audits the **complete candidate JSONL**, not the upstream already-filtered `validated_instances.jsonl`. It reads the source Parquet file, a candidate JSONL projection, and the per-task output directory from upstream `evaluation.validation`.

It verifies the source Parquet SHA-256 against the caller-supplied preregistered digest and exact candidate-row count. Candidate IDs must be unique and safe as filesystem components. Every candidate must have a status JSON, a non-empty baseline `pre_patch_log.txt`, and a non-empty raw `post_patch_log.txt` with each marker `eval No.0`, `eval No.1`, and `eval No.2` exactly once.

For each task, it recomputes from `pre_patch_status` and `post_patch_status`:

- **Fail-to-pass:** test is explicitly `fail` before the gold patch and explicitly `pass` in the three-run aggregate after it. A missing key or skip is not a failure observation.
- **Pass-to-pass:** every baseline-passing test remains part of the expected pass-to-pass set; it cannot be dropped merely because the gold patch regresses it.
- **Task eligibility:** baseline and gold test-ID sets must match, baseline status may not be skipped, all gold-aggregate tests must pass, the upstream F2P/P2P lists must agree with the stricter derived sets, and at least one explicit fail-to-pass transition must exist.

If validation evidence is missing or malformed for any candidate, the program writes a non-lockable audit report and exits nonzero without writing a task manifest. If evidence is complete, it emits a manifest of strictly eligible tasks and retains every rejected task with reasons in a separate audit report. It hashes the prompt/gold/test patch content and both baseline and gold logs into the manifest instead of copying prompt, patch or test contents.

## Pinned runtime evidence overlay (required before validation)

The current upstream validator discards command exit codes, ignores the boolean returns from `apply_patch`, writes only an aggregate of the three gold logs, and does not preserve each run's resolved Docker image ID. PR #8513 includes a patch against exactly SWE-bench-Live commit `62d02936a0726fbf4597b05bd07b09fb99eddae6`:

`docs/research/patches/swe-bench-live-validation-evidence.patch`

Apply and syntax-check it in a fresh clone before running gold validation:

```bash
git clone https://github.com/microsoft/SWE-bench-Live.git /tmp/SWE-bench-Live
git -C /tmp/SWE-bench-Live checkout --detach 62d02936a0726fbf4597b05bd07b09fb99eddae6
git -C /tmp/SWE-bench-Live apply --check /path/to/symthaea/docs/research/patches/swe-bench-live-validation-evidence.patch
git -C /tmp/SWE-bench-Live apply /path/to/symthaea/docs/research/patches/swe-bench-live-validation-evidence.patch
python3 -m py_compile /tmp/SWE-bench-Live/evaluation/validation.py
git -C /tmp/SWE-bench-Live submodule update --init launch
```

Run the upstream three-repeat gold validation only after the overlay is applied. It records `validation_evidence.json`, a separate `gold_run_0.log`, `gold_run_1.log`, and `gold_run_2.log` per task.

Run the full candidate split through the patched gold validator (use the complete JSONL as input; do not feed its filtered output back as the candidate denominator):

```bash
cd /tmp/SWE-bench-Live
python3 -m evaluation.validation \
  --input_dir /path/to/rust-candidates.jsonl \
  --platform linux \
  --workers 4 \
  --output_dir /path/to/validation-output \
  --overwrite 1
```

The output directory must be the same `validation-dir` passed to `rust-repair-manifest`. The Rust auditor expects each candidate's subdirectory to contain `status.json`, `pre_patch_log.txt`, `post_patch_log.txt`, `validation_evidence.json`, and `gold_run_0.log` through `gold_run_2.log`. Evidence includes actual shell exit codes, whether the test and gold patches applied, parsed status per run, raw-log SHA-256, Docker container ID, configured image reference and the immutable Docker image ID (`sha256:...`). Missing runtime metadata is not replaced with a requested image label.

The Rust auditor requires that evidence for every candidate. It verifies all three per-run logs against their SHA-256 values; checks baseline and per-run parsed maps; reconstructs the gold aggregate; rejects timeout/incomplete commands and mismatched test patch states; verifies that the four container IDs are distinct and their actual Docker image IDs agree; and binds the resolved image ID, all container IDs and the runtime evidence hash into the eligible manifest. The evaluator-only controller manifest carries both the requested image reference and the observed immutable image ID.

This instrumentation still is not cryptographic host attestation: a compromised runner can fabricate these files. It does close the observable gap between “a tag was requested” and “the Docker API reported this container/image ID and these commands returned these exit codes.” Preserve the raw files and independently rerun the validator for publication.

## Invocation

First export the full Rust split from the pinned dataset snapshot to JSONL without filtering rows. Keep the dataset task fields intact. Run the upstream three-run validation on that complete input; do not use its filtered output as the candidate denominator. Then run:

```bash
cargo test --locked --bin rust-repair-manifest --features rust-repair-manifest

cargo run --locked --bin rust-repair-manifest \
  --features rust-repair-manifest -- \
  --candidate-jsonl /path/to/rust-candidates.jsonl \
  --validation-dir /path/to/upstream-validation-output \
  --source-parquet /path/to/rust-00000-of-00001.parquet \
  --expected-parquet-sha256 02ac78e2c51a84eb174ac393bb07b77478f1f96cf260af18835a711dd8074ebc \
  --dataset-revision 3638632e8153a10ca422c1022bed79023084b5c9 \
  --upstream-code-revision 62d02936a0726fbf4597b05bd07b09fb99eddae6 \
  --expected-candidate-count 171 \
  --manifest-out build/rust-repair/eligible-task-manifest.json \
  --report-out build/rust-repair/task-audit-report.json \
  --agent-prompts-out build/rust-repair/agent-prompts.jsonl \
  --controller-manifest-out build/rust-repair/controller-manifest.evaluator-only.json
```

The source digest and candidate count above are the currently recorded pins; the binary is expected to refuse a mismatch. Do not override a mismatch to force a pass. Re-investigate the upstream snapshot/file metadata and correct the preregistration first.



## Agent prompt projection and data-leakage controls

The auditor now emits two separate artifacts only when the full candidate set has complete validation evidence and at least one task survives the strict eligibility audit:

- `agent-prompts.jsonl`: one JSON object per eligible task, containing only `problem_statement`. It contains no instance ID, repository, base commit, gold patch, test patch, hint, expected test labels, or commit URL. Line order is pinned by sorted task identity through the separate controller map.
- `controller-manifest.evaluator-only.json`: maps zero-based line number to instance ID and requested Docker image, and binds the ordered eligible-ID digest plus safe prompt projection digest. This artifact is explicitly evaluator-only and must never enter the agent's prompt, filesystem, retrieval context or tool outputs.

The unit tests serialize the projection structs and assert exact allowed fields plus absence of representative prohibited fields. The runtime-evidence tests also reject tampered per-run logs and image-ID drift across baseline/gold containers. This prevents accidental field copying at the data-contract level; it does not enforce operating-system access controls. The runner still must mount or expose only the prompt and sandbox to the agent and keep the controller/evaluator files in a separate trust domain.

## Digest contract

The manifest contains both the pinned source Parquet SHA-256 and the exact candidate-JSONL SHA-256. On a lockable audit, the tool also emits an agent-facing JSONL projection that serializes **only** `problem_statement` per line and a separate evaluator-only controller manifest containing the line index, instance ID and requested sandbox image. The report pins hashes of both projections. Never place the controller manifest, gold patch, test patch, hints, or evaluator status fields in the coding agent's context. Keep controller metadata access-separated from the agent process. Eligible task IDs are sorted lexicographically and encoded as UTF-8 IDs joined with LF and a final LF before SHA-256. `task_records_sha256` is SHA-256 over the compact serde JSON serialization of the emitted task-record array after sorting by `instance_id`. The audit report retains the exact status/log hashes and per-task reasons.

The binary deliberately does not expose raw prompt, gold patch, hidden test patch or test names in its manifest. It does not prove that the candidate JSONL was correctly generated from the Parquet; the source and projection digests allow independent regeneration and comparison. A hash proves byte identity, not source authenticity or execution. Independent reviewers must preserve and re-run the evidence.

## Validation status

The implementation includes unit tests for explicit fail-to-pass transitions, retention of baseline passes under a regression, repeated-run markers, canonical IDs and SHA-256. The authoring environment has no Rust compiler or Cargo, so these tests have **not** been executed locally. The dedicated GitHub Actions path is configured to run formatting, tests and Clippy; its result must be read at the exact head and must not be inferred from a queued workflow.
