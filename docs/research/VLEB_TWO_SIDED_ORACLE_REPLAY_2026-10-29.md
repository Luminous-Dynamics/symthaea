# VLEB reproducible two-sided oracle controls — 2026-10-29

## Motivation

An oracle should be admitted only if it distinguishes a known-bad empty baseline from a known-good reference fix. SWE-Bench Pro V2 uses this two-sided gate as part of task release, and its 2026 refresh also re-grades candidate changes on pristine images because an agent-controlled worktree can create a false pass (release: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2). The same research shows why benchmark maintenance must audit instructions, tests, source images and evaluation together rather than treating a green test result as ground truth (AgentSuite / COBA, ICML 2026: https://proceedings.mlr.press/v306/suh26a.html).

VLEB's receipt auditor already checks pinned baseline and reference logs. The new tools/vleb/verify_oracle_controls.py independently replays those controls from a locked source revision to reduce reliance on historical declarations and opaque log bundles.

## Protocol

The verifier takes:
- a local Git source repository whose HEAD is clean and equals the externally pinned 40-character source revision;
- a strict, externally pinned JSON control manifest whose commands are fixed argv arrays (never shell strings), explicit per-command environment maps, working directories, timeout ceilings, and test start/pass/fail regex markers;
- an externally pinned reference patch;
- an externally pinned ZIP containing independent tests that may add files but may not overwrite source files;
- an externally pinned verifier implementation digest and a fresh evidence output directory.

It creates two independent shallow Git repositories fetched at the pinned source commit, with no source refs and no shared object alternates; it verifies that each checkout exposes exactly one reachable commit. It disables Git hooks in both checkouts, adds identical test-bundle files, and applies the reference patch only in the reference checkout. It then runs the same frozen test commands in both arms with exactly the declared environment—ambient runner environment variables are not inherited. The result records the names and SHA-256 digest of each command environment without echoing its values. A command is considered a demonstrated test-level failure only if its test-start marker and fail-summary marker are present, its pass marker is absent, and its exit status is non-zero. A command passes only if it starts, emits the pass marker without the fail marker, and exits zero. Compile errors, missing markers, conflicting summaries and timeouts are unresolved, not kills.

The gate passes only when all baseline commands are resolved, at least one baseline command has a test-level failure, and every reference command passes. Combined command output is saved as evidence artifacts with SHA-256 digests. The report records source, manifest, test-bundle, reference-patch and verifier digests, plus the extracted test-file digests and a per-command digest of the complete execution environment. The verifier hashes the tracked source diff before and after command execution and rechecks every pinned test-bundle file; changes to tracked source or test-bundle bytes invalidate the control replay.

## Suggested test-manifest shape

The top-level fields are exactly schema_version and commands, where schema_version must be vleb-oracle-control-v1. Each command contains exactly command_id, argv, environment, working_directory, timeout_seconds, test_start_pattern, pass_pattern, and fail_pattern. Environment is a complete name-to-string map; secret-bearing variable names are rejected.

Example for Python unittest (JSON):

    {
      "schema_version": "vleb-oracle-control-v1",
      "commands": [
        {
          "command_id": "python-contract",
          "argv": ["python3", "-m", "unittest", "test_contract"],
          "environment": {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp/vleb-home",
            "PYTHONDONTWRITEBYTECODE": "1"
          },
          "working_directory": ".",
          "timeout_seconds": 300,
          "test_start_pattern": "Ran [1-9]\\d* tests?",
          "pass_pattern": "^OK$",
          "fail_pattern": "^FAILED(?: \\(|$)"
        }
      ]
    }

Use markers that correspond to actual summary lines for the selected test framework. Start patterns should exclude zero-test runs (for Python unittest, use a nonzero count such as Ran [1-9]\\d* tests? rather than accepting Ran 0 tests). A tool invocation should pin every input independently:

    python tools/vleb/verify_oracle_controls.py \
      --source-repo /trusted/source \
      --expected-source-revision "$SOURCE_REVISION" \
      --manifest oracle-controls.json \
      --expected-manifest-sha256 "$MANIFEST_SHA256" \
      --reference-patch reference.patch \
      --expected-reference-patch-sha256 "$REFERENCE_PATCH_SHA256" \
      --test-bundle tests.zip \
      --expected-test-bundle-sha256 "$TEST_BUNDLE_SHA256" \
      --expected-verifier-sha256 "$VERIFIER_SHA256" \
      --evidence-dir "$RUNNER_TEMP/oracle-control-evidence" \
      --output oracle-control-result.json

## Trust boundary and limits

The verifier is deliberately not a general sandbox. It does not enforce network isolation, container/VM isolation, CPU/memory/output limits or cryptographically signed runner provenance; those remain duties of protected evaluation infrastructure. It does not run candidate agents or grant benchmark score eligibility. The report always sets analysis_only=true, score_eligible=false, runner_attestation_required=true, and network_isolation_independently_verified=false. A valid two-sided replay is necessary oracle evidence, not sufficient proof that a task belongs in a scored benchmark.

The test bundle must be independently authored and pinned by the trusted evaluator. A weak test can pass both controls for the wrong reason, and regex markers are only as sound as the trusted manifest's chosen patterns. The verifier checks actual execution against those markers; it does not infer semantic oracle quality from the markers alone.

The integration suite creates a temporary Git repository, applies a real reference patch, and demonstrates empty-baseline failure/reference-patch success. It also checks that inverted tests, a behaviorally wrong reference patch, timeouts, bad identity pins, dirty source checkouts, path traversal, ZIP symlinks, output-directory misuse, and attempts to mutate source/test bytes during execution fail closed. These synthetic tests validate mechanics only; they are not VLEB task measurements.
