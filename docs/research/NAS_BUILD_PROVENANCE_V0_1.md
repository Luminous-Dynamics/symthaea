# Authenticated and independently reproducible deployment builds v0.1

Tracking: NAS-BUILD-PROVENANCE-001 / #9681  
Depends on: #9594 / PR #9595  
Coordinates with: #9598 staged qualification; #9601 physical readback; #9515 cross-layer co-design

## Thesis

An artifact digest answers **which bytes are claimed**. It does not answer **who built them**, **from which resolved materials**, **under which invocation**, or **whether an independent builder can reproduce them**.

The deployment-artifact contract in PR #9595 correctly creates content identity for a model-binary + firmware-binary pair. This proposal adds an independently verifiable provenance layer around that identity.

The unit of evidence becomes:

`source materials → authenticated build invocation → model + firmware subjects → independent rebuild comparison`

Physical flash readback and measurement provenance remain separate, later evidence gates. A signed build attestation is not proof that a device was flashed, and a reproducible build is not proof of measured performance.

## Evidence tiers (do not collapse these statuses)

| Tier | Required evidence | Permitted claim | Explicitly does not prove |
|---|---|---|---|
| B0 — content identified | #9595 artifact identity verifies | These exact model and firmware digests are bound to this pipeline/build receipt | Who built them; physical deployment |
| B1 — build authenticated | Valid SLSA v1 provenance statement; signature and signer/builder policy verify; subjects equal #9595 binary digests; required materials are resolved | An allowlisted builder attested to producing these exact outputs from these declared materials and invocation | Independent reproducibility; flash contents |
| B2 — independently reproduced | B1 from two independently administered builder trust domains; exact source/material/configuration agreement; byte-for-byte output equality | The declared build is independently reproducible under the two recorded build environments | Physical deployment; performance/accuracy |
| D1 — physically deployed | B1 or a benchmark-policy-accepted build tier plus #9601 verified readback or separately qualified attestation | The expected bytes were verified on the identified device/session | Measurement validity or fleet generalization |
| M1 — physically measured | D1 plus the #9595 artifact-bound objective receipt and the #9601 session linkage, with raw acquisition evidence available | These observations were acquired for this verified artifact on this device/session | Population-level performance outside the sampled fleet/environment |

B1 and B2 must be distinct outcomes. A build can have valid authenticated provenance while independent reproduction is not attempted or is impossible. Such a result may be retained as B1, but must not be reported as verified-reproducible. A benchmark that requires B2 must exclude B1-only artifacts from its B2 cohort.

No status may be inferred from a missing receipt. Use explicit states such as `VERIFIED`, `INVALID`, `NOT_ATTEMPTED`, `FAILED`, and `INCOMPLETE`; never encode missing evidence as a successful empty/default value.

## Provenance format and subject binding

Use the in-toto attestation envelope with the SLSA Build Provenance v1 predicate type:

`https://slsa.dev/provenance/v1`

Do not invent a parallel statement format. The verifier should parse the pinned major-version contract, reject malformed required fields, and retain the original signed envelope bytes for independent audit.

The attestation subject set must contain:
- the exact model binary SHA-256 from the #9595 build receipt;
- the exact firmware binary SHA-256 from the #9595 build receipt;
- the complete flash image/package digest when the toolchain produces a single aggregate image or signed update package.

The set is compared as a set of (name, digest) pairs; order is not identity. Missing, extra, swapped, or mismatched required subjects fail. An aggregate package subject does not replace the model and firmware subjects unless the benchmark explicitly defines the aggregate as the only deployable output.

## Required resolved materials

The provenance verifier should require an explicit material inventory. At minimum:

| Material | Required identity |
|---|---|
| Source | Repository URI, immutable commit digest, tree digest, clean/dirty state |
| Submodules and vendored trees | Each immutable commit/tree digest |
| Rust dependencies | Cargo.lock digest and resolved package/source digests |
| Nix build closure, when used | flake.lock digest and realized derivation/output identities |
| Compiler/linker/toolchain | Exact executable or toolchain bundle digest, version, target triple |
| Build scripts and generators | Script digest, generator identity/version, generated-input digests |
| Target configuration | MCU, silicon/board revision, ABI/feature flags, linker script and memory layout |
| Model inputs | Architecture/config digest, weights, quantization and calibration artifacts |
| Operator/runtime dependencies | Exact source/binary digests and selected kernel variants |
| Invocation | Build type, evaluated configuration digest, command/arguments and output-affecting environment |

All output-affecting inputs must be enumerated or the record is `INCOMPLETE`. A version string alone is not a material digest. A mutable tag, branch name, floating dependency range, or unpinned container tag is not an immutable dependency.

Secrets must never be copied into public provenance. If a secret can affect generated bytes, record an approved opaque commitment and its trust-policy reference; otherwise mark the build non-reproducible/incomplete. Do not pretend that a digest of an unknown secret proves that the secret was correctly supplied.

## Builder authentication policy

A signature is useful only relative to an explicit trust policy.

The policy must be a separately content-addressed, versioned input and must pin:
- accepted signer identity / certificate or key identity;
- accepted builder identity and builder version;
- accepted build type and workflow/source constraints;
- trust-root digest and validity/revocation evidence;
- policy version and verifier version.

The verifier must check the signed envelope, not a detached caller-provided boolean such as `signature_valid: true`. It must bind signer and builder identities to the policy, verify statement subject digests, and check the expected repository/commit and build invocation. Unknown, expired, revoked, untrusted, or mismatched identities fail closed.

A self-hosted builder is not automatically trusted because it signs its own output. Builder identity describes the trust base that the consumer has explicitly chosen to accept.

## Independent rebuild protocol

A B2 comparison requires two builds that are independent in administration and credential/control-plane trust, not merely two jobs on the same runner pool.

Both attestations must agree on:
- source tree and immutable repository commit;
- submodule and dependency materials;
- build type and evaluated invocation parameters;
- target triple, compiler/linker and toolchain digests;
- model weights, quantization, calibration and kernel selections;
- all output-affecting environment parameters.

The verifier compares each required output digest byte-for-byte. The result is one of:

- `EXACT_MATCH`: both builders produced every required subject with identical digests;
- `DIFFERENT`: one or more required output bytes/digests differ;
- `FAILED`: a required build or verification attempt failed;
- `NOT_ATTEMPTED`: no second independent build was performed;
- `INCOMPLETE`: input or output provenance is missing.

Only `EXACT_MATCH` creates a `VerifiedReproducibleBuild` capability. Do not silently normalize timestamps, metadata, padding, section order, signatures, or binary regions. If a target requires a documented non-reproducible field, the normalization must be specified and frozen before evaluation, preserve all executable/loadable bytes exactly, and produce a separately named result such as `NORMALIZED_MATCH`; it must never be relabelled `EXACT_MATCH`.

A second builder that shares the same administrator, signing key, mutable dependency cache, or privileged control plane is not independent for the affected trust claim. Independence should be recorded as evidence and evaluated against a preregistered policy, not inferred from different machine names.

## Verification algorithm

1. Read the original #9595 pipeline and build receipts; recompute their canonical digests.
2. Extract the exact model/firmware subject digests and the pipeline's benchmark, candidate, target, compiler, runtime and quantization identities.
3. Parse the in-toto envelope and SLSA v1 predicate; preserve raw envelope bytes and reject malformed required fields.
4. Verify the signature against the pinned trust root and policy; establish the signer/builder allowlist match.
5. Verify that the statement subject set equals the exact #9595 outputs.
6. Verify source URI/commit/tree, submodules, lockfile, toolchain, build scripts, target configuration, model/calibration inputs, and all resolved dependency commitments.
7. Verify build type, invocation digest, output-affecting parameters, builder identity and run identity.
8. Produce B1 only when all required provenance checks pass.
9. For B2, verify the second builder's independent trust domain and repeat steps 3–8; compare the two source/material/invocation identities and all output digests byte-for-byte.
10. Emit a deterministic verification receipt containing policy/verifier digests, attestation digests, verified subjects, resolved-material set digest, builder identities, outcome, and explicit failure codes.

Verification must be possible offline from exported signed envelopes, pinned public trust material, policy, source/material manifests, and output binaries. The verifier must not need access to a private CI control plane merely to recompute the result.

## Adversarial control matrix

| ID | Mutation or attack | Expected result |
|---|---|---|
| BP-001 | Change a byte in the signed statement | Signature verification fails |
| BP-002 | Unknown signer or builder | Policy rejects |
| BP-003 | Valid signature from a signer not allowed for this builder | Policy rejects |
| BP-004 | Substitute source commit/tree | Material binding fails |
| BP-005 | Substitute Cargo.lock / flake.lock / resolved dependency | Material binding fails |
| BP-006 | Use dirty or untracked output-affecting source | B1 rejected or explicitly INCOMPLETE |
| BP-007 | Omit a build script, generator, or environment input | Material completeness fails |
| BP-008 | Model and firmware subjects swapped | Subject-set comparison fails |
| BP-009 | Add, remove, or alter a required output subject | Subject-set comparison fails |
| BP-010 | Claim two independent builds from one trust domain | B2 independence check fails |
| BP-011 | Change one output byte on builder 2 | Outcome is DIFFERENT |
| BP-012 | Second build is missing or failed | NOT_ATTEMPTED / FAILED, never B2 |
| BP-013 | Replay a run identity where uniqueness is required | Replay policy rejects |
| BP-014 | Revoke or expire the trust root after a policy-defined cutoff | Verification reflects the frozen revocation/validity policy |
| BP-015 | Alter provenance policy or verifier digest after the run | Verification receipt/policy binding fails |
| BP-016 | Present a correct build attestation but no physical readback | B1/B2 may pass; D1 must not be issued |

Every control must be executed against the independent verifier. A test that only checks a helper function is insufficient; at least one end-to-end receipt path must prove that the final capability is not issued.

## Benchmark and reporting plan

### Phase A — fixture-level verification

No hardware is required. Build synthetic attestations and binaries with deterministic fixtures. Exercise every adversarial control, and test verifier agreement between two independent implementations or languages. Keep all cryptographic trust roots test-only and clearly marked.

### Phase B — one reproducible MCU build

Choose one NATS-Bench-MCU v2 architecture that is already in the deployable 14,281 cohort. Pin the source, firmware, quantization, compiler, toolchain, and environment. Run independent builders and check exact output equality before any performance claim.

### Phase C — matched NATS-Bench-MCU cohort

Rebuild a preregistered subset, then expand to the full 14,281 deployable rows if resource and licensing constraints permit. Keep the 1,344 flash-infeasible architectures in the outcome table; never assign them fabricated energy/latency values. Use the corrected v2 export and correction overlay as the data authority.

### Phase D — MLPerf Tiny

Use the current MLPerf Tiny rules and keep Closed/Open division claims separate. The Closed division is the stronger apples-to-apples comparison when the reference model is required; Open division allows model/retraining changes and must not be mixed into the same leaderboard.

### Report separately

- authenticated provenance coverage;
- resolved-material completeness;
- signer and builder-policy rejection rate;
- independent-builder trust-domain diversity;
- exact rebuild rate and normalized-match rate (if separately allowed);
- mismatch and build-failure rates;
- verification time and provenance storage overhead;
- end-to-end cost per B1 and per B2 artifact;
- deploy/readback and measurement yield after B1/B2;
- accuracy, p50/p95/p99 latency, energy/inference, flash, static RAM, tensor arena, stack, heap and worst-target performance.

Do not publish a single weighted score that hides failed or unreproducible builds. Keep `FAILED`, `NOT_ATTEMPTED`, `INCOMPLETE`, and `DIFFERENT` in the candidate funnel.

## Research targets (not achieved results)

- 100% of promoted B1 artifacts have authenticated provenance for all declared output-affecting materials.
- 100% of promoted B2 artifacts have two independent builder attestations and exact subject equality.
- Zero accepted BP-001 through BP-016 negative controls.
- At least 2× lower total resource cost per B2 + D1 + M1 deployment than the strongest matched baseline, at equal quality/target constraints.
- No performance claim may be attributed to provenance verification alone; report the time/cost overhead separately.

## Claim ceiling

- A digest is identity, not authentication.
- A signature authenticates an attestation under a trust policy; it does not prove the builder is honest outside that policy.
- Independent byte equality supports reproducibility for the declared materials and two builders; it does not prove all possible builders will agree.
- Neither B1 nor B2 proves physical flash contents. That is #9601.
- Neither B1 nor B2 proves measurement validity. That requires readback-coupled acquisition and objective receipts.
- No SOTA or superiority claim is justified until matched baselines, target constraints, total resource accounting, and independent verification are published.

## Sources

- SLSA Build Provenance v1: https://slsa.dev/spec/v1.0/provenance
- SLSA build provenance specification: https://github.com/slsa-framework/slsa/blob/main/spec/build-provenance.md
- SLSA reproducible versus verified-reproducible builds: https://slsa.dev/spec/v1.0/faq
- MLPerf Tiny benchmark and current suite overview: https://mlcommons.org/benchmarks/inference-tiny/
- MLPerf Tiny reproducibility rules: https://github.com/mlcommons/tiny/blob/master/benchmark/MLPerfTiny_Rules.adoc
- NATS-Bench-MCU v2 corrected dataset: https://zenodo.org/records/21644470

