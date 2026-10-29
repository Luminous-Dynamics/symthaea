# Source-derived drift manifest validator

**Status:** schema/provenance validator only. It does not fetch datasets, implement detectors, or claim benchmark results.

This dependency-free Rust tool is the first implementation step for [issue #8829](https://github.com/Luminous-Dynamics/symthaea/issues/8829). It enforces a canonical line-based manifest for real-source semisynthetic drift experiments without copying dataset payloads into the repository.

## Commands

From the repository root:

```bash
cargo run --locked --manifest-path tools/source-derived-drift-manifest/Cargo.toml -- validate tools/source-derived-drift-manifest/fixtures/example.manifest

cargo run --locked --manifest-path tools/source-derived-drift-manifest/Cargo.toml -- canonicalize input.manifest canonical.manifest

cargo run --locked --manifest-path tools/source-derived-drift-manifest/Cargo.toml -- validate-receipt canonical.manifest result.receipt artifacts-root/
```

The canonicalizer accepts key-order variation and emits the fixed schema order. The validator rejects unknown/duplicate/missing fields, noncanonical key order or numeric formatting, non-SHA-256 digests, placeholder license values, missing HTTPS license-evidence URLs, held-out-source leakage into calibration, ambiguous zero-shift shams, inconsistent abrupt/gradual change bounds, target/feature overlap, incompatible label-reveal budgets, and unsafe artifact paths. The receipt command takes an artifact root directory and resolves the receipt's safe relative `artifact_path` beneath it before recomputing the output digest.

## Manifest contract

The manifest pins the source ID, HTTPS source URL/version, SPDX-style license identifier and its evidence URL, intended-use profile, source SHA-256, row-order semantics, canonicalized feature list and target semantics, transform family/severity and transition interval, random seed, calibration source list, held-out source, detector/config digest, policy version, and label reveal/delay/sentinel budget. The intended-use field is either research_only or commercial_compatible. As a conservative guardrail, commercial_compatible rejects obvious non-commercial license identifiers and unreviewed LicenseRef identifiers.

Change interval indices are one-based observation indices. Transform severity is normalized to the inclusive range [0, 1] with six decimal places. A zero-shift sham must explicitly declare `transform_family=none`, `transform_severity=0.000000`, `transition_shape=none`, and `NA` change bounds. For a non-sham, abrupt intervals have the same positive start/end index; gradual intervals have `0 < change_start < change_end`.

The receipt binds the run ID, SHA-256 of the canonical manifest, source SHA-256, output SHA-256, and a safe relative artifact path. `validate-receipt` recomputes the manifest and output hashes using the host `sha256sum` executable. The source hash is cross-checked against the manifest; source acquisition must separately verify the downloaded payload against that digest.

## Audit boundaries

The intended-use guard is deliberately limited: it does not resolve SPDX expressions or determine whether a license really grants the intended use, whether a supplied identifier exists in the current SPDX registry, or whether a specific transform is a derivative work. A human/source audit must verify each license evidence URL, data terms, attribution obligations, derivative-work conditions, and reuse conditions before a source is approved. The example manifest is only a schema fixture using `example.org`, placeholder hashes, and no real dataset; it is not a dataset, license decision, or experimental result.

Feature and source identifiers use canonical ASCII tokens with comma-separated sorted lists. Normalize external column names in an audited preprocessing manifest rather than relying on hidden ad hoc renaming. The initial source register under issue #8829 lists candidate streams; licensing and exact upstream source/version remain hard gates.

## Qualification

The accompanying workflow runs `cargo fmt --check`, locked tests, Clippy with warnings denied, CLI manifest validation, canonicalization parity, receipt hash verification, and negative controls. No Rust pass is claimed until the workflow reports an exact-head success.
