# SCTK Pb SCDFT Legacy Replay Baseline

**Contract:** SCDFT-SCTK-REPLAY-0.1  
**Status:** upstream reference inputs/outputs pinned; local QE/SCTK replay not executed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12854

## Purpose

This is a conventional Pb SCDFT software/replay baseline, separate from the 2026 finite-center-of-mass-momentum SCDFT response bundle in #12840.

Upstream repository:
- https://github.com/mitsuaki1987/sctk
- inspected upstream main commit: 4bc883e5dfd8a2b38f47f2cb6efba26b773d5b33
- commit date: 9 October 2026
- replay fixture source: examples/Pb/README.md, examples/Pb/withSF, examples/Pb/withoutSF

The upstream Pb README says this contributed example was extracted from the parameter set used in M. Kawamura, Y. Hizume and T. Ozaki, Phys. Rev. B 101, 134511 (2020), modified from the SCTK Al/MgB2 tutorials. It reports approximately 4.4 K for SCDFT without spin fluctuations and 3.7 K with spin fluctuations. It also explicitly warns that:
- the example does **not** include spin-orbit interaction;
- the calculations require a supercomputing facility;
- the outputs include the node and parallelization requirements.

It is not the Pb calculation in the 2026 PRL finite-Q paper. That paper's arXiv-v2 table includes Pb with SOI and reports a different SCDFT Tc (5.33 K), plus xi0/lambda/Jdp observables. Do not compare these as a matched-method result.

## Frozen reference results

The exact upstream tc.out reference files were read from the pinned repository commit. Both end in a convergence marker and JOB DONE.

| Case | Spin fluctuations | Reference Tc | Converged iterations | Output termination date |
|---|---|---:|---:|---|
| withoutSF | off | 4.4251 K | 41 | 28 November 2023 |
| withSF | on | 3.7233 K | 45 | 6 December 2023 |

The upstream README's rounded values (4.4 K and 3.7 K) agree with these outputs. The reference file's exact numerical line and complete upstream Git blob SHA are in data/research/scdft_sctk_pb_legacy_replay_manifest_v0_1.json.

The withSF kel.out reference reports a substantial screened Coulomb/spin-fluctuation calculation (about 14h44 wall time in its recorded run). This is a reference environment datum, not a forecast for a modern machine.

## Important model boundary

This baseline:
- uses elemental Pb in the fcc structure at the example's ambient-pressure cell;
- is a conventional phonon-mediated SCDFT calculation;
- compares toggling the spin-fluctuation setting;
- does not include spin-orbit coupling;
- does not implement finite-Q Cooper-pair response, xi0, lambda_L, or the 2026 PRL method;
- does not establish unconventional pairing, PDW order, or multicomponent GL behaviour.

The manifest pins the current public upstream SCTK source snapshot. It does **not** assert that this is the exact SCTK source revision used to generate the 2023 reference output, and the matching QE version is not yet extracted.

## Source manifest and local Git-blob verifier

Files:
- schema: data/schemas/scdft_sctk_replay_manifest_v0_1.schema.json
- manifest: data/research/scdft_sctk_pb_legacy_replay_manifest_v0_1.json
- auditor: tools/scdft-sctk-replay-auditor

The manifest contains 31 upstream file paths and Git blob SHA-1 identifiers for:
- source README and pseudopotential;
- shared and case-specific QE/SCTK inputs;
- reference outputs for both withSF and withoutSF branches.

Git blob identity is computed as:

    SHA1("blob " + decimal_byte_length + NUL + raw_file_bytes)

This is not the ordinary SHA-1 of file bytes. The local verifier checks this canonical Git object identity, rejects unsafe relative paths, and reports exactly which files were verified.

After independently cloning the source at the pinned commit, run:

    cargo run --locked \
      --manifest-path tools/scdft-sctk-replay-auditor/Cargo.toml \
      --bin scdft_sctk_replay_audit -- \
      --schema data/schemas/scdft_sctk_replay_manifest_v0_1.schema.json \
      --manifest data/research/scdft_sctk_pb_legacy_replay_manifest_v0_1.json \
      --root /path/to/sctk-checkout \
      --out sctk-pb-local-blob-verification.json

The --root option verifies local materialization of every listed input and reference-output file. Without --root, the auditor validates the schema/manifest only and explicitly reports that local bytes have not been checked.

This command does **not** run Quantum ESPRESSO or SCTK. It only verifies source files against pinned upstream Git blob IDs.

## Calculation replay checklist

A genuine replay must be performed on a host with QE and the matching SCTK patch/build, suitable MPI/OpenMP execution and enough memory/storage for the response matrices.

Before running:
1. Verify all 31 manifest file blob IDs against a local checkout.
2. Record exact QE source commit, SCTK patch and build commit; do not assume today's SCTK main is the 2023 calculation revision.
3. Record compiler, MPI, BLAS/LAPACK, FFT, OpenMP and hardware configuration.
4. Preserve pseudopotential licensing/source and verify its blob digest.
5. Capture all run inputs, seeds, environment, stdout/stderr, exit code, runtime and peak memory.
6. Run the no-SF and with-SF cases as separate immutable executions; only the intended spin-fluctuation switch should differ in the SCDFT input contract.
7. Compare exact Tc outputs with the pinned references using a preregistered numerical tolerance. Also compare intermediate kel.out, gap convergence, and key response files.
8. Retain failed/non-converged runs; do not silently alter the input mesh or convergence threshold to force agreement.

## Acceptance ladder

- **Gate 0 — manifest integrity:** schema passes; each local Git blob matches upstream SHA-1.
- **Gate 1 — environment closure:** exact QE/SCTK/toolchain and hardware are recorded.
- **Gate 2 — reference replay:** both Pb cases complete, with iteration/energy convergence and Tc within a preregistered tolerance of 4.4251 K and 3.7233 K respectively.
- **Gate 3 — independent physical calculation:** a separately implemented or independently built reference path reproduces the relevant observables.
- **Gate 4 — 2026 source comparison:** keep this legacy, no-SOI calculation separate from the finite-Q, SOI-aware PRL/v2 dataset in #12840.

## Current evidence status

- Upstream commit and 31 file blob IDs are pinned.
- Exact tc.out convergence values were read from upstream reference files.
- The README confirms the no-SOI/HPC limitations.
- The local Git blob verifier and CI contract are source-implemented.
- No upstream checkout was materialized into a local calculation workspace.
- No QE/SCTK compilation, SCF/DFPT, Coulomb/spin-fluctuation kernel, or Tc replay was run.
- No independent reproduction is claimed.

## Claim ceiling

This is a source-pinned conventional Pb SCDFT reference manifest. It is not a new superconductivity result, not the 2026 finite-Q SCDFT method, not spin-orbit coupled, and not experimental confirmation of Pb's Tc.
