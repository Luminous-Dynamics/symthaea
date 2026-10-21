# Native KiCad CLI smoke benchmark v0.1

## Purpose

This fixture exercises the next link beyond manifest validation: a pinned, real EDA command line must parse a native KiCad PCB, accept a clean positive control under the configured design-rule checks, reject a board with a deliberately shorted net as a negative control, and emit actual Gerber and Excellon drill files. The receipt binds the board bytes, tool image digest, reported tool version, outputs and exact Git commit.

The fixture is a small two-layer, passive continuity coupon. It is intentionally not a complete product: no schematic parity, BOM procurement check, assembly plan, test electronics, physical metrology or electrical safety qualification is implied.

## Toolchain and reproducibility

- Official `kicad/kicad:9.0.9` Linux/amd64 image, pinned by manifest digest `sha256:cef4a572df1465b98fa63cdac88780e0f8d39f03c5493434802bcd92f62123fb`.
- Native project board: `docs/research/fixtures/atomfab-kicad-cli-smoke/continuity-coupon.kicad_pcb`.
- Workflow: `.github/workflows/atomfab-kicad-cli-smoke.yml`.
- Receipt/artifacts: `atomfab-kicad-cli-smoke-receipt`.

The workflow uses `kicad-cli pcb drc` in JSON mode with violation exit codes enabled. It records a successful clean-board run, then changes the GND trace coordinates to coincide with the VCC trace and requires exit code 5 on the deliberately shorted fixture. It parses both JSON reports and exports copper, mask, silkscreen and outline Gerbers plus Excellon drill output. The workflow reruns the clean DRC report and repeats both exports to separate directories, requiring byte-identical reports/files and matching output filenames. Checksums include the design, negative-control input, reports, primary/repeated exports and receipt.

KiCad's [official CLI reference](https://docs.kicad.org/9.0/en/cli/cli.html) documents DRC's exit code (0 when no violations are found, 5 when violations exist), Gerber's per-layer output and drill export. The [official image manifest](https://hub.docker.com/layers/kicad/kicad/9.0.9/images/sha256-cef4a572df1465b98fa63cdac88780e0f8d39f03c5493434802bcd92f62123fb) identifies the pinned Linux/amd64 KiCad 9.0.9 image.

## Acceptance contract

- The native board is read by the pinned KiCad CLI.
- Clean DRC returns 0 and produces parseable JSON.
- The intentionally shorted board returns 5 and produces parseable JSON.
- Gerber and drill outputs exist and are non-empty.
- The receipt includes the exact checkout SHA, the native input-board SHA-256, pinned image identity, tool version output, the clean/negative-control exit codes, and output counts.
- Repeated outputs are tied to deterministic inputs and recorded by SHA-256; no test output grants machine authority.

## Claim ceiling and next step

A passing run proves only this bounded CAD command/export smoke contract, not that the board is manufacturable, physically conforming, safe, correctly assembled, or functionally qualified. The next step is a source-native project with schematic/netlist parity, an explicit design/acceptance profile, pinned DRC rule assumptions, a BOM/placement export contract and clean/violated fixtures. Physical claims require fabricated boards, calibrated metrology, functional tests, full denominator/sample sizes and independent inspection evidence.


## Exact-head and container write semantics

The receipt runner uses a shallow sparse checkout of the required fixture directory and checks out the pull request head SHA rather than the synthetic merge ref, so the `git_head` is the proposed source commit without downloading unrelated repository history/blobs. The pinned container runs non-root; the CI workspace's output directory is made writable before launch, and the output inventory is checksum-bound after the tool has exited. This addresses reproducibility and artifact collection only; it is not a security sandbox for untrusted board files.


The PR trigger is intentionally limited to changes in this benchmark document and its fixture directory, so adding or editing the CI contract in the parent roadmap PR does not run a smoke benchmark against a branch that lacks the board fixture.


On the first execution, KiCad found two violations on the purported clean coupon and returned its documented violation code. This is a fixture defect until the report proves otherwise; the workflow now preserves and prints full DRC diagnostics even when a gate fails, rather than stopping before the evidence is retained. The fixture is not called clean unless the pinned tool says exit code 0.


## Failure-driven hardening and replay semantics

Observed exact-head experiment on `f49ccec0bc8db0a4ed5a1ba23f9e2de6af13acc0`:

- The board passed the clean DRC control with zero violations and zero unconnected items after removing the false standard-library identity from its self-contained smoke-only footprints.
- The first invalid-board mutation returned DRC exit code 5 but was *not* a true short: the report contained zero `violations` and one `unconnected_items` entry. That is an invalid negative control, not success.
- Gerber replay exposed creation-time metadata (`%TF.CreationDate` and a generated-by KiCad date comment), so raw Gerber bytes differed across valid runs even though geometry output was expected to match.

The next run now changes the GND trace into a continuous crossover through the VCC trace, asserts a KiCad `shorting_items` violation and zero unconnected items, and compares the normalized DRC/export payload after excluding only documented time fields. Original reports/exports are still retained and raw-file SHA-256 checksums still include those timestamps. We do **not** claim raw output bytes are identical when KiCad intentionally writes wall-clock metadata; the repeatability target is identical design/export payload plus separately hashed provenance-bearing files.


The native-output gate now verifies each expected layer file explicitly: `F_Cu`, `B_Cu`, `F_Mask`, `B_Mask`, `F_Silkscreen` and `Edge_Cuts`, alongside `continuity-coupon.drl` with an Excellon `M48` header and drill tool declaration. This avoids counting six unrelated nonempty files as equivalent to a complete fabrication set.


## Proof-carrying CAM manifest

The CI bundle now includes `artifact-manifest.json`: a deterministic, path-sorted inventory of the native input board, negative-control board, KiCad version output, both clean DRC reports, the short-control report, each of the six primary Gerbers, each repeated Gerber, and both Excellon files. Each entry records a unique role, relative path, byte length and SHA-256 computed from the actual bytes. The manifest itself binds the exact Git head, board digest, pinned image identity, tool version and the bounded claim ceiling; the final receipt binds the manifest's SHA-256.

The raw files and a checksum list are preserved. For repeatability comparisons, the validator removes only known clock fields from DRC JSON, Gerber and Excellon files; raw file hashes still include those clock values. The clean DRC must have zero violations and zero unconnected items. The deliberately shorted board must have a `shorting_items` or `tracks_crossing` violation and zero unconnected items; a bare nonzero process exit is insufficient.


The machine-readable manifest deliberately uses the existing strict `atomfab-pcb-artifact-inventory-v0.1` contract: `design_id`, `design_revision`, `required_roles`, and one `{role, path, expected_sha256}` entry per artifact, with the inventory evaluator's exact authority and claim-ceiling markers. Paths are relative to the artifact directory. This keeps native EDA output inventory compatible with the independent Rust byte-hashing verifier rather than introducing a second manifest schema.


## Latest failure-driven refinement (2026-10-21)

The latest executed run confirmed the connected negative-control mutation is now doing the right thing: the shorted-board DRC report records **two `tracks_crossing` violations and zero unconnected items**. The true-short negative control is therefore supported by report semantics, not merely exit code.

The remaining replay failure came from KiCad's generated `continuity-coupon-job.gbrjob` JSON. It contains `Header.CreationDate`, which changes between otherwise equivalent exports; this is separate from the timestamp lines in the Gerber layer files. The workflow now compares `.gbrjob` files with canonical JSON after deleting only `Header.CreationDate`, compares raw Gerber layer payloads after removing the known creation-date lines, and preserves raw files plus SHA-256 hashes.

The artifact manifest includes the two job-metadata files in addition to the six primary Gerber layers, six repeat layers, two Excellon outputs, three DRC reports, native/negative board, and tool-version output: **22 uniquely role-bound files**. The manifest uses the same strict schema as the Rust actual-byte inventory gate in parent PR #8222. This rerun still needs to validate the latest code; no green native-EDA claim is made until it passes at its exact head.


## First complete native-EDA smoke pass and portable bundle follow-up

Exact-head run [#39549790807](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39549790807) succeeded for commit `ac67123180d6537235acddab89bf9452a477c7ce`. The receipt reports KiCad 9.0.9, zero clean-board DRC violations/unconnected items, two `tracks_crossing` violations on the deliberately crossed-net control and zero unconnected items there, repeat-equivalent DRC/exports after explicit date normalization, six named Gerbers, one validated Excellon drill file, and a 22-entry actual-byte manifest.

Receipt artifact: [`atomfab-kicad-cli-smoke-receipt`](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39549790807/artifacts/12322464083), uploaded ZIP digest `sha256:1ba5f03077abaa24823a33f937cd2587244ca70e81cf27db5e9b5d9c77ab881c`. The embedded receipt binds the manifest digest `0913ce50225fb1b238fdfc764193f60e8bbc2d547441febdabc2cfeee655f1b8` and the exact board/tool revision.

Reviewing the downloaded bundle then revealed a handoff issue: `SHA256SUMS` recorded runner-absolute paths, so the manifest and CAM outputs were intact but the checksum file could not verify after extraction on another host. The workflow now writes artifact-root-relative checksums and runs `sha256sum -c SHA256SUMS` before upload. The next exact-head run must prove the exported ZIP is portable, not merely that its files were generated on the hosted runner.


## Portable checksum verification

The checksum list initially used absolute paths from the hosted runner. I fixed that follow-up in the parent workflow: it now changes into the artifact root, hashes every file except `SHA256SUMS` with sorted relative paths, and immediately executes `sha256sum -c SHA256SUMS`. This adds a portability check to the exact-head run rather than assuming an artifact can be verified only on its producer machine.
