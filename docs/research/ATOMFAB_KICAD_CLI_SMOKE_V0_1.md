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
