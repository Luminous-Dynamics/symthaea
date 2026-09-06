# HCP-MMP1 Lineage-B run capture v1

Status: **operator-side manifest capture only; no HCP acquisition or transform execution**

This profile converts operator-selected local files into the closed-world run-manifest
schema consumed by the Lineage-B derivation mechanism in PR #523.

## Authority boundary

The capturer may:

- validate the pinned #523 method manifest;
- resolve and hash the exact fourteen local scientific inputs;
- hash the exact `wb_command` executable;
- invoke exactly `wb_command -version`;
- hash that probe's exact stdout+stderr;
- preserve `execution_id` and `authorization_reference` as descriptive execution metadata;
- emit a canonical run manifest.

The capturer may **not** download HCP/BALSA or neuromaps data, execute any scientific
Workbench transform, infer legal entitlement from `authorization_reference`, assert
scientific independence, or produce atlas/neural evidence.

## Stable capture window

Workbench identity is bound as:

`hash-before -> wb_command -version -> hash-after`

and the capture fails if those executable hashes differ.

Every scientific input is also hashed again after the version probe. If any selected
input changed during capture, no run manifest is emitted.

## Output custody

The complete manifest contains local paths and descriptive provenance, so the CLI does
**not** echo it to stdout. Stdout contains only a minimal capture profile and the SHA-256
of the completed run-manifest file.

The manifest is created atomically without overwrite through a restrictive `0600`
temporary file linked into place. Existing output paths fail closed.

## WN56 source pair

The candidate manifest is revalidated through #523's canonical `load_run()` boundary.
Therefore the exact fourteen v1 roles are required and WN56/RVVG left (`npz0`) and
right (`pkN9`) source byte roots must be distinct.

## Qualification

The dependency-free suite covers fourteen contracts:

1. valid capture;
2. version-only Workbench invocation;
3. missing role rejection;
4. duplicate role rejection;
5. unknown role rejection;
6. identical WN56 hemisphere-root rejection;
7. directory substitution rejection;
8. output no-overwrite;
9. restrictive atomic output;
10. digest-only CLI receipt with no provenance/path echo;
11. execution metadata separated from scientific roots;
12. input mutation during the version probe rejected;
13. Workbench mutation during the version probe rejected;
14. failed Workbench version probe rejected.

Successful capture still establishes neither authorized acquisition nor transform
execution. Those remain separate evidence boundaries.
