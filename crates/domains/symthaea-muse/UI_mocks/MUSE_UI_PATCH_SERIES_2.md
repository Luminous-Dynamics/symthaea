# Muse UI Alignment Patch Series 2

**Date:** 2026-07-18
**Applies after:** Muse UI Alignment Patch Series 1
**Scope:** Versioned musical evidence, score/performance visualization, and reproducibility

## Purpose

Series 1 aligned the executable bridge with the visual language and information architecture of the Listen and Research specifications. Series 2 replaces the most important remaining visual approximations with inspectable backend data.

The governing rule is:

> A Muse visualization may be beautiful, but every structural claim must remain traceable to a score, performance, analysis, or provenance artifact.

## Patch order

### 1. `feat(protocol): add versioned piece analysis bundles`

Adds shared protocol contracts for:

- composition time, tempo, and meter;
- sections and phrases;
- exact symbolic notes;
- performed notes and source-note mappings;
- artifact hashes and bounded reproducibility claims;
- shared versioned envelopes and warnings.

### 2. `feat(studio-api): expose grounded piece bundles and provenance`

Adds:

- `GET /api/piece/{id}/listen-bundle`;
- `GET /api/piece/{id}/performance-bundle`;
- `GET /api/piece/{id}/provenance`.

The server emits score truth and rendered truth separately. It conservatively reconstructs section and phrase regions from actual score annotations, records the derivation method, and refuses to fabricate motif, harmony, cadence-type, or orchestration-role analysis.

### 3. `feat(listen): render score-grounded structure and compare views`

Adds:

- Hybrid, Form, Score, Compare, and Resonance purposes;
- real section arcs and seekable section labels;
- symbolic note rendering;
- composed-versus-performed overlays;
- current-section context during playback;
- source-note timing connectors where mappings exist.

### 4. `feat(research): link structural evidence and reproducibility`

Adds:

- observed form, section, phrase, and symbolic-note facts;
- a linked composition/performance timeline;
- selected-moment evidence using score and performance event IDs;
- accurate availability states;
- recipe, score, and audio hashes;
- engine, theory, and renderer versions;
- separate exactness flags and declared limitations.

### 5. `docs(ui): record grounded bundle foundation and remaining gaps`

Updates the implementation ledger and documents the remaining work:

- explicit composer-owned section plans;
- motif, harmony, cadence, orchestration, and identity-event bundles;
- versioned resonance methods;
- persistent Leptos stores;
- the semantic Studio edit loop.

## Verification performed

- `git diff --check`
- JavaScript syntax validation with `node --check`
- headless Chromium smoke tests with mocked versioned API responses
- visual inspection of Listen and Research at desktop resolution
- fresh application of the generated patch series to the Series 1 baseline

Rust compilation was not available in the patch-generation environment because Cargo was not installed. The protocol and server changes must therefore be compiled and tested in the canonical Nix workspace before merge.

## Merge guidance

Apply all patches in order. Do not cherry-pick only the frontend patches: the visible structure and provenance views depend on the new protocol and server endpoints.

After application, run the workspace's normal Nix verification lane, with particular attention to:

- `symthaea-muse-protocol` serialization tests;
- `muse_studio` endpoint and helper tests;
- clippy under the `studio` feature;
- browser checks against real composed candidates and every supported form.
