# Muse UI Alignment Patch Series 3

**Date:** 2026-07-18
**Applies after:** Muse UI Alignment Patch Series 2
**Scope:** Inspectable motif, harmony, cadence, orchestration, and structural-activity evidence

## Purpose

Series 2 established separate composition, performance, and provenance bundles. Series 3 makes several of the previously unavailable musical layers real without overstating what the current engine knows.

The governing rule is:

> Every visible musical claim carries an epistemic status, a producing method, inspectable identifiers where applicable, and explicit limitations.

## Patch order

### 1. `feat(protocol): add inspectable musical evidence layers`

Bumps the composition bundle to v2 and adds backward-compatible, serde-defaulted contracts for:

- evidence status and basis;
- recipe motif definitions;
- motif occurrences and transformations;
- cadential arrival markers;
- score-side sonority regions;
- structural voice activity and orchestration regions;
- score-derived resonance samples.

Compatibility tests prove that v1 payloads remain readable with empty v2 evidence fields.

### 2. `feat(studio-api): emit motif harmony and orchestration evidence`

Adds bounded evidence producers for:

- the exact motif chosen by the resolved recipe;
- conservative score-window motif matching with source-note IDs;
- exact cadential-emphasis note binding;
- active pitch-class sets per beat;
- declared-home-key degree and function only for exact diatonic triads;
- symbolic voice activity by structural region;
- a score-derived energy, density, and motion proxy.

The endpoint emits limitations instead of claiming authored motif occurrence, modulation, cadence type, rendered prominence, or objective emotion.

### 3. `feat(listen): add inspectable musical evidence layers`

Adds Listen purposes for:

- Motifs;
- Harmony;
- Orchestration.

Also adds:

- motif and cadence markers to the Hybrid view;
- score-derived structural activity with performed-density fallback;
- a current-moment evidence card;
- evidence-aware mode availability;
- bounded explanatory language throughout.

### 4. `feat(research): add linked evidence workspaces`

Adds dedicated Research routes for:

- Motifs;
- Harmony;
- Orchestration.

The routes include linked, seekable evidence canvases, status badges, method summaries, limitations, identifiers in the explanation inspector, and accurate analysis availability. The shared timeline now shows motif occurrences and cadence markers without merging them into the performed layer.

### 5. `docs(ui): record inspectable evidence boundaries`

Updates the implementation ledger with:

- the exact claims now supported;
- the narrower meaning of each new panel;
- the remaining composer-owned and analytical data gaps;
- the next Leptos, Research-selection, performance-analysis, and semantic Studio priorities.

## Epistemic boundaries

This series deliberately does **not** claim:

- that inferred motif matches are authored occurrences;
- that a local pitch-class set establishes a key or modulation;
- that a cadential-emphasis marker determines cadence type;
- that symbolic note share equals rendered prominence;
- that score density or interval motion is objective emotion;
- that missing lineage can be reconstructed from similarity.

## Verification performed in the patch-generation environment

- `git diff --check`
- JavaScript syntax validation with `node --check`
- HTML ID uniqueness validation
- protocol compatibility tests added in source
- server helper tests added in source
- fresh application of all generated patches to the Series 2 baseline
- static frontend contract review against mocked v2 bundle shapes

Cargo and Nix are not installed in the patch-generation environment. A managed-browser policy also blocked local and `file:` preview URLs, so browser runtime checks could not be completed here. The Rust changes and browser behavior must be compiled, tested, linted, and visually checked in the canonical Nix workspace before merge.

## Merge guidance

Apply all five patches in order. The new Listen and Research views depend on composition bundle v2 and the new server evidence producers.

After application, run the normal Nix verification lane, with particular attention to:

- v1/v2 protocol deserialization compatibility;
- motif threshold and transformation behavior over representative styles;
- false-positive rates for exact home-key triad labels;
- cadence source-note binding;
- empty and partial evidence bundles;
- every Listen visualization purpose;
- each Research evidence route;
- reduced-motion and keyboard seeking;
- playback isolation from visualization work.
