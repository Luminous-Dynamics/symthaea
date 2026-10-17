<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Conventional display benchmark corpus v0.1

**Owner:** [VIS-BASE-001 #7853](https://github.com/Luminous-Dynamics/symthaea/issues/7853)  
**Corpus:** `VIS-DISP-CORPUS-001`  
**Status:** `UNQUALIFIED` — design fixtures only; no panel measurements or performance improvements claimed.

## Purpose

This corpus gives the first TV/monitor work an executable future contract. It separates four lanes:

- **Software math:** known-answer reference equations and invalid-input handling.
- **Software synthetic:** deterministic scenes, ramps and transformation comparisons.
- **Physical measurement:** named hardware + pinned mode + calibration + raw measurement + uncertainty.
- **Human task study:** consented, preregistered user tasks and statistics.

They are not interchangeable. A green sRGB/PQ/HLG calculation cannot establish measured panel luminance, HDR conformance, motion clarity, readability or comfort. The corpus includes 32 cases; every case deliberately remains `UNQUALIFIED`.

## Current standards and how they are used

- **VESA DisplayHDR CTS 1.2** is the reference vocabulary for peak and sustained luminance, black levels, local dimming, color volume/accuracy, PQ EOTF tracking, rise time, black crush and subtitle flicker. Our planned cases are *aligned to this vocabulary*, not claimed to reproduce the official pattern set or certify a product. [Criteria](https://displayhdr.org/performance-criteria/).
- **VESA ClearMR** is the motion-blur comparator. Its Clear Motion Ratio uses the ratio of clear to blurry pixels under defined test conditions. Refresh rate alone is not an acceptable proxy for motion clarity. A result only counts as ClearMR-compatible if the pinned test version and measurement method are followed. [Overview](https://vesa.org/featured-articles/vesa-brings-clarity-to-motion-blur-in-digital-displays-with-new-compliance-test-specification-and-logo-program/).
- **ITU-R BT.2100-3 (2025)** is the current in-force HDR-TV recommendation covering PQ and HLG. The contract pins the transform semantics so PQ's absolute display-referred signal mapping is not mixed with HLG's scene-referred OETF. [Recommendation](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en).
- **W3C CSS Color Module Level 4** provides the current sRGB transfer-function definition. [Specification](https://www.w3.org/TR/2026/CRD-css-color-4-20261008/).

## Files

- `visual-display-workload-corpus-v0.1.json`: 32 proposed cases with explicit evidence lanes, stimuli, expected behavior, metrics, failure modes and claim limits.
- `visual-display-workload-corpus.schema.json`: JSON Schema 2020-12 structure contract.

## Validator obligations

A first validator should:
1. Check schema validity, unique case IDs, reference-ID resolution, and `case_count == len(cases)`.
2. Reject duplicate IDs, unknown reference IDs, invalid lane/evidence-class combinations and any attempt to set status beyond `UNQUALIFIED` without new evidence.
3. Add independent known-answer oracles for software math before integrating candidate optimizers.
4. Preserve a clear distinction among software-math, software-synthetic, physical-measurement and human-study results.
5. Reject claims of official DisplayHDR/ClearMR conformance unless the complete applicable official procedure has been followed with required equipment.

## Proposed targets

- 100% pass of independently authored known-answer fixtures within declared numerical tolerance.
- Zero false accepts for invalid metadata, non-finite values, stale profile/calibration, missing raw artifacts and corpus mutations.
- Target ≥10% lower GPU time or total-system power on a declared workload **only at matched perceptual quality**; GPU time and wall power are separate metrics.
- Report p95/p99 frame time and temporal regressions; an average frame rate does not clear motion artifacts.
- Any physical comparison binds device/mode/firmware, instrument/calibration, pattern/window size, APL, duration, ambient illuminance, raw data and measurement uncertainty.

These are proposed gates, not measured results. The JSON corpus does not yet have an executable validator and the schema/corpus themselves have not been validated by the dedicated workflow in this PR.

## Negative results are first-class

A candidate fails the objective if it reduces power by crushing shadows, clips important highlights, causes subtitle-induced luminance changes, degrades moving-edge clarity, increases tail latency, or makes the test evidence incomplete. Keep per-dimension results and uncertainty visible; do not collapse them into one quality score.
