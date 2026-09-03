# Spore Boot Ecology v0.3.3 — Temporal Visual Evidence Contract

Status: **stacked evidence tranche; no renderer authority**

This document defines how exact Spore frames should be reviewed over time. It does not define boot truth, health, readiness, Last Known Good, DRM ownership, activation, or aesthetic acceptance.

## Why this exists

The existing Spore visual gate already provides unusually strong raw evidence:

- exact CPU renderer pixels from the DRM/KMS path;
- 16 factual lifecycle cases;
- serialized `BootStateReceipt`, `MorphologyLineage`, and `BootGenome` inputs;
- lifecycle and inoculation galleries;
- descriptive pixel lint;
- renderer-cost evidence;
- sealed evidence manifests and hashes.

What it does not yet provide is a stable answer to **which moments should be compared** when judging choreography and hierarchy.

A midpoint thumbnail is insufficient. A raw animation loop is useful to a human but difficult to compare reproducibly across versions. Sampling by absolute frame number is also wrong because semantic stage durations can change.

v0.3.3 therefore adds a semantic-time evidence layer.

## Core rule

Temporal comparison is aligned to the serialized `BootGenome.stages`, not to arbitrary frame indices.

For each lifecycle case, evidence requests:

1. sequence start;
2. the midpoint of every `BootStage` in stage order;
3. sequence final.

Each requested semantic time is matched to the nearest **already captured exact renderer frame**. Ties choose the earlier frame deterministically.

The evidence record stores both requested and actual time. Approximation is never silently presented as exact.

## Why stage midpoint

One midpoint per stage is deliberately modest.

It is enough to show whether the stage has a distinct visual hero and whether secondary layers support or compete with that hero, while avoiding hundreds of review frames or a fake automated aesthetic score.

When a specific transition needs deeper study, a later focused tranche may add stage-entry/stage-exit samples for that transition only. The default evidence contract should remain cheap.

## Required per-sample identity

Every semantic sample records:

- stable sample key;
- role (`sequence-start`, `stage-midpoint`, or `sequence-final`);
- stage index and `BootStageKind` where applicable;
- serialized visual-only stage intensity;
- requested elapsed milliseconds;
- actual captured elapsed milliseconds;
- timing error milliseconds;
- whether semantic time was exact;
- exact source-frame path;
- SHA-256 of the complete PPM file.

This makes the contact sheet an index into evidence rather than an untraceable screenshot collage.

## Descriptive metrics, never beauty scores

The temporal tool computes only fixed, interpretable pixel descriptors:

- mean luminance;
- p95 luminance;
- fraction above a fixed bright-luma threshold;
- fraction above a fixed very-bright threshold;
- fraction above a fixed near-black threshold;
- normalized luminance-weighted centroid above a fixed threshold.

Thresholds are written into every metric record.

These values may answer questions such as:

- did Handoff actually become quieter over time?
- did a recovery focal event move attention into a local region?
- did Rich increase secondary visual occupancy without changing semantic identity?
- did a change accidentally make every stage equally bright?

They must **not** be combined into a scalar quality score, pass/fail beauty threshold, or optimizer target.

A renderer can become aesthetically worse while making any one of these numbers look 'better'. Human temporal review remains authoritative for composition quality.

## Contact sheets

Each case receives one deterministic contact sheet in both PPM and PNG form.

The sheet contains the selected exact source pixels in semantic-sample order with black gutters only. No labels, arrows, generated annotations, resampling, or image enhancement are applied to the pixels.

The accompanying JSON provides the semantic labels and exact source hashes.

This keeps visual evidence mechanically simple and lets a reviewer inspect composition without changing the thing being reviewed.

## Terminal-frame honesty

The current `render_preview` implementation computes:

`frame_count = ceil(duration_ms * fps / 1000)`

and begins capture at frame index zero.

Therefore the exact sequence endpoint is not guaranteed to be present. For example, a 2,000 ms preview at 2 fps naturally captures 0, 500, 1,000, and 1,500 ms under that count, not 2,000 ms.

The temporal evidence layer does not synthesize the missing endpoint. Instead it records:

- `terminal_frame_exact`;
- `terminal_timing_error_ms`.

This matters especially for v0.3.3 standalone Handoff, whose contract resolves below the renderer work threshold at the final semantic instant.

A future **capture-only** patch should explicitly include the terminal renderer frame. That patch must be applied symmetrically to control and treatment evidence generation before exact endpoint comparisons are used as a visual claim.

## Control versus treatment

The intended experiment remains:

- **control:** #301 — v0.3.2 renderer semantics, mechanical rustfmt reconciliation only;
- **treatment:** #300 — v0.3.3 perceptual composition policy and exact renderer integration.

The same temporal evidence tool and sampling rule must be used for both sides.

Comparison should pair samples by lifecycle case + semantic sample key, not by filename or raw frame index.

## Review questions

For each paired contact sheet, a human reviewer should answer a small set of concrete questions:

1. **Hero:** Is the intended semantic event visually dominant without reading the label?
2. **Competition:** Are unrelated holography, caustics, bloom, mesh, or identity elements stealing attention?
3. **Continuity:** Does existing morphology remain recognizable where the semantic event should preserve identity?
4. **Restraint:** Does the event use negative space and local contrast rather than global brightness?
5. **Causality:** Does motion/growth/retraction visually follow the event being represented?
6. **Resolution:** Does Handoff monotonically simplify rather than ending in a crowded or abrupt frame?

These are review prompts, not machine gates.

## Cost budget

Temporal evidence is intentionally a postprocessor over frames the workflow already renders.

It introduces:

- no additional boot runtime work;
- no renderer dependency;
- no network dependency;
- no image library dependency;
- no extra source of boot truth;
- no aesthetic threshold.

The only incremental CI cost is reading existing PPM files, computing simple descriptors, writing contact sheets/JSON, and sealing those outputs with the existing evidence artifact.

## Exit criteria for the evidence tranche

The temporal evidence layer is ready to join the focused Spore workflow when:

- its standard-library self-test passes;
- it rejects malformed/mismatched frame manifests;
- sample selection is deterministic;
- exact semantic timestamps are distinguished from nearest-frame approximations;
- contact-sheet pixels are byte-derived only from exact source PPM pixels plus fixed black gutters;
- all metric thresholds are explicit;
- policy text states that metrics are descriptive and not aesthetic scores;
- control and treatment can use the same schema without version-specific logic.

No physical-host activation is authorized by this evidence tranche.
