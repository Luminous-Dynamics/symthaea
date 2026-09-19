# SYM-RSI-SEM-004 — Independent Behavior Canaries

Status: **CROSS-IMPLEMENTATION PREDICTION — NOT EXECUTABLE QUALIFICATION**

Parent preregistration:

`2a11924821831137a58253ec6013ba40a63f0f3d`

Ambiguity-band amendment:

`bdd779afedc7603978009278e4a661b9d568b443`

Generator amendment:

`2182c2bb8d6f3c6d3bd72bf6c187253397f49183`

## Purpose

These values were computed independently after the complete SEM-004 protocol and generator were frozen. They predict the deterministic result of the preregistered gated-relative retrieval rule.

They must not be used to alter the SEM-004 generator, `G_D`, `C_D`, `A_D`, `R_D`, maximum set size, held-out criteria, or disposition rules.

If a future Rust implementation agrees with these canaries, SEM-004's correct scientific disposition is `NotEstablished`.

## Corpus integrity

Independent reconstruction produced exactly:

- `288` candidate memories;
- `96` labelled clean calibration queries;
- `48` ambiguous calibration queries;
- `96` unrelated calibration-control queries;
- `96` clean held-out queries;
- `72` ambiguous held-out queries;
- `72` unrelated held-out queries;
- `48` OOD held-out queries.

Across all SEM-004 partitions there were `816` exact-distinct contexts and zero internal exact-identity overlap.

The reconstruction also found zero exact BLAKE3 identity overlap with the frozen SEM-003 synthetic corpus identities.

## Frozen calibration canaries

| Dimension | `G_D` absolute gate | `G_D` bits | `C_D` clean gap | `A_D` ambiguous gap | `R_D` final band |
|---:|---:|---:|---:|---:|---:|
| 4 | `0.994384765625` | `0x3f7e9000` | `0.0` | `0.0045166015625` | `0.0045166015625` |
| 8 | `0.94873046875` | `0x3f72e000` | `0.0` | `0.00372314453125` | `0.00372314453125` |
| 16 | `0.912841796875` | `0x3f69b000` | `0.0` | `0.002197265625` | `0.002197265625` |

Relative-band bit patterns:

- 4-D `R_D`: `0x3b940000`;
- 8-D `R_D`: `0x3b740000`;
- 16-D `R_D`: `0x3b100000`.

For every dimension the clean-source gap calibration quantile is exactly zero: the designated clean calibration source is already top-1 throughout the calibration split.

The ambiguous calibration partition successfully gives the relative band a non-zero width. Both clean-calibration source coverage and ambiguous-calibration dual-parent coverage are predicted to be `1.0` in all three dimensions.

The strict `q95` unrelated-control gate admits `1 / 32 = 0.03125` calibration controls per dimension, as expected from the frozen nearest-rank rule plus strict `>` comparison.

## Held-out clean retrieval

### 4-D

- admission rate: `21 / 32 = 0.65625`;
- designated-source coverage: `21 / 32 = 0.65625`;
- overload rate: `0.0`;
- mean returned set size among admitted/non-overload queries: approximately `1.0952380952`.

### 8-D

- admission rate: `32 / 32 = 1.0`;
- designated-source coverage: `32 / 32 = 1.0`;
- overload rate: `0.0`;
- mean returned set size: `1.0`.

### 16-D

- admission rate: `32 / 32 = 1.0`;
- designated-source coverage: `32 / 32 = 1.0`;
- overload rate: `0.0`;
- mean returned set size: `1.0`.

### Aggregate

- admission rate: `85 / 96 = 0.8854166666666666`;
- designated-source coverage: `85 / 96 = 0.8854166666666666`;
- overload rate: `0.0`;
- mean returned set size among admitted/non-overload clean queries: approximately `1.0235294118`.

The aggregate clean admission and designated-source coverage therefore miss the frozen `>= 0.90` criteria.

## Held-out constructed ambiguity

### 4-D

- admission rate: `0 / 24 = 0.0`;
- at-least-one-parent inclusion: `0.0`;
- dual-parent inclusion: `0.0`;
- singleton forced-choice rate: `0.0`;
- overload rate: `0.0`.

### 8-D

- admission rate: `24 / 24 = 1.0`;
- at-least-one-parent inclusion: `1.0`;
- dual-parent inclusion: `1.0`;
- singleton forced-choice rate: `0.0`;
- overload rate: `0.0`.

### 16-D

- admission rate: `24 / 24 = 1.0`;
- at-least-one-parent inclusion: `1.0`;
- dual-parent inclusion: `1.0`;
- singleton forced-choice rate: `0.0`;
- overload rate: `0.0`.

### Aggregate

- admission rate: `48 / 72 = 0.6666666666666666`;
- at-least-one-parent inclusion: `48 / 72 = 0.6666666666666666`;
- dual-parent inclusion: `48 / 72 = 0.6666666666666666`;
- singleton forced-choice rate: `0.0`;
- overload rate: `0.0`.

The ambiguity-band mechanism itself succeeds once the query passes the absolute gate. The dominant failure is the 4-D absolute admission gate.

## Held-out unrelated retrieval

Predicted operational-guidance rates:

- 4-D: `0 / 24 = 0.0`;
- 8-D: `0 / 24 = 0.0`;
- 16-D: `1 / 24 = 0.041666666666666664`;
- aggregate: `1 / 72 = 0.013888888888888888`.

Predicted ambiguity-overload rate is `0.0` in every dimension.

The unrelated guard therefore passes comfortably.

## OOD boundary

By frozen generator construction, all `48 / 48` OOD queries leave nominal scalar support.

Expected operational support rejection rate:

`1.0`.

## Why 4-D fails

The independent top-1 similarity geometry is qualitatively different at 4-D.

With the current `SemanticContextEncoder`, every same-dimensional context bundles its four role-bound scalar components together with the **same deterministic length hypervector**.

For an even number of scalar components, that shared fifth vector resolves per-bit ties in the same direction for every context of a given width. Tie probability is largest at low width, so this shared tie-breaking influence is strongest for 4-D contexts.

The independent reconstruction observes:

- 4-D unrelated calibration top-1 similarities extending to roughly `0.995`;
- frozen `G_4 = 0.994384765625`;
- clean held-out top-1 values overlapping that gate;
- constructed ambiguous 4-D top-1 similarities below that gate.

By contrast, 8-D and 16-D unrelated top-1 similarities separate substantially from clean/ambiguous queries.

This is evidence about the frozen synthetic geometry only. It does not establish that the shared length vector is the sole cause without a separate encoder-ablation protocol.

## Expected primary disposition

SEM-004 is predicted to fail at least these frozen criteria:

- aggregate clean designated-source coverage `>= 0.90`;
- aggregate clean admission `>= 0.90`;
- ambiguous admission `>= 0.90`;
- ambiguous at-least-one-parent inclusion `>= 0.90`;
- ambiguous dual-parent inclusion `>= 0.70`.

The clean coverage criterion fails, so the expected frozen disposition is:

`NotEstablished`

## Interpretation

This negative result is more informative than SEM-003's failure.

The relative ambiguity band appears to solve the intended set-valued ambiguity problem in 8-D and 16-D while preserving very low unrelated activation. The remaining failure is concentrated in the low-dimensional absolute similarity geometry.

The appropriate next question is therefore representation-level:

> Can a semantic context composition rule preserve locality while removing same-dimension tie-breaking bias, especially at low numeric context width?

That question belongs to a new preregistered experiment. Do not retune SEM-004 after observing these values.

## Claim boundary

These are cross-implementation deterministic predictions only. They do not establish executable Rust qualification, live retrieval utility, factual truth, evidence authority, confidence, consciousness, or recursive self-improvement.
