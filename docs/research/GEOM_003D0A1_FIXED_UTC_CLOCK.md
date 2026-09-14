# GEOM-003D0A1 — Fixed UTC scientific clock

Status: preregistered implementation candidate. No target-system lesion outcomes were inspected or executed to define this tranche.

Parent program: #3157 (deterministic experimental environment authority)
Implementation issue: #3159
Exact predecessor: `01c21fcad34d7c9a0651815ba40647086af67ea5`

## Purpose

The real cognitive loop reconstructs its `Biorhythm` from wall-clock UTC on the scheduled biorhythm refresh. Circadian state then influences neuromodulation and sleep/wake-dependent behavior. A matched GEOM experiment therefore cannot treat wall time as harmless telemetry.

D0A1 freezes only the UTC input to the existing chronobiology semantics. It does **not** change how timezone offset, phase offset, entrainment, circadian phase, or sleep/wake logic are interpreted.

## Boundary

The override is compiled for unit tests and for the existing `scientific_method` feature. Normal production builds retain the existing ambient-clock path and gain no new top-level `CognitiveLoopService` field.

An experiment may call:

- `install_experimental_fixed_utc_hour(hour)`
- `experimental_fixed_utc_hour()`
- `clear_experimental_fixed_utc_hour()`

No raw mutable `BiorhythmManager` or `Biorhythm` handle is exposed.

## Semantics

For a fixed UTC hour `h`, where `0 <= h < 24`:

1. Installation validates `h` and fails closed on NaN, infinity, negative values, or 24+.
2. Installation immediately rebuilds the current rhythm from `Biorhythm::for_hour(h)` and reapplies the manager timezone. This removes ambient construction time before the first normal refresh boundary.
3. Every later manager refresh uses the same fixed `h` instead of `Utc::now()`.
4. The existing refresh behavior still preserves the prior `phase_offset` and `entrainment_rate`.
5. Clearing the authority returns the manager to the ordinary `Biorhythm::current_with_tz(...)` path.

The fixed value is **UTC input**, not a claim that the stored `phase` equals local/effective circadian phase. Any correction to existing timezone/phase semantics belongs to a separate evidence lineage.

## Required controls

The candidate includes unit controls for:

- immediate installation;
- repeated-refresh stability;
- timezone preservation;
- phase-offset and entrainment preservation;
- invalid-hour rejection without partial installation;
- clearing back to ambient-clock mode.

## Scientific use

D0/D1 must record the fixed UTC hour in the pre-run environment manifest. All matched arms in one campaign use the same value unless a future independent circadian-factor experiment preregisters multiple clock conditions.

This tranche does not authorize GEOM-003D0 or D1 execution by itself. Persistence isolation, deterministic initialization, exact sham reproducibility, and the statistical authority remain separate prerequisites.
