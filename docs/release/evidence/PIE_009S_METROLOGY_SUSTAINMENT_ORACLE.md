# PIE-009S metrology sustainment oracle

Purpose: freeze an implementation-independent reference for long-campaign calibration continuity.

The governing distinction is:

`measurement qualified now` != `critical measurements can remain qualified continuously`.

Each measurement channel has an explicit maximum qualified age. Recalibration is a finite industrial service with limited bench capacity and finite reference/standard uses. A recalibration started in campaign step `N` becomes effective only at the next step boundary, so it cannot retroactively repair an already-lapsed opening state.

The final candidate self-test was executed locally with Python 3 on 2026-09-13 and returned `ok`.

The synthetic fixtures demonstrate a staggered one-slot schedule that preserves two critical channels, an explicit qualification lapse when one channel is delayed, same-step non-retroactivity, single-spend standard inventory, finite bench capacity, and fail-closed duplicate/unknown calibration targets.

This reference models scheduling and resource accounting only. It does not prove physical calibration quality, drift behavior, uncertainty budgets, workforce availability, certification, economics, or real lunar/Mars metrology performance.
