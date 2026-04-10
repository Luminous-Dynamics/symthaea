# Research Findings: Deep Exploration of V10 Output

## Critical Bugs Found

### 1. Trauma is Dead (SEVERITY: HIGH)
Agent `trauma_level` starts at 0.0 and is ONLY modified in `factions.rs`
during faction recruitment. Since factions are almost always 0, trauma
is always 0.0 across all 1000 years.

**Disasters don't cause trauma.** Population crashes don't cause trauma.
ECLSS failures don't cause trauma. The 6D Spinozist affect system uses
`trauma_level` as an input (line 588) but it's always 0.0.

**Fix**: Disasters should increment trauma for affected agents. Population
crashes should increment trauma for survivors. The magnitude should scale
with disaster severity and proximity (named characters get more trauma).

### 2. Phi Inversely Correlates with Population (SEVERITY: HIGH)
Phi at 4K pop: 0.224. At 18K: 0.165. At 47K: 0.149. At 71K: 0.185.

IIT predicts more integrated systems should have HIGHER Phi. The current
implementation computes per-agent Phi and averages — but per-agent Phi
doesn't increase with colony size. It may even decrease because:
- More agents → more "disconnected" agents (no social ties)
- Phi computation doesn't account for organizational structure
- The consciousness system is per-agent, not per-colony

**Fix**: Colony-level Phi should be a function of organizational integration
(governance stability × trust × faction coherence), not just mean agent Phi.
Larger, well-governed colonies should have HIGHER collective Phi.

### 3. CVS Decreases Over Time (SEVERITY: MEDIUM)
CVS at year 50: 0.680. At year 1000: 0.642. The civilization gets LESS
viable over 1000 years despite 7 tech milestones and 10x population growth.

Root cause: Phi dropping (see #2) pulls the CVS down. The 5-component
CVS formula weights Phi at 20%. A Phi drop from 0.224 to 0.149 costs
0.015 in CVS, which is most of the 0.038 total decline.

**Fix**: Fix the Phi computation (#2) and CVS will naturally improve.

### 4. Factions Almost Never Form (SEVERITY: MEDIUM)
Across 1000 years, factions appear at tick 50 and tick 1000 — that's it.
A civilization of 70,000 people across 5 worlds with independence movements
should have constant political faction activity.

Root cause: Faction recruitment requires `trauma_level > 0.3` or
`allostatic_load > 0.7`. With trauma always 0.0 and load now ~0.3,
recruitment threshold is never met.

**Fix**: Lower faction recruitment threshold OR wire trauma properly so
disasters create recruitable discontent.

### 5. Unexplored Modules (4,976 LOC)
Eight modules exist that weren't examined this session:
- `generation_ship.rs` (550 LOC) — interstellar transit
- `earth_regions.rs` (605 LOC) — 12 macro-regions
- `consciousness_epidemiology.rs` (1,118 LOC) — consciousness spread
- `interplanetary_consciousness.rs` (808 LOC) — cross-world consciousness
- `maglev_network.rs` (500 LOC) — global transit
- `spaceport.rs` (463 LOC) — launch funnel
- `empirical.rs` (282 LOC) — cited constants (well-calibrated)
- `harmony.rs` (650 LOC) — Eight Harmonies

Several of these may duplicate or conflict with systems we built.

## Positive Findings

### 1. Empirical Constants Are Excellent
`empirical.rs` contains 282 lines of NASA-cited data: SPE rates, ECLSS MTBF,
crop yields (Wheeler 1996), moonquake rates (Nakamura 1982), Mars atmospheric
composition. This is the best-calibrated module in the codebase.

### 2. Load Recalibration Produced Correct World Variance
Earth: 0.034 (comfortable, large population, full SS)
Moon: 0.148 (moderate, small population, dependent)
Mars: 0.395 (stressed, harsh environment)
This ordering is physically correct.

### 3. Narrative Bridge Transformed Output
From 41 events (80% identical) to 162 diverse events with 38 named characters,
18 projects, 35 explorations, 7 Dunbar transitions.

## Priority Fixes

| # | Bug | Impact | Effort |
|---|-----|--------|--------|
| 1 | Wire trauma to disasters | Enables faction system + affects | 30 min |
| 2 | Fix colony-level Phi | CVS stops declining over time | 1 hour |
| 3 | Lower faction threshold | Political dynamics emerge | 15 min |
| 4 | Audit unexplored modules | Eliminate duplication | 2 hours |
