# Battery inverse design and experiment selection v0.1

Status: research contract and synthetic fixture seed. Not a battery design, synthesis instruction, safety qualification, or measured performance claim.

## Purpose

Connect Symthaea materials discovery to evidence-qualified cell and pack capability. Given a mission, bounded candidate portfolio, operating scenarios, and a finite experiment set, a read-only planner returns eligible Pareto candidates and a next-experiment recommendation.

Compose with MAT-RD-001 #4997, MAT-MLIP-001 #4992, ION-001 #4996, SSE-001 #4993, battery observation/rights work #11249 and #11421, multi-output target work #11495, dynamic electrothermal work #11485, pack capability #11271, mission evaluator #11589, and dynamic diagnostics #11404. Do not duplicate their physics, target, rights, or protocol owners.

## Paradigm shift

Optimize the material + electrode + process window + cell architecture + operating policy for a declared mission. A chemistry label alone is not a design identity. Keep candidate generation, surrogate predictions, local calculations, physical measurements, independent-lab evidence, and qualification decisions distinct.

## Contract

- Mission: application, exact duty-cycle identity, temperature/load scenarios, life, required energy/power/reserve, and system boundary.
- Candidate: immutable material/composition references, electrode formulation/loading/microstructure, process recipe/window and maturity, cell format and inactive-mass boundary, and optional pack/thermal/control configuration.
- Metrics: unit, boundary, scenario, value, uncertainty, and evidence reference. Never mix active-material, electrode, cell, pack, and installed-system metrics.
- Hard gates: rights, safety evidence, mission feasibility, evidence support, manufacturing eligibility, and reserve. FAIL or UNKNOWN cannot be compensated by a high score elsewhere.
- Experiment option: question, expected information/decision value, duration, cost, material, risk, facility needs, and estimate evidence. STOP / NO_EXPERIMENT remains available.
- Read-only: no scheduling, purchasing, synthesis, lab equipment operation, hazardous-work authorization, BMS control, or deployment approval.

## Pareto and experiment tournament

Compare random, DOE/grid, single-objective Bayesian optimization, multi-objective Bayesian optimization, uncertainty-only, diversity/novelty, reproducible expert-prioritized, and mission-conditioned evidence-weighted value-of-information policies under equal frozen budgets.

Report feasible-design discovery, time/cost/material to first feasible candidate, normalized hypervolume, simple/Pareto regret, prospective top-K hit rate on blind physical specimens, calibration by chemistry/structure/process family, OOD/abstention quality, duplicate experiments, diversity, sensitivity to failed/censored/non-converged outcomes, compute time/energy, and adaptation burden.

Use chemistry-family, structure-family, process-batch, line/equipment, and physical-cell holdouts. Random row splits alone do not establish transfer.

## Industry reference envelope

USABC High Energy EV Performance Goals (6 October 2025) publish aspirational cell-level end-of-life goals including 450 Wh/kg usable specific energy, 850 Wh/L, 15-year calendar life, 750 DST cycles with 25% fast charge, 10-minute fast charge to 80% usable energy, and a stated $60/kWh cost target at the specified production volume. These are programme goals, not commercial-cell averages or universal requirements: https://uscar.org/download/246/energy-storage-system-goals/16331/usabc-high-energy-ev-performance-targets.pdf

IEA Global EV Outlook 2026 reports LFP above 55% of global EV battery deployment in 2025; latest cell values reported up to 175 Wh/kg sodium-ion, 205 Wh/kg LFP, and 265 Wh/kg NMC. These are market context, not matched independent comparisons: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries

The 3 November 2026 MRS perspective on AI/ML for energy materials emphasizes data quality, uncertainty, physical validation, autonomy level, and lifecycle benefit; computational candidate screens are not experimental discovery: https://doi.org/10.1557/s43581-026-00171-8

Battery Data Commons is a discovery/metadata resource; each original dataset still requires independent rights review: https://batterydatacommons.org/

## v0.1 software acceptance

- Every objective has unit, boundary, mission/scenario, and evidence reference.
- Every material, process, cell, model, and data dependency resolves to a versioned identity.
- FAIL/UNKNOWN hard gates never enter the qualified Pareto set.
- Synthetic, assumption, vendor, literature, surrogate, calculation, physical-measurement, and independent-lab evidence remain distinct.
- Canonical reports are deterministic under input permutation.
- Every recommendation includes reason, evidence lineage, support/uncertainty state, and resource budget.
- Negative controls cover dominated/incomparable candidates, scenario failure, OOD prediction, unknown safety, rights-blocked data, censoring, invalid units/boundaries, and over-budget experiments.

These are software integrity targets, not achieved battery performance. A passing contract audit cannot establish synthesis, cell performance, safety, lifetime, manufacturing yield, cost, or commercial superiority.

## Next increment

Implement a standalone Rust reference audit for schema/identity checks, hard-gate eligibility, exact small-set Pareto dominance, finite-budget experiment selection, adversarial fixtures, and permutation tests. Add learned optimizers only after the small reference semantics replay independently.