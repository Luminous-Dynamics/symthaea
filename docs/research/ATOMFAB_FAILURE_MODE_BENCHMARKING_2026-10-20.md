# AtomFab structured failure-mode benchmark v1

**Status:** Executable accounting contract and synthetic fixture; not a validated materials-model result.

## Paradigm shift: do not let average accuracy hide rare catastrophic errors

Model evaluation typically reports energy MAE/RMSE, force error, or one task-level leaderboard score. These are necessary but insufficient for an evidence-first fabrication pipeline: low average error may coexist with occasional non-finite forces, collapsed geometries, unstable relaxations, severe ranking inversions, or unsupported transfer to a new material/support family. An out-of-domain abstention is also different from an out-of-domain prediction: the former may be correct risk management; the latter may be a scientific failure.

The 2025 MS25 benchmark explicitly evaluates derived observables as well as energies/forces and recommends focusing on failure modes, transferability, and how errors affect physical observables rather than only marginal changes in MAE/RMSE. [MS25, JCIM (2025)](https://doi.org/10.1021/acs.jcim.5c01262). This v1 implements that recommendation as record-level, typed evidence rather than unstructured commentary.

## Failure-mode taxonomy

The Rust enum freezes these categories:

- Numerical: non-finite energy, non-finite force, solver non-convergence, force spike.
- Structure/physics: geometry collapse, interatomic overlap, unsupported species, energy-ordering violation, observable violation, uncertainty undercoverage.
- Transfer/interface/process: out-of-domain, interface breakdown, infeasible process.

Each assessment has exactly one disposition:

- **Pass:** no failure mode or severity attached.
- **Failed:** requires a failure mode and severity.
- **Abstained:** requires a reason mode and severity; retained separately from failure counts.

Severity is informational, degraded, or critical. In particular, an out-of-domain abstention counts as an abstention reason, not as a failed physical prediction; an out-of-domain prediction that is nevertheless emitted must be recorded as a failure. The taxonomy is intended to be frozen before the test campaign. If the taxonomy must change, version it instead of silently remapping prior failures.

## Comparison validity and denominators

A summary is computed only over one exact matched scope: campaign, model digest, evaluator digest, evaluation-scope digest, material family, split, and assessment basis. Validation, random test, family holdout, prospective evidence and distinct model checkpoints cannot be pooled. Every record carries a parent-system and sample identity plus a digest of the evidence artifact. Duplicate record IDs are rejected; summary counts and IDs are rebuilt from the records to detect a forged summary.

Report at minimum total assessed, pass/fail/abstain counts, critical failure count, failure counts by mode and severity, and abstention reasons by mode. Keep the denominator visible. Count failed relaxations, non-convergence and invalid candidates rather than reporting only the best surviving structures. Publish energy/force errors and derived-observable error separately, grouped by material family and holdout split.

## Initial comparative gates

1. Reproduce a published reference baseline on a pinned task before modifying the policy.
2. Keep matched-structure energy/force error, derived observables, structural-discovery quality, and failure-mode distributions as separate score families.
3. Report critical failures by family/surface/size and retain failed samples in denominators; no pooled mean can compensate for a critical failure.
4. Separate failure and abstention rates. A reject-all model cannot succeed because pass, fail, and abstain counts are all required and denominators remain visible.
5. Compare fixed domain-specific and universal MLIPs, conventional random/local search, and the same model/search with the Symthaea decision layer disabled.
6. Verify transfer to unseen parent systems and support surfaces; related configurations from one parent system remain in one split.
7. Distinguish reference, simulation, measurement and independent reproduction. Software tests do not validate atomistic physics, and workcell simulation does not prove a real fabrication route.

## Industry baseline and scientific ceiling

MS25 spans MgO surfaces, liquid water, zeolites, Pt surface reaction, high-entropy alloys and disordered Zr oxides and evaluates physical observables in addition to direct energy/force errors. Its recommendation matches the direction taken here: identify failure modes and transferability limits, not just tiny improvements on an aggregate score. [MS25 data record, Argonne ACDC](https://acdc.alcf.anl.gov/mdf/detail/74d6b3d9-c33c-47c4-9c9a-c14ca773d3c8-1.0/).

For supported Cu/Al2O3 nanoparticle search, compare against the exact 2026 published protocol and pinned DP-UniAlCu, MACE-OMAT and MatterSim checkpoints when terms and compute permit. The published benchmark shows why the split matters: a universal potential with larger binding-energy deviations found more stable structures in some cases, while the domain-specific baseline was much faster for the reported MD comparison. This is evidence for separate scoreboards—not evidence AtomFab already wins. [Xu et al., npj Computational Materials (2026)](https://doi.org/10.1038/s41524-026-02043-8).

## Implemented artifacts

- Rust module: `crates/domains/symthaea-atomfab/src/failure_modes.rs`
- Record schema: `crates/domains/symthaea-atomfab/schemas/failure-mode-records-v1.schema.json`
- Summary schema: `crates/domains/symthaea-atomfab/schemas/failure-mode-summary-v1.schema.json`
- Synthetic fixture: `crates/domains/symthaea-atomfab/fixtures/failure-mode-records-v1.json`
- Benchmark registry: MS25, supported-nanoparticle, and experimental reality-gap lanes explicitly include failure-mode metrics and denominator/transferability gates.

The fixture has four synthetic examples (one pass, one critical force failure, one degraded energetic-ordering failure, and one out-of-domain abstention) to test the accounting contract. These are not observed model outputs, experimental results or safety qualifications.
