# Ambient-Pressure High-Tc Superconductivity: Evidence and Engineering Programme
**Research note v0.1 — 17 November 2026**  
**Tracking issue:** [Symthaea #12617](https://github.com/Luminous-Dynamics/symthaea/issues/12617)  
**Status:** Evidence review and programme proposal; no new material, synthesis, or performance is claimed.

## Executive conclusion

There are two different opportunities:

1. **Make existing superconductors work better and at larger scale.** This is already possible in specialized applications. NbTi and Nb3Sn are mature low-temperature conductor technologies; REBCO is a commercial high-temperature-superconductor tape technology; MgB2 and Bi-2212 occupy more specialized positions. Actual deployed HTS cable systems exist, including the SuperRail railway-grid installation in Paris. The immediate engineering problem is cost and reliability per delivered kA·m, field/temperature performance, joints, strain, AC loss, cryogenics, protection and manufacturing yield—not discovering room-temperature physics first.
2. **Discover a stable, synthesizable, ambient-pressure material with much higher Tc, potentially 293–300 K.** This remains a major open scientific challenge. The current evidence does not establish a reproducible room-temperature, ambient-pressure bulk superconductor. High-pressure hydrides, ambient-pressure nickelates, conventional electron–phonon candidates and unconventional correlated materials must remain distinct research lanes.

Luminous can make a real contribution by creating a reproducible materials-discovery-to-engineering workflow, not by presenting a predicted high-Tc compound as a discovery.

## 1. Current experimental landscape

### 1.1 Ambient-pressure high-Tc materials

The PNAS 2026 programmatic perspective notes that the long-standing ambient-pressure record is about 133 K in cuprates. It identifies nickelates, iron-based superconductors, high-pressure hydrides, interfaces and low-dimensional engineered systems as important research directions. It also emphasizes that prediction and synthesis must be connected: dynamically stable is not necessarily thermodynamically synthesizable, and metastable phases may decompose or transform.

- PNAS (9 March 2026): https://doi.org/10.1073/pnas.2520324123

### 1.2 Nickelates: a compelling near-term ambient-pressure route

A March 2026 paper reports epitaxial (La,Pr)3Ni2O7 thin films at ambient pressure:
- superconducting onset up to approximately 63 K;
- zero resistance up to approximately 37 K;
- diamagnetic transition beginning around approximately 23 K;
- growth via a highly oxidizing non-equilibrium epitaxial process.

Source: Zhou et al., *Superconductivity onset above 60 K in ambient-pressure nickelate films*, National Science Review (2026-03-10): https://doi.org/10.1093/nsr/nwag151

This is an important materials and synthesis result, but the three temperatures are not interchangeable. The gap between pairing onset, zero resistance and diamagnetic response makes phase coherence, superconducting fraction, defect/oxygen control and reproducible growth central engineering questions.

A distinct bulk bilayer-nickelate result reports:
- onset around 92 K and zero resistance around 73 K at 21.6 GPa;
- Meissner effect around 60 K at 20.6 GPa;
- onset up to 96 K in a related composition.

Source: *Bulk superconductivity up to 96 K in pressurized nickelate single crystals*, Nature 649 (2026): https://doi.org/10.1038/s41586-025-09954-4

This is a high-pressure result, not an ambient-pressure operating material. The useful research direction is to examine whether chemical pressure, epitaxial strain, interfacial design, stoichiometry control or metastable phase stabilization can preserve the desired state after pressure is removed.

### 1.3 Hydrides: high Tc, extreme pressure

Hydrides demonstrate that very high transition temperatures can occur in hydrogen-rich structures under extreme compression. LaH10 has reported transition temperatures around 250 K under pressures of order 170–200 GPa. This demonstrates a remarkable physical regime, but a diamond-anvil-cell sample at planetary pressure is not a practical power conductor.

The 2026 PNAS perspective explicitly identifies synthesis thermodynamics, stable structure prediction and metastability as major bottlenecks:
https://doi.org/10.1073/pnas.2520324123

A candidate reported near 298 K at approximately 260 GPa remains a high-pressure research result, not a room-temperature ambient-pressure material. Any such claim needs exact specimen identity, independent reproduction, pressure history, secondary-phase analysis and evidence that all superconducting signatures belong to the same sample.

### 1.4 Limits of conventional electron–phonon discovery

A 2025 Nature Communications analysis examined over 20,000 metals with electron–phonon calculations and identified a trade-off between phonon frequency and coupling strength. It suggests conventional ambient-pressure compounds predicted near 100–120 K are already difficult to synthesize and often thermodynamically unstable. It does not establish a universal physical prohibition against unconventional room-temperature superconductivity.

Source: https://www.nature.com/articles/s41467-025-63702-w

Implication: ambient-pressure room-temperature superconductivity should not be approached by merely training a Tc regressor on conventional BCS/Eliashberg labels and extrapolating.

## 2. AI discovery baseline: reproduce before claiming superiority

The January 2026 BEE-NET workflow is the baseline to challenge.

It predicts Eliashberg spectral functions and DFT/Allen–Dynes Tc labels. Its paper reports:
- 0.87 K mean absolute error against DFT-based Allen–Dynes labels;
- a 99.4% true-negative rate;
- screening of over 1.3 million candidate structures to 741 dynamically and thermodynamically stable candidates with DFT-confirmed Tc above 5 K;
- synthesis and experimental confirmation of two previously unreported superconductors.

Paper: https://www.nature.com/articles/s41524-026-01964-8  
Code and data: https://github.com/henniggroup/BEE-NET

**Reproducibility caveat from direct repository audit (17 November 2026):** the BEE-NET README says the full database and model checkpoints are not tracked in Git. GitHub contents checks return 404 for database.json and models/, the referenced workflow/pkl_files intermediate is absent, no GitHub releases are listed, and the full pipeline contains site-specific /blue/hennig paths. The README also warns that the bootstrap index files contain DataFrame row labels while the index column holds CIF filenames; substituting one for the other can cause held-out test leakage. This establishes that a clean clone is not a complete reproducibility package; it does not prove that data/checkpoints are unavailable from Materials Cloud, another artifact host or the authors. The work to close these dependencies is tracked in [BEE-NET-REPRO-001 #12619](https://github.com/Luminous-Dynamics/symthaea/issues/12619).

The 0.87 K error is against computational labels, not measured experimental Tc. The workflow is conventional-superconductor-focused; its authors note that its ambient-pressure training and alpha2F frequency range do not directly cover high-pressure hydrides or unconventional superconductors.

Our first task is to reproduce the exact released dataset, code, checkpoint, environment and original train/test split. Then test chemistry/family/source/temporal/pressure holdouts. A model that interpolates within a known family is not evidence of discovery generalization.

## 3. Luminous work already exists

Compose rather than duplicate:
- Symthaea [MAT-RD-001 #4997](https://github.com/Luminous-Dynamics/symthaea/issues/4997): staged materials discovery, including an explicit gate for electron–phonon/Eliashberg and pressure-dependent stability readiness.
- Symthaea [MAT-MLIP-001 #4992](https://github.com/Luminous-Dynamics/symthaea/issues/4992): calibrated interatomic-potential evaluator and OOD discipline.
- Symthaea [HTS-FUSION-MAGNET-001 #11736](https://github.com/Luminous-Dynamics/symthaea/issues/11736): REBCO conductor-to-coil engineering, quench, AC loss, joints, strain and cryogenic burden.
- Symthaea [ENG-MAG-001 #5677](https://github.com/Luminous-Dynamics/symthaea/issues/5677) and [ENG-THERM-001 #5681](https://github.com/Luminous-Dynamics/symthaea/issues/5681): reusable magnetic and thermal models/evidence.
- Symthaea [FUS-MAT-001 #4348](https://github.com/Luminous-Dynamics/symthaea/issues/4348) and [FUS-SYS-001 #7509](https://github.com/Luminous-Dynamics/symthaea/issues/7509): fusion material architecture and whole-plant systems boundary.

The new research programme is [SC-AMB-PRESSURE-001 #12617](https://github.com/Luminous-Dynamics/symthaea/issues/12617). It owns ambient-pressure high-Tc discovery, superconductivity evidence semantics and material-to-manufacturability gates. #11736 retains REBCO conductor and magnet engineering.

## 4. Terminology: what “quasi-superconducting” can and cannot mean

“Quasi-superconductor” is not an acceptable pass state for a practical conductor. Separate:
- superconducting fluctuations / preformed Cooper pairs above global phase coherence;
- proximity-induced superconductivity in a neighboring normal layer;
- granular/percolative superconducting paths;
- pseudogap or anomalous low-resistance normal-state behavior;
- low-temperature high-conductivity normal metals;
- confirmed bulk superconductivity.

The first four may be valuable physics or device research, but they do not automatically provide a macroscopic zero-resistance conductor. A percolating filament can give a near-zero resistance measurement while most of a sample remains normal. A local diamagnetic signal does not establish a bulk Meissner state. A high-conductivity metal is not a superconductor.

For nickelates, the separation between onset, zero-resistance and diamagnetic temperatures is itself a target: improve bulk phase coherence, interlayer coupling, uniform oxygen stoichiometry, defect control and reproducible epitaxial growth.

## 5. Paradigm shifts for discovery

### A. Optimize synthesizable free-energy landscapes, not predicted Tc alone

Candidate generation should jointly estimate:
- target structure and composition;
- electronic state and pairing mechanism;
- phonons/dynamical stability;
- thermodynamic hull and competing phases at relevant pressure/temperature;
- finite-temperature, anharmonic and nuclear quantum effects where important;
- metastable lifetime and kinetic barriers;
- synthesis route, substrate/interface and pressure/decompression history.

A high-Tc prediction attached to an impossible or unrecoverable phase should be low priority.

### B. Search the process–structure–property space

For nickelates and other complex oxides, composition alone is insufficient. Growth oxygen chemical potential, substrate, epitaxial strain, temperature, deposition kinetics, oxygen stoichiometry, defects, layer sequence and annealing history may determine whether the desired phase can form.

### C. Treat pressure as a synthesis/control variable, not automatically an operating requirement

Search for pressure-created structures that can be stabilized after decompression through chemical pressure, epitaxy, interfaces, confinement, metastable quenching or substitution. Each route needs a phase-retention and time-stability test. Never count an ambient-pressure prediction as an experimentally recoverable phase.

### D. Separate conventional and unconventional physics

For conventional electron–phonon candidates, use qualified DFPT, phonon and pressure-resolved Eliashberg calculations, with Allen–Dynes/McMillan estimates only inside their validity envelope.

For cuprates, nickelates, iron-based systems and interface materials, do not force an electron–phonon Tc model to generate scientific authority. Preserve separate mechanisms, observables and theory confidence.

### E. Close the loop with negative results

Record failed synthesis, decomposition, impurity phases, non-zero residual resistance, missing magnetic evidence, low shielding fraction, poor current density and unstable batches. Candidate search must learn from why an attractive material failed.

## 6. Benchmarks

### Discovery model
- Reproduce the published BEE-NET test set before making comparison claims.
- Report model-label error separately from measured-Tc error.
- Add source-, structure-, chemistry-, family-, time- and pressure-held-out tests.
- Report precision/recall for rare superconductors, false discoveries, OOD abstention, calibration, compute budget and synthesis success.
- Compare top-K candidate yield to random, diversity, simple heuristic and DFT-only controls under matched resources.

### Experimental evidence
For the same exact specimen and conditions, preserve:
- calibrated four-probe resistivity with geometry and noise floor;
- magnetic flux exclusion/diamagnetic shielding and shielding fraction;
- heat-capacity/thermodynamic evidence where feasible;
- field and current suppression;
- structural and composition evidence, including secondary-phase exclusion;
- sample/batch provenance and independent replication.

Keep onset Tc, zero-resistance Tc0, diamagnetic onset, bulk fraction and irreversibility separate.

### Room-temperature/ambient-pressure gate
A claim must show the state at 293–300 K and approximately 1 bar in the same identified sample. There must be no hidden high-pressure operating requirement. Transport, magnetic and bulk-sensitive evidence must refer to the same sample/state and be independently reproduced by other laboratories.

### Engineering translation
Measure:
- Jc(B,T,theta) and engineering Je across the whole composite cross-section;
- field, temperature margin, strain and anisotropy;
- AC losses, joints, stabilizer, current sharing and quench behavior;
- usable conductor length, yield, winding, bending, fatigue and inspection;
- cryogenic energy, availability, maintenance and lifecycle cost.

Compare on a matched application and operating point. Do not compare self-field tape Ic to high-field coil current, or material-only current density to full-conductor engineering current density.

## 7. Can superconductors work at scale today?

Yes, in specialized applications. REBCO tape is commercially manufactured and used in high-current cable/magnet programmes. A 2025 SuperRail report describes an HTS cable system in commercial operation on a Paris railway grid, with two cables rated at 2.25 MW each. This is real engineering scale, but it is not room-temperature operation or a universal grid replacement:
https://doi.org/10.1049/icp.2025.2154

A 2026 conductor translation review reports approximate routine REBCO piece lengths of 0.3–2 km and estimated yields of 80–90%; these are review-derived estimates, not universal supplier guarantees. It highlights cost, manufacturing throughput, jointing and mechanical robustness. Commercial REBCO is therefore scalable but not yet a drop-in low-cost replacement for all copper/aluminium systems.

For room-temperature ambient-pressure superconductors, there is currently no equivalent qualified conductor supply chain because no independently reproduced material has reached that state.

## 8. First execution sequence

1. Freeze source revisions and evidence records for BEE-NET, 2026 nickelate films, high-pressure nickelate crystals, hydrides and the REBCO conductor roadmap.
2. Reproduce BEE-NET on its original released split and record exact environment/data/code identities.
3. Create the specimen-level evidence schema and the onset/zero/diamagnetic/bulk transition distinction.
4. Audit MAT-MLIP/MAT-RD capability and define missing pressure-resolved electron–phonon, synthesis thermodynamics and unconventional-mechanism adapters.
5. Build a small candidate/positive/negative control set with measured and computational labels separated.
6. Select one application-specific REBCO/NbTi/Nb3Sn/MgB2 benchmark to connect new materials discovery with #11736.
7. Only after these gates, rank a small prospective candidate set for partner-lab synthesis and independent replication.

## Claim ceiling

This is a research and qualification programme. It does not claim that Luminous has discovered, synthesized, measured, independently replicated or manufactured a room-temperature superconductor. It does not establish ambient-pressure room-temperature superconductivity as likely or impossible.
