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

****Frozen upstream audit identity:** BEE-NET main resolved to commit f21c260a9117830f2d74f2c5ac844ab671c0c606 (commit message: “updated readme, with dataset clarification”) on 17 November 2026. This exact commit, not mutable main, is the source snapshot for the first audit.

Reproducibility caveat from direct repository audit (17 November 2026):** the BEE-NET README says the full database and model checkpoints are not tracked in Git. GitHub contents checks return 404 for database.json and models/, the referenced workflow/pkl_files intermediate is absent, no GitHub releases are listed, and the full pipeline contains site-specific /blue/hennig paths. The README also warns that the bootstrap index files contain DataFrame row labels while the index column holds CIF filenames; substituting one for the other can cause held-out test leakage. This establishes that a clean clone is not a complete reproducibility package; it does not prove that data/checkpoints are unavailable from Materials Cloud, another artifact host or the authors. The work to close these dependencies is tracked in [BEE-NET-REPRO-001 #12619](https://github.com/Luminous-Dynamics/symthaea/issues/12619).

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

Yes, in specialized applications. REBCO tape is commercially manufactured and used in high-current cable/magnet programmes, without requiring room-temperature superconductivity.

A 2025 PLD-REBCO industry perspective reports that top-tier manufacturers exceed 3,000 km/year of 12 mm tape, and cites representative performance above 200 A per 4 mm at 20 K, 20 T (field parallel to c-axis), with an approximate $20/m price. These are industry-perspective figures and specific product claims, not universal supplier guarantees:
https://doi.org/10.1016/j.supcon.2025.100188

A 2025 SuperRail paper reports a commercial HTS DC cable installation for Paris Montparnasse using two 60 m, 1.5 kV, 3.5 kA cables. The system was type-tested for thermal cycles, dielectric performance, pressure, system losses and a 67 kA/100 ms fault-current duty. It demonstrates a real niche where constrained right-of-way can justify HTS infrastructure, not a universal grid cost advantage:
https://doi.org/10.1049/icp.2025.2154

A separate 2025 REBCO cable-in-conduit prototype reports 80 kA at 10.85 T and 4.5 K, with cyclic Lorentz loading to at least 830 kN/m. This is an important fusion-magnet conductor milestone, not room-temperature or standard utility-grid operation:
https://doi.org/10.1016/j.eng.2025.05.015

A 2026 conductor translation review estimates approximate routine REBCO piece lengths of 0.3–2 km and yields of 80–90%, with costs around 4–10 USD/kA·m at 77 K self-field versus an illustrative grid-scale target at or below 1.5 USD/kA·m. These values are review-derived and sensitive to specification; compare only at matched field, temperature, angle and conductor cross-section:
https://www.sciencedirect.com/science/article/pii/S2949822826015558

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


## 9. Prospective discovery tournament — 17 November 2026

A new child work package, [SC-DISCOVERY-TOURNAMENT-001 #12634](https://github.com/Luminous-Dynamics/symthaea/issues/12634), turns the discovery benchmark into a budgeted decision problem. This is a deliberate paradigm change: optimize how many credible, synthesisable candidates survive independent validation at a fixed compute and laboratory budget, rather than optimizing retrospective Tc MAE alone.

### Research update

1. **Genome-inspired AI (29 October 2026).** The early-access *Scientific Reports* paper reports a 53,196-material DataG Non-Sc dataset and a screening workflow over ICSD, COD and Materials Project. It predicts 22 Pu-containing candidates above 50 K, including approximately 89 K for Pu2Co17, plus more than 40 hydrides with predicted Tc up to 100 K. These are ML predictions, not experimental confirmations. The dataset, training/test split, model weights, features, negative-label definition and full pipeline need to be obtained and independently reproduced before its metrics are admitted. Actinide-containing candidates are computational-only in the general programme pending specialist institutional review.
   - https://doi.org/10.1038/s41598-026-76977-w

2. **Metal-embedded hydrogen units (August 2026).** A computational study screened more than one million M-Be-H configurations. It predicts CaAuH6 at approximately 70 K under 30 GPa and approximately 44 K at ambient pressure; it also reports an AcBeH6 metastable phase approximately 131 meV/atom above the convex hull with predicted Tc around 52 K at 30 GPa. These are calculated predictions. The nominal ambient-pressure result must not be conflated with experimentally synthesized or decompression-retained material.
   - https://doi.org/10.1016/j.mtphys.2026.102142

3. **Ambient-pressure nickelate film (March 2026).** The (La,Pr)3Ni2O7 epitaxial-film report gives approximately 63 K onset, 37 K zero resistance and a diamagnetic transition beginning around 23 K. Those temperatures describe distinct observables; exact specimen-to-observable linkage must be extracted from the source and supplement before combining them as one specimen-level record.
   - https://doi.org/10.1093/nsr/nwag151

4. **Bulk nickelate requiring pressure.** La2SmNi2O7-delta crystals were synthesized at ambient pressure, but the reported superconducting state occurs at approximately 21.6 GPa (92 K onset, 73 K zero resistance), with Meissner response around 60 K at 20.6 GPa. This is a high-pressure operating material, not an ambient-pressure superconductor.
   - https://doi.org/10.1038/s41586-025-09954-4

### Benchmark design lessons

- **BEE-NET** is a conventional electron–phonon / Eliashberg workflow whose 0.87 K result is an error against DFT/Allen–Dynes labels. Do not compare that number directly with experimental Tc error or with empirical composition-model accuracy.
- **HTSC-2025** has 155 computational candidate records and its own structural-system splits. It is an independent computational benchmark, not a specimen-level experimental dataset.
- **DataG Non-Sc** is a new negative-material dataset from the 29 October 2026 study. It should not be considered admitted until its schema, row identities, split manifest, license and provenance are recovered.
- **Matbench Discovery** is the methodological model: evaluate selection quality along a ranked campaign, including precision/recall as the validation budget K increases. Its discovery-acceleration result is for crystal stability, not superconductivity, and is used only as a benchmark-design precedent.
  - https://doi.org/10.1038/s42256-025-01055-1
- A 2026 review of superconductivity ML specifically warns that random splits can place doped variants of the same parent compound in both train and test sets; family/prototype/formula grouping and OOD evaluation are therefore mandatory.
  - https://doi.org/10.1021/acsomega.6c01100

### New machine-readable corpus

- [Candidate and source register v0.1](https://github.com/Luminous-Dynamics/symthaea/blob/research/sc-ambient-pressure-001/data/research/sc_superconductor_discovery_cases_v0_1.json)
- Contains five candidate/reference records, seven source records and three benchmark-catalog entries.
- Preserves computational predictions separately from reported experimental transitions, and pressure of synthesis separately from pressure of operation.
- This is a small hand-auditable seed corpus, not a complete training dataset or leaderboard. The register deliberately flags unresolved specimen identity, missing phase-stability evidence, and unverified dataset/checkpoint access.

### Decision-centric benchmark protocol

For each mechanism-specific track, use identical frozen candidate pools and report precision@K / verified yield@K at K = 10, 25, 50, 100 and 500; discovery acceleration versus random and simple heuristics; novelty by family/prototype; calibration and selective risk; OOD abstention; synthesis/phase-retention outcomes; and cost per independently verified result. Use paired confidence intervals, record all failed/non-converged candidates and preserve uncertainty.

Mandatory holdouts include reduced-formula groups, parent family, structure prototype, chemistry family, source/lab, time and pressure. All preprocessing, feature selection, calibration and model selection must remain inside the training partition. Freeze the prospective candidate list before expensive DFT or external evidence retrieval.

The stretch targets in #12634 are proposed research targets, not current results: at K <= 50, seek at least 1.5x validated-candidate enrichment over the strongest admitted baseline on a frozen prospective pool, and at least 2x fewer expensive follow-ups per independently verified candidate than the best simple heuristic. If the cohort is too small for meaningful confidence intervals, report inconclusive.

### Cross-domain design rule

A prediction becomes a candidate dossier; a candidate dossier becomes a prospective test; a prospective test becomes a specimen-level evidence chain; and only independently verified material enters the conductor/device programme. No weighted score may compensate for a failed hard constraint such as pressure release, unsafe composition, or inability to retain a usable phase. Existing REBCO conductor-to-coil engineering remains with [HTS-FUSION-MAGNET-001 #11736](https://github.com/Luminous-Dynamics/symthaea/issues/11736).

### Claim ceiling

This addendum records new papers, benchmark design and an offline JSON corpus. It does not reproduce the genome-inspired model, run DFT/Eliashberg calculations, synthesize candidates, or claim experimental qualification.


## 10. Executable registry contract — 17 November 2026

A focused, standalone Rust audit harness now accompanies the reference corpus:

- Harness: [tools/superconductivity-discovery-harness](https://github.com/Luminous-Dynamics/symthaea/tree/research/sc-ambient-pressure-001/tools/superconductivity-discovery-harness)
- It validates the registry against the committed Draft 2020-12 schema, refuses schema reference indirection, verifies unique candidate/source IDs and source closure, prevents predicted Tc from being represented as observed transitions, and requires actinide records to carry an explicit computational-only/exclusion gate.
- The receipt includes exact SHA-256 digests for both schema and registry and record counts.
- Negative controls cover duplicate candidate IDs, orphan source references, prediction-to-observation promotion and remote schema references.
- The harness is a standalone Cargo workspace with exact direct dependency pins and JSON Schema reference-resolution features disabled. First-time Cargo dependency acquisition may still require a package mirror; once built, the audit reads only the explicit local files and performs no evidence-source network lookups.

Run commands and claim limits are documented in [the harness README](https://github.com/Luminous-Dynamics/symthaea/blob/research/sc-ambient-pressure-001/tools/superconductivity-discovery-harness/README.md). A path-scoped exact-head workflow was added at [.github/workflows/superconductivity-discovery.yml](https://github.com/Luminous-Dynamics/symthaea/blob/research/sc-ambient-pressure-001/.github/workflows/superconductivity-discovery.yml).

**Execution status:** the source, schema and negative-control tests have been authored; the Rust binary has not yet been compiled or run in this environment. The workflow is intended to provide the first exact-head compiler/test/clippy receipt. No test pass is claimed until that receipt reports success.


## 11. HTSC-2025 benchmark admission audit — 17 November 2026

The public [HTSC-2025 Hugging Face dataset](https://huggingface.co/datasets/xiao-qi/HTSC-2025) is now recorded as a promising but **not yet admitted** external benchmark.

- The paper describes 155 samples in its benchmark overview; the current Hugging Face README table totals 140 material entries (19 + 12 + 23 + 15 + 18 + 53). The viewer shows one top-level JSON row containing material-keyed records and a 166 kB JSON file. This count discrepancy is unresolved; no count is asserted as canonical until per-material keys are reconciled.
- The visible Hugging Face dataset card displays an empty/missing YAML metadata warning and no visible license field. The paper's article license must not be assumed to grant rights to independently hosted JSON/CIF files. Do not mirror or redistribute upstream records until data rights are clarified.
- The paper's own protocol uses DFT-computed Tc targets for theoretically predicted BCS candidates, and says test benchmark materials should be excluded from training to prevent leakage. These metrics do not measure experimental Tc and should not be pooled with the ambient nickelate specimen evidence.

Child work package: [HTSC-2025-REPRO-001 #12635](https://github.com/Luminous-Dynamics/symthaea/issues/12635), which covers count reconciliation, rights, CIF/source provenance, canonical record IDs, exact split reconstruction and grouped holdouts.

The registry now preserves paper-reported count 155 separately from hosted README table count 140, marks license status as unresolved / redistribution unauthorized, and keeps the HTSC-2025 record out of any admitted or reproduced leaderboard. This is a source-audit update; no raw hosted CIF/JSON content was copied into Symthaea.

## 12. Genome-inspired AI reproduction gate — 17 November 2026

The 29 October 2026 *Scientific Reports* early-access paper reports a 53,196-record DataG Non-Sc set, 10 ML and 9 neural models, and high-Tc predictions from ICSD / COD / Materials Project. Its reported Pu-based and hydride candidates remain computational predictions. The source says this early-access article version may be replaced automatically by the final Version of Record.

Child work package: [GENOME-SC-AI-REPRO-001 #12636](https://github.com/Luminous-Dynamics/symthaea/issues/12636). Before admitting it, require original data artifacts, data rights, label provenance, exact splits, candidate list, checkpoints, code and training environment. Plutonium candidates remain computational-only for the general programme pending specialist institutional radiological review.

The programme now has separate reproducibility tracks for BEE-NET (#12619), HTSC-2025 (#12635) and genome-inspired AI (#12636). Their tasks and label spaces remain distinct.


## 13. Budgeted discovery metrics runner — 17 November 2026

A deterministic ranking-metrics executable was added to the standalone Rust harness:

- source: tools/superconductivity-discovery-harness/src/bin/discovery_metrics.rs
- binary: sc_discovery_metrics
- test-only plumbing fixture: tools/superconductivity-discovery-harness/fixtures/discovery_metrics_smoke_v0_1.json

It accepts a frozen list of candidate IDs, finite ranking scores and outcome labels. It computes precision@K, recall@K, random-prevalence expectation, enrichment over random, average precision and an input SHA-256. Equal scores use candidate-ID ordering for deterministic replay; duplicate candidate IDs and invalid candidate budgets fail closed. Scores are explicitly ranking-only and cannot be interpreted as calibrated probabilities. No-positive test sets produce null recall/enrichment rather than an invented value.

The dedicated workflow now tests both the evidence registry auditor and ranking-metrics runner, executes the runner only on a clearly synthetic plumbing fixture, generates an isolated Cargo.lock, runs subsequent Cargo commands with --locked, and retains the lockfile, toolchain versions, source digests and machine-readable receipts as a run artifact.

**Metric scope:** confidence intervals, probability calibration, OOD selective risk, cost-normalized comparisons, family/temporal splits, and prospective synthesis yield are not implemented by this small ranking executable. Those remain separate tournament requirements in #12634. The synthetic fixture is a plumbing check, not evidence of model quality or a materials discovery.

The hosted source snapshot was pinned from the Hugging Face tree at verified revision ab53472 on 17 November 2026. The visible tree lists HTSC-2025.json (reported 166 kB), README.md and .gitattributes. This source revision is a locator for future reproduction, not permission to redistribute the data.
