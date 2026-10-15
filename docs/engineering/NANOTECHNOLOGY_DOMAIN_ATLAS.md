# Nanotechnology Domain Atlas: Beyond Atomistic Materials Discovery

**Status:** Research and engineering strategy; not a validated capability inventory or claim of a discovered material.  
**Date:** 2026-10-15  
**Main portfolio:** [NANOTECHNOLOGY_FRONTIER_PORTFOLIO.md](./NANOTECHNOLOGY_FRONTIER_PORTFOLIO.md)  
**Paradigm shifts:** [NANOMATERIALS_PARADIGM_SHIFT_ROADMAP.md](./NANOMATERIALS_PARADIGM_SHIFT_ROADMAP.md)  
**Closed-loop method benchmark:** [NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md)  
**Follow-on domain gate:** [NANO-002](https://github.com/Luminous-Dynamics/symthaea/issues/7507)  
**Evidence status:** literature review and source-planning only; no new domain data or models were run.

## 1. The key reframing: nanotechnology is not one field or one model

“Nanotechnology” spans several different physical regimes and engineering workflows. A quantum dot, a DNA-origami assembly, a nanopore, a plasmonic metasurface, a carbon nanotube transistor, a nano-enabled drug carrier, and an atomic-force microscope probe are not interchangeable kinds of object. They have different state variables, governing physics, process dependencies, measurements, and failure modes.

Organize a program along three axes:

1. **What is the engineered object?** Particle, surface, wire/tube, 2D layer, porous network, interface, molecular assembly, optical structure, electronic device, sensor, or manufacturing process.
2. **What physics or system response matters?** Electronic structure, bonding and kinetics; electromagnetic response; transport; fluid flow; mechanics; self-assembly; biochemical interaction; sensing; or process yield.
3. **What evidence level answers the question?** Data record, descriptor, approximate simulation, validated domain model, measured specimen, replicated result, or operationally qualified product.

A shared AI coordinator can help choose and organize evidence across these levels. It cannot make the underlying solvers interchangeable or transfer a result from one physical regime to another.

## 2. A wider map of nanotechnology fields

The entries below are opportunity hypotheses. No entry implies Symthaea already has a qualified implementation. Recommended tools are method *families*, not mandatory dependencies; select and pin an actual solver only after the task, license, input/output contract, and baseline are audited.

| Domain / object | Important questions and measurable outputs | Physics, tool, or evidence family that fits | Plausible Symthaea contribution | Main proof obligation |
|---|---|---|---|---|
| **1. 0D nanoparticles and quantum dots** | Size/composition/shape versus absorption, emission, quantum yield, stability, defect sensitivity and batch variation | Electronic-structure and excited-state methods where appropriate; optical spectroscopy; microscopy; synthesis/process records | Search diverse candidates and conditions; retrieve failure histories; choose the most informative optical/characterization measurement | Same sample identity across structure, process, spectrum, and aging; predictive calibration on held-out composition/size families |
| **2. 1D carbon nanotubes and nanowires** | Conductivity, contact resistance, defect sensitivity, mechanical limits, thermal transport, chirality, and dispersion | Atomistic/electronic models, transport calculations, mechanical characterization and fabrication records | Candidate grouping, defect/condition-aware retrieval, multi-objective device selection | Device-level tests must account for contact and fabrication effects, not just isolated-material values |
| **3. 2D materials and heterostructures** | Band structure, bandgap, defects, strain, stacking, charge transport, thermal behavior, interfaces and stability | DFT or qualified electronic-structure methods, tight-binding/transport models, microscopy and device measurements | Explore compositional/stacking/strain families, detect data leakage by parent structure, identify model disagreement | Group splits by related structures/material families; check DFT setup and measured-device mismatch |
| **4. Nanoelectronics and spintronics** | Switching, leakage, mobility, variability, spin transport, contact behavior, reliability, energy per operation | Electronic structure plus suitable transport/device simulation; fabrication and electrical test data | Constraint-aware inverse design; prioritize reliability experiments; connect material, process, and device records | A material-level predictor is not a device predictor; validate against operating conditions and manufacturing variation |
| **5. Nanophotonics, plasmonics and metasurfaces** | Spectral response, phase, polarization, efficiency, losses, angle sensitivity, bandwidth and fabrication tolerance | Maxwell-equation solvers, FDTD/FEM/RCWA or adjoint design; optical measurement | High-dimensional search, candidate diversity, Pareto analysis and expensive-simulation selection | Exact electromagnetic solver/settings; independent forward re-simulation; fabrication-tolerance and held-out geometry tests |
| **6. Photonic crystals and nanoscale optical cavities** | Bandgap, mode frequency, quality factor, coupling, disorder tolerance and emitter interaction | Electromagnetic eigenmode solvers and optical characterization | Explore parameter regions and identify discriminating simulations | Keep idealized simulation distinct from measured loss, disorder, and manufacturability |
| **7. Nanomagnetics and magnetic nanoparticles** | Switching fields, domain structure, anisotropy, heating response, stability and particle interaction | Micromagnetics, electronic methods where required, magnetometry and imaging | Select costly parameter sweeps; map uncertainty over composition, geometry and interactions | Validate the appropriate length/time scales; avoid conflating single-particle and ensemble behavior |
| **8. Nanopores, nanofluidics and selective separation** | Permeability, selectivity, flux, fouling, electro-osmosis, ion transport, hydration and energy per separation | Molecular dynamics, continuum transport, fluid/electrochemical models and real-feed experiments | Multi-objective selection across selectivity/flux/stability; choose the next mixed-feed or fouling test | Validate under relevant solution chemistry, pressure, concentration, and long-duration conditions |
| **9. Nano-biosensors and chemical nanosensors** | Sensitivity, detection limit, selectivity, response time, drift, matrix interference, calibration and false-positive rate | Surface chemistry, electrochemical/optical models, signal processing, spectroscopy and reference-standard measurements | Calibrated sensor triage, drift detection, experimental design and cross-modal evidence integration | Independent calibration, negative/blank/interference controls, instrument and batch holdouts; a low detection limit alone is not utility |
| **10. Nanomedicine and nanoparticle drug delivery** | Encapsulation, release, colloidal stability, biological barriers, biodistribution, immune response, toxicity and scale-up | Formulation data, physicochemical characterization, qualified biological studies and clinical evidence | Map evidence gaps, detect contradictory studies, rank formulation hypotheses for expert review | No clinical/safety inference from generic similarity; control for dose, route, assay, species, batch and formulation; require expert oversight |
| **11. Nano-bio interfaces and biomolecular nanotechnology** | Protein corona, membrane interaction, molecular recognition, DNA/protein assembly, biocompatibility and fouling | Molecular simulation and biophysical/biochemical measurements appropriate to the system | Maintain competing interaction hypotheses and recommend measurements that discriminate them | Biological context and uncertainty must be represented; do not generalize from a single assay or simplified model |
| **12. DNA origami, programmable self-assembly and colloids** | Assembly yield/fidelity, nucleation barriers, kinetic traps, cooperativity, hysteresis and defect rates | Coarse-grained molecular models, kinetic models, microscopy and time-resolved assembly data | Temporal reasoning on real trajectories; active selection of sequence/condition tests; failure-family memory | Sequence/condition lineage and measured yield; unordered snapshots cannot be treated as trajectories |
| **13. Nanozymes and catalytic / electrocatalytic surfaces** | Reaction-specific selectivity, adsorption, activation barriers, turnover, poisoning, durability and operating-condition dependence | DFT/validated interatomic models plus microkinetics, electrochemistry, surface characterization and experiments | Diverse candidate selection, hypothesis discrimination, cost-aware reference calculation requests | Total-energy benchmarks are not turnover-rate benchmarks; include potential, pH, solvent, coverage, kinetics and degradation as applicable |
| **14. Nanocomposites, polymer nanocomposites and hybrid interfaces** | Dispersion, interfacial adhesion, fracture/fatigue, barrier properties, percolation, rheology and aging | Multiscale mechanics, molecular simulation, process/quality data and mechanical testing | Relate process histories to variability; retain negative results; optimize Pareto frontiers | Processing history and specimen variability; properties must be measured under specified test conditions |
| **15. Nanoscale mechanical systems and NEMS** | Resonance, damping, force sensitivity, fatigue, contact and electromechanical coupling | Continuum/multiphysics FEM, atomistic treatment where essential, device metrology | Parameter search and diagnostic experiment selection | Solver verification and calibration against a physical device; do not apply a rigid-body simulator beyond its model class |
| **16. Nano-enabled energy and resource systems** | Battery/electrode interfaces, ion transport, degradation, catalytic energy conversion, thermal management and resource recovery | Electrochemical, atomistic and continuum models combined with cycling/aging experiments | Optimize performance–lifetime–cost–criticality trade-offs; learn from failures | Measure lifecycle degradation and operating conditions; do not rank solely on an ideal initial state |
| **17. Nanomanufacturing and process engineering** | Yield, size distribution, morphology, batch variation, throughput, process window, energy, solvent/precursor use and reproducibility | Process logs, designed experiments, in-situ sensing, microscopy/spectroscopy, qualified process models | Sequential process optimization, anomaly detection, experiment selection and batch traceability | Freeze machine/protocol versions and operating range; distinguish within-tool optimization from transfer between facilities |
| **18. Nano-characterization, metrology and microscopy** | Segmentation, particle-size distributions, defect detection, spectral interpretation, calibration, measurement uncertainty and cross-instrument agreement | SEM/TEM/AFM/STM, spectroscopy, scattering, image analysis and reference artifacts | Multimodal retrieval, disagreement detection, uncertainty-aware triage, reproducible analysis workflows | Calibration standards, operator/instrument holdouts, known detection limits and auditable processing; image quality is not physical truth |
| **19. Environmental nanotechnology and remediation** | Adsorption, photocatalysis, contaminant removal, regeneration, transformation products, ecotoxicity and secondary waste | Environmental chemistry, transport, material characterization, exposure and lifecycle assessment | Optimize removal versus energy, regeneration and secondary impact; keep uncertain hazards visible | Test relevant matrices and realistic exposure; quantify transformation/by-products and full waste burden |
| **20. Safe-and-sustainable-by-design / nanomaterial lifecycle** | Exposure, dissolution, aggregation, persistence, mobility, criticality, worker protections, recovery and disposal | OECD SSbD/Safe(r) Innovation approaches, material-specific hazard/exposure evidence and lifecycle tools | Detect missing/contradictory evidence; trace claims and sample history; support transparent multi-objective review | Absence of hazard data is not evidence of safety; safety and sustainability are constraints from early design through end-of-life |

**The central insight:** nanotechnology has *families of physics and evidence contracts*. A reusable discovery coordinator is plausible; a universal “nano model” that predicts all 20 rows from one metric is not a credible initial architecture.

## 3. Paradigm shifts that unite these diverse fields

### Shift A — Design from desired function, then solve backward

Start with a measurable use-case target and constraints, then search structures, geometry, composition, process, and operating conditions that can plausibly achieve it. This is inverse design, but candidate generation is only step one.

For every candidate, require: constraint check → forward-model evaluation → uncertainty/applicability assessment → manufacturing/measurement check → independent review. For photonics this may mean a Maxwell solver; for atomistic catalysts it may mean a validated energy/force model and appropriate reaction analysis; for nanomedicine it means much more than predicting a nanoparticle descriptor.

### Shift B — Optimize the whole functional stack, not an isolated nanoparticle

The working object is often a stack: base material + nanoscale architecture + surface functionalization + process + substrate/device + operating environment. Performance may be dominated by the interface or the manufacturing process rather than the isolated material.

Represent each link in the stack with versioned identity and evidence; measure how uncertainty propagates downstream. Do not assume the bulk-material encoder or an atomistic model covers every layer.

### Shift C — Make process and characterization first-class data

Record how the sample was made and measured: equipment/protocol revision, environment, precursor or starting-material identity, process parameters, batch, operator/instrument where appropriate, calibration, failed attempts, and raw-data hashes. The same nominal material prepared differently may not have equivalent size distribution, defects, coating, or performance.

The 2026 review of AI in nanotechnology covers design, synthesis/process optimization, microscopy, spectroscopy, particle tracking and nanofluid simulation, emphasizing that the workflow extends beyond prediction alone: https://link.springer.com/article/10.1186/s44301-026-00038-4

### Shift D — Select measurements by expected decision value

A model's uncertainty is not automatically useful. Ask which next measurement would distinguish competing hypotheses or change a design decision. Compare standard Bayesian/uncertainty acquisition, diversity selection, random selection, and one Symthaea policy under the same measurement budget.

Use explicit utility, calibration and abstention. Report the full curve against cost/query count—not a single cherry-picked final score.

### Shift E — Build distinct fidelity ladders with typed bridges

Use cheap screening for triage, qualified simulations for domain-specific predictions, and physical measurements for claims that require experiments. Create a typed interface between solvers; never silently compare energies, optical spectra, biological endpoints or process yields that have different units or semantics.

MADE's convex-hull policy benchmark, OC22's initial-to-relaxed surface-energy task, nanophotonic electromagnetic optimization and nano-bio formulation evidence should remain separate experiments.

### Shift F — Design for safety and circularity before the candidate is “good”

Integrate hazard and lifecycle constraints from the beginning. The OECD's 2025 report on safe-and-sustainable-by-design tools inventories methods for combining risk assessment and sustainability when evidence is still incomplete: https://doi.org/10.1787/e411a4b7-en. Keep environmental transformation, exposure, regeneration, secondary waste and end-of-life separate from predicted performance.

### Shift G — Use negative evidence as a shared asset

Capture failed simulations, non-convergence, unstable phases, poor assembly yield, sensor drift, toxicological ambiguity, manufacturing defects and negative measurements. Record the conditions and limits of the observation. “Failed under this method and condition” must not be converted into “the material never works.”

### Shift H — Treat measurement quality as a discovery bottleneck

If microscopy segmentation, spectrum interpretation, metrology, or batch identity is unreliable, more generative search may increase the volume of bad evidence. Invest in calibration, reference specimens, cross-instrument comparisons, robust data lineage and blinded evaluations. Sometimes the most valuable next step is to improve the measurement, not generate another candidate.

### Shift I — Share evidence and policy, not scientific assumptions

Mycelix can preserve source provenance, rights, review, sample lineage and supersession. Symthaea can propose and prioritize evidence-seeking actions. Symtropy can visualize/scenario-test supported system models. Domain solvers still own their physics. No project's consensus, coherence or score can certify physical truth by itself.

## 4. What Symthaea should contribute—and what it should not pretend to do

| Capability to evaluate | Potential use across domains | What it cannot establish by itself |
|---|---|---|
| HDC representation/retrieval | Coarse metadata grouping, candidate-family retrieval, coverage and redundancy triage | Atomic, optical, biological or process-property prediction; physical novelty |
| Temporal models | Real kinetics, sensor drift, aging, assembly trajectories and cycling degradation | Temporal behavior inferred from unordered snapshots or absent time measurements |
| Active inference / predictive processing | Evidence-action selection where the action space, observations, costs and update rules are explicit | Thermodynamic free energy, calibrated uncertainty or physical truth by naming a cognitive metric “free energy” |
| Multi-hypothesis/counterfactual reasoning | Distinguishing explanations and choosing discriminating calculations/measurements | Causal discovery without designed contrasts and valid evidence |
| Metacognition / abstention | Flag unsupported claims, out-of-domain cases, missing data and conflicts | Safety certification or calibrated probabilities without held-out calibration |
| Multi-objective/Pareto methods | Expose trade-offs and avoid hidden single-score collapse | Objective weights, policy choices or values that were not declared |
| Independent verification/evidence contracts | Check splits, units, digests, query budgets, metrics and control outcomes | Scientific validity beyond the explicitly tested contract |
| Mycelix provenance/review | Track sources, licenses, claims, sample history and independent reviewer attestations | Truth from consensus |
| Symtropy models/visualization | Scenario analysis or presentation using appropriate, validated external models | Nanoscopic physics outside its implemented and validated model class |

Implementation rule: one contribution per ablation. Keep candidate-generation, forward solver/oracle, budget, and evaluation protocol fixed while testing the Symthaea part. If the result cannot be attributed to a component, do not claim that component helped.

## 5. Domain selection: how to choose the next real application

NANO-002 should choose one domain only after a source and tool audit. Use a written decision matrix, not intuition about which technology sounds exciting.

Score each candidate from 0 to 2 on each of these *readiness questions* (scores are provisional until evidence is collected):
1. **Target clarity:** Is there a numerical/observable target, unit, measurement condition and defined decision?
2. **Data lineage:** Can we freeze source bytes, rights, labels, split groups, and sample/protocol lineage?
3. **Oracle/model:** Is there an appropriate forward solver or experimental reference and a usable baseline?
4. **Leakage resistance:** Can related structures/batches/devices/trajectories be kept together for holdout?
5. **Compute/measurement feasibility:** Can a meaningful benchmark run within known hardware/cost/time limits?
6. **Symthaea differentiation:** Is there a testable selection, diversity, reasoning, retrieval, or calibration hypothesis beyond ordinary ML?
7. **Safety/readiness:** Can the initial work remain retrospective/computational with clear limitations and adequate expert oversight?

Do not total these into one misleading scalar if a critical gate fails. Rights, leakage control, valid target semantics and an appropriate baseline are hard gates. A high score elsewhere does not compensate for one of those failures.

### Candidate pilot shortlist (not yet validated by data audit)

- **Nanophotonics/metasurface inverse design:** promising for an independent physics domain because the forward model and target response can be specified; requires a pinned electromagnetic solver, geometry/fabrication constraints and independent forward re-simulation. Recent reviews describe both AI-assisted metadevice design and the shift toward scalable, manufacturable photonic inverse design:
  - https://doi.org/10.1039/D5NR02043C
  - https://www.nature.com/articles/s41578-026-00915-5
- **2D materials/electronic properties:** valuable for nanoelectronics and device-adjacent screening, but requires careful family-aware holdouts, DFT setup matching and a bridge from idealized calculated properties to measurements:
  - https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00155b
  - https://www.frontiersin.org/journals/nanotechnology/articles/10.3389/fnano.2025.1627210/full
- **Nano-characterization/metrology:** potentially useful because imaging and spectroscopy are central across many application families; needs reference annotations, cross-instrument splits and calibration-aware uncertainty rather than image-only accuracy.
- **DNA/colloidal self-assembly:** promising if accessible time-resolved data are available; coarse-grained models may answer many mechanistic questions, but assembly kinetics and defects need direct validation:
  - https://doi.org/10.1039/D4CS01095G
- **Nanofluidics/separations:** high societal value; likely needs mixed-feed, fouling, regeneration and operating-condition data, not just ideal single-molecule transport.

This is an audit shortlist, not a declaration that the datasets have already been acquired or that one candidate won a readiness score. NANO-002 is the gate for that evidence.

## 6. Recommended implementation sequence

1. **Complete the existing infrastructure gates.** Continue the OC22 data-integrity and pair-crosswalk work; keep the MADE method benchmark separate and reproduce its baseline.
2. **Audit three follow-on pilots against the readiness questions.** Suggested comparison: photonic inverse design, 2D-material properties, and nanoscale characterization. Identify one actual source, license, target, solver/model and baseline for each before choosing.
3. **Select one follow-on domain** based on real data and reference-tool feasibility; document why the other pilots were deferred.
4. **Define a minimal domain adapter** that preserves source identity, units, conditions, uncertainty, forbidden data and failure states. Prefer Rust-first Symthaea interfaces; external scientific tools may retain their own runtimes in isolated, pinned research environments.
5. **Run an equal-budget acquisition experiment** against random and standard baseline policies, one Symthaea contribution at a time.
6. **Independent verification and domain review.** Recompute metrics from frozen receipts and preserve failures/null results.
7. **Only then explore physical experiments** with domain experts, reviewed protocols, safe facilities, sample traceability, controls and independent characterization. Nothing here authorizes autonomous synthesis, biological work, environmental release, medical use or lab actuation.

## 7. Research references

- Majedi (2026), “Artificial intelligence in nanotechnology: current applications and future perspectives,” *Health Nanotechnology*: https://link.springer.com/article/10.1186/s44301-026-00038-4
- Kim & Kim (2025), “Inverse design of nanophotonic devices enabled by optimization algorithms and deep learning,” *Nanophotonics*: https://doi.org/10.1515/nanoph-2024-0536
- Schul et al. (2026), “Inverse design for scalable photonic systems,” *Nature Reviews Materials*: https://www.nature.com/articles/s41578-026-00915-5
- Mishchenko et al. (2026), “Deep learning methods for 2D material electronic properties,” *Digital Discovery*: https://doi.org/10.1039/D5DD00155B
- Cumberworth & Reinhardt (2025), “Models and simulations of structural DNA nanotechnology reveal fundamental principles of self-assembly,” *Chemical Society Reviews*: https://doi.org/10.1039/D4CS01095G
- OECD (2025), “Safe and sustainable by design tools, integrative systems and platforms for nanomaterials and nano-enabled products”: https://doi.org/10.1787/e411a4b7-en
- Han et al. (2026), “Artificial intelligence as a driver of sustainable materials and circularity,” *Nature Reviews Materials*: https://www.nature.com/articles/s41578-026-00912-8
- Panagiotakopoulou et al. (2026), “Artificial intelligence and machine learning in nanoparticle drug delivery systems,” *Nature Reviews Bioengineering*: https://www.nature.com/articles/s44222-026-00495-7

**Success condition:** this atlas is useful if it helps select a tractable, well-defined second domain and prevents scientifically invalid transfer of models, metrics or claims—not if every subfield is turned into code at once.
