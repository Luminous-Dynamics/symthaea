# First-Principles Chemical Engineering: Physics-Stack Audit and Execution Plan

**Prepared:** 2026-10-10  
**Canonical repository:** `Luminous-Dynamics/symthaea`  
**Baseline commit inspected:** `77b872fd116c7b6f44fedd82bb8c6100240caa73`  
**Document class:** source-level research and engineering plan; not experimental or numerical qualification evidence.

## Executive decision

Yes: Symthaea should be developed toward first-principles chemical engineering. The useful architecture is a **hierarchy of physical models with explicit interfaces, applicability ranges, uncertainty, verification and evidence maturity**, not a new monolithic chemistry solver.

This work belongs in the dedicated `Luminous-Dynamics/symthaea` repository. Its root contains the Rust workspace and canonical crate paths such as `crates/domains/symthaea-process-discovery`. The monorepo has overlapping snapshots and some potentially more advanced bridge implementations; those differences must be explicitly reconciled rather than silently treated as canonical. This PR therefore adds only research/audit documentation to the dedicated repository and proposes no automatic code migration from the monorepo.

The overarching physical chain is:

```
electronic structure / interatomic models
→ molecular and materials properties
→ statistical mechanics and thermodynamics
→ equilibrium and reaction kinetics
→ species / momentum / energy transport
→ unit operations and process flowsheets
→ constrained optimization, economics, and safety case
```

For each question, select the cheapest model whose **domain of validity has been demonstrated**. An electronic energy is not automatically a reaction enthalpy, Gibbs energy, activation barrier, rate constant, reactor prediction, or plant design. Where physical models require constitutive relations, empirical correlations, experimental data, or a calibrated mechanism, those are explicit inputs with provenance—not invisible “first-principles” magic.

Symthaea's comparative advantage should be to formulate hypotheses, choose among supported models, connect them across scales, track assumptions/uncertainty, plan the next informative calculation, discover contradictions, and produce reproducible evidence-bearing design comparisons. HDC equation similarity and generative reasoning can help propose models; neither is a numerical solution or physical validation.

## Existing architecture to compose, not duplicate

The repository already has a broad relevant engineering stack and active issue ownership. This document is a cross-cutting audit and execution sequence, not another quantity system, manufacturing ontology, evidence store, solver orchestrator, or authority pathway.

| Existing owner | Role in this program | Canonical coordination |
|---|---|---|
| `symthaea-organic-chemistry` | Molecular structure and cheminformatics representation | Existing chemistry parser / explicit representation limits |
| `symthaea-quantum-chemistry` | Bounded electronic-structure calculations | Existing Q0–Q6 work and PySCF comparison environment |
| `symthaea-process-discovery` | Structural reaction candidates, validity/scope and candidate certificates | Its current oracle intentionally does not claim quantum-chemical feasibility |
| `symthaea-materials` | Materials properties, candidates and aging/discovery interfaces | #6185, #6224, #6225, #6227 and existing MAT programs |
| `symthaea-thermofluids` | Applied hydraulic / thermal formulas | Physics type system #6868 and engineering coupling #6173 |
| `symthaea-continuum-physics` | Native PDE/field prototypes, including 2D incompressible Navier–Stokes | This audit's CFD qualification follow-up #7316; maturity envelope #6871 |
| `symthaea-numerical` | Shared roots, quadrature, interpolation, fixed-step ODE routines | Use only within validated numerical scope; chemistry stiffness needs a suitable solver |
| `symthaea-sim-bridge` | Solver-agnostic simulation orchestration and run/result types | #3681 / #5659 and #6173 |
| `symthaea-openfoam-bridge` | External CFD adapter | #5659 owns transitive solver-input closure; do not invent a competing digest contract |
| `symthaea-engineering` / ETK / `symthaea-formal-safety` | Engineering orchestration and assurance boundaries | SEC-002B #5124, SE-VV #3697, SE-MODEL #3759 and #6871 |
| MFG-PROC / WATER-MFG | Manufacturing-process semantics and water-system application owner | #5686 and #5966 |

**Ownership rule:** use #6868 for compositional physical types and units; #6871 for model/faculty maturity; #3697 for verification/validation/credibility; #6173 for inter-model interface transfer and conservation; #5659 for transitive external-solver input closure; #5124 for removing implicit safety-obligation discharge. This work must not create parallel versions of those primitives.

## Static source audit at the exact baseline

The observations below were made by fetching source from the dedicated repository at the baseline commit above. They are **static findings**, not claims that a failure was reproduced at runtime. Tests were not executed as part of this documentation PR.

### 1. Reaction discovery stops before 3D energetic feasibility

`crates/domains/symthaea-process-discovery/src/oracle.rs` describes its gates as normalization → structural validity → scope, with a composition-stability estimate kept advisory. It explains that quantum-chemistry feasibility is deferred because the candidate pipeline does not yet have a qualified 2D-to-3D geometry generator. The crate manifest has no dependency on `symthaea-quantum-chemistry`.

That is an appropriate honest boundary. A structurally conserved reaction candidate is not proof that the reaction is energetically favorable, kinetically accessible, selective, experimentally realizable, or safe. Do not add an implicit quantum “pass” until input representation, geometry, method applicability, convergence, reference comparison, and evidence status are represented explicitly.

`ProcessCertificate` records a normalized molecular graph and says the item is a computational candidate, not a synthesis instruction. Preserve that separation while extending the evidence vocabulary.

### 2. Composition-stability model must remain advisory

The current `symthaea-materials::compound_stability` source uses a simplified electronegativity-variance estimate and ideal-mixing entropy. It does not include the structural/phase/reference-state model required to treat it as a general molecular stability oracle; its comments themselves characterize it as first-order and note the need for DFT for accuracy. The process-discovery oracle documents why its sign structure makes it unsuitable as a hard discriminator for this organic candidate space.

Keep its numeric output clearly labeled as heuristic telemetry. Do not promote `is_stable`, a confidence field, or a scalar score into an acceptance/rejection gate without a separately defined model and benchmark.

### 3. Native continuum Navier–Stokes needs numerical qualification

`crates/domains/symthaea-continuum-physics/src/navier_stokes.rs` implements a 2D incompressible projection-method prototype.

Source-level issues to qualify:
- `NavierStokes2D::new` chooses `dt = 0.1 * dx * dx / nu.max(1e-6)`, a diffusion criterion, even though the velocity update also discretizes advection. No velocity-dependent advective CFL condition is visible in that constructor.
- The pressure Poisson solve uses a fixed 50 Jacobi iterations and exposes no residual-based convergence result.
- The current tests check a quiescent flow, nonzero finite kinetic energy / positive enstrophy in a lid-driven case, and Reynolds number. Those are smoke tests, not quantitative error bounds, divergence/mass-conservation evidence, or grid/time-step convergence.

These findings are now recorded in dedicated repository issue [#7316](https://github.com/Luminous-Dynamics/symthaea/issues/7316), composed with #6871 and #3697. Until the validation targets are met, classify this as a research prototype / fast-screening solver, not general CFD engineering evidence.

### 4. Thermofluids formulas are useful, but raw-float interfaces are weak contracts

`symthaea-thermofluids` already provides Reynolds number/flow regime, Bernoulli head, Darcy–Weisbach head loss, continuity, Carnot efficiency, Fourier conduction, Newton cooling, and idealized engine work. These are useful hand-checkable building blocks.

The public functions accept raw `f64` inputs and do not validate all physically required domains. Before these become generic engineering primitives, integrate with the gradual physical-type work in #6868 and define explicit validated entry points for finite SI values, positive viscosity/dimensions/temperatures where applicable, denominator constraints, and physical ordering of heat reservoirs. Preserve current valid-input compatibility where practical. Add tests for invalid inputs as well as analytic reference values.

Do not invent another quantity/unit subsystem: #6868 owns the broader physical type system and the engineering semantics effort owns transfer meaning.

### 5. Molecular-to-continuum approximations need unit/domain contracts

`crates/domains/symthaea-continuum-physics/src/coarse_graining.rs` includes Clausius–Mossotti, a rough HOMO–LUMO-gap-to-polarizability estimate, an Einstein-model heat capacity expression, gap-based relative conductivity, and other physics approximations.

A concrete contract ambiguity: the Clausius–Mossotti formula uses SI polarizability, while its nearby documentation says `m³ or atomic units`; conversion is implemented separately. Several other functions do not make returned unit conventions explicit or guard all nonphysical inputs (for example nonpositive temperature, non-finite values, or a singular denominator). This warrants dimensional and domain validation; it does not imply every approximation is useless.

Keep approximations separately labeled, and only promote a result after formula, unit conversion, limits, numerical stability and reference cases are checked.

### 6. OpenFOAM adapter in the dedicated repository is thin

At the same baseline, `crates/bridges/symthaea-openfoam-bridge/src/lib.rs` is a small adapter exposing a solver command (default `simpleFoam`), case directory, and a dry-run path. Its domain allowlist accepts Aerospace, Environmental, Mechanical and Systems, but not the existing `EngineeringDomain::ChemicalProcess` variant. It does not currently expose the larger input-tree digest / fresh instrumented metrics boundary observed in a separate monorepo snapshot.

This is an important **cross-repository drift finding**: do not assume a monorepo implementation is already present in canonical Symthaea, and do not paste it wholesale without checking current owner contracts. #5659 already owns transitive external-solver input closure; #3681 owns the bridge spine. Port any stronger adapter behavior to this repository only through those semantics, with exact-input identity, fresh output artifacts, solver/version and parser identity, convergence/residual metrics, bounded execution, and adversarial tests.

Adding `ChemicalProcess` to the allowlist alone would not qualify reacting flow. A real reacting case additionally requires a declared solver family, mesh and boundary conditions, thermodynamic/transport models, reaction mechanism digest, species mapping, valid regime, solver residuals, conservation checks, and a separately reviewed model applicability envelope.

### 7. Engineering safety evidence must not be inferred from generic convergence

`EngineeringManager::evaluate_concept` in the inspected `symthaea-engineering` source marks all obligations with `EvidenceKind::Simulation` as legacy `Discharged` when any simulation result has `converged=true`. That status mutation is not visibly matched to each exact requirement/result. This is already captured by canonical [SEC-002B #5124](https://github.com/Luminous-Dynamics/symthaea/issues/5124), whose implementation sequence is strict; this audit does not create a duplicate fix issue or substitute for that prescribed branch lineage.

Likewise, generic convergence is not proof that the model applies, that conservation holds, that an external result is current, or that a requirement-specific criterion was met. Compose with #5124, #3697 and #6871.

### 8. Equation HDC retrieval is not numerical physics

The `symthaea-physics-bridge` equation catalog encodes equation identity/structure, symmetries and dimensional signatures into hypervectors. This can help retrieve relevant equations and form analogies. It does not solve the equation, demonstrate dimensional correctness of an arbitrary generated expression, verify boundary conditions, establish solver stability, or validate a model against reality.

Use catalog similarity to propose/retrieve candidate model forms. Require symbolic/type checks, executable numerical methods, reference tests and model-maturity evidence before those models inform engineering decisions.

## First-principles program: staged plan

### Phase 0 — Baseline and source-of-truth control

- Pin the exact Symthaea baseline, Rust toolchain, workspace lockfile, and relevant fixture/data digests.
- Record commands and outputs for `cargo test -p symthaea-process-discovery`, `cargo test -p symthaea-quantum-chemistry`, `cargo test -p symthaea-thermofluids`, `cargo test -p symthaea-numerical`, `cargo test -p symthaea-continuum-physics`, `cargo test -p symthaea-sim-bridge`, `cargo test -p symthaea-openfoam-bridge`, and `cargo test -p symthaea-engineering`.
- Run the chemical corpus auditor in offline/replay mode as well as any opt-in network mode. Preserve the raw report and counts per outcome, not only the headline certification ratio.
- Compare canonical Symthaea and monorepo implementations file-by-file before porting any changes; record which source is selected and why. No copied code is presumed authoritative by path/name alone.
- If toolchain limits, build time, or known unrelated failures prevent a test, preserve the exact failure; old documentation is not fresh passing evidence.

**Exit:** reproducible baseline evidence, known source drift, and no unverified claims carried forward.

### Phase 1 — Physical type, conservation and evidence contracts

Compose the types work in #6868 and maturity envelope in #6871. For scientific parameters and fields record, at minimum:

- quantity kind, coherent unit and conversions;
- scalar domain and domain refinements (finite, positive, nonzero, bounded, normalized, etc.);
- state/derivative/PDE semantics and relevant coordinate/frame/mesh/time basis;
- phase, temperature, pressure, composition, charge/spin/standard state where relevant;
- whether data are observed, assumed, inferred, calculated, calibrated, or experimentally validated;
- model/backend/version, configuration and dataset digests, applicability limits and error bounds;
- convergence/residual status, uncertainty, and explicit unsupported/inconclusive/error reason.

For each reaction mechanism, use a stoichiometric matrix (S) and element/charge-composition matrix (A); verify (A S = 0) for the declared conserved quantities. Do not hide incomplete species definitions, non-stoichiometric transformations, or unit mismatches with fallback values.

A physical solver result should have explicit statuses such as `NotAttempted`, `SupportedWithinDomain`, `UnsupportedDomain`, `Inconclusive`, or `FailedCheck`. Do not reduce model maturity, convergence, conservation, validation and authority to a single scalar or `verified` boolean.

### Phase 2 — Qualify molecular geometry and electronic structure

- Keep external stereochemical/charge/isotope input identity explicit. The existing internal SMILES parser's representation losses cannot be silently treated as irrelevant for geometry or energetics.
- Evaluate an optional, pinned RDKit conformer-generation adapter (ETKDG parameters, random seed, conformer count, optimizer/force-field identity, status, and atom/coordinate digests). A conformer proposal is not proof of a global minimum or a reaction pathway.
- Connect `symthaea-quantum-chemistry` as an optional calculation provider only after geometry and model applicability are explicit.
- Benchmark matching species/calculations against PySCF with identical method, basis, charge/spin, geometry, and convergence settings. Prefer invariant observables and energy/eigenvalue comparisons over raw matrix entries where orbital sign conventions can differ.
- Keep electronic energy, zero-point-corrected energy, enthalpy, Gibbs energy, barrier, rate constant and equilibrium constant as different quantities with different prerequisites.

### Phase 3 — Thermodynamic property and equilibrium layer

Build on qualified reference data and external property packages before writing a broad new package. Needed capabilities include species thermochemistry, heat capacities, phase identity, equations of state, activities/fugacities, chemical potentials, and phase/equilibrium calculations, each with a declared model and applicable temperature/pressure/composition range.

Use NIST Chemistry WebBook records where the species and state match. Preserve exact record/source, phase, temperature, pressure/standard-state convention, unit conversions, uncertainty when reported, and every derived formula. Never compare raw electronic `ΔE` directly against an experimental `ΔH` or `ΔG` without the required thermal, entropy, phase and reference-state treatment.

### Phase 4 — Kinetics with qualified mechanisms

Represent reactions as typed species/rate-law/mechanism records with stoichiometry, units, parameter source, and applicability envelope. Verify elemental and charge conservation, positivity of concentrations where required, dimensional correctness of rates, and thermodynamic consistency/detailed balance where the mechanism claims them.

Use a qualified stiff integrator for stiff reaction systems. The existing fixed-step RK4 remains appropriate for selected elementary ODE references but should not become the default engine for stiff chemistry. Cantera is a strong candidate for an optional adapter because its science reference describes thermodynamic phases, chemical rates, transport properties, and 0D/1D reactor models. Pin the precise Cantera release and mechanism file digests.

### Phase 5 — Couple species, momentum and energy transport

The generic reacting-mixture equations should be implemented only with explicit assumptions and compatible closure models:

- species: accumulation + convection = diffusion + chemical source;
- momentum: the correctly chosen incompressible/compressible flow regime;
- energy: accumulation/advection/conduction plus the appropriate pressure/work and reaction-heat terms;
- constitutive relationships: equation of state, diffusion/multicomponent transport, viscosity, conductivity, phases and interfacial transfer;
- geometry/mesh/boundary and initial conditions with exact provenance;
- time-step limits, solver tolerances, residuals, conservation diagnostics and refinement evidence.

Do not merely attach source terms to a CFD field. Prove that the chosen coupling conserves its declared invariants, converges as specified, and matches an independently qualified reference. #6173 owns semantic cross-domain field transfer; #5659 owns the external solver input closure. Use OpenFOAM's published verification/validation cases and independently versioned inputs for differential benchmarks where suitable.

### Phase 6 — Process flowsheets and one bounded application

Only after the lower layers have qualified models should we compose pumps, heat exchangers, mixing/splitting, reactors, separators/membranes, recycle, water/waste and utility systems into a process flowsheet. DWSIM and IDAES are candidates for broader process-system modeling; use them behind versioned adapters rather than mandatory Rust build dependencies. Audit all dependency and distribution licenses before shipping integrated packages.

The best early application is a **simulation-only, non-potable low-head circulation/filtration test module** aligned to WATER-MFG #5966 and the existing engineering/product pilot portfolio. Begin with hydraulic and thermal characterization and a transparent baseline; only add treatment chemistry after the relevant feedstream, speciation, separation mechanism, material state, measurement plan and data sources are declared. This pilot must not claim potability, pathogen removal, public-health safety or operating authority.

A successful end-to-end report must include the baseline, model scope, mass/element/charge/energy balances, unit-operation efficiencies, waste/residual pathways, sensitivity analysis, uncertainty, model/data versions, output digests, reference comparisons, limitations and a clear list of unanswered questions.

## External research and reusable tools

Use proven tools in their appropriate roles; keep every adapter optional and evidence-carrying.

- **RDKit** — molecular representation and reproducible conformer proposals. The 2026.03 documentation describes ETKDG parameter controls including an explicit random seed; still record all relevant settings and optimizer outputs. https://rdkit.org/docs/RDKit_Book.html
- **PySCF** — independent electronic-structure comparison for methods that match the in-repo calculation. https://pyscf.org/user/index.html
- **NIST Chemistry WebBook, SRD 69** — reference thermochemistry for enthalpy of formation, entropy, heat capacity and reaction thermochemistry, with phase and standard-state distinctions that must be preserved. https://webbook.nist.gov/chemistry/ and https://webbook.nist.gov/chemistry/guide/index.html.en-us.en
- **Cantera** — thermodynamics, chemical kinetics, transport and reactor networks. https://www.cantera.org/stable/reference/index.html and https://cantera.org/
- **OpenFOAM** — external CFD; official verification/validation catalogue includes reference cases for laminar/turbulent flow, heat transfer, combustion and chemistry. https://doc.openfoam.com/2306/examples/verification-validation/
- **DWSIM** — process simulator with thermodynamic property models and unit-operation/flowsheet scope. https://dwsim.org/help/main/index.html
- **IDAES** — process systems modeling and optimization. https://idaes.org/overview/

External tool capability is not local qualification evidence. Each selected version, mechanism/property package and case must be qualified against the exact intended task.

## Explicit non-goals

- No novel-reaction generation or synthesis-instruction output is added by this audit.
- No solver result is automatically allowed to discharge a safety obligation; SEC-002B #5124 remains the prescribed workstream.
- No external solver is added as a mandatory workspace dependency in this documentation phase.
- No drinking-water/health, process safety, regulatory compliance, production yield, scale-up, or plant-readiness claim is made.
- No monorepo implementation is silently copied into canonical Symthaea. Source drift is a separate item to resolve through owners and the appropriate existing interfaces.

## Implementation follow-on (2026-10-10)

The first additive code slice is under review in [THERM-VALID-001 #7318](https://github.com/Luminous-Dynamics/symthaea/issues/7318) and [PR #7319](https://github.com/Luminous-Dynamics/symthaea/pull/7319). It adds checked numeric-domain entry points for the existing thermofluid formulas while preserving legacy APIs. This is intentionally narrower than unit/dimensional validation, solver qualification, or model validation; those remain owned by #6868 / #3697 / #6871. As of this update, PR #7319 is ready for review and its CI jobs are queued; no passing test result is claimed.

## Evidence status at submission

- Canonical Symthaea source inspected at exact HEAD `77b872fd116c7b6f44fedd82bb8c6100240caa73`: process-discovery oracle/certificate/manifest, quantum-chemistry linkage, thermofluids, numerical ODE routines, continuum Navier–Stokes/coarse-graining, materials stability, engineering facade, simulation bridge, OpenFOAM bridge and physics equation catalog.
- Existing issue ownership inspected: #5124, #5659, #5686, #5966, #6173, #6185, #6224/#6227, #6868, #6871.
- External docs reviewed: Cantera, RDKit ETKDG, NIST Chemistry WebBook, OpenFOAM V&V, DWSIM, PySCF and IDAES.
- Code changes in this document-only PR: none.
- Tests run: none.
- Runtime defect reproduction: none. Navier–Stokes and obligation-discharge findings remain source-level findings pending their canonical qualification work.
