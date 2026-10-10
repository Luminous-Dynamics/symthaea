# Quantum Chemistry Integration and Qualification Plan

**Status date:** 2026-10-10  
**Scope:** Source-level audit and design plan. This note does not claim Rust tests, the native solver, RDKit, or PySCF were executed in this session.

## Decision

**Keep and improve the existing symthaea-quantum-chemistry crate. Do not create a duplicate quantum-chemistry implementation.**

The repository already contains a substantial pure-Rust electronic-structure implementation. The next high-value step is to connect it to the chemistry graph/candidate path through a typed, provenance-preserving interface while fixing known validation gaps. Quantum chemistry should be one evidence-producing layer in a larger chemical-engineering stack, not a universal chemistry oracle.

Tracked integration issue: [#7321 — CHEM-QC-001](https://github.com/Luminous-Dynamics/symthaea/issues/7321).

## 1. What already exists

Source inspected at base commit 77b872fd116c7b6f44fedd82bb8c6100240caa73:

| Area | Existing capability | Important boundary |
|---|---|---|
| Molecular electronic structure | Gaussian basis functions and molecular integrals; overlap, kinetic, nuclear-attraction and electron-repulsion integrals; Schwarz screening | Numerical conditioning and reference agreement must be established per supported method/basis/system |
| Ground-state SCF | Restricted Hartree–Fock (RHF) and unrestricted Hartree–Fock (UHF), DIIS, optional direct-integral mode | RHF rejects open-shell multiplicities with an assertion; malformed inputs need typed preflight before solver calls |
| Density-functional theory | Self-consistent LDA Kohn–Sham calculation | PBE exchange is documented as post-hoc only; PBE correlation is not implemented. Do not call this a general PBE-DFT engine |
| Post-HF / excited states | MP2 (including frozen-core/SCS variants), CIS | Each method needs its own reference and applicability envelope |
| Geometry / vibration / thermochemistry | Geometry optimization; numerical-gradient/numerical-Hessian vibration pipeline; statistical-mechanics calculations | Finite-difference sensitivity and restricted thermochemistry scope are documented. Optimization/convergence does not prove a global minimum |
| Validation | Native benchmark helpers and a Nix .#qc-verify PySCF lane with comparison/diagnostic examples | Existing source documents unresolved energy discrepancies; validation needs machine-readable, tight, method-specific thresholds |
| Process discovery | SMILES parsing, normalization, valence/structure checks, scope policy, human-reviewable process certificates | The oracle explicitly does not perform QC: it lacks a graph-to-3D geometry path |

Primary code anchors:

- [Quantum chemistry crate overview](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/crates/domains/symthaea-quantum-chemistry/src/lib.rs)
- [Native QC validation helpers](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/crates/domains/symthaea-quantum-chemistry/src/validation.rs)
- [Molecular geometry representation](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/crates/domains/symthaea-quantum-chemistry/src/molecule.rs)
- [Reaction candidate oracle](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/crates/domains/symthaea-process-discovery/src/oracle.rs)
- [SMILES parser and declared unsupported features](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/crates/domains/symthaea-organic-chemistry/src/smiles.rs)
- [Existing PySCF verification environment](https://github.com/Luminous-Dynamics/symthaea/blob/77b872fd116c7b6f44fedd82bb8c6100240caa73/flake.nix)

## 2. Known scientific qualification debt

The QC source code itself documents these unresolved HF energy discrepancies:

- N₂ / STO-3G: reported error approximately +458 kcal/mol.
- H₂O / 6-31G: reported error approximately −206 kcal/mol.
- CH₄ / 6-31G: reported error approximately +27 kcal/mol.

These are source-documented historical measurements, **not re-run in this session**. The validation source also notes wide tolerances that can conceal material discrepancies. The correct next step is diagnosis against an independent implementation and primary reference values, not threshold relaxation.

Other material limits:

- The CCD coupled-cluster module is explicitly marked non-production.
- The native LDA SCF path is narrower than the umbrella phrase “DFT”; post-hoc PBE exchange is not self-consistent PBE.
- The current SMILES parser states that stereochemical syntax is tolerated but stereochemical information is discarded. Isotopes, disconnected graphs, and radicals are outside its declared scope.
- The native QC molecule representation expects explicit 3D atom positions in **Bohr**; the current reaction path has a graph and implicit hydrogen counts, not validated 3D coordinates.
- Some native APIs use assertions/panics for unsupported states (including RHF multiplicity and invalid electron count). A fallible request boundary should validate these preconditions first.

Therefore, do not use a converged SCF, internal consistency alone, composition-based stability estimate, orbital-Phi score, or HDC similarity as the decision predicate for reaction feasibility.

## 3. Recommended stack and division of responsibility

### Geometry proposal: RDKit ETKDG, behind an opt-in adapter

Use RDKit ETKDG as the first external conformer-proposal path, not as a quantum result or ground-truth geometry. Pin its version and random seed; preserve atom identity/order; add hydrogens explicitly; record stereochemistry; generate an ensemble for flexible molecules; and reject ambiguous/unsupported structures rather than inventing missing state.

The RDKit documentation describes ETKDG as distance geometry supplemented by torsion preferences and exposes a reproducibility seed. It is a useful proposal generator; conformer generation remains distinct from high-level geometry optimization and proof of a global minimum.

- [RDKit Book: conformer generation and random seeds](https://www.rdkit.org/docs/RDKit_Book.html)
- [RDKit Cookbook: ETKDG options](https://www.rdkit.org/docs/Cookbook.html)

The integration boundary must explicitly convert external coordinates (commonly Å) to the native engine's Bohr representation. Persist pre/post-conversion coordinate digest, unit, atom mapping, hydrogens, geometry-generator version/seed, conformer index, and optimization history.

### Electronic structure: native engine plus independent reference

Keep the native Rust engine as a backend. Use the repository's existing Nix .#qc-verify / PySCF lane as the first independent comparison path. PySCF supports HF/KS-SCF and a broad family of correlated and excited-state methods. Do not assume that two implementations agreeing proves experimental truth; agreement is one qualification signal within a declared method/basis/geometry envelope.

- [PySCF user guide](https://pyscf.org/user/index.html)
- [PySCF SCF guide](https://pyscf.org/user/scf.html)

When native and PySCF results disagree, preserve both raw outputs and compare integral/electronic/nuclear energy components using identical geometry, charge, multiplicity, method, basis, integration settings, and thresholds. Psi4 is a reasonable second independent implementation for unresolved or consequential cases, not a prerequisite for the first milestone.

- [Psi4 official manual](https://psicode.org/psi4manual/master/introduction.html)

### Reference data: NIST first, curated datasets later

Use NIST's Computational Chemistry Comparison and Benchmark Database (CCCBDB) for documented small-molecule calculated/experimental thermochemical properties, geometries, vibrational frequencies, and method/basis comparisons. Match the reference property and level of theory: experimental heat of formation is not the same quantity as raw electronic energy.

- [NIST CCCBDB summary](https://cccbdb.nist.gov/summaryx.asp)
- [NIST CCCBDB property index](https://cccbdb.nist.gov/cccbdbindexx.asp)

For larger regression collections later, evaluate QCArchive/QCFractal. Do not ingest a huge dataset before the narrow method/geometry/receipt protocol is stable.

- [MolSSI QCArchive documentation](https://qcarchive.molssi.org/)

### Thermokinetics and process scale are downstream

Quantum chemistry can contribute energies, gradients, frequencies and other calculated observables within supported methods. It does **not** automatically supply reliable free energies, barriers, solvent corrections, competing pathways, rate constants, selectivity, yield, reactor performance, or process safety.

Use an explicitly reviewed thermodynamic/kinetic mechanism and dedicated process solver downstream. Cantera, for example, has specific thermodynamic, chemical-kinetic, transport, reactor-network, and 1D-flame models. A QC result is one input to this chain, not a substitute for it.

- [Cantera official documentation](https://cantera.org/stable/reference/index.html)

## 4. Target architecture

SMILES/source structure  
→ parse + normalize + atom mapping  
→ explicit chemistry state (charge, spin, protonation, stereo, isotopes)  
→ bounded conformer proposal (versioned generator, fixed seed, ensemble)  
→ geometry sanity/valence/atom-order/units checks  
→ typed QC request  
→ native Symthaea QC backend  
→ independent reference backend (PySCF initially)  
→ exact comparison + method-specific validation policy  
→ qualified observable with uncertainty and provenance  
→ balanced reaction-energy calculation / explicit thermochemistry corrections  
→ separate kinetic model and process simulation  
→ human review and evidence-bounded certificate

Each arrow is a validation boundary. The request/result protocol must be solver-neutral but scientifically specific enough to prevent comparing unlike quantities.

### Required request identity

A stable QC job identity must bind:

- canonical molecular graph, atom mapping, and normalization history;
- charge, multiplicity, electron count, isotope and stereochemical state;
- coordinate bytes/digest, units, atom order, conformer ID, and optimization state;
- method, functional, basis-set identity/version, numerical thresholds and integration controls;
- environment/solvation/embedding assumptions, or an explicit “gas phase / none” value;
- solver name/version/build, runtime options, and resource policy.

### Required result receipt

A result receipt should capture:

- exact request and geometry digests;
- run state (not-run, rejected, failed, non-converged, converged, independently compared, qualified) with typed reason;
- energy components, explicit units, SCF iterations, convergence thresholds, finite-value and matrix-conditioning checks;
- solver version and basis-data identity;
- raw input/output artifact digests and machine-readable summaries;
- reference backend/version, comparable settings, reference energy, signed and absolute error, allowed threshold, and comparison state;
- limitations and permitted claims.

A single boolean is inadequate because it hides whether the run was never attempted, skipped, numerically converged but unverified, or compared to an inapplicable reference.

## 5. Ordered implementation milestones

### Q0 — Make current status testable

1. Keep the three known benchmark discrepancies as explicit failures/known defects; do not widen tolerances.
2. Make the existing PySCF diagnostic lane emit structured JSON/CSV for every benchmark entry, including failures and skipped runs.
3. Bind records to exact geometry, method, basis, package version, and repository HEAD.
4. Separate native unit tests from external-reference tests; report skipped distinctly from passed.
5. Root-cause N₂/STO-3G, H₂O/6-31G and CH₄/6-31G by comparing components and matrices against PySCF with identical inputs.

### Q1 — Add a typed, fallible request boundary

1. Validate finite/nondegenerate geometry, supported elements/basis, electron count, and spin/multiplicity consistency before invoking panic-prone APIs.
2. Return typed preflight errors without silently changing the underlying physical model.
3. Carry model maturity and supported-input envelope explicitly.
4. Return NotEvaluated/Rejected/Failed states honestly; never use a default value that looks like an energy.

### Q2 — Reproducible 2D-to-3D proposal

1. Add an optional RDKit ETKDG adapter with version pin, fixed seed, explicit hydrogens and atom map.
2. Reject structures whose stereochemistry was discarded when stereo matters; do not backfill a guess.
3. Generate multiple conformers for flexible molecules and retain each result, failed embedding, and optimization status.
4. Test atom-order mapping, symmetry, rigid-motion invariance, unit conversion, minimum interatomic distance, and digest stability.

### Q3 — Non-gating certificate attachment

Attach QC outcomes to ProcessCertificate behind an opt-in integration. Existing structural-validity and scope-policy semantics remain unchanged until scientific qualification is complete. Certificates are for human review; neither the generator nor a QC result may synthesize, procure, or act.

### Q4 — Reaction energetics and downstream chemistry

Only after Q0–Q3: require atom and charge balance; compute matched reaction energies using the same method, basis, environment and state convention; evaluate conformer sensitivity; add explicit thermal/solvation/standard-state corrections where supported; then hand a versioned mechanism to a separately qualified kinetic/process model. Transition-state barriers and competing pathways require dedicated validated work.

## 6. Minimal regression corpus

Start with H₂, HeH⁺, LiH, HF, H₂O, NH₃, CH₄ and N₂, which already appear in the native validation helpers. For each entry, capture the exact fixture geometry, expected reference property/source, exact method/basis/settings, native convergence/result, comparable PySCF result, signed/absolute errors, chemically justified threshold, and qualification status. Do not use a broad tolerance to average over molecules. Keep invalid/distorted inputs in separate rejection tests.

## 7. Qualification vocabulary

Treat these as independent states rather than one chemically-feasible flag:

1. **Graph-valid:** supported parser and declared graph rules accept the structure.
2. **Geometry-proposed:** a named generator emitted coordinates with known units and atom map.
3. **Geometry-checked:** finite/nondegenerate geometry and required atom/stereo state survived checks.
4. **SCF-converged:** a named solver reached its stated numerical criteria.
5. **Reference-compared:** an independent implementation with comparable settings is available and discrepancy is computed.
6. **Method-qualified:** this method/basis/property/input regime passed a declared benchmark and applicability envelope.
7. **Reaction-energy-estimated:** balanced reaction and comparable species calculations produced an estimate with conformer/model uncertainty.
8. **Thermochemistry-qualified / kinetic-model-qualified:** only where corresponding corrections, models, and evidence exist.
9. **Experimentally validated:** only when independent experimental evidence actually supports the claim.

None of levels 1–8 implies level 9. A favorable reaction energy is not a barrier; a barrier estimate is not selectivity or yield; a yield model is not safe process design.

## External references consulted

- RDKit ETKDG: https://www.rdkit.org/docs/RDKit_Book.html
- PySCF user guide / SCF: https://pyscf.org/user/index.html and https://pyscf.org/user/scf.html
- Psi4: https://psicode.org/psi4manual/master/introduction.html
- NIST CCCBDB: https://cccbdb.nist.gov/summaryx.asp
- MolSSI QCArchive: https://qcarchive.molssi.org/
- Cantera: https://cantera.org/stable/reference/index.html


## Current implementation checkpoint (2026-10-10)

The first code increment is proposed in [PR #7322: typed, resource-bounded QC request preflight](https://github.com/Luminous-Dynamics/symthaea/pull/7322). Its current head at this checkpoint is `6ef6cb3264b6ccdcbc62ec7d977a013865ff22b1`.

The proposal adds a serializable request schema, explicit method/basis/environment/Bohr declarations, charge/electron/spin consistency checks, finite and non-coincident geometry checks, supported basis-data checks, SCF-setting validation, and request-level atom/basis/iteration budgets. It retains the built basis and overlap-derived independent-basis rank in an immutable preflight result, checks electron capacity after linear-dependence removal, validates the maximum residual in Xᵀ S X − I against 1e-6, and uses signed arithmetic for the spin split. It does not run SCF or claim energy accuracy. A separate focused workflow targets the request tests. The residual gate is not a solver convergence proof; #7325 tracks convergence reporting for the Jacobi eigensolver.

**Verification state:** the PR's current GitHub Actions workflows were queued at the last status query; earlier attempts on prior commits were cancelled when the branch advanced. No build or test pass is claimed. Check the live PR checks for the current head before treating the code as verified.


## Machine-readable PySCF regression comparison (2026-10-10)

A follow-on implementation is proposed in [PR #7323](https://github.com/Luminous-Dynamics/symthaea/pull/7323):

- `crates/domains/symthaea-quantum-chemistry/examples/qc_native_reference_fixtures.rs` emits schema v2 with exact geometry, charge, multiplicity, basis/method labels, native status/result, legacy reference target, and the native AO overlap matrix. It preserves solver panics/non-convergence as explicit records rather than silently dropping a fixture.
- `scripts/qc_reference_compare.py` verifies the declared Git commit/tree relationship, checks atom symbols against atomic numbers and RHF spin preconditions, evaluates the *same serialized coordinates* in PySCF, records solver version/settings and the exact input-report SHA-256, and includes every input case in the result. It requires equal AO basis-function counts and elementwise native/PySCF overlap-matrix agreement (maximum absolute residual ≤ 1e-8) before comparing total, electronic, and nuclear-repulsion energies plus each backend's energy-decomposition residual. The separate energy tolerance defaults to 1e-6 Hartree; that is also a hard maximum, and the CLI rejects larger values. It refuses to overwrite the native input and atomically publishes the comparison report before returning failure.
- The legacy reference tolerances stay as contextual metadata and are **not** reused as the cross-backend comparison threshold. The current known N2/STO-3G, H2O/6-31G and CH4/6-31G issues are expected to stay visible until root-caused.



The end-to-end numerical run is also wired as `.github/workflows/qc-numerical-diagnostic.yml`. It intentionally allows expected scientific discrepancies to make the diagnostic workflow fail, while uploading `/tmp/qc-native.json` and `/tmp/qc-comparison.json` as artifacts. This gives us the actual discrepancies to debug without turning report generation into a scientific pass.
### Usage

From the repository root in a Rust-enabled environment:

```sh
cargo run -p symthaea-quantum-chemistry --example qc_native_reference_fixtures > /tmp/qc-native.json
```

Then in the PySCF verification environment:

```sh
nix develop .#qc-verify --command python scripts/qc_reference_compare.py \
  --input /tmp/qc-native.json --output /tmp/qc-comparison.json
```

The command returns nonzero when any case fails or cannot be compared. A nonzero result with a complete JSON report is a useful failure artifact, not an excuse to loosen thresholds. Cross-backend agreement is a software-validation signal, not a claim of chemical accuracy or experimental validation.

**Verification status for the new scripts:** proposed in PR #7323. One focused workflow compiles the native example, runs crate unit tests, and exercises comparator policy with a mocked backend. A separate numerical diagnostic runs the real PySCF comparison and preserves reports as artifacts on failure. The fixture producer refuses to start on a dirty source tree and checks that HEAD/tree remain unchanged after calculation; the comparator verifies commit/tree consistency against local Git objects, requires the report revision/tree to equal the current checkout's HEAD/tree, and rejects a dirty worktree. This is consistency checking, not a signature or build attestation. The PySCF shell explicitly includes Git for source verification. No local execution or actual PySCF numerical result is claimed here; the real comparison must still run in `.#qc-verify`.
