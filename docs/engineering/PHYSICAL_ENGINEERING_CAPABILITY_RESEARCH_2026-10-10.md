# Physical Engineering Capability Research

**Research date:** 2026-10-10  
**Status:** Research and architecture recommendation; not a capability qualification  
**Scope:** Robotics, modular tooling, CAD/manufacturing, circuits, optics/photonics, acoustics, fluid/thermal systems, and plasma modeling.

## Decision

Symthaea should grow into an evidence-bounded physical engineering system: it can propose and compare designs, orchestrate specialist solvers, generate manufacturing artifacts, coordinate calibrated experiments, and update models from measured discrepancies. It should not be represented as a qualified autonomous engineer until each claimed capability has passed a domain-appropriate, reproducible evidence program.

Do not create a parallel engineering framework. Build on the existing engineering roadmap and issue graph, especially:

- [ROB-DESIGN-000 — evidence-bounded recursive robotics design bootstrap (#4795)](https://github.com/Luminous-Dynamics/symthaea/issues/4795)
- [ROB-EXP-001 — canonical robotics experiment protocol and evidence envelope (#4800)](https://github.com/Luminous-Dynamics/symthaea/issues/4800)
- [ROB-JOINTLAB-001 — instrumented single-joint physical robotics laboratory (#4814)](https://github.com/Luminous-Dynamics/symthaea/issues/4814)
- [SIM-EVID-001 — external-solver reproducibility evidence (#4440)](https://github.com/Luminous-Dynamics/symthaea/issues/4440)
- [ENG-SPICE-001 — parse real ngspice output (#5633)](https://github.com/Luminous-Dynamics/symthaea/issues/5633)
- [ENG-DESIGN-002 — executable engineering design pipeline (#6090)](https://github.com/Luminous-Dynamics/symthaea/issues/6090)
- [Typed physical-model faculty (#6870)](https://github.com/Luminous-Dynamics/symthaea/issues/6870) and [model evidence/maturity envelopes (#6871)](https://github.com/Luminous-Dynamics/symthaea/issues/6871)
- [EAC-000 — electro-acoustic engineering ownership (#5630)](https://github.com/Luminous-Dynamics/symthaea/issues/5630)
- [Engineering Trust Kernel (#1738)](https://github.com/Luminous-Dynamics/symthaea/issues/1738)

The central distinction is:

`proposal != simulation != calibrated measurement != qualification != actuation authority`

## Repository baseline observed on 2026-10-10

The [existing engineering roadmap](SYMTHAEA_ENGINEERING_ROADMAP.md) already calls for solver-neutral requests/results, digital-twin primitives, formal safety obligations, fabrication geometry, and narrowly scoped adapters. The current architecture is better described as a substantial scaffold with several real domain modules than as a complete autonomous design-to-manufacture system.

Concrete findings from the current source:

1. **External solver honesty is already fail-closed in important places.** The MuJoCo, OpenFOAM, and ngspice bridges can return deterministic dry-run fixture results. In their non-dry-run paths, they deliberately return an adapter error after execution because trustworthy result parsing and convergence evidence are not yet available. That prevents successful process launch from being mistaken for valid engineering metrics. Preserve this behavior until parser tests pass.
2. **The ngspice path is a good first vertical slice.** Issue #5633 already specifies real output parsing, raw artifact identity, solver identity, independently calculated RC/RLC expectations, missing-metric failure, and explicit separation of dry-run from external-solver evidence.
3. **Physical experiments already have a defined starting point.** Issue #4814 chooses a single instrumented joint instead of beginning with a full arm or humanoid. Issue #4800 defines experiment identity, test-article identity, calibration, environment, timing, and raw/normalized observations.
4. **The engineering data model needs stronger physical semantics.** Issue #6870 identifies string-valued units and requested signals as a source of invalid cross-physics connections. A type check should reject incompatible quantities before dispatch, not merely annotate them in a report.
5. **Model maturity must flow through every result.** Issue #6871 correctly separates textbook/analytical models, validated numerical methods, calibrated empirical models, research prototypes, frontier hypotheses, and synthetic/instrumental outputs. A result must never acquire a stronger evidence class merely because it completed a computation.
6. **CAD/manufacturing is not a blank slate.** The roadmap records existing STEP/NURBS/mesh and STL/3MF/toolpath work. External CAD should extend that foundation rather than replace it prematurely.
7. **ROS integration is not equivalent to qualified hardware control.** Use ROS 2 and `ros2_control` as interoperability layers where useful, but keep actuation authority in bounded, separately enforced control and interlock paths. A bridge, message type, or simulated controller is not proof of real hardware behavior.

## Source-level execution audit

A follow-up source review of the current `main` branch found concrete gaps that should shape the first implementation. These are code observations, not guesses about future behavior.

### What the current contracts already do well

- `CommandSolver` starts processes without invoking a shell, validates that command/timeout/output limits are non-empty and bounded, drains stdout/stderr without blocking forever on inherited pipes, terminates on timeout/output-limit breaches, and rejects a non-zero process exit.
- `SimulationResult::dry_run` stamps results with `ExecutionMode::DryRun` and a warning that the metrics are fixtures.
- `SimulationResult::is_engineering_evidence()` refuses to recognize a result with invalid normalized values, no metrics, no convergence claim, a non-`ExternalSolver` mode, or absent provenance fields.
- MuJoCo, OpenFOAM, and ngspice bridges deliberately refuse to promote a successful external process to engineering evidence until real result parsing is implemented.

These should be preserved as invariants during refactoring.

### Specific gaps visible in the current code

- `SimulationEvidence` currently stores mode, backend, solver version, input digest, output digest and parser version. Its completeness check requires these fields to be non-empty strings; it does not validate digest encoding, re-compute digests against retained artifacts, or bind the executable/environment/parser artifacts by digest.
- `SimulationResult` has a single `converged: bool`. It does not itself represent separate process, parse, convergence and requested-output statuses. `is_engineering_evidence()` also does not receive the originating request and therefore cannot prove that every requested metric was present; the adapter must enforce that before creating an accepted normalized result.
- `CommandSolver::execute()` captures stderr but returns only stdout on success. stderr is included in an error string for non-zero exit, but successful-run warnings are not returned as a first-class artifact. The process also inherits the calling process environment because the code adds configured environment variables without clearing the ambient environment.
- `CommandSolver` does not expose a per-run working directory. The current ngspice adapter invokes `ngspice -b input.sp` using a fixed relative filename; it does not pass explicit raw-output/log paths and does not archive the raw result. A result left over from an earlier run must never be able to satisfy the current run.
- The adapter currently discards successful stdout content after measuring its length in the placeholder error. It therefore has no parser-consumable evidence output yet.
- String-valued units and metric names are convenient for prototypes but cannot prevent dimensionally invalid coupling. Work #6870 should own the typed physical semantics rather than creating a second unit system here.

### First vertical-slice execution contract

For each run, create a unique workspace and a frozen manifest. Pin the absolute solver executable/package identity and version, bind the Nix closure/container image or other environment manifest, and use a controlled environment with an explicit allowlist. Use no shell interpolation; pass arguments as separate values. The working directory, solver inputs, all transitive `.include`/`.lib` dependencies, command arguments, configuration files, and output paths must be explicit and included in the run manifest.

The ngspice Version 47 manual documents batch output using a rawfile and log, e.g. `ngspice -b -r result.raw -o result.log input.cir`. Use a supported, deterministic output format and parse the actual raw/log artifacts; do not infer convergence from process exit or scrape an unconstrained terminal transcript alone. Pin the chosen solver version and validate the parser against that version's actual outputs.

Keep an execution report even for failed runs, separately tracking:

- **Process:** `NotStarted`, `Exited(code)`, `TimedOut`, `OutputLimitExceeded`, or `Terminated`.
- **Parsing:** `NotAttempted`, `Passed(parser_digest)`, or `Failed(reason)`.
- **Convergence:** `Unknown`, `Converged(criteria_id)`, or `NotConverged(reason)`.
- **Requested metrics:** `Complete` or `Missing(names)`, including units and source vector/column.
- **Artifacts:** digests and retrievable references for the request, dependency bundle, stdout, stderr, solver log, raw numerical output, parser, and normalized result.

A failed process or parser should still produce an auditable failure report, never a synthetic successful `SimulationResult`. Keep this run report distinct from a later engineering acceptance or safety-case decision.

### Failure-oriented test matrix for #5633 / #4440

The ngspice parser should not be considered ready from a single happy-path fixture. Its tests should include:

| Case | Required disposition |
|---|---|
| RC step-response netlist; expected outputs present | Parse, converge classification, units, and independent analytic comparison all succeed within declared tolerances |
| RLC/DC fixtures with independently calculated expected values | Golden numeric values and units match the declared solver/parser version |
| Invalid netlist or solver exits non-zero | Process failure recorded; no normalized engineering result |
| Solver exits zero but emits a convergence failure/warning for the requested analysis | Process success recorded separately; convergence remains failed/unknown unless solver-specific rules prove otherwise |
| Raw/log file missing, truncated, malformed, or from a previous run | Parser failure; stale artifacts rejected by unique workspace plus manifest/digest checks |
| Requested metric absent or renamed | Explicit missing-metric result; no partial pass for the full request |
| NaN/Inf, duplicate/conflicting metric, unsupported vector, or unit ambiguity | Fail closed or mark the individual metric invalid; do not silently coerce |
| stdout/stderr/raw/log exceed configured bounds or process times out | Run terminates within policy; failure artifacts/status are retained as allowed |
| Same netlist but changed included model library, solver binary, parser, or environment | The corresponding provenance identity changes; a previous result cannot be reused as if identical |
| Ambient `.spiceinit` or unrelated environment variable changes solver behavior | Isolated run remains deterministic or the changed configuration is explicitly bound |
| Dry-run produces numerically plausible fixture values | Test orchestration works, but result remains `DryRun` and cannot satisfy external-solver evidence |
| Tampered input/output bytes under an existing digest | Digest verification fails; no engineering evidence accepted |

The minimum numerical benchmark should be intentionally small. For example, an ideal first-order RC step response has time constant \(\tau = RC\) and capacitor voltage \(V_C(t)=V_{step}(1-e^{-t/\tau})\) for zero initial voltage. Compare a declared sample/measurement at \(t=\tau\) to \(1-e^{-1}\approx0.6321\) of the step, while accounting for the exact source, initial conditions, transient time-step settings, tolerances, and any model simplifications. This proves only that specific circuit model and parser path; it says nothing about hardware qualification.

## Review of the existing ngspice implementation PR

The repository already has an implementation lane in [ENG-SPICE-001 PR #5658](https://github.com/Luminous-Dynamics/symthaea/pull/5658). Do not create a duplicate implementation. Its design narrows accepted input to built-in R/C/L elements, independent V/I sources, a small set of analyses/output directives, explicitly requested scalar `.measure` results, and rejection of model/control/file-loading surfaces until the transitive input closure is bound.

### Verified framing correction on the PR branch

The official [ngspice Version 47 manual, §2.1.1](https://ngspice.sourceforge.io/docs/ngspice-47-manual.pdf) states that the first physical line is always the title and the last line must be `.end` followed by a newline delimiter. The validator initially checked every line as if it were a candidate netlist statement and did not require these framing rules. A follow-up on the existing PR branch now requires an explicit comment-style title beginning with `*`, validates only statements after that title, requires terminal `.end`, and rejects a missing final newline. Regression cases cover an element in title position, blank title, missing `.end`, missing newline, and a statement after `.end`.

### Narrow physical-dimension guard added

At exact PR head `d121e9c5d783f0182a1a91d31509fa286073c29d`, the adapter now performs a deliberately constrained unit-dimension check for requested measurement forms. It recognizes `v(node)`, `i(source)`, `mag(v(node))`, and `mag(i(source))`; for `FIND` it requires an explicit `AT=` point, and summary operations are restricted to `MAX/MIN/AVG/RMS/PP`. It rejects a requested voltage measure declared in amperes (and the inverse) and fails closed on unsupported expressions. A subsequent source review also tightened exact vector parsing so compound expressions cannot pass merely because they start and end like a simple vector.

This is only a tranche-one whitelist, not a general SPICE expression type system or proof of physical validity. The longer-term owner must be the existing typed physical model work (#6870). The exact-head CI and PR Governance runs have been queued, but no result for that head has yet been verified. No new code or tests are claimed as passing until that gate completes.

### Remaining acceptance gates

1. **The output artifact is hashed then deleted.** The PR hashes the log and removes it immediately; the digest alone does not make the bytes retrievable. Add an immutable/content-addressed artifact reference in the evidence layer (#4440), or explicitly scope this tranche as parser provenance rather than full independent reproducibility. Capture successful-run stderr/warnings as artifacts in the generic subprocess evidence work.
2. **Version string is not full solver identity.** The PR records the first non-empty line from the version command, plus netlist and log digests. It does not yet bind the solver executable/package digest, runtime closure, controlled environment, or parser binary digest. Those belong in #4440 and must be completed before making stronger reproducibility claims.
3. **Real-solver qualification remains separate.** The three ngspice integration tests are marked ignored because they require the external solver. They must be executed against a pinned version and exact head with raw outputs retained; library unit tests alone do not qualify the real solver path.

The recorded CI run from 2026-09-25 passed the workspace nextest job, the default-feature `symthaea` library tests, security audit, and PR-governance check, but the overall run was red due to other workspace jobs (unrelated orphan modules, formatting, Clippy, Muse-theory, and Spore-size failures). Those results are historical and do not qualify the current PR head.

## Existing domain ownership discovered in follow-up research

A new repository pass shows that photonics, plasma engineering, and shared device composition already have dedicated owners. Future work should follow this issue graph rather than add separate parallel domain frameworks:

- [PHOT-ENG-000 #5668](https://github.com/Luminous-Dynamics/symthaea/issues/5668) owns photonics/laser design and solver architecture; [PHOT-001 #5679](https://github.com/Luminous-Dynamics/symthaea/issues/5679) owns exact photonic assembly/model semantics; [FIELD-004 #3593](https://github.com/Luminous-Dynamics/symthaea/issues/3593) owns read-only calibrated optical observations.
- [PLASMA-ENG-000 #5669](https://github.com/Luminous-Dynamics/symthaea/issues/5669) owns regime-explicit plasma engineering from cold atmospheric plasma through kinetic and fusion regimes; [FIELD-005 #3594](https://github.com/Luminous-Dynamics/symthaea/issues/3594) owns calibrated plasma diagnostic observations.
- [ENG-DEVICE-000 #5670](https://github.com/Luminous-Dynamics/symthaea/issues/5670) owns reusable power, magnetics, vacuum/gas, thermal, mechanical and component subjects shared across device families; [ENG-THERM-001 #5681](https://github.com/Luminous-Dynamics/symthaea/issues/5681) owns thermal networks and boundary/evidence semantics.
- [ENG-DEVICE-001 #5684](https://github.com/Luminous-Dynamics/symthaea/issues/5684) is the natural first *cross-domain composition benchmark*. It deliberately proposes two benign fixtures rather than one artificial machine: (A) a passive/low-power magneto-optic sensor bench using permanent-magnet bias, and (B) a low-energy vacuum/gas-manifold characterization bench using inert gas/air only. It can demonstrate canonical component identity → subsystem topology → prediction → as-built configuration → calibrated observation → preserved discrepancy, without requiring laser-source optimization or plasma ignition.

The implication is that the near-term physical systems program should have **one evidence and device-composition spine**, with physics-specific modules plugged into it. For an initial built instrument, choose #5684 fixture A if optical/magnetic/thermal measurement is available; otherwise start with the single-joint robotics lab #4814 to establish actuator identity and sim-to-real mechanics. Fixture B is a separate gas/vacuum capability and should not be bundled into fixture A merely to increase the number of subsystems exercised.

This is an architecture recommendation, **not permission to start the integrated benchmark before its foundations qualify**. The issue's own execution-gate note says not to copy catalog, vacuum, magnetic, photonic or thermal semantics into the integrated fixture. The current PR train includes [catalog identity #5676](https://github.com/Luminous-Dynamics/symthaea/pull/5676), [vacuum/gas #5674](https://github.com/Luminous-Dynamics/symthaea/pull/5674), [magnetics #5678](https://github.com/Luminous-Dynamics/symthaea/pull/5678), [photonics #5680](https://github.com/Luminous-Dynamics/symthaea/pull/5680), [thermal #5682](https://github.com/Luminous-Dynamics/symthaea/pull/5682), and [shared reference closure #5685](https://github.com/Luminous-Dynamics/symthaea/pull/5685). As observed in this review, several domain children remain draft; build and qualify the canonical identities/evidence corpus first, then compose by consuming those exact IDs. The benchmark's own milestone ladder (I0 identity composition → I1 analytical reproducibility → I2 closure-bound external numerical evidence → I3 exact as-built + calibrated observation binding → I4 preserved discrepancy → I5 preregistered redesign) should remain explicit. A successful synthetic fixture is not a physical pass.

### Regime-aware solver hierarchy

The existing [PLASMA-ENG-000 issue](https://github.com/Luminous-Dynamics/symthaea/issues/5669) correctly forbids one generic "plasma solver" capability. The adapter contract should include a mandatory regime profile (cold atmospheric, low-pressure low-temperature, thermal arc/jet, magnetized-fluid/MHD, kinetic/PIC, high-temperature fusion, laser-plasma interaction, or a specifically qualified profile), and each model/solver should declare its applicable regime, assumptions, species/chemistry coverage, collision/closure model, geometry/boundary assumptions, and evidence maturity.

The current repository's preferred initial solver progression is more precise than a generic PICLas-first strategy:

1. **Analytical and reduced-order checks:** existing Symthaea equations plus [PlasmaPy](https://docs.plasmapy.org/en/latest/) for documented formulary, particles, dispersion and diagnostic calculations. Treat it as a formula/analysis adapter, not a general high-fidelity plasma device simulator.
2. **Optical design / ray path:** [Optiland](https://www.optiland.org/docs/) for lenses, mirrors, ray tracing, tolerancing and optimization. Keep it distinct from full-wave EM simulation.
3. **Full-wave optics:** [Meep](https://meep.readthedocs.io/en/master/) for FDTD problems within its supported material/geometry models, and [MPB](https://mpb.readthedocs.io/en/stable/) for supported eigenmode/band-structure problems. They are complementary—not interchangeable—and GPL licensing/distribution implications must be reviewed before packaging.
4. **Fluid plasma / fusion edge cases:** evaluate [BOUT++](https://bout-dev.readthedocs.io/en/stable/user_docs/introduction.html) where its fluid equations and curvilinear/fusion-plasma domain fit. Evaluate Gkeyll separately for appropriate kinetic/fluid models.
5. **Particle-in-cell:** adopt a [PICMI](https://picmi.readthedocs.io/en/latest/) interchange projection with an explicit capability/loss report; the standard is still evolving and implementations need not support every option. Then qualify one concrete backend—[WarpX](https://warpx.readthedocs.io/) is the current issue's first candidate—before considering a second PIC backend.
6. **Acoustics/ultrasound:** keep existing acoustic domain ownership and FIELD measurement ownership; use k-Wave for the use cases its model supports, noting that k-Wave-II currently describes itself as under construction and in pre-release, with MATLAB R2023b or later required ([project status](https://github.com/ucl-bug/k-wave-ii)).

This order is based on the domain-owner issues and current public tool documentation, not a claim that these adapters are implemented or qualified. A regime/model compatibility error should block dispatch before compute resources are spent.

## Proposed external toolchain

Treat every external program as an independently versioned instrument behind a typed adapter. Keep native/heavy solver dependencies out of default workspace builds; run them in explicit development or deployment environments with pinned versions and reproducible inputs.

| Domain | Preferred candidate | What it contributes | Integration stance |
|---|---|---|---|
| Multibody robotics | [MuJoCo](https://mujoco.readthedocs.io/) | Contact, articulated dynamics, mechanism and controller simulation | First simulation adapter to harden; parse actual trajectories and diagnostics before reporting metrics |
| Robot middleware and hardware abstractions | [ROS 2](https://docs.ros.org/) / [ros2_control](https://control.ros.org/) | Device, actuator, sensor and controller interoperability | Integrate after command authority, lifecycle, limits, watchdog and independent interlock contracts are verified |
| Circuit/electrical analysis | [ngspice](https://ngspice.sourceforge.io/docs/ngspice-44-manual.pdf) | DC, AC and transient circuit simulation | Highest-leverage real-solver vertical slice; request machine-readable outputs and preserve rawfiles |
| Structural mechanics | [OpenSees](https://opensees.berkeley.edu/) | Structural and seismic analysis | Add one narrow frame/truss/beam profile first; require independent reference solutions |
| CFD and thermal-fluid behavior | [OpenFOAM](https://www.openfoam.com/documentation/) | Flow, heat-transfer and other continuum cases | Preserve case directory, mesh, dictionaries, solver logs, residual histories and requested field outputs |
| Parametric CAD | [FreeCAD](https://www.freecad.org/) and [Open CASCADE](https://dev.opencascade.org/) | Editable geometry, STEP exchange, constraints and manufacturing exports | Start with file-based/CLI or Python automation behind the existing geometry kernel; defer direct kernel embedding until warranted |
| System-level design optimization | [OpenMDAO](https://openmdao.org/newdocs/versions/latest/) | Connect analysis disciplines and perform multidisciplinary optimization | Optional orchestration/optimization service after adapters emit typed, reproducible results; do not make Python a core dependency |
| Optics and electromagnetics | [Meep](https://meep.readthedocs.io/en/master/) | Scriptable finite-difference time-domain electromagnetic simulation | Recommended first photonics research adapter after generic evidence capsules and physical types are established |
| Acoustics and ultrasound | [k-Wave](https://www.k-wave.org/documentation.php) | Time-domain wave propagation, heterogeneous media and acoustic/ultrasound field modeling | Align with #5630 and existing acoustics/FIELD observation ownership; confirm runtime and license/deployment implications before packaging |
| Plasma and rarefied-flow research | [PICLas](https://piclas.readthedocs.io/en/latest/) | Particle-in-cell, DSMC and related particle/field models for specialist plasma and flow cases | Research-grade later adapter; computational cost, physical assumptions, species/chemistry data and validation must be explicit |
| Coupled model exchange | [FMI](https://fmi-standard.org/) | Standardized model exchange and co-simulation boundaries | Represent its semantics at the interchange layer where useful; no requirement to embed a particular modeling environment in core Rust |

Tool choice must remain domain- and version-specific. This is a shortlist for controlled evaluation, not an assertion that any tool supports every device or process. Review the exact version, license, solver models, supported platforms, redistribution terms, and external dependencies before incorporating a tool into a deliverable product.

### Why this is the right sequence

- **Mechanics and circuits first:** they connect directly to existing bridges and have small reference cases that can be checked against analytical solutions.
- **CFD/structural and CAD next:** these provide geometry-linked physical design, but they introduce larger meshes, boundary-condition assumptions, solver-specific convergence semantics, and harder-to-compare artifacts.
- **Optics and acoustics after evidence plumbing:** both have useful scriptable open-source tools, but credible predictions still depend on mesh/grid resolution, material properties, boundaries, calibration, and exact sensor provenance.
- **Plasma later:** this is not one uniform solver problem. Cold-plasma process equipment, thermal plasma, discharge physics, particle kinetics, gas flow, surfaces, chemistry, and radiation can need materially different models. Begin with literature-backed, low-risk measurement and simulation cases; do not generalize a plasma benchmark into a qualified device-design capability.

## Universal module contract

Create one shared, versioned module manifest for physical tools, instruments and end effectors. It should be a schema in the existing physical-type/observation/engineering ownership model—not a second source of truth.

Each module manifest should bind:

- **Identity:** module design ID, revision, vendor/build identity where applicable, firmware, serial/device identity and as-built configuration.
- **Mechanical interface:** mount datum, coordinate frame, envelope, allowable loads, fasteners and tolerance profile.
- **Electrical interface:** supply range, current/power limits, grounding/isolation class, connector identity and fault behavior.
- **Signals:** typed input/output channels, physical quantity, SI-compatible unit, frame, sample rate, timing semantics and whether a value is directly measured or derived.
- **Operating envelope:** permitted temperature, pressure, speed, force/torque, optical/acoustic exposure and other domain-specific limits.
- **Control semantics:** supported modes, rate/latency bounds, command expiry, cancellation behavior, state transitions and fault response.
- **Safety and authority:** hazards, containment requirements, interlock identity, safe state, permission prerequisites and prohibited operations.
- **Calibration and maintenance:** calibration record and epoch, uncertainty, drift checks, wear state, service interval and recalibration triggers.
- **Evidence links:** source requirements, analysis outputs, fabrication artifacts, inspection records, calibration certificates, experiment runs, model versions and qualification decisions.

The system should be able to discover that a tool is compatible with a task while separately determining whether that tool is currently calibrated, within its operating envelope, and authorized for the requested use.

## Evidence contract: do not collapse the lifecycle into one “pass”

Represent state and evidence kind separately. At minimum, distinguish:

1. **Proposal:** a candidate generated by reasoning or optimization; not yet an engineering result.
2. **Analytical prediction:** output of a stated analytical model with assumptions and validity envelope.
3. **Dry-run fixture:** deterministic test fixture used to exercise orchestration/parser code; never external-solver evidence.
4. **External process execution:** exact solver command/process completed and inputs/outputs were captured; completion alone says nothing about convergence.
5. **Parsed numerical result:** the declared parser extracted requested quantities and units from bound raw outputs.
6. **Numerically converged result:** solver-specific convergence criteria are satisfied and preserved in the evidence.
7. **Independently checked result:** a separate checker, analytical reference, benchmark, or accepted cross-solver comparison supports the stated claim.
8. **Calibrated physical observation:** measurement is bound to device identity, calibration identity/currentness, clock, physical units, frame, raw payload and uncertainty.
9. **Model calibration/validation:** a versioned model was compared against sufficiently independent physical data for a bounded domain of use.
10. **Design verification:** exact design revision satisfies its declared requirements under the accepted verification plan.
11. **Intended-use validation:** evidence supports that the design meets the actual use needs in its declared conditions.
12. **Physical qualification/release:** separate, explicit authority decision for a specific as-built article and operating envelope.

A higher state requires its own obligations; it is never assigned just because a preceding tool returned success. External approval or regulatory conformity, when required, remains external and must not be inferred from an internal report.

Every external-solver evidence capsule should bind at least: request/design/requirement IDs; exact solver and version; executable or package digest; execution environment/closure digest; rendered input bundle and digest; exact command/workflow manifest; exit status; stdout/stderr/raw output references and digests; parser identity/digest; requested metrics and units; convergence semantics; warnings; model maturity envelope; uncertainty; timestamps; and a reproducible rerun recipe. This expands on #4440 and #5633.

## First end-to-end pilot: single-joint laboratory

Use the existing #4814 proposal, with the frozen protocol and observation semantics from #4800. It exercises the hardest recurring issues while bounding cost and consequence better than a humanoid or a laser/plasma process rig.

The first pilot should produce:

- A frozen test-article design and as-built identity.
- Instrumentation and calibration records before a run is treated as valid.
- A declared excitation/measurement protocol and operating envelope.
- Synchronized raw observations of command/feedback timing, joint position/velocity, current, voltage, temperature, and load/torque when instrumentation supports it.
- Clear separation of directly measured channels from derived estimates.
- Repeated trials, measurement uncertainty, and explicit exclusions.
- A model fit/identification result with train/holdout separation where appropriate.
- Simulation-to-measurement residuals and drift/discrepancy reports.
- A new model revision and bounded requalification of only the claims supported by the affected evidence.

Core rules: `command sent != command latched != motion observed != torque delivered`; and `single-joint pass != full-arm qualification`. Human-proximity and powered testing must wait for a documented hazard assessment and independent, appropriately rated hardware safety path.

## Incremental execution plan

### Gate 0 — Close the evidence contract

1. Advance #4440 so external solver results carry reproducible executable/environment/input/output/parser provenance.
2. Advance #5633 as the first fully implemented real-solver parser. Use small deterministic circuit reference cases, including invalid input, missing metrics, non-convergence, warnings, and altered output fixtures.
3. Require separate booleans/statuses for process completion, parse completion, convergence and requested-metric availability. Do not use a single `converged` bit as a proxy for all four.
4. Keep dry-run fixtures testable on machines without external solvers, but make it impossible to admit them as real-solver qualification evidence.

### Gate 1 — Establish physical semantics and maturity

1. Integrate #6870's typed physical quantities into request inputs, outputs, constraints and coupling edges.
2. Integrate #6871's model/evidence maturity envelope so a frontier or uncalibrated model cannot satisfy a higher-grade engineering obligation.
3. Reject missing units, non-finite values, incompatible dimensions, unbound coordinate frames, stale calibration, and unsupported model regimes before solver execution.
4. Preserve domain ownership: raw observation/calibration semantics stay with the observation layer, optimization stays with the optimization owner, and final safety/release authority stays outside the proposing model.

### Gate 2 — One real numerical design loop

1. Harden the actual MuJoCo output path with a tiny morphology and golden trajectories; report only parsed outputs.
2. Implement a narrow OpenSees or equivalent structural reference case only after the generic result capsule and parser harness are stable.
3. Add geometry-bound request/result identity and reproducible fixtures before expanding to large meshes or coupled physics.
4. Benchmark against known reference values and include explicit failure cases.

### Gate 3 — One physical loop

1. Execute #4800's experiment protocol prerequisites, test-article identity and calibration semantics.
2. Build #4814's single-joint lab behind an independent bounded hardware control/interlock path.
3. Compare simulation predictions to calibrated measured observations; record sim-to-real discrepancy rather than retuning away inconvenient residuals.
4. Use results to refine the model and design in a new revision with change-impact analysis—not to mutate an already qualified model or controller silently.

### Gate 4 — Reusable physical modules and fabrication

1. Stabilize the shared module manifest and manufacturing artifact identity.
2. Connect the existing fabrication kernel to external parametric CAD and inspection/metrology records.
3. Establish repeatable tool-changing, module discovery, calibration checks, fault handling and safe-state transitions.
4. Only then broaden from a single joint to an interchangeable robotic workcell.

### Gate 5 — Domain expansion

Start with low-risk, measurement-centered projects:

- **Acoustics:** transducer characterization and calibrated acoustic measurements through #5630/FIELD ownership; use k-Wave only for problems its model supports.
- **Optics:** optical metrology or low-power bench experiments with defined eye-safety controls; use Meep for relevant electromagnetic field modeling.
- **Cold-plasma research:** begin with model review and externally supervised, contained test equipment only after plasma-specific hazard assessment and measurement protocols exist.
- **Hot plasma/high-energy equipment:** remain a later research tier requiring qualified laboratory infrastructure, specialist review, and domain-specific analysis and protection systems.

This sequence does not prohibit ambitious capabilities. It prevents an impressive simulation or generated CAD file from being presented as proof that a real apparatus works safely.

## Safety and release principles

- Treat machine hazards through an explicit risk-assessment and risk-reduction workflow. [ISO 12100:2010](https://www.iso.org/standard/51528.html) is the currently published machinery risk-assessment baseline at the time of this research, but ISO lists a successor draft under development; review applicable/current standards again for each release and jurisdiction.
- Lasers, high voltage, vacuum, plasma, ionizing radiation, hot surfaces, pressure systems, hazardous gases, and high-intensity ultrasound require distinct hazard analysis; there is no single generic “lab safe” flag.
- Use hardwired or independently enforced interlocks where the hazard assessment requires them. The reasoning process may request an action, but cannot override an interlock, extend its own authority, or self-certify safety.
- Bind authorization to the exact module, firmware, calibration epoch, procedure, environment and operating envelope. Expired/stale/unknown identity must fail closed.
- Test failure paths deliberately: lost heartbeat, stale sensor, invalid units, sensor disagreement, overtemperature, overcurrent, limit switch, parser ambiguity, interrupted process, stale configuration, corrupted evidence and power loss.
- Separate simulation, physical measurements, model qualification and release approval. A digitally signed report is not evidence that its underlying engineering claim is true unless its evidence and authority semantics are valid.

## Research references

Primary documentation reviewed for the tool shortlist:

- [MuJoCo model formats and model editing](https://mujoco.readthedocs.io/en/stable/modeling.html) — programmable multibody model boundary.
- [ROS 2 control hardware components](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/hardware_components_userdoc.html) — actuator/sensor/system interfaces and lifecycle.
- [ngspice Version 47 manual: batch analyses and raw output](https://ngspice.sourceforge.io/docs/ngspice-47-manual.pdf) — batch execution and rawfile/log support.
- [OpenFOAM residuals and convergence](https://doc.openfoam.com/2306/tools/processing/numerics/solvers/residuals/) — solver-specific residual semantics; do not reduce all convergence to process exit.
- [OpenFOAM user guide](https://www.openfoam.com/documentation/user-guide) — cases, meshes, boundary conditions, solvers, monitoring and post-processing.
- [OpenMDAO documentation](https://openmdao.org/newdocs/versions/latest/) — multidisciplinary optimization over coupled analysis components.
- [Meep documentation](https://meep.readthedocs.io/en/master/) — open-source scriptable FDTD electromagnetics.
- [k-Wave documentation](https://www.k-wave.org/documentation.php) and [k-Wave-II status](https://github.com/ucl-bug/k-wave-ii) — acoustics/ultrasound wave simulation; the successor project is under active development and should not be treated as API-stable.
- [PICLas documentation](https://piclas.readthedocs.io/en/latest/) — specialized PIC/DSMC and field/particle modeling. The project documentation notes that parts of its species/reaction database are still being verified, so data provenance and model-specific validation are particularly important.
- [ISO 12100:2010](https://www.iso.org/standard/51528.html) — machinery hazard assessment and risk reduction.

- [ngspice Version 47 User's Manual](https://ngspice.sourceforge.io/docs/ngspice-47-manual.pdf) — explicit rawfile and log output options for batch runs.
- [Meep licensing](https://meep.readthedocs.io/en/latest/License_and_Copyright/) — GPL-2-or-later; evaluate packaging/distribution consequences before bundling binaries or linking native code.
- [PICLas documentation](https://piclas.readthedocs.io/en/latest/) — particle-in-cell / DSMC plasma-flow methods, GPLv3, and high-order electromagnetic field solvers.
- [k-Wave-II project status](https://github.com/ucl-bug/k-wave-ii) — active rewrite/pre-release and requires MATLAB R2023b or newer; not an assumed portable, self-contained runtime.
- [ROS 2 control hardware components (Kilted)](https://control.ros.org/kilted/doc/ros2_control/hardware_interface/doc/hardware_components_userdoc.html) — useful interface/lifecycle model, but not a substitute for independent physical safety enforcement.
- [FMI 3.0.2 specification](https://fmi-standard.org/docs/3.0.2/) — Model Exchange and Co-Simulation interchange boundaries for later coupled-model integration.

## Exit criterion for this research phase

This memo is successful only when it drives a narrow, reproducible path through the existing issue graph. The first demonstrated claim should be modest and concrete:

> Symthaea can submit a frozen, typed engineering request to a pinned external solver, preserve the exact inputs and raw outputs, parse and classify its result without inventing missing data, compare it with an independent reference, and attach the result to a review whose acceptance rules remain separate from the solver.

After that is reproducible, the same evidence and module contracts can support robotics, CAD, optics, acoustics, and eventually plasma research. That is a credible path from engineering scaffolding to a self-improving physical engineering system.
