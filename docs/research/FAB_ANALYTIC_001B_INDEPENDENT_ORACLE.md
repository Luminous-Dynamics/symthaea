# FAB-ANALYTIC-001B — independent beam-equation oracle

**Status:** Initial executable oracle tranche; code/test execution pending.  
**Parent:** [FAB-ANALYTIC-001 #4933](https://github.com/Luminous-Dynamics/symthaea/issues/4933)  
**Current reference source:** `crates/domains/symthaea-fabrication-kernel/src/analytical.rs`  
**Claim ceiling:** Formula/code checks for the bounded idealized model only. Not physical validation, FEA equivalence, structural safety, code compliance, or fabrication qualification.

## Paradigm shift: qualify the equations, not the story

The engineering kernel's simple-beam equations are useful only if they are mathematically implemented as declared. In-source tests that calculate expected values through the same production helpers risk repeating the same implementation bug. This oracle instead uses hand-derived reference constants and invariants that do not call production functions to calculate expected results.

The work composes with #4933's 001B tranche and does not duplicate:
- #6081's source repair for the dimensionally incorrect generic reaction-force implementation;
- #6084 / #6082's source-bound compile/regression qualifier;
- #6078's later three-way structural-kernel differential/consolidation campaign;
- #3697's lifecycle verification, validation and credibility ownership.

## Executable corpus

The new integration file is:

`crates/domains/symthaea-fabrication-kernel/tests/analytical_known_answer_oracle.rs`

It has 11 test functions spanning the following frozen hand-derived cases:

| Family | Independent reference or invariant |
|---|---|
| Rectangular section | `A = b h`, `I = b h³/12`, `Z = I/(h/2)` for `b=0.02 m, h=0.04 m` |
| Solid circular section | `A = πr²`, `I = πr⁴/4`, `Z=I/r` for `r=0.01 m` |
| Hollow circular section | `A = π(ro²-ri²)`, `I=π(ro⁴-ri⁴)/4` for `ro=0.02 m, ri=0.01 m` |
| Symmetric I-beam | Hand-derived area, centroidal second moment and elastic section modulus for `bf=0.1 m, tf=0.01 m, hw=0.08 m, tw=0.01 m` |
| Axial loading | `σ=F/A=1.25 MPa` for `F=1000 N` on the reference rectangle |
| Center-point beam loading | `Mmax=PL/4`, `σmax=M/Z=33.75 MPa`, `δmax=PL³/(48EI)=1.0125 mm` for `P=600 N, L=1.2 m, E=200 GPa` |
| Combined scalar stress | Hand-derived 1.25 MPa axial stress, 33.75 MPa bending stress, 0.375 MPa simplified average-shear proxy, and independently combined scalar result |
| Elastic-modulus metamorphic check | For fixed geometry/load in this linear model, doubling `E` leaves elastic stress invariant and halves deflection |
| Load/span scaling | Doubling point load doubles stress/deflection; doubling span doubles the center-load bending stress and multiplies deflection by eight |
| Zero load | Zero computed stresses/deflection and explicit infinite yield/stress ratio semantics; that ratio is not a safety qualification |
| Adversarial geometry | Checked construction rejects zero-width rectangles and hollow-circle inner radius equal to outer radius |

The oracle uses stringent relative tolerance plus a small absolute floor, reports non-finite actual values as failures, and keeps all expected numeric constants within the test file. The test target is separate from production unit-test code, and production code never imports or invokes the oracle.

## Promotion gates and proposed benchmarks

These are acceptance targets, not current achieved results.

### Tier A — frozen closed-form oracle

- All hand-authored known-answer cases pass on the exact commit.
- Zero false accepts for invalid geometry and unsupported numeric states in the declared corpus.
- Oracle expectations remain independently specified and reviewed.
- Any mutation that changes the equations (for example, rectangle inertia exponent, center-load moment factor or deflection denominator) must fail at least one protected case.
- Record test command, toolchain, exact commit SHA, package revision and result exit status.

### Tier B — semantic and dimensional controls

Add distinct tests or typed results for:
- force versus energy dimensionality at the generic `PhysicsBackend` boundary;
- load-position and support-condition identity;
- the distinction between visualization proxy and physical displacement/strain;
- material preset identity versus exact as-built material evidence;
- model assumptions and explicit out-of-profile rejection.

Do not fold these into a generic `safe=true` label.

### Tier C — independent differential review

Freeze a shared input corpus and compare:
1. independent hand/calculated oracle;
2. `symthaea-structural` implementation;
3. fabrication-kernel analytical implementation.

A three-way mismatch should be retained and diagnosed rather than decided by majority vote. Do not consolidate overlapping equations until the exact three implementations have independently been compared at frozen inputs and units. This is owned by #6078.

### Tier D — numerical-solver verification

For numerical FEM/CFD solvers, supplement simple closed-form tests with manufactured solutions, grid/time-step refinement, conservation, boundary-condition tests, convergence order and fail-to-converge controls. A known-answer beam test is not a substitute for numerical solver verification.

### Tier E — physical validation

Only a declared physical simple-beam campaign with calibrated metrology, fixture properties, measurement uncertainty, blinded predictions and held-out conditions can support physical-validation claims. Measurements do not automatically validate extrapolation to other geometries, materials, load cases or failure modes.

## Risk-weighted fidelity escalation

Fidelity should scale with both model influence and the consequence of the engineering decision. A cheap analytical profile can screen candidate concepts, but uncertainty near a hard constraint should trigger more expensive numerical analysis or measurement. If the probability model is not defensible, retain interval bounds and worst-case regret rather than inventing expected values.

A future Symthaea evaluator should emit separate outputs for:
- numerical/code correctness;
- operating-envelope applicability;
- external-reference agreement;
- uncertainty and discrepancy;
- constraint robustness and decision consequence;
- evidence completeness and currentness;
- permitted claim ceiling.

No aggregate score may erase a failed critical control.

## Verification status

The integration test file is committed on a dedicated public-repository branch and fetched back from GitHub. The available shell has Git but no `cargo`, `rustc`, or `rustfmt`, so this tranche has **not** been compiled or executed in this environment. A queued workflow is not a pass. The PR must remain draft until the frozen exact-head command has actually run or a source-bound hosted qualifier reports the full result.

Suggested command:

```bash
cargo test --locked -p symthaea-fabrication-kernel --test analytical_known_answer_oracle
```

## Research references

- NASA, [NASA-STD-7009B: Standard for Models and Simulations](https://standards.nasa.gov/standard/NASA/NASA-STD-7009), active public standard. Supports program/project-defined acceptance criteria and lifecycle model-credibility assessment.
- NASA, [NASA-HDBK-7009B: Implementation Guide](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009), issued 2026-02-03. Guidance for practical modeling and simulation credibility processes.
- Srinivasan, V. (2026), NIST, [On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices](https://doi.org/10.6028/NIST.AMS.100-85), NIST Advanced Manufacturing Series 100-85, published 2026-10-05. Presents a risk-based credibility framing that scales VVUQ effort with model influence and decision consequence.
- Roache, P. J. (2002), [Code Verification by the Method of Manufactured Solutions](https://doi.org/10.1115/1.1436090), *Journal of Fluids Engineering*. Establishes manufactured solutions as a general means of code verification, with systematic refinement exposing order-of-accuracy errors.
- Salari, K. and Knupp, P. (2000), Sandia, [Code Verification by the Method of Manufactured Solutions](https://www.sandia.gov/research/publications/details/code-verification-by-the-method-of-manufactured-solutions-2000-06-01/), SAND2000-1444. Describes how manufactured-solution procedures can expose coding mistakes that affect numerical order accuracy.

These references justify the benchmark methodology; they do not certify this code or any physical structure.
