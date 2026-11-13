# AIRROAD-MIMO-ROBUST-001 — exact three-port hidden-mode and uncertainty-boundary benchmark

**Status:** V0.1 synthetic exact-rational mathematical reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11801  
**Composes with:** #11761 (two-port hidden dynamics), #11764 (MIMO impedance passport), #11788 (complex phasor uncertainty enclosures)

## 1. Objective

Extend the asymmetric DC-bus and two-port interaction work into a three-port reference that tests whether all cross-port paths survive the stability decision. Add a small exact-rational uncertainty family whose robust classification can be verified without a frequency grid or numerical root finder.

This is a frozen mathematical benchmark and research contract. It does not implement a general generalized-Nyquist method and does not describe a physical converter.

## 2. Exact model

Let the loop transfer matrix be:

    L(s, gamma) = M(gamma) / (s + 1)

    M(gamma) = [  0       2       0     ]
               [  0       0       2     ]
               [ -2gamma  0       0     ]

The port order is [bus-a, bus-b, bus-c]. The feedback convention is negative unity feedback and the matrix convention is row = observed output, column = driven input.

All diagonal entries are identically zero:

    L_aa(s) = L_bb(s) = L_cc(s) = 0

There is nevertheless a complete directed three-port coupling cycle. The network is intentionally asymmetric and nonreciprocal.

## 3. Characteristic derivation

The state-space realization is:

    A = -I_3
    B =  I_3
    C =  M(gamma)
    D =  0

so L(s,gamma) = C (sI - A)^(-1) B. Under negative unity feedback, A_cl = A - B C = -I_3 - M(gamma).

Equivalently:

    det(I + L(s,gamma))
      = det((s + 1) I_3 + M(gamma)) / (s + 1)^3
      = ((s + 1)^3 - 8 gamma) / (s + 1)^3

The open-loop realization has three poles at -1. The closed-loop characteristic polynomial is:

    p(s,gamma) = (s + 1)^3 - 8 gamma
               = s^3 + 3s^2 + 3s + (1 - 8 gamma)

For gamma > 0, let c = (8 gamma)^(1/3). The closed-loop poles are:

    s_0 = c - 1
    s_1,2 = -1 - c/2 +/- j (sqrt(3) c/2)

The complex pair always has negative real part. The real pole changes sign exactly when c = 1, i.e. gamma = 1/8. Equivalently, the cubic Routh-Hurwitz conditions reduce to gamma < 1/8 for this declared positive parameter domain.

Therefore:

- 0 < gamma < 1/8: all three closed-loop poles are strictly in the left half-plane;
- gamma = 1/8: one closed-loop pole is exactly zero;
- gamma > 1/8: exactly one closed-loop pole is in the right half-plane.

This is the key counterexample: all diagonal SISO loop projections remain identically zero for every gamma, while the complete three-port network changes from stable to unstable.

## 4. Frozen uncertainty sets

All interval endpoints are exact rational numbers.

| Case | gamma set | Expected disposition | Reason |
|---|---:|---|---|
| Stable | [1/32, 1/16] | ROBUSTLY_STABLE | Entire set lies strictly below 1/8 |
| Boundary | [1/16, 1/8] | NOT_ESTABLISHED | Contains a marginal realization at 1/8 |
| Mixed | [1/16, 1/4] | NOT_ESTABLISHED | Contains stable, marginal and unstable realizations |
| Unstable | [3/16, 1/4] | ROBUSTLY_UNSTABLE | Every realization lies strictly above 1/8 |

For this fixture, the exact set-level decision rule is:

- robustly stable iff gamma_max < 1/8;
- robustly unstable iff gamma_min > 1/8;
- otherwise not established.

A set touching the boundary is not robustly stable. A set containing both stable and unstable realizations is not labelled robustly unstable; it is not established.


## 5A. Existential destabilization witness versus universal instability

Do not collapse two distinct propositions:

- **DESTABILIZATION_WITNESS:** there exists at least one parameter realization inside the admitted uncertainty set that is unstable.
- **ROBUSTLY_UNSTABLE:** every parameter realization in the admitted uncertainty set is unstable.

For the mixed set [1/16, 1/4], gamma=1/4 is a concrete destabilizing witness, but the set is not robustly unstable because it also contains stable realizations. Its universal disposition remains NOT_ESTABLISHED. At gamma=1/8 the system is marginal, not strictly unstable.

This distinction matters for industry comparisons. A robust-stability margin upper bound or worst-case destabilizing perturbation is generally an existential witness, not proof that the entire uncertainty set is unstable. Record the witness, its exact admissibility and its closed-loop pole evidence separately from a universal set-level classification.

## 5. Independent exact-rational oracle

The verifier at tools/airroad-mimo-robust-qualifier/verify_three_port_reference.py:

- uses only Python standard-library Fraction arithmetic;
- checks the frozen three-port matrix and zero diagonal;
- derives the cubic coefficients exactly from each gamma endpoint;
- checks the open-loop pole declaration and closed-loop pole formula;
- reproduces the four set dispositions using exact rational comparisons;
- refuses malformed, non-positive or reversed parameter intervals;
- never samples a frequency grid or estimates roots numerically.

The machine-readable corpus is docs/research/fixtures/airroad-mimo-three-port-robust-v0.1.json.


## 5B. Current industry benchmark contracts (R2026b / current documentation)

The primary commercial robust-stability baseline is MATLAB Robust Control Toolbox `robstab` for uncertain state-space models (`uss` / `genss`):

- `LowerBound` is a guaranteed lower bound on the normalized robust-stability margin. For uncertainty up to that level, the model is guaranteed stable.
- `UpperBound` is a guaranteed upper bound on that margin and is associated with an existential destabilizing perturbation returned as `wcu`. Substituting `wcu` and checking poles is the documented confirmation step.
- The `uss` / `genss` state-space algorithm searches across frequency and does not rely on frequency gridding.
- Passing an explicit frequency vector, or using `ufrd` / `genfrd`, evaluates a frequency grid and gives no guarantee between the supplied points.

Primary reference: https://www.mathworks.com/help/robust/ref/dynamicsystem.robstab.html

This requires a sharper AIRROAD output contract:

| Result | Meaning | Must not be confused with |
|---|---|---|
| `ROBUSTLY_STABLE` | Every admissible realization in the declared set is internally stable | Nominal stability |
| `ROBUST_STABILITY_MARGIN_LOWER_BOUND` | Guaranteed stable uncertainty scaling up to a normalized margin | A witness |
| `DESTABILIZATION_WITNESS` | One admissible realization with a verified unstable/marginal closed-loop mode | Universal instability of the entire set |
| `ROBUST_STABILITY_MARGIN_UPPER_BOUND` | A destabilizing perturbation exists at or below the bound | Proof all points in the set are unstable |
| `ROBUSTLY_UNSTABLE` | Every admitted realization is proven unstable | An upper robust-stability-margin bound |
| `NOT_ESTABLISHED` | Neither universal proposition has been proven | PASS |
| `NOT_EVALUABLE` | Inputs, evidence or model applicability are insufficient | Stable-by-default |

MathWorks documents that a robust-stability margin above 1 means stable for all values of the modeled uncertainty, while below 1 means there exist values that destabilize. That is not the same proposition as "every member is unstable." AIRROAD must compare these results separately and preserve the witness identity and exact pole evidence.

### Additional baselines

- **IQC Toolbox / IQClab** provides worst-case stability/performance analysis for uncertainty and nonlinearities using integral quadratic constraints. It is an important complementary baseline for dynamic/nonlinear uncertainty, not simply another frequency-sweep solver: https://iqctoolbox.github.io/iqcToolbox/ and https://www.ll.mit.edu/partner-us/available-technologies/integral-quadratic-constraints-iqc-toolbox
- **RobustAndOptimalControl.jl** is a useful reproducible open-source control baseline. Current docs describe structured singular-value support for diagonal complex perturbations only; its documented `robstab` method uses a frequency vector by default: https://juliacontrol.github.io/RobustAndOptimalControl.jl/stable/ and https://juliacontrol.github.io/RobustAndOptimalControl.jl/stable/uncertainty/
- **PLECS 5.0** has AC analysis, frequency sweeps and frequency response at discrete perturbation frequencies. Its loop-gain analysis block is scalar-signal-only; AC analysis can linearize state-space/MNA equations at an operating point. It is a circuit-analysis/frequency-response baseline, not directly equivalent to structured robust-stability analysis: https://docs.plexim.com/plecs/5.0/analysis/frequency-response/ and https://docs.plexim.com/plecs/latest/components-by-category/loopgainanalysis/

### Matched benchmark protocol

For every candidate, pin:
- identical nominal model, state-space realization, source/load partition and feedback signs;
- identical uncertainty parameterization, shared-vs-independent dependencies and admissible set;
- identical nominal stability and internal-mode/cancellation semantics;
- identical time units, frequency domain, and uncertainty scaling convention;
- solver/tool version and all options.

Report:
- exact stability margin when analytically known;
- MATLAB lower and upper margin bounds, critical frequency, `wcu`, and substituted closed-loop poles;
- IQC certificate status, multiplier class/order, solver and conservatism where applicable;
- Julia `mu` lower/upper capabilities and perturbation structure;
- AIRROAD enclosure width, subdivision/work budget, proof coverage, witness validation and abstention;
- false robust-pass, false witness, false abstention, runtime and memory.

AIRROAD's defensible differentiation remains the evidence and physical applicability bridge: calibrated per-port provenance, joint parameter semantics across frequency, internal-mode preservation, replayable margins/witnesses, configuration identity and composition with fault/supply/mission evidence. It is not a claim that MATLAB lacks continuous-frequency state-space robust analysis.

## 6. What this benchmark establishes

It establishes an independently replayable analytic example where diagonal SISO inspection loses all relevant coupling information, and it creates exact stable / boundary / mixed / unstable parameter families for future robust-analysis implementations.

It does not yet establish:

- a generalized-Nyquist winding calculation;
- validated continuous-frequency determinant enclosures;
- a general structured-uncertainty solver;
- robustness for measured impedance matrices;
- correctness for delays, harmonic state-space models or arbitrary descriptor systems;
- physical converter, aircraft, HIL/PHIL or mission capability.

The next algorithmic stage must preserve a single physical uncertainty realization across the whole frequency contour. Independent interval boxes at each frequency are not a substitute for that joint model.

## 7. Benchmark targets

Targets, not achieved results:

- 100% exact agreement with the four rational parameter-set dispositions;
- zero false ROBUSTLY_STABLE outcomes when a set touches or crosses gamma = 1/8;
- zero false ROBUSTLY_UNSTABLE outcomes when the set includes stable members;
- exact retention of the three open-loop poles at -1;
- zero diagonal-only SISO PASS for the coupled three-port fixture;
- deterministic replay under port-order permutation;
- no frequency-grid-only robust decision.

## 8. Next research step

Implement the narrowest useful continuous-frequency robust-stability method only after freezing this corpus: structured rational state-space inputs, shared parameter domains, validated contour subdivision, explicit argument-change orientation, independent open-loop RHP pole accounting, and a direct closed-loop state-space reference on exact fixtures.

Any inability to exclude zero from a determinant enclosure, establish pole counts, or close the uncertainty proof must produce NOT_ESTABLISHED or NOT_EVALUABLE, never a PASS.

## 9. Claim ceiling

V0.1 is an exact synthetic mathematical reference. It is not a physical converter model, aircraft electrical qualification, flight-safety analysis, airworthiness evidence or certification. No energized fault-injection or hardware test is authorized.
