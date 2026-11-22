# AIRROAD-MIMO-MODEL-IDENTIFICATION-001 — exact finite-knot non-identifiability reference

**Status:** V0.1 synthetic exact-rational analytical counterexample  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11865  
**Composes with:** #11764 impedance passport, #11788 complex enclosures, #11858 shared calibration, #11801 robust MIMO stability, #4801 ROB-SYSID, #4919 ENG-MEAS, #4918 common-mode evidence

## 1. Objective

Freeze the smallest useful counterexample proving that finite MIMO impedance knots do not uniquely identify a continuous-frequency matrix model. The work owns only AIRROAD's domain-specific profile and benchmark; the generic system-identification, measurement-equation, calibration and evidence-independence owners remain canonical.

## 2. Exact hidden-crossport model pair

The measured angular-frequency set is:

    omega = {1, 2, 4} rad/s

The first model is:

    Z_A(s) = [ 2  0 ]
             [ 0  3 ]

The second model is:

    Delta(s) = 1000*(s^2+1)*(s^2+4)*(s^2+16)/(s+5)^7

    Z_B(s) = [ 2       Delta(s) ]
             [ 0          3     ]

Delta is proper (numerator degree 6, denominator degree 7), real-rational and has only a repeated pole at s=-5. Both models are stable as rational transfer matrices in the sense that neither has a right-half-plane pole. No passivity claim is made.

At each measured knot s=j*omega_k with omega_k in {1,2,4}, one numerator factor is zero exactly, hence:

    Z_A(j*omega_k) = Z_B(j*omega_k)

Their full 2x2 matrices, not merely diagonal SISO projections, are identical at every training knot.

At the held-out frequency omega=3:

    (1-9)*(4-9)*(16-9) = 280

so Delta(j*3) is nonzero. The exact rational value is frozen in the fixture:

    Re(Delta(j*3)) = -51419200000 / 52523350144
    Im(Delta(j*3)) =  38391360000 / 52523350144

This pair fits all three training frequency knots with zero error but disagrees at an off-grid cross-port observation. It is a direct counterexample to promoting a finite-knot fit into a continuous-frequency model claim without additional assumptions/evidence.

## 3. What an honest model-identification result must separate

Keep these products distinct:
1. training fit residual;
2. held-out frequency/sweep/operating-point residual;
3. model structure/order ambiguity;
4. measurement uncertainty (including shared calibration factors);
5. model discrepancy/residual envelope;
6. continuous-frequency validity proof or explicit gap.

Dense frequency sampling is a useful diagnostic and refinement trigger, but is not automatically a proof between points. If the model order, residual envelope or continuous-frequency remainder is not bounded, the system must return NOT_ESTABLISHED_FOR_CONTINUOUS_DOMAIN.

## 4. Rational realization and physical properties

A common-pole MIMO rational form such as H(s)=D+sE+sum_k R_k/(s-p_k) is an established candidate representation, not AIRROAD novelty.

Stable poles do not imply passivity. Passivity should only be enforced when the physical element is known to be passive under the declared port/reference conventions. Converter/source models can be active and may exhibit negative incremental impedance; forcing those fits to passivity can erase real dynamics.

Any passivity correction changes the fitted model. The correction size, resulting residual, and revalidation must be reported. Finite-sample passivity checks do not establish continuous-frequency passivity.

## 5. Industry / SOTA baseline

- SINTEF Vector Fitting provides common-pole MIMO rational fitting and state-space conversion, with passivity enforcement as a separate operation: https://www.sintef.no/en/software/vector-fitting/
- SINTEF's passivity page states the positive-real / positive-semidefinite Hermitian-part criterion and provides residue-perturbation methods: https://www.sintef.no/en/software/vector-fitting/passivity/
- 2026 MFT-NNLS paper reports passive macromodeling of large MIMO immittance and scattering data, with stable poles, passivity and reciprocity for passive targets: DOI 10.1109/TPWRD.2026.3669657; https://doi.org/10.1109/TPWRD.2026.3669657
- MATLAB R2026b idfrd represents frequency-response data/models and supports frequency-domain estimation/validation workflows: https://www.mathworks.com/help/ident/ref/idfrd.html
- MATLAB R2026b n4sid accepts continuous-time frequency-response data for continuous-time identification: https://www.mathworks.com/help/ident/ref/n4sid.html
- MATLAB system-identification workflow separates preprocessing, model structures and validation: https://www.mathworks.com/help/ident/gs/system-identification-workflow.html

Do not claim novelty for vector fitting, common-pole MIMO fitting, passivity enforcement, rational-to-state-space conversion, or frequency-response system identification. The defensible AIRROAD opportunity is the evidence-qualified transition from the exact MIMO measurement passport and joint uncertainty semantics to a held-out, discrepancy-bounded, continuous-frequency model artifact whose validity can be consumed by #11801.

## 6. Required qualification ladder

1. Exact rational reproduction of the hidden-crossport pair.
2. Held-out knots that distinguish the two models.
3. Model-order and pole/residue provenance.
4. Joint calibration and per-record uncertainty dependency preservation.
5. Held-out operating-point/sweep validation with independent-evidence assessment.
6. Residual envelope with explicit domain coverage.
7. Continuous-frequency discrepancy bound, or NOT_ESTABLISHED.
8. Only then feed the admitted uncertain rational model set to robust stability.

## 7. Benchmark targets (targets, not results)

- 100% exact agreement with every frozen matrix value.
- Zero promotion from knot-only fit to continuous-domain validity.
- Zero false-negative detection of the hidden off-grid cross-port term in the fixture.
- 100% model poles/residues bind source sweep, preprocessing and port identities.
- 100% held-out data are provenance-disjoint from fit/model-order/weight selection under the declared independence profile.
- Report complex matrix residual per element/frequency, max norm, weighted norm, singular-value residual, pole error, model order, passivity status, continuous-domain proof coverage and runtime separately.
- Compare Vector Fitting, MF-NNLS, MATLAB frequency-domain estimation and PLECS/ngspice response on identical model/data/uncertainty/holdout conditions.
- Zero robust-stability PASS when the model discrepancy is unbounded over any required frequency subdomain.

## 8. Claim ceiling

V0.1 is an exact analytic non-identifiability counterexample. It does not implement a vector fitter, qualify a measured aircraft impedance, establish a continuous-frequency bound for physical data, establish passivity of converter models, validate an aircraft electrical network, or create flight-safety, airworthiness or certification evidence.
