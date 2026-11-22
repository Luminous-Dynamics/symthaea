# AIRROAD-MIMO-CONTINUOUS-ENVELOPE-001 — exact continuous-band rational discrepancy certificate

**Status:** V0.1 synthetic exact-rational analytic reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11882  
**Composes with:** #11865 finite-knot identifiability, #11858 shared calibration, #11868 active excitation, #11801 robust MIMO stability, ROB-SYSID #4801 and ENG-MEAS #4919

## 1. Proof-carrying model contract

A fitted rational model must not enter a robust-stability calculation as a bare transfer function. Its evidence package must preserve model/source identity, measurement and calibration dependencies, fit/holdout provenance, continuous-domain discrepancy, proof coverage and unsupported gaps.

A dense grid is diagnostic, not a universal certificate. Finite noisy samples alone cannot identify arbitrary dynamics between knots. A physical continuous-frequency error bound requires additional assumptions such as a bounded model family/order/pole region, bounded disturbances, an independently justified residual model, or other validated physical constraints. Without these, the result remains NOT_ESTABLISHED_FOR_CONTINUOUS_DOMAIN.

## 2. Frozen rational model-pair error

Reuse the hidden cross-port term from #11865:

    Delta(s) = 1000*(s^2+1)*(s^2+4)*(s^2+16)/(s+5)^7

The 2x2 matrix error has exactly one nonzero element, row bus-a / column driven by bus-b. Consequently, its spectral norm equals |Delta(j*omega)|.

The complete certificate band is:

    omega in [5/2, 7/2] rad/s
    x=omega^2 in [25/4,49/4]

The exact squared magnitude is:

    f(x) = 10^6*(x-1)^2*(x-4)^2*(16-x)^2/(25+x)^7

No sampled-grid approximation is used in the proof.

## 3. Exact stationary-point derivation

Differentiate log(f) with respect to x. After putting the terms over the positive common denominator:

    D(x)=(x-1)*(x-4)*(16-x)*(25+x)

the derivative sign is the sign of:

    p(x)=x^3-213*x^2+2520*x-4648

On the entire band:
- p''(x)=6*x-426 is negative, including at the upper endpoint;
- therefore p'(x) is decreasing;
- p'(25/4)=-405/16, so p' is negative throughout;
- p(25/4)=193653/64 is positive;
- p(49/4)=-249795/64 is negative.

Thus p has exactly one root in the band, and f increases before it and decreases after it. The only possible minimum points are the two band endpoints.

The unique maximum root is isolated in:

    x_star in [1017/100,1018/100]

because p(1017/100)=1916213/1000000 is positive and p(1018/100)=-1640421/125000 is negative.

## 4. Exact continuous-band bounds

The smaller endpoint value is the exact minimum squared discrepancy:

    min f(x)=13908900096/30517578125 > 0

The maximum is bounded below by one exact point evaluation at omega=319/100:

    max f(x) >=
    1089363281328919378211341745610000000000 /
    666398192890950766041034480156534871921

The maximum is bounded above by evaluating the monotone numerator factors at their extremal ends over the isolating interval and using the smallest denominator:

    max f(x) <=
    10939549105801166400000000 /
    6655896779165488119029173

The upper/lower ratio of these squared H-infinity bounds is strictly less than 1005435/1000000, i.e. less than 1.005435. The continuous-band H-infinity norm enclosure is obtained by taking the square roots. All comparisons are exact rational comparisons.

The strictly positive minimum proves this particular pair of model matrices remains separated at every frequency in the band. It is not a physical measurement or a bound on an unknown aircraft plant.

## 5. Adversarial coarse-grid demonstration

The three-point grid omega={5/2,3,7/2} has a maximum squared sample of 612500000/410338673. The unsampled point omega=319/100 has a strictly larger value. Thus even the maximum of this small grid is below a known in-band value and cannot serve as a universal supremum certificate.

More generally, a finite scan can miss a narrow peak, zero, resonance or cross-port mode. The workflow must not infer continuous coverage merely from a large sample count.

## 6. Industry / SOTA baseline

Do not claim novelty for rational fitting, passivity enforcement, interval arithmetic, data-consistent uncertainty modeling, KYP or SOS.

Relevant current methods and references:
- SINTEF Vector Fitting and 2026 MF-NNLS perform rational multiport fitting; MF-NNLS applies passivity enforcement as a distinct post-fit procedure: https://www.sintef.no/en/software/vector-fitting/ and https://www.sintef.no/en/software/vector-fitting/downloads/mft-nnls/
- MATLAB supports frequency-response model estimation and validation: https://www.mathworks.com/help/ident/ref/idfrd.html and https://www.mathworks.com/help/ident/ref/n4sid.html
- Hindi, Seong and Boyd derive data-consistent MIMO frequency-domain uncertainty sets through LMI feasibility/SDP: https://web.stanford.edu/~boyd/papers/unc_fdd.html
- Frequency-domain MIMO identification with bounded unstructured model error and H-infinity uncertainty is described here: https://pure.tue.nl/ws/portalfiles/portal/3525408/388453.pdf
- NIST DLMF documents interval arithmetic as validated inclusion: https://dlmf.nist.gov/3.1.ii
- Bernstein-polynomial methods provide range bounds for complex polynomials and rational functions: https://doi.org/10.1016/j.cam.2020.113377

The defensible AIRROAD contribution is the proof-carrying evidence package, full-band proof coverage, uncertainty provenance and fail-closed composition with robust stability, not invention of those numerical techniques.

## 7. Next certificate ladder

1. Exact analytic stationary-point certificates (this reference).
2. Exact rational interval subdivision and interval-Horner enclosures.
3. Bernstein bounds for polynomial/rational range enclosures.
4. Derivative/Lipschitz remainder certificates under explicit assumptions.
5. SOS/KYP positive-real or matrix-inequality certificates where applicable.
6. Data-consistent nonparametric MIMO uncertainty from calibration/noise bounds.
7. Adaptive subdivision guided by enclosure width and robust stability boundaries, while retaining complete frequency coverage.
8. Compose admitted nominal-plus-discrepancy sets into #11801.

A gridded scan is only a diagnostic/refinement hint, never a certificate.

## 8. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with the frozen fixture.
- Zero universal bounds inferred from a finite grid alone.
- Zero uncovered frequency intervals in any successful certificate.
- Exact root isolation and sign validation in all adversarial fixtures.
- V0.1 squared H-infinity upper/lower ratio below 1.005435.
- Zero false positive positive-separation claims for any band containing a zero of Delta.
- Report conservatism, proof time, partition/root count, memory and unsupported-domain rate separately.
- Compare analytic extrema, interval subdivision, Bernstein, derivative/Lipschitz and SOS/KYP under identical rational functions and bands.
- No physical robust-stability PASS while measurement-derived model discrepancy is unbounded.

## 9. Claim ceiling

V0.1 certifies one exact rational model-pair discrepancy over one synthetic finite band. It does not prove the discrepancy of a real plant from finite noisy measurements, implement a general interval transfer-matrix engine, qualify aircraft converter models, establish physical passivity or create flight-safety/airworthiness/certification evidence.
