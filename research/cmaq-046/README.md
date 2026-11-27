# CMAQ-046 — Convex-target approachability e-process

## Why this is a paradigm shift

Predictive-rank monitoring is valuable when the null can be expressed as exchangeability against a frozen calibration reference. But many operational questions are naturally multivariate and adaptive:

- Is sensor residual bias inside an allowed engineering envelope?
- Do tracking, localization and actuation errors remain inside a joint tolerance set?
- Are energy, thermal, vibration and control residual means simultaneously within acceptable ranges?
- Is the conditional mean of a vector of safety indicators inside an admissible region, even when the environment changes over time?

Reducing these to one scalar before monitoring can hide combinations of failure modes. CMAQ-046 instead treats the null as a **predeclared convex set of acceptable conditional means** and learns a separating direction online.

This changes the null from fixed-reference exchangeability to a conditional-mean restriction. It does not replace PRM; it addresses a different statistical contract.

## Null and evidence process

Let r_t ∈ R^d be a bounded residual vector and S a frozen compact convex target. The null is

    H0: E[r_t | F_(t-1)] ∈ S   for every t.

For a box target S = [lower, upper], the support function is

    h_S(w) = sup_(s∈S) <w,s>
           = sum_i (w_i >= 0 ? w_i*upper_i : w_i*lower_i).

An online convex optimization (OCO) learner selects a predictable unit-ball normal w_t using only the history F_(t-1). The support residual is

    q_t = <w_t, r_t> - h_S(w_t).

Since w_t is predictable and the conditional mean belongs to S,

    E[q_t | F_(t-1)] <= 0.

For observations with ||r_t||_2 <= R and target radius S_max = max_(s∈S)||s||_2, use B = R + S_max, which guarantees |q_t| <= B for ||w_t||_2 <= 1. With fixed lambda=1/2,

    K_t = K_(t-1) * (1 + lambda*q_t/B),   K_0 = 1.

Every factor is at least 1/2 and has conditional expectation at most one. Thus K_t is a nonnegative test supermartingale; Ville's inequality gives

    P_H0(sup_t K_t >= 1/alpha) <= alpha.

This is an **anytime conditional-mean guarantee**. It does not rely on exchangeability or stationarity, but it does require the conditional mean to remain inside the frozen set at each time and a valid norm bound on every observation.

## Predictable OCO normal

CMAQ-046 uses projected online gradient descent on

    f_t(w) = h_S(w) - <w,r_t>

over the unit L2 ball. At time t, w_t is already fixed. Only after r_t arrives does the learner compute a support-function subgradient s_t ∈ ∂h_S(w_t) and update

    w_(t+1) = Proj_unit_ball(w_t + eta_t*(r_t-s_t)),
    eta_t = 1/(B*sqrt(t)).

At a zero coordinate, the implementation selects a deterministic valid subgradient by clamping zero into the corresponding box interval. The OGD update uses the pre-observation normal; it cannot select its own current bet after seeing the outcome.

The implementation uses a fixed bet, not the ONS bet used in the paper's optimized betting-regret result. The OCO learner has the standard O(B*sqrt(T)) regret scale on the unit ball. This crate does not claim to reproduce every finite-time rate or theorem in Zhao (2026).

## Pathwise residual-regret certificate

For the empirical mean r_bar_T and any compact convex target S, the online convex geometry identity is

    dist(r_bar_T, S)
      = (1/T)*sum_t q_t + Regret_T/T.

The monitor reports:
- empirical mean residual vector;
- Euclidean distance from that mean to the box;
- cumulative support residual;
- realized OCO regret T*distance - sum_t q_t;
- the predictable normal used and next normal;
- log e-value and Ville threshold.

This identity is a pathwise optimization diagnostic. It is distinct from the conditional-mean null guarantee. A positive empirical distance can arise temporarily under a valid stochastic null; it is not itself a valid stopping rule. The e-process, not the geometric distance, controls anytime rejection.

## Implementation and failure behavior

The dependency-free Rust crate provides:
- finite, dimension-checked BoxTarget with lower/upper vectors;
- support function, a valid deterministic support subgradient and Euclidean distance;
- bounded observation vectors in the L2 norm;
- projected OGD in the unit ball;
- log-domain fixed betting process and Ville threshold;
- distance/residual/regret certificate;
- atomic validation and checked observation counter;
- permanent quarantine after threshold crossing.

Invalid dimension, NaN/Infinity, an observation outside the declared norm bound, non-finite arithmetic, or counter overflow returns an error before any state mutation.

## Tests

The deterministic tests cover:
- box support and distance;
- support subgradient validity including zero coordinates;
- nonpositive expected support residual for conditional means inside the box;
- predictable-normal ordering;
- pathwise distance/residual/regret identity;
- residual and betting-factor bounds;
- atomic rejection of invalid vectors;
- evidence accumulation under a separated mean;
- quarantine permanence;
- counter overflow and invalid target/alpha settings.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-046/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-046/Cargo.toml --locked --all-targets -- -D warnings

## Paired unknown-direction probe

Run from the repository root:

    cargo +1.100.0 run --release --manifest-path research/cmaq-046/Cargo.toml --locked --bin approachability_probe -- 1000 2000 20261127 1000 > cmaq-046-results.csv

The executable compares:
- adaptive OCO approachability e-process;
- fixed equal mixture over eight compass-direction e-processes.

Both receive identical two-dimensional observations, target box, norm bound, alpha and residual normalization. Under the null, each coordinate receives independent symmetric noise with zero conditional mean inside the target box. Under the alternative, after the frozen pre-change prefix, the mean shifts in one of eight orientations unknown to the detector; the same bounded noise remains.

The probe reports full-stream null alarm counts, pre-change false alarms, post-change detections, misses and median delay from true onset. It is a narrow synthetic test of direction adaptation, not a general result or comparison against PRM/CCTM.

The CI run is deliberately small and is a wiring smoke check only. Full statistical qualification needs at least 10,000 independent null streams per primary cell and 5,000 alternatives per orientation, with confidence intervals and CMAQ-044 paired effective-block inference.

## Required comparisons

Compare the adaptive OCO direction against:
1. a single fixed direction;
2. a pre-registered equal mixture over compass directions;
3. coordinate-wise scalar e-processes;
4. Hotelling / multivariate sequential baselines under matched conditional-mean assumptions;
5. CMAQ-031/033/038/045 only after mapping their inputs to the exact same bounded residual stream.

Use paired bounded vector streams with unknown violation orientation, conditional-mean nulls inside the box, marginally stable variance changes, persistent mean departures, gradual shifts, and cross-coordinate cancellation. Report full-horizon null false-alarm probability with confidence intervals, miss probability, delay from true onset (with miss penalties), worst orientation/family, update p99, memory and target-dimension scaling.

Do not compare this conditional-mean result directly with PRM/CCTM as though their null assumptions were identical. If a variance/tail shift leaves the conditional mean in S, this monitor may not reject; a separate scale/tail e-process is required.

## Literature

Jinze Zhao (10 Aug 2026), *From Approachability Residuals to Anytime-Valid Evidence: The Online Convex Geometry of Testing by Betting*, arXiv:2608.09450. The paper connects Blackwell approachability / OCO residuals to anytime-valid e-processes under adaptive conditional-mean constraints. CMAQ-046 implements a bounded box-target, projected-OGD, fixed-bet subset of this framework; it is not a reproduction of all optimized betting-regret results or the paper's full controlled Blackwell game.

## Claim boundary

The target set, residual definitions, source sensors and norm bound must be justified independently and frozen before deployment. This software does not authenticate sensor receipts, prove the target represents physical safety, establish completeness of residual channels, or certify the system. Variance-only and tail-only shifts can be invisible when conditional means remain in the target. The OCO certificate and the e-process guarantee are separate claims and must be audited separately.
