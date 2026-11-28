# CMAQ-060 — Compute-sparse predictable expert selection

Status: research design / theorem-obligation draft. No implementation or performance result is claimed.

## 1. Motivation

CMAQ-057's full-information adaptive policy emits a factor for every registered Fourier expert per observation. This enables C058's kernel × restart portfolio, but feature transformation work is O(KD) per pair for K candidate experts and product-feature dimension D.

CMAQ-060 investigates a different architecture: choose one expert before seeing the current response pair, calculate only that expert's factor, and update one global e-process. If the selected factor is itself conditionally mean-one, predictable selection among experts does not require a per-expert alpha split.

The desired compute point is O(D) selected feature work plus O(K) selector state. It is not guaranteed to be more powerful than the full-information K-expert portfolio.

## 2. Information filtration and timing

At observation t, the permitted order is:

1. Receive the current covariate Z_t, which is guaranteed available before X_t and Y_t.
2. Use only F_(t-1), Z_t, and optional fresh selection randomness independent of the current response pair to choose J_t.
3. Read the selected expert's past-only direction u_(J_t,t).
4. Evaluate only that expert's response and covariate maps on the current pair.
5. Form its factor and append it to the one global evidence process.
6. Update the selected expert's history and the selector using the observed factor.
7. Commit sequence, pair identity, policy state, and evidence atomically.

No decision in steps 1–3 may depend on X_t or Y_t. The selection RNG and policy version must be frozen and recorded. A feature map, target or norm bound must be part of the content-addressed inventory before use.

## 3. Conditional validity sketch

For each fixed expert k, assume:
- its response/covariate feature maps are content-addressed and fixed before monitoring;
- its predictable direction u_(k,t) is measurable from the full past and selected-expert history before the current pair;
- the current covariate Z_t is observed before J_t;
- under H0, Law(X_t | Z_t,F_(t-1)) = Law(Y_t | Z_t,F_(t-1));
- the paired feature difference r_(k,t) is conditionally centered and bounded in norm by one.

Choose J_t using F_(t-1), Z_t, and randomness conditionally independent of the current pair. Then, conditional on the selected index,

    E[ r_(J_t,t) | F_(t-1), Z_t, J_t ] = 0.

For fixed 0 < lambda < 1, define

    F_t = 1 + lambda * <u_(J_t,t), r_(J_t,t)>.

The norm bound gives F_t >= 1-lambda > 0, and conditional centering gives

    E[ F_t | F_(t-1), Z_t, J_t ] = 1.

Therefore M_t = product_(s<=t) F_s is a nonnegative martingale (or supermartingale if factors are conservative), relative to the filtration that includes the selector's randomness and decisions. Ville's inequality then supplies the single global anytime threshold 1/alpha.

This is a proof sketch and a list of obligations, not an implementation proof. In particular, the filtration must include the exact timing of covariate availability and selector randomization. If selection randomness or the current covariate leaks future response information, the argument fails.

## 4. Bandit selector

The proposed reward is the observed bounded log-factor:

    g_t = log(F_t) in [log(1-lambda), log(1+lambda)].

Candidate policies include:
- EXP3-IX as an initial robust bandit baseline;
- Tsallis-INF as an alternative exploration/regret baseline;
- deterministic greedy selection with a declared exploration floor;
- round-robin and fixed-expert controls.

The bandit selector is an efficiency/power optimization policy. The e-process guarantee comes from the factor's conditional validity, not from EXP3 or a regret theorem.

Because per-expert state is updated only on selected observations, formal regret must be defined against the same selected-subsequence state semantics. It must not be presented as regret against C057's full-information per-expert state. C057/C058 remain direct full-information performance comparators; report paired power/delay gaps separately.

## 5. Scalar capital restart

A single selected factor can be consumed by a dyadic all-start capital restart:

    R_t = F_t * (R_(t-1) + nu_t)
    E_t = U_t + R_t.

This is O(1) numeric restart bookkeeping per observation, in addition to selected feature work and O(K) selector state. It is a separate scalar adaptive strategy, not C058's per-kernel posterior mixture.

A scalar restart can also be run without a restart layer as an ablation. Compare restart and no-restart arms on identical source streams.

## 6. Integrity and failure behavior

- Selection, factor calculation, selector update and restart update are transactional.
- Invalid maps, nonfinite features, norm violations, stale prepared state, duplicate IDs and out-of-order observations fail closed.
- Selection probability floors and numerical clipping are part of the policy manifest.
- No post-hoc expert maximum, best-arm-on-current-pair selection, or unaccounted hyperparameter choice is allowed.
- Hashes bind content but do not authenticate source truth or establish stochastic assumptions.
- A deterministic negative control deliberately selects from current-pair outcomes; it must be rejected or demonstrate the expected failure in an isolated invalid-control track.

## 7. Decision metrics

Report:
- Full-horizon null ever-alarm with exact pointwise and simultaneous confidence bounds.
- Unconditional power, early-alarm rate and restricted delay with misses at H+1.
- Paired detection-delay and power differences against C057 and C058.
- Bandit regret only under matched selected-subsequence state semantics.
- Number of selected map evaluations, feature coordinates computed, bytes per pair, p50/p99 feature latency, selector latency and peak memory.
- Performance by sparse/local, high-frequency, late, oscillating and explicit blind-spot alternatives.
- Posterior or selection-probability concentration, exploration counts and cold-start cost.

## 8. Claim boundary

The project tests whether the compute saved by sparse expert evaluation is worth the exploration cost. It does not assume:
- zero selection penalty;
- domination of C057/C058;
- a new general theorem for adaptive testing;
- statistical Type-I calibration from deterministic tests;
- SOTA or deployment superiority.

Primary external comparison is SCPT on the same direct conditional paired null. SKCI (conditional independence) and Online RFF-MMD (unconditional two-sample change detection) must remain adjacent-task tables.
