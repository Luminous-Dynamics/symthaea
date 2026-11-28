# CMAQ-064 — Predictable multi-fidelity evidence allocation

Status: **research design and proof obligations**. This is a new composition hypothesis. It is not yet an executable monitor, a proof-reviewed method, a benchmark result, or an SOTA claim.

## 1. Research question

Can one predictable controller allocate a bounded per-observation compute budget across **context region × temporal scale × response representation** and improve qualified detection power per unit cost over both a full-information monitor and a single-expert bandit monitor, while preserving one global anytime-valid false-alarm guarantee?

The system should spend more work where prior evidence says it is useful, but never let the current response pair decide which evidence process gets to inspect that same pair.

## 2. Existing ownership — compose, do not fork

- C060 (#14209 / PR #14210) owns one-expert predictable bandit selection, selector RNG and selected-subsequence history.
- C062 (#14253 / PR #14255) owns fixed-prior localized context-tree evidence with one global mixture.
- C061 (#14211 / PR #14254) owns the previsible bounded representation primitive; its parent evidence receipt must be integrated before using it as a live action.
- C059 (#14207 / PR #14208) owns common streams, SCPT and finite-sample reporting.
- C063 (#14320 / PR #14320) owns the common manifest and qualification protocol.
- C053/C056 own global risk composition and typed null contracts; C054/C055/C057/C058 remain comparators.

Do not add a parallel PRNG, alpha budget, null definition, stream generator, SCPT clone, or restart mechanism.

## 3. Core proposition: predictable convex allocation of valid factors

Let ℓ_t denote the information available after observing the current pre-pair covariate but before observing the response pair:

    G_t = sigma(F_(t-1), Z_t, Xi_t)

where F_(t-1) is the complete past, Z_t is known before (X_t,Y_t), and Xi_t is optional selection randomness generated independently of the current pair conditional on F_(t-1),Z_t.

For every selectable action j, suppose its factor F_(j,t) is:
- measurable using only G_t and the current pair;
- nonnegative (preferably bounded away from zero);
- conditionally mean-one under the registered direct conditional paired-equality null:

    E[F_(j,t) | G_t] = 1.

Choose a finite evaluated subset S_t and weights q_(j,t) before seeing X_t,Y_t, with:

    q_(j,t) >= 0,  sum_(j in S_t) q_(j,t) = 1.

The weights and subset are G_t-measurable. Form:

    G_t^* = sum_(j in S_t) q_(j,t) F_(j,t).

Then:

    E[G_t^* | G_t] = sum_j q_(j,t) E[F_(j,t)|G_t] = 1.

Thus the product:

    E_t = product_(s=1..t) G_s^*

is a nonnegative mean-one martingale (or e-process under the corresponding supermartingale assumptions) and Ville's inequality supplies one global threshold 1/alpha.

The single-action case S_t={J_t} is the C060 construction. A fixed set and fixed q is a factor-mixture baseline. A time-varying q_t chosen from past evidence and current pre-pair Z_t is the new adaptive-allocation hypothesis.

This proposition is only valid when every action in the mixture obeys the same registered null, filtration, predictable timing and positivity/centering contract. A factor with another null or a leaky representation cannot be repaired by convex mixing.

### Important distinction from C062

C062 maintains a fixed-prior mixture of each node's accumulated wealth:

    E_t^tree = sum_j pi_j product_(s<=t) F_(j,s).

C064 investigates a **product of per-time convex combinations**:

    E_t^alloc = product_(s<=t) sum_j q_(j,s) F_(j,s).

These are different processes. Both can be valid under their own assumptions, but their power, compute and regret properties differ. C064 must not silently replace C062's accumulated fixed-prior process or claim equivalent power.

## 4. Selection and compute-budget semantics

At time t:
1. Consume the pre-pair context Z_t and past-only state.
2. Derive a bounded action budget B_t and candidate action inventory from a frozen policy manifest.
3. Select a subset S_t and normalized weights q_t using only past state, Z_t and precommitted independent randomness.
4. Freeze all selected feature/representation versions and all betting directions/rates.
5. Evaluate only the selected actions on both current X_t and Y_t. Every response action applies the same unary transform to X and Y.
6. Form the convex factor G_t^* and prepare the global evidence/trace update.
7. Commit the evidence factor, action receipts, selector/RNG state and replay identity atomically.
8. Only after successful evidence commit, update selector weights and any representation for t+1.

An action is a typed tuple:

    action = (context_region, temporal_scale, representation, witness/betting_policy)

The selected subset has a frozen maximum cardinality and explicit per-action cost contract. Any unselected action receives zero mixture mass for that observation; no uncomputed factor may be substituted into the arithmetic mixture. The weights over actually evaluated actions must sum to one. A sentinel/fallback action is included if a policy could otherwise leave an observation with no evaluated action.

### Resource-allocation policies to compare

- Fixed-prior C062-style tree: reference for broad coverage and localized accumulated wealth.
- Single-action C060-style bandit selector: one selected factor per pair.
- Predictable convex subset: evaluate 2, 4, 8, or 16 factors per pair and combine them.
- Hierarchical selector: choose context level/region first, then scale/representation within the chosen branch.
- Multi-fidelity sentinel: always evaluate one cheap bounded sentinel; allocate expensive factors only when a pre-pair policy predicts higher expected value of information.
- Deterministic predictable routing: action selected from Z_t and past state, no randomization.
- Randomized contextual routing: a frozen RNG and explicit exploration floor, to protect rare contexts and reduce blind spots.

No policy is presumed superior. Compute savings and power gaps must be measured on matched streams.

## 5. Selector objective and its statistical boundary

Potential reward:

    g_t = normalize(log(G_t^*)) in [0,1]

or a selected-action reward derived from its committed log-factor, with a fixed documented normalization.

The selector is updated only after the evidence commit. A reward can be used for optimization, but the selector is not the source of validity. The only statistical guarantee comes from the per-action conditional mean-one property and the predictable convex combination.

Do not claim a standard EXP3/Hedge regret theorem just because the implementation resembles one. The correct comparator must specify:
- full-information versus bandit feedback;
- selected-subsequence versus counterfactual action-state semantics;
- how each expert's direction/history would evolve if it were selected;
- reward boundedness and importance weighting;
- whether action availability is context-dependent ("sleeping experts");
- the exact filtration and randomization model.

Until this is resolved, report adaptation quality as paired power/delay and cost versus C057/C058/C060/C062, not as a regret guarantee.

## 6. Representation contract

Each action's representation is content-addressed and previsible. For C061-derived actions:
- the current version is frozen before the pair;
- the same exact unary map and preprocessing are used for X_t and Y_t;
- normalization and any context transform are included in the norm proof;
- staged updates are bound to the current pair digest and encoder version;
- the parent evidence commit returns a typed receipt binding sequence, pair digest, encoder digest, factor and trace;
- only a matching accepted receipt permits the next version to become current.

Current-pair gradients, batch statistics, checkpoint selection, feature-dimension selection, or preprocessing updates before the current factor are integrity violations.

Fixed Fourier and SCPT-style actions remain in the inventory as controls. A learned action must not be allowed to replace all fixed coverage before its blind spots and adaptation costs are qualified.

## 7. Temporal and spatial structure

The action inventory should cover:
- context regions: global, coarse, fine, conjunction, oblique/learned region after a separate proof/contract;
- temporal scales: abrupt, gradual, periodic, high-frequency, temporary, late, repeated on/off;
- response representations: fixed Fourier, multi-band kernel, random neural, bounded learned encoder, conditional context map;
- witness styles: fixed direction, predictable adaptive direction, selected-subsequence direction, SCPT's past-only KNN conditional witness.

Every region/scale/representation inventory and prior/action-cost map is frozen in the policy manifest. Online selection may change weights and chosen actions predictably, but cannot retroactively change the candidate set or use an observed current factor to select itself.

## 8. Falsifiable hypotheses

H1. A convex subset of 2–4 actions can recover much of full-information C057/C058 detection power at materially lower feature cost.

H2. Context-conditioned action weights improve sparse/local alternatives relative to a global bandit selector with no local state.

H3. A cheap always-on sentinel plus selectively activated high-cost witnesses reduces mean work while preserving high-frequency/short-lived sensitivity.

H4. Exploration floors improve worst-family performance when contexts are rare or move across fixed region boundaries.

H5. Per-time allocation can outperform a static prior mixture in changing regimes, but may underperform when evidence is weak or the selector learns too slowly.

H6. Representation adaptation improves high-dimensional conditioning only when its representation update cost, finite-map blind spots and pair transaction are fully included.

Each hypothesis can fail. Negative and adversarial cells are first-class benchmark outcomes.

## 9. Qualification matrix

Use C063's common frozen manifest when available. Until then, use C059's common generator and same full-horizon replicate units.

### Comparators

C054, C055, C057, C058, C060, C062, faithful SCPT, and C061 only after parent receipt integration. Add ablations:
- static q versus time-varying q;
- deterministic versus randomized selection;
- 1/2/4/8/16 evaluated actions;
- global selector versus context-conditioned selector;
- sentinel-only versus sentinel + gated expensive action;
- fixed Fourier versus learned representation;
- fixed-prior mixture versus per-time convex factor product.

### Scenarios

- Valid conditional nulls: predictable covariate drift, heteroskedasticity, serial dependence and source/common-cause blocks while preserving the registered conditional paired equality.
- Support sizes 0.5%, 1%, 5%, 10%, 25%.
- Axis-aligned, oblique, conjunction, smooth-bump, narrow-manifold, rare, moving and partition-boundary contexts.
- Location, scale, tails, multimodality and equal-marginal pairing alternatives.
- Abrupt, gradual, periodic, high-/low-frequency, temporary, delayed and repeated changes.
- High-dimensional contexts and finite-feature blind spots.
- Negative controls: current-pair action/representation selection, look-ahead, coupled RNG, invalid action factor, unbounded feature, mutated inventory, replay, duplicate ID and partial commit.

### Resource budgets

Evaluate 1, 2, 4, 8 and 16 action/factor budgets per pair. Count:
- actual feature coordinates and kernel evaluations;
- router comparisons and selection operations;
- RNG calls and action-probability vector updates;
- representation inference/training;
- persistent state, memory traffic and peak memory;
- p50/p99 full-horizon valid-null latency and alternative live-to-alarm latency.

### Statistical metrics

- Exact per-cell false-alarm intervals and simultaneous bounds across the frozen null family.
- Unconditional power and restricted delay with early alarm/miss = H+1.
- p50/p90/p95 restricted delay.
- Paired differences on identical streams, with familywise uncertainty for preregistered primary contrasts.
- Worst support, context boundary, time scale and representation family.
- Power versus compute Pareto frontier subject to the same valid alpha.

Use at least C059's 10,000 null and 5,000 alternative replicate targets per primary cell, with the predeclared independent precision expansion rule. A small smoke run is not qualification.

## 10. Prospective acceptance gates (not achieved)

- Independent review of the conditional-centering/convex-allocation proof under the exact filtration.
- All actions satisfy the same direct conditional-pair null and bounded factor contract.
- Selector and representation are previsible; a current-pair-selection mutation must fail.
- Evidence, selector RNG/state and representation version commit atomically; a rejected evidence receipt changes none of them.
- One global alpha threshold with no action-count alpha split and no false-alarm-bound violation.
- At least 4× fewer feature-map evaluations than C057/C058 in the registered compute-constrained cells.
- No more than 2 percentage points of worst-family power loss versus the full-information reference.
- At least 20% lower restricted mean delay on three preregistered sparse/local alternatives versus the designated global and SCPT comparators, without degradation hidden by pooling.
- No unacceptable regression for periodic, temporary, late, high-frequency, rare-context or boundary-adversarial alternatives.
- Exact-head format/test/Clippy, mutation/property tests, reproducible raw artifacts and independent review before any claim of superiority.

## 11. Implementation sequence

1. Write a formal proof note for predictable convex action factors, including randomized and context-conditioned selection.
2. Specify a typed action descriptor and factor receipt that reuses C056 null semantics and C063 manifest identity.
3. Add a pure deterministic policy oracle first: given prior state, Z_t and RNG receipt, emit the selected subset and q_t. No current X/Y argument is available to this interface.
4. Add a bounded multi-action transaction that stages all factor outputs and selector state, then commits atomically.
5. Add mutation tests for pair leakage, RNG reuse/coupling, q sum != 1, omitted factor mass, invalid action null, stale representation and rollback.
6. Compare C060 one-action, subset allocation, C062 fixed mixture, and C057 full information on the same C059 streams.
7. Only after deterministic correctness, run C063 confirmatory calibration and power/cost grid.

The intended result is not an unconstrained adaptive detector. It is a **predictable compute scheduler whose every action is evidence-valid**, with an auditable global process and a measured resource-performance frontier.
