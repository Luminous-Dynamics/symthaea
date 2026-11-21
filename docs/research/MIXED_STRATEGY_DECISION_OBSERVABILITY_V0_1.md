# Mixed-Strategy Decision Observability Bounds (v0.1)

**Tracking:** NEURO-SILICON-012 / #9563  
**Implementation:** `research/temporal-evaluator/src/decision_observability_mixed.rs`  
**Composes with:** #13152 deterministic decision-observability evaluator, #9637 signal aliasing/observability evaluator, #9616/#13151 candidate-side sensing policies, and #9954 irregular temporal decision benchmark  
**Status:** source-authored evaluator extension; tests are authored but not executed

## 1. Why this extension exists

The deterministic v0.1 evaluator in #13152 is correct for deterministic policies with an identical candidate view and frozen seed. Its deterministic minimax-regret value is **not** a lower bound for randomized policies.

A simple counterexample has two indistinguishable worlds and two actions:

| Hidden world | left | right |
|---|---:|---:|
| world 1 | 0 | 1 |
| world 2 | 1 | 0 |

A deterministic action has worst-case regret 1. A policy that chooses each action with probability 0.5 has expected regret 0.5 in either world. Scoring the randomized policy against the deterministic ambiguity floor would incorrectly label unavoidable randomization value as an improvement over a supposedly universal lower bound.

The extension adds a finite mixed-strategy evaluator without changing candidate policy code or the existing deterministic certificate.

## 2. Evaluator contract

For every hidden world in one equivalence class, bind the same:

- candidate-visible observation-history digest;
- candidate recurrent-state digest;
- model/configuration digest and frozen policy seed;
- candidate policy implementation/profile digest;
- exact candidate action-probability distribution over a common admissible action set.

Each hidden world separately carries its evaluator-only identity, hidden-world digest and action-loss vector. Those fields must remain outside candidate-visible artifacts.

Candidate probability mass is represented as integer parts per million (PPM), and must sum exactly to 1,000,000. Losses are normalized to the same range. Worlds must have unique IDs and distinct hidden-world digests. The action set must be identical and complete across worlds.

This is a one-decision probability-kernel contract. It does not model a sequential randomized policy whose later distributions depend on new observations or private state transitions.

## 3. Mixed minimax-regret formulation

For hidden world \(h\) and admissible action \(a\):

\[
r(h,a)=L(h,a)-\min_{a'}L(h,a')
\]

The mixed-strategy ambiguity floor is the value of a finite zero-sum game:

\[
R_{\mathrm{mix}}=\min_{p\in\Delta(A)}\max_h
\sum_a p(a)r(h,a)
\]

Here \(p\) is a probability distribution over actions; the maximizing player selects the worst hidden world (or, equivalently for the finite game, a distribution over worlds).

For the candidate's supplied distribution \(p_c\):

\[
R_c=\max_h\sum_a p_c(a)r(h,a)
\]

The evaluator uses exponential-weights/no-regret updates for the action player and the hidden-world player. It averages the iterates and quantizes each final distribution to PPM using largest remainders with stable tie-breaking. The solver is approximate; the final strategy distributions are valid, and the evaluator computes outward-rounded bounds using integer arithmetic:

\[
\mathrm{LB}=\min_a\sum_h q(h)r(h,a)
\]

\[
\mathrm{UB}=\max_h\sum_a p(a)r(h,a)
\]

For the returned primal strategy \(p\) and dual strategy \(q\), weak duality gives:

\[
\mathrm{LB}\le R_{\mathrm{mix}}\le\mathrm{UB}
\]

The evaluator reports the duality gap \(\mathrm{UB}-\mathrm{LB}\). It reports solver tolerance as met only when this measured gap is at or below the frozen configured tolerance. The actual avoidable regret is interval-valued:

\[
\max(0,R_c-\mathrm{UB})\le
R_{\mathrm{avoidable}}\le
\max(0,R_c-\mathrm{LB})
\]

This interval is intentional. A fixed iteration count is not a convergence proof.

## 4. Boundedness and reproducibility

The current profile caps the evaluator at 128 hidden worlds, 64 actions and 8,192 iterations. The default is 4,096 iterations, a 50,000-PPM learning rate and a 10,000-PPM maximum duality gap. The evaluator returns the actual gap even when tolerance is not met.

Worlds are sorted by evaluator world ID before floating-point iteration. Action order follows the frozen candidate distribution. Final probabilities are quantized to integer PPM; the returned lower and upper bounds use checked integer accumulation and outward rounding. The floating-point iterative solver is an optimizer, not the source of the final bound arithmetic.

The solver profile, candidate policy digest, candidate view identity and output certificate must be included in the run receipt. Confirmatory benchmark configuration must freeze the solver profile before evaluation.

## 5. Required controls and tests

Ten Rust test functions are authored for:

1. the opposed-world case where mixing halves the deterministic worst-case regret;
2. bounding a poor candidate policy's excess regret above the mixed floor;
3. a common optimal action across hidden worlds;
4. rejecting a different candidate distribution across indistinguishable worlds;
5. rejecting mismatched candidate view or policy identity;
6. rejecting probability mass that does not sum to one million PPM;
7. rejecting incomplete or inconsistent action sets;
8. accepting a deterministic point-mass distribution as a mixed strategy;
9. checking bound ordering and exact reported duality gap;
10. rejecting invalid/unbounded solver profiles.

**No Rust test, build, formatting or Clippy result is claimed.** The repository's stated execution blocker remains the unavailable local Rust toolchain and skipped CI workflows. These tests are authored, not executed.

Before a confirmatory use, add property-based and exhaustive small-matrix tests against a separately implemented LP/reference solver, plus randomized action/world permutation invariance and quantization-bound tests. In particular, test degenerate matrices, tied equilibria, sparse support, maximum matrix dimensions and worst-case iteration cost.

## 6. Relationship to prior art and claim boundary

Mixed strategies, minimax regret and no-regret learning for finite zero-sum games are established subjects. Exponential-weights methods and their average-strategy convergence are known prior art; this PR does not claim to invent them.

The implementation contribution is a task-specific evaluator adapter that:

- binds the candidate probability kernel to an observational-equivalence identity;
- separates evaluator-only hidden worlds and losses from candidate inputs;
- computes a robust mixed-policy ambiguity interval;
- reports the primal/dual gap and interval-valued avoidable regret alongside the deterministic result.

The result is only a lower/upper bound for the declared finite hidden-world and action set under the declared loss matrix. It is not a universal impossibility theorem, a proof that the hidden-world set is complete, a sequential-game solution, a secure oracle boundary, or evidence of improved deployed decision quality. #9437 process/OS isolation and signed receipts remain prerequisites for a security claim. No SOTA, hardware, energy or performance claim is made.
