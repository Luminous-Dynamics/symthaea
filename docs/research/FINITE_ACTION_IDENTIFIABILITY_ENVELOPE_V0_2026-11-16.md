# Finite-action scientific identifiability envelope v0

**Date:** 2026-11-16  
**Owner:** Symthaea #12454  
**Implementation:** standalone falsification-power-reference Rust crate  
**Composes with:** #8669, #8628, #11060, #12402, #12426

## 1. Research question

Before asking whether a researcher selected the right mechanism, determine whether the available experiment actions could identify a unique mechanism at all.

For a finite candidate set and a frozen action set, this reference:
1. partitions mechanisms by exact finite IEEE-754 expected-mean signatures over every admissible action, canonicalizing signed zero;
2. retains mechanisms in an irreducible action-space equivalence class as a set;
3. searches for the minimum-cost integer replication schedule that separates all distinct classes at a declared family-wise alpha and pairwise target power;
4. returns BudgetInsufficient rather than pretending that an underpowered schedule proves equivalence.

This is a finite evaluator oracle, not a posterior sampler or an action-selection policy.

## 2. Formal contract

Each action a has:
- known positive Gaussian measurement standard deviation sigma_a;
- positive cost per independent replicate c_a.

Each mechanism m declares an evaluator-side mean mu_(m,a) for every action. Actions and mechanisms are canonicalized by their IDs before reporting.

The response signature preserves exact finite IEEE-754 mean identity per action, canonicalizing +0 and -0 to the same representation. Any nonzero mean difference is a distinct mechanism alternative, however small relative to measurement noise. This avoids turning a practical detection-limit problem into a false claim of mathematical equivalence. The predicted means and canonical serialization are part of the evaluator input manifest.

For a pair of mechanisms, a replication vector r_a yields simple Gaussian likelihood-ratio information:

    I_(m,n)(r) = sum_a r_a * (mu_(m,a) - mu_(n,a))^2 / sigma_a^2

Let P be the number of cross-class mechanism pairs (all choose(M,2) mechanism pairs minus the irreducible within-class pairs). The v0 reference uses the conservative Bonferroni allocation across every pair that the report evaluates:

    alpha_pair = alpha_familywise / P

For a one-sided known-direction Gaussian likelihood-ratio test:

    power = Phi(sqrt(I) - Phi_inverse(1-alpha_pair))

Thus a pair reaches target power beta when:

    I >= [Phi_inverse(1-alpha_pair) + Phi_inverse(beta)]^2

The search enumerates every nonnegative integer replication vector whose total count is within the declared maximum (at most 16 in v0). A schedule qualifies only when the *worst pair across all mechanism members of every pair of distinct signature classes* reaches the target. This guards against an equivalence-class representative that is easier to separate than another member.

Among qualifying schedules, choose minimum total cost; ties resolve by observation count and then lexicographic replication vector. If no schedule meets the target, report the bounded schedule with the highest worst-pair power, then minimum cost/size/lexicographic tie breaks.

## 3. Output and dispositions

- SingleEquivalenceClass: all candidate mechanisms share one exact action-response signature. No admissible replication schedule can uniquely identify a member under this finite action set.
- TargetAchievable: a minimum-cost schedule meets the declared all-pairs information requirement.
- BudgetInsufficient: the classes are distinct, but no schedule under the observation cap reaches the declared target. This is not observational equivalence.

The report includes class membership and multiplicity, irreducible equivalent-pair fraction, family-wise and pairwise alpha, required information, action-level replicate counts, total observation count and cost, and the worst separated mechanism pair with its achieved information and power.

## 4. Implementation and controls

The Rust module provides:
- strict finite/unique input validation;
- canonical sorting independent of caller order;
- exact per-action response signatures (signed zero canonicalized);
- deterministic equivalence classes;
- exact bounded enumeration for at most six actions, twelve mechanisms and sixteen total observations;
- conservative family-wise pairwise information requirement;
- cost-minimal target-achieving schedule or explicit budget-insufficient schedule.

Tests cover:
- mechanisms observationally equivalent over every available action;
- narrow/aliased design resolved by a complementary action;
- a minimum-replicate known answer;
- budget insufficiency without false equivalence;
- cheaper/noisier versus expensive/precise measurement costs;
- a multi-mechanism case where each action alone leaves an unresolved pair but the joint schedule separates all classes;
- family-wise alpha tightening as class count increases;
- order-independent deterministic reports;
- a tiny nonzero mean difference stays a distinct, potentially underpowered alternative;
- malformed/non-finite inputs fail closed.

## 5. What this does not establish

The oracle assumes independent Gaussian measurements, known action-specific sigma, fixed expected responses, a simple pairwise model and a frozen action set. It does not qualify unknown variance, serial dependence, composite hypotheses, adaptive future actions, causal interference, sensor calibration, simulator validity or physical measurement systems.

The equivalence ceiling is only relative to the supplied candidate mechanisms and the declared finite action set. A missing mechanism in the candidate set is invisible to this oracle; an unrepresented action could split a class; a flawed noise/cost contract can make the output misleading. The report must preserve these assumptions and be attached to its exact input manifest.

The one-sided pairwise tests use evaluator-known directions. Bonferroni supplies a conservative family-wise reference for the declared finite pair set; it is not a universal model-selection guarantee.

## 6. Portfolio ownership

- #8669 continues to own posterior mass over functional-equivalence classes, prior-splitting invariance and support omission.
- #8628 continues to own law-drift versus measurement-drift diagnosis and the diagnostic action semantics.
- #11060 / #11042 continue to own blind world/action protocols and structural discovery.
- #12402 / PR #12405 continue to own predictable Gaussian sequential falsification.
- #12426 / PR #12433 continue to own fixed-design Gaussian terminal-test induction.
- #12454 owns only this finite-action evaluator identifiability/separation-cost oracle.

No second posterior, action selector or evidence authority is introduced.

## 7. Research comparators

Liu, Maini & Baker, *Optimal experiment design for practical parameter identifiability and model discrimination*, Mathematical Biosciences 399 (September 2026), DOI 10.1016/j.mbs.2026.109710, explicitly studies design for parameter identifiability and model discrimination using profile likelihood and optimal control. This work is not claiming that model-discriminating experiment design is new. The narrower increment is an exact, bounded, auditable finite-action oracle that reports an identifiability ceiling, irreducible set-valued classes and minimum replication cost for a declared simple Gaussian family.

## 8. Claim boundary

A passing exact-head Rust workflow qualifies implementation behavior and known-answer fixtures. It does not establish new scientific discovery, superiority on Unsatz/NewtonBench/FrontierPhysics, real-world mechanism identification or physical validity. Campaign use still requires independent action/response truth custody, a pinned observation model, matched resource accounting and sealed confirmation.
