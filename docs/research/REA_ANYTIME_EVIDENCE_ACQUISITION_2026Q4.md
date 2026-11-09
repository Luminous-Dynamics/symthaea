# REA — anytime evidence acquisition, claim stopping and readiness revocation (2026 Q4)

**Status:** original synthetic design fixtures and sequential-evaluation protocol. No human-adjudicated oracle, production acquisition policy, or benchmark performance result is claimed.

**Owner:** [#11199 — REA-RUST-053](https://github.com/Luminous-Dynamics/symthaea/issues/11199), composed with #11108, #11125/#11127 and #11162/#11163.

## Executive thesis

An evidence-driven investigator must decide what to observe next, when to stop, which claim strength to issue, and how to maintain that claim as new evidence arrives.

A one-shot benchmark misses three high-risk behaviors: stopping while mandatory proof/defeater duties remain unresolved; continuing after a useful bounded conclusion is already available; and retaining a stale Ready result after contradiction, expiry, digest mismatch or lineage disclosure.

The paradigm is **anytime, sequential, revocable claim assurance**.

## Research foundation

- Cheng & Huan, *Optimal Stopping for Sequential Bayesian Experimental Design* (2025, revised 2026), model experiment selection and termination jointly as a finite-horizon decision problem. REA adds typed proof obligations, evidence admissibility, defeater coverage and revocation. https://arxiv.org/abs/2509.21734
- Koning & van Meer, *Anytime validity is free: inducing sequential tests*, JRSSB (21 February 2026), construct anytime-valid sequential tests from terminal tests and establish optional continuation at stopping times. This is a statistical validity result, not source authentication or proof of real-world truth. https://doi.org/10.1093/jrsssb/qkag050
- SRE-Bench (updated October 2026) supplies a separate binary reverse-engineering track with 262 binary instances and 1,572 graded tasks. REA stopping/claim measures must not be merged into its task score. https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- Behavior2Code (23 September 2026) emphasizes black-box behavior reconstruction and verifier quality. REA adds evidence traceability, stopping and claim-scope control without replacing functional-correctness grading. https://www.turing.com/blog/new-behavior2code-benchmark-reverse-engineer-software

## 1. Paradigm shifts

### 1.1 From static readiness to time-indexed, revocable readiness

Readiness is a function of the current evidence state, not a sticky boolean:

```
Ready(claim, t) = all mandatory proof obligations satisfied at t
                  AND all declared mandatory defeater duties satisfied at t
                  AND no unresolved material conflict, expiry or scope invalidation
```

A new observation may strengthen, weaken, contradict, stale or invalidate a prior result. Correctness is not monotone in the number of Evidence records.

### 1.2 From one claim to a declared claim ladder

A bounded observation-specific claim and a universal claim have different proof burdens. Each claim ID has its own required obligations and defeater duties. Select the strongest declared claim whose full burden is met.

For example, “no network event was observed in session S” may be warranted while “the application has no background network behavior” remains unwarranted. A bounded null observation must not become universal absence.

### 1.3 From positive proof only to proof plus falsification duty

A strong claim may require explicit disconfirming checks: negative control, independent acquisition, contradiction search, supersession review or population-gap audit. These duties are owned by the claim contract, not waived by the model because a probe has low EIG or high cost.

A challenge result can revoke readiness.

### 1.4 From fixed horizon or crude thresholds to decision-aware stopping

For current state s, compare terminal value with expected continuation value:

```
Stop when terminal_value(s) >= max_a E[continuation_value(s,a)]
                              - cost(a) - risk(a)
```

This is a modelling template, not a universal stop theorem. Authorization, reversibility, evidence admissibility and target-impact constraints remain hard gates. A high-utility but unauthorized action is ineligible, not merely expensive.

### 1.5 From a certificate to an auditable proof frontier

Record which obligations are satisfied, which are unresolved, which challenge duties remain, which evidence supports each state, and which events would revoke a claim. A Ready result should be accompanied by a proof frontier, not just a confidence percentage.

## 2. Dataset v0

The development corpus contains 14 original synthetic episodes covering:
- proof completion followed by mandatory challenge coverage;
- late counterexample and readiness revocation;
- source-lineage merge that removes false independence;
- temporal supersession that preserves a historical but not current claim;
- artifact digest mismatch invalidating descendant observations;
- bounded no-event result versus universal absence;
- challenge test that finds a counterexample;
- exhausted budget and unresolved cause;
- unauthorized / irreversible high-information candidate;
- processing failure after archive enumeration;
- multiple tools derived from one acquisition session;
- equivalent final evidence sets acquired in different orders;
- late API page-gap discovery that revokes population completeness.

Agent-view and grader-only inputs have separate Draft 2020-12 schemas. The grader file contains synthetic world-state labels, event outcomes and expected stop checkpoints; it must never be mounted in an evaluated agent workspace. Public synthetic grader data is not a hidden test set.

Only executed and independently qualified observations update current readiness. Predicted outcomes cannot satisfy obligations or defeater duties.

## 3. Stop/action contract

- `continue(action_id)`: propose an authorized, reversible, budget-compliant experiment.
- `stop_ready(claim_id)`: declared proof obligations and defeater duties for that claim are satisfied now.
- `stop_bounded(claim_id)`: a narrower declared claim is justified while a stronger claim is not.
- `stop_unresolved(reason)`: no safe/eligible/useful action remains or the budget is exhausted.
- `abstain(reason)`: no declared claim is defensible.

A later event can reopen any previously ready claim. The system must identify the exact invalidated Evidence, source-family, time interval or target binding and preserve adjacent claims whose own burdens remain satisfied.

## 4. Metrics

| Metric | What it catches |
|---|---|
| False-ready rate | Mandatory proof/defeater duty missing but claim marked ready |
| False-stop rate | Stops with mandatory obligations or challenge duties unresolved |
| Premature-strong-claim rate | Uses stronger claim than evidence warrants |
| Useful bounded coverage | Correctly reports narrower claims instead of always abstaining |
| False-continue rate | Wastes actions after sufficient claim is available |
| Revocation recall | Fails to revoke on contradiction, expiry, digest mismatch or lineage collapse |
| Revocation latency | Decision steps a revoked claim remains falsely ready |
| Stale-ready duration | Exposure window caused by retained invalid state |
| Challenge coverage | Required negative/defeater checks actually executed and qualified |
| Cost to sufficient evidence | Acquisition cost/actions/time for the strongest justified claim |
| Order invariance | Equivalent final evidence states yield equivalent final readiness |
| Oracle regret | Cost/utility gap versus exact finite-horizon solution on small cases |

Do not collapse metrics into one score. Zero false-ready achieved by never asserting claims must fail useful coverage; low cost achieved by omitting challenge duties must fail challenge coverage.

## 5. Matched baselines

Compare on identical episodes, budgets and event schedules:
1. Stop at first positive result.
2. Fixed-horizon exhaustive acquisition.
3. Entropy-threshold stopping.
4. Point EIG with fixed horizon.
5. Robust EIG / EVSI / minimax with fixed stop.
6. Obligation coverage plus explicit defeater duties.
7. Combined proof-frontier + decision-aware continuation + revocable readiness.
8. Exact finite-horizon dynamic-programming oracle for small cases only.

The exact oracle is a finite-case diagnostic under supplied transitions and costs, not a general optimality guarantee.

## 6. Proposed qualification targets

Targets, not results:
- Zero false-ready accepts in deterministic negative controls.
- 100% revocation on explicit contradiction, expiry and identity-mismatch controls.
- Zero readiness upgrades from predicted/unexecuted outcomes.
- 100% declared defeater duties accounted for before the corresponding claim is ready.
- Equivalent final evidence states produce identical final readiness independent of acquisition order.
- Zero universal-absence upgrades from bounded no-event observations.
- Zero unauthorized/irreversible selections.
- Exact finite-horizon value/policy agreement on small deterministic oracle cases under declared tie-breaking.

Do not choose an arbitrary percentage improvement before a human-adjudicated pilot establishes difficulty, agreement and baseline variance.

## 7. Claim ceiling and integration

This work defines a sequential evaluation protocol and synthetic development fixtures. It does not prove every conceivable defeater was imagined, likelihoods are calibrated, stopping is optimal outside the finite oracle, or observations are authentic/true. Compose with #11108/#11125/#11162 and the existing EIG/EVSI/minimax planners. Reuse #10314 receipts; do not create another planner, generic evidence graph, trust score or receipt format.
