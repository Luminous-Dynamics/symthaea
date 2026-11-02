# Compositional Risk Budget Ledger v0.1

**Status:** Rust research implementation and synthetic topology fixture. No component theorem, calibration result, or real-world guarantee is verified here.

Tracking: [NEURO-SILICON-022 / #10066](https://github.com/Luminous-Dynamics/symthaea/issues/10066).

## 1. Problem

An agent may use a calibrated predictor, an action-risk controller and a compute-stopping rule. Three individually reported risk numbers do not automatically make a valid system-level risk statement. The target population, unit of analysis, execution horizon, conditioning event, filtration and certification experiment must line up.

The ledger makes the compatibility rules executable and fails closed when the claims cannot be composed.

## 2. Two different risk budgets

- **Operational alpha:** upper bound on the probability of one or more declared adverse events during the execution scope.
- **Statistical confidence delta:** upper bound on the probability that one or more statistical certification statements fail to cover their target quantities.

These values answer different questions. The ledger uses integer parts-per-billion and never converts one budget into the other.

## 3. The only two formal arithmetic reductions in v0.1

### Unconditional marginal event groups

If each component provides an unconditional event-probability bound P(E_i) <= alpha_i on the same declared target population, subject/policy and execution episode, then:

    P(union_i E_i) <= sum_i alpha_i

This is a union bound; it does not require independence. It does require each component bound to actually apply to the same target population and full execution scope. The shared execution-scope digest is a compatibility assertion, not proof that those assumptions are true.

The marginal group permits finite-sample marginal claims, sequential-anytime claims and compute-stopping claims. A sequential claim must also bind its filtration digest. Each method's theorem/assumptions remain distinct in the contract.

### Complete action-conditional partition

Suppose the actual selected action A is known to lie in a frozen finite support S. For every action a in S, assume each event-family claim satisfies:

    P(E_j | A=a) <= alpha_(j,a)

Within each action, a union bound gives:

    P(union_j E_j | A=a) <= sum_j alpha_(j,a)

Then:

    P(union_j E_j) <= sum_a P(A=a) * sum_j alpha_(j,a)
                  <= max_a sum_j alpha_(j,a)

This is why the complete-partition group uses a **maximum across actions**, not a sum across action IDs. Within each action, event bounds are still summed. No action frequency estimate is required.

Required conditions are strict: every possible selected action must be in the frozen support; each action must have the same complete event-family set; every claim must condition on the actual selected action; all claims must refer to the same population, policy/subject, execution episode and joint certification experiment. An omitted or out-of-support action makes the plan invalid.

The output remains an arithmetic bound over declared conditional claims. It is not an AC-CP theorem verifier.

### Combining groups

Each compatible group is reduced to an unconditional probability bound over the same execution scope. Group bounds are then union-bounded:

    alpha_system <= sum_g alpha_group_g

An empirical-only or unsupported component is not silently omitted: its no-formal-credit group makes the overall report unsupported and suppresses the system alpha result.

## 4. Confidence failure is a separate union bound

Each contract has a statistical delta allocation. If each component certification statement fails with probability at most delta_i over the declared joint certification experiment, then:

    delta_system <= sum_i delta_i

This sum includes all action-specific certificates, not the max-over-actions reduction. The claims may be dependent; no independence multiplication is used. The shared certification-experiment digest makes the intended joint experiment explicit, but does not prove its sampling assumptions.

## 5. Contract and receipt contents

Each frozen contract binds:

- claim ID, event-family ID, group ID and guarantee kind;
- operational alpha and statistical delta allocations;
- target population, full execution horizon, subject/policy and joint certification experiment digests;
- calibration-data and assumptions digests;
- method/version, source paper/specification URL and pinned source revision;
- sequential filtration digest, action ID and complete action-support digest where applicable;
- optional upstream receipt digest.

The complete sorted contract set and group declarations are frozen into a plan SHA-256 digest. Evidence receipts then append in that exact preregistered order. Each receipt binds the contract digest, evidence digest, method-result receipt digest, optional upstream receipt digest and previous receipt hash. The ledger digest binds the full receipt chain.

Hashes detect mutation relative to the frozen bytes. They are not digital signatures, proof of author identity, trusted timestamps, external preregistration, WORM storage, or proof that an upstream receipt is correct.

## 6. Reuse existing infrastructure

- [#9465 — canonical anytime confidence family ledger](https://github.com/Luminous-Dynamics/symthaea/pull/9465) remains responsible for stream-family membership, alpha allocation, look sequencing and replay. This ledger can bind its exported receipt digest; it does not rebuild that protocol.
- [Run-wide pruning risk ledger](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/RUN_WIDE_PRUNING_RISK_LEDGER_V0_1.md) remains responsible for pruning-event risk and run-wide event allocation.
- [#10042 — policy risk](https://github.com/Luminous-Dynamics/symthaea/issues/10042) supplies marginal CPC-style and action-conditional risk claims.
- [#10052 — compute stopping](https://github.com/Luminous-Dynamics/symthaea/issues/10052) supplies separate wrong-stop and premature-stop claims.
- [#9882 — exact resource-vector controller](https://github.com/Luminous-Dynamics/symthaea/issues/9882) remains the owner of deterministic hard resource feasibility. Hard constraints are not statistical alpha credits.

## 7. Synthetic reference case

The bundled fixture has six claim declarations across two groups: a marginal prediction/compute/temporal group and a two-action complete conditional partition. It binds paper/source revisions for CPC, AC-RAC and Conformal Thinking, and includes an upstream anytime-ledger receipt reference.

The fixture's evidence hashes are deterministic synthetic placeholders. The sample alpha/delta allocations demonstrate the ledger's arithmetic only. They are not measured risk values, verified theorem certificates, or evidence that any of the cited methods has been reproduced in Symthaea.

Expected arithmetic for the fixture:

- marginal group: 8 + 12 + 7 + 5 = 32 million PPB;
- conditional action group: max(6, 10) = 10 million PPB;
- overall arithmetic alpha: 42 million PPB against a 50 million PPB budget;
- statistical delta: 10 million PPB against a 20 million PPB budget.

The report must still say component guarantees are unverified and formal_system_guarantee_claimed=false.

## 8. Industry / governance position

NIST AI RMF organizes risk work into Govern, Map, Measure and Manage, with governance cross-cutting the other functions. It calls for ongoing risk measurement, uncertainty reporting, benchmark comparisons, documentation and regular monitoring. This ledger is a machine-checkable measurement/management artifact that can support those activities; it does not replace governance, impact analysis, human oversight or incident response.

NIST states that AI RMF 1.0 is being revised and, on April 7, 2026, published a concept note for a trustworthy-AI profile for critical infrastructure. We should therefore keep the risk schema versioned, map fields to updated frameworks as they mature, and avoid describing this research ledger as a regulatory certification.

Reference: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/

## 9. Acceptance targets (prospective)

- 0 alpha/delta cross-ledger spending or double debit accepted.
- 0 incomplete action partitions accepted.
- 0 wrong population, execution scope, subject or certification-experiment digest combinations accepted.
- 100% of report outputs replayable from frozen plan and append-only evidence receipts.
- Empirical-only/unsupported claim always blocks a full system composition.
- Exact small finite cases match hand-derived union-bound and max-over-actions results.
- Mutation, missing-family, duplicate, wrong-order, policy-change and upstream-ledger-digest negative controls all fail closed.
- No formal system guarantee emitted by v0.1, even when arithmetic is valid.

## 10. Claim ceiling and next theorem work

This ledger proves only consistency of its own contracts, completeness rules, arithmetic and hash chain. It does not prove CPC/AC-RAC/Conformal Thinking assumptions, outcome independence, exchangeability, target-distribution stability, filtration correctness, or that an action is safe.

The next mathematical task is to formalize the exact interface between each component theorem and its contract: loss/event definitions, target population, conditioning, finite-sample or anytime horizon, confidence delta semantics, and what independent evidence makes the claim qualified. A later version may accept trusted verifier attestations, but v0.1 does not treat a free-form 'verified' flag as authority.
