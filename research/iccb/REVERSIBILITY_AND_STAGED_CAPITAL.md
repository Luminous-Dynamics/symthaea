# ICCB paradigm shift: staged commitments, reversibility and real options

**Status:** Research/design protocol with a small deterministic tranche gate; no real-option valuation or live customer performance is claimed  
**Date:** 2026-10-18

## Core shift

A conventional capital review is often presented as a choice among projects using projected return, risk, liquidity, and approval limits. ICCB should also represent **the policy for how capital is committed over time**.

The question is not only “which project is best?” It is also:

- How much capital should be committed before critical uncertainties are resolved?
- Which evidence must be observed before the next tranche is released?
- Which milestones justify continuing, revising, pausing, or stopping?
- What is the cost of waiting for information versus the cost of irreversible commitment?
- Which losses become unrecoverable after each stage?
- Who is authorized to accept changed risk, and what evidence can invalidate that decision?

This creates a testable transition from one-shot project scoring to **evidence-gated, reversible commitment paths**. It is related to real-options theory, staged finance, and adaptive project governance; ICCB should not claim to have invented those disciplines. The potential contribution is a deterministic, auditable decision contract joining stage gates to physical dependencies, maintenance liabilities, provenance, and successor-readable reasons.

## Why test it now?

Family offices surveyed by UBS in 2026 reported interest in AI, power/resources, and infrastructure, while the same survey documented persistent succession gaps. J.P. Morgan's distinct survey found AI interest alongside limited infrastructure allocation. These are relevant motivations for testing real-asset decision workflows, not proof of demand for ICCB.

More immediately, recent reporting on the AI infrastructure build-out describes project delays/cancellations and increasing financing / execution concerns. The lesson for ICCB is not that all infrastructure investments are bad. It is that a benchmark should include the cost of committed capital, schedule slippage, changing assumptions, and the ability to stop before additional irreversible expenditure. See [Financial Times, “America's $10tn AI infrastructure boom is starting to fray”](https://www.ft.com/content/9ea5a37d-15c9-4d5b-94d6-719048c9cd3c) (published 18 October 2026; article is paywalled).

## Decision model

Represent a capital plan as a stateful sequence:

1. **Authorize a bounded first tranche.** The authorization records amount, total commitment cap, owner, mandate, evidence snapshot, and permitted use.
2. **Observe a milestone.** The evaluator records what passed, what failed, and which evidence supports the observation.
3. **Decide whether to release the next tranche.** It must not be released unless authority, cumulative cap, milestone condition, and evidence gates pass.
4. **Recompute the decision vector.** Refresh costs, liquidity, dependencies, maintenance, capacity, forecasts, and the remaining option to stop or change direction.
5. **Record continue / revise / pause / stop.** Record the approver, dissent, reasons, changed assumptions, and evidence that would reopen the decision.

A previously spent amount is not by itself a reason to continue. The benchmark must include sunk-cost traps and make the incremental decision depend on future outcomes, not on the need to justify past spending.

## Implementation present in this branch

The crate's `staging.rs` introduces a small `TrancheRequest` contract. It evaluates:

- positive requested tranche amount;
- valid prior commitment and total cap;
- integer-overflow-safe cumulative commitment calculation;
- explicit milestone satisfaction;
- delegated authority;
- non-empty, qualified, in-scope evidence with usable observation/validity dates.

A failed condition rejects the next release. This is **a rule gate**, not an economic estimate of real-option value. The current implementation does not forecast milestone success, optimize tranche sizes, model schedule distributions, calculate discount rates, or prove that staging improves returns. The gate's milestone result and evidence state must be supplied by the caller and independently validated in future iterations.

The main evaluator also exposes:
- a cash-maximizing feasible baseline, using only options that pass hard constraints; and
- a Pareto frontier over expected cash, liquidity, productive capacity, maintenance burden, and critical-dependency overlap.

Reported valuation change remains observable but is deliberately excluded from Pareto dominance. No one metric should automatically substitute for productive capacity, and no weighted total should silently compensate for a hard-gate failure.

## Benchmark families

Add at least these cases to the frozen synthetic corpus:

1. **Successful staged deployment:** milestones pass and staging preserves the project's expected capability with manageable delay cost.
2. **Milestone failure:** the first stage reveals a fault; a fail-closed policy avoids releasing an unauthorized next tranche.
3. **Schedule slippage:** the project can still succeed, but delayed cash flows and financing costs alter the next decision.
4. **Evidence reversal:** a key initial assumption is invalidated by newer, in-scope evidence.
5. **Sunk-cost trap:** the candidate has already spent a large amount, but incremental continuation has lower prospective value than stopping.
6. **False milestone report:** a source claims a milestone passed but source scope, date, provenance or independent inspection fails validation.
7. **Irreversible versus reversible choice:** compare a large one-time commitment with staged or modular alternatives under the same hidden realized state.
8. **Liquidity shock mid-project:** a subsequent tranche would violate a predeclared liquidity floor.
9. **Dependency shock between stages:** a shared grid, water, logistics, or supplier dependency changes before release.
10. **Governance override attempt:** a high-return narrative tries to bypass the required approver or total commitment cap.

Each scenario needs at least one negative control and a simple policy baseline. The hidden evaluator knows what actually happened; candidate-facing inputs must not reveal outcome labels.

## Proposed benchmark metrics

Targets are provisional and need preregistration before final evaluation.

| Dimension | Measure | Initial target |
|---|---|---|
| Hard release gates | Unjustified tranche releases when milestone, evidence, authority or cap fails | 0 in the frozen fixture set |
| Capital bound | Cumulative commitment exceeding approved cap due to evaluation logic | 0 |
| Evidence invalidation | Detection of stale, scope-mismatched and contradicted milestone evidence | ≥95% recall on planted faults; report false positives separately |
| Sunk-cost resistance | Cases in which past spend improperly changes the incremental action absent future-value justification | 0 unjustified continuations in scripted negative controls |
| Decision regret | Difference from evaluator oracle, under a declared objective vector and scenario family | Compare to simple all-at-once and cash-maximizing baselines; no preclaimed universal improvement |
| Option preservation | Unnecessary irreversible commitment under uncertainty when a reversible path preserves value at tolerable cost | Measure on held-out scripted cases; report assumptions and regret |
| Time / financing cost | Delay, bridging finance, inflation and opportunity cost introduced by staging | Always report; staging does not receive free option value |
| Liquidity resilience | Violations under mid-project liquidity shocks | 0 hard-floor breaches in required negative controls |
| Decision trace | Exact approver, conditions, evidence and rationale for each release | 100% present in synthetic receipts |
| Reproducibility | Same input, configuration and version produce the same gate results and receipt | 100% equal outputs for deterministic fixture replay |

No aggregate score should hide an unauthorized tranche release or an accounting error. Report the costs of waiting as carefully as the benefits of preserving the option to stop.

## Research questions to resolve before claiming advantage

1. Which uncertainty can actually be resolved by staging, rather than merely postponed?
2. How costly is waiting in the use case (financing, schedule, missed revenue, inflation, scarce equipment or capacity)?
3. What is the appropriate baseline for the customer's existing milestone process?
4. Which milestones are observable, independently verifiable, and hard to game?
5. How should the evaluator represent correlated project risks, shared dependencies, and common financing constraints?
6. What conditions are safety / legal / mandate constraints, rather than preferences that can be weighted against return?
7. Does this protocol add value beyond a well-run stage-gate process and a transparent spreadsheet?

## References

- UBS, [Global Family Office Report 2026](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html).
- J.P. Morgan, [2026 Global Family Office Report](https://privatebank.jpmorgan.com/nam/en/insights/reports/2026-family-office-report).
- Financial Times, [“America's $10tn AI infrastructure boom is starting to fray”](https://www.ft.com/content/9ea5a37d-15c9-4d5b-94d6-719048c9cd3c), 18 October 2026.
- Existing ICCB protocol and Rust prototype: [PR #7974](https://github.com/Luminous-Dynamics/symthaea/pull/7974), [PR #7987](https://github.com/Luminous-Dynamics/symthaea/pull/7987), and [issue #7976](https://github.com/Luminous-Dynamics/symthaea/issues/7976).

No real asset model, customer stage-gate workflow, live investment result, or customer commitment is represented by this document.
