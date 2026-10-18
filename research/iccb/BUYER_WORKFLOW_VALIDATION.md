# ICCB buyer-workflow validation plan

**Status:** Research protocol; no customer discovery interviews or paid-pilot commitments are claimed  
**Date:** 2026-10-18  
**Purpose:** Falsify the initial commercial hypothesis before building a broad family-office product

## Executive recommendation

Do **not** position ICCB as a replacement for family-office portfolio reporting, accounting, private-market document ingestion, or custodian aggregation. These are established categories with incumbent vendors. Masttro, for example, markets consolidated reporting across family entities and assets, real-time wealth visibility, data feeds, and AI-powered intelligence; Addepar is also a significant portfolio data and analytics platform. J.P. Morgan's 2026 survey finds substantial family-office interest in AI while reporting that many offices have limited direct exposure to venture/growth markets and infrastructure. These are adjacent signals, not proof of demand for ICCB.

The hypothesis worth testing is narrower:

> A family-owned enterprise or long-horizon asset owner with real operating assets has material decisions that require connecting financial projections to physical dependencies, maintenance liabilities, service continuity, evidence quality, and institutional context. Existing workflows may not connect those dimensions adequately. ICCB would be valuable only if it improves one repeated decision beyond the customer's existing tools.

If interviews show that existing EAM/ERP, engineering, portfolio, and advisory tools already solve the nominated job well enough, ICCB should integrate with them or abandon that use case. A broad “dynasty OS” pitch is not a substitute for a costly, recurring workflow.

## Competitive boundary

| Existing category | Existing solution examples | Do not claim ICCB replaces it |
|---|---|---|
| Consolidated wealth and investment analytics | [Addepar](https://addepar.com/), [Masttro](https://masttro.com/family-office-software) | Custodian aggregation, performance reporting, whole-estate visibility, entity and asset reporting |
| Family-office accounting / operational ERP | [Eton Solutions](https://eton-solutions.com/), [FundCount](https://fundcount.com/) | General ledger, entity accounting, tax workflows, accounting close or statutory records |
| Alternative-investment document ingestion | [Canoe Intelligence](https://canoeintelligence.com/) | Automated collection and extraction of capital-call notices, statements and other investment documents |
| Asset / engineering / maintenance systems | Existing EAM, CMMS, GIS, ERP and engineering-specific tooling selected by a customer | Equipment maintenance execution, engineering safety cases, authoritative asset registers or work orders |
| General-purpose AI and consulting | Existing approved LLMs, spreadsheets, advisory teams | General intelligence, fiduciary judgment, legal or engineering responsibility |

This table is an initial category map, not an exhaustive vendor review or independently validated feature comparison. Recheck current product capabilities in each prospect's actual environment before claiming a gap.

## Best first customer profile to test

Prioritize an owner-connected operating business or holding company that controls physical assets—such as energy, water, logistics, manufacturing, forestry, real estate, or multi-site facilities—and makes recurring capital allocation decisions across those assets.

Look for observable properties, not merely high net worth:

- Multiple projects compete for a constrained capital budget.
- Service continuity, asset condition, maintenance backlog, or shared dependencies change the economics of the decision.
- Relevant information is split across finance, engineering, maintenance, geography, suppliers and prior decision records.
- A named operator owns the decision, a finance lead owns the budget, and an executive sponsor can authorize a pilot.
- There is a repeated workflow and a measurable baseline to compare against.

A passive portfolio consisting mostly of liquid securities may not benefit enough from ICCB's physical-world capability graph to be the first target. Sovereign wealth funds and transnational investors remain longer-horizon targets until a private, repeatable deployment demonstrates value.

## First workflow hypothesis

**Capital-plan review for a family-owned operating company.**

The workflow is intentionally a decision-support pilot, not an autonomous investment agent.

Input:
- A consented, bounded list of proposed capital projects.
- Existing project cash-flow estimates and approved budget limits.
- Selected asset/site dependencies (power, water, logistics, suppliers, workforce or shared facilities).
- Known maintenance and replacement obligations.
- Source dates, uncertainty and known data gaps.
- The customer's decision authority and constraints.
- A small set of historical decisions and actual outcomes for retrospective evaluation if available.

Output:
- A comparison of project outcomes as a vector, not one opaque score.
- Explicit hard-constraint failures and assumptions.
- Critical shared dependencies and scenario-specific failure paths.
- Evidence freshness, conflict, scope and missingness flags.
- Sensitivity and stress scenarios with assumptions visible.
- A decision memo and a machine-readable run receipt.
- Human approver, dissent, final decision and later outcome record.

Do not include confidential family data in the public benchmark. Begin with synthetic fixtures; use customer data only after explicit permission, data-minimization, security and retention terms.

## Discovery protocol

Conduct 12–15 semi-structured interviews as an initial qualitative exploration, not as a statistically representative survey. Seek multiple roles and multiple organizations, rather than 15 respondents from one office. Suggested roles include family-office COO/CIO, operating-company CFO, asset/maintenance director, portfolio-company executive and the adviser who currently assembles the decision evidence.

Ask about past behavior before asking about the product:

1. “Walk me through the last significant capital allocation decision that involved competing projects. What triggered it?”
2. “Which people and systems supplied the data? Where did the numbers disagree?”
3. “What did you not know at the time that later mattered?”
4. “How were maintenance, shared suppliers, utilities and service continuity accounted for?”
5. “What alternatives were rejected, and how could a new decision-maker reconstruct why?”
6. “How long did the process take, how much rework occurred, and what was the cost of a wrong or delayed decision?”
7. “Which current product or adviser already performs this job well? What is still manual or missing?”
8. “What data could you actually permit a pilot to process, and under what controls?”
9. “Who owns the operational problem, who controls the budget, and what procurement step comes next?”
10. “What outcome would justify continuing, and what outcome would cause you to stop?”

Do not lead with “capability compounding,” consciousness, a five-project architecture, or claims that the stack is superior. Describe the workflow only after understanding the customer's existing process. Capture verbatim problem descriptions with permission; avoid publishing identifiable or confidential information.

## Falsification tests

The commercial hypothesis weakens materially if any of the following repeatedly occurs:

- The target buyer does not make a repeated decision for which finance and physical dependency information must be combined.
- Existing portfolio/ERP/EAM/GIS tools already deliver the needed outcome at acceptable cost.
- Data is too fragmented, stale, legally restricted or costly to normalize for a useful pilot.
- There is no identifiable budget owner or the problem is too infrequent to justify a product.
- Operators do not trust the proposed outputs enough to review them, even in non-executing mode.
- The tool merely produces a more attractive memo without detecting material issues or reducing human effort.
- No prospect will agree to a defined pilot scope, baseline and decision date after seeing a concrete demo.
- Integration and support costs exceed the value the workflow can reasonably deliver.

Negative interviews and reasons for rejection should be recorded alongside positive responses. Expressions of interest, conference praise and letters without a defined evaluation path are not evidence of product-market fit.

## Pilot design and acceptance

### Stage A — Synthetic proof

Use the ICCB public fixtures to show one coherent example: a high projected-cash option fails an explicit maintenance or authority gate while an alternative has lower cash but better capability outcomes. Compare against a simple spreadsheet-equivalent baseline. Show limitations and source gaps openly.

### Stage B — Retrospective shadow evaluation

With permission, select 5–10 previously completed decisions with recorded inputs and known outcomes. Give the tool only information available at the original decision date to avoid look-ahead bias. Have a domain expert independently review whether the tool identifies material dependencies, missing evidence and constraints. Separate facts known at decision time from outcomes revealed later. This small sample is for finding failure modes, not for estimating investment alpha.

### Stage C — Prospective non-executing pilot

For a single organization, choose a bounded repeated workflow and preregister the baseline and success conditions before the system sees the final evaluation cases. Recommendations do not execute trades, purchase orders or control actions. Humans retain authority.

Candidate measures:
- Analyst/engineering hours per decision package.
- Time from kickoff to review-ready evidence package.
- Material errors or rework found by independent reviewers.
- Recall of planted/known critical dependencies and maintenance obligations.
- Percentage of material claims with usable provenance or explicit unknown status.
- Constraint violations and inappropriate confidence.
- Forecast error only where outcomes and adequate sample sizes exist.
- Fully burdened implementation, support and governance cost.
- Sponsor's decision on whether to renew or expand.

### Proposed go/no-go rules

These are provisional pilot gates to agree with the customer before testing, not established results:
- Zero hard-gate violations in the frozen synthetic acceptance set.
- At least 90% recall of pre-labelled material dependencies in the agreed test set, with false positives shown.
- All material factual claims are evidence-linked or explicitly unknown in the synthetic test set.
- At least one meaningful, customer-defined improvement in review time, error discovery, or decision quality versus the existing workflow, without unacceptable regressions.
- At least one named operational owner and budget owner accept the result and specify a next procurement step.
- The fully burdened cost and support plan are plausible relative to the validated benefit.

A first pilot does not prove general financial outperformance. It can establish only that the workflow is useful for the evaluated customer and scenario.

## Commercial evidence ladder

Treat evidence as a sequence, not as interchangeable “traction”:

1. Problem interview: the respondent describes an actual repeated pain from recent experience.
2. Workflow access: a prospect agrees to share a bounded, non-sensitive or properly governed example.
3. Baseline agreement: both parties agree on the current process, metric, and comparison.
4. Pilot sponsor: an operational owner and budget authority are named.
5. Paid pilot or formally budgeted evaluation: real willingness to allocate resources is demonstrated.
6. Repeat use: the workflow is used again in a new decision cycle.
7. Renew/expand: the customer pays for sustained deployment after measuring results.
8. Referenceable case study: public claims are explicitly approved and independently reviewable.

Do not present a higher rung when only a lower rung has been reached.

## What this means for product positioning

Lead with **evidence-backed capital planning for asset-heavy family enterprises**, not “one operating system for dynasties.” The first product should integrate with the systems customers already trust and improve a high-value gap across financial and physical decision evidence.

Only expand to family-wide institutional memory, portfolio-level dependency analysis, cross-company learning and sovereign-scale partnerships when the first workflow shows repeatable value, security, governance and acceptable deployment cost.

## Sources and limits

- UBS, [Global Family Office Report 2026](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html), 28 May 2026. Survey of 307 UBS family-office clients; treat as a sample, not the complete market.
- J.P. Morgan Private Bank, [2026 Global Family Office Report](https://privatebank.jpmorgan.com/nam/en/insights/reports/2026-family-office-report), survey of 333 offices across 30 countries.
- International Forum of Sovereign Wealth Funds, [Santiago Principles](https://www.ifswf.org/santiago-principles-landing/santiago-principles), governance and investment principles for sovereign wealth funds.
- [Addepar](https://addepar.com/) and [Masttro](https://masttro.com/family-office-software) public product descriptions. Vendor statements are not independent performance evaluations.
- [Canoe Intelligence](https://canoeintelligence.com/) public product descriptions; verify the current scope directly with each prospect.
- NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). A reference for organizing AI risk work, not a certification.

No customer interviews, customer data access, paid pilots, independent security assessment, or live investment performance are claimed by this research plan.
