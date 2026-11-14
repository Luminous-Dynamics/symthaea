# Verified Capability Cost Accounting Crosswalk V0.1 — 2026-11-13

**Status:** design-only research crosswalk; no Luminous cost comparison has been measured.  
**Parent:** [INDUSTRY-STACK-COMP-001 #11774](https://github.com/Luminous-Dynamics/symthaea/issues/11774)  
**Composition:** canonical benchmark receipt #10314 / #10370; qualified task judges #11772; comparative-claim statistics #10619 / #10624; workload owners #10543, #10771 and Symtropy #3776.

## 1. The missing comparison question

The right competitive unit is not tokens per dollar, model price, benchmark score, or a demo's time-to-first-answer. It is:

> What is the fully burdened cost of producing one independently accepted, evidence-qualified outcome for a fixed eligible workload, while meeting every declared safety, authority, correctness, provenance, service and physical constraint?

The cost boundary must follow the workflow from input to accepted outcome. It includes work that is easy to hide in a model-only comparison: tool calls, retries, sandboxing, independent verification, human review, exception handling, integration, maintenance, rework and allocated infrastructure.

This crosswalk supplies a common interpretation of cost fields for #11774. It does **not** add a second benchmark receipt, task denominator, invoice ledger, judge, evidence service or cost calculator. The run receipt remains authoritative for the task universe, arm identity, budget and actual resource-use reconciliation.

## 2. Three accounting lenses — never mix them

| Lens | What it measures | Reporting rule |
|---|---|---|
| **Benchmark campaign** | Arm execution, benchmark setup, gold annotation, independent judging and study-only instrumentation | Report separately from production workflow cost. Include only recurring evaluation work in deployed TCO. |
| **Workflow delivery** | Model/provider, orchestration, tools, compute, isolation, verification, human operation, retries and rework | Charge the entire eligible cohort, not only successful attempts. |
| **Deployed lifecycle** | Workflow delivery plus integration, migration, licenses, support, maintenance and allocated fixed infrastructure | Declare the horizon, workload volume, utilization and amortization method; show sensitivity scenarios. |

A research study may incur expensive independent gold annotation that does not recur in deployment. Conversely, a production system may require ongoing human exception handling, support and security review that a benchmark leaderboard does not count. Keep those boundaries visible.

## 3. Full-cost dimensions

| Dimension | Include | Evidence to retain | Double-counting guard |
|---|---|---|---|
| Model/provider inference | Billed input/output/reasoning/cache and multimodal units; embeddings; provider retries | Invoice, usage export, dated regional rate card, request-level usage | Do not add estimated provider GPU cost to the same hosted inference invoice. |
| Agent runtime/orchestration | Workflow runtime, gateway, queues, state, traces, audit and managed orchestration | Invoice, usage export, runtime meter | Assign bundled service charges once. |
| Tools/external services | Code hosting, search, CAD/CAE/PLM, databases, data acquisition and external APIs | Tool/API usage and invoice | Do not split a bundled platform charge into a second invented fee. |
| Compute/storage/network | CPU/GPU/accelerator time, memory, storage, egress and dedicated runner capacity | Cloud bill, resource meter, hardware utilization | For a shared asset, use either invoice allocation or owned-asset allocation for the same interval. |
| Isolation/verification | Sandbox, independent observer, artifact reconstruction, proof verification and evidence retention | Runner/service bill, verifier record, storage meter | If verifier compute is already in compute cost, record verifier utilization diagnostically but charge it once. |
| Human operation/governance | Operator, exception handler, reviewer, approver, security/privacy review and on-call support | Role-specific time logs and disclosed loaded labor-rate basis | Do not count the same time in task logs and generic overhead. |
| Failure/recovery/rework | Failed attempts, retries, rollback, duplicate-effect cleanup, manual correction and revalidation | Canonical dispositions, incident/recovery records, human time and invoices | Retry spend already present in usage totals is not added a second time. |
| Integration/maintenance/migration | Connectors, schema mapping, security review, upgrades, provider migration and training | Project records, work items, contracts and invoices | Show one-time and recurring costs separately; disclose the amortization denominator. |
| Licenses/support/commitments | Seats, platform minimums, reserved capacity, support tiers and residency add-ons | Contract, invoice, dated price sheet and utilization | Report unused commitment; do not count a bundled license twice. |
| Physical resource vector | Electricity, cooling energy, water, materials and measured carbon | Metered kWh/litres, provider data and source-qualified lifecycle factors | Keep physical units separate from money; do not rebill energy already inside a facility/cloud invoice. |
| Independent evaluation | Gold annotation, blind adjudication, external judge fees and study-only instrumentation | Adjudicator time, evaluator invoice and calibration record | Report as campaign cost unless it is a recurring production activity. |

The machine-readable companion enumerates the same dimensions, their evidence sources, monetary treatment and double-counting rules.

## 4. Monetary cost is not the whole resource picture

Publish at least two monetary views:

1. **Cash outlay:** actual invoices, paid labor and contractual charges over the declared period.
2. **Fully burdened economic cost:** cash outlay plus disclosed allocations for internal labor, fixed infrastructure, integration, maintenance and failure/rework.

Also publish non-monetary resource vectors: kWh, water volume, CPU/GPU time, human minutes, latency and (where properly sourced) CO2e. Do not quietly monetize environmental externalities or convert physical resource use into a cost number without publishing the tariff or valuation method.

For every cost amount, preserve currency, tax treatment, region, billing period and observation date. Report net invoiced cost and gross/list-price equivalent separately when discounts, credits, free tiers or negotiated commitments apply. A one-off credit must not make an arm appear structurally cheaper. If values are normalized across currencies, preserve the original currency amount, FX source, timestamp and conversion method.

## 5. Evidence quality and missing cost

Every cost observation must be identified as one of:

- **Invoiced/contractual:** bill or contract with date, region, currency and billing boundary.
- **Metered:** direct resource observation with a named meter, period and source metadata.
- **Provider-reported:** provider usage export without independent meter access.
- **Measured human time:** role-specific observed minutes and the time-capture method.
- **Estimated allocation:** an explicit formula, source inputs, horizon and uncertainty range.
- **Unknown:** unavailable or unmeasured.

Unknown is never zero. If a material cost category is unknown, publish the lower bound and the missing category; do not make a fully burdened cost-superiority claim unless the uncertainty can be bounded tightly enough to preserve the conclusion. Estimates must remain distinguishable from invoices and meters.

## 6. Primary economic metrics

Use the canonical receipt's eligible denominator and the independently qualified judge's accepted-outcome labels. Do not create a new denominator or redefine refusal, abstention, unsupported, timeout, tool failure, ungraded or not-attempted states.

Let the complete cohort cost include every cost incurred for the frozen eligible set, including failures and nonproductive attempts.

- **Fully burdened cost per eligible task** = complete cohort workflow-delivery cost / eligible task count.
- **Fully burdened cost per verified success** = complete cohort workflow-delivery cost / independently accepted outcomes.
- **Deployed lifecycle cost per verified success** = deployed-lifecycle cost over the declared horizon / accepted outcomes over that same horizon.
- **Human minutes per verified success** = all operational and review minutes / accepted outcomes, with role breakdowns.
- **Resource per verified success** = each measured physical quantity / accepted outcomes.
- **Matched cost delta** = candidate cohort cost minus control cohort cost over the same frozen eligible task set.

If an arm has zero accepted outcomes, cost per verified success is undefined; retain its costs and report zero accepted outcomes. Never make the failures disappear by excluding them from cost. Report mean, median, p50/p95 tail behavior and paired task-level distributions where sample size supports them.

## 7. Comparison discipline

Keep two comparison modes separate:

**Same-model matched-harness ablation.** Hold model, prompt, tools, task, input, context, budget, runner, grader and authority fixed; change one system component. This estimates the causal contribution of the component.

**Best-available-product comparison.** Let each eligible vendor use its strongest documented configuration. Pin model/product revision, region, contract and tool stack; report the differences rather than attributing the whole result to one model.

Both modes require paired eligible tasks, full task accounting, all-attempt cost, independent outcome adjudication and cluster-aware uncertainty. Price snapshots must be source-pinned by date and region. Unavailable or ineligible vendor arms remain visible with a reason; do not quietly remove them from a comparison narrative.

## 8. Non-compensatory gates precede cost ranking

A lower cost is not an improvement when the arm achieves it by missing critical consequences, accepting stale evidence, weakening authority, omitting verification, violating safety or failing the declared service requirement.

Cost or speed superiority may be claimed only after the candidate is non-inferior on task correctness and every hard constraint under a preregistered margin. Critical failures are vetoes, not negative points in a weighted score. After the gates pass, report a Pareto vector over correctness, evidence completeness, security, cost, latency, human effort, energy and resource use. No universal weighted score across software, manufacturing, robotics and datacentre workloads is defined.

The existing #10619 / #10624 statistical rules own the claim class, paired inference, cluster structure, power and Pareto reporting. This crosswalk does not weaken or replace them.

## 9. Synthetic arithmetic example — not a result

Suppose a fictional 20-task cohort incurs USD 160 in provider charges, USD 20 in other directly metered compute, USD 140 in measured human work, USD 50 in rework and USD 30 in allocated integration cost. Its full workflow-delivery cost is USD 400. If the independent judge accepts 12 outcomes, cost per verified success is USD 33.33; cost per eligible task is USD 20.00.

The two non-attempted tasks and six non-accepted outcomes remain in the cohort. The example is arithmetic only: it does not describe a Luminous run, current vendor pricing or a measured productivity result. Independent benchmark adjudication cost would be reported separately as campaign cost unless it is part of the deployed service.

## 10. Research targets and claim ceilings

The parent comparison's prospective target remains at least 10% lower cost per independently verified successful task **or** at least 10% lower human review time, but only after task correctness and hard constraints are non-inferior and the study is adequately powered. This is a hypothesis, not an achieved result or universal threshold.

Before any economic claim:
- all eligible tasks and all terminal dispositions reconcile to the canonical receipt;
- all material cost categories are sourced or bounded, with unknowns explicit;
- cost allocations and fixed-cost horizons are frozen before unblinding;
- human time and physical resource measurements are not inferred from model usage;
- the judge is independent and qualified under #11772;
- comparisons and statistical claims follow #10619 / #10624;
- hard safety, security, authority, evidence and service gates pass.

This crosswalk does not prove cost savings, ROI, capability superiority, operational security, physical performance or vendor interoperability. Those require measured matched workflows and independent evidence on exact versions and workloads.

## 11. Runner-independent work completed in this tranche

- Defined three non-interchangeable accounting lenses.
- Mapped eleven cost dimensions to source evidence and double-counting rules.
- Distinguished invoiced, metered, provider-reported, human-measured, estimated and unknown cost.
- Defined cohort-level metrics that keep unsuccessful attempts in the denominator and numerator cost.
- Reused the canonical receipt, judge, statistics and domain owners; no new receipt or ledger was introduced.
- Added a machine-readable companion and linked both artefacts in the research index.

**Validation boundary:** static JSON parsing and structural crosswalk checks only. No repository-controlled code, candidate workflow, vendor account, production data, benchmark candidate, invoice, meter, human time study or physical system was executed or measured.
