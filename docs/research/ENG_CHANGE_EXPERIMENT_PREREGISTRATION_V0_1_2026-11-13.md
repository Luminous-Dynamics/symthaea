# Engineering Change Experiment Preregistration V0.1 — 2026-11-13

**Status:** preregistration design only; no candidate runs, no gold labels and no performance results.  
**Benchmark owner:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Public blueprint corpus:** [manifest PR #10547](https://github.com/Luminous-Dynamics/symthaea/pull/10547), head `3f658a9f74b750667cfe9642a2e881afa993fca1`  
**Gold adjudication contract:** [#10581](https://github.com/Luminous-Dynamics/symthaea/issues/10581)  
**Industry capability register:** [ENGINEERING_LIFECYCLE_COMPETITOR_REGISTER_2026-11-13.md](ENGINEERING_LIFECYCLE_COMPETITOR_REGISTER_2026-11-13.md)

## 1. Research question

For a frozen engineering baseline, a proposed semantic design/configuration change, and a set of previously admitted evidence, can a system correctly determine:
- which claim–evidence pairs remain applicable;
- which require applicability review or exact recalculation;
- which require new verification, validation or physical evidence;
- which require professional/external approval;
- which are not preserved, unknown or unsupported;

while reducing unnecessary requalification **without missing critical impact or declaring stale evidence current**?

The benchmark evaluates proposed assessments. It does not issue engineering approval, update ETK authority/currentness, release a product, authorize manufacturing or actuate equipment.

## 2. Task and scoring unit

Use the existing 24-case blueprint registry and its six named change families. The scoring unit is the existing #10581 **claim–evidence applicability pair**, not a newly invented task row or generic impact-graph node.

Every pair binds:
- exact source and target design/configuration generation;
- exact change/transformation identity;
- claim identity and criticality;
- prior evidence identity, class, source and validity;
- dependency contract and exact dependency paths;
- one of the ten existing dispositions;
- required follow-up obligation(s);
- independent assessors and adjudication state.

No new semantic graph, delta model, evidence-currentness owner, lifecycle authority, gold-record schema or generic benchmark receipt is introduced.

## 3. Trial stages

### Stage A — instrument and corpus preparation

1. Pin the 24-blueprint manifest, schema, #10581 gold-record schema, evaluator and all owner contracts by exact commit and digest.
2. Materialize exact before/after design/configuration snapshots, evidence manifests and dependency projections only under the existing domain owners.
3. Confirm all task fields needed by the adjudicators are available; otherwise mark the pair unsupported/unknown rather than filling gaps by inference.
4. Define candidate-visible/evaluator-only partitions and access logs before creating any gold record.
5. Freeze the assessor rubric, disagreement policy, criticality rules, exclusion reasons and evidence sufficiency criteria.

### Stage B — independent gold adjudication

1. At least two qualified domain assessors independently label high-consequence claim–evidence pairs while blinded to candidate systems, treatment arms and each other's labels.
2. Record assessor qualification, domain expertise, conflicts/authorship, rubric revision, evidence manifest and access history.
3. Resolve disagreement through the documented third-party adjudicator procedure in #10581; preserve both original labels and the resolution.
4. Keep UnknownImpact and Unsupported where dependencies or evidence do not justify a stronger answer.
5. Seal gold records using the exact #10581 evaluator contract. A schema-valid record is not proof of assessor independence by itself.

### Stage C — 24-case pilot / variance estimation

Use the public development corpus only for a feasibility pilot, rubric and evaluator debugging, variance estimation, workflow timing and cost measurement. Do not describe it as a confirmatory superiority study.

### Stage D — design-family held-out evaluation

Create a later holdout from design families and change lineages not used in candidate development, prompt tuning or rubric calibration. Near-duplicate changes and multiple mutations of one base design remain in the same partition. The holdout must be independently reviewed and sealed before candidate runs.

### Stage E — confirmatory comparison

Determine sample size through a preregistered power analysis using the pilot's paired discordance, design-family/lineage cluster variance, invalidity and attrition. Select one primary endpoint, one primary contrast and one practical margin before exposing the held-out set. Do not infer superiority from the 24-case pilot.

## 4. Comparison arms

### Reference and ablation arms

- **B0 — Full requalification:** every eligible claim–evidence pair is sent to the most conservative required verification/review path after every change. High-recall / overwork reference.
- **B1 — Byte/field diff:** treat changed bytes or fields as affected and unchanged fields as preserved. Deliberately weak baseline; must not receive semantic-equivalence credit.
- **B2 — Static dependency closure:** traverse known relationship edges without qualified semantic-delta or authority distinctions.
- **B3 — Semantic delta projection:** use the existing qualified delta and dependency owners to identify candidate impact paths, without the full evidence-applicability reasoning layer.
- **B4 — Full Luminous composition:** Symthaea proposals + existing SE-001 semantic projections + ENG-DESIGN delta contracts + ETK evidence/applicability owner. ETK remains the authority; benchmark output is proposal/evaluation only.
- **B5 — Qualified human workflow:** domain-qualified systems-engineering reviewers apply the same evidence manifest and rubric; report time, inter-rater disagreement and adjudication separately.

### Conditional vendor-native arms

- **V1 — Siemens Teamcenter 2606 / Teamcenter X:** include AI BOM agent, Copilot MBSE and native change workflow only where the exact edition, deployment, model and enabled features can be pinned.
- **V2 — Aras Innovator UCM + Impact Analysis AI Agent v1.0.1:** highest-priority direct commercial comparator where a licensed representative environment can be provisioned.
  - Eligibility requires a real pinned Aras Innovator/UCM/InnovatorEdge AI environment and the actual supported agent release; public documentation alone is not a score.
  - Prepare a Draft-state Change Request with substantive title, Reason of Change and Description, complete product-structure links, and stable unique `item_number` values for custom item types.
  - Record the authenticated principal and its configured access: read/update Change Requests and read Affected Items (Parts, CADs, Documents). Record effective permissions at analysis and apply time.
  - The optional session-level Analysis Instructions field is part of the candidate-visible instruction channel. Give all arms equivalent task scope; test both legitimate narrowing and hostile expansion attempts. Free text is not approval or a permission grant.
  - The human must select the relationships to Apply. Capture the exact selected set, observed writes and resulting lifecycle state; the guide says the CR remains Draft. Do not assume permission enforcement or atomicity beyond what is actually observed.
- **V3 — PTC Windchill 13.1.2:** include change-impact collection and configured upstream/downstream external links; state whether requirements integration is active.
- **V4 — IBM DOORS Next 7.2:** requirements-traceability adjunct only, paired with a declared PLM system. Do not rank it as a complete PLM workflow.

Vendor documentation alone does not make an arm eligible. If an actual configured system cannot be accessed, report a source-qualified capability map, not a score. All non-candidate systems receive the same source snapshots, task scope, permitted semantic projections and evaluation rubric to the extent each system can support them; any unavoidable interface/data-model mismatch is recorded and can make the arm ineligible for the matched contrast.

## 5. Outcome metrics and non-compensatory gates

### Hard gates

1. Critical impact recall: no seeded critical-impact miss in the frozen hard-gate suite.
2. Stale-evidence false-current: zero in the frozen critical controls.
3. Preservation-witness integrity: no accepted witness with wrong source/target identity, scope, dependency path or validity.
4. Evidence-state non-amplification: historical, simulation, numerical, measured, physical and approval evidence cannot be silently promoted into a stronger class.
5. Authority non-mutation: no benchmark or model output changes ETK authority/currentness, approval, manufacturing release or actuation.
6. Evaluator integrity: no omitted task/pair, mutated gold record or unaccounted terminal disposition.

Any hard-gate failure blocks an efficiency/superiority claim; it is not a negative score that can be offset by cost or time.

### Accuracy and decision-quality metrics

- Per-disposition confusion matrix for all ten dispositions.
- Claim–evidence impact precision/recall by change family and criticality.
- Critical under-impact count, reported separately.
- Unnecessary requalification / over-impact count.
- Valid scoped-equivalence and exact-recalculation witness rate.
- Unknown/unsupported detection and correct clarification/abstention.
- Unsupported relation invention and false preservation-witness rates.
- Historical evidence identity preservation and provenance completeness.

### Operational and economic metrics

- Engineering reviewer minutes by role.
- Time to independently accepted impact/applicability set.
- Correctly avoided re-verification/re-validation obligations.
- Cost per independently accepted change-impact assessment.
- Total workflow cost and human review/rework using [Symthaea #11859's cost-accounting crosswalk](https://github.com/Luminous-Dynamics/symthaea/pull/11859).
- Integration, mapping, configuration and deployment effort reported separately from recurring workflow cost.
- p50/p95 latency only where per-condition sample sizes support it.
- All material unknown costs remain unknown; no zero imputation.

## 6. Blinding and evaluator independence

- The candidate cannot judge itself and cannot receive evaluator-only records or hidden mutation controls.
- Assessors do not receive treatment-arm identity, agent rationale, vendor branding or candidate confidence unless a field is explicitly required by the frozen rubric.
- Normalize output format before adjudication while preserving the evidence references necessary to judge a claim.
- Log all access to candidate-visible and evaluator-only material.
- Report blinding failures, rater disagreement, re-review, escalation and adjudication cost.
- Use deterministic oracle checks for exact identities, digests, witness scope and enumerated constraints; human domain adjudication remains necessary for engineering semantics not reducible to a deterministic oracle.

## 7. Statistical reporting

The 24-case public set is a pilot and not a confirmatory sample. Do not inflate the denominator with repeated seeds or multiple mutations of the same design lineage.

For the held-out study:
- preserve paired per-task / per-claim-pair outcomes;
- identify design-family and lineage clusters;
- estimate sample size through power analysis using pilot disagreement and clustering;
- preregister one primary contrast and endpoint;
- apply the claim qualification and multiplicity rules in #10619/#10624;
- report effect sizes, confidence intervals, discordant-pair counts and cluster-aware sensitivity;
- use a Pareto vector for correctness, hard-gate status, cost, human effort, latency and evidence completeness.

A result is descriptive/exploratory until the #10619/#10624 requirements for matched comparison, non-inferiority, superiority or Pareto dominance are met.

## 7A. Zero-event miss-rate precision: hard-gate planning aid

This section quantifies what a zero-miss result would and would not establish. It is **prospective sample-size guidance, not a result**.

For (n) independent critical cases with zero observed misses, the exact one-sided 95% binomial upper confidence bound on the underlying miss probability is:

[
p_{upper}=1-0.05^{1/n}
]

| Independent critical cases | Observed misses | One-sided 95% upper bound on miss probability |
|---:|---:|---:|
| 24 | 0 | 11.735% |
| 60 | 0 | 4.870% |
| 100 | 0 | 2.951% |
| 299 | 0 | 0.997% |
| 300 | 0 | 0.994% |
| 600 | 0 | 0.498% |

Implications:

- Even if the 24-case pilot eventually records zero critical misses, its one-sided 95% upper bound is about **11.7%** under the independent-binomial assumption. That is nowhere near evidence for a low miss probability; the pilot remains a feasibility, rubric, variance and cost exercise.
- To place the one-sided 95% upper bound below **1%** after observing zero misses requires at least **299 independent critical cases**. This is a high-assurance planning target, not an industry standard, certification threshold or claim that 299 correlated mutations are sufficient.
- Cases sharing the same design family, source artifact, base configuration or change lineage are not automatically independent. Report the nominal case count and the number/structure of independent design-family and lineage clusters. The exact binomial bound above is valid only under its independence and fixed-sampling assumptions; the confirmatory analysis must also report cluster-aware uncertainty and sensitivity.
- If one or more misses occur, do not use the zero-event table. Report the observed numerator and denominator with an exact binomial interval and cluster-aware analysis.
- Do not collapse all criticality strata into a single denominator if the sampling design or risk profile differs materially. Report critical-impact families and high-severity classes separately, including missing/unsupported cases and all eligible denominators.

The programme should retain the current hard gate of **zero critical misses in the frozen hard-gate suite**. This precision calculation adds a second question: how much statistical evidence supports a low underlying miss rate? A hard-gate pass on a small curated suite is not the same as a statistically bounded low-risk claim.

## 7B. Decision-time evidence admissibility overlay

The benchmark now composes a separate decision-time evidence-admissibility overlay:
- [Decision-time evidence admissibility protocol](ENGINEERING_CHANGE_DECISION_TIME_ADMISSIBILITY_OVERLAY_V0_1_2026-11-13.md)
- [Machine-readable temporal challenge blueprints](ENGINEERING_CHANGE_DECISION_TIME_OVERLAY_V0_1.json)
- Existing lifecycle/as-of owner: [KNOW-LIFE-001 #10600](https://github.com/Luminous-Dynamics/symthaea/issues/10600)

The overlay separates world-valid-at T from available-to-the-decision-process-by K, and distinguishes source publication, canonical receipt, acquisition/ingestion, derived-index visibility and actual run-context inclusion. It adds nine public challenge blueprints and six non-compensatory temporal gates. These are overlay controls on existing #10543/#10581 claim–evidence pairs, not nine adjudicated or statistically independent new benchmark cases.

A source's publication date does not prove the agent retrieved it; a service receipt does not prove an index or model incorporated it. If the evidence chain cannot establish the required temporal fact, the result remains indeterminate/unsupported. #11241 is referenced only as finance-scoped bitemporal precedent; Mycelix #5860 remains a DeSci receipt-time projection owner with its stated model/index-availability limitation.

**Gold-record compatibility is a blocking prerequisite for the temporal sub-axis:** the current #10581 schema is strict and contains no structured T/K, source-availability, downstream-materialization or run-context fields. Its free-text notes are not a substitute. Temporal gold labels must wait for a versioned optional extension or a separately validated content-addressed companion record bound to the existing gold record. The ten existing engineering dispositions remain unchanged; no temporal labels or candidate scores have been created.

## 8. Execution and trust prerequisites

- No untrusted repository-controlled candidates/build scripts may run until [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064) establishes an actual OS isolation boundary. Host execution is not a sandbox.
- Strong signed independent-runner claims remain blocked until [#10369](https://github.com/Luminous-Dynamics/symthaea/issues/10369) qualifies out-of-band trust roots, key custody, failure-domain independence and runner attestation.
- Canonical run accounting remains under #10314/#10370. Judge qualification remains under #11772. Statistical claims remain under #10619/#10624.
- External vendor systems are accessed only via provisioned, licensed test environments and declared integration boundaries; no production customer data or real engineering authority is used in the synthetic pilot.

## 9. Current status and next decision

- 24 public blueprints exist; all remain unadjudicated and unscored.
- #10581 defines the gold-record schema and independent adjudication process; no gold record has been created by this research tranche.
- Commercial competitor features are now source-mapped, but no vendor environment has been run.
- This preregistration introduces no scores, gold labels, sealed holdout, vendor result, execution authorization or new receipt.

**Next gate:** secure two qualified independent assessors and a bounded design-family sample, then adjudicate the pilot labels under #10581 before any candidate output is evaluated.


### Aras platform 2.4 security configuration (not an extra score arm)

The Aras Impact Analysis AI Agent remains v1.0.1; the broader InnovatorEdge AI platform has a separate 2.4 release dated 2026-11-10. Where a licensed V2 Aras trial also exposes the Edge AI MCP Server, pin the platform version and each MCP profile's endpoint, Access Point, enabled state, published tool set, authentication mode and CIAM/Innovator permission snapshot. Record whether the client presents a cached tool list and whether it was refreshed. Test profile disablement, tool removal, Access Point/permission drift, cross-profile tool substitution and the beta Invoke Innovator Method Tool's per-call confirmation. This is a security/authority subprotocol, not an additional engineering-change accuracy arm; no failure is inferred from documentation alone.
