# Engineering Lifecycle / PLM Competitor Register — 2026-11-13

**Status:** public-source capability map; no vendor system was accessed or benchmarked.  
**Parent comparison:** [INDUSTRY-STACK-COMP-001 #11774](https://github.com/Luminous-Dynamics/symthaea/issues/11774)  
**Benchmark owner:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Gold protocol:** [#10581](https://github.com/Luminous-Dynamics/symthaea/issues/10581)

## Executive finding

The competitive baseline is stronger than a generic “PLM plus manual review” comparison.

As of November 13, 2026, public product documentation describes:
- Siemens Teamcenter 2606 with an AI BOM agent, AI-assisted MBSE, requirement checks and test-case recommendations.
- Aras InnovatorEdge Impact Analysis AI Agent v1.0.1, a generally available five-specialist recommendation workflow grounded in product structure and historical change orders.
- PTC Windchill change impact and upstream/downstream collection of local and external objects.
- IBM DOORS Next requirements traceability across requirements, development artifacts and test artifacts.

Therefore, **AI-assisted impact analysis, multiple specialist agents, human-in-the-loop recommendations and traceable change workflows are not sufficient differentiation by themselves.**

The stronger research question is whether Luminous can independently qualify *which claim–evidence pairs remain applicable*, determine what must be reviewed or revalidated, detect stale evidence, preserve exact provenance, and do so at lower fully burdened cost without missing critical engineering consequences.

## 1. Capability comparison

| Platform | Publicly documented capability | Benchmark role | Important caveat |
|---|---|---|---|
| **Siemens Teamcenter 2606 / Teamcenter X** | AI BOM agent for guided multi-step changes with human oversight; AI-assisted MBSE requirement checks and test recommendations; rules-based change routing; impact analysis across products, requirements and connected information. | Conditional direct PLM/MBSE vendor arm. | Exact edition, deployment, licensing, integrations, model/agent revision and supported task scope must be pinned. |
| **Aras Innovator UCM + Impact Analysis AI Agent** | GA v1.0.0 announced March 23, 2026; v1.0.1 maintenance update August 18, 2026. Five AI specialists inspect change-request context, product structure and historical change orders; a recommendation validator consolidates the set; users review and apply affected-item relationships. | Highest-priority direct AI change-impact comparator. | Needs substantive CR fields, complete product structure and valid item numbers. Applying recommendations writes affected-item relationships, but the CR remains Draft and normal approval continues. Five AI specialists are not independent human reviewers. |
| **PTC Windchill 13.1.2** | Change-impact analysis can collect affected local and external objects, including upstream/downstream linked objects and remote requirements. | Conditional conventional PLM / traceability vendor arm. | Pin exact modules and integrations. Collected objects are not automatically independent claim-evidence applicability judgments. |
| **IBM DOORS Next 7.2** | Requirements traceability through links, tags, attributes, workflows and dashboards; traceability connects requirements to requirements, development and test artifacts. | Requirements-management adjunct, paired with a named PLM system if available. | Not a stand-alone full PLM change-execution comparator. Do not compare it as if it owned BOM, CAD, manufacturing and physical evidence. |

## 2. Source-qualified capability notes

### Siemens Teamcenter

Siemens' June 12, 2026 Teamcenter 2606 material describes:
- an AI BOM agent for guided, agentic BOM change workflows;
- AI-powered MBSE within Teamcenter Copilot;
- configurable INCOSE-based requirement checks;
- parameter linking and AI-recommended test cases / coverage;
- AI recommendations and rules-based change workflow routing;
- faster digital-thread visualization and large-scale impact analysis.

Its August 11, 2026 Teamcenter X workflow guide describes connected change requests, impacted products, supporting documentation, approvals and release. This is a serious baseline, not a straw-man workflow.

Sources:
- https://blogs.sw.siemens.com/teamcenter/new-in-teamcenter/
- https://blogs.sw.siemens.com/teamcenter/teamcenter-2606/
- https://blogs.sw.siemens.com/teamcenter/engineering-change-workflow/

### Aras InnovatorEdge

Aras' release notes state that Impact Analysis AI Agent v1.0.0 became generally available on March 23, 2026; v1.0.1 was released August 18, 2026. The user guide describes:
- five parallel specialists: Change Management Expert, Part Model Checker, Rationale Analyzer, Prior Change Order Checker and ARAS Expert;
- a final Recommendation Validator;
- recommendations with affected item, suggested action and rationale;
- user review and selective application;
- an operation that writes selected affected-item relationships in one operation;
- a change request that remains in Draft after application.

Important documented prerequisites include substantive change-request title/reason, correct product-structure links, unique item numbers for custom types and Draft-state selection. The agent is a direct benchmark target, not a hypothetical future feature.
The user guide additionally exposes an optional free-text **Analysis Instructions** field for session-level focus/constraining. Its documented account prerequisites include read/update access to Change Requests and read access to Affected Items (Parts, CADs and Documents). The agent is user-invoked: recommendations are not written until the user selects Apply Changes; the Change Request remains Draft afterward. The actual effective principal, permissions, selected items and resulting state must be recorded in a trial.

A key research distinction: the five specialist agents and final validator are components of one product workflow. Their count is not evidence of independent adjudication, operational independence or separate failure domains.

Sources:
- https://docs.aras.com/aras-innovatoredge-ai-book/aras-innovatoredge-ai-agents-release-notes
- https://docs.aras.com/aras-innovatoredge-ai-book/impact-analysis-ai-agent-user-guide
- https://docs.aras.com/unified-change-management-beta

### PTC Windchill

Windchill 13.1.2 documentation describes change impact collection for local and external objects, including upstream and downstream linked objects. This is a meaningful baseline for deterministic relationship traversal and integration with remote requirements.

Sources:
- https://support.ptc.com/help/windchill/r13.1.2.0/en/Windchill_Help_Center/reqmgmt/RMChangeImpact.html
- https://support.ptc.com/help/windchill/r2027.0.0.0/en/Windchill_Help_Center/reqmgmt/RMTraceabilityAbout.html
- https://www.ptc.com/community/windchill/8/tutorials/codebase/wt/helpfiles/help_en/tutorials/pdm/R7PDMChProc/index.htm

### IBM DOORS Next

IBM DOORS Next 7.2 documentation describes requirement traceability using links, tags, attributes, filters, workflows and dashboards, connecting requirements to related requirements, development artifacts and test artifacts. It should be assessed as the requirements-management portion of a combined engineering lifecycle stack, not as a complete PLM system.

Source:
- https://www.ibm.com/docs/en/engineering-lifecycle-management-suite/doors-next/7.2.0?topic=requirements-managing

## 3. What Luminous must prove to differentiate

The commercial products above already own strong systems of record, product structures, workflows, approvals, impact lists and audit trails. Luminous should not attempt to replace them.

The testable differentiation hypothesis is narrower:

1. **Claim–evidence granularity.** Decide for each claim/evidence pair, not just each affected item, whether it is preserved by identity, preserved by a scoped qualified-equivalence witness, preserved by exact recalculation, requires applicability review, requires verification, requires physical evidence, requires external approval, not preserved, unknown or unsupported.
2. **Both sides of the error.** Measure critical under-impact and unnecessary over-requalification separately. A system that requalifies everything should not receive a good score merely for high recall; a system that preserves everything should fail critical controls.
3. **Evidence reconstruction.** Every decision cites the exact source/target generation, dependency path, evidence identity, validity window and preservation witness. A generated rationale is not itself a witness.
4. **Uncertainty is a valid result.** Missing product structure, non-unique identifiers, unknown dependency edges, stale configuration or unqualified equivalence must produce Unknown/NeedsReview/Unsupported—not a guessed preservation decision.
5. **Authority remains external.** A benchmark or AI recommendation cannot approve a change, mint ETK evidence, set evidence currentness, sign a qualification, release manufacturing or actuate equipment.
6. **Cross-vendor operation.** Retain semantics, provenance and evidence references when the system of record is Siemens, Aras, PTC, IBM or an interoperable combination.
7. **Whole-cost efficiency.** Report human review, integration, model/provider, runtime, verification, rework and lifecycle allocation per independently accepted assessment.

These are hypotheses to test against vendor workflows, not claims that the vendors cannot provide them. Public documents do not establish the complete internal capability boundary of each configured product.

## 4. Recommended benchmark arm design

Compose [#10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543) and its existing 24-case corpus; do not create another task universe.

| Arm | Definition | Purpose |
|---|---|---|
| B0 | Full requalification of all declared claim/evidence pairs after every change | Conservative high-recall / overwork reference |
| B1 | Naive byte/field-diff matching | Minimal automation baseline |
| B2 | Static untyped dependency-graph traversal | Conventional relationship-closure baseline |
| B3 | Qualified semantic delta plus impact-path projection | Isolates semantic-change contribution |
| B4 | Full Luminous proposal/applicability composition using existing SE-001, ENG-DESIGN and ETK owners | Candidate system |
| B5 | Qualified human systems-engineering review | Human accuracy/time/disagreement reference |
| V1 | Siemens Teamcenter 2606 / Teamcenter X native workflow | Conditional vendor-native arm |
| V2 | Aras Innovator UCM + Impact Analysis AI Agent v1.0.1 | Priority conditional vendor-native arm |
| V3 | PTC Windchill 13.1.2 change impact with configured external links | Conditional vendor-native arm |
| V4 | IBM DOORS Next 7.2 paired with a named PLM system | Requirements-traceability adjunct, not a standalone PLM arm |

A vendor arm is eligible only when the actual licensed system, product revision, integrations, configured data model and workflow are available and pinned. Public feature descriptions alone are not benchmark results. If access is unavailable, retain a documented capability-map entry and do not assign a score.

## 5. Evidence and statistical discipline

- Use #10581's evaluator-only claim–evidence pair contract and dual independent assessor / third-party adjudication process.
- Keep candidate-visible inputs separate from gold labels, expected dispositions and evaluator-only mutations.
- Preserve all 24 blueprints as an authoring/pilot corpus; it is too small and clustered to support broad superiority claims.
- Split any later holdout by design family and change lineage, not random rows or repeated seeds.
- Report per-disposition confusion matrices, critical false negatives, stale-evidence false-current errors, over-requalification, uncertainty/abstention, reviewer time and total cost.
- Select one primary confirmatory endpoint after pilot variance and assessor disagreement are known; preregister the primary contrast and practical margin before unblinding.
- Use #11772 for judge qualification; #10619/#10624 for paired, cluster-aware inference and claim class; #10314/#10370 for canonical task/run accounting.
- Do not execute untrusted candidate code until #10064's isolation boundary is qualified. Do not claim strong signed independent-runner evidence until #10369's trust-root and runner independence are qualified.

## 6. Status

This register is based on public vendor documentation only. No Siemens, Aras, PTC or IBM system was accessed, configured, licensed, integrated, run or scored. No vendor price, implementation effort or comparative result is asserted.

The 24-case corpus remains unadjudicated and unscored. The next valid milestone is an independently adjudicated pilot and an instrument-qualified, matched baseline comparison—not a vendor leaderboard assembled from documentation.


## 7. Aras platform update — InnovatorEdge AI 2.4 (2026-11-10)

The competitor register now distinguishes the **Impact Analysis AI Agent v1.0.1** from the wider **InnovatorEdge AI platform 2.4** release announced on November 10, 2026. The platform update documents multiple MCP Server profiles per tenant; each profile has a distinct endpoint, Access Point, published-tool selection and enabled/disabled state. The administrator guide documents End User authentication for the Edge AI MCP Server, a cached connector tool inventory in Claude that may require manual refresh, and a beta Innovator Method tool with per-call user confirmation.

This does not mean the Impact Analysis agent is version 2.4, and it is not evidence of a security defect. For a licensed comparison, record the agent release, platform release, MCP profile configuration, endpoint, Access Point, tool-set digest, enablement state, authentication mode, CIAM roles and effective Innovator permissions separately. Treat cached tool inventory as client discovery state—not authority. The mock security overlay now tests stale cached tools after disable/removal, cross-profile tool substitution, permission/Access Point drift and method invocation without the required confirmation.

Sources:
- [Aras Edge AI Release Notes](https://docs.aras.com/aras-innovatoredge-ai-book/edge-ai-release-notes)
- [Aras InnovatorEdge AI Administrator Guide](https://docs.aras.com/aras-innovatoredge-ai-book/innovatoredge-ai-administrator-guide)

## 8. Implication for the SOTA comparison

The benchmark must separate **capability discovery**, **recommendation quality**, **authority at effect entry** and **observed effect**. Commercial platforms already provide AI-assisted impact analysis, governed write operations, configurable tools and human confirmation. Therefore an assistant that only adds another approval screen or MCP integration is not differentiated. The proposed Luminous advantage must be demonstrated through independently replayable decision evidence, non-anticipating source/version admissibility, claim-level evidence applicability, and an independent reconstruction of the exact effect boundary. The Aras MCP profile case is a security/authority control, not an additional engineering-change accuracy score.
