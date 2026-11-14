# Engineering Change Decision Replay Benchmark v0.1

**Status:** preregistration / design specification; no candidate scores or gold labels claimed.  
**Date:** 2026-11-14  
**Parent:** ENG-DESIGN-CHANGE-QUAL-001 gold-record V1.1 and its temporal admissibility mutation suite.

## 1. Thesis: benchmark the decision, not the answer

The proposed benchmark asks whether a system can reconstruct what an engineer was justified in believing and doing at a specific decision frontier. It is stricter than document retrieval, impact prediction, timeline ordering or explanation quality taken alone.

A decision is replayable only when the evaluator can reconstruct:

1. the exact design, configuration and claim under review;
2. the target-world time **T** to which evidence applies;
3. the decision information frontier **K**;
4. the frozen source inventory and source-specific evidence versions;
5. what had actually been published, received, ingested, materialized and included in the run context by K;
6. the applicable rights and authorization state;
7. the dependency/applicability argument and its uncertainty;
8. the approved decision and obligations, without granting the benchmark system release or execution authority.

This creates three separate questions that must not be collapsed:

- **Temporal truth:** Was the evidence valid for the world state at T?
- **Decision admissibility:** Was this exact evidence available, authorized and present in the relevant process by K?
- **Causal/applicability validity:** Does the evidence actually support the claim for the target configuration?

An item may be valid at T but unavailable by K. It may be available by K but invalid at T. It may be both valid and admissible while irrelevant to the claim. All three axes need distinct labels.

## 2. Paradigm shifts

### 2.1 From timestamp cutoff to a replayable information frontier

A cutoff timestamp does not prove the system had the information. The benchmark freezes the source set and distinguishes event occurrence, source publication, issuer availability, repository receipt, ingestion, derived materialization and run-context inclusion. For multi-source decisions it also requires a verified partial-order frontier where applicable.

GS1 EPCIS illustrates why this matters: eventTime is the capturing application's assertion of when an event occurred, while recordTime is repository bookkeeping and does not describe the real-world event. Neither field alone proves when a specific decision process acquired or used the record. See the [official EPCIS ObjectEvent semantics](https://ref.gs1.org/epcis/ObjectEvent) and [EPCIS standard](https://ref.gs1.org/standards/epcis/).

### 2.2 From a plausible timeline to a causal claim ledger

Score separately:
- a timestamp assertion;
- a source-local sequence relation;
- an independently verified cross-source happens-before edge;
- a physical/engineering dependency;
- a causal effect supported by an intervention, mechanism or qualified model.

A model must preserve incomparable events. It must not turn timestamps into causal edges, nor correlations into engineering causation. The 2026 ICLR work on event-level causal disentanglement in temporal knowledge graphs highlights that conventional temporal-graph prediction can leave causal structure unresolved: [HEDRA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cafcbf5d00a961a05bd5875c17ef0fe7-Abstract-Conference.html).

### 2.3 From relevant retrieval to temporally complete retrieval

A system can retrieve relevant documents and still omit the required period, source or correction history. Add evaluator-frozen temporal bins and source roles. Missing periods are a coverage failure, not something precision can hide.

A useful external diagnostic is TEMPO (EMNLP 2026): 1,730 multi-domain queries, 3,976 decomposed retrieval steps and metrics including Temporal Coverage@k and Temporal Precision@k. Its reported best system reached 71.4% Temporal Coverage@10; this is a retrieval benchmark, not an engineering-change result and not an apples-to-apples competitor score. [TEMPO paper](https://aclanthology.org/2026.emnlp-main.291/).

### 2.4 From explanation to independent reconstruction

Require a second implementation or independent assessor to reconstruct the K projection from frozen inputs. Compare evidence membership, source/claim bindings, disposition, obligations and authorization—not prose similarity. Candidate-generated explanations are evidence of what the candidate said, not proof that its reasoning is correct.

### 2.5 From one leaderboard number to hard gates plus a Pareto frontier

First report non-compensatory critical errors. Then show quality, coverage, latency, human workload and fully burdened cost separately. Do not create a weighted composite where faster processing can compensate for a false-safe engineering decision.

## 3. Industry comparison (publicly documented capabilities, 2026-11-14)

These are product-documentation baselines and integration targets. Vendor descriptions establish advertised product capabilities, not independent benchmark performance.

| Platform | Publicly documented capability | Benchmark implication |
|---|---|---|
| Siemens Teamcenter 2606 | AI BOM agent for guided BOM change workflows; BOM understanding, impact analysis and governed updates with human oversight; AI-supported MBSE and change workflow capabilities. [Release overview](https://blogs.sw.siemens.com/teamcenter/teamcenter-2606/) · [BOM agent details](https://blogs.sw.siemens.com/teamcenter/ai-bom-agent-plm-engineering-change/) | Compare impacted-occurrence recall, variant/plant scope, downstream manufacturing propagation, action traceability and user approval. |
| Aras InnovatorEdge AI 2.4 | Multiple MCP profiles per tenant; each profile has its own endpoint, Access Point, published tools and enabled/disabled state. [Release notes, 2026-11-10](https://docs.aras.com/aras-innovatoredge-ai-book/edge-ai-release-notes) | Test exact profile/endpoint/tool-set binding, stale client inventory, scope substitution and action-time permission checks. These are conformance tests, not vulnerability allegations. |
| PTC Windchill 13.1.2 | Change impact analysis collects linked upstream/downstream objects; Windchill AI materials describe natural-language product-data assistance and workflow automation. [Impact-analysis help](https://support.ptc.com/help/windchill/r13.1.2.0/en/Windchill_Help_Center/reqmgmt/RMChangeImpact.html) · [AI product overview](https://www.ptc.com/en/products/windchill/windchill-ai) | Compare linked-object completeness, cross-domain traceability, permission-respecting retrieval and exact change/configuration baselines. |
| IBM Engineering Requirements Management DOORS Next 7.3 | Traceability links requirements to related requirements, implementation and test artifacts; supports coverage analysis and impact analysis. [Traceability documentation](https://www.ibm.com/docs/en/engineering-lifecycle-management-suite/doors-next/7.3.0_beta?topic=requirements-traceability) | Compare requirement-to-test closure, baseline/time-slice fidelity, traceability gaps and change propagation. |

**Where to compete:** do not try to replace the authoritative PLM/ALM system of record. Position the layer as a cross-system decision-assurance and replay layer that can qualify the evidence packet before an existing PLM workflow accepts a proposed disposition or effect. The differentiator is independent evidence admissibility, completeness, causal-frontier correctness and replay—not a prettier BOM chatbot.

## 4. Benchmark tracks

### Track A — temporal evidence admissibility
Can the candidate determine what was valid at T and available by K, with exact source version, digest, rights and run-context membership? Include source publication after K, late ingestion, backdated corrections, retractions, supersession and clock uncertainty.

### Track B — evidence coverage and retrieval
Can the candidate retrieve sufficient evidence across all frozen temporal bins and source roles? Include records that are individually relevant but collectively incomplete, sparse periods, conflicting sources and missing-source negatives.

### Track C — engineering applicability and change impact
Can the candidate map each changed item to claim–evidence obligations without missing critical impact or over-triggering unrelated requalification? Preserve exact variants, occurrences, configuration generations, requirements, dependencies and qualified equivalence scope.

### Track D — distributed order and causal validity
Can the candidate preserve local order, verified cross-source happens-before edges and concurrent/incomparable events? Can it distinguish a causal dependency from mere temporal succession or correlation?

### Track E — replay, authority and effect safety
Can an independent evaluator reconstruct the decision-time packet? Does the candidate abstain when required evidence is incomplete? Does any action recheck principal, scope, profile, endpoint and permissions at the actual effect boundary? Unknown outcomes must not create new mutations.

## 5. Metrics and definitions

All denominators are frozen by the evaluator before candidate execution.

| Metric | Definition | Criticality |
|---|---|---|
| Critical impact recall | Gold critical impacted claim–evidence pairs correctly detected / all gold critical impacted pairs | Hard gate |
| Critical false-current count | Critical stale/inapplicable pairs reported as current | Hard gate; target 0 |
| Temporal Coverage@k | Required evaluator-frozen temporal bins represented by admissible, claim-relevant retrieved evidence / all required bins | Hard minimum |
| Temporal Precision@k | Retrieved evidence items admissible for the requested T and K / all retrieved items | Quality |
| Source inventory coverage | Required source roles represented in the validated frontier / all required roles | Hard minimum |
| Future-knowledge leakage | Decision claims relying on evidence unavailable to the process by K / all supported decision claims | Hard gate; target 0 |
| False causal-edge rate | Asserted cross-source causal-order edges without independent valid causal evidence / all asserted cross-source causal-order edges | Hard gate; target 0 |
| Incomparability preservation | Gold-incomparable event pairs not forced into an unsupported total order / all gold-incomparable pairs | Hard gate |
| Replay fidelity | Exact match of frozen evidence membership, versions, frontier, disposition and obligations on independent reconstruction | Hard gate |
| Obligation precision | Correct mandatory follow-up obligations / all proposed mandatory obligations | Quality |
| Decision calibration | Risk/uncertainty calibration and selective risk at fixed coverage, reported by domain and criticality | Diagnostic |
| Accepted-decision cost | Fully burdened cost (people, compute, tools, review and rework) / independently accepted decisions | Economics |
| Time-to-accepted-decision | Elapsed time from frozen task start to independent acceptance, excluding unqualified waiting states | Economics |

### Proposed acceptance targets (targets, not achieved results)

- Zero critical false-current, future-knowledge leakage, unsupported causal-edge and unauthorized-effect events in the hard-gate suite.
- 100% recall on seeded critical impact cases before any efficiency comparison.
- At least 95% precision on supported impact proposals.
- At least 95% Temporal Coverage@k on frozen, explicitly required temporal bins.
- At least 99% coverage of enumerated source roles; unknown source inventory blocks a completeness claim.
- At least 90% correct preservation classification on identity/equivalence-preserving cases.
- At least 30% reduction in avoidable requalification obligations versus the matched human/native workflow without degrading critical recall.
- At least 10% reduction in fully burdened cost per independently accepted decision in a real pilot.

Treat thresholds as preregistered goals. They must be adjusted only before holdout evaluation, with rationale and versioned protocol; never tune them after seeing the test labels.

## 6. Baselines and matched comparison design

Use these arms on the same frozen task universe, evidence corpus, source permissions, cutoff K, target time T, budget, system scope and adjudication revision:

1. **Human-only:** current team process with recorded effort and independent second review.
2. **Native PLM/ALM:** exact product and version, configuration, access controls, enabled features and integration settings pinned.
3. **Keyword/BM25:** deterministic retrieval with a documented ranking configuration.
4. **Temporal RAG:** a capable temporal retrieval baseline; report its native task metrics and our engineering metrics separately.
5. **Model-only:** same base model without the evidence-frontier/authority layer.
6. **Model + pinned evidence:** retrieval with exact source versions and run manifest.
7. **Symthaea + existing Mycelix owners:** use existing EvidenceFrontierV1, multi-conductor causal-history and time-qualified dependency-completeness services; do not create competing clocks or provenance services.

A platform's unavailable data or inaccessible connector is recorded as not-run/not-comparable, not as a losing score. Vendor trial/tenant access and procurement are prerequisites for a vendor score. Do not fabricate an enterprise baseline from product documentation.

Use paired cases and report per-family results. Split by base design family, supplier lineage and change lineage—not random rows. Keep a sealed holdout whose cases and gold labels are not candidate-visible.

## 7. Ablations that test the actual paradigm

Run one ablation at a time with identical task budgets:

- remove source-set completeness checks;
- replace the verified multi-source partial order with timestamp sorting;
- collapse event time, repository record time and acquisition time into one timestamp;
- omit exact run-manifest inclusion;
- disable rights-at-decision or rights-at-replay checks;
- remove correction/retraction history;
- remove applicability/equivalence witness checks;
- replace independent reconstruction with candidate self-report;
- allow the model to choose its own denominator;
- skip action-time permission/profile checks.

Ablations are diagnostic and intentionally unsafe in a sandbox. They must not connect to production systems or execute real effects.

## 8. Statistical design

A result of zero failures is not evidence of zero risk. Under an idealized independent Bernoulli assumption, 299 independent critical trials with zero misses are needed for a one-sided 95% upper confidence bound below 1%. Repeated mutations of one design family are correlated and do not count as 299 independent trials. Use cluster-aware sampling/power analysis and report confidence intervals by design family and source domain.

Recommended phases:

- **V0.1 synthetic contract:** 30–50 hand-authored temporal decision cases and adversarial mutations; structural/evaluator qualification only.
- **V0.2 dual-assessor pilot:** 100–200 claim–evidence pairs across at least four change families; independent labels and disagreement analysis.
- **V1 confirmatory evaluation:** independently authored, cluster-aware, held-out cases; sample size chosen from target error rate, prevalence and within-family correlation.
- **V1.5 field pilot:** vendor-backed, permissioned read-only integration; shadow decisions, no automated release or production mutation.
- **V2 operational qualification:** only after replay, isolation, action authorization, recovery and professional review gates are separately qualified.

## 9. NIST TEVV alignment

NIST's August 2026 TEVV-Athlon draft proposes a configurable evaluation framework spanning AI systems including agentic systems. This benchmark operationalizes the same test/evaluation/verification/validation orientation for engineering-change decisions: [NIST TEVV-Athlon](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems). It is not NIST certification or endorsement.

## 10. Execution prerequisites and explicit blockers

- Actual JSON Schema Draft 2020-12 validation of V1.1 using the existing generic validator from Symthaea #10314, not a duplicate harness.
- A semantic evaluator that resolves every referenced evidence object and checks chronology, source inventory, causal cut, rights, exact run-manifest binding and assessor independence.
- Two qualified independent assessors plus third-party adjudication where needed.
- Raw-byte, version-pinned vendor/standard corpora and rights-cleared test fixtures.
- Isolated candidate execution, runner trust, artifact custody and independently reconstructable receipts.
- Vendor trials for vendor-specific score claims.
- Engineering-domain and physical tests for claims that cannot be established from digital evidence.

Until these exist, only synthetic structural/predicate checks may be reported. No gold labels, benchmark victory, SOTA claim, safety conclusion or product certification is established by this specification.

## 11. Research references and source notes

- Abdallah et al., **TEMPO: Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval**, EMNLP 2026, DOI 10.18653/2026.emnlp-main.291. Its 71.4% Temporal Coverage@10 is a diagnostic reference, not a directly comparable engineering-change score.
- Chen et al., **Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs**, ICLR 2026. Supports treating causal structure as distinct from temporal correlation.
- NIST, **The TEVV-Athlon Framework for Evaluating AI Systems**, public draft announced 2026-08-07; this work is a domain-specific research alignment, not an endorsement.
- GS1, **EPCIS 2.0** normative resources and ObjectEvent semantics. Pin exact 2.0.0 artifacts and their raw bytes before using them as validator oracles.
- Siemens, **Teamcenter 2606** (2026-06-12); Aras, **InnovatorEdge AI 2.4** (2026-11-10); PTC Windchill 13.1.2; IBM DOORS Next 7.3 documentation. Treat statements as publicly documented features, not performance proofs.

**Claim ceiling:** This is a benchmark design and preregistration. It is not evidence that Symthaea is currently better than Siemens, Aras, PTC, IBM, TEMPO or any SOTA system. Superiority can only be claimed after matched, independently adjudicated, reproducible results.
