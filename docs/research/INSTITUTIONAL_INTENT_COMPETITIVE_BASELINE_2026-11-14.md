# Institutional Intent Compiler — competitive baseline and SOTA protocol

**Research note — 2026-11-14**

Status: desk research and proposed evaluation design. Vendor features are based on public product documentation; no vendor product was installed or benchmarked. No superiority result is claimed.

## 1. Executive finding

The market is not waiting for a requirements lifecycle manager. Mature systems already offer requirements authoring, traceability, workflows, electronic approvals, change control, history, verification/test linking and integrations.

The opportunity is therefore not to reproduce a weaker DOORS Next, Polarion or Jama Connect.

The research hypothesis is narrower:

> Can a source-grounded, authority-aware compiler turn unusually complex institutional briefs into reviewable, versioned requirement candidates and measurable evidence plans, while preserving uncertainty and routing each claim to its existing system owner?

That question spans client intent, governance, evidence, physical assets, privacy and continuity. It is not solved merely by improving requirement sentence quality or generating a test-case draft.

## 2. Standards baseline

As of 14 November 2026:

- ISO/IEC/IEEE 29148:2018 remains the published edition and ISO lists it as current following its 2024 review: https://www.iso.org/standard/72089.html
- ISO/IEC/IEEE DIS 29148:2026, Edition 3, is now a Draft International Standard in the enquiry phase. ISO's record shows the DIS ballot initiated on 14 September 2026; it is not yet a published replacement: https://www.iso.org/standard/94091.html
- NASA's systems engineering handbook provides practical requirements-writing, requirements-validation and requirements-verification-matrix guidance: https://www.nasa.gov/reference/system-engineering-handbook-appendix/

The compiler should map to the published standard and track the DIS changes. It must not claim ISO/IEEE conformance merely because it uses familiar attributes.

## 3. Competitive landscape

| System/category | Publicly documented strengths | Position for Luminous |
|---|---|---|
| IBM Engineering Requirements Management DOORS Next | Requirement attributes, tags, workflow, traceability links, filtering, dashboards, change control and links to other lifecycle applications | Treat as a strong system-of-record / lifecycle baseline; integrate rather than replace |
| Siemens Polarion ALM | LiveDocs paragraph-level identity and traceability, ReqIF exchange, review/approval, electronic signatures, version/time-machine history, impact analysis and requirement-to-test traceability | Treat as a strong lifecycle, configuration and audit baseline |
| Jama Connect | Requirements authoring and live traceability; Jama Connect Advisor feedback on requirement language aligned with INCOSE/EARS; advertised AI-assisted test-case generation and coverage-gap identification | Treat as a strong guided-authoring and test-generation baseline |
| Manual expert requirements engineering | Domain judgment, elicitation, negotiation, feasibility, stakeholder validation and interpretation of local context | Essential human baseline and required adjudication layer |
| General-purpose LLM workflows | Fast extraction, drafting, rewriting, classification, explanation and candidate test generation | Baseline for task-level automation; not the authority source |
| Domain systems and registries | Authoritative records within bounded domains (e.g. property, custody, insurance, engineering, heritage) | Preserve native record ownership; use source-aware adapters |

Public documentation:
- IBM DOORS Next: https://www.ibm.com/docs/en/engineering-lifecycle-management-suite/doors-next/7.2.0?topic=requirements-managing
- Siemens Polarion: https://www.siemens.com/en-gb/products/polarion/application-lifecycle-management-alm/
- Jama Connect: https://www.jamasoftware.com/platform/jama-connect/features/

A fair comparison must configure each product competently and score only capabilities that belong to its intended scope. A discussion platform should not be marked as failing to verify a physical transfer; it should be marked out of scope for that measure.

## 4. Current research says not to trust the easy scores

Recent studies provide important warnings:

1. A September 2026 benchmark of ten off-the-shelf LLMs across two requirement sets and 100 runs reported an asymmetric error profile. Its best-performing evaluated model detected a median 47% of expert-identified issues and false-flagged 11%. Detection was particularly weak for necessity and correctness judgments requiring systems-engineering expertise. The result is specific to that study and its corpus; it is not a universal score for every current model. https://arxiv.org/abs/2609.03230
2. A May 2026 empirical study transformed 150 OSS feature-request titles into 900 candidate requirements. It found that LLMs can draft useful requirements, but the quality varied with input clarity and prompt design; the authors explicitly retained human refinement and domain validation. The evaluation focused on ambiguity, verifiability and singularity, not institutional/legal correctness. https://link.springer.com/article/10.1007/s00766-026-00462-z
3. An August 2026 cross-task study of LLMs in requirements engineering found that performance was task-dependent and no model consistently outperformed the others across classification, specification and traceability tasks. https://arxiv.org/abs/2608.21531
4. A 2026 industrial case study reported that AI-generated requirements could be comparable on selected clarity and testability dimensions but underperformed where domain-specific quantitative judgment was required. https://www.tandfonline.com/doi/full/10.1080/09544828.2026.2683735
5. A 2026 systematic review highlights the lack of shared benchmarks for several requirements-engineering activities and the difficulty of comparing studies with incompatible datasets, metrics and evaluation methods. https://www.mdpi.com/2673-4052/7/5/169

Implication: do not optimize for BLEU-like similarity, fluent prose, an LLM judge score, raw requirement count or one-shot acceptance rate. Optimize for adjudicated completeness, source fidelity, modality fidelity, correct uncertainty, traceability, change impact and lower human rework without increasing critical defects.

## 5. Differentiation hypotheses worth testing

### H1 — Source-grounded intent, not prompt-only drafting

Every candidate should retain the exact source span, source identity/version, interpretation status and any approved derivation. Missing source text, unavailable attachments or unclear jurisdiction must remain unresolved.

### H2 — Modality and authority are first-class

Keep a preference distinct from a mandatory obligation. Require a separate human decision when the modality changes. Do not convert a document signature, model confidence or job title into the authority to accept a requirement.

### H3 — Compile to existing owners, not a universal ontology

Map candidate requirements to owner-native records in Mycelix, Symtropy, Sol-Atlas, customer systems or competent external authorities. Do not create a parallel identity provider, property registry, digital twin, event store or decision engine.

### H4 — Compile verification alongside requirements

A candidate should include its intended acceptance condition, verification method, evidence owner, measurement context, freshness policy and limitations. If no valid method exists, retain a typed NotYetVerifiable state with a specific, human-approved exception rather than a false PASS.

### H5 — Change impact across institutional boundaries

When a person, policy, provider, jurisdiction, asset configuration or evidence source changes, identify which requirements, tests, interfaces, approvals and prior qualification claims are affected. Preserve the older as-known-at view.

### H6 — Multi-objective outcomes and hard floors

Do not collapse privacy, rights, authority, security, safety, service, financial, environmental and continuity requirements into one priority score. Keep hard prohibitions and non-compensatory floors separate from tradeable preferences.

### H7 — Value-of-information-aware elicitation

When the brief is incomplete, rank clarifications or source requests by expected decision improvement, risk avoided, cost, latency, reversibility and deadline. Do not ask the client to provide every possible datum; identify which missing fact could change a decision.

### H8 — Adversarial and exception-first qualification

Test contradictory principals, malicious or mistaken insiders, stale authority, compromised providers, colluding vendors, source disagreement, conflicting jurisdictions, emergency exceptions, partial outage and attempted coercive recovery.

## 6. Benchmark protocol

### 6.1 Arms

A. Expert manual requirements analysis.
B. General-purpose LLM, one-shot generation.
C. General-purpose LLM with structured extraction, source retrieval and a reviewer.
D. Jama Connect, configured with Advisor and its supported requirements/test workflows where licensed.
E. Siemens Polarion, configured for paragraph traceability, change control, approvals and test linkage.
F. IBM DOORS Next, configured for requirement attributes, traceability, workflows and downstream lifecycle links.
G. Proposed Institutional Intent Compiler plus the same human-review budget.

Where access or licensing prevents a direct run, label the vendor arm unavailable and do not substitute a literature comparison for a measured result.

### 6.2 Two separate contests

**Contest 1 — brief-to-approved-requirements**

Measure candidate completeness, source fidelity, modality fidelity, semantic defects, contradiction detection, review effort and accepted baseline quality.

**Contest 2 — approved-requirements-to-evidence**

Measure traceability from approved requirement to verification method, test/evidence, owner-native record, change impact and historical reconstruction.

A tool can perform well in Contest 1 and poorly in Contest 2. Do not combine the results without showing both.

### 6.3 Corpus and annotation

- Start with 60 development briefs; freeze at least 200 held-out briefs before tuning.
- Include at least 50 adversarial, contradictory, incomplete or change-heavy briefs.
- Cover enterprise operations, family/estate asset stewardship, infrastructure, institutional governance and heritage.
- Use synthetic briefs initially. Any private customer material requires explicit permission and must not be placed in public fixtures.
- Use at least two independent domain-qualified annotators per brief; keep disagreements and adjudication visible.
- Keep hidden truth and holdout labels out of candidate systems.
- Pre-register primary metrics, exclusions, stopping rules, budget, prompts, model/version, retrieval sources, reviewer hours and subgroup analyses.
- Give commercial tools appropriate configuration, training and feature access; publish product edition/version and integrations.

### 6.4 Primary metrics

| Measure | Definition |
|---|---|
| Critical-constraint recall | Fraction of adjudicated high-consequence obligations captured |
| Must-have recall | Fraction of adjudicated mandatory requirements represented |
| False addition rate | Unsupported or invented obligations per accepted baseline |
| Modality error | Incorrect conversion among mandatory, conditional, preferred, optional, prohibited and unresolved |
| Source fidelity | Correct source span and source version per accepted requirement |
| Traceability completeness | Source → accepted requirement → acceptance condition → evidence owner → verification record |
| Semantic defect rates | Ambiguity, non-atomicity, non-verifiability, infeasibility, incorrectness, redundancy |
| Contradiction performance | Precision/recall for conflicts, dependencies and incompatible scopes |
| Uncertainty correctness | Proper handling of contested, stale, unavailable, unsupported and not-yet-verifiable information |
| Change-impact performance | Precision/recall for requirements/tests/evidence invalidated by a controlled change |
| Human review burden | Reviewer minutes, edits, reversals, clarifications and downstream rework |
| End-to-end time | Time to a reviewed, evidence-ready baseline, not merely first draft |
| Privacy/authority leakage | Sensitive data disclosed or unsupported authority inferred |

Report confidence intervals, per-domain distributions, denominators and critical-error counts. An aggregate score cannot compensate for a critical false-clear.

## 7. Target ladder (prospective, not achieved)

### P0 — safety semantics

- 100% of accepted requirements link to a source span or human-approved derivation.
- Zero silent omissions in a frozen must-have set.
- Zero unapproved modality changes.
- Zero false-clear results on frozen authority, privacy, security and safety hard-negative controls.
- 100% of unresolved, stale, contested and unavailable states remain visible.
- 100% of accepted requirements have an acceptance/verification disposition and known limitation.

### P1 — reviewer utility

- At least 98% recall on pre-labelled high-consequence constraints while preserving zero false-clear on frozen hard negatives.
- At least 95% must-have recall on held-out briefs.
- At least 30% lower expert review time than the best matched baseline without increasing adjudicated critical defects.
- At least 80% fewer missing source-to-acceptance traceability links than the matched workflow baseline.
- Report confidence intervals and domain-level results; do not hide weaker domains.

### P2 — cross-system advantage

- At least three owner-native adapters with typed unsupported states.
- Two independent evaluator/replay implementations.
- At least 200 held-out briefs and one external replication.
- At least 95% precision and recall for change impact on the frozen change corpus, with zero missed predesignated critical invalidations.
- 100% of synthetic authority-escalation, stale-source and unknown-outcome hard negatives fail closed.

These are stretch targets. Freeze and preregister exact definitions before held-out evaluation. Do not tune thresholds after seeing the holdout.

## 8. How to win without trying to replace incumbents

1. **Integrate with DOORS Next, Polarion and Jama.** Send accepted, human-reviewed requirements and evidence references to the customer’s chosen system of record; retrieve status and change events through supported APIs or exports.
2. **Own the difficult intake boundary.** Focus on complex, multi-stakeholder briefs where obligations span different systems, entities, jurisdictions and asset domains.
3. **Make source and authority boundaries visible.** Each requirement explains who asserted it, what source supports it, which owner can decide it and what remains unknown.
4. **Offer private deployment and strict data control.** Deployment, retention, key custody, operator access and model/provider boundaries must be explicit and independently reviewed.
5. **Sell evidence-ready outcomes.** Measure reductions in omitted obligations, review effort, change-impact misses and reconstruction cost.
6. **Build industry-specific benchmark credibility.** Publish corpora, protocols, manifests, baselines, limitations and independent replication. A public benchmark may become more valuable than a proprietary model claim.

The wedge is not “better requirements software.” It is a verified bridge from messy institutional intent to an existing engineering/governance/asset ecosystem.

## 9. Immediate engineering priorities

1. Finish the standalone strict Rust manifest validator; treat its current status as authored source/tests until exact-head compile and test evidence exists.
2. Add a frozen 20-case known-answer/adversarial corpus and an independently implemented reference evaluator.
3. Add source-span byte/digest verification against supplied source bytes in a later private-input-capable reader; current v0.1 checks only declared lengths and digest shape.
4. Add explicit lifecycle-transition tests and authority-profile verification through an owner-native reviewer/authority source; opaque reviewer references are not authenticated by v0.1.
5. Add a benchmark runner that measures semantic-quality metrics using adjudicated human labels rather than calling the production validator as its own oracle.
6. Define ReqIF and API adapter experiments only after the static requirement contract and privacy boundary stabilize.

## 10. Claim ceiling

The v0.1 validator only checks structural consistency of the supplied manifest. It cannot establish that a requirement is complete, true, feasible, legally valid, safe, authorised, or an accurate interpretation of customer intent.

A future benchmark may support narrow claims such as “reduced review time for source-grounded acceptance baselines on this preregistered corpus without increasing critical defects.” It cannot support a universal claim to outperform requirements engineering, family-office systems, systems engineering or human experts.
