# Institutional assurance differentiation and competitive qualification

**Research note:** 2026-11-14  
**Status:** public-document desk research and proposed comparison. No vendor system was installed, configured or independently benchmarked. Vendor performance claims are not treated as neutral evidence.

## 1. The competitive correction

The product landscape is more capable than a comparison against "generic AI requirements tools" suggests.

### Requirements and engineering lifecycle

| Product | Publicly documented capabilities | What Luminous must not claim as unique |
|---|---|---|
| Siemens Polarion ALM | Requirements traceability, automatic change control, audit trails, electronic signatures, historical "time machine", impact analysis, configuration/version management and APIs | Traceability, historical browsing, audit history, change impact or lifecycle workflows |
| IBM DOORS Next | Requirement attributes, relationships/traceability, baselines, configuration management, link validity and change-impact workflows | Baselines, traceability to tests/work items, impact views or requirement lifecycle management |
| Jama Connect + Jama Intelligence | INCOSE/EARS language analysis, individual and batch reviews, rewriting, AI-generated test cases, relationship discovery, PDF parsing, traceability and review controls | AI requirement feedback, test generation, traceability-gap detection, relationship discovery or document-to-structured-item extraction |

Sources:
- Siemens Polarion: https://www.siemens.com/en-gb/products/polarion/application-lifecycle-management-alm/
- IBM DOORS Next: https://www.ibm.com/docs/en/engineering-lifecycle-management-suite/doors-next/7.2.0?topic=requirements-managing
- Jama feature overview: https://www.jamasoftware.com/platform/jama-connect/features/
- Jama Intelligence capability overview: https://help2.jamasoftware.com/ah/en/jama-intelligence-.html
- Jama Intelligence support page, published October 16 and updated October 21, 2026: https://support.jamasoftware.com/hc/en-us/articles/49682592826253-What-Is-the-Jama-Intelligence-AI-Assistant-Icon

Jama's own support documentation says its generated test cases remain reviewable by engineering staff and do not guarantee complete coverage. Its feature set now includes more than grammar/quality advice. A fair evaluation must configure it competently and compare only in-scope functions.

### Family-office and wealth operations

| Product | Publicly documented capabilities | What Luminous should integrate with rather than reproduce |
|---|---|---|
| Addepar | Multi-custodian and alternative-asset aggregation, advanced entity/ownership graphs, look-through exposure, open APIs, portfolio analytics and scalable reporting | Wealth aggregation, investment reporting, complex entity modelling and accounting/CRM integrations |
| Masttro | Direct custodian feeds, consolidated public/private/real-asset/collectible views, entity mapping, alternatives ingestion, reporting, role/entity permissions | Family-office asset visibility, custodian feeds, alternative asset reporting and whole-office dashboards |
| Eton Solutions AtlasFive | Entity and portfolio management, GL, trust/partnership accounting, tax, transaction processing, document management and extensive operational workflows | General-ledger, trust/partnership accounting, bill pay, tax and family-office ERP functionality |

Sources:
- Addepar operations: https://addepar.com/operations
- Masttro platform: https://masttro.com/family-office-software
- Eton Solutions AtlasFive: https://eton-solutions.com/solutions/atlasfive/

These are vendor descriptions, not independent performance results. Masttro's connection counts and Eton's scale claims should be treated as vendor-published until independently verified.

## 2. The defensible product boundary

The product should not be another requirements repository, portfolio dashboard, custodian aggregator or family-office ERP.

The proposed boundary is a **decision-integrity layer** that consumes bounded references and returns a reconstructable decision packet.

A consequential decision should be answerable along these independent dimensions:

1. **Intent:** Which accepted, human-approved obligation and source version governs?
2. **Claim:** Which exact claim dimension is being asserted?
3. **Competence:** Is each source competent for that claim, purpose and jurisdiction?
4. **Provenance:** Which bytes/version, adapter build, upstream provider and observation receipt were used?
5. **Authority:** Which actor/entity/delegation is valid for this action, asset, purpose and execution time?
6. **Time:** What was effective, what was known, and what was learned later?
7. **Execution:** Did an external side effect occur, fail, or remain unknown pending reconciliation?
8. **Change:** Which downstream decisions and qualifications are invalidated, potentially affected or unchanged?
9. **Sufficiency:** What is the minimum next competent evidence request, and why is it sufficient?

The differentiated output is not a green status. It is a bounded, replayable decision with its unresolved dependencies and safe next step.

## 3. Proposed category: cross-system institutional decision integrity

The category should be defined by capabilities no single incumbent is positioned to own across all the others:

- Claim-scoped evidence competence across unrelated source systems.
- Issuer trust, subject binding, delegation scope and revocation at execution time.
- Effective-time and knowledge-time reconstruction across different owner-native systems.
- Reconciliation of unknown external outcomes and detection of duplicate side effects.
- Cross-system change invalidation without rebuilding every source system's own dependency model.
- Decision-sufficient evidence acquisition that minimizes unnecessary collection and disclosure.
- Independent, replayable evidence packets with exact corpus, source, adapter, policy and evaluator digests.

These are differentiation hypotheses, not proven gaps in every incumbent product. The competitive test must examine configured integrations and real workflows, not marketing-page absences.

## 4. Research benchmark programme

### IIC-Eval

Measure semantic fidelity, modality, condition/exception preservation, source-span/version binding, contradiction handling, prompt-injection resistance and decision-time evidence coverage.

The September 2026 LLM requirements-assessment study reports that its best evaluated off-the-shelf model detected a median 47% of expert-identified issues and false-flagged 11%. This is one study's result on its corpus, not a universal model score. It supports a benchmark that distinguishes misses from false alarms and necessity/correctness judgments from prose clarity.

### IPB-S6

Measure claim separation, source competence, temporal applicability, authority-at-execution-time, common-cause dependence, revocation, stale/digest-mismatched evidence, unknown-outcome reconciliation, replay and unauthorized side effects.

### DTEC — Decision-Time Evidence Coverage

TEMPO provides a useful retrieval baseline: it evaluates 1,730 multi-domain temporal queries, 3,976 decomposed steps and reports 71.4% Temporal Coverage@10 for its best retrieval system. DTEC is not directly comparable to TEMPO. It asks whether *all critical evidence references needed for each specific decision-time slice* were recovered with correct source version, competence, authority and effective/knowledge-time semantics.

### VOI / decision-sufficient acquisition

Measure whether the next evidence request resolves the decision-critical blocker with the least unnecessary cost, delay and disclosure, while preserving hard constraints. Compare fixed checklists, next-missing-field rules, LLM suggestions, expert planning and competence/dependency-aware planning.

### Integrated qualification

Before claiming a new category, create at least 100 end-to-end cases, spanning multiple domains and owner systems. Require:
- two independent adjudicators;
- hidden held-out cases;
- preregistered comparators, thresholds and reviewer budgets;
- exact system/configuration/corpus/evaluator receipts;
- per-case errors and denominators;
- an external replication.

Prospective targets (not results):
- zero false clears on frozen hard-negative safety/authority controls;
- at least 98% DTEC on adjudicated held-out slices;
- at least 95% authority-at-execution-time correctness;
- at least 90% change-impact recall with less than 5% false invalidation of unaffected controls;
- at least 30% reduction in expert reconstruction time without increased critical defects;
- at least 30% fewer unnecessary evidence requests than a fixed checklist, without increased disclosure burden.

Zero failures on a finite test set does not prove zero real-world risk. Publish confidence intervals and scale the hard-negative set. For example, zero failures in 300 independent trials gives only an approximate 95% upper bound of 1% (rule of three).

## 5. Initial commercial wedge

Offer a bounded **Institutional Decision Integrity Review** alongside the customer's existing systems:

- one institution and one decision family;
- a limited source dossier and its approved purpose;
- explicit owner-native source and authority boundaries;
- a human-approved intent baseline;
- claim-scoped evidence and temporal replay;
- a decision packet, unresolved blockers and minimum evidence requests;
- measured expert time, false inferences, source completeness and review rework.

The customer keeps its incumbent requirements, portfolio, ERP, registry and custody systems. Luminous supplies cross-system assurance and evidence reconstruction.

Do not promise a generic efficiency percentage or legal/financial assurance. Measure an actual baseline and report the limitations.

## 6. Strategic conclusion

Incumbents already solve much of the local workflow. The defensible opportunity lies in the joins between their boundaries: the exact obligation, the source competent to establish a claim, the actor authorized at the instant of execution, the evidence available at the decision time, and the consequences of later corrections.

Luminous should win by proving fewer false inferences, faster reconstruction and lower evidence-request burden on a matched workload—not by claiming more features or a larger AI model.
