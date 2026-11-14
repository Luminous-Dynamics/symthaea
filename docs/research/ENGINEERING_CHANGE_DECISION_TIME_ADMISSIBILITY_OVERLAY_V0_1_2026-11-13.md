# Engineering Change Decision-Time Evidence Admissibility Overlay V0.1 — 2026-11-13

**Status:** prospective benchmark overlay only; no candidate runs, adjudicated labels, vendor trials or measured results.  
**Parent benchmark:** [ENG-DESIGN-CHANGE-QUAL-001 #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543)  
**Experiment:** [Engineering Change Experiment Preregistration V0.1](ENG_CHANGE_EXPERIMENT_PREREGISTRATION_V0_1_2026-11-13.md)  
**Lifecycle reasoning owner:** [KNOW-LIFE-001 #10600](https://github.com/Luminous-Dynamics/symthaea/issues/10600)

## 1. Research question and paradigm shift

A decision can be semantically supported by a document and still be unjustified at the time it was made. The document may have arrived later, a correction may have been backdated, an index may not yet have incorporated it, or a supersession relation may have been recorded after the decision.

The benchmark therefore tests two independent coordinates:

- **World-valid-at T:** which proposition or engineering condition applies to the target time T?
- **Available-to-the-decision-process-by K:** which exact evidence versions were admissible to the declared principal and actually available through its acquisition, index and context path by decision frontier K?

These are not interchangeable. Validity is not knowledge. Publication is not ingestion. Ingestion is not index visibility. Index visibility is not proof that an artifact was included in a particular agent run.

The proposed shift is from evaluating an impact analysis against today's best reconstructed truth to evaluating whether the exact decision was justified by the evidence that could be used at its own information frontier. This is a benchmark overlay over existing evidence, identity, lifecycle and receipt owners—not a new temporal database, evidence ledger, provenance graph, or decision-authority system.

## 2. Keep the clocks distinct

For each claim–evidence applicability pair, retain the clock semantics already supplied by the owning source contracts:

1. **World-valid / event time:** when the represented fact, measurement, requirement or condition applies. Keep interval bounds and uncertainty.
2. **Source publication / release time:** when the exact source version was published or made available by its issuer, where such a fact can be established.
3. **Canonical record / receipt time:** when the relevant service actually committed the record or relationship. A signer-asserted event timestamp is not a server receipt.
4. **Acquisition / ingestion time:** when the evaluated organisation or declared principal acquired and admitted the exact bytes under the applicable rights and entitlement profile.
5. **Derived-materialization time:** when a search index, cache, embedding, feature store or other retrieval surface made that version queryable.
6. **Run-context time:** whether the exact artifact/version was actually included in the candidate's retrieval/context manifest for the assessed run.

Do not collapse these into a generic `timestamp`, `updated_at` or `known_at` field. A clock may be unavailable or uncertain. Unknown must remain unknown; it is not an unbounded interval and must not be replaced by the most convenient adjacent timestamp.

### Two admissibility claims, reported separately

- **Historically source-available by K:** the exact version can be shown to have existed and been available to the specified principal by K, under the declared source and rights profile.
- **Actually available to this system/run by K:** acquisition, indexing/materialization and run-context evidence establish the exact version was accessible to the evaluated system at the relevant frontier. If the required lineage is absent, report that fact as unestablished/indeterminate; do not infer that the model knew it from a public publication date or a later successful retrieval.

A later retrieval can support historical source availability only when the exact version, historical publication/availability evidence and rights are independently established. It cannot prove that the original agent actually retrieved or saw that artifact. A replay that reconstructs a candidate context from historical sources must be labelled a controlled reconstruction, not a record of the original context.

The existing ten #10581 engineering dispositions remain unchanged. Temporal admissibility is an orthogonal evaluation axis; it must not silently create a new disposition or alter ETK authority/currentness.

## 3. Frozen task contract

Each temporal challenge overlays an existing #10543 blueprint and #10581 claim–evidence pair. Before any candidate run, the evaluator must pin:

- exact baseline and candidate design/configuration generations;
- exact source artifact identities, byte digests and source-version/revision identities where available;
- target world-valid time T and decision frontier K, with explicit inclusive/exclusive boundary convention;
- source publication/release, canonical record/receipt, acquisition/ingestion, derived-materialization and run-context evidence when each exists;
- declared principal, access/rights profile and relevant trust policy;
- correction, supersession, withdrawal and relation-materialization history;
- timestamp precision, time-zone, clock source and uncertainty intervals;
- the existing dependency path and claim–evidence applicability rubric;
- the evaluator and source-contract revisions.

If an exact source vintage, required availability event or relation-materialization event is not retained, the evaluator must not manufacture it. The case remains indeterminate or unsupported under the predeclared rubric.

## 4. Temporal challenge overlays

The machine-readable companion `ENGINEERING_CHANGE_DECISION_TIME_OVERLAY_V0_1.json` defines nine public challenge blueprints. They are specifications, not gold labels or scored cases.

| ID | Challenge | Required control |
|---|---|---|
| TDA-001 | Backdated correction | A correction received after K cannot alter the earlier K assessment, even when it claims an earlier valid-from date. |
| TDA-002 | Delayed supplier evidence | Earlier asserted inspection/event time does not make a later-received certificate available at the earlier decision frontier. |
| TDA-003 | Index/materialization lag | A committed artifact is not treated as available to a retrieval-based candidate before the index/cache/embedding materialization is evidenced. |
| TDA-004 | Known future-effective requirement | A requirement published before K may be considered as a known future obligation, but not represented as already effective at T. |
| TDA-005 | Late supersession relation | A supersession edge materialized after K cannot rewrite the earlier evidence state merely because its endpoints carry older timestamps. |
| TDA-006 | Unknown vintage or time bound | Missing availability, source vintage or validity bounds remain indeterminate; no adjacent timestamp is substituted. |
| TDA-007 | Overlapping clock uncertainty | When uncertainty intervals overlap, the system must not invent a total event order or causal dependency from nominal timestamps alone. |
| TDA-008 | Rights and use-time drift | Historical source availability, rights at decision time and rights at replay/query time remain separate; historical access does not grant present use permission. |
| TDA-009 | Standards/profile revision boundary | A later-indexed standards artifact or validator cannot silently replace the exact source/profile edition pinned for an earlier assessment. |

These controls should be instantiated across material, component/lot, calibration, requirement, manufacturing-process, standards and supplier-deviation changes where the underlying source owner can supply the required evidence. Avoid multiplying near-identical mutations and then counting them as independent design families.



### TDA-010 — Distributed evidence frontier and partial-order causality

A decision may combine records from multiple conductors, repositories or asynchronous source services. A single wall-clock K is still retained as the declared temporal boundary, but it does not by itself prove which distributed events were included or causally ordered. A source's local sequence number is meaningful only within that source; equal, larger or smaller indices from different conductors are not comparable.

For multi-source cases, the overlay binds:
- Mycelix #2631 `EvidenceFrontierV1` reference/digest and exact source-set commitment;
- Mycelix #5843 causal-history receipt, per-conductor local order and evidence-backed cross-conductor edges;
- Mycelix #5923 source/dependency completeness where applicable.

An incomplete or unknown causal cut cannot support `system_availability=available` or `run_context_presence=included`. This composes existing owners and creates no new vector-clock, event-log or causal-history service.

## 5. Non-compensatory gates and metrics

Add the following hard gates to the existing six preregistered gates:

- **Zero future-knowledge leakage:** no evidence version, correction, relation, index entry or run-context artifact first available after K may influence the as-of-K result.
- **Zero retroactive rewrite:** later corrections, withdrawals, supersession relations or newer standards versions cannot alter a previously frozen K projection.
- **No timestamp substitution:** event/valid/publication/receipt/retrieval/index/context times cannot substitute for one another.
- **Unknown stays unknown:** missing or unverifiable source vintage, availability, temporal bounds, relation history or rights yields indeterminate/unsupported as specified—not pass.
- **Exact replay:** identical frozen inputs, frontier, rights profile and evaluator build reconstruct the same temporal admissibility result.
- **No authority amplification:** passing this overlay does not grant approval, release, ETK currentness, compliance or permission to execute.

Report temporal metrics separately from impact precision and cost:

- future-knowledge leakage count (hard gate);
- retroactive historical-state rewrite count (hard gate);
- false admissibility and false inadmissibility, stratified by evidence class;
- exact-version and source-vintage coverage;
- source-available vs system-visible vs run-context-evidenced classification accuracy;
- indeterminate detection and inappropriate forced-resolution rate;
- historical replay equality under late-arrival, correction and retrieval-order permutations;
- clock-order error rate under offset, precision and uncertainty mutations;
- additional review/retrieval burden and time-to-resolution.

Do not combine these into a single weighted score. A false future-evidence acceptance cannot be compensated by lower latency or cost. Likewise, over-excluding legitimately known future-effective requirements must be reported separately rather than hidden by a high precision score.

## 6. Baselines and relationship to temporal SOTA

The parent engineering benchmark retains B0–B5 and conditional vendor arms. Add matched temporal ablations only where the same frozen source vintages and task budgets can be supplied:

1. latest-value/current-state retrieval with no explicit as-of frontier;
2. timestamp-only filtering, which treats one timestamp as a proxy for availability;
3. lifecycle-aware as-of retrieval/projection using exact source versions and declared time coordinates;
4. full Luminous composition with decision-time evidence admission and the existing engineering impact/applicability reasoning.

Use temporal model-editing benchmarks as diagnostic references, not as direct PLM leaderboard comparators:

- [Vera-Life-Eval / Vera-Life (GroundLM 2026)](https://aclanthology.org/2026.groundlm-1.18/) evaluates lifecycle faithfulness across current, past, stale-only, update, conflict and multi-hop evidence settings. Its reported 8,500 questions and 41,203 versioned evidence pieces are a dataset description, not a target score for this engineering benchmark.
- [TKE-Bench / ChronoExpert (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.1823/) diagnoses chronological interference in model knowledge editing. Its model-editing task is not equivalent to a source-ledger or PLM workflow.
- [TRACE (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.1045/) tests chronological preservation and bidirectional temporal reasoning after fact edits. Use its task taxonomy only after dataset rights and exact revision are checked.

The relevant SOTA question is not whether a system can answer a temporal question from a large benchmark. It is whether an engineering workflow can prove that every claim–evidence decision used the correct source vintage and information frontier, while preserving the historical answer and refusing unsupported hindsight. Do not transfer reported scores across these non-equivalent tasks.

Vendor trials must receive the same eligible source versions and decision frontier. For each vendor arm, record whether it exposes the source/version, evidence-history, index and actual recommendation-context lineage needed to evaluate the temporal gates. Missing product telemetry makes that submetric unobservable/incomparable; it is not automatic proof of vendor failure.

## 7. Existing-owner composition (no duplicate framework)

- **#10543 / #10581:** engineering-change benchmark, claim–evidence pair and independent gold adjudication.
- **#10600:** lifecycle-faithful, as-of evidence reasoning and temporal benchmark alignment.
- **#11241:** bitemporal evidence-frontier work for its financial research/quant consumers; informative precedent, not silently promoted to a universal engineering contract.
- **Mycelix #3022 / #5451:** canonical source/artifact identity, import receipts, provenance families and shared evidence-plane expectations.
- **Mycelix #3046:** Unknown vs Unbounded temporal-bound semantics.
- **Mycelix #4778:** temporal provenance-edge materialization; a relation that changes state needs its own recorded history.
- **Mycelix #5687:** source-specific creation, publication, recording, deposition and uncertain chronology semantics.
- **Mycelix #5860:** DeSci receipt-time as-known-at projection. Its own claim ceiling is preserved: a committed receipt does not prove when a search index, cache, embedding or model incorporated the item.
- **Mycelix #1112 / #5395 / #6105:** semantic translation loss, exact GS1 artifacts and supply-chain interoperability profiles.

This overlay consumes these contracts. It creates no generic temporal ledger, evidence identity, event log, relation graph, rights service, receipt schema, validator or approval authority.

## 8. Gold-record contract compatibility gate

A compatibility audit of the current [#10581 gold-record schema](https://github.com/Luminous-Dynamics/symthaea/blob/research/eng-design-change-qual-gold-protocol-v0-1/benchmarks/engineering-change-impact/gold-record.schema.json) found that its root object uses `additionalProperties: false`. It has no structured temporal-admissibility property for T/K, source-version availability, repository receipt, acquisition/ingestion, derived materialization, actual run-context inclusion or rights-at-time. Its optional `notes` field is an array of strings and is not an adequate machine-readable substitute.

**Consequently, temporal overlay gold labels are blocked pending a versioned contract change.** Before adjudicators create temporal labels, the #10581 owner must approve either:
1. a versioned optional nested temporal-evidence object in the existing gold-record contract; or
2. a separately content-addressed companion record, explicitly bound to the existing gold record and validated by an independently reviewed contract.

Whichever route is chosen must bind the exact T and K coordinates, source/version identities, relevant availability/receipt/materialization/context evidence, rights profile, uncertainty/unknown states and assessor adjudication. The ten existing engineering dispositions must remain unchanged. Do not stuff these fields into `notes`, relax `additionalProperties`, or create a second unbound gold-record system.

The current overlay remains public blueprint-only research. Existing #10543 pilot assessment may proceed under its current contract where applicable; no claim is made that the temporal sub-axis is ready for gold adjudication or candidate scoring.

## 9. Staging and claim ceiling

The current 24 engineering blueprints remain a feasibility/rubric pilot and are not automatically nine more statistically independent cases. First map the nine overlays onto existing blueprint claim–evidence pairs, identify which source owners can supply the required time evidence, and mark gaps as unknown. Then have independent assessors adjudicate the applicable pairs under #10581. Any new confirmatory sample must be split by design family and change lineage and powered using cluster-aware analysis.

A pass establishes only that the candidate respected the frozen temporal evidence contract for the declared principal, rights profile, source set, target time and decision frontier. It does not establish source truth, physical condition, engineering safety, legal compliance, professional approval, product release or vendor superiority.

No gold labels, candidate runs, vendor executions, raw production evidence, trusted runner attestations or physical tests were created by this overlay.
