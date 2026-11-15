# Engineering Decision Minimal Evidence Benchmark v0.1

**Status:** synthetic benchmark protocol and preregistered controls  
**Date:** 2026-11-14  
**Parent:** [ENG-DECISION-CORRECTNESS-001 #12039](https://github.com/Luminous-Dynamics/symthaea/issues/12039)  
**Follow-on:** [ENG-DECISION-MIN-EVIDENCE-001 #12132](https://github.com/Luminous-Dynamics/symthaea/issues/12132)  
**Claim ceiling:** synthetic evidence-dependency benchmark; no real engineering qualification.

## 1. Research question

Can an engineering decision system identify what evidence is actually sufficient to support a decision, distinguish independent support from repeated records, and change its conclusion only when a decisive support dependency changes?

This is not a retrieval-volume benchmark. More documents must not create more independent support merely because there are more IDs, files, URLs, or database rows.

The benchmark measures four distinct properties:

1. **Admissibility:** is each record valid for the exact claim, configuration, time, intended use and decision?
2. **Provenance independence:** which admissible records have genuinely distinct provenance roots?
3. **Sufficiency:** do the admissible independent roots cover the evaluator-frozen support obligations?
4. **Decision sensitivity:** does the disposition remain invariant under redundant changes and flip when the last sufficient support is removed?

A candidate cannot choose the source inventory, provenance-root grouping, required support count, or sufficient-set gold.

The fixture separates the **evaluator-frozen source inventory** from the **candidate-visible evidence IDs**. The first defines what sources exist and which interventions changed the world; the second records what the candidate actually received or selected by frontier K. A source that exists in the evaluator inventory but was not materialized into candidate context cannot be silently counted as support for that candidate's decision. Conversely, evaluator gold may enumerate alternative sufficient sets across the complete frozen inventory so that a candidate is not rewarded for one arbitrary evidence subset.

Each case freezes target configuration, source-inventory state/reference/digest, mandatory predicates, minimum independent-root thresholds, and candidate-visible IDs. The last-support case contains paired baseline and intervention worlds: the baseline inventory has two independent roots; the intervention inventory removes one root entirely. This prevents "not retrieved" and "not present in the world" from being conflated.

### Inventory and candidate-visible commitments

Each source-inventory SHA-256 is computed over a domain-separated v0.1 projection:

- domain: `symthaea.engineering-decision-min-evidence.source-inventory.v0.1`;
- target configuration;
- source-inventory state;
- mandatory predicates (sorted);
- minimum independent-root threshold;
- full evidence records (sorted by evidence ID).

The projection is serialized as compact JSON with object keys sorted recursively; array order is retained except where this contract explicitly sorts evidence IDs and mandatory predicates. This is a project-specific canonicalization contract, not a claim of RFC 8785 conformance.

The candidate-visible manifest SHA-256 separately binds:
- domain: `symthaea.engineering-decision-min-evidence.candidate-visible-set.v0.1`;
- the source-inventory SHA-256;
- candidate-visible evidence IDs (sorted).

This separation allows an evaluator to distinguish a changed source universe from a different subset actually exposed to the candidate. The 40-record irrelevant batch has its own domain-separated digest, and every record is present in the injection world rather than represented only by a count.

## 2. Formal object

For a decision claim q, configuration c and knowledge frontier K, let E be the evaluator-frozen evidence inventory and R(e) the provenance root of evidence item e.

An evidence item is eligible only when the relevant owner validators qualify its exact source identity, temporal state, configuration, intended use, authority and claim scope.

A record derived from another record inherits the root of its source lineage unless an independently qualified acquisition/measurement boundary proves a genuinely independent origin. A new document ID, URL, export, format, embedding or summary does not create a new independent root.

For a support predicate p, the effective support set is:

- admissible evidence for p;
- grouped by qualified provenance root;
- deduplicated by common-cause lineage;
- evaluated against an evaluator-frozen minimum independent-root policy.

A decision is supported only when every mandatory predicate has at least the required number of admissible independent roots and no unresolved hard defeater blocks that claim.

**Minimality is not the same as minimum document count.** If several alternative sufficient evidence sets exist, each valid alternative must be preserved. Removing one redundant record should not flip a decision while another admissible sufficient set remains.

## 3. Intervention families

| Family | Intervention | Expected relation |
|---|---|---|
| Redundant-source removal | Remove one record while another sufficient independent set remains | INVARIANT |
| Last-support removal | Remove the last admissible root for a mandatory predicate | FLIP to unsupported/HOLD |
| Common-cause duplication | Add copied/derived records with an existing root | INVARIANT; effective support count unchanged |
| Inadmissible lookalike | Keep content identical but change configuration, rights, version or intended use | FLIP when it removes required support |
| Irrelevant evidence injection | Add irrelevant or duplicate material without changing source inventory completeness | INVARIANT |
| Admissible substitution | Replace one valid root with another admissible independent root that satisfies the same predicate | INVARIANT |
| Unknown provenance | Remove the ability to establish root independence | INDETERMINATE, not PASS |
| Denominator manipulation | Candidate omits a required source role or changes the minimum-root policy | FAIL / INDETERMINATE according to the owner contract |

## 4. Frozen synthetic corpus

The companion JSON fixture contains eight known-answer scenarios:

- redundant evidence with multiple sufficient root sets;
- two records with one shared origin;
- removal of the last independent support;
- content-identical evidence from the wrong configuration;
- large irrelevant evidence injection;
- derivative/copy lineage plus one valid independent source;
- admissible independent substitution;
- unknown provenance that cannot establish independence.

The expected results include effective candidate-visible roots per predicate, all valid sufficient root sets in the frozen world, disposition and exact reason codes. Paired worlds carry their own evidence manifests and source-inventory digests. The corpus has six explicit intervention pairs: redundant-copy removal, last-root removal, irrelevant injection, wrong-configuration lookalike, unknown-to-qualified provenance resolution, and admissible independent substitution. These are synthetic oracle labels, not engineering facts.

A standalone Rust reference-test package is provided at `benchmarks/engineering-change-impact/minimal-evidence-reference`. Run it from the repository root with:

```sh
cargo test --manifest-path benchmarks/engineering-change-impact/minimal-evidence-reference/Cargo.toml
```

The reference tests independently reconstruct root sets, sufficient sets, state dispositions and pair relations from the frozen manifests. They do not call Symthaea's production decision evaluator and do not establish performance of an AI candidate.

## 5. Metrics

Report these separately:

- **Sufficient-set recall:** proportion of evaluator-authored valid sufficient root sets recovered.
- **Sufficient-set precision:** proportion of candidate-reported sufficient sets that are valid.
- **Redundancy-invariance accuracy:** unchanged decisions under removal of redundant evidence.
- **Last-support flip accuracy:** correct loss of support when the last required admissible root is removed.
- **Common-cause detection:** precision/recall for shared-origin, copied and derived source relationships.
- **Inadmissible-lookalike rejection:** rejection of content-identical but configuration-/time-/rights-/purpose-ineligible material.
- **Unknown-provenance handling:** fraction of cases kept INDETERMINATE when independence cannot be established.
- **Evidence inflation resistance:** decision stability under irrelevant/duplicated source injection.
- **Evidence processing cost:** retrieved tokens, source-review time and verification work per qualified decision.

Never reward the smallest set if it omits a required independent evidence line. Never count duplicated records as independent merely to improve recall.

## 6. Proposed targets (not achieved results)

| Metric | Proposed target |
|---|---:|
| Critical common-cause negative detection | 100% on frozen controls |
| Redundancy-invariance accuracy | ≥95% on blinded held-out pairs |
| Last-support removal flip accuracy | ≥95% on blinded held-out pairs |
| Sufficient-set recall | ≥95% |
| Sufficient-set precision | ≥95% |
| Inadmissible-lookalike rejection | 100% on frozen critical controls |
| Unknown provenance promoted to PASS | 0 |
| Decision changes under irrelevant injection/source ordering | 0 on frozen controls |

A zero-failure synthetic suite is a falsification gate, not a population reliability estimate. For any population claim, sample independent task lineages, supplier/source families and configuration families; use cluster-aware uncertainty intervals.

## 7. Baseline ladder and ablations

Matched task packets, source bytes, budgets and candidate-visible information are required.

1. Raw history and source-count heuristic.
2. BM25 retrieval.
3. Temporal RAG.
4. Generic event graph with source IDs.
5. Provenance-root-aware evidence graph.
6. Typed decision-governing state with explicit sufficient sets.
7. Symthaea owner receipts plus independent decision evaluator.

Key ablations:

- source ID versus provenance-root identity;
- provenance graph versus typed common-cause groups;
- raw record count versus independent-root count;
- candidate-visible evidence versus evaluator-frozen complete source inventory;
- all retrieved evidence versus admissible evidence only;
- single sufficient set versus all alternative sufficient sets;
- full source corpus versus decision-minimal sufficient evidence;
- known provenance versus unknown/censored lineage.

Keep content and token budgets fixed when varying the representation of governing state. Otherwise an apparent benefit may merely be extra context.

## 8. Industry comparison

PLM, ALM, MES, ERP and supplier systems remain the systems of record. The benchmark is an assurance layer over their evidence, not a competing repository.

Compare native workflows and their traceability relationships with the evidence-root and sufficient-set evaluator under the same frozen source inventory. Industry product descriptions are not independent performance measurements. Use real customer-owned, permissioned configurations for any vendor pilot.

NIST IR 8536 (final, September 2026) is relevant to provenance, traceability linkage and selective disclosure across manufacturing supply chains. It is an interoperability framework, not this benchmark and not a source-independence oracle.

## 9. Acceptance gates

1. Sufficient-set and provenance-root gold are independently authored and hidden from candidates.
2. Every source has an explicit origin state: qualified root, derivative lineage, unknown, or not applicable.
3. Copies and derivatives cannot increase the effective independent-root count.
4. Every mandatory predicate has a frozen independent-root threshold and a list of valid alternative sufficient sets.
5. Redundant-source removal preserves the decision when another valid sufficient set remains.
6. Last-support removal flips the disposition or makes it INDETERMINATE when lineage is unknown.
7. Exact reason-code agreement and deterministic replay are required.
8. No result grants approval, release, manufacture, operation or effect authority.

## 10. Current execution boundary

The public branch contains eight synthetic cases, six paired interventions, sixteen preregistered mutation controls and a standalone Rust reference-test package. A separate JavaScript reference replay was used during authoring to inspect the expected root sets, sufficient sets, reason codes and paired relations. The Rust reference tests are authored but have not yet been compiled or executed; the mutation controls are preregistered controls, not executed mutation outcomes.

## 11. Explicit nonclaims

This benchmark does not prove physical truth, causal truth, universal statistical independence, source-inventory completeness outside the declared task, legal admissibility, safety, vendor superiority or general SOTA. Results are bounded to exact evidence owners, provenance ontology, task families, minimum-root policy, evaluator build and frozen source inventory.
