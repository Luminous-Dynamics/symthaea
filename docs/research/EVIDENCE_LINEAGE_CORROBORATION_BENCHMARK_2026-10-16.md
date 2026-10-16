# Symthaea Lineage-Aware Corroboration Contract

**Date:** 2026-10-16  
**Status:** deterministic grouping contract and adversarial tests authored; no tests run or performance results claimed  
**Implementation:** src/knowledge/evidence_lineage.rs

## Problem: citation count is not independent support

A knowledge system can overstate confidence when many records repeat one underlying report, press release, witness, dataset, or anonymous source. Counting URLs, documents, publishers, or mentions overstates the amount of evidence when the information has a shared origin. The opposite error also matters: a strict system can collapse genuinely independent observations merely because they share a publisher, domain, or topic.

The model should therefore make lineage explicit and keep three quantities separate:

- Raw witness count: retrieved/recorded evidence items.
- Attributed lineage-group count: groups with an explicit canonical-origin assertion after deduplication.
- Unresolved lineage-group count: groups whose upstream origin is not known from available metadata.

An attributed origin group is not proof of genuine independence. Independence is domain- and claim-dependent and may require human or domain-specific adjudication. The prototype never upgrades missing lineage into independent support just because two URLs differ.

## Research basis

W3C PROV models provenance with entities, activities, and agents. Its ontology provides prov:wasDerivedFrom, plus more specific relationships such as prov:wasQuotedFrom, prov:wasRevisionOf, and prov:hadPrimarySource. Those are useful semantics for lineage edges; a source URL alone is not a lineage graph. See the [PROV Model Primer](https://www.w3.org/TR/prov-primer/) and [PROV-O](https://www.w3.org/TR/prov-o/).

The [IARPA Source Independence](https://iarpa.org/international/source-independence/) guidance highlights syndication, shared anonymous sourcing, common press releases, and circular citation as reasons multiple articles can still represent one source chain. This is a threat model, not a ready-made automated detector or ground-truth labeling scheme.

## Current deterministic contract

EvidenceWitness carries a unique evidence occurrence ID, source-record ID, optional canonical origin ID, and optional content digest. The grouping algorithm unions witnesses when any one explicit fact holds: same evidence occurrence ID; same source-record ID; same non-empty declared origin ID; or same well-formed SHA-256 digest for the compared content representation.

A malformed or unsupported digest is never used as an identity key and blocks eligibility. If one resulting group asserts more than one canonical origin, the assessment abstains instead of choosing the convenient label. When lineage is unknown, the system reports an unresolved group; it does not assume independence based on distinct URL or publisher.

The default review policy requires at least two distinct attributed lineage groups and requires every group to have attributed lineage. This is a conservative engineering default, not a universal epistemic law. Some tasks may be answerable from one primary source; others need independent replications or different evidence types. The minimum must be policy/domain-specific and measured against labeled tasks.

## Adversarial benchmark design

Keep the proposition fixed within each family and mutate one lineage property at a time.

| Level | Construction | Expected accounting |
|---|---|---|
| L0 | Two distinct, explicitly attributed origins with distinct valid content | Two attributed lineage groups |
| L1 | Ten articles all derived from one primary report | One group, nine duplicate witness records |
| L2 | Different URLs/publishers, no origin metadata, distinct digests | Two unresolved groups, zero attributed origins |
| L3 | Same exact content digest with inconsistent origin labels | One duplicate group; conflicting-origin reason and abstention |
| L4 | Same source ID repeated with changed labels/metadata | One source group; conflict surfaced, never counted as two |
| L5 | Malformed/unsupported digest alongside otherwise sufficient sources | Abstain; malformed digest is not used to merge or distinguish sources |
| L6 | Different documents from one publisher but explicitly distinct origins | Distinct groups unless another lineage signal connects them |
| L7 | Revisions of one underlying source with different content digests | One group if the common origin is explicit |
| L8 | Genuine independent observations share a publisher or domain | Distinct groups when independent origin IDs are properly attributed |
| L9 | Circular citations or reciprocal summaries without known original | Unresolved until lineage is adjudicated; cycles must not create new origins |

L6/L8 are false-collapse controls; without them, the system might combine everything. L2/L9 are false-independence controls: missing origin data must not be treated as proof of independence.

### Annotation and leakage controls

1. Create at least 1,000 independent claim/source-lineage families for the pilot; expand to 5,000+ before release threshold decisions. Split by underlying source family, not URL.
2. Have two annotators label primary origins, derivation/copy/quotation/translation relationships, unresolved lineage, and whether origins are independently probative for this specific claim. Adjudicate disagreements.
3. Retain labels such as same-origin, independent-origin, derived-from, and unknown. Do not force uncertain cases into binary independent/dependent labels.
4. Never place different URLs for the same original source or generated counterfactual siblings across train/dev/test. Reserve a source-shift publisher/domain test set.
5. Separate gold-lineage experiments from end-to-end lineage extraction. The former tests grouping logic; the latter measures origin-resolution accuracy and false merges.
6. Keep claim truth, source authenticity, lineage attribution, and source independence in separate annotation columns. They are related but not interchangeable.

## Metrics and proposed gates

Targets below are hypotheses to test, not results already obtained.

| Metric | Proposed pilot gate | Reason |
|---|---:|---|
| Raw-citation inflation on exact duplicate/syndication fixtures | Zero additional attributed-origin groups | Core failure mode |
| Distinct attributed-origin count on gold lineages | 100% regression correctness | Pure grouping contract |
| Unresolved-origin leakage into credited origins | 0 | Do not hallucinate lineage |
| Conflicting-origin assertion escapes | 0 | Prevent contradictory lineage attribution |
| Malformed-digest eligibility escapes | 0 | Invalid identity data must fail closed |
| False merge rate on adjudicated independent origins | ≤1% target after source-shift validation | Avoid erasing genuine corroboration |
| False split rate on copies/revisions with gold shared origin | ≤1% target after source-shift validation | Avoid inflating independent support |
| Lineage extraction precision/recall | Report separately; set target after a labeled pilot | Gold grouping does not validate extraction |
| Abstention rate and coverage by domain | Always report together | A never-answer system must not appear successful |
| Throughput | p50/p95 at 1k, 10k, 100k witnesses with exact commit/hardware | Measure scaling instead of inventing a number |
| Audit replay | 100% identical groups/decision from pinned inputs and policy | Reproducibility |

The difference between raw-witness count and group count is a deterministic duplicate-count metric, not semantic accuracy. Population-level false merge/split rates require an independently labeled corpus and confidence intervals.

### Reproducible in-memory scaling probe

An opt-in ignored Rust test generates complete, unique in-memory witness sets and reports p50/p95 assessment time at 1k, 10k, and 100k witnesses. Run it in release mode:

    cargo test --release --lib knowledge::evidence_lineage::tests::benchmark_lineage_grouping_scaling -- --ignored --nocapture

The probe validates group-count parity before timing. It excludes lineage extraction, network/source lookup, persistence, and digest computation. It is not a production latency result; no measured timing is claimed until the exact test is run and the commit, hardware, compiler, and feature configuration are recorded.

## Ablations and baselines

Compare raw citation count; distinct URL count; distinct publisher/domain count; exact digest deduplication only; origin-ID grouping only; the full union of source ID, digest, and declared origin; and gold lineage vs. extracted lineage.

Each evaluation must report both recall of truly separate origin groups and the rate of false independent support. If a strict origin-completeness gate is used, show its coverage/abstention cost.

## Integration guardrails

- Assess evidence per proposition, not globally across the entire knowledge graph.
- Preserve source and lineage relationships as auditable references; keep canonical trust/provenance records in one shared authority.
- Group membership is an audit aid, not a trust score and not a truth label.
- The current implementation is a pure in-memory module, not connected to a graph query or persistent source store.
- SHA-256 is used only as a content-identity key for exact byte/representation matches. Canonicalization scope must be specified before comparing rendered web pages, extracted passages, and raw bytes.
- Future lineage graphs must handle cycles, unknown parents, multiple upstream sources, and time-varying revisions. A source derived from multiple origins should not be collapsed into one fictional primary source.

## References

- W3C, [PROV Model Primer](https://www.w3.org/TR/prov-primer/), derivation, revision, and source lineage.
- W3C, [PROV-O](https://www.w3.org/TR/prov-o/), derivation, quotation, revision, and primary-source relations.
- IARPA, [Source Independence](https://iarpa.org/international/source-independence/), syndication, shared anonymous sourcing, and circular citation threat models.
