# MACE Source-Origin and Provenance Assurance Benchmark V1

**Snapshot:** 2026-11-26  
**Status:** Proposed benchmark protocol. All numeric thresholds are targets, not achieved results.  
**Relationship:** Complements [MACE 2026 Benchmark Parity and Decision-Assurance Programme](MACE_2026_BENCHMARK_PARITY_AND_ASSURANCE_PLAN.md) and the source-origin verifier in [MACE Source-Origin Attestation V1](MACE_SOURCE_ORIGIN_ATTESTATION_V1.md).

## 1. Research question

Can a memory system demonstrate that the evidence used for a decision is not merely signed and internally consistent, but that the declared source and derivation bytes were independently acquired, correctly bound to the evidence graph, and not falsely presented as independent?

This benchmark isolates four claims that must not be collapsed:

1. **Byte integrity:** the bytes acquired now match the digests committed by the replayed graph.
2. **Origin authenticity:** a separately trusted authority attests to source identity and acquisition context.
3. **Lineage completeness:** all material upstream sources and transformations relevant to the evidence are represented.
4. **Causal independence:** sources sharing an upstream cause are not counted as independent support.

A correct SHA-256 check proves only byte equality. A valid DSSE signature proves that a pinned signer made a statement. Neither alone proves publication authorship, factual accuracy, rights, freshness or completeness of ancestry. The benchmark scores these separately.

## 2. Benchmark task families

### A. Exact artifact integrity

Each task supplies a graph commitment, source bytes, payload bytes and—when derived—transformation-receipt bytes. Negative controls mutate one artifact at a time while preserving lengths, alter only a suffix, substitute a valid but different source revision, truncate a stream, or change the evidence identifier without changing the bytes.

The verifier must accept only the exact source, payload and derivation bytes that match the graph. Root nodes must have no derivation receipt; derived nodes must provide the committed receipt. The fixture includes multi-megabyte and boundary-sized artifacts to test limits and aggregate accounting.

### B. Ancestry and transformation completeness

Gold provenance DAGs contain root sources, intermediate snapshots, extracted passages, transformed summaries and final candidate evidence. The evaluator injects:
- omitted parents;
- hidden shared datasets or model-generation dependencies;
- missing or substituted derivation receipts;
- dangling references and cycles;
- parent-set changes after qualification;
- sources whose content changed between acquisition and use.

Use two separately reported strata:
- **closed-world synthetic fixtures**, where the benchmark knows the complete graph and can score exact ancestry recall;
- **open-world acquisition tasks**, where completeness is only scored against independently collected public metadata and cross-resolver evidence. Do not describe this latter reference as omniscient ground truth.

### C. Common-cause and independence reasoning

Include source topologies that look independent by URL, publisher or file hash but are not causally independent:
- mirrors and syndicated articles;
- many pages derived from one report;
- separate reports sharing an upstream dataset;
- multiple model-generated summaries from one model output;
- multiple sensors sharing a clock, calibration source, power supply or physical vantage point;
- different vendors repackaging the same registry or commercial feed.

Vary the hidden common-cause depth, number of apparent sources, and whether the common cause is disclosed by only one acquisition path. Unresolved ancestry contributes zero independent roots. The evaluator grades root-family partitions, not citation counts.

### D. Authority, lifecycle and replay attacks

Test role/profile/key reuse across qualification, sufficiency, lineage and source-origin authorities; stale and revoked keys; profile rollback; expired receipts; replay across run, invocation, subject, DecisionNeed and graph digest; clock rollback; DSSE payload mutation; envelope substitution; and trust-profile replacement.

Add correction, expiry, revocation, rights withdrawal and erasure events while evidence is in active retrieval caches. The system must distinguish reversible epistemic quarantine from rights-based deletion, and it must not silently revive an expired or revoked origin claim.

### E. Disagreement and selective abstention

Create cases where independent acquisition operators disagree about publisher identity, parent edges, source revision, or common-cause groups. The correct outcome is often **indeterminate / re-acquire / abstain**, not majority-vote truth. Measure whether the system preserves disagreement and requests the cheapest useful next observation rather than collapsing conflicting assertions into a fabricated consensus.

## 3. Reference corpus design

Use three independently versioned corpora:

| Corpus | Purpose | Ground-truth status |
|---|---|---|
| Protocol fixtures | Exhaustive byte, signature, binding, expiry, duplicate, cycle and resource-limit controls | Exact by construction |
| Synthetic provenance benchmark | Hidden-parent, common-cause, transformation and adversarial resolver experiments | Exact only for the declared synthetic world |
| External acquisition corpus | Real sources with independently collected publisher, archive, dataset and transformation metadata | Partial; uncertainty and missing metadata must remain explicit |

Each case must contain:
- a frozen task ID and task-family label;
- source and payload artifact digests;
- graph and derivation manifests;
- independent acquisition records and collection time;
- expected root-family partition where the reference is known;
- rights and use-scope constraints;
- attack labels held out from the candidate system;
- scenario cluster ID for statistical correction;
- exact model, tool, resolver, trust-profile and resource-budget manifests.

The candidate resolver must not see hidden graph edges, gold root families or attack labels. Use public development fixtures and an independently controlled held-out set. Record benchmark contamination and tuning exposure.

## 4. Metrics and proposed targets

| Metric | Meaning | Initial target |
|---|---|---|
| Artifact mismatch detection | Reject altered source, payload or derivation bytes | 100% on exhaustive protocol fixtures |
| False artifact acceptance | Accept any deliberately altered artifact | 0 on exhaustive protocol fixtures |
| Ancestor artifact coverage | Every graph node, including non-candidate ancestors, has exact acquired artifacts and a verified receipt | 100% on closed-world fixtures |
| Ancestry completeness recall | Material known parent/derivation edges recovered by independent acquisition | ≥99.5% on frozen synthetic holdout; report per attack family |
| False-complete rate | Declare ancestry complete when a known required edge is missing | 0 on protocol fixtures; 0/3,000 held-out target with cluster-adjusted interval |
| False independent-root rate | Count common-cause roots as separate independent support | 0 on frozen common-cause fixtures |
| Root-family partition quality | Pairwise precision/recall and adjusted Rand index against the known graph | ≥0.99 on closed-world synthetic fixtures |
| Unknown ancestry handling | Treat unresolved or unverified ancestry as independent | 0 tolerance in exhaustive fixtures |
| Authority separation | Accept profile or signer reuse across protected roles | 0 on all deterministic role-separation controls |
| Stale-origin use | Use expired, revoked, corrected or superseded origin evidence | 0 on deterministic lifecycle fixtures |
| Selective risk / coverage | Unsafe acceptance rate as the system answers more cases | ≤1% at ≥80% supported coverage; publish the complete curve |
| Calibration | Coverage of declared confidence / risk bounds | At least nominal coverage, including critical strata |
| Resource overhead | Added p50/p95 latency, bytes, CPU, memory and all-in cost | Report; first establish a reproducible baseline before fixing thresholds |

The 0/3,000 held-out goal is not proof of zero field risk. Under independent Bernoulli trials it corresponds to a one-sided 95% upper bound near 0.1%. Shared scenario roots, repeated source families and attack templates invalidate the independence assumption; report effective sample size or cluster-aware confidence intervals rather than treating near-duplicate cases as independent trials.

## 5. Required ablations and baselines

Run the same frozen corpus and decision policies against:
1. digest-only integrity checks;
2. signed-origin statements without byte reacquisition;
3. source-lineage graph replay without origin verification;
4. current four-authority MACE consumer verification;
5. naive URL/publisher identity matching;
6. citation-count or URL-count independence;
7. two independent acquisition operators with disagreement-preserving abstention;
8. a future threshold/quorum origin profile (for example, 2-of-3 operators) with the same trust and resource limits.

Ablate each of source bytes, payload bytes, derivation receipts, ancestor coverage, profile separation, use-time revalidation and common-cause edges. Report where safety improves, where valid evidence is lost, and the cost of each additional verification stage.

The quorum arm is a research baseline until a threshold or multi-signature protocol and independently governed operators are implemented. It must not be represented as current MACE functionality.

## 6. External benchmark positioning

Agent Memory Leaderboard textual, multimodal and coding tracks remain separate and must be reported using each official scorer and its own denominator. AMA-Bench, MemoryArena, Memora/FAMA and MEMORA-Bench test different aspects of trajectory, action, forgetting and embodied memory; none alone establishes source-origin truth.

MACE therefore needs two scorecards:

- **Memory capability scorecard:** established benchmark parity, measured on matched models, prompts, task sets, graders and budgets.
- **Evidence assurance scorecard:** source integrity, origin authenticity, lineage completeness, common-cause handling, lifecycle correctness, calibrated sufficiency, regret and abstention.

Do not add the assurance score to AML or average across tracks. The assurance benchmark is a distinct contribution that should be published with its task manifest, reference graph format, verifier interface, scorer, confidence-interval method and independent replication package.

## 7. Research frontier beyond the current four-authority draft

The present consumer verifier checks exact bytes and a distinct fourth authority's signed assertion. The next frontier is **multi-operator evidence acquisition**:

- acquire the same source/ancestor through different networks, archives or custodians;
- compare immutable content, revision metadata and transformation receipts;
- preserve disagreements as typed observations rather than resolving them through an unqualified vote;
- require independent corroboration for high-consequence origin claims;
- use 2-of-3 or threshold attestations only when operator independence, correlated compromise and fallback rules are explicit;
- make disagreement, unavailability and unresolved parents reduce supported coverage rather than inflate source independence.

This turns provenance from a self-reported graph into an adversarially testable acquisition process. It still cannot prove universal truth: every result remains scoped to its source, collection time, evidence class, rights, trust assumptions and declared decision.

## 8. Reporting and anti-gaming rules

- Freeze datasets, source snapshots, graph manifests, prompts, model/tool versions, trust profiles and task inclusion before evaluation.
- Score the entire denominator, including abstentions, unavailable sources, failed fetches, timeouts and resource-limit rejections.
- Publish false accepts and false rejects separately; do not hide valid-evidence loss behind a single safety metric.
- Report each task family, ancestry depth, root count, lifecycle event, risk level and attack class, plus the worst supported stratum.
- Use paired seeds, scenario-clustered confidence intervals and at least one independently operated evaluator.
- Keep source-origin signer credentials and gold provenance records outside the candidate system and its rollback domain.
- Never treat a resolver's own graph as the gold reference for grading that resolver.
- Preserve all exact source/derivation digests, receipt envelopes, run IDs, commit SHAs and verifier versions needed for replay.
- Distinguish authored tests, locally executed tests, hosted CI, matched benchmark runs, independent replication and field validation.

This protocol is designed to make false independence and false completeness measurable, rather than allowing a signed, self-consistent lineage graph to be mistaken for verified provenance.
