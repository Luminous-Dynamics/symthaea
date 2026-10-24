# HDC Retrieval Trust Contract — DSE-007

**Status:** Research design; implementation and Rust execution pending  
**Related work:** PR #8510; issue #8639; DSE-005 and DSE-006  
**Core question:** When is a nearest vector a useful candidate, and when is it justified to claim that the intended memory was found?

## 1. Motivation

Exact Hamming search answers a narrow question: which stored vector minimizes distance to the query under a declared tie rule? It does not answer whether the query belongs to a stored record, whether a similar record is the intended identity, whether that record is current, or whether the result is authorized to support a decision.

DSE-005 exposed an observational-ambiguity case: the same cue bytes can be generated as a corrupted cue for a known value or as a cue for a distinct absent value. A function receiving only those bytes cannot always assign the two different identity labels correctly. This is an information deficit, not a SIMD or threshold-tuning defect.

Therefore, treat retrieval as a pipeline with separate, typed contracts:

1. **Search:** produce reproducible candidates and scores.
2. **Interpretation:** estimate whether the candidate is relevant to the declared task.
3. **Identity verification:** establish record identity/version using independent evidence when identity matters.
4. **Decision:** accept, return a tentative candidate, reject, or abstain.
5. **Audit:** preserve enough information to reproduce the decision.

This DSE defines experiments and contracts. It does not claim a current performance, semantic-quality or energy improvement.

## 2. Typed decision model

The caller must declare the task mode. Never silently convert semantic similarity into identity confirmation.

| Result kind | Meaning | Permitted interpretation |
|---|---|---|
| ExactIdentity | Requested stable record ID and version were located and verified against the relevant record metadata/digest | May claim the requested identity if all required checks pass |
| AcceptedSemantic | Candidate passes the explicitly declared semantic/relevance policy | Claim relevance under that policy, not unique identity |
| CandidateOnly | Search found a plausible nearest candidate but the evidence is insufficient | Display or pass to a downstream verifier as tentative |
| NoMatch | The open-set policy rejects all candidates | No stored result accepted |
| Abstain | Evidence is ambiguous, inconsistent, stale, unavailable, or outside the calibrated operating domain | Do not guess; ask for context or invoke a suitable verifier |

Reference pseudocode for the public contract (not a committed Rust API):

    RetrievalDecision =
        ExactIdentity { record_id, version, evidence_digest }
      | AcceptedSemantic { candidate_id, policy_id, score, margin }
      | CandidateOnly { candidate_id, score, margin, reason }
      | NoMatch { policy_id, reason }
      | Abstain { reason, missing_evidence }

A score threshold is one possible feature inside a policy. It is not a universal identity proof. If the caller asks for a specific record ID, identity should be checked through that ID and the version/provenance binding—not inferred solely from whichever vector wins. If the caller asks an exploratory semantic question, a near match may be useful but should remain labeled as semantic.

## 3. Evidence envelope

A decision receipt should bind the result to the exact inputs and policy that produced it. The initial schema should consider:

- task mode and caller-supplied identity/version request, if any;
- candidate ID, winner score/distance, target rank where known, and top-1/top-2 margin;
- query digest, corpus/index digest, dimension, encoding and byte-order identifiers;
- policy version, threshold/calibration artifact digest and calibration-domain identifier;
- provenance and version/validity metadata for the candidate;
- which independent checks ran and their outcomes;
- whether the decision was accepted, tentative, rejected, or abstained;
- source revision, toolchain/build identity, seed and benchmark case ID for test receipts.

Do not store sensitive raw queries by default when digests and controlled references are sufficient. A digest proves byte identity, not semantic truth, freshness, authorization, or the legitimacy of the source. Those are separate evidence claims.

## 4. Threat and ambiguity model

The evaluation corpus must include at least these distinct cases:

1. Known record, clean cue.
2. Known record, cue corrupted at fixed rates.
3. Absent value sampled independently from known values.
4. Absent value deliberately close to a stored value.
5. Same-byte observational twins whose external task labels differ; test that the vector-only baseline cannot resolve the case and that the system either uses declared side information or abstains.
6. Distinct records sharing constituents or overlapping heavily after bundling.
7. Stale, superseded, revoked, deleted or version-mismatched records.
8. Query from a source distribution or encoding version not represented during calibration.
9. Ties, duplicate vectors, corrupted metadata, missing provenance, and malformed evidence.
10. Empty and one-element corpus; non-power-of-two sizes; deterministic repeated runs.

Split train/calibration/test partitions by underlying entity and generation seed—not only by query row—so near-duplicate derivations of the same fact cannot leak across partitions. Freeze calibration before held-out evaluation. Include independently produced fixtures or an independently implemented oracle.

## 5. Benchmark protocol

### Search correctness

Compare optimized top-1 IDs and distances to an independent scalar Hamming oracle. Specify byte order, dimension, candidate IDs, duplicate handling and the lowest-ID tie rule before timing. All optimized code paths must pass the same fixtures.

### Semantic retrieval

Report target Hit@1 / Hit@k, target rank, top-1/top-2 similarity margin, coverage and results by bundle arity, corpus size, overlap and corruption level. Do not describe a semantic miss as an exact-search defect when the optimized search agrees with the oracle.

### Acceptance and abstention

Report false-accept and false-reject rates for each negative profile separately; an aggregate number must not hide near-miss performance. Also report coverage, selective risk versus coverage, calibration plots and a calibration metric such as Brier score or expected calibration error where meaningful. Report confidence intervals and the tested operating domain. Compare score-only thresholding, margin-aware rules, provenance/ID-aware verification and explicit abstention.

For zero observed false accepts, a useful approximate 95% upper bound is 3/n under independent identically distributed test cases (the rule of three). Thus zero errors in 30,000 cases only supports an approximate <1e-4 bound for that tested distribution; 300,000 zero-error cases are needed for a similar <1e-5 bound. Correlated samples or distribution shift invalidate the simple interpretation. These are evidence thresholds, not universal safety guarantees.

### Runtime and resource costs

Report p50/p95/p99 latency, throughput, index-build time, query preparation time, memory footprint including sidecars, and setup-inclusive versus resident-index time. For energy, measure joules/query and joules/successful semantic retrieval on named hardware with calibrated instrumentation and a declared system boundary. Corpus bytes visited or CPU TDP are not energy measurements.

## 6. Industry comparison

Use identical packed bytes, candidate IDs, queries, tie policy and timing boundaries for:

- Symthaea exhaustive exact scan;
- FAISS IndexBinaryFlat as an exhaustive Hamming baseline;
- Symthaea early-abandon, pivot-bound, tiled and other exact variants;
- FAISS indexed/approximate alternatives on a separate recall/latency frontier;
- metadata-aware second-stage verification;
- a score-only threshold baseline as a deliberately limited control.

For approximate methods, state recall relative to the exact oracle and also report task-level semantic hit@k and acceptance errors. Do not compare the exact path's speed with an approximate path while presenting both as exact. Include the same query batch sizes, warm/cold-cache condition, resident/storage boundary, build amortization, CPU features, compiler flags and machine identity.

External protocol references:
- FAISS binary indexes: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes
- MLPerf Tiny benchmark principles, fixed workloads, latency and energy rules: https://mlcommons.org/working-groups/benchmarks/tiny/

## 7. Go/no-go criteria

Do not promote the design to a production identity claim until:

- the search result matches the independent exact oracle for all fixed correctness fixtures;
- task mode and identity policy are explicit at the call boundary;
- a score-only ambiguous case produces an ambiguity/abstention result when required side evidence is absent;
- requested-ID and version checks fail closed on wrong, stale, revoked or missing metadata;
- held-out false accepts/rejects, coverage and uncertainty are published per query profile;
- no threshold is retuned on the test partition;
- same-byte FAISS baselines are recorded for performance claims;
- device-energy claims are supported by direct measurements;
- the output receipt binds the candidate, evidence, policy and source revision.

A speed improvement is useful only if the same task-level quality and decision contract are preserved. A better semantic representation may matter more than a faster kernel when the target rank or margin is poor.

## 8. Phased execution

**Phase A — Contract fixtures:** implement task modes, decision states and same-byte ambiguity/abstention tests; no algorithm changes yet.

**Phase B — Held-out corpus:** add larger multi-seed sets, entity-disjoint partitions, hard negatives, overlapping records, stale versions and missing provenance; emit machine-readable raw results.

**Phase C — Verifier comparison:** test requested-ID/digest/provenance checks and independent second-stage verifiers versus threshold-only controls.

**Phase D — Matched baselines:** compare exact scan and FAISS IndexBinaryFlat, then indexed alternatives, with build and query costs separated.

**Phase E — Target-device qualification:** repeat the frozen workload on mobile/edge hardware; measure thermally sustained latency and joules per successful result.

## 9. Evidence status

This is a design specification, not a test report. The PR #8510 benchmark sources still need exact-head Rust compilation and execution. Static file presence, Python pilots and workflow states that are skipped or in progress are not passes. No SOTA, Rust runtime, device-energy or production safety claim follows from this document.


## 10. Initial Rust contract prototype

A first implementation now exists at `crates/core/symthaea-core/src/hdc/retrieval_contract.rs` and is exposed through the HDC module.

It adds explicit modes for exact identity, semantic retrieval and open-set retrieval, plus decision outcomes for verified identity, accepted semantic result, tentative candidate, no match and abstention. The contract includes unit tests for:

- high similarity failing to override a requested-ID mismatch;
- exact version mismatch and missing version;
- missing or failed content-integrity evidence;
- semantic acceptance requiring an independent relevance-verifier status;
- open-set low-score rejection versus high-score/low-margin abstention;
- NaN score and invalid policy rejection;
- empty candidate results.

This module is deliberately a policy boundary, not a cryptographic implementation. The EvidenceStatus values must come from real upstream verifiers. It does not currently replace existing StatisticalRetriever call sites or prove compatibility with every consumer.

**Validation status:** source and tests have been committed, but this environment has no Cargo, rustc or Nix executable, so the crate has not been compiled and the tests have not run here. The observed GitHub Actions runs for the branch ended as skipped; they are not passes. Treat API/type correctness as pending until exact-head Rust checks execute.

### Next implementation gate

1. Run `cargo test -p symthaea-core retrieval_contract` and `cargo clippy -p symthaea-core --all-targets -- -D warnings` at the exact PR head.
2. Add property-based tests for all EvidenceStatus combinations and boundary values.
3. Add a deterministic benchmark corpus that includes bit-identical observations with different external identity labels; verify that score-only input yields CandidateOnly/Abstain, never ExactIdentityVerified.
4. Only after the API is stable, adapt one real caller behind an opt-in path and measure false acceptance, abstention/coverage and latency against its existing policy.
5. Add receipt serialization and bind the decision to policy, corpus/index digest and source revision before production adoption.
