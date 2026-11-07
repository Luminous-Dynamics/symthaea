# QVF-R — Query-Visible Frontier Replay v0.1

## Purpose

QVF-R is a synthetic qualification lane for historical retrieval. It asks whether an exact source-derived representation was query-visible to a particular serving replica by the requested knowledge cutoff.

It does not create a new temporal ledger, source registry, execution receipt, evidence graph, rights registry, or benchmark runner. It joins references owned by existing contracts:

- Symthaea #10651: immutable, anchor-preserving retrieval plan.
- Symthaea #9075 and Mycelix #2631: valid-time / known-at and evidence-frontier semantics.
- Symthaea #10816: receipt-backed plan / execution / index-visibility join.
- Symthaea #5482 and Mycelix #2779: read-only execution receipt interpretation.
- Mycelix #2825 / #2831 / #2832: capture/transform provenance, actual coverage and authority chain.
- Symthaea #9492: Temporal Error Debt (TED) consumes the resulting answer trajectory; QVF-R does not redefine TED.
- Symthaea #10719: one-shot / fixed-iteration / adaptive-agent causal attribution; QVF-R supplies the availability conditions, not a second agency runner.

## The invariant

A source's server receipt time is not its embedding time, index commit time, or serving-replica visibility time.

For a representation R, index revision I, replica P, and historical cutoff K:

- R must have an exact materialization observation no later than K.
- The exact index revision containing R must be shown visible to P no later than K.
- The requested valid-time and known-at constraints must independently pass.
- The current rights decision must permit the payload at K.
- Every derived representation must inherit parent source-family and rights dependencies. A revoked parent can invalidate the derived payload even if the bridge fact remains physically present in a stale vector index.

Missing visibility evidence means Unknown, not visible and not not-visible. An index build timestamp or source receipt cannot substitute for a serving-replica visibility observation.

## Files

- qvf-r-case.schema.json — closed Draft 2020-12 schema for a synthetic replay case.
- synthetic-cases.jsonl — deterministic examples. All timestamps and references are synthetic test data.
- EVALUATION_PROTOCOL.md — preregistered benchmark tracks, metric denominators, ablations, SOTA targets, statistical design and anti-contamination rules.
- This README — semantics, ownership and claim ceiling.

## Proposition-version and source-span controls

The fixture also models an overlapping amendment:
- v1 and v2 represent the same logical proposition but have distinct canonical version refs and exact source-span refs.
- The v2 amendment has a typed replacement-relation reference to v1.
- Both versions may be present and visible in the same index revision.
- A pre-amendment query resolves to v1 even when v2 is already indexed and visible.
- A post-amendment query resolves to v2; the semantically similar old clause must not be substituted.
- A query whose known-at cutoff precedes v2 materialization cannot retrieve v2, even when its valid-time resolution points to the amended clause.

The benchmark carries opaque canonical statement/version, relation, temporal-resolution and source-span references. It does not define a second proposition graph, version ledger or selector ontology. Exact span contents and coordinate semantics remain owned by the source/provenance contracts.

## Included negative controls

1. Source receipt predates the query, but embedding materialization follows the cutoff.
2. Index revision is materialized, but the selected serving replica is updated after the cutoff.
3. A representation is eligible only after its replica-visibility observation.
4. A different replica has no visibility observation; result must remain Unknown.
5. A derived bridge fact inherits two source families; revocation of one parent makes the derived payload ineligible under the fixture's explicit rights policy.
6. An amended clause is present and visible but is not the version selected by the requested valid-time resolution.
7. The pre-amendment clause remains the correct result even when the amended version is already indexed and visible.
8. The amended clause is temporally selected but not yet materialized by the query's known-at cutoff.

## Industry adapter baselines

QVF-R is provider-neutral. Product comparisons should record observable capabilities and opaque internals separately rather than assuming a provider lacks an unexposed feature.

- **Azure AI Search:** hybrid BM25/vector retrieval with reciprocal-rank fusion and a separate semantic-reranker score. Its vector filter modes have different recall/cost behavior: prefilter can maximize filtered recall; postfilter and strict postfilter can miss eligible records under selective filters or small k. Benchmark each filter mode and record eligible-candidate false-negative rates rather than treating filter success as equivalent across modes. See [hybrid ranking](https://learn.microsoft.com/azure/search/hybrid-search-ranking), [filter modes](https://learn.microsoft.com/en-us/azure/search/vector-search-filters), and [hybrid search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview).
- **Amazon Bedrock Knowledge Bases:** managed retrieval provides hybrid search, metadata filtering, reranking and agentic multi-query retrieval with trace events. Document-level ACL filtering is supported for listed connectors, with documented exceptions. QVF-R should test which exact request/trace/ACL/index-availability evidence is actually exposed to an adapter; do not infer historical replica visibility from a successful Retrieve response. See [Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html), [managed retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html), and [AgenticRetrieveStream](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-retrieval.html).
- **Google Vertex AI RAG Engine with Vector Search 2.0:** the documented managed RAG collection does not automatically have an index; unindexed retrieval uses brute-force KNN. The documented RAG collection schema has no sparse-vector field (so hybrid search is unavailable for that corpus path), and its v1 index replica defaults are two. Metadata filtering is resolved to file IDs. QVF-R should compare collection, index and serving configuration separately, and treat provider-internal historical visibility as Unknown unless exposed. See [RAG Engine Vector Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/use-rag-managed-vertex-ai-vector-search).

These are documentation-backed product configurations, not claims about private vendor internals or a complete product-wide capability inventory.

## Known v0.1 gap: requested constraints vs applied constraints

A retrieval plan's declared constraints and a committed request/filter digest establish what was intended or serialized; they do not, by themselves, prove how the provider executed those constraints.

A follow-on adapter observation should bind, per retrieval step:
- exact planned constraint refs;
- exact post-adaptation request commitment;
- provider/adapter filter mode: pre-filter, post-filter, strict post-filter, client-side validation, provider-managed opaque, or unknown;
- which constraints were actually applied and the evidence supporting that observation;
- whether a client-side verifier checked every returned candidate against the required constraints.

Do not claim server-side filtering merely because the request contains a filter. When filter execution is opaque, preserve that limitation. Client-side checking can reject an ineligible returned item, but it cannot prove that the provider returned every eligible item.

Measure filtered candidate recall and false-negative rate against a frozen eligible corpus, stratified by filter selectivity, k, shard/partition count and provider filter mode. Keep this separate from temporal-version correctness and query-visible precision.

## Required next validator checks

The schema is structural only. An offline semantic validator must additionally reject:

- query execution before the declared known-at cutoff or mismatched query/index/replica identity;
- required planned constraints not represented in the exact adapted request or a typed application observation;
- opaque provider-side filters incorrectly treated as exhaustive or independently verified;
- visibility evidence for a different representation, revision or serving replica;
- materialization after cutoff;
- rights denial/revocation effective at or before cutoff;
- active derived payload when any required parent dependency is invalidated or rights-denied;
- missing or duplicated event identities, impossible ordering, non-monotonic visibility watermarks and malformed SHA-256 commitments;
- an expected EligibleForRetrieval result if any one of the time, visibility, rights or authorization gates is missing.

These rules should be implemented inside the existing knowledge/evidence benchmark harness after the branch containing that harness is refreshed onto current main. This v0.1 tranche intentionally adds no new runner and makes no CI or runtime claim.

## Metrics to report separately

- query-visible precision/recall at K;
- filtered candidate recall and filter false-negative rate, stratified by provider filter mode/selectivity/k;
- proposition-version accuracy under overlapping amendments;
- exact source-span locator precision/recall;
- obsolete-clause substitution rate and amendment propagation correctness;
- post-cutoff visibility leakage;
- unknown-visibility rate;
- stale/revoked derived-payload leakage;
- per-required-period Temporal Coverage@k;
- source-family recall (never URL count);
- replica/index freshness lag;
- deterministic replay agreement;
- p50/p95 latency and cost when a real runtime is evaluated.

Do not combine these into one groundedness score. Do not treat any result as world completeness or as a claim-evidence warrant.

## Claim ceiling

A passing structural/schema check establishes only that the synthetic case has the declared shape. It does not establish real serving telemetry, historical system knowledge, source authenticity, truth, semantic entailment, provenance independence, rights compliance, retrieval performance, or superiority to SOTA.
