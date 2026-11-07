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
- This README — semantics, ownership and claim ceiling.

## Included negative controls

1. Source receipt predates the query, but embedding materialization follows the cutoff.
2. Index revision is materialized, but the selected serving replica is updated after the cutoff.
3. A representation is eligible only after its replica-visibility observation.
4. A different replica has no visibility observation; result must remain Unknown.
5. A derived bridge fact inherits two source families; revocation of one parent makes the derived payload ineligible under the fixture's explicit rights policy.

## Required next validator checks

The schema is structural only. An offline semantic validator must additionally reject:

- query execution before the declared known-at cutoff or mismatched query/index/replica identity;
- visibility evidence for a different representation, revision or serving replica;
- materialization after cutoff;
- rights denial/revocation effective at or before cutoff;
- active derived payload when any required parent dependency is invalidated or rights-denied;
- missing or duplicated event identities, impossible ordering, non-monotonic visibility watermarks and malformed SHA-256 commitments;
- an expected EligibleForRetrieval result if any one of the time, visibility, rights or authorization gates is missing.

These rules should be implemented inside the existing knowledge/evidence benchmark harness after the branch containing that harness is refreshed onto current main. This v0.1 tranche intentionally adds no new runner and makes no CI or runtime claim.

## Metrics to report separately

- query-visible precision/recall at K;
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
