# Current industry baseline facts for the Assurance Gap Challenge

**Snapshot:** 2026-11-25. This document separates vendor-reported performance, documented execution semantics, standards, and local SFEB measurements. It is not an end-to-end benchmark result.

## DBOS: durable-workflow throughput

DBOS's October 28, 2026 vendor post reports more than 1 million durable workflows/second, scaling from about 20,000/s on one Postgres server to 1 million/s across 50 shards.

Published test configuration:
- 50 RDS Postgres `db.m7i.4xlarge` shards, each 16 cores and 64 GB RAM.
- 48 worker processes per shard, 2,400 total at 50 shards.
- 20,000 workflow enqueues per shard per second.
- Workflows are described as simple no-ops with no steps, distributed evenly over shard keys.

**Interpretation:** this is a valuable W0 orchestration/queue ceiling and a strong durability/scaling result. It does not measure an external action, destination idempotency, independent state readback, COSE/SCITT receipts, witnesses or offline audit. It must not be compared to W4–W6 throughput.

Source: https://www.dbos.dev/blog/one-million-workflows

## Temporal: retry and effect semantics

Temporal's Activity documentation states that Activities can be retried and executed more than once. A Workflow may observe an Activity as completed exactly once while the Activity's code—and partial side effects—may have executed more than once. Temporal recommends idempotent business Activities and describes using a stable idempotency key checked by the destination service.

**Interpretation:** compare Temporal on a matched W2/W4 workload using stable operation IDs and sink-side deduplication. A Temporal completion record alone does not establish the number or truth of external sink effects.

Sources:
- https://docs.temporal.io/activity-definition
- https://docs.temporal.io/activities

## Rekor / Sigstore: transparency infrastructure

Sigstore describes Rekor as a verifiable transparency log for signed metadata, with entry inclusion proofs and integrity/consistency monitoring. Its documentation points to monitor tooling and omniwitness. The public instance advertises a 99.5% availability SLO.

**Interpretation:** do not build a replacement log or witness network merely to reproduce established capabilities. SFEB's comparison should use Rekor plus its available monitoring path as a W5 building block, then ask whether the external effect and capture completeness are independently established. Log availability and a valid inclusion proof are not sink-effect truth.

Sources:
- https://docs.sigstore.dev/logging/overview/
- https://docs.sigstore.dev/about/security/

## IETF standard foundations

- RFC 9942 (June 2026, Proposed Standard) defines COSE receipts for VDS proofs including inclusion and consistency proofs.
- RFC 9943 defines the SCITT transparency architecture, requires COSE receipts, transparent registration-policy/trust-anchor history, and a VDS with append-only, non-equivocation and replayability properties.

These are foundations for interoperability, not new SFEB formats.

Sources:
- https://www.rfc-editor.org/rfc/rfc9942
- https://www.rfc-editor.org/rfc/rfc9943

## Agent evidence drafts

Use the current interoperability map for AEE -02, SCITT Agent Execution -01 and Vaara Receipt -15. Distinguish each draft's profile specification, test vectors, author-reported prototype/implementation status, and independently reproduced deployments. Do not call an Internet-Draft a published standard.

## What the current SFEB numbers mean

MON-008's local five-run median verifier path is 132.729 µs, 202.110 µs and 292.231 µs for one, two and three witnesses. It measures in-process signature verification, quorum processing and local state checks only. It excludes network, durable storage, COSE/SCITT registration, sink mutation, independent readback and offline audit.

Those microbenchmarks cannot be ranked against DBOS's W0 no-op workflow throughput, and they do not establish SFEB is faster than Temporal, DBOS, Rekor or SCITT. No matched W0–W6 industry run has yet been completed.

## Competitive test matrix

| Claim | Current relevant baseline | SFEB evidence today | Next fair test |
|---|---|---|---|
| Durable workflow throughput | DBOS W0, Temporal orchestration | None end-to-end | Reproduce W0 on same hardware and durability settings |
| At-least-once external execution | Temporal Activity retry semantics | MON-007 stable operation ID / sink predicate model | Inject lost ACK after commit and retry same ID at sink |
| Transparent signed statement | RFC 9942/9943 and Rekor | MON-007 RFC 9162 proof verifier; MON-008 custom test checkpoint only | COSE/SCITT verifier adapter + official vectors |
| Two-sided caller/service trace | AEE -02 profile and vectors | Not implemented yet | Pin profile revision, verify linked bundle and negative controls |
| Decision/execution evidence pair | Vaara Receipt -15 | MON-007 action/authority/sink/readback semantic core, not Vaara adapter | Independent implementation of profile vectors |
| Sink truth | Native conditional-write control and separate principal | Semantic model only; no production sink driver | Real test sink, actual atomic fencing/dedupe, signed sink receipt/readback |
| Omission/completeness | AEE harness/serving party evidence + remote telemetry composition | Benchmark protocol only | Out-of-band and suppressed-record red-team corpus |
| Operator-independent audit | Offline SCITT/AEE/Vaara bundles | Preflight bundle; not complete offline effect auditor | Remove original DB/credentials and verify only from portable evidence |

## Performance reporting contract

Report W0, W1 and W2–W6 separately. Every comparison must publish hardware, worker/shard counts, database durability and isolation, payload distribution, concurrency, retry policy, sink semantics, proof/receipt type, witness count and independence, readback path, fault seeds, exact versions, p50/p95/p99 and raw samples.

The primary success measure is independently audited external effects/s at a declared assurance tier, accompanied by false accepts, duplicate effects, indeterminate age, observation coverage and cost per qualified effect. A faster incomplete path is not a better assurance system.
