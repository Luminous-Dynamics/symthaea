# KAMA-PRIV-001B — Dependency-Aware Intimate Memory Invalidation

Status: source-design candidate
Issue: #4899
Parent: KAMA-EVAL-001B / #4886
Authority: privacy dependency bookkeeping only; **no consent, clinical, content, or physical authority**

## Core theorem

```text
source retracted
!= every derivative erased everywhere
```

A privacy-safe memory system therefore needs explicit derivation edges and evidence-bearing retraction outcomes rather than a boolean `deleted=true` on one store.

## Metadata-only graph

`IntimateMemoryPrivacyGraphV1` tracks artifact identity and dependency metadata, not raw intimate content.

Artifact kinds include transcript, session summary, explicit/inferred preference, psychology evidence, reflective memory, fantasy history, embedding/index, and external export.

Each artifact declares:

- retention class;
- dependency completeness;
- derivation policy;
- whether it can materially reconstruct sensitive source content;
- exact source artifact IDs;
- optional durability-authorization evidence;
- optional export receipt evidence.

## Admission boundaries

A derived artifact with complete dependency metadata must reference already-active sources. This makes the graph acyclic by construction order.

An ephemeral source cannot silently become a durable derived artifact:

```text
ephemeral source
+ durable derivative
+ no durability authorization evidence
-> REJECT
```

Unknown dependency metadata is allowed only as an explicit fail-closed state and is never retrievable.

## Retraction actions

When a governed source is retracted, descendants resolve to explicit actions:

- `DeleteLocal` — remove from active retrieval;
- `RecomputeRequired` — remove until recomputed without the retracted source;
- `RetainIndependentBasis` — keep only when another explicit active source provides independent basis;
- `ExternalExportNotice` — remove local active metadata while reporting that external deletion cannot be proven here;
- `BlockUnknownDependency` — remove an artifact whose dependency completeness is unknown.

Reconstructive derivatives always collapse to local deletion rather than surviving on a weaker policy.

## Tombstones

Artifact IDs remain in `known_ids` after retraction. This prevents a sensitive item from being silently reintroduced under the same identity after its payload was removed.

Repeated retraction is idempotent and returns `AlreadyRetracted` with no new actions.

## Fan-out and fan-in

Retraction propagates transitively through fan-out chains such as:

```text
transcript -> summary -> embedding
```

For fan-in, `MayRetainWithIndependentBasis` can preserve a non-reconstructive derivative only when another declared active source remains. The removed source edge is deleted from the retained artifact.

## External exports

`ExternalExportRequiresNotice` requires an opaque export receipt reference and produces `external_deletion_unproven=true` if a source is later retracted.

This is intentionally more conservative than claiming an exported copy was erased.

## Tests

Source + integration fixtures cover:

- transitive reconstructive deletion;
- fan-in independent-basis retention;
- ephemeral-to-durable promotion gating;
- external-export notice semantics;
- unknown-dependency fail-close behavior;
- idempotent repeated retraction;
- artifact-ID tombstoning/replay resistance.

## Nonclaims

This graph does not prove deletion from third-party systems, user-made copies, external backups, previously authorized exports, or any system outside the governed dependency domain. It also does not itself execute recomputation; `RecomputeRequired` removes retrieval eligibility until a separately governed derivation creates a new artifact identity.
