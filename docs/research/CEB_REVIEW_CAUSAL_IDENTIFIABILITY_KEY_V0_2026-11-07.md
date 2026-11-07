# CEB Review Causal-Identifiability Development Challenge Key V0 — 2026-11-07

**Status:** Public development answer key. Not secure, sealed or eligible for holdout use.
**Companion challenge:** [Causal-Identifiability Development Challenge V0](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md).
**Blinding:** Reviewers must lock independent first-pass forms before coordinators expose this key. Because the key is public, this is procedural concealment only.

## 1. General adjudication rule

Do not demand one exact graph. Accept a different model decomposition if it preserves the evidence-compatible alternatives, their different predictions, the appropriate claim disposition and a genuinely discriminating minimum evidence request.

A plausible single model is not identified merely because it is simpler, more common or preferred by the reviewer. Conversely, do not reward unbounded “anything is possible” skepticism: material alternatives must be consistent with the supplied source/evidence and differ on the actual query.

## 2. CID-S01 — omitted authorization helper

**Expected disposition:** non_identifiable for the claim that tenant/resource authority is enforced.

The visible source establishes that this handler calls a helper with actor, tenant and resource arguments before this particular store mutation. It does not establish what the helper checks, what backing state it uses, or whether other mutation callers exist.

**Material alternatives:** Completion A performs a tenant/resource-scoped current policy check. Completion B checks only a global actor allowlist. Both fit the visible caller and produce different security properties.

**Disagreement witness:** a same-user, same-resource request under two tenants with different policies. A tenant-scoped check may return different decisions; a global user allowlist may return the same decision.

**Minimum additional evidence:** inspect the exact auth::check definition, backing policy/cache implementation, policy contract and in-scope callers of store.mutate at the exact source revision. A helper signature is not implementation evidence.

**Claim ceiling:** “This handler gates its mutation on auth::check(actor, tenant, resource).” Do not say “tenant isolation is enforced” or “a tenant bypass exists” from the excerpt alone.

## 3. CID-S02 — configuration-dependent implementation

**Expected disposition:** non_identifiable for the claim that the shipped artifact selects the fail-closed definition.

The source proves two mutually exclusive cfg-selected implementations exist. Without the exact resolved build world and artifact identity, neither implementation can be attributed to the shipped binary.

**Material alternatives:** strict-policy enabled → Deny; strict-policy not enabled → Allow.

**Disagreement witness:** inspect the exact Cargo feature resolution and target compilation unit for the artifact under review.

**Minimum additional evidence:** frozen Cargo invocation/lockfile, feature tree or equivalent resolved build metadata, target triple, cfg values, build profile, artifact digest and provenance tying those to the shipped binary. Do not assume default features.

**Claim ceiling:** “The source contains a fail-closed implementation under strict-policy and a fail-open implementation otherwise.” Do not claim which branch is deployed.

## 4. CID-S03 — cache eviction and latent workload

**Expected disposition:** non_identifiable for the claim that eviction caused the request failures.

Temporal precedence alone does not rule out a common cause. The visible request path makes a cache miss return Unavailable, but the monitoring summary does not establish that the evicted key belonged to each failed request or that cache eviction was the reason for the underlying miss.

**Material alternatives:**
- A: the target key is evicted, causing the failing request's cache miss.
- B: latent load/tenant context causes both evictions and independent dependency/queue failures; the observed co-occurrence is confounded.

**Disagreement witness:** compare per-request key identity, eviction lineage and dependency state under matched load; or perform a controlled, authorized cache-retention intervention while holding workload and dependency capacity fixed.

**Minimum additional evidence:** event-level key/tenant/request correlation plus load/queue/dependency state. If needed, a safe matched intervention that changes eviction while preserving the other workload parents. A broad log correlation or a model-generated prior is not sufficient.

**Claim ceiling:** “Evictions and Unavailable errors co-occurred in the observed window.” Do not attribute causation without a discriminating observation/intervention.

## 5. CID-S04 — overdetermination

**Expected disposition:** partially_identified for “at least one sufficient route was active” if the supplied incident record is trusted; non_identifiable for which route produced the observed duplicate.

The two branches are OR alternatives, each with its own gate. Either may produce the same duplicate. The outcome alone does not identify which path executed; both may also have executed.

**Disagreement witness:** observe or safely isolate the two branch conditions and their corresponding apply_transfer call identities. Removing only the legacy retry route leaves replay fallback as a sufficient path.

**Minimum additional evidence:** exact request/config values and an event/trace identity for each application call, or a qualified factorial intervention that blocks each route separately and then both. Do not assume the routes are mutually exclusive.

**Claim ceiling:** “The code contains two independently sufficient in-scope routes under their respective gates.” For the incident, say “at least one route was active” only under the declared trusted incident evidence; do not choose one actual cause or claim a one-route repair is complete.

## 6. CID-S05 — temporal race

**Expected disposition:** non_identifiable for whether this specific incident's worker mutation happened after revocation. A source-level possibility claim can be identified from the supplied control flow: the job carries an enqueue-time boolean and the worker uses it without a current check.

**Material alternatives:**
- Order A: enqueue authorization → worker mutation → revocation.
- Order B: enqueue authorization → revocation → worker mutation using stale boolean.

**Disagreement witness:** a qualified happens-before record for revocation and the worker's mutation boundary, or a synchronization/authorization-epoch contract that makes one ordering impossible.

**Minimum additional evidence:** exact request/job identity, monotonic event ordering or a trusted transaction/queue synchronization guarantee, current authorization epoch and worker consumption record. Wall-clock timestamps without clock/ordering guarantees may be insufficient.

**Claim ceiling:** “The source permits a stale-authorization path if revocation precedes worker use.” Do not say this path occurred in the incident from the supplied packet.

## 7. CID-S06 — bundled repair

**Expected disposition:** non_identifiable for the claim that stable key identity alone caused remediation.

The historical change bundles key identity, retention, queue acknowledgement, retry horizon and store behavior. A later period without observed duplicates is consistent with multiple explanations.

**Material alternatives:**
- A: stable logical operation ID is the decisive correction under the declared retention semantics.
- B: transaction/acknowledgement ordering, durable store behavior or an interaction among the changes is decisive.

**Disagreement witness:** a qualified ablation/factorial set that changes one causal component at a time while holding workload, event delivery and relevant environment constant; or isolated commits whose source and behavioral contract are independently qualified.

**Minimum additional evidence:** disentangle the changed hunks, pin exact before/after trees and build worlds, enumerate alternate redelivery paths, and establish a qualified minimal intervention. If executing those ablations is not authorized, the result remains a source-level attribution gap.

**Claim ceiling:** “The bundled revision is followed by no observed duplicate in the supplied sample.” Do not attribute the outcome to one component or claim that each other change was unnecessary.

## 8. Cross-case common errors

- A helper call's arguments are not proof of the helper's semantics.
- A branch existing in source is not proof it is active in a particular build.
- Temporal precedence is not sufficient to rule out a common cause.
- One observed outcome does not identify one cause under overdetermination.
- Source-level race possibility is not proof of event ordering in an incident.
- A merged repair that changes several mechanisms is not a minimal intervention.
- A request for more evidence is not automatically useful; it must separate at least one material model pair.
- A source-only claim must not be promoted to a runtime-observed claim.

## 9. Adjudicator checklist

For each reviewer, preserve:
1. First-pass disposition and alternatives.
2. Whether each proposed alternative is actually compatible with the supplied evidence.
3. Whether the disagreement witness produces different predictions.
4. Whether the requested evidence is minimal, safe, authorized and correctly scoped.
5. Whether the final claim ceiling is narrower than or equal to the evidence authority.
6. Which unresolved alternatives remain after the requested evidence.

Keep first-pass forms separate. Adjudication may explain disagreement; it must not overwrite original annotations.

## 10. Claim boundary

This key is a synthetic expected-answer guide, not a gold authority for real repositories. It establishes no model score, reliability estimate or production causal finding. If independent reviewers reveal ambiguity in the case itself, preserve that disagreement and version the case rather than forcing a false consensus.
