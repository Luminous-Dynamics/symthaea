# Symthaea Music Theory Patch Series 21 Plan

**Date:** 2026-07-21  
**Base:** exact Patch Series 20 final tree, to be pinned before authoring  
**Theme:** Post-recovery publication resumption, trust-segment continuity, and mutation-boundary enforcement

## Executive summary

Patch Series 19 made witness policy rotatable and preserved authenticated gossip conflicts. Patch Series 20 added conservative incident attribution, governed quarantine, exceptional recovery, and a recovered policy anchor. The next unresolved operational question is:

> When may ordinary publication safely resume, and how can a third party prove that the resumed catalog did not erase, bypass, or silently cross the incident boundary?

Series 21 introduces an explicit trust-segment and resumption protocol. Recovery chooses an authorized branch and establishes a new witness-policy anchor. Resumption is a separate event requiring fresh evidence produced **after** that anchor.

## Security invariants

1. **Recovery is not resumption.** A recovered anchor remains `AwaitingFreshWitnesses` until a strictly later checkpoint is authenticated by the recovered policy.
2. **No silent trust carryover.** Pre-recovery delegation allowances, witness statements, mirror observations, and policy decisions cannot satisfy post-recovery gates.
3. **Exact branch binding.** The first resumable head must extend the recovery-selected checkpoint through an explicit catalog lineage.
4. **Mutation-time reauthentication.** Publication must verify the exact resumption authorization again at the catalog mutation boundary.
5. **Incident history remains reachable.** Every post-recovery head bundle binds the incident-response package and recovered anchor identities.
6. **Global ordinals never reset.** Recovery creates a new trust segment, not a new history with reusable publication or event ordinals.
7. **Cross-segment status changes are explicit.** Supersession or revocation of a pre-recovery publication requires a separately authenticated cross-segment transition.
8. **Quarantine remains effective.** Active containment that removes quorum capacity blocks readiness rather than being ignored.
9. **Freshness is logical, not temporal.** The crate uses caller-supplied logical epochs and does not claim wall-clock freshness.
10. **Authentication is external.** Algorithms, key custody, enrollment, signer independence, and legal authority remain outside the crate.

## New contracts

### `CalibrationPublicationTrustSegment`

A canonical segment identity derived from:

- selected recovery checkpoint;
- incident-response package SHA-256;
- recovered policy-anchor SHA-256;
- recovered witness-policy epoch and policy SHA-256;
- segment genesis logical epoch;
- prior segment identity when one exists.

The identifier must be content-derived. Operator-provided display labels are excluded.

### `CalibrationPublicationResumptionReadiness`

A deterministic assessment separating:

- recovery package structurally valid;
- recovered anchor structurally valid;
- fresh checkpoint strictly after anchor;
- catalog lineage extends the selected branch;
- recovered-policy witness threshold externally authenticated;
- configured mirror policy satisfied;
- no unresolved authenticated conflict affects the candidate head;
- active quarantine still leaves sufficient witness and observer capacity;
- no pre-recovery authority artifact is being reused;
- ready/not-ready decision and exact blockers.

### `CalibrationPublicationResumptionAuthorization`

An externally authenticated envelope binding:

- trust-segment identity;
- exact incident-response package;
- exact recovered policy anchor;
- exact first fresh checkpoint and catalog head;
- exact recovered witness policy;
- readiness report SHA-256;
- resumption-authority policy identity;
- logical authorization epoch;
- distinct signer envelopes.

Structural validity and external authorization remain separate results.

### `CalibrationPublicationSegmentBridge`

A narrowly scoped authorization for a post-recovery event that changes the status of a pre-recovery publication. It must bind both segment identities, the historical publication record, the exact proposed event, the incident-response package, and a configured bridge-authority policy. Ordinary publication delegation cannot create this bridge.

### `CalibrationPublicationResumedHeadBundle`

A portable package containing:

- current catalog and checkpoint;
- exact post-recovery lineage from selected checkpoint to current head;
- trust segment and recovered anchor;
- incident-response package;
- resumption readiness and authenticated authorization;
- fresh head witnesses under the recovered policy;
- mirror/gossip evidence and conflict set;
- segment-aware publication status proofs;
- mandatory machine-readable limitations.

## Landing order

1. Audit inherited Series 18–20 invariants and freeze prerequisite behavior.
2. Add trust-segment identities and canonical integrity.
3. Bind recovered anchors into segment genesis records.
4. Add post-anchor fresh-checkpoint readiness assessment.
5. Add quarantine-capacity and mirror/conflict readiness gates.
6. Add canonical resumption authorization payloads and envelopes.
7. Authenticate resumption through caller-supplied verifiers.
8. Enforce exact resumption authorization at publication mutation time.
9. Refuse delegation and allowance carryover across segments.
10. Make publication records, events, and status proofs segment-aware.
11. Add authenticated cross-segment status bridges.
12. Build resumed-head bundles and exact branch audit.
13. Add operator tools for readiness, signing, authorization, and bundle export.
14. Append persistence roles without renumbering Series 20.
15. Add adversarial regression corpus.
16. Add end-to-end recovery-to-resumption integration test.
17. Decompose modules and remove panic assumptions.
18. Document, checksum, replay, and package the release.

## Explicit non-goals

Series 21 does not implement distributed consensus, network transport, automated key management, universal mirror coverage, wall-clock timestamping, legal adjudication, or automatic deletion/repair of disputed external copies.
