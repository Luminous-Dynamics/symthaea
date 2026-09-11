# Actuation Enforcement Qualification Invalidation Matrix V1

Status: DESIGN / EVIDENCE CONTRACT ONLY — NOT QUALIFIED

This document complements `continuity-actuation-enforcement-campaign-v1.md` and freezes when a future verifier-qualified actuation-enforcement campaign must stop being reusable.

It does not change runtime authority and does not claim any physical backend is qualified.

## Core theorem

`EvidenceOnceQualified != EvidenceValidForever`.

A campaign may remain relevant only while every identity and environmental assumption declared authoritative by its verifier profile remains compatible with the exact qualified campaign.

## Hard invalidation: full requalification required

The following changes invalidate the qualified enforcement campaign and require a new campaign root rather than a freshness refresh:

- `ExecutionBackendId` changes;
- backend implementation digest changes;
- backend generation changes;
- `ActuationEnforcementBoundaryProfileId` changes;
- enforcement-boundary implementation digest changes;
- one-use-mechanism digest changes;
- enforcement profile generation changes;
- `ActuationInterlockAuthenticationProfileId` changes;
- authentication implementation digest changes;
- authentication root identity changes unexpectedly;
- authentication root epoch changes unless the verifier profile explicitly defines an authenticated root-rotation transition and requires new evidence under the successor root;
- qualification harness implementation digest changes;
- scenario-suite manifest changes;
- environment manifest changes in any verifier-authoritative field;
- topology/dependency manifest changes in any verifier-authoritative field;
- hardware/firmware capability manifest changes in any verifier-authoritative field;
- toolchain/container/Nix realization digest changes;
- the fixed enforcement-obligation schema changes;
- required evidence-basis mapping changes;
- evidence campaign manifest changes or any of the canonical nine record IDs changes;
- verifier profile implementation/semantic version changes in a way that alters admission rules;
- verifier-adoption lineage supersedes the verifier that admitted the campaign and the successor does not explicitly re-admit the exact campaign.

A hard-invalidated campaign must not be resurrected by changing a timestamp or issuing a new freshness challenge.

## Fresh-currentness invalidation: re-attestation may be enough

These changes do not by themselves require rerunning all nine enforcement scenarios, but they prevent immediate reliance until a fresh currentness/adoption theorem re-establishes the same campaign under the unchanged hard identities:

- current-verifier challenge expires/is consumed;
- verifier currentness advances while retaining the same admitted campaign and exact verifier semantics;
- trusted-time epoch advances normally without implementation/root rollback;
- a campaign-validity policy requires periodic currentness confirmation without declaring the evidence stale by age alone.

V1 must not use a magic wall-clock TTL as a substitute for identity/currentness rules.

## Attempt-local invalidation: campaign may remain valid, transition must requalify

These changes invalidate one physical transition attempt without invalidating the underlying backend enforcement campaign:

- owner/operator authority is revoked or superseded;
- distributed transition context changes;
- local subject/target realization changes;
- active LKG selection changes;
- execution journal/trusted epoch advances to a different attempt;
- current actuation fence advances to a newer generation;
- a `Deny` / emergency-stop generation is established;
- one-use permit is consumed;
- resource is temporarily unavailable while implementation identities remain unchanged.

The campaign can still describe the backend's enforcement properties, but a new exact transition must earn its own eligibility and actuation decision.

## Diagnostic-only drift

A verifier profile may explicitly classify fields as diagnostic-only. Changes to those fields do not preserve authority by default; they are diagnostic-only only if the exact verifier profile says so.

Examples that may be diagnostic depending on policy:

- human-readable host/lab labels;
- log storage location;
- UI/report formatting;
- non-semantic comments;
- evidence packaging filename.

A field is not diagnostic merely because it is inconvenient to reproduce.

## Root rotation

Root rotation deserves special treatment:

1. old root currentness must be established;
2. successor root identity/epoch must be authenticated by a separately defined rotation theorem;
3. the verifier must decide whether campaign evidence is portable across the rotation;
4. if portable, a new admission object must bind the old campaign ID to the new root lineage;
5. silent root substitution always invalidates qualification.

## Firmware and microcode

Firmware/microcode changes are hard invalidation whenever they can affect the enforcement boundary, durable fence storage, authentication path, atomicity, crash behavior or device actuation.

Examples include:

- BMC firmware;
- switch/router NOS or ASIC SDK/SAI implementation;
- storage-controller firmware;
- TPM/secure-element firmware;
- NIC/HBA firmware when the boundary depends on it;
- hypervisor/host kernel components implementing the fence;
- database/controller versions that own the resource mutation transaction.

A future verifier may prove a broader compatibility class, but V1 should default to exact digests rather than inferred compatibility.

## Topology and dependency drift

Topology is hard-invalidating when the enforcement theorem depends on where durable state or mutation authority lives.

Examples:

- moving the fence store from local TPM/NV to a remote service;
- changing HA leader/follower ownership of the mutation boundary;
- adding a proxy that can bypass the fence check;
- changing controller quorum or datastore semantics;
- changing out-of-band management path assumptions used by crash/recovery evidence.

## Adversarial tests

The future qualification implementation must reject at least:

- old campaign + new backend implementation with same human-readable name;
- old campaign + new firmware but same backend ID incorrectly reused by an adapter;
- old campaign + successor authentication root without explicit root-rotation admission;
- old campaign + changed toolchain/container realization;
- old campaign + one updated scenario record while retaining the previous campaign ID;
- fresh verifier challenge used to revive a hard-invalidated campaign;
- old verifier admission reused after a semantic verifier upgrade;
- attempt-local owner revocation incorrectly treated as campaign invalidation instead of transition denial;
- temporary resource outage incorrectly causing permanent campaign invalidation;
- diagnostic label changes incorrectly altering qualified identity.

## Relationship to #1550 and #1528

#1550 must bind qualification to this invalidation model through the exact verifier profile. The verifier should expose why a campaign is invalidated and whether the required remedy is full rerun, fresh re-attestation or transition-local requalification.

#1528 remains stronger: even a currently valid campaign only establishes qualified evidence about enforcement. Live actuation still requires proof that the exact resource boundary enforces the newest fence generation at the mutation point.
