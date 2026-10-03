# RFC: Sovereign State Compiler

**Status:** Architecture exploration / naming RFC  
**Branch:** `rfc/sovereign-state-compiler`  
**Date:** 2026-10-03

## Summary

The deployment subsystem currently emerging from Spore/Nixward should become a platform-neutral deployment protocol and plan compiler rather than a larger installer embedded in Spore.

The working name is **Sovereign State Compiler**.

The core transformation is:

`DeploymentIntent -> Sovereign Deployment IR -> Target-Native Plan -> Authorized Execution -> Observed Verification -> Receipt`

The abstraction is OS-neutral. Execution is target-specific.

## Why not "Sovereign Deploy"?

"Sovereign Deploy" is clear, but it describes only the most visible action. The proposed system also discovers targets, negotiates capabilities, binds artifacts and authority, plans installation/configuration/update/rollback/recovery, and records evidence.

The system is therefore closer to a compiler than to an installer.

"State" is important because the desired outcome is a target state, not a command sequence.

"Compiler" is important because the system lowers a platform-neutral desired state into a platform-specific execution plan.

## Naming findings

Research on 2026-10-03 found substantial use of "Sovereign Deployment", including products/services from multiple vendors. "Sovereign Forge" is also already in active use by unrelated software/infrastructure organizations. "StateWeave" is already used by multiple agent-state projects. These names should therefore not be adopted for this project.

"Sovereign State Compiler" has a clearer semantic relationship to the architecture and did not surface an obvious direct software collision in the initial search.

This is a naming decision for the architecture, not a trademark clearance.

## Architectural boundary

### Symthaea

Owns cognition, planning, interpretation, and optional generation of deployment intent.

It does not receive privileged execution authority merely because it is intelligent or because a cognitive metric crosses a threshold.

### Symthaea-WASM

Provides a first-class portable Symthaea runtime for browser/WASM environments.

It may inspect a target profile and construct a deployment intent, but it should not contain arbitrary privileged installers.

### Spore

Remains a minimal/portable Symthaea seed and browser/edge embodiment.

Spore should not be the universal deployment implementation.

### Sovereign State Compiler

Owns the neutral deployment model:

- target identity
- capabilities
- desired state
- artifacts and dependencies
- preconditions
- authorization
- execution plan
- verification policy
- rollback/recovery policy
- provenance and receipts

### Nixward

Becomes the NixOS/Nix-specific target intelligence and adapter.

Nixward can lower the neutral plan into NixOS-native mechanisms such as flakes, disko, nixos-rebuild, generations, and Nix-specific verification.

### Xenia

Owns consented remote transport/session establishment where remote execution is required.

### nix-signature-policy

Owns artifact trust/admission and cryptographic signature policy.

It is not the deployment planner.

## Target model

The primary portability boundary must be capability negotiation, not an OS enum.

Example capabilities:

- ObserveHardware
- InstallApplication
- RemoveApplication
- ConfigureSystem
- UpdateSystem
- Rollback
- Reboot
- ReplaceOS
- ModifyBootChain
- ConfigureSecureBoot
- EncryptStorage
- CreateRecoveryEnvironment
- RemoteExecution
- AttestState

A target advertises the capabilities it actually exposes. The compiler may only produce plans within that capability envelope.

This allows:

- NixOS to expose broad local control.
- Linux distributions to expose agent/package/configuration control.
- Windows to use native image/provisioning/DSC/MDM mechanisms.
- macOS/iOS/iPadOS to use Apple's device-management/declarative mechanisms where authorized.
- Android to use Android Enterprise / Android Management API mechanisms where authorized.
- Embedded/firmware targets to expose only the signed update and recovery operations they support.

## Deployment IR

The core object should be a canonical, serializable deployment intent/plan rather than a shell command.

Conceptual fields:

`plan_digest`
`target_identity`
`artifact_manifest`
`desired_state`
`capability_requirements`
`authorization_evidence`
`preconditions`
`actions`
`verification_policy`
`rollback_policy`
`recovery_policy`
`expiry`
`nonce`

Optional environment evidence can include:

`hardware_observation_digest`
`configuration_digest`
`source_bundle_digest`
`flake_lock_digest`
`secure_boot_intent`
`credential_policy`

## Security invariant

UI state, cognitive state, or inferred confidence must never become deployment authority.

The privileged executor should accept an authorized deployment plan whose authority is bound to:

1. target identity;
2. exact artifact/configuration identity;
3. allowed capabilities;
4. authorization evidence;
5. policy;
6. precondition state;
7. validity window.

A value such as Phi may inform planning or human-facing readiness, but cannot by itself authorize destructive mutation.

## Lifecycle

The system should treat deployment as a lifecycle:

`Discover -> Plan -> Authorize -> Stage -> Execute -> Verify -> Commit -> Maintain -> Rollback/Recover`

This covers:

- fresh OS installation
- application installation
- configuration
- updates
- migrations
- rollback
- disaster recovery
- reconstitution
- remote deployment
- offline/air-gapped deployment

## Immediate extraction from current code

The first NixOS adapter proof is now separated from the neutral compiler as
`sovereign-state-compiler-nix`. It lowers an explicit NixOS state vocabulary
into abstract lifecycle steps and rejects unknown state properties rather than
silently discarding them. It intentionally has no Nixward, shell, transport, or
privileged-execution dependency.

The existing Spore/Nixward code suggests the following ownership transfer:

### Move toward Sovereign State Compiler

- typed deployment intent/plan
- capability model
- authorization envelope
- generic verification/rollback contract
- evidence/receipt model
- generic target/session abstractions
- lifecycle state machine

### Keep in Nixward

- NixOS observation
- Nix parsing
- NixOS command definitions
- flake/config generation
- NixOS generation handling
- NixOS rollback implementation
- Nix-specific preconditions and verification

### Remove from Spore over time

- generic SSH/WebSocket deployment relay
- generic installer orchestration
- NixOS-specific provisioning
- Secure Boot orchestration
- PXE orchestration
- disk installation logic
- arbitrary shell transport

Spore can retain browser-facing configuration UX and portable cognitive runtime interfaces.

## First implementation tranche

1. Add neutral deployment types in a new crate.
2. Add capability negotiation.
3. Add immutable deployment-plan digesting.
4. Add authorization-evidence binding.
5. Define target-adapter trait.
6. Implement a NixOS adapter backed by existing Nixward logic.
7. Add deterministic plan vectors.
8. Add negative authorization tests.
9. Move one Spore installation path through the neutral plan without changing behavior.
10. Only after the neutral path is proven, extract the remaining relay/installer code.

## Success criterion

A deployment of Symthaea and a deployment of an unrelated application should be representable by the same neutral model even when the target execution mechanisms are completely different.

The compiler should be able to say:

> "This target can realize these requested state transitions, using these native mechanisms, under this authority, with this verification and recovery contract."

It should not need to understand the application semantically beyond the artifact
and declared requirements.

The NixOS adapter is the first proof of this principle: the same neutral intent,
artifact, capability, authorization, and verification contract can describe a
system rebuild and an unrelated application installation. The adapter chooses
the NixOS lowering semantics while the core remains oblivious to Nix.

## Naming recommendation

Recommended architecture/product name:

**Sovereign State Compiler**

Likely repository name:

`sovereign-state-compiler`

Possible CLI:

`ssc`

Possible protocol namespace:

`ssc/v1`

"Sovereign Deploy" can remain an informal descriptive phrase during migration, but the architecture should avoid freezing the shorter name into public APIs if a more exact name is accepted.


## Interoperability and canonicalization

The v0.1 Rust crate uses deterministic serde/JSON serialization for its internal
preview digests. This is sufficient to bind objects inside the same
implementation, but it is deliberately **not** presented as a cross-language
cryptographic canonicalization standard.

Before external signing/interoperability is stabilized, the protocol should
adopt an explicit canonical representation. JSON Canonicalization Scheme
(JCS, RFC 8785) is a candidate because it defines deterministic property
ordering and primitive serialization for hash/signature operations. The choice
should be frozen in a protocol version rather than inferred from a library's
serializer.

Supply-chain evidence should likewise use established attestation/artifact
formats by reference where practical. in-toto provides a stable attestation
framework; SLSA defines provenance concepts; OCI already models
content-addressable artifacts and platform-specific image variants. The
compiler should bind these references into its plan rather than inventing
parallel provenance semantics.

## New invariants in the v0.1 prototype

Authorization validation now requires:

- a non-empty authority identifier;
- a non-empty nonce;
- intent digest equality;
- target-profile digest equality;
- exact compiled-plan digest equality;
- a valid time window;
- every granted capability to be supported by the target;
- every required/step capability to be both supported and granted.

This deliberately separates the four sets that matter:

`required ∩ granted ∩ supported`

with the additional invariant:

`granted ⊆ supported`.

The first relationship prevents privilege escalation through an unsupported or
unapproved requested operation. The second prevents an authorization object
from claiming authority the target cannot actually realize.

## Platform evidence model

Target discovery should eventually return more than an OS label. The target
profile needs to evolve toward a structured capability and evidence document
containing target identity, platform family, architecture, management channel,
available operations, and freshness/lifecycle metadata.

This is important because modern platforms increasingly expose declarative
desired-state and status channels rather than unrestricted command execution.
Microsoft DSC 3 is explicitly declarative and cross-platform; Apple exposes
device-management capabilities and status reports; Android Management API
represents managed-device configuration as policy. These should map into the
same neutral capability model without pretending their authority surfaces are
identical.

References:

- RFC 8785: JSON Canonicalization Scheme.
- in-toto specifications and Attestation Framework.
- SLSA provenance specification.
- OCI Image Specification.
- Microsoft Desired State Configuration.
- Apple Device Management declarations/status.
- Android Management API policies.


## Resource binding

Capabilities alone are insufficient for operations that mutate a concrete
resource. An intent therefore carries required resource identities, and the
fresh target snapshot advertises the resources actually observed.

Examples include:

- a specific block device selected for OS replacement or encryption;
- a specific firmware slot;
- a specific VM or hypervisor resource;
- a specific managed application/package identity.

The compiler validates:

`required_resources ⊆ observed_resources`

and the compiled plan digest binds the complete resource set. This prevents a
generic capability such as `EncryptStorage` from silently widening into
authority over an arbitrary storage device.

For destructive operations, the executor should re-observe resource identity
immediately before mutation and compare that observation to the authorized
snapshot, rather than trusting a UI-selected device path.


## Authorization lifetime

An intent expiry is an upper bound on authorization. Authorization MUST NOT
extend beyond the intent's expiry, and an executor MUST reject an intent that
has already expired.

This makes the authority window monotone:

`execution_now <= authorization_expiry <= intent_expiry`

when both expiries are present.

Short-lived authorization is preferred for privileged operations; persistent
management should be implemented as repeated, freshly authorized reconciliation
rather than an indefinitely valid mutation grant.
