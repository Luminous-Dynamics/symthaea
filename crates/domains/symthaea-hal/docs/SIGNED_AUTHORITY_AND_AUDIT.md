# Signed Authority and Audit Operations

## Purpose

The v7 layer separates physical control from operator identity and private-key
custody. `symthaea-hal` defines canonical claims and verification policy, but it
does not store private keys or implement a production signing agent. Deployments
should connect `EvidenceSigner` and `EvidenceVerifier` to an independently
protected agent, TPM, HSM, or operating-system credential service.

## Evidence identity

New v7 authorization and audit artifacts use recursively canonicalized JSON,
explicit domain separation, and SHA-256 digests. Existing calibration, HIL, and
fault-ledger FNV fingerprints remain unchanged so old evidence is not silently
reinterpreted. A migration to stronger legacy fingerprints must be treated as a
new calibration/HIL evidence generation.

## Signed startup seals

A startup seal is first issued only after boot admission passes. It can then be
wrapped in `SignedStartupSeal` using the `startup-seal-signature-v1` domain.
Admission consumers must verify:

1. the trusted key ID and allowed algorithm;
2. signature bytes through the external verifier;
3. issue/expiry bounds and clock skew;
4. the canonical payload digest;
5. startup-seal internal structure and deployment identity.

An unsigned startup seal remains diagnostic evidence only.

## Critical operator actions

`AuthorizationPolicy` supports action-specific approval counts. Recovery,
maintenance clearing, and configuration approval default to two approvals.
Every `OperatorGrant` is bound to:

- one deployment;
- one exact request digest;
- one action;
- one ceremony nonce;
- one operator identity and authority key;
- one short validity interval.

Production policy should require distinct operator identities and distinct
keys. A nonce must never be reused after success or failure.

## Command ingress

Remote producers must not supply Rust `Instant` values. A trusted local ingress
process constructs `ReceivedCommandEnvelope` on receipt. `IngressCommandGuard`
then rejects excessive queue dwell, command floods, implausible sequence jumps,
replay, reordering, and live session takeover.

The ingress process should be local, privilege-separated, resource-limited, and
unable to release the physical output gate directly.

## Clock integrity

`TimeIntegrityGuard` detects control-loop gaps consistent with suspend/resume or
severe scheduler stalls, wall-clock regression, and disagreement between wall
and monotonic progress. Install it on the runtime when signatures, expiry, or
remote command admission are required. A violation is a fail-stop event and
must require the normal reviewed recovery path.

## Supervisor quorum

`QuorumSupervisorNotifier` attempts all configured channels. Heartbeats and
normal states require a configured minimum; terminal `Faulted` and `Stopped`
states can require delivery to every channel. Supervisor notification is
observability, not power removal, and never substitutes for OE/e-stop wiring.

## Safety case

`DeploymentSafetyCase` binds reviewed evidence digests to named assurance
claims and expiry times. A policy should require claims such as:

- `hil.shutdown`;
- `hil.current-decay`;
- `watchdog.independent`;
- `procedure.recovery`;
- `procedure.maintenance-lockout`;
- `operator.dual-control`;
- `configuration.reviewed`.

Run `hal-safety-case-verify` during deployment admission. Missing, future-dated,
or expired claims must block readiness.

## Audit export

`hal-audit-export` combines:

- the signed startup seal;
- the final run marker;
- the verified fault-ledger chain and checkpoint;
- optional pre-fault incident context.

The resulting JSON is bound by a canonical SHA-256 digest. The tool verifies
source structure and ledger integrity, but cryptographic verification of the
startup signature remains the responsibility of the deployment verifier that
has access to the trust store.

Audit exports can contain operationally sensitive details. Store them with
restricted permissions and require signed `ExportAuditEvidence` authorization
before transfer outside the robot's security boundary.
