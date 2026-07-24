# Deployment Provenance and Anti-Rollback

Powered exoskeleton transport must not arm from an unidentified binary.
`symthaea-hal::provenance` requires a release manifest containing source,
artifact, SBOM, and runtime-configuration digests; a security epoch; a build
number; and an approved signer identity.

The HAL does not hard-code a signature algorithm. The deployment environment
must provide a `ReleaseSignatureVerifier` backed by the approved trust store and
signature suite. A successful verification produces an opaque,
hardware/calibration-bound permit with a short monotonic lifetime.

## Required production behavior

- Persist the rollback guard in tamper-resistant or otherwise authoritative
  storage across restarts.
- Increase the security epoch for emergency revocations or incompatible trust
  migrations.
- Reject lower epochs and lower build numbers within the active epoch.
- Measure the running artifact and resolved runtime configuration, then require
  those digests to match the signed manifest before issuing a permit.
- Bind the permit to hardware identity, firmware revision, and calibration
  digest.
- Disable output when the permit expires or the actuator frame carries a
  different calibration binding.
- Archive the exact manifest, verifier policy, signer chain, and evidence report
  with each release.

The included test verifier is deterministic test scaffolding only; it is not a
cryptographic implementation.
