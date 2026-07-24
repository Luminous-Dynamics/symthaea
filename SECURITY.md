# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x | Yes |
| 1.9.x | Security fixes only |
| < 1.9.0 | No |

## Reporting Vulnerabilities

**Do NOT open public issues for security vulnerabilities.**

Email: tristan.stoltz@evolvingresonantcocreationism.com

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days

Include a description of the vulnerability, steps to reproduce, and any relevant logs or configuration details.

## Security Posture

- **BLAKE3 integrity attestation** with 6 canary subsystems for runtime tamper detection
- **Model integrity**: SHA-256 verification before unsafe mmap loading
- **27 sub-crates** enforce `#![deny(unsafe_code)]`
- **Moral algebra** gates every action with Safe/Caution/Blocked classification
- **Epistemic gating** in Broca prevents hallucination at the logit level

## CI Security Checks

- `cargo-audit` scans for known vulnerabilities in dependencies
- `cargo-deny` enforces license and advisory policies

## Control-Plane Hardening

- The service daemon uses a versioned JSON-line protocol with bearer auth for privileged requests.
- The benchmark API uses HTTP bearer auth for protected routes and handler-level privacy checks for result access.
- `execute_gated` is intentionally read-only over the daemon control plane.
- Reserved daemon verbs must stay explicitly documented as `not_implemented` until they are real.
- Control-plane changes should keep these CI jobs green:
  - `Hardened Lib Regressions`
  - `Hardened Daemon Regressions`
  - `Hardened API Regressions`
  - `Hardened Nix Regressions`

## Disclosure Policy

We follow coordinated disclosure. Please allow us reasonable time to address reported issues before any public disclosure.
