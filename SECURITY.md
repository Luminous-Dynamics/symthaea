# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.9.x (latest main) | Yes |
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

## Disclosure Policy

We follow coordinated disclosure. Please allow us reasonable time to address reported issues before any public disclosure.
