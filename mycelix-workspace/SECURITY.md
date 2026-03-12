# Security Policy

## Supported Versions

Only the latest `main` branch is actively supported with security updates.

## Reporting a Vulnerability

**Do NOT open public GitHub issues for security vulnerabilities.**

Please report security issues by emailing:

**tristan.stoltz@evolvingresonantcocreationism.com**

Include as much detail as possible: affected component, reproduction steps, and potential impact.

### Response Timeline

- **48 hours**: Acknowledgment of your report.
- **7 days**: Initial assessment and severity classification.
- Fixes will be developed privately and disclosed responsibly once a patch is available.

## Security Posture

### Post-Quantum Cryptography

PQ hardening is complete across all clusters. Mycelix is prepared for post-quantum threat models.

### Input Validation

All zomes enforce `is_finite()` guards on every f64 and f32 field in entry types, preventing NaN/Infinity injection attacks.

### Unsafe Code

`#![deny(unsafe_code)]` is enforced across all crates. No unsafe Rust is permitted.

### Fuzz Testing

Critical financial logic (demurrage, currency parameters) is continuously fuzz-tested with cargo-fuzz.
