# Security Policy

## Supported Versions

| Component | Version | Supported |
|-----------|---------|-----------|
| Symthaea | 1.9.x | Yes |
| Mycelix clusters | All active | Yes |
| Luminous Nix | 0.4.x | Yes |

## Reporting a Vulnerability

**Do NOT open public GitHub issues for security vulnerabilities.**

Please report vulnerabilities via email to:

- **Primary**: security@luminousdynamics.org
- **Fallback**: tristan.stoltz@evolvingresonantcocreationism.com

Your report should include:

- Description of the vulnerability
- Steps to reproduce
- Impact assessment (confidentiality, integrity, availability)
- Affected component (e.g., Symthaea core, Mycelix bridge-common, identity zomes)

### Response Timeline

- **48 hours**: Acknowledgment of report
- **7 days**: Initial assessment and severity classification
- **30 days**: Fix developed for critical/high severity
- **90 days**: Coordinated public disclosure

## Scope

### In Scope

- All Rust crates (Symthaea, Mycelix zomes, bridge-common, bridge-entry-types)
- Python federated learning code (0TML)
- Bridge and consciousness gating logic
- Cryptographic implementations (PQC, STARK proofs, credential validation)
- Identity, governance, and finance zomes
- SDK code (Rust, TypeScript, Python, WASM)

### Out of Scope

- **Holochain conductor**: Report to the [Holochain team](https://github.com/holochain/holochain/security)
- **Third-party dependencies**: Report upstream to the maintainer
- **Documentation-only issues**: Not considered security vulnerabilities

## Disclosure Policy

We follow a **90-day coordinated disclosure** timeline:

1. Reporter submits vulnerability privately.
2. We acknowledge, assess, and develop a fix.
3. Fix is released and reporter is notified.
4. Public disclosure after 90 days from initial report, or upon fix release (whichever comes first).

Extensions are negotiable for complex fixes requiring coordinated multi-component patches.

Reporters will be credited in release notes unless they prefer to remain anonymous.

## Security Contacts

| Role | Name | Contact |
|------|------|---------|
| Primary | Tristan Stoltz | security@luminousdynamics.org |

**PGP key**: Available on request.
