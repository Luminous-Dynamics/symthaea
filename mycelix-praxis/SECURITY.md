# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

**Note**: During alpha/beta phases, only the latest release receives security updates.

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

### Preferred Method: GitHub Security Advisories

1. Navigate to the [Security Advisories](https://github.com/Luminous-Dynamics/mycelix-praxis/security/advisories) page
2. Click "Report a vulnerability"
3. Fill out the form with detailed information
4. Submit privately to maintainers

### Alternative Method: Email

Send an email to **security@mycelix.org** with:

- **Subject**: `[SECURITY] Brief description`
- **Description**: Detailed explanation of the vulnerability
- **Impact**: Potential severity and affected components
- **Reproduction**: Step-by-step instructions to reproduce
- **Proof of concept**: Code, screenshots, or logs (if applicable)
- **Suggested fix**: If you have one (optional)
- **Contact info**: How we can reach you for follow-up

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Status updates**: Every 14 days until resolved
- **Disclosure timeline**: Coordinated with you, typically 90 days
- **Credit**: You'll be credited in release notes (unless you prefer anonymity)

## Embargo Policy

- **Embargo period**: 90 days from report acceptance
- **Early disclosure**: Only if vulnerability is being actively exploited
- **Coordination**: We'll work with you to time public disclosure
- **CVE assignment**: We'll request a CVE for qualifying vulnerabilities

## Scope

### In Scope

Security issues in:

- **Holochain zomes** (validation, membrane, links)
- **Federated Learning protocol** (aggregation, poisoning resistance)
- **Verifiable Credentials** (issuance, verification, revocation)
- **Web client** (XSS, CSRF, injection attacks)
- **Rust crates** (memory safety, crypto misuse)
- **CI/CD pipeline** (supply chain attacks)
- **Documentation** (misleading security guidance)

### Out of Scope

- **Third-party dependencies**: Report to upstream projects (we'll track via advisories)
- **Social engineering**: User education issues
- **Physical access**: Local machine compromise scenarios
- **Denial of service**: Unless it enables other attacks
- **Deprecated code**: Marked as such in docs

## Security Best Practices

### For Contributors

- **Validate all inputs**: Never trust external data
- **Use safe Rust**: Minimize `unsafe`, audit carefully
- **Audit dependencies**: Check `cargo audit` and `npm audit`
- **Follow threat model**: See [docs/threat-model.md](docs/threat-model.md)
- **Minimize secrets**: No keys, passwords, or tokens in code
- **Test edge cases**: Fuzz inputs, test boundary conditions
- **Document security assumptions**: Note in code and ADRs

### For Users

- **Keep updated**: Apply security patches promptly
- **Verify releases**: Check signatures and checksums (when available)
- **Review permissions**: Understand what data you're sharing
- **Report suspicious behavior**: See reporting section above
- **Use secure channels**: HTTPS, encrypted comms for sensitive data

## Vulnerability Disclosure Timeline

| Day | Action |
|-----|--------|
| 0   | Vulnerability reported |
| 2   | Acknowledgment sent |
| 7   | Initial assessment and severity rating |
| 30  | Fix developed and tested (target) |
| 60  | Fix released in patched version (target) |
| 90  | Public disclosure (if not already released) |

**Note**: Timeline may vary based on severity and complexity.

## Security Measures

### Current

- **Input validation**: Holochain entry validation rules
- **Robust aggregation**: Trimmed mean/median for FL updates
- **Update clipping**: L2 norm clipping to bound contributions
- **Code review**: All PRs require maintainer approval
- **CI checks**: Automated linting, testing, and format checks
- **Dependency scanning**: Manual review of new dependencies

### Roadmap

- **Automated dependency scanning**: `cargo audit` and `npm audit` in CI
- **Fuzzing**: Continuous fuzzing of critical paths
- **Signing**: Binary and release signing with verified keys
- **Reproducible builds**: Deterministic build process
- **Penetration testing**: External security audit (pre-v1.0)
- **Bug bounty**: Community-driven vulnerability rewards (post-v1.0)

## Threat Model

For a comprehensive threat analysis, including:

- **Adversary models** (malicious users, validators, coordinators)
- **Attack vectors** (poisoning, sybil, backdoor, censorship)
- **Mitigations** (technical, cryptographic, governance)

See [docs/threat-model.md](docs/threat-model.md).

## Cryptographic Practices

- **Hashing**: BLAKE3 for content addressing
- **Signatures**: Ed25519 via Holochain agent keys
- **Random number generation**: Use OS-provided CSPRNGs
- **Key management**: Agent keys managed by Holochain lair keystore
- **No custom crypto**: Use well-audited libraries only

## Security Champions

Current security-focused maintainers:

- **@Luminous-Dynamics/maintainers** — Primary security contact

## Hall of Fame

Contributors who have responsibly disclosed security issues:

- *(None yet — be the first!)*

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Holochain Security Model](https://developer.holochain.org/concepts/security/)
- [Rust Security Guidelines](https://anssi-fr.github.io/rust-guide/)
- [Federated Learning Security](https://arxiv.org/abs/1811.12470)

## Questions?

For security-related questions that are **not vulnerability reports**:

- Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions) with `security` tag
- Email **security@mycelix.org** for private inquiries

---

**Thank you for helping keep Mycelix Praxis and our community safe!**
