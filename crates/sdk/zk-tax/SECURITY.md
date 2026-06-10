# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: `security@luminousdynamics.org`
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

### Security Best Practices

#### For Users

1. **Always verify proofs** before trusting them
2. **Use real proofs in production** - dev mode is NOT cryptographically secure
3. **Keep dependencies updated** - run `cargo audit` regularly
4. **Validate inputs** - check jurisdiction codes, tax years, and income values

#### For Developers

1. **Never log income values** - they are private by design
2. **Use secure random number generation** for any cryptographic operations
3. **Implement rate limiting** when exposing proof generation as a service
4. **Validate all user inputs** before passing to proof generation

### Known Security Considerations

#### Dev Mode Proofs

Dev mode proofs (`TaxBracketProver::dev_mode()`) are:
- ❌ NOT cryptographically secure
- ❌ NOT suitable for production
- ✅ Fast for testing and development
- ✅ Marked with a special indicator (`0xDEADBEEF`)

Always check `is_dev_mode` before trusting a proof in production:

```rust
if proof.receipt_bytes == vec![0xDE, 0xAD, 0xBE, 0xEF] {
    // This is a dev mode proof - do not trust!
}
```

#### Proof Verification

Always verify proofs before acting on them:

```rust
match proof.verify() {
    Ok(_) => { /* Safe to trust */ }
    Err(e) => { /* Do not trust - verification failed */ }
}
```

#### Commitment Binding

Proofs are cryptographically bound to specific bracket parameters. Tampering with any field will cause verification to fail.

### Cryptographic Primitives

This library uses:
- **RISC Zero zkVM**: STARK-based zero-knowledge proofs
- **SHA3-256**: Commitment hashing
- **FNV-1a**: Fast commitment fingerprinting (non-cryptographic, for indexing)

### Audit Status

- [ ] Formal security audit (planned for v1.0)
- [x] Automated security scanning (cargo-audit)
- [x] Fuzz testing infrastructure

## Bug Bounty

We do not currently have a formal bug bounty program. However, we deeply appreciate security researchers who responsibly disclose vulnerabilities and will:

1. Publicly acknowledge your contribution (with permission)
2. Provide a letter of appreciation
3. Consider financial rewards for critical vulnerabilities (case-by-case)

## Contact

- Security issues: `security@luminousdynamics.org`
- General inquiries: `dev@luminousdynamics.org`
- GitHub: [Luminous-Dynamics/mycelix](https://github.com/Luminous-Dynamics/mycelix)
