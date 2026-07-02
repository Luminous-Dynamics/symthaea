# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### For Critical Vulnerabilities

**DO NOT** open a public GitHub issue.

Instead:

1. Email security concerns to: **security@mycelix.org** (coming soon)
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. You should receive a response within 48 hours
4. We will work with you to understand and address the issue

### For Non-Critical Security Issues

You may open a GitHub issue for:
- General security questions
- Best practices discussions
- Documentation improvements

## Security Best Practices

### For Users

- Always verify dataset hashes before use
- Use the latest stable version
- Keep dependencies updated
- Review epistemic tier classifications
- Validate cryptographic proofs

### For Developers

- Never commit secrets or API keys
- Use environment variables for configuration
- Implement proper input validation
- Follow secure coding guidelines
- Run security audits regularly

## Cryptographic Components

This project uses:

- **zk-STARKs** (Risc0) for zero-knowledge proofs
- **Ed25519** for digital signatures
- **BLAKE3** for hashing
- **Adaptive Differential Privacy** for data protection

### Known Limitations

- zk-STARK implementation is experimental
- Adaptive DP requires careful parameter tuning
- PoGQ assumes honest majority > 55%

## Security Audits

- **Planned Q4 2026**: External security audit by reputable firm
- Scope: Smart contracts, cryptographic implementations, PoGQ algorithm
- Budget: $30,000

## Bug Bounty Program

Coming in Phase 2 (Q2-Q4 2026):

- **Critical**: Up to $10,000
- **High**: $1,000 - $5,000
- **Medium**: $500 - $1,000
- **Low**: $100 - $500

Eligibility:
- First to report
- Clear reproduction steps
- Responsible disclosure
- No public disclosure before fix

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 2**: Initial response from team
3. **Day 7**: Assessment complete
4. **Day 30**: Fix developed and tested
5. **Day 45**: Patch released
6. **Day 60**: Public disclosure (coordinated with reporter)

## Security Updates

Security patches will be released as:
- Patch versions (0.1.x) for minor fixes
- Minor versions (0.x.0) for moderate issues
- Major versions (x.0.0) for critical vulnerabilities

Subscribe to:
- GitHub Security Advisories
- Project mailing list (coming soon)

## Acknowledgments

We thank the security researchers who responsibly disclose vulnerabilities:

- [List will be updated as reports are received]

## Contact

- **Security Email**: security@mycelix.org (coming soon)
- **General Contact**: dev@mycelix.org (coming soon)
- **GitHub**: https://github.com/luminousdynamics/mycelix-desci

## References

- [Mycelix Protocol Security](https://github.com/luminousdynamics/mycelix-core/blob/main/SECURITY.md)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)

---

Last updated: 2025-11-15
