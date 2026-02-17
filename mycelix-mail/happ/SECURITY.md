# Security Policy

## Supported Versions

Mycelix Mail is currently in **Phase 1** (Core DNA development). Security updates are provided for the main branch only.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

**Please do not file public issues for security vulnerabilities.**

If you discover a security vulnerability in Mycelix Mail, please report it by emailing:

**tristan.stoltz@evolvingresonantcocreationism.com**

Please include:

1. **Description**: A clear description of the vulnerability
2. **Impact**: How the vulnerability could be exploited
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Suggested Fix**: If you have ideas on how to fix it (optional)
5. **Your Contact Info**: So we can follow up with you

### What to expect:

- **Initial Response**: Within 48 hours
- **Status Update**: Within 1 week
- **Fix Timeline**: Varies by severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### Security Best Practices

When using Mycelix Mail:

1. **Keep Dependencies Updated**: Regularly update Holochain and Rust
2. **Use Strong DIDs**: Follow DID best practices for key generation
3. **Trust Score Thresholds**: Set appropriate minimum trust scores for your use case
4. **Monitor Logs**: Watch for unusual activity in Holochain logs
5. **Backup Keys**: Securely backup your agent keys

### Security Features

Mycelix Mail implements multiple security layers:

- **E2E Encryption**: All message bodies encrypted
- **Trust-Based Filtering**: Prevents spam at protocol level
- **Holochain Validation**: Distributed validation of all messages
- **MATL Byzantine Tolerance**: Up to 45% adversarial resistance
- **Open Source**: Auditable by anyone

### Known Limitations (Phase 1)

- MATL integration is simulated (trust scores not yet real)
- SMTP bridge not yet implemented (no email interop)
- UI not yet built (CLI only)
- No formal security audit yet (planned for Phase 3)

### Future Security Work

- [ ] Professional security audit (Phase 3)
- [ ] Penetration testing
- [ ] Formal verification of validation rules
- [ ] Bug bounty program
- [ ] Security documentation

---

Thank you for helping keep Mycelix Mail secure! 🍄
