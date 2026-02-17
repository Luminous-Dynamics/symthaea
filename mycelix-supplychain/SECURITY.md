# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@luminous-dynamics.dev**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Best Practices

### For Users

1. **Keep dependencies updated** - Run `cargo update` and `npm update` regularly
2. **Use environment variables** - Never hardcode secrets; use `.env` files (gitignored)
3. **Enable TLS** - Always use HTTPS/TLS in production
4. **Validate inputs** - The service validates against JSON schemas, but add application-level checks
5. **Audit logs** - Enable audit logging for all claim operations
6. **Rotate keys** - Rotate DID signing keys per your organization's policy

### For Contributors

1. **No secrets in code** - Use environment variables or secret management
2. **Dependency scanning** - Dependabot is enabled; review alerts promptly
3. **Code scanning** - CodeQL runs on all PRs; fix identified issues
4. **Minimal permissions** - Follow principle of least privilege
5. **Input validation** - Always validate and sanitize external inputs
6. **Secure defaults** - Default configurations should be secure

## Known Security Considerations

### Cryptographic Operations

- This project uses **ed25519** for signing (via `ed25519-dalek`)
- SD-JWT and BBS+ implementations are **experimental** - audit before production use
- Key management is **out of scope** - integrate with HSM/KMS for production

### Supply Chain Security

- Dependencies are scanned via Dependabot
- Cargo.lock and package-lock.json are committed for reproducible builds
- Consider using `cargo-audit` and `npm audit` in CI/CD

### Data Privacy

- **Selective disclosure** (SD-JWT/BBS+) allows hiding sensitive fields
- Review which fields are included in claims before publishing to DKG
- PII should be minimized or excluded from on-chain data

### Network Security

- Service expects to run behind a reverse proxy (nginx, Envoy, etc.)
- Rate limiting and authentication should be implemented at the proxy layer
- Consider mTLS for service-to-service communication

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patched versions ASAP
5. Publish a security advisory on GitHub

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request.
