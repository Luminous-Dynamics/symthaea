# Security Policy

Mycelix Music takes security seriously. Before any mainnet deployment we intend to:

- Commission an external smart contract audit (target budget $15-25K)
- Run internal reviews for reentrancy, authorization, and token-transfer safety
- Add comprehensive tests and fuzzing for edge cases (fee math, split sums, min payments)
- Prefer SafeERC20 for token operations and follow OpenZeppelin best practices

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to: security@mycelix.music
3. Include as much detail as possible:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 72 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Critical issues within 30 days
- **Disclosure**: Coordinated disclosure with reporter credit

## Scope

**In Scope:**
- Solidity contracts under `contracts/src/`
- API server in `/apps/api`
- Any off-chain components that affect on-chain state or funds
- SDK in `/packages/sdk`
- Infrastructure configurations in `/k8s`

**Out of Scope:**
- Demo/mock endpoints and placeholder integrations
- Third-party dependencies (report to respective maintainers)
- Social engineering attacks
- Denial of service attacks

## Security Features

### Smart Contracts
- OpenZeppelin libraries for battle-tested implementations
- Reentrancy guards on state-changing functions
- SafeERC20 for token operations
- Access control with owner/admin separation

### API Security
- **Helmet.js** - Security headers (HSTS, CSP, X-Frame-Options)
- **Rate limiting** - Configurable per-endpoint limits
- **Input validation** - Zod schemas for all endpoints
- **SQL injection protection** - Parameterized queries only
- **Signature verification** - EIP-712 typed data signing
- **Replay protection** - Nonce and timestamp validation

### Infrastructure
- Non-root containers with read-only filesystems
- Kubernetes NetworkPolicies for isolation
- Resource limits on all pods
- TLS everywhere with cert-manager

## For Contributors

1. Never commit secrets - use environment variables
2. Validate all inputs using `/apps/api/src/validators`
3. Use parameterized queries - never concatenate SQL
4. Follow the principle of least privilege
5. Keep dependencies updated (Dependabot enabled)

Thank you for helping keep Mycelix Music safe.
