# Security Policy

> **@mycelix/sdk** is a Byzantine-resistant trust infrastructure SDK. Security is foundational to our mission.

## Table of Contents

- [Reporting Vulnerabilities](#reporting-vulnerabilities)
- [Security Model](#security-model)
- [Cryptographic Assumptions](#cryptographic-assumptions)
- [Trust Boundaries](#trust-boundaries)
- [Threat Model](#threat-model)
- [Security Guarantees](#security-guarantees)
- [Known Limitations](#known-limitations)
- [Security Best Practices](#security-best-practices)
- [Audit Status](#audit-status)
- [Bug Bounty Program](#bug-bounty-program)
- [Supported Versions](#supported-versions)

---

## Reporting Vulnerabilities

### 🔐 Private Disclosure (Preferred)

For security vulnerabilities, please report **privately** to allow us time to develop and test fixes before public disclosure.

**Email:** security@luminousdynamics.org

**Please include:**
1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Suggested fix (if any)
5. Your contact information for follow-up

### PGP Encryption (Optional)

For sensitive reports, encrypt your message using our PGP key:

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
[Key will be published at https://luminousdynamics.org/.well-known/security.txt]
-----END PGP PUBLIC KEY BLOCK-----
```

### Response Timeline

| Phase | Timeline |
|-------|----------|
| Initial response | Within 48 hours |
| Triage & severity assessment | Within 5 business days |
| Fix development | Based on severity (see below) |
| Coordinated disclosure | 90 days from report |

**Severity-based fix timeline:**
- 🔴 **Critical:** 7 days
- 🟠 **High:** 14 days
- 🟡 **Medium:** 30 days
- 🟢 **Low:** 90 days

### What NOT to Do

- ❌ Do not disclose publicly before coordinated disclosure
- ❌ Do not test vulnerabilities on production systems
- ❌ Do not access, modify, or delete data belonging to others
- ❌ Do not perform DoS attacks

---

## Security Model

### Byzantine Fault Tolerance

The Mycelix SDK implements **34% validated Byzantine Fault Tolerance (BFT)**, meaning the system remains secure when up to 34% of participants are malicious or faulty.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BYZANTINE TOLERANCE MODEL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Traditional BFT: n ≥ 3f + 1  (tolerates ~33% Byzantine)       │
│   Mycelix MATL:    n ≥ 2.94f   (tolerates ~34% Byzantine)       │
│                                                                  │
│   This is achieved through:                                      │
│   • Proof of Gradient Quality (PoGQ) scoring                    │
│   • Adaptive reputation tracking                                 │
│   • Trust-weighted aggregation                                   │
│   • Anomaly detection with adaptive thresholds                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Defense Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYERS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 5: Application    │ Input validation, rate limiting      │
│  Layer 4: Protocol       │ MATL scoring, Byzantine detection    │
│  Layer 3: Cryptographic  │ Signatures, hashing, secure random   │
│  Layer 2: Transport      │ Holochain DHT, encrypted channels    │
│  Layer 1: Infrastructure │ CodeQL, npm audit, Dependabot        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cryptographic Assumptions

### Algorithms Used

| Purpose | Algorithm | Security Level |
|---------|-----------|----------------|
| Hashing | SHA-256 | 128-bit |
| HMAC | HMAC-SHA-256 | 128-bit |
| Random | Web Crypto API (`crypto.getRandomValues`) | CSPRNG |
| Signatures | Ed25519 (via Holochain) | 128-bit |
| Key derivation | HKDF-SHA-256 | 128-bit |

### Security Assumptions

1. **SHA-256 collision resistance:** Finding two inputs with the same hash is computationally infeasible
2. **Ed25519 unforgeability:** Signatures cannot be forged without the private key
3. **CSPRNG quality:** `crypto.getRandomValues` provides cryptographically secure randomness
4. **Holochain DHT integrity:** The underlying DHT provides eventual consistency and tamper evidence

### What We Do NOT Assume

- ❌ Network availability (we handle partitions gracefully)
- ❌ Honest majority (we tolerate up to 34% validated Byzantine actors)
- ❌ Synchronous communication (we're asynchronous-first)
- ❌ Trust in any single node (we aggregate trust across sources)

---

## Trust Boundaries

### Boundary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR APPLICATION                            │
│                        (Trusted)                                 │
├─────────────────────────────────────────────────────────────────┤
│                    @mycelix/sdk                                  │
│                     (Trusted)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ INPUT VALIDATION BOUNDARY                                 │   │
│  │ • Zod schema validation                                   │   │
│  │ • Rate limiting                                           │   │
│  │ • Input sanitization                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                HOLOCHAIN CONDUCTOR                               │
│                  (Semi-Trusted)                                  │
│  • Local conductor: Trusted                                      │
│  • Remote conductor: Verify all responses                        │
├─────────────────────────────────────────────────────────────────┤
│                    DHT NETWORK                                   │
│                   (Untrusted)                                    │
│  • All data is validated cryptographically                       │
│  • Byzantine actors are detected and penalized                   │
│  • Reputation is aggregated across sources                       │
└─────────────────────────────────────────────────────────────────┘
```

### Crossing Boundaries

When data crosses a trust boundary:

1. **Inbound from network:** Validate signatures, check schemas, verify provenance
2. **Outbound to network:** Sign data, apply rate limits, sanitize metadata
3. **Between hApps:** Use Bridge validation schemas, verify credentials

---

## Threat Model

### In-Scope Threats

| Threat | Mitigation |
|--------|------------|
| **Byzantine participants** | MATL scoring, PoGQ detection, reputation penalties |
| **Sybil attacks** | Stake requirements, proof-of-work alternatives, reputation building cost |
| **Gradient poisoning (FL)** | Krum, TrimmedMean, coordinate median aggregation |
| **Replay attacks** | Nonces, timestamps, sequence numbers |
| **Man-in-the-middle** | End-to-end encryption via Holochain |
| **Denial of service** | Rate limiting, circuit breakers, timeout policies |
| **Data tampering** | Cryptographic signatures, content addressing |

### Out-of-Scope Threats

| Threat | Reason |
|--------|--------|
| **Compromised local machine** | SDK cannot protect against root access |
| **Side-channel attacks** | Requires hardware-level mitigations |
| **Quantum attacks** | Current algorithms are not post-quantum (planned for v2.0) |
| **Social engineering** | User education, not code mitigation |
| **Physical access attacks** | Out of software scope |

### Attack Surface

```
┌─────────────────────────────────────────────────────────────────┐
│                      ATTACK SURFACE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HIGH RISK (Carefully audited)                                   │
│  ├── src/security/          # Crypto, rate limiting              │
│  ├── src/matl/              # Byzantine detection                │
│  ├── src/fl/                # Federated learning aggregation     │
│  └── src/validation/        # Input validation schemas           │
│                                                                  │
│  MEDIUM RISK (Standard review)                                   │
│  ├── src/bridge/            # Cross-hApp communication           │
│  ├── src/client/            # Conductor connectivity             │
│  └── src/epistemic/         # Claim verification                 │
│                                                                  │
│  LOWER RISK (Functionality focus)                                │
│  ├── src/integrations/      # Domain-specific adapters           │
│  ├── src/react/             # UI bindings                        │
│  └── src/graphql/           # Query layer                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Guarantees

### What We Guarantee

✅ **Byzantine tolerance up to 34%** - System remains correct with up to 34% validated malicious participants

✅ **Cryptographic integrity** - All data is signed and verified

✅ **Timing-attack resistance** - Constant-time comparison for sensitive operations

✅ **Secure randomness** - All random values use CSPRNG

✅ **Input validation** - All external inputs are validated against Zod schemas

✅ **Rate limiting** - Configurable rate limits prevent abuse

### What We Do NOT Guarantee

❌ **Privacy** - The SDK does not provide encryption at rest (use your own)

❌ **Anonymity** - Agent public keys are visible on the DHT

❌ **Perfect security** - No system is unbreakable; we minimize risk

❌ **Quantum resistance** - Current algorithms are classical (roadmapped for v2.0)

---

## Known Limitations

### Current Limitations

| Limitation | Impact | Planned Fix |
|------------|--------|-------------|
| Classical cryptography | Vulnerable to future quantum computers | v2.0: Post-quantum algorithms |
| In-memory secrets | Secrets visible in memory dumps | v1.1: Secure memory wrappers |
| No HSM support | Private keys in software | v1.2: HSM integration |
| Single-node conductor | Conductor is single point of failure | v1.3: Conductor redundancy |

### Byzantine Tolerance Caveats

The 34% validated Byzantine tolerance applies under these conditions:
- Participants have similar influence (stake/reputation)
- Network latency is bounded
- At least 66% of participants are online and responsive

---

## Security Best Practices

### For SDK Users

```typescript
// ✅ DO: Validate all external inputs
import { validateOrThrow, IdentitySchemas } from '@mycelix/sdk/validation';
validateOrThrow(IdentitySchemas.QueryIdentityInput, userInput);

// ✅ DO: Use secure random
import { security } from '@mycelix/sdk';
const id = security.secureUUID();

// ✅ DO: Check reputation before trusting
const reputation = await client.queryCrossHappReputation(agent);
if (reputation.aggregate < 0.5) {
  throw new Error('Insufficient reputation');
}

// ❌ DON'T: Trust input without validation
const data = JSON.parse(untrustedInput); // Dangerous!

// ❌ DON'T: Use Math.random for anything security-related
const id = Math.random().toString(36); // Predictable!

// ❌ DON'T: Ignore Byzantine detection
if (matl.isByzantine(pogq)) {
  // DO handle this - don't ignore!
  rejectParticipant(participantId);
}
```

### For SDK Contributors

1. **Security module changes require two reviewers**
2. **All crypto code must have test vectors**
3. **Never commit secrets, even for testing**
4. **Use `security/detect-*` ESLint rules**
5. **Run `npm audit` before every commit**

---

## Audit Status

### Completed Audits

| Date | Auditor | Scope | Status |
|------|---------|-------|--------|
| - | - | - | No formal audits yet |

### Planned Audits

| Q1 2026 | Independent security firm | Full SDK audit |
| Q2 2026 | Cryptography specialist | MATL/FL modules |

### Continuous Scanning

- ✅ **CodeQL:** Weekly deep scans + PR checks
- ✅ **npm audit:** Daily vulnerability checks
- ✅ **Dependabot:** Automated dependency updates
- ✅ **ESLint security plugin:** Static analysis on every commit

---

## Bug Bounty Program

### Program Status: 🟡 Coming Soon

We are establishing a formal bug bounty program. In the meantime:

**Informal Rewards:**
- Acknowledgment in SECURITY.md and release notes
- Swag and recognition
- Potential future bounty backpay for critical finds

**Planned Bounty Ranges (when launched):**
| Severity | Range |
|----------|-------|
| Critical | $1,000 - $5,000 |
| High | $500 - $1,000 |
| Medium | $100 - $500 |
| Low | Recognition |

### Hall of Fame

*No entries yet - be the first!*

---

## Supported Versions

| Version | Supported | Security Updates |
|---------|-----------|------------------|
| 0.6.x | ✅ | Yes |
| 0.5.x | ⚠️ | Critical only |
| < 0.5 | ❌ | No |

**Upgrade policy:** We support the current major version and one previous minor version with security patches.

---

## Contact

- **Security issues:** security@luminousdynamics.org
- **General questions:** dev@luminousdynamics.org
- **GitHub Security Advisories:** [Repository Security Tab](https://github.com/Luminous-Dynamics/mycelix/security)

---

*Last updated: January 2026*
*Version: 1.0.0*
