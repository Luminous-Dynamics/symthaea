# Security Policy

## Reporting Security Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

### How to Report

1. **Email**: Send details to **security@luminous-dynamics.com**
2. **GitHub Security**: Use [GitHub Security Advisories](https://github.com/Luminous-Dynamics/mycelix-praxis/security/advisories/new)

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: <7 days
  - High: <30 days
  - Medium: <90 days
  - Low: Best effort

### Disclosure Policy

- **Embargo Period**: 90 days minimum before public disclosure
- **Coordinated Disclosure**: We'll work with you on timing
- **Credit**: You'll be credited in security advisories (unless you prefer anonymity)

---

## Security Measures

### Infrastructure Security

#### Web Application

**HTTP Security Headers** (`docker/nginx.conf`):

```nginx
# Prevent clickjacking
X-Frame-Options: SAMEORIGIN

# Prevent MIME type sniffing
X-Content-Type-Options: nosniff

# XSS protection (legacy browsers)
X-XSS-Protection: 1; mode=block

# Referrer policy
Referrer-Policy: strict-origin-when-cross-origin

# Content Security Policy
Content-Security-Policy:
  default-src 'self';
  script-src 'self' 'unsafe-inline' 'unsafe-eval';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self' data:;
  connect-src 'self';
  frame-ancestors 'self';
  base-uri 'self';
  form-action 'self';

# Permissions Policy (Feature Policy)
Permissions-Policy:
  geolocation=(),
  microphone=(),
  camera=(),
  payment=(),
  usb=()

# HSTS (when using HTTPS)
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

**CSP Explanation**:
- `default-src 'self'`: Only load resources from same origin
- `script-src 'unsafe-inline' 'unsafe-eval'`: Required for React development build (remove in production)
- `img-src data: https:`: Allow data URIs and HTTPS images
- `connect-src 'self'`: API calls only to same origin

**Note**: For production, tighten CSP by:
1. Remove `'unsafe-inline'` and use nonces/hashes
2. Remove `'unsafe-eval'`
3. Specify exact CDN origins instead of wildcards

#### Rate Limiting

**nginx Example** (not yet implemented):
```nginx
# Define rate limit zone
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

# Apply to API endpoints
location /api/ {
    limit_req zone=api burst=20 nodelay;
    limit_req_status 429;
}
```

**Holochain Conductor**:
- Built-in rate limiting per agent
- Configurable in conductor config
- See [Holochain docs](https://developer.holochain.org/)

### Application Security

#### Input Validation

**Rust** (`crates/praxis-core/src/validation.rs`):
```rust
/// Validate course title
pub fn validate_course_title(title: &str) -> Result<(), String> {
    if title.is_empty() {
        return Err("Title cannot be empty".into());
    }
    if title.len() > 200 {
        return Err("Title too long (max 200 chars)".into());
    }
    if !title.chars().all(|c| c.is_alphanumeric() || c.is_whitespace() || ":-.,!?()".contains(c)) {
        return Err("Title contains invalid characters".into());
    }
    Ok(())
}

/// Validate email format
pub fn validate_email(email: &str) -> Result<(), String> {
    let re = regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        .unwrap();
    if re.is_match(email) {
        Ok(())
    } else {
        Err("Invalid email format".into())
    }
}
```

**TypeScript** (`apps/web/src/utils/validation.ts`):
```typescript
export function sanitizeInput(input: string): string {
  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
}

export function validateCourseId(id: string): boolean {
  return /^[a-z0-9-]+$/.test(id) && id.length <= 50;
}
```

#### Cryptographic Operations

**Only Use Vetted Libraries**:
- ✅ `ed25519-dalek` for signatures
- ✅ `sha2` for hashing
- ✅ `rand` with OS RNG for randomness
- ❌ Never roll your own crypto

**Example** (`crates/praxis-core/src/crypto.rs`):
```rust
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};
use sha2::{Sha256, Digest};

pub fn sign_data(keypair: &Keypair, data: &[u8]) -> Signature {
    keypair.sign(data)
}

pub fn verify_signature(
    public_key: &ed25519_dalek::PublicKey,
    data: &[u8],
    signature: &Signature,
) -> Result<(), SignatureError> {
    public_key.verify(data, signature)
}

pub fn hash_data(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}
```

#### Dependency Security

**Automated Auditing**:
- `cargo audit` runs on every PR (GitHub Actions)
- `npm audit` runs on every PR
- Dependabot configured for automatic updates

**Manual Review**:
- All new dependencies require maintainer approval
- Check for:
  - Active maintenance
  - Security track record
  - License compatibility
  - Code quality

### Privacy Protection

#### Federated Learning Privacy

**Gradient Clipping** (prevents gradient leakage):
```rust
pub fn clip_gradients(gradients: &[f32], max_norm: f32) -> Vec<f32> {
    let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        gradients.iter().map(|g| g * (max_norm / norm)).collect()
    } else {
        gradients.to_vec()
    }
}
```

**Differential Privacy** (optional, configurable):
```rust
use rand_distr::{Normal, Distribution};

pub fn add_dp_noise(
    data: &[f32],
    epsilon: f64,
    delta: f64,
    sensitivity: f32,
) -> Vec<f32> {
    let sigma = (2.0 * sensitivity.powi(2) * (1.25 / delta).ln()) / epsilon;
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut rng = rand::thread_rng();

    data.iter()
        .map(|x| x + normal.sample(&mut rng) as f32)
        .collect()
}
```

**Privacy Parameters**:
- Default ε (epsilon): 1.0 (strong privacy)
- Default δ (delta): 1e-5
- Adjustable per round based on sensitivity

#### Data Minimization

**What We Store**:
- ✅ Course metadata (titles, descriptions)
- ✅ FL round state and aggregated models
- ✅ Verifiable credentials (public)
- ❌ User training data (never leaves device)
- ❌ Raw model updates (aggregated only)

**What We Never See**:
- Individual training datasets
- User behavior analytics
- Personal information beyond public keys

### Holochain Security

#### Agent-Centric Security

- **Agent Keys**: Each user has unique ed25519 keypair
- **Source Chain**: Local, append-only, cryptographically signed
- **DHT**: Distributed, no central point of failure
- **Validation**: All entries validated by peers

#### Zome Security

**Entry Validation**:
```rust
#[hdk_extern]
pub fn validate_create_course(
    _action: EntryCreationAction,
    course: Course,
) -> ExternResult<ValidateCallbackResult> {
    // Validate title
    if course.title.is_empty() || course.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid course title".into()
        ));
    }

    // Validate instructor signature
    if !verify_instructor_signature(&course) {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid instructor signature".into()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
```

**Capability Grants**:
- Fine-grained permissions
- Time-limited tokens
- Revocable access

### Attack Surface Analysis

#### Identified Threats

| Threat | Likelihood | Impact | Mitigation |
|--------|------------|--------|------------|
| **Model Poisoning** | High | High | Robust aggregation (trimmed mean), outlier detection |
| **Gradient Leakage** | Medium | High | Differential privacy, gradient clipping |
| **Sybil Attack** | Medium | Medium | Signed commitments, reputation system |
| **XSS** | Low | Medium | CSP, input sanitization, React auto-escaping |
| **CSRF** | Low | Medium | SameSite cookies, CORS policy |
| **DDoS** | Medium | Low | Rate limiting, Cloudflare (production) |
| **Supply Chain** | Low | High | Dependency auditing, lock files, vendoring |

#### Not Yet Implemented (Future Work)

- [ ] Secure Multi-Party Computation (SMPC) for aggregation
- [ ] Zero-Knowledge Proofs for credential privacy
- [ ] Homomorphic encryption for computation on encrypted data
- [ ] Byzantine-fault-tolerant consensus beyond trimmed mean

### Compliance

#### GDPR (if applicable)

- **Right to Erasure**: Agents can delete source chain (local)
- **Data Portability**: Export credentials as JSON
- **Consent**: Explicit opt-in for FL participation
- **Privacy by Design**: Data minimization, local-first

#### Data Protection

- No PII collected beyond public keys
- No analytics or tracking
- User controls all data
- Transparent data flows

---

## Security Best Practices for Contributors

### Code Review Checklist

When reviewing PRs, check for:

**Input Validation**:
- [ ] All external inputs validated
- [ ] String lengths checked
- [ ] Regex patterns safe (no ReDoS)
- [ ] Numeric bounds enforced

**Cryptography**:
- [ ] Using vetted libraries only
- [ ] No hardcoded keys or secrets
- [ ] Randomness from secure sources
- [ ] Signatures verified

**Authentication/Authorization**:
- [ ] Capability grants checked
- [ ] Agent permissions verified
- [ ] No privilege escalation possible

**Injection Prevention**:
- [ ] No SQL injection (N/A - no SQL)
- [ ] No command injection
- [ ] No XSS via proper escaping
- [ ] No path traversal

**Dependencies**:
- [ ] New deps justified
- [ ] Deps audited for vulnerabilities
- [ ] License compatible
- [ ] Minimal version constraints

### Secure Coding Guidelines

**DO**:
- ✅ Validate all inputs at boundaries
- ✅ Use type systems to enforce invariants
- ✅ Fail securely (return errors, don't panic)
- ✅ Log security events
- ✅ Keep secrets out of code (use env vars)

**DON'T**:
- ❌ Trust user input
- ❌ Use `unwrap()` on external data
- ❌ Implement custom crypto
- ❌ Commit secrets to git
- ❌ Disable security features "temporarily"

---

## Incident Response Plan

### Severity Levels

**Critical (P0)**:
- Remote code execution
- Data breach
- Authentication bypass
- Fix: <24 hours

**High (P1)**:
- Privilege escalation
- Significant DoS
- Crypto weakness
- Fix: <7 days

**Medium (P2)**:
- Local exploits
- Information disclosure
- Minor DoS
- Fix: <30 days

**Low (P3)**:
- Minor bugs
- Cosmetic issues
- Fix: Best effort

### Response Procedure

1. **Triage** (1 hour)
   - Assess severity
   - Identify affected versions
   - Determine exploitability

2. **Containment** (varies)
   - Disable affected features (if possible)
   - Notify affected users
   - Document incident

3. **Fix** (timeline above)
   - Develop patch
   - Security review
   - Test thoroughly

4. **Release** (1-2 days)
   - Cut security release
   - Update security advisory
   - Notify users

5. **Post-Mortem** (1 week)
   - Root cause analysis
   - Process improvements
   - Update threat model

---

## Security Contacts

- **Email**: security@luminous-dynamics.com
- **GitHub**: [@Luminous-Dynamics](https://github.com/Luminous-Dynamics)
- **PGP Key**: TBD

---

## Version History

- **1.0** (2025-11-15): Initial security policy
- Security headers, CSP, Permissions Policy
- Dependency auditing automation
- Input validation guidelines

---

**Last Updated**: 2025-11-15
**Next Review**: 2026-02-15 (quarterly)
