# Mycelix Security Hardening Guide

This document provides comprehensive security hardening guidelines for deploying Mycelix in production environments.

## Table of Contents

1. [Threat Model](#threat-model)
2. [Network Security](#network-security)
3. [Authentication & Authorization](#authentication--authorization)
4. [Data Protection](#data-protection)
5. [Byzantine Attack Resistance](#byzantine-attack-resistance)
6. [Container Security](#container-security)
7. [Secrets Management](#secrets-management)
8. [Monitoring & Alerting](#monitoring--alerting)
9. [Compliance Considerations](#compliance-considerations)
10. [Audit Checklist](#audit-checklist)
11. [Holochain Zome Checklist](#holochain-zome-checklist)

---

## Threat Model

### Assets

| Asset | Sensitivity | Impact if Compromised |
|-------|-------------|----------------------|
| Model weights | HIGH | IP theft, competitive advantage loss |
| Gradient updates | MEDIUM | Privacy leakage, model poisoning |
| Node credentials | CRITICAL | Full system compromise |
| Reputation scores | MEDIUM | Unfair attribution, gaming |
| Aggregated models | HIGH | Incorrect ML outputs |

### Threat Actors

1. **Malicious Participants** (Byzantine nodes)
   - Goal: Poison model, steal data, disrupt training
   - Capability: Control up to 45% of nodes
   - Mitigation: Byzantine detection, reputation system

2. **External Attackers**
   - Goal: Network intrusion, data exfiltration
   - Capability: Network-level attacks, exploitation
   - Mitigation: TLS, firewalls, input validation

3. **Curious Aggregator**
   - Goal: Learn individual contributions
   - Capability: Access to gradient aggregation
   - Mitigation: Differential privacy, secure aggregation

4. **Insider Threats**
   - Goal: Data theft, sabotage
   - Capability: Legitimate system access
   - Mitigation: Audit logging, least privilege

### Attack Vectors

| Vector | Likelihood | Impact | Mitigation |
|--------|------------|--------|------------|
| Gradient poisoning | HIGH | HIGH | Byzantine detection |
| Model inversion | MEDIUM | HIGH | Differential privacy |
| Sybil attack | MEDIUM | HIGH | Identity verification |
| DoS attack | MEDIUM | MEDIUM | Rate limiting |
| Replay attack | LOW | MEDIUM | Nonce-based auth |
| Side-channel | LOW | HIGH | Constant-time ops |

---

## Network Security

### TLS Configuration

```yaml
# Recommended TLS settings
tls:
  min_version: "1.3"
  cipher_suites:
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
  client_auth: require  # mTLS for node-to-node
```

### Firewall Rules

```bash
# Coordinator
ufw allow 3000/tcp  # FL API (internal only)
ufw allow 9090/tcp  # Prometheus (internal only)
ufw deny 22         # No SSH in production

# Holochain node
ufw allow 8888/tcp  # Conductor (internal only)
ufw allow 5000/udp  # DHT gossip (specific peers only)

# Block all other inbound
ufw default deny incoming
```

### Network Segmentation

```
┌─────────────────────────────────────────────────────────┐
│                     DMZ / Load Balancer                 │
└─────────────────────┬───────────────────────────────────┘
                      │ (TLS termination)
┌─────────────────────┴───────────────────────────────────┐
│                   Application Tier                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Coordinator │  │ Coordinator │  │ Coordinator │     │
│  │   (Active)  │  │  (Standby)  │  │  (Standby)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼────────────────┼────────────────┼─────────────┘
          │ (mTLS)         │                │
┌─────────┴────────────────┴────────────────┴─────────────┐
│                     Data Tier                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Holochain  │  │  Holochain  │  │  Holochain  │     │
│  │    Node 1   │  │    Node 2   │  │    Node 3   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## Authentication & Authorization

### Node Authentication

Every FL participant must authenticate using Ed25519 signatures:

```rust
// Node identity verification
pub struct NodeIdentity {
    pub public_key: [u8; 32],
    pub signature: [u8; 64],
    pub timestamp: u64,
    pub nonce: [u8; 32],
}

impl NodeIdentity {
    pub fn verify(&self, challenge: &[u8]) -> Result<bool> {
        let message = [
            challenge,
            &self.timestamp.to_le_bytes(),
            &self.nonce,
        ].concat();

        ed25519::verify(&self.public_key, &message, &self.signature)
    }
}
```

### Authorization Levels

| Role | Permissions | Example |
|------|-------------|---------|
| Participant | Submit gradients, view own scores | FL training node |
| Validator | Verify proofs, vote on consensus | Guardian node |
| Coordinator | Aggregate, manage rounds | Training orchestrator |
| Admin | System configuration, key rotation | Ops team |

### API Authentication

```rust
// JWT-based API auth (for HTTP endpoints)
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,        // Node ID
    pub role: String,       // Authorization role
    pub exp: usize,         // Expiration (short-lived)
    pub iat: usize,         // Issued at
    pub jti: String,        // Unique token ID (for revocation)
}

// Token lifetime: 15 minutes for API, 1 hour for session
const API_TOKEN_LIFETIME: Duration = Duration::from_secs(900);
```

---

## Data Protection

### Gradient Privacy

```python
# Differential privacy parameters
class PrivacyConfig:
    epsilon: float = 1.0           # Privacy budget
    delta: float = 1e-5            # Failure probability
    max_gradient_norm: float = 1.0  # Clipping bound
    noise_multiplier: float = 1.1   # Gaussian noise scale

def add_differential_privacy(gradient: np.ndarray, config: PrivacyConfig) -> np.ndarray:
    # Clip gradient to bound sensitivity
    norm = np.linalg.norm(gradient)
    if norm > config.max_gradient_norm:
        gradient = gradient * (config.max_gradient_norm / norm)

    # Add calibrated Gaussian noise
    noise_std = config.noise_multiplier * config.max_gradient_norm / config.epsilon
    noise = np.random.normal(0, noise_std, gradient.shape)

    return gradient + noise
```

### Data at Rest

```yaml
# Encryption configuration
encryption:
  algorithm: AES-256-GCM
  key_derivation: HKDF-SHA256

storage:
  # Holochain DHT entries are encrypted with agent keys
  dht_encryption: enabled

  # PostgreSQL (if used for metrics)
  postgres:
    ssl_mode: verify-full
    encryption: tde  # Transparent Data Encryption
```

### Data in Transit

All communications use TLS 1.3 with mTLS for node-to-node:

```rust
// Gradient submission includes integrity proof
pub struct SignedGradient {
    pub gradient: Vec<f32>,
    pub signature: [u8; 64],
    pub proof_hash: [u8; 32],
    pub timestamp: u64,
}
```

---

## Byzantine Attack Resistance

### Detection Configuration

```toml
# config/byzantine.toml

[detection]
# Multi-layer detection with configurable thresholds
krum_threshold = 0.7
trimmed_mean_trim_ratio = 0.1
foolsgold_similarity_threshold = 0.8

# Temporal analysis
temporal_window_size = 10
temporal_volatility_threshold = 2.0

# Cartel detection (TCDM)
cartel_detection_enabled = true
cartel_similarity_threshold = 0.95
min_cartel_size = 3

[adaptive_defense]
enabled = true
initial_method = "trimmed_mean"
escalation_threshold = 0.3  # Escalate if >30% anomalies
max_escalation_level = 3

[reputation]
initial_score = 0.5
decay_rate = 0.01
recovery_bonus = 0.05
min_rounds_for_trust = 5
```

### Attack-Specific Mitigations

| Attack Type | Detection Method | Response |
|-------------|------------------|----------|
| Gradient scaling | Norm analysis | Flag + investigate |
| Sign flip | Direction variance | Immediate flag |
| Random noise | Statistical outlier | Flag + reduce weight |
| Adaptive | Ensemble detection | Escalate defense |
| Free-rider | Contribution analysis | Shapley = 0 |
| Cartel/Sybil | TCDM + similarity | Group ban |

---

## Container Security

### Base Image Hardening

```dockerfile
# Use distroless or minimal base
FROM gcr.io/distroless/cc-debian12

# Run as non-root
USER nonroot:nonroot

# No shell access
SHELL ["/nonexistent"]

# Read-only filesystem
# (Configured in orchestration)
```

### Kubernetes Security

```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: mycelix-restricted
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
```

### Resource Limits

```yaml
resources:
  limits:
    cpu: "2"
    memory: "4Gi"
    ephemeral-storage: "1Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"
```

---

## Secrets Management

### Sealed Secrets (Kubernetes)

```yaml
# Use Bitnami Sealed Secrets
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mycelix-credentials
spec:
  encryptedData:
    database-password: AgBy3i4OJSWK+PiTySYZZA...
    jwt-secret: AgAR5N8JFH3sGh2D+k1pQA...
```

### Key Rotation

```bash
# Rotate signing keys every 90 days
#!/bin/bash
# key_rotation.sh

# Generate new keypair
NEW_KEY=$(openssl rand -hex 32)

# Update Kubernetes secret
kubectl create secret generic mycelix-keys \
  --from-literal=signing-key=$NEW_KEY \
  --dry-run=client -o yaml | kubectl apply -f -

# Notify nodes of rotation (gradual rollout)
curl -X POST https://coordinator/admin/key-rotation \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"rotation_window": "24h"}'
```

### Environment Variables

```bash
# NEVER store secrets in:
# - Environment variables in code
# - Container build args
# - Git repositories
# - Log output

# DO use:
# - Kubernetes Secrets (encrypted at rest)
# - HashiCorp Vault
# - AWS Secrets Manager
# - Sealed Secrets
```

---

## Monitoring & Alerting

### Security Metrics

```yaml
# Prometheus security alerts
groups:
  - name: security
    rules:
      - alert: HighByzantineDetectionRate
        expr: rate(fl_byzantine_detected_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Elevated Byzantine detection rate"

      - alert: AuthenticationFailures
        expr: rate(auth_failures_total[5m]) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High authentication failure rate"

      - alert: UnauthorizedAccessAttempt
        expr: unauthorized_access_attempts_total > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Unauthorized access attempt detected"
```

### Audit Logging

```rust
// Comprehensive audit trail
#[derive(Debug, Serialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub actor: String,
    pub resource: String,
    pub action: String,
    pub outcome: Outcome,
    pub details: serde_json::Value,
    pub ip_address: Option<IpAddr>,
    pub request_id: Uuid,
}

pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    ConfigurationChange,
    GradientSubmission,
    AggregationRound,
    ByzantineDetection,
    ReputationChange,
    KeyRotation,
}
```

---

## Compliance Considerations

### GDPR

| Requirement | Implementation |
|-------------|----------------|
| Data minimization | Only collect necessary gradients |
| Right to erasure | DHT tombstone support |
| Data portability | Export in standard formats |
| Consent | Node registration agreement |
| Privacy by design | Differential privacy built-in |

### HIPAA (Healthcare FL)

| Safeguard | Implementation |
|-----------|----------------|
| Access controls | Role-based authorization |
| Audit controls | Comprehensive logging |
| Integrity controls | Gradient signatures |
| Transmission security | TLS 1.3 + mTLS |
| PHI protection | Never transmit raw data |

### SOC 2

| Trust Principle | Controls |
|-----------------|----------|
| Security | All sections above |
| Availability | HA deployment, monitoring |
| Processing Integrity | Byzantine detection |
| Confidentiality | Encryption, access control |
| Privacy | Differential privacy |

---

## Audit Checklist

### Pre-Deployment

- [ ] All secrets stored in secure secret manager
- [ ] TLS certificates valid and properly configured
- [ ] Network segmentation verified
- [ ] Firewall rules reviewed and minimal
- [ ] Container images scanned for vulnerabilities
- [ ] Dependencies audited (`cargo audit`, `pip-audit`)
- [ ] RBAC policies configured
- [ ] Backup procedures tested

### Runtime

- [ ] Monitoring dashboards operational
- [ ] Alerting configured and tested
- [ ] Log aggregation working
- [ ] Byzantine detection enabled
- [ ] Rate limiting active
- [ ] Health checks passing

### Periodic (Monthly)

- [ ] Review access logs for anomalies
- [ ] Rotate API keys and tokens
- [ ] Update dependencies
- [ ] Review Byzantine detection effectiveness
- [ ] Test incident response procedures
- [ ] Verify backup restoration

### Incident Response

```
1. DETECT
   - Alert fires or anomaly reported
   - Gather initial evidence

2. CONTAIN
   - Isolate affected nodes
   - Block suspicious IPs
   - Preserve logs

3. ERADICATE
   - Identify root cause
   - Remove malicious nodes
   - Patch vulnerabilities

4. RECOVER
   - Restore from clean state
   - Verify integrity
   - Resume operations

5. LEARN
   - Post-incident review
   - Update detection rules
   - Improve procedures
```

---

## Holochain Zome Checklist

For all Holochain zomes (agents, FL, bridge, identity, governance), apply the concrete patterns in:

- [`docs/ZOME_SECURITY_CHECKLIST.md`](./ZOME_SECURITY_CHECKLIST.md)

That checklist translates this guide into specific practices at the zome boundary (input validation, rate limiting, authorization/ownership checks, DoS mitigation, and Byzantine-aware behavior).

---

## Quick Reference

### Security Contacts

- Security Team: security@luminousdynamics.com
- Emergency: +1-XXX-XXX-XXXX
- Bug Bounty: hackerone.com/mycelix

### Useful Commands

```bash
# Check for vulnerabilities
cargo audit
pip-audit
npm audit

# Verify TLS configuration
openssl s_client -connect coordinator:3000 -tls1_3

# Test Byzantine detection
cargo run --bin fl-demo -- --byzantine 5 --attack adaptive

# Generate audit report
mycelix audit --output audit-report.json
```

---

*Last updated: January 2026*
*Version: 1.0.0*
