# Praxis Threat Model v0.1

This document analyzes security threats to Mycelix Praxis and our mitigation strategies.

## Threat Actors

### 1. Malicious Learner

**Capabilities**:
- Control one or more agent identities
- Submit arbitrary FL updates
- Withhold participation or data

**Motivation**:
- Degrade model quality
- Inject backdoors
- Gain unfair credentials

### 2. Malicious Coordinator

**Capabilities**:
- Create FL rounds
- Request gradients from participants
- Perform aggregation

**Motivation**:
- Infer private training data
- Bias model toward specific outcomes
- Censor participants

### 3. Malicious Issuer

**Capabilities**:
- Issue verifiable credentials
- Revoke credentials

**Motivation**:
- Issue fraudulent credentials
- Selectively revoke legitimate credentials
- Reputation attacks

### 4. Sybil Attacker

**Capabilities**:
- Create many fake identities
- Coordinate malicious updates across identities

**Motivation**:
- Amplify poisoning attacks
- Manipulate DAO votes
- Overwhelm aggregation defenses

### 5. Passive Adversary (Observer)

**Capabilities**:
- Read DHT entries
- Analyze network traffic
- Correlate agent activities

**Motivation**:
- Infer private information
- De-anonymize agents
- Track learning patterns

---

## Attack Vectors

### A. Model Poisoning

**Description**: Attacker submits malicious gradients to degrade model accuracy

**Impact**: HIGH - Model becomes unusable for all learners

**Scenarios**:
1. **Random noise**: Submit random gradients → model diverges
2. **Label flipping**: Train on flipped labels → model learns wrong patterns
3. **Targeted poisoning**: Degrade specific classes/features

**Mitigations**:
- ✅ **Trimmed mean aggregation** (default): Removes top/bottom 10% outliers
- ✅ **Median aggregation** (option): 50% breakdown point
- ✅ **Clipping**: Bounds gradient norm (prevents extreme values)
- ⏳ **Validation**: Coordinator tests aggregated model on holdout set
- ⏳ **Reputation**: Weight updates by historical accuracy
- 🔮 **Byzantine-robust aggregators**: Krum, Bulyan (future)

**Residual Risk**: MEDIUM - Sophisticated attacks within trimmed range

---

### B. Backdoor Injection

**Description**: Attacker injects hidden behavior (e.g., misclassify when trigger present)

**Impact**: HIGH - Compromises model integrity, hard to detect

**Scenarios**:
1. Trigger-based: Model misbehaves on specific input pattern
2. Semantic: Backdoor activates on natural inputs (e.g., certain phrases)

**Mitigations**:
- ✅ **Robust aggregation**: Dilutes backdoor signal
- ⏳ **Anomaly detection**: Flag updates with unusual activation patterns
- ⏳ **Holdout testing**: Test for known backdoor triggers
- 🔮 **Certified defenses**: Provable backdoor removal (future)
- 🔮 **Community auditing**: DAO-funded security reviews

**Residual Risk**: MEDIUM-HIGH - Subtle backdoors hard to detect

---

### C. Sybil Attacks

**Description**: Attacker creates multiple identities to amplify influence

**Impact**: HIGH - Defeats robust aggregation, skews DAO votes

**Scenarios**:
1. **FL amplification**: 30% fake identities coordinate poisoning → breaks 10% trim threshold
2. **DAO manipulation**: Vote brigading on governance proposals

**Mitigations**:
- ⏳ **Identity proofing**: Require device attestation or social proof
- ⏳ **Stake-weighted participation**: Agents must bond tokens
- ⏳ **Reputation**: New agents have lower voting power
- 🔮 **Proof of personhood**: Integration with Gitcoin Passport, BrightID, etc.
- 🔮 **Rate limiting**: Cap rounds per agent per time period

**Residual Risk**: HIGH - Hard problem without central authority

---

### D. Inference Attacks

**Description**: Infer private training data from gradients or model updates

**Impact**: MEDIUM - Privacy violation, potential PII leakage

**Scenarios**:
1. **Gradient inversion**: Reconstruct inputs from gradients (especially in small batches)
2. **Membership inference**: Determine if specific data point was in training set
3. **Model inversion**: Extract features of training data

**Mitigations**:
- ✅ **Gradient clipping**: Bounds information leakage per update
- ✅ **Commitment scheme**: Only publish hash, not gradient (until aggregation)
- ⏳ **Differential privacy**: Add calibrated noise to gradients
- ⏳ **Minimum batch size**: Require ≥32 samples per update
- 🔮 **Secure aggregation**: Encrypt gradients, aggregate in MPC

**Residual Risk**: MEDIUM - DP degrades utility, secure agg. is complex

---

### E. Data Poisoning (Local)

**Description**: Attacker trains on malicious data before submitting update

**Impact**: MEDIUM - Similar to model poisoning but harder to detect

**Scenarios**:
1. Mislabeled training data
2. Adversarial examples in dataset

**Mitigations**:
- ✅ **Same as model poisoning mitigations**
- ⏳ **Validation loss checks**: Reject updates with suspiciously high local loss
- 🔮 **Peer review**: Participants cross-validate each other's datasets (opt-in)

**Residual Risk**: MEDIUM

---

### F. Denial of Service

**Description**: Disrupt FL rounds or DHT operations

**Impact**: LOW-MEDIUM - Temporary degradation, not persistent

**Scenarios**:
1. **Spam rounds**: Create many fake rounds
2. **Ghost participants**: Join but never submit updates (stall rounds)
3. **DHT flooding**: Publish junk entries

**Mitigations**:
- ✅ **Timeouts**: Rounds auto-fail if insufficient participation
- ⏳ **Bonding**: Require stake to create rounds or join
- ⏳ **Rate limiting**: Max rounds per agent per day
- 🔮 **Proof of work**: Small PoW to publish entries

**Residual Risk**: LOW - Economic disincentives

---

### G. Credential Forgery

**Description**: Issue or present fraudulent credentials

**Impact**: MEDIUM - Undermines trust in credentials

**Scenarios**:
1. **Fake issuer**: Attacker poses as legitimate issuer
2. **Credential tampering**: Modify credential after issuance
3. **Replay**: Present someone else's credential

**Mitigations**:
- ✅ **Cryptographic signatures**: Ed25519 signatures on VCs
- ✅ **Issuer registry**: DAO-curated list of trusted issuers
- ✅ **Holder binding**: Credential tied to agent public key
- ⏳ **Revocation checking**: Verifiers check `credentialStatus`
- 🔮 **DID integration**: Use DIDs for issuers and holders

**Residual Risk**: LOW - Crypto prevents tampering; social layer for issuer trust

---

### H. Malicious Coordinator

**Description**: Coordinator abuses aggregation role

**Impact**: MEDIUM-HIGH - Can infer data, bias results

**Scenarios**:
1. **Gradient theft**: Request gradients but don't aggregate
2. **Biased aggregation**: Cherry-pick updates, ignore others
3. **Censorship**: Exclude specific participants

**Mitigations**:
- ⏳ **Transparent logging**: All coordinator actions on-chain
- ⏳ **Multi-coordinator verification**: Multiple coordinators aggregate independently, compare
- ⏳ **Participant auditing**: Participants verify their update was included
- 🔮 **Decentralized coordination**: DAO-elected coordinator rotation
- 🔮 **Verifiable aggregation**: ZK proof of correct aggregation

**Residual Risk**: MEDIUM - Requires trust or complex crypto

---

### I. Censorship

**Description**: Prevent agents from participating or publishing

**Impact**: MEDIUM - Harms inclusivity, potential for bias

**Scenarios**:
1. **Network-level**: Block DHT connections
2. **Application-level**: Coordinator rejects join requests
3. **Social**: DAO votes to ban agents

**Mitigations**:
- ✅ **Decentralized DHT**: No single point of control
- ⏳ **Appeal mechanism**: Rejected agents can appeal to DAO
- ⏳ **Multiple coordinators**: Alternative rounds if one is hostile
- 🔮 **Privacy-preserving participation**: Anonymous join (ZK credentials)

**Residual Risk**: LOW-MEDIUM - Holochain's P2P nature limits censorship

---

### J. Rollback / Fork Attacks

**Description**: Create conflicting chain histories

**Impact**: LOW - Holochain's agent-centric model limits impact

**Scenarios**:
1. Agent publishes conflicting entries on different branches

**Mitigations**:
- ✅ **Holochain validation**: Invalid chains rejected by peers
- ✅ **Timestamps**: Detect temporal inconsistencies
- ⏳ **Checkpointing**: Periodic immutable snapshots

**Residual Risk**: LOW - Agent reputation damaged

---

## Threat Summary Table

| Threat | Impact | Likelihood | Mitigation Status | Residual Risk |
|--------|--------|------------|-------------------|---------------|
| Model Poisoning | HIGH | MEDIUM | ✅ Implemented | MEDIUM |
| Backdoor Injection | HIGH | LOW | ⏳ Partial | MEDIUM-HIGH |
| Sybil Attacks | HIGH | MEDIUM | ⏳ Partial | HIGH |
| Inference Attacks | MEDIUM | MEDIUM | ✅ Implemented | MEDIUM |
| Data Poisoning | MEDIUM | MEDIUM | ✅ Implemented | MEDIUM |
| Denial of Service | LOW-MEDIUM | LOW | ⏳ Partial | LOW |
| Credential Forgery | MEDIUM | LOW | ✅ Implemented | LOW |
| Malicious Coordinator | MEDIUM-HIGH | LOW | ⏳ Planned | MEDIUM |
| Censorship | MEDIUM | LOW | ✅ Implemented | LOW-MEDIUM |
| Rollback Attacks | LOW | LOW | ✅ Implemented | LOW |

**Legend**:
- ✅ Implemented
- ⏳ Partial / Planned
- 🔮 Future / Research

---

## Risk Acceptance

We accept the following residual risks for v0.1:

1. **Sybil attacks**: Hard problem; iterating on identity solutions
2. **Backdoor injection**: Requires community auditing + DAO oversight
3. **Malicious coordinator**: Starting with trusted coordinators; decentralizing in v1.0

---

## Monitoring & Incident Response

### Detection

- **Anomaly detection**: Flag rounds with high gradient variance or low final accuracy
- **Community reports**: DAO members can report suspicious behavior
- **Automated tests**: Continuous validation on known-good datasets

### Response

1. **Fast path**: Maintainer can emergency-halt active rounds (see `GOVERNANCE.md`)
2. **Investigation**: Review DHT entries, gradient norms, participant histories
3. **Remediation**: Revert to checkpoint, exclude malicious agents, patch vulnerabilities
4. **Disclosure**: Public incident report (90-day embargo if needed)

---

## Future Work

- **Formal security audit**: Pre-v1.0 external review
- **Bug bounty**: Community-driven vulnerability discovery
- **Threat intelligence**: Track attacks on similar FL systems (e.g., Google FL, OpenMined)
- **Red team exercises**: Simulate attacks to test defenses

---

**Version**: 0.1.0
**Last Updated**: 2025-11-15
**Next Review**: 2026-02-15 (quarterly)
