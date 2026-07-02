# Praxis Privacy Model v0.1

This document describes privacy protections and data handling in Mycelix Praxis.

## Core Principles

1. **Data minimization**: Collect only what's necessary
2. **Local-first**: Training data never leaves device
3. **User control**: Agents control data sharing and disclosure
4. **Transparency**: Clear data inventory and flows
5. **Privacy by design**: Privacy built into architecture, not bolted on

---

## Data Inventory

### Private Data (Never Shared)

| Data Type | Location | Purpose | Retention |
|-----------|----------|---------|-----------|
| Learning content interactions | Local device | Personalization | User-controlled |
| Raw training data | Local device | FL model training | User-controlled |
| Quiz answers / exercises | Local device | Assessment | User-controlled |
| Full learning history | Local device | Progress tracking | User-controlled |

**Protection**: Stored in agent's local source chain; never published to DHT.

---

### Limited Disclosure (Opt-in Sharing)

| Data Type | Shared With | Purpose | Protection |
|-----------|-------------|---------|------------|
| Gradient updates | FL coordinator | Model aggregation | Clipped + commitment |
| Validation loss | FL coordinator | Quality assessment | Aggregated statistics |
| Sample count | FL coordinator | Weighted aggregation | No PII |
| Course enrollment | DHT (public) | Discovery / matching | Agent pubkey only |
| Credentials | Verifier (peer-to-peer) | Prove achievement | Holder controls disclosure |

**Protection**: Cryptographic commitments, aggregation, selective disclosure.

---

### Public Data (DHT)

| Data Type | Visibility | Purpose |
|-----------|-----------|---------|
| Course metadata | Public | Discovery |
| FL round announcements | Public | Coordination |
| DAO proposals | Public | Governance |
| Votes | Public | Transparency |
| Gradient commitments | Public | Provenance |

**Note**: All public data is **content-addressed** and **agent-signed**, ensuring integrity.

---

## Privacy Mechanisms

### 1. Federated Learning

**Goal**: Train models without sharing raw data

**How it works**:
1. **Local training**: Each agent trains model on private data
2. **Gradient extraction**: Compute parameter updates (gradients)
3. **Clipping**: Bound gradient norm to threshold (default: 1.0)
4. **Commitment**: Publish only BLAKE3 hash of gradient
5. **Aggregation**: Coordinator combines gradients (trimmed mean/median)
6. **Model update**: Aggregated gradient applied to global model

**Privacy guarantee**: Raw data never leaves device; only gradients (which are aggregates over many samples) shared.

**Limitations**:
- Gradients can leak info about small batches (requires ≥32 samples recommended)
- Sophisticated attacks can partially reconstruct data (see §3 Differential Privacy)

---

### 2. Gradient Clipping

**Goal**: Bound influence of any single agent

**How it works**:
```
if ||gradient|| > clip_norm:
    gradient = gradient * (clip_norm / ||gradient||)
```

**Privacy benefit**:
- Limits information leakage per update
- Prevents outliers from dominating aggregation
- Prerequisite for differential privacy

**Parameters**:
- `clip_norm`: Default 1.0 (configurable per round)

---

### 3. Differential Privacy (Optional)

**Goal**: Formal privacy guarantee against inference attacks

**How it works** (client-side):
1. Clip gradient to norm `C`
2. Add Gaussian noise: `gradient += N(0, σ²I)` where `σ = C * noise_multiplier`
3. Publish noisy gradient

**Privacy guarantee**: (ε, δ)-differential privacy
- ε (epsilon): Privacy budget (smaller = more private)
- δ (delta): Failure probability (typically 10⁻⁵)

**Example parameters**:
- ε = 8, δ = 10⁻⁵, C = 1.0 → noise_multiplier ≈ 0.4

**Trade-off**: Privacy vs. model accuracy
- More noise → stronger privacy → lower accuracy
- Tuning required per application

**Implementation status**: ⏳ Planned for v0.2

---

### 4. Commitment Schemes

**Goal**: Prove data existence without revealing content

**How it works**:
1. Compute commitment: `c = BLAKE3(data)`
2. Publish `c` to DHT
3. Later reveal `data` to specific parties
4. Verifier checks: `BLAKE3(data) == c`

**Use cases**:
- Gradient updates (commit before revealing to coordinator)
- Credential attributes (selective disclosure)

**Privacy benefit**: Public provenance without public data

---

### 5. Selective Disclosure (Credentials)

**Goal**: Prove attributes without revealing full credential

**How it works** (future):
1. Issuer creates credential with attributes `{score: 95, course: "Math101", ...}`
2. Holder stores credential locally
3. Verifier requests proof of `score >= 80`
4. Holder generates ZK proof: "I have a credential where score >= 80" (without revealing exact score)

**Implementation status**: 🔮 Future (requires ZK-SNARK/BBS+ integration)

**Current approach**: Holder shares full credential (verifier sees all attributes)

---

### 6. Pseudonymity

**Agent identity**: Holochain agent public key (Ed25519)

**Properties**:
- **Persistent**: Same key across sessions (for reputation)
- **Pseudonymous**: No direct link to real-world identity (unless agent chooses to reveal)
- **Linkable**: All actions by same agent linkable (by design, for accountability)

**Privacy considerations**:
- Behavioral fingerprinting: Patterns may de-anonymize agents
- Mitigation: Rotate agent keys periodically (breaks reputation linkage)
- Future: ZK credentials for anonymous participation

---

## Data Flow Diagrams

### FL Round (Private Data Stays Local)

```
┌─────────────────┐
│  Learner Agent  │
│  ┌───────────┐  │
│  │  Raw Data │  │  ← NEVER LEAVES DEVICE
│  └───────────┘  │
│        ↓        │
│  ┌───────────┐  │
│  │   Train   │  │  ← LOCAL COMPUTATION
│  └───────────┘  │
│        ↓        │
│  ┌───────────┐  │
│  │ Gradients │  │  ← CLIPPED + NOISED (optional)
│  └───────────┘  │
│        ↓        │
│   [Commitment]  │  → Published to DHT
└────────┬────────┘
         │
         ↓
   ┌──────────┐
   │ DHT      │  ← Only commitments + metadata
   └──────────┘
         │
         ↓
   ┌──────────────┐
   │ Coordinator  │  ← Requests gradients (P2P)
   │   Aggregate  │  ← Combines with robust method
   └──────────────┘
```

### Credential Issuance (Holder Controls Disclosure)

```
┌────────────┐                  ┌────────────┐
│   Issuer   │  1. Issue VC     │   Holder   │
│            │ ───────────────> │  (Agent)   │
└────────────┘                  └──────┬─────┘
                                       │
                                       │ 2. Store locally
                                       │
                                       ↓
                              ┌─────────────────┐
                              │ Local Storage   │
                              │  (Private)      │
                              └────────┬────────┘
                                       │
                                       │ 3. Selective share
                                       │
                                       ↓
                              ┌─────────────────┐
                              │   Verifier      │  ← Holder chooses
                              │   (P2P)         │    what to share
                              └─────────────────┘
```

---

## Privacy Risk Assessment

### High-Risk Scenarios

1. **Small batch gradient leakage**
   - **Risk**: Gradients from <10 samples may reveal individual data points
   - **Mitigation**: Require min_batch_size = 32
   - **Status**: ⏳ To be enforced in v0.2

2. **Correlation attacks**
   - **Risk**: Multiple rounds + same participants → infer data distribution
   - **Mitigation**: Rotate participants, limit rounds per agent
   - **Status**: ⏳ Roadmap

3. **Behavioral fingerprinting**
   - **Risk**: Learning patterns de-anonymize agents
   - **Mitigation**: Option to create new agent identity, mix networks
   - **Status**: 🔮 Future

### Medium-Risk Scenarios

1. **Metadata leakage**
   - **Risk**: Course enrollment, timing patterns reveal interests
   - **Mitigation**: Obfuscate timing, dummy enrollments (opt-in)
   - **Status**: 🔮 Future

2. **Credential linking**
   - **Risk**: Multiple credentials from same holder linkable
   - **Mitigation**: Use pairwise DIDs per verifier
   - **Status**: 🔮 Future

---

## Compliance Considerations

### GDPR (EU)

- **Right to erasure**: Agents control local data; can delete anytime
- **Data minimization**: Only gradients + commitments published (not raw data)
- **Purpose limitation**: Data used only for stated purpose (learning, credentialing)
- **Consent**: Explicit opt-in for FL participation and credential issuance
- **Data portability**: Agents own their source chain (exportable)

**Note**: DHT entries are public and immutable (like blockchain); design for non-PII only.

### FERPA (US Education Privacy)

- **Student records**: Learning data stays local (compliant)
- **Directory information**: Course enrollment public (similar to public directory)
- **Consent**: Required for credential disclosure (holder controls)

### COPPA (Children's Privacy)

- **Age verification**: Not implemented in v0.1 (assume 13+ users)
- **Parental consent**: Deferred to application layer (e.g., school district)

---

## User Controls

### What Users Can Control

1. **Participation**: Opt-in to FL rounds (per round)
2. **Data retention**: Delete local training data anytime
3. **Credential disclosure**: Choose when/who to share credentials with
4. **Agent identity**: Create new agent key (breaks linkability)
5. **DP noise**: Opt-in to stronger privacy (at cost of accuracy)

### What Users Cannot Control

1. **Public DHT data**: Course metadata, proposals (immutable once published)
2. **Commitments**: Once published, can't be deleted (but don't reveal underlying data)
3. **Aggregated models**: Agents can't un-contribute to model (aggregation is lossy)

---

## Privacy Notices

### For Learners

**What we collect**:
- Course enrollment (public)
- Gradient commitments (public hash)
- Validation loss (aggregated)
- Credentials (you control disclosure)

**What we DON'T collect**:
- Your learning content (stays on your device)
- Your quiz answers (stays on your device)
- Your raw training data (never leaves your device)

**Your rights**:
- Delete local data anytime
- Opt out of FL rounds
- Control credential sharing
- Create new agent identity

### For Course Creators

**What you publish**:
- Course metadata (title, description, tags) → public
- Creator agent key → public

**What stays private**:
- Draft materials (until published)
- Learner progress (only visible to learner)

---

## Privacy by Default

### Design Choices

1. **Gradients off-DHT**: Only coordinator can request (not public)
2. **Credentials P2P**: Holder sends directly to verifier (not published)
3. **Progress private**: LearnerProgress entry only readable by agent
4. **Clipping mandatory**: Can't submit unclipped gradients

---

## Future Enhancements

- **Secure aggregation**: Encrypt gradients, aggregate without seeing individuals (MPC)
- **Anonymous credentials**: ZK-SNARKs for attribute proofs
- **Mix networks**: Obfuscate communication patterns
- **Trusted execution**: TEEs for coordinator (SGX, SEV)
- **Auditable privacy**: ZK proofs of correct DP noise addition

---

## Privacy Audit Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2025-11-15 | Initial privacy model | Bootstrap v0.1 |

---

**Version**: 0.1.0
**Last Updated**: 2025-11-15
**Next Review**: 2026-02-15 (quarterly)
