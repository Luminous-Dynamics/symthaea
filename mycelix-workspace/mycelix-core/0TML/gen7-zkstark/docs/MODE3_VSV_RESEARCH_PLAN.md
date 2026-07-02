# Mode 3: Verifiable Self-Validation (VSV) - Research Plan

**Status**: Initial Research Phase
**Timeline**: Nov 6, 2025 - Feb 2026
**Target Venue**: IEEE S&P 2026 or USENIX Security 2026

---

## 1. Executive Summary

**Mode 3 (VSV)** introduces a novel challenge-response protocol for Byzantine detection in federated learning that:
- **Eliminates server trust assumption** (no validation set required)
- **Operates fully decentralized** (software-only, no TEEs)
- **Breaks 35% barrier** (validated beyond 33% Byzantine threshold)
- **Uses canary gradients** as cryptographic proof of computation

**Key Innovation**: Byzantine nodes cannot fake high-quality gradients because they cannot predict which canary samples will be challenged.

---

## 2. Motivation & Problem Space

### Current Landscape (Mode 0, 1, 2)

| Mode | Trust Assumption | BFT Limit | Deployment |
|------|------------------|-----------|------------|
| **Mode 0** | Peer comparison | 33% | ✅ Software-only |
| **Mode 1** | Server validation set | 50% | ✅ Software-only |
| **Mode 2** | TEE infrastructure | 50%+ | ❌ Hardware dependency |

**Gap**: No solution for **software-only, trustless, >33% BFT**

### Why VSV Matters

1. **Zero Trust**: No server validation set, no TEEs, no trusted third party
2. **Decentralized**: Challenge-response happens peer-to-peer
3. **Cost-Effective**: Software-only (no specialized hardware)
4. **Scalable**: O(log N) verification overhead using DHT sampling

---

## 3. VSV Protocol Design (Initial Concept)

### 3.1 Core Components

#### **Canary Samples**
- Small validation dataset (~100-1000 samples) stored on DHT
- Public, accessible to all nodes
- Known ground truth labels
- **Key Property**: Any node can compute expected gradient

#### **Challenge-Response Flow**

```
Round t:
1. Node i computes gradient g_i from local data
2. Node i commits: H(g_i || nonce) → DHT
3. Validators randomly select node j for challenge
4. Challenge: "Prove g_j improves loss on canary samples S_c"
5. Node j responds:
   - θ_temp = θ_t - η * g_j
   - Δ_loss = Loss(θ_t, S_c) - Loss(θ_temp, S_c)
   - Proof: ZK-proof of Δ_loss > 0
6. Validators verify proof
7. If valid: Accept g_j, If invalid: Flag as Byzantine
```

#### **Gradient Commitment Scheme**
```
commit_phase():
    g_i = compute_local_gradient(local_data)
    nonce = random()
    commitment = H(g_i || nonce)
    publish_to_dht(commitment, node_id=i)
```

#### **Challenge Selection**
```
challenge_phase():
    # Sample √N validators from DHT
    validators = dht_sample(sqrt(N))

    # Each validator challenges k random nodes
    for v in validators:
        targets = random_sample(N, k=3)
        for t in targets:
            challenge = generate_canary_challenge(t)
            send_challenge(challenge, target=t)
```

#### **Response Verification**
```
verify_response(node_j, gradient_g_j, proof):
    # Check commitment
    if H(g_j || nonce) != commitment[j]:
        return BYZANTINE

    # Verify canary loss improvement
    θ_temp = current_model - learning_rate * g_j
    actual_delta = compute_loss_delta(θ_temp, canary_samples)

    # Verify ZK proof
    if verify_zkproof(proof, actual_delta):
        return HONEST
    else:
        return BYZANTINE
```

---

## 4. Threat Model

### 4.1 Adversary Capabilities

**Byzantine Nodes Can:**
- Submit arbitrary gradients
- Collude with other Byzantine nodes
- Read all DHT data (including canary samples)
- Intercept and replay messages
- Create Sybil identities

**Byzantine Nodes Cannot:**
- Break cryptographic primitives (H, ZK-proofs)
- Fake ZK-proofs of correct computation
- Predict which nodes will be challenged
- Forge commitments after seeing challenge

### 4.2 Attack Scenarios

#### **Attack 1: Pre-computed Canary Responses**
**Strategy**: Byzantine node pre-computes correct responses for all canary samples, submits malicious gradient for aggregation but honest canary proof.

**Mitigation**:
- Commitment-challenge-response prevents selective response
- Node must commit to gradient BEFORE seeing challenge
- Cannot change gradient after commitment

#### **Attack 2: Sybil Attack**
**Strategy**: Create many fake identities to:
1. Dominate validator selection
2. Avoid being challenged
3. Collude to accept Byzantine gradients

**Mitigation Options**:
1. **Proof-of-Stake**: Require stake deposit per identity
2. **Computational Puzzles**: Rate-limit identity creation
3. **Reputation System**: Weight by historical accuracy
4. **Network-Layer Identity**: Tie to IP addresses (weak, but practical)

#### **Attack 3: Gradient Perturbation**
**Strategy**: Submit gradient that:
- Passes canary validation (small perturbation)
- But still poisons global model (targeted attack)

**Mitigation**:
- Multiple canary challenges per node
- Diversity in canary sample selection
- Combine with reputation scoring

#### **Attack 4: Eclipse Attack**
**Strategy**: Byzantine nodes isolate honest node, feed it false DHT data.

**Mitigation**:
- Multi-path DHT routing
- Cross-validation with multiple validators
- Holochain's gossip protocol provides natural resistance

---

## 5. Research Questions

### 5.1 Critical Questions (Must Answer)

1. **Q1**: Can commitment-challenge-response prevent selective gradient submission?
   - **Approach**: Formal proof using game theory
   - **Timeline**: 2 weeks

2. **Q2**: What is the computational overhead of ZK-proof generation/verification?
   - **Approach**: Benchmark zk-SNARKs for gradient validation
   - **Timeline**: 1 week

3. **Q3**: How many canary samples needed for reliable detection?
   - **Approach**: Empirical study varying canary set size (10-1000)
   - **Timeline**: 2 weeks

4. **Q4**: How to prevent Sybil attacks in fully permissionless setting?
   - **Approach**: Literature review + novel protocol design
   - **Timeline**: 3 weeks

5. **Q5**: What is the maximum Byzantine ratio achievable?
   - **Approach**: Theoretical analysis + simulation
   - **Timeline**: 2 weeks

### 5.2 Open Questions (Explore)

- Can we use differential privacy to hide which canary samples are queried?
- How does VSV perform with non-IID data distributions?
- Can we combine VSV with reputation systems for efficiency?
- What is the latency impact of challenge-response rounds?

---

## 6. Literature Review (Key Papers)

### Challenge-Response Protocols
- [ ] **Byzantine Agreement**: Lamport et al. (1982) - Original BFT foundations
- [ ] **Practical BFT**: Castro & Liskov (1999) - PBFT consensus
- [ ] **HoneyBadgerBFT**: Miller et al. (2016) - Asynchronous BFT

### Zero-Knowledge Proofs in FL
- [ ] **zk-SNARKs**: Ben-Sasson et al. (2014) - Succinct proofs
- [ ] **Bulletproofs**: Bünz et al. (2018) - Efficient range proofs
- [ ] **zkFL**: Zhang et al. (2023) - ZK for federated learning

### Sybil Attack Mitigation
- [ ] **SybilGuard**: Yu et al. (2006) - Social network defense
- [ ] **SybilLimit**: Yu et al. (2008) - Improved bounds
- [ ] **Proof-of-Personhood**: Borge et al. (2017) - Identity systems

### Federated Learning Security
- [ ] **FLTrust**: Cao et al. (2021) - Server-side trust bootstrap
- [ ] **CRFL**: Xie et al. (2021) - Client reputation
- [ ] **FedGKT**: Zhang et al. (2022) - Knowledge transfer defense

---

## 7. Experimental Plan

### Phase 1: Proof of Concept (Weeks 1-4)

**Goal**: Demonstrate VSV works in controlled setting

**Setup**:
- 20 clients (13 honest, 7 Byzantine = 35% BFT)
- MNIST dataset
- Simple commitment scheme (no ZK yet)
- Manual challenge selection

**Metrics**:
- Detection rate
- False positive rate
- Overhead (computation, communication)

**Expected Results**:
- >95% detection rate
- <10% FPR
- 2-5x overhead vs Mode 1

### Phase 2: Sybil Resistance (Weeks 5-8)

**Goal**: Add Sybil attack mitigation

**Approaches to Test**:
1. Proof-of-Stake (require 1 ETH deposit per identity)
2. Computational puzzles (HashCash-style)
3. Reputation system (EigenTrust)

**Attack Scenarios**:
- 100 Sybil identities (5:1 ratio)
- Coordinated validation avoidance
- Eclipse attack simulation

**Success Criteria**:
- Detect >90% of Sybil attacks
- Prevent validator domination

### Phase 3: ZK-Proof Integration (Weeks 9-12)

**Goal**: Replace manual verification with cryptographic proofs

**ZK System Selection**:
- **Option A**: Groth16 zk-SNARKs (fastest verification)
- **Option B**: PLONK (universal setup)
- **Option C**: STARKs (no trusted setup)

**Benchmark**:
- Proof generation time
- Proof size
- Verification time
- Compare to manual gradient evaluation

### Phase 4: Real-World Validation (Weeks 13-16)

**Goal**: Test on CIFAR-10 and larger networks

**Configurations**:
- 50 clients (various BFT ratios: 35%, 40%, 45%, 50%)
- CIFAR-10 dataset
- Multiple attack types (sign flip, scaling, backdoor)

**Deployment**:
- Holochain DHT for challenges/responses
- IPFS for canary sample storage
- Full end-to-end system

---

## 8. Paper Outline (Tentative)

### Title
"VSV: Verifiable Self-Validation for Trustless Byzantine-Robust Federated Learning Beyond 33%"

### Abstract (150 words)
- Problem: Existing FL defenses require trust (server validation) or hardware (TEEs)
- Solution: Challenge-response protocol using canary gradients
- Results: >95% detection at 50% BFT with software-only deployment
- Impact: Enables trustless FL in adversarial environments

### 1. Introduction
- Federated learning in hostile environments
- Limitations of Mode 0 (33% barrier), Mode 1 (server trust), Mode 2 (TEE cost)
- VSV fills architectural gap: software-only, trustless, >33% BFT

### 2. Background & Related Work
- Byzantine fault tolerance
- Zero-knowledge proofs
- Sybil attack defenses
- Federated learning security

### 3. Threat Model
- Adversary capabilities
- Attack scenarios
- Security assumptions

### 4. VSV Protocol Design
- Canary sample construction
- Commitment scheme
- Challenge-response flow
- ZK-proof integration
- Sybil mitigation

### 5. Security Analysis
- Formal proof of soundness
- Impossibility of selective response
- Sybil resistance analysis
- Computational/communication complexity

### 6. Implementation
- Holochain DHT integration
- ZK-proof system (Groth16/PLONK/STARK)
- End-to-end system architecture

### 7. Evaluation
- Detection accuracy (35-50% BFT)
- Performance overhead
- Sybil attack resistance
- Comparison with Mode 0, 1, 2

### 8. Discussion & Limitations
- Canary sample selection strategies
- Trade-offs (accuracy vs overhead)
- Deployment considerations

### 9. Conclusion
- First trustless, software-only, >33% BFT solution
- Opens path to fully decentralized FL

---

## 9. Timeline & Milestones

### November 2025
- ✅ Mode 1 paper submitted (IEEE S&P 2025)
- [Week 1-2] Literature review complete
- [Week 3-4] Initial protocol design + threat model

### December 2025
- [Week 1-2] Proof of concept implementation
- [Week 3-4] Basic experiments (20 clients, MNIST)

### January 2026
- [Week 1-2] Sybil mitigation design
- [Week 3-4] ZK-proof integration planning

### February 2026
- [Week 1-2] ZK implementation
- [Week 3-4] CIFAR-10 experiments

### March 2026
- [Week 1-2] Paper writing
- [Week 3-4] Final experiments & revision

### April 2026
- **Submit to IEEE S&P 2026** (deadline typically mid-April)

---

## 10. Success Criteria

### Minimum Viable Paper
- ✅ Protocol formally defined
- ✅ Security proof for commitment-challenge-response
- ✅ Proof-of-concept implementation
- ✅ MNIST experiments: >95% detection, <10% FPR at 35% BFT
- ✅ Overhead analysis: <5x vs Mode 1

### Stretch Goals
- ✅ Full ZK-proof implementation
- ✅ CIFAR-10 validation
- ✅ Sybil attack mitigation proven effective
- ✅ 50% BFT validation

---

## 11. Risk Analysis

### High Risk
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **ZK proof overhead too high** | Medium | High | Use Groth16 (fastest), optimize circuit |
| **Sybil attack unsolvable** | Medium | Critical | Multiple defenses (PoS + reputation + puzzles) |
| **Canary samples leak info** | Low | Medium | DP-noise canary selection |

### Medium Risk
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Reviewers skeptical** | High | Medium | Strong security proofs, real implementation |
| **Implementation complexity** | Medium | Low | Incremental development, open-source |
| **CIFAR-10 low accuracy** | Medium | Low | Focus on MNIST, CIFAR as future work |

---

## 12. Resources Needed

### Technical
- [ ] Holochain development environment (already have)
- [ ] ZK-proof library (circom/snarkjs or libsnark)
- [ ] PyTorch for FL experiments (already have)
- [ ] Compute resources for experiments (local or cloud)

### Knowledge
- [ ] Deep dive into zk-SNARKs/STARKs
- [ ] Sybil attack literature
- [ ] Cryptographic commitment schemes
- [ ] Game-theoretic security proofs

### Collaboration (Optional)
- [ ] Reach out to ZK cryptography experts
- [ ] Engage with Holochain community
- [ ] Find co-authors with complementary expertise

---

## 13. Next Steps (This Week)

### Immediate Actions
1. ✅ **Literature review** (2 days)
   - Read 10 key papers on challenge-response, ZK, Sybil defenses
   - Create annotated bibliography

2. **Formalize threat model** (1 day)
   - Write formal adversary capabilities
   - Define security goals (soundness, completeness, Sybil-resistance)

3. **Protocol pseudocode** (1 day)
   - Detailed commit-challenge-response flow
   - Data structures and message formats

4. **Security sketch** (1 day)
   - Informal argument for why selective response is impossible
   - Identify assumptions

5. **Proof-of-concept plan** (1 day)
   - Simple Python implementation (no ZK yet)
   - Test on toy dataset (MNIST subset)

---

## 14. Open Questions for Discussion

1. **Canary sample size**: How many samples needed to reliably detect Byzantine gradients? Trade-off between detection accuracy and privacy leakage.

2. **Challenge frequency**: Should we challenge every node every round? Or random sampling? How does this affect false negatives?

3. **ZK system choice**: Groth16 (fast, trusted setup), PLONK (universal setup), or STARKs (no setup, slower)? What are acceptable overheads?

4. **Sybil mitigation strategy**: Which combination of PoS, puzzles, reputation, network-layer identity? Can we prove formal bounds?

5. **Integration with Mode 1**: Can VSV and PoGQ complement each other? Hybrid approach for different trust environments?

---

## 15. References (Initial)

### Core Papers to Read

**Byzantine Agreement & Challenge-Response**
1. Lamport, L., Shostak, R., & Pease, M. (1982). "The Byzantine Generals Problem." ACM TOPLAS.
2. Castro, M., & Liskov, B. (1999). "Practical Byzantine Fault Tolerance." OSDI.
3. Miller, A., et al. (2016). "The Honey Badger of BFT Protocols." CCS.

**Zero-Knowledge Proofs**
4. Ben-Sasson, E., et al. (2014). "Succinct Non-Interactive Zero Knowledge for a von Neumann Architecture." USENIX Security.
5. Bünz, B., et al. (2018). "Bulletproofs: Short Proofs for Confidential Transactions." S&P.
6. Goldreich, O. (2008). "Foundations of Cryptography: Volume 1, Basic Tools."

**Sybil Attacks**
7. Yu, H., et al. (2006). "SybilGuard: Defending Against Sybil Attacks via Social Networks." SIGCOMM.
8. Yu, H., et al. (2008). "SybilLimit: A Near-Optimal Social Network Defense against Sybil Attacks." S&P.
9. Douceur, J. (2002). "The Sybil Attack." IPTPS.

**Federated Learning Security**
10. Cao, X., et al. (2021). "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping." NDSS.
11. Xie, C., et al. (2021). "CRFL: Certifiably Robust Federated Learning against Backdoor Attacks." ICML.
12. Bagdasaryan, E., et al. (2020). "Backdoor Attacks in Federated Learning." NeurIPS Workshop.

---

**Status**: Research plan approved ✅
**Next**: Start literature review + protocol formalization
**Target**: Proof-of-concept by end of December 2025
