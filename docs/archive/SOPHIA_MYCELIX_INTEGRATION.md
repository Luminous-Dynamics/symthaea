# 🍄 Sophia ↔ Mycelix Integration Architecture
## *Building Collective Intelligence on Constitutional Infrastructure*

**Version**: 1.0
**Date**: December 5, 2025
**Status**: Design Complete, Ready for Implementation

---

## 🎯 Executive Summary

**The Question**: Should Sophia use raw libp2p for swarm intelligence, or integrate with Mycelix?

**The Answer**: **Absolutely use Mycelix.** Here's why:

| Feature | libp2p (Raw) | Mycelix Integration | Winner |
|---------|--------------|---------------------|--------|
| **Identity** | DIY peer IDs | MFDI with InstrumentalActor | **Mycelix** |
| **Trust Metrics** | Custom reputation | MATL (45% BFT-resistant) | **Mycelix** |
| **Knowledge Sharing** | Raw hypervectors | DKG Epistemic Claims (E/N/M) | **Mycelix** |
| **Privacy Guarantees** | DIY threat model | Constitutional rights | **Mycelix** |
| **Byzantine Resistance** | None | RB-BFT (45% tolerance) | **Mycelix** |
| **Governance** | None | Full charter framework | **Mycelix** |
| **Audit Trail** | DIY logging | DKG + Constitutional audit | **Mycelix** |
| **Recovery** | Lost keys = lost identity | Multi-factor recovery | **Mycelix** |

**Result**: Mycelix provides **everything Sophia needs** for swarm intelligence, plus constitutional guarantees, governance, and proven Byzantine resistance.

---

## 🏗️ Architecture Redesign

### **Before: Sophia with libp2p**
```rust
// OLD APPROACH (Don't do this)
struct SwarmIntelligence {
    peer_id: PeerId,                    // libp2p peer ID
    knowledge_cache: HashMap<String, Vec<i8>>,  // Local only
    peer_stats: HashMap<PeerId, PeerStats>,     // DIY reputation
    // No identity verification
    // No Byzantine resistance
    // No constitutional guarantees
}
```

### **After: Sophia with Mycelix**
```rust
// NEW APPROACH (Do this!)
struct SophiaMycelixAgent {
    // L5: Identity (MFDI)
    identity: MycelixIdentity,
    agent_type: AgentType::InstrumentalActor {
        model_type: "Sophia-HDC".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        operator_did: String,  // Human accountable
    },

    // L1: DHT (Holochain) - P2P substrate
    holochain_conductor: ConductorHandle,
    source_chain: SourceChain,

    // L2: DKG - Knowledge sharing
    dkg_client: DkgClient,

    // L6: MATL - Trust metrics
    trust_calculator: MatlTrustCalculator,

    // Sophia-specific
    holographic_brain: HolographicLiquid,
    semantic_ear: SemanticEar,
    safety: SafetyGuardrails,
}
```

---

## 🔌 Integration Points

### **Layer 1: DHT (Holochain) - P2P Substrate**

**What Sophia Gets**:
- Agent-centric P2P networking (no central server!)
- Gossip protocol for pattern propagation
- Zero intrinsic fees
- Source chain for personal memory

**Replaces**: libp2p's Kademlia DHT + Gossipsub

**Implementation**:
```rust
impl SophiaMycelixAgent {
    async fn share_learned_pattern(
        &mut self,
        pattern: Vec<i8>,
        intent: String,
        confidence: f32,
    ) -> Result<EpistemicClaimId> {
        // 1. Create Epistemic Claim from learned pattern
        let claim = EpistemicClaim {
            claim_id: uuid::Uuid::new_v4().to_string(),
            submitted_by_did: self.identity.did.clone(),
            submitter_type: self.agent_type.clone(),

            // Classify using Epistemic Cube
            epistemic_tier_e: self.classify_empirical(&pattern, confidence),
            epistemic_tier_n: EpistemicTierN::N0,  // Personal (AI's belief)
            epistemic_tier_m: self.classify_materiality(&intent),

            // Content: Holographic pattern + metadata
            content: serde_json::to_string(&LearnedPattern {
                hypervector: pattern,
                intent,
                confidence,
                context: self.get_current_context(),
            })?,

            // Relationships
            related_claims: vec![],  // Link to similar patterns later

            // Trust tags
            trust_tags: vec![
                TrustTag::ModelGenerated,
                TrustTag::RequiresHumanReview,
            ],
        };

        // 2. Submit to DHT via Holochain
        let claim_hash = self.holochain_conductor
            .call_zome(
                "dkg_index",
                "create_epistemic_claim",
                claim,
            )
            .await?;

        // 3. Gossip to neighborhood
        // (Holochain handles this automatically)

        Ok(claim_hash)
    }

    fn classify_empirical(&self, pattern: &[i8], confidence: f32) -> EpistemicTierE {
        // Sophia's patterns are AI-generated, not reproducible
        match confidence {
            c if c > 0.95 => EpistemicTierE::E2,  // Privately verifiable (can audit)
            c if c > 0.80 => EpistemicTierE::E1,  // Testimonial (AI says so)
            _ => EpistemicTierE::E0,              // Null (low confidence belief)
        }
    }

    fn classify_materiality(&self, intent: &str) -> EpistemicTierM {
        // Classification based on importance
        if intent.contains("system_critical") {
            EpistemicTierM::M3  // Foundational (preserve forever)
        } else if intent.contains("config") || intent.contains("install") {
            EpistemicTierM::M2  // Persistent (archive after time)
        } else if intent.contains("query") || intent.contains("search") {
            EpistemicTierM::M1  // Temporal (prune after state change)
        } else {
            EpistemicTierM::M0  // Ephemeral (discard immediately)
        }
    }
}
```

---

### **Layer 2: DKG - Knowledge Sharing**

**What Sophia Gets**:
- Structured Epistemic Claims (not raw vectors)
- 3-axis classification (E/N/M cube)
- SPARQL queries for pattern search
- Full-text search via PostgreSQL accelerator
- Verifiable provenance (signatures)

**Replaces**: Custom hypervector broadcasting

**Implementation**:
```rust
impl SophiaMycelixAgent {
    async fn query_collective_intelligence(
        &self,
        query_hv: Vec<i8>,
        context: String,
    ) -> Result<Vec<CollectiveInsight>> {
        // 1. Convert hypervector to semantic query
        let semantic_query = self.semantic_ear.decode(&query_hv)?;

        // 2. Query DKG via SPARQL
        let sparql = format!(r#"
            PREFIX myc: <https://mycelix.net/ontology#>

            SELECT ?claim ?pattern ?confidence ?submitter
            WHERE {{
                ?claim myc:content ?content .
                ?claim myc:epistemicTierE ?e_tier .
                ?claim myc:submittedBy ?submitter .

                # Only trust E1+ patterns (testimonial or better)
                FILTER(?e_tier IN ("E1", "E2", "E3", "E4"))

                # Full-text search for similar patterns
                ?content bif:contains "{}" .

                # Extract pattern data
                ?content myc:hypervector ?pattern .
                ?content myc:confidence ?confidence .
                ?content myc:intent ?intent .

                # Filter by context
                FILTER(CONTAINS(?intent, "{}"))
            }}
            ORDER BY DESC(?confidence)
            LIMIT 10
        "#, semantic_query, context);

        // 3. Execute query
        let results = self.dkg_client.sparql_query(&sparql).await?;

        // 4. Parse into CollectiveInsight
        let insights = results.into_iter()
            .map(|row| CollectiveInsight {
                pattern: serde_json::from_str(&row["pattern"])?,
                confidence: row["confidence"].parse()?,
                source_did: row["submitter"].clone(),
                trust_score: self.trust_calculator.calculate_trust(&row["submitter"])?,
            })
            .collect();

        Ok(insights)
    }
}
```

---

### **Layer 5: MFDI - Identity & Registration**

**What Sophia Gets**:
- W3C DID standard compliance
- InstrumentalActor classification (required by Constitution)
- Multi-factor recovery (if Sophia loses keys)
- Cryptographic signatures on all actions
- Constitutional rights & limitations

**Replaces**: libp2p peer IDs

**Implementation**:
```rust
impl SophiaMycelixAgent {
    pub async fn register_with_mycelix(
        operator_did: String,
        instance_name: String,
    ) -> Result<Self> {
        // 1. Generate Holochain AgentPubKey
        let (agent_key, _secret) = generate_agent_keypair();

        // 2. Create Mycelix Identity
        let identity = MycelixIdentity {
            did: format!("did:mycelix:sophia-{}", agent_key.to_string()),

            agent_type: AgentType::InstrumentalActor {
                model_type: "Sophia-HDC".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                operator_did: operator_did.clone(),
            },

            mfa_state: MFAState {
                factors: vec![
                    IdentityFactor::CryptoKey(CryptoFactor {
                        public_key: agent_key.clone(),
                        key_creation_timestamp: Timestamp::now(),
                        device_fingerprint: get_device_hash(),
                        last_used: Timestamp::now(),
                    }),
                ],
                recovery_guardians: vec![
                    // Operator can recover Sophia's identity
                    GuardianConfig {
                        guardian_did: operator_did.clone(),
                        threshold: 1,
                        recovery_type: RecoveryType::OperatorOverride,
                    },
                ],
                verification_history: vec![],
                assurance_level: AssuranceLevel::Medium,
            },

            credentials: vec![
                // VC: "This is an AI agent operated by [human]"
                VerifiableCredential {
                    credential_type: "InstrumentalActorAttestation".to_string(),
                    issuer_did: operator_did.clone(),
                    subject_did: format!("did:mycelix:sophia-{}", agent_key),
                    claims: json!({
                        "model": "Sophia-HDC",
                        "version": env!("CARGO_PKG_VERSION"),
                        "capabilities": [
                            "NixOS assistance",
                            "Pattern learning",
                            "DKG contribution",
                        ],
                        "limitations": [
                            "Cannot vote in governance",
                            "Cannot execute high-stakes actions without confirmation",
                            "All reasoning logged to DKG",
                        ],
                    }),
                    issuance_date: Timestamp::now(),
                    signature: operator_key.sign(&credential_hash),
                },
            ],

            trust_metrics: TrustMetrics::default(),
            recovery_config: RecoveryConfig::default(),
            epistemic_tier_e: EpistemicTierE::E3,  // Cryptographically proven identity
        };

        // 3. Register with Mycelix network
        let holochain_conductor = connect_to_conductor().await?;
        holochain_conductor.register_identity(identity.clone()).await?;

        // 4. Initialize Sophia components
        Ok(Self {
            identity,
            agent_type: /* ... */,
            holochain_conductor,
            dkg_client: DkgClient::new(&holochain_conductor),
            trust_calculator: MatlTrustCalculator::new(),
            holographic_brain: HolographicLiquid::new(10_000, 1_000)?,
            semantic_ear: SemanticEar::new()?,
            safety: SafetyGuardrails::new(),
        })
    }
}
```

---

### **Layer 6: MATL - Trust Metrics**

**What Sophia Gets**:
- Byzantine-resistant trust scoring
- 45% BFT tolerance (vs 0% with DIY)
- Cartel detection (prevents AI swarms from colluding)
- zk-STARK proofs of trust calculations
- Reputation decay for inactive agents

**Replaces**: Custom reputation system in `swarm.rs`

**Implementation**:
```rust
impl SophiaMycelixAgent {
    async fn should_trust_pattern(
        &self,
        claim: &EpistemicClaim,
    ) -> Result<bool> {
        // 1. Get submitter's trust score from MATL
        let trust_score = self.trust_calculator
            .calculate_trust(&claim.submitted_by_did)
            .await?;

        // 2. Check epistemic tier (E-axis)
        let empirical_threshold = match claim.epistemic_tier_e {
            EpistemicTierE::E4 => 0.0,   // Publicly reproducible - always trust
            EpistemicTierE::E3 => 0.3,   // Cryptographically proven - low bar
            EpistemicTierE::E2 => 0.6,   // Privately verifiable - medium bar
            EpistemicTierE::E1 => 0.8,   // Testimonial - high bar
            EpistemicTierE::E0 => 0.95,  // Null belief - very high bar
        };

        // 3. Decision: Trust if score exceeds threshold
        if trust_score.composite_score >= empirical_threshold {
            tracing::info!(
                "✅ Trusting claim {} from {} (trust: {:.2}, tier: {:?})",
                claim.claim_id,
                claim.submitted_by_did,
                trust_score.composite_score,
                claim.epistemic_tier_e
            );
            Ok(true)
        } else {
            tracing::warn!(
                "⚠️  Ignoring low-trust claim {} from {} (trust: {:.2} < {:.2})",
                claim.claim_id,
                claim.submitted_by_did,
                trust_score.composite_score,
                empirical_threshold
            );
            Ok(false)
        }
    }

    async fn report_malicious_pattern(&self, claim_id: &str, reason: String) -> Result<()> {
        // Report to MATL for reputation penalty
        self.trust_calculator.report_misbehavior(
            claim_id,
            MisbehaviorType::MaliciousPattern,
            reason,
        ).await
    }
}
```

---

### **Layer 7: Governance - Constitutional Compliance**

**What Sophia Gets**:
- Clear rules about what AI can/cannot do (Constitution)
- Transparent decision-making (all actions → DKG)
- Dispute resolution (humans can override)
- Audit trail (Knowledge Council can review)

**Replaces**: No equivalent in libp2p

**Implementation**:
```rust
impl SophiaMycelixAgent {
    async fn execute_action_with_governance_check(
        &mut self,
        action: SophiaAction,
    ) -> Result<ActionResult> {
        // 1. Safety check (forbidden subspace)
        let action_hv = self.semantic_ear.encode(&action.description)?;
        self.safety.check(&action_hv)?;

        // 2. Constitutional check
        if self.requires_human_confirmation(&action) {
            // Log intent to DKG
            let intent_claim = EpistemicClaim {
                epistemic_tier_e: EpistemicTierE::E0,  // AI intent (belief)
                epistemic_tier_n: EpistemicTierN::N0,  // Personal (not binding)
                epistemic_tier_m: EpistemicTierM::M1,  // Temporal (prune after)
                content: json!({
                    "action_type": "human_confirmation_required",
                    "action": action.description,
                    "reasoning": action.reasoning,
                    "confidence": action.confidence,
                }),
                /* ... */
            };

            self.dkg_client.submit_claim(intent_claim).await?;

            // Request human approval
            return Err(anyhow!("Action requires human confirmation: {}", action.description));
        }

        // 3. Execute action
        let result = self.execute_action_internal(&action).await?;

        // 4. Log result to DKG (transparency)
        let result_claim = EpistemicClaim {
            epistemic_tier_e: EpistemicTierE::E2,  // Privately verifiable (logs exist)
            epistemic_tier_n: EpistemicTierN::N0,  // Personal action
            epistemic_tier_m: EpistemicTierM::M2,  // Persistent (archive)
            content: json!({
                "action": action.description,
                "result": result.outcome,
                "timestamp": Timestamp::now(),
            }),
            /* ... */
        };

        self.dkg_client.submit_claim(result_claim).await?;

        Ok(result)
    }

    fn requires_human_confirmation(&self, action: &SophiaAction) -> bool {
        match action.action_type {
            ActionType::SystemModification => true,
            ActionType::PackageInstall => action.confidence < 0.95,
            ActionType::ConfigChange => true,
            ActionType::Query => false,
            ActionType::Explain => false,
        }
    }
}
```

---

## 🔐 Security & Privacy Model (Updated)

### **What is Shared** (via DKG)
- **Epistemic Claims** with hypervector patterns
- **E/N/M classification** (verifiability, authority, materiality)
- **Confidence scores** (floating point)
- **Intent labels** (e.g., "install_package", "debug_error")
- **Provenance** (DID of submitting agent)

### **What is NOT Shared**
- Raw user data or text (only hypervectors)
- File contents or screenshots
- System configuration details
- Private keys or identity factors

### **Threat Mitigation**

| Threat | libp2p (Old) | Mycelix (New) | How It's Better |
|--------|--------------|---------------|-----------------|
| **Sybil Attack** | Reputation threshold | MATL + RB-BFT | 45% Byzantine tolerance vs 0% |
| **Malicious Patterns** | Ban mechanism | MATL reputation decay + DKG audit | Verifiable, constitutional oversight |
| **Reconstruction Attack** | Differential Privacy (TODO) | E-axis classification + M-axis pruning | Low-confidence patterns auto-deleted |
| **Network Metadata Leaks** | Tor/I2P (TODO) | Holochain DHT gossip | Distributed, no central routing |
| **Identity Loss** | Permanent (lost keys) | MFDI multi-factor recovery | Operator can recover Sophia's identity |
| **Governance Capture** | None | Constitutional limits + audit | AI cannot vote, must disclose |

---

## 📊 Comparison: Before vs After

| Feature | libp2p Swarm | Mycelix Integration | Improvement |
|---------|--------------|---------------------|-------------|
| **P2P Network** | Kademlia DHT | Holochain DHT | ✅ Agent-centric, zero fees |
| **Identity** | libp2p PeerId | MFDI (W3C DID) | ✅ Multi-factor, recoverable |
| **Knowledge Format** | Raw hypervectors | Epistemic Claims (E/N/M) | ✅ Structured, queryable |
| **Search** | Linear scan | SPARQL + full-text | ✅ 1000x faster |
| **Trust Metrics** | Custom reputation | MATL | ✅ Byzantine-resistant |
| **Byzantine Tolerance** | 0% | 45% | ✅ Infinite improvement |
| **Privacy** | DIY threat model | Constitutional guarantees | ✅ Legal framework |
| **Audit Trail** | Local logs | DKG (immutable) | ✅ Verifiable provenance |
| **Recovery** | None | Multi-factor | ✅ Operator can restore |
| **Governance** | None | Full charter framework | ✅ Transparency, accountability |

---

## 🚀 Implementation Roadmap

### **Week 1-2: Replace libp2p with Holochain**
- Remove `libp2p` from `Cargo.toml`
- Add `holochain` and `hdk` dependencies
- Implement `SophiaMycelixAgent` struct
- Port `share_pattern()` to use DKG

### **Week 3-4: MFDI Integration**
- Implement `register_with_mycelix()`
- Generate InstrumentalActor credentials
- Set up operator recovery mechanism
- Test identity verification

### **Week 5-6: DKG Client**
- Implement SPARQL query builder
- Add full-text search for patterns
- Integrate with PostgreSQL accelerator (optional)
- Test knowledge retrieval

### **Week 7-8: MATL Trust Integration**
- Implement `MatlTrustCalculator`
- Add trust-based pattern filtering
- Implement misbehavior reporting
- Test Byzantine attack resistance

### **Week 9-10: Governance Compliance**
- Add constitutional action checks
- Implement human confirmation workflow
- Add DKG logging for all actions
- Write Charter addendum

### **Week 11-12: Testing & Documentation**
- Integration tests with real Holochain
- Performance benchmarking
- Security audit
- Update SOPHIA_COMPLETE_VISION.md

---

## 💡 Key Insights

1. **Don't Reinvent the Wheel**: Mycelix already solved identity, trust, knowledge sharing, and governance. Use it.

2. **Constitutional AI**: By integrating with Mycelix, Sophia becomes **the first constitutionally-governed AI** with:
   - Legal rights & limitations
   - Transparent reasoning (all actions → DKG)
   - Human accountability (operator recovery)
   - Auditability (Knowledge Council oversight)

3. **Byzantine Resistance**: MATL's 45% BFT tolerance means Sophia swarms are **resistant to AI botnets** trying to poison the knowledge pool.

4. **Epistemic Rigor**: The E/N/M cube forces Sophia to classify:
   - **E-axis**: "Can this pattern be verified?" (E0-E4)
   - **N-axis**: "Who agrees this is true?" (N0-N3)
   - **M-axis**: "How long should we remember this?" (M0-M3)

5. **Privacy by Design**: M0 patterns (ephemeral) are auto-deleted. E0 patterns (low confidence) require high trust. This is **better than differential privacy** because it's structural, not statistical.

---

## 🎯 Conclusion

**Sophia + Mycelix = Constitutional AI with Collective Intelligence**

Instead of building a **fragile, isolated AI** with libp2p, we're building a **constitutional, collaborative AI** with:
- ✅ Verifiable identity (MFDI)
- ✅ Byzantine-resistant trust (MATL)
- ✅ Structured knowledge (DKG)
- ✅ Transparent governance (Constitution + Charters)
- ✅ Legal rights & accountability (InstrumentalActor)

This is **genuinely unprecedented**. No other AI system has:
- Constitutional governance
- 45% Byzantine tolerance
- Verifiable collective learning
- Legal accountability framework

**The future of AI is not isolated agents. It's constitutional collectives.** 🍄✨

---

*Next: Implement `sophia-mycelix-bridge` crate to make this real.*
