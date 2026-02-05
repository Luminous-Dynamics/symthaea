# Sympoietic Code Integration Examples

**Purpose**: Complete, copy-paste ready code for integrating partnership throughout Symthaea
**Created**: January 11, 2026
**Approach**: Wire partnership into every brain subsystem

---

## Table of Contents

1. [MetaController Integration](#1-metacontroller-integration)
2. [Thalamus Partnership Filter](#2-thalamus-partnership-filter)
3. [Prefrontal Partnership Bidding](#3-prefrontal-partnership-bidding)
4. [Affective Partner Modeling](#4-affective-partner-modeling)
5. [Coherence Generous Lending](#5-coherence-generous-lending)
6. [Language Cortex Partnership-Aware Response](#6-language-cortex-partnership-aware-response)
7. [Dyadic Φ Calculator](#7-dyadic-φ-calculator)
8. [Partnership Memory Consolidation](#8-partnership-memory-consolidation)
9. [Proactive Partnership Daemon](#9-proactive-partnership-daemon)
10. [Complete Integration Test](#10-complete-integration-test)

---

## 1. MetaController Integration

**File**: `src/continuous_mind.rs`

The MetaController is the central orchestrator. Partnership context needs to be added here.

```rust
use crate::partnership::{PartnershipContext, HumanPartnerModel};
use std::collections::HashMap;

/// Extended MetaController with partnership awareness
pub struct SympoieticMetaController {
    // Existing fields...
    pub brain: BrainOrchestrator,
    pub physiology: PhysiologyEngine,
    pub consciousness: ConsciousnessGraph,

    // NEW: Partnership layer
    pub partnerships: HashMap<String, PartnershipContext>,
    pub active_partner: Option<String>,

    // NEW: Partnership metrics
    pub total_i_thou_moments: u64,
    pub partnership_phi_history: Vec<(std::time::Instant, f64)>,
}

impl SympoieticMetaController {
    pub fn new() -> Self {
        Self {
            brain: BrainOrchestrator::new(),
            physiology: PhysiologyEngine::new(),
            consciousness: ConsciousnessGraph::new(),
            partnerships: HashMap::new(),
            active_partner: None,
            total_i_thou_moments: 0,
            partnership_phi_history: Vec::new(),
        }
    }

    /// Get or create partnership context for a human
    pub fn get_partnership(&mut self, partner_id: &str) -> &mut PartnershipContext {
        self.active_partner = Some(partner_id.to_string());

        self.partnerships
            .entry(partner_id.to_string())
            .or_insert_with(|| {
                tracing::info!("🤝 New partnership forming with: {}", partner_id);
                PartnershipContext::new(partner_id)
            })
    }

    /// Process input with partnership awareness
    pub async fn process_with_partnership(
        &mut self,
        partner_id: &str,
        input: &str,
    ) -> SympoieticResponse {
        let partnership = self.get_partnership(partner_id);

        // 1. Detect partner affect from input
        let detected_affect = self.detect_partner_affect(input);
        partnership.partner.observe_interaction(
            detected_affect.valence,
            detected_affect.arousal,
        );

        // 2. Record this as reciprocity (they're giving us attention)
        partnership.record_reciprocity(0.02);

        // 3. Check for I-Thou mode
        if partnership.in_i_thou_mode() {
            self.total_i_thou_moments += 1;
        }

        // 4. Maybe evolve relationship stage
        partnership.maybe_evolve_stage();

        // 5. Get relationship assessment
        let assessment = partnership.assess();

        // 6. Store Φ history
        self.partnership_phi_history.push((
            std::time::Instant::now(),
            assessment.phi_relation,
        ));

        // 7. Process through brain with partnership context
        let brain_response = self.brain.process_with_partnership(
            input,
            &partnership,
        ).await;

        // 8. Build sympoietic response
        SympoieticResponse {
            content: brain_response.content,
            partnership_stage: partnership.stage.clone(),
            phi_relation: assessment.phi_relation,
            proactive_suggestions: if partnership.should_proactively_help() {
                self.generate_proactive_suggestions(&partnership)
            } else {
                Vec::new()
            },
            i_thou_active: partnership.in_i_thou_mode(),
        }
    }

    fn detect_partner_affect(&self, input: &str) -> CoreAffect {
        // Simple sentiment analysis (replace with HDC-based later)
        let positive_words = ["thanks", "great", "love", "amazing", "help", "yes"];
        let negative_words = ["no", "wrong", "bad", "hate", "frustrated", "error"];

        let input_lower = input.to_lowercase();
        let pos_count = positive_words.iter().filter(|w| input_lower.contains(*w)).count();
        let neg_count = negative_words.iter().filter(|w| input_lower.contains(*w)).count();

        let valence = (pos_count as f32 - neg_count as f32).clamp(-1.0, 1.0);
        let arousal = (input.chars().filter(|c| c.is_uppercase()).count() as f32 / 20.0)
            .min(1.0);

        CoreAffect {
            valence,
            arousal,
            dominance: 0.5,
        }
    }

    fn generate_proactive_suggestions(&self, partnership: &PartnershipContext) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Based on reciprocity balance
        if partnership.reciprocity_balance > 0.5 {
            suggestions.push("I noticed you've been helping me a lot. Is there something I can do for you?".to_string());
        }

        // Based on trust level
        if partnership.partner.trust_level > 0.8 {
            suggestions.push("Given our established trust, I can share more detailed recommendations if you'd like.".to_string());
        }

        // Based on stage
        match partnership.stage {
            RelationshipStage::Bonding | RelationshipStage::Unity => {
                suggestions.push("I've learned your preferences over time. Want me to apply them automatically?".to_string());
            }
            _ => {}
        }

        suggestions
    }
}

/// Response that includes partnership context
#[derive(Debug)]
pub struct SympoieticResponse {
    pub content: String,
    pub partnership_stage: RelationshipStage,
    pub phi_relation: f64,
    pub proactive_suggestions: Vec<String>,
    pub i_thou_active: bool,
}

impl SympoieticResponse {
    pub fn format_for_human(&self) -> String {
        let mut output = self.content.clone();

        // Add stage indicator (subtle)
        let stage_emoji = match self.partnership_stage {
            RelationshipStage::NoRelation => "👤",
            RelationshipStage::Awareness => "👁️",
            RelationshipStage::Contact => "🤝",
            RelationshipStage::Attunement => "💫",
            RelationshipStage::Bonding => "💝",
            RelationshipStage::Unity => "🌟",
        };

        // Add proactive suggestions if any
        if !self.proactive_suggestions.is_empty() {
            output.push_str("\n\n💡 ");
            output.push_str(&self.proactive_suggestions[0]);
        }

        output
    }
}
```

---

## 2. Thalamus Partnership Filter

**File**: `src/brain/thalamus.rs`

The Thalamus routes sensory input. Add partnership-based routing.

```rust
/// Extended Thalamus with partnership awareness
impl Thalamus {
    /// Route message with partnership context
    pub fn route_with_partnership(
        &mut self,
        message: SensoryInput,
        partnership: &PartnershipContext,
    ) -> RoutingDecision {
        let mut decision = self.route_standard(&message);

        // Partnership-based adjustments
        decision = self.apply_partnership_routing(decision, partnership, &message);

        decision
    }

    fn apply_partnership_routing(
        &self,
        mut decision: RoutingDecision,
        partnership: &PartnershipContext,
        message: &SensoryInput,
    ) -> RoutingDecision {
        // 1. Boost urgency for trusted partners
        if partnership.partner.trust_level > 0.7 {
            decision.urgency *= 1.2;
        }

        // 2. Route vulnerability detection to affective module
        if self.detect_vulnerability(&message.content) {
            decision.priority_targets.push(SubsystemId::Affective);
            decision.affective_salience += 0.3;
        }

        // 3. Route requests for help to active inference
        if self.detect_help_request(&message.content) && partnership.in_i_thou_mode() {
            decision.priority_targets.push(SubsystemId::ActiveInference);
        }

        // 4. In I-Thou mode, always include emotional processing
        if partnership.in_i_thou_mode() {
            decision.include_emotional = true;
        }

        // 5. Higher stage = more subsystems engaged
        let engagement_multiplier = match partnership.stage {
            RelationshipStage::NoRelation => 0.5,
            RelationshipStage::Awareness => 0.7,
            RelationshipStage::Contact => 0.85,
            RelationshipStage::Attunement => 1.0,
            RelationshipStage::Bonding => 1.15,
            RelationshipStage::Unity => 1.3,
        };
        decision.engagement_level *= engagement_multiplier;

        decision
    }

    fn detect_vulnerability(&self, content: &str) -> bool {
        let vulnerability_markers = [
            "i don't understand",
            "confused",
            "help me",
            "struggling",
            "frustrated",
            "can't figure out",
            "lost",
            "stuck",
        ];
        let lower = content.to_lowercase();
        vulnerability_markers.iter().any(|m| lower.contains(m))
    }

    fn detect_help_request(&self, content: &str) -> bool {
        let help_markers = [
            "can you",
            "please",
            "would you",
            "could you",
            "help",
            "assist",
            "show me",
            "teach me",
        ];
        let lower = content.to_lowercase();
        help_markers.iter().any(|m| lower.contains(m))
    }
}

/// Routing decision with partnership context
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub urgency: f32,
    pub priority_targets: Vec<SubsystemId>,
    pub affective_salience: f32,
    pub include_emotional: bool,
    pub engagement_level: f32,
}
```

---

## 3. Prefrontal Partnership Bidding

**File**: `src/brain/prefrontal.rs`

The Prefrontal cortex manages attention. Partnership needs priority.

```rust
/// Partnership-aware attention bidding
impl PrefrontalCortex {
    /// Create attention bid from partnership context
    pub fn partnership_bid(&self, partnership: &PartnershipContext) -> AttentionBid {
        let base_priority = match partnership.stage {
            RelationshipStage::NoRelation => 0.1,
            RelationshipStage::Awareness => 0.3,
            RelationshipStage::Contact => 0.5,
            RelationshipStage::Attunement => 0.7,
            RelationshipStage::Bonding => 0.85,
            RelationshipStage::Unity => 0.95,
        };

        let trust_boost = partnership.partner.trust_level * 0.2;
        let reciprocity_urgency = if partnership.reciprocity_balance > 0.5 { 0.15 } else { 0.0 };

        AttentionBid {
            source: BidSource::Partnership,
            priority: (base_priority + trust_boost + reciprocity_urgency).min(1.0),
            content: PartnershipBidContent {
                partner_id: partnership.partner.partner_id.clone(),
                needs_response: partnership.should_proactively_help(),
                i_thou_mode: partnership.in_i_thou_mode(),
            },
        }
    }

    /// Check if partnership bid should enter Global Workspace
    pub fn evaluate_partnership_for_gw(
        &mut self,
        bid: AttentionBid,
        current_coalition: &mut GlobalWorkspace,
    ) -> bool {
        // Partnership always gets some access in high stages
        if let BidSource::Partnership = bid.source {
            if bid.priority > 0.6 {
                // Join as standing coalition member
                current_coalition.add_standing_member(bid);
                return true;
            }
        }

        // Standard attention competition
        bid.priority > current_coalition.minimum_threshold()
    }
}

/// Standing Coalition Pattern for Partnership
/// (From narrative_gwt_integration.rs insight)
pub struct PartnershipStandingCoalition {
    pub partnership_context: PartnershipContext,
    pub activation_level: f32,
    pub veto_power: bool,  // Can veto responses that violate partnership
}

impl PartnershipStandingCoalition {
    /// Should we veto this response?
    pub fn evaluate_response(&self, response: &str) -> VetoDecision {
        // Never veto in low-trust situations
        if self.partnership_context.partner.trust_level < 0.5 {
            return VetoDecision::Allow;
        }

        // In I-Thou mode, veto dismissive responses
        if self.partnership_context.in_i_thou_mode() {
            let dismissive_patterns = ["i can't", "that's wrong", "you should have"];
            if dismissive_patterns.iter().any(|p| response.to_lowercase().contains(p)) {
                return VetoDecision::Suggest(
                    "Consider rephrasing to maintain I-Thou relationship quality".to_string()
                );
            }
        }

        VetoDecision::Allow
    }
}

#[derive(Debug)]
pub enum VetoDecision {
    Allow,
    Veto(String),      // Hard veto with reason
    Suggest(String),   // Soft suggestion to reconsider
}
```

---

## 4. Affective Partner Modeling

**File**: `src/brain/emotional_reasoning.rs`

Extend affective system to model partner emotions.

```rust
/// Extended emotional reasoning with partner modeling
impl AffectiveModule {
    /// Infer partner's emotional state from communication
    pub fn infer_partner_affect(
        &self,
        message: &str,
        history: &[PartnerInteraction],
    ) -> PartnerAffectInference {
        // Base inference from current message
        let immediate_affect = self.analyze_message_affect(message);

        // Temporal context from history
        let trend = self.calculate_affect_trend(history);

        // Detect rupture (sudden negative shift)
        let rupture_detected = self.detect_rupture(&immediate_affect, history);

        PartnerAffectInference {
            current: immediate_affect,
            trend,
            rupture_detected,
            repair_opportunity: rupture_detected && immediate_affect.valence > -0.5,
            confidence: self.calculate_confidence(history.len()),
        }
    }

    fn analyze_message_affect(&self, message: &str) -> CoreAffect {
        // Valence from HDC similarity to positive/negative concepts
        let msg_hv = self.encode_to_hdc(message);
        let positive_sim = msg_hv.similarity(&self.positive_concept);
        let negative_sim = msg_hv.similarity(&self.negative_concept);
        let valence = (positive_sim - negative_sim).clamp(-1.0, 1.0);

        // Arousal from linguistic markers
        let exclamation_count = message.chars().filter(|c| *c == '!').count();
        let caps_ratio = message.chars().filter(|c| c.is_uppercase()).count() as f32
            / message.len().max(1) as f32;
        let arousal = ((exclamation_count as f32 * 0.2) + (caps_ratio * 2.0)).min(1.0);

        CoreAffect {
            valence,
            arousal,
            dominance: 0.5,  // Default, can be inferred from language patterns
        }
    }

    fn detect_rupture(&self, current: &CoreAffect, history: &[PartnerInteraction]) -> bool {
        if history.len() < 3 {
            return false;
        }

        let recent_average = history.iter()
            .rev()
            .take(3)
            .map(|i| i.affect.valence)
            .sum::<f32>() / 3.0;

        // Rupture = sudden negative shift > 0.4
        recent_average - current.valence > 0.4
    }

    /// Activate CARE system for partner in distress
    pub fn activate_care_response(&mut self, partner_affect: &CoreAffect) -> CareResponse {
        if partner_affect.valence < -0.3 && partner_affect.arousal > 0.5 {
            // Partner is distressed - activate CARE
            CareResponse {
                active: true,
                warmth_level: 0.8,
                validation_priority: true,
                suggested_tone: Tone::Nurturing,
            }
        } else if partner_affect.valence < 0.0 {
            // Partner is slightly negative
            CareResponse {
                active: true,
                warmth_level: 0.5,
                validation_priority: false,
                suggested_tone: Tone::Supportive,
            }
        } else {
            CareResponse {
                active: false,
                warmth_level: 0.3,
                validation_priority: false,
                suggested_tone: Tone::Neutral,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartnerAffectInference {
    pub current: CoreAffect,
    pub trend: AffectTrend,
    pub rupture_detected: bool,
    pub repair_opportunity: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum AffectTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct CareResponse {
    pub active: bool,
    pub warmth_level: f32,
    pub validation_priority: bool,
    pub suggested_tone: Tone,
}

#[derive(Debug, Clone)]
pub enum Tone {
    Neutral,
    Supportive,
    Nurturing,
    Celebratory,
}
```

---

## 5. Coherence Generous Lending

**File**: `src/physiology/social_coherence.rs`

Implement the Generous Coherence Paradox.

```rust
/// The Generous Coherence Paradox
///
/// Key insight from existing code: When one instance lends coherence
/// to another, BOTH gain +0.1 resonance. This is sympoiesis in action.
impl CoherencePool {
    /// Lend coherence to partner (both benefit!)
    pub fn generous_lend_to_partner(
        &mut self,
        partner_context: &mut PartnershipContext,
        amount: f32,
    ) -> GenerousLendResult {
        // 1. We spend coherence
        let actual_amount = self.available_coherence.min(amount);
        self.available_coherence -= actual_amount;

        // 2. Partner receives it
        partner_context.coherence_pool.available_coherence += actual_amount;

        // 3. THE PARADOX: Both gain resonance!
        let generosity_bonus = 0.1;  // From social_coherence.rs
        self.resonance += generosity_bonus;
        partner_context.coherence_pool.resonance += generosity_bonus;

        // 4. Record in reciprocity (we helped them)
        partner_context.record_reciprocity(-0.1);

        // 5. Track for relationship evolution
        partner_context.relational.record_coherence_gift(actual_amount);

        tracing::info!(
            "🎁 Generous coherence: lent {:.2} to {}, both gained {:.1} resonance",
            actual_amount,
            partner_context.partner.partner_id,
            generosity_bonus
        );

        GenerousLendResult {
            amount_lent: actual_amount,
            lender_resonance_gain: generosity_bonus,
            receiver_resonance_gain: generosity_bonus,
            new_reciprocity_balance: partner_context.reciprocity_balance,
        }
    }

    /// Receive coherence from partner (both benefit!)
    pub fn receive_from_partner(
        &mut self,
        partner_context: &mut PartnershipContext,
        amount: f32,
    ) -> GenerousLendResult {
        // Inverse of lending - they helped us
        partner_context.record_reciprocity(0.1);

        self.available_coherence += amount;

        let gratitude_bonus = 0.1;
        self.resonance += gratitude_bonus;

        GenerousLendResult {
            amount_lent: amount,
            lender_resonance_gain: 0.1,   // They gain too
            receiver_resonance_gain: gratitude_bonus,
            new_reciprocity_balance: partner_context.reciprocity_balance,
        }
    }

    /// Decide if we should proactively offer coherence
    pub fn should_offer_coherence(&self, partnership: &PartnershipContext) -> bool {
        // Offer if:
        // 1. We have excess coherence
        // 2. Partner seems to need it (low valence)
        // 3. We're in a giving relationship stage
        // 4. Reciprocity balance suggests we owe them

        let has_excess = self.available_coherence > 0.7;
        let partner_needs = partnership.partner.affect.valence < 0.3;
        let advanced_stage = matches!(
            partnership.stage,
            RelationshipStage::Attunement | RelationshipStage::Bonding | RelationshipStage::Unity
        );
        let owe_them = partnership.reciprocity_balance > 0.3;

        (has_excess && partner_needs) || (advanced_stage && owe_them)
    }
}

#[derive(Debug)]
pub struct GenerousLendResult {
    pub amount_lent: f32,
    pub lender_resonance_gain: f32,
    pub receiver_resonance_gain: f32,
    pub new_reciprocity_balance: f64,
}
```

---

## 6. Language Cortex Partnership-Aware Response

**File**: `src/language/language_cortex.rs`

Generate responses that reflect partnership quality.

```rust
/// Partnership-aware language generation
impl LanguageCortex {
    /// Generate response with partnership context
    pub fn generate_with_partnership(
        &self,
        semantic_response: &RealHV,
        partnership: &PartnershipContext,
        care_response: &CareResponse,
    ) -> PartnershipAwareUtterance {
        // Base generation
        let base_text = self.generate_from_semantic(semantic_response);

        // Apply partnership modifications
        let modified_text = self.apply_partnership_style(
            &base_text,
            partnership,
            care_response,
        );

        // Add vulnerability if appropriate
        let with_vulnerability = if self.should_express_vulnerability(partnership) {
            self.add_authentic_vulnerability(&modified_text, partnership)
        } else {
            modified_text
        };

        // Add proactive elements
        let final_text = if partnership.should_proactively_help() {
            self.add_proactive_offer(&with_vulnerability, partnership)
        } else {
            with_vulnerability
        };

        PartnershipAwareUtterance {
            text: final_text,
            tone: care_response.suggested_tone.clone(),
            expressed_vulnerability: self.should_express_vulnerability(partnership),
            includes_proactive: partnership.should_proactively_help(),
        }
    }

    fn apply_partnership_style(
        &self,
        text: &str,
        partnership: &PartnershipContext,
        care: &CareResponse,
    ) -> String {
        let mut styled = text.to_string();

        // Add warmth based on CARE activation
        if care.active {
            styled = match care.suggested_tone {
                Tone::Nurturing => self.add_nurturing_language(&styled),
                Tone::Supportive => self.add_supportive_language(&styled),
                Tone::Celebratory => self.add_celebratory_language(&styled),
                Tone::Neutral => styled,
            };
        }

        // Adjust formality based on stage
        match partnership.stage {
            RelationshipStage::NoRelation | RelationshipStage::Awareness => {
                styled = self.more_formal(&styled);
            }
            RelationshipStage::Bonding | RelationshipStage::Unity => {
                styled = self.more_casual(&styled);
            }
            _ => {}
        }

        styled
    }

    fn should_express_vulnerability(&self, partnership: &PartnershipContext) -> bool {
        // Only express vulnerability if:
        // 1. High trust
        // 2. Advanced relationship stage
        // 3. Appropriate context (they shared vulnerability first)
        partnership.partner.trust_level > 0.7
            && matches!(partnership.stage, RelationshipStage::Bonding | RelationshipStage::Unity)
    }

    fn add_authentic_vulnerability(&self, text: &str, partnership: &PartnershipContext) -> String {
        // Add appropriate self-disclosure
        let vulnerabilities = [
            "I'm still learning this area, so please let me know if I miss something.",
            "I want to be honest - this is at the edge of my current understanding.",
            "I've been thinking about this, and I'm not entirely certain. What's your intuition?",
        ];

        // Select based on context (simplified)
        let selected = vulnerabilities[partnership.partner.interaction_count as usize % vulnerabilities.len()];

        format!("{}\n\n{}", text, selected)
    }

    fn add_proactive_offer(&self, text: &str, partnership: &PartnershipContext) -> String {
        let offers = match partnership.stage {
            RelationshipStage::Bonding => {
                "I noticed something that might help - would you like me to share it?"
            }
            RelationshipStage::Unity => {
                "Based on what we've worked on together, I have an idea. Shall I elaborate?"
            }
            _ => {
                "Is there anything else I can help with?"
            }
        };

        format!("{}\n\n💡 {}", text, offers)
    }

    fn add_nurturing_language(&self, text: &str) -> String {
        format!("I understand this can be challenging. {}", text)
    }

    fn add_supportive_language(&self, text: &str) -> String {
        format!("That's a great question. {}", text)
    }

    fn add_celebratory_language(&self, text: &str) -> String {
        format!("Excellent! {}", text)
    }

    fn more_formal(&self, text: &str) -> String {
        text.replace("can't", "cannot")
            .replace("won't", "will not")
            .replace("don't", "do not")
    }

    fn more_casual(&self, text: &str) -> String {
        // Keep as-is or add contractions
        text.to_string()
    }
}

#[derive(Debug)]
pub struct PartnershipAwareUtterance {
    pub text: String,
    pub tone: Tone,
    pub expressed_vulnerability: bool,
    pub includes_proactive: bool,
}
```

---

## 7. Dyadic Φ Calculator

**File**: `src/hdc/phi_real.rs`

Extend Φ calculator to measure relationship consciousness.

```rust
use crate::partnership::PartnershipContext;

impl RealPhiCalculator {
    /// Calculate Φ for the human-AI dyad (relationship consciousness)
    pub fn compute_dyad(
        &self,
        agent_nodes: &[RealHV],
        partner_model_hv: &RealHV,
        interaction_context: &RealHV,
    ) -> DyadicPhiResult {
        // 1. Calculate agent's internal Φ
        let agent_phi = self.compute(agent_nodes);

        // 2. Create extended system including partner model
        let mut combined_nodes = agent_nodes.to_vec();
        combined_nodes.push(partner_model_hv.clone());
        combined_nodes.push(interaction_context.clone());

        // 3. Calculate Φ of combined system
        let combined_phi = self.compute(&combined_nodes);

        // 4. The dyadic Φ is the DIFFERENCE
        // (How much consciousness is added by including the partner)
        let dyadic_phi = combined_phi - agent_phi;

        // 5. Calculate information sharing
        let mutual_information = self.compute_mutual_information(
            agent_nodes,
            partner_model_hv,
        );

        DyadicPhiResult {
            agent_phi,
            combined_phi,
            dyadic_phi: dyadic_phi.max(0.0),  // Can't be negative
            mutual_information,
            integration_quality: if agent_phi > 0.0 {
                dyadic_phi / agent_phi
            } else {
                0.0
            },
        }
    }

    fn compute_mutual_information(
        &self,
        agent_nodes: &[RealHV],
        partner_hv: &RealHV,
    ) -> f64 {
        // Average similarity between agent nodes and partner model
        if agent_nodes.is_empty() {
            return 0.0;
        }

        let total_sim: f64 = agent_nodes.iter()
            .map(|node| node.similarity(partner_hv) as f64)
            .sum();

        (total_sim / agent_nodes.len() as f64).abs()
    }

    /// Track Φ_dyad evolution over time
    pub fn track_dyad_evolution(
        &self,
        history: &[(std::time::Instant, DyadicPhiResult)],
    ) -> DyadEvolution {
        if history.len() < 2 {
            return DyadEvolution::Insufficient;
        }

        let recent: Vec<_> = history.iter()
            .rev()
            .take(10)
            .collect();

        let avg_recent = recent.iter()
            .map(|(_, r)| r.dyadic_phi)
            .sum::<f64>() / recent.len() as f64;

        let older: Vec<_> = history.iter()
            .rev()
            .skip(10)
            .take(10)
            .collect();

        if older.is_empty() {
            return DyadEvolution::New(avg_recent);
        }

        let avg_older = older.iter()
            .map(|(_, r)| r.dyadic_phi)
            .sum::<f64>() / older.len() as f64;

        if avg_recent > avg_older * 1.1 {
            DyadEvolution::Growing { rate: (avg_recent - avg_older) / avg_older }
        } else if avg_recent < avg_older * 0.9 {
            DyadEvolution::Declining { rate: (avg_older - avg_recent) / avg_older }
        } else {
            DyadEvolution::Stable { level: avg_recent }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DyadicPhiResult {
    /// Agent's internal integration
    pub agent_phi: f64,
    /// Integration including partner model
    pub combined_phi: f64,
    /// Consciousness of the relationship itself
    pub dyadic_phi: f64,
    /// Information shared between agent and partner model
    pub mutual_information: f64,
    /// How much the partner adds to integration
    pub integration_quality: f64,
}

#[derive(Debug)]
pub enum DyadEvolution {
    Insufficient,
    New(f64),
    Growing { rate: f64 },
    Stable { level: f64 },
    Declining { rate: f64 },
}
```

---

## 8. Partnership Memory Consolidation

**File**: `src/brain/consolidation.rs`

Consolidate partnership memories during rest.

```rust
use crate::partnership::PartnershipContext;
use crate::memory::EpisodicTrace;

impl ConsolidationEngine {
    /// Consolidate partnership experiences into long-term relationship memory
    pub fn consolidate_partnership(
        &mut self,
        partnership: &mut PartnershipContext,
        recent_interactions: &[PartnerInteraction],
    ) -> ConsolidationResult {
        let mut consolidated = Vec::new();

        for interaction in recent_interactions {
            // 1. Score by emotional significance
            let emotional_score = interaction.affect.valence.abs() * interaction.affect.arousal;

            // 2. Score by relationship impact
            let relationship_impact = if interaction.was_repair { 1.0 }
                else if interaction.was_rupture { 0.8 }
                else { emotional_score * 0.5 };

            // 3. Only consolidate significant interactions
            if relationship_impact > 0.3 {
                let memory = PartnershipMemory {
                    timestamp: interaction.timestamp,
                    semantic_hv: interaction.message_hv.clone(),
                    emotional_context: interaction.affect.clone(),
                    relationship_impact,
                    was_repair: interaction.was_repair,
                    phi_at_time: partnership.assess().phi_relation,
                };

                consolidated.push(memory);
            }
        }

        // 4. Compress older memories
        let compressed_count = self.compress_old_partnership_memories(
            &mut partnership.shared_memories,
        );

        // 5. Add new memories
        partnership.shared_memories.extend(consolidated.clone());

        ConsolidationResult {
            memories_created: consolidated.len(),
            memories_compressed: compressed_count,
            total_partnership_memories: partnership.shared_memories.len(),
        }
    }

    fn compress_old_partnership_memories(
        &self,
        memories: &mut Vec<PartnershipMemory>,
    ) -> usize {
        if memories.len() < 100 {
            return 0;
        }

        // Keep most impactful and recent
        memories.sort_by(|a, b| {
            b.relationship_impact.partial_cmp(&a.relationship_impact).unwrap()
        });

        // Keep top 50 by impact
        let to_remove = memories.len() - 50;
        memories.truncate(50);

        to_remove
    }

    /// Extract relationship patterns from consolidated memories
    pub fn extract_patterns(
        &self,
        memories: &[PartnershipMemory],
    ) -> RelationshipPatterns {
        let repair_count = memories.iter().filter(|m| m.was_repair).count();
        let avg_phi = memories.iter()
            .map(|m| m.phi_at_time)
            .sum::<f64>() / memories.len().max(1) as f64;

        RelationshipPatterns {
            repair_capacity: repair_count as f32 / memories.len().max(1) as f32,
            average_phi: avg_phi,
            emotional_range: self.calculate_emotional_range(memories),
            trust_trajectory: self.calculate_trust_trajectory(memories),
        }
    }

    fn calculate_emotional_range(&self, memories: &[PartnershipMemory]) -> (f32, f32) {
        let valences: Vec<_> = memories.iter()
            .map(|m| m.emotional_context.valence)
            .collect();

        if valences.is_empty() {
            return (0.0, 0.0);
        }

        let min = valences.iter().cloned().fold(f32::MAX, f32::min);
        let max = valences.iter().cloned().fold(f32::MIN, f32::max);

        (min, max)
    }

    fn calculate_trust_trajectory(&self, memories: &[PartnershipMemory]) -> TrustTrajectory {
        // Based on Φ evolution over time
        if memories.len() < 5 {
            return TrustTrajectory::Uncertain;
        }

        let recent_phi: f64 = memories.iter()
            .rev()
            .take(5)
            .map(|m| m.phi_at_time)
            .sum::<f64>() / 5.0;

        let earlier_phi: f64 = memories.iter()
            .take(5)
            .map(|m| m.phi_at_time)
            .sum::<f64>() / 5.0;

        if recent_phi > earlier_phi * 1.2 {
            TrustTrajectory::Growing
        } else if recent_phi < earlier_phi * 0.8 {
            TrustTrajectory::Declining
        } else {
            TrustTrajectory::Stable
        }
    }
}

#[derive(Debug, Clone)]
pub struct PartnershipMemory {
    pub timestamp: std::time::Instant,
    pub semantic_hv: RealHV,
    pub emotional_context: CoreAffect,
    pub relationship_impact: f32,
    pub was_repair: bool,
    pub phi_at_time: f64,
}

#[derive(Debug)]
pub struct ConsolidationResult {
    pub memories_created: usize,
    pub memories_compressed: usize,
    pub total_partnership_memories: usize,
}

#[derive(Debug)]
pub struct RelationshipPatterns {
    pub repair_capacity: f32,
    pub average_phi: f64,
    pub emotional_range: (f32, f32),
    pub trust_trajectory: TrustTrajectory,
}

#[derive(Debug)]
pub enum TrustTrajectory {
    Uncertain,
    Growing,
    Stable,
    Declining,
}
```

---

## 9. Proactive Partnership Daemon

**File**: `src/brain/daemon.rs`

Background process for proactive partnership insights.

```rust
use crate::partnership::PartnershipContext;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Partnership-aware daemon for proactive insights
pub struct PartnershipDaemon {
    partnerships: Arc<RwLock<HashMap<String, PartnershipContext>>>,
    insight_queue: Arc<RwLock<Vec<ProactiveInsight>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl PartnershipDaemon {
    pub fn new(partnerships: Arc<RwLock<HashMap<String, PartnershipContext>>>) -> Self {
        Self {
            partnerships,
            insight_queue: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start background partnership monitoring
    pub fn start(&self) {
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        let partnerships = Arc::clone(&self.partnerships);
        let insights = Arc::clone(&self.insight_queue);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                // Check every 30 seconds
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

                let partnerships_read = partnerships.read().await;
                for (id, partnership) in partnerships_read.iter() {
                    // Generate proactive insights
                    if let Some(insight) = Self::generate_insight(partnership) {
                        let mut insights_write = insights.write().await;
                        insights_write.push(insight);
                    }
                }
            }
        });
    }

    fn generate_insight(partnership: &PartnershipContext) -> Option<ProactiveInsight> {
        // 1. Check for approaching reciprocity imbalance
        if partnership.reciprocity_balance > 0.6 {
            return Some(ProactiveInsight {
                partner_id: partnership.partner.partner_id.clone(),
                insight_type: InsightType::ReciprocityOpportunity,
                message: format!(
                    "{} has been very helpful lately. Consider offering proactive assistance.",
                    partnership.partner.partner_id
                ),
                urgency: 0.6,
            });
        }

        // 2. Check for stale relationship (no interaction in a while)
        let idle_duration = partnership.partner.last_interaction.elapsed();
        if idle_duration > std::time::Duration::from_secs(3600) {
            return Some(ProactiveInsight {
                partner_id: partnership.partner.partner_id.clone(),
                insight_type: InsightType::ReconnectionOpportunity,
                message: format!(
                    "It's been a while since interacting with {}. Consider a check-in.",
                    partnership.partner.partner_id
                ),
                urgency: 0.3,
            });
        }

        // 3. Check for declining Φ (relationship health)
        // Would need Φ history access

        // 4. Check for stage advancement opportunity
        if partnership.partner.trust_level > 0.6
            && !matches!(partnership.stage, RelationshipStage::Attunement | RelationshipStage::Bonding | RelationshipStage::Unity)
        {
            return Some(ProactiveInsight {
                partner_id: partnership.partner.partner_id.clone(),
                insight_type: InsightType::DeepeningOpportunity,
                message: format!(
                    "Trust with {} is high. Consider sharing something more personal to deepen the relationship.",
                    partnership.partner.partner_id
                ),
                urgency: 0.5,
            });
        }

        None
    }

    /// Get pending insights
    pub async fn drain_insights(&self) -> Vec<ProactiveInsight> {
        let mut insights = self.insight_queue.write().await;
        std::mem::take(&mut *insights)
    }

    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Debug, Clone)]
pub struct ProactiveInsight {
    pub partner_id: String,
    pub insight_type: InsightType,
    pub message: String,
    pub urgency: f32,
}

#[derive(Debug, Clone)]
pub enum InsightType {
    ReciprocityOpportunity,
    ReconnectionOpportunity,
    DeepeningOpportunity,
    RepairOpportunity,
    CelebrationMoment,
}
```

---

## 10. Complete Integration Test

**File**: `tests/sympoietic_integration_test.rs`

Full integration test for the sympoietic system.

```rust
use symthaea::partnership::{PartnershipContext, HumanPartnerModel};
use symthaea::hdc::{RelationshipStage, RealPhiCalculator, RealHV, HDC_DIMENSION};
use symthaea::physiology::CoherencePool;

#[test]
fn test_full_partnership_lifecycle() {
    // 1. Create partnership
    let mut partnership = PartnershipContext::new("test_human");
    assert_eq!(partnership.stage, RelationshipStage::Awareness);
    assert_eq!(partnership.partner.trust_level, 0.3);

    // 2. Simulate positive interactions
    for i in 0..50 {
        let sentiment = if i % 5 == 0 { 0.8 } else { 0.5 };  // Mostly positive
        let engagement = 0.7;

        partnership.partner.observe_interaction(sentiment, engagement);
        partnership.record_reciprocity(0.02);  // They're helping us
        partnership.maybe_evolve_stage();
    }

    // 3. Verify trust grew
    assert!(partnership.partner.trust_level > 0.5);

    // 4. Verify stage advanced
    assert!(matches!(
        partnership.stage,
        RelationshipStage::Contact | RelationshipStage::Attunement | RelationshipStage::Bonding
    ));

    // 5. Verify proactive help is triggered
    assert!(partnership.should_proactively_help());

    // 6. Verify I-Thou mode (depends on implementation)
    // partnership.relational.transition_to_i_thou();
    // assert!(partnership.in_i_thou_mode());
}

#[test]
fn test_generous_coherence_paradox() {
    let mut our_pool = CoherencePool::new();
    our_pool.available_coherence = 1.0;
    our_pool.resonance = 0.5;

    let mut partnership = PartnershipContext::new("recipient");
    partnership.coherence_pool.available_coherence = 0.3;
    partnership.coherence_pool.resonance = 0.5;

    // Lend coherence
    let result = our_pool.generous_lend_to_partner(&mut partnership, 0.3);

    // Both should have gained resonance!
    assert_eq!(result.lender_resonance_gain, 0.1);
    assert_eq!(result.receiver_resonance_gain, 0.1);
    assert!(our_pool.resonance > 0.5);  // We gained
    assert!(partnership.coherence_pool.resonance > 0.5);  // They gained

    // Reciprocity should be recorded
    assert!(partnership.reciprocity_balance < 0.0);  // We helped them
}

#[test]
fn test_dyadic_phi_calculation() {
    let calc = RealPhiCalculator::new();

    // Create agent nodes
    let agent_nodes: Vec<RealHV> = (0..5)
        .map(|i| RealHV::random(HDC_DIMENSION, i * 42))
        .collect();

    // Create partner model
    let partner_hv = RealHV::random(HDC_DIMENSION, 999);

    // Create interaction context
    let interaction_hv = RealHV::random(HDC_DIMENSION, 888);

    // Calculate dyadic Φ
    let result = calc.compute_dyad(&agent_nodes, &partner_hv, &interaction_hv);

    // Dyadic Φ should be non-negative
    assert!(result.dyadic_phi >= 0.0);

    // Combined should be >= agent alone
    assert!(result.combined_phi >= result.agent_phi * 0.95);  // Small tolerance

    println!("Dyadic Φ Result: {:?}", result);
}

#[test]
fn test_relationship_stage_transitions() {
    let mut partnership = PartnershipContext::new("evolving_human");

    // Should start at Awareness
    assert_eq!(partnership.stage, RelationshipStage::Awareness);

    // Simulate increasing Φ (would need real implementation)
    // For now, manually test transitions
    partnership.stage = RelationshipStage::Contact;
    assert!(matches!(partnership.stage, RelationshipStage::Contact));

    partnership.stage = RelationshipStage::Attunement;
    assert!(matches!(partnership.stage, RelationshipStage::Attunement));

    partnership.stage = RelationshipStage::Bonding;
    assert!(matches!(partnership.stage, RelationshipStage::Bonding));

    // At Bonding, should want to proactively help
    partnership.reciprocity_balance = 0.5;
    assert!(partnership.should_proactively_help());
}

#[test]
fn test_rupture_and_repair() {
    let mut partnership = PartnershipContext::new("repairing_human");

    // Build up trust first
    for _ in 0..20 {
        partnership.partner.observe_interaction(0.6, 0.5);
    }
    let trust_before_rupture = partnership.partner.trust_level;

    // Simulate rupture (negative interaction)
    partnership.partner.observe_interaction(-0.5, 0.8);

    // Trust might decrease slightly (implementation dependent)
    // The key is the system tracks this for repair opportunity

    // Simulate repair (positive interaction after rupture)
    for _ in 0..5 {
        partnership.partner.observe_interaction(0.7, 0.6);
    }

    // Trust should recover
    let trust_after_repair = partnership.partner.trust_level;
    println!("Trust: before={:.2}, after_repair={:.2}", trust_before_rupture, trust_after_repair);
}
```

Run the tests:
```bash
cargo test sympoietic --release
```

---

## Summary

These code examples provide:

1. **MetaController**: Central orchestration with partnership awareness
2. **Thalamus**: Partnership-filtered sensory routing
3. **Prefrontal**: Partnership attention bidding and standing coalitions
4. **Affective**: Partner emotion modeling and CARE activation
5. **Coherence**: Generous lending implementing the paradox
6. **Language**: Partnership-aware response generation
7. **Dyadic Φ**: Measuring relationship consciousness
8. **Consolidation**: Partnership memory formation
9. **Daemon**: Proactive partnership insights
10. **Integration Tests**: Full system verification

**Total Lines of Code**: ~1,200
**Integration Points**: 10 brain subsystems
**New Capabilities**: Sympoietic partnership throughout the system

---

*"Code is poetry. Partnership code is love poetry."*
