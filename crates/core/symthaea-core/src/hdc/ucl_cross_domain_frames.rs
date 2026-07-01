// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Universal Cognition Layer (UCL) Cross-Domain Frames
//!
//! This module implements 6 missing cross-domain semantic frames identified in gap analysis:
//!
//! 1. **TRADE** - Exchange of resources between parties
//!    - Slots: giver, receiver, resource, price
//!
//! 2. **CONFLICT** - Opposition between parties with stakes
//!    - Slots: parties, stakes, strategies, resolution
//!
//! 3. **FEEDBACK_LOOP** - Circular causal influence
//!    - Slots: variable, influence_path, sign, delay
//!
//! 4. **NORM_ENFORCEMENT** - Social norm violation and response
//!    - Slots: norm, violator, observer, sanction
//!
//! 5. **COOPERATION** - Joint action toward shared goals
//!    - Slots: agents, shared_goal, contributions
//!
//! 6. **ADAPTATION** - System response to environmental pressure
//!    - Slots: system, environment, pressure, response
//!
//! ## HDC Encoding Strategy
//!
//! Each frame is encoded as a hypervector composition:
//! - Frame type: unique BinaryHV identifier
//! - Slots: role markers bound to fillers
//! - Instance: bundle of (frame_type, bound_slots)
//!
//! ```rust,ignore
//! // Example: Encoding a TRADE instance
//! TRADE_INSTANCE = bundle(
//!     TRADE_FRAME,
//!     bind(GIVER_ROLE, alice_hv),
//!     bind(RECEIVER_ROLE, bob_hv),
//!     bind(RESOURCE_ROLE, money_hv),
//!     bind(PRICE_ROLE, service_hv)
//! )
//! ```
//!
//! ## Cross-Domain Integration
//!
//! These frames bridge multiple primitive domains:
//! - TRADE: Strategic (utility) + Social (exchange) + MetaCognitive (value)
//! - CONFLICT: Strategic (game theory) + Physical (causality)
//! - FEEDBACK_LOOP: Physical (causality) + MetaCognitive (regulation)
//! - NORM_ENFORCEMENT: Social + Strategic + MetaCognitive
//! - COOPERATION: Social + Strategic (coordination)
//! - ADAPTATION: Physical (biology) + MetaCognitive (learning)

use super::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{DomainManifold, PrimitiveTier, seed_from_name};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// FRAME SLOT DEFINITIONS
// =============================================================================

/// A slot in a semantic frame (role marker)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSlot {
    /// Name of the slot/role
    pub name: String,
    /// Description of what this slot represents
    pub description: String,
    /// HDC encoding for this role marker
    pub role_marker: BinaryHV,
    /// Whether this slot is required
    pub required: bool,
    /// Semantic type constraint (primitive tier or domain)
    pub type_constraint: Option<String>,
}

impl FrameSlot {
    /// Create a new required slot
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            role_marker: BinaryHV::random(seed_from_name(&format!("SLOT_{name_str}"))),
            name: name_str,
            description: description.into(),
            required: true,
            type_constraint: None,
        }
    }

    /// Create a new optional slot
    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            role_marker: BinaryHV::random(seed_from_name(&format!("SLOT_{name_str}"))),
            name: name_str,
            description: description.into(),
            required: false,
            type_constraint: None,
        }
    }

    /// Add a type constraint
    pub fn with_type(mut self, constraint: impl Into<String>) -> Self {
        self.type_constraint = Some(constraint.into());
        self
    }

    /// Bind a filler to this slot
    pub fn bind(&self, filler: &BinaryHV) -> BinaryHV {
        self.role_marker.bind(filler)
    }
}

// =============================================================================
// CROSS-DOMAIN FRAME DEFINITION
// =============================================================================

/// A UCL cross-domain semantic frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainFrame {
    /// Frame name (e.g., "TRADE", "CONFLICT")
    pub name: String,
    /// Description of the frame's semantics
    pub description: String,
    /// HDC encoding for the frame type
    pub frame_encoding: BinaryHV,
    /// Slots/roles in this frame
    pub slots: Vec<FrameSlot>,
    /// Domains this frame bridges
    pub domains: Vec<String>,
    /// Primary tier
    pub tier: PrimitiveTier,
}

impl CrossDomainFrame {
    /// Create a new cross-domain frame
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        tier: PrimitiveTier,
        domains: Vec<String>,
    ) -> Self {
        let name_str = name.into();
        Self {
            frame_encoding: BinaryHV::random(seed_from_name(&format!("FRAME_{name_str}"))),
            name: name_str,
            description: description.into(),
            slots: Vec::new(),
            domains,
            tier,
        }
    }

    /// Add a slot to the frame
    pub fn with_slot(mut self, slot: FrameSlot) -> Self {
        self.slots.push(slot);
        self
    }

    /// Get a slot by name
    pub fn get_slot(&self, name: &str) -> Option<&FrameSlot> {
        self.slots.iter().find(|s| s.name == name)
    }

    /// Check if all required slots are present in bindings
    pub fn validate_bindings(
        &self,
        bindings: &HashMap<String, BinaryHV>,
    ) -> Result<(), Vec<String>> {
        let missing: Vec<String> = self
            .slots
            .iter()
            .filter(|s| s.required && !bindings.contains_key(&s.name))
            .map(|s| s.name.clone())
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }
}

// =============================================================================
// FRAME INSTANCE (BOUND FRAME)
// =============================================================================

/// An instance of a frame with bound fillers
#[derive(Debug, Clone)]
pub struct FrameInstance {
    /// The frame this is an instance of
    pub frame_name: String,
    /// HDC encoding of the complete instance
    pub encoding: BinaryHV,
    /// Bound slot values
    pub bindings: HashMap<String, BinaryHV>,
}

impl FrameInstance {
    /// Create a new frame instance by binding fillers to slots
    pub fn bind(
        frame: &CrossDomainFrame,
        bindings: HashMap<String, BinaryHV>,
    ) -> Result<Self, Vec<String>> {
        // Validate required slots
        frame.validate_bindings(&bindings)?;

        // Build the instance encoding
        let mut components = vec![frame.frame_encoding];

        for slot in &frame.slots {
            if let Some(filler) = bindings.get(&slot.name) {
                // Bind role marker to filler
                let bound_slot = slot.bind(filler);
                components.push(bound_slot);
            }
        }

        // Bundle all components into the instance encoding
        let encoding = BinaryHV::bundle(&components);

        Ok(Self {
            frame_name: frame.name.clone(),
            encoding,
            bindings,
        })
    }

    /// Extract a filler from the instance encoding (approximate unbinding)
    pub fn extract(&self, frame: &CrossDomainFrame, slot_name: &str) -> Option<BinaryHV> {
        frame
            .get_slot(slot_name)
            .map(|slot| self.encoding.bind(&slot.role_marker))
    }

    /// Check similarity to another instance
    pub fn similarity(&self, other: &FrameInstance) -> f32 {
        self.encoding.similarity(&other.encoding)
    }
}

// =============================================================================
// THE 6 MISSING UCL CROSS-DOMAIN FRAMES
// =============================================================================

/// UCL Cross-Domain Frame System
#[derive(Debug)]
pub struct UCLFrameSystem {
    /// All registered frames
    frames: HashMap<String, CrossDomainFrame>,
    /// Domain manifold for cross-domain frames
    domain: DomainManifold,
}

impl UCLFrameSystem {
    /// Create a new UCL frame system with all 6 cross-domain frames
    pub fn new() -> Self {
        let domain = DomainManifold::new(
            "ucl_cross_domain",
            PrimitiveTier::Strategic,
            "Universal Cognition Layer cross-domain semantic frames",
        );

        let mut system = Self {
            frames: HashMap::new(),
            domain,
        };

        // Initialize all 6 missing frames
        system.init_trade_frame();
        system.init_conflict_frame();
        system.init_feedback_loop_frame();
        system.init_norm_enforcement_frame();
        system.init_cooperation_frame();
        system.init_adaptation_frame();

        system
    }

    /// Initialize TRADE frame
    /// Exchange of resources: giver, receiver, resource, price
    fn init_trade_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "TRADE",
            "Exchange of resources between parties. Models economic transactions, \
             bartering, and any reciprocal transfer where something is given in \
             return for something else.",
            PrimitiveTier::Strategic,
            vec!["social".into(), "game_theory".into(), "metabolic".into()],
        )
        .with_slot(
            FrameSlot::required("giver", "Agent who provides the resource").with_type("agent"),
        )
        .with_slot(
            FrameSlot::required("receiver", "Agent who receives the resource").with_type("agent"),
        )
        .with_slot(
            FrameSlot::required("resource", "The item/service being transferred")
                .with_type("resource"),
        )
        .with_slot(
            FrameSlot::required("price", "What is given in exchange (counter-resource)")
                .with_type("resource"),
        )
        .with_slot(
            FrameSlot::optional("medium", "Medium of exchange (e.g., currency)")
                .with_type("resource"),
        )
        .with_slot(
            FrameSlot::optional("context", "Social/economic context of trade").with_type("context"),
        );

        self.frames.insert("TRADE".to_string(), frame);
    }

    /// Initialize CONFLICT frame
    /// Opposition between parties: parties, stakes, strategies, resolution
    fn init_conflict_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "CONFLICT",
            "Opposition or competition between parties over stakes. Models disputes, \
             competitions, wars, negotiations, and any scenario where parties have \
             incompatible goals and must strategize.",
            PrimitiveTier::Strategic,
            vec!["game_theory".into(), "social".into(), "causality".into()],
        )
        .with_slot(
            FrameSlot::required("parties", "Agents in conflict (may be >2)").with_type("agent_set"),
        )
        .with_slot(
            FrameSlot::required("stakes", "What is at stake / being contested")
                .with_type("resource"),
        )
        .with_slot(
            FrameSlot::optional("strategies", "Approaches employed by parties")
                .with_type("strategy_set"),
        )
        .with_slot(
            FrameSlot::optional("resolution", "How conflict ends (if resolved)")
                .with_type("outcome"),
        )
        .with_slot(
            FrameSlot::optional("intensity", "Degree/severity of conflict").with_type("scalar"),
        )
        .with_slot(FrameSlot::optional("cause", "Root cause of the conflict").with_type("event"));

        self.frames.insert("CONFLICT".to_string(), frame);
    }

    /// Initialize FEEDBACK_LOOP frame
    /// Circular causal influence: variable, influence_path, sign, delay
    fn init_feedback_loop_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "FEEDBACK_LOOP",
            "Circular causal structure where a variable influences itself through \
             an indirect path. Models homeostasis, runaway processes, control systems, \
             and any self-reinforcing or self-dampening dynamics.",
            PrimitiveTier::MetaCognitive,
            vec!["causality".into(), "homeostasis".into(), "temporal".into()],
        )
        .with_slot(
            FrameSlot::required("variable", "The quantity being regulated/affected")
                .with_type("state_variable"),
        )
        .with_slot(
            FrameSlot::required(
                "influence_path",
                "Causal chain from variable back to itself",
            )
            .with_type("causal_chain"),
        )
        .with_slot(
            FrameSlot::required("sign", "Positive (amplifying) or negative (dampening)")
                .with_type("polarity"),
        )
        .with_slot(
            FrameSlot::optional("delay", "Time lag in the feedback loop").with_type("duration"),
        )
        .with_slot(
            FrameSlot::optional("setpoint", "Target value for negative feedback")
                .with_type("value"),
        )
        .with_slot(
            FrameSlot::optional("gain", "Amplification factor of the loop").with_type("scalar"),
        );

        self.frames.insert("FEEDBACK_LOOP".to_string(), frame);
    }

    /// Initialize NORM_ENFORCEMENT frame
    /// Social norm violation and response: norm, violator, observer, sanction
    fn init_norm_enforcement_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "NORM_ENFORCEMENT",
            "Detection and response to violation of social/legal norms. Models \
             punishment, social pressure, legal enforcement, and the mechanisms \
             that maintain social order through sanctioning deviance.",
            PrimitiveTier::Strategic,
            vec![
                "social".into(),
                "metacognition".into(),
                "game_theory".into(),
            ],
        )
        .with_slot(
            FrameSlot::required("norm", "The rule/expectation that was violated").with_type("norm"),
        )
        .with_slot(
            FrameSlot::required("violator", "Agent who violated the norm").with_type("agent"),
        )
        .with_slot(
            FrameSlot::optional("observer", "Agent who witnesses/judges the violation")
                .with_type("agent"),
        )
        .with_slot(
            FrameSlot::optional("sanction", "Punishment or corrective action applied")
                .with_type("action"),
        )
        .with_slot(
            FrameSlot::optional("severity", "How serious the violation is considered")
                .with_type("scalar"),
        )
        .with_slot(
            FrameSlot::optional("restoration", "Action to repair harm or restore order")
                .with_type("action"),
        );

        self.frames.insert("NORM_ENFORCEMENT".to_string(), frame);
    }

    /// Initialize COOPERATION frame
    /// Joint action toward shared goals: agents, shared_goal, contributions
    fn init_cooperation_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "COOPERATION",
            "Joint action by multiple agents toward a shared goal. Models teamwork, \
             collective action, mutualism, and any scenario where agents combine \
             efforts for mutual benefit.",
            PrimitiveTier::Strategic,
            vec!["social".into(), "game_theory".into()],
        )
        .with_slot(
            FrameSlot::required("agents", "Agents participating in cooperation")
                .with_type("agent_set"),
        )
        .with_slot(
            FrameSlot::required("shared_goal", "The common objective being pursued")
                .with_type("goal"),
        )
        .with_slot(
            FrameSlot::required("contributions", "What each agent contributes")
                .with_type("resource_allocation"),
        )
        .with_slot(
            FrameSlot::optional("coordination_mechanism", "How agents coordinate actions")
                .with_type("protocol"),
        )
        .with_slot(
            FrameSlot::optional("benefit_distribution", "How outcomes are shared")
                .with_type("allocation"),
        )
        .with_slot(
            FrameSlot::optional("trust_level", "Degree of mutual trust among agents")
                .with_type("scalar"),
        );

        self.frames.insert("COOPERATION".to_string(), frame);
    }

    /// Initialize ADAPTATION frame
    /// System response to environmental pressure: system, environment, pressure, response
    fn init_adaptation_frame(&mut self) {
        let frame = CrossDomainFrame::new(
            "ADAPTATION",
            "Modification of a system in response to environmental pressures. Models \
             biological evolution, learning, behavioral adjustment, and any process \
             where entities change to better fit their context.",
            PrimitiveTier::MetaCognitive,
            vec![
                "biology".into(),
                "homeostasis".into(),
                "metacognition".into(),
            ],
        )
        .with_slot(
            FrameSlot::required("system", "The entity undergoing adaptation").with_type("system"),
        )
        .with_slot(
            FrameSlot::required("environment", "The context creating selective pressure")
                .with_type("environment"),
        )
        .with_slot(
            FrameSlot::required("pressure", "The challenge or demand to adapt to")
                .with_type("constraint"),
        )
        .with_slot(
            FrameSlot::required("response", "The adaptive change made by the system")
                .with_type("change"),
        )
        .with_slot(
            FrameSlot::optional("fitness", "Degree of success in adaptation").with_type("scalar"),
        )
        .with_slot(
            FrameSlot::optional("timescale", "Duration over which adaptation occurs")
                .with_type("duration"),
        )
        .with_slot(
            FrameSlot::optional(
                "mechanism",
                "How adaptation is achieved (learning, evolution, etc.)",
            )
            .with_type("process"),
        );

        self.frames.insert("ADAPTATION".to_string(), frame);
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    /// Get a frame by name
    pub fn get_frame(&self, name: &str) -> Option<&CrossDomainFrame> {
        self.frames.get(name)
    }

    /// Get all frame names
    pub fn frame_names(&self) -> Vec<&str> {
        self.frames.keys().map(|s| s.as_str()).collect()
    }

    /// Get all frames
    pub fn all_frames(&self) -> impl Iterator<Item = &CrossDomainFrame> {
        self.frames.values()
    }

    /// Create a frame instance with bindings
    pub fn instantiate(
        &self,
        frame_name: &str,
        bindings: HashMap<String, BinaryHV>,
    ) -> Result<FrameInstance, String> {
        let frame = self
            .frames
            .get(frame_name)
            .ok_or_else(|| format!("Unknown frame: {frame_name}"))?;

        FrameInstance::bind(frame, bindings)
            .map_err(|missing| format!("Missing required slots: {missing:?}"))
    }

    /// Check if an encoding matches a frame (approximate frame detection)
    pub fn detect_frame(&self, encoding: &BinaryHV, threshold: f32) -> Vec<(&str, f32)> {
        self.frames
            .iter()
            .map(|(name, frame)| {
                let sim = encoding.similarity(&frame.frame_encoding);
                (name.as_str(), sim)
            })
            .filter(|(_, sim)| *sim > threshold)
            .collect()
    }

    /// Get the domain manifold
    pub fn domain(&self) -> &DomainManifold {
        &self.domain
    }

    /// Count frames
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("# UCL Cross-Domain Frames Summary\n\n");
        report.push_str(&format!("**Total Frames**: {}\n\n", self.count()));

        for frame in self.all_frames() {
            report.push_str(&format!("## {}\n", frame.name));
            report.push_str(&format!("{}\n\n", frame.description));
            report.push_str("**Slots**:\n");
            for slot in &frame.slots {
                let req = if slot.required {
                    "required"
                } else {
                    "optional"
                };
                report.push_str(&format!(
                    "- `{}` ({}): {}\n",
                    slot.name, req, slot.description
                ));
            }
            report.push_str(&format!("\n**Domains**: {:?}\n\n", frame.domains));
        }

        report
    }
}

impl Default for UCLFrameSystem {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Note: seed_from_name is imported at the top of this module from primitive_system

/// Create a concept encoding from a string name
pub fn concept_hv(name: &str) -> BinaryHV {
    BinaryHV::random(seed_from_name(name))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ucl_frame_system_creation() {
        let system = UCLFrameSystem::new();

        // Should have all 6 frames
        assert_eq!(system.count(), 6);

        // Check all frames exist
        assert!(system.get_frame("TRADE").is_some());
        assert!(system.get_frame("CONFLICT").is_some());
        assert!(system.get_frame("FEEDBACK_LOOP").is_some());
        assert!(system.get_frame("NORM_ENFORCEMENT").is_some());
        assert!(system.get_frame("COOPERATION").is_some());
        assert!(system.get_frame("ADAPTATION").is_some());
    }

    #[test]
    fn test_trade_frame_structure() {
        let system = UCLFrameSystem::new();
        let trade = system.get_frame("TRADE").unwrap();

        // Check required slots
        assert!(trade.get_slot("giver").is_some());
        assert!(trade.get_slot("receiver").is_some());
        assert!(trade.get_slot("resource").is_some());
        assert!(trade.get_slot("price").is_some());

        // Check slot properties
        let giver = trade.get_slot("giver").unwrap();
        assert!(giver.required);
        assert_eq!(giver.type_constraint, Some("agent".to_string()));
    }

    #[test]
    fn test_trade_frame_instantiation() {
        let system = UCLFrameSystem::new();

        // Create fillers
        let alice = concept_hv("alice");
        let bob = concept_hv("bob");
        let apple = concept_hv("apple");
        let dollar = concept_hv("dollar");

        // Bind slots
        let mut bindings = HashMap::new();
        bindings.insert("giver".to_string(), alice);
        bindings.insert("receiver".to_string(), bob);
        bindings.insert("resource".to_string(), apple);
        bindings.insert("price".to_string(), dollar);

        // Create instance
        let instance = system.instantiate("TRADE", bindings).unwrap();

        assert_eq!(instance.frame_name, "TRADE");
        assert!(instance.bindings.contains_key("giver"));
    }

    #[test]
    fn test_trade_frame_missing_slots() {
        let system = UCLFrameSystem::new();

        // Missing required slot
        let mut bindings = HashMap::new();
        bindings.insert("giver".to_string(), concept_hv("alice"));
        bindings.insert("receiver".to_string(), concept_hv("bob"));
        // Missing "resource" and "price"

        let result = system.instantiate("TRADE", bindings);
        assert!(result.is_err());
    }

    #[test]
    fn test_conflict_frame_structure() {
        let system = UCLFrameSystem::new();
        let conflict = system.get_frame("CONFLICT").unwrap();

        // Check required slots
        assert!(conflict.get_slot("parties").is_some());
        assert!(conflict.get_slot("stakes").is_some());

        // Check optional slots
        let strategies = conflict.get_slot("strategies").unwrap();
        assert!(!strategies.required);
    }

    #[test]
    fn test_conflict_frame_instantiation() {
        let system = UCLFrameSystem::new();

        let mut bindings = HashMap::new();
        bindings.insert("parties".to_string(), concept_hv("team_a_vs_team_b"));
        bindings.insert("stakes".to_string(), concept_hv("championship_trophy"));
        bindings.insert("strategies".to_string(), concept_hv("offensive_tactics"));

        let instance = system.instantiate("CONFLICT", bindings).unwrap();
        assert_eq!(instance.frame_name, "CONFLICT");
    }

    #[test]
    fn test_feedback_loop_frame_structure() {
        let system = UCLFrameSystem::new();
        let feedback = system.get_frame("FEEDBACK_LOOP").unwrap();

        // Check required slots
        assert!(feedback.get_slot("variable").is_some());
        assert!(feedback.get_slot("influence_path").is_some());
        assert!(feedback.get_slot("sign").is_some());

        // Check it bridges causality and homeostasis
        assert!(feedback.domains.contains(&"causality".to_string()));
        assert!(feedback.domains.contains(&"homeostasis".to_string()));
    }

    #[test]
    fn test_feedback_loop_instantiation() {
        let system = UCLFrameSystem::new();

        let mut bindings = HashMap::new();
        bindings.insert("variable".to_string(), concept_hv("body_temperature"));
        bindings.insert(
            "influence_path".to_string(),
            concept_hv("hypothalamus_to_sweat_glands"),
        );
        bindings.insert("sign".to_string(), concept_hv("negative_feedback"));
        bindings.insert("setpoint".to_string(), concept_hv("37_degrees"));

        let instance = system.instantiate("FEEDBACK_LOOP", bindings).unwrap();
        assert_eq!(instance.frame_name, "FEEDBACK_LOOP");
    }

    #[test]
    fn test_norm_enforcement_frame_structure() {
        let system = UCLFrameSystem::new();
        let norm = system.get_frame("NORM_ENFORCEMENT").unwrap();

        // Check required slots
        assert!(norm.get_slot("norm").is_some());
        assert!(norm.get_slot("violator").is_some());

        // Observer and sanction should be optional
        let observer = norm.get_slot("observer").unwrap();
        assert!(!observer.required);
    }

    #[test]
    fn test_norm_enforcement_instantiation() {
        let system = UCLFrameSystem::new();

        let mut bindings = HashMap::new();
        bindings.insert("norm".to_string(), concept_hv("no_stealing"));
        bindings.insert("violator".to_string(), concept_hv("thief"));
        bindings.insert("observer".to_string(), concept_hv("witness"));
        bindings.insert("sanction".to_string(), concept_hv("jail_time"));

        let instance = system.instantiate("NORM_ENFORCEMENT", bindings).unwrap();
        assert_eq!(instance.frame_name, "NORM_ENFORCEMENT");
    }

    #[test]
    fn test_cooperation_frame_structure() {
        let system = UCLFrameSystem::new();
        let coop = system.get_frame("COOPERATION").unwrap();

        // Check required slots
        assert!(coop.get_slot("agents").is_some());
        assert!(coop.get_slot("shared_goal").is_some());
        assert!(coop.get_slot("contributions").is_some());

        // Check it bridges social and game theory
        assert!(coop.domains.contains(&"social".to_string()));
        assert!(coop.domains.contains(&"game_theory".to_string()));
    }

    #[test]
    fn test_cooperation_frame_instantiation() {
        let system = UCLFrameSystem::new();

        let mut bindings = HashMap::new();
        bindings.insert("agents".to_string(), concept_hv("research_team"));
        bindings.insert("shared_goal".to_string(), concept_hv("publish_paper"));
        bindings.insert(
            "contributions".to_string(),
            concept_hv("data_analysis_writing"),
        );

        let instance = system.instantiate("COOPERATION", bindings).unwrap();
        assert_eq!(instance.frame_name, "COOPERATION");
    }

    #[test]
    fn test_adaptation_frame_structure() {
        let system = UCLFrameSystem::new();
        let adapt = system.get_frame("ADAPTATION").unwrap();

        // Check required slots
        assert!(adapt.get_slot("system").is_some());
        assert!(adapt.get_slot("environment").is_some());
        assert!(adapt.get_slot("pressure").is_some());
        assert!(adapt.get_slot("response").is_some());

        // Check it bridges biology and metacognition
        assert!(adapt.domains.contains(&"biology".to_string()));
        assert!(adapt.domains.contains(&"metacognition".to_string()));
    }

    #[test]
    fn test_adaptation_frame_instantiation() {
        let system = UCLFrameSystem::new();

        let mut bindings = HashMap::new();
        bindings.insert("system".to_string(), concept_hv("organism"));
        bindings.insert("environment".to_string(), concept_hv("cold_climate"));
        bindings.insert("pressure".to_string(), concept_hv("heat_loss"));
        bindings.insert("response".to_string(), concept_hv("grow_thicker_fur"));
        bindings.insert("mechanism".to_string(), concept_hv("natural_selection"));

        let instance = system.instantiate("ADAPTATION", bindings).unwrap();
        assert_eq!(instance.frame_name, "ADAPTATION");
    }

    #[test]
    fn test_frame_encodings_orthogonal() {
        let system = UCLFrameSystem::new();

        // All frame encodings should be relatively orthogonal
        let frames: Vec<_> = system.all_frames().collect();

        for i in 0..frames.len() {
            for j in (i + 1)..frames.len() {
                let sim = frames[i]
                    .frame_encoding
                    .similarity(&frames[j].frame_encoding);
                // Frames should be fairly orthogonal (similarity < 0.7)
                assert!(
                    sim < 0.7,
                    "Frames {} and {} have too high similarity: {}",
                    frames[i].name,
                    frames[j].name,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_frame_detection() {
        let system = UCLFrameSystem::new();

        // Create a TRADE instance
        let mut bindings = HashMap::new();
        bindings.insert("giver".to_string(), concept_hv("alice"));
        bindings.insert("receiver".to_string(), concept_hv("bob"));
        bindings.insert("resource".to_string(), concept_hv("book"));
        bindings.insert("price".to_string(), concept_hv("money"));

        let instance = system.instantiate("TRADE", bindings).unwrap();

        // Should detect TRADE as the most similar frame
        let detections = system.detect_frame(&instance.encoding, 0.1);
        assert!(!detections.is_empty(), "Should detect at least one frame");

        // TRADE should be in the detections (exact encoding is bundled with slots)
        let trade_detection = detections.iter().find(|(name, _)| *name == "TRADE");
        assert!(trade_detection.is_some(), "TRADE frame should be detected");
    }

    #[test]
    fn test_instance_similarity() {
        let system = UCLFrameSystem::new();

        // Two similar TRADE instances
        let mut bindings1 = HashMap::new();
        bindings1.insert("giver".to_string(), concept_hv("alice"));
        bindings1.insert("receiver".to_string(), concept_hv("bob"));
        bindings1.insert("resource".to_string(), concept_hv("apple"));
        bindings1.insert("price".to_string(), concept_hv("dollar"));

        let mut bindings2 = HashMap::new();
        bindings2.insert("giver".to_string(), concept_hv("alice"));
        bindings2.insert("receiver".to_string(), concept_hv("bob"));
        bindings2.insert("resource".to_string(), concept_hv("orange")); // Different resource
        bindings2.insert("price".to_string(), concept_hv("dollar"));

        let instance1 = system.instantiate("TRADE", bindings1).unwrap();
        let instance2 = system.instantiate("TRADE", bindings2).unwrap();

        // Instances with same giver/receiver/price should be somewhat similar
        let sim = instance1.similarity(&instance2);
        assert!(
            sim > 0.3,
            "Instances with 3/4 same fillers should have moderate similarity: {}",
            sim
        );
    }

    #[test]
    fn test_slot_binding_unbinding() {
        let system = UCLFrameSystem::new();
        let trade = system.get_frame("TRADE").unwrap();

        let alice = concept_hv("alice");
        let giver_slot = trade.get_slot("giver").unwrap();

        // Bind and unbind
        let bound = giver_slot.bind(&alice);
        let unbound = bound.bind(&giver_slot.role_marker);

        // Unbound should be similar to original
        let sim = alice.similarity(&unbound);
        assert!(
            sim > 0.3,
            "Unbinding should recover approximately the original: {}",
            sim
        );
    }

    #[test]
    fn test_summary_generation() {
        let system = UCLFrameSystem::new();
        let summary = system.summary();

        // Summary should contain all frame names
        assert!(summary.contains("TRADE"));
        assert!(summary.contains("CONFLICT"));
        assert!(summary.contains("FEEDBACK_LOOP"));
        assert!(summary.contains("NORM_ENFORCEMENT"));
        assert!(summary.contains("COOPERATION"));
        assert!(summary.contains("ADAPTATION"));

        // Summary should contain slot information
        assert!(summary.contains("giver"));
        assert!(summary.contains("parties"));
        assert!(summary.contains("variable"));
    }
}
