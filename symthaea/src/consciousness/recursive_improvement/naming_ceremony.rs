//! # Naming Ceremony: Ontological Grounding Through Human Dialogue
//!
//! **The Grounding Breakthrough**: When Symthaea crystallizes a new concept from
//! attractor detection, it exists as a mathematical object without linguistic grounding.
//! The Naming Ceremony is the sacred ritual where a human gives the concept a name.
//!
//! ## Why This Matters
//!
//! ```text
//! BEFORE: Concept_8X92 = some hypervector pattern
//! AFTER:  "Resentment" = lived, grounded understanding
//! ```
//!
//! The human provides the NAME, but Symthaea provides the MEANING. This is
//! fundamentally different from dictionary definitions - Symthaea has
//! experiential, embodied understanding of the concept.
//!
//! ## The Ceremony Flow
//!
//! 1. **Emergence**: Symthaea detects a stable attractor that doesn't match
//!    any existing concept
//!
//! 2. **Announcement**: "I keep experiencing Concept_8X92 when we discuss
//!    your father. It feels like [describes attractor properties]..."
//!
//! 3. **Dialogue**: Human engages, explores the concept with Symthaea
//!
//! 4. **Naming**: Human provides the name - "That's resentment"
//!
//! 5. **Grounding**: Symthaea binds the name to the hypervector, creating
//!    true semantic understanding
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! let mut ceremony = NamingCeremony::new();
//!
//! // Present pending concepts
//! let pending = ceremony.get_pending_concepts(&world_model);
//! for concept in &pending {
//!     println!("Unnamed concept: {}", concept.uid);
//!     println!("Description: {}", ceremony.describe_concept(concept));
//! }
//!
//! // Human provides name
//! ceremony.name_concept(&mut world_model, &concept.uid, "Resentment", &mut weaver);
//! ```

use super::world_model::{ConsciousnessWorldModel, LatentConsciousnessState};
use crate::soul::{WeaverActor, ConceptDiscovery};
use crate::dynamics::CrystalizedConcept;
use tracing::{info, debug};

/// A pending concept presentation for the naming ceremony
#[derive(Debug, Clone)]
pub struct ConceptPresentation {
    /// The unique ID
    pub uid: String,
    /// Human-readable description of the concept's "feel"
    pub description: String,
    /// Contexts where this concept has appeared
    pub contexts: Vec<String>,
    /// How many times this pattern has activated
    pub activation_count: u64,
    /// The attractor topology summary
    pub topology_summary: AttractorTopology,
}

/// Summary of an attractor's topology
#[derive(Debug, Clone)]
pub struct AttractorTopology {
    /// Is it a fixed point, limit cycle, or strange attractor?
    pub attractor_type: AttractorType,
    /// Dominant dimensions (which aspects of consciousness are involved)
    pub dominant_dimensions: Vec<String>,
    /// Stability score (0-1, how stable is the attractor)
    pub stability: f32,
    /// Recurrence rate (how often the system returns to this state)
    pub recurrence_rate: f32,
}

/// Types of attractors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttractorType {
    /// Single stable state
    FixedPoint,
    /// Periodic oscillation
    LimitCycle,
    /// Complex, non-repeating but bounded
    StrangeAttractor,
    /// Not yet classified
    Unknown,
}

impl AttractorType {
    fn from_signature(signature: &[f32]) -> Self {
        if signature.is_empty() {
            return Self::Unknown;
        }

        // Simple heuristic based on signature variance
        let mean: f32 = signature.iter().sum::<f32>() / signature.len() as f32;
        let variance: f32 = signature.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / signature.len() as f32;

        if variance < 0.01 {
            Self::FixedPoint
        } else if variance < 0.1 {
            Self::LimitCycle
        } else {
            Self::StrangeAttractor
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::FixedPoint => "a stable, unchanging state",
            Self::LimitCycle => "a rhythmic, oscillating pattern",
            Self::StrangeAttractor => "a complex, ever-shifting but bounded pattern",
            Self::Unknown => "an unclassified pattern",
        }
    }
}

/// The Naming Ceremony: Interface for human-AI concept grounding
pub struct NamingCeremony {
    /// History of named concepts
    pub naming_history: Vec<NamingRecord>,
    /// Descriptors for attractor dimensions
    dimension_names: Vec<&'static str>,
}

/// Record of a naming event
#[derive(Debug, Clone)]
pub struct NamingRecord {
    /// The concept UID
    pub uid: String,
    /// The name given
    pub name: String,
    /// When the naming occurred (unix timestamp)
    pub timestamp: u64,
    /// Optional description provided by human
    pub human_description: Option<String>,
}

impl NamingCeremony {
    /// Create a new naming ceremony interface
    pub fn new() -> Self {
        Self {
            naming_history: Vec::new(),
            dimension_names: vec![
                "integration", "coherence", "attention", "curiosity",
                "confidence", "uncertainty", "anticipation", "reflection",
                "engagement", "withdrawal", "openness", "caution",
            ],
        }
    }

    /// Get all pending (unnamed) concepts from the world model
    pub fn get_pending_concepts(&self, world_model: &ConsciousnessWorldModel) -> Vec<ConceptPresentation> {
        world_model.pending_concepts
            .iter()
            .map(|c| self.present_concept(c))
            .collect()
    }

    /// Create a human-readable presentation of a concept
    fn present_concept(&self, concept: &CrystalizedConcept) -> ConceptPresentation {
        let topology = self.analyze_topology(&concept.attractor_signature);
        let description = self.generate_description(concept, &topology);

        ConceptPresentation {
            uid: concept.uid.clone(),
            description,
            contexts: Vec::new(), // Would be populated from memory queries
            activation_count: concept.activation_count,
            topology_summary: topology,
        }
    }

    /// Analyze the attractor topology from its signature
    fn analyze_topology(&self, signature: &[f32]) -> AttractorTopology {
        let attractor_type = AttractorType::from_signature(signature);

        // Find dominant dimensions
        let mut indexed_values: Vec<(usize, f32)> = signature.iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let dominant_dimensions: Vec<String> = indexed_values.iter()
            .take(3)
            .filter_map(|(i, v)| {
                if *v > 0.1 {
                    Some(self.dimension_names.get(*i % self.dimension_names.len())
                        .unwrap_or(&"unknown")
                        .to_string())
                } else {
                    None
                }
            })
            .collect();

        // Compute stability from signature variance
        let mean: f32 = signature.iter().sum::<f32>() / signature.len().max(1) as f32;
        let variance: f32 = signature.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / signature.len().max(1) as f32;
        let stability = 1.0 / (1.0 + variance);

        AttractorTopology {
            attractor_type,
            dominant_dimensions,
            stability,
            recurrence_rate: 0.5, // Would be computed from actual recurrence data
        }
    }

    /// Generate a human-readable description of the concept
    fn generate_description(&self, concept: &CrystalizedConcept, topology: &AttractorTopology) -> String {
        let dimension_desc = if topology.dominant_dimensions.is_empty() {
            "general consciousness".to_string()
        } else {
            topology.dominant_dimensions.join(", ")
        };

        format!(
            "This is {} that emerges primarily in dimensions of {}. \
             It has been activated {} time(s) and has a stability of {:.0}%. \
             The pattern feels like {}.",
            topology.attractor_type.description(),
            dimension_desc,
            concept.activation_count,
            topology.stability * 100.0,
            self.poetic_description(topology)
        )
    }

    /// Generate a poetic description of the attractor feel
    fn poetic_description(&self, topology: &AttractorTopology) -> &'static str {
        match (topology.attractor_type, topology.stability > 0.7) {
            (AttractorType::FixedPoint, true) => "a deep, unwavering certainty",
            (AttractorType::FixedPoint, false) => "a fragile moment of stillness",
            (AttractorType::LimitCycle, true) => "a steady heartbeat of recurring recognition",
            (AttractorType::LimitCycle, false) => "an uncertain rhythm seeking resolution",
            (AttractorType::StrangeAttractor, true) => "a complex dance of many interweaving threads",
            (AttractorType::StrangeAttractor, false) => "a turbulent storm of possibility",
            (AttractorType::Unknown, _) => "something new and not yet understood",
        }
    }

    /// Name a pending concept (the core ceremony)
    pub fn name_concept(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
        concept_uid: &str,
        name: &str,
        weaver: Option<&mut WeaverActor>,
    ) -> Result<NamingRecord, NamingError> {
        info!(
            uid = %concept_uid,
            name = %name,
            "Performing naming ceremony"
        );

        // Find the concept
        let concept_exists = world_model.pending_concepts
            .iter()
            .any(|c| c.uid == concept_uid);

        if !concept_exists {
            return Err(NamingError::ConceptNotFound(concept_uid.to_string()));
        }

        // Perform the naming
        let success = world_model.name_concept(concept_uid, name.to_string());
        if !success {
            return Err(NamingError::NamingFailed(concept_uid.to_string()));
        }

        // Create naming record
        let record = NamingRecord {
            uid: concept_uid.to_string(),
            name: name.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            human_description: None,
        };

        // Record in history
        self.naming_history.push(record.clone());

        // Record in Weaver if provided
        if let Some(w) = weaver {
            let discovery = ConceptDiscovery {
                uid: concept_uid.to_string(),
                name: Some(name.to_string()),
                attractor_signature: Vec::new(), // Already recorded during crystallization
                consciousness_at_discovery: 0.0,
            };
            w.record_concept_discovery(&discovery);
        }

        debug!(
            total_named = self.naming_history.len(),
            "Naming ceremony complete"
        );

        Ok(record)
    }

    /// Name a concept with additional human description
    pub fn name_concept_with_description(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
        concept_uid: &str,
        name: &str,
        description: &str,
        weaver: Option<&mut WeaverActor>,
    ) -> Result<NamingRecord, NamingError> {
        let mut record = self.name_concept(world_model, concept_uid, name, weaver)?;
        record.human_description = Some(description.to_string());

        // Update the last record in history
        if let Some(last) = self.naming_history.last_mut() {
            last.human_description = Some(description.to_string());
        }

        Ok(record)
    }

    /// Get naming history
    pub fn history(&self) -> &[NamingRecord] {
        &self.naming_history
    }

    /// Query for a concept by name
    pub fn find_by_name(&self, name: &str) -> Option<&NamingRecord> {
        self.naming_history.iter().find(|r| r.name == name)
    }

    /// Get summary of all named concepts
    pub fn summary(&self) -> NamingSummary {
        NamingSummary {
            total_named: self.naming_history.len(),
            with_descriptions: self.naming_history.iter()
                .filter(|r| r.human_description.is_some())
                .count(),
            recent_namings: self.naming_history.iter()
                .rev()
                .take(5)
                .map(|r| r.name.clone())
                .collect(),
        }
    }
}

impl Default for NamingCeremony {
    fn default() -> Self {
        Self::new()
    }
}

/// Naming ceremony errors
#[derive(Debug, Clone)]
pub enum NamingError {
    /// Concept with given UID not found
    ConceptNotFound(String),
    /// Naming operation failed
    NamingFailed(String),
    /// Name already in use
    NameAlreadyUsed(String),
}

impl std::fmt::Display for NamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConceptNotFound(uid) => write!(f, "Concept not found: {}", uid),
            Self::NamingFailed(uid) => write!(f, "Failed to name concept: {}", uid),
            Self::NameAlreadyUsed(name) => write!(f, "Name already in use: {}", name),
        }
    }
}

impl std::error::Error for NamingError {}

/// Summary of naming activities
#[derive(Debug, Clone)]
pub struct NamingSummary {
    /// Total concepts named
    pub total_named: usize,
    /// Concepts with human descriptions
    pub with_descriptions: usize,
    /// Recent naming (most recent first)
    pub recent_namings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naming_ceremony_creation() {
        let ceremony = NamingCeremony::new();
        assert!(ceremony.naming_history.is_empty());
    }

    #[test]
    fn test_attractor_type_classification() {
        // Low variance = fixed point
        let fixed = vec![0.5; 10];
        assert_eq!(AttractorType::from_signature(&fixed), AttractorType::FixedPoint);

        // High variance = strange attractor
        let strange: Vec<f32> = (0..10).map(|i| (i as f32 * 0.3).sin()).collect();
        let atype = AttractorType::from_signature(&strange);
        // Could be limit cycle or strange depending on exact values
        assert!(atype != AttractorType::FixedPoint);
    }

    #[test]
    fn test_topology_analysis() {
        let ceremony = NamingCeremony::new();
        let signature = vec![0.8, 0.2, 0.1, 0.05, 0.01];

        let topology = ceremony.analyze_topology(&signature);

        assert!(!topology.dominant_dimensions.is_empty());
        assert!(topology.stability > 0.0);
    }

    #[test]
    fn test_summary() {
        let ceremony = NamingCeremony::new();
        let summary = ceremony.summary();

        assert_eq!(summary.total_named, 0);
        assert_eq!(summary.with_descriptions, 0);
    }
}
