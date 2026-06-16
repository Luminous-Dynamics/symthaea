// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Graceful Ignorance System (GIS)
//!
//! The Graceful Ignorance System enables AI to **explicitly track, communicate,
//! and collectively resolve** knowledge gaps. It transforms ignorance from a
//! hidden liability into a visible asset.
//!
//! ## Architecture Evolution
//!
//! - **v1.0 (Hygiene)**: Detect and classify ignorance
//! - **v2.0 (Immune System)**: Actively hunt and resolve ignorance
//! - **v3.0 (Benevolent Intelligence)**: Wisdom, empathy, and moral reasoning
//!
//! ## Core Components
//!
//! 1. **Ignorance Taxonomy**: 5 types of ignorance (κ, ι₁, ι₂, ι₃, ι∞)
//! 2. **3D Uncertainty**: Epistemic, aleatoric, and structural dimensions
//! 3. **Dark Spot DHT**: Distributed hash table of collective ignorance
//! 4. **Zero-Knowledge Signatures**: Privacy-preserving ignorance publication
//! 5. **Curiosity Engine**: Active ignorance resolution via EIG
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::{GracefulIgnoranceSystem, IgnoranceType};
//!
//! let mut gis = GracefulIgnoranceSystem::new();
//!
//! // Detect ignorance
//! let ignorance = gis.detect_ignorance("What is the capital of Mars?");
//! assert_eq!(ignorance.ignorance_type, IgnoranceType::Impossible);
//!
//! // Publish to Dark Spot DHT (privacy-preserving)
//! let signature = gis.publish_ignorance(ignorance)?;
//!
//! // Check for matches
//! if let Some(knowledge) = gis.check_matches(&signature) {
//!     println!("Found knowledge to fill gap!");
//! }
//! ```

pub mod curiosity;
pub mod dht;
pub mod epistemic_mirror;
pub mod ignorance_types;
pub mod persistence;
pub mod rashomon;
pub mod socratic;
pub mod uncertainty;
pub mod zk_proofs;

pub use curiosity::*;
pub use dht::*;
pub use epistemic_mirror::*;
pub use ignorance_types::*;
pub use persistence::*;
pub use rashomon::*;
pub use socratic::*;
pub use uncertainty::*;
pub use zk_proofs::*;

// EpistemicClassification is available via crate::mycelix::types if needed
use std::collections::HashMap;
use std::time::SystemTime;

/// The main Graceful Ignorance System
pub struct GracefulIgnoranceSystem {
    /// Active ignorance records
    ignorance_records: HashMap<String, IgnoranceRecord>,

    /// Dark Spot DHT connection
    dark_spot_dht: DarkSpotDHT,

    /// Curiosity engine for active resolution
    curiosity_engine: CuriosityEngine,

    /// Configuration
    config: GISConfig,
}

impl GracefulIgnoranceSystem {
    /// Create a new GIS instance
    pub fn new() -> Self {
        Self::with_config(GISConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GISConfig) -> Self {
        Self {
            ignorance_records: HashMap::new(),
            dark_spot_dht: DarkSpotDHT::new(),
            curiosity_engine: CuriosityEngine::new(config.curiosity_config.clone()),
            config,
        }
    }

    /// Detect ignorance in a query
    ///
    /// Analyzes the query to determine what type of ignorance is present
    /// and quantifies the uncertainty.
    pub fn detect_ignorance(&self, query: &str) -> IgnoranceDetection {
        // 1. Classify the query domain
        let domain = self.classify_domain(query);

        // 2. Check against known knowledge
        let knowledge_check = self.check_knowledge(query);

        // 3. Determine ignorance type
        let ignorance_type = match (&domain, &knowledge_check) {
            (_, KnowledgeCheck::Known) => IgnoranceType::None,
            (Domain::Undefined, _) => IgnoranceType::Impossible,
            (_, KnowledgeCheck::TemporallyUnknown) => IgnoranceType::Known,
            (_, KnowledgeCheck::DerivableUnknown) => IgnoranceType::KnownUnknown,
            (_, KnowledgeCheck::StructurallyUnknown) => IgnoranceType::Unknown,
            (_, KnowledgeCheck::FundamentallyUnknown) => IgnoranceType::Impossible,
        };

        // 4. Calculate 3D uncertainty
        let uncertainty = self.calculate_uncertainty(query, &ignorance_type);

        // 5. Calculate Expected Information Gain
        let eig = self.calculate_eig(query, &ignorance_type, &uncertainty);

        IgnoranceDetection {
            query: query.to_string(),
            ignorance_type,
            uncertainty,
            domain,
            eig,
            detected_at: SystemTime::now(),
        }
    }

    /// Record ignorance for tracking
    pub fn record_ignorance(&mut self, detection: IgnoranceDetection) -> String {
        let id = self.generate_ignorance_id(&detection);

        let record = IgnoranceRecord {
            id: id.clone(),
            detection,
            status: IgnoranceStatus::Active,
            resolution: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.ignorance_records.insert(id.clone(), record);
        id
    }

    /// Publish ignorance to Dark Spot DHT
    ///
    /// Creates a privacy-preserving signature and publishes to the DHT.
    pub fn publish_ignorance(
        &mut self,
        detection: &IgnoranceDetection,
    ) -> Result<ZKIgnoranceSignature, GISError> {
        // Only publish if EIG is above threshold
        if detection.eig < self.config.publish_eig_threshold {
            return Err(GISError::EIGBelowThreshold {
                actual: detection.eig,
                required: self.config.publish_eig_threshold,
            });
        }

        // Create ZK signature
        let signature = self.dark_spot_dht.create_signature(
            &detection.query,
            &detection.domain.to_string(),
            detection.eig,
        )?;

        // Publish to DHT
        self.dark_spot_dht.publish(&signature)?;

        Ok(signature)
    }

    /// Check for matches in the network
    pub fn check_matches(&self, signature: &ZKIgnoranceSignature) -> Vec<KnowledgeMatch> {
        self.dark_spot_dht.find_matches(signature)
    }

    /// Generate response based on ignorance detection
    pub fn generate_response(&self, detection: &IgnoranceDetection) -> GracefulResponse {
        match detection.ignorance_type {
            IgnoranceType::None => GracefulResponse::Answer {
                confidence: 1.0 - detection.uncertainty.total(),
            },

            IgnoranceType::Known => GracefulResponse::Deferred {
                reason: "This information exists but is not currently accessible".to_string(),
                resolution_hint: Some("Will be available when: [condition]".to_string()),
            },

            IgnoranceType::KnownUnknown => GracefulResponse::Uncertain {
                confidence: 0.3,
                uncertainty_breakdown: detection.uncertainty,
                caveat: "This is a known unknown - the answer exists but I don't have it"
                    .to_string(),
            },

            IgnoranceType::Unknown => GracefulResponse::Unknown {
                explanation: "I don't know what I don't know here".to_string(),
                could_be_wrong: true,
                request_for_context: Some("Could you provide more context?".to_string()),
            },

            IgnoranceType::Impossible => GracefulResponse::OutOfDomain {
                reason: "This question may be unanswerable in principle".to_string(),
                suggestions: vec![
                    "Rephrase the question".to_string(),
                    "Check if the premises are valid".to_string(),
                ],
            },
        }
    }

    /// Request active resolution of ignorance
    pub fn request_resolution(&mut self, id: &str) -> Result<ResolutionRequest, GISError> {
        let record = self
            .ignorance_records
            .get_mut(id)
            .ok_or_else(|| GISError::NotFound(id.to_string()))?;

        if record.status != IgnoranceStatus::Active {
            return Err(GISError::InvalidStatus {
                current: record.status,
                required: IgnoranceStatus::Active,
            });
        }

        record.status = IgnoranceStatus::ResolutionRequested;
        record.updated_at = SystemTime::now();

        self.curiosity_engine.queue_resolution(&record.detection)
    }

    /// Get ignorance statistics
    pub fn statistics(&self) -> GISStatistics {
        let mut stats = GISStatistics::default();

        for record in self.ignorance_records.values() {
            match record.detection.ignorance_type {
                IgnoranceType::None => stats.known_count += 1,
                IgnoranceType::Known => stats.known_unknown_count += 1,
                IgnoranceType::KnownUnknown => stats.known_unknown_count += 1,
                IgnoranceType::Unknown => stats.unknown_unknown_count += 1,
                IgnoranceType::Impossible => stats.impossible_count += 1,
            }

            match record.status {
                IgnoranceStatus::Active => stats.active_count += 1,
                IgnoranceStatus::ResolutionRequested => stats.pending_resolution += 1,
                IgnoranceStatus::Resolved => stats.resolved_count += 1,
                IgnoranceStatus::Expired => stats.expired_count += 1,
            }
        }

        stats.total_count = self.ignorance_records.len();
        stats.average_eig = if stats.total_count > 0 {
            self.ignorance_records
                .values()
                .map(|r| r.detection.eig)
                .sum::<f32>()
                / stats.total_count as f32
        } else {
            0.0
        };

        stats
    }

    /// Get the dominant ignorance type across all active records.
    ///
    /// Returns the most frequently occurring ignorance type. If no records exist,
    /// returns `IgnoranceType::None` (no ignorance detected).
    pub fn dominant_ignorance_type(&self) -> IgnoranceType {
        if self.ignorance_records.is_empty() {
            return IgnoranceType::None;
        }

        let mut counts = [0u32; 5]; // None, Known, KnownUnknown, Unknown, Impossible
        for record in self.ignorance_records.values() {
            let idx = match record.detection.ignorance_type {
                IgnoranceType::None => 0,
                IgnoranceType::Known => 1,
                IgnoranceType::KnownUnknown => 2,
                IgnoranceType::Unknown => 3,
                IgnoranceType::Impossible => 4,
            };
            counts[idx] += 1;
        }

        let (max_idx, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap_or((0, &0));

        match max_idx {
            0 => IgnoranceType::None,
            1 => IgnoranceType::Known,
            2 => IgnoranceType::KnownUnknown,
            3 => IgnoranceType::Unknown,
            4 => IgnoranceType::Impossible,
            _ => IgnoranceType::None,
        }
    }

    // --- Private helpers ---

    fn classify_domain(&self, query: &str) -> Domain {
        // Simple domain classification based on keywords
        // In production, this would use HDC semantic matching
        let query_lower = query.to_lowercase();

        if query_lower.contains("capital of mars") || query_lower.contains("impossible") {
            Domain::Undefined
        } else if query_lower.contains("math") || query_lower.contains("calculate") {
            Domain::Mathematics
        } else if query_lower.contains("physics") || query_lower.contains("gravity") {
            Domain::Physics
        } else if query_lower.contains("history") || query_lower.contains("when did") {
            Domain::History
        } else if query_lower.contains("feel") || query_lower.contains("opinion") {
            Domain::Subjective
        } else {
            Domain::General
        }
    }

    fn check_knowledge(&self, _query: &str) -> KnowledgeCheck {
        // Placeholder - would query knowledge base
        KnowledgeCheck::StructurallyUnknown
    }

    fn calculate_uncertainty(&self, _query: &str, ignorance_type: &IgnoranceType) -> Uncertainty3D {
        // Calculate uncertainty based on ignorance type
        match ignorance_type {
            IgnoranceType::None => Uncertainty3D::new(0.05, 0.05, 0.05),
            IgnoranceType::Known => Uncertainty3D::new(0.2, 0.1, 0.1),
            IgnoranceType::KnownUnknown => Uncertainty3D::new(0.5, 0.3, 0.2),
            IgnoranceType::Unknown => Uncertainty3D::new(0.8, 0.5, 0.5),
            IgnoranceType::Impossible => Uncertainty3D::new(1.0, 0.8, 0.8),
        }
    }

    fn calculate_eig(
        &self,
        _query: &str,
        ignorance_type: &IgnoranceType,
        uncertainty: &Uncertainty3D,
    ) -> f32 {
        // Expected Information Gain = Potential value × Likelihood of resolution
        let potential_value = match ignorance_type {
            IgnoranceType::None => 0.0,
            IgnoranceType::Known => 0.9,
            IgnoranceType::KnownUnknown => 0.7,
            IgnoranceType::Unknown => 0.4,
            IgnoranceType::Impossible => 0.1,
        };

        let likelihood = 1.0 - uncertainty.epistemic;

        potential_value * likelihood
    }

    fn generate_ignorance_id(&self, detection: &IgnoranceDetection) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        detection.query.hash(&mut hasher);
        detection.detected_at.hash(&mut hasher);
        format!("ign_{:016x}", hasher.finish())
    }
}

impl Default for GracefulIgnoranceSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// GIS Configuration
#[derive(Debug, Clone)]
pub struct GISConfig {
    /// Minimum EIG to publish to DHT
    pub publish_eig_threshold: f32,
    /// Maximum ignorance records to keep
    pub max_records: usize,
    /// Curiosity engine configuration
    pub curiosity_config: CuriosityConfig,
}

impl Default for GISConfig {
    fn default() -> Self {
        Self {
            publish_eig_threshold: 0.3,
            max_records: 10000,
            curiosity_config: CuriosityConfig::default(),
        }
    }
}

/// GIS Statistics
#[derive(Debug, Default)]
pub struct GISStatistics {
    pub total_count: usize,
    pub known_count: usize,
    pub known_unknown_count: usize,
    pub unknown_unknown_count: usize,
    pub impossible_count: usize,
    pub active_count: usize,
    pub pending_resolution: usize,
    pub resolved_count: usize,
    pub expired_count: usize,
    pub average_eig: f32,
}

/// Result of ignorance detection
#[derive(Debug, Clone)]
pub struct IgnoranceDetection {
    /// Original query
    pub query: String,
    /// Type of ignorance
    pub ignorance_type: IgnoranceType,
    /// 3D uncertainty quantification
    pub uncertainty: Uncertainty3D,
    /// Query domain
    pub domain: Domain,
    /// Expected Information Gain
    pub eig: f32,
    /// When detected
    pub detected_at: SystemTime,
}

/// Knowledge check result
#[derive(Debug)]
#[allow(dead_code)] // RESERVED(future): knowledge classification variants
enum KnowledgeCheck {
    Known,
    TemporallyUnknown,
    DerivableUnknown,
    StructurallyUnknown,
    FundamentallyUnknown,
}

/// Graceful response to ignorance
#[derive(Debug)]
pub enum GracefulResponse {
    /// Have the answer
    Answer { confidence: f32 },

    /// Know the answer exists but can't access it now
    Deferred {
        reason: String,
        resolution_hint: Option<String>,
    },

    /// Uncertain but have partial answer
    Uncertain {
        confidence: f32,
        uncertainty_breakdown: Uncertainty3D,
        caveat: String,
    },

    /// Don't know what we don't know
    Unknown {
        explanation: String,
        could_be_wrong: bool,
        request_for_context: Option<String>,
    },

    /// Question is outside answerable domain
    OutOfDomain {
        reason: String,
        suggestions: Vec<String>,
    },
}

/// GIS Errors
#[derive(Debug)]
pub enum GISError {
    NotFound(String),
    EIGBelowThreshold {
        actual: f32,
        required: f32,
    },
    InvalidStatus {
        current: IgnoranceStatus,
        required: IgnoranceStatus,
    },
    DHTError(String),
    CryptoError(String),
}

impl std::fmt::Display for GISError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Ignorance record not found: {id}"),
            Self::EIGBelowThreshold { actual, required } => {
                write!(f, "EIG {actual} below threshold {required}")
            }
            Self::InvalidStatus { current, required } => {
                write!(f, "Invalid status: {current:?}, required: {required:?}")
            }
            Self::DHTError(msg) => write!(f, "DHT error: {msg}"),
            Self::CryptoError(msg) => write!(f, "Crypto error: {msg}"),
        }
    }
}

impl std::error::Error for GISError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gis_creation() {
        let gis = GracefulIgnoranceSystem::new();
        assert_eq!(gis.statistics().total_count, 0);
    }

    #[test]
    fn test_ignorance_detection() {
        let gis = GracefulIgnoranceSystem::new();

        let detection = gis.detect_ignorance("What is the capital of Mars?");
        assert_eq!(detection.ignorance_type, IgnoranceType::Impossible);
        assert!(detection.eig < 0.5);

        let detection2 = gis.detect_ignorance("What is 2 + 2?");
        assert!(detection2.uncertainty.total() > 0.0);
    }

    #[test]
    fn test_record_ignorance() {
        let mut gis = GracefulIgnoranceSystem::new();

        let detection = gis.detect_ignorance("Some unknown question");
        let id = gis.record_ignorance(detection);

        assert!(gis.ignorance_records.contains_key(&id));
        assert_eq!(gis.statistics().total_count, 1);
    }

    #[test]
    fn test_graceful_response() {
        let gis = GracefulIgnoranceSystem::new();

        let detection = gis.detect_ignorance("What is the capital of Mars?");
        let response = gis.generate_response(&detection);

        match response {
            GracefulResponse::OutOfDomain { .. } => (),
            _ => panic!("Expected OutOfDomain response"),
        }
    }
}
