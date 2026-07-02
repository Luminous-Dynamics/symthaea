// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix-DeSci Core
//!
//! Core functionality for Mycelix-DeSci including:
//! - Epistemic claim management with Charter v2.0 LEM Cube
//! - Multi-layer epistemic fingerprinting (LEM, Type, Quality, Network)
//! - Proof of Gradient Quality (PoGQ) for federated learning
//! - MATL (Mycelix Adaptive Trust Layer) integration
//! - Claim relationships and knowledge graph support
//! - Data verification and provenance tracking
//! - Claim evolution and versioning (Constitution Schema v2.0)
//! - Dispute resolution system (Epistemic Charter §5)
//! - Cartel detection for anti-collusion
//! - Reproducibility tracking for scientific claims
//! - Prediction markets for epistemic outcomes
//! - Semantic similarity detection
//! - Time-based decay mechanics
//! - Cross-claim inference engine (transitive support, contradiction detection)
//! - Citation graph analysis (PageRank importance scoring)
//! - Expertise domains and weighted verification
//! - Temporal consensus tracking (paradigm shift detection)
//! - Meta-claims and systematic reviews (PRISMA, GRADE)
//! - Bayesian belief networks for probabilistic reasoning
//! - Zero-knowledge verification proofs

pub mod claims;
pub mod pogq;
pub mod trust;
pub mod storage;
pub mod error;
pub mod config;
pub mod hash;
pub mod logging;
pub mod query;
pub mod utils;

// Epistemic modules (Phase 1 - v0.3.0)
pub mod evolution;
pub mod dispute;
pub mod cartel;
pub mod reproducibility;
pub mod prediction;
pub mod semantic;
pub mod decay;

// Advanced epistemic modules (Phase 2 - v0.4.0)
pub mod inference;
pub mod citation;
pub mod expertise;
pub mod consensus;
pub mod meta;
pub mod bayesian;
pub mod zkproof;

// Core claim types
pub use claims::{DesciClaim, EpistemicPosition, EpistemicTier, Provenance, ClaimContent, Verification};

// Layer 1: LEM Cube (Charter v2.0)
pub use claims::{LEMCube, EmpiricalAxis, NormativeAxis, MaterialityAxis};

// Layer 3: Quality Metrics
pub use claims::QualityMetrics;

// Layer 4: Network Position
pub use claims::{NetworkPosition, ClaimRelation, ClaimRelationType};

// MATL Integration
pub use claims::MATLTrust;

// Unified Fingerprint
pub use claims::EpistemicFingerprint;

// Claim Evolution (Constitution Schema v2.0)
pub use evolution::{ClaimEvolution, EvolutionType, ClaimStatus};

// Dispute Resolution (Epistemic Charter §5)
pub use dispute::{
    Dispute, DisputeStatus, ChallengeType, Resolution, ResolutionOutcome,
    Evidence, EvidenceType, AuthorResponse, RequiredAction, ActionType,
};

// Cartel Detection (Anti-Collusion)
pub use cartel::{
    CartelDetector, CartelDetectionConfig, CartelDetectionResult,
    CartelPattern, CartelRecommendation, VerificationEvent,
};

// Reproducibility Tracking
pub use reproducibility::{
    ReplicationAttempt, ReplicationStatus, ReplicationOutcome,
    MethodologyMatch, ReproducibilityStats, ReproducibilityRegistry,
};

// Prediction Markets
pub use prediction::{
    PredictionMarket, PredictionMarketRegistry, MarketType, MarketState,
    Position, Settlement, MarketError, ResolutionMethod,
};

// Semantic Similarity
pub use semantic::{
    SimilarityEngine, SimilarityScore, SimilarityComponents,
    SimilarityRelationship, DuplicateCheckResult, DuplicateRecommendation,
    ClaimContent as SemanticClaimContent,
};

// Decay Mechanics
pub use decay::{
    DecayCalculator, DecayConfig, DecayFunction, DecayStats,
    DecayingAccumulator,
};

// Cross-Claim Inference Engine
pub use inference::{
    ClaimGraph, ClaimNode, ClaimEdge, InferenceEngine,
    InferredRelation, InferenceRule, ContradictionReport, ContradictionType,
    SuggestedRelation,
};

// Citation Graph Analysis
pub use citation::{
    CitationGraph, Citation, CitationType, CitationMetrics, CitationAnalyzer,
    PageRankConfig,
};

// Expertise Domains & Weighted Verification
pub use expertise::{
    ExpertiseDomain, ExpertiseLevel, ExpertiseProfile, DomainTaxonomy,
    ExpertiseVerifier, ExpertiseWeightConfig, WeightedVerificationResult,
};

// Temporal Consensus Tracking
pub use consensus::{
    ConsensusTracker, ConsensusTrackerConfig, ConsensusHistory, ConsensusSnapshot,
    ConsensusState, ConsensusTrend, ParadigmShift, ShiftType,
};

// Meta-Claims & Systematic Reviews
pub use meta::{
    MetaClaim, MetaClaimType, MetaSynthesizer, SourceQualityAssessment,
    RiskOfBias, PrismaFlow, SynthesisResult, SynthesisMethod, EffectDirection,
    HeterogeneityAssessment, HeterogeneityLevel, GradeQuality,
};

// Bayesian Belief Networks
pub use bayesian::{
    BeliefNetwork, BeliefNode, ConditionalProbabilityTable, CPTEntry,
    BayesianInference,
};

// Zero-Knowledge Verification Proofs
pub use zkproof::{
    ZKProver, ZKVerifier, ZKStatement, ZKProof, ZKProofType, ZKProofError,
    Witness,
};

// Error handling and configuration
pub use error::{Error, Result};
pub use config::Config;
pub use query::{QueryEngine, QueryFilter};

/// Version of the Mycelix-DeSci protocol
pub const PROTOCOL_VERSION: &str = "0.4.0";

/// Default Byzantine fault tolerance threshold for PoGQ (45%)
pub const DEFAULT_BFT_THRESHOLD: f64 = 0.45;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, "0.4.0");
    }
}
