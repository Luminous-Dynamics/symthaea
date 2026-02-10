//! TypeScript Binding Generation
//!
//! This module contains a test that generates TypeScript bindings
//! for all types annotated with #[ts(export)].
//!
//! Run with: cargo test export_all_bindings --features ts-export -- --nocapture

#[cfg(all(test, feature = "ts-export"))]
mod tests {
    use std::fs;
    use std::path::Path;
    use ts_rs::TS;

    /// Generate all TypeScript bindings
    ///
    /// This test exports all ts-rs annotated types to the bindings/ directory.
    /// Run: cargo test export_all_bindings --features ts-export -- --nocapture
    #[test]
    fn export_all_bindings() {
        // Ensure bindings directories exist
        let binding_dirs = [
            "bindings/matl",
            "bindings/epistemic",
            "bindings/dkg",
            "bindings/agentic",
        ];

        for dir in binding_dirs {
            let path = Path::new(dir);
            if !path.exists() {
                fs::create_dir_all(path).expect("Failed to create bindings directory");
            }
        }

        // Export MATL types
        use crate::matl::{
            GovernanceTier, KVector, KVectorDimension, KVectorWeights, ProofOfGradientQuality,
            RoundState, Vote, VoteType,
        };

        KVector::export_all().expect("Failed to export KVector");
        KVectorWeights::export_all().expect("Failed to export KVectorWeights");
        KVectorDimension::export_all().expect("Failed to export KVectorDimension");
        GovernanceTier::export_all().expect("Failed to export GovernanceTier");
        ProofOfGradientQuality::export_all().expect("Failed to export ProofOfGradientQuality");
        RoundState::export_all().expect("Failed to export RoundState");
        VoteType::export_all().expect("Failed to export VoteType");
        Vote::export_all().expect("Failed to export Vote");

        // Export Epistemic types
        use crate::epistemic::{
            EmpiricalLevel, EpistemicClassification, EpistemicClassificationExtended,
            HarmonicLevel, MaterialityLevel, NormativeLevel,
        };

        EmpiricalLevel::export_all().expect("Failed to export EmpiricalLevel");
        NormativeLevel::export_all().expect("Failed to export NormativeLevel");
        MaterialityLevel::export_all().expect("Failed to export MaterialityLevel");
        HarmonicLevel::export_all().expect("Failed to export HarmonicLevel");
        EpistemicClassification::export_all().expect("Failed to export EpistemicClassification");
        EpistemicClassificationExtended::export_all()
            .expect("Failed to export EpistemicClassificationExtended");

        // Export DKG types
        use crate::dkg::{
            ConfidenceFactors, ConfidenceScore, ConfidenceThresholds, DKGConfig, EpistemicType,
            StoredTriple, TripleValue, VerifiableTriple, URI,
        };

        EpistemicType::export_all().expect("Failed to export EpistemicType");
        URI::export_all().expect("Failed to export URI");
        TripleValue::export_all().expect("Failed to export TripleValue");
        VerifiableTriple::export_all().expect("Failed to export VerifiableTriple");
        StoredTriple::export_all().expect("Failed to export StoredTriple");
        DKGConfig::export_all().expect("Failed to export DKGConfig");
        ConfidenceScore::export_all().expect("Failed to export ConfidenceScore");
        ConfidenceFactors::export_all().expect("Failed to export ConfidenceFactors");
        ConfidenceThresholds::export_all().expect("Failed to export ConfidenceThresholds");

        // Export Agentic types (MIP-E-004: Epistemic-Aware AI Agency)
        // Note: Only exporting self-contained API types to avoid dependency cascades
        use crate::agentic::{
            // API types (self-contained)
            ApiError,
            CalibrationSummary,
            EscalationResolutionResponse,
            EscalationSummary,
            // Gaming detection
            GamingAttackType,
            GamingResponse,
            KVectorValues,
            MoralActionGuidance,
            // GIS integration (Graceful Ignorance System)
            MoralUncertaintyType,
        };

        MoralUncertaintyType::export_all().expect("Failed to export MoralUncertaintyType");
        MoralActionGuidance::export_all().expect("Failed to export MoralActionGuidance");
        GamingAttackType::export_all().expect("Failed to export GamingAttackType");
        GamingResponse::export_all().expect("Failed to export GamingResponse");
        ApiError::export_all().expect("Failed to export ApiError");
        KVectorValues::export_all().expect("Failed to export KVectorValues");
        EscalationSummary::export_all().expect("Failed to export EscalationSummary");
        EscalationResolutionResponse::export_all()
            .expect("Failed to export EscalationResolutionResponse");
        CalibrationSummary::export_all().expect("Failed to export CalibrationSummary");

        // Print summary
        println!("\n=== TypeScript bindings exported successfully! ===\n");
        println!("Output directories:");
        println!("  - bindings/matl/");
        println!("  - bindings/epistemic/");
        println!("  - bindings/dkg/");
        println!("  - bindings/agentic/");
        println!();
        println!("Exported types:");
        println!("  MATL: KVector, KVectorWeights, KVectorDimension, GovernanceTier,");
        println!("        ProofOfGradientQuality, RoundState, VoteType, Vote");
        println!("  Epistemic: EmpiricalLevel, NormativeLevel, MaterialityLevel,");
        println!(
            "             HarmonicLevel, EpistemicClassification, EpistemicClassificationExtended"
        );
        println!("  DKG: EpistemicType, URI, TripleValue, VerifiableTriple, StoredTriple,");
        println!("       DKGConfig, ConfidenceScore, ConfidenceFactors, ConfidenceThresholds");
        println!("  Agentic: MoralUncertaintyType, MoralActionGuidance, GamingAttackType,");
        println!("           GamingResponse, ApiError, KVectorValues,");
        println!("           EscalationSummary, EscalationResolutionResponse, CalibrationSummary");
    }
}
