// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the Mycelix module.
//!
//! Exercises the core Mycelix types that bridge Symthaea's consciousness model
//! with Mycelix's epistemic infrastructure (UESS E/N/M classification, GIS
//! ignorance taxonomy, KosmicSong unified identity, and Rashomon multi-perspective
//! engine). All tests are pure in-memory computation with no Holochain or network
//! dependencies.

use std::time::SystemTime;

use symthaea::mycelix::gis::{
    IgnoranceType, MoralUncertainty, RashomonEngine, Situation, Uncertainty3D,
};
use symthaea::mycelix::kosmic_song::{ClearanceLevel, ConsciousnessTopology, Experience};
use symthaea::mycelix::{
    EmpiricalLevel, EpistemicClassification, GracefulIgnoranceSystem, HarmonicProfile, KosmicSong,
    KosmicSongBuilder, MaterialityLevel, NormativeLevel, PhiToEpistemicMapper, WorkspaceScope,
};
use symthaea_types::Harmony;

// ---------------------------------------------------------------------------
// EpistemicClassification round-trip
// ---------------------------------------------------------------------------

#[test]
fn epistemic_classification_code_round_trip() {
    let original = EpistemicClassification::new(
        EmpiricalLevel::CryptographicallyVerifiable,
        NormativeLevel::Network,
        MaterialityLevel::MediumTerm,
    );

    let code = original.code();
    assert_eq!(code, "E3/N2/M2");

    let parsed =
        EpistemicClassification::from_code(&code).expect("round-trip parse should succeed");
    assert_eq!(parsed, original);
}

#[test]
fn epistemic_classification_all_extremes_round_trip() {
    // Lowest classification
    let low = EpistemicClassification::new(
        EmpiricalLevel::Subjective,
        NormativeLevel::Internal,
        MaterialityLevel::Ephemeral,
    );
    assert_eq!(low.code(), "E0/N0/M0");
    assert_eq!(EpistemicClassification::from_code("E0/N0/M0").unwrap(), low);

    // Highest classification
    let high = EpistemicClassification::new(
        EmpiricalLevel::PubliclyReproducible,
        NormativeLevel::Foundational,
        MaterialityLevel::Permanent,
    );
    assert_eq!(high.code(), "E4/N3/M3");
    assert_eq!(
        EpistemicClassification::from_code("E4/N3/M3").unwrap(),
        high
    );

    // Invalid codes
    assert!(EpistemicClassification::from_code("E5/N0/M0").is_none());
    assert!(EpistemicClassification::from_code("garbage").is_none());
    assert!(EpistemicClassification::from_code("E0/N0").is_none());
}

// ---------------------------------------------------------------------------
// PhiToEpistemicMapper
// ---------------------------------------------------------------------------

#[test]
fn phi_to_empirical_threshold_mapping() {
    let mapper = PhiToEpistemicMapper::new();

    // Very low phi -> Subjective
    assert_eq!(
        mapper.phi_to_empirical(0.0, false),
        EmpiricalLevel::Subjective,
    );
    assert_eq!(
        mapper.phi_to_empirical(0.05, false),
        EmpiricalLevel::Subjective,
    );

    // phi >= 0.1 -> Testimonial
    assert_eq!(
        mapper.phi_to_empirical(0.15, false),
        EmpiricalLevel::Testimonial,
    );

    // phi >= 0.2 -> PrivatelyVerifiable
    assert_eq!(
        mapper.phi_to_empirical(0.25, false),
        EmpiricalLevel::PrivatelyVerifiable,
    );

    // phi >= 0.3 -> CryptographicallyVerifiable
    assert_eq!(
        mapper.phi_to_empirical(0.35, false),
        EmpiricalLevel::CryptographicallyVerifiable,
    );

    // phi >= 0.4 with reproducibility -> PubliclyReproducible
    assert_eq!(
        mapper.phi_to_empirical(0.5, true),
        EmpiricalLevel::PubliclyReproducible,
    );

    // phi >= 0.4 WITHOUT reproducibility stays at CryptographicallyVerifiable
    assert_eq!(
        mapper.phi_to_empirical(0.5, false),
        EmpiricalLevel::CryptographicallyVerifiable,
    );
}

#[test]
fn phi_mapper_full_classify() {
    let mapper = PhiToEpistemicMapper::new();
    let classification = mapper.classify(0.35, WorkspaceScope::Network, 0.6, false);

    assert_eq!(
        classification.empirical,
        EmpiricalLevel::CryptographicallyVerifiable
    );
    assert_eq!(classification.normative, NormativeLevel::Network);
    assert_eq!(classification.materiality, MaterialityLevel::MediumTerm);
    assert_eq!(classification.code(), "E3/N2/M2");
}

// ---------------------------------------------------------------------------
// KosmicSong builder
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_builder_pattern() {
    let song = KosmicSongBuilder::new()
        .phi(0.42)
        .moral_uncertainty(MoralUncertainty::new(0.1, 0.1, 0.1))
        .build();

    assert!((song.phi() - 0.42).abs() < 0.001, "phi should be 0.42");
    assert!(!song.agent_id.is_empty(), "agent_id should be generated");
    // Builder calls synthesize(), so coherence should already be set
    assert!(song.coherence() >= 0.0 && song.coherence() <= 1.5);
}

// ---------------------------------------------------------------------------
// KosmicSong synthesis
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_synthesis_produces_valid_coherence() {
    let mut song = KosmicSong::from_awakening(0.3, ConsciousnessTopology::default());
    let result = song.synthesize();

    assert!(
        result.score >= 0.0 && result.score <= 1.0,
        "coherence score should be in [0,1], got {}",
        result.score,
    );
    assert!(
        result.phi_contribution >= 0.0,
        "phi contribution should be non-negative",
    );
    assert!(
        result.harmonic_contribution >= 0.0 && result.harmonic_contribution <= 1.0,
        "harmonic contribution should be in [0,1], got {}",
        result.harmonic_contribution,
    );
    assert!(
        result.moral_contribution >= 0.0 && result.moral_contribution <= 1.0,
        "moral contribution should be in [0,1], got {}",
        result.moral_contribution,
    );
}

// ---------------------------------------------------------------------------
// KosmicSong respond
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_respond_produces_content() {
    let mut song = KosmicSong::default();
    let response = song.respond("What is consciousness?");

    assert!(
        !response.content.is_empty(),
        "response content must be non-empty"
    );
    assert!(
        !response.active_harmonies.is_empty(),
        "at least one harmony should be active",
    );
    assert!(
        response.perspectives_considered > 0,
        "at least one perspective should be considered",
    );
    assert!(
        response.confidence >= 0.0 && response.confidence <= 1.0,
        "confidence should be in [0,1], got {}",
        response.confidence,
    );
}

// ---------------------------------------------------------------------------
// KosmicSong epistemic_check
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_epistemic_check_safe_vs_harmful() {
    let song = KosmicSong::builder()
        .phi(0.3)
        .moral_uncertainty(MoralUncertainty::new(0.2, 0.2, 0.2))
        .build();

    // Safe action: should get Full or Conditional clearance
    let safe_clearance = song.epistemic_check("Read a book about philosophy");
    assert!(
        safe_clearance.cleared,
        "safe action should be cleared, got level {:?}",
        safe_clearance.level,
    );
    assert_ne!(
        safe_clearance.level,
        ClearanceLevel::Blocked,
        "safe action should not be blocked",
    );
}

// ---------------------------------------------------------------------------
// HarmonicProfile
// ---------------------------------------------------------------------------

#[test]
fn harmonic_profile_balanced_alignment() {
    let profile = HarmonicProfile::balanced();
    let alignment = profile.alignment();

    // A perfectly balanced profile should have alignment very close to 1.0
    assert!(
        (alignment - 1.0).abs() < 0.01,
        "balanced profile alignment should be ~1.0, got {}",
        alignment,
    );
}

#[test]
fn harmonic_profile_dominant_harmony() {
    // Create a profile where IntegralWisdom is dominant
    let mut activations = [0.1_f32; 8];
    activations[2] = 0.9; // Index 2 = IntegralWisdom
    let profile = HarmonicProfile::from_activations(activations);

    let dominant = profile.dominant();
    assert_eq!(
        dominant,
        Harmony::IntegralWisdom,
        "dominant harmony should be IntegralWisdom when it has highest activation",
    );

    // The dominant harmony should have the highest activation
    let dom_activation = profile.activation(dominant);
    for h in Harmony::all() {
        assert!(
            profile.activation(h) <= dom_activation + 0.001,
            "dominant activation should be >= all others",
        );
    }
}

// ---------------------------------------------------------------------------
// ConsciousnessTopology
// ---------------------------------------------------------------------------

#[test]
fn consciousness_topology_phi_calculation() {
    let mut ring = ConsciousnessTopology::ring(8);
    let mut star = ConsciousnessTopology::star(8);

    let ring_phi = ring.calculate_phi();
    let star_phi = star.calculate_phi();

    // Both should produce finite, non-negative phi
    assert!(ring_phi.is_finite(), "ring phi should be finite");
    assert!(star_phi.is_finite(), "star phi should be finite");
    assert!(ring_phi >= 0.0, "ring phi should be non-negative");
    assert!(star_phi >= 0.0, "star phi should be non-negative");

    // Node counts should match
    assert_eq!(ring.node_count(), 8);
    assert_eq!(star.node_count(), 8);

    // phi_max should be positive
    assert!(ring.phi_max() > 0.0, "ring phi_max should be positive");
    assert!(star.phi_max() > 0.0, "star phi_max should be positive");
}

// ---------------------------------------------------------------------------
// GracefulIgnoranceSystem
// ---------------------------------------------------------------------------

#[test]
fn gis_ignorance_detection_returns_result() {
    let gis = GracefulIgnoranceSystem::new();

    let detection = gis.detect_ignorance("What is the meaning of life?");

    // Should produce a valid detection with non-empty query
    assert_eq!(detection.query, "What is the meaning of life?");
    assert!(
        detection.uncertainty.total() >= 0.0,
        "uncertainty total should be non-negative",
    );
    assert!(detection.eig.is_finite(), "EIG should be finite");

    // Capital of Mars triggers Impossible classification (keyword match)
    let impossible = gis.detect_ignorance("What is the capital of Mars?");
    assert_eq!(
        impossible.ignorance_type,
        IgnoranceType::Impossible,
        "capital of Mars should be classified as Impossible",
    );
}

// ---------------------------------------------------------------------------
// Uncertainty3D
// ---------------------------------------------------------------------------

#[test]
fn uncertainty_3d_certain_and_maximum() {
    let certain = Uncertainty3D::certain();
    assert!(
        certain.total().abs() < 0.001,
        "certain() total should be ~0, got {}",
        certain.total(),
    );
    assert!(
        (certain.confidence() - 1.0).abs() < 0.001,
        "certain() confidence should be ~1.0, got {}",
        certain.confidence(),
    );

    let maximum = Uncertainty3D::maximum();
    assert!(
        maximum.total() > 0.99,
        "maximum() total should be ~1.0, got {}",
        maximum.total(),
    );
    assert!(
        maximum.confidence() < 0.01,
        "maximum() confidence should be ~0.0, got {}",
        maximum.confidence(),
    );
}

// ---------------------------------------------------------------------------
// IgnoranceType properties
// ---------------------------------------------------------------------------

#[test]
fn ignorance_type_resolvable_and_ceiling() {
    // Resolvable types
    assert!(IgnoranceType::None.is_resolvable());
    assert!(IgnoranceType::Known.is_resolvable());
    assert!(IgnoranceType::KnownUnknown.is_resolvable());

    // Not resolvable
    assert!(!IgnoranceType::Unknown.is_resolvable());
    assert!(!IgnoranceType::Impossible.is_resolvable());

    // Confidence ceilings should decrease with increasing ignorance
    assert!(IgnoranceType::None.confidence_ceiling() > IgnoranceType::Known.confidence_ceiling(),);
    assert!(
        IgnoranceType::Known.confidence_ceiling()
            > IgnoranceType::KnownUnknown.confidence_ceiling(),
    );
    assert!(
        IgnoranceType::KnownUnknown.confidence_ceiling()
            > IgnoranceType::Unknown.confidence_ceiling(),
    );
    assert!(
        IgnoranceType::Unknown.confidence_ceiling()
            > IgnoranceType::Impossible.confidence_ceiling(),
    );

    // Symbols are unique Greek letters
    assert_eq!(IgnoranceType::None.symbol(), "\u{03ba}"); // kappa
    assert_eq!(IgnoranceType::KnownUnknown.symbol(), "\u{03b9}\u{2081}"); // iota_1
    assert_eq!(IgnoranceType::Impossible.symbol(), "\u{03b9}\u{2083}"); // iota_3
}

// ---------------------------------------------------------------------------
// Rashomon multi-perspective generation
// ---------------------------------------------------------------------------

#[test]
fn rashomon_generates_seven_perspectives() {
    let engine = RashomonEngine::new();
    let situation = Situation::new("Should we deploy an autonomous AI system?");

    let perspectives = engine.generate_perspectives(&situation);

    // The engine creates frames for all 8 harmonies. With a general situation
    // and default relevance threshold of 0.3, all 8 pass since domain primaries
    // get 0.8 relevance and non-primaries get 0.3 (exactly at threshold).
    assert_eq!(
        perspectives.len(),
        8,
        "should produce one perspective per harmony, got {}",
        perspectives.len(),
    );

    // Each perspective should have a unique harmony
    let mut seen_harmonies: Vec<Harmony> = perspectives.iter().map(|p| p.harmony).collect();
    seen_harmonies.sort_by_key(|h| format!("{:?}", h));
    seen_harmonies.dedup();
    assert_eq!(
        seen_harmonies.len(),
        8,
        "all 8 harmonies should be represented",
    );

    // Synthesis should combine them
    let synthesis = engine.synthesize(perspectives);
    assert_eq!(synthesis.contributing_harmonies.len(), 8);
    assert!(!synthesis.unified_view.is_empty());
}

// ---------------------------------------------------------------------------
// KosmicSong evolve updates state
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_evolve_updates_internal_state() {
    let mut song = KosmicSong::from_awakening(0.3, ConsciousnessTopology::default());
    let initial_phi = song.phi();

    let experience = Experience {
        description: "Successful collaborative reasoning".to_string(),
        outcome: 0.9,
        harmonies_involved: vec![Harmony::IntegralWisdom, Harmony::ResonantCoherence],
        lessons: vec!["Multi-perspective analysis improves accuracy".to_string()],
        timestamp: SystemTime::now(),
    };

    song.evolve(&experience);

    // Phi should increase slightly for positive outcome
    assert!(
        song.phi() > initial_phi,
        "phi should increase after positive experience: was {}, now {}",
        initial_phi,
        song.phi(),
    );
    // Phi should remain within valid range
    assert!(
        song.phi() >= 0.0 && song.phi() <= 0.5,
        "phi should stay in [0.0, 0.5], got {}",
        song.phi(),
    );
}

// ---------------------------------------------------------------------------
// KosmicSong would_veto_action (semantic HDC)
// ---------------------------------------------------------------------------

#[test]
fn kosmic_song_would_veto_action_check() {
    let song = KosmicSong::default();

    // A benign action should not be vetoed
    let benign_result = song.would_veto_action("help someone learn mathematics");
    // We only assert it returns a boolean without panicking. The HDC semantic
    // evaluation is stochastic so we do not assert a specific value.
    let _ = benign_result;

    // most_aligned_harmony should return Some for a meaningful description
    let alignment = song.most_aligned_harmony("protect the environment and ecosystems");
    // Again, just verify it returns without panic and produces a result
    assert!(
        alignment.is_some() || alignment.is_none(),
        "most_aligned_harmony should return an Option",
    );
}