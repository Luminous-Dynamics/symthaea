// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Collective Identity — Community-Level KosmicSong
//!
//! Aggregates individual KosmicCredentials into a community-level identity,
//! enabling harmonic routing of proposals to aligned agents and detection
//! of the community's collective mode.
//!
//! ## Community Modes
//!
//! Based on the dominant harmonies across all agents:
//! - **Exploratory**: high Infinite Play + curiosity → community is experimenting
//! - **Protective**: high Pan-Sentient Care + Mutual Reciprocity → community is nurturing
//! - **Creative**: high Infinite Play + Evolutionary Progression → community is building
//! - **Reflective**: high Sacred Stillness + Integral Wisdom → community is integrating

use super::gis::Harmony;
use super::kosmic_song::KosmicCredential;

/// The collective identity of a community of agents.
#[derive(Debug, Clone)]
pub struct CollectiveKosmicSong {
    /// The dominant harmony across all agents.
    pub dominant_harmony: Harmony,
    /// Mean coherence across all agents (0.0–1.0).
    pub coherence: f64,
    /// The community's current operational mode.
    pub community_mode: CommunityMode,
    /// Number of agents contributing.
    pub n_agents: usize,
    /// Mean maturation signal across agents (0.0–1.0).
    pub mean_maturation: f64,
}

/// Community operational modes derived from collective harmonic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityMode {
    /// High Infinite Play + curiosity → experimentation and exploration.
    Exploratory,
    /// High Pan-Sentient Care + Mutual Reciprocity → nurturing and protection.
    Protective,
    /// High Infinite Play + Evolutionary Progression → building and creation.
    Creative,
    /// High Sacred Stillness + Integral Wisdom → integration and reflection.
    Reflective,
}

impl CommunityMode {
    /// Static string for telemetry (avoids `format!("{:?}")`).
    pub fn as_str(&self) -> &'static str {
        match self {
            CommunityMode::Exploratory => "Exploratory",
            CommunityMode::Protective => "Protective",
            CommunityMode::Creative => "Creative",
            CommunityMode::Reflective => "Reflective",
        }
    }
}

impl CollectiveKosmicSong {
    /// Aggregate individual credentials into a collective identity.
    pub fn from_credentials(credentials: &[KosmicCredential]) -> Self {
        if credentials.is_empty() {
            return Self {
                dominant_harmony: Harmony::ResonantCoherence,
                coherence: 0.0,
                community_mode: CommunityMode::Reflective,
                n_agents: 0,
                mean_maturation: 0.0,
            };
        }

        let n = credentials.len();

        // Mean coherence
        let coherence = credentials
            .iter()
            .map(|c| c.coherence_score as f64)
            .sum::<f64>()
            / n as f64;

        // Mean maturation
        let mean_maturation = credentials
            .iter()
            .map(|c| c.maturation_signal as f64)
            .sum::<f64>()
            / n as f64;

        // Count dominant harmonies across agents
        let mut harmony_counts = [0u32; 8];
        for cred in credentials {
            let idx = harmony_to_index(cred.dominant_harmony);
            harmony_counts[idx] += 1;
        }

        let (dominant_idx, _) = harmony_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap_or((0, &0));

        let dominant_harmony = index_to_harmony(dominant_idx);
        let community_mode = detect_community_mode(&harmony_counts);

        Self {
            dominant_harmony,
            coherence,
            community_mode,
            n_agents: n,
            mean_maturation,
        }
    }
}

/// Detect community mode from harmony distribution.
fn detect_community_mode(counts: &[u32; 8]) -> CommunityMode {
    // Index mapping (from harmony_to_index, matches symthaea_types declaration order):
    // 0: ResonantCoherence, 1: PanSentientFlourishing, 2: IntegralWisdom
    // 3: InfinitePlay, 4: UniversalInterconnectedness, 5: SacredReciprocity
    // 6: EvolutionaryProgression, 7: SacredStillness

    let play = counts[3]; // InfinitePlay
    let care = counts[1]; // PanSentientFlourishing
    let reciprocity = counts[5]; // SacredReciprocity
    let evolution = counts[6]; // EvolutionaryProgression
    let stillness = counts[7]; // SacredStillness
    let wisdom = counts[2]; // IntegralWisdom

    let exploratory_score = play + counts[4]; // play + interconnectedness
    let protective_score = care + reciprocity;
    let creative_score = play + evolution;
    let reflective_score = stillness + wisdom;

    let max_score = exploratory_score
        .max(protective_score)
        .max(creative_score)
        .max(reflective_score);

    if max_score == 0 {
        return CommunityMode::Reflective; // default
    }

    // Tie-break: prefer protective > reflective > creative > exploratory
    if protective_score == max_score {
        CommunityMode::Protective
    } else if reflective_score == max_score {
        CommunityMode::Reflective
    } else if creative_score == max_score {
        CommunityMode::Creative
    } else {
        CommunityMode::Exploratory
    }
}

/// Map Harmony enum to index (matches symthaea_types declaration order).
fn harmony_to_index(h: Harmony) -> usize {
    match h {
        Harmony::ResonantCoherence => 0,
        Harmony::PanSentientFlourishing => 1,
        Harmony::IntegralWisdom => 2,
        Harmony::InfinitePlay => 3,
        Harmony::UniversalInterconnectedness => 4,
        Harmony::SacredReciprocity => 5,
        Harmony::EvolutionaryProgression => 6,
        Harmony::SacredStillness => 7,
    }
}

/// Map index back to Harmony enum.
fn index_to_harmony(idx: usize) -> Harmony {
    match idx {
        0 => Harmony::ResonantCoherence,
        1 => Harmony::PanSentientFlourishing,
        2 => Harmony::IntegralWisdom,
        3 => Harmony::InfinitePlay,
        4 => Harmony::UniversalInterconnectedness,
        5 => Harmony::SacredReciprocity,
        6 => Harmony::EvolutionaryProgression,
        7 => Harmony::SacredStillness,
        _ => Harmony::ResonantCoherence,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelix::gis::{IgnoranceType, MoralUncertainty};

    fn cred(harmony: Harmony, coherence: f32, maturation: f32) -> KosmicCredential {
        KosmicCredential {
            agent_id: "test".to_string(),
            dominant_harmony: harmony,
            coherence_score: coherence,
            moral_uncertainty: MoralUncertainty::default(),
            gis_type: IgnoranceType::None,
            maturation_signal: maturation,
        }
    }

    #[test]
    fn test_empty_credentials() {
        let song = CollectiveKosmicSong::from_credentials(&[]);
        assert_eq!(song.n_agents, 0);
        assert!((song.coherence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_credential_generation() {
        let credentials = vec![
            cred(Harmony::IntegralWisdom, 0.8, 0.9),
            cred(Harmony::IntegralWisdom, 0.7, 0.8),
            cred(Harmony::PanSentientFlourishing, 0.6, 0.5),
        ];
        let song = CollectiveKosmicSong::from_credentials(&credentials);

        assert_eq!(song.n_agents, 3);
        assert_eq!(song.dominant_harmony, Harmony::IntegralWisdom);
        assert!((song.coherence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_harmonic_affinity() {
        let mut ks = crate::mycelix::KosmicSong::default();
        ks.harmonic_profile
            .set_activation(Harmony::PanSentientFlourishing, 0.9);

        let affinity = ks.harmonic_affinity(&[Harmony::PanSentientFlourishing]);
        assert!(
            affinity > 0.5,
            "Care-dominant should match care proposal: {}",
            affinity
        );

        let low_affinity = ks.harmonic_affinity(&[Harmony::InfinitePlay]);
        assert!(
            low_affinity < affinity,
            "Non-dominant harmony should have lower affinity"
        );
    }

    #[test]
    fn test_community_mode_protective() {
        let credentials = vec![
            cred(Harmony::PanSentientFlourishing, 0.8, 0.5),
            cred(Harmony::PanSentientFlourishing, 0.7, 0.6),
            cred(Harmony::SacredReciprocity, 0.8, 0.7),
        ];
        let song = CollectiveKosmicSong::from_credentials(&credentials);
        assert_eq!(song.community_mode, CommunityMode::Protective);
    }

    #[test]
    fn test_community_mode_exploratory() {
        let credentials = vec![
            cred(Harmony::InfinitePlay, 0.8, 0.5),
            cred(Harmony::InfinitePlay, 0.7, 0.6),
            cred(Harmony::UniversalInterconnectedness, 0.8, 0.7),
        ];
        let song = CollectiveKosmicSong::from_credentials(&credentials);
        assert_eq!(song.community_mode, CommunityMode::Exploratory);
    }

    #[test]
    fn test_community_mode_reflective() {
        let credentials = vec![
            cred(Harmony::SacredStillness, 0.8, 0.9),
            cred(Harmony::IntegralWisdom, 0.7, 0.8),
            cred(Harmony::SacredStillness, 0.6, 0.7),
        ];
        let song = CollectiveKosmicSong::from_credentials(&credentials);
        assert_eq!(song.community_mode, CommunityMode::Reflective);
    }

    #[test]
    fn test_epistemic_check_via_kosmic_song() {
        let ks = crate::mycelix::KosmicSong::default();
        let cred = ks.governance_credential();
        assert!(!cred.agent_id.is_empty());
        assert!(cred.coherence_score >= 0.0);
    }
}
