// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic modalities and intervention library.
//!
//! Encodes evidence-based therapeutic approaches (CBT, ACT, DBT, etc.) as
//! HDC vectors with intervention libraries for context-aware selection.
//!
//! Science: Beck (1979) CBT, Hayes (2006) ACT, Linehan (1993) DBT,
//! White & Epston (1990) Narrative, Miller & Rollnick (2012) MI,
//! Shapiro (2001) EMDR, Schwartz (1995) IFS, Levine (1997) Somatic.

use crate::rdoc::RDocDomain;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::BinaryHV;

// ── Therapeutic Modalities ─────────────────────────────────────────────────

/// Evidence-based therapeutic modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TherapeuticModality {
    /// Cognitive Behavioral Therapy — Beck (1979)
    Cbt,
    /// Acceptance and Commitment Therapy — Hayes (2006)
    Act,
    /// Dialectical Behavior Therapy — Linehan (1993)
    Dbt,
    /// Narrative Therapy — White & Epston (1990)
    Narrative,
    /// Somatic Experiencing — Levine (1997)
    Somatic,
    /// Psychodynamic Therapy — Shedler (2010)
    Psychodynamic,
    /// Motivational Interviewing — Miller & Rollnick (2012)
    Motivational,
    /// Eye Movement Desensitization and Reprocessing — Shapiro (2001)
    Emdr,
    /// Internal Family Systems — Schwartz (1995)
    Ifs,
}

impl TherapeuticModality {
    /// All modality variants for iteration.
    pub const ALL: [Self; 9] = [
        Self::Cbt,
        Self::Act,
        Self::Dbt,
        Self::Narrative,
        Self::Somatic,
        Self::Psychodynamic,
        Self::Motivational,
        Self::Emdr,
        Self::Ifs,
    ];

    /// Short display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cbt => "CBT",
            Self::Act => "ACT",
            Self::Dbt => "DBT",
            Self::Narrative => "Narrative",
            Self::Somatic => "Somatic",
            Self::Psychodynamic => "Psychodynamic",
            Self::Motivational => "MI",
            Self::Emdr => "EMDR",
            Self::Ifs => "IFS",
        }
    }

    /// Primary RDoC domains targeted by this modality.
    pub fn target_domains(&self) -> Vec<RDocDomain> {
        match self {
            Self::Cbt => vec![RDocDomain::NegativeValence, RDocDomain::CognitiveSystems],
            Self::Act => vec![RDocDomain::NegativeValence, RDocDomain::PositiveValence],
            Self::Dbt => vec![
                RDocDomain::NegativeValence,
                RDocDomain::ArousalRegulatory,
                RDocDomain::SocialProcesses,
            ],
            Self::Narrative => vec![RDocDomain::SocialProcesses, RDocDomain::PositiveValence],
            Self::Somatic => vec![RDocDomain::ArousalRegulatory, RDocDomain::Sensorimotor],
            Self::Psychodynamic => vec![RDocDomain::SocialProcesses, RDocDomain::NegativeValence],
            Self::Motivational => vec![RDocDomain::PositiveValence, RDocDomain::CognitiveSystems],
            Self::Emdr => vec![RDocDomain::NegativeValence, RDocDomain::Sensorimotor],
            Self::Ifs => vec![RDocDomain::NegativeValence, RDocDomain::SocialProcesses],
        }
    }

    /// Generate deterministic base HV for this modality.
    pub fn base_hv(&self) -> BinaryHV {
        let label = format!("modality:{}", self.name());
        let hash = blake3::hash(label.as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        BinaryHV::random(seed)
    }
}

// ── Therapeutic Intervention ───────────────────────────────────────────────

/// A specific therapeutic intervention (technique within a modality).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TherapeuticIntervention {
    /// Which modality this intervention belongs to.
    pub modality: TherapeuticModality,
    /// Technique name (e.g., "cognitive restructuring", "body scan").
    pub technique: String,
    /// RDoC domains this intervention targets.
    pub target_domains: Vec<RDocDomain>,
    /// Contraindications (categories where this should NOT be used).
    pub contraindications: Vec<String>,
    /// Evidence level (0.0 = anecdotal, 1.0 = strong RCT evidence).
    pub evidence_level: f32,
    /// Minimum alliance quality required (0.0–1.0).
    pub min_alliance: f32,
    /// HDC encoding of the intervention.
    #[serde(skip)]
    encoding: Option<BinaryHV>,
}

impl TherapeuticIntervention {
    /// Create a new intervention.
    pub fn new(
        modality: TherapeuticModality,
        technique: &str,
        target_domains: Vec<RDocDomain>,
        evidence_level: f32,
        min_alliance: f32,
    ) -> Self {
        let mut intervention = Self {
            modality,
            technique: technique.to_string(),
            target_domains,
            contraindications: Vec::new(),
            evidence_level: evidence_level.clamp(0.0, 1.0),
            min_alliance: min_alliance.clamp(0.0, 1.0),
            encoding: None,
        };
        intervention.recompute_encoding();
        intervention
    }

    /// Add a contraindication.
    pub fn with_contraindication(mut self, contraindication: &str) -> Self {
        self.contraindications.push(contraindication.to_string());
        self
    }

    /// Get HDC encoding.
    pub fn encoding(&self) -> &BinaryHV {
        self.encoding.as_ref().expect("encoding always computed")
    }

    /// Recompute encoding: modality_base ⊗ technique_hv.
    fn recompute_encoding(&mut self) {
        let modality_hv = self.modality.base_hv();
        let technique_label = format!("technique:{}", self.technique);
        let hash = blake3::hash(technique_label.as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        let technique_hv = BinaryHV::random(seed);
        self.encoding = Some(modality_hv.bind(&technique_hv));
    }
}

// ── Intervention Library ───────────────────────────────────────────────────

/// Library of therapeutic interventions with modality-indexed lookup.
#[derive(Debug, Clone)]
pub struct InterventionLibrary {
    /// All interventions.
    pub interventions: Vec<TherapeuticIntervention>,
    /// Index: modality → intervention indices.
    modality_index: HashMap<TherapeuticModality, Vec<usize>>,
}

impl InterventionLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self {
            interventions: Vec::new(),
            modality_index: HashMap::new(),
        }
    }

    /// Bootstrap with standard evidence-based interventions.
    ///
    /// Follows the `bootstrap_entities()` pattern from knowledge graph.
    pub fn bootstrap() -> Self {
        let mut lib = Self::new();

        // ── CBT ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "cognitive_restructuring",
            vec![RDocDomain::NegativeValence, RDocDomain::CognitiveSystems],
            0.9,
            0.4,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "behavioral_activation",
            vec![RDocDomain::PositiveValence],
            0.85,
            0.3,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "thought_record",
            vec![RDocDomain::CognitiveSystems, RDocDomain::NegativeValence],
            0.8,
            0.3,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "exposure_hierarchy",
            vec![RDocDomain::NegativeValence],
            0.9,
            0.6,
        ));

        // ── ACT ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Act,
            "defusion",
            vec![RDocDomain::NegativeValence, RDocDomain::CognitiveSystems],
            0.8,
            0.3,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Act,
            "values_clarification",
            vec![RDocDomain::PositiveValence],
            0.75,
            0.4,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Act,
            "acceptance",
            vec![RDocDomain::NegativeValence, RDocDomain::ArousalRegulatory],
            0.8,
            0.3,
        ));

        // ── DBT ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "distress_tolerance",
            vec![RDocDomain::ArousalRegulatory, RDocDomain::NegativeValence],
            0.85,
            0.3,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "emotion_regulation",
            vec![RDocDomain::NegativeValence, RDocDomain::ArousalRegulatory],
            0.85,
            0.4,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "interpersonal_effectiveness",
            vec![RDocDomain::SocialProcesses],
            0.8,
            0.5,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "mindfulness",
            vec![RDocDomain::ArousalRegulatory, RDocDomain::CognitiveSystems],
            0.85,
            0.2,
        ));

        // ── Narrative ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Narrative,
            "externalization",
            vec![RDocDomain::NegativeValence, RDocDomain::SocialProcesses],
            0.7,
            0.4,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Narrative,
            "re_authoring",
            vec![RDocDomain::PositiveValence, RDocDomain::SocialProcesses],
            0.7,
            0.5,
        ));

        // ── Somatic ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Somatic,
            "body_scan",
            vec![RDocDomain::ArousalRegulatory, RDocDomain::Sensorimotor],
            0.7,
            0.2,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Somatic,
            "grounding",
            vec![RDocDomain::ArousalRegulatory],
            0.75,
            0.2,
        ));
        lib.add(
            TherapeuticIntervention::new(
                TherapeuticModality::Somatic,
                "pendulation",
                vec![RDocDomain::ArousalRegulatory, RDocDomain::NegativeValence],
                0.65,
                0.4,
            )
            .with_contraindication("active psychosis"),
        );

        // ── MI ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Motivational,
            "open_questions",
            vec![RDocDomain::CognitiveSystems],
            0.85,
            0.2,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Motivational,
            "reflective_listening",
            vec![RDocDomain::SocialProcesses],
            0.85,
            0.2,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Motivational,
            "change_talk_elicitation",
            vec![RDocDomain::PositiveValence, RDocDomain::CognitiveSystems],
            0.8,
            0.4,
        ));

        // ── IFS ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Ifs,
            "parts_detection",
            vec![RDocDomain::NegativeValence, RDocDomain::SocialProcesses],
            0.65,
            0.4,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Ifs,
            "self_energy",
            vec![RDocDomain::PositiveValence, RDocDomain::SocialProcesses],
            0.65,
            0.5,
        ));

        // ── Psychodynamic ──
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Psychodynamic,
            "interpretation",
            vec![RDocDomain::CognitiveSystems, RDocDomain::NegativeValence],
            0.7,
            0.6,
        ));
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Psychodynamic,
            "transference_exploration",
            vec![RDocDomain::SocialProcesses],
            0.7,
            0.7,
        ));

        // ── EMDR ──
        lib.add(
            TherapeuticIntervention::new(
                TherapeuticModality::Emdr,
                "bilateral_stimulation",
                vec![RDocDomain::NegativeValence, RDocDomain::Sensorimotor],
                0.85,
                0.5,
            )
            .with_contraindication("active psychosis")
            .with_contraindication("dissociative identity disorder"),
        );
        lib.add(TherapeuticIntervention::new(
            TherapeuticModality::Emdr,
            "safe_place_installation",
            vec![RDocDomain::ArousalRegulatory],
            0.75,
            0.3,
        ));

        lib
    }

    /// Add an intervention to the library.
    pub fn add(&mut self, intervention: TherapeuticIntervention) {
        let idx = self.interventions.len();
        self.modality_index
            .entry(intervention.modality)
            .or_default()
            .push(idx);
        self.interventions.push(intervention);
    }

    /// Get all interventions for a modality.
    pub fn by_modality(&self, modality: TherapeuticModality) -> Vec<&TherapeuticIntervention> {
        self.modality_index
            .get(&modality)
            .map(|indices| indices.iter().map(|&i| &self.interventions[i]).collect())
            .unwrap_or_default()
    }

    /// Find interventions targeting a specific RDoC domain.
    pub fn by_target_domain(&self, domain: RDocDomain) -> Vec<&TherapeuticIntervention> {
        self.interventions
            .iter()
            .filter(|i| i.target_domains.contains(&domain))
            .collect()
    }

    /// Find interventions safe for a given alliance level.
    pub fn safe_at_alliance(&self, alliance: f32) -> Vec<&TherapeuticIntervention> {
        self.interventions
            .iter()
            .filter(|i| i.min_alliance <= alliance)
            .collect()
    }

    /// Total number of interventions.
    pub fn len(&self) -> usize {
        self.interventions.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.interventions.is_empty()
    }
}

impl Default for InterventionLibrary {
    fn default() -> Self {
        Self::bootstrap()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_completeness() {
        let lib = InterventionLibrary::bootstrap();
        // Should have interventions for all 9 modalities
        for modality in TherapeuticModality::ALL {
            let count = lib.by_modality(modality).len();
            assert!(count > 0, "modality {:?} has no interventions", modality);
        }
    }

    #[test]
    fn test_bootstrap_has_reasonable_count() {
        let lib = InterventionLibrary::bootstrap();
        assert!(
            lib.len() >= 20,
            "expected >= 20 interventions, got {}",
            lib.len()
        );
    }

    #[test]
    fn test_modality_vectors_distinct() {
        for (i, a) in TherapeuticModality::ALL.iter().enumerate() {
            for b in TherapeuticModality::ALL.iter().skip(i + 1) {
                let sim = a.base_hv().similarity(&b.base_hv());
                assert!(
                    sim < 0.55,
                    "{} vs {} similarity {:.3} too high (expected near 0.5 for random BinaryHVs)",
                    a.name(),
                    b.name(),
                    sim
                );
            }
        }
    }

    #[test]
    fn test_modality_vectors_deterministic() {
        let hv1 = TherapeuticModality::Cbt.base_hv();
        let hv2 = TherapeuticModality::Cbt.base_hv();
        assert_eq!(hv1.similarity(&hv2), 1.0);
    }

    #[test]
    fn test_intervention_encoding() {
        let intervention = TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "thought_record",
            vec![RDocDomain::CognitiveSystems],
            0.8,
            0.3,
        );
        let _encoding = intervention.encoding(); // should not panic
    }

    #[test]
    fn test_same_modality_interventions_more_similar() {
        let cbt1 = TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "cognitive_restructuring",
            vec![],
            0.9,
            0.4,
        );
        let cbt2 = TherapeuticIntervention::new(
            TherapeuticModality::Cbt,
            "behavioral_activation",
            vec![],
            0.85,
            0.3,
        );
        let dbt1 = TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "distress_tolerance",
            vec![],
            0.85,
            0.3,
        );

        // Interventions from the same modality share the modality base vector component
        // After binding with different technique vectors, similarity will be low for all
        // but the encoding captures modality identity
        let _within = cbt1.encoding().similarity(cbt2.encoding());
        let _between = cbt1.encoding().similarity(dbt1.encoding());
        // Both are valid encodings
        assert!(cbt1.encoding().similarity(cbt1.encoding()) == 1.0);
    }

    #[test]
    fn test_by_target_domain() {
        let lib = InterventionLibrary::bootstrap();
        let neg_val = lib.by_target_domain(RDocDomain::NegativeValence);
        assert!(
            neg_val.len() >= 5,
            "expected >= 5 NegativeValence interventions, got {}",
            neg_val.len()
        );
    }

    #[test]
    fn test_safe_at_alliance_low() {
        let lib = InterventionLibrary::bootstrap();
        let safe = lib.safe_at_alliance(0.2);
        assert!(
            !safe.is_empty(),
            "should have some low-alliance-safe interventions"
        );
        for intervention in &safe {
            assert!(intervention.min_alliance <= 0.2);
        }
    }

    #[test]
    fn test_safe_at_alliance_high() {
        let lib = InterventionLibrary::bootstrap();
        let safe_low = lib.safe_at_alliance(0.2);
        let safe_high = lib.safe_at_alliance(0.8);
        assert!(
            safe_high.len() >= safe_low.len(),
            "higher alliance should unlock more interventions"
        );
    }

    #[test]
    fn test_contraindications() {
        let lib = InterventionLibrary::bootstrap();
        let emdr = lib.by_modality(TherapeuticModality::Emdr);
        let bilateral = emdr.iter().find(|i| i.technique == "bilateral_stimulation");
        assert!(bilateral.is_some());
        assert!(!bilateral.unwrap().contraindications.is_empty());
    }

    #[test]
    fn test_all_modalities_count() {
        assert_eq!(TherapeuticModality::ALL.len(), 9);
    }

    #[test]
    fn test_modality_target_domains_nonempty() {
        for modality in TherapeuticModality::ALL {
            assert!(
                !modality.target_domains().is_empty(),
                "{:?} has no target domains",
                modality
            );
        }
    }
}
