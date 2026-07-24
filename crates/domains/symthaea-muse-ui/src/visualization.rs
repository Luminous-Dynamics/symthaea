// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed capability contract for Listen/Research visualizations.
//!
//! A renderer declares the evidence it needs rather than probing a bag of
//! optional JSON values. This keeps honest empty states and makes neutral
//! piano reductions distinct from missing orchestration evidence.

use symthaea_muse_protocol::{
    AnalystPieceBundle, ListenCompositionBundle, ListenPerformanceBundle,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualizationKind {
    Hybrid,
    Form,
    Score,
    Compare,
    Resonance,
    Motifs,
    Harmony,
    Orchestration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceRequirement {
    Composition,
    Performance,
    Analyst,
    Motifs,
    Harmony,
    Orchestration,
    Resonance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualizationContract {
    pub kind: VisualizationKind,
    pub required: &'static [EvidenceRequirement],
    pub supports_seek: bool,
    pub follows_current_moment: bool,
    pub supports_reduced_motion: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualizationAvailability {
    pub available: bool,
    pub reason: Option<&'static str>,
}

pub struct VisualizationEvidence<'a> {
    pub composition: Option<&'a ListenCompositionBundle>,
    pub performance: Option<&'a ListenPerformanceBundle>,
    pub analyst: Option<&'a AnalystPieceBundle>,
}

impl VisualizationContract {
    pub fn availability(self, evidence: &VisualizationEvidence<'_>) -> VisualizationAvailability {
        for requirement in self.required {
            let present = match requirement {
                EvidenceRequirement::Composition => evidence.composition.is_some(),
                EvidenceRequirement::Performance => evidence.performance.is_some(),
                EvidenceRequirement::Analyst => evidence.analyst.is_some(),
                EvidenceRequirement::Motifs => evidence
                    .composition
                    .is_some_and(|bundle| !bundle.motif_occurrences.is_empty()),
                EvidenceRequirement::Harmony => evidence
                    .composition
                    .is_some_and(|bundle| !bundle.sonorities.is_empty()),
                EvidenceRequirement::Orchestration => evidence
                    .composition
                    .is_some_and(|bundle| !bundle.orchestration.is_empty()),
                EvidenceRequirement::Resonance => evidence
                    .composition
                    .is_some_and(|bundle| bundle.resonance.is_some()),
            };
            if !present {
                return VisualizationAvailability {
                    available: false,
                    reason: Some(match requirement {
                        EvidenceRequirement::Composition => "composition evidence is unavailable",
                        EvidenceRequirement::Performance => {
                            "performed-note evidence is unavailable"
                        }
                        EvidenceRequirement::Analyst => "verified Analyst evidence is unavailable",
                        EvidenceRequirement::Motifs => {
                            "this piece has no emitted motif occurrences"
                        }
                        EvidenceRequirement::Harmony => {
                            "this piece has no harmonic-region evidence"
                        }
                        EvidenceRequirement::Orchestration => {
                            "this rendition has no orchestration allocation evidence"
                        }
                        EvidenceRequirement::Resonance => {
                            "this piece has no structural-resonance curve"
                        }
                    }),
                };
            }
        }
        VisualizationAvailability {
            available: true,
            reason: None,
        }
    }
}

pub const FORM: VisualizationContract = VisualizationContract {
    kind: VisualizationKind::Form,
    required: &[EvidenceRequirement::Composition],
    supports_seek: true,
    follows_current_moment: true,
    supports_reduced_motion: true,
};

pub const COMPARE: VisualizationContract = VisualizationContract {
    kind: VisualizationKind::Compare,
    required: &[
        EvidenceRequirement::Composition,
        EvidenceRequirement::Performance,
    ],
    supports_seek: true,
    follows_current_moment: true,
    supports_reduced_motion: true,
};

pub const MOTIFS: VisualizationContract = VisualizationContract {
    kind: VisualizationKind::Motifs,
    required: &[
        EvidenceRequirement::Composition,
        EvidenceRequirement::Motifs,
    ],
    supports_seek: true,
    follows_current_moment: true,
    supports_reduced_motion: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_evidence_produces_an_honest_reason() {
        let evidence = VisualizationEvidence {
            composition: None,
            performance: None,
            analyst: None,
        };
        let result = COMPARE.availability(&evidence);
        assert!(!result.available);
        assert_eq!(result.reason, Some("composition evidence is unavailable"));
    }
}
