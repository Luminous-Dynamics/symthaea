// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Studio-facing Preserve/Change contracts derived from cognitive proposals.
//!
//! Symthaea proposes *why* a change matters; the Studio needs an explicit,
//! reviewable contract describing what may change and what must remain fixed.
//! This module performs that translation without generating notes.

use crate::cognitive_bridge::{
    ActionScope, CognitiveDecisionTrace, CognitiveObligationDemand, CognitiveSection,
    PreserveInvariant, SymbolicAction,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// User-visible musical dimensions that can be preserved or changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StudioDimension {
    MotifIdentity,
    MotifDevelopment,
    ThematicPlacement,
    Melody,
    Harmony,
    TonalCenter,
    Rhythm,
    Meter,
    FormLength,
    ClimaxLocation,
    Ending,
    Texture,
    Density,
    Counterpoint,
    Cadence,
    ExistingManualEdits,
}

/// The concrete musical intention exposed to Studio and candidate generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeIntent {
    Maintain,
    DevelopMotif,
    IntroduceContrast,
    IncreaseHarmonicInstability,
    ModulateToRelatedKey,
    IncreaseDensity,
    StrengthenCadence,
    AddCounterline,
    ReturnOpeningMaterial,
    ThinTexture,
}

impl From<SymbolicAction> for ChangeIntent {
    fn from(value: SymbolicAction) -> Self {
        match value {
            SymbolicAction::Maintain => Self::Maintain,
            SymbolicAction::DevelopMotif => Self::DevelopMotif,
            SymbolicAction::IntroduceContrast => Self::IntroduceContrast,
            SymbolicAction::IncreaseHarmonicInstability => Self::IncreaseHarmonicInstability,
            SymbolicAction::ModulateToRelatedKey => Self::ModulateToRelatedKey,
            SymbolicAction::IncreaseDensity => Self::IncreaseDensity,
            SymbolicAction::StrengthenCadence => Self::StrengthenCadence,
            SymbolicAction::AddCounterline => Self::AddCounterline,
            SymbolicAction::ReturnOpeningMaterial => Self::ReturnOpeningMaterial,
            SymbolicAction::ThinTexture => Self::ThinTexture,
        }
    }
}

/// Where the contract applies in the current piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudioTarget {
    pub section: CognitiveSection,
    pub scope: ActionScope,
}

/// A reviewable edit contract for candidate generation or direct manipulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioEditContract {
    pub target: StudioTarget,
    pub intent: ChangeIntent,
    pub preserve: BTreeSet<StudioDimension>,
    pub change: BTreeSet<StudioDimension>,
    pub urgency: f32,
    pub confidence: f32,
    pub obligation_ids: Vec<u64>,
    /// Full typed promises available to the candidate generator.
    #[serde(default)]
    pub obligation_demands: Vec<CognitiveObligationDemand>,
    /// The specific promise that selected this intent, when one did.
    #[serde(default)]
    pub driving_obligation_id: Option<u64>,
    #[serde(default)]
    pub supporting_obligation_ids: Vec<u64>,
    #[serde(default)]
    pub deferred_obligation_ids: Vec<u64>,
    pub rationale: Vec<String>,
}

/// A structural contradiction in a Preserve/Change contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractIssue {
    PreservedAndChanged(StudioDimension),
    MaintainRequestsChange,
    ActionHasNoChangeDimension,
    ManualEditsNotPreserved,
    UnknownDrivingObligation(u64),
}

impl StudioEditContract {
    /// Translate a cognitive trace into a deterministic Studio contract.
    pub fn from_trace(trace: &CognitiveDecisionTrace) -> Self {
        let mut preserve: BTreeSet<_> = trace
            .proposal
            .preserve
            .iter()
            .copied()
            .map(dimension_for_invariant)
            .collect();
        // Symthaea may propose around artist work, but it may not silently
        // overwrite it. A future explicit artist command can remove this
        // invariant before realization.
        preserve.insert(StudioDimension::ExistingManualEdits);

        Self {
            target: StudioTarget {
                section: trace.observation.section,
                scope: trace.proposal.scope,
            },
            intent: trace.proposal.action.into(),
            preserve,
            change: change_dimensions_for(trace.proposal.action),
            urgency: trace.proposal.urgency.clamp(0.0, 1.0),
            confidence: trace.proposal.confidence.clamp(0.0, 1.0),
            obligation_ids: trace.observation.overdue_obligations.clone(),
            obligation_demands: trace.observation.obligation_demands.clone(),
            driving_obligation_id: trace.proposal.driving_obligation_id,
            supporting_obligation_ids: trace.proposal.supporting_obligation_ids.clone(),
            deferred_obligation_ids: trace.proposal.deferred_obligation_ids.clone(),
            rationale: trace.proposal.rationale.clone(),
        }
    }

    /// Return every contradiction rather than failing on only the first one.
    pub fn validate(&self) -> Vec<ContractIssue> {
        let mut issues = Vec::new();
        for dimension in self.preserve.intersection(&self.change) {
            issues.push(ContractIssue::PreservedAndChanged(*dimension));
        }
        if self.intent == ChangeIntent::Maintain && !self.change.is_empty() {
            issues.push(ContractIssue::MaintainRequestsChange);
        }
        if self.intent != ChangeIntent::Maintain && self.change.is_empty() {
            issues.push(ContractIssue::ActionHasNoChangeDimension);
        }
        if !self
            .preserve
            .contains(&StudioDimension::ExistingManualEdits)
        {
            issues.push(ContractIssue::ManualEditsNotPreserved);
        }
        if let Some(id) = self.driving_obligation_id {
            if !self.obligation_demands.iter().any(|demand| demand.id == id) {
                issues.push(ContractIssue::UnknownDrivingObligation(id));
            }
        }
        issues
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

fn dimension_for_invariant(value: PreserveInvariant) -> StudioDimension {
    match value {
        PreserveInvariant::MotifIdentity => StudioDimension::MotifIdentity,
        PreserveInvariant::Melody => StudioDimension::Melody,
        PreserveInvariant::Harmony => StudioDimension::Harmony,
        PreserveInvariant::Meter => StudioDimension::Meter,
        PreserveInvariant::FormLength => StudioDimension::FormLength,
        PreserveInvariant::ClimaxLocation => StudioDimension::ClimaxLocation,
        PreserveInvariant::Ending => StudioDimension::Ending,
        PreserveInvariant::ExistingManualEdits => StudioDimension::ExistingManualEdits,
    }
}

fn change_dimensions_for(action: SymbolicAction) -> BTreeSet<StudioDimension> {
    use StudioDimension as D;
    let dimensions: &[D] = match action {
        SymbolicAction::Maintain => &[],
        SymbolicAction::DevelopMotif => &[D::MotifDevelopment],
        SymbolicAction::IntroduceContrast => &[D::Harmony, D::TonalCenter, D::Texture],
        SymbolicAction::IncreaseHarmonicInstability => &[D::Harmony],
        SymbolicAction::ModulateToRelatedKey => &[D::Harmony, D::TonalCenter],
        SymbolicAction::IncreaseDensity => &[D::Density, D::Texture],
        SymbolicAction::StrengthenCadence => &[D::Cadence, D::Harmony],
        SymbolicAction::AddCounterline => &[D::Counterpoint, D::Texture],
        SymbolicAction::ReturnOpeningMaterial => &[D::ThematicPlacement, D::TonalCenter],
        SymbolicAction::ThinTexture => &[D::Density, D::Texture],
    };
    dimensions.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{
        CognitiveGoal, CognitiveObligationDemand, InferenceEvidence, PredictedMusicalOutcome,
        SymbolicActionProposal, SymbolicMusicObservation,
    };
    use crate::musical_inference::MusicAction;

    fn trace(action: SymbolicAction, preserve: Vec<PreserveInvariant>) -> CognitiveDecisionTrace {
        CognitiveDecisionTrace {
            observation: SymbolicMusicObservation {
                section: CognitiveSection::Development,
                active_goal: Some(CognitiveGoal::BuildClimax),
                goal_urgency: 0.8,
                valence: 0.0,
                arousal: 0.7,
                prediction_error: 0.2,
                consciousness_level: 0.6,
                dominant_harmony: 0,
                dominant_harmony_activation: 0.8,
                pending_obligations: 1,
                overdue_obligations: vec![7],
                obligation_demands: Vec::new(),
                obligation_pressure: 0.9,
            },
            inference: InferenceEvidence {
                source_action: MusicAction::IncreaseComplexity,
                free_energy: 0.2,
                prediction_error: 0.2,
                surprise: 0.1,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            proposal: SymbolicActionProposal {
                action,
                driving_obligation_id: None,
                supporting_obligation_ids: Vec::new(),
                deferred_obligation_ids: Vec::new(),
                scope: ActionScope::CurrentSection,
                preserve,
                urgency: 0.8,
                confidence: 0.7,
                rationale: vec!["test evidence".into()],
            },
            predicted_outcome: PredictedMusicalOutcome {
                tension_delta: 0.1,
                density_delta: 0.2,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
        }
    }

    #[test]
    fn cognitive_proposal_becomes_a_valid_preserve_change_contract() {
        let contract = StudioEditContract::from_trace(&trace(
            SymbolicAction::IncreaseDensity,
            vec![
                PreserveInvariant::Melody,
                PreserveInvariant::Harmony,
                PreserveInvariant::Meter,
                PreserveInvariant::FormLength,
            ],
        ));

        assert!(contract.is_valid());
        assert!(contract.change.contains(&StudioDimension::Density));
        assert!(contract.preserve.contains(&StudioDimension::Melody));
        assert!(
            contract
                .preserve
                .contains(&StudioDimension::ExistingManualEdits)
        );
        assert_eq!(contract.obligation_ids, vec![7]);
    }

    #[test]
    fn typed_obligation_survives_contract_translation() {
        let mut trace = trace(
            SymbolicAction::ReturnOpeningMaterial,
            vec![PreserveInvariant::MotifIdentity],
        );
        trace.observation.obligation_demands = vec![CognitiveObligationDemand {
            id: 7,
            priority: 1.0,
            due_by: symthaea_music_theory::Duration::quarter(),
            overdue: true,
            kind: symthaea_music_theory::ObligationKind::ReturnMotif {
                motif_id: "opening".into(),
                transformation: symthaea_music_theory::ReturnTransformation::Literal,
            },
        }];
        trace.proposal.driving_obligation_id = Some(7);

        let contract = StudioEditContract::from_trace(&trace);
        assert!(contract.is_valid());
        assert_eq!(contract.driving_obligation_id, Some(7));
        assert_eq!(contract.obligation_demands[0].id, 7);
    }

    #[test]
    fn contradictory_contracts_are_rejected() {
        let mut contract = StudioEditContract::from_trace(&trace(
            SymbolicAction::IncreaseHarmonicInstability,
            vec![PreserveInvariant::Melody],
        ));
        contract.preserve.insert(StudioDimension::Harmony);

        assert_eq!(
            contract.validate(),
            vec![ContractIssue::PreservedAndChanged(StudioDimension::Harmony)]
        );
    }

    #[test]
    fn recapitulation_changes_placement_without_erasing_identity() {
        let contract = StudioEditContract::from_trace(&trace(
            SymbolicAction::ReturnOpeningMaterial,
            vec![
                PreserveInvariant::MotifIdentity,
                PreserveInvariant::Meter,
                PreserveInvariant::FormLength,
                PreserveInvariant::Ending,
            ],
        ));

        assert!(contract.is_valid());
        assert!(contract.preserve.contains(&StudioDimension::MotifIdentity));
        assert!(
            contract
                .change
                .contains(&StudioDimension::ThematicPlacement)
        );
    }
}
