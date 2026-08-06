// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bridge from Symthaea's cognitive state and active inference into symbolic
//! compositional decisions.
//!
//! The music-theory crate remains responsible for musical validity. This
//! module gives Symthaea a narrow, inspectable role above it: observe the
//! unfolding work, remember outstanding obligations, choose a meaningful
//! symbolic action, predict its effect, and retain a decision trace.
//!
//! Nothing here writes notes directly. A caller translates the resulting
//! [`SymbolicActionProposal`] into a theory-layer edit contract or candidate
//! generation request.

use crate::MusicalState;
use crate::composer_mind::{ComposerMind, GoalTarget, SectionArc};
use crate::musical_inference::{MusicAction, MusicInferenceResult};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{Duration, ObligationKind, ObligationLedger};

/// A stable, serializable section label for cognitive traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveSection {
    Exposition,
    Development,
    Climax,
    Recapitulation,
    Coda,
}

impl From<SectionArc> for CognitiveSection {
    fn from(value: SectionArc) -> Self {
        match value {
            SectionArc::Exposition => Self::Exposition,
            SectionArc::Development => Self::Development,
            SectionArc::Climax => Self::Climax,
            SectionArc::Recapitulation => Self::Recapitulation,
            SectionArc::Coda => Self::Coda,
        }
    }
}

/// A stable, serializable compositional goal label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveGoal {
    BuildClimax,
    Resolve,
    Recapitulate,
    Contrast,
    FadeToSilence,
    Sustain,
}

impl From<GoalTarget> for CognitiveGoal {
    fn from(value: GoalTarget) -> Self {
        match value {
            GoalTarget::BuildClimax => Self::BuildClimax,
            GoalTarget::Resolve => Self::Resolve,
            GoalTarget::Recapitulate => Self::Recapitulate,
            GoalTarget::Contrast => Self::Contrast,
            GoalTarget::FadeToSilence => Self::FadeToSilence,
            GoalTarget::Sustain => Self::Sustain,
        }
    }
}

/// One typed prospective-memory demand visible to the cognitive planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveObligationDemand {
    pub id: u64,
    pub priority: f32,
    pub due_by: Duration,
    pub overdue: bool,
    pub kind: ObligationKind,
}

/// What Symthaea currently believes about the symbolic composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicMusicObservation {
    pub section: CognitiveSection,
    pub active_goal: Option<CognitiveGoal>,
    pub goal_urgency: f32,
    pub valence: f32,
    pub arousal: f32,
    pub prediction_error: f32,
    pub consciousness_level: f32,
    pub dominant_harmony: usize,
    pub dominant_harmony_activation: f32,
    pub pending_obligations: usize,
    pub overdue_obligations: Vec<u64>,
    /// Typed pending promises, ordered by priority and then due time.
    #[serde(default)]
    pub obligation_demands: Vec<CognitiveObligationDemand>,
    /// Priority-weighted deadline pressure from the prospective-memory ledger.
    #[serde(default)]
    pub obligation_pressure: f32,
}

impl SymbolicMusicObservation {
    /// Build an observation from the current cognitive state, composer memory,
    /// and theory-layer prospective-memory ledger.
    pub fn capture(
        state: &MusicalState,
        mind: &ComposerMind,
        obligations: &ObligationLedger,
        now: Duration,
    ) -> Self {
        let (dominant_harmony, dominant_harmony_activation) = state
            .harmony_activations
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0));
        let active_goal = mind.primary_goal();
        let obligation_pressure = obligations.pressure_at(now);
        let obligation_demands = obligations
            .pending()
            .into_iter()
            .map(|item| CognitiveObligationDemand {
                id: item.id,
                priority: item.priority,
                due_by: item.due_by,
                overdue: item.is_due_at(now),
                kind: item.kind.clone(),
            })
            .collect();

        Self {
            section: mind.section().into(),
            active_goal: active_goal.map(|goal| goal.target.into()),
            goal_urgency: active_goal.map_or(0.0, |goal| goal.urgency.clamp(0.0, 1.0)),
            valence: state.valence.clamp(-1.0, 1.0),
            arousal: state.arousal.clamp(0.0, 1.0),
            prediction_error: state.prediction_error.clamp(0.0, 1.0),
            consciousness_level: state.consciousness_level.clamp(0.0, 1.0),
            dominant_harmony,
            dominant_harmony_activation,
            pending_obligations: obligation_pressure.pending_count,
            overdue_obligations: obligations
                .overdue_at(now)
                .into_iter()
                .map(|item| item.id)
                .collect(),
            obligation_demands,
            obligation_pressure: obligation_pressure.weighted_pressure,
        }
    }
}

/// Symbolic operations that a theory-aware realization layer can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicAction {
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

/// Semantic invariants to preserve while realizing a cognitive action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreserveInvariant {
    MotifIdentity,
    Melody,
    Harmony,
    Meter,
    FormLength,
    ClimaxLocation,
    Ending,
    ExistingManualEdits,
}

/// The default scope of a proposed action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionScope {
    CurrentPhrase,
    CurrentSection,
    WholePiece,
}

/// One aggregate vote cast by compatible prospective-memory demands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObligationActionVote {
    pub action: SymbolicAction,
    pub obligation_ids: Vec<u64>,
    pub overdue_count: usize,
    pub aggregate_urgency: f32,
    pub earliest_due_by: Duration,
}

/// Transparent arbitration among simultaneously actionable promises.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObligationArbitration {
    pub selected_action: Option<SymbolicAction>,
    pub driving_obligation_id: Option<u64>,
    pub supporting_obligation_ids: Vec<u64>,
    pub deferred_obligation_ids: Vec<u64>,
    pub votes: Vec<ObligationActionVote>,
}

/// An auditable symbolic action proposal produced from cognitive evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicActionProposal {
    pub action: SymbolicAction,
    /// The promise that selected this action, when prospective memory overrode
    /// the local active-inference action or long-range goal.
    #[serde(default)]
    pub driving_obligation_id: Option<u64>,
    /// All promises whose compatible action supports this proposal.
    #[serde(default)]
    pub supporting_obligation_ids: Vec<u64>,
    /// Actionable promises deliberately deferred because they requested a
    /// conflicting action with lower aggregate urgency.
    #[serde(default)]
    pub deferred_obligation_ids: Vec<u64>,
    pub scope: ActionScope,
    pub preserve: Vec<PreserveInvariant>,
    pub urgency: f32,
    pub confidence: f32,
    pub rationale: Vec<String>,
}

/// Expected directional effects of a proposed action.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PredictedMusicalOutcome {
    pub tension_delta: f32,
    pub density_delta: f32,
    pub familiarity_delta: f32,
    pub tonal_displacement_delta: f32,
}

/// Measured directional effects after theory realization and rendering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservedMusicalOutcome {
    pub tension_delta: f32,
    pub density_delta: f32,
    pub familiarity_delta: f32,
    pub tonal_displacement_delta: f32,
}

/// Version of the deterministic score-side measurement contract.
pub const SYMBOLIC_MEASUREMENT_VERSION: &str = "score-cognitive-profile-v2";

/// Auditable symbolic evidence retained when a candidate is compared with its
/// baseline. Renderer and listener evidence remain separate channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicMeasurementEvidence {
    pub measurement_version: String,
    pub baseline: symthaea_music_theory::ScoreCognitiveProfile,
    pub candidate: symthaea_music_theory::ScoreCognitiveProfile,
    pub observed_outcome: ObservedMusicalOutcome,
}

impl SymbolicMeasurementEvidence {
    pub fn new(
        baseline: symthaea_music_theory::ScoreCognitiveProfile,
        candidate: symthaea_music_theory::ScoreCognitiveProfile,
    ) -> Self {
        let delta = baseline.delta_to(candidate);
        Self {
            measurement_version: SYMBOLIC_MEASUREMENT_VERSION.into(),
            baseline,
            candidate,
            observed_outcome: ObservedMusicalOutcome {
                tension_delta: delta.tension_delta,
                density_delta: delta.density_delta,
                familiarity_delta: delta.familiarity_delta,
                tonal_displacement_delta: delta.tonal_displacement_delta,
            },
        }
    }
}

/// Interpretable error between predicted and observed musical effects.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MusicalOutcomeError {
    pub tension_error: f32,
    pub density_error: f32,
    pub familiarity_error: f32,
    pub tonal_displacement_error: f32,
    pub mean_absolute_error: f32,
}

impl PredictedMusicalOutcome {
    /// Compare the cognitive prediction with measured symbolic/audio effects.
    pub fn error(self, observed: ObservedMusicalOutcome) -> MusicalOutcomeError {
        let tension_error = observed.tension_delta - self.tension_delta;
        let density_error = observed.density_delta - self.density_delta;
        let familiarity_error = observed.familiarity_delta - self.familiarity_delta;
        let tonal_displacement_error =
            observed.tonal_displacement_delta - self.tonal_displacement_delta;
        let mean_absolute_error = (tension_error.abs()
            + density_error.abs()
            + familiarity_error.abs()
            + tonal_displacement_error.abs())
            / 4.0;

        MusicalOutcomeError {
            tension_error,
            density_error,
            familiarity_error,
            tonal_displacement_error,
            mean_absolute_error,
        }
    }
}

/// Minimal active-inference evidence retained with a decision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InferenceEvidence {
    pub source_action: MusicAction,
    pub free_energy: f64,
    pub prediction_error: f64,
    pub surprise: f64,
    pub sensory_precision: f64,
    pub prior_precision: f64,
}

impl From<&MusicInferenceResult> for InferenceEvidence {
    fn from(value: &MusicInferenceResult) -> Self {
        Self {
            source_action: value.action,
            free_energy: value.free_energy,
            prediction_error: value.prediction_error,
            surprise: value.surprise,
            sensory_precision: value.sensory_precision,
            prior_precision: value.prior_precision,
        }
    }
}

/// Complete trace from observation through symbolic proposal and prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveDecisionTrace {
    pub observation: SymbolicMusicObservation,
    pub inference: InferenceEvidence,
    pub proposal: SymbolicActionProposal,
    pub predicted_outcome: PredictedMusicalOutcome,
}

/// Convert a live active-inference result into a symbolic, constrained action.
pub fn propose_symbolic_action(
    inference: &MusicInferenceResult,
    observation: SymbolicMusicObservation,
) -> CognitiveDecisionTrace {
    let arbitration = arbitrate_obligations(&observation);
    let (action, driving_obligation_id) = action_for(inference.action, &observation, &arbitration);
    let proposal = SymbolicActionProposal {
        action,
        driving_obligation_id,
        supporting_obligation_ids: arbitration.supporting_obligation_ids.clone(),
        deferred_obligation_ids: arbitration.deferred_obligation_ids.clone(),
        scope: scope_for(action),
        preserve: invariants_for(action),
        urgency: observation
            .goal_urgency
            .max(observation.obligation_pressure)
            .max(inference.surprise.clamp(0.0, 1.0) as f32),
        confidence: (inference.prior_precision / 2.0).clamp(0.0, 1.0) as f32,
        rationale: rationale_for(
            inference,
            &observation,
            action,
            driving_obligation_id,
            &arbitration,
        ),
    };
    let predicted_outcome = default_predicted_outcome(action);

    CognitiveDecisionTrace {
        observation,
        inference: inference.into(),
        proposal,
        predicted_outcome,
    }
}

/// Group compatible promises, score their aggregate urgency, and make any
/// conflict explicit. An overdue actionable promise always participates;
/// otherwise prospective memory takes control only once aggregate deadline
/// pressure reaches 0.75.
pub fn arbitrate_obligations(observation: &SymbolicMusicObservation) -> ObligationArbitration {
    let mut votes: Vec<ObligationActionVote> = Vec::new();
    for demand in &observation.obligation_demands {
        let Some(action) = action_for_obligation(&demand.kind) else {
            continue;
        };
        let urgency = demand.priority.clamp(0.0, 1.0)
            * if demand.overdue {
                1.5
            } else {
                0.5 + 0.5 * observation.obligation_pressure.clamp(0.0, 1.0)
            };
        if let Some(vote) = votes.iter_mut().find(|vote| vote.action == action) {
            vote.obligation_ids.push(demand.id);
            vote.overdue_count += usize::from(demand.overdue);
            vote.aggregate_urgency += urgency;
            if demand.due_by.beats() < vote.earliest_due_by.beats() {
                vote.earliest_due_by = demand.due_by;
            }
        } else {
            votes.push(ObligationActionVote {
                action,
                obligation_ids: vec![demand.id],
                overdue_count: usize::from(demand.overdue),
                aggregate_urgency: urgency,
                earliest_due_by: demand.due_by,
            });
        }
    }
    for vote in &mut votes {
        vote.obligation_ids.sort_unstable();
    }
    votes.sort_by(|left, right| {
        right
            .overdue_count
            .cmp(&left.overdue_count)
            .then_with(|| right.aggregate_urgency.total_cmp(&left.aggregate_urgency))
            .then_with(|| {
                left.earliest_due_by
                    .beats()
                    .total_cmp(&right.earliest_due_by.beats())
            })
            .then_with(|| action_rank(left.action).cmp(&action_rank(right.action)))
    });

    let prospective_control =
        votes.iter().any(|vote| vote.overdue_count > 0) || observation.obligation_pressure >= 0.75;
    let selected = if prospective_control {
        votes.first()
    } else {
        None
    };
    let selected_action = selected.map(|vote| vote.action);
    let supporting_obligation_ids = selected
        .map(|vote| vote.obligation_ids.clone())
        .unwrap_or_default();
    let driving_obligation_id = selected.and_then(|vote| {
        vote.obligation_ids
            .iter()
            .filter_map(|id| {
                observation
                    .obligation_demands
                    .iter()
                    .find(|demand| demand.id == *id)
            })
            .max_by(|left, right| {
                left.overdue
                    .cmp(&right.overdue)
                    .then_with(|| left.priority.total_cmp(&right.priority))
                    .then_with(|| right.due_by.beats().total_cmp(&left.due_by.beats()))
            })
            .map(|demand| demand.id)
    });
    let deferred_obligation_ids = if selected_action.is_some() {
        votes
            .iter()
            .skip(1)
            .flat_map(|vote| vote.obligation_ids.iter().copied())
            .collect()
    } else {
        Vec::new()
    };

    ObligationArbitration {
        selected_action,
        driving_obligation_id,
        supporting_obligation_ids,
        deferred_obligation_ids,
        votes,
    }
}

fn action_for(
    source: MusicAction,
    observation: &SymbolicMusicObservation,
    arbitration: &ObligationArbitration,
) -> (SymbolicAction, Option<u64>) {
    if let Some(action) = arbitration.selected_action {
        return (action, arbitration.driving_obligation_id);
    }

    let action = match observation.active_goal {
        Some(CognitiveGoal::Recapitulate) => SymbolicAction::ReturnOpeningMaterial,
        Some(CognitiveGoal::Resolve) => SymbolicAction::StrengthenCadence,
        Some(CognitiveGoal::FadeToSilence) => SymbolicAction::ThinTexture,
        Some(CognitiveGoal::Contrast) if source == MusicAction::Maintain => {
            SymbolicAction::IntroduceContrast
        }
        _ => match source {
            MusicAction::FollowHarmony => SymbolicAction::Maintain,
            MusicAction::ChromaticExplore => SymbolicAction::IncreaseHarmonicInstability,
            MusicAction::RepeatMotif => SymbolicAction::DevelopMotif,
            MusicAction::ModulateKey => SymbolicAction::ModulateToRelatedKey,
            MusicAction::IncreaseComplexity => SymbolicAction::IncreaseDensity,
            MusicAction::ResolveTension => SymbolicAction::StrengthenCadence,
            MusicAction::AddCountermelody => SymbolicAction::AddCounterline,
            MusicAction::Maintain => SymbolicAction::Maintain,
        },
    };
    (action, None)
}

fn action_rank(action: SymbolicAction) -> u8 {
    match action {
        SymbolicAction::ReturnOpeningMaterial => 0,
        SymbolicAction::StrengthenCadence => 1,
        SymbolicAction::ModulateToRelatedKey => 2,
        SymbolicAction::IncreaseDensity => 3,
        SymbolicAction::AddCounterline => 4,
        SymbolicAction::Maintain => 5,
        SymbolicAction::DevelopMotif => 6,
        SymbolicAction::IntroduceContrast => 7,
        SymbolicAction::IncreaseHarmonicInstability => 8,
        SymbolicAction::ThinTexture => 9,
    }
}

fn action_for_obligation(kind: &ObligationKind) -> Option<SymbolicAction> {
    match kind {
        ObligationKind::ReturnMotif { .. } | ObligationKind::RestoreIdentity { .. } => {
            Some(SymbolicAction::ReturnOpeningMaterial)
        }
        ObligationKind::ReachKey { .. } => Some(SymbolicAction::ModulateToRelatedKey),
        ObligationKind::Cadence { .. } | ObligationKind::ResolveAlteredDegree { .. } => {
            Some(SymbolicAction::StrengthenCadence)
        }
        ObligationKind::EnterVoice { .. } => Some(SymbolicAction::AddCounterline),
        ObligationKind::ReachClimax => Some(SymbolicAction::IncreaseDensity),
        ObligationKind::Custom { .. } => None,
    }
}

fn scope_for(action: SymbolicAction) -> ActionScope {
    match action {
        SymbolicAction::ReturnOpeningMaterial | SymbolicAction::ModulateToRelatedKey => {
            ActionScope::CurrentSection
        }
        SymbolicAction::IntroduceContrast
        | SymbolicAction::IncreaseDensity
        | SymbolicAction::ThinTexture => ActionScope::CurrentSection,
        SymbolicAction::Maintain
        | SymbolicAction::DevelopMotif
        | SymbolicAction::IncreaseHarmonicInstability
        | SymbolicAction::StrengthenCadence
        | SymbolicAction::AddCounterline => ActionScope::CurrentPhrase,
    }
}

fn invariants_for(action: SymbolicAction) -> Vec<PreserveInvariant> {
    match action {
        SymbolicAction::Maintain => vec![
            PreserveInvariant::MotifIdentity,
            PreserveInvariant::Melody,
            PreserveInvariant::Harmony,
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
        ],
        SymbolicAction::DevelopMotif => vec![
            PreserveInvariant::MotifIdentity,
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
        ],
        SymbolicAction::IntroduceContrast => vec![
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
            PreserveInvariant::ClimaxLocation,
        ],
        SymbolicAction::IncreaseHarmonicInstability
        | SymbolicAction::ModulateToRelatedKey
        | SymbolicAction::StrengthenCadence => vec![
            PreserveInvariant::MotifIdentity,
            PreserveInvariant::Melody,
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
        ],
        SymbolicAction::IncreaseDensity
        | SymbolicAction::AddCounterline
        | SymbolicAction::ThinTexture => vec![
            PreserveInvariant::Melody,
            PreserveInvariant::Harmony,
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
        ],
        SymbolicAction::ReturnOpeningMaterial => vec![
            PreserveInvariant::MotifIdentity,
            PreserveInvariant::Meter,
            PreserveInvariant::FormLength,
            PreserveInvariant::Ending,
        ],
    }
}

pub fn default_predicted_outcome(action: SymbolicAction) -> PredictedMusicalOutcome {
    match action {
        SymbolicAction::Maintain => PredictedMusicalOutcome {
            tension_delta: 0.0,
            density_delta: 0.0,
            familiarity_delta: 0.1,
            tonal_displacement_delta: 0.0,
        },
        SymbolicAction::DevelopMotif => PredictedMusicalOutcome {
            tension_delta: 0.1,
            density_delta: 0.0,
            familiarity_delta: 0.15,
            tonal_displacement_delta: 0.0,
        },
        SymbolicAction::IntroduceContrast => PredictedMusicalOutcome {
            tension_delta: 0.2,
            density_delta: 0.1,
            familiarity_delta: -0.25,
            tonal_displacement_delta: 0.2,
        },
        SymbolicAction::IncreaseHarmonicInstability => PredictedMusicalOutcome {
            tension_delta: 0.3,
            density_delta: 0.0,
            familiarity_delta: -0.1,
            tonal_displacement_delta: 0.15,
        },
        SymbolicAction::ModulateToRelatedKey => PredictedMusicalOutcome {
            tension_delta: 0.15,
            density_delta: 0.0,
            familiarity_delta: -0.1,
            tonal_displacement_delta: 0.4,
        },
        SymbolicAction::IncreaseDensity => PredictedMusicalOutcome {
            tension_delta: 0.15,
            density_delta: 0.35,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        },
        SymbolicAction::StrengthenCadence => PredictedMusicalOutcome {
            tension_delta: -0.35,
            density_delta: -0.05,
            familiarity_delta: 0.2,
            tonal_displacement_delta: -0.2,
        },
        SymbolicAction::AddCounterline => PredictedMusicalOutcome {
            tension_delta: 0.1,
            density_delta: 0.25,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        },
        SymbolicAction::ReturnOpeningMaterial => PredictedMusicalOutcome {
            tension_delta: -0.2,
            density_delta: 0.0,
            familiarity_delta: 0.5,
            tonal_displacement_delta: -0.35,
        },
        SymbolicAction::ThinTexture => PredictedMusicalOutcome {
            tension_delta: -0.15,
            density_delta: -0.4,
            familiarity_delta: 0.0,
            tonal_displacement_delta: 0.0,
        },
    }
}

fn rationale_for(
    inference: &MusicInferenceResult,
    observation: &SymbolicMusicObservation,
    action: SymbolicAction,
    driving_obligation_id: Option<u64>,
    arbitration: &ObligationArbitration,
) -> Vec<String> {
    let mut rationale = vec![format!(
        "active inference selected {:?} with prediction error {:.3}",
        inference.action, inference.prediction_error
    )];
    if let Some(goal) = observation.active_goal {
        rationale.push(format!(
            "current {:?} goal has urgency {:.2}",
            goal, observation.goal_urgency
        ));
    }
    if observation.pending_obligations > 0 {
        rationale.push(format!(
            "{} compositional obligation(s) exert {:.2} deadline pressure",
            observation.pending_obligations, observation.obligation_pressure
        ));
    }
    if !observation.overdue_obligations.is_empty() {
        rationale.push(format!(
            "{} compositional obligation(s) are overdue",
            observation.overdue_obligations.len()
        ));
    }
    if arbitration.supporting_obligation_ids.len() > 1 {
        rationale.push(format!(
            "{} compatible obligations jointly support this action",
            arbitration.supporting_obligation_ids.len()
        ));
    }
    if !arbitration.deferred_obligation_ids.is_empty() {
        rationale.push(format!(
            "deferred conflicting obligation(s): {:?}",
            arbitration.deferred_obligation_ids
        ));
    }
    if let Some(demand) = driving_obligation_id.and_then(|id| {
        observation
            .obligation_demands
            .iter()
            .find(|demand| demand.id == id)
    }) {
        rationale.push(format!(
            "typed obligation {} ({:?}) selected this action",
            demand.id, demand.kind
        ));
    }
    rationale.push(format!("proposed symbolic action: {action:?}"));
    rationale
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{CompositionalObligation, ObligationKind};

    /// PINS THE HONEST CONTRACT: on the live studio path the proposed action is
    /// decided by the caller's hardcoded goal, NOT by the cognitive session.
    ///
    /// `muse_studio.rs` calls `bridge_observation(..., Some(CognitiveGoal::Recapitulate), ...)`
    /// with that goal as a literal, and `action_for` matches it before ever
    /// reading the FEP inference. So every `MusicAction` the HDC/CfC session
    /// could possibly infer collapses to the same `SymbolicAction`.
    ///
    /// This test exists so that stays TRUE-and-documented or becomes
    /// FALSE-and-noticed. If someone wires the inference in, this fails — and
    /// the correct response is to update `CognitiveSession::bridge_observation`'s
    /// "provenance, not control" doc section rather than to weaken the test.
    #[test]
    fn studio_goal_pins_the_action_regardless_of_what_the_session_inferred() {
        let obs = observation(Some(CognitiveGoal::Recapitulate));
        let every_action = [
            MusicAction::FollowHarmony,
            MusicAction::ChromaticExplore,
            MusicAction::RepeatMotif,
            MusicAction::ModulateKey,
            MusicAction::IncreaseComplexity,
            MusicAction::Maintain,
        ];
        for a in every_action {
            let trace = propose_symbolic_action(&inference(a), obs.clone());
            assert_eq!(
                trace.proposal.action,
                SymbolicAction::ReturnOpeningMaterial,
                "inference {a:?} changed the proposed action — the session now influences \
                 output, so bridge_observation's 'provenance, not control' doc is stale"
            );
        }
    }

    fn inference(action: MusicAction) -> MusicInferenceResult {
        MusicInferenceResult {
            action,
            free_energy: 0.2,
            prediction_error: 0.3,
            surprise: 0.4,
            is_surprised: false,
            learning_rate_mod: 1.0,
            sensory_precision: 1.0,
            prior_precision: 1.2,
        }
    }

    fn observation(goal: Option<CognitiveGoal>) -> SymbolicMusicObservation {
        SymbolicMusicObservation {
            section: CognitiveSection::Development,
            active_goal: goal,
            goal_urgency: 0.8,
            valence: 0.0,
            arousal: 0.6,
            prediction_error: 0.3,
            consciousness_level: 0.7,
            dominant_harmony: 0,
            dominant_harmony_activation: 0.8,
            pending_obligations: 0,
            overdue_obligations: Vec::new(),
            obligation_demands: Vec::new(),
            obligation_pressure: 0.0,
        }
    }

    #[test]
    fn active_inference_maps_to_symbolic_action_with_invariants() {
        let trace =
            propose_symbolic_action(&inference(MusicAction::ChromaticExplore), observation(None));
        assert_eq!(
            trace.proposal.action,
            SymbolicAction::IncreaseHarmonicInstability
        );
        assert!(trace.proposal.preserve.contains(&PreserveInvariant::Melody));
        assert!(trace.predicted_outcome.tension_delta > 0.0);
    }

    #[test]
    fn long_range_goal_can_override_local_maintain_action() {
        let trace = propose_symbolic_action(
            &inference(MusicAction::Maintain),
            observation(Some(CognitiveGoal::Recapitulate)),
        );
        assert_eq!(trace.proposal.action, SymbolicAction::ReturnOpeningMaterial);
        assert!(trace.predicted_outcome.familiarity_delta > 0.0);
    }

    #[test]
    fn capture_brings_theory_obligations_into_cognitive_state() {
        let state = MusicalState::default();
        let mind = ComposerMind::new();
        let mut obligations = ObligationLedger::new();
        obligations.add(CompositionalObligation::new(
            9,
            Duration::zero(),
            Duration::quarter(),
            1.0,
            ObligationKind::ReachClimax,
        ));

        let captured =
            SymbolicMusicObservation::capture(&state, &mind, &obligations, Duration::quarter());
        assert_eq!(captured.pending_obligations, 1);
        assert_eq!(captured.overdue_obligations, vec![9]);
        assert_eq!(captured.obligation_demands.len(), 1);
        assert_eq!(captured.obligation_demands[0].id, 9);
        assert_eq!(captured.obligation_pressure, 1.0);
    }

    #[test]
    fn obligation_pressure_can_raise_proposal_urgency() {
        let mut observed = observation(None);
        observed.goal_urgency = 0.1;
        observed.obligation_pressure = 0.9;
        observed.pending_obligations = 1;
        let trace = propose_symbolic_action(&inference(MusicAction::Maintain), observed);
        assert_eq!(trace.proposal.urgency, 0.9);
        assert!(
            trace
                .proposal
                .rationale
                .iter()
                .any(|line| line.contains("deadline pressure"))
        );
    }

    #[test]
    fn overdue_typed_obligation_selects_a_responsible_action() {
        let mut observed = observation(Some(CognitiveGoal::BuildClimax));
        observed.obligation_pressure = 1.0;
        observed.overdue_obligations = vec![41];
        observed.obligation_demands = vec![CognitiveObligationDemand {
            id: 41,
            priority: 1.0,
            due_by: Duration::quarter(),
            overdue: true,
            kind: ObligationKind::ReturnMotif {
                motif_id: "opening".into(),
                transformation: symthaea_music_theory::ReturnTransformation::Literal,
            },
        }];

        let trace = propose_symbolic_action(&inference(MusicAction::IncreaseComplexity), observed);
        assert_eq!(trace.proposal.action, SymbolicAction::ReturnOpeningMaterial);
        assert_eq!(trace.proposal.driving_obligation_id, Some(41));
    }

    #[test]
    fn prediction_error_is_channel_specific() {
        let predicted = default_predicted_outcome(SymbolicAction::IncreaseDensity);
        let error = predicted.error(ObservedMusicalOutcome {
            tension_delta: predicted.tension_delta,
            density_delta: 0.0,
            familiarity_delta: predicted.familiarity_delta,
            tonal_displacement_delta: predicted.tonal_displacement_delta,
        });
        assert_eq!(error.tension_error, 0.0);
        assert!(error.density_error < 0.0);
        assert!(error.mean_absolute_error > 0.0);
    }

    #[test]
    fn compatible_obligations_aggregate_into_one_action_vote() {
        let mut observed = observation(None);
        observed.obligation_pressure = 0.9;
        observed.obligation_demands = vec![
            CognitiveObligationDemand {
                id: 10,
                priority: 0.7,
                due_by: Duration::half(),
                overdue: false,
                kind: ObligationKind::Cadence {
                    arrival_degree: symthaea_music_theory::AlteredDegree::diatonic(1),
                },
            },
            CognitiveObligationDemand {
                id: 11,
                priority: 0.8,
                due_by: Duration::whole(),
                overdue: false,
                kind: ObligationKind::ResolveAlteredDegree {
                    degree: symthaea_music_theory::AlteredDegree::new(4, 1).unwrap(),
                },
            },
        ];
        let arbitration = arbitrate_obligations(&observed);
        assert_eq!(
            arbitration.selected_action,
            Some(SymbolicAction::StrengthenCadence)
        );
        assert_eq!(arbitration.supporting_obligation_ids, vec![10, 11]);
        assert!(arbitration.deferred_obligation_ids.is_empty());
    }

    #[test]
    fn conflicting_overdue_promises_are_ranked_and_deferred_explicitly() {
        let mut observed = observation(None);
        observed.obligation_pressure = 1.0;
        observed.obligation_demands = vec![
            CognitiveObligationDemand {
                id: 20,
                priority: 1.0,
                due_by: Duration::quarter(),
                overdue: true,
                kind: ObligationKind::ReturnMotif {
                    motif_id: "opening".into(),
                    transformation: symthaea_music_theory::ReturnTransformation::Literal,
                },
            },
            CognitiveObligationDemand {
                id: 21,
                priority: 0.4,
                due_by: Duration::quarter(),
                overdue: true,
                kind: ObligationKind::ReachKey {
                    key: symthaea_music_theory::Key::major(symthaea_music_theory::PitchClass::G),
                },
            },
        ];
        let trace = propose_symbolic_action(&inference(MusicAction::Maintain), observed);
        assert_eq!(trace.proposal.action, SymbolicAction::ReturnOpeningMaterial);
        assert_eq!(trace.proposal.supporting_obligation_ids, vec![20]);
        assert_eq!(trace.proposal.deferred_obligation_ids, vec![21]);
        assert!(
            trace
                .proposal
                .rationale
                .iter()
                .any(|line| line.contains("deferred"))
        );
    }
}
