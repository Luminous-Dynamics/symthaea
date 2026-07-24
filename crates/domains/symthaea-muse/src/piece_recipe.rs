// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, serializable recipes and version provenance for symbolic pieces.
//!
//! A seed alone is not a complete reproduction contract. The resolved intent,
//! composition specification, renderer identity, model versions, cognitive
//! decisions, and artifact digests must travel together.

use crate::MusicalState;
use crate::adaptive_prediction::{InterventionCalibrationEvidence, PredictionCalibrationEvidence};
use crate::cognitive_bridge::{
    CognitiveDecisionTrace, MusicalOutcomeError, ObservedMusicalOutcome,
    SymbolicMeasurementEvidence,
};
use crate::cognitive_session::CognitiveSessionTrace;
use crate::intervention::InterventionDescriptor;
use crate::musical_policy::MusicalPolicyPreference;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_music_theory::{CompositionSpec, MusicalIntent};

pub const PIECE_RECIPE_SCHEMA_VERSION: u32 = 3;

/// Renderer and model identity required to interpret or reproduce an export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererRecipe {
    pub renderer_name: String,
    pub sample_rate_hz: u32,
    pub muse_engine_version: String,
    pub theory_engine_version: String,
    /// Exact external renderer version when one participates in the export.
    #[serde(default)]
    pub renderer_version: Option<String>,
    /// Source revisions are separate from package versions because local builds
    /// may contain untagged changes.
    #[serde(default)]
    pub muse_source_revision: Option<String>,
    #[serde(default)]
    pub theory_source_revision: Option<String>,
    pub soundfont_sha256: Option<String>,
    #[serde(default)]
    pub renderer_binary_sha256: Option<String>,
    pub performance_model_sha256: Option<String>,
    /// Digest of a lockfile, Nix closure, OCI image, or equivalent renderer
    /// environment manifest.
    #[serde(default)]
    pub render_environment_sha256: Option<String>,
}

impl RendererRecipe {
    pub fn new(
        renderer_name: impl Into<String>,
        sample_rate_hz: u32,
        muse_engine_version: impl Into<String>,
        theory_engine_version: impl Into<String>,
    ) -> Self {
        Self {
            renderer_name: renderer_name.into(),
            sample_rate_hz,
            muse_engine_version: muse_engine_version.into(),
            theory_engine_version: theory_engine_version.into(),
            renderer_version: None,
            muse_source_revision: None,
            theory_source_revision: None,
            soundfont_sha256: None,
            renderer_binary_sha256: None,
            performance_model_sha256: None,
            render_environment_sha256: None,
        }
    }

    pub fn with_renderer_version(mut self, version: impl Into<String>) -> Self {
        self.renderer_version = Some(version.into());
        self
    }

    pub fn with_source_revisions(
        mut self,
        muse_revision: impl Into<String>,
        theory_revision: impl Into<String>,
    ) -> Self {
        self.muse_source_revision = Some(muse_revision.into());
        self.theory_source_revision = Some(theory_revision.into());
        self
    }
}

/// One explicit artist edit rather than an opaque edit identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManualEditRecord {
    pub id: String,
    /// Stable operation vocabulary, for example `transpose`, `trim`, or
    /// `replace-notes`. New operations may be introduced without a schema bump.
    pub operation: String,
    pub target: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, String>,
    #[serde(default)]
    pub parent_score_sha256: Option<String>,
    #[serde(default)]
    pub result_score_sha256: Option<String>,
}

/// Missing information that prevents independent artifact reconstruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReproductionGap {
    RendererVersion,
    MuseSourceRevision,
    TheorySourceRevision,
    SoundfontDigest,
    RendererBinaryDigest,
    RenderEnvironmentDigest,
}

/// The artist's disposition toward one cognitive proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionDisposition {
    Proposed,
    /// A generated alternative was rendered for review but not accepted.
    Previewed,
    Accepted,
    Edited,
    Rejected,
}

/// Invalid artist-response updates to a cognitive decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionResponseError {
    DecisionNotFound(u32),
    MissingSelectedAlternative,
    InvalidCognitiveSession,
}

/// One cognitive decision and the evidence accumulated after realization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecipeDecision {
    pub sequence: u32,
    pub trace: CognitiveDecisionTrace,
    pub disposition: DecisionDisposition,
    pub selected_alternative_id: Option<String>,
    pub observed_outcome: Option<ObservedMusicalOutcome>,
    pub prediction_error: Option<MusicalOutcomeError>,
    /// Exact score-side profiles that produced the observed directional delta.
    pub symbolic_measurement: Option<SymbolicMeasurementEvidence>,
    /// Legacy action/context calibration retained for schema compatibility.
    #[serde(default)]
    pub prediction_calibration: Option<PredictionCalibrationEvidence>,
    /// Exact temporal Symthaea trajectory that produced the terminal inference.
    #[serde(default)]
    pub cognitive_session: Option<CognitiveSessionTrace>,
    /// Exact parameterized intervention proposed in the v6 path.
    #[serde(default)]
    pub intervention_descriptor: Option<InterventionDescriptor>,
    /// World-model evidence for that exact intervention.
    #[serde(default)]
    pub intervention_prediction: Option<InterventionCalibrationEvidence>,
    /// Frozen utility policy used to choose among valid alternatives.
    #[serde(default)]
    pub policy_preference: Option<MusicalPolicyPreference>,
    pub artist_note: Option<String>,
}

impl RecipeDecision {
    pub fn proposed(sequence: u32, trace: CognitiveDecisionTrace) -> Self {
        Self {
            sequence,
            trace,
            disposition: DecisionDisposition::Proposed,
            selected_alternative_id: None,
            observed_outcome: None,
            prediction_error: None,
            symbolic_measurement: None,
            prediction_calibration: None,
            cognitive_session: None,
            intervention_descriptor: None,
            intervention_prediction: None,
            policy_preference: None,
            artist_note: None,
        }
    }

    /// Attach the validated temporal cognitive trajectory that produced this decision.
    pub fn attach_cognitive_session(
        &mut self,
        session: CognitiveSessionTrace,
    ) -> Result<(), DecisionResponseError> {
        if !session.is_valid() {
            return Err(DecisionResponseError::InvalidCognitiveSession);
        }
        self.cognitive_session = Some(session);
        Ok(())
    }

    /// Attach measured outcome evidence and compute the channel-specific error.
    pub fn observe(&mut self, observed: ObservedMusicalOutcome) {
        self.prediction_error = Some(self.trace.predicted_outcome.error(observed));
        self.observed_outcome = Some(observed);
    }

    /// Retain the score profiles as well as their directional outcome.
    pub fn observe_symbolic(&mut self, evidence: SymbolicMeasurementEvidence) {
        self.observe(evidence.observed_outcome);
        self.symbolic_measurement = Some(evidence);
    }
}

/// Complete deterministic symbolic recipe plus auditable cognitive history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceRecipe {
    pub schema_version: u32,
    pub intent: MusicalIntent,
    pub resolved_spec: CompositionSpec,
    /// Exact cognitive/performance state used by Muse realization.
    #[serde(default)]
    pub initial_musical_state: MusicalState,
    pub renderer: RendererRecipe,
    pub cognitive_decisions: Vec<RecipeDecision>,
    /// Legacy identifiers retained for schema-v1 readers.
    #[serde(default)]
    pub manual_edit_ids: Vec<String>,
    /// Structured, replayable artist operations.
    #[serde(default)]
    pub manual_edits: Vec<ManualEditRecord>,
}

impl PieceRecipe {
    pub fn new(
        intent: MusicalIntent,
        resolved_spec: CompositionSpec,
        renderer: RendererRecipe,
    ) -> Self {
        Self {
            schema_version: PIECE_RECIPE_SCHEMA_VERSION,
            intent,
            resolved_spec,
            initial_musical_state: MusicalState::default(),
            renderer,
            cognitive_decisions: Vec::new(),
            manual_edit_ids: Vec::new(),
            manual_edits: Vec::new(),
        }
    }

    pub fn with_initial_musical_state(mut self, state: MusicalState) -> Self {
        self.initial_musical_state = state;
        self
    }

    pub fn record_manual_edit(&mut self, edit: ManualEditRecord) {
        self.manual_edit_ids.push(edit.id.clone());
        self.manual_edits.push(edit);
    }

    pub fn reproduction_gaps(&self) -> Vec<ReproductionGap> {
        let mut gaps = Vec::new();
        if self
            .renderer
            .renderer_version
            .as_deref()
            .is_none_or(str::is_empty)
        {
            gaps.push(ReproductionGap::RendererVersion);
        }
        if self
            .renderer
            .muse_source_revision
            .as_deref()
            .is_none_or(str::is_empty)
        {
            gaps.push(ReproductionGap::MuseSourceRevision);
        }
        if self
            .renderer
            .theory_source_revision
            .as_deref()
            .is_none_or(str::is_empty)
        {
            gaps.push(ReproductionGap::TheorySourceRevision);
        }
        if self.renderer.renderer_name != "native" {
            if self
                .renderer
                .soundfont_sha256
                .as_deref()
                .is_none_or(str::is_empty)
            {
                gaps.push(ReproductionGap::SoundfontDigest);
            }
            if self
                .renderer
                .renderer_binary_sha256
                .as_deref()
                .is_none_or(str::is_empty)
            {
                gaps.push(ReproductionGap::RendererBinaryDigest);
            }
        }
        if self
            .renderer
            .render_environment_sha256
            .as_deref()
            .is_none_or(str::is_empty)
        {
            gaps.push(ReproductionGap::RenderEnvironmentDigest);
        }
        gaps
    }

    pub fn record_decision(&mut self, trace: CognitiveDecisionTrace) -> u32 {
        let sequence = self.cognitive_decisions.len() as u32;
        self.cognitive_decisions
            .push(RecipeDecision::proposed(sequence, trace));
        sequence
    }

    /// Record that one generated alternative was rendered for review without
    /// treating the preview as artist acceptance.
    pub fn record_preview(
        &mut self,
        sequence: u32,
        alternative_id: String,
    ) -> Result<(), DecisionResponseError> {
        if alternative_id.trim().is_empty() {
            return Err(DecisionResponseError::MissingSelectedAlternative);
        }
        let decision = self
            .cognitive_decisions
            .get_mut(sequence as usize)
            .filter(|decision| decision.sequence == sequence)
            .ok_or(DecisionResponseError::DecisionNotFound(sequence))?;
        decision.disposition = DecisionDisposition::Previewed;
        decision.selected_alternative_id = Some(alternative_id);
        Ok(())
    }

    /// Record the artist's explicit response to a cognitive proposal.
    ///
    /// Accepted and edited decisions must identify the concrete alternative
    /// that entered the piece. Rejection may omit an alternative identifier.
    pub fn record_artist_response(
        &mut self,
        sequence: u32,
        disposition: DecisionDisposition,
        selected_alternative_id: Option<String>,
        artist_note: Option<String>,
    ) -> Result<(), DecisionResponseError> {
        if matches!(
            disposition,
            DecisionDisposition::Accepted | DecisionDisposition::Edited
        ) && selected_alternative_id
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(DecisionResponseError::MissingSelectedAlternative);
        }

        let decision = self
            .cognitive_decisions
            .get_mut(sequence as usize)
            .filter(|decision| decision.sequence == sequence)
            .ok_or(DecisionResponseError::DecisionNotFound(sequence))?;
        decision.disposition = disposition;
        decision.selected_alternative_id = selected_alternative_id;
        decision.artist_note = artist_note;
        Ok(())
    }

    pub fn validate(&self) -> Vec<RecipeIssue> {
        let mut issues = Vec::new();
        if !(1..=PIECE_RECIPE_SCHEMA_VERSION).contains(&self.schema_version) {
            issues.push(RecipeIssue::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.renderer.renderer_name.trim().is_empty() {
            issues.push(RecipeIssue::MissingRendererName);
        }
        if self.renderer.sample_rate_hz == 0 {
            issues.push(RecipeIssue::InvalidSampleRate);
        }
        if self.renderer.muse_engine_version.trim().is_empty() {
            issues.push(RecipeIssue::MissingMuseVersion);
        }
        if self.renderer.theory_engine_version.trim().is_empty() {
            issues.push(RecipeIssue::MissingTheoryVersion);
        }
        for digest in [
            self.renderer.soundfont_sha256.as_deref(),
            self.renderer.renderer_binary_sha256.as_deref(),
            self.renderer.performance_model_sha256.as_deref(),
            self.renderer.render_environment_sha256.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !is_sha256(digest) {
                issues.push(RecipeIssue::InvalidSha256(digest.to_owned()));
            }
        }
        let mut edit_ids = BTreeSet::new();
        for edit in &self.manual_edits {
            if edit.id.trim().is_empty() {
                issues.push(RecipeIssue::MissingManualEditId);
            } else if !edit_ids.insert(edit.id.as_str()) {
                issues.push(RecipeIssue::DuplicateManualEditId(edit.id.clone()));
            }
            if edit.operation.trim().is_empty() {
                issues.push(RecipeIssue::MissingManualEditOperation(edit.id.clone()));
            }
            if edit.target.trim().is_empty() {
                issues.push(RecipeIssue::MissingManualEditTarget(edit.id.clone()));
            }
            for digest in [
                edit.parent_score_sha256.as_deref(),
                edit.result_score_sha256.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                if !is_sha256(digest) {
                    issues.push(RecipeIssue::InvalidSha256(digest.to_owned()));
                }
            }
        }
        for (index, decision) in self.cognitive_decisions.iter().enumerate() {
            if decision.sequence as usize != index {
                issues.push(RecipeIssue::NonContiguousDecisionSequence {
                    expected: index as u32,
                    found: decision.sequence,
                });
            }
            if decision
                .cognitive_session
                .as_ref()
                .is_some_and(|session| !session.is_valid())
            {
                issues.push(RecipeIssue::InvalidCognitiveSession(decision.sequence));
            }
        }
        issues
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

/// Exported artifact kinds recorded by one version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactKind {
    SymbolicScore,
    Midi,
    Audio,
    MusicXml,
    Stem,
    Other,
}

/// Digest and media identity for one exported artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub kind: ArtifactKind,
    pub file_name: String,
    pub media_type: String,
    pub sha256: String,
}

/// Version-level provenance. The recipe remains the canonical source input;
/// artifact digests prove which concrete files came from this version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceVersionProvenance {
    pub version_id: String,
    pub parent_version_id: Option<String>,
    pub created_at_unix_s: u64,
    pub recipe: PieceRecipe,
    pub artifacts: Vec<ArtifactRecord>,
}

impl PieceVersionProvenance {
    pub fn validate(&self) -> Vec<RecipeIssue> {
        let mut issues = self.recipe.validate();
        if self.version_id.trim().is_empty() {
            issues.push(RecipeIssue::MissingVersionId);
        }
        for artifact in &self.artifacts {
            if artifact.file_name.trim().is_empty() {
                issues.push(RecipeIssue::MissingArtifactFileName);
            }
            if !is_sha256(&artifact.sha256) {
                issues.push(RecipeIssue::InvalidSha256(artifact.sha256.clone()));
            }
        }
        issues
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecipeIssue {
    UnsupportedSchemaVersion(u32),
    MissingRendererName,
    InvalidSampleRate,
    MissingMuseVersion,
    MissingTheoryVersion,
    InvalidSha256(String),
    NonContiguousDecisionSequence { expected: u32, found: u32 },
    InvalidCognitiveSession(u32),
    MissingVersionId,
    MissingArtifactFileName,
    MissingManualEditId,
    DuplicateManualEditId(String),
    MissingManualEditOperation(String),
    MissingManualEditTarget(String),
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{
        ActionScope, CognitiveSection, InferenceEvidence, PredictedMusicalOutcome, SymbolicAction,
        SymbolicActionProposal, SymbolicMusicObservation,
    };
    use crate::intervention::InterventionDescriptor;
    use crate::musical_inference::MusicAction;
    use crate::musical_policy::MusicalPolicyPreference;
    use symthaea_music_theory::Style;

    fn trace() -> CognitiveDecisionTrace {
        CognitiveDecisionTrace {
            observation: SymbolicMusicObservation {
                section: CognitiveSection::Development,
                active_goal: None,
                goal_urgency: 0.4,
                valence: 0.0,
                arousal: 0.5,
                prediction_error: 0.2,
                consciousness_level: 0.5,
                dominant_harmony: 0,
                dominant_harmony_activation: 0.7,
                pending_obligations: 0,
                overdue_obligations: Vec::new(),
                obligation_demands: Vec::new(),
                obligation_pressure: 0.0,
            },
            inference: InferenceEvidence {
                source_action: MusicAction::RepeatMotif,
                free_energy: 0.1,
                prediction_error: 0.2,
                surprise: 0.1,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            proposal: SymbolicActionProposal {
                action: SymbolicAction::DevelopMotif,
                driving_obligation_id: None,
                supporting_obligation_ids: Vec::new(),
                deferred_obligation_ids: Vec::new(),
                scope: ActionScope::CurrentPhrase,
                preserve: Vec::new(),
                urgency: 0.4,
                confidence: 0.5,
                rationale: vec!["motif development".into()],
            },
            predicted_outcome: PredictedMusicalOutcome {
                tension_delta: 0.1,
                density_delta: 0.0,
                familiarity_delta: 0.1,
                tonal_displacement_delta: 0.0,
            },
        }
    }

    #[test]
    fn recipe_carries_the_complete_intent_and_resolved_spec() {
        let intent = MusicalIntent {
            seed: 42,
            bars: 8,
            ..MusicalIntent::default()
        };
        let recipe = PieceRecipe::new(
            intent,
            Style::Nocturne.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );

        assert_eq!(recipe.intent, intent);
        assert_eq!(recipe.resolved_spec.name, "Nocturne");
        assert!(recipe.is_valid());
    }

    #[test]
    fn recipe_records_exact_musical_state_and_reports_external_gaps() {
        let state = MusicalState {
            dopamine: 0.8,
            arousal: 0.7,
            ..MusicalState::default()
        };
        let recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0")
                .with_renderer_version("native-v1")
                .with_source_revisions("muse-commit", "theory-commit"),
        )
        .with_initial_musical_state(state.clone());

        assert_eq!(recipe.initial_musical_state, state);
        assert_eq!(
            recipe.reproduction_gaps(),
            vec![ReproductionGap::RenderEnvironmentDigest]
        );
    }

    #[test]
    fn schema_v1_recipes_deserialize_with_safe_defaults() {
        let recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        let mut legacy = serde_json::to_value(recipe).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.insert("schema_version".into(), serde_json::json!(1));
        object.remove("initial_musical_state");
        object.remove("manual_edits");
        let renderer = object
            .get_mut("renderer")
            .and_then(serde_json::Value::as_object_mut)
            .unwrap();
        for field in [
            "renderer_version",
            "muse_source_revision",
            "theory_source_revision",
            "renderer_binary_sha256",
            "render_environment_sha256",
        ] {
            renderer.remove(field);
        }

        let restored: PieceRecipe = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.schema_version, 1);
        assert_eq!(restored.initial_musical_state, MusicalState::default());
        assert!(restored.manual_edits.is_empty());
        assert!(restored.is_valid());
    }

    #[test]
    fn structured_manual_edits_are_validated_and_indexed() {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        recipe.record_manual_edit(ManualEditRecord {
            id: "edit-1".into(),
            operation: "transpose".into(),
            target: "recapitulation.primary".into(),
            parameters: BTreeMap::from([("semitones".into(), "-2".into())]),
            parent_score_sha256: None,
            result_score_sha256: None,
        });

        assert_eq!(recipe.manual_edit_ids, vec!["edit-1"]);
        assert_eq!(recipe.manual_edits[0].operation, "transpose");
        assert!(recipe.is_valid());
    }

    #[test]
    fn observed_outcomes_are_retained_with_channel_errors() {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        let sequence = recipe.record_decision(trace());
        recipe.cognitive_decisions[sequence as usize].observe(ObservedMusicalOutcome {
            tension_delta: 0.0,
            density_delta: 0.0,
            familiarity_delta: 0.1,
            tonal_displacement_delta: 0.0,
        });

        assert_eq!(sequence, 0);
        assert!(recipe.cognitive_decisions[0].prediction_error.is_some());
        assert!(recipe.is_valid());
    }

    #[test]
    fn artist_response_records_the_selected_alternative() {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        let sequence = recipe.record_decision(trace());
        recipe
            .record_artist_response(
                sequence,
                DecisionDisposition::Edited,
                Some("alternative-b".into()),
                Some("kept the harmony, shortened the ending".into()),
            )
            .unwrap();

        let decision = &recipe.cognitive_decisions[sequence as usize];
        assert_eq!(decision.disposition, DecisionDisposition::Edited);
        assert_eq!(
            decision.selected_alternative_id.as_deref(),
            Some("alternative-b")
        );
        assert!(decision.artist_note.is_some());
    }

    #[test]
    fn accepted_response_requires_a_concrete_alternative() {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Classical.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        let sequence = recipe.record_decision(trace());
        assert_eq!(
            recipe.record_artist_response(sequence, DecisionDisposition::Accepted, None, None,),
            Err(DecisionResponseError::MissingSelectedAlternative)
        );
    }

    #[test]
    fn malformed_artifact_digests_are_rejected() {
        let provenance = PieceVersionProvenance {
            version_id: "v1".into(),
            parent_version_id: None,
            created_at_unix_s: 0,
            recipe: PieceRecipe::new(
                MusicalIntent::default(),
                Style::Classical.spec(),
                RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
            ),
            artifacts: vec![ArtifactRecord {
                kind: ArtifactKind::Audio,
                file_name: "piece.wav".into(),
                media_type: "audio/wav".into(),
                sha256: "not-a-digest".into(),
            }],
        };

        assert_eq!(
            provenance.validate(),
            vec![RecipeIssue::InvalidSha256("not-a-digest".into())]
        );
    }

    #[test]
    fn preview_is_not_misrepresented_as_artist_acceptance() {
        let mut recipe = PieceRecipe::new(
            MusicalIntent::default(),
            Style::Sonata.spec(),
            RendererRecipe::new("native", 48_000, "0.1.0", "0.1.0"),
        );
        let sequence = recipe.record_decision(trace());
        recipe
            .record_preview(sequence, "literal-register".into())
            .unwrap();
        assert_eq!(
            recipe.cognitive_decisions[0].disposition,
            DecisionDisposition::Previewed
        );
        assert_eq!(
            recipe.cognitive_decisions[0]
                .selected_alternative_id
                .as_deref(),
            Some("literal-register")
        );
    }
}
