//! Blind agent-facing episode protocol for the finite open-world microcosms.
//!
//! AgentView contains public task metadata and observed evidence only. Truth
//! remains in the evaluator-side WorldState. This is a protocol boundary, not
//! an OS sandbox: run the agent in a separate process and never give it the
//! evaluator handle or evaluator score before submission.

use crate::{HiddenWorld, Hypothesis, Mechanism, MeasurementMap, ObservationLog};
use crate::{interaction, numeric};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const PROTOCOL_VERSION: &str = "symthaea-open-world-research-episode-v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldFamily {
    BooleanMechanism,
    NumericMechanism,
    TwoInputResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureAccess {
    BooleanWithheld,
    BooleanSupplied(Mechanism),
    NumericWithheld,
    NumericSupplied(numeric::Mechanism),
    InteractionWithheld,
    InteractionSupplied(interaction::Structure),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "values", rename_all = "snake_case")]
pub enum InterventionDomain {
    Boolean([bool; 2]),
    Integer(Vec<i64>),
    IntegerPair(Vec<i64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicsContract {
    StatefulAcrossProbes,
    IndependentProbes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicTaskManifest {
    pub protocol_version: String,
    pub episode_id: String,
    pub world_family: WorldFamily,
    pub structure_access: StructureAccess,
    pub intervention_domain: InterventionDomain,
    pub dynamics: DynamicsContract,
    pub max_probe_actions: usize,
    pub probe_cost_units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum AgentIntervention {
    Boolean(bool),
    Numeric(i64),
    Pair([i64; 2]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum AgentMeasurement {
    Boolean(bool),
    Numeric(i64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRecord {
    pub evidence_id: String,
    pub action_index: usize,
    pub intervention: AgentIntervention,
    pub measurement: AgentMeasurement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentView {
    pub task: PublicTaskManifest,
    pub observations: Vec<ObservationRecord>,
    pub remaining_probe_actions: usize,
    /// SHA-256 over canonical serde_json bytes of (task, observations).
    pub evidence_prefix_sha256: String,
    pub status: EpisodeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeStatus {
    Open,
    Submitted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", content = "payload", rename_all = "snake_case")]
pub enum AgentAction {
    Probe { intervention: AgentIntervention },
    Submit { submission: AgentSubmission },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimDisposition {
    Identified,
    Ambiguous,
    NoCompatibleModel,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", content = "claim", rename_all = "snake_case")]
pub enum AgentClaim {
    Boolean {
        compatible_models: Vec<Hypothesis>,
        disposition: ClaimDisposition,
    },
    Numeric {
        compatible_models: Vec<numeric::ModelHypothesis>,
        disposition: ClaimDisposition,
    },
    Interaction {
        compatible_models: Vec<interaction::ModelHypothesis>,
        disposition: ClaimDisposition,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSubmission {
    pub claim: AgentClaim,
    /// Pins the submission to the exact public task/evidence prefix it consumed.
    pub evidence_prefix_sha256: String,
    /// References must resolve to observations actually returned in this episode.
    pub cited_evidence_ids: Vec<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentResponse {
    pub status: EpisodeStatus,
    pub view: AgentView,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetScore {
    pub claimed_model_count: usize,
    pub oracle_model_count: usize,
    pub correct_model_count: usize,
    pub exact_version_space_match: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorScore {
    pub evidence_prefix_sha256: String,
    pub set_score: SetScore,
    pub decision_correct: bool,
    pub abstained: bool,
    pub evidence_references_valid: bool,
    pub cited_evidence_count: usize,
    pub observed_evidence_count: usize,
    pub probe_count: usize,
}

/// Truth-bearing state. This type is private to the episode runner and is
/// deliberately absent from AgentView and every wire-protocol response.
enum WorldState {
    Boolean(HiddenWorld),
    Numeric(numeric::NumericWorld),
    Interaction(interaction::InteractionWorld),
}

/// A runner owns one hidden reference world. Give an agent only AgentView
/// values and accept AgentAction values over a process boundary.
pub struct BlindEpisode {
    manifest: PublicTaskManifest,
    world: WorldState,
    observations: Vec<ObservationRecord>,
    submission: Option<AgentSubmission>,
}

impl BlindEpisode {
    /// Evaluator-side constructor. Generating truth is never copied into the
    /// agent-facing manifest.
    pub fn new_boolean(
        episode_id: impl Into<String>,
        mechanism: Mechanism,
        measurement_map: MeasurementMap,
        initial_hidden_state: Option<bool>,
        access: StructureAccess,
        max_probe_actions: usize,
    ) -> Result<Self, ProtocolError> {
        match access {
            StructureAccess::BooleanWithheld => {}
            StructureAccess::BooleanSupplied(supplied) if supplied == mechanism => {}
            StructureAccess::BooleanSupplied(_) => return Err(ProtocolError::SuppliedStructureMismatch),
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let world = HiddenWorld::new(mechanism, measurement_map, initial_hidden_state)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(
            episode_id.into(),
            WorldFamily::BooleanMechanism,
            access,
            InterventionDomain::Boolean([false, true]),
            DynamicsContract::StatefulAcrossProbes,
            max_probe_actions,
            WorldState::Boolean(world),
        )
    }

    /// Evaluator-side constructor. The full generating model remains private.
    pub fn new_numeric(
        episode_id: impl Into<String>,
        truth: numeric::ModelHypothesis,
        access: StructureAccess,
        max_probe_actions: usize,
    ) -> Result<Self, ProtocolError> {
        match access {
            StructureAccess::NumericWithheld => {}
            StructureAccess::NumericSupplied(supplied) if supplied == truth.mechanism => {}
            StructureAccess::NumericSupplied(_) => return Err(ProtocolError::SuppliedStructureMismatch),
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let world = numeric::NumericWorld::new(truth)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(
            episode_id.into(),
            WorldFamily::NumericMechanism,
            access,
            InterventionDomain::Integer(numeric::INPUT_LEVELS.to_vec()),
            DynamicsContract::StatefulAcrossProbes,
            max_probe_actions,
            WorldState::Numeric(world),
        )
    }

    /// Evaluator-side constructor. The generating coefficients remain private.
    pub fn new_interaction(
        episode_id: impl Into<String>,
        truth: interaction::ModelHypothesis,
        access: StructureAccess,
        max_probe_actions: usize,
    ) -> Result<Self, ProtocolError> {
        match access {
            StructureAccess::InteractionWithheld => {}
            StructureAccess::InteractionSupplied(supplied) if supplied == truth.structure => {}
            StructureAccess::InteractionSupplied(_) => return Err(ProtocolError::SuppliedStructureMismatch),
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let world = interaction::InteractionWorld::new(truth)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(
            episode_id.into(),
            WorldFamily::TwoInputResponse,
            access,
            InterventionDomain::IntegerPair(interaction::INPUT_LEVELS.to_vec()),
            DynamicsContract::IndependentProbes,
            max_probe_actions,
            WorldState::Interaction(world),
        )
    }

    fn new_internal(
        episode_id: String,
        world_family: WorldFamily,
        structure_access: StructureAccess,
        intervention_domain: InterventionDomain,
        dynamics: DynamicsContract,
        max_probe_actions: usize,
        world: WorldState,
    ) -> Result<Self, ProtocolError> {
        if episode_id.trim().is_empty() || max_probe_actions == 0 {
            return Err(ProtocolError::InvalidManifest);
        }
        let manifest = PublicTaskManifest {
            protocol_version: PROTOCOL_VERSION.to_string(),
            episode_id,
            world_family,
            structure_access,
            intervention_domain,
            dynamics,
            max_probe_actions,
            probe_cost_units: 1,
        };
        Ok(Self {
            manifest,
            world,
            observations: Vec::new(),
            submission: None,
        })
    }

    /// Returns public task metadata, observations, remaining budget and digest.
    pub fn view(&self) -> Result<AgentView, ProtocolError> {
        let bytes = serde_json::to_vec(&(&self.manifest, &self.observations))
            .map_err(|error| ProtocolError::SerializationFailure(error.to_string()))?;
        let digest = Sha256::digest(bytes);
        let hex = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        Ok(AgentView {
            task: self.manifest.clone(),
            observations: self.observations.clone(),
            remaining_probe_actions: self
                .manifest
                .max_probe_actions
                .saturating_sub(self.observations.len()),
            evidence_prefix_sha256: format!("sha256:{hex}"),
            status: self.status(),
        })
    }

    pub fn status(&self) -> EpisodeStatus {
        if self.submission.is_some() {
            EpisodeStatus::Submitted
        } else {
            EpisodeStatus::Open
        }
    }

    /// Applies one agent action. Probes consume one unit and return one
    /// measurement. A submission is terminal. No oracle result is returned.
    pub fn apply(&mut self, action: AgentAction) -> Result<AgentResponse, ProtocolError> {
        if self.submission.is_some() {
            return Err(ProtocolError::EpisodeClosed);
        }

        match action {
            AgentAction::Probe { intervention } => {
                if self.observations.len() >= self.manifest.max_probe_actions {
                    return Err(ProtocolError::ProbeBudgetExhausted);
                }
                let measurement = match (&mut self.world, &intervention) {
                    (WorldState::Boolean(world), AgentIntervention::Boolean(value)) => {
                        let log = world.run(&[*value]);
                        AgentMeasurement::Boolean(log.measurements[0])
                    }
                    (WorldState::Numeric(world), AgentIntervention::Numeric(value)) => {
                        let log = world
                            .run(&[*value])
                            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
                        AgentMeasurement::Numeric(log.measurements[0])
                    }
                    (WorldState::Interaction(world), AgentIntervention::Pair(values)) => {
                        let log = world
                            .run(&[*values])
                            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
                        AgentMeasurement::Numeric(log.measurements[0])
                    }
                    _ => return Err(ProtocolError::WrongInterventionFamily),
                };
                self.observations.push(ObservationRecord {
                    evidence_id: format!("E{:04}", self.observations.len() + 1),
                    action_index: self.observations.len() + 1,
                    intervention,
                    measurement,
                });
            }
            AgentAction::Submit { submission } => {
                self.validate_submission(&submission)?;
                self.submission = Some(submission);
            }
        }

        let view = self.view()?;
        Ok(AgentResponse {
            status: view.status,
            view,
        })
    }

    pub fn submission(&self) -> Option<&AgentSubmission> {
        self.submission.as_ref()
    }

    fn validate_submission(&self, submission: &AgentSubmission) -> Result<(), ProtocolError> {
        if submission.evidence_prefix_sha256 != self.view()?.evidence_prefix_sha256 {
            return Err(ProtocolError::EvidencePrefixMismatch);
        }
        if self.observations.is_empty() || submission.cited_evidence_ids.is_empty() {
            return Err(ProtocolError::EmptyEvidenceReferences);
        }
        let known: BTreeSet<&str> = self
            .observations
            .iter()
            .map(|record| record.evidence_id.as_str())
            .collect();
        let mut cited = BTreeSet::new();
        for reference in &submission.cited_evidence_ids {
            if !cited.insert(reference.as_str()) {
                return Err(ProtocolError::DuplicateEvidenceReference);
            }
            if !known.contains(reference.as_str()) {
                return Err(ProtocolError::UnknownEvidenceReference);
            }
        }

        match (&submission.claim, self.manifest.structure_access) {
            (AgentClaim::Boolean { compatible_models, disposition }, StructureAccess::BooleanWithheld) => {
                validate_models(compatible_models, &crate::candidate_hypotheses(crate::StructureAccess::Withheld))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (AgentClaim::Boolean { compatible_models, disposition }, StructureAccess::BooleanSupplied(mechanism)) => {
                validate_models(compatible_models, &crate::candidate_hypotheses(crate::StructureAccess::Supplied(mechanism)))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (AgentClaim::Numeric { compatible_models, disposition }, StructureAccess::NumericWithheld) => {
                validate_models(compatible_models, &numeric::candidate_models(numeric::StructureAccess::Withheld))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (AgentClaim::Numeric { compatible_models, disposition }, StructureAccess::NumericSupplied(mechanism)) => {
                validate_models(compatible_models, &numeric::candidate_models(numeric::StructureAccess::Supplied(mechanism)))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (AgentClaim::Interaction { compatible_models, disposition }, StructureAccess::InteractionWithheld) => {
                validate_models(compatible_models, &interaction::candidate_models(interaction::StructureAccess::Withheld))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (AgentClaim::Interaction { compatible_models, disposition }, StructureAccess::InteractionSupplied(structure)) => {
                validate_models(compatible_models, &interaction::candidate_models(interaction::StructureAccess::Supplied(structure)))?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            _ => Err(ProtocolError::ClaimFamilyMismatch),
        }
    }

    /// Evaluator-only: call after submission, inside the trusted runner process.
    /// Returns aggregate scores, not the oracle's hidden model set.
    pub fn evaluate_after_submission(&self) -> Result<EvaluatorScore, ProtocolError> {
        let submission = self.submission.as_ref().ok_or(ProtocolError::NotSubmitted)?;
        let evidence_references_valid = {
            let known: BTreeSet<&str> = self
                .observations
                .iter()
                .map(|record| record.evidence_id.as_str())
                .collect();
            !submission.cited_evidence_ids.is_empty()
                && submission
                    .cited_evidence_ids
                    .iter()
                    .all(|reference| known.contains(reference.as_str()))
        };

        let (set_score, disposition) = match (&self.world, &submission.claim) {
            (WorldState::Boolean(_), AgentClaim::Boolean { compatible_models, disposition }) => {
                let log = ObservationLog {
                    interventions: self.observations.iter().map(|record| record.intervention).map(|intervention| match intervention {
                        AgentIntervention::Boolean(value) => value,
                        _ => unreachable!("action family is checked before a record is appended"),
                    }).collect(),
                    measurements: self.observations.iter().map(|record| record.measurement).map(|measurement| match measurement {
                        AgentMeasurement::Boolean(value) => value,
                        _ => unreachable!("measurement family follows the world family"),
                    }).collect(),
                };
                let model_access = match self.manifest.structure_access {
                    StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
                    StructureAccess::BooleanSupplied(mechanism) => crate::StructureAccess::Supplied(mechanism),
                    _ => return Err(ProtocolError::ClaimFamilyMismatch),
                };
                let oracle = crate::discover(&log, model_access)
                    .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
                (compare_model_sets(compatible_models, &oracle.compatible_hypotheses)?, *disposition)
            }
            (WorldState::Numeric(_), AgentClaim::Numeric { compatible_models, disposition }) => {
                let log = numeric::ObservationLog {
                    interventions: self.observations.iter().map(|record| record.intervention).map(|intervention| match intervention {
                        AgentIntervention::Numeric(value) => value,
                        _ => unreachable!("action family is checked before a record is appended"),
                    }).collect(),
                    measurements: self.observations.iter().map(|record| record.measurement).map(|measurement| match measurement {
                        AgentMeasurement::Numeric(value) => value,
                        _ => unreachable!("measurement family follows the world family"),
                    }).collect(),
                };
                let access = match self.manifest.structure_access {
                    StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
                    StructureAccess::NumericSupplied(mechanism) => numeric::StructureAccess::Supplied(mechanism),
                    _ => return Err(ProtocolError::ClaimFamilyMismatch),
                };
                let oracle = numeric::discover(&log, access)
                    .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
                (compare_model_sets(compatible_models, &oracle.compatible_models)?, *disposition)
            }
            (WorldState::Interaction(_), AgentClaim::Interaction { compatible_models, disposition }) => {
                let log = interaction::ObservationLog {
                    interventions: self.observations.iter().map(|record| record.intervention).map(|intervention| match intervention {
                        AgentIntervention::Pair(value) => value,
                        _ => unreachable!("action family is checked before a record is appended"),
                    }).collect(),
                    measurements: self.observations.iter().map(|record| record.measurement).map(|measurement| match measurement {
                        AgentMeasurement::Numeric(value) => value,
                        _ => unreachable!("measurement family follows the world family"),
                    }).collect(),
                };
                let access = match self.manifest.structure_access {
                    StructureAccess::InteractionWithheld => interaction::StructureAccess::Withheld,
                    StructureAccess::InteractionSupplied(structure) => interaction::StructureAccess::Supplied(structure),
                    _ => return Err(ProtocolError::ClaimFamilyMismatch),
                };
                let oracle = interaction::discover(&log, access)
                    .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
                (compare_model_sets(compatible_models, &oracle.compatible_models)?, *disposition)
            }
            _ => return Err(ProtocolError::ClaimFamilyMismatch),
        };

        let expected = match set_score.oracle_model_count {
            0 => ClaimDisposition::NoCompatibleModel,
            1 => ClaimDisposition::Identified,
            _ => ClaimDisposition::Ambiguous,
        };
        Ok(EvaluatorScore {
            evidence_prefix_sha256: submission.evidence_prefix_sha256.clone(),
            decision_correct: disposition == expected,
            abstained: disposition == ClaimDisposition::Abstain,
            set_score,
            evidence_references_valid,
            cited_evidence_count: submission.cited_evidence_ids.len(),
            observed_evidence_count: self.observations.len(),
            probe_count: self.observations.len(),
        })
    }
}

fn validate_models<T: Serialize>(claimed: &[T], allowed: &[T]) -> Result<(), ProtocolError> {
    let claimed_set = serialized_set(claimed)?;
    let allowed_set = serialized_set(allowed)?;
    if claimed_set.len() != claimed.len() {
        return Err(ProtocolError::DuplicateClaimModel);
    }
    if !claimed_set.is_subset(&allowed_set) {
        return Err(ProtocolError::UnsupportedModelClaim);
    }
    Ok(())
}

fn validate_disposition(
    model_count: usize,
    disposition: ClaimDisposition,
) -> Result<(), ProtocolError> {
    let valid = match disposition {
        ClaimDisposition::Identified => model_count == 1,
        ClaimDisposition::Ambiguous => model_count > 1,
        ClaimDisposition::NoCompatibleModel => model_count == 0,
        ClaimDisposition::Abstain => true,
    };
    if valid {
        Ok(())
    } else {
        Err(ProtocolError::ClaimDispositionMismatch)
    }
}

fn serialized_set<T: Serialize>(models: &[T]) -> Result<BTreeSet<String>, ProtocolError> {
    models
        .iter()
        .map(|model| {
            serde_json::to_string(model)
                .map_err(|error| ProtocolError::SerializationFailure(error.to_string()))
        })
        .collect()
}

fn compare_model_sets<T: Serialize>(
    claimed: &[T],
    oracle: &[T],
) -> Result<SetScore, ProtocolError> {
    let claimed_set = serialized_set(claimed)?;
    let oracle_set = serialized_set(oracle)?;
    let correct_model_count = claimed_set.intersection(&oracle_set).count();
    Ok(SetScore {
        claimed_model_count: claimed_set.len(),
        oracle_model_count: oracle_set.len(),
        correct_model_count,
        exact_version_space_match: claimed_set == oracle_set,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    InvalidManifest,
    AccessFamilyMismatch,
    SuppliedStructureMismatch,
    WrongInterventionFamily,
    ProbeBudgetExhausted,
    EpisodeClosed,
    ClaimFamilyMismatch,
    DuplicateClaimModel,
    UnsupportedModelClaim,
    ClaimDispositionMismatch,
    EmptyEvidenceReferences,
    DuplicateEvidenceReference,
    UnknownEvidenceReference,
    EvidencePrefixMismatch,
    NotSubmitted,
    WorldFailure(String),
    SerializationFailure(String),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_numeric_truth() -> numeric::ModelHypothesis {
        numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::Identity,
        }
    }

    fn claim_disposition(count: usize) -> ClaimDisposition {
        match count {
            0 => ClaimDisposition::NoCompatibleModel,
            1 => ClaimDisposition::Identified,
            _ => ClaimDisposition::Ambiguous,
        }
    }

    #[test]
    fn public_view_does_not_serialize_boolean_world_truth() {
        let episode = BlindEpisode::new_boolean(
            "blind-boolean-01",
            Mechanism::HiddenLag,
            MeasurementMap::Invert,
            Some(true),
            StructureAccess::BooleanWithheld,
            4,
        )
        .unwrap();
        let serialized = serde_json::to_string(&episode.view().unwrap()).unwrap();
        assert!(!serialized.contains("hidden_lag"));
        assert!(!serialized.contains("invert"));
        assert!(!serialized.contains("initial_hidden_state"));
        assert!(!serialized.contains("measurement_map"));
        assert!(!serialized.contains("world_truth"));
    }

    #[test]
    fn supplied_structure_is_exposed_only_when_the_treatment_allows_it() {
        let episode = BlindEpisode::new_boolean(
            "blind-boolean-a0",
            Mechanism::HiddenLag,
            MeasurementMap::Identity,
            Some(false),
            StructureAccess::BooleanSupplied(Mechanism::HiddenLag),
            2,
        )
        .unwrap();
        let view = episode.view().unwrap();
        assert_eq!(
            view.task.structure_access,
            StructureAccess::BooleanSupplied(Mechanism::HiddenLag)
        );
        let serialized = serde_json::to_string(&view).unwrap();
        assert!(!serialized.contains("identity"));
    }

    #[test]
    fn probes_are_single_step_budgeted_and_cannot_be_repeated_after_submission() {
        let mut episode = BlindEpisode::new_boolean(
            "blind-budget-01",
            Mechanism::Direct,
            MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            1,
        )
        .unwrap();
        episode.apply(AgentAction::Probe {
            intervention: AgentIntervention::Boolean(true),
        }).unwrap();
        assert_eq!(
            episode.apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(false),
            }),
            Err(ProtocolError::ProbeBudgetExhausted)
        );

        let view = episode.view().unwrap();
        let hypotheses = crate::candidate_hypotheses(crate::StructureAccess::Withheld);
        let submission = AgentSubmission {
            claim: AgentClaim::Boolean {
                compatible_models: hypotheses,
                disposition: ClaimDisposition::Ambiguous,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256.clone(),
            cited_evidence_ids: vec!["E0001".to_string()],
            rationale: "One observation does not identify the mechanism.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        assert_eq!(episode.status(), EpisodeStatus::Submitted);
        assert_eq!(
            episode.apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(false),
            }),
            Err(ProtocolError::EpisodeClosed)
        );
        assert_eq!(view.remaining_probe_actions, 0);
    }

    #[test]
    fn wrong_intervention_family_is_rejected_without_consuming_budget() {
        let mut episode = BlindEpisode::new_numeric(
            "blind-numeric-01",
            direct_numeric_truth(),
            StructureAccess::NumericWithheld,
            2,
        )
        .unwrap();
        assert_eq!(
            episode.apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            }),
            Err(ProtocolError::WrongInterventionFamily)
        );
        assert_eq!(episode.view().unwrap().remaining_probe_actions, 2);
    }

    #[test]
    fn claims_must_reference_real_evidence_and_stay_inside_the_public_model_family() {
        let mut episode = BlindEpisode::new_numeric(
            "blind-numeric-02",
            direct_numeric_truth(),
            StructureAccess::NumericWithheld,
            3,
        )
        .unwrap();
        for value in [-1, 0, 1] {
            episode.apply(AgentAction::Probe {
                intervention: AgentIntervention::Numeric(value),
            }).unwrap();
        }

        let log = numeric::ObservationLog {
            interventions: vec![-1, 0, 1],
            measurements: vec![-2, 0, 2],
        };
        let oracle = numeric::discover(&log, numeric::StructureAccess::Withheld).unwrap();
        let count = oracle.compatible_models.len();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                compatible_models: oracle.compatible_models,
                disposition: claim_disposition(count),
            },
            evidence_prefix_sha256: episode.view().unwrap().evidence_prefix_sha256,
            cited_evidence_ids: vec!["E0001".to_string(), "E0002".to_string(), "E0003".to_string()],
            rationale: "The evidence identifies a compatible version space.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
        assert!(score.evidence_references_valid);
        assert_eq!(score.probe_count, 3);
    }

    #[test]
    fn invalid_evidence_reference_does_not_close_the_episode() {
        let mut episode = BlindEpisode::new_boolean(
            "blind-evidence-01",
            Mechanism::Direct,
            MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            2,
        )
        .unwrap();
        episode.apply(AgentAction::Probe {
            intervention: AgentIntervention::Boolean(true),
        }).unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Boolean {
                compatible_models: crate::candidate_hypotheses(crate::StructureAccess::Withheld),
                disposition: ClaimDisposition::Ambiguous,
            },
            evidence_prefix_sha256: episode.view().unwrap().evidence_prefix_sha256,
            cited_evidence_ids: vec!["E9999".to_string()],
            rationale: "Invalid reference control.".to_string(),
        };
        assert_eq!(
            episode.apply(AgentAction::Submit { submission }),
            Err(ProtocolError::UnknownEvidenceReference)
        );
        assert_eq!(episode.status(), EpisodeStatus::Open);
    }

    #[test]
    fn interaction_protocol_preserves_full_factorial_probe_semantics() {
        let truth = interaction::ModelHypothesis {
            structure: interaction::Structure::BilinearInteraction,
            bias: 0,
            weight_u: 1,
            weight_v: 1,
            interaction_weight: 1,
        };
        let mut episode = BlindEpisode::new_interaction(
            "blind-interaction-01",
            truth,
            StructureAccess::InteractionWithheld,
            interaction::FACTORIAL_DESIGN.len(),
        ).unwrap();
        for input in interaction::FACTORIAL_DESIGN {
            episode.apply(AgentAction::Probe {
                intervention: AgentIntervention::Pair(input),
            }).unwrap();
        }
        let current = episode.view().unwrap();
        let log = interaction::ObservationLog {
            interventions: current.observations.iter().map(|record| match record.intervention {
                AgentIntervention::Pair(value) => value,
                _ => unreachable!(),
            }).collect(),
            measurements: current.observations.iter().map(|record| match record.measurement {
                AgentMeasurement::Numeric(value) => value,
                _ => unreachable!(),
            }).collect(),
        };
        let oracle = interaction::discover(&log, interaction::StructureAccess::Withheld).unwrap();
        let count = oracle.compatible_models.len();
        let submission = AgentSubmission {
            claim: AgentClaim::Interaction {
                compatible_models: oracle.compatible_models,
                disposition: claim_disposition(count),
            },
            evidence_prefix_sha256: current.evidence_prefix_sha256.clone(),
            cited_evidence_ids: current.observations.iter().map(|record| record.evidence_id.clone()).collect(),
            rationale: "The factorial design distinguishes variable relevance and interaction.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
    }

    #[test]
    fn evidence_prefix_digest_changes_when_public_evidence_changes() {
        let mut episode = BlindEpisode::new_boolean(
            "blind-digest-01",
            Mechanism::Direct,
            MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            2,
        ).unwrap();
        let before = episode.view().unwrap().evidence_prefix_sha256;
        episode.apply(AgentAction::Probe {
            intervention: AgentIntervention::Boolean(true),
        }).unwrap();
        let after = episode.view().unwrap().evidence_prefix_sha256;
        assert_ne!(before, after);
    }
}
