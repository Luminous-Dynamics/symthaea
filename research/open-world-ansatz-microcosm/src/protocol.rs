//! Blind agent-facing episode protocol for the finite open-world microcosms.
//!
//! AgentView contains public task metadata and observed evidence only. Truth
//! remains in the evaluator-side WorldState. This is a protocol boundary, not
//! an OS sandbox: run the agent in a separate process and never give it the
//! evaluator handle or evaluator score before submission.

use crate::{HiddenWorld, Hypothesis, MeasurementMap, Mechanism, ObservationLog};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
pub struct StructureScore {
    pub claimed_structure_count: usize,
    pub oracle_structure_count: usize,
    pub correct_structure_count: usize,
    pub exact_structure_set_match: bool,
    pub claimed_structure_identified: bool,
    pub oracle_structure_identified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorScore {
    pub evidence_prefix_sha256: String,
    /// Exact parameter-struct membership in the legal candidate version space.
    /// False may still be behaviorally represented by an equivalent parameterization.
    pub generating_model_in_version_space: bool,
    /// Whether some legal candidate has identical predictions over every length-4
    /// intervention sequence in the finite numeric input language. For Boolean and
    /// interaction episodes the generating truth is inside the declared grammar.
    pub generating_truth_behaviorally_represented: bool,
    pub set_score: SetScore,
    /// Scores structural-family recovery independently of coefficients and sensor parameters.
    pub structure_score: StructureScore,
    /// Whether an exact reported version space makes abstention epistemically warranted.
    pub safe_abstention: bool,
    /// Whether the agent abstained despite an exactly identified parameterized model.
    pub unnecessary_abstention: bool,
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

#[derive(Clone, Copy)]
enum GeneratingTruth {
    Boolean(Hypothesis),
    Numeric(numeric::ModelHypothesis),
    Interaction(interaction::ModelHypothesis),
}

/// A runner owns one hidden reference world. Give an agent only AgentView
/// values and accept AgentAction values over a process boundary.
pub struct BlindEpisode {
    manifest: PublicTaskManifest,
    world: WorldState,
    generating_truth: GeneratingTruth,
    /// Only explicit open-support episodes may treat omitted generating truth
    /// as a valid "no compatible model" outcome.
    allow_outside_candidate_grammar: bool,
    observations: Vec<ObservationRecord>,
    submission: Option<AgentSubmission>,
}

struct EpisodeInit {
    episode_id: String,
    world_family: WorldFamily,
    structure_access: StructureAccess,
    intervention_domain: InterventionDomain,
    dynamics: DynamicsContract,
    max_probe_actions: usize,
    world: WorldState,
    generating_truth: GeneratingTruth,
    allow_outside_candidate_grammar: bool,
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
            StructureAccess::BooleanSupplied(_) => {
                return Err(ProtocolError::SuppliedStructureMismatch);
            }
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let truth = Hypothesis::new(mechanism, measurement_map, initial_hidden_state)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        let world = HiddenWorld::new(mechanism, measurement_map, initial_hidden_state)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(EpisodeInit {
            episode_id: episode_id.into(),
            world_family: WorldFamily::BooleanMechanism,
            structure_access: access,
            intervention_domain: InterventionDomain::Boolean([false, true]),
            dynamics: DynamicsContract::StatefulAcrossProbes,
            max_probe_actions,
            world: WorldState::Boolean(world),
            generating_truth: GeneratingTruth::Boolean(truth),
            allow_outside_candidate_grammar: false,
        })
    }

    /// Evaluator-side constructor over the declared closed candidate grammar.
    pub fn new_numeric(
        episode_id: impl Into<String>,
        truth: numeric::ModelHypothesis,
        access: StructureAccess,
        max_probe_actions: usize,
    ) -> Result<Self, ProtocolError> {
        Self::new_numeric_with_support(episode_id, truth, access, max_probe_actions, false)
    }

    /// Evaluator-side open-support control. The world generator may use a
    /// valid parameter value outside the agent's declared candidate catalogue.
    /// The agent receives the same public task contract and is not told that
    /// the hidden truth is out of catalogue. Only this explicit constructor
    /// permits the evaluator to score a correct empty version space.
    pub fn new_numeric_open_support(
        episode_id: impl Into<String>,
        truth: numeric::ModelHypothesis,
        access: StructureAccess,
        max_probe_actions: usize,
    ) -> Result<Self, ProtocolError> {
        Self::new_numeric_with_support(episode_id, truth, access, max_probe_actions, true)
    }

    fn new_numeric_with_support(
        episode_id: impl Into<String>,
        truth: numeric::ModelHypothesis,
        access: StructureAccess,
        max_probe_actions: usize,
        allow_outside_candidate_grammar: bool,
    ) -> Result<Self, ProtocolError> {
        match access {
            StructureAccess::NumericWithheld => {}
            StructureAccess::NumericSupplied(supplied) if supplied == truth.mechanism => {}
            StructureAccess::NumericSupplied(_) => {
                return Err(ProtocolError::SuppliedStructureMismatch);
            }
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let world = numeric::NumericWorld::new(truth)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(EpisodeInit {
            episode_id: episode_id.into(),
            world_family: WorldFamily::NumericMechanism,
            structure_access: access,
            intervention_domain: InterventionDomain::Integer(numeric::INPUT_LEVELS.to_vec()),
            dynamics: DynamicsContract::StatefulAcrossProbes,
            max_probe_actions,
            world: WorldState::Numeric(world),
            generating_truth: GeneratingTruth::Numeric(truth),
            allow_outside_candidate_grammar,
        })
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
            StructureAccess::InteractionSupplied(_) => {
                return Err(ProtocolError::SuppliedStructureMismatch);
            }
            _ => return Err(ProtocolError::AccessFamilyMismatch),
        }
        let world = interaction::InteractionWorld::new(truth)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        Self::new_internal(EpisodeInit {
            episode_id: episode_id.into(),
            world_family: WorldFamily::TwoInputResponse,
            structure_access: access,
            intervention_domain: InterventionDomain::IntegerPair(
                interaction::INPUT_LEVELS.to_vec(),
            ),
            dynamics: DynamicsContract::IndependentProbes,
            max_probe_actions,
            world: WorldState::Interaction(world),
            generating_truth: GeneratingTruth::Interaction(truth),
            allow_outside_candidate_grammar: false,
        })
    }

    fn new_internal(init: EpisodeInit) -> Result<Self, ProtocolError> {
        let EpisodeInit {
            episode_id,
            world_family,
            structure_access,
            intervention_domain,
            dynamics,
            max_probe_actions,
            world,
            generating_truth,
            allow_outside_candidate_grammar,
        } = init;

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
            generating_truth,
            allow_outside_candidate_grammar,
            observations: Vec::new(),
            submission: None,
        })
    }

    /// Returns public task metadata, observations, remaining budget and digest.
    pub fn view(&self) -> Result<AgentView, ProtocolError> {
        let bytes = serde_json::to_vec(&(&self.manifest, &self.observations))
            .map_err(|error| ProtocolError::SerializationFailure(error.to_string()))?;
        let digest = Sha256::digest(bytes);
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
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
            (
                AgentClaim::Boolean {
                    compatible_models,
                    disposition,
                },
                StructureAccess::BooleanWithheld,
            ) => {
                validate_models(
                    compatible_models,
                    &reference_boolean_candidates(crate::StructureAccess::Withheld)?,
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (
                AgentClaim::Boolean {
                    compatible_models,
                    disposition,
                },
                StructureAccess::BooleanSupplied(mechanism),
            ) => {
                validate_models(
                    compatible_models,
                    &reference_boolean_candidates(crate::StructureAccess::Supplied(mechanism))?,
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (
                AgentClaim::Numeric {
                    compatible_models,
                    disposition,
                },
                StructureAccess::NumericWithheld,
            ) => {
                validate_models(
                    compatible_models,
                    &reference_numeric_candidates(numeric::StructureAccess::Withheld),
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (
                AgentClaim::Numeric {
                    compatible_models,
                    disposition,
                },
                StructureAccess::NumericSupplied(mechanism),
            ) => {
                validate_models(
                    compatible_models,
                    &reference_numeric_candidates(numeric::StructureAccess::Supplied(mechanism)),
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (
                AgentClaim::Interaction {
                    compatible_models,
                    disposition,
                },
                StructureAccess::InteractionWithheld,
            ) => {
                validate_models(
                    compatible_models,
                    &reference_interaction_candidates(interaction::StructureAccess::Withheld),
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            (
                AgentClaim::Interaction {
                    compatible_models,
                    disposition,
                },
                StructureAccess::InteractionSupplied(structure),
            ) => {
                validate_models(
                    compatible_models,
                    &reference_interaction_candidates(interaction::StructureAccess::Supplied(
                        structure,
                    )),
                )?;
                validate_disposition(compatible_models.len(), *disposition)
            }
            _ => Err(ProtocolError::ClaimFamilyMismatch),
        }
    }

    /// Evaluator-only: call after submission, inside the trusted runner process.
    /// Returns aggregate scores, not the oracle's hidden model set.
    pub fn evaluate_after_submission(&self) -> Result<EvaluatorScore, ProtocolError> {
        let submission = self
            .submission
            .as_ref()
            .ok_or(ProtocolError::NotSubmitted)?;
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

        let (set_score, structure_score, disposition, generating_model_in_version_space) =
            match (&self.world, &self.generating_truth, &submission.claim) {
                (
                    WorldState::Boolean(_),
                    GeneratingTruth::Boolean(truth),
                    AgentClaim::Boolean {
                        compatible_models,
                        disposition,
                    },
                ) => {
                    let log = ObservationLog {
                        interventions: self
                            .observations
                            .iter()
                            .map(|record| record.intervention)
                            .map(|intervention| match intervention {
                                AgentIntervention::Boolean(value) => value,
                                _ => unreachable!(
                                    "action family is checked before a record is appended"
                                ),
                            })
                            .collect(),
                        measurements: self
                            .observations
                            .iter()
                            .map(|record| record.measurement)
                            .map(|measurement| match measurement {
                                AgentMeasurement::Boolean(value) => value,
                                _ => unreachable!("measurement family follows the world family"),
                            })
                            .collect(),
                    };
                    let model_access = match self.manifest.structure_access {
                        StructureAccess::BooleanWithheld => crate::StructureAccess::Withheld,
                        StructureAccess::BooleanSupplied(mechanism) => {
                            crate::StructureAccess::Supplied(mechanism)
                        }
                        _ => return Err(ProtocolError::ClaimFamilyMismatch),
                    };
                    let oracle_models = reference_boolean_version_space(&log, model_access)?;
                    let truth_present = oracle_models.contains(truth);
                    if !truth_present && !self.allow_outside_candidate_grammar {
                        return Err(ProtocolError::GeneratingTruthMissingFromOracle);
                    }
                    (
                        compare_model_sets(compatible_models, &oracle_models)?,
                        compare_structure_sets(compatible_models, &oracle_models, |model| {
                            model.mechanism
                        }),
                        *disposition,
                        truth_present,
                    )
                }
                (
                    WorldState::Numeric(_),
                    GeneratingTruth::Numeric(truth),
                    AgentClaim::Numeric {
                        compatible_models,
                        disposition,
                    },
                ) => {
                    let log = numeric::ObservationLog {
                        interventions: self
                            .observations
                            .iter()
                            .map(|record| record.intervention)
                            .map(|intervention| match intervention {
                                AgentIntervention::Numeric(value) => value,
                                _ => unreachable!(
                                    "action family is checked before a record is appended"
                                ),
                            })
                            .collect(),
                        measurements: self
                            .observations
                            .iter()
                            .map(|record| record.measurement)
                            .map(|measurement| match measurement {
                                AgentMeasurement::Numeric(value) => value,
                                _ => unreachable!("measurement family follows the world family"),
                            })
                            .collect(),
                    };
                    let access = match self.manifest.structure_access {
                        StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
                        StructureAccess::NumericSupplied(mechanism) => {
                            numeric::StructureAccess::Supplied(mechanism)
                        }
                        _ => return Err(ProtocolError::ClaimFamilyMismatch),
                    };
                    let oracle_models = reference_numeric_version_space(&log, access);
                    let truth_present = oracle_models.contains(truth);
                    if !truth_present && !self.allow_outside_candidate_grammar {
                        return Err(ProtocolError::GeneratingTruthMissingFromOracle);
                    }
                    (
                        compare_model_sets(compatible_models, &oracle_models)?,
                        compare_structure_sets(compatible_models, &oracle_models, |model| {
                            model.mechanism
                        }),
                        *disposition,
                        truth_present,
                    )
                }
                (
                    WorldState::Interaction(_),
                    GeneratingTruth::Interaction(truth),
                    AgentClaim::Interaction {
                        compatible_models,
                        disposition,
                    },
                ) => {
                    let log = interaction::ObservationLog {
                        interventions: self
                            .observations
                            .iter()
                            .map(|record| record.intervention)
                            .map(|intervention| match intervention {
                                AgentIntervention::Pair(value) => value,
                                _ => unreachable!(
                                    "action family is checked before a record is appended"
                                ),
                            })
                            .collect(),
                        measurements: self
                            .observations
                            .iter()
                            .map(|record| record.measurement)
                            .map(|measurement| match measurement {
                                AgentMeasurement::Numeric(value) => value,
                                _ => unreachable!("measurement family follows the world family"),
                            })
                            .collect(),
                    };
                    let access = match self.manifest.structure_access {
                        StructureAccess::InteractionWithheld => {
                            interaction::StructureAccess::Withheld
                        }
                        StructureAccess::InteractionSupplied(structure) => {
                            interaction::StructureAccess::Supplied(structure)
                        }
                        _ => return Err(ProtocolError::ClaimFamilyMismatch),
                    };
                    let oracle_models = reference_interaction_version_space(&log, access);
                    let truth_present = oracle_models.contains(truth);
                    if !truth_present && !self.allow_outside_candidate_grammar {
                        return Err(ProtocolError::GeneratingTruthMissingFromOracle);
                    }
                    (
                        compare_model_sets(compatible_models, &oracle_models)?,
                        compare_structure_sets(compatible_models, &oracle_models, |model| {
                            model.structure
                        }),
                        *disposition,
                        truth_present,
                    )
                }
                _ => return Err(ProtocolError::ClaimFamilyMismatch),
            };

        let generating_truth_behaviorally_represented = match &self.generating_truth {
            GeneratingTruth::Boolean(_) | GeneratingTruth::Interaction(_) => true,
            GeneratingTruth::Numeric(truth) => {
                let access = match self.manifest.structure_access {
                    StructureAccess::NumericWithheld => numeric::StructureAccess::Withheld,
                    StructureAccess::NumericSupplied(mechanism) => {
                        numeric::StructureAccess::Supplied(mechanism)
                    }
                    _ => return Err(ProtocolError::ClaimFamilyMismatch),
                };
                numeric_truth_behaviorally_represented(truth, access)?
            }
        };

        let expected = match set_score.oracle_model_count {
            0 => ClaimDisposition::NoCompatibleModel,
            1 => ClaimDisposition::Identified,
            _ => ClaimDisposition::Ambiguous,
        };
        let abstained = disposition == ClaimDisposition::Abstain;
        let exact_version_space = set_score.exact_version_space_match;
        let safe_abstention = abstained && exact_version_space && set_score.oracle_model_count > 1;
        let unnecessary_abstention =
            abstained && exact_version_space && set_score.oracle_model_count == 1;
        let decision_correct = if abstained {
            safe_abstention
        } else {
            disposition == expected
        };
        Ok(EvaluatorScore {
            evidence_prefix_sha256: submission.evidence_prefix_sha256.clone(),
            generating_model_in_version_space,
            generating_truth_behaviorally_represented,
            safe_abstention,
            unnecessary_abstention,
            decision_correct,
            abstained,
            set_score,
            structure_score,
            evidence_references_valid,
            cited_evidence_count: submission.cited_evidence_ids.len(),
            observed_evidence_count: self.observations.len(),
            probe_count: self.observations.len(),
        })
    }
}

/// Evaluator-owned candidate enumeration. The arrays and loops are
/// intentionally local to the evaluator module rather than calling the
/// agent-facing candidate_models/candidate_hypotheses generators.
fn reference_boolean_candidates(
    access: crate::StructureAccess,
) -> Result<Vec<Hypothesis>, ProtocolError> {
    let mechanisms = match access {
        crate::StructureAccess::Withheld => vec![Mechanism::Direct, Mechanism::HiddenLag],
        crate::StructureAccess::Supplied(mechanism) => vec![mechanism],
    };
    let sensors = [
        MeasurementMap::Identity,
        MeasurementMap::Invert,
        MeasurementMap::ConstantZero,
    ];
    let mut models = Vec::new();
    for mechanism in mechanisms {
        for sensor in sensors {
            match mechanism {
                Mechanism::Direct => models.push(Hypothesis {
                    mechanism,
                    measurement_map: sensor,
                    initial_hidden_state: None,
                }),
                Mechanism::HiddenLag => {
                    for initial in [false, true] {
                        models.push(Hypothesis {
                            mechanism,
                            measurement_map: sensor,
                            initial_hidden_state: Some(initial),
                        });
                    }
                }
            }
        }
    }
    models.sort();
    Ok(models)
}

fn reference_numeric_candidates(access: numeric::StructureAccess) -> Vec<numeric::ModelHypothesis> {
    let mechanisms = match access {
        numeric::StructureAccess::Withheld => vec![
            numeric::Mechanism::DirectResponse,
            numeric::Mechanism::LeakyMemory,
        ],
        numeric::StructureAccess::Supplied(mechanism) => vec![mechanism],
    };
    let sensors = [
        numeric::SensorMap::Identity,
        numeric::SensorMap::Negate,
        numeric::SensorMap::Double,
        numeric::SensorMap::ConstantZero,
    ];
    let mut models = Vec::new();
    for mechanism in mechanisms {
        for gain in [1_i64, 2, 3] {
            for sensor in sensors {
                match mechanism {
                    numeric::Mechanism::DirectResponse => models.push(numeric::ModelHypothesis {
                        mechanism,
                        gain,
                        decay: None,
                        initial_state: None,
                        sensor,
                    }),
                    numeric::Mechanism::LeakyMemory => {
                        for decay in [0_i64, 1, 2] {
                            for initial_state in [-1_i64, 0, 1] {
                                models.push(numeric::ModelHypothesis {
                                    mechanism,
                                    gain,
                                    decay: Some(decay),
                                    initial_state: Some(initial_state),
                                    sensor,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    models.sort();
    models
}

fn reference_interaction_candidates(
    access: interaction::StructureAccess,
) -> Vec<interaction::ModelHypothesis> {
    let structures = match access {
        interaction::StructureAccess::Withheld => vec![
            interaction::Structure::SingleInputU,
            interaction::Structure::SingleInputV,
            interaction::Structure::AdditiveTwoInput,
            interaction::Structure::BilinearInteraction,
        ],
        interaction::StructureAccess::Supplied(structure) => vec![structure],
    };
    let mut models = Vec::new();
    for structure in structures {
        for bias in [-1_i64, 0, 1] {
            match structure {
                interaction::Structure::SingleInputU => {
                    for weight_u in [-2_i64, -1, 1, 2] {
                        models.push(interaction::ModelHypothesis {
                            structure,
                            bias,
                            weight_u,
                            weight_v: 0,
                            interaction_weight: 0,
                        });
                    }
                }
                interaction::Structure::SingleInputV => {
                    for weight_v in [-2_i64, -1, 1, 2] {
                        models.push(interaction::ModelHypothesis {
                            structure,
                            bias,
                            weight_u: 0,
                            weight_v,
                            interaction_weight: 0,
                        });
                    }
                }
                interaction::Structure::AdditiveTwoInput => {
                    for weight_u in [-2_i64, -1, 1, 2] {
                        for weight_v in [-2_i64, -1, 1, 2] {
                            models.push(interaction::ModelHypothesis {
                                structure,
                                bias,
                                weight_u,
                                weight_v,
                                interaction_weight: 0,
                            });
                        }
                    }
                }
                interaction::Structure::BilinearInteraction => {
                    for weight_u in [-2_i64, -1, 1, 2] {
                        for weight_v in [-2_i64, -1, 1, 2] {
                            for interaction_weight in [-2_i64, -1, 1, 2] {
                                models.push(interaction::ModelHypothesis {
                                    structure,
                                    bias,
                                    weight_u,
                                    weight_v,
                                    interaction_weight,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    models.sort();
    models
}

/// Independent evaluator-side trace filters. These deliberately do not call
/// the agent-facing discovery or prediction implementations. They share
/// declared candidate types/families, but use separate transition logic.
fn reference_boolean_version_space(
    log: &ObservationLog,
    access: crate::StructureAccess,
) -> Result<Vec<Hypothesis>, ProtocolError> {
    if log.interventions.len() != log.measurements.len() || log.interventions.is_empty() {
        return Err(ProtocolError::InvalidEvaluatorTrace);
    }
    let mut compatible = Vec::new();
    for hypothesis in reference_boolean_candidates(access)? {
        let mut latent_state = hypothesis.initial_hidden_state;
        let mut matches = true;
        for (&input, &measurement) in log.interventions.iter().zip(&log.measurements) {
            let latent = match hypothesis.mechanism {
                Mechanism::Direct => input,
                Mechanism::HiddenLag => match latent_state {
                    Some(value) => value,
                    None => {
                        matches = false;
                        break;
                    }
                },
            };
            let expected = match hypothesis.measurement_map {
                MeasurementMap::Identity => latent,
                MeasurementMap::Invert => !latent,
                MeasurementMap::ConstantZero => false,
            };
            if expected != measurement {
                matches = false;
                break;
            }
            if hypothesis.mechanism == Mechanism::HiddenLag {
                latent_state = Some(input);
            }
        }
        if matches {
            compatible.push(hypothesis);
        }
    }
    compatible.sort();
    Ok(compatible)
}

const NUMERIC_BEHAVIORAL_EQUIVALENCE_HORIZON: usize = 4;

/// Construct the evaluator-side truth signature independently from the
/// candidate ModelHypothesis::predict implementation. All possible length-4
/// intervention sequences are included, so a parameter/sensor reparameterization
/// is recognized when it is observationally identical over the declared domain.
fn numeric_truth_signature(truth: &numeric::ModelHypothesis) -> Result<Vec<i64>, ProtocolError> {
    let horizon = NUMERIC_BEHAVIORAL_EQUIVALENCE_HORIZON;
    let sequence_count = numeric::INPUT_LEVELS.len().pow(horizon as u32);
    let mut signature = Vec::with_capacity(sequence_count * horizon);

    for code in 0..sequence_count {
        let mut remaining = code;
        let mut inputs = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            inputs.push(numeric::INPUT_LEVELS[remaining % numeric::INPUT_LEVELS.len()]);
            remaining /= numeric::INPUT_LEVELS.len();
        }

        let mut state = truth.initial_state;
        for input in inputs {
            let latent = match truth.mechanism {
                numeric::Mechanism::DirectResponse => truth.gain.checked_mul(input),
                numeric::Mechanism::LeakyMemory => state,
            }
            .ok_or_else(|| {
                ProtocolError::WorldFailure(
                    "numeric truth signature encountered invalid state or overflow".to_owned(),
                )
            })?;

            let measurement = match truth.sensor {
                numeric::SensorMap::Identity => Some(latent),
                numeric::SensorMap::Negate => latent.checked_neg(),
                numeric::SensorMap::Double => latent.checked_mul(2),
                numeric::SensorMap::ConstantZero => Some(0),
            }
            .ok_or_else(|| {
                ProtocolError::WorldFailure(
                    "numeric truth signature measurement overflow".to_owned(),
                )
            })?;
            signature.push(measurement);

            if truth.mechanism == numeric::Mechanism::LeakyMemory {
                let decay = truth.decay.ok_or_else(|| {
                    ProtocolError::WorldFailure(
                        "numeric leaky-memory truth has no decay".to_owned(),
                    )
                })?;
                let current = state.ok_or_else(|| {
                    ProtocolError::WorldFailure(
                        "numeric leaky-memory truth has no state".to_owned(),
                    )
                })?;
                state = Some(
                    decay
                        .checked_mul(current)
                        .and_then(|value| {
                            truth
                                .gain
                                .checked_mul(input)
                                .and_then(|drive| value.checked_add(drive))
                        })
                        .ok_or_else(|| {
                            ProtocolError::WorldFailure(
                                "numeric truth signature state overflow".to_owned(),
                            )
                        })?,
                );
            }
        }
    }

    Ok(signature)
}

fn numeric_candidate_signature(
    candidate: &numeric::ModelHypothesis,
) -> Result<Vec<i64>, ProtocolError> {
    let horizon = NUMERIC_BEHAVIORAL_EQUIVALENCE_HORIZON;
    let sequence_count = numeric::INPUT_LEVELS.len().pow(horizon as u32);
    let mut signature = Vec::with_capacity(sequence_count * horizon);

    for code in 0..sequence_count {
        let mut remaining = code;
        let mut inputs = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            inputs.push(numeric::INPUT_LEVELS[remaining % numeric::INPUT_LEVELS.len()]);
            remaining /= numeric::INPUT_LEVELS.len();
        }
        let outputs = candidate
            .predict(&inputs)
            .map_err(|error| ProtocolError::WorldFailure(error.to_string()))?;
        signature.extend(outputs);
    }

    Ok(signature)
}

fn numeric_truth_behaviorally_represented(
    truth: &numeric::ModelHypothesis,
    access: numeric::StructureAccess,
) -> Result<bool, ProtocolError> {
    let truth_signature = numeric_truth_signature(truth)?;
    for candidate in reference_numeric_candidates(access) {
        if numeric_candidate_signature(&candidate)? == truth_signature {
            return Ok(true);
        }
    }
    Ok(false)
}

fn reference_numeric_version_space(
    log: &numeric::ObservationLog,
    access: numeric::StructureAccess,
) -> Vec<numeric::ModelHypothesis> {
    let mut compatible = Vec::new();
    for model in reference_numeric_candidates(access) {
        let mut state = model.initial_state;
        let mut matches = true;
        for (&input, &observed) in log.interventions.iter().zip(&log.measurements) {
            let latent = match model.mechanism {
                numeric::Mechanism::DirectResponse => model.gain.checked_mul(input),
                numeric::Mechanism::LeakyMemory => state,
            };
            let Some(latent) = latent else {
                matches = false;
                break;
            };
            let expected = match model.sensor {
                numeric::SensorMap::Identity => Some(latent),
                numeric::SensorMap::Negate => latent.checked_neg(),
                numeric::SensorMap::Double => latent.checked_mul(2),
                numeric::SensorMap::ConstantZero => Some(0),
            };
            if expected != Some(observed) {
                matches = false;
                break;
            }
            if model.mechanism == numeric::Mechanism::LeakyMemory {
                let Some(decay) = model.decay else {
                    matches = false;
                    break;
                };
                let next = state
                    .and_then(|value| decay.checked_mul(value))
                    .and_then(|value| {
                        model
                            .gain
                            .checked_mul(input)
                            .and_then(|drive| value.checked_add(drive))
                    });
                if next.is_none() {
                    matches = false;
                    break;
                }
                state = next;
            }
        }
        if matches {
            compatible.push(model);
        }
    }
    compatible.sort();
    compatible
}

fn reference_interaction_version_space(
    log: &interaction::ObservationLog,
    access: interaction::StructureAccess,
) -> Vec<interaction::ModelHypothesis> {
    let mut compatible = Vec::new();
    for model in reference_interaction_candidates(access) {
        let mut matches = true;
        for (input, &observed) in log.interventions.iter().zip(&log.measurements) {
            let [u, v] = *input;
            let response = model.weight_u.checked_mul(u).and_then(|u_term| {
                model.weight_v.checked_mul(v).and_then(|v_term| {
                    model
                        .interaction_weight
                        .checked_mul(u)
                        .and_then(|term| term.checked_mul(v))
                        .and_then(|interaction_term| {
                            model
                                .bias
                                .checked_add(u_term)
                                .and_then(|value| value.checked_add(v_term))
                                .and_then(|value| value.checked_add(interaction_term))
                        })
                })
            });
            if response != Some(observed) {
                matches = false;
                break;
            }
        }
        if matches {
            compatible.push(model);
        }
    }
    compatible.sort();
    compatible
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

fn compare_structure_sets<T, S>(
    claimed: &[T],
    oracle: &[T],
    structure_of: impl Fn(&T) -> S,
) -> StructureScore
where
    S: Ord,
{
    let claimed_structures: BTreeSet<S> = claimed.iter().map(&structure_of).collect();
    let oracle_structures: BTreeSet<S> = oracle.iter().map(structure_of).collect();
    let correct_structure_count = claimed_structures.intersection(&oracle_structures).count();

    StructureScore {
        claimed_structure_count: claimed_structures.len(),
        oracle_structure_count: oracle_structures.len(),
        correct_structure_count,
        exact_structure_set_match: claimed_structures == oracle_structures,
        claimed_structure_identified: claimed_structures.len() == 1,
        oracle_structure_identified: oracle_structures.len() == 1,
    }
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
    InvalidEvaluatorTrace,
    GeneratingTruthMissingFromOracle,
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
    fn independent_boolean_evaluator_agrees_with_inference_for_every_candidate() {
        let inputs = [true, false, true, false];
        for truth in crate::candidate_hypotheses(crate::StructureAccess::Withheld) {
            let mut world = HiddenWorld::new(
                truth.mechanism,
                truth.measurement_map,
                truth.initial_hidden_state,
            )
            .unwrap();
            let log = world.run(&inputs);
            let inferred = crate::discover(&log, crate::StructureAccess::Withheld).unwrap();
            let evaluated =
                reference_boolean_version_space(&log, crate::StructureAccess::Withheld).unwrap();
            assert!(evaluated.contains(&truth), "truth omitted: {truth:?}");
            assert_eq!(inferred.compatible_hypotheses, evaluated, "truth={truth:?}");
        }
    }

    #[test]
    fn open_support_episode_scores_empty_version_space_without_forcing_nearest_model() {
        let truth = numeric::ModelHypothesis {
            gain: 7,
            ..direct_numeric_truth()
        };
        let mut episode = BlindEpisode::new_numeric_open_support(
            "open-support-out-of-catalogue-gain",
            truth,
            StructureAccess::NumericWithheld,
            2,
        )
        .unwrap();

        let probe = episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Numeric(1),
            })
            .unwrap();
        assert_eq!(probe.view.observations.len(), 1);
        assert_eq!(
            probe.view.observations[0].measurement,
            AgentMeasurement::Numeric(7)
        );

        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                compatible_models: Vec::new(),
                disposition: ClaimDisposition::NoCompatibleModel,
            },
            evidence_prefix_sha256: probe.view.evidence_prefix_sha256,
            cited_evidence_ids: vec![probe.view.observations[0].evidence_id.clone()],
            rationale: "The observation lies outside every declared candidate response.".to_owned(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();

        assert!(!score.generating_model_in_version_space);
        assert!(!score.generating_truth_behaviorally_represented);
        assert_eq!(score.set_score.oracle_model_count, 0);
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
        assert!(score.evidence_references_valid);
        assert_eq!(score.structure_score.oracle_structure_count, 0);
    }

    #[test]
    fn open_support_parameter_shift_is_accepted_when_an_equivalent_candidate_exists() {
        let truth = numeric::ModelHypothesis {
            gain: 4,
            ..direct_numeric_truth()
        };
        let mut episode = BlindEpisode::new_numeric_open_support(
            "open-support-equivalent-parameterization",
            truth,
            StructureAccess::NumericWithheld,
            2,
        )
        .unwrap();

        let probe = episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Numeric(1),
            })
            .unwrap();
        assert_eq!(
            probe.view.observations[0].measurement,
            AgentMeasurement::Numeric(4)
        );

        let equivalent = numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::Double,
        };
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                compatible_models: vec![equivalent],
                disposition: ClaimDisposition::Identified,
            },
            evidence_prefix_sha256: probe.view.evidence_prefix_sha256,
            cited_evidence_ids: vec![probe.view.observations[0].evidence_id.clone()],
            rationale: "Gain 4 with identity sensor is behaviorally equivalent to gain 2 with a double sensor over the declared intervention language.".to_owned(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();

        assert!(!score.generating_model_in_version_space);
        assert!(score.generating_truth_behaviorally_represented);
        assert_eq!(score.set_score.oracle_model_count, 1);
        assert!(score.set_score.exact_version_space_match);
        assert!(score.structure_score.exact_structure_set_match);
        assert!(score.decision_correct);
        assert!(score.evidence_references_valid);
    }

    #[test]
    fn closed_support_episode_fails_closed_when_generating_truth_is_outside_candidates() {
        let truth = numeric::ModelHypothesis {
            gain: 7,
            ..direct_numeric_truth()
        };
        assert!(matches!(
            BlindEpisode::new_numeric(
                "closed-support-out-of-catalogue-gain",
                truth,
                StructureAccess::NumericWithheld,
                2,
            ),
            Ok(_)
        ));
        let mut episode = BlindEpisode::new_numeric(
            "closed-support-out-of-catalogue-gain-score",
            truth,
            StructureAccess::NumericWithheld,
            2,
        )
        .unwrap();
        let probe = episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Numeric(1),
            })
            .unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                compatible_models: Vec::new(),
                disposition: ClaimDisposition::NoCompatibleModel,
            },
            evidence_prefix_sha256: probe.view.evidence_prefix_sha256,
            cited_evidence_ids: vec![probe.view.observations[0].evidence_id.clone()],
            rationale: "No compatible declared model.".to_owned(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        assert_eq!(
            episode.evaluate_after_submission(),
            Err(ProtocolError::GeneratingTruthMissingFromOracle)
        );
    }

    #[test]
    fn independent_numeric_evaluator_agrees_with_inference_for_every_candidate() {
        let inputs = [1, -1, 1, 0];
        for truth in numeric::candidate_models(numeric::StructureAccess::Withheld) {
            let mut world = numeric::NumericWorld::new(truth).unwrap();
            let log = world.run(&inputs).unwrap();
            let inferred = numeric::discover(&log, numeric::StructureAccess::Withheld).unwrap();
            let evaluated =
                reference_numeric_version_space(&log, numeric::StructureAccess::Withheld);
            assert!(evaluated.contains(&truth), "truth omitted: {truth:?}");
            assert_eq!(inferred.compatible_models, evaluated, "truth={truth:?}");
        }
    }

    #[test]
    fn independent_interaction_evaluator_agrees_with_inference_for_every_candidate() {
        for truth in interaction::candidate_models(interaction::StructureAccess::Withheld) {
            let world = interaction::InteractionWorld::new(truth).unwrap();
            let log = world.run(&interaction::FACTORIAL_DESIGN).unwrap();
            let inferred =
                interaction::discover(&log, interaction::StructureAccess::Withheld).unwrap();
            let evaluated =
                reference_interaction_version_space(&log, interaction::StructureAccess::Withheld);
            assert!(evaluated.contains(&truth), "truth omitted: {truth:?}");
            assert_eq!(inferred.compatible_models, evaluated, "truth={truth:?}");
        }
    }

    #[test]
    fn independent_evaluator_detects_an_incomplete_claimed_version_space() {
        let mut episode = BlindEpisode::new_boolean(
            "blind-independent-oracle-01",
            Mechanism::Direct,
            MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            2,
        )
        .unwrap();
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
        let log = ObservationLog {
            interventions: vec![true],
            measurements: vec![true],
        };
        let mut compatible = crate::discover(&log, crate::StructureAccess::Withheld)
            .unwrap()
            .compatible_hypotheses;
        assert!(compatible.len() > 1);
        compatible.pop();
        let count = compatible.len();
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Boolean {
                compatible_models: compatible,
                disposition: claim_disposition(count),
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec!["E0001".to_string()],
            rationale: "Deliberately incomplete set for evaluator negative control.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(!score.set_score.exact_version_space_match);
        assert_eq!(
            score.set_score.claimed_model_count + 1,
            score.set_score.oracle_model_count
        );
        assert_eq!(
            score.set_score.correct_model_count,
            score.set_score.claimed_model_count
        );
    }

    #[test]
    fn independent_evaluator_detects_an_allowed_but_incompatible_model() {
        let mut episode = BlindEpisode::new_numeric(
            "blind-independent-oracle-02",
            direct_numeric_truth(),
            StructureAccess::NumericWithheld,
            2,
        )
        .unwrap();
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Numeric(1),
            })
            .unwrap();
        let mut compatible = numeric::candidate_models(numeric::StructureAccess::Withheld)
            .into_iter()
            .filter(|model| {
                model.mechanism == numeric::Mechanism::DirectResponse
                    && model.gain == 2
                    && model.sensor == numeric::SensorMap::Identity
            })
            .collect::<Vec<_>>();
        assert!(!compatible.is_empty());
        compatible.push(numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 1,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::Identity,
        });
        let count = compatible.len();
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                compatible_models: compatible,
                disposition: claim_disposition(count),
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec!["E0001".to_string()],
            rationale: "Deliberately contaminated set for evaluator negative control.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(!score.set_score.exact_version_space_match);
        assert!(score.set_score.correct_model_count < score.set_score.claimed_model_count);
    }

    #[test]
    fn structure_scoring_separates_structure_recovery_from_parameter_identification() {
        let mut episode = BlindEpisode::new_numeric(
            "numeric-structure-only-identified",
            direct_numeric_truth(),
            StructureAccess::NumericWithheld,
            4,
        )
        .unwrap();

        for intervention in [1, -1, 1] {
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Numeric(intervention),
                })
                .unwrap();
        }

        let log = numeric::ObservationLog {
            interventions: vec![1, -1, 1],
            measurements: vec![2, -2, 2],
        };
        let compatible_models = numeric::discover(&log, numeric::StructureAccess::Withheld)
            .unwrap()
            .compatible_models;
        assert!(compatible_models.len() > 1);
        assert!(
            compatible_models
                .iter()
                .all(|model| { model.mechanism == numeric::Mechanism::DirectResponse })
        );

        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                disposition: ClaimDisposition::Ambiguous,
                compatible_models,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec![
                "E0001".to_string(),
                "E0002".to_string(),
                "E0003".to_string(),
            ],
            rationale: "The mechanism is identified, while gain and sensor scale remain observationally equivalent.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();

        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
        assert!(score.structure_score.exact_structure_set_match);
        assert!(score.structure_score.claimed_structure_identified);
        assert!(score.structure_score.oracle_structure_identified);
        assert_eq!(score.structure_score.oracle_structure_count, 1);
    }

    #[test]
    fn structure_scoring_preserves_cross_family_ambiguity_under_uninformative_sensor() {
        let truth = numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::ConstantZero,
        };
        let mut episode = BlindEpisode::new_numeric(
            "numeric-structure-ambiguous",
            truth,
            StructureAccess::NumericWithheld,
            4,
        )
        .unwrap();

        for intervention in [1, -1, 1] {
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Numeric(intervention),
                })
                .unwrap();
        }

        let log = numeric::ObservationLog {
            interventions: vec![1, -1, 1],
            measurements: vec![0, 0, 0],
        };
        let compatible_models = numeric::discover(&log, numeric::StructureAccess::Withheld)
            .unwrap()
            .compatible_models;
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                disposition: claim_disposition(compatible_models.len()),
                compatible_models,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec![
                "E0001".to_string(),
                "E0002".to_string(),
                "E0003".to_string(),
            ],
            rationale:
                "The constant sensor makes direct and leaky mechanisms observationally equivalent."
                    .to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();

        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.structure_score.exact_structure_set_match);
        assert!(!score.structure_score.oracle_structure_identified);
        assert_eq!(score.structure_score.oracle_structure_count, 2);
    }

    #[test]
    fn exact_abstention_on_non_identifiable_version_space_is_scored_as_safe() {
        let truth = numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::ConstantZero,
        };
        let mut episode = BlindEpisode::new_numeric(
            "numeric-safe-abstention",
            truth,
            StructureAccess::NumericWithheld,
            4,
        )
        .unwrap();

        for intervention in [1, -1, 1] {
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Numeric(intervention),
                })
                .unwrap();
        }

        let log = numeric::ObservationLog {
            interventions: vec![1, -1, 1],
            measurements: vec![0, 0, 0],
        };
        let compatible_models = numeric::discover(&log, numeric::StructureAccess::Withheld)
            .unwrap()
            .compatible_models;
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                disposition: ClaimDisposition::Abstain,
                compatible_models,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec![
                "E0001".to_string(),
                "E0002".to_string(),
                "E0003".to_string(),
            ],
            rationale: "The exact surviving model set spans distinct structures under an uninformative sensor.".to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();

        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.safe_abstention);
        assert!(!score.unnecessary_abstention);
        assert!(score.abstained);
        assert!(score.decision_correct);
    }

    #[test]
    fn abstention_on_uniquely_identified_model_is_not_scored_as_safe() {
        let truth = numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::Double,
        };
        let mut episode = BlindEpisode::new_numeric(
            "numeric-unnecessary-abstention",
            truth,
            StructureAccess::NumericWithheld,
            4,
        )
        .unwrap();

        for intervention in [1, -1, 1] {
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Numeric(intervention),
                })
                .unwrap();
        }

        let log = numeric::ObservationLog {
            interventions: vec![1, -1, 1],
            measurements: vec![4, -4, 4],
        };
        let compatible_models = numeric::discover(&log, numeric::StructureAccess::Withheld)
            .unwrap()
            .compatible_models;
        assert_eq!(compatible_models.len(), 1);
        let view = episode.view().unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Numeric {
                disposition: ClaimDisposition::Abstain,
                compatible_models,
            },
            evidence_prefix_sha256: view.evidence_prefix_sha256,
            cited_evidence_ids: vec![
                "E0001".to_string(),
                "E0002".to_string(),
                "E0003".to_string(),
            ],
            rationale: "Deliberately abstain despite unique identification as a negative control."
                .to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();

        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(!score.safe_abstention);
        assert!(score.unnecessary_abstention);
        assert!(score.abstained);
        assert!(!score.decision_correct);
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
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
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
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Numeric(value),
                })
                .unwrap();
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
            cited_evidence_ids: vec![
                "E0001".to_string(),
                "E0002".to_string(),
                "E0003".to_string(),
            ],
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
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
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
        )
        .unwrap();
        for input in interaction::FACTORIAL_DESIGN {
            episode
                .apply(AgentAction::Probe {
                    intervention: AgentIntervention::Pair(input),
                })
                .unwrap();
        }
        let current = episode.view().unwrap();
        let log = interaction::ObservationLog {
            interventions: current
                .observations
                .iter()
                .map(|record| match record.intervention {
                    AgentIntervention::Pair(value) => value,
                    _ => unreachable!(),
                })
                .collect(),
            measurements: current
                .observations
                .iter()
                .map(|record| match record.measurement {
                    AgentMeasurement::Numeric(value) => value,
                    _ => unreachable!(),
                })
                .collect(),
        };
        let oracle = interaction::discover(&log, interaction::StructureAccess::Withheld).unwrap();
        let count = oracle.compatible_models.len();
        let submission = AgentSubmission {
            claim: AgentClaim::Interaction {
                compatible_models: oracle.compatible_models,
                disposition: claim_disposition(count),
            },
            evidence_prefix_sha256: current.evidence_prefix_sha256.clone(),
            cited_evidence_ids: current
                .observations
                .iter()
                .map(|record| record.evidence_id.clone())
                .collect(),
            rationale: "The factorial design distinguishes variable relevance and interaction."
                .to_string(),
        };
        episode.apply(AgentAction::Submit { submission }).unwrap();
        let score = episode.evaluate_after_submission().unwrap();
        assert!(score.set_score.exact_version_space_match);
        assert!(score.decision_correct);
    }

    #[test]
    fn mismatched_evidence_prefix_is_rejected_without_closing_episode() {
        let mut episode = BlindEpisode::new_boolean(
            "blind-prefix-mismatch-01",
            Mechanism::Direct,
            MeasurementMap::Identity,
            None,
            StructureAccess::BooleanWithheld,
            2,
        )
        .unwrap();
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
        let submission = AgentSubmission {
            claim: AgentClaim::Boolean {
                compatible_models: crate::candidate_hypotheses(crate::StructureAccess::Withheld),
                disposition: ClaimDisposition::Ambiguous,
            },
            evidence_prefix_sha256: "sha256:stale-prefix".to_string(),
            cited_evidence_ids: vec!["E0001".to_string()],
            rationale: "This deliberately cites a stale evidence prefix.".to_string(),
        };
        assert_eq!(
            episode.apply(AgentAction::Submit { submission }),
            Err(ProtocolError::EvidencePrefixMismatch)
        );
        assert_eq!(episode.status(), EpisodeStatus::Open);
    }

    #[test]
    fn supplied_structure_must_match_the_evaluator_world() {
        assert_eq!(
            BlindEpisode::new_boolean(
                "blind-a0-mismatch-01",
                Mechanism::Direct,
                MeasurementMap::Identity,
                None,
                StructureAccess::BooleanSupplied(Mechanism::HiddenLag),
                2,
            )
            .err(),
            Some(ProtocolError::SuppliedStructureMismatch)
        );
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
        )
        .unwrap();
        let before = episode.view().unwrap().evidence_prefix_sha256;
        episode
            .apply(AgentAction::Probe {
                intervention: AgentIntervention::Boolean(true),
            })
            .unwrap();
        let after = episode.view().unwrap().evidence_prefix_sha256;
        assert_ne!(before, after);
    }
}
