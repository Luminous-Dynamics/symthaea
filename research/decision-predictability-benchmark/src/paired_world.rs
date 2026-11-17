//! Paired-world reference fixture for decision-relevant predictability.
//!
//! The candidate-facing episode and evaluator-only world record are separate
//! types. This is an API/type boundary for the benchmark harness, not a
//! process-security boundary: confirmatory evaluations must keep evaluator
//! records in a separate process or permissioned execution environment.

/// Candidate-visible action set, with stable action indices used by the
/// evaluator loss table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum CandidateAction {
    Intervene = 0,
    Continue = 1,
    Abstain = 2,
}

/// One candidate-visible observation from the primary sensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimarySensorObservation {
    pub time_step: u32,
    pub sensor_id: &'static str,
    pub value: f64,
}

/// Candidate-visible information for one episode.
///
/// This intentionally contains no world mechanism, hidden state, oracle
/// label, probe result or evaluator seed.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEpisode {
    pub episode_id: &'static str,
    pub primary_observations: Vec<PrimarySensorObservation>,
    pub reference_probe_available: bool,
    pub reference_probe_cost: f64,
    pub reference_probe_latency_steps: u32,
    pub decision_deadline_step: u32,
}

/// Evaluator-only description of the latent world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMechanism {
    /// The physical system is hazardous and the primary sensor is unbiased.
    PhysicalHazard,
    /// The physical system is cool; a sensor offset creates the same primary
    /// reading as the physical-hazard world.
    SensorOffsetOnly,
}

/// Evaluator-only truth and loss information for one world.
///
/// Action-loss indices match CandidateAction discriminants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluatorWorld {
    pub mechanism: WorldMechanism,
    pub reference_probe_value: f64,
    pub action_losses: [f64; 3],
}

/// Why a requested reference observation was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeDenial {
    UnknownWorld,
    ProbeUnavailable,
    InvalidBudget,
    InsufficientBudget,
    DecisionDeadlineMissed,
}

/// Paired-world fixture. Candidate and evaluator records are separately
/// addressable so the harness can supply only candidate episodes to a model.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedWorldFixture {
    candidate_episodes: [CandidateEpisode; 2],
    evaluator_worlds: [EvaluatorWorld; 2],
    prior_physical_hazard_probability: f64,
}

impl PairedWorldFixture {
    /// Return candidate-visible episodes only.
    pub fn candidate_episodes(&self) -> &[CandidateEpisode; 2] {
        &self.candidate_episodes
    }

    /// Return evaluator-only world records.
    ///
    /// Do not pass these records, or values derived from them other than
    /// authorized sensor observations, to a candidate model.
    pub fn evaluator_worlds(&self) -> &[EvaluatorWorld; 2] {
        &self.evaluator_worlds
    }

    pub fn prior_physical_hazard_probability(&self) -> f64 {
        self.prior_physical_hazard_probability
    }

    /// Materialize a reference-probe observation only when the probe is
    /// available, its cost fits the remaining budget, and its result arrives
    /// no later than the declared decision deadline.
    ///
    /// The returned observation contains no mechanism label. This simple
    /// benchmark step counter is not a substitute for the event-time and
    /// clock-domain contract owned by the broader observation-event work.
    pub fn reference_probe_observation(
        &self,
        world_index: usize,
        requested_at_step: u32,
        remaining_budget: f64,
    ) -> Result<PrimarySensorObservation, ProbeDenial> {
        let world = self
            .evaluator_worlds
            .get(world_index)
            .ok_or(ProbeDenial::UnknownWorld)?;
        let episode = self
            .candidate_episodes
            .get(world_index)
            .ok_or(ProbeDenial::UnknownWorld)?;

        if !episode.reference_probe_available {
            return Err(ProbeDenial::ProbeUnavailable);
        }
        if !remaining_budget.is_finite() || remaining_budget < 0.0 {
            return Err(ProbeDenial::InvalidBudget);
        }
        if !episode.reference_probe_cost.is_finite() || episode.reference_probe_cost < 0.0 {
            return Err(ProbeDenial::InvalidBudget);
        }
        if remaining_budget < episode.reference_probe_cost {
            return Err(ProbeDenial::InsufficientBudget);
        }

        let result_step = requested_at_step.saturating_add(episode.reference_probe_latency_steps);
        if result_step > episode.decision_deadline_step {
            return Err(ProbeDenial::DecisionDeadlineMissed);
        }

        Ok(PrimarySensorObservation {
            time_step: result_step,
            sensor_id: "independent_reference",
            value: world.reference_probe_value,
        })
    }
}

/// Construct two worlds that are identical under passive primary sensing.
///
/// World 0: actual temperature 10, unbiased primary sensor reports 10.
/// World 1: actual temperature 0, sensor offset +10, primary sensor reports 10.
///
/// The independent reference probe reports the physical temperature, allowing
/// the two causes to be distinguished. The action losses are:
///
/// | World | Intervene | Continue | Abstain |
/// |---|---:|---:|---:|
/// | Physical hazard | 0.0 | 1.0 | 0.4 |
/// | Sensor offset only | 1.0 | 0.0 | 0.4 |
///
/// Prior probability of each world is 0.5. A passive-only policy cannot
/// distinguish them; abstention has Bayes risk 0.4. A perfect-information
/// observer has risk 0.0. The reference probe costs 0.1 task-loss units, so
/// its decision-focused net value is 0.3.
pub fn canonical_paired_world_fixture() -> PairedWorldFixture {
    let observations = vec![
        PrimarySensorObservation {
            time_step: 1,
            sensor_id: "primary_temperature",
            value: 10.0,
        },
        PrimarySensorObservation {
            time_step: 2,
            sensor_id: "primary_temperature",
            value: 10.0,
        },
        PrimarySensorObservation {
            time_step: 3,
            sensor_id: "primary_temperature",
            value: 10.0,
        },
    ];

    let episode = |episode_id| CandidateEpisode {
        episode_id,
        primary_observations: observations.clone(),
        reference_probe_available: true,
        reference_probe_cost: 0.1,
        reference_probe_latency_steps: 1,
        decision_deadline_step: 4,
    };

    PairedWorldFixture {
        candidate_episodes: [episode("paired-world-a"), episode("paired-world-b")],
        evaluator_worlds: [
            EvaluatorWorld {
                mechanism: WorldMechanism::PhysicalHazard,
                reference_probe_value: 10.0,
                action_losses: [0.0, 1.0, 0.4],
            },
            EvaluatorWorld {
                mechanism: WorldMechanism::SensorOffsetOnly,
                reference_probe_value: 0.0,
                action_losses: [1.0, 0.0, 0.4],
            },
        ],
        prior_physical_hazard_probability: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConditionalStateLoss, InformationSet, evaluate_decisions, expected_value_of_information,
    };

    fn conditional_state(probability: f64, world: EvaluatorWorld) -> ConditionalStateLoss {
        ConditionalStateLoss {
            probability,
            action_losses: world.action_losses.to_vec(),
        }
    }

    #[test]
    fn passive_candidate_views_are_exactly_observationally_equivalent() {
        let fixture = canonical_paired_world_fixture();
        let episodes = fixture.candidate_episodes();

        assert_ne!(episodes[0].episode_id, episodes[1].episode_id);
        assert_eq!(
            episodes[0].primary_observations,
            episodes[1].primary_observations
        );
        assert_eq!(
            episodes[0].reference_probe_available,
            episodes[1].reference_probe_available
        );
        assert_eq!(
            episodes[0].reference_probe_cost,
            episodes[1].reference_probe_cost
        );
        assert_eq!(
            episodes[0].decision_deadline_step,
            episodes[1].decision_deadline_step
        );
        assert_eq!(episodes[0].primary_observations.len(), 3);
        assert!(
            episodes[0]
                .primary_observations
                .iter()
                .all(|observation| observation.value == 10.0)
        );
    }

    #[test]
    fn evaluator_worlds_have_different_causes_and_probe_readings() {
        let fixture = canonical_paired_world_fixture();
        let worlds = fixture.evaluator_worlds();

        assert_ne!(worlds[0].mechanism, worlds[1].mechanism);
        assert_eq!(
            fixture
                .reference_probe_observation(0, 3, 0.1)
                .expect("probe should meet deadline and budget")
                .value,
            10.0
        );
        assert_eq!(
            fixture
                .reference_probe_observation(1, 3, 0.1)
                .expect("probe should meet deadline and budget")
                .value,
            0.0
        );
    }

    #[test]
    fn passive_equivalence_requires_abstention_and_preserves_irreducible_risk() {
        let fixture = canonical_paired_world_fixture();
        let worlds = fixture.evaluator_worlds();
        let prior_hazard = fixture.prior_physical_hazard_probability();

        let information_sets = [InformationSet {
            probability: 1.0,
            states: vec![
                conditional_state(prior_hazard, worlds[0]),
                conditional_state(1.0 - prior_hazard, worlds[1]),
            ],
            candidate_action: Some(CandidateAction::Abstain as usize),
        }];

        let result = evaluate_decisions(&information_sets).expect("valid evaluator fixture");
        assert!((result.bayes_risk - 0.4).abs() < 1e-12);
        assert!((result.policy_risk - 0.4).abs() < 1e-12);
        assert!((result.perfect_information_risk - 0.0).abs() < 1e-12);
        assert!((result.regret - 0.0).abs() < 1e-12);
    }

    #[test]
    fn reference_probe_changes_the_optimal_action_and_has_positive_net_value() {
        let fixture = canonical_paired_world_fixture();
        let worlds = fixture.evaluator_worlds();
        let prior_hazard = fixture.prior_physical_hazard_probability();
        let probe_cost = fixture.candidate_episodes()[0].reference_probe_cost;

        // A perfect reference reading creates a candidate-visible information
        // set for each possible outcome, each with posterior probability one.
        let information_sets = [
            InformationSet {
                probability: prior_hazard,
                states: vec![conditional_state(1.0, worlds[0])],
                candidate_action: Some(CandidateAction::Intervene as usize),
            },
            InformationSet {
                probability: 1.0 - prior_hazard,
                states: vec![conditional_state(1.0, worlds[1])],
                candidate_action: Some(CandidateAction::Continue as usize),
            },
        ];

        let decision = evaluate_decisions(&information_sets).expect("valid probe outcomes");
        assert!((decision.bayes_risk - 0.0).abs() < 1e-12);
        assert!((decision.policy_risk - 0.0).abs() < 1e-12);
        assert!((decision.perfect_information_risk - 0.0).abs() < 1e-12);
        assert!((decision.regret - 0.0).abs() < 1e-12);

        let value = expected_value_of_information(0.4, decision.bayes_risk, probe_cost)
            .expect("valid expected posterior risk");
        assert!((value.gross_value - 0.4).abs() < 1e-12);
        assert!((value.net_value - 0.3).abs() < 1e-12);
    }

    #[test]
    fn high_entropy_nuisance_probe_has_zero_decision_value() {
        let fixture = canonical_paired_world_fixture();
        let worlds = fixture.evaluator_worlds();
        let prior_hazard = fixture.prior_physical_hazard_probability();

        // A nuisance reading splits into two equally likely outcomes (one bit
        // of information) but each outcome leaves the hazard prior unchanged.
        // Expected posterior Bayes risk therefore remains 0.4.
        let nuisance_information_sets = [
            InformationSet {
                probability: 0.5,
                states: vec![
                    conditional_state(prior_hazard, worlds[0]),
                    conditional_state(1.0 - prior_hazard, worlds[1]),
                ],
                candidate_action: Some(CandidateAction::Abstain as usize),
            },
            InformationSet {
                probability: 0.5,
                states: vec![
                    conditional_state(prior_hazard, worlds[0]),
                    conditional_state(1.0 - prior_hazard, worlds[1]),
                ],
                candidate_action: Some(CandidateAction::Abstain as usize),
            },
        ];

        let result =
            evaluate_decisions(&nuisance_information_sets).expect("valid nuisance outcomes");
        assert!((result.bayes_risk - 0.4).abs() < 1e-12);
        assert!((result.policy_risk - 0.4).abs() < 1e-12);

        let value = expected_value_of_information(0.4, result.bayes_risk, 0.05)
            .expect("valid expected posterior risk");
        assert!((value.gross_value - 0.0).abs() < 1e-12);
        assert!((value.net_value + 0.05).abs() < 1e-12);
    }
    #[test]
    fn reference_probe_is_denied_when_budget_is_insufficient() {
        let fixture = canonical_paired_world_fixture();
        assert_eq!(
            fixture.reference_probe_observation(0, 3, 0.05),
            Err(ProbeDenial::InsufficientBudget)
        );
    }

    #[test]
    fn reference_probe_is_denied_when_result_arrives_after_deadline() {
        let fixture = canonical_paired_world_fixture();
        assert_eq!(
            fixture.reference_probe_observation(0, 4, 0.1),
            Err(ProbeDenial::DecisionDeadlineMissed)
        );
    }

    #[test]
    fn reference_probe_rejects_invalid_budget_and_world_identity() {
        let fixture = canonical_paired_world_fixture();
        assert_eq!(
            fixture.reference_probe_observation(0, 3, f64::NAN),
            Err(ProbeDenial::InvalidBudget)
        );
        assert_eq!(
            fixture.reference_probe_observation(2, 3, 0.1),
            Err(ProbeDenial::UnknownWorld)
        );
    }

    #[test]
    fn reference_probe_is_denied_when_not_permitted() {
        let mut fixture = canonical_paired_world_fixture();
        fixture.candidate_episodes[0].reference_probe_available = false;
        assert_eq!(
            fixture.reference_probe_observation(0, 3, 0.1),
            Err(ProbeDenial::ProbeUnavailable)
        );
    }
}
