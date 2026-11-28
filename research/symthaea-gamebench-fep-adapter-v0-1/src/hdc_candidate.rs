#![forbid(unsafe_code)]

//! Spatial HDC candidate using the repository's actual symthaea-core
//! ContinuousHV implementation. This is a one-step representation/policy
//! baseline; it is not a trained world model or full HDC+CfC agent.

use std::error::Error;
use std::f32::consts::TAU;
use std::fmt::{Display, Formatter};

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_gamebench::{
    Action, CandidateDecision, CandidateFeedback, CandidateInput, FeatureForecast, ForecastMass,
    ForecastValue, InteractiveCandidate, Observation, validate_decision,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdcCandidateError {
    ZeroDimension,
    MissingGoal,
    IncompleteMap,
    EmptyActionSet,
    NoRemainingBudget,
    InvalidDecision,
}

impl Display for HdcCandidateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "HDC spatial dimensions must be nonzero"),
            Self::MissingGoal => write!(f, "HDC spatial candidate requires a visible goal"),
            Self::IncompleteMap => write!(f, "HDC greedy baseline requires a complete map"),
            Self::EmptyActionSet => write!(f, "HDC candidate received no offered actions"),
            Self::NoRemainingBudget => write!(f, "HDC candidate received no remaining action budget"),
            Self::InvalidDecision => write!(f, "HDC candidate produced an invalid decision"),
        }
    }
}

impl Error for HdcCandidateError {}

#[derive(Debug, Clone, Copy)]
struct FourierFeature {
    omega: f32,
    phase: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialHdcEncoder {
    width: u16,
    height: u16,
    features: Vec<FourierFeature>,
    x_role: ContinuousHV,
    y_role: ContinuousHV,
}

impl SpatialHdcEncoder {
    pub(crate) fn new(width: u16, height: u16, seed: u64) -> Result<Self, HdcCandidateError> {
        if width == 0 || height == 0 {
            return Err(HdcCandidateError::ZeroDimension);
        }
        let mut rng = nonzero_seed(seed);
        let features = (0..HDC_DIMENSION)
            .map(|_| {
                let u1 = next_uniform(&mut rng).max(1e-12);
                let u2 = next_uniform(&mut rng);
                let gaussian = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
                FourierFeature {
                    // RBF random Fourier features: nearby coordinates should
                    // retain higher expected cosine similarity.
                    omega: gaussian * 3.0,
                    phase: next_uniform(&mut rng) * TAU,
                }
            })
            .collect();

        Ok(Self {
            width,
            height,
            features,
            x_role: ContinuousHV::random(HDC_DIMENSION, seed ^ 0x5841_5849_535F_524F),
            y_role: ContinuousHV::random(HDC_DIMENSION, seed ^ 0x5941_5849_535F_524F),
        })
    }

    pub(crate) fn encode(&self, x: u16, y: u16) -> ContinuousHV {
        let x_norm = if self.width > 1 {
            f32::from(x) / f32::from(self.width - 1)
        } else {
            0.0
        };
        let y_norm = if self.height > 1 {
            f32::from(y) / f32::from(self.height - 1)
        } else {
            0.0
        };

        let x_features = ContinuousHV::from_values(
            self.features
                .iter()
                .map(|feature| (feature.omega * x_norm + feature.phase).cos())
                .collect(),
        );
        let y_features = ContinuousHV::from_values(
            self.features
                .iter()
                .map(|feature| (feature.omega * y_norm + feature.phase).cos())
                .collect(),
        );

        // Role binding distinguishes the x- and y-coordinate channels;
        // superposition binds them into one spatial-state hypervector.
        let x_term = self.x_role.bind(&x_features);
        let y_term = self.y_role.bind(&y_features);
        ContinuousHV::bundle(&[&x_term, &y_term])
    }
}

#[derive(Debug, Clone)]
pub struct HdcSpatialCandidate {
    encoder: SpatialHdcEncoder,
}

impl HdcSpatialCandidate {
    pub fn new(width: u16, height: u16, seed: u64) -> Result<Self, HdcCandidateError> {
        Ok(Self {
            encoder: SpatialHdcEncoder::new(width, height, seed)?,
        })
    }

    fn next_position(observation: &Observation, action: Action) -> (u16, u16) {
        let current = observation.position;
        let candidate = match action {
            Action::North => current.y.checked_sub(1).map(|y| (current.x, y)),
            Action::East => current.x.checked_add(1).map(|x| (x, current.y)),
            Action::South => current.y.checked_add(1).map(|y| (current.x, y)),
            Action::West => current.x.checked_sub(1).map(|x| (x, current.y)),
        };

        let Some((x, y)) = candidate else {
            return (current.x, current.y);
        };
        if x >= observation.width || y >= observation.height {
            return (current.x, current.y);
        }
        let is_wall = observation.visible_cells.iter().any(|cell| {
            cell.position.x == x && cell.position.y == y && cell.cell == symthaea_gamebench::Cell::Wall
        });
        if is_wall {
            (current.x, current.y)
        } else {
            (x, y)
        }
    }
}

impl InteractiveCandidate<Observation, Action> for HdcSpatialCandidate {
    type Error = HdcCandidateError;

    fn reset(&mut self) -> Result<(), Self::Error> {
        // All random frequencies are frozen at construction; reset does not
        // change the representation or leak an evaluator seed.
        Ok(())
    }

    fn decide(
        &mut self,
        input: CandidateInput<'_, Observation, Action>,
    ) -> Result<CandidateDecision<Action>, Self::Error> {
        if input.remaining_actions == 0 {
            return Err(HdcCandidateError::NoRemainingBudget);
        }
        let observation = input.observation;
        let goal = observation.known_goal.ok_or(HdcCandidateError::MissingGoal)?;
        let expected_cells = usize::from(observation.width)
            .checked_mul(usize::from(observation.height))
            .ok_or(HdcCandidateError::IncompleteMap)?;
        if observation.visible_cells.len() != expected_cells {
            return Err(HdcCandidateError::IncompleteMap);
        }
        if input.legal_actions.is_empty() {
            return Err(HdcCandidateError::EmptyActionSet);
        }

        let goal_hv = self.encoder.encode(goal.x, goal.y);
        let mut best: Option<(f32, Action, (u16, u16))> = None;
        for action in input.legal_actions {
            let position = Self::next_position(observation, *action);
            let candidate_hv = self.encoder.encode(position.0, position.1);
            let similarity = candidate_hv.similarity(&goal_hv);
            if best
                .as_ref()
                .is_none_or(|(best_similarity, _, _)| similarity > *best_similarity)
            {
                best = Some((similarity, *action, position));
            }
        }

        let (_, action, (x, y)) = best.ok_or(HdcCandidateError::EmptyActionSet)?;
        let decision = CandidateDecision {
            action,
            // Similarity is a representation score, not a calibrated
            // probability of task success.
            confidence: None,
            forecasts: vec![
                FeatureForecast {
                    feature_id: 0,
                    outcomes: vec![ForecastMass {
                        value: ForecastValue::Integer(i64::from(x)),
                        probability: 1.0,
                    }],
                },
                FeatureForecast {
                    feature_id: 1,
                    outcomes: vec![ForecastMass {
                        value: ForecastValue::Integer(i64::from(y)),
                        probability: 1.0,
                    }],
                },
            ],
        };
        validate_decision(&decision, input.legal_actions)
            .map_err(|_| HdcCandidateError::InvalidDecision)?;
        Ok(decision)
    }

    fn observe_transition(
        &mut self,
        _feedback: CandidateFeedback<'_, Observation, Action>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn nonzero_seed(seed: u64) -> u64 {
    if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

fn next_uniform(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32) / (1_u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_gamebench::{Cell, Game, GridSpec, ObservationMode, Position};

    fn fixture() -> Observation {
        let spec = GridSpec::new(
            3,
            3,
            vec![
                Cell::Floor, Cell::Floor, Cell::Floor,
                Cell::Floor, Cell::Floor, Cell::Floor,
                Cell::Floor, Cell::Floor, Cell::Goal,
            ],
            Position::new(0, 0),
            Position::new(2, 2),
            12,
        )
        .expect("valid map");
        Game::new(spec).observe(ObservationMode::Full)
    }

    #[test]
    fn hdc_spatial_encoding_is_deterministic() {
        let encoder_a = SpatialHdcEncoder::new(5, 5, 42).expect("encoder");
        let encoder_b = SpatialHdcEncoder::new(5, 5, 42).expect("encoder");
        let a = encoder_a.encode(2, 3);
        let b = encoder_b.encode(2, 3);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nearby_positions_have_higher_expected_similarity_than_far_positions() {
        let encoder = SpatialHdcEncoder::new(10, 10, 7).expect("encoder");
        let origin = encoder.encode(4, 4);
        let nearby = encoder.encode(5, 4);
        let far = encoder.encode(9, 9);
        assert!(origin.similarity(&nearby) > origin.similarity(&far));
    }

    #[test]
    fn hdc_candidate_chooses_a_move_toward_goal_using_core_continuous_hv() {
        let observation = fixture();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcSpatialCandidate::new(3, 3, 99).expect("candidate");
        let decision = candidate
            .decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 12,
                episode_step: 0,
            })
            .expect("decision");
        assert!(matches!(decision.action, Action::East | Action::South));
        assert_eq!(decision.confidence, None);
    }

    #[test]
    fn hdc_candidate_rejects_zero_action_budget() {
        let observation = fixture();
        let actions = [Action::North, Action::East, Action::South, Action::West];
        let mut candidate = HdcSpatialCandidate::new(3, 3, 6).expect("candidate");
        assert_eq!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 0,
                episode_step: 0,
            }),
            Err(HdcCandidateError::NoRemainingBudget)
        );
    }

    #[test]
    fn hdc_candidate_rejects_partial_map() {
        let spec = GridSpec::new(
            3,
            1,
            vec![Cell::Floor, Cell::Floor, Cell::Goal],
            Position::new(0, 0),
            Position::new(2, 0),
            4,
        )
        .expect("valid map");
        let observation = Game::new(spec).observe(ObservationMode::ManhattanRadius(0));
        let actions = [Action::East, Action::West];
        let mut candidate = HdcSpatialCandidate::new(3, 1, 5).expect("candidate");
        assert!(matches!(
            candidate.decide(CandidateInput {
                observation: &observation,
                legal_actions: &actions,
                remaining_actions: 4,
                episode_step: 0,
            }),
            Err(HdcCandidateError::IncompleteMap)
        ));
    }
}
