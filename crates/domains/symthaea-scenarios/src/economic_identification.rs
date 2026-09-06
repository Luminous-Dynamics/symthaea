// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! ESE-A5: blind synthetic mechanism identification.
//!
//! A5 asks a narrower question than real-world inference: when the data-generating
//! mechanism is known to the experiment harness but hidden from candidate models,
//! can the scientific layer identify the generator from held-out synthetic
//! outcomes? All candidate prediction pairs are frozen from generator-free case
//! inputs before any hidden generator is revealed.
//!
//! Error remains vector-valued. No arbitrary scalar exchange rate between demand
//! atoms and jobs is introduced. Observational equivalence, equal fit, and
//! cross-metric incomparability are represented separately rather than collapsed
//! into a forced winner.
//!
//! This remains synthetic model behavior only. The in-memory frozen-prediction
//! boundary is not durable preregistration and confers no empirical, policy, or
//! governance authority. Durable probabilistic forecasting belongs to the
//! Symthaea Futures Laboratory in a later tranche.

use symthaea_economics::{EconomicsError, Result};

use crate::economic_science::{
    SyntheticEconomicOutcome, aggregate_system_dynamics_prediction,
    heterogeneous_agent_prediction,
};

/// Which known synthetic mechanism generated a revealed A5 world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticGeneratorKind {
    HeterogeneousThresholds,
    AggregateProportional,
}

/// Result of comparing two candidate mechanisms without scalarizing errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentificationVerdict {
    HeterogeneousThresholds,
    AggregateProportional,
    /// Candidate mechanisms produced exactly the same observable prediction.
    Indistinguishable,
    /// Candidate predictions differed, but their dimension-preserving error
    /// vectors were exactly tied against the revealed continuation.
    EqualFit,
    /// Neither candidate Pareto-dominates the other because each is better on a
    /// different observable.
    Incomparable,
}

/// Exact, dimension-preserving prediction error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorVector {
    pub demand_atoms: u64,
    pub employment: u64,
}

impl ErrorVector {
    fn between(predicted: SyntheticEconomicOutcome, actual: SyntheticEconomicOutcome) -> Self {
        Self {
            demand_atoms: predicted
                .shocked_demand_atoms
                .abs_diff(actual.shocked_demand_atoms),
            employment: predicted
                .shocked_employment
                .abs_diff(actual.shocked_employment),
        }
    }

    fn pareto_dominates(self, other: Self) -> bool {
        let no_worse = self.demand_atoms <= other.demand_atoms
            && self.employment <= other.employment;
        let strictly_better = self.demand_atoms < other.demand_atoms
            || self.employment < other.employment;
        no_worse && strictly_better
    }
}

/// Candidate predictions computed without access to the hidden generator.
/// Fields are private and there are no mutators: after construction, the
/// prediction pair is frozen in memory for the reveal/compare phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenCandidatePredictions {
    shock_bps: u16,
    heterogeneous: SyntheticEconomicOutcome,
    aggregate: SyntheticEconomicOutcome,
}

impl FrozenCandidatePredictions {
    pub fn shock_bps(&self) -> u16 {
        self.shock_bps
    }

    pub fn heterogeneous(&self) -> SyntheticEconomicOutcome {
        self.heterogeneous
    }

    pub fn aggregate(&self) -> SyntheticEconomicOutcome {
        self.aggregate
    }
}

/// One revealed blind-identification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindIdentificationRecord {
    pub case_id: u8,
    pub shock_bps: u16,
    pub generator: SyntheticGeneratorKind,
    pub actual: SyntheticEconomicOutcome,
    pub heterogeneous_error: ErrorVector,
    pub aggregate_error: ErrorVector,
    pub verdict: IdentificationVerdict,
}

impl BlindIdentificationRecord {
    /// `Some(true/false)` when the observations identify one generator;
    /// `None` when the result correctly refuses to choose.
    pub fn identified_correctly(self) -> Option<bool> {
        match self.verdict {
            IdentificationVerdict::HeterogeneousThresholds => {
                Some(self.generator == SyntheticGeneratorKind::HeterogeneousThresholds)
            }
            IdentificationVerdict::AggregateProportional => {
                Some(self.generator == SyntheticGeneratorKind::AggregateProportional)
            }
            IdentificationVerdict::Indistinguishable
            | IdentificationVerdict::EqualFit
            | IdentificationVerdict::Incomparable => None,
        }
    }
}

/// Aggregate qualification summary. Counts stay separate rather than collapsing
/// success, abstention, and ambiguity into one headline score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindIdentificationSummary {
    pub total_cases: usize,
    pub identifiable_cases: usize,
    pub correct_identifications: usize,
    pub incorrect_identifications: usize,
    pub indistinguishable_cases: usize,
    pub equal_fit_cases: usize,
    pub incomparable_cases: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlindCaseInput {
    case_id: u8,
    shock_bps: u16,
}

/// Generator-free candidate input table. Nothing here reveals which synthetic
/// mechanism will later generate a continuation.
const BLIND_INPUTS: [BlindCaseInput; 8] = [
    BlindCaseInput {
        case_id: 1,
        shock_bps: 500,
    },
    BlindCaseInput {
        case_id: 2,
        shock_bps: 500,
    },
    BlindCaseInput {
        case_id: 3,
        shock_bps: 1_000,
    },
    BlindCaseInput {
        case_id: 4,
        shock_bps: 1_000,
    },
    BlindCaseInput {
        case_id: 5,
        shock_bps: 2_000,
    },
    BlindCaseInput {
        case_id: 6,
        shock_bps: 2_000,
    },
    BlindCaseInput {
        case_id: 7,
        shock_bps: 3_000,
    },
    BlindCaseInput {
        case_id: 8,
        shock_bps: 3_000,
    },
];

/// Hidden generator table. It carries no shock values and is not consulted while
/// candidate predictions are being frozen.
const HIDDEN_GENERATORS: [(u8, SyntheticGeneratorKind); 8] = [
    (1, SyntheticGeneratorKind::HeterogeneousThresholds),
    (2, SyntheticGeneratorKind::AggregateProportional),
    // Intentional null-identifiability controls: at a 10% shock both candidate
    // mechanisms produce the same observables, regardless of hidden generator.
    (3, SyntheticGeneratorKind::HeterogeneousThresholds),
    (4, SyntheticGeneratorKind::AggregateProportional),
    (5, SyntheticGeneratorKind::HeterogeneousThresholds),
    (6, SyntheticGeneratorKind::AggregateProportional),
    (7, SyntheticGeneratorKind::HeterogeneousThresholds),
    (8, SyntheticGeneratorKind::AggregateProportional),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrozenBlindCase {
    input: BlindCaseInput,
    predictions: FrozenCandidatePredictions,
}

fn invalid(context: &'static str) -> EconomicsError {
    EconomicsError::InvalidParameter { context }
}

/// Candidate models receive only the intervention size. The hidden generator is
/// absent from this function's input surface by construction.
fn freeze_candidate_predictions(shock_bps: u16) -> Result<FrozenCandidatePredictions> {
    Ok(FrozenCandidatePredictions {
        shock_bps,
        heterogeneous: heterogeneous_agent_prediction(shock_bps)?,
        aggregate: aggregate_system_dynamics_prediction(shock_bps)?,
    })
}

/// Freeze every candidate pair before any hidden generator is looked up.
fn freeze_blind_suite() -> Result<Vec<FrozenBlindCase>> {
    BLIND_INPUTS
        .iter()
        .copied()
        .map(|input| {
            Ok(FrozenBlindCase {
                input,
                predictions: freeze_candidate_predictions(input.shock_bps)?,
            })
        })
        .collect()
}

fn hidden_generator(case_id: u8) -> Result<SyntheticGeneratorKind> {
    HIDDEN_GENERATORS
        .iter()
        .find_map(|(hidden_case_id, generator)| {
            (*hidden_case_id == case_id).then_some(*generator)
        })
        .ok_or(invalid("A5 hidden generator is missing"))
}

/// Reveal the held-out synthetic continuation after the full candidate suite has
/// already been frozen.
fn reveal_world(
    input: BlindCaseInput,
) -> Result<(SyntheticGeneratorKind, SyntheticEconomicOutcome)> {
    let generator = hidden_generator(input.case_id)?;
    let actual = match generator {
        SyntheticGeneratorKind::HeterogeneousThresholds => {
            heterogeneous_agent_prediction(input.shock_bps)?
        }
        SyntheticGeneratorKind::AggregateProportional => {
            aggregate_system_dynamics_prediction(input.shock_bps)?
        }
    };
    Ok((generator, actual))
}

fn identify(
    predictions: FrozenCandidatePredictions,
    actual: SyntheticEconomicOutcome,
) -> (ErrorVector, ErrorVector, IdentificationVerdict) {
    let heterogeneous_error = ErrorVector::between(predictions.heterogeneous, actual);
    let aggregate_error = ErrorVector::between(predictions.aggregate, actual);

    let verdict = if predictions.heterogeneous == predictions.aggregate {
        IdentificationVerdict::Indistinguishable
    } else if heterogeneous_error == aggregate_error {
        IdentificationVerdict::EqualFit
    } else if heterogeneous_error.pareto_dominates(aggregate_error) {
        IdentificationVerdict::HeterogeneousThresholds
    } else if aggregate_error.pareto_dominates(heterogeneous_error) {
        IdentificationVerdict::AggregateProportional
    } else {
        IdentificationVerdict::Incomparable
    };

    (heterogeneous_error, aggregate_error, verdict)
}

/// Execute the fixed ESE-A5 blind-identification suite.
pub fn run_a5_blind_identification_suite() -> Result<Vec<BlindIdentificationRecord>> {
    // This call completes for every case before reveal begins. The hidden
    // generator table therefore cannot influence candidate construction through
    // the experiment runner.
    let frozen_cases = freeze_blind_suite()?;

    frozen_cases
        .into_iter()
        .map(|frozen_case| {
            let (generator, actual) = reveal_world(frozen_case.input)?;
            let (heterogeneous_error, aggregate_error, verdict) =
                identify(frozen_case.predictions, actual);

            Ok(BlindIdentificationRecord {
                case_id: frozen_case.input.case_id,
                shock_bps: frozen_case.input.shock_bps,
                generator,
                actual,
                heterogeneous_error,
                aggregate_error,
                verdict,
            })
        })
        .collect()
}

/// Summarize an A5 run without turning heterogeneous outcomes into a scalar
/// scientific score.
pub fn summarize_a5(records: &[BlindIdentificationRecord]) -> Result<BlindIdentificationSummary> {
    if records.is_empty() {
        return Err(invalid("A5 identification records are empty"));
    }

    let mut identifiable_cases = 0_usize;
    let mut correct_identifications = 0_usize;
    let mut incorrect_identifications = 0_usize;
    let mut indistinguishable_cases = 0_usize;
    let mut equal_fit_cases = 0_usize;
    let mut incomparable_cases = 0_usize;

    for record in records {
        match record.identified_correctly() {
            Some(true) => {
                identifiable_cases += 1;
                correct_identifications += 1;
            }
            Some(false) => {
                identifiable_cases += 1;
                incorrect_identifications += 1;
            }
            None => match record.verdict {
                IdentificationVerdict::Indistinguishable => indistinguishable_cases += 1,
                IdentificationVerdict::EqualFit => equal_fit_cases += 1,
                IdentificationVerdict::Incomparable => incomparable_cases += 1,
                IdentificationVerdict::HeterogeneousThresholds
                | IdentificationVerdict::AggregateProportional => {
                    return Err(invalid("A5 identification state is inconsistent"));
                }
            },
        }
    }

    Ok(BlindIdentificationSummary {
        total_cases: records.len(),
        identifiable_cases,
        correct_identifications,
        incorrect_identifications,
        indistinguishable_cases,
        equal_fit_cases,
        incomparable_cases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_freeze_surface_has_no_generator_input() {
        let frozen = freeze_candidate_predictions(2_000).unwrap();
        assert_eq!(frozen.shock_bps(), 2_000);
        assert_eq!(frozen.heterogeneous().shocked_employment, 6);
        assert_eq!(frozen.aggregate().shocked_employment, 8);
    }

    #[test]
    fn complete_candidate_suite_freezes_before_any_reveal() {
        let frozen = freeze_blind_suite().unwrap();
        assert_eq!(frozen.len(), BLIND_INPUTS.len());
        assert_eq!(frozen[0].input.case_id, 1);
        assert_eq!(frozen[7].input.case_id, 8);
    }

    #[test]
    fn blind_and_hidden_case_ids_match_exactly() {
        let blind_ids: Vec<_> = BLIND_INPUTS.iter().map(|input| input.case_id).collect();
        let hidden_ids: Vec<_> = HIDDEN_GENERATORS
            .iter()
            .map(|(case_id, _)| *case_id)
            .collect();
        assert_eq!(blind_ids, hidden_ids);
    }

    #[test]
    fn discriminating_cases_recover_the_hidden_generator() {
        let records = run_a5_blind_identification_suite().unwrap();
        for record in records
            .iter()
            .filter(|record| record.shock_bps != 1_000)
        {
            assert_eq!(record.identified_correctly(), Some(true));
        }
    }

    #[test]
    fn observational_equivalence_forces_abstention() {
        let records = run_a5_blind_identification_suite().unwrap();
        let controls: Vec<_> = records
            .iter()
            .filter(|record| record.shock_bps == 1_000)
            .collect();
        assert_eq!(controls.len(), 2);
        for record in controls {
            assert_eq!(record.heterogeneous_error, ErrorVector {
                demand_atoms: 0,
                employment: 0,
            });
            assert_eq!(record.aggregate_error, ErrorVector {
                demand_atoms: 0,
                employment: 0,
            });
            assert_eq!(record.verdict, IdentificationVerdict::Indistinguishable);
            assert_eq!(record.identified_correctly(), None);
        }
    }

    #[test]
    fn equal_error_does_not_imply_observational_equivalence() {
        let actual = SyntheticEconomicOutcome {
            baseline_demand_atoms: 1_000,
            shocked_demand_atoms: 700,
            baseline_employment: 10,
            shocked_employment: 7,
        };
        let frozen = FrozenCandidatePredictions {
            shock_bps: 2_000,
            heterogeneous: SyntheticEconomicOutcome {
                baseline_demand_atoms: 1_000,
                shocked_demand_atoms: 600,
                baseline_employment: 10,
                shocked_employment: 6,
            },
            aggregate: SyntheticEconomicOutcome {
                baseline_demand_atoms: 1_000,
                shocked_demand_atoms: 800,
                baseline_employment: 10,
                shocked_employment: 8,
            },
        };
        let (heterogeneous, aggregate, verdict) = identify(frozen, actual);
        assert_eq!(heterogeneous, aggregate);
        assert_eq!(verdict, IdentificationVerdict::EqualFit);
    }

    #[test]
    fn pareto_rule_refuses_cross_metric_tradeoffs() {
        let actual = SyntheticEconomicOutcome {
            baseline_demand_atoms: 1_000,
            shocked_demand_atoms: 700,
            baseline_employment: 10,
            shocked_employment: 7,
        };
        let frozen = FrozenCandidatePredictions {
            shock_bps: 3_000,
            heterogeneous: SyntheticEconomicOutcome {
                baseline_demand_atoms: 1_000,
                shocked_demand_atoms: 700,
                baseline_employment: 10,
                shocked_employment: 5,
            },
            aggregate: SyntheticEconomicOutcome {
                baseline_demand_atoms: 1_000,
                shocked_demand_atoms: 650,
                baseline_employment: 10,
                shocked_employment: 7,
            },
        };
        let (heterogeneous, aggregate, verdict) = identify(frozen, actual);
        assert_eq!(heterogeneous, ErrorVector {
            demand_atoms: 0,
            employment: 2,
        });
        assert_eq!(aggregate, ErrorVector {
            demand_atoms: 50,
            employment: 0,
        });
        assert_eq!(verdict, IdentificationVerdict::Incomparable);
    }

    #[test]
    fn fixed_suite_has_six_correct_and_two_indistinguishable_cases() {
        let records = run_a5_blind_identification_suite().unwrap();
        let summary = summarize_a5(&records).unwrap();
        assert_eq!(summary.total_cases, 8);
        assert_eq!(summary.identifiable_cases, 6);
        assert_eq!(summary.correct_identifications, 6);
        assert_eq!(summary.incorrect_identifications, 0);
        assert_eq!(summary.indistinguishable_cases, 2);
        assert_eq!(summary.equal_fit_cases, 0);
        assert_eq!(summary.incomparable_cases, 0);
    }

    #[test]
    fn suite_is_exactly_deterministic() {
        assert_eq!(
            run_a5_blind_identification_suite().unwrap(),
            run_a5_blind_identification_suite().unwrap()
        );
    }

    #[test]
    fn empty_summary_fails_closed() {
        assert!(summarize_a5(&[]).is_err());
    }
}
