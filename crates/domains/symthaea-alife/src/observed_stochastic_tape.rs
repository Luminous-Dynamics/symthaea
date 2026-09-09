// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical evidence contract for stochastic decisions observed on a natural Genesis trajectory.
//!
//! Exact snapshot/restore is sufficient to continue one world. It is not sufficient to explain a
//! rescue counterfactually after the fact: the sham replay must receive the same stochastic input
//! that the observed natural trajectory received, while the intervention branch must reuse that
//! input for logically shared events without forcing the same downstream action or phenotype.
//!
//! Different stochastic owners expose different semantics, so this module deliberately does not
//! flatten every observation into one pseudo-random scalar:
//!
//! - FEP action selection, RandomPeer source selection, and mutation use scalar unit variates;
//! - offspring construction uses an exact nonzero seed;
//! - the legacy encounter scheduler uses Fisher-Yates, so its natural evidence is the exact input
//!   cohort plus resulting shuffled order rather than a misleading per-agent scalar.
//!
//! This tape is evidence, not execution authority. Live FEP/scheduler/evolution recording and
//! replay adapters are intentionally separate tranches. Validation proves canonical structure and
//! interval/identity/value semantics; it does not authenticate where the bytes came from.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{AgentId, GenomeTraitV1};

/// Behavior/lifecycle boundary that identifies the execution interval covered by one tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedStochasticTapeSubjectV1 {
    pub start_behavior_next_tick: u64,
    pub start_lifecycle_next_sequence: u64,
    pub end_behavior_next_tick: u64,
}

/// Scalar stochastic surfaces whose natural variate/seed can be replayed directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservedStochasticDomainV1 {
    InheritanceSourceSample,
    MutationOccurrence,
    MutationPerturbation,
    OffspringConstructionSeed,
    AgentActionSample,
}

/// Semantic address of one observed scalar stochastic input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObservedStochasticKeyV1 {
    pub domain: ObservedStochasticDomainV1,
    pub tick: u64,
    pub subject_id: AgentId,
    pub counterpart_id: Option<AgentId>,
    pub ordinal: u64,
}

impl ObservedStochasticKeyV1 {
    pub fn inheritance_source_sample(
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
    ) -> Self {
        Self {
            domain: ObservedStochasticDomainV1::InheritanceSourceSample,
            tick,
            subject_id: reproductive_parent_id,
            counterpart_id: None,
            ordinal: birth_ordinal,
        }
    }

    pub fn mutation_occurrence(
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
    ) -> Self {
        Self {
            domain: ObservedStochasticDomainV1::MutationOccurrence,
            tick,
            subject_id: reproductive_parent_id,
            counterpart_id: Some(genome_source_id),
            ordinal: combine_birth_trait_ordinal(birth_ordinal, trait_id),
        }
    }

    pub fn mutation_perturbation(
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
    ) -> Self {
        Self {
            domain: ObservedStochasticDomainV1::MutationPerturbation,
            tick,
            subject_id: reproductive_parent_id,
            counterpart_id: Some(genome_source_id),
            ordinal: combine_birth_trait_ordinal(birth_ordinal, trait_id),
        }
    }

    pub fn offspring_construction_seed(
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
    ) -> Self {
        Self {
            domain: ObservedStochasticDomainV1::OffspringConstructionSeed,
            tick,
            subject_id: reproductive_parent_id,
            counterpart_id: None,
            ordinal: birth_ordinal,
        }
    }

    pub fn agent_action_sample(tick: u64, agent_id: AgentId, draw_ordinal: u64) -> Self {
        Self {
            domain: ObservedStochasticDomainV1::AgentActionSample,
            tick,
            subject_id: agent_id,
            counterpart_id: None,
            ordinal: draw_ordinal,
        }
    }
}

/// Exact persisted value for one observed scalar stochastic input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservedStochasticValueV1 {
    /// IEEE-754 bits for a finite scalar in `[0, 1)`.
    UnitF64Bits(u64),
    /// Exact nonzero seed for a constructor/RNG surface that forbids all-zero state.
    NonZeroU64(u64),
}

impl ObservedStochasticValueV1 {
    pub fn unit(value: f64) -> Self {
        Self::UnitF64Bits(value.to_bits())
    }

    pub fn nonzero_u64(value: u64) -> Self {
        Self::NonZeroU64(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedStochasticDrawV1 {
    pub key: ObservedStochasticKeyV1,
    pub value: ObservedStochasticValueV1,
}

/// Which legacy scheduler shuffle produced one observed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservedSchedulerOrderKindV1 {
    /// Full living population used by `PairingMode::Random`.
    RandomPopulation,
    /// Unassigned/widowed eligible cohort shuffled by `PairingMode::FixedPartners`.
    FixedPartnerEligible,
}

/// Exact natural input cohort and post-Fisher-Yates order for one scheduler call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedSchedulerOrderV1 {
    pub tick: u64,
    pub kind: ObservedSchedulerOrderKindV1,
    pub input_order: Vec<AgentId>,
    pub shuffled_order: Vec<AgentId>,
}

impl ObservedSchedulerOrderV1 {
    /// Project the historical random order onto identities still present in a counterfactual.
    ///
    /// This preserves the relative stochastic order of every shared identity. Novel identities
    /// are deliberately not inserted here; a later adapter must merge them using the stateless
    /// counterfactual field under an explicitly qualified rule.
    pub fn project_shared_order(&self, present: &BTreeSet<AgentId>) -> Vec<AgentId> {
        self.shuffled_order
            .iter()
            .copied()
            .filter(|id| present.contains(id))
            .collect()
    }
}

/// Raw persistence for one observed natural stochastic interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedStochasticTapeV1 {
    pub subject: ObservedStochasticTapeSubjectV1,
    pub scalar_draws: Vec<ObservedStochasticDrawV1>,
    pub scheduler_orders: Vec<ObservedSchedulerOrderV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedObservedStochasticTapeV1 {
    inner: ObservedStochasticTapeV1,
}

impl ValidatedObservedStochasticTapeV1 {
    pub fn persisted(&self) -> &ObservedStochasticTapeV1 {
        &self.inner
    }

    pub fn subject(&self) -> ObservedStochasticTapeSubjectV1 {
        self.inner.subject
    }

    pub fn unit_for(&self, key: ObservedStochasticKeyV1) -> Option<f64> {
        let index = self
            .inner
            .scalar_draws
            .binary_search_by_key(&key, |draw| draw.key)
            .ok()?;
        match self.inner.scalar_draws[index].value {
            ObservedStochasticValueV1::UnitF64Bits(bits) => Some(f64::from_bits(bits)),
            ObservedStochasticValueV1::NonZeroU64(_) => None,
        }
    }

    pub fn nonzero_u64_for(&self, key: ObservedStochasticKeyV1) -> Option<u64> {
        let index = self
            .inner
            .scalar_draws
            .binary_search_by_key(&key, |draw| draw.key)
            .ok()?;
        match self.inner.scalar_draws[index].value {
            ObservedStochasticValueV1::UnitF64Bits(_) => None,
            ObservedStochasticValueV1::NonZeroU64(value) => Some(value),
        }
    }

    pub fn scheduler_order(
        &self,
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
    ) -> Option<&ObservedSchedulerOrderV1> {
        let target = (tick, kind);
        let index = self
            .inner
            .scheduler_orders
            .binary_search_by_key(&target, |entry| (entry.tick, entry.kind))
            .ok()?;
        self.inner.scheduler_orders.get(index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedStochasticTapeErrorV1 {
    BehaviorIntervalReversed {
        start: u64,
        end: u64,
    },
    ScalarOrderNotCanonical {
        index: usize,
    },
    SchedulerOrderNotCanonical {
        index: usize,
    },
    TickOutsideSubject {
        tick: u64,
        start: u64,
        end: u64,
    },
    ReservedIdentity {
        key: ObservedStochasticKeyV1,
    },
    CounterpartPolicyMismatch {
        key: ObservedStochasticKeyV1,
    },
    ValueKindMismatch {
        domain: ObservedStochasticDomainV1,
    },
    InvalidUnitVariate {
        key: ObservedStochasticKeyV1,
        bits: u64,
    },
    ZeroConstructionSeed {
        key: ObservedStochasticKeyV1,
    },
    SchedulerOrderTooSmall {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
        len: usize,
    },
    SchedulerLengthMismatch {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
        input_len: usize,
        shuffled_len: usize,
    },
    SchedulerDuplicateIdentity {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
    },
    SchedulerReservedIdentity {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
    },
    SchedulerMembershipMismatch {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
    },
}

impl ObservedStochasticTapeV1 {
    pub fn validate(self) -> Result<ValidatedObservedStochasticTapeV1, ObservedStochasticTapeErrorV1> {
        let subject = self.subject;
        if subject.end_behavior_next_tick < subject.start_behavior_next_tick {
            return Err(ObservedStochasticTapeErrorV1::BehaviorIntervalReversed {
                start: subject.start_behavior_next_tick,
                end: subject.end_behavior_next_tick,
            });
        }

        for (index, pair) in self.scalar_draws.windows(2).enumerate() {
            if pair[0].key >= pair[1].key {
                return Err(ObservedStochasticTapeErrorV1::ScalarOrderNotCanonical {
                    index: index + 1,
                });
            }
        }
        for draw in &self.scalar_draws {
            validate_tick(draw.key.tick, subject)?;
            validate_scalar_draw(*draw)?;
        }

        for (index, pair) in self.scheduler_orders.windows(2).enumerate() {
            if (pair[0].tick, pair[0].kind) >= (pair[1].tick, pair[1].kind) {
                return Err(ObservedStochasticTapeErrorV1::SchedulerOrderNotCanonical {
                    index: index + 1,
                });
            }
        }
        for order in &self.scheduler_orders {
            validate_tick(order.tick, subject)?;
            validate_scheduler_order(order)?;
        }

        Ok(ValidatedObservedStochasticTapeV1 { inner: self })
    }
}

fn validate_tick(
    tick: u64,
    subject: ObservedStochasticTapeSubjectV1,
) -> Result<(), ObservedStochasticTapeErrorV1> {
    if tick < subject.start_behavior_next_tick || tick >= subject.end_behavior_next_tick {
        return Err(ObservedStochasticTapeErrorV1::TickOutsideSubject {
            tick,
            start: subject.start_behavior_next_tick,
            end: subject.end_behavior_next_tick,
        });
    }
    Ok(())
}

fn validate_scalar_draw(
    draw: ObservedStochasticDrawV1,
) -> Result<(), ObservedStochasticTapeErrorV1> {
    let key = draw.key;
    if key.subject_id == AgentId::UNALLOCATED || key.counterpart_id == Some(AgentId::UNALLOCATED) {
        return Err(ObservedStochasticTapeErrorV1::ReservedIdentity { key });
    }

    let requires_counterpart = matches!(
        key.domain,
        ObservedStochasticDomainV1::MutationOccurrence
            | ObservedStochasticDomainV1::MutationPerturbation
    );
    if requires_counterpart != key.counterpart_id.is_some() {
        return Err(ObservedStochasticTapeErrorV1::CounterpartPolicyMismatch { key });
    }

    match (key.domain, draw.value) {
        (
            ObservedStochasticDomainV1::OffspringConstructionSeed,
            ObservedStochasticValueV1::NonZeroU64(value),
        ) => {
            if value == 0 {
                return Err(ObservedStochasticTapeErrorV1::ZeroConstructionSeed { key });
            }
        }
        (
            ObservedStochasticDomainV1::OffspringConstructionSeed,
            ObservedStochasticValueV1::UnitF64Bits(_),
        ) => {
            return Err(ObservedStochasticTapeErrorV1::ValueKindMismatch {
                domain: key.domain,
            });
        }
        (_, ObservedStochasticValueV1::UnitF64Bits(bits)) => {
            let value = f64::from_bits(bits);
            if !value.is_finite() || !(0.0..1.0).contains(&value) {
                return Err(ObservedStochasticTapeErrorV1::InvalidUnitVariate { key, bits });
            }
        }
        (_, ObservedStochasticValueV1::NonZeroU64(_)) => {
            return Err(ObservedStochasticTapeErrorV1::ValueKindMismatch {
                domain: key.domain,
            });
        }
    }
    Ok(())
}

fn validate_scheduler_order(
    order: &ObservedSchedulerOrderV1,
) -> Result<(), ObservedStochasticTapeErrorV1> {
    if order.input_order.len() < 2 {
        return Err(ObservedStochasticTapeErrorV1::SchedulerOrderTooSmall {
            tick: order.tick,
            kind: order.kind,
            len: order.input_order.len(),
        });
    }
    if order.input_order.len() != order.shuffled_order.len() {
        return Err(ObservedStochasticTapeErrorV1::SchedulerLengthMismatch {
            tick: order.tick,
            kind: order.kind,
            input_len: order.input_order.len(),
            shuffled_len: order.shuffled_order.len(),
        });
    }
    if order
        .input_order
        .iter()
        .chain(order.shuffled_order.iter())
        .any(|&id| id == AgentId::UNALLOCATED)
    {
        return Err(ObservedStochasticTapeErrorV1::SchedulerReservedIdentity {
            tick: order.tick,
            kind: order.kind,
        });
    }

    let input = order.input_order.iter().copied().collect::<BTreeSet<_>>();
    let shuffled = order
        .shuffled_order
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if input.len() != order.input_order.len() || shuffled.len() != order.shuffled_order.len() {
        return Err(ObservedStochasticTapeErrorV1::SchedulerDuplicateIdentity {
            tick: order.tick,
            kind: order.kind,
        });
    }
    if input != shuffled {
        return Err(ObservedStochasticTapeErrorV1::SchedulerMembershipMismatch {
            tick: order.tick,
            kind: order.kind,
        });
    }
    Ok(())
}

fn combine_birth_trait_ordinal(birth_ordinal: u64, trait_id: GenomeTraitV1) -> u64 {
    let trait_tag = match trait_id {
        GenomeTraitV1::SetPoint => 0,
        GenomeTraitV1::ForageEfficiency => 1,
        GenomeTraitV1::ActionTemperature => 2,
        GenomeTraitV1::PerceptualGrain => 3,
    };
    birth_ordinal
        .checked_mul(4)
        .and_then(|base| base.checked_add(trait_tag))
        .expect("birth ordinal too large for Genome-v1 stochastic addressing")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn ids(n: usize) -> Vec<AgentId> {
        let mut allocator = AgentIdAllocator::new();
        (0..n).map(|_| allocator.allocate()).collect()
    }

    fn subject() -> ObservedStochasticTapeSubjectV1 {
        ObservedStochasticTapeSubjectV1 {
            start_behavior_next_tick: 10,
            start_lifecycle_next_sequence: 7,
            end_behavior_next_tick: 20,
        }
    }

    #[test]
    fn canonical_tape_round_trip_preserves_exact_observed_values_and_orders() {
        let agents = ids(4);
        let action_key = ObservedStochasticKeyV1::agent_action_sample(11, agents[0], 0);
        let seed_key = ObservedStochasticKeyV1::offspring_construction_seed(12, agents[1], 0);
        let mut scalar_draws = vec![
            ObservedStochasticDrawV1 {
                key: seed_key,
                value: ObservedStochasticValueV1::nonzero_u64(991),
            },
            ObservedStochasticDrawV1 {
                key: action_key,
                value: ObservedStochasticValueV1::unit(0.375),
            },
        ];
        scalar_draws.sort_by_key(|draw| draw.key);
        let tape = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws,
            scheduler_orders: vec![ObservedSchedulerOrderV1 {
                tick: 13,
                kind: ObservedSchedulerOrderKindV1::RandomPopulation,
                input_order: agents.clone(),
                shuffled_order: vec![agents[2], agents[0], agents[3], agents[1]],
            }],
        };
        let encoded = serde_json::to_string(&tape).expect("serialize tape");
        let decoded: ObservedStochasticTapeV1 =
            serde_json::from_str(&encoded).expect("deserialize tape");
        let validated = decoded.validate().expect("validate tape");
        assert_eq!(validated.persisted(), &tape);
        assert_eq!(validated.unit_for(action_key), Some(0.375));
        assert_eq!(validated.nonzero_u64_for(seed_key), Some(991));
        assert_eq!(
            validated
                .scheduler_order(13, ObservedSchedulerOrderKindV1::RandomPopulation)
                .unwrap()
                .shuffled_order,
            vec![agents[2], agents[0], agents[3], agents[1]]
        );
    }

    #[test]
    fn scalar_draws_must_be_strictly_canonical_and_unique() {
        let agents = ids(1);
        let key = ObservedStochasticKeyV1::agent_action_sample(11, agents[0], 0);
        let draw = ObservedStochasticDrawV1 {
            key,
            value: ObservedStochasticValueV1::unit(0.25),
        };
        let tape = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws: vec![draw, draw],
            scheduler_orders: vec![],
        };
        assert!(matches!(
            tape.validate(),
            Err(ObservedStochasticTapeErrorV1::ScalarOrderNotCanonical { index: 1 })
        ));
    }

    #[test]
    fn unit_draws_reject_nan_one_and_wrong_value_kind() {
        let agents = ids(1);
        let action = ObservedStochasticKeyV1::agent_action_sample(11, agents[0], 0);
        for value in [f64::NAN, 1.0, f64::INFINITY] {
            let tape = ObservedStochasticTapeV1 {
                subject: subject(),
                scalar_draws: vec![ObservedStochasticDrawV1 {
                    key: action,
                    value: ObservedStochasticValueV1::unit(value),
                }],
                scheduler_orders: vec![],
            };
            assert!(matches!(
                tape.validate(),
                Err(ObservedStochasticTapeErrorV1::InvalidUnitVariate { .. })
            ));
        }

        let tape = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws: vec![ObservedStochasticDrawV1 {
                key: action,
                value: ObservedStochasticValueV1::nonzero_u64(5),
            }],
            scheduler_orders: vec![],
        };
        assert!(matches!(
            tape.validate(),
            Err(ObservedStochasticTapeErrorV1::ValueKindMismatch { .. })
        ));
    }

    #[test]
    fn scheduler_order_must_be_an_exact_permutation_of_its_input_cohort() {
        let agents = ids(4);
        let missing_member = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws: vec![],
            scheduler_orders: vec![ObservedSchedulerOrderV1 {
                tick: 12,
                kind: ObservedSchedulerOrderKindV1::RandomPopulation,
                input_order: agents.clone(),
                shuffled_order: vec![agents[0], agents[1], agents[2], agents[2]],
            }],
        };
        assert!(matches!(
            missing_member.validate(),
            Err(ObservedStochasticTapeErrorV1::SchedulerDuplicateIdentity { .. })
        ));

        let different_member = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws: vec![],
            scheduler_orders: vec![ObservedSchedulerOrderV1 {
                tick: 12,
                kind: ObservedSchedulerOrderKindV1::RandomPopulation,
                input_order: agents[..3].to_vec(),
                shuffled_order: vec![agents[0], agents[1], agents[3]],
            }],
        };
        assert!(matches!(
            different_member.validate(),
            Err(ObservedStochasticTapeErrorV1::SchedulerMembershipMismatch { .. })
        ));
    }

    #[test]
    fn projecting_a_counterfactual_subset_preserves_shared_relative_scheduler_order() {
        let agents = ids(5);
        let observed = ObservedSchedulerOrderV1 {
            tick: 14,
            kind: ObservedSchedulerOrderKindV1::FixedPartnerEligible,
            input_order: agents.clone(),
            shuffled_order: vec![agents[3], agents[1], agents[4], agents[0], agents[2]],
        };
        let present = [agents[3], agents[4], agents[2]]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            observed.project_shared_order(&present),
            vec![agents[3], agents[4], agents[2]]
        );
    }

    #[test]
    fn every_observation_must_fall_inside_the_subject_behavior_interval() {
        let agents = ids(2);
        let tape = ObservedStochasticTapeV1 {
            subject: subject(),
            scalar_draws: vec![ObservedStochasticDrawV1 {
                key: ObservedStochasticKeyV1::inheritance_source_sample(20, agents[0], 0),
                value: ObservedStochasticValueV1::unit(0.5),
            }],
            scheduler_orders: vec![],
        };
        assert!(matches!(
            tape.validate(),
            Err(ObservedStochasticTapeErrorV1::TickOutsideSubject {
                tick: 20,
                start: 10,
                end: 20
            })
        ));
    }

    #[test]
    fn mutation_keys_distinguish_birth_ordinal_trait_and_occurrence_from_perturbation() {
        let agents = ids(2);
        let mut keys = Vec::new();
        for birth in 0..2 {
            for trait_id in [
                GenomeTraitV1::SetPoint,
                GenomeTraitV1::ForageEfficiency,
                GenomeTraitV1::ActionTemperature,
                GenomeTraitV1::PerceptualGrain,
            ] {
                keys.push(ObservedStochasticKeyV1::mutation_occurrence(
                    11, agents[0], agents[1], birth, trait_id,
                ));
                keys.push(ObservedStochasticKeyV1::mutation_perturbation(
                    11, agents[0], agents[1], birth, trait_id,
                ));
            }
        }
        let unique = keys.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), keys.len());
    }
}
