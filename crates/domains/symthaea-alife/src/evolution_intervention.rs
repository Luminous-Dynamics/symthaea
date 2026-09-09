// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure counterfactual intervention contract for one already-prepared evolutionary birth plan.
//!
//! The natural [`crate::EvolutionBirthPlanV1`] is prepared first, so genome-source selection and
//! mutation RNG draws have already occurred. An intervention then either leaves that exact plan
//! unchanged (sham control) or copies selected mutated traits back from the already-selected source
//! genome. The intervention consumes no RNG and cannot change parent/source ancestry.
//!
//! This module deliberately does **not** wire interventions into live `Population` execution yet.
//! It defines the exact subject, fail-closed preconditions, deterministic transformation, and
//! receipt needed for a later matched-randomness counterfactual execution profile.

use serde::{Deserialize, Serialize};

use crate::{AgentId, EvolutionBirthPlanV1, Genome, GenomeEvidenceV1};

/// Stable identifiers for the complete currently-heritable Genome v1 surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GenomeTraitV1 {
    SetPoint,
    ForageEfficiency,
    ActionTemperature,
    PerceptualGrain,
}

/// Exact natural birth subject an intervention is authorized to transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvolutionBirthSubjectV1 {
    tick: u64,
    reproductive_parent_index: u64,
    reproductive_parent_id: AgentId,
    reproductive_parent_genome: GenomeEvidenceV1,
    genome_source_index: u64,
    genome_source_id: AgentId,
    genome_source_genome: GenomeEvidenceV1,
    natural_offspring_genome: GenomeEvidenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BirthMutationInterventionKindV1 {
    /// Procedural control: bind to the same exact natural birth plan but make no genome change.
    Sham,
    /// Copy only these naturally-mutated traits back to the selected source genome.
    /// Traits must be non-empty and strictly sorted by [`GenomeTraitV1`].
    RevertTraits { traits: Vec<GenomeTraitV1> },
}

/// Serializable intervention request. Deserialization alone is not authority: application still
/// requires an exact live natural plan matching every subject field bit-for-bit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BirthMutationInterventionV1 {
    subject: EvolutionBirthSubjectV1,
    kind: BirthMutationInterventionKindV1,
}

/// Evidence that one exact natural plan was transformed into one exact counterfactual plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BirthMutationInterventionReceiptV1 {
    subject: EvolutionBirthSubjectV1,
    kind: BirthMutationInterventionKindV1,
    counterfactual_offspring_genome: GenomeEvidenceV1,
}

/// Pure applied result: the modified birth plan plus a serializable intervention receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedBirthMutationInterventionV1 {
    pub plan: EvolutionBirthPlanV1,
    pub receipt: BirthMutationInterventionReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BirthMutationInterventionErrorV1 {
    IndexTooLarge { field: &'static str, value: usize },
    NonFiniteGenome { role: &'static str, field: &'static str },
    TickMismatch { expected: u64, observed: u64 },
    ReproductiveParentIndexMismatch { expected: u64, observed: u64 },
    ReproductiveParentIdMismatch,
    ReproductiveParentGenomeMismatch,
    GenomeSourceIndexMismatch { expected: u64, observed: u64 },
    GenomeSourceIdMismatch,
    GenomeSourceGenomeMismatch,
    NaturalOffspringGenomeMismatch,
    EmptyTraitSet,
    TraitsNotStrictlySorted,
    TraitWasNotMutated { trait_id: GenomeTraitV1 },
    NoNaturalMutation,
}

impl EvolutionBirthSubjectV1 {
    pub fn from_plan(
        tick: u64,
        plan: &EvolutionBirthPlanV1,
    ) -> Result<Self, BirthMutationInterventionErrorV1> {
        validate_genome("reproductive_parent", plan.reproductive_parent_genome)?;
        validate_genome("genome_source", plan.genome_source_genome)?;
        validate_genome("natural_offspring", plan.offspring_genome)?;
        Ok(Self {
            tick,
            reproductive_parent_index: index_to_u64(
                "reproductive_parent_index",
                plan.reproductive_parent_index,
            )?,
            reproductive_parent_id: plan.reproductive_parent_id,
            reproductive_parent_genome: plan.reproductive_parent_genome.into(),
            genome_source_index: index_to_u64("genome_source_index", plan.genome_source_index)?,
            genome_source_id: plan.genome_source_id,
            genome_source_genome: plan.genome_source_genome.into(),
            natural_offspring_genome: plan.offspring_genome.into(),
        })
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn reproductive_parent_id(&self) -> AgentId {
        self.reproductive_parent_id
    }

    pub fn genome_source_id(&self) -> AgentId {
        self.genome_source_id
    }

    pub fn natural_offspring_genome(&self) -> GenomeEvidenceV1 {
        self.natural_offspring_genome
    }
}

impl BirthMutationInterventionV1 {
    pub fn sham_for_plan(
        tick: u64,
        plan: &EvolutionBirthPlanV1,
    ) -> Result<Self, BirthMutationInterventionErrorV1> {
        Ok(Self {
            subject: EvolutionBirthSubjectV1::from_plan(tick, plan)?,
            kind: BirthMutationInterventionKindV1::Sham,
        })
    }

    pub fn revert_traits_for_plan(
        tick: u64,
        plan: &EvolutionBirthPlanV1,
        traits: Vec<GenomeTraitV1>,
    ) -> Result<Self, BirthMutationInterventionErrorV1> {
        validate_trait_set(&traits)?;
        let subject = EvolutionBirthSubjectV1::from_plan(tick, plan)?;
        for &trait_id in &traits {
            if !trait_changed_exact(
                trait_id,
                subject.genome_source_genome,
                subject.natural_offspring_genome,
            ) {
                return Err(BirthMutationInterventionErrorV1::TraitWasNotMutated {
                    trait_id,
                });
            }
        }
        Ok(Self {
            subject,
            kind: BirthMutationInterventionKindV1::RevertTraits { traits },
        })
    }

    /// Revert every trait that actually differs between the selected source and natural offspring.
    pub fn revert_all_changed_for_plan(
        tick: u64,
        plan: &EvolutionBirthPlanV1,
    ) -> Result<Self, BirthMutationInterventionErrorV1> {
        let subject = EvolutionBirthSubjectV1::from_plan(tick, plan)?;
        let traits = changed_traits_exact(
            subject.genome_source_genome,
            subject.natural_offspring_genome,
        );
        if traits.is_empty() {
            return Err(BirthMutationInterventionErrorV1::NoNaturalMutation);
        }
        Ok(Self {
            subject,
            kind: BirthMutationInterventionKindV1::RevertTraits { traits },
        })
    }

    pub fn subject(&self) -> &EvolutionBirthSubjectV1 {
        &self.subject
    }

    pub fn kind(&self) -> &BirthMutationInterventionKindV1 {
        &self.kind
    }

    /// Apply the intervention to an exact natural birth plan without consuming any RNG.
    pub fn apply(
        &self,
        tick: u64,
        natural_plan: &EvolutionBirthPlanV1,
    ) -> Result<AppliedBirthMutationInterventionV1, BirthMutationInterventionErrorV1> {
        validate_subject_match(&self.subject, tick, natural_plan)?;

        let mut plan = *natural_plan;
        match &self.kind {
            BirthMutationInterventionKindV1::Sham => {}
            BirthMutationInterventionKindV1::RevertTraits { traits } => {
                validate_trait_set(traits)?;
                for &trait_id in traits {
                    if !trait_changed_exact(
                        trait_id,
                        self.subject.genome_source_genome,
                        self.subject.natural_offspring_genome,
                    ) {
                        return Err(BirthMutationInterventionErrorV1::TraitWasNotMutated {
                            trait_id,
                        });
                    }
                    copy_trait_from_source(
                        trait_id,
                        &mut plan.offspring_genome,
                        natural_plan.genome_source_genome,
                    );
                }
            }
        }

        let receipt = BirthMutationInterventionReceiptV1 {
            subject: self.subject.clone(),
            kind: self.kind.clone(),
            counterfactual_offspring_genome: plan.offspring_genome.into(),
        };
        Ok(AppliedBirthMutationInterventionV1 { plan, receipt })
    }
}

impl BirthMutationInterventionReceiptV1 {
    pub fn subject(&self) -> &EvolutionBirthSubjectV1 {
        &self.subject
    }

    pub fn kind(&self) -> &BirthMutationInterventionKindV1 {
        &self.kind
    }

    pub fn natural_offspring_genome(&self) -> GenomeEvidenceV1 {
        self.subject.natural_offspring_genome
    }

    pub fn counterfactual_offspring_genome(&self) -> GenomeEvidenceV1 {
        self.counterfactual_offspring_genome
    }
}

pub fn changed_traits_for_plan_v1(
    plan: &EvolutionBirthPlanV1,
) -> Result<Vec<GenomeTraitV1>, BirthMutationInterventionErrorV1> {
    validate_genome("genome_source", plan.genome_source_genome)?;
    validate_genome("natural_offspring", plan.offspring_genome)?;
    Ok(changed_traits_exact(
        plan.genome_source_genome.into(),
        plan.offspring_genome.into(),
    ))
}

fn validate_subject_match(
    expected: &EvolutionBirthSubjectV1,
    tick: u64,
    plan: &EvolutionBirthPlanV1,
) -> Result<(), BirthMutationInterventionErrorV1> {
    let observed = EvolutionBirthSubjectV1::from_plan(tick, plan)?;
    if observed.tick != expected.tick {
        return Err(BirthMutationInterventionErrorV1::TickMismatch {
            expected: expected.tick,
            observed: observed.tick,
        });
    }
    if observed.reproductive_parent_index != expected.reproductive_parent_index {
        return Err(
            BirthMutationInterventionErrorV1::ReproductiveParentIndexMismatch {
                expected: expected.reproductive_parent_index,
                observed: observed.reproductive_parent_index,
            },
        );
    }
    if observed.reproductive_parent_id != expected.reproductive_parent_id {
        return Err(BirthMutationInterventionErrorV1::ReproductiveParentIdMismatch);
    }
    if observed.reproductive_parent_genome != expected.reproductive_parent_genome {
        return Err(BirthMutationInterventionErrorV1::ReproductiveParentGenomeMismatch);
    }
    if observed.genome_source_index != expected.genome_source_index {
        return Err(BirthMutationInterventionErrorV1::GenomeSourceIndexMismatch {
            expected: expected.genome_source_index,
            observed: observed.genome_source_index,
        });
    }
    if observed.genome_source_id != expected.genome_source_id {
        return Err(BirthMutationInterventionErrorV1::GenomeSourceIdMismatch);
    }
    if observed.genome_source_genome != expected.genome_source_genome {
        return Err(BirthMutationInterventionErrorV1::GenomeSourceGenomeMismatch);
    }
    if observed.natural_offspring_genome != expected.natural_offspring_genome {
        return Err(BirthMutationInterventionErrorV1::NaturalOffspringGenomeMismatch);
    }
    Ok(())
}

fn validate_trait_set(traits: &[GenomeTraitV1]) -> Result<(), BirthMutationInterventionErrorV1> {
    if traits.is_empty() {
        return Err(BirthMutationInterventionErrorV1::EmptyTraitSet);
    }
    if traits.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(BirthMutationInterventionErrorV1::TraitsNotStrictlySorted);
    }
    Ok(())
}

fn changed_traits_exact(source: GenomeEvidenceV1, natural: GenomeEvidenceV1) -> Vec<GenomeTraitV1> {
    const ALL: [GenomeTraitV1; 4] = [
        GenomeTraitV1::SetPoint,
        GenomeTraitV1::ForageEfficiency,
        GenomeTraitV1::ActionTemperature,
        GenomeTraitV1::PerceptualGrain,
    ];
    ALL.into_iter()
        .filter(|&trait_id| trait_changed_exact(trait_id, source, natural))
        .collect()
}

fn trait_changed_exact(
    trait_id: GenomeTraitV1,
    source: GenomeEvidenceV1,
    natural: GenomeEvidenceV1,
) -> bool {
    match trait_id {
        GenomeTraitV1::SetPoint => source.set_point_bits != natural.set_point_bits,
        GenomeTraitV1::ForageEfficiency => {
            source.forage_efficiency_bits != natural.forage_efficiency_bits
        }
        GenomeTraitV1::ActionTemperature => {
            source.action_temperature_bits != natural.action_temperature_bits
        }
        GenomeTraitV1::PerceptualGrain => {
            source.perceptual_grain_bits != natural.perceptual_grain_bits
        }
    }
}

fn copy_trait_from_source(trait_id: GenomeTraitV1, target: &mut Genome, source: Genome) {
    match trait_id {
        GenomeTraitV1::SetPoint => target.set_point = source.set_point,
        GenomeTraitV1::ForageEfficiency => target.forage_efficiency = source.forage_efficiency,
        GenomeTraitV1::ActionTemperature => target.action_temperature = source.action_temperature,
        GenomeTraitV1::PerceptualGrain => target.perceptual_grain = source.perceptual_grain,
    }
}

fn validate_genome(
    role: &'static str,
    genome: Genome,
) -> Result<(), BirthMutationInterventionErrorV1> {
    for (field, value) in [
        ("set_point", genome.set_point),
        ("forage_efficiency", genome.forage_efficiency),
        ("action_temperature", genome.action_temperature),
    ] {
        if !value.is_finite() {
            return Err(BirthMutationInterventionErrorV1::NonFiniteGenome { role, field });
        }
    }
    if genome.perceptual_grain.is_some_and(|value| !value.is_finite()) {
        return Err(BirthMutationInterventionErrorV1::NonFiniteGenome {
            role,
            field: "perceptual_grain",
        });
    }
    Ok(())
}

fn index_to_u64(
    field: &'static str,
    value: usize,
) -> Result<u64, BirthMutationInterventionErrorV1> {
    u64::try_from(value).map_err(|_| BirthMutationInterventionErrorV1::IndexTooLarge { field, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, EvolutionRngStreamsV1, InheritanceMode, Organism, OrganismConfig,
        prepare_evolution_birth_v1,
    };

    fn controlled_plan() -> EvolutionBirthPlanV1 {
        let mut ids = AgentIdAllocator::new();
        let parent_id = ids.allocate();
        let source_id = ids.allocate();
        let parent = Genome {
            set_point: 0.60,
            forage_efficiency: 0.50,
            action_temperature: 0.90,
            perceptual_grain: Some(0.20),
        };
        let source = Genome {
            set_point: 0.55,
            forage_efficiency: 0.70,
            action_temperature: 1.10,
            perceptual_grain: Some(0.25),
        };
        let natural = Genome {
            set_point: 0.58,
            forage_efficiency: 0.74,
            action_temperature: 1.10,
            perceptual_grain: Some(0.21),
        };
        EvolutionBirthPlanV1 {
            reproductive_parent_index: 0,
            reproductive_parent_id: parent_id,
            reproductive_parent_genome: parent,
            genome_source_index: 1,
            genome_source_id: source_id,
            genome_source_genome: source,
            offspring_genome: natural,
        }
    }

    #[test]
    fn selective_revert_changes_only_the_requested_natural_mutations() {
        let plan = controlled_plan();
        let intervention = BirthMutationInterventionV1::revert_traits_for_plan(
            17,
            &plan,
            vec![GenomeTraitV1::SetPoint, GenomeTraitV1::PerceptualGrain],
        )
        .expect("valid selective intervention");
        let applied = intervention.apply(17, &plan).expect("apply intervention");

        assert_eq!(applied.plan.offspring_genome.set_point.to_bits(), plan.genome_source_genome.set_point.to_bits());
        assert_eq!(applied.plan.offspring_genome.perceptual_grain.map(f64::to_bits), plan.genome_source_genome.perceptual_grain.map(f64::to_bits));
        assert_eq!(applied.plan.offspring_genome.forage_efficiency.to_bits(), plan.offspring_genome.forage_efficiency.to_bits());
        assert_eq!(applied.plan.offspring_genome.action_temperature.to_bits(), plan.offspring_genome.action_temperature.to_bits());
    }

    #[test]
    fn revert_all_changed_traits_recovers_the_selected_source_genome_exactly() {
        let plan = controlled_plan();
        let intervention = BirthMutationInterventionV1::revert_all_changed_for_plan(9, &plan)
            .expect("natural plan contains mutations");
        let applied = intervention.apply(9, &plan).expect("apply full revert");
        assert_eq!(
            GenomeEvidenceV1::from_genome(applied.plan.offspring_genome),
            GenomeEvidenceV1::from_genome(plan.genome_source_genome)
        );
    }

    #[test]
    fn sham_control_preserves_the_natural_plan_bit_exactly() {
        let plan = controlled_plan();
        let intervention = BirthMutationInterventionV1::sham_for_plan(4, &plan).unwrap();
        let applied = intervention.apply(4, &plan).unwrap();
        assert_eq!(
            GenomeEvidenceV1::from_genome(applied.plan.offspring_genome),
            GenomeEvidenceV1::from_genome(plan.offspring_genome)
        );
        assert_eq!(applied.plan, plan);
    }

    #[test]
    fn intervention_consumes_no_additional_evolution_rng() {
        let mut ids = AgentIdAllocator::new();
        let mut organisms = Vec::new();
        for i in 0..4u64 {
            let mut cfg = OrganismConfig::default();
            cfg.perceptual_grain = Some(0.2);
            organisms.push(Organism::new(cfg, 100 + i).with_id(ids.allocate()));
        }
        let mut rng = EvolutionRngStreamsV1::new(0xA11F_CA55);
        let plan = prepare_evolution_birth_v1(
            &organisms,
            1,
            InheritanceMode::RandomPeer,
            1.0,
            0.04,
            &mut rng,
        )
        .expect("natural plan");
        let rng_after_natural_plan = rng.snapshot();
        let intervention = BirthMutationInterventionV1::revert_all_changed_for_plan(22, &plan)
            .expect("always-mutate plan should differ");
        let _ = intervention.apply(22, &plan).expect("apply intervention");
        assert_eq!(rng.snapshot(), rng_after_natural_plan);
    }

    #[test]
    fn exact_subject_binding_rejects_wrong_tick_and_tampered_natural_genome() {
        let plan = controlled_plan();
        let intervention = BirthMutationInterventionV1::sham_for_plan(7, &plan).unwrap();
        assert_eq!(
            intervention.apply(8, &plan).unwrap_err(),
            BirthMutationInterventionErrorV1::TickMismatch {
                expected: 7,
                observed: 8,
            }
        );

        let mut tampered = plan;
        tampered.offspring_genome.set_point = f64::from_bits(
            tampered.offspring_genome.set_point.to_bits() ^ 1,
        );
        assert_eq!(
            intervention.apply(7, &tampered).unwrap_err(),
            BirthMutationInterventionErrorV1::NaturalOffspringGenomeMismatch
        );
    }

    #[test]
    fn revert_set_must_be_canonical_and_must_name_real_mutations() {
        let plan = controlled_plan();
        assert_eq!(
            BirthMutationInterventionV1::revert_traits_for_plan(
                1,
                &plan,
                vec![GenomeTraitV1::PerceptualGrain, GenomeTraitV1::SetPoint],
            )
            .unwrap_err(),
            BirthMutationInterventionErrorV1::TraitsNotStrictlySorted
        );
        assert_eq!(
            BirthMutationInterventionV1::revert_traits_for_plan(
                1,
                &plan,
                vec![GenomeTraitV1::ActionTemperature],
            )
            .unwrap_err(),
            BirthMutationInterventionErrorV1::TraitWasNotMutated {
                trait_id: GenomeTraitV1::ActionTemperature,
            }
        );
    }
}
