// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Behavior-preserving bridge from the live Forge mutator to proposal-exposure evidence.
//!
//! The mutator owns proposal mechanics. This module does not resample or reconstruct a choice from
//! RNG state; it validates and converts the exact [`crate::mutations::RecordedMutation`] emitted by
//! that proposal into the typed [`crate::proposal_exposure::ForgeProposalExposure`] contract.

use crate::mutations::{Mutator, RecordedMutation};
use crate::proposal_exposure::{
    ForgeFamilyOpportunity, ForgeProposalDecision, ForgeProposalExposure,
    ForgeProposalExposureError, ForgeProposalPolicy, ForgeProposalSelection,
};
use crate::trace::ForgeAttemptId;
use symthaea_algorithms::discovery::{DiscoveryError, DiscoveryRun};
use symthaea_algorithms::ContentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalRecordingError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Exposure(#[from] ForgeProposalExposureError),
    #[error("proposal policy generator does not match the semantic DiscoveryRun generator")]
    PolicyRunMismatch,
    #[error("Forge attempt does not belong to the semantic DiscoveryRun seed/generation budget")]
    AttemptRunMismatch,
    #[error("recorded mutator opportunity order does not match the frozen proposal policy")]
    OpportunityOrderMismatch,
    #[error("recorded mutator opportunity count cannot be represented in the exposure contract")]
    OpportunityCountOverflow,
    #[error("recorded mutator total does not equal the checked opportunity sum")]
    OpportunityTotalMismatch,
    #[error("recorded mutator selection is inconsistent with its opportunity set")]
    SelectionMismatch,
    #[error("recorded mutation operator does not equal the operator selected by the proposal draw")]
    MutationOperatorMismatch,
}

/// Build the exact `UniformEligibleSiteV1` policy corresponding to this mutator's registered
/// operator order under the semantic run's exact generator identity.
pub fn proposal_policy_for_mutator(
    run: &DiscoveryRun,
    mutator: &Mutator,
) -> Result<ForgeProposalPolicy, ForgeProposalRecordingError> {
    run.validate()?;
    Ok(ForgeProposalPolicy::uniform_eligible_site(
        run.generator_id.clone(),
        mutator.operator_names(),
    )?)
}

/// Convert one already-executed mutator draw into a typed proposal-exposure record.
///
/// This function performs no RNG operations and never reapplies a mutation. The exact live draw is
/// accepted only if its ordered opportunities, total, selected pair and optional mutation operator
/// agree with the supplied generator-scoped policy.
pub fn exposure_from_recorded_mutation(
    run: &DiscoveryRun,
    attempt_id: ForgeAttemptId,
    parent_artifact_id: ContentId,
    policy: &ForgeProposalPolicy,
    recorded: &RecordedMutation,
) -> Result<ForgeProposalExposure, ForgeProposalRecordingError> {
    run.validate()?;
    policy.validate()?;
    if policy.generator_id() != &run.generator_id {
        return Err(ForgeProposalRecordingError::PolicyRunMismatch);
    }
    if attempt_id.seed() != run.seed || attempt_id.generation() >= run.budget.max_generations {
        return Err(ForgeProposalRecordingError::AttemptRunMismatch);
    }
    if recorded.opportunities.len() != policy.families().len() {
        return Err(ForgeProposalRecordingError::OpportunityOrderMismatch);
    }

    let mut opportunities = Vec::with_capacity(recorded.opportunities.len());
    let mut checked_total = 0u64;
    for (recorded_opportunity, family) in recorded.opportunities.iter().zip(policy.families()) {
        if recorded_opportunity.operator != family.operator() {
            return Err(ForgeProposalRecordingError::OpportunityOrderMismatch);
        }
        let eligible_sites = u64::try_from(recorded_opportunity.eligible_sites)
            .map_err(|_| ForgeProposalRecordingError::OpportunityCountOverflow)?;
        checked_total = checked_total
            .checked_add(eligible_sites)
            .ok_or(ForgeProposalRecordingError::OpportunityCountOverflow)?;
        opportunities.push(ForgeFamilyOpportunity::new(family.clone(), eligible_sites));
    }

    let recorded_total = u64::try_from(recorded.total_eligible_sites)
        .map_err(|_| ForgeProposalRecordingError::OpportunityCountOverflow)?;
    if checked_total != recorded_total {
        return Err(ForgeProposalRecordingError::OpportunityTotalMismatch);
    }

    let decision = match &recorded.selection {
        None => {
            if recorded_total != 0 || recorded.mutation.is_some() {
                return Err(ForgeProposalRecordingError::SelectionMismatch);
            }
            ForgeProposalDecision::NoEligibleSites
        }
        Some(selection) => {
            if recorded_total == 0 {
                return Err(ForgeProposalRecordingError::SelectionMismatch);
            }
            let recorded_opportunity = recorded
                .opportunities
                .get(selection.operator_index)
                .ok_or(ForgeProposalRecordingError::SelectionMismatch)?;
            if recorded_opportunity.operator != selection.operator {
                return Err(ForgeProposalRecordingError::SelectionMismatch);
            }
            if recorded
                .mutation
                .as_ref()
                .is_some_and(|mutation| mutation.operator != selection.operator)
            {
                return Err(ForgeProposalRecordingError::MutationOperatorMismatch);
            }
            let family = policy
                .families()
                .get(selection.operator_index)
                .ok_or(ForgeProposalRecordingError::SelectionMismatch)?
                .clone();
            if family.operator() != selection.operator {
                return Err(ForgeProposalRecordingError::SelectionMismatch);
            }
            let site_index = u64::try_from(selection.site_index)
                .map_err(|_| ForgeProposalRecordingError::OpportunityCountOverflow)?;
            let global_pair_index = u64::try_from(selection.global_pair_index)
                .map_err(|_| ForgeProposalRecordingError::OpportunityCountOverflow)?;
            ForgeProposalDecision::Selected(ForgeProposalSelection::new(
                family,
                site_index,
                global_pair_index,
            ))
        }
    };

    Ok(ForgeProposalExposure::new(
        run.id.clone(),
        attempt_id,
        parent_artifact_id,
        policy.clone(),
        opportunities,
        decision,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use crate::mutations::{ComparisonOperatorSwap, NumericLiteralPerturb};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        DeterminismRequirement, DiscoveryRisk, ProblemSpec, SemanticGuarantee,
    };

    fn run() -> DiscoveryRun {
        let problem = ProblemSpec::new(
            "proposal-recording-test",
            "Exact test contract.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge-recorded-v1".as_slice()]),
            "abc123",
            SearchBudget::new(16, 4, 16).unwrap(),
            19,
        )
        .unwrap()
    }

    fn parse_body(src: &str) -> syn::Block {
        let file: syn::File = syn::parse_str(src).unwrap();
        match &file.items[0] {
            syn::Item::Fn(function) => (*function.block).clone(),
            _ => panic!("expected free function fixture"),
        }
    }

    #[test]
    fn live_recorded_draw_converts_without_resampling() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32, y: i32) -> bool { x < 5 && y + 2 > 9 }");
        let mut rng = StdRng::seed_from_u64(41);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);

        let exposure = exposure_from_recorded_mutation(
            &run,
            attempt,
            baseline,
            &policy,
            &recorded,
        )
        .unwrap();

        assert_eq!(
            exposure.total_eligible_sites(),
            u64::try_from(recorded.total_eligible_sites).unwrap()
        );
        assert_eq!(exposure.opportunities().len(), recorded.opportunities.len());
        match (exposure.decision(), recorded.selection.as_ref()) {
            (ForgeProposalDecision::Selected(observed), Some(recorded_selection)) => {
                assert_eq!(
                    observed.family_id(),
                    &policy.families()[recorded_selection.operator_index]
                );
                assert_eq!(observed.family_id().operator(), recorded_selection.operator);
                assert_eq!(observed.site_index(), recorded_selection.site_index as u64);
                assert_eq!(
                    observed.global_pair_index(),
                    recorded_selection.global_pair_index as u64
                );
            }
            (ForgeProposalDecision::NoEligibleSites, None) => {}
            _ => panic!("typed exposure must preserve the live mutator decision"),
        }
    }

    #[test]
    fn selected_pair_without_source_change_remains_selected_exposure() {
        let run = run();
        let mutator = Mutator::new(vec![Box::new(NumericLiteralPerturb { max_fraction: 0.1 })]);
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f() -> i64 { 0 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f() -> i64 { 0 }");
        let mut rng = StdRng::seed_from_u64(7);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        assert!(recorded.selection.is_some());
        assert!(recorded.mutation.is_none());

        let exposure = exposure_from_recorded_mutation(
            &run,
            attempt,
            baseline,
            &policy,
            &recorded,
        )
        .unwrap();
        assert!(matches!(exposure.decision(), ForgeProposalDecision::Selected(_)));
    }

    #[test]
    fn tampered_operator_slot_fails_closed() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32, y: i32) -> bool { x < 5 && y + 2 > 9 }");
        let mut rng = StdRng::seed_from_u64(41);
        let mut recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let selection = recorded.selection.as_mut().unwrap();
        selection.operator_index = (selection.operator_index + 1) % recorded.opportunities.len();

        assert!(matches!(
            exposure_from_recorded_mutation(
                &run,
                attempt,
                baseline,
                &policy,
                &recorded,
            ),
            Err(ForgeProposalRecordingError::SelectionMismatch)
        ));
    }

    #[test]
    fn reordered_opportunities_fail_closed() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32) -> bool { x < 5 }");
        let mut rng = StdRng::seed_from_u64(3);
        let mut recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        recorded.opportunities.swap(0, 1);

        assert!(matches!(
            exposure_from_recorded_mutation(
                &run,
                attempt,
                baseline,
                &policy,
                &recorded,
            ),
            Err(ForgeProposalRecordingError::OpportunityOrderMismatch)
        ));
    }

    #[test]
    fn zero_site_record_converts_to_no_eligible_sites() {
        let run = run();
        let mutator = Mutator::new(vec![Box::new(ComparisonOperatorSwap)]);
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f() -> &'static str { \"x\" }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f() -> &'static str { \"x\" }");
        let mut rng = StdRng::seed_from_u64(2);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);

        let exposure = exposure_from_recorded_mutation(
            &run,
            attempt,
            baseline,
            &policy,
            &recorded,
        )
        .unwrap();
        assert!(matches!(exposure.decision(), ForgeProposalDecision::NoEligibleSites));
        assert_eq!(exposure.total_eligible_sites(), 0);
    }
}
