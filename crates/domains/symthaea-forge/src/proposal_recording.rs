// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Behavior-preserving proposal-evidence binding.
//!
//! The live mutator owns proposal mechanics. Generator-local raw records preserve that exact draw;
//! this module is the only semantic bridge from those records into `ForgeProposalExposure`.
//! Binding performs no RNG operations and never reapplies a mutation.

use crate::mutations::{Mutator, RecordedMutation};
use crate::proposal_exposure::{
    ForgeFamilyOpportunity, ForgeProposalDecision, ForgeProposalExposure,
    ForgeProposalExposureError, ForgeProposalPolicy, ForgeProposalSelection,
};
use crate::proposal_trace::{
    ForgeRawProposalDecision, ForgeRawProposalError, ForgeRawProposalRecord,
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
    #[error(transparent)]
    Raw(#[from] ForgeRawProposalError),
    #[error("proposal policy generator does not match the semantic DiscoveryRun generator")]
    PolicyRunMismatch,
    #[error("Forge attempt does not belong to the semantic DiscoveryRun seed/generation budget")]
    AttemptRunMismatch,
    #[error("raw proposal opportunity order does not match the frozen semantic policy")]
    OpportunityOrderMismatch,
    #[error("raw proposal selected registration slot does not match the frozen semantic policy")]
    SelectionMismatch,
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

/// Convenience path for an in-memory live draw.
///
/// The live draw is first frozen as a generator-local raw record; semantic binding then follows the
/// same path used by persisted/reloaded evidence. There is intentionally one semantic converter.
pub fn exposure_from_recorded_mutation(
    run: &DiscoveryRun,
    attempt_id: ForgeAttemptId,
    parent_artifact_id: ContentId,
    policy: &ForgeProposalPolicy,
    recorded: &RecordedMutation,
) -> Result<ForgeProposalExposure, ForgeProposalRecordingError> {
    let raw = ForgeRawProposalRecord::from_recorded(attempt_id, parent_artifact_id, recorded)?;
    exposure_from_raw_record(run, policy, &raw)
}

/// Bind one self-validating generator-local proposal record to a semantic discovery run/policy.
///
/// A raw record may faithfully describe generator behavior that this v1 family policy cannot
/// represent (for example duplicate operator-family registrations). Such evidence is retained but
/// fails semantic qualification rather than being rewritten.
pub fn exposure_from_raw_record(
    run: &DiscoveryRun,
    policy: &ForgeProposalPolicy,
    raw: &ForgeRawProposalRecord,
) -> Result<ForgeProposalExposure, ForgeProposalRecordingError> {
    run.validate()?;
    policy.validate()?;
    raw.validate()?;

    if policy.generator_id() != &run.generator_id {
        return Err(ForgeProposalRecordingError::PolicyRunMismatch);
    }
    let attempt = raw.attempt_id();
    if attempt.seed() != run.seed || attempt.generation() >= run.budget.max_generations {
        return Err(ForgeProposalRecordingError::AttemptRunMismatch);
    }
    if raw.opportunities().len() != policy.families().len() {
        return Err(ForgeProposalRecordingError::OpportunityOrderMismatch);
    }

    let opportunities = raw
        .opportunities()
        .iter()
        .zip(policy.families())
        .enumerate()
        .map(|(index, (raw_opportunity, family))| {
            let expected_index = u64::try_from(index)
                .map_err(|_| ForgeProposalRecordingError::OpportunityOrderMismatch)?;
            if raw_opportunity.operator_index() != expected_index
                || raw_opportunity.operator() != family.operator()
            {
                return Err(ForgeProposalRecordingError::OpportunityOrderMismatch);
            }
            Ok(ForgeFamilyOpportunity::new(
                family.clone(),
                raw_opportunity.eligible_sites(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let decision = match raw.decision() {
        ForgeRawProposalDecision::NoEligibleSites => ForgeProposalDecision::NoEligibleSites,
        ForgeRawProposalDecision::Selected { selection } => {
            let index = usize::try_from(selection.operator_index())
                .map_err(|_| ForgeProposalRecordingError::SelectionMismatch)?;
            let family = policy
                .families()
                .get(index)
                .ok_or(ForgeProposalRecordingError::SelectionMismatch)?;
            if family.operator() != selection.operator() {
                return Err(ForgeProposalRecordingError::SelectionMismatch);
            }
            ForgeProposalDecision::Selected(ForgeProposalSelection::new(
                family.clone(),
                selection.site_index(),
                selection.global_pair_index(),
            ))
        }
    };

    let exposure = ForgeProposalExposure::new(
        run.id.clone(),
        raw.attempt_id().clone(),
        raw.parent_artifact_id().clone(),
        policy.clone(),
        opportunities,
        decision,
    )?;
    if exposure.total_eligible_sites() != raw.total_eligible_sites() {
        return Err(ForgeProposalRecordingError::OpportunityOrderMismatch);
    }
    Ok(exposure)
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
    fn live_and_raw_semantic_binding_are_identical() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32, y: i32) -> bool { x < 5 && y + 2 > 9 }");
        let mut rng = StdRng::seed_from_u64(41);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);

        let direct = exposure_from_recorded_mutation(
            &run,
            attempt.clone(),
            baseline.clone(),
            &policy,
            &recorded,
        )
        .unwrap();
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        let rebound = exposure_from_raw_record(&run, &policy, &raw).unwrap();
        assert_eq!(direct, rebound);
    }

    #[test]
    fn persisted_raw_round_trip_preserves_semantic_exposure() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32, y: i32) -> bool { x < 5 && y + 2 > 9 }");
        let mut rng = StdRng::seed_from_u64(53);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        let bytes = serde_json::to_vec(&raw).unwrap();
        let decoded: ForgeRawProposalRecord = serde_json::from_slice(&bytes).unwrap();
        decoded.validate().unwrap();
        assert_eq!(
            exposure_from_raw_record(&run, &policy, &raw).unwrap(),
            exposure_from_raw_record(&run, &policy, &decoded).unwrap()
        );
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
    fn reordered_raw_opportunities_fail_closed() {
        let run = run();
        let mutator = Mutator::default();
        let policy = proposal_policy_for_mutator(&run, &mutator).unwrap();
        let baseline = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&baseline, run.seed, 0, 0);
        let mut body = parse_body("fn f(x: i32) -> bool { x < 5 }");
        let mut rng = StdRng::seed_from_u64(3);
        let recorded = mutator.mutate_one_recorded(&mut body, &mut rng);
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();

        // A different policy order is semantically a different sampler interpretation.
        let reversed = ForgeProposalPolicy::uniform_eligible_site(
            run.generator_id.clone(),
            mutator.operator_names().collect::<Vec<_>>().into_iter().rev(),
        )
        .unwrap();
        assert!(matches!(
            exposure_from_raw_record(&run, &reversed, &raw),
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
        let raw = ForgeRawProposalRecord::from_recorded(attempt, baseline, &recorded).unwrap();
        let exposure = exposure_from_raw_record(&run, &policy, &raw).unwrap();
        assert!(matches!(exposure.decision(), ForgeProposalDecision::NoEligibleSites));
        assert_eq!(exposure.total_eligible_sites(), 0);
    }
}
