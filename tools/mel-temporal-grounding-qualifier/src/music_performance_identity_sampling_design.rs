//! Sampling-design admission for property-relative musical identity inference.
//!
//! This module classifies work / performer / session / take incidence before
//! the source-work-only comparator is used. It does not estimate effects,
//! authenticate people, or implement crossed-cluster inference.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::music_performance_identity::{
    MusicalPerformanceIdentityCaseV1, MusicalPerformanceIdentityTriadV1,
    MusicalIdentityVariantV1,
};
use crate::music_performance_identity_inference::{
    PairedMusicalIdentityComparisonCaseV1, PairedMusicalIdentityComparisonReportV1,
    audit_paired_musical_identity_comparison,
};

const DESIGN_SCHEMA_V1: &str = "mel-music-performance-identity-sampling-design-v1";
const REPORT_SCHEMA_V1: &str = "mel-music-performance-identity-sampling-design-report-v1";
const QUALIFIED_REPORT_SCHEMA_V1: &str =
    "mel-music-performance-identity-sampling-design-qualified-comparison-report-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceIdentitySamplingDesignV1 {
    pub schema: String,
    pub sampling_frame_id: String,
    pub target_population: String,
    pub inclusion_exclusion_policy: String,
    pub pseudonymization_scope: String,
    /// A study-design declaration, not proof that works were sampled independently.
    pub independent_source_work_sampling_claimed: bool,
    pub independence_basis: String,
    pub observations: Vec<PerformanceIdentitySamplingObservationV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceIdentitySamplingObservationV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_work_lineage_sha256: String,
    pub clean: PerformanceIdentityArmMembershipV1,
    pub performance_stay: PerformanceIdentityArmMembershipV1,
    pub musical_flip: PerformanceIdentityArmMembershipV1,
}

/// The scalar performer identity is joined to the current #13947 reducer.
/// The membership vector records every declared participant in an ensemble;
/// it must include the reducer-bound performer identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceIdentityArmMembershipV1 {
    pub performer_identity_sha256: String,
    pub performer_membership_sha256: Vec<String>,
    pub session_identity_sha256: String,
    pub take_identity_sha256: String,
    pub performance_identity_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingDesignShapeV1 {
    WorkDisjointClusters,
    NestedWithinWork,
    CrossedPerformerSessionOrTake,
    InsufficientlyIdentified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingDesignDispositionV1 {
    ConditionalSourceWorkBootstrap,
    CrossedInferenceRequired,
    DescriptiveOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PerformanceIdentitySamplingDesignReportV1 {
    pub schema: &'static str,
    pub design_shape: SamplingDesignShapeV1,
    pub disposition: SamplingDesignDispositionV1,
    pub observation_count: usize,
    pub source_work_cluster_count: usize,
    pub performer_identity_count: usize,
    pub session_identity_count: usize,
    pub take_identity_count: usize,
    pub performance_identity_count: usize,
    pub crossed_performer_count: usize,
    pub crossed_session_count: usize,
    pub crossed_take_count: usize,
    pub crossed_performance_count: usize,
    pub multi_performer_arm_count: usize,
    pub independent_source_work_sampling_claimed: bool,
    pub independence_basis_present: bool,
    /// This is a conditional admission flag, not an authenticated independence claim.
    pub work_cluster_bootstrap_permitted_under_declared_assumption: bool,
    pub crossed_inference_required: bool,
    pub performer_identity_authenticated: bool,
    pub source_work_identity_authenticated: bool,
    pub sampling_independence_authenticated: bool,
    pub population_generalization_established: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SamplingDesignQualifiedComparisonReportV1 {
    pub schema: &'static str,
    pub sampling_design: PerformanceIdentitySamplingDesignReportV1,
    pub paired_comparison: PairedMusicalIdentityComparisonReportV1,
}

/// Classify the supplied incidence structure without making an inferential claim.
///
/// A repeated performer, session, or take across source works is a crossed
/// dependency. A multi-performer ensemble confined to one work is nested,
/// not crossed. Missing pseudonyms or design metadata force descriptive-only.
pub fn audit_performance_identity_sampling_design(
    design: &PerformanceIdentitySamplingDesignV1,
) -> Result<PerformanceIdentitySamplingDesignReportV1, String> {
    if design.schema != DESIGN_SCHEMA_V1 {
        return Err(format!(
            "unsupported sampling design schema: {}",
            design.schema
        ));
    }

    let mut incomplete = false;
    if design.sampling_frame_id.trim().is_empty()
        || design.target_population.trim().is_empty()
        || design.inclusion_exclusion_policy.trim().is_empty()
        || design.pseudonymization_scope.trim().is_empty()
    {
        incomplete = true;
    }

    let mut observation_ids = BTreeSet::new();
    let mut work_ids = BTreeSet::new();
    let mut performer_works = BTreeMap::<String, BTreeSet<String>>::new();
    let mut session_works = BTreeMap::<String, BTreeSet<String>>::new();
    let mut take_works = BTreeMap::<String, BTreeSet<String>>::new();
    let mut performance_works = BTreeMap::<String, BTreeSet<String>>::new();
    let mut work_performers = BTreeMap::<String, BTreeSet<String>>::new();
    let mut work_sessions = BTreeMap::<String, BTreeSet<String>>::new();
    let mut work_takes = BTreeMap::<String, BTreeSet<String>>::new();
    let mut performer_work_sessions =
        BTreeMap::<(String, String), BTreeSet<String>>::new();
    let mut multi_performer_arm_count = 0usize;

    for observation in &design.observations {
        if observation.item_id.trim().is_empty() || observation.pair_id.trim().is_empty() {
            incomplete = true;
        }
        if observation.item_id.trim().is_empty() {
            continue;
        }
        if !observation_ids.insert(observation.item_id.clone()) {
            return Err(format!(
                "duplicate sampling-design item_id: {}",
                observation.item_id
            ));
        }

        let work = observation.source_work_lineage_sha256.trim();
        if work.is_empty() {
            incomplete = true;
        } else {
            crate::music_listen_bench::validate_digest(
                work,
                "source_work_lineage_sha256",
            )?;
            work_ids.insert(work.to_string());
        }

        for arm in [
            &observation.clean,
            &observation.performance_stay,
            &observation.musical_flip,
        ] {
            let session = arm.session_identity_sha256.trim();
            let take = arm.take_identity_sha256.trim();
            let performance = arm.performance_identity_sha256.trim();
            let primary = arm.performer_identity_sha256.trim();

            if session.is_empty() || take.is_empty() || performance.is_empty() || primary.is_empty()
            {
                incomplete = true;
            }
            if !session.is_empty() {
                crate::music_listen_bench::validate_digest(
                    session,
                    "session_identity_sha256",
                )?;
                if !work.is_empty() {
                    session_works
                        .entry(session.to_string())
                        .or_default()
                        .insert(work.to_string());
                    work_sessions
                        .entry(work.to_string())
                        .or_default()
                        .insert(session.to_string());
                }
            }
            if !take.is_empty() {
                crate::music_listen_bench::validate_digest(take, "take_identity_sha256")?;
                if !work.is_empty() {
                    take_works
                        .entry(take.to_string())
                        .or_default()
                        .insert(work.to_string());
                    work_takes
                        .entry(work.to_string())
                        .or_default()
                        .insert(take.to_string());
                }
            }
            if !performance.is_empty() {
                crate::music_listen_bench::validate_digest(
                    performance,
                    "performance_identity_sha256",
                )?;
                if !work.is_empty() {
                    performance_works
                        .entry(performance.to_string())
                        .or_default()
                        .insert(work.to_string());
                }
            }
            if !primary.is_empty() {
                crate::music_listen_bench::validate_digest(
                    primary,
                    "performer_identity_sha256",
                )?;
            }

            if arm.performer_membership_sha256.len() > 1 {
                multi_performer_arm_count += 1;
            }
            let mut arm_members = BTreeSet::new();
            for member in &arm.performer_membership_sha256 {
                let member = member.trim();
                if member.is_empty() {
                    incomplete = true;
                    continue;
                }
                crate::music_listen_bench::validate_digest(
                    member,
                    "performer_membership_sha256",
                )?;
                if !arm_members.insert(member.to_string()) {
                    return Err(format!(
                        "duplicate performer identity in one arm for item {}",
                        observation.item_id
                    ));
                }
                if !work.is_empty() {
                    performer_works
                        .entry(member.to_string())
                        .or_default()
                        .insert(work.to_string());
                    work_performers
                        .entry(work.to_string())
                        .or_default()
                        .insert(member.to_string());
                    if !session.is_empty() {
                        performer_work_sessions
                            .entry((work.to_string(), member.to_string()))
                            .or_default()
                            .insert(session.to_string());
                    }
                }
            }
            if arm_members.is_empty() || (!primary.is_empty() && !arm_members.contains(primary)) {
                incomplete = true;
            }
        }
    }

    if design.observations.is_empty() {
        incomplete = true;
    }

    let crossed_performer_count = performer_works
        .values()
        .filter(|works| works.len() > 1)
        .count();
    let crossed_session_count = session_works
        .values()
        .filter(|works| works.len() > 1)
        .count();
    let crossed_take_count = take_works.values().filter(|works| works.len() > 1).count();
    let crossed_performance_count = performance_works
        .values()
        .filter(|works| works.len() > 1)
        .count();
    let crossed = crossed_performer_count > 0
        || crossed_session_count > 0
        || crossed_take_count > 0
        || crossed_performance_count > 0;

    let nested = work_performers.values().any(|performers| performers.len() > 1)
        || work_sessions.values().any(|sessions| sessions.len() > 1)
        || work_takes.values().any(|takes| takes.len() > 1)
        || performer_work_sessions
            .values()
            .any(|sessions| sessions.len() > 1);

    let design_shape = if incomplete {
        SamplingDesignShapeV1::InsufficientlyIdentified
    } else if crossed {
        SamplingDesignShapeV1::CrossedPerformerSessionOrTake
    } else if nested {
        SamplingDesignShapeV1::NestedWithinWork
    } else {
        SamplingDesignShapeV1::WorkDisjointClusters
    };

    let independence_basis_present = !design.independence_basis.trim().is_empty();
    let work_cluster_bootstrap_permitted_under_declared_assumption = !incomplete
        && !crossed
        && design.independent_source_work_sampling_claimed
        && independence_basis_present;

    let disposition = if crossed {
        SamplingDesignDispositionV1::CrossedInferenceRequired
    } else if work_cluster_bootstrap_permitted_under_declared_assumption {
        SamplingDesignDispositionV1::ConditionalSourceWorkBootstrap
    } else {
        SamplingDesignDispositionV1::DescriptiveOnly
    };

    let crossed_inference_required = crossed;
    let mut reasons = Vec::new();
    if incomplete {
        reasons.push(
            "one or more required work, performer, session, take, or sampling-frame identities are missing"
                .into(),
        );
    }
    if crossed_performer_count > 0 {
        reasons.push(format!(
            "{crossed_performer_count} performer identities occur across multiple source works"
        ));
    }
    if crossed_session_count > 0 {
        reasons.push(format!(
            "{crossed_session_count} session identities occur across multiple source works"
        ));
    }
    if crossed_take_count > 0 {
        reasons.push(format!(
            "{crossed_take_count} take identities occur across multiple source works"
        ));
    }
    if crossed_performance_count > 0 {
        reasons.push(format!(
            "{crossed_performance_count} performance identities occur across multiple source works"
        ));
    }
    if nested && !crossed {
        reasons.push(
            "performers, sessions, or takes are nested within one or more source works".into(),
        );
    }
    if !design.independent_source_work_sampling_claimed {
        reasons.push("independent source-work sampling was not declared".into());
    }
    if design.independent_source_work_sampling_claimed && !independence_basis_present {
        reasons.push("independence was claimed without a documented design basis".into());
    }
    if work_cluster_bootstrap_permitted_under_declared_assumption {
        reasons.push(
            "source-work bootstrap is only conditionally admitted under the supplied independence declaration"
                .into(),
        );
    }

    Ok(PerformanceIdentitySamplingDesignReportV1 {
        schema: REPORT_SCHEMA_V1,
        design_shape,
        disposition,
        observation_count: observation_ids.len(),
        source_work_cluster_count: work_ids.len(),
        performer_identity_count: performer_works.len(),
        session_identity_count: session_works.len(),
        take_identity_count: take_works.len(),
        performance_identity_count: performance_works.len(),
        crossed_performer_count,
        crossed_session_count,
        crossed_take_count,
        crossed_performance_count,
        multi_performer_arm_count,
        independent_source_work_sampling_claimed: design
            .independent_source_work_sampling_claimed,
        independence_basis_present,
        work_cluster_bootstrap_permitted_under_declared_assumption,
        crossed_inference_required,
        performer_identity_authenticated: false,
        source_work_identity_authenticated: false,
        sampling_independence_authenticated: false,
        population_generalization_established: false,
        reasons,
    })
}

/// Safe composition entry point for a source-work-only paired comparison.
///
/// This binds every design record to one exact item, pair, work, and CLEAN /
/// STAY / FLIP performer/take/performance identity in both baseline and
/// candidate cases. Crossed, unidentified, or undeclared-independent designs
/// are rejected before the #14019 comparator is invoked.
pub fn audit_paired_comparison_with_sampling_design(
    comparison: &PairedMusicalIdentityComparisonCaseV1,
    design: &PerformanceIdentitySamplingDesignV1,
) -> Result<SamplingDesignQualifiedComparisonReportV1, String> {
    let sampling_design = audit_performance_identity_sampling_design(design)?;
    validate_design_against_comparison(design, comparison)?;
    if !sampling_design.work_cluster_bootstrap_permitted_under_declared_assumption {
        return Err(format!(
            "source-work-only inference refused: {:?}; {:?}",
            sampling_design.design_shape, sampling_design.disposition
        ));
    }
    let paired_comparison = audit_paired_musical_identity_comparison(comparison)?;
    Ok(SamplingDesignQualifiedComparisonReportV1 {
        schema: QUALIFIED_REPORT_SCHEMA_V1,
        sampling_design,
        paired_comparison,
    })
}

fn validate_design_against_comparison(
    design: &PerformanceIdentitySamplingDesignV1,
    comparison: &PairedMusicalIdentityComparisonCaseV1,
) -> Result<(), String> {
    let baseline = triads_by_item_id(&comparison.baseline)?;
    let candidate = triads_by_item_id(&comparison.candidate)?;
    if baseline.len() != candidate.len() || baseline.len() != design.observations.len() {
        return Err(
            "sampling design must contain exactly one observation for every paired comparison item"
                .into(),
        );
    }

    let mut seen = BTreeSet::new();
    for observation in &design.observations {
        if !seen.insert(observation.item_id.as_str()) {
            return Err(format!(
                "duplicate sampling-design item_id: {}",
                observation.item_id
            ));
        }
        let baseline_triad = baseline
            .get(observation.item_id.as_str())
            .ok_or_else(|| format!("sampling design has unknown item_id {}", observation.item_id))?;
        let candidate_triad = candidate
            .get(observation.item_id.as_str())
            .ok_or_else(|| format!("candidate is missing item_id {}", observation.item_id))?;

        if observation.pair_id != baseline_triad.pair_id
            || observation.pair_id != candidate_triad.pair_id
            || observation.source_work_lineage_sha256 != baseline_triad.source_work_lineage_sha256
            || observation.source_work_lineage_sha256
                != candidate_triad.source_work_lineage_sha256
        {
            return Err(format!(
                "sampling design item/pair/source-work identity mismatch for {}",
                observation.item_id
            ));
        }

        validate_arm_join(
            &observation.clean,
            &baseline_triad.clean,
            &candidate_triad.clean,
            &observation.item_id,
            "clean",
        )?;
        validate_arm_join(
            &observation.performance_stay,
            &baseline_triad.performance_stay,
            &candidate_triad.performance_stay,
            &observation.item_id,
            "performance_stay",
        )?;
        validate_arm_join(
            &observation.musical_flip,
            &baseline_triad.musical_flip,
            &candidate_triad.musical_flip,
            &observation.item_id,
            "musical_flip",
        )?;
    }
    Ok(())
}

fn triads_by_item_id(
    case: &MusicalPerformanceIdentityCaseV1,
) -> Result<
    BTreeMap<&str, &MusicalPerformanceIdentityTriadV1>,
    String,
> {
    let mut output = BTreeMap::new();
    for triad in &case.triads {
        if output.insert(triad.item_id.as_str(), triad).is_some() {
            return Err(format!(
                "duplicate comparison item_id: {}",
                triad.item_id
            ));
        }
    }
    Ok(output)
}

fn validate_arm_join(
    membership: &PerformanceIdentityArmMembershipV1,
    baseline: &MusicalIdentityVariantV1,
    candidate: &MusicalIdentityVariantV1,
    item_id: &str,
    arm_name: &str,
) -> Result<(), String> {
    if membership.performer_identity_sha256 != baseline.performer_identity_sha256
        || membership.performer_identity_sha256 != candidate.performer_identity_sha256
        || membership.take_identity_sha256 != baseline.take_identity_sha256
        || membership.take_identity_sha256 != candidate.take_identity_sha256
        || membership.performance_identity_sha256 != baseline.performance_identity_sha256
        || membership.performance_identity_sha256 != candidate.performance_identity_sha256
    {
        return Err(format!(
            "{arm_name} performer/take/performance identity mismatch for item {item_id}"
        ));
    }
    if !membership
        .performer_membership_sha256
        .iter()
        .any(|member| member == &membership.performer_identity_sha256)
    {
        return Err(format!(
            "{arm_name} participant set omits the reducer-bound performer for item {item_id}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_performance_identity_inference::PairedMusicalIdentityComparisonCaseV1;

    fn crossed_fixture() -> PerformanceIdentitySamplingDesignV1 {
        serde_json::from_str(include_str!(
            "../fixtures/mel-music-performance-identity-sampling-design.crossed.synthetic.json"
        ))
        .expect("crossed sampling-design fixture must parse")
    }

    fn comparison_fixture() -> PairedMusicalIdentityComparisonCaseV1 {
        serde_json::from_str(include_str!(
            "../fixtures/mel-music-performance-identity-comparison.synthetic.json"
        ))
        .expect("paired comparison fixture must parse")
    }

    fn digest(value: u64) -> String {
        format!("{value:064x}")
    }

    fn arm(
        performer: &str,
        session: &str,
        take: &str,
        performance: &str,
        members: Vec<String>,
    ) -> PerformanceIdentityArmMembershipV1 {
        PerformanceIdentityArmMembershipV1 {
            performer_identity_sha256: performer.into(),
            performer_membership_sha256: members,
            session_identity_sha256: session.into(),
            take_identity_sha256: take.into(),
            performance_identity_sha256: performance.into(),
        }
    }

    fn observation(
        item_id: &str,
        pair_id: &str,
        work: &str,
        clean_performer: &str,
        stay_performer: &str,
        flip_performer: &str,
        session_seed: u64,
    ) -> PerformanceIdentitySamplingObservationV1 {
        PerformanceIdentitySamplingObservationV1 {
            item_id: item_id.into(),
            pair_id: pair_id.into(),
            source_work_lineage_sha256: work.into(),
            clean: arm(
                clean_performer,
                &digest(session_seed),
                &digest(session_seed + 1),
                &digest(session_seed + 2),
                vec![clean_performer.into()],
            ),
            performance_stay: arm(
                stay_performer,
                &digest(session_seed + 3),
                &digest(session_seed + 4),
                &digest(session_seed + 5),
                vec![stay_performer.into()],
            ),
            musical_flip: arm(
                flip_performer,
                &digest(session_seed + 6),
                &digest(session_seed + 7),
                &digest(session_seed + 8),
                vec![flip_performer.into()],
            ),
        }
    }

    fn design(observations: Vec<PerformanceIdentitySamplingObservationV1>) -> PerformanceIdentitySamplingDesignV1 {
        PerformanceIdentitySamplingDesignV1 {
            schema: DESIGN_SCHEMA_V1.into(),
            sampling_frame_id: "synthetic-frame-v1".into(),
            target_population: "synthetic test population".into(),
            inclusion_exclusion_policy: "all fixture records".into(),
            pseudonymization_scope: "synthetic fixture only".into(),
            independent_source_work_sampling_claimed: true,
            independence_basis: "works allocated independently in the synthetic fixture".into(),
            observations,
        }
    }

    fn make_work_disjoint_comparison(
        comparison: &mut PairedMusicalIdentityComparisonCaseV1,
    ) {
        let item_factors = comparison
            .baseline
            .triads
            .iter()
            .enumerate()
            .map(|(index, triad)| {
                (
                    index,
                    triad.item_id.clone(),
                    triad.performance_stay_factor,
                )
            })
            .collect::<Vec<_>>();
        for (index, item_id, stay_factor) in item_factors {
            let clean = digest(10_000 + index as u64 * 10);
            let stay = if stay_factor
                == crate::music_performance_identity::PerformanceStayFactorV1::DifferentPerformerSameInstrument
            {
                digest(10_001 + index as u64 * 10)
            } else {
                clean.clone()
            };
            let flip = clean.clone();
            set_item_performers(&mut comparison.baseline, &item_id, &clean, &stay, &flip);
            set_item_performers(&mut comparison.candidate, &item_id, &clean, &stay, &flip);
        }
    }

    fn set_item_performers(
        case: &mut MusicalPerformanceIdentityCaseV1,
        item_id: &str,
        clean: &str,
        stay: &str,
        flip: &str,
    ) {
        let triad = case.triads.iter_mut().find(|value| value.item_id == item_id).unwrap();
        triad.clean.performer_identity_sha256 = clean.into();
        triad.performance_stay.performer_identity_sha256 = stay.into();
        triad.musical_flip.performer_identity_sha256 = flip.into();
    }

    fn design_for_comparison(
        comparison: &PairedMusicalIdentityComparisonCaseV1,
    ) -> PerformanceIdentitySamplingDesignV1 {
        let observations = comparison
            .baseline
            .triads
            .iter()
            .enumerate()
            .map(|(index, triad)| {
                PerformanceIdentitySamplingObservationV1 {
                    item_id: triad.item_id.clone(),
                    pair_id: triad.pair_id.clone(),
                    source_work_lineage_sha256: triad.source_work_lineage_sha256.clone(),
                    clean: arm(
                        &triad.clean.performer_identity_sha256,
                        &digest(20_000 + index as u64 * 10),
                        &triad.clean.take_identity_sha256,
                        &triad.clean.performance_identity_sha256,
                        vec![triad.clean.performer_identity_sha256.clone()],
                    ),
                    performance_stay: arm(
                        &triad.performance_stay.performer_identity_sha256,
                        &digest(20_001 + index as u64 * 10),
                        &triad.performance_stay.take_identity_sha256,
                        &triad.performance_stay.performance_identity_sha256,
                        vec![triad.performance_stay.performer_identity_sha256.clone()],
                    ),
                    musical_flip: arm(
                        &triad.musical_flip.performer_identity_sha256,
                        &digest(20_002 + index as u64 * 10),
                        &triad.musical_flip.take_identity_sha256,
                        &triad.musical_flip.performance_identity_sha256,
                        vec![triad.musical_flip.performer_identity_sha256.clone()],
                    ),
                }
            })
            .collect();
        design(observations)
    }

    #[test]
    fn crossed_performer_fixture_is_not_eligible_for_source_work_bootstrap() {
        let report = audit_performance_identity_sampling_design(&crossed_fixture()).unwrap();
        assert_eq!(
            report.design_shape,
            SamplingDesignShapeV1::CrossedPerformerSessionOrTake
        );
        assert_eq!(
            report.disposition,
            SamplingDesignDispositionV1::CrossedInferenceRequired
        );
        assert_eq!(report.source_work_cluster_count, 2);
        assert_eq!(report.crossed_performer_count, 1);
        assert!(report.crossed_inference_required);
        assert!(!report.work_cluster_bootstrap_permitted_under_declared_assumption);
        assert!(!report.performer_identity_authenticated);
        assert!(!report.population_generalization_established);
    }

    #[test]
    fn multiple_performers_and_sessions_nested_inside_work_are_not_crossed() {
        let work_a = digest(1);
        let work_b = digest(2);
        let performer_a = digest(11);
        let performer_b = digest(12);
        let performer_c = digest(13);
        let report = audit_performance_identity_sampling_design(&design(vec![
            observation("item-a1", "pair-a1", &work_a, &performer_a, &performer_b, &performer_a, 100),
            observation("item-a2", "pair-a2", &work_a, &performer_b, &performer_a, &performer_b, 200),
            observation("item-b1", "pair-b1", &work_b, &performer_c, &performer_c, &performer_c, 300),
        ]))
        .unwrap();
        assert_eq!(report.design_shape, SamplingDesignShapeV1::NestedWithinWork);
        assert_eq!(report.disposition, SamplingDesignDispositionV1::ConditionalSourceWorkBootstrap);
        assert_eq!(report.crossed_performer_count, 0);
        assert!(report.work_cluster_bootstrap_permitted_under_declared_assumption);
        assert_eq!(report.multi_performer_arm_count, 0);
    }

    #[test]
    fn explicitly_multi_performer_arm_is_counted_and_remains_nested() {
        let work = digest(31);
        let lead = digest(41);
        let guest = digest(42);
        let mut row = observation("ensemble-item", "ensemble-pair", &work, &lead, &lead, &lead, 400);
        row.clean.performer_membership_sha256.push(guest);
        let report = audit_performance_identity_sampling_design(&design(vec![row])).unwrap();
        assert_eq!(report.design_shape, SamplingDesignShapeV1::NestedWithinWork);
        assert_eq!(report.multi_performer_arm_count, 1);
        assert_eq!(report.crossed_performer_count, 0);
    }

    #[test]
    fn a_session_reused_across_works_is_crossed_even_when_performers_are_disjoint() {
        let performer_a = digest(51);
        let performer_b = digest(52);
        let shared_session = digest(53);
        let mut first = observation("session-a", "pair-a", &digest(54), &performer_a, &performer_a, &performer_a, 500);
        let mut second = observation("session-b", "pair-b", &digest(55), &performer_b, &performer_b, &performer_b, 600);
        first.clean.session_identity_sha256 = shared_session.clone();
        second.clean.session_identity_sha256 = shared_session;
        let report = audit_performance_identity_sampling_design(&design(vec![first, second])).unwrap();
        assert_eq!(report.design_shape, SamplingDesignShapeV1::CrossedPerformerSessionOrTake);
        assert_eq!(report.crossed_performer_count, 0);
        assert_eq!(report.crossed_session_count, 1);
        assert_eq!(report.disposition, SamplingDesignDispositionV1::CrossedInferenceRequired);
    }

    #[test]
    fn a_take_reused_across_works_is_crossed_even_when_performers_are_disjoint() {
        let performer_a = digest(151);
        let performer_b = digest(152);
        let shared_take = digest(153);
        let mut first = observation(
            "take-a",
            "pair-take-a",
            &digest(154),
            &performer_a,
            &performer_a,
            &performer_a,
            1_500,
        );
        let mut second = observation(
            "take-b",
            "pair-take-b",
            &digest(155),
            &performer_b,
            &performer_b,
            &performer_b,
            1_600,
        );
        first.performance_stay.take_identity_sha256 = shared_take.clone();
        second.performance_stay.take_identity_sha256 = shared_take;

        let report = audit_performance_identity_sampling_design(&design(vec![first, second])).unwrap();
        assert_eq!(
            report.design_shape,
            SamplingDesignShapeV1::CrossedPerformerSessionOrTake
        );
        assert_eq!(report.crossed_performer_count, 0);
        assert_eq!(report.crossed_session_count, 0);
        assert_eq!(report.crossed_take_count, 1);
        assert_eq!(
            report.disposition,
            SamplingDesignDispositionV1::CrossedInferenceRequired
        );
    }

    #[test]
    fn secondary_ensemble_member_reused_across_works_is_crossed() {
        let lead_a = digest(161);
        let lead_b = digest(162);
        let shared_member = digest(163);
        let mut first = observation(
            "ensemble-a",
            "pair-ensemble-a",
            &digest(164),
            &lead_a,
            &lead_a,
            &lead_a,
            1_700,
        );
        let mut second = observation(
            "ensemble-b",
            "pair-ensemble-b",
            &digest(165),
            &lead_b,
            &lead_b,
            &lead_b,
            1_800,
        );
        first.clean.performer_membership_sha256.push(shared_member.clone());
        second.clean.performer_membership_sha256.push(shared_member);

        let report = audit_performance_identity_sampling_design(&design(vec![first, second])).unwrap();
        assert_eq!(
            report.design_shape,
            SamplingDesignShapeV1::CrossedPerformerSessionOrTake
        );
        assert_eq!(report.crossed_performer_count, 1);
        assert_eq!(report.crossed_session_count, 0);
        assert_eq!(report.crossed_take_count, 0);
        assert_eq!(report.multi_performer_arm_count, 2);
    }


    #[test]
    fn a_performance_realization_reused_across_works_is_crossed() {
        let performer_a = digest(171);
        let performer_b = digest(172);
        let shared_performance = digest(173);
        let mut first = observation(
            "performance-a",
            "pair-performance-a",
            &digest(174),
            &performer_a,
            &performer_a,
            &performer_a,
            1_900,
        );
        let mut second = observation(
            "performance-b",
            "pair-performance-b",
            &digest(175),
            &performer_b,
            &performer_b,
            &performer_b,
            2_000,
        );
        first.performance_stay.performance_identity_sha256 = shared_performance.clone();
        second.performance_stay.performance_identity_sha256 = shared_performance;

        let report =
            audit_performance_identity_sampling_design(&design(vec![first, second])).unwrap();
        assert_eq!(
            report.design_shape,
            SamplingDesignShapeV1::CrossedPerformerSessionOrTake
        );
        assert_eq!(report.crossed_performer_count, 0);
        assert_eq!(report.crossed_session_count, 0);
        assert_eq!(report.crossed_take_count, 0);
        assert_eq!(report.crossed_performance_count, 1);
        assert_eq!(
            report.disposition,
            SamplingDesignDispositionV1::CrossedInferenceRequired
        );
    }

    #[test]
    fn unidentified_performer_or_session_forces_descriptive_only() {
        let mut row = observation("missing-item", "missing-pair", &digest(61), &digest(62), &digest(62), &digest(62), 700);
        row.performance_stay.performer_membership_sha256.clear();
        row.musical_flip.session_identity_sha256.clear();
        let report = audit_performance_identity_sampling_design(&design(vec![row])).unwrap();
        assert_eq!(report.design_shape, SamplingDesignShapeV1::InsufficientlyIdentified);
        assert_eq!(report.disposition, SamplingDesignDispositionV1::DescriptiveOnly);
        assert!(!report.work_cluster_bootstrap_permitted_under_declared_assumption);
    }

    #[test]
    fn a_missing_independence_basis_blocks_conditional_bootstrap_admission() {
        let mut input = design(vec![observation(
            "basis-item",
            "basis-pair",
            &digest(71),
            &digest(72),
            &digest(72),
            &digest(72),
            800,
        )]);
        input.independence_basis.clear();
        let report = audit_performance_identity_sampling_design(&input).unwrap();
        assert_eq!(report.design_shape, SamplingDesignShapeV1::WorkDisjointClusters);
        assert_eq!(report.disposition, SamplingDesignDispositionV1::DescriptiveOnly);
        assert!(!report.work_cluster_bootstrap_permitted_under_declared_assumption);
    }

    #[test]
    fn item_order_does_not_change_sampling_design_classification() {
        let mut input = crossed_fixture();
        let original = audit_performance_identity_sampling_design(&input).unwrap();
        input.observations.reverse();
        let reversed = audit_performance_identity_sampling_design(&input).unwrap();
        assert_eq!(original, reversed);
    }

    #[test]
    fn duplicate_performer_membership_within_one_arm_is_rejected() {
        let performer = digest(91);
        let mut row = observation("duplicate-member", "duplicate-pair", &digest(92), &performer, &performer, &performer, 900);
        row.clean.performer_membership_sha256.push(performer);
        assert!(audit_performance_identity_sampling_design(&design(vec![row])).is_err());
    }

    #[test]
    fn duplicate_item_membership_is_rejected() {
        let performer = digest(101);
        let row = observation("duplicate-item", "pair-a", &digest(102), &performer, &performer, &performer, 1_000);
        let mut second = row.clone();
        second.pair_id = "pair-b".into();
        let report = design(vec![row, second]);
        assert!(audit_performance_identity_sampling_design(&report).is_err());
    }

    #[test]
    fn crossed_design_is_refused_before_the_paired_comparator_runs() {
        let comparison = comparison_fixture();
        let crossed = design_for_comparison(&comparison);
        let error =
            audit_paired_comparison_with_sampling_design(&comparison, &crossed).unwrap_err();
        assert!(error.contains("CrossedPerformerSessionOrTake"));
        assert!(error.contains("CrossedInferenceRequired"));
    }

    #[test]
    fn work_disjoint_design_is_joined_to_exact_comparison_items_before_admission() {
        let mut comparison = comparison_fixture();
        make_work_disjoint_comparison(&mut comparison);
        let sampling_design = design_for_comparison(&comparison);
        let report = audit_paired_comparison_with_sampling_design(&comparison, &sampling_design)
            .expect("complete work-disjoint design should be conditionally admitted");
        assert_eq!(
            report.sampling_design.design_shape,
            SamplingDesignShapeV1::NestedWithinWork
        );
        assert!(
            report
                .sampling_design
                .work_cluster_bootstrap_permitted_under_declared_assumption
        );
        assert_eq!(report.paired_comparison.source_work_clusters, 30);
    }

    #[test]
    fn design_cannot_be_attached_to_another_source_work_or_performer() {
        let mut comparison = comparison_fixture();
        make_work_disjoint_comparison(&mut comparison);
        let mut sampling_design = design_for_comparison(&comparison);
        sampling_design.observations[0].source_work_lineage_sha256 = digest(999);
        assert!(
            audit_paired_comparison_with_sampling_design(&comparison, &sampling_design).is_err()
        );

        let mut sampling_design = design_for_comparison(&comparison);
        sampling_design.observations[0].clean.performer_identity_sha256 = digest(1_001);
        assert!(
            audit_paired_comparison_with_sampling_design(&comparison, &sampling_design).is_err()
        );
    }
}
