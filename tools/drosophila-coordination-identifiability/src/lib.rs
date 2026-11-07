//! Synthetic mechanism-equivalence auditor for six-leg coordination.
//!
//! The crate demonstrates why baseline mean frequency / aggregate phase
//! locking cannot identify the mechanism that produces a coordinated gait.
//! It compares exact integer signatures across summary, full-baseline, and
//! feasible-intervention observation views. It is not a fly simulator,
//! intervention selector, biological estimator, or mechanism proof.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEG_COUNT: usize = 6;
pub const LEG_PAIR_COUNT: usize = 15;
pub const STANCE_COUNT_BINS: usize = LEG_COUNT + 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaselineSummary {
    pub mean_frequency_millihz: u32,
    pub aggregate_phase_locking_milli: u16,
    pub tripod_occupancy_milli: u16,
}

/// Full baseline shape omitted by the common scalar summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaselineTrajectory {
    pub pairwise_phase_offsets_microturns: [i32; LEG_PAIR_COUNT],
    /// Probability mass over 0..=6 legs in stance, in per-thousand units.
    pub stance_count_histogram_milli: [u16; STANCE_COUNT_BINS],
}

/// One time point of a phase-response trajectory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseFrame {
    pub phase_reset_by_leg_microturns: [i32; LEG_COUNT],
    pub cycle_slips_by_leg: [i32; LEG_COUNT],
    pub frequency_by_leg_millihz: [u32; LEG_COUNT],
    pub tripod_occupancy_milli: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionPrediction {
    pub intervention_id: String,
    /// Ordered response frames on one common, predeclared time grid.
    pub frames: Vec<ResponseFrame>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismWorld {
    pub world_id: String,
    pub mechanism_family: String,
    pub source_revision_sha256: String,
    pub world_spec_sha256: String,
    pub baseline_summary: BaselineSummary,
    pub baseline_trajectory: BaselineTrajectory,
    pub interventions: Vec<InterventionPrediction>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionSpec {
    pub intervention_id: String,
    pub feasible: bool,
    pub feasibility_receipt_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationView {
    SummaryOnly,
    FullBaseline,
    FullFeasiblePortfolio,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismAudit {
    pub evidence_tier: &'static str,
    pub summary_only_classes: Vec<Vec<String>>,
    pub full_baseline_classes: Vec<Vec<String>>,
    pub feasible_portfolio_classes: Vec<Vec<String>>,
    pub total_world_pairs: usize,
    pub summary_separated_pairs: usize,
    pub baseline_separated_pairs: usize,
    pub feasible_portfolio_separated_pairs: usize,
    pub feasible_portfolio_unresolved_pairs: usize,
    pub status: &'static str,
    pub mechanism_certificate_issued: bool,
    pub unavailable_candidate_interventions: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentifiabilityError(pub String);

impl fmt::Display for IdentifiabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for IdentifiabilityError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn append_summary(summary: &BaselineSummary, out: &mut Vec<i64>) {
    out.push(i64::from(summary.mean_frequency_millihz));
    out.push(i64::from(summary.aggregate_phase_locking_milli));
    out.push(i64::from(summary.tripod_occupancy_milli));
}

fn append_baseline(trajectory: &BaselineTrajectory, out: &mut Vec<i64>) {
    out.extend(
        trajectory
            .pairwise_phase_offsets_microturns
            .iter()
            .map(|value| i64::from(*value)),
    );
    out.extend(
        trajectory
            .stance_count_histogram_milli
            .iter()
            .map(|value| i64::from(*value)),
    );
}

fn append_intervention(prediction: &InterventionPrediction, out: &mut Vec<i64>) {
    for frame in &prediction.frames {
        out.extend(
            frame
                .phase_reset_by_leg_microturns
                .iter()
                .map(|value| i64::from(*value)),
        );
        out.extend(
            frame
                .cycle_slips_by_leg
                .iter()
                .map(|value| i64::from(*value)),
        );
        out.extend(
            frame
                .frequency_by_leg_millihz
                .iter()
                .map(|value| i64::from(*value)),
        );
        out.push(i64::from(frame.tripod_occupancy_milli));
    }
}

fn signature(world: &MechanismWorld, view: ObservationView, feasible_ids: &[String]) -> Vec<i64> {
    let mut out = Vec::new();
    append_summary(&world.baseline_summary, &mut out);
    match view {
        ObservationView::SummaryOnly => {}
        ObservationView::FullBaseline => append_baseline(&world.baseline_trajectory, &mut out),
        ObservationView::FullFeasiblePortfolio => {
            append_baseline(&world.baseline_trajectory, &mut out);
            for id in feasible_ids {
                // The validator guarantees exactly one response per registered
                // intervention ID in every world.
                let prediction = world
                    .interventions
                    .iter()
                    .find(|candidate| &candidate.intervention_id == id)
                    .expect("validated intervention coverage");
                append_intervention(prediction, &mut out);
            }
        }
    }
    out
}

fn partition(
    worlds: &[MechanismWorld],
    view: ObservationView,
    feasible_ids: &[String],
) -> Vec<Vec<String>> {
    let mut groups: BTreeMap<Vec<i64>, Vec<String>> = BTreeMap::new();
    for world in worlds {
        groups
            .entry(signature(world, view, feasible_ids))
            .or_default()
            .push(world.world_id.clone());
    }
    let mut result: Vec<Vec<String>> = groups.into_values().collect();
    for group in &mut result {
        group.sort();
    }
    result.sort_by(|left, right| left[0].cmp(&right[0]));
    result
}

fn separated_pair_count(classes: &[Vec<String>], world_count: usize) -> usize {
    let total = world_count * world_count.saturating_sub(1) / 2;
    let unresolved: usize = classes
        .iter()
        .map(|group| group.len() * group.len().saturating_sub(1) / 2)
        .sum();
    total - unresolved
}

fn validate_world(world: &MechanismWorld) -> Result<(), IdentifiabilityError> {
    if world.world_id.is_empty() || world.mechanism_family.is_empty() {
        return Err(IdentifiabilityError(
            "world_id and mechanism_family must be non-empty".into(),
        ));
    }
    if !valid_sha256(&world.source_revision_sha256) || !valid_sha256(&world.world_spec_sha256) {
        return Err(IdentifiabilityError(format!(
            "{} has a malformed source/world digest",
            world.world_id
        )));
    }
    let summary = &world.baseline_summary;
    if summary.mean_frequency_millihz == 0
        || summary.aggregate_phase_locking_milli > 1_000
        || summary.tripod_occupancy_milli > 1_000
    {
        return Err(IdentifiabilityError(format!(
            "{} has an invalid baseline summary",
            world.world_id
        )));
    }
    for offset in world.baseline_trajectory.pairwise_phase_offsets_microturns {
        if !(-500_000..500_000).contains(&offset) {
            return Err(IdentifiabilityError(format!(
                "{} has a non-canonical phase offset",
                world.world_id
            )));
        }
    }
    let mass: u32 = world
        .baseline_trajectory
        .stance_count_histogram_milli
        .iter()
        .map(|value| u32::from(*value))
        .sum();
    if mass != 1_000 {
        return Err(IdentifiabilityError(format!(
            "{} stance-count histogram must sum exactly to 1000",
            world.world_id
        )));
    }

    let mut intervention_ids = BTreeSet::new();
    for prediction in &world.interventions {
        if prediction.intervention_id.is_empty()
            || !intervention_ids.insert(prediction.intervention_id.as_str())
        {
            return Err(IdentifiabilityError(format!(
                "{} has an empty or duplicate intervention ID",
                world.world_id
            )));
        }
        if prediction.frames.len() < 2 {
            return Err(IdentifiabilityError(format!(
                "{} / {} requires at least two response frames",
                world.world_id, prediction.intervention_id
            )));
        }
        for frame in &prediction.frames {
            if frame.tripod_occupancy_milli > 1_000
                || frame.frequency_by_leg_millihz.contains(&0)
                || frame
                    .phase_reset_by_leg_microturns
                    .iter()
                    .any(|value| !(-500_000..500_000).contains(value))
            {
                return Err(IdentifiabilityError(format!(
                    "{} / {} has an invalid response frame",
                    world.world_id, prediction.intervention_id
                )));
            }
        }
    }
    Ok(())
}

/// Audit exact observational-equivalence classes for the registered candidate
/// worlds. Infeasible interventions are recorded but excluded from the
/// feasible portfolio. This crate does not select an intervention.
pub fn audit_mechanism_identifiability(
    worlds: &[MechanismWorld],
    interventions: &[InterventionSpec],
) -> Result<MechanismAudit, IdentifiabilityError> {
    if worlds.len() < 2 {
        return Err(IdentifiabilityError(
            "at least two candidate mechanism worlds are required".into(),
        ));
    }
    if interventions.is_empty() {
        return Err(IdentifiabilityError(
            "candidate intervention registry is empty".into(),
        ));
    }

    let mut world_ids = BTreeSet::new();
    for world in worlds {
        validate_world(world)?;
        if !world_ids.insert(world.world_id.as_str()) {
            return Err(IdentifiabilityError("duplicate world_id".into()));
        }
    }

    let mut intervention_ids = BTreeSet::new();
    for intervention in interventions {
        if intervention.intervention_id.is_empty()
            || !intervention_ids.insert(intervention.intervention_id.as_str())
        {
            return Err(IdentifiabilityError(
                "empty or duplicate intervention registry ID".into(),
            ));
        }
        if !valid_sha256(&intervention.feasibility_receipt_sha256) {
            return Err(IdentifiabilityError(format!(
                "{} has a malformed feasibility receipt digest",
                intervention.intervention_id
            )));
        }
    }

    let registered_ids: BTreeSet<&str> = interventions
        .iter()
        .map(|item| item.intervention_id.as_str())
        .collect();
    for world in worlds {
        let world_ids: BTreeSet<&str> = world
            .interventions
            .iter()
            .map(|item| item.intervention_id.as_str())
            .collect();
        if world_ids != registered_ids {
            return Err(IdentifiabilityError(format!(
                "{} predictions do not cover the exact intervention registry",
                world.world_id
            )));
        }
    }

    // Every model pair must use the same time grid for a given intervention.
    for intervention in interventions {
        let frame_count = worlds[0]
            .interventions
            .iter()
            .find(|item| item.intervention_id == intervention.intervention_id)
            .expect("coverage checked")
            .frames
            .len();
        for world in worlds.iter().skip(1) {
            let candidate = world
                .interventions
                .iter()
                .find(|item| item.intervention_id == intervention.intervention_id)
                .expect("coverage checked");
            if candidate.frames.len() != frame_count {
                return Err(IdentifiabilityError(format!(
                    "{} response time grid differs for {}",
                    world.world_id, intervention.intervention_id
                )));
            }
        }
    }

    let mut feasible_ids: Vec<String> = interventions
        .iter()
        .filter(|item| item.feasible)
        .map(|item| item.intervention_id.clone())
        .collect();
    feasible_ids.sort();
    if feasible_ids.is_empty() {
        return Err(IdentifiabilityError(
            "NO_FEASIBLE_DISCRIMINATING_INTERVENTION: no candidate is feasible".into(),
        ));
    }

    let summary_only_classes = partition(worlds, ObservationView::SummaryOnly, &[]);
    let full_baseline_classes = partition(worlds, ObservationView::FullBaseline, &[]);
    let feasible_portfolio_classes = partition(
        worlds,
        ObservationView::FullFeasiblePortfolio,
        &feasible_ids,
    );

    let total_world_pairs = worlds.len() * (worlds.len() - 1) / 2;
    let summary_separated_pairs = separated_pair_count(&summary_only_classes, worlds.len());
    let baseline_separated_pairs = separated_pair_count(&full_baseline_classes, worlds.len());
    let feasible_portfolio_separated_pairs =
        separated_pair_count(&feasible_portfolio_classes, worlds.len());
    let feasible_portfolio_unresolved_pairs =
        total_world_pairs - feasible_portfolio_separated_pairs;
    let unavailable_candidate_interventions = interventions
        .iter()
        .filter(|item| !item.feasible)
        .map(|item| item.intervention_id.clone())
        .collect();

    let identifiable = feasible_portfolio_unresolved_pairs == 0;
    Ok(MechanismAudit {
        evidence_tier: "SYNTHETIC_KNOWN_TRUTH_ONLY",
        summary_only_classes,
        full_baseline_classes,
        feasible_portfolio_classes,
        total_world_pairs,
        summary_separated_pairs,
        baseline_separated_pairs,
        feasible_portfolio_separated_pairs,
        feasible_portfolio_unresolved_pairs,
        status: if identifiable {
            "DISTINGUISHABLE_WITHIN_DECLARED_FEASIBLE_PORTFOLIO"
        } else {
            "NOT_IDENTIFIABLE_UNDER_DECLARED_FEASIBLE_PORTFOLIO"
        },
        mechanism_certificate_issued: false,
        unavailable_candidate_interventions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn summary() -> BaselineSummary {
        BaselineSummary {
            mean_frequency_millihz: 11_000,
            aggregate_phase_locking_milli: 920,
            tripod_occupancy_milli: 800,
        }
    }

    fn baseline(unique_mech: bool) -> BaselineTrajectory {
        let mut offsets = [50_000; LEG_PAIR_COUNT];
        if unique_mech {
            offsets[0] = -100_000;
            offsets[4] = 150_000;
            offsets[9] = -175_000;
        }
        BaselineTrajectory {
            pairwise_phase_offsets_microturns: offsets,
            stance_count_histogram_milli: [0, 0, 50, 700, 200, 50, 0],
        }
    }

    fn frame(response: [i32; LEG_COUNT], tripod: u16) -> ResponseFrame {
        ResponseFrame {
            phase_reset_by_leg_microturns: response,
            cycle_slips_by_leg: [0; LEG_COUNT],
            frequency_by_leg_millihz: [11_000; LEG_COUNT],
            tripod_occupancy_milli: tripod,
        }
    }

    fn trajectory(response: [i32; LEG_COUNT]) -> Vec<ResponseFrame> {
        let recovered = response.map(|value| value / 2);
        vec![
            frame([0; LEG_COUNT], 800),
            frame(response, 800),
            frame(recovered, 800),
        ]
    }

    fn local() -> [i32; LEG_COUNT] {
        [100_000, 0, 0, 0, 0, 0]
    }

    fn central() -> [i32; LEG_COUNT] {
        [100_000, 40_000, 25_000, 30_000, 20_000, 10_000]
    }

    fn sensory() -> [i32; LEG_COUNT] {
        [100_000, 0, 30_000, 0, 20_000, 0]
    }

    fn mechanical() -> [i32; LEG_COUNT] {
        [100_000, 0, 25_000, 0, 10_000, 0]
    }

    fn world(
        id: &str,
        family: &str,
        unique_mech_baseline: bool,
        air: [i32; LEG_COUNT],
        contact: [i32; LEG_COUNT],
        silenced: [i32; LEG_COUNT],
        bridge_block: [i32; LEG_COUNT],
    ) -> MechanismWorld {
        MechanismWorld {
            world_id: id.into(),
            mechanism_family: family.into(),
            source_revision_sha256: digest('a'),
            world_spec_sha256: digest('b'),
            baseline_summary: summary(),
            baseline_trajectory: baseline(unique_mech_baseline),
            interventions: vec![
                InterventionPrediction {
                    intervention_id: "left_fore_kick_air".into(),
                    frames: trajectory(air),
                },
                InterventionPrediction {
                    intervention_id: "left_fore_kick_contact".into(),
                    frames: trajectory(contact),
                },
                InterventionPrediction {
                    intervention_id: "left_fore_kick_contact_afferents_silenced".into(),
                    frames: trajectory(silenced),
                },
                InterventionPrediction {
                    intervention_id: "candidate_central_bridge_block".into(),
                    frames: trajectory(bridge_block),
                },
            ],
        }
    }

    fn fixture() -> (Vec<MechanismWorld>, Vec<InterventionSpec>) {
        let worlds = vec![
            world(
                "W-local",
                "LOCAL_CPG",
                false,
                local(),
                local(),
                local(),
                local(),
            ),
            world(
                "W-central",
                "CENTRAL_NEURAL_COUPLING",
                false,
                central(),
                central(),
                central(),
                local(),
            ),
            world(
                "W-sensory",
                "PROPRIOCEPTIVE_LOAD_FEEDBACK",
                false,
                local(),
                sensory(),
                local(),
                local(),
            ),
            world(
                "W-mechanical",
                "BODY_ENVIRONMENT_COUPLING",
                true,
                local(),
                sensory(),
                mechanical(),
                mechanical(),
            ),
            // Adversarial omitted-mechanism challenger: it mimics every
            // feasible response of central coupling, so those mechanisms
            // cannot be distinguished without a selective bridge intervention.
            world(
                "W-shared-drive",
                "OMITTED_SHARED_DRIVE",
                false,
                central(),
                central(),
                central(),
                central(),
            ),
        ];
        let interventions = vec![
            InterventionSpec {
                intervention_id: "left_fore_kick_air".into(),
                feasible: true,
                feasibility_receipt_sha256: digest('c'),
            },
            InterventionSpec {
                intervention_id: "left_fore_kick_contact".into(),
                feasible: true,
                feasibility_receipt_sha256: digest('d'),
            },
            InterventionSpec {
                intervention_id: "left_fore_kick_contact_afferents_silenced".into(),
                feasible: true,
                feasibility_receipt_sha256: digest('e'),
            },
            InterventionSpec {
                intervention_id: "candidate_central_bridge_block".into(),
                feasible: false,
                feasibility_receipt_sha256: digest('f'),
            },
        ];
        (worlds, interventions)
    }

    #[test]
    fn aggregate_baseline_metrics_hide_all_mechanism_worlds() {
        let (worlds, interventions) = fixture();
        let report = audit_mechanism_identifiability(&worlds, &interventions).unwrap();
        assert_eq!(report.summary_only_classes.len(), 1);
        assert_eq!(report.summary_only_classes[0].len(), 5);
        assert_eq!(report.summary_separated_pairs, 0);
        assert_eq!(report.total_world_pairs, 10);
    }

    #[test]
    fn full_baseline_trajectory_exposes_a_summary_metric_collision() {
        let (worlds, interventions) = fixture();
        let report = audit_mechanism_identifiability(&worlds, &interventions).unwrap();
        assert_eq!(report.full_baseline_classes.len(), 2);
        assert_eq!(report.baseline_separated_pairs, 4);
        assert_eq!(report.summary_separated_pairs, 0);
    }

    #[test]
    fn feasible_interventions_resolve_three_of_four_mechanism_groups() {
        let (worlds, interventions) = fixture();
        let report = audit_mechanism_identifiability(&worlds, &interventions).unwrap();
        assert_eq!(report.feasible_portfolio_classes.len(), 4);
        assert_eq!(report.feasible_portfolio_separated_pairs, 9);
        assert_eq!(report.feasible_portfolio_unresolved_pairs, 1);
        assert_eq!(
            report.feasible_portfolio_classes,
            vec![
                vec!["W-central".to_string(), "W-shared-drive".to_string()],
                vec!["W-local".to_string()],
                vec!["W-mechanical".to_string()],
                vec!["W-sensory".to_string()]
            ]
        );
        assert_eq!(
            report.status,
            "NOT_IDENTIFIABLE_UNDER_DECLARED_FEASIBLE_PORTFOLIO"
        );
        assert!(!report.mechanism_certificate_issued);
        assert_eq!(
            report.unavailable_candidate_interventions,
            vec!["candidate_central_bridge_block".to_string()]
        );
    }

    #[test]
    fn unavailable_bridge_block_would_break_the_adversarial_equivalence() {
        let (worlds, mut interventions) = fixture();
        interventions
            .iter_mut()
            .find(|item| item.intervention_id == "candidate_central_bridge_block")
            .unwrap()
            .feasible = true;
        let report = audit_mechanism_identifiability(&worlds, &interventions).unwrap();
        assert_eq!(report.feasible_portfolio_unresolved_pairs, 0);
        assert_eq!(
            report.status,
            "DISTINGUISHABLE_WITHIN_DECLARED_FEASIBLE_PORTFOLIO"
        );
        assert!(!report.mechanism_certificate_issued);
    }

    #[test]
    fn incomplete_intervention_prediction_matrix_is_rejected() {
        let (mut worlds, interventions) = fixture();
        worlds[2].interventions.pop();
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());
    }

    #[test]
    fn inconsistent_response_horizon_is_rejected() {
        let (mut worlds, interventions) = fixture();
        worlds[2].interventions[1].frames.pop();
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());
    }

    #[test]
    fn duplicate_world_and_intervention_ids_are_rejected() {
        let (mut worlds, interventions) = fixture();
        worlds[1].world_id = worlds[0].world_id.clone();
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());

        let (worlds, mut interventions) = fixture();
        interventions[1].intervention_id = interventions[0].intervention_id.clone();
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());
    }

    #[test]
    fn malformed_receipts_and_summary_values_are_rejected() {
        let (mut worlds, interventions) = fixture();
        worlds[0].world_spec_sha256 = "not-a-digest".into();
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());

        let (mut worlds, interventions) = fixture();
        worlds[0].baseline_summary.aggregate_phase_locking_milli = 1_001;
        assert!(audit_mechanism_identifiability(&worlds, &interventions).is_err());
    }

    #[test]
    fn infeasible_intervention_cannot_be_counted_as_discriminating_evidence() {
        let (worlds, mut interventions) = fixture();
        interventions[3].feasible = false;
        let report = audit_mechanism_identifiability(&worlds, &interventions).unwrap();
        assert_eq!(report.feasible_portfolio_unresolved_pairs, 1);
        assert_eq!(report.unavailable_candidate_interventions.len(), 1);
    }
}
