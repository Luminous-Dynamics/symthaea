//! Capture/scenario-level aggregation for leakage-resistant cybersecurity evaluation.
//!
//! The orderless operation treats one source capture or labeled scenario as one
//! statistical example. The separate ordered operation binds event vectors to
//! coarse relative record-position roles. Neither operation establishes that a
//! source label describes every event in a capture; neither is a production
//! incident detector or a causal attack-chain model.

use crate::encoder::{HDC_DIM, Hdv, bind, bundle, random_hv};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const POSITION_BUCKETS: usize = 8;
const POSITION_ROLE_SEED: u64 = 0x5343_454E_4152_0000;

/// Version of the experimental relative-record-position encoder.
pub const ORDERED_SCENARIO_ENCODER_VERSION: &str = "relative-record-position-v1";

/// The semantics of a scenario vector. Keeping this explicit prevents consumers
/// from accidentally comparing orderless and relative-position representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioEncodingMode {
    /// Permutation-invariant bundle; event order is discarded.
    OrderlessBundle,
    /// Event vectors are bound to coarse relative parser-record-position roles.
    RelativeRecordPosition,
}

/// One independently labeled capture/scenario represented by a bundle of its events.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioEmbedding {
    /// Stable source-file or scenario-directory identity.
    pub scenario_id: String,
    /// Evaluator-only ground-truth label; never included in the vector.
    pub label: String,
    /// Number of event vectors bundled into the scenario representation.
    pub event_count: usize,
    /// Mode is metadata, not part of the vector.
    pub encoding_mode: ScenarioEncodingMode,
    /// Scenario representation. Its interpretation is defined by encoding_mode.
    pub vector: Hdv,
}

/// Validate that source and held-out scenario sets are non-empty and carry the
/// same representation mode. This keeps feature contracts explicit at the point
/// where vectors from separate corpora are compared.
pub fn validate_matching_scenario_modes(
    source: &[ScenarioEmbedding],
    held_out: &[ScenarioEmbedding],
) -> Result<ScenarioEncodingMode, String> {
    let Some(first) = source.first() else {
        return Err("source scenario set is empty".to_string());
    };
    if held_out.is_empty() {
        return Err("held-out scenario set is empty".to_string());
    }
    let expected_mode = first.encoding_mode;
    if source
        .iter()
        .chain(held_out.iter())
        .any(|scenario| scenario.encoding_mode != expected_mode)
    {
        return Err("scenario encoder modes differ between source and held-out data".to_string());
    }
    Ok(expected_mode)
}

fn position_roles() -> &'static [Hdv; POSITION_BUCKETS] {
    static ROLES: OnceLock<[Hdv; POSITION_BUCKETS]> = OnceLock::new();
    ROLES.get_or_init(|| {
        std::array::from_fn(|index| random_hv(POSITION_ROLE_SEED.wrapping_add(index as u64)))
    })
}

fn validated_groups(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    if hvs.len() != labels.len() || labels.len() != groups.len() {
        return Err(format!(
            "length mismatch: vectors={}, labels={}, groups={}",
            hvs.len(),
            labels.len(),
            groups.len()
        ));
    }
    if hvs.is_empty() {
        return Err("cannot aggregate an empty corpus".to_string());
    }
    if hvs.iter().any(|hv| hv.len() != HDC_DIM) {
        return Err(format!(
            "every event vector must have exactly {HDC_DIM} dimensions"
        ));
    }
    if labels.iter().any(|label| label.is_empty()) || groups.iter().any(|group| group.is_empty()) {
        return Err("labels and scenario IDs must be non-empty".to_string());
    }

    let mut by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut group_labels: BTreeMap<&str, &str> = BTreeMap::new();
    for (index, (label, group)) in labels.iter().zip(groups.iter()).enumerate() {
        if let Some(existing) = group_labels.get(group.as_str()) {
            if *existing != label.as_str() {
                return Err(format!(
                    "scenario {group:?} contains conflicting labels: {existing:?} and {label:?}"
                ));
            }
        } else {
            group_labels.insert(group.as_str(), label.as_str());
        }
        by_group.entry(group.clone()).or_default().push(index);
    }
    Ok(by_group)
}

fn aggregate_with_mode(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
    ordered: bool,
) -> Result<Vec<ScenarioEmbedding>, String> {
    let by_group = validated_groups(hvs, labels, groups)?;
    let mut scenarios = Vec::with_capacity(by_group.len());

    for (scenario_id, indices) in by_group {
        let (vector, encoding_mode) = if ordered {
            let roles = position_roles();
            let count = indices.len();
            let position_bound: Vec<Hdv> = indices
                .iter()
                .enumerate()
                .map(|(rank, &index)| {
                    let bucket = ((rank * POSITION_BUCKETS) / count).min(POSITION_BUCKETS - 1);
                    bind(&hvs[index], &roles[bucket])
                })
                .collect();
            (
                bundle(&position_bound),
                ScenarioEncodingMode::RelativeRecordPosition,
            )
        } else {
            let event_vectors: Vec<Hdv> =
                indices.iter().map(|&index| hvs[index].clone()).collect();
            (bundle(&event_vectors), ScenarioEncodingMode::OrderlessBundle)
        };

        scenarios.push(ScenarioEmbedding {
            scenario_id,
            label: labels[indices[0]].clone(),
            event_count: indices.len(),
            encoding_mode,
            vector,
        });
    }
    Ok(scenarios)
}

/// Bundle event vectors so that each source capture/scenario contributes exactly
/// one orderless sample to training or evaluation.
///
/// Group membership is deterministic and output is sorted by scenario ID. A group
/// must have one non-empty label; conflicting labels fail closed rather than
/// arbitrarily picking one. The bundle is invariant to event row order, and neither
/// the label nor the group ID contributes to the vector.
///
/// This function does not establish that the label accurately describes every
/// event in a capture. Mixed-technique captures need finer scenario annotations.
pub fn aggregate_scenarios(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
) -> Result<Vec<ScenarioEmbedding>, String> {
    aggregate_with_mode(hvs, labels, groups, false)
}

/// Bundle events after binding each vector to one of eight coarse relative record
/// positions. The caller must provide events in deterministic record order within
/// each group; rows within a group are consumed in input order.
///
/// This experimental representation retains coarse order while discarding absolute
/// timestamps and time gaps. It is not a temporal graph, does not identify causal
/// links, and does not repair missing or out-of-order source records. Compare it to
/// aggregate_scenarios on exactly the same grouped splits before drawing conclusions.
pub fn aggregate_ordered_scenarios(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
) -> Result<Vec<ScenarioEmbedding>, String> {
    aggregate_with_mode(hvs, labels, groups, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::random_hv;

    #[test]
    fn aggregation_returns_one_sorted_scenario_per_group_with_counts() {
        let hvs = vec![random_hv(1), random_hv(2), random_hv(3)];
        let labels = vec!["execution".into(), "execution".into(), "persistence".into()];
        let groups = vec!["evtx:b".into(), "evtx:b".into(), "evtx:a".into()];

        let scenarios = aggregate_scenarios(&hvs, &labels, &groups).unwrap();
        assert_eq!(scenarios.len(), 2);
        assert_eq!(scenarios[0].scenario_id, "evtx:a");
        assert_eq!(scenarios[0].label, "persistence");
        assert_eq!(scenarios[0].event_count, 1);
        assert_eq!(scenarios[0].encoding_mode, ScenarioEncodingMode::OrderlessBundle);
        assert_eq!(scenarios[0].vector, hvs[2]);
        assert_eq!(scenarios[1].scenario_id, "evtx:b");
        assert_eq!(scenarios[1].label, "execution");
        assert_eq!(scenarios[1].event_count, 2);
        assert_eq!(scenarios[1].vector, bundle(&hvs[..2]));
    }

    #[test]
    fn aggregation_is_invariant_to_input_order() {
        let hvs = vec![random_hv(11), random_hv(22), random_hv(33), random_hv(44)];
        let labels = vec![
            "execution".into(),
            "execution".into(),
            "persistence".into(),
            "persistence".into(),
        ];
        let groups = vec![
            "evtx:a".into(),
            "evtx:a".into(),
            "evtx:b".into(),
            "evtx:b".into(),
        ];
        let forward = aggregate_scenarios(&hvs, &labels, &groups).unwrap();

        let reverse_hvs: Vec<Hdv> = hvs.iter().rev().cloned().collect();
        let reverse_labels: Vec<String> = labels.iter().rev().cloned().collect();
        let reverse_groups: Vec<String> = groups.iter().rev().cloned().collect();
        let reversed = aggregate_scenarios(&reverse_hvs, &reverse_labels, &reverse_groups).unwrap();
        assert_eq!(forward, reversed);
    }

    #[test]
    fn ordered_aggregation_is_deterministic_and_position_sensitive() {
        let hvs = vec![random_hv(11), random_hv(22), random_hv(33)];
        let labels = vec!["execution".into(), "execution".into(), "execution".into()];
        let groups = vec!["same-capture".into(), "same-capture".into(), "same-capture".into()];

        let first = aggregate_ordered_scenarios(&hvs, &labels, &groups).unwrap();
        let repeated = aggregate_ordered_scenarios(&hvs, &labels, &groups).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(
            first[0].encoding_mode,
            ScenarioEncodingMode::RelativeRecordPosition
        );

        let reversed_hvs: Vec<Hdv> = hvs.iter().rev().cloned().collect();
        let reversed_labels: Vec<String> = labels.iter().rev().cloned().collect();
        let reversed_groups: Vec<String> = groups.iter().rev().cloned().collect();
        let reversed =
            aggregate_ordered_scenarios(&reversed_hvs, &reversed_labels, &reversed_groups)
                .unwrap();
        assert_ne!(first[0].vector, reversed[0].vector);
    }

    #[test]
    fn aggregation_does_not_encode_ground_truth_label_or_group_id() {
        let vector = random_hv(0xCAFE);
        let a = aggregate_scenarios(
            std::slice::from_ref(&vector),
            &["execution".into()],
            &["source-a".into()],
        )
        .unwrap();
        let b = aggregate_scenarios(
            std::slice::from_ref(&vector),
            &["persistence".into()],
            &["source-b".into()],
        )
        .unwrap();

        assert_ne!(a[0].label, b[0].label);
        assert_ne!(a[0].scenario_id, b[0].scenario_id);
        assert_eq!(a[0].vector, b[0].vector);

        let ordered_a = aggregate_ordered_scenarios(
            std::slice::from_ref(&vector),
            &["execution".into()],
            &["source-a".into()],
        )
        .unwrap();
        let ordered_b = aggregate_ordered_scenarios(
            std::slice::from_ref(&vector),
            &["persistence".into()],
            &["source-b".into()],
        )
        .unwrap();
        assert_eq!(ordered_a[0].vector, ordered_b[0].vector);
    }

    #[test]
    fn mode_validation_accepts_matching_representations() {
        let vector = random_hv(123);
        let labels = vec!["execution".into()];
        let groups = vec!["source-a".into()];
        let source = aggregate_scenarios(&[vector.clone()], &labels, &groups).unwrap();
        let held_out = aggregate_scenarios(&[vector.clone()], &labels, &["otrf-a".into()]).unwrap();
        assert_eq!(
            validate_matching_scenario_modes(&source, &held_out).unwrap(),
            ScenarioEncodingMode::OrderlessBundle
        );

        let ordered_source =
            aggregate_ordered_scenarios(&[vector.clone()], &labels, &groups).unwrap();
        let ordered_held_out =
            aggregate_ordered_scenarios(&[vector], &labels, &["otrf-a".into()]).unwrap();
        assert_eq!(
            validate_matching_scenario_modes(&ordered_source, &ordered_held_out).unwrap(),
            ScenarioEncodingMode::RelativeRecordPosition
        );
    }

    #[test]
    fn mode_validation_rejects_empty_or_mixed_representations() {
        let vector = random_hv(123);
        let labels = vec!["execution".into()];
        let source = aggregate_scenarios(
            std::slice::from_ref(&vector),
            &labels,
            &["source-a".into()],
        )
        .unwrap();
        let held_out_ordered = aggregate_ordered_scenarios(
            std::slice::from_ref(&vector),
            &labels,
            &["otrf-a".into()],
        )
        .unwrap();
        assert!(
            validate_matching_scenario_modes(&source, &held_out_ordered)
                .unwrap_err()
                .contains("modes differ")
        );
        assert!(validate_matching_scenario_modes(&[], &held_out_ordered).is_err());
        assert!(validate_matching_scenario_modes(&source, &[]).is_err());
    }

    #[test]
    fn aggregation_rejects_conflicting_labels_within_a_scenario() {
        let hvs = vec![random_hv(1), random_hv(2)];
        let labels = vec!["execution".into(), "persistence".into()];
        let groups = vec!["same-capture".into(), "same-capture".into()];
        let error = aggregate_scenarios(&hvs, &labels, &groups).unwrap_err();
        assert!(error.contains("conflicting labels"));
    }

    #[test]
    fn aggregation_rejects_empty_or_mismatched_inputs() {
        let hvs = vec![random_hv(1), random_hv(2)];
        let labels = vec!["execution".into(), "execution".into()];
        let groups = vec!["a".into(), "b".into()];
        assert!(aggregate_scenarios(&[], &[], &[]).is_err());
        assert!(aggregate_scenarios(&hvs, &labels[..1], &groups).is_err());
        assert!(aggregate_scenarios(&hvs, &labels, &groups[..1]).is_err());
    }

    #[test]
    fn aggregation_rejects_invalid_dimensions_and_empty_metadata() {
        let hvs = vec![vec![1, -1]];
        let labels = vec!["execution".into()];
        let groups = vec!["evtx:a".into()];
        assert!(aggregate_scenarios(&hvs, &labels, &groups).unwrap_err().contains("dimensions"));

        let valid_hvs = vec![random_hv(1)];
        assert!(aggregate_scenarios(&valid_hvs, &["".into()], &groups).is_err());
        assert!(aggregate_scenarios(&valid_hvs, &labels, &["".into()]).is_err());
    }
}
