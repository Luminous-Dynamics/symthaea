//! Capture/scenario-level aggregation for leakage-resistant cybersecurity evaluation.
//!
//! This module deliberately treats one source capture or labeled scenario as one
//! statistical example. Event labels and group IDs are evaluator metadata; only
//! event vectors contribute to the aggregate representation. The v1 operation is
//! an orderless HDC bundle, not a temporal sequence model. Keeping that limitation
//! explicit prevents scenario aggregation from being overclaimed as attack-chain
//! correlation.

use crate::encoder::{HDC_DIM, Hdv, bundle};
use std::collections::BTreeMap;

/// One independently labeled capture/scenario represented by a bundle of its events.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioEmbedding {
    /// Stable source-file or scenario-directory identity.
    pub scenario_id: String,
    /// Evaluator-only ground-truth label; never included in the vector.
    pub label: String,
    /// Number of event vectors bundled into the scenario representation.
    pub event_count: usize,
    /// Orderless HDC bundle of the scenario's event vectors.
    pub vector: Hdv,
}

/// Bundle event vectors so that each source capture/scenario contributes exactly
/// one sample to training or evaluation.
///
/// Group membership is deterministic and output is sorted by scenario ID. A group
/// must have one non-empty label; conflicting labels fail closed rather than
/// arbitrarily picking one. The underlying bundle is permutation-invariant, so
/// the order returned by filesystem traversal does not affect the scenario vector.
///
/// This function does not establish that the label accurately describes every
/// event in a capture. Mixed-technique captures need finer scenario annotations.
pub fn aggregate_scenarios(
    hvs: &[Hdv],
    labels: &[String],
    groups: &[String],
) -> Result<Vec<ScenarioEmbedding>, String> {
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

    let mut scenarios = Vec::with_capacity(by_group.len());
    for (scenario_id, indices) in by_group {
        let event_vectors: Vec<Hdv> = indices.iter().map(|&index| hvs[index].clone()).collect();
        scenarios.push(ScenarioEmbedding {
            scenario_id,
            label: labels[indices[0]].clone(),
            event_count: indices.len(),
            vector: bundle(&event_vectors),
        });
    }
    Ok(scenarios)
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
