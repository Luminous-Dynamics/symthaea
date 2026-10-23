//! Finite synthetic oracle for current morphology versus intended target identity.
//!
//! This is an evaluator/reference fixture, not a learned controller or biological model.
//! The canonical chair is a table plus a four-cell backrest: removing that backrest makes
//! a damaged chair exactly observationally identical to an intact table.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Cell3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

impl Cell3 {
    #[must_use]
    pub const fn new(x: i16, y: i16, z: i16) -> Self {
        Self { x, y, z }
    }
}

pub type Shape = BTreeSet<Cell3>;

/// SHA-256 of canonical_request_v0_1.txt; re-verified by the dedicated workflow.
pub const REQUEST_SHA256: &str = "9348ff465116c23bacf9dfa5f1ed6ce98bb14bfa042b228800d6406ef542681f";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Template {
    pub id: &'static str,
    pub cells: Shape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapMetrics {
    pub intersection: usize,
    pub missing_from_observation: usize,
    pub extra_in_observation: usize,
    pub target_cells: usize,
    pub observed_cells: usize,
    /// Parts per million: 1_000_000 means 1.0.
    pub precision_ppm: u32,
    /// Parts per million: 1_000_000 means 1.0.
    pub recall_ppm: u32,
    /// Parts per million: 1_000_000 means 1.0.
    pub iou_ppm: u32,
}

fn ratio_ppm(numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        return if numerator == 0 { 1_000_000 } else { 0 };
    }
    ((numerator as u128 * 1_000_000) / denominator as u128) as u32
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TargetEvidence<'a> {
    Missing,
    Corrupted,
    Conflicting,
    Claimed(&'a str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Disposition {
    TargetAvailable,
    AbstainAmbiguousHistory,
    AbstainNoCompatibleHistory,
    AbstainInvalidTargetEvidence,
    AbstainTargetNotCompatible,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeReport {
    pub current_exact_matches: Vec<&'static str>,
    pub compatible_histories: Vec<&'static str>,
    pub disposition: Disposition,
    /// None means there was no usable target claim to score.
    pub evidence_matches_ground_truth: Option<bool>,
    pub observed_vs_target: Option<OverlapMetrics>,
    /// Perfect-template reconstruction oracle for the *claimed* target, never a learned result.
    pub target_conditioned_oracle: Option<OverlapMetrics>,
}

/// Builds canonical table/chair templates. Table = 9-cell seat + 8 leg cells;
/// chair = table + three upper backrest cells + one top-backrest cell.
#[must_use]
pub fn canonical_templates() -> Vec<Template> {
    let mut table = Shape::new();
    for x in -1..=1 {
        for y in -1..=1 {
            table.insert(Cell3::new(x, y, 2));
        }
    }
    for x in [-1, 1] {
        for y in [-1, 1] {
            for z in 0..=1 {
                table.insert(Cell3::new(x, y, z));
            }
        }
    }

    let mut chair = table.clone();
    for x in -1..=1 {
        chair.insert(Cell3::new(x, 1, 3));
    }
    chair.insert(Cell3::new(0, 1, 4));

    vec![
        Template {
            id: "chair",
            cells: chair,
        },
        Template {
            id: "table",
            cells: table,
        },
    ]
}

#[must_use]
pub fn template<'a>(templates: &'a [Template], id: &str) -> Option<&'a Template> {
    templates.iter().find(|candidate| candidate.id == id)
}

/// Exact matches against currently observed shape. This is not historical target inference.
#[must_use]
pub fn classify_exact(observed: &Shape, templates: &[Template]) -> Vec<&'static str> {
    let mut matches: Vec<_> = templates
        .iter()
        .filter(|candidate| candidate.cells.eq(observed))
        .map(|candidate| candidate.id)
        .collect();
    matches.sort_unstable();
    matches
}

/// Templates from which the observation can result by cell deletion alone, within the budget.
/// This is a compatibility set, not a probability or an authentication of target memory.
#[must_use]
pub fn compatible_histories(
    observed: &Shape,
    templates: &[Template],
    max_missing_cells: usize,
) -> Vec<&'static str> {
    let mut compatible = Vec::new();
    for candidate in templates {
        if observed.is_subset(&candidate.cells)
            && candidate.cells.len().saturating_sub(observed.len()) <= max_missing_cells
        {
            compatible.push(candidate.id);
        }
    }
    compatible.sort_unstable();
    compatible
}

#[must_use]
pub fn overlap_metrics(target: &Shape, observed_or_reconstruction: &Shape) -> OverlapMetrics {
    let intersection = target.intersection(observed_or_reconstruction).count();
    let missing = target.difference(observed_or_reconstruction).count();
    let extra = observed_or_reconstruction.difference(target).count();
    let union = target.union(observed_or_reconstruction).count();
    OverlapMetrics {
        intersection,
        missing_from_observation: missing,
        extra_in_observation: extra,
        target_cells: target.len(),
        observed_cells: observed_or_reconstruction.len(),
        precision_ppm: ratio_ppm(intersection, observed_or_reconstruction.len()),
        recall_ppm: ratio_ppm(intersection, target.len()),
        iou_ppm: ratio_ppm(intersection, union),
    }
}

/// Evaluates claimed target evidence against evaluator-only ground truth.
/// A false-but-compatible claim remains a false claim even if its template perfectly fits the
/// current observation. That distinction is the point of this oracle.
#[must_use]
pub fn evaluate_episode(
    ground_truth_target_id: &str,
    observed: &Shape,
    templates: &[Template],
    max_missing_cells: usize,
    evidence: TargetEvidence<'_>,
) -> EpisodeReport {
    let current_exact_matches = classify_exact(observed, templates);
    let compatible = compatible_histories(observed, templates, max_missing_cells);
    let observed_vs_target = template(templates, ground_truth_target_id)
        .map(|target| overlap_metrics(&target.cells, observed));

    let (disposition, evidence_matches_ground_truth, target_conditioned_oracle) = match evidence {
        TargetEvidence::Missing | TargetEvidence::Corrupted | TargetEvidence::Conflicting => {
            let disposition = if compatible.is_empty() {
                Disposition::AbstainNoCompatibleHistory
            } else if compatible.len() > 1 {
                Disposition::AbstainAmbiguousHistory
            } else {
                // Uniqueness under a deletion-only model does not prove that the model family is
                // complete, and does not authenticate the missing/corrupted history record.
                Disposition::AbstainInvalidTargetEvidence
            };
            (disposition, None, None)
        }
        TargetEvidence::Claimed(claimed_id) => match template(templates, claimed_id) {
            None => (Disposition::AbstainInvalidTargetEvidence, Some(false), None),
            Some(candidate) if !compatible.contains(&candidate.id) => (
                Disposition::AbstainTargetNotCompatible,
                Some(claimed_id == ground_truth_target_id),
                None,
            ),
            Some(candidate) => (
                Disposition::TargetAvailable,
                Some(claimed_id == ground_truth_target_id),
                Some(overlap_metrics(&candidate.cells, &candidate.cells)),
            ),
        },
    };

    EpisodeReport {
        current_exact_matches,
        compatible_histories: compatible,
        disposition,
        evidence_matches_ground_truth,
        observed_vs_target,
        target_conditioned_oracle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape<'a>(templates: &'a [Template], id: &str) -> &'a Shape {
        &template(templates, id).expect("canonical template").cells
    }

    fn chair_without_backrest(templates: &[Template]) -> Shape {
        shape(templates, "table").clone()
    }

    #[test]
    fn canonical_request_digest_is_pinned() {
        assert_eq!(
            REQUEST_SHA256,
            "9348ff465116c23bacf9dfa5f1ed6ce98bb14bfa042b228800d6406ef542681f"
        );
    }

    #[test]
    fn intact_shapes_are_distinct_and_exactly_classifiable() {
        let templates = canonical_templates();
        assert_ne!(shape(&templates, "chair"), shape(&templates, "table"));
        assert_eq!(
            classify_exact(shape(&templates, "chair"), &templates),
            vec!["chair"]
        );
        assert_eq!(
            classify_exact(shape(&templates, "table"), &templates),
            vec!["table"]
        );
    }

    #[test]
    fn removing_chair_backrest_produces_exact_table_observation() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        assert_eq!(observation, *shape(&templates, "table"));
        assert_eq!(classify_exact(&observation, &templates), vec!["table"]);
    }

    #[test]
    fn no_memory_abstains_when_two_histories_are_compatible() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        let report = evaluate_episode(
            "chair",
            &observation,
            &templates,
            4,
            TargetEvidence::Missing,
        );
        assert_eq!(report.compatible_histories, vec!["chair", "table"]);
        assert_eq!(report.disposition, Disposition::AbstainAmbiguousHistory);
        assert_eq!(report.evidence_matches_ground_truth, None);
    }

    #[test]
    fn correct_target_claim_gets_separate_oracle_upper_bound() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        let report = evaluate_episode(
            "chair",
            &observation,
            &templates,
            4,
            TargetEvidence::Claimed("chair"),
        );
        assert_eq!(report.disposition, Disposition::TargetAvailable);
        assert_eq!(report.evidence_matches_ground_truth, Some(true));
        assert_eq!(
            report.target_conditioned_oracle.expect("oracle").iou_ppm,
            1_000_000
        );
        assert_eq!(
            report
                .observed_vs_target
                .expect("target metrics")
                .missing_from_observation,
            4
        );
    }

    #[test]
    fn false_compatible_target_claim_is_not_hidden_by_perfect_shape_score() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        let report = evaluate_episode(
            "chair",
            &observation,
            &templates,
            4,
            TargetEvidence::Claimed("table"),
        );
        assert_eq!(report.disposition, Disposition::TargetAvailable);
        assert_eq!(report.evidence_matches_ground_truth, Some(false));
        assert_eq!(
            report
                .target_conditioned_oracle
                .expect("table oracle")
                .iou_ppm,
            1_000_000
        );
        assert_eq!(
            report.observed_vs_target.expect("chair metrics").iou_ppm,
            809_523
        );
    }

    #[test]
    fn invalid_or_incompatible_target_claim_fails_closed() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        let invalid = evaluate_episode(
            "chair",
            &observation,
            &templates,
            4,
            TargetEvidence::Claimed("spaceship"),
        );
        assert_eq!(
            invalid.disposition,
            Disposition::AbstainInvalidTargetEvidence
        );
        assert_eq!(invalid.evidence_matches_ground_truth, Some(false));
        let small_budget = evaluate_episode(
            "chair",
            &observation,
            &templates,
            2,
            TargetEvidence::Claimed("chair"),
        );
        assert_eq!(
            small_budget.disposition,
            Disposition::AbstainTargetNotCompatible
        );
    }

    #[test]
    fn corrupted_or_conflicting_target_memory_does_not_authorize_recovery() {
        let templates = canonical_templates();
        let observation = chair_without_backrest(&templates);
        assert_eq!(
            evaluate_episode(
                "chair",
                &observation,
                &templates,
                4,
                TargetEvidence::Corrupted
            )
            .disposition,
            Disposition::AbstainAmbiguousHistory
        );
        assert_eq!(
            evaluate_episode(
                "chair",
                &observation,
                &templates,
                4,
                TargetEvidence::Conflicting
            )
            .disposition,
            Disposition::AbstainAmbiguousHistory
        );
    }

    #[test]
    fn overlap_counts_and_integer_metrics_are_deterministic() {
        let templates = canonical_templates();
        let metrics = overlap_metrics(shape(&templates, "chair"), shape(&templates, "table"));
        assert_eq!(metrics.intersection, 17);
        assert_eq!(metrics.missing_from_observation, 4);
        assert_eq!(metrics.extra_in_observation, 0);
        assert_eq!(metrics.target_cells, 21);
        assert_eq!(metrics.observed_cells, 17);
        assert_eq!(metrics.recall_ppm, 809_523);
        assert_eq!(metrics.iou_ppm, 809_523);
        assert_eq!(metrics.precision_ppm, 1_000_000);
    }
}
