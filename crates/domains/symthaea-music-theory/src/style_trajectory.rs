// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, score-time contract for compositions that change style grammar.
//!
//! M0 deliberately validates only the trajectory envelope: source/profile
//! digest shape, exact rational segment partitioning, and one declared
//! transition for each adjacent segment boundary. It does not resolve profile
//! references, prove that a grammar is executable, validate a realized score,
//! or claim that a transition sounds musically convincing.

use std::cmp::Ordering;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::rhythm::Duration;

/// Current serialized trajectory contract version.
pub const STYLE_TRAJECTORY_SCHEMA_VERSION: u16 = 1;

/// A style trajectory over one frozen symbolic score.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleTrajectoryPlanV1 {
    pub schema_version: u16,
    /// Digest of the exact source score bytes under the declared schema below.
    pub source_score_sha256: String,
    pub source_score_digest_schema: String,
    /// Exact symbolic composition length in quarter-note beats.
    pub total_beats: Duration,
    /// Ordered, contiguous, half-open intervals covering the complete score.
    pub segments: Vec<StyleTrajectorySegmentV1>,
    /// Exactly one transition record for every adjacent segment pair.
    pub transitions: Vec<StyleTrajectoryTransitionV1>,
}

/// One interval governed by one declared structural grammar owner.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleTrajectorySegmentV1 {
    pub segment_id: String,
    pub start_beat: Duration,
    pub end_beat: Duration,
    /// Opaque canonical grammar/profile reference. Resolution is a later gate.
    pub structural_owner_ref: String,
    /// Opaque style-profile reference used to resolve the non-structural axes.
    pub profile_ref: String,
    /// SHA-256 of the exact immutable profile revision.
    pub profile_sha256: String,
}

/// How two adjacent structural segments relate at their exact shared boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleTransitionMode {
    /// The new grammar begins at the shared boundary without bridge material.
    HardBoundary,
    /// Bridge material is represented as its own segment with its own owner.
    ComposedBridge,
    /// A gradual change; M0 records the intent but does not interpolate axes.
    GradualMorph,
    /// A bounded ensemble overlay; structural ownership remains singular.
    EnsembleOverlay,
}

/// A declared transition between two adjacent segments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StyleTrajectoryTransitionV1 {
    pub transition_id: String,
    pub from_segment_id: String,
    pub to_segment_id: String,
    pub mode: StyleTransitionMode,
}

/// A deterministic validation finding. Codes are stable machine-readable IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyleTrajectoryIssue {
    pub code: &'static str,
    pub segment_id: Option<String>,
    pub transition_id: Option<String>,
}

/// Validate the M0 trajectory envelope without executing or mutating a score.
///
/// Time comparisons use exact rational cross-products in i128. The f64
/// transport projection in Muse Listen is intentionally not involved.
pub fn validate_style_trajectory(
    plan: &StyleTrajectoryPlanV1,
) -> Result<(), Vec<StyleTrajectoryIssue>> {
    let mut issues = Vec::new();
    let mut segment_ids = HashSet::new();
    let mut transition_ids = HashSet::new();

    if plan.schema_version != STYLE_TRAJECTORY_SCHEMA_VERSION {
        push_issue(&mut issues, "unsupported_schema_version", None, None);
    }
    if !is_lower_hex_sha256(&plan.source_score_sha256) {
        push_issue(&mut issues, "invalid_source_score_sha256", None, None);
    }
    if plan.source_score_digest_schema.trim().is_empty() {
        push_issue(&mut issues, "missing_source_score_digest_schema", None, None);
    }
    if !is_positive_position(plan.total_beats) {
        push_issue(&mut issues, "invalid_total_beats", None, None);
    }
    if plan.segments.is_empty() {
        push_issue(&mut issues, "empty_segments", None, None);
    }

    let zero = Duration::zero();
    let mut previous_end: Option<Duration> = None;

    for segment in &plan.segments {
        let segment_id = Some(segment.segment_id.clone());

        if segment.segment_id.trim().is_empty() {
            push_issue(&mut issues, "empty_segment_id", segment_id.clone(), None);
        } else if !segment_ids.insert(segment.segment_id.as_str()) {
            push_issue(&mut issues, "duplicate_segment_id", segment_id.clone(), None);
        }
        if segment.structural_owner_ref.trim().is_empty() {
            push_issue(&mut issues, "missing_structural_owner_ref", segment_id.clone(), None);
        }
        if segment.profile_ref.trim().is_empty() {
            push_issue(&mut issues, "missing_profile_ref", segment_id.clone(), None);
        }
        if !is_lower_hex_sha256(&segment.profile_sha256) {
            push_issue(&mut issues, "invalid_profile_sha256", segment_id.clone(), None);
        }
        if !is_nonnegative_position(segment.start_beat)
            || !is_nonnegative_position(segment.end_beat)
            || compare_exact(segment.start_beat, segment.end_beat) != Some(Ordering::Less)
        {
            push_issue(&mut issues, "invalid_segment_interval", segment_id.clone(), None);
            previous_end = Some(segment.end_beat);
            continue;
        }

        if let Some(previous) = previous_end {
            match compare_exact(segment.start_beat, previous) {
                Some(Ordering::Equal) => {}
                Some(Ordering::Greater) => {
                    push_issue(&mut issues, "segment_partition_gap", segment_id.clone(), None);
                }
                Some(Ordering::Less) => {
                    push_issue(&mut issues, "segment_partition_overlap", segment_id.clone(), None);
                }
                None => {
                    push_issue(
                        &mut issues,
                        "invalid_segment_time_denominator",
                        segment_id.clone(),
                        None,
                    );
                }
            }
        } else if compare_exact(segment.start_beat, zero) != Some(Ordering::Equal) {
            push_issue(
                &mut issues,
                "segment_partition_must_start_at_zero",
                segment_id.clone(),
                None,
            );
        }

        if compare_exact(segment.end_beat, plan.total_beats) == Some(Ordering::Greater) {
            push_issue(&mut issues, "segment_exceeds_total_beats", segment_id.clone(), None);
        }
        previous_end = Some(segment.end_beat);
    }

    if let Some(last) = plan.segments.last() {
        if compare_exact(last.end_beat, plan.total_beats) != Some(Ordering::Equal) {
            push_issue(
                &mut issues,
                "segment_partition_does_not_end_at_total",
                Some(last.segment_id.clone()),
                None,
            );
        }
    }

    let expected_transitions = plan.segments.len().saturating_sub(1);
    if plan.transitions.len() != expected_transitions {
        push_issue(&mut issues, "transition_count_mismatch", None, None);
    }

    for (index, transition) in plan.transitions.iter().enumerate() {
        let transition_id = Some(transition.transition_id.clone());
        if transition.transition_id.trim().is_empty() {
            push_issue(&mut issues, "empty_transition_id", None, transition_id.clone());
        } else if !transition_ids.insert(transition.transition_id.as_str()) {
            push_issue(&mut issues, "duplicate_transition_id", None, transition_id.clone());
        }

        let expected_from = plan.segments.get(index).map(|segment| segment.segment_id.as_str());
        let expected_to = plan.segments.get(index + 1).map(|segment| segment.segment_id.as_str());
        if expected_from != Some(transition.from_segment_id.as_str())
            || expected_to != Some(transition.to_segment_id.as_str())
        {
            push_issue(&mut issues, "transition_order_mismatch", None, transition_id);
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

fn push_issue(
    issues: &mut Vec<StyleTrajectoryIssue>,
    code: &'static str,
    segment_id: Option<String>,
    transition_id: Option<String>,
) {
    issues.push(StyleTrajectoryIssue {
        code,
        segment_id,
        transition_id,
    });
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_nonnegative_position(value: Duration) -> bool {
    value.den() > 0 && value.num() >= 0
}

fn is_positive_position(value: Duration) -> bool {
    value.den() > 0 && value.num() > 0
}

/// Compare two rational beat positions without converting them to floating point.
///
/// Products of two i64 values fit in i128, so this comparison is exact and
/// cannot overflow for values representable by Duration.
fn compare_exact(left: Duration, right: Duration) -> Option<Ordering> {
    if left.den() <= 0 || right.den() <= 0 {
        return None;
    }
    let left_scaled = i128::from(left.num()) * i128::from(right.den());
    let right_scaled = i128::from(right.num()) * i128::from(left.den());
    Some(left_scaled.cmp(&right_scaled))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCORE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PROFILE_DIGEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn segment(id: &str, start: Duration, end: Duration) -> StyleTrajectorySegmentV1 {
        StyleTrajectorySegmentV1 {
            segment_id: id.into(),
            start_beat: start,
            end_beat: end,
            structural_owner_ref: format!("grammar:{id}"),
            profile_ref: format!("style:{id}"),
            profile_sha256: PROFILE_DIGEST.into(),
        }
    }

    fn transition(from: &str, to: &str) -> StyleTrajectoryTransitionV1 {
        StyleTrajectoryTransitionV1 {
            transition_id: format!("{from}-to-{to}"),
            from_segment_id: from.into(),
            to_segment_id: to.into(),
            mode: StyleTransitionMode::HardBoundary,
        }
    }

    fn plan(
        total_beats: Duration,
        segments: Vec<StyleTrajectorySegmentV1>,
        transitions: Vec<StyleTrajectoryTransitionV1>,
    ) -> StyleTrajectoryPlanV1 {
        StyleTrajectoryPlanV1 {
            schema_version: STYLE_TRAJECTORY_SCHEMA_VERSION,
            source_score_sha256: SCORE_DIGEST.into(),
            source_score_digest_schema: "muse-score-serde-json-sha256-v1".into(),
            total_beats,
            segments,
            transitions,
        }
    }

    fn has_code(result: &Result<(), Vec<StyleTrajectoryIssue>>, code: &str) -> bool {
        result
            .as_ref()
            .err()
            .is_some_and(|issues| issues.iter().any(|issue| issue.code == code))
    }

    #[test]
    fn accepts_single_segment_covering_complete_score() {
        let candidate = plan(
            Duration::new(4, 1),
            vec![segment("a", Duration::zero(), Duration::new(4, 1))],
            vec![],
        );
        assert_eq!(validate_style_trajectory(&candidate), Ok(()));
    }

    #[test]
    fn accepts_exact_rational_partition_and_ordered_transition() {
        let candidate = plan(
            Duration::new(1, 1),
            vec![
                segment("a", Duration::zero(), Duration::new(1, 3)),
                segment("b", Duration::new(1, 3), Duration::new(2, 3)),
                segment("c", Duration::new(2, 3), Duration::new(1, 1)),
            ],
            vec![transition("a", "b"), transition("b", "c")],
        );
        assert_eq!(validate_style_trajectory(&candidate), Ok(()));
    }

    #[test]
    fn rejects_partition_gap() {
        let candidate = plan(
            Duration::new(2, 1),
            vec![
                segment("a", Duration::zero(), Duration::new(1, 1)),
                segment("b", Duration::new(3, 2), Duration::new(2, 1)),
            ],
            vec![transition("a", "b")],
        );
        assert!(has_code(&validate_style_trajectory(&candidate), "segment_partition_gap"));
    }

    #[test]
    fn rejects_partition_overlap() {
        let candidate = plan(
            Duration::new(2, 1),
            vec![
                segment("a", Duration::zero(), Duration::new(3, 2)),
                segment("b", Duration::new(1, 1), Duration::new(2, 1)),
            ],
            vec![transition("a", "b")],
        );
        assert!(has_code(&validate_style_trajectory(&candidate), "segment_partition_overlap"));
    }

    #[test]
    fn rejects_missing_transition_for_adjacent_segments() {
        let candidate = plan(
            Duration::new(2, 1),
            vec![
                segment("a", Duration::zero(), Duration::new(1, 1)),
                segment("b", Duration::new(1, 1), Duration::new(2, 1)),
            ],
            vec![],
        );
        assert!(has_code(&validate_style_trajectory(&candidate), "transition_count_mismatch"));
    }

    #[test]
    fn rejects_transition_that_skips_a_segment() {
        let candidate = plan(
            Duration::new(3, 1),
            vec![
                segment("a", Duration::zero(), Duration::new(1, 1)),
                segment("b", Duration::new(1, 1), Duration::new(2, 1)),
                segment("c", Duration::new(2, 1), Duration::new(3, 1)),
            ],
            vec![transition("a", "c"), transition("b", "c")],
        );
        assert!(has_code(&validate_style_trajectory(&candidate), "transition_order_mismatch"));
    }

    #[test]
    fn rejects_invalid_digests_and_empty_owner() {
        let mut candidate = plan(
            Duration::new(1, 1),
            vec![segment("a", Duration::zero(), Duration::new(1, 1))],
            vec![],
        );
        candidate.source_score_sha256 = "not-a-digest".into();
        candidate.segments[0].profile_sha256 = "not-a-digest".into();
        candidate.segments[0].structural_owner_ref.clear();
        let result = validate_style_trajectory(&candidate);
        assert!(has_code(&result, "invalid_source_score_sha256"));
        assert!(has_code(&result, "invalid_profile_sha256"));
        assert!(has_code(&result, "missing_structural_owner_ref"));
    }

    #[test]
    fn rejects_negative_or_zero_length_intervals() {
        let candidate = plan(
            Duration::new(1, 1),
            vec![segment("a", Duration::new(-1, 1), Duration::new(1, 1))],
            vec![],
        );
        assert!(has_code(&validate_style_trajectory(&candidate), "invalid_segment_interval"));

        let zero_length = plan(
            Duration::new(1, 1),
            vec![segment("a", Duration::zero(), Duration::zero())],
            vec![],
        );
        assert!(has_code(&validate_style_trajectory(&zero_length), "invalid_segment_interval"));
    }
}
