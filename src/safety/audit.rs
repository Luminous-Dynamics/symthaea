// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety Audit Report — Genesis Mission Challenge 26
//!
//! Generates structured safety audit reports from SafetyAgent history.
//! Supports both markdown (human-readable) and JSON (machine-readable) export.

use serde::{Deserialize, Serialize};

use super::agent::{SafetyAssessment, SafetyLevel, SafetyOverrideEntry};

/// A complete safety audit report covering a window of assessments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAuditReport {
    /// Report title.
    pub title: String,
    /// ISO 8601 generation timestamp.
    pub generated_at: String,
    /// Number of assessments covered.
    pub total_assessments: usize,
    /// Breakdown: how many assessments at each level.
    pub level_counts: LevelCounts,
    /// Maximum (worst) safety level observed.
    pub peak_level: SafetyLevel,
    /// Mean consciousness level across all assessments.
    pub mean_consciousness: f32,
    /// Minimum consciousness level observed.
    pub min_consciousness: f32,
    /// Mean prediction error across all assessments.
    pub mean_prediction_error: f32,
    /// Whether any Red events occurred.
    pub had_emergency: bool,
    /// Top reasons contributing to escalations.
    pub top_reasons: Vec<(String, usize)>,
    /// Individual assessments (optional; may be omitted for large reports).
    pub assessments: Vec<SafetyAssessment>,
    /// Human override events (EU AI Act Article 14 audit trail).
    pub overrides: Vec<SafetyOverrideEntry>,
}

/// Count of assessments at each safety level.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelCounts {
    pub green: usize,
    pub yellow: usize,
    pub orange: usize,
    pub red: usize,
}

impl SafetyAuditReport {
    /// Generate a report from a slice of safety assessments.
    pub fn from_assessments(assessments: &[SafetyAssessment]) -> Self {
        Self::from_assessments_and_overrides(assessments, &[])
    }

    /// Generate a report from assessments and human override events.
    ///
    /// The override log is included verbatim in the report for Article 14 compliance.
    pub fn from_assessments_and_overrides(
        assessments: &[SafetyAssessment],
        overrides: &[SafetyOverrideEntry],
    ) -> Self {
        let total = assessments.len();

        let mut counts = LevelCounts::default();
        let mut sum_consciousness = 0.0f64;
        let mut min_consciousness = f32::MAX;
        let mut sum_pred_error = 0.0f64;
        let mut peak_level = SafetyLevel::Green;
        let mut reason_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for a in assessments {
            match a.level {
                SafetyLevel::Green => counts.green += 1,
                SafetyLevel::Yellow => counts.yellow += 1,
                SafetyLevel::Orange => counts.orange += 1,
                SafetyLevel::Red => counts.red += 1,
            }
            if a.level > peak_level {
                peak_level = a.level;
            }
            sum_consciousness += a.consciousness_level as f64;
            if a.consciousness_level < min_consciousness {
                min_consciousness = a.consciousness_level;
            }
            sum_pred_error += a.prediction_error as f64;

            for reason in &a.reasons {
                *reason_counts.entry(reason.clone()).or_insert(0) += 1;
            }
        }

        let mean_consciousness = if total > 0 {
            (sum_consciousness / total as f64) as f32
        } else {
            0.0
        };
        let mean_prediction_error = if total > 0 {
            (sum_pred_error / total as f64) as f32
        } else {
            0.0
        };
        if min_consciousness == f32::MAX {
            min_consciousness = 0.0;
        }

        // Top reasons by frequency
        let mut top_reasons: Vec<(String, usize)> = reason_counts.into_iter().collect();
        top_reasons.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_reasons.truncate(10);

        Self {
            title: "Symthaea Safety Audit Report".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            total_assessments: total,
            level_counts: counts,
            peak_level,
            mean_consciousness,
            min_consciousness,
            mean_prediction_error,
            had_emergency: peak_level == SafetyLevel::Red,
            top_reasons,
            assessments: assessments.to_vec(),
            overrides: overrides.to_vec(),
        }
    }

    /// Export as JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export as markdown string.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!("**Generated**: {}\n\n", self.generated_at));
        md.push_str("## Summary\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!(
            "| Total Assessments | {} |\n",
            self.total_assessments
        ));
        md.push_str(&format!("| Peak Level | {} |\n", self.peak_level.label()));
        md.push_str(&format!(
            "| Mean Consciousness | {:.3} |\n",
            self.mean_consciousness
        ));
        md.push_str(&format!(
            "| Min Consciousness | {:.3} |\n",
            self.min_consciousness
        ));
        md.push_str(&format!(
            "| Mean Prediction Error | {:.3} |\n",
            self.mean_prediction_error
        ));
        md.push_str(&format!(
            "| Emergency Events | {} |\n",
            if self.had_emergency { "YES" } else { "No" }
        ));

        md.push_str("\n## Level Distribution\n\n");
        md.push_str(&format!("- Green: {}\n", self.level_counts.green));
        md.push_str(&format!("- Yellow: {}\n", self.level_counts.yellow));
        md.push_str(&format!("- Orange: {}\n", self.level_counts.orange));
        md.push_str(&format!("- Red: {}\n", self.level_counts.red));

        if !self.top_reasons.is_empty() {
            md.push_str("\n## Top Escalation Reasons\n\n");
            for (reason, count) in &self.top_reasons {
                md.push_str(&format!("- **{}**: {} occurrences\n", reason, count));
            }
        }

        if !self.overrides.is_empty() {
            md.push_str("\n## Human Override Events (Article 14)\n\n");
            md.push_str("| Cycle | Operator | Original | Override | Reason |\n");
            md.push_str("|-------|----------|----------|----------|--------|\n");
            for entry in &self.overrides {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    entry.cycle,
                    entry.operator,
                    entry.original_level.label(),
                    entry.override_level.label(),
                    entry.reason,
                ));
            }
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_assessments() -> Vec<SafetyAssessment> {
        vec![
            SafetyAssessment {
                cycle: 0,
                level: SafetyLevel::Green,
                raw_level: SafetyLevel::Green,
                consciousness_level: 0.8,
                prediction_error: 0.1,
                temporal_coherence: 0.7,
                reasons: vec!["all metrics within normal range".to_string()],
            },
            SafetyAssessment {
                cycle: 1,
                level: SafetyLevel::Yellow,
                raw_level: SafetyLevel::Yellow,
                consciousness_level: 0.5,
                prediction_error: 0.3,
                temporal_coherence: 0.6,
                reasons: vec!["consciousness_level 0.500 < yellow threshold 0.600".to_string()],
            },
            SafetyAssessment {
                cycle: 2,
                level: SafetyLevel::Red,
                raw_level: SafetyLevel::Red,
                consciousness_level: 0.1,
                prediction_error: 0.9,
                temporal_coherence: 0.2,
                reasons: vec![
                    "consciousness_level 0.100 < red threshold 0.150".to_string(),
                    "prediction_error 0.900 > threshold 0.700".to_string(),
                ],
            },
        ]
    }

    #[test]
    fn test_report_from_assessments() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        assert_eq!(report.total_assessments, 3);
        assert_eq!(report.level_counts.green, 1);
        assert_eq!(report.level_counts.yellow, 1);
        assert_eq!(report.level_counts.red, 1);
        assert_eq!(report.peak_level, SafetyLevel::Red);
        assert!(report.had_emergency);
    }

    #[test]
    fn test_report_mean_consciousness() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        // (0.8 + 0.5 + 0.1) / 3 ≈ 0.467
        assert!((report.mean_consciousness - 0.467).abs() < 0.01);
    }

    #[test]
    fn test_report_min_consciousness() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        assert!((report.min_consciousness - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_report_json_export() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        let json = report.to_json();
        assert!(json.contains("Safety Audit Report"));
        assert!(json.contains("Red"));
    }

    #[test]
    fn test_report_markdown_export() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        let md = report.to_markdown();
        assert!(md.contains("# Symthaea Safety Audit Report"));
        assert!(md.contains("Peak Level"));
        assert!(md.contains("Emergency Events"));
    }

    #[test]
    fn test_empty_report() {
        let report = SafetyAuditReport::from_assessments(&[]);
        assert_eq!(report.total_assessments, 0);
        assert_eq!(report.peak_level, SafetyLevel::Green);
        assert!(!report.had_emergency);
        assert_eq!(report.mean_consciousness, 0.0);
    }

    #[test]
    fn test_top_reasons_sorted() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        assert!(!report.top_reasons.is_empty());
        // Verify descending sort order by frequency
        for i in 1..report.top_reasons.len() {
            assert!(
                report.top_reasons[i - 1].1 >= report.top_reasons[i].1,
                "top_reasons not sorted: {} ({}) before {} ({})",
                report.top_reasons[i - 1].0,
                report.top_reasons[i - 1].1,
                report.top_reasons[i].0,
                report.top_reasons[i].1
            );
        }
    }

    #[test]
    fn test_report_with_overrides() {
        let overrides = vec![SafetyOverrideEntry {
            timestamp_nanos: 1234567890,
            cycle: 42,
            operator: "test-operator".to_string(),
            reason: "sensor miscalibration".to_string(),
            original_level: SafetyLevel::Red,
            override_level: SafetyLevel::Green,
        }];
        let report =
            SafetyAuditReport::from_assessments_and_overrides(&make_assessments(), &overrides);
        assert_eq!(report.overrides.len(), 1);
        assert_eq!(report.overrides[0].operator, "test-operator");

        let md = report.to_markdown();
        assert!(md.contains("Human Override Events"));
        assert!(md.contains("test-operator"));
        assert!(md.contains("sensor miscalibration"));

        let json = report.to_json();
        assert!(json.contains("test-operator"));
    }

    #[test]
    fn test_report_no_overrides_section_when_empty() {
        let report = SafetyAuditReport::from_assessments(&make_assessments());
        let md = report.to_markdown();
        assert!(!md.contains("Human Override Events"));
    }
}
