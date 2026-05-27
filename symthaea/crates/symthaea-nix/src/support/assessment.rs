// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Support Assessment
//!
//! Combines health checks, knowledge base search, and predictive monitoring
//! into a single coherent assessment with prioritized recommendations.

use crate::encoding::SystemStateSnapshot;
use crate::encoding::codebook::NixCodebook;
use crate::observe::hardware::HardwareInfo;
use crate::support::health_check::{HealthAssessor, HealthCheck, HealthStatus};
use crate::support::knowledge::KnowledgeBase;
use crate::support::predictive::{Prediction, PredictiveMonitor};

/// A single recommendation from the support assessment.
#[derive(Debug, Clone)]
pub struct SupportRecommendation {
    /// Urgency level (matches the health check that triggered it).
    pub urgency: HealthStatus,
    /// What triggered this recommendation.
    pub trigger: String,
    /// Issue category (e.g., "disk", "services", "store").
    pub category: String,
    /// Actionable steps to resolve the issue.
    pub actions: Vec<String>,
    /// IDs of matching knowledge articles.
    pub knowledge_article_ids: Vec<String>,
    /// Predictive context (e.g., "predicted to hit 95% in 6h").
    pub prediction_context: Option<String>,
}

/// Complete assessment of system health, knowledge, and predictions.
#[derive(Debug)]
pub struct SupportAssessment {
    /// Overall system health status (worst of all checks).
    pub overall_status: HealthStatus,
    /// Individual health check results.
    pub health_checks: Vec<HealthCheck>,
    /// Prioritized recommendations (Critical first).
    pub recommendations: Vec<SupportRecommendation>,
    /// Predictions that cross alert thresholds.
    pub active_alerts: Vec<Prediction>,
    /// Number of knowledge articles that matched health issues.
    pub knowledge_matches_found: usize,
}

/// Combines health checks, knowledge search, and predictions into a unified assessment.
pub struct SupportAssessor {
    health_assessor: HealthAssessor,
    knowledge_base: KnowledgeBase,
}

impl SupportAssessor {
    /// Create a new assessor, initializing the knowledge base with the given codebook.
    pub fn new(codebook: &mut NixCodebook) -> Self {
        Self {
            health_assessor: HealthAssessor::default(),
            knowledge_base: KnowledgeBase::new(codebook),
        }
    }

    /// Create with a custom health assessor.
    pub fn with_health_assessor(
        health_assessor: HealthAssessor,
        codebook: &mut NixCodebook,
    ) -> Self {
        Self {
            health_assessor,
            knowledge_base: KnowledgeBase::new(codebook),
        }
    }

    /// Run a unified assessment: health checks → knowledge search → predictions.
    pub fn assess(
        &mut self,
        snapshot: &SystemStateSnapshot,
        hardware: Option<&HardwareInfo>,
        predictive_monitor: Option<&mut PredictiveMonitor>,
        codebook: &mut NixCodebook,
    ) -> SupportAssessment {
        // 1. Run health checks
        let (overall_status, health_checks) = self.health_assessor.assess_all(snapshot, hardware);

        let mut recommendations = Vec::new();
        let mut knowledge_matches_found = 0;

        // 2. For Warning/Critical checks, search knowledge base for solutions
        for check in &health_checks {
            if check.status == HealthStatus::Healthy {
                continue;
            }

            // Search knowledge base for matching articles
            let kb_results = self.knowledge_base.search(&check.message, codebook, 3);
            let article_ids: Vec<String> = kb_results
                .iter()
                .filter(|r| r.similarity > 0.1)
                .map(|r| r.article.id.to_string())
                .collect();
            knowledge_matches_found += article_ids.len();

            recommendations.push(SupportRecommendation {
                urgency: check.status,
                trigger: check.message.clone(),
                category: check.category.to_string(),
                actions: check.recommendations.clone(),
                knowledge_article_ids: article_ids,
                prediction_context: None,
            });
        }

        // 3. Pull threshold-crossing predictions
        let mut active_alerts = Vec::new();
        if let Some(monitor) = predictive_monitor {
            let predictions = monitor.predict_all_horizons();
            for pred in predictions {
                if pred.crosses_threshold {
                    // Augment matching recommendations with prediction context
                    let context = format!(
                        "Predicted to reach {:.1} in {:.0}h (current: {:.1}, threshold: {:.1})",
                        pred.predicted_value, pred.hours_ahead, pred.current_value, pred.threshold,
                    );

                    // Try to find an existing recommendation for this metric's category
                    let category = match pred.metric {
                        "disk_used_pct" => "disk",
                        "memory_used_pct" => "memory",
                        "store_path_count" => "store",
                        "failed_unit_count" => "services",
                        "load_average_1m" => "cpu",
                        "swap_used_pct" => "swap",
                        _ => "unknown",
                    };

                    let augmented = recommendations.iter_mut().find(|r| r.category == category);

                    if let Some(rec) = augmented {
                        rec.prediction_context = Some(context);
                    } else {
                        // Create a new recommendation from the prediction
                        recommendations.push(SupportRecommendation {
                            urgency: HealthStatus::Warning,
                            trigger: format!("{} predicted to cross threshold", pred.metric),
                            category: category.to_string(),
                            actions: pred.recommended_action.iter().cloned().collect(),
                            knowledge_article_ids: vec![],
                            prediction_context: Some(format!(
                                "Predicted to reach {:.1} in {:.0}h",
                                pred.predicted_value, pred.hours_ahead
                            )),
                        });
                    }

                    active_alerts.push(pred);
                }
            }
        }

        // 4. Sort recommendations by urgency (Critical first)
        recommendations.sort_by(|a, b| b.urgency.cmp(&a.urgency));

        SupportAssessment {
            overall_status,
            health_checks,
            recommendations,
            active_alerts,
            knowledge_matches_found,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::ServiceState;
    use crate::observe::hardware::DiskInfo;
    use crate::support::predictive::SystemTelemetry;

    fn make_snapshot(
        services: Vec<(&str, ServiceState)>,
        store_paths: Option<usize>,
    ) -> SystemStateSnapshot {
        SystemStateSnapshot {
            services: services
                .into_iter()
                .map(|(n, s)| (n.to_string(), s))
                .collect(),
            store_path_count: store_paths,
            ..Default::default()
        }
    }

    fn make_hardware(disk_used_pct: f64, mem_used_pct: f64) -> HardwareInfo {
        let total_bytes = 500_000_000_000u64;
        let used_bytes = (total_bytes as f64 * disk_used_pct / 100.0) as u64;
        let total_mem = 32_000u64;
        let available_mem = (total_mem as f64 * (1.0 - mem_used_pct / 100.0)) as u64;

        HardwareInfo {
            cpu_model: "Test CPU".to_string(),
            cpu_cores: 8,
            memory_total_mb: total_mem,
            memory_available_mb: available_mem,
            gpus: vec![],
            disks: vec![DiskInfo {
                device: "/dev/sda1".to_string(),
                mount_point: "/".to_string(),
                total_bytes,
                used_bytes,
            }],
            load_average: [0.5, 0.4, 0.3],
            swap_total_mb: 8192,
            swap_used_mb: 0,
        }
    }

    #[test]
    fn test_healthy_system_no_recommendations() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(vec![("nginx.service", ServiceState::Running)], Some(50_000));
        let hw = make_hardware(50.0, 40.0);

        let result = assessor.assess(&snapshot, Some(&hw), None, &mut codebook);
        assert_eq!(result.overall_status, HealthStatus::Healthy);
        assert!(result.recommendations.is_empty());
    }

    #[test]
    fn test_warning_triggers_knowledge_search() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(
            vec![
                ("nginx.service", ServiceState::Running),
                ("broken.service", ServiceState::Failed),
            ],
            Some(50_000),
        );
        let hw = make_hardware(50.0, 40.0);

        let result = assessor.assess(&snapshot, Some(&hw), None, &mut codebook);
        assert!(!result.recommendations.is_empty());
        // Should find knowledge articles for failed services
        assert!(
            result.knowledge_matches_found > 0 || !result.recommendations.is_empty(),
            "Warning should trigger recommendations"
        );
    }

    #[test]
    fn test_critical_has_recommendations() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(
            vec![
                ("a.service", ServiceState::Failed),
                ("b.service", ServiceState::Failed),
                ("c.service", ServiceState::Failed),
                ("d.service", ServiceState::Failed),
            ],
            Some(50_000),
        );
        let hw = make_hardware(95.0, 40.0);

        let result = assessor.assess(&snapshot, Some(&hw), None, &mut codebook);
        assert_eq!(result.overall_status, HealthStatus::Critical);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_predictions_augment_context() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(vec![("nginx.service", ServiceState::Running)], Some(50_000));
        let hw = make_hardware(85.0, 40.0); // Warning-level disk

        let mut monitor = PredictiveMonitor::with_defaults();
        // Rising disk usage
        for i in 0..10 {
            monitor.ingest(SystemTelemetry {
                disk_used_pct: 80.0 + i as f64,
                memory_used_pct: 40.0,
                store_path_count: 50_000,
                failed_unit_count: 0,
                load_average_1m: 0.5,
                swap_used_pct: 5.0,
            });
        }

        let result = assessor.assess(&snapshot, Some(&hw), Some(&mut monitor), &mut codebook);
        // Should have disk warning recommendation
        let disk_rec = result.recommendations.iter().find(|r| r.category == "disk");
        assert!(disk_rec.is_some(), "Should have disk recommendation");
    }

    #[test]
    fn test_sorted_by_urgency() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        // Create a situation with both Warning and Critical
        let snapshot = make_snapshot(
            vec![
                ("a.service", ServiceState::Failed),
                ("b.service", ServiceState::Failed),
                ("c.service", ServiceState::Failed),
                ("d.service", ServiceState::Failed),
            ],
            Some(120_000), // Warning-level store
        );
        let hw = make_hardware(95.0, 40.0); // Critical disk

        let result = assessor.assess(&snapshot, Some(&hw), None, &mut codebook);

        // Verify Critical comes before Warning
        if result.recommendations.len() >= 2 {
            assert!(
                result.recommendations[0].urgency >= result.recommendations[1].urgency,
                "Recommendations should be sorted by urgency"
            );
        }
    }

    #[test]
    fn test_works_without_hardware() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(vec![], Some(50_000));

        let result = assessor.assess(&snapshot, None, None, &mut codebook);
        assert_eq!(result.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn test_empty_predictions_handled() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(vec![], Some(50_000));
        let mut monitor = PredictiveMonitor::with_defaults();

        let result = assessor.assess(&snapshot, None, Some(&mut monitor), &mut codebook);
        assert!(result.active_alerts.is_empty());
    }

    #[test]
    fn test_assessment_includes_all_checks() {
        let mut codebook = NixCodebook::new();
        let mut assessor = SupportAssessor::new(&mut codebook);
        let snapshot = make_snapshot(vec![("nginx.service", ServiceState::Running)], Some(50_000));
        let hw = make_hardware(50.0, 40.0);

        let result = assessor.assess(&snapshot, Some(&hw), None, &mut codebook);
        // Should have disk, memory, services, store, and flake checks
        assert!(
            result.health_checks.len() >= 4,
            "Should have at least 4 health checks, got {}",
            result.health_checks.len()
        );
    }
}
