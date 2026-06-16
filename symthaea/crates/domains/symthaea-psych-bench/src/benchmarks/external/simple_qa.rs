// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Simple QA Adapter
//!
//! Basic factual question answering with known ground truth.
//! Tests whether HDC representations encode factual knowledge.

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use std::collections::BTreeMap;

pub struct SimpleQAAdapter;

impl PsychBenchmark for SimpleQAAdapter {
    fn name(&self) -> &str {
        "External::SimpleQA"
    }
    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut metrics = BTreeMap::new();
        // Placeholder — requires trained HDC knowledge base to be meaningful
        metrics.insert("accuracy".into(), MetricValue::from_samples(&[0.0]));
        metrics.insert(
            "confidence_calibration".into(),
            MetricValue::from_samples(&[0.0]),
        );
        BenchmarkResult {
            benchmark: self.name().to_string(),
            config_label: Some("placeholder".into()),
            metrics,
            elapsed_ms: 0,
            conditions: 0,
            trials_per_condition: 0,
            trial_trace: vec![],
            notes: vec![
                "Placeholder — requires trained model for meaningful results".into(),
                "Requires trained Broca model with knowledge grounding. \
                 Current evaluation uses untrained random HDC vectors."
                    .into(),
            ],
        }
    }
}