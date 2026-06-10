// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Substrate Proprioception — Sensing the physical hardware as a metabolic organ.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use systemstat::{Platform, System};

/// Real-world hardware metrics from the laptop substrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateMetrics {
    pub cpu_temp_c: f32,
    pub cpu_load_pct: f32,
    pub gpu_temp_c: Option<f32>,
    pub memory_used_gb: f32,
}

pub struct Proprioceptor {
    sys: System,
}

impl Proprioceptor {
    pub fn new() -> Self {
        Self { sys: System::new() }
    }

    /// Capture current hardware state from the laptop.
    pub fn sense_substrate(&self) -> SubstrateMetrics {
        let cpu_temp = self.sys.cpu_temp().unwrap_or(45.0);
        let cpu_load = self
            .sys
            .cpu_load_aggregate()
            .ok()
            .map(|l| {
                std::thread::sleep(std::time::Duration::from_millis(100));
                l.done().ok().map(|c| c.user * 100.0).unwrap_or(10.0)
            })
            .unwrap_or(5.0);

        let mem = self.sys.memory().ok();
        let mem_used = mem
            .map(|m| (m.total.as_u64() - m.free.as_u64()) as f32 / 1_000_000_000.0)
            .unwrap_or(4.0);

        SubstrateMetrics {
            cpu_temp_c: cpu_temp,
            cpu_load_pct: cpu_load,
            gpu_temp_c: None, // NVML requires optional feature
            memory_used_gb: mem_used,
        }
    }

    /// Encode substrate stress into a 16,384-dimensional hypervector.
    pub fn encode_stress(&self, metrics: &SubstrateMetrics) -> ContinuousHV {
        let dim = 16384;
        let mut values = vec![0.0f32; dim];

        // Map CPU temp to the first segment
        let temp_norm = (metrics.cpu_temp_c / 100.0).clamp(0.0, 1.0);
        for v in values.iter_mut().take(dim / 4) {
            *v = temp_norm;
        }

        // Map CPU load to the second segment
        let load_norm = (metrics.cpu_load_pct / 100.0).clamp(0.0, 1.0);
        for v in values.iter_mut().skip(dim / 4).take(dim / 4) {
            *v = load_norm;
        }

        ContinuousHV::from_vec(values)
    }
}
