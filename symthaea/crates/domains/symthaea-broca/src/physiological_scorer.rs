// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PhysiologicalScorer — Hardware-Aware Active Inference
//!
//! Executes generated code in a sandbox and returns real performance metrics.
//! This grounds Symthaea's self-optimization in actual machine reality.

use std::process::{Command, Stdio};
use std::time::Instant;
use sysinfo::{Pid, System};

#[derive(Debug, Clone)]
pub struct PhysiologicalProfile {
    pub cpu_percent: f32,
    pub memory_mb: f32,
    pub latency_ms: f32,
    pub error_rate: f32,
    pub success: bool,
}

pub struct PhysiologicalScorer {
    sys: System,
}

impl PhysiologicalScorer {
    pub fn new() -> Self {
        Self {
            sys: System::new_all(),
        }
    }

    pub fn profile_execution(&mut self, service_name: &str, _config: &str) -> PhysiologicalProfile {
        let start = Instant::now();

        // 1. Spawn a dry-run or eval process
        let mut child = Command::new("cargo")
            .arg("check")
            .arg("-p")
            .arg(service_name)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn verification process");

        let pid = child.id();

        // 2. Measure physiology during execution
        let mut max_mem: f32 = 0.0;
        let mut max_cpu: f32 = 0.0;

        while child.try_wait().unwrap().is_none() {
            self.sys.refresh_all();
            if let Some(process) = self.sys.process(Pid::from_u32(pid)) {
                max_mem = max_mem.max(process.memory() as f32 / 1024.0 / 1024.0);
                max_cpu = max_cpu.max(process.cpu_usage());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let duration = start.elapsed();
        let status = child.wait().unwrap();

        PhysiologicalProfile {
            cpu_percent: max_cpu,
            memory_mb: max_mem,
            latency_ms: duration.as_millis() as f32,
            error_rate: if status.success() { 0.0 } else { 1.0 },
            success: status.success(),
        }
    }

    pub fn to_surprisal(&self, profile: &PhysiologicalProfile) -> f32 {
        if !profile.success {
            return 0.95;
        }
        // Normalized metrics: higher usage -> higher surprisal (energy cost)
        let cpu_p = (profile.cpu_percent / 100.0).clamp(0.0, 1.0);
        let mem_p = (profile.memory_mb / 1024.0).clamp(0.0, 1.0);
        let lat_p = (profile.latency_ms / 5000.0).clamp(0.0, 1.0);

        0.4 * cpu_p + 0.3 * mem_p + 0.3 * lat_p
    }
}

impl Default for PhysiologicalScorer {
    fn default() -> Self {
        Self::new()
    }
}
