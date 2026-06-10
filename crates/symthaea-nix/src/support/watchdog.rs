// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Post-Rebuild Consciousness Watchdog
//!
//! After `nixos-rebuild test`, monitors system health using HDC surprise
//! detection. If the system degrades (cosine distance from baseline exceeds
//! the surprise threshold for N consecutive checks), the watchdog auto-reverts
//! to the pre-rebuild generation.
//!
//! This is the auto-rollback feature NixOS has been missing since 2019:
//! `nixos-rebuild test` applies changes but auto-reverts on reboot;
//! the watchdog adds *runtime* degradation detection with conscious rollback.

use std::time::{Duration, Instant};

use symthaea_core::hdc::ContinuousHV;

use crate::action::generation_manager::GenerationManager;
use crate::encoding::{NixCodebook, SystemStateEncoder, SystemStateSnapshot};
use crate::observe::SystemObserver;
use crate::support::health_check::{HealthAssessor, HealthCheck, HealthStatus};

/// Controls how aggressively the watchdog responds to degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutonomyLevel {
    /// Only report degradation, never take action (default).
    ReportOnly,
    /// Report degradation and include the rollback command, but don't execute.
    DryRun,
    /// Automatically execute rollback when degradation is detected.
    FullAutonomous,
}

/// Watchdog configuration.
#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// Maximum monitoring duration before declaring stable.
    pub timeout: Duration,
    /// Interval between health checks.
    pub check_interval: Duration,
    /// Cosine distance threshold — above this means "surprised".
    pub surprise_threshold: f64,
    /// Number of consecutive failures before reverting.
    pub consecutive_failures_to_revert: u32,
    /// Controls rollback behavior: report-only, dry-run, or fully autonomous.
    pub autonomy_level: AutonomyLevel,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),       // 5 minutes
            check_interval: Duration::from_secs(10), // every 10 seconds
            surprise_threshold: 0.3,
            consecutive_failures_to_revert: 3,
            autonomy_level: AutonomyLevel::ReportOnly,
        }
    }
}

/// Verdict after the watchdog monitoring period.
#[derive(Debug, Clone)]
pub enum WatchdogVerdict {
    /// System stabilized within the timeout — safe to promote to `switch`.
    Stabilized {
        health: Vec<HealthCheck>,
        duration: Duration,
        checks_performed: u32,
    },
    /// System degraded — surprise threshold exceeded consistently.
    Degraded {
        reason: String,
        surprise: f64,
        health: Vec<HealthCheck>,
        checks_performed: u32,
    },
    /// System was automatically reverted to pre-rebuild generation.
    Reverted { reason: String, pre_gen: u64 },
    /// Monitoring failed (couldn't take snapshots).
    Error { message: String },
}

/// Post-rebuild consciousness monitor.
pub struct Watchdog {
    config: WatchdogConfig,
    assessor: HealthAssessor,
}

impl Default for Watchdog {
    fn default() -> Self {
        Self::new(WatchdogConfig::default())
    }
}

impl Watchdog {
    /// Create a new watchdog with the given configuration.
    pub fn new(config: WatchdogConfig) -> Self {
        Self {
            config,
            assessor: HealthAssessor::default(),
        }
    }

    /// Monitor the system after a rebuild. Blocks until a verdict is reached.
    ///
    /// # Arguments
    /// * `codebook` — shared codebook for encoding system state as HDC vectors
    /// * `baseline_hv` — the HDC-encoded system state *before* the rebuild
    /// * `pre_gen` — the NixOS generation number before the rebuild (used for rollback)
    ///
    /// # Returns
    /// A `WatchdogVerdict` indicating whether the system stabilized, degraded,
    /// or was reverted.
    pub fn monitor(
        &self,
        codebook: &mut NixCodebook,
        baseline_hv: &ContinuousHV,
        pre_gen: u64,
    ) -> WatchdogVerdict {
        let start = Instant::now();
        let mut consecutive_failures: u32 = 0;
        let mut last_surprise: f64;
        let mut last_health = Vec::new();
        let mut checks_performed: u32 = 0;

        loop {
            // Check timeout
            if start.elapsed() >= self.config.timeout {
                return WatchdogVerdict::Stabilized {
                    health: last_health,
                    duration: start.elapsed(),
                    checks_performed,
                };
            }

            // Wait for check interval
            std::thread::sleep(self.config.check_interval);
            checks_performed += 1;

            // Take a snapshot
            let snapshot = match SystemObserver::snapshot() {
                Ok(s) => s,
                Err(e) => {
                    consecutive_failures += 1;
                    if consecutive_failures >= self.config.consecutive_failures_to_revert {
                        return WatchdogVerdict::Error {
                            message: format!(
                                "Failed to snapshot {} consecutive times: {}",
                                consecutive_failures, e
                            ),
                        };
                    }
                    continue;
                }
            };

            // Encode and compare to baseline
            let current_hv = {
                let mut encoder = SystemStateEncoder::new(codebook);
                encoder.encode_snapshot(&snapshot)
            };

            let similarity = current_hv.similarity(baseline_hv);
            let surprise = 1.0 - similarity as f64;
            last_surprise = surprise;

            // Run health assessment
            let hw = crate::observe::hardware::HardwareObserver::probe().ok();
            let (overall, health) = self.assessor.assess_all(&snapshot, hw.as_ref());
            last_health = health;

            // Check for degradation
            let is_degraded =
                surprise > self.config.surprise_threshold || overall == HealthStatus::Critical;

            if is_degraded {
                consecutive_failures += 1;
                if consecutive_failures >= self.config.consecutive_failures_to_revert {
                    let reason = format!(
                        "System degraded: surprise={:.3}, health={}, {} consecutive failures",
                        surprise, overall, consecutive_failures
                    );

                    match self.config.autonomy_level {
                        AutonomyLevel::ReportOnly => {
                            return WatchdogVerdict::Degraded {
                                reason,
                                surprise: last_surprise,
                                health: last_health,
                                checks_performed,
                            };
                        }
                        AutonomyLevel::DryRun => {
                            let cmd = GenerationManager::switch_to(pre_gen as u32);
                            let (bin, args) = cmd.to_command();
                            let cmd_str = format!("{} {}", bin, args.join(" "));
                            return WatchdogVerdict::Degraded {
                                reason: format!("{}; would run: {}", reason, cmd_str),
                                surprise: last_surprise,
                                health: last_health,
                                checks_performed,
                            };
                        }
                        AutonomyLevel::FullAutonomous => {
                            let cmd = GenerationManager::switch_to(pre_gen as u32);
                            let (bin, args) = cmd.to_command();
                            let result = std::process::Command::new(&bin).args(&args).status();
                            match result {
                                Ok(status) if status.success() => {
                                    return WatchdogVerdict::Reverted { reason, pre_gen };
                                }
                                Ok(status) => {
                                    return WatchdogVerdict::Degraded {
                                        reason: format!(
                                            "{}; rollback failed (exit {})",
                                            reason,
                                            status.code().unwrap_or(-1)
                                        ),
                                        surprise: last_surprise,
                                        health: last_health,
                                        checks_performed,
                                    };
                                }
                                Err(e) => {
                                    return WatchdogVerdict::Degraded {
                                        reason: format!("{}; rollback exec error: {}", reason, e),
                                        surprise: last_surprise,
                                        health: last_health,
                                        checks_performed,
                                    };
                                }
                            }
                        }
                    }
                }
            } else {
                // Reset consecutive failure counter on a good check
                consecutive_failures = 0;
            }
        }
    }

    /// Take a baseline snapshot and encode it as an HDC vector.
    ///
    /// Call this *before* the rebuild to establish the pre-rebuild state.
    pub fn capture_baseline(
        codebook: &mut NixCodebook,
    ) -> Option<(ContinuousHV, SystemStateSnapshot)> {
        let snapshot = SystemObserver::snapshot().ok()?;
        let hv = {
            let mut encoder = SystemStateEncoder::new(codebook);
            encoder.encode_snapshot(&snapshot)
        };
        Some((hv, snapshot))
    }

    /// Get the watchdog configuration.
    pub fn config(&self) -> &WatchdogConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_config_defaults() {
        let config = WatchdogConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.check_interval, Duration::from_secs(10));
        assert!((config.surprise_threshold - 0.3).abs() < 1e-10);
        assert_eq!(config.consecutive_failures_to_revert, 3);
    }

    #[test]
    fn test_watchdog_creation() {
        let watchdog = Watchdog::default();
        assert_eq!(watchdog.config().timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_watchdog_custom_config() {
        let config = WatchdogConfig {
            timeout: Duration::from_secs(60),
            check_interval: Duration::from_secs(5),
            surprise_threshold: 0.5,
            consecutive_failures_to_revert: 5,
            autonomy_level: AutonomyLevel::DryRun,
        };
        let watchdog = Watchdog::new(config);
        assert_eq!(watchdog.config().timeout, Duration::from_secs(60));
        assert_eq!(watchdog.config().consecutive_failures_to_revert, 5);
        assert_eq!(watchdog.config().autonomy_level, AutonomyLevel::DryRun);
    }

    #[test]
    fn test_capture_baseline_encodes_snapshot() {
        // This test verifies the encoding path, not the actual system state
        // (SystemObserver::snapshot() will fail in test environments)
        let mut codebook = NixCodebook::new();
        // Manually create a baseline from a mock snapshot
        let snapshot = SystemStateSnapshot {
            services: vec![(
                "test.service".to_string(),
                crate::encoding::ServiceState::Running,
            )],
            ..Default::default()
        };
        let hv = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snapshot)
        };
        assert!(hv.dim() > 0);
        assert!(hv.norm() > 0.0);
    }

    #[test]
    fn test_similar_snapshots_low_surprise() {
        let mut codebook = NixCodebook::new();

        let snap_a = SystemStateSnapshot {
            services: vec![
                (
                    "nginx.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
                (
                    "sshd.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
            ],
            ..Default::default()
        };

        let snap_b = SystemStateSnapshot {
            services: vec![
                (
                    "nginx.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
                (
                    "sshd.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
            ],
            ..Default::default()
        };

        let hv_a = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snap_a)
        };
        let hv_b = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snap_b)
        };

        let surprise = 1.0 - hv_a.similarity(&hv_b) as f64;
        assert!(
            surprise < 0.1,
            "Identical snapshots should have very low surprise, got {}",
            surprise
        );
    }

    #[test]
    fn test_different_snapshots_high_surprise() {
        let mut codebook = NixCodebook::new();

        let snap_a = SystemStateSnapshot {
            services: vec![
                (
                    "nginx.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
                (
                    "sshd.service".to_string(),
                    crate::encoding::ServiceState::Running,
                ),
            ],
            ..Default::default()
        };

        let snap_b = SystemStateSnapshot {
            services: vec![
                (
                    "nginx.service".to_string(),
                    crate::encoding::ServiceState::Failed,
                ),
                (
                    "sshd.service".to_string(),
                    crate::encoding::ServiceState::Failed,
                ),
                (
                    "broken.service".to_string(),
                    crate::encoding::ServiceState::Failed,
                ),
            ],
            ..Default::default()
        };

        let hv_a = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snap_a)
        };
        let hv_b = {
            let mut encoder = SystemStateEncoder::new(&mut codebook);
            encoder.encode_snapshot(&snap_b)
        };

        let similarity = hv_a.similarity(&hv_b);
        let surprise = 1.0 - similarity as f64;
        assert!(
            surprise > 0.0,
            "Different snapshots should have non-zero surprise, got sim={}, surprise={}",
            similarity,
            surprise
        );
    }

    #[test]
    fn test_verdict_display() {
        let verdict = WatchdogVerdict::Stabilized {
            health: vec![],
            duration: Duration::from_secs(120),
            checks_performed: 12,
        };
        assert!(matches!(verdict, WatchdogVerdict::Stabilized { .. }));

        let verdict = WatchdogVerdict::Degraded {
            reason: "test".to_string(),
            surprise: 0.5,
            health: vec![],
            checks_performed: 3,
        };
        assert!(matches!(verdict, WatchdogVerdict::Degraded { .. }));
    }

    #[test]
    fn test_default_autonomy_is_report_only() {
        let config = WatchdogConfig::default();
        assert_eq!(config.autonomy_level, AutonomyLevel::ReportOnly);
    }

    #[test]
    fn test_config_with_autonomy_level() {
        let config = WatchdogConfig {
            autonomy_level: AutonomyLevel::FullAutonomous,
            ..Default::default()
        };
        assert_eq!(config.autonomy_level, AutonomyLevel::FullAutonomous);
    }

    #[test]
    fn test_reverted_verdict_construction() {
        let verdict = WatchdogVerdict::Reverted {
            reason: "degradation detected".to_string(),
            pre_gen: 42,
        };
        match verdict {
            WatchdogVerdict::Reverted { pre_gen, .. } => assert_eq!(pre_gen, 42),
            _ => panic!("Expected Reverted"),
        }
    }

    #[test]
    fn test_dryrun_includes_command() {
        // Verify that DryRun mode produces a degraded verdict with the rollback command
        let config = WatchdogConfig {
            autonomy_level: AutonomyLevel::DryRun,
            ..Default::default()
        };
        // We can't easily run the full monitor in tests, but we can verify
        // the GenerationManager produces the expected command
        let cmd = crate::action::generation_manager::GenerationManager::switch_to(42);
        let (bin, args) = cmd.to_command();
        let cmd_str = format!("{} {}", bin, args.join(" "));
        assert!(
            cmd_str.contains("42"),
            "Command should reference generation 42"
        );
        assert_eq!(config.autonomy_level, AutonomyLevel::DryRun);
    }
}
