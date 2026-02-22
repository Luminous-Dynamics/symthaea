//! Threshold-Based Health Assessment
//!
//! Aggregates data from existing observers (`SystemObserver`, `HardwareObserver`,
//! `StoreObserver`) into actionable health verdicts with recommendations.
//!
//! The existing observers collect raw data; this module interprets it against
//! configurable thresholds and produces a prioritized list of health checks.

use crate::encoding::{ServiceState, SystemStateSnapshot};
use crate::observe::flake_registry::FlakeInfo;
use crate::observe::hardware::HardwareInfo;
use serde::{Deserialize, Serialize};

/// Overall health status (worst-of-all-checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "HEALTHY"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Category of a health issue for grouping and routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    Disk,
    Memory,
    Services,
    Store,
    Flake,
}

impl std::fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disk => write!(f, "disk"),
            Self::Memory => write!(f, "memory"),
            Self::Services => write!(f, "services"),
            Self::Store => write!(f, "store"),
            Self::Flake => write!(f, "flake"),
        }
    }
}

/// A single health check result with status, message, and recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: &'static str,
    pub status: HealthStatus,
    pub message: String,
    pub category: IssueCategory,
    pub recommendations: Vec<String>,
}

/// Configurable thresholds for health assessment.
#[derive(Debug, Clone)]
pub struct HealthAssessor {
    pub disk_warn_pct: f32,
    pub disk_crit_pct: f32,
    pub memory_warn_pct: f32,
    pub memory_crit_pct: f32,
    pub store_warn_paths: u64,
    pub max_failed_services: usize,
    /// Directories to search for flake.nix files.
    pub flake_search_paths: Vec<String>,
    /// Days before flake.lock age triggers a warning.
    pub flake_warn_age_days: u32,
    /// Days before flake.lock age triggers a critical alert.
    pub flake_crit_age_days: u32,
}

impl Default for HealthAssessor {
    fn default() -> Self {
        Self {
            disk_warn_pct: 80.0,
            disk_crit_pct: 90.0,
            memory_warn_pct: 80.0,
            memory_crit_pct: 95.0,
            store_warn_paths: 100_000,
            max_failed_services: 0,
            flake_search_paths: vec!["/etc/nixos".to_string()],
            flake_warn_age_days: 30,
            flake_crit_age_days: 90,
        }
    }
}

impl HealthAssessor {
    /// Run all health checks and return the overall status plus individual results.
    pub fn assess_all(
        &self,
        snapshot: &SystemStateSnapshot,
        hardware: Option<&HardwareInfo>,
    ) -> (HealthStatus, Vec<HealthCheck>) {
        let mut checks = Vec::new();

        if let Some(hw) = hardware {
            checks.push(self.check_disk(hw));
            checks.push(self.check_memory(hw));
        }
        checks.push(self.check_services(snapshot));
        checks.push(self.check_store(snapshot));
        checks.push(self.check_flake_freshness());

        let overall = checks
            .iter()
            .map(|c| c.status)
            .max()
            .unwrap_or(HealthStatus::Healthy);

        (overall, checks)
    }

    /// Check disk usage across all mounted filesystems.
    pub fn check_disk(&self, hardware: &HardwareInfo) -> HealthCheck {
        let mut worst_pct: f32 = 0.0;
        let mut worst_mount = String::new();

        for disk in &hardware.disks {
            if disk.total_bytes == 0 {
                continue;
            }
            let used_pct = (disk.used_bytes as f64 / disk.total_bytes as f64 * 100.0) as f32;
            if used_pct > worst_pct {
                worst_pct = used_pct;
                worst_mount = disk.mount_point.clone();
            }
        }

        let (status, message, recs) = if worst_pct >= self.disk_crit_pct {
            (
                HealthStatus::Critical,
                format!(
                    "{} is {:.1}% full — system may become unresponsive",
                    worst_mount, worst_pct
                ),
                vec![
                    "Run: nix-collect-garbage -d --delete-older-than 7d".to_string(),
                    "Run: nix store optimise".to_string(),
                    "Check for large /tmp or log files".to_string(),
                ],
            )
        } else if worst_pct >= self.disk_warn_pct {
            (
                HealthStatus::Warning,
                format!(
                    "{} is {:.1}% full — consider cleanup",
                    worst_mount, worst_pct
                ),
                vec![
                    "Run: nix-collect-garbage -d --delete-older-than 30d".to_string(),
                    "Run: nix store optimise".to_string(),
                ],
            )
        } else {
            (
                HealthStatus::Healthy,
                format!("All disks OK (worst: {:.1}% on {})", worst_pct, worst_mount),
                vec![],
            )
        };

        HealthCheck {
            name: "disk_usage",
            status,
            message,
            category: IssueCategory::Disk,
            recommendations: recs,
        }
    }

    /// Check memory pressure.
    pub fn check_memory(&self, hardware: &HardwareInfo) -> HealthCheck {
        if hardware.memory_total_mb == 0 {
            return HealthCheck {
                name: "memory",
                status: HealthStatus::Healthy,
                message: "Memory info unavailable".to_string(),
                category: IssueCategory::Memory,
                recommendations: vec![],
            };
        }

        let used_mb = hardware
            .memory_total_mb
            .saturating_sub(hardware.memory_available_mb);
        let used_pct = (used_mb as f64 / hardware.memory_total_mb as f64 * 100.0) as f32;

        let (status, message, recs) = if used_pct >= self.memory_crit_pct {
            (
                HealthStatus::Critical,
                format!(
                    "Memory {:.1}% used ({} MiB / {} MiB) — OOM risk",
                    used_pct, used_mb, hardware.memory_total_mb
                ),
                vec![
                    "Identify high-memory processes: systemd-cgtop".to_string(),
                    "Consider adding swap or increasing RAM".to_string(),
                    "Restart memory-leaking services".to_string(),
                ],
            )
        } else if used_pct >= self.memory_warn_pct {
            (
                HealthStatus::Warning,
                format!(
                    "Memory {:.1}% used ({} MiB / {} MiB)",
                    used_pct, used_mb, hardware.memory_total_mb
                ),
                vec!["Monitor with: systemd-cgtop".to_string()],
            )
        } else {
            (
                HealthStatus::Healthy,
                format!(
                    "Memory OK ({:.1}% used, {} MiB available)",
                    used_pct, hardware.memory_available_mb
                ),
                vec![],
            )
        };

        HealthCheck {
            name: "memory",
            status,
            message,
            category: IssueCategory::Memory,
            recommendations: recs,
        }
    }

    /// Check for failed systemd services.
    pub fn check_services(&self, snapshot: &SystemStateSnapshot) -> HealthCheck {
        let failed: Vec<&str> = snapshot
            .services
            .iter()
            .filter(|(_, state)| *state == ServiceState::Failed)
            .map(|(name, _)| name.as_str())
            .collect();

        let (status, message, recs) = if failed.len() > self.max_failed_services + 2 {
            (
                HealthStatus::Critical,
                format!("{} services failed: {}", failed.len(), failed.join(", ")),
                vec![
                    format!("Check logs: journalctl -u {}", failed[0]),
                    "Review recent changes: nixos-rebuild list-generations".to_string(),
                    "Consider rollback: nixos-rebuild switch --rollback".to_string(),
                ],
            )
        } else if !failed.is_empty() {
            (
                HealthStatus::Warning,
                format!("{} service(s) failed: {}", failed.len(), failed.join(", ")),
                failed
                    .iter()
                    .map(|svc| format!("Check: journalctl -u {} --no-pager -n 20", svc))
                    .collect(),
            )
        } else {
            (
                HealthStatus::Healthy,
                format!("{} services running, none failed", snapshot.services.len()),
                vec![],
            )
        };

        HealthCheck {
            name: "services",
            status,
            message,
            category: IssueCategory::Services,
            recommendations: recs,
        }
    }

    /// Check flake.lock freshness by discovering flakes and comparing lock age.
    pub fn check_flake_freshness(&self) -> HealthCheck {
        let search_refs: Vec<&str> = self.flake_search_paths.iter().map(|s| s.as_str()).collect();
        let flakes = crate::observe::flake_registry::FlakeRegistry::discover(&search_refs)
            .unwrap_or_default();
        self.check_flake_freshness_from_infos(&flakes)
    }

    /// Testable helper: check flake freshness from pre-collected FlakeInfo.
    pub fn check_flake_freshness_from_infos(&self, flakes: &[FlakeInfo]) -> HealthCheck {
        if flakes.is_empty() {
            return HealthCheck {
                name: "flake_freshness",
                status: HealthStatus::Healthy,
                message: "No flakes discovered".to_string(),
                category: IssueCategory::Flake,
                recommendations: vec![],
            };
        }

        let now = chrono::Utc::now();
        let mut worst_age_days: i64 = 0;
        let mut worst_path = String::new();

        for flake in flakes {
            if let Some(ref modified_str) = flake.last_modified {
                if let Ok(modified) =
                    chrono::NaiveDateTime::parse_from_str(modified_str, "%Y-%m-%d %H:%M:%S UTC")
                {
                    let modified_utc = modified.and_utc();
                    let age_days = (now - modified_utc).num_days();
                    if age_days > worst_age_days {
                        worst_age_days = age_days;
                        worst_path = flake.path.clone();
                    }
                }
            }
        }

        let (status, message, recs) = if worst_age_days >= self.flake_crit_age_days as i64 {
            (
                HealthStatus::Critical,
                format!(
                    "Flake lock at {} is {} days old — severely outdated",
                    worst_path, worst_age_days
                ),
                vec![
                    format!("Run: nix flake update --flake {}", worst_path),
                    "Review changelogs before updating".to_string(),
                ],
            )
        } else if worst_age_days >= self.flake_warn_age_days as i64 {
            (
                HealthStatus::Warning,
                format!(
                    "Flake lock at {} is {} days old — consider updating",
                    worst_path, worst_age_days
                ),
                vec![format!("Run: nix flake update --flake {}", worst_path)],
            )
        } else {
            (
                HealthStatus::Healthy,
                format!(
                    "All flake locks fresh (worst: {} days at {})",
                    worst_age_days, worst_path
                ),
                vec![],
            )
        };

        HealthCheck {
            name: "flake_freshness",
            status,
            message,
            category: IssueCategory::Flake,
            recommendations: recs,
        }
    }

    /// Check nix store health.
    pub fn check_store(&self, snapshot: &SystemStateSnapshot) -> HealthCheck {
        let path_count = snapshot.store_path_count.unwrap_or(0) as u64;

        let (status, message, recs) = if path_count > self.store_warn_paths * 2 {
            (
                HealthStatus::Critical,
                format!(
                    "Nix store has {} paths — extremely bloated",
                    path_count
                ),
                vec![
                    "Run: nix-collect-garbage -d".to_string(),
                    "Delete old generations: nix-env --delete-generations +5 -p /nix/var/nix/profiles/system".to_string(),
                    "Run: nix store optimise".to_string(),
                ],
            )
        } else if path_count > self.store_warn_paths {
            (
                HealthStatus::Warning,
                format!("Nix store has {} paths — consider cleanup", path_count),
                vec!["Run: nix-collect-garbage -d --delete-older-than 30d".to_string()],
            )
        } else {
            let size_str = snapshot
                .store_size_bytes
                .map(|b| format!(", {:.1} GiB", b as f64 / 1_073_741_824.0))
                .unwrap_or_default();
            (
                HealthStatus::Healthy,
                format!("Nix store OK ({} paths{})", path_count, size_str),
                vec![],
            )
        };

        HealthCheck {
            name: "store",
            status,
            message,
            category: IssueCategory::Store,
            recommendations: recs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::hardware::DiskInfo;

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
        }
    }

    #[test]
    fn test_healthy_system() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(
            vec![
                ("nginx.service", ServiceState::Running),
                ("sshd.service", ServiceState::Running),
            ],
            Some(50_000),
        );
        let hw = make_hardware(50.0, 40.0);
        let (overall, checks) = assessor.assess_all(&snapshot, Some(&hw));

        assert_eq!(overall, HealthStatus::Healthy);
        assert!(checks.iter().all(|c| c.status == HealthStatus::Healthy));
    }

    #[test]
    fn test_disk_warning() {
        let assessor = HealthAssessor::default();
        let hw = make_hardware(85.0, 40.0);
        let check = assessor.check_disk(&hw);

        assert_eq!(check.status, HealthStatus::Warning);
        assert!(check.message.contains("85.0%"));
        assert!(!check.recommendations.is_empty());
    }

    #[test]
    fn test_disk_critical() {
        let assessor = HealthAssessor::default();
        let hw = make_hardware(95.0, 40.0);
        let check = assessor.check_disk(&hw);

        assert_eq!(check.status, HealthStatus::Critical);
    }

    #[test]
    fn test_memory_warning() {
        let assessor = HealthAssessor::default();
        let hw = make_hardware(50.0, 85.0);
        let check = assessor.check_memory(&hw);

        assert_eq!(check.status, HealthStatus::Warning);
    }

    #[test]
    fn test_memory_critical() {
        let assessor = HealthAssessor::default();
        let hw = make_hardware(50.0, 97.0);
        let check = assessor.check_memory(&hw);

        assert_eq!(check.status, HealthStatus::Critical);
        assert!(check.message.contains("OOM"));
    }

    #[test]
    fn test_failed_services_warning() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(
            vec![
                ("nginx.service", ServiceState::Running),
                ("broken.service", ServiceState::Failed),
            ],
            None,
        );
        let check = assessor.check_services(&snapshot);

        assert_eq!(check.status, HealthStatus::Warning);
        assert!(check.message.contains("broken.service"));
    }

    #[test]
    fn test_many_failed_services_critical() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(
            vec![
                ("a.service", ServiceState::Failed),
                ("b.service", ServiceState::Failed),
                ("c.service", ServiceState::Failed),
                ("d.service", ServiceState::Failed),
            ],
            None,
        );
        let check = assessor.check_services(&snapshot);

        assert_eq!(check.status, HealthStatus::Critical);
    }

    #[test]
    fn test_store_warning() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(vec![], Some(120_000));
        let check = assessor.check_store(&snapshot);

        assert_eq!(check.status, HealthStatus::Warning);
    }

    #[test]
    fn test_store_critical() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(vec![], Some(250_000));
        let check = assessor.check_store(&snapshot);

        assert_eq!(check.status, HealthStatus::Critical);
    }

    #[test]
    fn test_overall_is_worst() {
        let assessor = HealthAssessor::default();
        // Healthy disk/memory, but failed services
        let snapshot = make_snapshot(
            vec![
                ("a.service", ServiceState::Failed),
                ("b.service", ServiceState::Failed),
                ("c.service", ServiceState::Failed),
                ("d.service", ServiceState::Failed),
            ],
            Some(1000),
        );
        let hw = make_hardware(50.0, 40.0);
        let (overall, _) = assessor.assess_all(&snapshot, Some(&hw));

        assert_eq!(overall, HealthStatus::Critical);
    }

    #[test]
    fn test_no_hardware_info() {
        let assessor = HealthAssessor::default();
        let snapshot = make_snapshot(vec![], Some(1000));
        let (overall, checks) = assessor.assess_all(&snapshot, None);

        // Should still produce service, store, and flake checks
        assert_eq!(overall, HealthStatus::Healthy);
        assert_eq!(checks.len(), 3); // services + store + flake
    }

    #[test]
    fn test_custom_thresholds() {
        let assessor = HealthAssessor {
            disk_warn_pct: 50.0,
            disk_crit_pct: 70.0,
            ..Default::default()
        };
        let hw = make_hardware(60.0, 40.0);
        let check = assessor.check_disk(&hw);

        assert_eq!(check.status, HealthStatus::Warning);
    }

    #[test]
    fn test_flake_freshness_no_flakes() {
        let assessor = HealthAssessor::default();
        let check = assessor.check_flake_freshness_from_infos(&[]);
        assert_eq!(check.status, HealthStatus::Healthy);
        assert_eq!(check.name, "flake_freshness");
    }

    #[test]
    fn test_flake_freshness_fresh_lock() {
        let assessor = HealthAssessor::default();
        let now = chrono::Utc::now();
        let fresh_ts = now.format("%Y-%m-%d %H:%M:%S UTC").to_string();
        let flakes = vec![FlakeInfo {
            path: "/etc/nixos".to_string(),
            description: None,
            inputs: vec![],
            last_modified: Some(fresh_ts),
        }];
        let check = assessor.check_flake_freshness_from_infos(&flakes);
        assert_eq!(check.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_flake_freshness_stale_lock_warning() {
        let assessor = HealthAssessor::default();
        let stale = chrono::Utc::now() - chrono::Duration::days(45);
        let stale_ts = stale.format("%Y-%m-%d %H:%M:%S UTC").to_string();
        let flakes = vec![FlakeInfo {
            path: "/etc/nixos".to_string(),
            description: None,
            inputs: vec![],
            last_modified: Some(stale_ts),
        }];
        let check = assessor.check_flake_freshness_from_infos(&flakes);
        assert_eq!(check.status, HealthStatus::Warning);
        assert!(!check.recommendations.is_empty());
    }

    #[test]
    fn test_flake_freshness_very_stale_critical() {
        let assessor = HealthAssessor::default();
        let old = chrono::Utc::now() - chrono::Duration::days(100);
        let old_ts = old.format("%Y-%m-%d %H:%M:%S UTC").to_string();
        let flakes = vec![FlakeInfo {
            path: "/etc/nixos".to_string(),
            description: None,
            inputs: vec![],
            last_modified: Some(old_ts),
        }];
        let check = assessor.check_flake_freshness_from_infos(&flakes);
        assert_eq!(check.status, HealthStatus::Critical);
        assert!(check.message.contains("100"));
    }
}
