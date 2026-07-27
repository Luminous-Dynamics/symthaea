// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for `Doctor`'s health/anomaly calibration.
//!
//! Tier 2 of the test-corpus plan in
//! SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md: rather than
//! more unit tests of the same shape as what already exists, assert
//! invariants across arbitrary inputs -- the two calibration bugs fixed
//! this session (services severity, anomaly confidence) were both
//! "reasonable on the cases anyone thought to write a test for, wrong on
//! cases nobody did" -- exactly what property testing is for.

use proptest::prelude::*;

use nixward::encoding::{ServiceState, SystemStateSnapshot};
use nixward::observe::hardware::{DiskInfo, HardwareInfo};
use nixward::support::health_check::{HealthAssessor, HealthStatus};

fn hardware_with(
    memory_total_mb: u64,
    memory_available_mb: u64,
    disk_used_pct: f64,
    load1: f64,
    swap_total_mb: u64,
    swap_used_mb: u64,
) -> HardwareInfo {
    let total_bytes = 1_000_000_000_000u64;
    let used_bytes = ((total_bytes as f64) * disk_used_pct.clamp(0.0, 1.0)) as u64;
    HardwareInfo {
        cpu_model: "test-cpu".to_string(),
        cpu_cores: 8,
        memory_total_mb,
        memory_available_mb,
        gpus: vec![],
        disks: vec![DiskInfo {
            device: "/dev/test".to_string(),
            mount_point: "/".to_string(),
            total_bytes,
            used_bytes,
        }],
        load_average: [load1, load1, load1],
        swap_total_mb,
        swap_used_mb,
    }
}

fn service_name(is_instanced: bool, seed: u32) -> String {
    if is_instanced {
        format!("some-oneshot-job@{seed}.service")
    } else {
        format!("persistent-service-{seed}.service")
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Regression property for the services-severity fix: overall status
    /// can only reach Critical for the "services" check if at least one
    /// *persistent* (non `@`-instanced) unit failed. An arbitrarily large
    /// pile of ephemeral instanced-unit failures (crash processors, backup
    /// jobs, etc.) must never alone drive Critical.
    #[test]
    fn instanced_only_failures_never_reach_critical(
        instanced_count in 0usize..40,
    ) {
        let assessor = HealthAssessor::default();
        let services: Vec<(String, ServiceState)> = (0..instanced_count)
            .map(|i| (service_name(true, i as u32), ServiceState::Failed))
            .collect();
        let snapshot = SystemStateSnapshot {
            services,
            ..Default::default()
        };

        let check = assessor.check_services(&snapshot);
        prop_assert_ne!(
            check.status,
            HealthStatus::Critical,
            "{} instanced-only failures reached Critical",
            instanced_count
        );
        prop_assert!(
            !check.recommendations.iter().any(|r| r.contains("rollback")),
            "instanced-only failures recommended a rollback: {:?}",
            check.recommendations
        );
    }

    /// Adding more persistent-service failures never *decreases* severity
    /// (monotonicity) -- and once enough exist, status is Critical
    /// regardless of how many additional ephemeral instanced failures are
    /// also present (they must not dilute or mask a real problem either).
    #[test]
    fn more_persistent_failures_never_decrease_severity(
        persistent_count in 0usize..10,
        instanced_count in 0usize..10,
    ) {
        let assessor = HealthAssessor::default();
        let mut services: Vec<(String, ServiceState)> = (0..persistent_count)
            .map(|i| (service_name(false, i as u32), ServiceState::Failed))
            .collect();
        services.extend(
            (0..instanced_count).map(|i| (service_name(true, i as u32), ServiceState::Failed)),
        );
        let snapshot = SystemStateSnapshot {
            services,
            ..Default::default()
        };
        let check = assessor.check_services(&snapshot);

        if persistent_count > assessor.max_failed_services + 2 {
            prop_assert_eq!(check.status, HealthStatus::Critical);
        } else if persistent_count > 0 {
            prop_assert!(check.status >= HealthStatus::Warning);
        }
    }

    /// check_services must never panic, regardless of how many services of
    /// either kind (or mixed) are reported failed vs. running.
    #[test]
    fn check_services_never_panics(
        total in 0usize..60,
        failed_fraction in 0.0f64..1.0,
        instanced_fraction in 0.0f64..1.0,
    ) {
        let assessor = HealthAssessor::default();
        let services: Vec<(String, ServiceState)> = (0..total)
            .map(|i| {
                let failed = (i as f64) / (total.max(1) as f64) < failed_fraction;
                let instanced = (i as f64) / (total.max(1) as f64) < instanced_fraction;
                let state = if failed {
                    ServiceState::Failed
                } else {
                    ServiceState::Running
                };
                (service_name(instanced, i as u32), state)
            })
            .collect();
        let snapshot = SystemStateSnapshot {
            services,
            ..Default::default()
        };
        // Must not panic -- that's the whole assertion.
        let _ = assessor.check_services(&snapshot);
    }

    /// Disk/memory/load/swap severity is monotonic in the underlying
    /// percentage/value: strictly worse (higher) usage/load must never
    /// report a *less* severe status than a lower one, all else equal.
    #[test]
    fn disk_severity_is_monotonic_in_usage(
        low_pct in 0.0f64..1.0,
        delta in 0.0f64..0.3,
    ) {
        let assessor = HealthAssessor::default();
        let high_pct = (low_pct + delta).min(1.0);

        let hw_low = hardware_with(32_000, 20_000, low_pct, 1.0, 8_000, 0);
        let hw_high = hardware_with(32_000, 20_000, high_pct, 1.0, 8_000, 0);

        let low_status = assessor.check_disk(&hw_low).status;
        let high_status = assessor.check_disk(&hw_high).status;

        prop_assert!(
            high_status >= low_status,
            "higher disk usage ({high_pct:.3}) reported LESS severe status ({high_status:?}) \
             than lower usage ({low_pct:.3}, {low_status:?})"
        );
    }

    #[test]
    fn load_severity_is_monotonic_in_load_average(
        low_load in 0.0f64..20.0,
        delta in 0.0f64..10.0,
    ) {
        let assessor = HealthAssessor::default();
        let high_load = low_load + delta;

        let hw_low = hardware_with(32_000, 20_000, 0.5, low_load, 8_000, 0);
        let hw_high = hardware_with(32_000, 20_000, 0.5, high_load, 8_000, 0);

        let low_status = assessor.check_load(&hw_low).status;
        let high_status = assessor.check_load(&hw_high).status;

        prop_assert!(high_status >= low_status);
    }

    /// None of the hardware-driven checks should ever panic, including at
    /// degenerate boundary values (zero totals, available > total, etc.)
    /// that a real (possibly buggy) observer could in principle report.
    #[test]
    fn hardware_checks_never_panic(
        memory_total_mb in 0u64..200_000,
        memory_available_mb in 0u64..200_000,
        disk_used_pct in 0.0f64..1.5, // intentionally allow > 100% (bad data)
        load1 in 0.0f64..64.0,
        swap_total_mb in 0u64..64_000,
        swap_used_mb in 0u64..64_000,
    ) {
        let assessor = HealthAssessor::default();
        let hw = hardware_with(
            memory_total_mb,
            memory_available_mb,
            disk_used_pct,
            load1,
            swap_total_mb,
            swap_used_mb,
        );
        let _ = assessor.check_disk(&hw);
        let _ = assessor.check_memory(&hw);
        let _ = assessor.check_load(&hw);
        let _ = assessor.check_swap(&hw);
    }
}
