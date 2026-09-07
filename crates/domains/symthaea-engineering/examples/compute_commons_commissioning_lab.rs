// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compute Commons commissioning-drift laboratory.
//!
//! Demonstrates that local survival authority is configuration-bound: a matching
//! commissioned image can operate with or without parent coordination, while an
//! uncommissioned plant/software configuration fails closed in both cases.

use chrono::{Duration, TimeZone, Utc};
use std::collections::BTreeSet;
use std::error::Error;
use symthaea_operating_authority::{LeaseAdmission, OperatingAuthorityRegistry};
use symthaea_operating_autonomy::{CoordinationMode, LocalSafetyEnvelope};
use symthaea_operating_commissioning::{
    CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningError,
    ConfigurationDigest, evaluate_commissioned_operation,
};
use symthaea_operating_envelope::{
    BoundaryMetric, ObservedResourceMetric, OperatingEnvelope, OperatingMode,
    OperatingPoint, ResourceConstraint,
};
use symthaea_resource_hierarchy::{NodeRole, NodeScale, ResourceHierarchy, ResourceNode};
use symthaea_resource_model::{
    ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
};

fn digest(byte: u8) -> ConfigurationDigest {
    ConfigurationDigest::Blake3_256([byte; 32])
}

fn watts(value: f64) -> ResourceAmount {
    ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
}

fn hierarchy() -> ResourceHierarchy {
    let mut hierarchy = ResourceHierarchy::default();
    hierarchy
        .insert_root(
            ResourceNode::new(
                "commons",
                "Community Compute Commons",
                NodeScale::Community,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::CommunityService),
        )
        .unwrap();
    hierarchy
        .insert_child(
            "commons",
            ResourceNode::new(
                "compute-campus",
                "Compute Campus",
                NodeScale::Facility,
                ResourceEnvelope::default(),
            )
            .with_role(NodeRole::Compute),
        )
        .unwrap();
    hierarchy
}

fn commissioned_local() -> CommissionedLocalSafetyEnvelope {
    let mut allowed_modes = BTreeSet::new();
    allowed_modes.insert(OperatingMode::Normal);
    allowed_modes.insert(OperatingMode::Islanded);
    let local = LocalSafetyEnvelope {
        id: "compute-campus-survival-v1".into(),
        subject_node_id: "compute-campus".into(),
        allowed_modes,
        resource_constraints: vec![ResourceConstraint {
            key: watts(0.0).key,
            metric: BoundaryMetric::Import,
            minimum: None,
            maximum: Some(250_000.0),
        }],
        critical_service_floor: 0.80,
        max_shed_fraction: 0.20,
    };
    CommissionedLocalSafetyEnvelope::new(
        local,
        CommissioningBinding::new(
            digest(0x42),
            "compute-commons/commissioning/compute-campus/v1",
        )
        .unwrap(),
    )
}

fn parent_envelope(t0: chrono::DateTime<Utc>) -> OperatingEnvelope {
    let mut allowed_modes = BTreeSet::new();
    allowed_modes.insert(OperatingMode::Normal);
    allowed_modes.insert(OperatingMode::Islanded);
    OperatingEnvelope {
        id: "commons-compute-g1".into(),
        issuer_node_id: "commons".into(),
        subject_node_id: "compute-campus".into(),
        generation: 1,
        valid_from: t0,
        valid_until: t0 + Duration::hours(1),
        allowed_modes,
        resource_constraints: vec![ResourceConstraint {
            key: watts(0.0).key,
            metric: BoundaryMetric::Import,
            minimum: None,
            maximum: Some(180_000.0),
        }],
        critical_service_floor: 0.85,
        max_shed_fraction: 0.15,
    }
}

fn point(import_w: f64) -> OperatingPoint {
    OperatingPoint {
        mode: OperatingMode::Normal,
        critical_service_fraction: 0.95,
        shed_fraction: 0.05,
        resources: vec![ObservedResourceMetric {
            key: watts(0.0).key,
            metric: BoundaryMetric::Import,
            value: import_w,
        }],
    }
}

#[derive(Debug)]
struct CommissioningLabReport {
    parent_coordination_seen: bool,
    drift_blocked_with_parent: bool,
    local_only_after_parent_expiry: bool,
    drift_blocked_after_parent_expiry: bool,
}

fn run_commissioning_lab() -> Result<CommissioningLabReport, Box<dyn Error>> {
    let hierarchy = hierarchy();
    let commissioned = commissioned_local();
    let current = digest(0x42);
    let drifted = digest(0x99);
    let t0 = Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap();

    let mut authority = OperatingAuthorityRegistry::new();
    assert_eq!(
        authority.admit(
            &hierarchy,
            parent_envelope(t0),
            t0 + Duration::minutes(1),
        )?,
        LeaseAdmission::Accepted
    );

    let coordinated = evaluate_commissioned_operation(
        &commissioned,
        current,
        &hierarchy,
        &authority,
        &point(170_000.0),
        t0 + Duration::minutes(10),
    )?;
    let parent_coordination_seen = matches!(
        coordinated.mode,
        CoordinationMode::ParentCoordinated { .. }
    );
    assert!(coordinated.compliant);
    assert!(parent_coordination_seen);

    let drift_with_parent = evaluate_commissioned_operation(
        &commissioned,
        drifted,
        &hierarchy,
        &authority,
        &point(170_000.0),
        t0 + Duration::minutes(10),
    );
    let drift_blocked_with_parent = matches!(
        drift_with_parent,
        Err(CommissioningError::ConfigurationDrift { .. })
    );
    assert!(drift_blocked_with_parent);

    let local_only = evaluate_commissioned_operation(
        &commissioned,
        current,
        &hierarchy,
        &authority,
        &point(170_000.0),
        t0 + Duration::hours(2),
    )?;
    let local_only_after_parent_expiry = local_only.mode == CoordinationMode::LocalOnly;
    assert!(local_only.compliant);
    assert!(local_only_after_parent_expiry);

    let drift_after_expiry = evaluate_commissioned_operation(
        &commissioned,
        drifted,
        &hierarchy,
        &authority,
        &point(170_000.0),
        t0 + Duration::hours(2),
    );
    let drift_blocked_after_parent_expiry = matches!(
        drift_after_expiry,
        Err(CommissioningError::ConfigurationDrift { .. })
    );
    assert!(drift_blocked_after_parent_expiry);

    Ok(CommissioningLabReport {
        parent_coordination_seen,
        drift_blocked_with_parent,
        local_only_after_parent_expiry,
        drift_blocked_after_parent_expiry,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let report = run_commissioning_lab()?;
    println!("Compute Commons commissioning report: {report:#?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commissioning_lab_executes_all_invariants() {
        let report = run_commissioning_lab().unwrap();
        assert!(report.parent_coordination_seen);
        assert!(report.drift_blocked_with_parent);
        assert!(report.local_only_after_parent_expiry);
        assert!(report.drift_blocked_after_parent_expiry);
    }
}
