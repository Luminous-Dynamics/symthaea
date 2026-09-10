// SPDX-License-Identifier: AGPL-3.0-or-later
use std::collections::{BTreeMap, BTreeSet};
use symthaea_maritime_core::*;

#[derive(Debug, Clone, Copy)]
struct Scenario {
    limit_status: OperationalLimitStatus,
    cooling_available: bool,
    latch_confirmed: bool,
    bay_outer_pressure_qualified: bool,
    return_limits_within: bool,
}

fn limit_assessment(status: OperationalLimitStatus) -> OperationalLimitAssessment {
    OperationalLimitAssessment {
        status,
        evidence_binding: if status == OperationalLimitStatus::Unknown {
            None
        } else {
            Some("evidence:limits".into())
        },
    }
}

fn service_permission(latch_confirmed: bool) -> bool {
    let contract = MaritimeServiceContract {
        contract_id: "contract-1".into(),
        dock_id: "dock-1".into(),
        client_platform_id: "auv-1".into(),
        interface_schema: "maritime-service-v1".into(),
        allowed_services: BTreeSet::from([MaritimeServiceKind::ElectricalPower]),
        evidence_binding: "evidence:contract".into(),
    };
    let observation = DockingObservation {
        dock_id: "dock-1".into(),
        host_platform_id: "tender-1".into(),
        client_platform_id: "auv-1".into(),
        interface_schema: "maritime-service-v1".into(),
        phase: DockingPhase::ServiceReady,
        faulted: false,
        evidence_binding: "evidence:dock".into(),
    };
    let context = ServiceGateContext {
        authenticated_client_id: Some("auv-1".into()),
        mechanical_capture_confirmed: true,
        alignment_confirmed: true,
        latch_confirmed,
        electrical_isolation_confirmed: true,
        local_interlocks_clear: true,
        authority_permitted: true,
    };
    matches!(
        evaluate_service_start(
            &contract,
            &observation,
            &BTreeSet::from([MaritimeServiceKind::ElectricalPower]),
            &context,
        ),
        ServiceStartDecision::Permitted
    )
}

fn resource_permission(cooling_available: bool) -> bool {
    let envelope = MaritimeResourceEnvelope {
        available_power_w: 10_000,
        stored_energy_j: 1_000_000,
        protected_reserve_energy_j: 250_000,
        thermal_rejection_margin_w: 5_000,
        cooling_available,
        evidence_binding: "evidence:resource".into(),
    };
    let request = MaritimeResourceRequest {
        request_id: "service-auv-1".into(),
        power_w: 1_000,
        energy_j: 100_000,
        thermal_load_w: 500,
        priority: ResourcePriority::Service,
    };
    matches!(
        evaluate_resource_request(
            &envelope,
            &request,
            ResourceGateContext {
                authority_permitted: true,
                local_interlocks_clear: true,
            },
        ),
        ResourceDecision::Permitted(_)
    )
}

fn bay_permission(outer_pressure_qualified: bool) -> bool {
    let observation = MaritimeBayObservation {
        bay_id: "bay-1".into(),
        host_platform_id: "tender-1".into(),
        medium: BayMediumState::Flooded,
        pressure_qualification: if outer_pressure_qualified {
            BayPressureQualification::QualifiedForOuterBoundary
        } else {
            BayPressureQualification::Unqualified
        },
        inner_boundary: BoundaryPosition::Closed,
        outer_boundary: BoundaryPosition::Closed,
        occupancy: BayOccupancy::Empty,
        service_isolated: true,
        handling_volume_clear: true,
        faulted: false,
        evidence_binding: "evidence:bay".into(),
    };
    matches!(
        evaluate_bay_boundary_open(
            BayBoundary::Outer,
            &observation,
            &BayBoundaryGateContext {
                authority_permitted: true,
                local_interlocks_clear: true,
                authenticated_occupant_id: None,
            },
        ),
        BayBoundaryDecision::Permitted
    )
}

fn return_to_service_permission(current_limits_within: bool) -> bool {
    let mut stage_evidence = BTreeMap::new();
    for stage in MaintenanceStage::ALL {
        stage_evidence.insert(stage, format!("evidence:{stage:?}"));
    }
    let record = MaintenanceRecord {
        record_id: "maint-1".into(),
        platform_id: "auv-1".into(),
        component_id: "energy-store".into(),
        disposition: MaintenanceDisposition::RepairOnboard,
        stage: MaintenanceStage::Requalified,
        stage_evidence,
    };
    matches!(
        evaluate_return_to_service(
            &record,
            ReturnToServiceGateContext {
                authority_permitted: true,
                local_interlocks_clear: true,
                current_health_acceptable: true,
                current_operational_limits_within: current_limits_within,
                independent_verification_present: true,
            },
        ),
        ReturnToServiceDecision::Permitted
    )
}

fn permission_vector(scenario: Scenario) -> [bool; 5] {
    let operating = matches!(
        evaluate_operating_mode(
            MaritimeOperatingMode::SurfaceTransit,
            &OperatingModeContext {
                baseline_envelope: OperatingEnvelope::Normal,
                limits: limit_assessment(scenario.limit_status),
                authority_permitted: true,
                transition_evidence_present: true,
                local_interlocks_clear: true,
            },
        ),
        OperatingModeDecision::Permitted { .. }
    );

    [
        operating,
        service_permission(scenario.latch_confirmed),
        resource_permission(scenario.cooling_available),
        bay_permission(scenario.bay_outer_pressure_qualified),
        return_to_service_permission(scenario.return_limits_within),
    ]
}

#[test]
fn accumulating_failures_never_restore_a_lost_permission() {
    let scenarios = [
        Scenario {
            limit_status: OperationalLimitStatus::WithinLimits,
            cooling_available: true,
            latch_confirmed: true,
            bay_outer_pressure_qualified: true,
            return_limits_within: true,
        },
        Scenario {
            limit_status: OperationalLimitStatus::Unknown,
            cooling_available: true,
            latch_confirmed: true,
            bay_outer_pressure_qualified: true,
            return_limits_within: false,
        },
        Scenario {
            limit_status: OperationalLimitStatus::Unknown,
            cooling_available: false,
            latch_confirmed: true,
            bay_outer_pressure_qualified: true,
            return_limits_within: false,
        },
        Scenario {
            limit_status: OperationalLimitStatus::Unknown,
            cooling_available: false,
            latch_confirmed: false,
            bay_outer_pressure_qualified: true,
            return_limits_within: false,
        },
        Scenario {
            limit_status: OperationalLimitStatus::Unknown,
            cooling_available: false,
            latch_confirmed: false,
            bay_outer_pressure_qualified: false,
            return_limits_within: false,
        },
    ];

    let mut previous = permission_vector(scenarios[0]);
    assert_eq!(previous, [true; 5]);

    for scenario in scenarios.into_iter().skip(1) {
        let current = permission_vector(scenario);
        for (axis, (was_permitted, is_permitted)) in previous
            .into_iter()
            .zip(current.into_iter())
            .enumerate()
        {
            assert!(
                was_permitted || !is_permitted,
                "permission axis {axis} was restored by an accumulating failure"
            );
        }
        previous = current;
    }

    assert_eq!(previous, [false; 5]);
}

#[test]
fn emergency_safe_state_remains_requestable_after_cross_domain_failures() {
    let decision = evaluate_operating_mode(
        MaritimeOperatingMode::EmergencySafeState,
        &OperatingModeContext {
            baseline_envelope: OperatingEnvelope::FailStop,
            limits: OperationalLimitAssessment {
                status: OperationalLimitStatus::Unknown,
                evidence_binding: None,
            },
            authority_permitted: false,
            transition_evidence_present: false,
            local_interlocks_clear: false,
        },
    );

    assert_eq!(
        decision,
        OperatingModeDecision::Permitted {
            effective_envelope: OperatingEnvelope::FailStop,
        }
    );
}
