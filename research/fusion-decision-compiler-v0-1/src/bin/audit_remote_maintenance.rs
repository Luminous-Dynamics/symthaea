use fusion_decision_compiler_contract::remote_maintenance_compiler::{
    IntegrityTestKind, InterfaceFamily, InterfaceProcedureBinding, MaintenanceAction,
    MaintenanceEvidenceClass, MaintenanceOperation, MaintenanceResource, MaintenanceTool,
    RemoteMaintenancePlan, ReplaceableAssembly, ServiceInterface, compile_remote_maintenance_plan,
};

fn operation(
    id: &str,
    action: MaintenanceAction,
    interface: Option<&str>,
    predecessors: &[&str],
    start: u64,
    end: u64,
) -> MaintenanceOperation {
    MaintenanceOperation {
        operation_id: id.to_string(),
        action,
        interface_id: interface.map(str::to_string),
        predecessor_ids: predecessors
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        required_capabilities: vec!["remote-maintenance".to_string()],
        resource_id: "remote-handler-1".to_string(),
        tool_id: "multi-service-tool-1".to_string(),
        start_second: start,
        end_second: end,
        minimum_duration_seconds: end - start,
        nominal_duration_seconds: end - start,
        maximum_duration_seconds: end - start,
        procedure_step_ref: format!("procedure-step-{id}"),
        evidence_ref: format!("synthetic-operation-evidence-{id}"),
        evidence_sha256: "a".repeat(64),
        evidence_class: MaintenanceEvidenceClass::Synthetic,
        clearance_reachability_ref: format!("synthetic-reachability-{id}"),
        clearance_reachability_sha256: "b".repeat(64),
        kinematic_model_ref: "synthetic-kinematic-model-v1".to_string(),
        validity_domain_ref: "synthetic-mockup-envelope-v1".to_string(),
        failure_recovery_ref: format!("synthetic-recovery-{id}"),
    }
}

fn interface(
    id: &str,
    mate: &str,
    assembly_side: bool,
    family: InterfaceFamily,
    compatibility: &str,
) -> ServiceInterface {
    ServiceInterface {
        interface_id: id.to_string(),
        mate_interface_id: mate.to_string(),
        is_assembly_side: assembly_side,
        family,
        compatibility_key: compatibility.to_string(),
        access_side_ref: format!("synthetic-access-{id}"),
        remote_actuation_capability: "remote-maintenance".to_string(),
        remote_actuation_proof_ref: format!("synthetic-actuation-proof-{id}"),
    }
}

fn main() {
    let plan = RemoteMaintenancePlan {
        campaign_id: "synthetic-blanket-module-changeout-v1".to_string(),
        procedure_revision: "synthetic-procedure-r1".to_string(),
        service_boundary_ref: "synthetic-plant-service-boundary-v1".to_string(),
        outage_start_second: 0,
        outage_end_second: 300,
        assembly: ReplaceableAssembly {
            assembly_id: "synthetic-blanket-module-7".to_string(),
            revision: "synthetic-module-r4".to_string(),
            mass_kg: 1200.0,
            centre_of_gravity_x_m: 0.1,
            centre_of_gravity_y_m: 0.1,
            centre_of_gravity_z_m: 0.1,
            removal_envelope_x_m: 2.0,
            removal_envelope_y_m: 1.5,
            removal_envelope_z_m: 1.2,
            handling_datum_ref: "synthetic-handling-datum".to_string(),
            removal_envelope_ref: "synthetic-removal-envelope".to_string(),
            shielded_cask_envelope_ref: "synthetic-shielded-cask-route".to_string(),
            hot_cell_interface_ref: "synthetic-hot-cell-interface".to_string(),
            hazard_profile_ref: "synthetic-activation-tritium-hazard-profile".to_string(),
            isolation_plan_ref: "synthetic-coolant-electrical-isolation-plan".to_string(),
        },
        interfaces: vec![
            interface(
                "coolant-module",
                "coolant-plant",
                true,
                InterfaceFamily::Coolant,
                "DN200-PN40",
            ),
            interface(
                "coolant-plant",
                "coolant-module",
                false,
                InterfaceFamily::Coolant,
                "DN200-PN40",
            ),
            interface(
                "power-module",
                "power-plant",
                true,
                InterfaceFamily::Electrical,
                "HV-remote-plug",
            ),
            interface(
                "power-plant",
                "power-module",
                false,
                InterfaceFamily::Electrical,
                "HV-remote-plug",
            ),
        ],
        interface_bindings: vec![
            InterfaceProcedureBinding {
                interface_id: "coolant-module".to_string(),
                isolation_operation_id: "isolate".to_string(),
                prove_safe_operation_id: "prove-safe".to_string(),
                preparation_operation_id: Some("purge-coolant".to_string()),
                disconnect_operation_id: "disconnect-coolant".to_string(),
                reconnect_operation_id: "reconnect-coolant".to_string(),
                integrity_test_operation_id: "pressure-test-coolant".to_string(),
            },
            InterfaceProcedureBinding {
                interface_id: "power-module".to_string(),
                isolation_operation_id: "isolate".to_string(),
                prove_safe_operation_id: "prove-safe".to_string(),
                preparation_operation_id: None,
                disconnect_operation_id: "disconnect-power".to_string(),
                reconnect_operation_id: "reconnect-power".to_string(),
                integrity_test_operation_id: "continuity-test-power".to_string(),
            },
        ],
        resources: vec![MaintenanceResource {
            resource_id: "remote-handler-1".to_string(),
            capabilities: vec!["remote-maintenance".to_string()],
        }],
        tools: vec![MaintenanceTool {
            tool_id: "multi-service-tool-1".to_string(),
            capabilities: vec!["remote-maintenance".to_string()],
            maximum_payload_kg: 2000.0,
            maximum_centre_of_gravity_offset_m: 0.5,
        }],
        operations: vec![
            operation(
                "isolate",
                MaintenanceAction::IsolateSystem,
                None,
                &[],
                0,
                10,
            ),
            operation(
                "prove-safe",
                MaintenanceAction::ProveSafeState,
                None,
                &["isolate"],
                10,
                20,
            ),
            operation(
                "purge-coolant",
                MaintenanceAction::PrepareServiceBoundary,
                None,
                &["prove-safe"],
                20,
                30,
            ),
            operation(
                "disconnect-coolant",
                MaintenanceAction::DisconnectInterface,
                Some("coolant-module"),
                &["prove-safe", "purge-coolant"],
                30,
                40,
            ),
            operation(
                "disconnect-power",
                MaintenanceAction::DisconnectInterface,
                Some("power-module"),
                &["prove-safe"],
                40,
                50,
            ),
            operation(
                "release-restraints",
                MaintenanceAction::ReleaseRestraints,
                None,
                &["disconnect-coolant", "disconnect-power"],
                50,
                60,
            ),
            operation(
                "extract",
                MaintenanceAction::ExtractModule,
                None,
                &["release-restraints"],
                60,
                80,
            ),
            operation(
                "transfer-cask",
                MaintenanceAction::TransferToShieldedCask,
                None,
                &["extract"],
                80,
                100,
            ),
            operation(
                "hot-cell-inspect",
                MaintenanceAction::HotCellInspectRepair,
                None,
                &["transfer-cask"],
                100,
                150,
            ),
            operation(
                "install",
                MaintenanceAction::InstallModule,
                None,
                &["hot-cell-inspect"],
                150,
                170,
            ),
            operation(
                "align",
                MaintenanceAction::AlignMetrology,
                None,
                &["install"],
                170,
                180,
            ),
            operation(
                "secure-restraints",
                MaintenanceAction::SecureRestraints,
                None,
                &["align"],
                180,
                190,
            ),
            operation(
                "reconnect-coolant",
                MaintenanceAction::ReconnectInterface,
                Some("coolant-module"),
                &["secure-restraints"],
                190,
                200,
            ),
            operation(
                "reconnect-power",
                MaintenanceAction::ReconnectInterface,
                Some("power-module"),
                &["secure-restraints"],
                200,
                210,
            ),
            operation(
                "pressure-test-coolant",
                MaintenanceAction::IntegrityTest(IntegrityTestKind::PressureLeak),
                Some("coolant-module"),
                &["reconnect-coolant"],
                210,
                220,
            ),
            operation(
                "continuity-test-power",
                MaintenanceAction::IntegrityTest(IntegrityTestKind::ElectricalContinuity),
                Some("power-module"),
                &["reconnect-power"],
                220,
                230,
            ),
            operation(
                "system-requalification",
                MaintenanceAction::SystemRequalification,
                None,
                &["pressure-test-coolant", "continuity-test-power"],
                230,
                240,
            ),
            operation(
                "restart",
                MaintenanceAction::RestartSystem,
                None,
                &["system-requalification"],
                240,
                250,
            ),
        ],
    };

    let report = compile_remote_maintenance_plan(&plan);
    println!("contract_version={}", report.contract_version);
    println!("campaign={}", report.campaign_id);
    println!("assembly={}", report.assembly_id);
    println!("disposition={:?}", report.disposition);
    println!("operation_count={}", report.operation_count);
    println!("interface_count={}", report.interface_count);
    println!(
        "scheduled_outage_seconds={}",
        report.scheduled_outage_seconds.unwrap_or_default()
    );
    println!(
        "critical_path_high_seconds={}",
        report.critical_path_high_seconds.unwrap_or_default()
    );
    for item in &report.scheduled_resource_utilization {
        println!(
            "resource={},busy_seconds={},idle_seconds={},operations={}",
            item.resource_id,
            item.busy_seconds,
            item.idle_seconds_within_campaign,
            item.operation_count
        );
    }
    for item in &report.evidence_class_counts {
        println!(
            "evidence_class={:?},operations={}",
            item.evidence_class, item.operation_count
        );
    }
    println!(
        "SCOPE: synthetic declared replacement procedure only; no physical reachability, activated maintenance or repair-time validation"
    );

    if report.disposition
        != fusion_decision_compiler_contract::remote_maintenance_compiler::DfrmDisposition::ValidDeclaredProcedure
    {
        eprintln!("synthetic DFRM procedure did not reconcile");
        for diagnostic in report.diagnostics {
            eprintln!("diagnostic={diagnostic:?}");
        }
        std::process::exit(1);
    }
}
