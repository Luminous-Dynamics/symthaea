use fusion_decision_compiler_contract::lifecycle_service_ledger::{
    LifecycleEvidenceClass, OutageCategory, ServiceImpact,
};
use fusion_decision_compiler_contract::maintenance_choreography::{
    MaintenancePhase, MaintenanceSchedulePlan, MaintenanceTask, RenewableResource, ResourceDemand,
    exact_small_instance_makespan, generate_maintenance_schedule, validate_maintenance_schedule,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn resource(id: &str, capacity: u32) -> RenewableResource {
    RenewableResource {
        resource_id: id.to_string(),
        capacity,
        evidence_ref: format!("synthetic-resource-evidence-{id}"),
        evidence_class: LifecycleEvidenceClass::Synthetic,
    }
}

fn task(
    id: &str,
    phase: MaintenancePhase,
    duration_seconds: u64,
    predecessors: &[&str],
    resource_ids: &[&str],
    impact: ServiceImpact,
) -> MaintenanceTask {
    MaintenanceTask {
        task_id: id.to_string(),
        phase,
        subsystem_id: format!("subsystem-{id}"),
        failure_mode_id: format!("failure-mode-{id}"),
        duration_seconds,
        predecessor_ids: predecessors.iter().map(|value| (*value).to_string()).collect(),
        resource_demands: resource_ids
            .iter()
            .map(|resource_id| ResourceDemand {
                resource_id: (*resource_id).to_string(),
                units: 1,
            })
            .collect(),
        outage_category: OutageCategory::Planned,
        service_impact: impact,
        task_definition_ref: format!("synthetic-task-definition-{id}"),
        task_definition_sha256: sha('a'),
        evidence_ref: format!("synthetic-task-evidence-{id}"),
        evidence_class: LifecycleEvidenceClass::Synthetic,
    }
}

fn main() {
    let tasks = vec![
        task(
            "isolate-sector-a",
            MaintenancePhase::IsolateAndVerify,
            100,
            &[],
            &["robot-a", "port-a"],
            ServiceImpact::FullOutage,
        ),
        task(
            "isolate-sector-b",
            MaintenancePhase::IsolateAndVerify,
            100,
            &[],
            &["robot-b", "port-b"],
            ServiceImpact::FullOutage,
        ),
        task(
            "transfer-a-to-hot-cell",
            MaintenancePhase::TransferToHotCell,
            40,
            &["isolate-sector-a"],
            &["robot-a", "cask-a"],
            ServiceImpact::FullOutage,
        ),
        task(
            "transfer-b-to-hot-cell",
            MaintenancePhase::TransferToHotCell,
            40,
            &["isolate-sector-b"],
            &["robot-b", "cask-b"],
            ServiceImpact::FullOutage,
        ),
        task(
            "replace-sector-a",
            MaintenancePhase::InsertAndAlign,
            160,
            &["transfer-a-to-hot-cell"],
            &["robot-a", "port-a"],
            ServiceImpact::FullOutage,
        ),
        task(
            "replace-sector-b",
            MaintenancePhase::InsertAndAlign,
            150,
            &["transfer-b-to-hot-cell"],
            &["robot-b", "port-b"],
            ServiceImpact::FullOutage,
        ),
        task(
            "precision-metrology-and-requalification",
            MaintenancePhase::RequalifyAndRestart,
            30,
            &["replace-sector-a", "replace-sector-b"],
            &["metrology"],
            ServiceImpact::FullOutage,
        ),
        task(
            "hot-cell-spare-inspection",
            MaintenancePhase::InspectOrRepair,
            120,
            &[],
            &["hot-cell-bay"],
            ServiceImpact::NoServiceLoss,
        ),
    ];

    let mut priority_order: Vec<String> = tasks.iter().map(|item| item.task_id.clone()).collect();
    priority_order.sort();

    let plan = MaintenanceSchedulePlan {
        campaign_id: "synthetic-parallel-blanket-changeout-v1".to_string(),
        campaign_start_second: 1_000,
        priority_order,
        resources: vec![
            resource("robot-a", 1),
            resource("robot-b", 1),
            resource("port-a", 1),
            resource("port-b", 1),
            resource("cask-a", 1),
            resource("cask-b", 1),
            resource("metrology", 1),
            resource("hot-cell-bay", 1),
        ],
        tasks,
    };

    let report = match generate_maintenance_schedule(&plan) {
        Ok(report) => report,
        Err(errors) => {
            eprintln!("maintenance scheduling refused: {errors:?}");
            std::process::exit(1);
        }
    };

    let validation = validate_maintenance_schedule(&plan, &report);
    if !validation.is_empty() {
        eprintln!("independent schedule validation failed: {validation:?}");
        std::process::exit(1);
    }

    let exact = match exact_small_instance_makespan(&plan) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("small-instance exact oracle failed: {error:?}");
            std::process::exit(1);
        }
    };

    println!("contract_version={}", report.contract_version);
    println!("campaign={}", report.campaign_id);
    println!("task_count={}", report.scheduled_tasks.len());
    println!("makespan_seconds={}", report.makespan_seconds);
    println!(
        "serial_baseline_makespan_seconds={}",
        report.serial_baseline_makespan_seconds
    );
    println!(
        "critical_path_lower_bound_seconds={}",
        report.critical_path_lower_bound_seconds
    );
    println!("exact_small_instance_makespan_seconds={exact}");
    println!(
        "parallel_speedup={}/{}",
        report.parallel_speedup_over_serial.0,
        report.parallel_speedup_over_serial.1
    );
    println!("full_outage_window_seconds={}", report.full_outage_window_seconds);
    println!("lifecycle_event_count={}", report.lifecycle_events.len());

    for scheduled in &report.scheduled_tasks {
        println!(
            "task={} phase={} start={} end={} resources={:?} impact={:?}",
            scheduled.task_id,
            scheduled.phase.id(),
            scheduled.start_second,
            scheduled.end_second,
            scheduled.resource_demands,
            scheduled.service_impact
        );
    }

    println!(
        "SCOPE: synthetic resource-constrained maintenance graph and exact small-instance comparison; no robot, remote-handling or plant maintenance performance is demonstrated"
    );

    if exact > report.makespan_seconds
        || report.makespan_seconds < report.critical_path_lower_bound_seconds
    {
        eprintln!("schedule violates its exact-oracle or critical-path benchmark");
        std::process::exit(1);
    }
}
