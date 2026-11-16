use fusion_decision_compiler_contract::remote_handling_benchmark::{
    BenchmarkEvidenceClass, OperationTrialEvent, OperationTrialOutcome,
    RemoteHandlingBenchmarkManifest, RemoteHandlingTrial, TrialStatus,
    summarize_remote_handling_benchmark,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn raw_digest(id: &str) -> String {
    let hash = id.bytes().fold(0xcbf29ce484222325_u64, |state, byte| {
        (state ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{hash:064x}")
}

fn manifest() -> RemoteHandlingBenchmarkManifest {
    RemoteHandlingBenchmarkManifest {
        benchmark_id: "synthetic-blanket-remote-handling-v1".to_string(),
        profile_id: "synthetic-blanket-module-profile-v1".to_string(),
        profile_sha256: sha('a'),
        assembly_revision: "synthetic-assembly-r1".to_string(),
        geometry_sha256: sha('b'),
        interface_manifest_sha256: sha('c'),
        procedure_revision: "synthetic-procedure-r1".to_string(),
        procedure_sha256: sha('d'),
        robot_revision: "synthetic-robot-r1".to_string(),
        tool_revision: "synthetic-tool-r1".to_string(),
        control_stack_revision: "synthetic-controller-r1".to_string(),
        environment_profile_id: "synthetic-dry-mockup-v1".to_string(),
        environment_profile_sha256: sha('e'),
        task_scope_sha256: sha('f'),
        required_operation_ids: vec![
            "isolate".to_string(),
            "disconnect".to_string(),
            "extract".to_string(),
            "install".to_string(),
            "reconnect".to_string(),
            "requalify".to_string(),
            "restart".to_string(),
        ],
        operator_competency_stratum: "synthetic-team-competency-v1".to_string(),
        clock_synchronization_ref: "synthetic-ptp-clock-v1".to_string(),
        metrology_calibration_ref: "synthetic-laser-tracker-calibration-v1".to_string(),
        evidence_class: BenchmarkEvidenceClass::Synthetic,
        one_sided_confidence_level: 0.95,
        completion_time_quantile: 0.90,
        alignment_tolerance_mm: 0.5,
    }
}

fn trial(
    manifest: &RemoteHandlingBenchmarkManifest,
    id: &str,
    duration: u64,
    status: TrialStatus,
    alignment_error: Option<f64>,
) -> RemoteHandlingTrial {
    let operation_events = if status == TrialStatus::Completed {
        manifest
            .required_operation_ids
            .iter()
            .enumerate()
            .map(|(index, operation_id)| OperationTrialEvent {
                operation_id: operation_id.clone(),
                start_offset_seconds: index as u64 * 10,
                end_offset_seconds: index as u64 * 10 + 5,
                outcome: OperationTrialOutcome::Completed,
                retry_count: 0,
                intervention_count: u32::from(index == 2),
                recovery_action_count: 0,
            })
            .collect()
    } else {
        Vec::new()
    };
    RemoteHandlingTrial {
        trial_id: id.to_string(),
        profile_sha256: manifest.profile_sha256.clone(),
        assembly_revision: manifest.assembly_revision.clone(),
        geometry_sha256: manifest.geometry_sha256.clone(),
        interface_manifest_sha256: manifest.interface_manifest_sha256.clone(),
        procedure_revision: manifest.procedure_revision.clone(),
        procedure_sha256: manifest.procedure_sha256.clone(),
        robot_revision: manifest.robot_revision.clone(),
        tool_revision: manifest.tool_revision.clone(),
        control_stack_revision: manifest.control_stack_revision.clone(),
        environment_profile_id: manifest.environment_profile_id.clone(),
        environment_profile_sha256: manifest.environment_profile_sha256.clone(),
        task_scope_sha256: manifest.task_scope_sha256.clone(),
        operator_competency_stratum: manifest.operator_competency_stratum.clone(),
        clock_synchronization_ref: manifest.clock_synchronization_ref.clone(),
        metrology_calibration_ref: alignment_error.map(|_| manifest.metrology_calibration_ref.clone()),
        evidence_class: manifest.evidence_class,
        raw_data_ref: format!("synthetic-raw-artifact-{id}"),
        raw_data_sha256: raw_digest(id),
        attempt_duration_seconds: duration,
        status,
        safe_terminal_state: status != TrialStatus::Failed,
        operation_events,
        measured_alignment_error_mm: alignment_error,
        dropped_grasp_count: 0,
        collision_contact_alarm_count: 0,
        connector_error_count: 0,
        tool_change_count: 1,
        rework_operation_count: 0,
    }
}

fn main() {
    let manifest = manifest();
    let mut trials: Vec<RemoteHandlingTrial> = (0..30)
        .map(|index| {
            trial(
                &manifest,
                &format!("completed-{index:02}"),
                600 + index as u64,
                TrialStatus::Completed,
                Some(if index == 3 { 0.8 } else { 0.2 }),
            )
        })
        .collect();
    trials.push(trial(&manifest, "failed-attempt-a", 420, TrialStatus::Failed, None));
    trials.push(trial(&manifest, "failed-attempt-b", 510, TrialStatus::Failed, None));
    trials.push(trial(&manifest, "aborted-attempt-a", 350, TrialStatus::Aborted, None));
    trials.push(trial(&manifest, "aborted-attempt-b", 390, TrialStatus::Aborted, None));

    let report = summarize_remote_handling_benchmark(&manifest, &trials);
    println!("contract_version={}", report.contract_version);
    println!("benchmark={}", report.benchmark_id);
    println!("disposition={:?}", report.disposition);
    println!("attempted_trials={}", report.attempted_trials);
    println!("completed_trials={}", report.completed_trials);
    println!("failed_trials={}", report.failed_trials);
    println!("aborted_trials={}", report.aborted_trials);
    println!("completion_fraction={:.6}", report.completion_fraction.unwrap_or(f64::NAN));
    println!(
        "completion_fraction_one_sided_lower_bound={:.6}",
        report.completion_fraction_one_sided_lower_bound.unwrap_or(f64::NAN)
    );
    println!("completed_time_p50_seconds={}", report.completed_duration_p50_seconds.unwrap_or_default());
    println!("completed_time_p90_seconds={}", report.completed_duration_p90_seconds.unwrap_or_default());
    println!(
        "completion_time_p90_upper_confidence_seconds={}",
        report.completion_time_quantile_upper_confidence_seconds.unwrap_or_default()
    );
    println!("completed_duration_sample_count={}", report.completed_duration_sample_count);
    println!("unsafe_terminal_trials={}", report.unsafe_terminal_trials);
    println!("interventions_total={}", report.interventions_total);
    println!("alignment_tolerance_failures={}", report.alignment_tolerance_failures);
    println!(
        "SCOPE: synthetic repeated-trial corpus and exact statistical contract only; no physical mockup or fusion-plant evidence"
    );

    if report.disposition
        != fusion_decision_compiler_contract::remote_handling_benchmark::BenchmarkDisposition::ValidBenchmarkSummary
        || report.attempted_trials != 34
        || report.completed_trials != 30
        || report.completion_time_quantile_upper_confidence_seconds != Some(629)
    {
        eprintln!("synthetic remote-handling benchmark did not match expected summary");
        for diagnostic in report.diagnostics {
            eprintln!("diagnostic={diagnostic:?}");
        }
        std::process::exit(1);
    }
}
