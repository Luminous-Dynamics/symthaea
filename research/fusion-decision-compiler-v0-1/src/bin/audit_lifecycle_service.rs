use fusion_decision_compiler_contract::lifecycle_service_ledger::{
    AvailabilityEvent, LifecycleEvidenceClass, LifecycleServicePlan, OutageCategory,
    PowerOperatingState, PowerStateSegment, ServiceImpact, evaluate_lifecycle_service_budget,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn event(
    event_id: &str,
    category: OutageCategory,
    subsystem_id: &str,
    start_second: u64,
    end_second: u64,
) -> AvailabilityEvent {
    AvailabilityEvent {
        event_id: event_id.to_string(),
        category,
        impact: ServiceImpact::FullOutage,
        subsystem_id: subsystem_id.to_string(),
        failure_mode_id: format!("failure-mode-{event_id}"),
        start_second,
        end_second,
        evidence_ref: format!("synthetic-event-evidence-{event_id}"),
        evidence_class: LifecycleEvidenceClass::Synthetic,
    }
}

fn segment(
    id: &str,
    start_second: u64,
    end_second: u64,
    net_power_mwe: f64,
    state: PowerOperatingState,
    active_event_ids: &[&str],
) -> PowerStateSegment {
    PowerStateSegment {
        segment_id: id.to_string(),
        start_second,
        end_second,
        net_power_mwe,
        state,
        active_event_ids: active_event_ids
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        power_balance_report_ref: format!("synthetic-power-balance-{id}"),
        output_sha256: sha('a'),
        evidence_ref: format!("synthetic-state-evidence-{id}"),
        evidence_class: LifecycleEvidenceClass::Synthetic,
    }
}

fn main() {
    let plan = LifecycleServicePlan {
        campaign_id: "synthetic-24-hour-lifecycle-v1".to_string(),
        horizon_ref: "24-hour-horizon-v1".to_string(),
        horizon_start_second: 0,
        horizon_end_second: 86_400,
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
        events: vec![
            event(
                "planned-blanket-changeout",
                OutageCategory::Planned,
                "blanket",
                3_600,
                7_200,
            ),
            event(
                "forced-bop-pump-failure",
                OutageCategory::Unplanned,
                "balance-of-plant-pump",
                5_400,
                9_000,
            ),
        ],
        power_segments: vec![
            segment(
                "generation-before",
                0,
                3_600,
                150.0,
                PowerOperatingState::PowerGeneration,
                &[],
            ),
            segment(
                "planned-maintenance",
                3_600,
                5_400,
                -5.0,
                PowerOperatingState::PlannedMaintenance,
                &["planned-blanket-changeout"],
            ),
            segment(
                "concurrent-outages",
                5_400,
                7_200,
                -8.0,
                PowerOperatingState::ConcurrentOutage,
                &["planned-blanket-changeout", "forced-bop-pump-failure"],
            ),
            segment(
                "forced-outage",
                7_200,
                9_000,
                -7.0,
                PowerOperatingState::ForcedOutage,
                &["forced-bop-pump-failure"],
            ),
            segment(
                "generation-after",
                9_000,
                86_400,
                150.0,
                PowerOperatingState::PowerGeneration,
                &[],
            ),
        ],
    };

    let report = evaluate_lifecycle_service_budget(&plan);
    println!("contract_version={}", report.contract_version);
    println!("campaign={}", report.campaign_id);
    println!("disposition={:?}", report.disposition);
    println!("horizon_seconds={}", report.horizon_seconds.unwrap_or_default());
    println!("available_seconds={}", report.available_seconds.unwrap_or_default());
    println!(
        "planned_only_unavailable_seconds={}",
        report.planned_only_unavailable_seconds.unwrap_or_default()
    );
    println!(
        "unplanned_only_unavailable_seconds={}",
        report.unplanned_only_unavailable_seconds.unwrap_or_default()
    );
    println!(
        "concurrent_unavailable_seconds={}",
        report.concurrent_unavailable_seconds.unwrap_or_default()
    );
    println!(
        "total_unavailable_seconds={}",
        report.total_unavailable_seconds.unwrap_or_default()
    );
    println!(
        "availability_fraction={:.6}",
        report.availability_fraction.unwrap_or(f64::NAN)
    );
    println!(
        "signed_net_energy_mwh={:.6}",
        report.signed_net_energy_mwh.unwrap_or(f64::NAN)
    );
    println!(
        "export_energy_mwh={:.6}",
        report.export_energy_mwh.unwrap_or(f64::NAN)
    );
    println!(
        "import_energy_mwh={:.6}",
        report.import_energy_mwh.unwrap_or(f64::NAN)
    );
    println!(
        "SCOPE: synthetic deterministic event and net-energy ledger; not an availability prediction or fusion-plant evidence"
    );

    if report.disposition
        != fusion_decision_compiler_contract::lifecycle_service_ledger::LifecycleServiceDisposition::ValidDeterministicLedger
    {
        eprintln!("synthetic lifecycle service ledger did not reconcile");
        std::process::exit(1);
    }
}
