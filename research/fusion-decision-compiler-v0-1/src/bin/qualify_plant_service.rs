use fusion_decision_compiler_contract::plant_service_gate::{
    D0FUS_3_2_0_ITER_SCHEMA_REF, D0FUS_3_2_0_ITER_SCHEMA_SHA256,
    PowerBalanceProfile, PowerBalanceRun, PowerObservable, PowerTermEvidence,
    evaluate_plant_power_balance,
};

fn term(
    observable: PowerObservable,
    value_mw: f64,
    tuple_index: usize,
) -> PowerTermEvidence {
    PowerTermEvidence {
        observable,
        value_mw,
        source_locator: format!("D0FUS-3.2.0-ITER:results[{tuple_index}]"),
        evidence_ref: format!(
            "d0fus-parity-receipt#source.core_observables.{}",
            observable.id()
        ),
    }
}

fn main() {
    // Values are taken from the pinned D0FUS v3.2.0 ITER output report and
    // run receipt. The report rounds several fields; the explicit tolerance
    // below preserves that provenance and does not imply higher precision.
    let profile = PowerBalanceProfile {
        schema_ref: D0FUS_3_2_0_ITER_SCHEMA_REF.to_string(),
        schema_sha256: D0FUS_3_2_0_ITER_SCHEMA_SHA256.to_string(),
        service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
        unit_ref: "MWe".to_string(),
        averaging_window_ref: "iter-run-steady-state-window-v1".to_string(),
        sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
        absolute_balance_tolerance_mw: 0.001,
        required_net_export_reserve_mw: 0.0,
        reserve_threshold_inclusive: true,
    };

    let run = PowerBalanceRun {
        run_id: "d0fus-3.2.0-iter-reference-run-2026-11-15".to_string(),
        solver_id: "D0FUS".to_string(),
        solver_revision: "a682c3511dc189b2a4cdd6f4b8bc545d3cdb8924".to_string(),
        candidate_revision: "D0FUS-ITER-reference-deck-v3.2.0".to_string(),
        output_sha256: "59d59114ba7ecbcf873ea386c6f27495611b24464d02101acd68f3c79ca305e7".to_string(),
        schema_ref: D0FUS_3_2_0_ITER_SCHEMA_REF.to_string(),
        schema_sha256: D0FUS_3_2_0_ITER_SCHEMA_SHA256.to_string(),
        service_boundary_ref: profile.service_boundary_ref.clone(),
        unit_ref: profile.unit_ref.clone(),
        averaging_window_ref: profile.averaging_window_ref.clone(),
        sign_convention_ref: profile.sign_convention_ref.clone(),
        receipt_ref: "symthaea-workflow:43729456208:artifact:14167288485".to_string(),
        terms: vec![
            term(PowerObservable::GrossElectric, 192.395, 153),
            term(PowerObservable::VariableRecirculated, 144.149, 160),
            term(PowerObservable::FixedRecirculated, 91.0951, 161),
            term(PowerObservable::TotalRecirculated, 235.244, 162),
            term(PowerObservable::NetElectric, -42.84920863494321, 25),
        ],
    };

    let report = evaluate_plant_power_balance(&profile, &run);
    println!("contract_version={}", report.contract_version);
    println!("solver={}", report.solver_id);
    println!("run_id={}", report.run_id);
    println!("disposition={:?}", report.disposition);
    println!(
        "gross_electric_mwe={:.6}",
        report.gross_electric_mw.unwrap_or(f64::NAN)
    );
    println!(
        "variable_recirculated_mwe={:.6}",
        report.variable_recirculated_mw.unwrap_or(f64::NAN)
    );
    println!(
        "fixed_recirculated_mwe={:.6}",
        report.fixed_recirculated_mw.unwrap_or(f64::NAN)
    );
    println!(
        "total_recirculated_mwe={:.6}",
        report.total_recirculated_mw.unwrap_or(f64::NAN)
    );
    println!(
        "calculated_net_electric_mwe={:.6}",
        report.calculated_net_electric_mw.unwrap_or(f64::NAN)
    );
    println!(
        "reported_net_electric_mwe={:.6}",
        report.reported_net_electric_mw.unwrap_or(f64::NAN)
    );
    println!(
        "maximum_absolute_balance_residual_mw={:.9}",
        report
            .maximum_absolute_balance_residual_mw
            .unwrap_or(f64::NAN)
    );
    println!(
        "normalized_balance_residual={:.12}",
        report.normalized_balance_residual.unwrap_or(f64::NAN)
    );
    println!(
        "claim_ceiling={}",
        report.claim_ceiling
    );
    println!(
        "SCOPE: actual pinned D0FUS ITER model output replay fixture; negative net export is not plant qualification or physical validation"
    );

    if report.disposition != fusion_decision_compiler_contract::plant_service_gate::PowerBalanceDisposition::NetImport {
        eprintln!("D0FUS ITER reference must remain NetImport under the declared service boundary");
        std::process::exit(1);
    }
}
