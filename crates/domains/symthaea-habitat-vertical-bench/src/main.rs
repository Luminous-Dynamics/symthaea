// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::{env, fs, process};

use symthaea_habitat_vertical_bench::scenario_campaign::{
    validate_scenario_campaign_str, EMBEDDED_SCENARIO_CAMPAIGN,
};
use symthaea_habitat_vertical_bench::resilience_options::{
    validate_resilience_options_str, EMBEDDED_RESILIENCE_OPTIONS,
};
use symthaea_habitat_vertical_bench::assumption_falsification::{
    validate_assumption_falsification_str, EMBEDDED_HABITAT_ASSUMPTIONS,
};
use symthaea_habitat_vertical_bench::event_replay::{validate_event_log_str, EMBEDDED_EVENT_LOG};
use symthaea_habitat_vertical_bench::material_ledger::{validate_material_ledger_str, EMBEDDED_MATERIAL_LEDGER};
use symthaea_habitat_vertical_bench::measurement_gate::{validate_measurement_gate_str, EMBEDDED_MEASUREMENT_GATE};
use symthaea_habitat_vertical_bench::audit_bundle::{validate_audit_bundle_str, EMBEDDED_AUDIT_BUNDLE};
use symthaea_habitat_vertical_bench::recall_drill::{validate_recall_drill_str, EMBEDDED_RECALL_DRILL};
use symthaea_habitat_vertical_bench::quantity_genealogy::{validate_quantity_genealogy_str, EMBEDDED_QUANTITY_GENEALOGY};
use symthaea_habitat_vertical_bench::{validate_fixture_str, EMBEDDED_FIXTURE};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 1 {
        eprintln!("usage: validate-habitat-vertical [fixture.json]");
        process::exit(2);
    }

    let input_storage = if let Some(path) = args.first() {
        match fs::read_to_string(path) {
            Ok(contents) => Some(contents),
            Err(error) => {
                eprintln!("could not read fixture {}: {error}", path);
                process::exit(2);
            }
        }
    } else {
        None
    };
    let input = input_storage.as_deref().unwrap_or(EMBEDDED_FIXTURE);

    let report = match validate_fixture_str(input) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("invalid JSON fixture: {error}");
            process::exit(2);
        }
    };

    if report.is_valid() {
        let value: serde_json::Value = match serde_json::from_str(input) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("fixture changed between validation and summary parsing: {error}");
                process::exit(2);
            }
        };
        let cost = match symthaea_habitat_vertical_bench::accounting::calculate_cost_comparison(&value) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("cost known-answer calculation failed after structural pass: {error}");
                process::exit(1);
            }
        };
        let mass_case_id = value
            .get("synthetic_material_balance_cases")
            .and_then(serde_json::Value::as_array)
            .and_then(|cases| cases.first())
            .and_then(|case| case.get("case_id"))
            .and_then(serde_json::Value::as_str);
        let Some(mass_case_id) = mass_case_id else {
            eprintln!("no valid material-balance case was found after structural pass");
            process::exit(1);
        };
        let mass = match symthaea_habitat_vertical_bench::accounting::calculate_material_balance(
            &value,
            mass_case_id,
        ) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("material-balance known-answer calculation failed after structural pass: {error}");
                process::exit(1);
            }
        };

        if input_storage.is_none() {
            let campaign = match validate_scenario_campaign_str(EMBEDDED_SCENARIO_CAMPAIGN) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded scenario campaign JSON: {error}");
                    process::exit(2);
                }
            };
            if !campaign.is_valid() {
                eprintln!(
                    "SCENARIO CAMPAIGN DEFINITION: FAIL ({} findings)",
                    campaign.issues.len()
                );
                for issue in campaign.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic disruption campaign definition: PASS ({} scenarios; numeric results={}; simulation=NOT RUN)",
                campaign.scenario_count, campaign.numeric_result_count
            );
            let options = match validate_resilience_options_str(EMBEDDED_RESILIENCE_OPTIONS) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded resilience-options JSON: {error}");
                    process::exit(2);
                }
            };
            if !options.is_valid() {
                eprintln!(
                    "RESILIENCE OPTIONS DEFINITION: FAIL ({} findings)",
                    options.issues.len()
                );
                for issue in options.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic resilience option matrix: PASS ({} options x {} scenarios = {} cells; robust frontier=NOT RUN; probabilities=NOT ASSUMED)",
                options.option_count, options.scenario_count, options.outcome_count
            );
            let assumptions = match validate_assumption_falsification_str(EMBEDDED_HABITAT_ASSUMPTIONS) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded assumption-falsification JSON: {error}");
                    process::exit(2);
                }
            };
            if !assumptions.is_valid() {
                eprintln!(
                    "ASSUMPTION FALSIFICATION DEFINITIONS: FAIL ({} findings)",
                    assumptions.issues.len()
                );
                for issue in assumptions.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "factory assumption/test definitions: PASS ({} assumptions; {} candidate tests; {} critical assumptions; selection=NOT RUN; observations=0)",
                assumptions.assumption_count,
                assumptions.experiment_count,
                assumptions.critical_assumption_count
            );
            let event_replay = match validate_event_log_str(EMBEDDED_EVENT_LOG, EMBEDDED_FIXTURE) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded event-replay JSON: {error}");
                    process::exit(2);
                }
            };
            if !event_replay.is_valid() {
                eprintln!("EVENT REPLAY DEFINITION: FAIL ({} findings)", event_replay.issues.len());
                for issue in event_replay.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic event replay contract: PASS ({} unique events; {} exact redeliveries; digest=NOT COMPUTED; physical evidence=NOT ESTABLISHED)",
                event_replay.unique_event_count,
                event_replay.duplicate_delivery_count
            );
            let material_ledger = match validate_material_ledger_str(
                EMBEDDED_MATERIAL_LEDGER,
                EMBEDDED_FIXTURE,
                EMBEDDED_EVENT_LOG,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded material-ledger JSON: {error}");
                    process::exit(2);
                }
            };
            if !material_ledger.is_valid() {
                eprintln!("MATERIAL LEDGER DEFINITION: FAIL ({} findings)", material_ledger.issues.len());
                for issue in material_ledger.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic material ledger: PASS ({} accounts; {} transactions; {} mg opening / {} mg closing; nonzero accounts={}; unit conversion=BLOCKED; physical evidence=NOT ESTABLISHED)",
                material_ledger.account_count,
                material_ledger.transaction_count,
                material_ledger.opening_mass_mg,
                material_ledger.closing_mass_mg,
                material_ledger.nonzero_closing_account_count
            );
            let measurement_gate = match validate_measurement_gate_str(EMBEDDED_MEASUREMENT_GATE) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded measurement-gate JSON: {error}");
                    process::exit(2);
                }
            };
            if !measurement_gate.is_valid() {
                eprintln!("MEASUREMENT GATE DEFINITION: FAIL ({} findings)", measurement_gate.issues.len());
                for issue in measurement_gate.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            let count_disposition = |name: &str| {
                measurement_gate.results.iter().filter(|result| result.disposition == name).count()
            };
            println!(
                "synthetic measurement decision-rule fixture: PASS ({} cases; conforming={}; nonconforming={}; indeterminate={}; blocked={}; not_evaluable={}; physical measurements=NONE)",
                measurement_gate.case_count,
                count_disposition("Conforming"),
                count_disposition("Nonconforming"),
                count_disposition("Indeterminate"),
                count_disposition("Blocked"),
                count_disposition("NotEvaluable")
            );
            let quantity_genealogy = match validate_quantity_genealogy_str(EMBEDDED_QUANTITY_GENEALOGY) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded quantity-genealogy JSON: {error}");
                    process::exit(2);
                }
            };
            if !quantity_genealogy.is_valid() {
                eprintln!("QUANTITY GENEALOGY: FAIL ({} findings)", quantity_genealogy.issues.len());
                for issue in quantity_genealogy.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic quantity genealogy definition: VALID ({} operations; opening={} mg; ending={} mg; affected terminals={}; physical evidence=NOT ESTABLISHED)",
                quantity_genealogy.operation_count,
                quantity_genealogy.initial_mass_mg,
                quantity_genealogy.ending_mass_mg,
                quantity_genealogy.affected_terminal_ids.len()
            );
            let audit_bundle = match validate_audit_bundle_str(
                EMBEDDED_AUDIT_BUNDLE,
                EMBEDDED_FIXTURE,
                EMBEDDED_EVENT_LOG,
                EMBEDDED_MATERIAL_LEDGER,
                EMBEDDED_MEASUREMENT_GATE,
                EMBEDDED_RECALL_DRILL,
                EMBEDDED_QUANTITY_GENEALOGY,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded cross-layer audit bundle JSON: {error}");
                    process::exit(2);
                }
            };
            if !audit_bundle.is_valid() {
                eprintln!("CROSS-LAYER AUDIT BUNDLE: FAIL ({} findings)", audit_bundle.issues.len());
                for issue in audit_bundle.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic cross-layer audit bundle definition: VALID ({} fixture fingerprints; {} events; {} material transactions; {} measurement cases; recall closure={}; unresolved affected={}; quantity operations={}; mass={}/{} mg; quantity-affected terminals={}; source authenticity=NOT ESTABLISHED)",
                audit_bundle.artifact_count,
                audit_bundle.event_count,
                audit_bundle.material_transaction_count,
                audit_bundle.measurement_case_count,
                audit_bundle.recall_closure_status,
                audit_bundle.recall_unresolved_affected_count,
                audit_bundle.quantity_genealogy_operation_count,
                audit_bundle.quantity_genealogy_initial_mass_mg,
                audit_bundle.quantity_genealogy_ending_mass_mg,
                audit_bundle.quantity_genealogy_affected_terminal_count
            );
            let recall_drill = match validate_recall_drill_str(
                EMBEDDED_RECALL_DRILL,
                EMBEDDED_FIXTURE,
                EMBEDDED_EVENT_LOG,
            ) {
                Ok(report) => report,
                Err(error) => {
                    eprintln!("invalid embedded recall-closure drill JSON: {error}");
                    process::exit(2);
                }
            };
            if !recall_drill.is_valid() {
                eprintln!("RECALL RESPONSE DRILL: FAIL ({} findings)", recall_drill.issues.len());
                for issue in recall_drill.issues {
                    eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
                }
                process::exit(1);
            }
            println!(
                "synthetic recall response drill definition: VALID ({} targets; affected={}; contained={}; unresolved={}; false-positive controls={}; closure={}; real recall=NOT PERFORMED)",
                recall_drill.target_count,
                recall_drill.affected_terminal_ids.len(),
                recall_drill.confirmed_contained_count,
                recall_drill.unresolved_affected_count,
                recall_drill.false_positive_control_count,
                recall_drill.closure_status
            );
        } else {
            println!("embedded disruption campaign: not checked against the alternate fixture");
            println!("embedded resilience option matrix: not checked against the alternate fixture");
            println!("embedded assumption/test definitions: not checked against the alternate fixture");
            println!("embedded event replay contract: not checked against the alternate fixture");
            println!("embedded material ledger: not checked against the alternate fixture");
            println!("embedded measurement gate: not checked against the alternate fixture");
            println!("embedded cross-layer audit bundle: not checked against the alternate fixture");
        }

        println!("STRUCTURAL FIXTURE VALIDATION: PASS");
        println!(
            "synthetic known-answer candidate/reference cost ratio: {} basis points; target_met={}",
            cost.ratio_basis_points_floor, cost.target_met
        );
        println!(
            "synthetic material-balance closure: {} basis points; accepted mass yield: {} basis points; unaccounted={} {}",
            mass.closure_basis_points_floor,
            mass.accepted_yield_basis_points_floor,
            mass.unaccounted_mass_mg,
            mass.mass_unit
        );
        println!("scope: synthetic arithmetic and structure only; no physical or product qualification");
        println!("negative controls are declared, not all executed by this binary");
        return;
    }

    eprintln!(
        "STRUCTURAL FIXTURE VALIDATION: FAIL ({} findings)",
        report.issues.len()
    );
    for issue in report.issues {
        eprintln!("{} {}: {}", issue.code, issue.path, issue.message);
    }
    process::exit(1);
}
