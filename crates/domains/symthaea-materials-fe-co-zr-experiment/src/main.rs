// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

mod readiness_gate;
mod readiness_recheck;

use readiness_gate::{
    RUNTIME_READINESS_FILE, RuntimeReadinessGate, execute_runtime_readiness,
};
use readiness_recheck::require_current_mysql_observation;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use symthaea_materials_authorized_audit_binding::AuthorizedAuditBinding;
use symthaea_materials_fe_co_zr_experiment::{
    LocalExperimentConfig, PreparedExperiment, acquire_snapshot, acquisition_receipt,
    audit_authorized_targets, construct_authorized_targets, decompress_snapshot,
    extract_historical_corpus, import_database, inventory_database, preflight_database,
    prepare_experiment, prepare_extraction_plan, prepare_import_plan, read_json,
    verify_stored_audit, write_json_new,
};
use symthaea_materials_history_tool::VerifiedHistoricalRun;
use symthaea_materials_oqmd_acquisition::OqmdHttpsAcquisitionAttempt;
use symthaea_materials_oqmd_db_state::VerifiedImportedDatabaseState;
use symthaea_materials_oqmd_extraction_executor::{
    CompletedQmpyExtraction, NormalizationAttempt, QmpyAdapterExecutionEvidence,
    QmpyExtractionPlan,
};
use symthaea_materials_oqmd_import_executor::{
    MysqlServerPreflightEvidence, OqmdDecompressionEvidence, OqmdLocalImportPlan,
    OqmdMysqlImportEvidence,
};
use symthaea_materials_snapshot_acquisition::HistoricalSnapshotAcquisitionReceipt;
use symthaea_materials_target_set::AuthorizedHistoricalTargetSet;

const PREPARED: &str = "prepared.json";
const ACQUISITION_ATTEMPT: &str = "acquisition-attempt.json";
const ACQUISITION_RECEIPT: &str = "acquisition-receipt.json";
const IMPORT_PLAN: &str = "import-plan.json";
const DECOMPRESSION: &str = "decompression.json";
const PREFLIGHT: &str = "server-preflight.json";
const IMPORT: &str = "mysql-import.json";
const DATABASE_STATE: &str = "database-state.json";
const EXTRACTION_PLAN: &str = "extraction-plan.json";
const RAW_EXTRACTION: &str = "raw-extraction.json";
const NORMALIZATION: &str = "normalization.json";
const COMPLETED_EXTRACTION: &str = "completed-extraction.json";
const AUTHORIZED_TARGETS: &str = "authorized-targets.json";
const HISTORICAL_RUN: &str = "historical-run.json";
const AUTHORIZED_AUDIT: &str = "authorized-audit-binding.json";

fn main() {
    if let Err(error) = run() {
        eprintln!("symthaea-fe-co-zr-experiment: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    let config_arg = args.next().ok_or_else(usage)?;
    let config_path = PathBuf::from(config_arg);
    let config: LocalExperimentConfig = read_json(&config_path)?;

    match command.as_str() {
        "prepare" => {
            require_no_more(args)?;
            let prepared = prepare_experiment(&config)?;
            write_json_new(&state(&config, PREPARED)?, &prepared)?;
            println!("prepared_sha256={}", prepared.prepared_sha256()?);
            println!("runtime_readiness=REQUIRED");
            println!("next=preflight-runtime");
        }
        "preflight-runtime" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let gate = execute_runtime_readiness(&config, &prepared)?;
            write_json_new(&state(&config, RUNTIME_READINESS_FILE)?, &gate)?;
            println!("runtime_readiness_gate_sha256={}", gate.gate_sha256()?);
            println!("qmpy_import_probe=PASS");
            println!("mysql_live_fixture=PASS");
            println!("execution_prerequisites_ready=true");
        }
        "ready" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let gate = require_runtime_readiness(&config, &prepared)?;
            println!("{}", serde_json::to_string_pretty(&gate)?);
            println!("execution_prerequisites_ready=true");
        }
        "acquire" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let attempt = acquire_snapshot(&config, &prepared)?;
            write_json_new(&state(&config, ACQUISITION_ATTEMPT)?, &attempt)?;
            println!("attempt_sha256={}", attempt.attempt_sha256()?);
            println!("assessment={:?}", attempt.assessment);
        }
        "bind-acquisition" => {
            let acquired_at_utc = args.next().ok_or_else(usage)?;
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let attempt = load::<OqmdHttpsAcquisitionAttempt>(&config, ACQUISITION_ATTEMPT)?;
            let receipt = acquisition_receipt(&config, &prepared, &attempt, &acquired_at_utc)?;
            write_json_new(&state(&config, ACQUISITION_RECEIPT)?, &receipt)?;
            println!("compressed_snapshot_sha256={}", receipt.compressed_snapshot_sha256);
            println!("compressed_snapshot_bytes={}", receipt.compressed_snapshot_bytes);
        }
        "prepare-import" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let plan = prepare_import_plan(&config, &prepared, &acquisition)?;
            write_json_new(&state(&config, IMPORT_PLAN)?, &plan)?;
            println!(
                "import_plan_sha256={}",
                plan.plan_sha256(
                    &symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol(),
                    &prepared.import_profile,
                    &acquisition,
                )?
            );
            println!("server_pid={}", plan.server_pid);
        }
        "decompress" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let plan = load::<OqmdLocalImportPlan>(&config, IMPORT_PLAN)?;
            let evidence = decompress_snapshot(&prepared, &acquisition, &plan)?;
            write_json_new(&state(&config, DECOMPRESSION)?, &evidence)?;
            println!("decompression_evidence_sha256={}", evidence.evidence_sha256()?);
        }
        "preflight-db" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let plan = load::<OqmdLocalImportPlan>(&config, IMPORT_PLAN)?;
            let evidence = preflight_database(&prepared, &acquisition, &plan)?;
            write_json_new(&state(&config, PREFLIGHT)?, &evidence)?;
            println!("preflight_evidence_sha256={}", evidence.evidence_sha256()?);
        }
        "import" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let plan = load::<OqmdLocalImportPlan>(&config, IMPORT_PLAN)?;
            let decompression = load::<OqmdDecompressionEvidence>(&config, DECOMPRESSION)?;
            let preflight = load::<MysqlServerPreflightEvidence>(&config, PREFLIGHT)?;
            let evidence = import_database(
                &prepared,
                &acquisition,
                &plan,
                &decompression,
                &preflight,
            )?;
            write_json_new(&state(&config, IMPORT)?, &evidence)?;
            println!("import_process_success={}", evidence.process_success());
            println!("import_evidence_sha256={}", evidence.evidence_sha256()?);
        }
        "inventory" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let plan = load::<OqmdLocalImportPlan>(&config, IMPORT_PLAN)?;
            let import = load::<OqmdMysqlImportEvidence>(&config, IMPORT)?;
            let database_state = inventory_database(&prepared, &acquisition, &plan, &import)?;
            write_json_new(&state(&config, DATABASE_STATE)?, &database_state)?;
            println!("database_state_sha256={}", database_state.evidence_sha256()?);
            println!("table_count={}", database_state.inventory.tables.len());
        }
        "prepare-extraction" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let database_state = load::<VerifiedImportedDatabaseState>(&config, DATABASE_STATE)?;
            let executable = env::current_exe()?;
            let plan = prepare_extraction_plan(
                &config,
                &prepared,
                &acquisition,
                &database_state,
                &executable,
            )?;
            write_json_new(&state(&config, EXTRACTION_PLAN)?, &plan)?;
            println!(
                "extraction_plan_sha256={}",
                plan.plan_sha256(
                    &symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol(),
                    &prepared.import_profile,
                    &acquisition,
                    &database_state,
                )?
            );
        }
        "extract" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let _gate = require_runtime_readiness(&config, &prepared)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let database_state = load::<VerifiedImportedDatabaseState>(&config, DATABASE_STATE)?;
            let plan = load::<QmpyExtractionPlan>(&config, EXTRACTION_PLAN)?;
            let (raw, normalization, completed) = extract_historical_corpus(
                &prepared,
                &acquisition,
                &database_state,
                &plan,
            )?;
            write_json_new(&state(&config, RAW_EXTRACTION)?, &raw)?;
            write_json_new(&state(&config, NORMALIZATION)?, &normalization)?;
            match completed {
                Some(completed) => {
                    write_json_new(&state(&config, COMPLETED_EXTRACTION)?, &completed)?;
                    println!("compact_corpus_records={}", completed.corpus.records.len());
                    println!(
                        "completed_extraction_sha256={}",
                        completed.evidence_sha256(
                            &symthaea_materials_historical_extraction::oqmd_v17_fe_co_zr_protocol()
                        )?
                    );
                }
                None => {
                    println!("source_multiplicity_unambiguous=false");
                    println!(
                        "multiplicity_report_sha256={}",
                        normalization.multiplicity.report_sha256()?
                    );
                    return Err(
                        "historical extraction refused: multiple standard-fit rows exist for at least one OQMD entry"
                            .into(),
                    );
                }
            }
        }
        "targets" => {
            require_no_more(args)?;
            let targets = construct_authorized_targets(&config)?;
            write_json_new(&state(&config, AUTHORIZED_TARGETS)?, &targets)?;
            println!("target_count={}", targets.targets.len());
            println!("target_set_sha256={}", targets.target_set_sha256()?);
        }
        "audit" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let database_state = load::<VerifiedImportedDatabaseState>(&config, DATABASE_STATE)?;
            let extraction = load::<CompletedQmpyExtraction>(&config, COMPLETED_EXTRACTION)?;
            let targets = load::<AuthorizedHistoricalTargetSet>(&config, AUTHORIZED_TARGETS)?;
            let executable = env::current_exe()?;
            let (run, binding) = audit_authorized_targets(
                &config,
                &prepared,
                &acquisition,
                &database_state,
                &extraction,
                &targets,
                &executable,
            )?;
            write_json_new(&state(&config, HISTORICAL_RUN)?, &run)?;
            write_json_new(&state(&config, AUTHORIZED_AUDIT)?, &binding)?;
            println!("verified_historical_run_sha256={}", run.verified_run_sha256()?);
            println!("authorized_audit_binding_sha256={}", binding.binding_sha256()?);
        }
        "verify" => {
            require_no_more(args)?;
            let prepared = load::<PreparedExperiment>(&config, PREPARED)?;
            let acquisition = load::<HistoricalSnapshotAcquisitionReceipt>(&config, ACQUISITION_RECEIPT)?;
            let database_state = load::<VerifiedImportedDatabaseState>(&config, DATABASE_STATE)?;
            let extraction = load::<CompletedQmpyExtraction>(&config, COMPLETED_EXTRACTION)?;
            let targets = load::<AuthorizedHistoricalTargetSet>(&config, AUTHORIZED_TARGETS)?;
            let run = load::<VerifiedHistoricalRun>(&config, HISTORICAL_RUN)?;
            let binding = load::<AuthorizedAuditBinding>(&config, AUTHORIZED_AUDIT)?;
            verify_stored_audit(
                &config,
                &prepared,
                &acquisition,
                &database_state,
                &extraction,
                &targets,
                &run,
                &binding,
                &env::current_exe()?,
            )?;
            println!("verification=PASS");
            println!("verified_historical_run_sha256={}", run.verified_run_sha256()?);
            println!("authorized_audit_binding_sha256={}", binding.binding_sha256()?);
        }
        _ => return Err(usage()),
    }
    Ok(())
}

fn require_runtime_readiness(
    config: &LocalExperimentConfig,
    prepared: &PreparedExperiment,
) -> Result<RuntimeReadinessGate, Box<dyn Error>> {
    let gate = load::<RuntimeReadinessGate>(config, RUNTIME_READINESS_FILE)?;
    gate.validate_against(config, prepared)?;
    require_current_mysql_observation(config, &gate.mysql_fixture.live_observation)?;
    Ok(gate)
}

fn load<T: serde::de::DeserializeOwned>(
    config: &LocalExperimentConfig,
    file_name: &str,
) -> Result<T, Box<dyn Error>> {
    Ok(read_json(&state(config, file_name)?)?)
}

fn state(config: &LocalExperimentConfig, file_name: &str) -> Result<PathBuf, Box<dyn Error>> {
    Ok(config.state_path(file_name)?)
}

fn require_no_more(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    if args.next().is_some() {
        Err(usage())
    } else {
        Ok(())
    }
}

fn usage() -> Box<dyn Error> {
    "usage: symthaea-fe-co-zr-experiment <prepare|preflight-runtime|ready|acquire|bind-acquisition|prepare-import|decompress|preflight-db|import|inventory|prepare-extraction|extract|targets|audit|verify> <absolute-or-relative-config.json> [stage args]\nbind-acquisition requires one additional YYYY-MM-DDTHH:MM:SSZ argument"
        .into()
}

#[allow(dead_code)]
fn _type_markers(
    _raw: QmpyAdapterExecutionEvidence,
    _normalization: NormalizationAttempt,
) {
}
