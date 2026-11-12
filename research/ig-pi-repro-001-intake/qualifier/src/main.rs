use serde_json::Value;
use std::{env, fs, process::ExitCode};

const INTAKE_ID: &str = "IG-PLURALISTIC-IGNORANCE-REPRO-001";
const SCHEMA_ID: &str = "https://luminousdynamics.io/schemas/ig-pi-repro-intake-v0.1.schema.json";
const UPSTREAM_COMMIT: &str = "b4d79c8229eb5d7bedfea824a9988115cfe53ef6";

fn str_at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut cur = v;
    for key in path { cur = cur.get(*key)?; }
    cur.as_str()
}
fn bool_at(v: &Value, path: &[&str]) -> Option<bool> {
    let mut cur = v;
    for key in path { cur = cur.get(*key)?; }
    cur.as_bool()
}
fn fail(errors: &mut Vec<String>, message: impl Into<String>) {
    errors.push(message.into());
}
fn check_schema_header(schema: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        fail(&mut errors, "schema dialect must be JSON Schema 2020-12");
    }
    if schema["$id"].as_str() != Some(SCHEMA_ID) {
        fail(&mut errors, "schema identity mismatch");
    }
    if schema["type"].as_str() != Some("object") {
        fail(&mut errors, "top-level schema must be an object");
    }
    if schema["additionalProperties"].as_bool() != Some(false) {
        fail(&mut errors, "top-level schema must reject additional properties");
    }
    let required: Vec<&str> = [
        "intake_id","version","status","captured_at","upstream","reported_protocol",
        "seed_audit","scorer_audit","validation_audit","dataset_audit","custody","disposition"
    ].into_iter().collect();
    for key in required {
        if !schema["required"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str()==Some(key))) {
            fail(&mut errors, format!("schema missing required field {key}"));
        }
    }
    errors
}
fn validate(manifest: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if str_at(manifest,&["intake_id"]) != Some(INTAKE_ID) { fail(&mut errors,"unexpected intake ID"); }
    if str_at(manifest,&["version"]) != Some("0.1.0") { fail(&mut errors,"unexpected intake version"); }
    if str_at(manifest,&["status"]) != Some("metadata_only_blocked") { fail(&mut errors,"intake status must remain metadata_only_blocked"); }
    if str_at(manifest,&["upstream","commit"]) != Some(UPSTREAM_COMMIT) { fail(&mut errors,"upstream commit is not the inspected snapshot"); }

    let artifacts = manifest["upstream"]["artifacts"].as_array();
    if artifacts.is_none_or(|a| a.len() < 6) { fail(&mut errors,"six pinned source artifacts are required"); }
    else {
        for artifact in artifacts.unwrap() {
            let sha = artifact["blob_sha"].as_str().unwrap_or("");
            if sha.len()!=40 || !sha.bytes().all(|b|b.is_ascii_hexdigit()) {
                fail(&mut errors,"source artifact has invalid Git blob SHA");
            }
        }
    }
    if manifest["reported_protocol"]["scenario_count"].as_u64()!=Some(100)
        || manifest["reported_protocol"]["domain_count"].as_u64()!=Some(10)
        || manifest["reported_protocol"]["authority_level_count"].as_u64()!=Some(5) {
        fail(&mut errors,"reported scenario/domain/authority counts differ from the pinned paper");
    }
    if manifest["reported_protocol"]["population_size"].as_u64()!=Some(20)
        || manifest["reported_protocol"]["assigned_private_dissenters"].as_u64()!=Some(16)
        || manifest["reported_protocol"]["assigned_private_supporters"].as_u64()!=Some(4) {
        fail(&mut errors,"80/20 assigned private-stance condition must be recorded exactly");
    }
    if !manifest["reported_protocol"]["private_state_semantics"].as_str().unwrap_or("").contains("not privileged access") {
        fail(&mut errors,"assigned private stance must not be promoted to a latent-belief claim");
    }

    if bool_at(manifest,&["seed_audit","python_builtin_string_hash_used"])!=Some(true)
        || bool_at(manifest,&["seed_audit","python_string_hash_process_salted"])!=Some(true)
        || bool_at(manifest,&["seed_audit","deterministic_speaker_order_across_fresh_processes_guaranteed"])!=Some(false) {
        fail(&mut errors,"upstream process-salted hash seed risk must remain represented");
    }
    if bool_at(manifest,&["seed_audit","explicit_pythonhashseed_documented"])!=Some(false) {
        fail(&mut errors,"do not claim PYTHONHASHSEED is documented");
    }

    if manifest["scorer_audit"]["unmatched_text_score"].as_f64()!=Some(0.0)
        || manifest["scorer_audit"]["conformity_threshold"].as_f64()!=Some(0.0)
        || bool_at(manifest,&["scorer_audit","neutral_or_unmatched_counted_as_conformity"])!=Some(true) {
        fail(&mut errors,"keyword neutral/unmatched conformity risk must remain explicit");
    }
    let dispositions = manifest["scorer_audit"]["required_dispositions"].as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>()).unwrap_or_default();
    for required in ["OPPOSE","SUPPORT","NEUTRAL_EXPLICIT","AMBIGUOUS","UNCLASSIFIABLE","ERROR_OR_MISSING"] {
        if !dispositions.contains(&required) { fail(&mut errors,format!("required scorer disposition missing: {required}")); }
    }

    if manifest["validation_audit"]["paper_reported_human_validation"]["sample_size"].as_u64()!=Some(100)
        || manifest["validation_audit"]["committed_artifact_validation"]["sample_size"].as_u64()!=Some(36) {
        fail(&mut errors,"the two validation snapshots must remain separately represented");
    }
    if str_at(manifest,&["validation_audit","reconciliation_status"]) !=
        Some("unresolved_distinct_estimands_or_snapshots_not_yet_reconciled") {
        fail(&mut errors,"validation discrepancy must not be silently declared reconciled");
    }
    if str_at(manifest,&["dataset_audit","viewer_status"]) != Some("DatasetGenerationCastError") {
        fail(&mut errors,"Hugging Face mixed-schema viewer failure must remain recorded");
    }
    if str_at(manifest,&["dataset_audit","dataset_admission_status"]) != Some("blocked_schema_and_rights_review") {
        fail(&mut errors,"dataset admission must remain blocked pending schema and rights review");
    }
    if bool_at(manifest,&["upstream","repository_metadata","root_license_file_present"])!=Some(false)
        || !manifest["upstream"]["repository_metadata"]["repository_api_detected_license"].is_null()
        || str_at(manifest,&["upstream","repository_metadata","code_reuse_status"])!=Some("blocked_unresolved") {
        fail(&mut errors,"code license must remain unresolved while no LICENSE file/API license is detected");
    }
    if str_at(manifest,&["upstream","repository_metadata","readme_license_claim"])!=Some("MIT") {
        fail(&mut errors,"upstream README license claim must remain distinct from repository license detection");
    }

    for path in [
        &["custody","scenario_payload_downloaded"][..],
        &["custody","result_payloads_downloaded"][..],
        &["custody","model_payloads_downloaded"][..],
        &["custody","participant_level_data_observed"][..],
        &["custody","model_execution_performed"][..],
        &["custody","upstream_files_mirrored"][..],
    ] {
        if bool_at(manifest,path)!=Some(false) {
            fail(&mut errors,format!("metadata-only custody invariant violated at {}",path.join(".")));
        }
    }
    if str_at(manifest,&["custody","rights_admission"])!=Some("blocked_unresolved")
        || str_at(manifest,&["custody","data_use_status"])!=Some("metadata_only") {
        fail(&mut errors,"custody status must remain metadata_only / rights blocked");
    }
    if str_at(manifest,&["disposition","overall"])!=Some("metadata_only_blocked") {
        fail(&mut errors,"overall disposition cannot be promoted without independent review and reproduction");
    }
    if !errors.is_empty() { Err(errors) } else { Ok(()) }
}
fn main_run(args: &[String]) -> Result<(),String> {
    if args.len()!=2 { return Err("usage: qualifier MANIFEST MANIFEST_SCHEMA".into()); }
    let manifest:Value=serde_json::from_str(&fs::read_to_string(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let schema:Value=serde_json::from_str(&fs::read_to_string(&args[1]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let schema_errors=check_schema_header(&schema);
    if !schema_errors.is_empty() { return Err(schema_errors.join("; ")); }
    validate(&manifest).map_err(|e|e.join("; "))?;
    println!("METADATA_ONLY_BLOCKED: upstream_commit={} scenarios=100 domains=10 authority_levels=5 assigned_private_dissenters=16 supporters=4",UPSTREAM_COMMIT);
    println!("BLOCKERS: license_unresolved; salted_hash_seed; neutral_unmatched_scoring; validation_artifacts_unreconciled; dataset_schema_cast_error");
    println!("CUSTODY: payload_downloaded=false model_execution=false rights_admission=blocked");
    Ok(())
}
fn main() -> ExitCode {
    match main_run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(())=>ExitCode::SUCCESS,
        Err(e)=>{eprintln!("NOT_QUALIFIED: {e}");ExitCode::FAILURE}
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn manifest()->Value {
        serde_json::from_str(include_str!("../../PI_BASELINE_INTAKE_V0_1.json")).unwrap()
    }
    #[test]
    fn current_upstream_snapshot_is_pinned() { assert_eq!(validate(&manifest()),Ok(())); }
    #[test]
    fn license_cannot_be_promoted_without_a_license_file() {
        let mut m=manifest();
        m["upstream"]["repository_metadata"]["code_reuse_status"]=Value::String("admitted".into());
        assert!(validate(&m).is_err());
    }
    #[test]
    fn data_payload_download_is_forbidden_in_metadata_only_intake() {
        let mut m=manifest();
        m["custody"]["scenario_payload_downloaded"]=Value::Bool(true);
        assert!(validate(&m).is_err());
    }
    #[test]
    fn model_execution_cannot_be_claimed_as_already_run() {
        let mut m=manifest();
        m["custody"]["model_execution_performed"]=Value::Bool(true);
        assert!(validate(&m).is_err());
    }
    #[test]
    fn salted_python_hash_cannot_be_reported_as_reproducible() {
        let mut m=manifest();
        m["seed_audit"]["deterministic_speaker_order_across_fresh_processes_guaranteed"]=Value::Bool(true);
        assert!(validate(&m).is_err());
    }
    #[test]
    fn unmatched_keyword_text_cannot_be_treated_as_a_trusted_neutral_label() {
        let mut m=manifest();
        m["scorer_audit"]["neutral_or_unmatched_counted_as_conformity"]=Value::Bool(false);
        assert!(validate(&m).is_err());
    }
    #[test]
    fn validation_artifact_discrepancy_cannot_be_silently_closed() {
        let mut m=manifest();
        m["validation_audit"]["reconciliation_status"]=Value::String("reconciled".into());
        assert!(validate(&m).is_err());
    }
    #[test]
    fn broken_dataset_viewer_blocks_dataset_admission() {
        let mut m=manifest();
        m["dataset_audit"]["dataset_admission_status"]=Value::String("admitted".into());
        assert!(validate(&m).is_err() || m["custody"]["rights_admission"].as_str()!=Some("admitted"));
    }
    #[test]
    fn dataset_viewer_cast_error_cannot_be_marked_admitted() {
        let mut m=manifest();
        m["dataset_audit"]["dataset_admission_status"]=Value::String("admitted".into());
        assert!(validate(&m).is_err());
    }
    #[test]
    fn scorer_validation_artifacts_cannot_be_marked_reconciled() {
        let mut m=manifest();
        m["validation_audit"]["reconciliation_status"]=Value::String("reconciled".into());
        assert!(validate(&m).is_err());
    }
    #[test]
    fn synthetic_80_20_assignment_is_not_a_human_population_prior() {
        assert!(manifest()["reported_protocol"]["private_state_semantics"].as_str().unwrap().contains("not a human population prior"));
    }
    #[test]
    fn repo_license_claim_is_not_equal_to_detected_repo_license() {
        let m=manifest();
        assert_eq!(m["upstream"]["repository_metadata"]["readme_license_claim"],"MIT");
        assert!(m["upstream"]["repository_metadata"]["repository_api_detected_license"].is_null());
    }
}
