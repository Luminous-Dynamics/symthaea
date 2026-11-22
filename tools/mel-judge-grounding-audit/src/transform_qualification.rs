use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-audio-transform-qualification-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformPanelV1 {
    schema_version: String,
    experiment_id: String,
    frozen_at_utc: String,
    task: TaskV1,
    policy: EquivalencePolicyV1,
    source: AudioAssetV1,
    variant: AudioAssetV1,
    transformation: TransformationV1,
    rights: RightsScopeV1,
    equivalence: EquivalenceReceiptV1,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskV1 {
    task_id: String,
    task_dimension: String,
    prompt_sha256: String,
    rubric_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EquivalencePolicyV1 {
    confidence_level: f64,
    minimum_listeners: usize,
    minimum_items: usize,
    equivalence_margin: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AudioAssetV1 {
    asset_id: String,
    sha256: String,
    sample_rate_hz: u32,
    channels: u8,
    duration_frames: u64,
    encoding: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransformationV1 {
    family: String,
    algorithm: String,
    implementation_sha256: String,
    parameters_sha256: String,
    intended_preserved_dimensions: Vec<String>,
    may_alter_dimensions: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RightsScopeV1 {
    purpose: String,
    rights_receipt_sha256: String,
    consent_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EquivalenceReceiptV1 {
    outcome: String,
    method: String,
    panel_protocol_sha256: String,
    cohort_commitment_sha256: String,
    analysis_plan_sha256: String,
    independence_attestation_sha256: String,
    listener_count: usize,
    item_count: usize,
    confidence_level: f64,
    mean_difference: f64,
    confidence_interval_lower: f64,
    confidence_interval_upper: f64,
    equivalence_margin: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransformQualificationReport {
    pub schema_version: String,
    pub experiment_id: String,
    pub task_id: String,
    pub task_dimension: String,
    pub transform_family: String,
    pub source_sha256_recomputed: String,
    pub variant_sha256_recomputed: String,
    pub listener_count: usize,
    pub item_count: usize,
    pub confidence_level: f64,
    pub observed_mean_difference: f64,
    pub confidence_interval: [f64; 2],
    pub equivalence_margin: f64,
    pub eligible_for_declared_task_dimension: bool,
    pub ineligibility_reasons: Vec<String>,
    pub claim_ceiling: Vec<String>,
}

pub fn parse_and_qualify(
    input: &str,
    source_path: &Path,
    variant_path: &Path,
) -> Result<TransformQualificationReport, String> {
    let panel: TransformPanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid transform qualification JSON: {error}"))?;
    validate_panel(&panel)?;

    let source_digest = sha256sum(source_path)?;
    if source_digest != panel.source.sha256 {
        return Err(format!(
            "source file digest mismatch: manifest {}, recomputed {}",
            panel.source.sha256, source_digest
        ));
    }

    let variant_digest = sha256sum(variant_path)?;
    if variant_digest != panel.variant.sha256 {
        return Err(format!(
            "variant file digest mismatch: manifest {}, recomputed {}",
            panel.variant.sha256, variant_digest
        ));
    }

    let mut reasons = Vec::new();
    let receipt = &panel.equivalence;
    if receipt.outcome != "equivalent" {
        reasons.push(format!("independent panel outcome is {}", receipt.outcome));
    }
    if receipt.listener_count < panel.policy.minimum_listeners {
        reasons.push(format!(
            "underpowered listener panel: {} < {}",
            receipt.listener_count, panel.policy.minimum_listeners
        ));
    }
    if receipt.item_count < panel.policy.minimum_items {
        reasons.push(format!(
            "underpowered item panel: {} < {}",
            receipt.item_count, panel.policy.minimum_items
        ));
    }
    let margin = panel.policy.equivalence_margin;
    if receipt.confidence_interval_lower <= -margin || receipt.confidence_interval_upper >= margin {
        reasons.push(
            "confidence interval is not wholly inside the preregistered equivalence bounds"
                .to_owned(),
        );
    }

    Ok(TransformQualificationReport {
        schema_version: "mel-audio-transform-qualification-report-v1".to_owned(),
        experiment_id: panel.experiment_id,
        task_id: panel.task.task_id,
        task_dimension: panel.task.task_dimension,
        transform_family: panel.transformation.family,
        source_sha256_recomputed: source_digest,
        variant_sha256_recomputed: variant_digest,
        listener_count: receipt.listener_count,
        item_count: receipt.item_count,
        confidence_level: receipt.confidence_level,
        observed_mean_difference: receipt.mean_difference,
        confidence_interval: [
            receipt.confidence_interval_lower,
            receipt.confidence_interval_upper,
        ],
        equivalence_margin: margin,
        eligible_for_declared_task_dimension: reasons.is_empty(),
        ineligibility_reasons: reasons,
        claim_ceiling: panel.claim_ceiling,
    })
}

fn validate_panel(panel: &TransformPanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION
        || panel.experiment_id.trim().is_empty()
        || !is_calendar_valid_utc(&panel.frozen_at_utc)
    {
        return Err("schema_version, experiment_id or frozen_at_utc is invalid".to_owned());
    }
    require_text(&panel.task.task_id, "task_id")?;
    require_text(&panel.task.task_dimension, "task_dimension")?;
    require_digest(&panel.task.prompt_sha256, "task prompt digest")?;
    require_digest(&panel.task.rubric_sha256, "task rubric digest")?;

    let policy = &panel.policy;
    if !policy.confidence_level.is_finite()
        || !(0.90..=0.99).contains(&policy.confidence_level)
        || policy.minimum_listeners == 0
        || policy.minimum_items == 0
        || !policy.equivalence_margin.is_finite()
        || policy.equivalence_margin <= 0.0
    {
        return Err("invalid preregistered equivalence policy".to_owned());
    }

    validate_asset(&panel.source, "source")?;
    validate_asset(&panel.variant, "variant")?;
    if panel.source.asset_id == panel.variant.asset_id {
        return Err("source and variant asset_id values must be distinct".to_owned());
    }

    let transform = &panel.transformation;
    require_text(&transform.family, "transform family")?;
    require_text(&transform.algorithm, "transform algorithm")?;
    require_digest(
        &transform.implementation_sha256,
        "transform implementation digest",
    )?;
    require_digest(&transform.parameters_sha256, "transform parameter digest")?;
    validate_dimension_set(
        &transform.intended_preserved_dimensions,
        "intended_preserved_dimensions",
    )?;
    validate_dimension_set(&transform.may_alter_dimensions, "may_alter_dimensions")?;

    let preserved: BTreeSet<&str> = transform
        .intended_preserved_dimensions
        .iter()
        .map(String::as_str)
        .collect();
    let altered: BTreeSet<&str> = transform
        .may_alter_dimensions
        .iter()
        .map(String::as_str)
        .collect();
    if !preserved.contains(panel.task.task_dimension.as_str()) {
        return Err("task_dimension is not declared as preserved".to_owned());
    }
    if altered.contains(panel.task.task_dimension.as_str()) {
        return Err("task_dimension is also declared as possibly altered".to_owned());
    }
    if preserved
        .iter()
        .any(|dimension| altered.contains(dimension))
    {
        return Err("a dimension cannot be both preserved and possibly altered".to_owned());
    }

    if panel.rights.purpose != "music-judge-qualification" {
        return Err("rights purpose must be music-judge-qualification".to_owned());
    }
    require_digest(&panel.rights.rights_receipt_sha256, "rights receipt digest")?;
    require_digest(
        &panel.rights.consent_receipt_sha256,
        "consent receipt digest",
    )?;

    let receipt = &panel.equivalence;
    if !matches!(
        receipt.outcome.as_str(),
        "equivalent" | "not_equivalent" | "inconclusive"
    ) {
        return Err("unsupported equivalence outcome".to_owned());
    }
    if !matches!(
        receipt.method.as_str(),
        "paired_cluster_bootstrap" | "crossed_mixed_effects"
    ) {
        return Err("equivalence method must account for listener/item dependence".to_owned());
    }
    require_digest(&receipt.panel_protocol_sha256, "panel protocol digest")?;
    require_digest(
        &receipt.cohort_commitment_sha256,
        "panel cohort commitment digest",
    )?;
    require_digest(&receipt.analysis_plan_sha256, "analysis plan digest")?;
    require_digest(
        &receipt.independence_attestation_sha256,
        "panel independence attestation digest",
    )?;
    for (name, value) in [
        ("receipt confidence level", receipt.confidence_level),
        ("mean difference", receipt.mean_difference),
        (
            "confidence interval lower",
            receipt.confidence_interval_lower,
        ),
        (
            "confidence interval upper",
            receipt.confidence_interval_upper,
        ),
        ("receipt equivalence margin", receipt.equivalence_margin),
    ] {
        if !value.is_finite() {
            return Err(format!("{name} must be finite"));
        }
    }
    if receipt.confidence_interval_lower > receipt.confidence_interval_upper
        || receipt.mean_difference < receipt.confidence_interval_lower
        || receipt.mean_difference > receipt.confidence_interval_upper
    {
        return Err("confidence interval must be ordered and contain the observed mean".to_owned());
    }
    if (receipt.confidence_level - policy.confidence_level).abs() > 1e-12
        || (receipt.equivalence_margin - policy.equivalence_margin).abs() > 1e-12
    {
        return Err("human receipt does not match the preregistered policy".to_owned());
    }

    if panel.claim_ceiling.is_empty()
        || panel
            .claim_ceiling
            .iter()
            .any(|claim| claim.trim().is_empty())
    {
        return Err("claim_ceiling must contain non-empty limits".to_owned());
    }
    Ok(())
}

fn validate_asset(asset: &AudioAssetV1, name: &str) -> Result<(), String> {
    require_text(&asset.asset_id, &format!("{name} asset_id"))?;
    require_digest(&asset.sha256, &format!("{name} audio digest"))?;
    if !(8_000..=384_000).contains(&asset.sample_rate_hz)
        || !(1..=8).contains(&asset.channels)
        || asset.duration_frames == 0
    {
        return Err(format!("{name} audio metadata is outside supported bounds"));
    }
    require_text(&asset.encoding, &format!("{name} encoding"))
}

fn validate_dimension_set(values: &[String], name: &str) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    let mut unique = BTreeSet::new();
    for value in values {
        require_text(value, name)?;
        if !unique.insert(value.as_str()) {
            return Err(format!("{name} contains a duplicate dimension"));
        }
    }
    Ok(())
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "{name} must be a 64-character lowercase SHA-256 digest"
        ))
    }
}

fn sha256sum(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg("--binary")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|error| {
            format!(
                "failed to execute sha256sum for {}: {error}",
                path.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "sha256sum failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("sha256sum returned non-UTF-8 output: {error}"))?;
    let digest = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| "sha256sum returned empty output".to_owned())?;
    require_digest(digest, "recomputed SHA-256 digest")?;
    Ok(digest.to_owned())
}

fn is_calendar_valid_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return false;
    }
    for (index, byte) in bytes.iter().enumerate() {
        if !matches!(index, 4 | 7 | 10 | 13 | 16 | 19) && !byte.is_ascii_digit() {
            return false;
        }
    }
    let year = value[0..4].parse::<u32>().unwrap_or(0);
    let month = value[5..7].parse::<u32>().unwrap_or(0);
    let day = value[8..10].parse::<u32>().unwrap_or(0);
    let hour = value[11..13].parse::<u32>().unwrap_or(99);
    let minute = value[14..16].parse::<u32>().unwrap_or(99);
    let second = value[17..19].parse::<u32>().unwrap_or(99);
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year > 0 && day > 0 && day <= days_in_month && hour < 24 && minute < 60 && second < 60
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-transform-qualification-v1.json");

    fn fixture_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        (
            root.join("fixtures/source-audio.bin"),
            root.join("fixtures/variant-audio.bin"),
        )
    }

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known transform fixture JSON")
    }

    fn qualify(input: &str) -> TransformQualificationReport {
        let (source, variant) = fixture_paths();
        parse_and_qualify(input, &source, &variant).expect("transform panel should validate")
    }

    #[test]
    fn known_answer_receipt_is_eligible_only_for_its_declared_task_dimension() {
        let report = qualify(FIXTURE);
        assert!(report.eligible_for_declared_task_dimension);
        assert_eq!(report.task_dimension, "arrangement_preference");
        assert_eq!(report.listener_count, 24);
        assert_eq!(report.item_count, 40);
        assert!(report.ineligibility_reasons.is_empty());
    }

    #[test]
    fn underpowered_panel_is_ineligible_not_a_pass() {
        let mut value = panel();
        value["equivalence"]["listener_count"] = serde_json::json!(8);
        let report = qualify(&value.to_string());
        assert!(!report.eligible_for_declared_task_dimension);
        assert!(
            report
                .ineligibility_reasons
                .iter()
                .any(|reason| reason.contains("underpowered listener"))
        );
    }

    #[test]
    fn confidence_interval_must_fit_inside_the_equivalence_margin() {
        let mut value = panel();
        value["equivalence"]["confidence_interval_upper"] = serde_json::json!(0.31);
        let report = qualify(&value.to_string());
        assert!(!report.eligible_for_declared_task_dimension);
        assert!(
            report
                .ineligibility_reasons
                .iter()
                .any(|reason| reason.contains("confidence interval"))
        );
    }

    #[test]
    fn non_equivalent_panel_result_is_ineligible() {
        let mut value = panel();
        value["equivalence"]["outcome"] = serde_json::json!("not_equivalent");
        let report = qualify(&value.to_string());
        assert!(!report.eligible_for_declared_task_dimension);
    }

    #[test]
    fn task_dimension_cannot_be_marked_as_altered() {
        let mut value = panel();
        value["transformation"]["may_alter_dimensions"] =
            serde_json::json!(["arrangement_preference"]);
        let (source, variant) = fixture_paths();
        let error = parse_and_qualify(&value.to_string(), &source, &variant)
            .expect_err("task-relevant transform must not be admitted as nuisance");
        assert!(error.contains("task_dimension is also declared"));
    }

    #[test]
    fn task_dimension_must_be_in_preserved_set() {
        let mut value = panel();
        value["transformation"]["intended_preserved_dimensions"] = serde_json::json!(["timbre"]);
        let (source, variant) = fixture_paths();
        let error = parse_and_qualify(&value.to_string(), &source, &variant)
            .expect_err("missing task dimension must fail closed");
        assert!(error.contains("task_dimension is not declared as preserved"));
    }

    #[test]
    fn rights_purpose_is_scoped_to_music_judge_qualification() {
        let mut value = panel();
        value["rights"]["purpose"] = serde_json::json!("training");
        let (source, variant) = fixture_paths();
        let error = parse_and_qualify(&value.to_string(), &source, &variant)
            .expect_err("other purposes must not pass");
        assert!(error.contains("rights purpose"));
    }

    #[test]
    fn tampered_source_bytes_are_rejected_by_recomputed_digest() {
        let value = panel();
        let (_, variant) = fixture_paths();
        let error = parse_and_qualify(&value.to_string(), &variant, &variant)
            .expect_err("source and variant bytes are not interchangeable");
        assert!(error.contains("source file digest mismatch"));
    }

    #[test]
    fn invalid_calendar_date_is_rejected() {
        let mut value = panel();
        value["frozen_at_utc"] = serde_json::json!("2026-02-30T10:00:00Z");
        let (source, variant) = fixture_paths();
        let error = parse_and_qualify(&value.to_string(), &source, &variant)
            .expect_err("invalid date must fail");
        assert!(error.contains("frozen_at_utc"));
    }
}
