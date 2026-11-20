//! Cross-layer, non-compensating verification for MUSE evidence bundles v1.
//!
//! The verifier consumes sealed component receipts and the render artifact
//! directory. It delegates receipt cryptography and domain checks to the
//! existing public render, symbolic and acoustic APIs. Score-to-audio and
//! human-acceptance components remain abstentions until their own public
//! verification APIs are registered.

use muse_render_receipt::{verify_bundle as verify_render_bundle, verify_receipt_bytes};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use symthaea_muse::{
    audio_edit_observer::verify_gain_envelope_observation_seal,
    edit_evidence::verify_result_seal,
};

pub const BUNDLE_SCHEMA_V1: &str = "muse-evidence-bundle-v1";
pub const BUNDLE_DIGEST_RECIPE_V1: &str = "muse-evidence-bundle-sorted-json-sha256-v1";
pub const BUNDLE_DIGEST_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-evidence-bundle-v1";
pub const BUNDLE_VERIFIER_ID_V1: &str = "muse-evidence-bundle-verifier";
pub const BUNDLE_VERIFIER_VERSION_V1: &str = "0.1.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatusV1 {
    Pass,
    Fail,
    Abstain,
    NotRun,
}

impl GateStatusV1 {
    fn effective(self, observed: Self) -> Self {
        use GateStatusV1::{Abstain, Fail, NotRun, Pass};
        match (self, observed) {
            (Fail, _) | (_, Fail) => Fail,
            (Abstain | NotRun, _) | (_, Abstain | NotRun) => Abstain,
            (Pass, Pass) => Pass,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Abstain => "abstain",
            Self::NotRun => "not_run",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleStatusV1 {
    Pass,
    Fail,
    Abstain,
}

impl BundleStatusV1 {
    fn from_gates(gates: &[GateStatusV1]) -> Self {
        if gates.iter().any(|status| *status == GateStatusV1::Fail) {
            Self::Fail
        } else if gates.iter().any(|status| {
            matches!(*status, GateStatusV1::Abstain | GateStatusV1::NotRun)
        }) {
            Self::Abstain
        } else {
            Self::Pass
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleVerifierIdentityV1 {
    pub verifier_id: String,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleArtifactsV1 {
    pub case_sha256: String,
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
    pub source_audio_sha256: String,
    pub candidate_audio_sha256: String,
    pub candidate_midi_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentBaseV1 {
    pub status: GateStatusV1,
    pub receipt_sha256: Option<String>,
    pub producer_id: Option<String>,
    pub producer_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicComponentV1 {
    #[serde(flatten)]
    pub base: ComponentBaseV1,
    pub case_sha256: String,
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderComponentV1 {
    #[serde(flatten)]
    pub base: ComponentBaseV1,
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
    pub candidate_midi_sha256: String,
    pub candidate_audio_sha256: String,
    pub attempt_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticComponentV1 {
    #[serde(flatten)]
    pub base: ComponentBaseV1,
    pub contract_sha256: Option<String>,
    pub source_audio_sha256: String,
    pub candidate_audio_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreAudioComponentV1 {
    #[serde(flatten)]
    pub base: ComponentBaseV1,
    pub candidate_score_sha256: String,
    pub candidate_audio_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HumanComponentV1 {
    #[serde(flatten)]
    pub base: ComponentBaseV1,
    pub case_sha256: String,
    pub candidate_score_sha256: String,
    pub candidate_audio_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleComponentsV1 {
    pub symbolic_edit: SymbolicComponentV1,
    pub render_execution: RenderComponentV1,
    pub acoustic_observation: AcousticComponentV1,
    pub score_audio_conformance: ScoreAudioComponentV1,
    pub human_acceptance: HumanComponentV1,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredOutcomeV1 {
    pub status: BundleStatusV1,
    pub required_gate_count: u8,
    pub passed_gate_count: u8,
    pub failed_gate_count: u8,
    pub abstained_gate_count: u8,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifestV1 {
    pub schema: String,
    pub bundle_id: String,
    pub verifier: BundleVerifierIdentityV1,
    pub digest_recipe: String,
    pub artifacts: BundleArtifactsV1,
    pub components: BundleComponentsV1,
    pub outcome: DeclaredOutcomeV1,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, Default)]
pub struct EvidenceInputsV1<'a> {
    pub symbolic_result_json: Option<&'a [u8]>,
    pub render_receipt_json: Option<&'a [u8]>,
    pub render_bundle_root: Option<&'a Path>,
    pub acoustic_result_json: Option<&'a [u8]>,
    /// Reserved for the v1 score-to-audio receipt. No registered public
    /// verifier exists yet, so presence alone cannot authorize a pass.
    pub score_audio_result_json: Option<&'a [u8]>,
    /// Reserved for the v1 human-acceptance receipt. No registered public
    /// verifier exists yet, so presence alone cannot authorize a pass.
    pub human_acceptance_json: Option<&'a [u8]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentReportV1 {
    pub declared_status: GateStatusV1,
    pub verified_status: GateStatusV1,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleVerificationReportV1 {
    pub schema: String,
    pub bundle_id: String,
    pub manifest_receipt_verified: bool,
    pub declared_status: BundleStatusV1,
    pub verified_status: BundleStatusV1,
    pub passed_gate_count: u8,
    pub failed_gate_count: u8,
    pub abstained_gate_count: u8,
    pub components: BundleReportComponentsV1,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleReportComponentsV1 {
    pub symbolic_edit: ComponentReportV1,
    pub render_execution: ComponentReportV1,
    pub acoustic_observation: ComponentReportV1,
    pub score_audio_conformance: ComponentReportV1,
    pub human_acceptance: ComponentReportV1,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    fn normalize(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort_unstable();
                let mut sorted = Map::new();
                for key in keys {
                    let child = object.get(key).expect("key came from this object");
                    sorted.insert(key.clone(), normalize(child));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&normalize(value))
        .map_err(|error| format!("bundle canonical JSON serialization failed: {error}"))
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

fn verify_manifest_seal(value: &Value, recorded: &str) -> Result<(), String> {
    if !is_sha256(recorded) {
        return Err("bundle receipt_sha256 is not lowercase SHA-256".into());
    }
    let mut unsealed = value.clone();
    let object = unsealed
        .as_object_mut()
        .ok_or("bundle manifest root must be an object")?;
    object.insert("receipt_sha256".into(), Value::Null);
    let actual = domain_digest(BUNDLE_DIGEST_DOMAIN_V1, &canonical_json_bytes(&unsealed)?);
    if actual != recorded {
        return Err(format!("bundle receipt seal mismatch: recorded {recorded}, recomputed {actual}"));
    }
    Ok(())
}

fn exact_fields(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let mut actual: Vec<&str> = object.keys().map(String::as_str).collect();
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    if actual != expected {
        let missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|field| !actual.contains(field))
            .collect();
        let extra: Vec<&str> = actual
            .iter()
            .copied()
            .filter(|field| !expected.contains(field))
            .collect();
        return Err(format!("{label} field set mismatch; missing={missing:?}, extra={extra:?}"));
    }
    Ok(())
}

fn validate_manifest_shape(value: &Value) -> Result<(), String> {
    exact_fields(
        value,
        &[
            "schema",
            "bundle_id",
            "verifier",
            "digest_recipe",
            "artifacts",
            "components",
            "outcome",
            "receipt_sha256",
        ],
        "bundle",
    )?;
    exact_fields(value.get("verifier").ok_or("bundle.verifier missing")?, &["verifier_id", "version"], "verifier")?;
    exact_fields(
        value.get("artifacts").ok_or("bundle.artifacts missing")?,
        &[
            "case_sha256",
            "source_score_sha256",
            "candidate_score_sha256",
            "source_audio_sha256",
            "candidate_audio_sha256",
            "candidate_midi_sha256",
        ],
        "artifacts",
    )?;
    let components = value.get("components").ok_or("bundle.components missing")?;
    exact_fields(
        components,
        &[
            "symbolic_edit",
            "render_execution",
            "acoustic_observation",
            "score_audio_conformance",
            "human_acceptance",
        ],
        "components",
    )?;
    let component_fields = [
        (
            "symbolic_edit",
            &[
                "status", "receipt_sha256", "producer_id", "producer_version",
                "case_sha256", "source_score_sha256", "candidate_score_sha256",
            ][..],
        ),
        (
            "render_execution",
            &[
                "status", "receipt_sha256", "producer_id", "producer_version",
                "source_score_sha256", "candidate_score_sha256", "candidate_midi_sha256",
                "candidate_audio_sha256", "attempt_id",
            ][..],
        ),
        (
            "acoustic_observation",
            &[
                "status", "receipt_sha256", "producer_id", "producer_version",
                "contract_sha256", "source_audio_sha256", "candidate_audio_sha256",
            ][..],
        ),
        (
            "score_audio_conformance",
            &[
                "status", "receipt_sha256", "producer_id", "producer_version",
                "candidate_score_sha256", "candidate_audio_sha256",
            ][..],
        ),
        (
            "human_acceptance",
            &[
                "status", "receipt_sha256", "producer_id", "producer_version",
                "case_sha256", "candidate_score_sha256", "candidate_audio_sha256",
            ][..],
        ),
    ];
    for (name, fields) in component_fields {
        exact_fields(
            components.get(name).ok_or_else(|| format!("components.{name} missing"))?,
            fields,
            &format!("components.{name}"),
        )?;
    }
    exact_fields(
        value.get("outcome").ok_or("bundle.outcome missing")?,
        &[
            "status",
            "required_gate_count",
            "passed_gate_count",
            "failed_gate_count",
            "abstained_gate_count",
            "reasons",
        ],
        "outcome",
    )?;
    Ok(())
}

fn validate_component_base(label: &str, base: &ComponentBaseV1) -> Result<(), String> {
    if let Some(digest) = base.receipt_sha256.as_deref() {
        if !is_sha256(digest) {
            return Err(format!("{label} receipt_sha256 is invalid"));
        }
    }
    if matches!(base.status, GateStatusV1::Pass)
        && (base.receipt_sha256.is_none()
            || base.producer_id.as_deref().unwrap_or("").trim().is_empty()
            || base.producer_version.as_deref().unwrap_or("").trim().is_empty())
    {
        return Err(format!("{label} cannot declare pass without a receipt digest and producer identity/version"));
    }
    Ok(())
}

fn validate_hashes(manifest: &BundleManifestV1) -> Result<(), String> {
    let values = [
        ("case_sha256", manifest.artifacts.case_sha256.as_str()),
        ("source_score_sha256", manifest.artifacts.source_score_sha256.as_str()),
        ("candidate_score_sha256", manifest.artifacts.candidate_score_sha256.as_str()),
        ("source_audio_sha256", manifest.artifacts.source_audio_sha256.as_str()),
        ("candidate_audio_sha256", manifest.artifacts.candidate_audio_sha256.as_str()),
        ("candidate_midi_sha256", manifest.artifacts.candidate_midi_sha256.as_str()),
        ("symbolic receipt", manifest.components.symbolic_edit.base.receipt_sha256.as_deref().unwrap_or("")),
        ("render receipt", manifest.components.render_execution.base.receipt_sha256.as_deref().unwrap_or("")),
        ("acoustic receipt", manifest.components.acoustic_observation.base.receipt_sha256.as_deref().unwrap_or("")),
        ("score-audio receipt", manifest.components.score_audio_conformance.base.receipt_sha256.as_deref().unwrap_or("")),
        ("human receipt", manifest.components.human_acceptance.base.receipt_sha256.as_deref().unwrap_or("")),
    ];
    for (label, digest) in values {
        if !digest.is_empty() && !is_sha256(digest) {
            return Err(format!("{label} is not lowercase SHA-256"));
        }
    }
    for (label, digest) in [
        ("symbolic case", manifest.components.symbolic_edit.case_sha256.as_str()),
        ("symbolic source Score", manifest.components.symbolic_edit.source_score_sha256.as_str()),
        ("symbolic candidate Score", manifest.components.symbolic_edit.candidate_score_sha256.as_str()),
        ("render source Score", manifest.components.render_execution.source_score_sha256.as_str()),
        ("render candidate Score", manifest.components.render_execution.candidate_score_sha256.as_str()),
        ("render candidate MIDI", manifest.components.render_execution.candidate_midi_sha256.as_str()),
        ("render candidate WAV", manifest.components.render_execution.candidate_audio_sha256.as_str()),
        ("acoustic source WAV", manifest.components.acoustic_observation.source_audio_sha256.as_str()),
        ("acoustic candidate WAV", manifest.components.acoustic_observation.candidate_audio_sha256.as_str()),
        ("score-audio candidate Score", manifest.components.score_audio_conformance.candidate_score_sha256.as_str()),
        ("score-audio candidate WAV", manifest.components.score_audio_conformance.candidate_audio_sha256.as_str()),
        ("human case", manifest.components.human_acceptance.case_sha256.as_str()),
        ("human candidate Score", manifest.components.human_acceptance.candidate_score_sha256.as_str()),
        ("human candidate WAV", manifest.components.human_acceptance.candidate_audio_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            return Err(format!("{label} binding is not lowercase SHA-256"));
        }
    }
    if let Some(digest) = manifest.components.acoustic_observation.contract_sha256.as_deref() {
        if !is_sha256(digest) {
            return Err("acoustic contract binding is not lowercase SHA-256".into());
        }
    }
    for (label, component) in [
        ("symbolic_edit", &manifest.components.symbolic_edit.base),
        ("render_execution", &manifest.components.render_execution.base),
        ("acoustic_observation", &manifest.components.acoustic_observation.base),
        ("score_audio_conformance", &manifest.components.score_audio_conformance.base),
        ("human_acceptance", &manifest.components.human_acceptance.base),
    ] {
        validate_component_base(label, component)?;
    }
    if manifest.components.render_execution.base.status == GateStatusV1::Pass
        && manifest.components.render_execution.attempt_id.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err("render_execution cannot declare pass without an attempt ID".into());
    }
    if manifest.components.acoustic_observation.base.status == GateStatusV1::Pass
        && manifest.components.acoustic_observation.contract_sha256.is_none()
    {
        return Err("acoustic_observation cannot declare pass without a contract digest".into());
    }
    Ok(())
}

fn declared_components_valid(manifest: &BundleManifestV1) -> Result<(), String> {
    if manifest.schema != BUNDLE_SCHEMA_V1
        || manifest.digest_recipe != BUNDLE_DIGEST_RECIPE_V1
        || manifest.verifier.verifier_id != BUNDLE_VERIFIER_ID_V1
        || manifest.verifier.version.is_empty()
        || manifest.verifier.version.len() > 64
        || manifest.bundle_id.trim().is_empty()
        || manifest.bundle_id.len() > 128
        || manifest.outcome.reasons.iter().any(|reason| reason.len() > 2048)
    {
        return Err("unsupported bundle schema, digest recipe, verifier identity or empty bundle ID".into());
    }
    let components = [
        manifest.components.symbolic_edit.base.status,
        manifest.components.render_execution.base.status,
        manifest.components.acoustic_observation.base.status,
        manifest.components.score_audio_conformance.base.status,
        manifest.components.human_acceptance.base.status,
    ];
    let passed = components.iter().filter(|status| **status == GateStatusV1::Pass).count() as u8;
    let failed = components.iter().filter(|status| **status == GateStatusV1::Fail).count() as u8;
    let abstained = components
        .iter()
        .filter(|status| matches!(**status, GateStatusV1::Abstain | GateStatusV1::NotRun))
        .count() as u8;
    if manifest.outcome.required_gate_count != 5
        || manifest.outcome.passed_gate_count != passed
        || manifest.outcome.failed_gate_count != failed
        || manifest.outcome.abstained_gate_count != abstained
        || manifest.outcome.status != BundleStatusV1::from_gates(&components)
    {
        return Err("declared outcome/status counts do not match the five component declarations".into());
    }
    Ok(())
}

fn hard_binding(reason: impl Into<String>) -> ComponentReportV1 {
    ComponentReportV1 {
        declared_status: GateStatusV1::Fail,
        verified_status: GateStatusV1::Fail,
        reasons: vec![reason.into()],
    }
}

fn gate_report(
    declared_status: GateStatusV1,
    observed_status: GateStatusV1,
    reason: Option<String>,
) -> ComponentReportV1 {
    let effective = declared_status.effective(observed_status);
    ComponentReportV1 {
        declared_status,
        verified_status: effective,
        reasons: reason.into_iter().collect(),
    }
}

fn parse_optional_json(bytes: Option<&[u8]>, label: &str) -> Result<Option<Value>, String> {
    bytes.map(|bytes| {
        serde_json::from_slice(bytes).map_err(|error| format!("{label} JSON is invalid: {error}"))
    }).transpose()
}

fn verify_symbolic(
    manifest: &BundleManifestV1,
    bytes: Option<&[u8]>,
) -> Result<ComponentReportV1, String> {
    let component = &manifest.components.symbolic_edit;
    let artifacts = &manifest.artifacts;
    if component.case_sha256 != artifacts.case_sha256
        || component.source_score_sha256 != artifacts.source_score_sha256
        || component.candidate_score_sha256 != artifacts.candidate_score_sha256
    {
        return Ok(hard_binding("symbolic component case/source/candidate Score binding mismatch"));
    }
    let Some(value) = parse_optional_json(bytes, "symbolic result")? else {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("symbolic result bytes were not supplied".into())));
    };
    verify_result_seal(&value).map_err(|error| format!("symbolic result seal invalid: {error}"))?;
    if value.get("schema").and_then(Value::as_str) != Some("muse-edit-evidence-result-v1") {
        return Err("unsupported symbolic result schema".into());
    }
    let system = value.get("system").ok_or("symbolic result system metadata missing")?;
    let actual_id = system.get("system_id").and_then(Value::as_str).unwrap_or("");
    let actual_version = system.get("version").and_then(Value::as_str).unwrap_or("");
    if component.base.producer_id.as_deref() != Some(actual_id)
        || component.base.producer_version.as_deref() != Some(actual_version)
    {
        return Ok(hard_binding("symbolic producer identity/version does not match the sealed result"));
    }
    if actual_id != "muse-symbolic-edit-evaluator" || actual_version != "0.3.0" {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("symbolic evaluator version is not in the v1 verifier allowlist".into())));
    }
    let result_receipt = value.get("artifacts").and_then(|v| v.get("evaluation_receipt_sha256")).and_then(Value::as_str);
    if result_receipt != component.base.receipt_sha256.as_deref() {
        return Ok(hard_binding("symbolic result receipt digest does not match the bundle reference"));
    }
    let result_artifacts = value.get("artifacts").ok_or("symbolic result artifacts missing")?;
    for (field, expected) in [
        ("case_sha256", artifacts.case_sha256.as_str()),
        ("source_score_sha256", artifacts.source_score_sha256.as_str()),
        ("candidate_score_sha256", artifacts.candidate_score_sha256.as_str()),
    ] {
        if result_artifacts.get(field).and_then(Value::as_str) != Some(expected) {
            return Ok(hard_binding(format!("symbolic result {field} does not match the bundle")));
        }
    }
    if value.get("schema").and_then(Value::as_str) != Some("muse-edit-evidence-result-v1") {
        return Err("unsupported symbolic result schema".into());
    }
    let outcome = value.get("outcome").ok_or("symbolic result outcome missing")?;
    let gates = ["target_pass", "preservation_pass", "correspondence_pass"];
    let failed: Vec<&str> = gates
        .iter()
        .copied()
        .filter(|field| outcome.get(field).and_then(Value::as_bool) == Some(false))
        .collect();
    if !failed.is_empty() {
        return Ok(gate_report(component.base.status, GateStatusV1::Fail, Some(format!("symbolic sub-gates failed: {failed:?}"))));
    }
    let complete = gates
        .iter()
        .all(|field| outcome.get(*field).and_then(Value::as_bool) == Some(true));
    if complete {
        Ok(gate_report(component.base.status, GateStatusV1::Pass, None))
    } else {
        Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("symbolic result lacks one or more required sub-gate verdicts".into())))
    }
}

fn render_evidence_status(verdict: &str) -> GateStatusV1 {
    match verdict {
        "pass" => GateStatusV1::Pass,
        "fail" => GateStatusV1::Fail,
        "abstain" => GateStatusV1::Abstain,
        _ => GateStatusV1::NotRun,
    }
}

fn verify_render_channel_binding(
    label: &str,
    evidence: &muse_render_receipt::EvidenceResultV1,
    component: &ComponentBaseV1,
) -> Result<(), String> {
    if let Some(expected) = component.receipt_sha256.as_deref() {
        if evidence.receipt_sha256.as_deref() != Some(expected) {
            return Err(format!("{label} receipt reference does not match the render receipt channel"));
        }
    } else if component.status == GateStatusV1::Pass {
        return Err(format!("{label} cannot pass without a receipt reference"));
    }
    if component.status == GateStatusV1::Pass && evidence.verdict != "pass" {
        return Err(format!("{label} manifest says pass but render receipt channel is {}", evidence.verdict));
    }
    if component.status == GateStatusV1::Fail && evidence.verdict != "fail" {
        return Err(format!("{label} manifest says fail but render receipt channel is {}", evidence.verdict));
    }
    Ok(())
}

fn verify_render(
    manifest: &BundleManifestV1,
    bytes: Option<&[u8]>,
    bundle_root: Option<&Path>,
) -> Result<ComponentReportV1, String> {
    let component = &manifest.components.render_execution;
    let artifacts = &manifest.artifacts;
    if component.source_score_sha256 != artifacts.source_score_sha256
        || component.candidate_score_sha256 != artifacts.candidate_score_sha256
        || component.candidate_midi_sha256 != artifacts.candidate_midi_sha256
        || component.candidate_audio_sha256 != artifacts.candidate_audio_sha256
    {
        return Ok(hard_binding("render component source/candidate Score, MIDI or WAV binding mismatch"));
    }
    let Some(bytes) = bytes else {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render receipt bytes were not supplied".into())));
    };
    if component.base.producer_id.as_deref() != Some("muse-render-receipt-verifier")
        || component.base.producer_version.as_deref() != Some("0.1.0")
    {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render verifier identity/version is not in the v1 allowlist".into())));
    }
    let receipt = verify_receipt_bytes(bytes)
        .map_err(|error| format!("render receipt failed public verification: {error}"))?;
    if component.base.receipt_sha256.as_deref() != Some(receipt.receipt_sha256.as_str()) {
        return Ok(hard_binding("render receipt digest does not match the bundle reference"));
    }
    if component.attempt_id.as_deref() != Some(receipt.attempt_id.as_str())
        || receipt.source.source_score_sha256 != artifacts.source_score_sha256
        || receipt.source.candidate_score_sha256 != artifacts.candidate_score_sha256
        || receipt.midi.score_sha256 != artifacts.candidate_score_sha256
        || receipt.midi.midi_sha256 != artifacts.candidate_midi_sha256
        || receipt.execution.input_midi_sha256.as_deref() != Some(artifacts.candidate_midi_sha256.as_str())
        || receipt.execution.output_audio_sha256.as_deref() != Some(artifacts.candidate_audio_sha256.as_str())
        || receipt.audio.audio_sha256.as_deref() != Some(artifacts.candidate_audio_sha256.as_str())
    {
        return Ok(hard_binding("render receipt has a source/candidate/MIDI/WAV cross-binding mismatch"));
    }
    for (label, evidence, component) in [
        (
            "protected_audio_preservation",
            &receipt.verification.protected_audio_preservation,
            &manifest.components.acoustic_observation.base,
        ),
        (
            "rendered_score_conformance",
            &receipt.verification.rendered_score_conformance,
            &manifest.components.score_audio_conformance.base,
        ),
        (
            "human_evaluation",
            &receipt.verification.human_evaluation,
            &manifest.components.human_acceptance.base,
        ),
    ] {
        if let Err(error) = verify_render_channel_binding(label, evidence, component) {
            return Ok(hard_binding(error));
        }
    }
    if receipt.attempt_state == "failed" || receipt.execution.state == "failed" {
        return Ok(gate_report(component.base.status, GateStatusV1::Fail, Some("render receipt records a failed attempt".into())));
    }
    if receipt.attempt_state == "outcome_unknown" || receipt.execution.state == "unknown" {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render outcome is unknown and must be reconciled".into())));
    }
    if receipt.attempt_state != "completed" || receipt.execution.state != "completed" {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render attempt is not complete".into())));
    }
    let Some(root) = bundle_root else {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render artifact bundle directory was not supplied; receipt-only validation is insufficient".into())));
    };
    let report = match verify_render_bundle(root, &receipt) {
        Ok(report) => report,
        Err(error) => {
            return Ok(gate_report(component.base.status, GateStatusV1::Fail, Some(format!("render artifact bundle verification failed: {error}"))));
        }
    };
    if !report.bundle_integrity_verified || !report.midi_event_manifest_verified || !report.wav_integrity_verified {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("render integrity verifier did not establish all artifact checks".into())));
    }
    let midi_conformance = render_evidence_status(&receipt.verification.midi_score_conformance.verdict);
    if midi_conformance != GateStatusV1::Pass {
        return Ok(gate_report(component.base.status, midi_conformance, Some("render receipt MIDI/Score conformance channel is not pass".into())));
    }
    // The current render verifier explicitly does not prove trusted executor
    // identity or that an untrusted renderer consumed the declared MIDI. Keep
    // this gate at abstain until a process-tree/runner attestation verifier is
    // registered; byte consistency alone is not execution provenance.
    Ok(gate_report(
        component.base.status,
        GateStatusV1::Abstain,
        Some(format!(
            "artifact integrity verified ({}) but trusted execution/input-consumption attestation is not registered",
            report.trust_level
        )),
    ))
}

fn verify_acoustic(
    manifest: &BundleManifestV1,
    bytes: Option<&[u8]>,
) -> Result<ComponentReportV1, String> {
    let component = &manifest.components.acoustic_observation;
    let artifacts = &manifest.artifacts;
    if component.source_audio_sha256 != artifacts.source_audio_sha256
        || component.candidate_audio_sha256 != artifacts.candidate_audio_sha256
    {
        return Ok(hard_binding("acoustic component source/candidate WAV binding mismatch"));
    }
    let Some(bytes) = bytes else {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("acoustic observation bytes were not supplied".into())));
    };
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("acoustic observation JSON invalid: {error}"))?;
    verify_gain_envelope_observation_seal(&value)
        .map_err(|error| format!("acoustic observation seal invalid: {error}"))?;
    let actual_receipt = value.get("receipt_sha256").and_then(Value::as_str);
    if actual_receipt != component.base.receipt_sha256.as_deref() {
        return Ok(hard_binding("acoustic receipt digest does not match bundle reference"));
    }
    if value.get("schema").and_then(Value::as_str) != Some("muse-acoustic-observation-v1") {
        return Err("unsupported acoustic observation schema".into());
    }
    let observer = value.get("observer").ok_or("acoustic observer metadata missing")?;
    let observer_id = observer.get("observer_id").and_then(Value::as_str).unwrap_or("");
    let observer_version = observer.get("version").and_then(Value::as_str).unwrap_or("");
    if component.base.producer_id.as_deref() != Some(observer_id)
        || component.base.producer_version.as_deref() != Some(observer_version)
    {
        return Ok(hard_binding("acoustic observer identity/version does not match the sealed result"));
    }
    if observer_id != "muse-windowed-rms-gain-observer" || observer_version != "0.1.0" {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("acoustic observer version is not in the v1 verifier allowlist".into())));
    }
    let bound = value.get("artifacts").ok_or("acoustic artifacts missing")?;
    if bound.get("source_audio_sha256").and_then(Value::as_str) != Some(artifacts.source_audio_sha256.as_str())
        || bound.get("candidate_audio_sha256").and_then(Value::as_str) != Some(artifacts.candidate_audio_sha256.as_str())
        || bound.get("contract_sha256").and_then(Value::as_str) != component.contract_sha256.as_deref()
    {
        return Ok(hard_binding("acoustic observation source/candidate/contract digest mismatch"));
    }
    let scope_kind = value.pointer("/scope/kind").and_then(Value::as_str).unwrap_or("");
    if scope_kind == "isolated_stem" {
        return Ok(gate_report(component.base.status, GateStatusV1::Abstain, Some("isolated-stem identity requires a separately verified stem/material lineage receipt".into())));
    }
    if scope_kind != "full_mix" {
        return Ok(gate_report(component.base.status, GateStatusV1::Fail, Some("acoustic observer scope kind is unsupported".into())));
    }
    let status = value.pointer("/outcome/acoustic_observation_status").and_then(Value::as_str).unwrap_or("abstain");
    let observed = match status {
        "pass" => GateStatusV1::Pass,
        "fail" => GateStatusV1::Fail,
        _ => GateStatusV1::Abstain,
    };
    Ok(gate_report(component.base.status, observed, None))
}

fn unsupported_component(
    declared: GateStatusV1,
    receipt_present: bool,
    name: &str,
) -> ComponentReportV1 {
    let reason = if receipt_present {
        format!("{name} receipt was supplied, but no registered public verifier is available")
    } else {
        format!("{name} receipt is not available")
    };
    gate_report(declared, GateStatusV1::Abstain, Some(reason))
}

/// Validate a sealed manifest and independently verify all currently
/// registered component receipts. Score-to-audio and human receipts deliberately
/// remain abstentions until their public verifiers are implemented.
pub fn verify_evidence_bundle_v1(
    manifest_bytes: &[u8],
    inputs: EvidenceInputsV1<'_>,
) -> Result<BundleVerificationReportV1, String> {
    let manifest_value: Value = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("bundle manifest JSON invalid: {error}"))?;
    validate_manifest_shape(&manifest_value)?;
    let manifest: BundleManifestV1 = serde_json::from_value(manifest_value.clone())
        .map_err(|error| format!("bundle manifest wire shape invalid: {error}"))?;
    verify_manifest_seal(&manifest_value, &manifest.receipt_sha256)?;
    declared_components_valid(&manifest)?;
    validate_hashes(&manifest)?;

    let mut reasons = Vec::new();
    let components = BundleReportComponentsV1 {
        symbolic_edit: verify_symbolic(&manifest, inputs.symbolic_result_json)?,
        render_execution: verify_render(&manifest, inputs.render_receipt_json, inputs.render_bundle_root)?,
        acoustic_observation: verify_acoustic(&manifest, inputs.acoustic_result_json)?,
        score_audio_conformance: unsupported_component(
            manifest.components.score_audio_conformance.base.status,
            inputs.score_audio_result_json.is_some(),
            "score-to-audio conformance",
        ),
        human_acceptance: unsupported_component(
            manifest.components.human_acceptance.base.status,
            inputs.human_acceptance_json.is_some(),
            "human acceptance",
        ),
    };
    for (name, component) in [
        ("symbolic_edit", &components.symbolic_edit),
        ("render_execution", &components.render_execution),
        ("acoustic_observation", &components.acoustic_observation),
        ("score_audio_conformance", &components.score_audio_conformance),
        ("human_acceptance", &components.human_acceptance),
    ] {
        for reason in &component.reasons {
            reasons.push(format!("{name}: {reason}"));
        }
    }
    let statuses = [
        components.symbolic_edit.verified_status,
        components.render_execution.verified_status,
        components.acoustic_observation.verified_status,
        components.score_audio_conformance.verified_status,
        components.human_acceptance.verified_status,
    ];
    let verified_status = BundleStatusV1::from_gates(&statuses);
    let passed_gate_count = statuses.iter().filter(|s| **s == GateStatusV1::Pass).count() as u8;
    let failed_gate_count = statuses.iter().filter(|s| **s == GateStatusV1::Fail).count() as u8;
    let abstained_gate_count = statuses.iter()
        .filter(|s| matches!(**s, GateStatusV1::Abstain | GateStatusV1::NotRun))
        .count() as u8;

    if manifest.outcome.status != verified_status {
        reasons.push(format!(
            "manifest declared outcome {} but registered verifiers derive {}",
            manifest.outcome.status.as_str(),
            verified_status.as_str()
        ));
    }
    Ok(BundleVerificationReportV1 {
        schema: "muse-evidence-bundle-verification-report-v1".into(),
        bundle_id: manifest.bundle_id,
        manifest_receipt_verified: true,
        declared_status: manifest.outcome.status,
        verified_status,
        passed_gate_count,
        failed_gate_count,
        abstained_gate_count,
        components,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn sealed_manifest(statuses: [GateStatusV1; 5]) -> Vec<u8> {
        let passed = statuses.iter().filter(|s| **s == GateStatusV1::Pass).count() as u8;
        let failed = statuses.iter().filter(|s| **s == GateStatusV1::Fail).count() as u8;
        let abstained = 5 - passed - failed;
        let status = BundleStatusV1::from_gates(&statuses);
        let value = json!({
            "schema": BUNDLE_SCHEMA_V1,
            "bundle_id": "unit.bundle.001",
            "verifier": {"verifier_id": BUNDLE_VERIFIER_ID_V1, "version": BUNDLE_VERIFIER_VERSION_V1},
            "digest_recipe": BUNDLE_DIGEST_RECIPE_V1,
            "artifacts": {
                "case_sha256": d('a'), "source_score_sha256": d('b'), "candidate_score_sha256": d('c'),
                "source_audio_sha256": d('d'), "candidate_audio_sha256": d('e'), "candidate_midi_sha256": d('f')
            },
            "components": {
                "symbolic_edit": {"status":statuses[0],"receipt_sha256":d('1'),"producer_id":"unit-test","producer_version":"1",
                    "case_sha256":d('a'),"source_score_sha256":d('b'),"candidate_score_sha256":d('c')},
                "render_execution": {"status":statuses[1],"receipt_sha256":d('2'),"producer_id":"unit-test","producer_version":"1",
                    "source_score_sha256":d('b'),"candidate_score_sha256":d('c'),"candidate_midi_sha256":d('f'),
                    "candidate_audio_sha256":d('e'),"attempt_id":"attempt.unit.001"},
                "acoustic_observation": {"status":statuses[2],"receipt_sha256":d('3'),"producer_id":"unit-test","producer_version":"1",
                    "contract_sha256":d('4'),"source_audio_sha256":d('d'),"candidate_audio_sha256":d('e')},
                "score_audio_conformance": {"status":statuses[3],"receipt_sha256":d('5'),"producer_id":"unit-test","producer_version":"1",
                    "candidate_score_sha256":d('c'),"candidate_audio_sha256":d('e')},
                "human_acceptance": {"status":statuses[4],"receipt_sha256":d('6'),"producer_id":"unit-test","producer_version":"1",
                    "case_sha256":d('a'),"candidate_score_sha256":d('c'),"candidate_audio_sha256":d('e')}
            },
            "outcome": {
                "status":status,"required_gate_count":5,"passed_gate_count":passed,
                "failed_gate_count":failed,"abstained_gate_count":abstained,"reasons":[]
            },
            "receipt_sha256":null
        });
        let bytes = canonical_json_bytes(&value).unwrap();
        let mut sealed = value;
        let digest = domain_digest(BUNDLE_DIGEST_DOMAIN_V1, &bytes);
        sealed["receipt_sha256"] = Value::String(digest);
        serde_json::to_vec(&sealed).unwrap()
    }

    #[test]
    fn failure_dominates_abstention() {
        let statuses = [
            GateStatusV1::Pass, GateStatusV1::Fail, GateStatusV1::Abstain,
            GateStatusV1::NotRun, GateStatusV1::Pass,
        ];
        let report = verify_evidence_bundle_v1(&sealed_manifest(statuses), EvidenceInputsV1::default()).unwrap();
        assert_eq!(report.verified_status, BundleStatusV1::Fail);
    }

    #[test]
    fn missing_required_verifiers_cannot_produce_pass() {
        let statuses = [GateStatusV1::Pass; 5];
        let report = verify_evidence_bundle_v1(&sealed_manifest(statuses), EvidenceInputsV1::default()).unwrap();
        assert_eq!(report.verified_status, BundleStatusV1::Abstain);
        assert_eq!(report.passed_gate_count, 0);
        assert_eq!(report.abstained_gate_count, 5);
    }

    #[test]
    fn stale_bundle_receipt_seal_is_rejected() {
        let mut value: Value = serde_json::from_slice(&sealed_manifest([GateStatusV1::Pass; 5])).unwrap();
        value["bundle_id"] = Value::String("changed".into());
        assert!(verify_evidence_bundle_v1(&serde_json::to_vec(&value).unwrap(), EvidenceInputsV1::default()).is_err());
    }

    #[test]
    fn cross_candidate_audio_binding_is_a_hard_failure() {
        let mut value: Value = serde_json::from_slice(&sealed_manifest([GateStatusV1::Pass; 5])).unwrap();
        value["components"]["acoustic_observation"]["candidate_audio_sha256"] = Value::String(d('8'));
        let mut unsealed = value;
        unsealed["receipt_sha256"] = Value::Null;
        let digest = domain_digest(BUNDLE_DIGEST_DOMAIN_V1, &canonical_json_bytes(&unsealed).unwrap());
        unsealed["receipt_sha256"] = Value::String(digest);
        let report = verify_evidence_bundle_v1(&serde_json::to_vec(&unsealed).unwrap(), EvidenceInputsV1::default()).unwrap();
        assert_eq!(report.verified_status, BundleStatusV1::Fail);
    }

    #[test]
    fn pass_component_requires_a_receipt_and_producer_identity() {
        let mut value: Value = serde_json::from_slice(&sealed_manifest([GateStatusV1::Pass; 5])).unwrap();
        value["components"]["human_acceptance"]["receipt_sha256"] = Value::Null;
        let mut unsealed = value;
        unsealed["receipt_sha256"] = Value::Null;
        let digest = domain_digest(BUNDLE_DIGEST_DOMAIN_V1, &canonical_json_bytes(&unsealed).unwrap());
        unsealed["receipt_sha256"] = Value::String(digest);
        let error = verify_evidence_bundle_v1(&serde_json::to_vec(&unsealed).unwrap(), EvidenceInputsV1::default()).unwrap_err();
        assert!(error.contains("cannot declare pass without a receipt digest"));
    }

    #[test]
    fn cross_case_symbolic_binding_is_a_hard_failure() {
        let mut value: Value = serde_json::from_slice(&sealed_manifest([GateStatusV1::Pass; 5])).unwrap();
        value["components"]["symbolic_edit"]["case_sha256"] = Value::String(d('9'));
        let mut object = value;
        object["receipt_sha256"] = Value::Null;
        let digest = domain_digest(BUNDLE_DIGEST_DOMAIN_V1, &canonical_json_bytes(&object).unwrap());
        object["receipt_sha256"] = Value::String(digest);
        let report = verify_evidence_bundle_v1(&serde_json::to_vec(&object).unwrap(), EvidenceInputsV1::default()).unwrap();
        assert_eq!(report.verified_status, BundleStatusV1::Fail);
    }

    #[test]
    fn component_order_does_not_change_bundle_seal() {
        let bytes = sealed_manifest([GateStatusV1::Pass; 5]);
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        let digest = original["receipt_sha256"].as_str().unwrap().to_owned();
        fn reverse(value: &Value) -> Value {
            match value {
                Value::Object(map) => {
                    let mut keys: Vec<&String> = map.keys().collect();
                    keys.sort_unstable_by(|a,b| b.cmp(a));
                    let mut result = Map::new();
                    for key in keys {
                        result.insert(key.clone(), reverse(&map[key]));
                    }
                    Value::Object(result)
                }
                Value::Array(items) => Value::Array(items.iter().map(reverse).collect()),
                _ => value.clone(),
            }
        }
        let reordered = reverse(&original);
        verify_manifest_seal(&reordered, &digest).unwrap();
    }
}
