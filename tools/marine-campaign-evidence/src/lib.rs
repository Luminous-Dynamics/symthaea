#![forbid(unsafe_code)]

//! Marine test-campaign evidence binding verifier.
//!
//! This crate validates cross-record evidence relationships over the canonical
//! ROB-INSTR / ROB-EXP / SE-OBS / EXEC-ID / SE-SEM / ROB-REALIZE references.
//! It deliberately does not define a second observation, calibration, quantity,
//! clock, raw-data or actuation ontology.
//!
//! A successful result means only "eligible for independent review" for the
//! declared campaign. It is not instrument calibration, physical qualification,
//! water-quality approval, electrical safety approval or operating authority.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CampaignDisposition {
    /// Structural or identity defect prevents a defensible campaign record.
    Invalid,
    /// A known prerequisite or hard evidence condition is not satisfied.
    NotAdmissible,
    /// Timing or evidence state is insufficient to make a defensible decision.
    Indeterminate,
    /// The fixture or raw bundle is not a new physical acquisition.
    SyntheticOnly,
    /// Declared evidence prerequisites are present; independent review is still required.
    EligibleForIndependentReview,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub disposition: CampaignDisposition,
    pub findings: Vec<Finding>,
}

impl VerificationReport {
    fn new() -> Self {
        Self {
            disposition: CampaignDisposition::EligibleForIndependentReview,
            findings: Vec::new(),
        }
    }

    fn add(
        &mut self,
        disposition: CampaignDisposition,
        code: &'static str,
        path: impl Into<String>,
        detail: impl Into<String>,
    ) {
        self.findings.push(Finding {
            code,
            path: path.into(),
            detail: detail.into(),
        });
        self.disposition = combine(self.disposition, disposition);
    }
}

fn combine(a: CampaignDisposition, b: CampaignDisposition) -> CampaignDisposition {
    use CampaignDisposition::*;
    fn rank(v: CampaignDisposition) -> u8 {
        match v {
            EligibleForIndependentReview => 0,
            SyntheticOnly => 1,
            Indeterminate => 2,
            NotAdmissible => 3,
            Invalid => 4,
        }
    }
    if rank(a) >= rank(b) { a } else { b }
}

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(value, |node, key| node.get(*key))
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    at(value, path)?.as_str()
}

fn number_at(value: &Value, path: &[&str]) -> Option<f64> {
    at(value, path)?.as_f64()
}

fn integer_at(value: &Value, path: &[&str]) -> Option<u64> {
    at(value, path)?.as_u64()
}

fn require_string(report: &mut VerificationReport, value: &Value, path: &[&str]) -> Option<String> {
    match string_at(value, path) {
        Some(v) if !v.trim().is_empty() => Some(v.to_owned()),
        _ => {
            report.add(
                CampaignDisposition::Invalid,
                "missing_or_empty_string",
                path.join("."),
                "A required non-empty reference or identity is absent.",
            );
            None
        }
    }
}

/// Validate the marine campaign's semantic joins and claim prerequisites.
///
/// JSON shape/type validation remains the responsibility of the versioned
/// campaign JSON Schema. This verifier checks the cross-record invariants that
/// require comparing fields across article, mission, channel, calibration,
/// sampling and clock records.
pub fn verify_campaign(value: &Value) -> VerificationReport {
    let mut report = VerificationReport::new();

    if !value.is_object() {
        report.add(
            CampaignDisposition::Invalid,
            "campaign_not_object",
            "$",
            "Campaign root must be a JSON object.",
        );
        return report;
    }

    require_string(&mut report, value, &["campaign_id"]);
    require_string(&mut report, value, &["protocol", "rob_exp_protocol_ref"]);
    require_string(&mut report, value, &["protocol", "protocol_digest"]);
    require_string(&mut report, value, &["test_article", "physical_article_ref"]);

    let article_config = require_string(
        &mut report,
        value,
        &["test_article", "as_built_configuration_digest"],
    );
    let mission_config = require_string(
        &mut report,
        value,
        &["mission_binding", "configuration_digest"],
    );

    if article_config.is_some() && article_config != mission_config {
        report.add(
            CampaignDisposition::NotAdmissible,
            "article_mission_configuration_mismatch",
            "test_article.as_built_configuration_digest",
            "The physical article configuration does not match the mission binding.",
        );
    }

    match at(value, &["protocol", "protocol_frozen_before_run"]).and_then(Value::as_bool) {
        Some(true) => {}
        Some(false) => report.add(
            CampaignDisposition::NotAdmissible,
            "protocol_not_frozen",
            "protocol.protocol_frozen_before_run",
            "The prospective protocol was not frozen before the run.",
        ),
        None => report.add(
            CampaignDisposition::Invalid,
            "protocol_freeze_state_missing",
            "protocol.protocol_frozen_before_run",
            "Protocol freeze state is absent or not boolean.",
        ),
    }

    let start = integer_at(value, &["run", "start_monotonic_ns"]);
    let end = integer_at(value, &["run", "end_monotonic_ns"]);
    match (start, end) {
        (Some(s), Some(e)) if e >= s => {}
        (Some(_), Some(_)) => report.add(
            CampaignDisposition::Invalid,
            "run_time_reversed",
            "run",
            "Run end time precedes run start time.",
        ),
        _ => report.add(
            CampaignDisposition::Invalid,
            "run_time_missing",
            "run",
            "Monotonic run start and end timestamps are required.",
        ),
    }

    match string_at(value, &["run", "run_disposition"]) {
        Some("synthetic_fixture") | Some("completed") => {}
        Some("aborted_safety")
        | Some("aborted_operator")
        | Some("instrumentation_failure")
        | Some("protocol_deviation")
        | Some("infrastructure_failure") => report.add(
            CampaignDisposition::NotAdmissible,
            "run_not_completed",
            "run.run_disposition",
            "Aborted, failed or protocol-deviant runs cannot satisfy a completed campaign analysis.",
        ),
        Some("inconclusive") => report.add(
            CampaignDisposition::Indeterminate,
            "run_inconclusive",
            "run.run_disposition",
            "The run is explicitly inconclusive.",
        ),
        _ => report.add(
            CampaignDisposition::Invalid,
            "run_disposition_missing",
            "run.run_disposition",
            "Run disposition is absent or unrecognized.",
        ),
    }

    if at(value, &["run", "abort_events"])
        .and_then(Value::as_array)
        .is_some_and(|events| !events.is_empty())
        && matches!(
            string_at(value, &["run", "run_disposition"]),
            Some("completed")
        )
    {
        report.add(
            CampaignDisposition::Invalid,
            "completed_run_contains_abort",
            "run.abort_events",
            "A run with recorded abort events cannot be labelled completed.",
        );
    }

    let channels = match at(value, &["channels"]).and_then(Value::as_array) {
        Some(items) if !items.is_empty() => items,
        _ => {
            report.add(
                CampaignDisposition::Invalid,
                "channels_missing",
                "channels",
                "At least one measurement channel is required.",
            );
            return report;
        }
    };

    let max_sync_uncertainty = number_at(value, &["analysis", "max_sync_uncertainty_us"]);
    let mut clock_uncertainty_exceeded = false;
    let mut clock_budget_unknown = false;
    let mut known_non_admissible = false;

    for (index, channel) in channels.iter().enumerate() {
        let base = format!("channels[{index}]");

        let channel_id = string_at(channel, &["channel_id"]).unwrap_or("<missing-channel-id>");
        match string_at(channel, &["identity_strength"]) {
            Some("unknown") | Some("logical_only") | None => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "instrument_identity_too_weak",
                    format!("{base}.identity_strength"),
                    format!(
                        "Channel {channel_id} does not bind an adequately identified physical or fixture-bound instrument endpoint."
                    ),
                );
            }
            _ => {}
        }

        match (
            number_at(channel, &["range_min"]),
            number_at(channel, &["range_max"]),
        ) {
            (Some(min), Some(max)) if min < max => {}
            (Some(_), Some(_)) => report.add(
                CampaignDisposition::Invalid,
                "instrument_range_invalid",
                format!("{base}.range_min"),
                format!("Channel {channel_id} has a non-increasing measurement range."),
            ),
            _ => report.add(
                CampaignDisposition::Invalid,
                "instrument_range_missing",
                format!("{base}.range_min"),
                format!("Channel {channel_id} lacks numeric range bounds."),
            ),
        }

        let channel_config = string_at(channel, &["as_built_configuration_digest"]);

        if article_config.is_some() && channel_config != article_config.as_deref() {
            known_non_admissible = true;
            report.add(
                CampaignDisposition::NotAdmissible,
                "channel_configuration_mismatch",
                format!("{base}.as_built_configuration_digest"),
                format!(
                    "Channel {channel_id} is not bound to the exact article as-built configuration."
                ),
            );
        }

        let sample_rate = number_at(channel, &["sampling", "sample_rate_hz"]);
        let minimum_sample_rate =
            number_at(channel, &["sampling", "minimum_required_sample_rate_hz"]);
        match (sample_rate, minimum_sample_rate) {
            (Some(actual), Some(required)) if actual >= required => {}
            (Some(actual), Some(required)) => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "sample_rate_below_method_minimum",
                    format!("{base}.sampling.sample_rate_hz"),
                    format!(
                        "Channel {channel_id} sample rate {actual} Hz is below the method-declared minimum {required} Hz."
                    ),
                );
            }
            _ => report.add(
                CampaignDisposition::Invalid,
                "sample_rate_requirement_missing",
                format!("{base}.sampling.minimum_required_sample_rate_hz"),
                format!("Channel {channel_id} lacks a numeric sample rate or method minimum."),
            ),
        }

        let analogue_bandwidth =
            number_at(channel, &["sampling", "analogue_bandwidth_hz"]);
        let required_bandwidth =
            number_at(channel, &["sampling", "required_claim_bandwidth_hz"]);
        match (analogue_bandwidth, required_bandwidth) {
            (Some(measured), Some(required)) if measured >= required => {}
            (Some(_), Some(_)) => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "analogue_bandwidth_below_claim",
                    format!("{base}.sampling"),
                    format!(
                        "Channel {channel_id} analogue bandwidth is below its declared claim bandwidth."
                    ),
                );
            }
            _ => report.add(
                CampaignDisposition::Invalid,
                "bandwidth_fields_missing",
                format!("{base}.sampling"),
                format!("Channel {channel_id} lacks bandwidth or claim-bandwidth values."),
            ),
        }

        match string_at(channel, &["sampling", "sampling_adequacy_review_ref"]) {
            Some(reference) if !reference.trim().is_empty() => {}
            _ => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "sampling_adequacy_review_missing",
                    format!("{base}.sampling.sampling_adequacy_review_ref"),
                    format!(
                        "Channel {channel_id} has no claim-specific sampling-adequacy review."
                    ),
                );
            }
        }

        let currentness = string_at(channel, &["calibration", "currentness"]);
        match currentness {
            Some("current") => {
                for field in ["calibration_id_ref", "epoch_ref", "qualification_ref", "uncertainty_profile_ref"] {
                    match string_at(channel, &["calibration", field]) {
                        Some(reference) if !reference.trim().is_empty() => {}
                        _ => {
                            known_non_admissible = true;
                            report.add(
                                CampaignDisposition::NotAdmissible,
                                "qualified_calibration_reference_missing",
                                format!("{base}.calibration.{field}"),
                                format!(
                                    "Channel {channel_id} claims current calibration but lacks {field}."
                                ),
                            );
                        }
                    }
                }
            }
            Some("not_required_for_claim") | Some("not_applicable") => {}
            Some("stale") | Some("unknown") => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "calibration_not_current",
                    format!("{base}.calibration.currentness"),
                    format!(
                        "Channel {channel_id} cannot support an analysis requiring current qualified calibration."
                    ),
                );
            }
            _ => report.add(
                CampaignDisposition::Invalid,
                "calibration_state_missing",
                format!("{base}.calibration.currentness"),
                format!("Channel {channel_id} has no recognized calibration state."),
            ),
        }

        match string_at(channel, &["observation", "sample_integrity"]) {
            Some("complete") => {}
            Some("gaps_reported")
            | Some("duplicates_reported")
            | Some("reordered_reported")
            | Some("saturated_reported")
            | Some("unknown") => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    "sample_integrity_not_complete",
                    format!("{base}.observation.sample_integrity"),
                    format!(
                        "Channel {channel_id} contains gaps, duplicates, reordering, saturation or unknown integrity."
                    ),
                );
            }
            _ => report.add(
                CampaignDisposition::Invalid,
                "sample_integrity_missing",
                format!("{base}.observation.sample_integrity"),
                format!("Channel {channel_id} has no recognized sample-integrity state."),
            ),
        }

        let sync_method = string_at(channel, &["clock", "sync_method"]);
        let sync_uncertainty = number_at(channel, &["clock", "sync_uncertainty_us"]);
        match (max_sync_uncertainty, sync_uncertainty, sync_method) {
            (Some(limit), Some(actual), Some(method))
                if method != "unknown" && actual <= limit => {}
            (Some(limit), Some(actual), Some(method))
                if method != "unknown" && actual > limit =>
            {
                clock_uncertainty_exceeded = true;
                report.add(
                    CampaignDisposition::Indeterminate,
                    "clock_uncertainty_exceeds_budget",
                    format!("{base}.clock.sync_uncertainty_us"),
                    format!(
                        "Channel {channel_id} synchronization uncertainty {actual} us exceeds the declared {limit} us analysis budget."
                    ),
                );
            }
            _ => {
                clock_budget_unknown = true;
                report.add(
                    CampaignDisposition::Indeterminate,
                    "clock_alignment_indeterminate",
                    format!("{base}.clock"),
                    format!(
                        "Channel {channel_id} has unknown synchronization method, uncertainty or campaign timing budget."
                    ),
                );
            }
        }

        for field in ["instrument_endpoint_ref", "quantity_kind_ref", "unit_ref", "frame_or_location_ref"] {
            match string_at(channel, &[field]) {
                Some(reference) if !reference.trim().is_empty() => {}
                _ => report.add(
                    CampaignDisposition::Invalid,
                    "channel_identity_reference_missing",
                    format!("{base}.{field}"),
                    format!("Channel {channel_id} lacks required {field}."),
                ),
            }
        }

        match string_at(channel, &["observation", "raw_artifact_digest"]) {
            Some(digest) if digest.len() == 64 => {}
            _ => report.add(
                CampaignDisposition::Invalid,
                "raw_observation_digest_missing",
                format!("{base}.observation.raw_artifact_digest"),
                format!("Channel {channel_id} has no 64-character raw artifact digest."),
            ),
        }
    }

    for (field, code) in [
        ("missing_sample_count", "missing_samples_present"),
        ("duplicate_sample_count", "duplicate_samples_present"),
        ("reordered_sample_count", "reordered_samples_present"),
        ("saturated_sample_count", "saturated_samples_present"),
    ] {
        match integer_at(value, &["run", field]) {
            Some(0) => {}
            Some(count) => {
                known_non_admissible = true;
                report.add(
                    CampaignDisposition::NotAdmissible,
                    code,
                    format!("run.{field}"),
                    format!("{count} sample-integrity events are recorded."),
                );
            }
            None => report.add(
                CampaignDisposition::Invalid,
                "run_integrity_counter_missing",
                format!("run.{field}"),
                "A required run integrity counter is absent or invalid.",
            ),
        }
    }

    let channel_classes: Vec<&str> = channels
        .iter()
        .filter_map(|channel| string_at(channel, &["evidence_class"]))
        .collect();

    match string_at(value, &["raw_bundle", "acquisition_class"]) {
        Some("measured_physical")
            if channel_classes.iter().any(|class| *class != "measured") =>
        {
            report.add(
                CampaignDisposition::NotAdmissible,
                "physical_acquisition_class_mismatch",
                "channels.evidence_class",
                "A physical acquisition bundle contains a channel not classified as measured.",
            );
        }
        Some("hil") if channel_classes.iter().any(|class| *class != "hil") => {
            report.add(
                CampaignDisposition::NotAdmissible,
                "hil_evidence_class_mismatch",
                "channels.evidence_class",
                "HIL bundle and channel evidence classes disagree.",
            );
        }
        Some("phil") if channel_classes.iter().any(|class| *class != "phil") => {
            report.add(
                CampaignDisposition::NotAdmissible,
                "phil_evidence_class_mismatch",
                "channels.evidence_class",
                "PHIL bundle and channel evidence classes disagree.",
            );
        }
        _ => {}
    }

    let acquisition_class = string_at(value, &["raw_bundle", "acquisition_class"]);
    let run_disposition = string_at(value, &["run", "run_disposition"]);
    if acquisition_class == Some("synthetic")
        || acquisition_class == Some("simulation")
        || acquisition_class == Some("replay")
        || run_disposition == Some("synthetic_fixture")
    {
        report.add(
            CampaignDisposition::SyntheticOnly,
            "not_new_physical_acquisition",
            "raw_bundle.acquisition_class",
            "This record is synthetic, simulated or replayed; it cannot establish a new physical observation.",
        );
    }

    if known_non_admissible {
        report.disposition =
            combine(report.disposition, CampaignDisposition::NotAdmissible);
    } else if clock_uncertainty_exceeded || clock_budget_unknown {
        report.disposition = combine(report.disposition, CampaignDisposition::Indeterminate);
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURES: &str = include_str!(
        "../../../docs/research/marine/liveaboard/marine_test_campaign_fixtures_synthetic_v0.1.json"
    );

    fn fixtures() -> Value {
        serde_json::from_str(FIXTURES).expect("synthetic campaign fixture JSON must parse")
    }

    fn expected(text: &str) -> CampaignDisposition {
        match text {
            "synthetic_only_no_physical_admission" => CampaignDisposition::SyntheticOnly,
            "not_admissible_missing_bandwidth_review"
            | "not_admissible_stale_calibration"
            | "not_admissible_mixed_configuration" => CampaignDisposition::NotAdmissible,
            "indeterminate_clock_alignment" => CampaignDisposition::Indeterminate,
            other => panic!("unknown expected admission {other}"),
        }
    }

    #[test]
    fn frozen_fixture_corpus_reproduces_expected_dispositions() {
        let root = fixtures();
        let campaigns = root["campaigns"].as_array().expect("campaigns array");
        assert_eq!(campaigns.len(), 5);
        for campaign in campaigns {
            let expected_text = campaign["expected_admission"].as_str().expect("expected disposition");
            let report = verify_campaign(campaign);
            assert_eq!(
                report.disposition,
                expected(expected_text),
                "campaign {}: {:?}",
                campaign["campaign_id"],
                report.findings
            );
        }
    }

    #[test]
    fn article_and_channel_configuration_mismatch_is_not_admissible() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["as_built_configuration_digest"] =
            Value::String("9".repeat(64));
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn stale_calibration_is_not_admissible_even_when_raw_data_is_retained() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["calibration"]["currentness"] =
            Value::String("stale".into());
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn missing_sampling_adequacy_review_is_not_admissible() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["sampling"]["sampling_adequacy_review_ref"] = Value::Null;
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn sample_rate_below_method_minimum_is_not_admissible() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["sampling"]["minimum_required_sample_rate_hz"] =
            Value::from(1000.0);
        campaign["channels"][0]["sampling"]["sample_rate_hz"] = Value::from(999.0);
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn clock_uncertainty_over_budget_is_indeterminate() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["analysis"]["max_sync_uncertainty_us"] = Value::from(100.0);
        campaign["channels"][0]["clock"]["sync_uncertainty_us"] = Value::from(101.0);
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::Indeterminate
        );
    }

    #[test]
    fn nonzero_missing_sample_count_is_not_admissible() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["run"]["missing_sample_count"] = Value::from(1);
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn logical_only_instrument_identity_is_not_admissible_for_physical_evidence() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["identity_strength"] = Value::String("logical_only".into());
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn safety_aborted_run_cannot_be_admitted_as_completed() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["run"]["run_disposition"] = Value::String("aborted_safety".into());
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }

    #[test]
    fn reversed_instrument_range_is_invalid() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["channels"][0]["range_min"] = Value::from(10.0);
        campaign["channels"][0]["range_max"] = Value::from(1.0);
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::Invalid
        );
    }

    #[test]
    fn unfrozen_protocol_is_not_admissible() {
        let mut campaign = fixtures()["campaigns"][0].clone();
        campaign["protocol"]["protocol_frozen_before_run"] = Value::Bool(false);
        assert_eq!(
            verify_campaign(&campaign).disposition,
            CampaignDisposition::NotAdmissible
        );
    }
}
