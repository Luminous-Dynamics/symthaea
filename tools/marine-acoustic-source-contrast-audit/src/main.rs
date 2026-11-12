use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const RECEIVED_LEVEL_QUANTITY: &str = "received_spl_db_re_1upa";

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Synthetic,
    Measured,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus {
    PublicOrPermitted,
    Unknown,
    Restricted,
}

#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum SourceKind {
    PropellerCavitation,
    PropellerHydrodynamic,
    PrimeMover,
    ShaftGearbox,
    AuxiliaryMachinery,
    HullAppendageFlow,
    AmbientBackground,
}

impl SourceKind {
    fn name(self) -> &'static str {
        match self {
            Self::PropellerCavitation => "propeller_cavitation",
            Self::PropellerHydrodynamic => "propeller_hydrodynamic",
            Self::PrimeMover => "prime_mover",
            Self::ShaftGearbox => "shaft_gearbox",
            Self::AuxiliaryMachinery => "auxiliary_machinery",
            Self::HullAppendageFlow => "hull_appendage_flow",
            Self::AmbientBackground => "ambient_background",
        }
    }

    fn all() -> BTreeSet<Self> {
        [
            Self::PropellerCavitation,
            Self::PropellerHydrodynamic,
            Self::PrimeMover,
            Self::ShaftGearbox,
            Self::AuxiliaryMachinery,
            Self::HullAppendageFlow,
            Self::AmbientBackground,
        ]
        .into_iter()
        .collect()
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum DesignStrategy {
    Randomized,
    Counterbalanced,
    MatchedRepeated,
    Uncontrolled,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    target_source: SourceKind,
    predeclared_intervention: bool,
    design_strategy: DesignStrategy,
    intervention_protocol_sha256: String,
    design_evidence_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    unobserved_sources: Vec<String>,
    band_frequency_tolerance_fraction: f64,
    baseline: Observation,
    candidate: Observation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEvidence {
    dataset_id: String,
    citation: String,
    raw_data_sha256: String,
    rights_status: RightsStatus,
    independently_reviewed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    run_id: String,
    hydrophone_instrument_id: String,
    calibration_certificate_sha256: String,
    timebase_receipt_sha256: String,
    acoustic_trace_sha256: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    source_states: Vec<SourceStateEvidence>,
    bands: Vec<BandMeasurement>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceStateEvidence {
    source: SourceKind,
    state: String,
    observable_kind: String,
    observable_id: String,
    evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BandMeasurement {
    band_id: String,
    quantity_id: String,
    center_frequency_hz: f64,
    received_level_db: f64,
    expanded_uncertainty_db: f64,
    confidence_level: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    target_source: &'static str,
    design_evidence_sha256: String,
    disposition: &'static str,
    identifiability: &'static str,
    baseline_run_id: String,
    candidate_run_id: String,
    band_deltas: Vec<BandDelta>,
    reasons: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Debug, Serialize)]
struct BandDelta {
    band_id: String,
    center_frequency_hz: f64,
    baseline_received_level_db: f64,
    candidate_received_level_db: f64,
    observed_received_level_delta_db: f64,
    conservative_delta_lower_db: f64,
    conservative_delta_upper_db: f64,
    combined_expanded_uncertainty_db: f64,
    confidence_level: f64,
    interpretation: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn validate_observation(observation: &Observation, label: &str, errors: &mut Vec<String>) {
    for (field, value) in [
        ("hydrophone_instrument_id", observation.hydrophone_instrument_id.as_str()),
        ("run_id", observation.run_id.as_str()),
    ] {
        if value.trim().is_empty() {
            errors.push(format!("{label}: {field} is required"));
        }
    }

    for (field, digest) in [
        ("calibration_certificate_sha256", &observation.calibration_certificate_sha256),
        ("timebase_receipt_sha256", &observation.timebase_receipt_sha256),
        ("acoustic_trace_sha256", &observation.acoustic_trace_sha256),
        ("service_fingerprint_sha256", &observation.service_fingerprint_sha256),
        ("environment_fingerprint_sha256", &observation.environment_fingerprint_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label}: {field} must be a 64-character SHA-256 hex digest"));
        }
    }

    if observation.source_states.is_empty() {
        errors.push(format!("{label}: source_states must not be empty"));
    }
    let mut seen_sources = BTreeSet::new();
    for state in &observation.source_states {
        if !seen_sources.insert(state.source) {
            errors.push(format!("{label}: duplicate source-state record for {}", state.source.name()));
        }
        if state.state.trim().is_empty()
            || state.observable_kind.trim().is_empty()
            || state.observable_id.trim().is_empty()
        {
            errors.push(format!("{label}: {} state and independent observable identity are required", state.source.name()));
        }
        if !is_sha256(&state.evidence_sha256) {
            errors.push(format!("{label}: {} evidence digest is malformed", state.source.name()));
        }
    }

    if observation.bands.is_empty() {
        errors.push(format!("{label}: at least one acoustic band is required"));
    }
    let mut band_ids = BTreeSet::new();
    for band in &observation.bands {
        if band.band_id.trim().is_empty() || !band_ids.insert(band.band_id.clone()) {
            errors.push(format!("{label}: band IDs must be non-empty and unique"));
        }
        if band.quantity_id != RECEIVED_LEVEL_QUANTITY {
            errors.push(format!("{label} / {}: only received SPL re 1 uPa is accepted in v0.1", band.band_id));
        }
        if !finite_positive(band.center_frequency_hz)
            || !band.received_level_db.is_finite()
            || !finite_positive(band.expanded_uncertainty_db)
            || !band.confidence_level.is_finite()
            || band.confidence_level <= 0.0
            || band.confidence_level >= 1.0
        {
            errors.push(format!("{label} / {}: invalid frequency, received level or expanded uncertainty", band.band_id));
        }
    }
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {}", case.schema_version, SCHEMA_VERSION));
    }
    if case.case_id.trim().is_empty() || case.source.dataset_id.trim().is_empty() || case.source.citation.trim().is_empty() {
        errors.push("case_id, source dataset_id and citation are required".to_string());
    }
    if !is_sha256(&case.source.raw_data_sha256)
        || !is_sha256(&case.intervention_protocol_sha256)
        || !is_sha256(&case.design_evidence_sha256)
        || !is_sha256(&case.service_fingerprint_sha256)
        || !is_sha256(&case.environment_fingerprint_sha256)
    {
        errors.push("source, intervention, service and environment digests must be SHA-256".to_string());
    }
    if !case.band_frequency_tolerance_fraction.is_finite()
        || case.band_frequency_tolerance_fraction <= 0.0
        || case.band_frequency_tolerance_fraction >= 1.0
    {
        errors.push("band_frequency_tolerance_fraction must be strictly between 0 and 1".to_string());
    }
    validate_observation(&case.baseline, "baseline", &mut errors);
    validate_observation(&case.candidate, "candidate", &mut errors);
    if case.baseline.run_id == case.candidate.run_id {
        errors.push("baseline and candidate run_id must differ".to_string());
    }
    if case.baseline.acoustic_trace_sha256 == case.candidate.acoustic_trace_sha256 {
        errors.push("baseline and candidate acoustic_trace_sha256 must differ".to_string());
    }
    errors
}

fn source_map(observation: &Observation) -> BTreeMap<SourceKind, &SourceStateEvidence> {
    observation.source_states.iter().map(|state| (state.source, state)).collect()
}

fn band_map(observation: &Observation) -> BTreeMap<&str, &BandMeasurement> {
    observation.bands.iter().map(|band| (band.band_id.as_str(), band)).collect()
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let invalid_input = !validation_errors.is_empty();
    let mut reasons = validation_errors;
    let mut band_deltas = Vec::new();
    let mut identifiability = "INCOMPLETE";

    if reasons.is_empty() {
        let baseline = &case.baseline;
        let candidate = &case.candidate;

        if baseline.service_fingerprint_sha256 != case.service_fingerprint_sha256
            || candidate.service_fingerprint_sha256 != case.service_fingerprint_sha256
            || baseline.environment_fingerprint_sha256 != case.environment_fingerprint_sha256
            || candidate.environment_fingerprint_sha256 != case.environment_fingerprint_sha256
        {
            reasons.push("baseline/candidate service or environment identity differs from the frozen case fingerprint".to_string());
            identifiability = "INCOMPARABLE";
        } else if baseline.hydrophone_instrument_id != candidate.hydrophone_instrument_id
            || baseline.calibration_certificate_sha256 != candidate.calibration_certificate_sha256
            || baseline.timebase_receipt_sha256 != candidate.timebase_receipt_sha256
        {
            reasons.push("baseline/candidate hydrophone, calibration certificate or timebase differs; no v0.1 cross-instrument transfer is defined".to_string());
            identifiability = "INCOMPARABLE";
        } else {
            let baseline_sources = source_map(baseline);
            let candidate_sources = source_map(candidate);
            let expected_sources = SourceKind::all();
            let baseline_ids: BTreeSet<_> = baseline_sources.keys().copied().collect();
            let candidate_ids: BTreeSet<_> = candidate_sources.keys().copied().collect();

            if baseline_ids != expected_sources || candidate_ids != expected_sources {
                reasons.push("the complete declared source-state inventory is not present in both observations".to_string());
                identifiability = "INCOMPLETE";
            } else if baseline_sources.values().chain(candidate_sources.values()).any(|state| state.state.eq_ignore_ascii_case("unknown")) {
                reasons.push("one or more source states are unknown".to_string());
                identifiability = "INCOMPLETE";
            } else {
                let target_baseline = baseline_sources.get(&case.target_source).expect("required source exists");
                let target_candidate = candidate_sources.get(&case.target_source).expect("required source exists");
                let source_observable_changed = expected_sources.iter().any(|source| {
                    let base = baseline_sources.get(source).expect("complete source inventory checked");
                    let cand = candidate_sources.get(source).expect("complete source inventory checked");
                    base.observable_kind != cand.observable_kind || base.observable_id != cand.observable_id
                });
                let non_target_changed = expected_sources
                    .iter()
                    .filter(|source| **source != case.target_source)
                    .any(|source| {
                        baseline_sources.get(source).map(|x| x.state.as_str())
                            != candidate_sources.get(source).map(|x| x.state.as_str())
                    });

                if source_observable_changed {
                    reasons.push("one or more source-state observable identities changed between baseline and candidate".to_string());
                    identifiability = "INCOMPARABLE";
                } else if non_target_changed {
                    reasons.push("at least one non-target source state changed between baseline and candidate".to_string());
                    identifiability = "INCOMPARABLE";
                } else if !case.unobserved_sources.is_empty() {
                    reasons.push("unobserved source contributors remain in the acoustic mixture".to_string());
                    identifiability = "UNRESOLVED_MIXTURE";
                } else if !case.predeclared_intervention
                    || target_baseline.state == target_candidate.state
                    || case.design_strategy == DesignStrategy::Uncontrolled
                {
                    reasons.push("no predeclared, discriminating and controlled target-source intervention is evidenced".to_string());
                    identifiability = "UNRESOLVED_MIXTURE";
                } else if case.design_strategy == DesignStrategy::MatchedRepeated {
                    reasons.push("matched repeated contrast is recorded, but randomization/counterbalancing is not established".to_string());
                    identifiability = "PARTIALLY_IDENTIFIABLE";
                } else {
                    identifiability = "IDENTIFIABLE_FOR_DECLARED_CONTRAST";
                }

                if identifiability == "IDENTIFIABLE_FOR_DECLARED_CONTRAST"
                    || identifiability == "PARTIALLY_IDENTIFIABLE"
                {
                    let base_bands = band_map(baseline);
                    let candidate_bands = band_map(candidate);
                    if base_bands.keys().copied().collect::<BTreeSet<_>>()
                        != candidate_bands.keys().copied().collect::<BTreeSet<_>>()
                    {
                        reasons.push("baseline/candidate acoustic band IDs differ".to_string());
                        identifiability = "INCOMPARABLE";
                        band_deltas.clear();
                    } else {
                        for (band_id, base) in &base_bands {
                            let cand = candidate_bands.get(band_id).expect("band set equality checked");
                            let relative_frequency_error =
                                (base.center_frequency_hz - cand.center_frequency_hz).abs()
                                    / base.center_frequency_hz;
                            if relative_frequency_error > case.band_frequency_tolerance_fraction {
                                reasons.push(format!("band {band_id} centre frequencies are not matched"));
                                identifiability = "INCOMPARABLE";
                                band_deltas.clear();
                                break;
                            }
                            if (base.confidence_level - cand.confidence_level).abs() > 1.0e-9 {
                                reasons.push(format!("band {band_id} expanded uncertainty confidence levels differ"));
                                identifiability = "INCOMPLETE";
                                band_deltas.clear();
                                break;
                            }
                            let delta = cand.received_level_db - base.received_level_db;
                            let combined_u = base.expanded_uncertainty_db + cand.expanded_uncertainty_db;
                            band_deltas.push(BandDelta {
                                band_id: (*band_id).to_string(),
                                center_frequency_hz: (base.center_frequency_hz + cand.center_frequency_hz) / 2.0,
                                baseline_received_level_db: base.received_level_db,
                                candidate_received_level_db: cand.received_level_db,
                                observed_received_level_delta_db: delta,
                                conservative_delta_lower_db: delta - combined_u,
                                conservative_delta_upper_db: delta + combined_u,
                                combined_expanded_uncertainty_db: combined_u,
                                confidence_level: base.confidence_level,
                                interpretation: "intervention_associated_change_in_total_received_level_not_additive_source_contribution",
                            });
                        }
                    }
                }
            }
        }
    }

    let evidence_eligible = case.evidence_kind == EvidenceKind::Measured
        && case.source.rights_status == RightsStatus::PublicOrPermitted
        && case.source.independently_reviewed;
    let disposition = if invalid_input {
        "INVALID"
    } else if identifiability == "INCOMPARABLE" {
        "INCOMPARABLE"
    } else if identifiability == "UNRESOLVED_MIXTURE" {
        "UNRESOLVED_MIXTURE"
    } else if identifiability == "PARTIALLY_IDENTIFIABLE" {
        "PARTIALLY_IDENTIFIABLE"
    } else if !evidence_eligible {
        "INCOMPLETE"
    } else if identifiability == "IDENTIFIABLE_FOR_DECLARED_CONTRAST" {
        "CONTRAST_ELIGIBLE"
    } else {
        "INCOMPLETE"
    };

    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("synthetic evidence cannot support measured source-attribution eligibility".to_string());
    }
    if case.source.rights_status != RightsStatus::PublicOrPermitted {
        reasons.push("source dataset reuse rights are not established as public_or_permitted".to_string());
    }
    if !case.source.independently_reviewed {
        reasons.push("source metadata has not been independently reviewed".to_string());
    }

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        target_source: case.target_source.name(),
        design_evidence_sha256: case.design_evidence_sha256.clone(),
        disposition,
        identifiability,
        baseline_run_id: case.baseline.run_id.clone(),
        candidate_run_id: case.candidate.run_id.clone(),
        band_deltas,
        reasons,
        claim_ceiling: "eligibility_for_declared_source_state_contrast_and_total_received_level_delta_only_no_additive_source_decomposition_source_level_ecological_or_regulatory_qualification",
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-acoustic-source-contrast-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-acoustic-source-contrast-audit <case.json>");
        process::exit(64);
    }
    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            process::exit(64);
        }
    };
    let case: Case = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid case JSON: {error}");
            process::exit(64);
        }
    };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "CONTRAST_ELIGIBLE" => {}
        "INVALID" | "INCOMPARABLE" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    fn states(target_state: &str, target_digest: char, machinery_state: &str) -> Vec<SourceStateEvidence> {
        [
            (SourceKind::PropellerCavitation, target_state),
            (SourceKind::PropellerHydrodynamic, "matched"),
            (SourceKind::PrimeMover, "running"),
            (SourceKind::ShaftGearbox, "matched"),
            (SourceKind::AuxiliaryMachinery, machinery_state),
            (SourceKind::HullAppendageFlow, "matched"),
            (SourceKind::AmbientBackground, "matched"),
        ]
        .into_iter()
        .map(|(source, state)| SourceStateEvidence {
            source,
            state: state.to_string(),
            observable_kind: "independent_sensor_or_state_record".to_string(),
            observable_id: format!("obs-{}", source.name()),
            evidence_sha256: digest(target_digest),
        })
        .collect()
    }

    fn observation(run_id: &str, target_state: &str, target_digest: char, level: f64, machinery_state: &str) -> Observation {
        Observation {
            run_id: run_id.to_string(),
            hydrophone_instrument_id: "hydrophone-H1".to_string(),
            calibration_certificate_sha256: digest('c'),
            timebase_receipt_sha256: digest('d'),
            acoustic_trace_sha256: digest(if run_id == "baseline" { 'e' } else { 'f' }),
            service_fingerprint_sha256: digest('1'),
            environment_fingerprint_sha256: digest('2'),
            source_states: states(target_state, target_digest, machinery_state),
            bands: vec![
                BandMeasurement {
                    band_id: "one-third-octave-100Hz".to_string(),
                    quantity_id: RECEIVED_LEVEL_QUANTITY.to_string(),
                    center_frequency_hz: 100.0,
                    received_level_db: level,
                    expanded_uncertainty_db: 0.5,
                    confidence_level: 0.95,
                },
            ],
        }
    }

    fn case() -> Case {
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-source-contrast-001".to_string(),
            evidence_kind: EvidenceKind::Synthetic,
            source: SourceEvidence {
                dataset_id: "synthetic-source-contrast".to_string(),
                citation: "Synthetic test fixture only".to_string(),
                raw_data_sha256: digest('a'),
                rights_status: RightsStatus::PublicOrPermitted,
                independently_reviewed: true,
            },
            target_source: SourceKind::PropellerCavitation,
            predeclared_intervention: true,
            design_strategy: DesignStrategy::Randomized,
            intervention_protocol_sha256: digest('3'),
            design_evidence_sha256: digest('7'),
            service_fingerprint_sha256: digest('1'),
            environment_fingerprint_sha256: digest('2'),
            unobserved_sources: vec![],
            band_frequency_tolerance_fraction: 0.01,
            baseline: observation("baseline", "non_cavitating", '4', 120.0, "running"),
            candidate: observation("candidate", "cavitating", '5', 117.0, "running"),
        }
    }

    fn make_measured(case: &mut Case) {
        case.evidence_kind = EvidenceKind::Measured;
    }

    #[test]
    fn source_observable_identity_change_is_incomparable() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.source_states.iter_mut()
            .find(|state| state.source == SourceKind::AmbientBackground)
            .unwrap().observable_id = "different-ambient-sensor".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn identical_run_ids_or_raw_traces_are_invalid() {
        let mut input = case();
        input.candidate.run_id = input.baseline.run_id.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
        let mut input = case();
        input.candidate.acoustic_trace_sha256 = input.baseline.acoustic_trace_sha256.clone();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn design_evidence_digest_is_preserved_in_receipt() {
        let report = audit(&case());
        assert_eq!(report.design_evidence_sha256, digest('7'));
    }

    #[test]
    fn counterbalanced_design_is_eligible_but_not_a_causal_effect_estimate() {
        let mut input = case();
        make_measured(&mut input);
        input.design_strategy = DesignStrategy::Counterbalanced;
        let report = audit(&input);
        assert_eq!(report.disposition, "CONTRAST_ELIGIBLE");
        assert!(report.claim_ceiling.contains("no_additive_source_decomposition"));
    }

    #[test]
    fn malformed_design_evidence_digest_is_invalid() {
        let mut input = case();
        input.design_evidence_sha256 = "missing".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn synthetic_fixture_is_never_contrast_eligible() {
        let report = audit(&case());
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.identifiability, "IDENTIFIABLE_FOR_DECLARED_CONTRAST");
        assert!((report.band_deltas[0].observed_received_level_delta_db + 3.0).abs() < 1e-12);
        assert!(report.claim_ceiling.contains("no_additive_source_decomposition"));
    }

    #[test]
    fn controlled_randomized_source_state_change_is_contrast_eligible() {
        let mut input = case();
        make_measured(&mut input);
        let report = audit(&input);
        assert_eq!(report.disposition, "CONTRAST_ELIGIBLE");
        assert_eq!(report.identifiability, "IDENTIFIABLE_FOR_DECLARED_CONTRAST");
    }

    #[test]
    fn passive_same_source_state_is_an_unresolved_mixture() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.source_states[0].state = input.baseline.source_states[0].state.clone();
        assert_eq!(audit(&input).disposition, "UNRESOLVED_MIXTURE");
    }

    #[test]
    fn changing_non_target_machinery_is_incomparable() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.source_states.iter_mut()
            .find(|state| state.source == SourceKind::AuxiliaryMachinery)
            .unwrap().state = "off".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn unobserved_sources_prevent_attribution() {
        let mut input = case();
        make_measured(&mut input);
        input.unobserved_sources.push("unknown_auxiliary".to_string());
        assert_eq!(audit(&input).disposition, "UNRESOLVED_MIXTURE");
    }

    #[test]
    fn uncontrolled_design_is_unresolved() {
        let mut input = case();
        make_measured(&mut input);
        input.design_strategy = DesignStrategy::Uncontrolled;
        assert_eq!(audit(&input).disposition, "UNRESOLVED_MIXTURE");
    }

    #[test]
    fn matched_repeated_design_is_only_partially_identifiable() {
        let mut input = case();
        make_measured(&mut input);
        input.design_strategy = DesignStrategy::MatchedRepeated;
        assert_eq!(audit(&input).disposition, "PARTIALLY_IDENTIFIABLE");
    }

    #[test]
    fn service_or_environment_change_is_incomparable() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.environment_fingerprint_sha256 = digest('9');
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn calibration_or_timebase_change_is_incomparable() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.calibration_certificate_sha256 = digest('8');
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn band_delta_uses_conservative_linear_uncertainty_sum() {
        let mut input = case();
        make_measured(&mut input);
        let report = audit(&input);
        let band = &report.band_deltas[0];
        assert!((band.combined_expanded_uncertainty_db - 1.0).abs() < 1e-12);
        assert!((band.conservative_delta_lower_db + 4.0).abs() < 1e-12);
        assert!((band.conservative_delta_upper_db + 2.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_evidence_digest_is_rejected() {
        let mut input = case();
        input.baseline.source_states[0].evidence_sha256 = "not-a-digest".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn mismatched_band_frequency_is_incomparable() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.bands[0].center_frequency_hz = 120.0;
        assert_eq!(audit(&input).disposition, "INCOMPARABLE");
    }

    #[test]
    fn unknown_source_state_is_incomplete() {
        let mut input = case();
        make_measured(&mut input);
        input.candidate.source_states[1].state = "unknown".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn missing_source_record_is_incomplete() {
        let mut input = case();
        make_measured(&mut input);
        input.baseline.source_states.pop();
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn uncontrolled_passive_record_never_reports_source_decomposition() {
        let mut input = case();
        input.predeclared_intervention = false;
        let report = audit(&input);
        assert_eq!(report.identifiability, "UNRESOLVED_MIXTURE");
        assert!(report.band_deltas.is_empty());
    }
}
