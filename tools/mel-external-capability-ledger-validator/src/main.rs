use serde::Deserialize;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const LEDGER_SCHEMA_V1: &str = "mel-external-capability-ledger-v1";
const LEDGER_FILE_V1: &str = "external-capability-ledger-2026-11-21.json";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LedgerV1 {
    schema_version: String,
    ledger_id: String,
    as_of: String,
    sources: Vec<SourceV1>,
    observations: Vec<ObservationV1>,
    comparison_rules: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceV1 {
    source_id: String,
    title: String,
    url: String,
    source_class: SourceClassV1,
    published_or_observed_at: String,
    retrieved_at: String,
    scope: String,
    version_snapshot: Option<String>,
    limitations: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SourceClassV1 {
    OfficialProductDocumentation,
    OfficialApiDocumentation,
    PeerReviewedOrConferencePaper,
    Preprint,
    BenchmarkLeaderboard,
    BenchmarkDataset,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationV1 {
    observation_id: String,
    source_id: String,
    system: String,
    system_version: String,
    capability_area: CapabilityAreaV1,
    evidence_kind: EvidenceKindV1,
    task_scope: String,
    metrics: Vec<MetricV1>,
    capabilities: Vec<String>,
    limitations: Vec<String>,
    comparability_group: String,
    directly_comparable_to_audio_elo: bool,
    claim_ceiling: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CapabilityAreaV1 {
    AudioGeneration,
    SymbolicGeneration,
    MusicEditing,
    LiveMusicInteraction,
    MusicUnderstanding,
    DatasetOrBenchmark,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceKindV1 {
    BlindPreferenceLeaderboard,
    HumanRatingStudy,
    PublishedDatasetStatistics,
    DocumentedProductCapability,
    DocumentedApiCapability,
    ResearchPrototype,
    TemporalControlBenchmark,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricV1 {
    name: String,
    value: f64,
    unit: String,
    plus_minus: Option<f64>,
    sample_size: Option<u64>,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    confidence_level: Option<f64>,
    scope: String,
}

fn valid_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 3 || bytes.len() > 101 {
        return false;
    }
    bytes[0].is_ascii_lowercase()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

fn valid_date(value: &str) -> bool {
    if value.len() != 10 {
        return false;
    }
    let parts: Vec<_> = value.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        parts[0].parse::<u32>(),
        parts[1].parse::<u32>(),
        parts[2].parse::<u32>(),
    ) else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_ledger(ledger: &LedgerV1) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if ledger.schema_version != LEDGER_SCHEMA_V1 {
        errors.push(format!("unsupported schema_version: {}", ledger.schema_version));
    }
    if !valid_id(&ledger.ledger_id) {
        errors.push("ledger_id is not a lowercase stable identifier".into());
    }
    if !valid_date(&ledger.as_of) {
        errors.push(format!("as_of is not a valid ISO date: {}", ledger.as_of));
    }
    if ledger.sources.is_empty() {
        errors.push("sources must not be empty".into());
    }
    if ledger.observations.is_empty() {
        errors.push("observations must not be empty".into());
    }
    if ledger.comparison_rules.is_empty()
        || ledger.comparison_rules.iter().any(|rule| !nonempty(rule))
    {
        errors.push("comparison_rules must contain non-empty rules".into());
    }

    let mut source_ids = BTreeSet::new();
    for source in &ledger.sources {
        if !valid_id(&source.source_id) {
            errors.push(format!("invalid source_id: {}", source.source_id));
        }
        if !source_ids.insert(source.source_id.as_str()) {
            errors.push(format!("duplicate source_id: {}", source.source_id));
        }
        if !nonempty(&source.title)
            || !nonempty(&source.scope)
            || !nonempty(&source.published_or_observed_at)
        {
            errors.push(format!("source {} has empty required text", source.source_id));
        }
        if !source.url.starts_with("https://") || source.url.chars().any(char::is_whitespace) {
            errors.push(format!("source {} has an invalid HTTPS URL", source.source_id));
        }
        if !valid_date(&source.retrieved_at) {
            errors.push(format!("source {} has an invalid retrieved_at date", source.source_id));
        }
        let _ = &source.source_class;
        let _ = &source.version_snapshot;
        let _ = &source.limitations;
    }

    let mut observation_ids = BTreeSet::new();
    for observation in &ledger.observations {
        if !valid_id(&observation.observation_id) {
            errors.push(format!(
                "invalid observation_id: {}",
                observation.observation_id
            ));
        }
        if !observation_ids.insert(observation.observation_id.as_str()) {
            errors.push(format!(
                "duplicate observation_id: {}",
                observation.observation_id
            ));
        }
        if !source_ids.contains(observation.source_id.as_str()) {
            errors.push(format!(
                "observation {} references unknown source {}",
                observation.observation_id, observation.source_id
            ));
        }
        if !nonempty(&observation.system)
            || !nonempty(&observation.system_version)
            || !nonempty(&observation.task_scope)
            || !nonempty(&observation.comparability_group)
            || !nonempty(&observation.claim_ceiling)
        {
            errors.push(format!(
                "observation {} has empty required text",
                observation.observation_id
            ));
        }
        if observation.capabilities.is_empty()
            || observation.capabilities.iter().any(|item| !nonempty(item))
        {
            errors.push(format!(
                "observation {} requires non-empty capability statements",
                observation.observation_id
            ));
        }
        if observation.limitations.iter().any(|item| !nonempty(item)) {
            errors.push(format!(
                "observation {} contains an empty limitation",
                observation.observation_id
            ));
        }

        for metric in &observation.metrics {
            if !nonempty(&metric.name) || !nonempty(&metric.unit) || !nonempty(&metric.scope) {
                errors.push(format!(
                    "observation {} contains a metric without name/unit/scope",
                    observation.observation_id
                ));
            }
            if !metric.value.is_finite() {
                errors.push(format!(
                    "observation {} contains a non-finite metric",
                    observation.observation_id
                ));
            }
            if metric.plus_minus.is_some_and(|value| !value.is_finite() || value < 0.0) {
                errors.push(format!(
                    "observation {} contains invalid metric uncertainty",
                    observation.observation_id
                ));
            }
            if metric.sample_size == Some(0) {
                errors.push(format!(
                    "observation {} contains a zero sample_size",
                    observation.observation_id
                ));
            }
            if metric.lower_bound.is_some() != metric.upper_bound.is_some() {
                errors.push(format!(
                    "observation {} has an incomplete confidence interval",
                    observation.observation_id
                ));
            }
            if let (Some(lower), Some(upper)) = (metric.lower_bound, metric.upper_bound) {
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    errors.push(format!(
                        "observation {} has an invalid confidence interval",
                        observation.observation_id
                    ));
                }
            }
            if metric.confidence_level.is_some_and(|level| {
                !level.is_finite() || !(0.0..1.0).contains(&level)
            }) {
                errors.push(format!(
                    "observation {} has an invalid confidence level",
                    observation.observation_id
                ));
            }
            if metric.unit == "Elo" && !observation.directly_comparable_to_audio_elo {
                errors.push(format!(
                    "observation {} uses Elo without declaring the native audio-Elo comparison group",
                    observation.observation_id
                ));
            }
        }

        if observation.directly_comparable_to_audio_elo {
            if !matches!(observation.capability_area, CapabilityAreaV1::AudioGeneration)
                || !matches!(observation.evidence_kind, EvidenceKindV1::BlindPreferenceLeaderboard)
                || !observation.comparability_group.starts_with("aa-music-")
                || !observation.metrics.iter().any(|metric| {
                    metric.name == "human_pairwise_preference_elo" && metric.unit == "Elo"
                })
            {
                errors.push(format!(
                    "observation {} claims direct audio-Elo comparability outside the matching blind-preference leaderboard cohort",
                    observation.observation_id
                ));
            }
        }

        let _ = &observation.capability_area;
        let _ = &observation.evidence_kind;
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn default_ledger_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/mel")
        .join(LEDGER_FILE_V1)
}

fn load_ledger(path: &Path) -> Result<LedgerV1, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid ledger JSON at {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let path = args.next().map(PathBuf::from).unwrap_or_else(default_ledger_path);
    if args.next().is_some() {
        return Err("usage: mel-external-capability-ledger-validator [ledger.json]".into());
    }
    let ledger = load_ledger(&path)?;
    validate_ledger(&ledger).map_err(|errors| errors.join("\n"))?;
    println!(
        "valid: {} sources, {} observations, as_of={}",
        ledger.sources.len(),
        ledger.observations.len(),
        ledger.as_of
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mel-external-capability-ledger-validator: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_ledger() -> LedgerV1 {
        load_ledger(&default_ledger_path()).expect("load checked-in ledger")
    }

    #[test]
    fn checked_in_ledger_has_valid_references_and_comparison_groups() {
        let ledger = current_ledger();
        validate_ledger(&ledger).expect("ledger must satisfy semantic validator");
        assert_eq!(ledger.sources.len(), 11);
        assert_eq!(ledger.observations.len(), 21);
    }

    #[test]
    fn duplicate_source_identity_is_rejected() {
        let mut ledger = current_ledger();
        ledger.sources[1].source_id = ledger.sources[0].source_id.clone();
        assert!(validate_ledger(&ledger).is_err());
    }

    #[test]
    fn dangling_observation_source_is_rejected() {
        let mut ledger = current_ledger();
        ledger.observations[0].source_id = "missing-source".into();
        assert!(validate_ledger(&ledger).is_err());
    }

    #[test]
    fn symbolic_rating_cannot_claim_audio_elo_comparability() {
        let mut ledger = current_ledger();
        let mirex = ledger
            .observations
            .iter_mut()
            .find(|item| item.observation_id == "mirex-damn-2026")
            .unwrap();
        mirex.directly_comparable_to_audio_elo = true;
        assert!(validate_ledger(&ledger).is_err());
    }

    #[test]
    fn metric_confidence_interval_requires_two_finite_ordered_bounds() {
        let mut ledger = current_ledger();
        {
            let metric = &mut ledger
                .observations
                .iter_mut()
                .find(|item| item.observation_id == "taan-timingbench")
                .unwrap()
                .metrics[0];
            metric.lower_bound = Some(99.0);
            metric.upper_bound = None;
        }
        assert!(validate_ledger(&ledger).is_err());

        {
            let metric = &mut ledger
                .observations
                .iter_mut()
                .find(|item| item.observation_id == "taan-timingbench")
                .unwrap()
                .metrics[0];
            metric.upper_bound = Some(98.0);
        }
        assert!(validate_ledger(&ledger).is_err());

        {
            let metric = &mut ledger
                .observations
                .iter_mut()
                .find(|item| item.observation_id == "taan-timingbench")
                .unwrap()
                .metrics[0];
            metric.upper_bound = Some(99.5);
            metric.confidence_level = Some(1.0);
        }
        assert!(validate_ledger(&ledger).is_err());
    }

    #[test]
    fn elo_metric_cannot_be_attached_to_non_comparable_cohort() {
        let mut ledger = current_ledger();
        let sme = ledger
            .observations
            .iter_mut()
            .find(|item| item.observation_id == "smebench-dataset-scope-2026")
            .unwrap();
        sme.metrics.push(MetricV1 {
            name: "human_pairwise_preference_elo".into(),
            value: 1000.0,
            unit: "Elo".into(),
            plus_minus: None,
            sample_size: None,
            lower_bound: None,
            upper_bound: None,
            confidence_level: None,
            scope: "invalid test metric".into(),
        });
        assert!(validate_ledger(&ledger).is_err());
    }
}
