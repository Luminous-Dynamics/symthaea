pub mod paired;
pub mod synthetic;

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub const PROTOCOL_ID: &str = "OPEN-OBS-001";
pub const PROTOCOL_VERSION: &str = "0.2";
pub const MANIFEST_HEADER: &str =
    "case_id\tbackground_group\tpartition\tstratum\ttruth\tdecision\tfamily_truth\tfamily_decision\tcapture\tdata_kind\tobservation_sha256\tbackground_sha256\tinstrument_profile_sha256\tscenario_profile_sha256\tdetector_profile_sha256";
const ONE_SIDED_95_Z: f64 = 1.644_853_626_951_472_2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Partition {
    Train,
    Calibration,
    Holdout,
}

impl Partition {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "train" => Ok(Self::Train),
            "calibration" => Ok(Self::Calibration),
            "holdout" => Ok(Self::Holdout),
            _ => Err(format!("unsupported partition {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Calibration => "calibration",
            Self::Holdout => "holdout",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Truth {
    Signal,
    NoSignal,
}

impl Truth {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "signal" => Ok(Self::Signal),
            "no_signal" => Ok(Self::NoSignal),
            _ => Err(format!("unsupported truth label {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::NoSignal => "no_signal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Decision {
    Candidate,
    Reject,
    Unknown,
}

impl Decision {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "candidate" => Ok(Self::Candidate),
            "reject" => Ok(Self::Reject),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("unsupported detector decision {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Reject => "reject",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum FamilyTruth {
    Known,
    HeldOut,
    NotApplicable,
}

impl FamilyTruth {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "known" => Ok(Self::Known),
            "held_out" => Ok(Self::HeldOut),
            "not_applicable" => Ok(Self::NotApplicable),
            _ => Err(format!("unsupported family truth {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Known => "known",
            Self::HeldOut => "held_out",
            Self::NotApplicable => "not_applicable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum FamilyDecision {
    Known,
    Unknown,
    NotAssigned,
}

impl FamilyDecision {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "known" => Ok(Self::Known),
            "unknown" => Ok(Self::Unknown),
            "not_assigned" => Ok(Self::NotAssigned),
            _ => Err(format!("unsupported family decision {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Known => "known",
            Self::Unknown => "unknown",
            Self::NotAssigned => "not_assigned",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Capture {
    Complete,
    Incomplete,
}

impl Capture {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "complete" => Ok(Self::Complete),
            "incomplete" => Ok(Self::Incomplete),
            _ => Err(format!("unsupported capture state {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Incomplete => "incomplete",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum DataKind {
    RawObservation,
    Derived,
    Simulated,
    IndependentMeasurement,
}

impl DataKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "raw_observation" => Ok(Self::RawObservation),
            "derived" => Ok(Self::Derived),
            "simulated" => Ok(Self::Simulated),
            "independent_measurement" => Ok(Self::IndependentMeasurement),
            _ => Err(format!("unsupported data kind {value:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawObservation => "raw_observation",
            Self::Derived => "derived",
            Self::Simulated => "simulated",
            Self::IndependentMeasurement => "independent_measurement",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub case_id: String,
    pub background_group: String,
    pub partition: Partition,
    pub stratum: String,
    pub truth: Truth,
    pub decision: Decision,
    pub family_truth: FamilyTruth,
    pub family_decision: FamilyDecision,
    pub capture: Capture,
    pub data_kind: DataKind,
    pub observation_sha256: String,
    pub background_sha256: String,
    pub instrument_profile_sha256: String,
    pub scenario_profile_sha256: String,
    pub detector_profile_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    pub sha256: String,
    pub cases: Vec<Case>,
}

fn digest_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_text(value: &str, line: usize, field: &str) -> Result<String, String> {
    if value.is_empty() || value.trim().is_empty() {
        return Err(format!("line {line}: {field} must not be empty"));
    }
    if value.trim() != value {
        return Err(format!("line {line}: {field} has leading/trailing whitespace"));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("line {line}: {field} contains a control character"));
    }
    Ok(value.to_owned())
}

pub fn parse_manifest(bytes: &[u8]) -> Result<Manifest, String> {
    if !bytes.ends_with(b"\n") {
        return Err("manifest must end with a single LF-terminated final record".to_owned());
    }
    if bytes.contains(&b'\r') {
        return Err("manifest must use canonical LF newlines, not CR/CRLF".to_owned());
    }

    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("manifest is not valid UTF-8: {e}"))?;
    let mut lines = text.split_terminator('\n');
    let header = lines.next().ok_or_else(|| "manifest is empty".to_owned())?;
    if header != MANIFEST_HEADER {
        return Err(format!(
            "unexpected header; required exact header is {MANIFEST_HEADER:?}"
        ));
    }

    let mut cases = Vec::new();
    let mut case_ids = HashSet::new();
    let mut observation_digests = HashSet::new();
    let mut background_digests = HashSet::new();
    let mut instrument_profile: Option<String> = None;
    let mut detector_profile: Option<String> = None;
    let mut scenario_profiles: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut group_partitions: BTreeMap<String, Partition> = BTreeMap::new();
    let mut used_background_groups: HashSet<String> = HashSet::new();
    let mut independent_trial_keys: HashSet<(String, String, String, String, String)> =
        HashSet::new();

    for (offset, line) in lines.enumerate() {
        let line_no = offset + 2;
        if line.is_empty() {
            return Err(format!("line {line_no}: blank records are not allowed"));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 15 {
            return Err(format!(
                "line {line_no}: expected 15 TSV fields, found {}",
                fields.len()
            ));
        }

        let case_id = required_text(fields[0], line_no, "case_id")?;
        let background_group = required_text(fields[1], line_no, "background_group")?;
        let partition_raw = required_text(fields[2], line_no, "partition")?;
        let stratum = required_text(fields[3], line_no, "stratum")?;
        let truth_raw = required_text(fields[4], line_no, "truth")?;
        let decision_raw = required_text(fields[5], line_no, "decision")?;
        let family_truth_raw = required_text(fields[6], line_no, "family_truth")?;
        let family_decision_raw = required_text(fields[7], line_no, "family_decision")?;
        let capture_raw = required_text(fields[8], line_no, "capture")?;
        let data_kind_raw = required_text(fields[9], line_no, "data_kind")?;
        let observation_sha256 = required_text(fields[10], line_no, "observation_sha256")?;
        let background_sha256 = required_text(fields[11], line_no, "background_sha256")?;
        let instrument_profile_sha256 =
            required_text(fields[12], line_no, "instrument_profile_sha256")?;
        let scenario_profile_sha256 =
            required_text(fields[13], line_no, "scenario_profile_sha256")?;
        let detector_profile_sha256 =
            required_text(fields[14], line_no, "detector_profile_sha256")?;

        for (field, value) in [
            ("observation_sha256", observation_sha256.as_str()),
            ("background_sha256", background_sha256.as_str()),
            ("instrument_profile_sha256", instrument_profile_sha256.as_str()),
            ("scenario_profile_sha256", scenario_profile_sha256.as_str()),
            ("detector_profile_sha256", detector_profile_sha256.as_str()),
        ] {
            if !valid_sha256(value) {
                return Err(format!(
                    "line {line_no}: {field} must be exactly 64 lowercase hexadecimal SHA-256 characters"
                ));
            }
        }

        if !case_ids.insert(case_id.clone()) {
            return Err(format!("line {line_no}: duplicate case_id {case_id:?}"));
        }
        if !observation_digests.insert(observation_sha256.clone()) {
            return Err(format!(
                "line {line_no}: duplicate observation bytes digest {observation_sha256:?}"
            ));
        }
        if !background_digests.insert(background_sha256.clone()) {
            return Err(format!(
                "line {line_no}: duplicate background bundle digest {background_sha256:?}; shared backgrounds must be one trial bundle"
            ));
        }
        match &instrument_profile {
            Some(previous) if previous != &instrument_profile_sha256 => {
                return Err(format!(
                    "line {line_no}: instrument profile changed within one manifest"
                ));
            }
            None => instrument_profile = Some(instrument_profile_sha256.clone()),
            _ => {}
        }
        match &detector_profile {
            Some(previous) if previous != &detector_profile_sha256 => {
                return Err(format!(
                    "line {line_no}: detector profile changed within one manifest"
                ));
            }
            None => detector_profile = Some(detector_profile_sha256.clone()),
            _ => {}
        }
        let scenario_key = (stratum.clone(), data_kind_raw.clone());
        match scenario_profiles.get(&scenario_key) {
            Some(previous) if previous != &scenario_profile_sha256 => {
                return Err(format!(
                    "line {line_no}: scenario profile changed within stratum {:?} and data kind {:?}",
                    stratum, data_kind_raw
                ));
            }
            None => {
                scenario_profiles.insert(scenario_key, scenario_profile_sha256.clone());
            }
            _ => {}
        }

        let partition = Partition::parse(&partition_raw)
            .map_err(|e| format!("line {line_no}: {e}"))?;
        let truth = Truth::parse(&truth_raw).map_err(|e| format!("line {line_no}: {e}"))?;
        let decision =
            Decision::parse(&decision_raw).map_err(|e| format!("line {line_no}: {e}"))?;
        let family_truth = FamilyTruth::parse(&family_truth_raw)
            .map_err(|e| format!("line {line_no}: {e}"))?;
        let family_decision = FamilyDecision::parse(&family_decision_raw)
            .map_err(|e| format!("line {line_no}: {e}"))?;
        let capture =
            Capture::parse(&capture_raw).map_err(|e| format!("line {line_no}: {e}"))?;
        let data_kind =
            DataKind::parse(&data_kind_raw).map_err(|e| format!("line {line_no}: {e}"))?;

        match group_partitions.get(&background_group) {
            Some(previous) if *previous != partition => {
                return Err(format!(
                    "line {line_no}: background_group {background_group:?} crosses {} and {} partitions",
                    previous.as_str(),
                    partition.as_str()
                ));
            }
            None => {
                group_partitions.insert(background_group.clone(), partition);
            }
            _ => {}
        }

        // V0 treats an independent background/acquisition group as one scored
        // trial bundle across all strata and truth labels. Paired ON/OFF or
        // target/control panels must live inside that bundle, not as multiple
        // rows that masquerade as independent observations.
        if !used_background_groups.insert(background_group.clone()) {
            return Err(format!(
                "line {line_no}: background_group {background_group:?} is reused across cases; aggregate paired panels into one trial bundle"
            ));
        }

        let trial_key = (
            background_group.clone(),
            partition.as_str().to_owned(),
            stratum.clone(),
            truth.as_str().to_owned(),
            data_kind.as_str().to_owned(),
        );
        if !independent_trial_keys.insert(trial_key) {
            return Err(format!(
                "line {line_no}: repeated independent trial for background_group={background_group:?}, partition={}, stratum={stratum:?}, truth={}",
                partition.as_str(),
                truth.as_str()
            ));
        }

        match truth {
            Truth::Signal if family_truth == FamilyTruth::NotApplicable => {
                return Err(format!(
                    "line {line_no}: a signal requires family_truth=known or held_out"
                ));
            }
            Truth::NoSignal if family_truth != FamilyTruth::NotApplicable => {
                return Err(format!(
                    "line {line_no}: no_signal requires family_truth=not_applicable"
                ));
            }
            _ => {}
        }

        if decision != Decision::Candidate && family_decision != FamilyDecision::NotAssigned {
            return Err(format!(
                "line {line_no}: a non-candidate disposition must use family_decision=not_assigned"
            ));
        }
        if capture == Capture::Incomplete
            && (decision != Decision::Unknown || family_decision == FamilyDecision::Known)
        {
            return Err(format!(
                "line {line_no}: incomplete capture must remain unknown and cannot carry a known-family assignment"
            ));
        }

        cases.push(Case {
            case_id,
            background_group,
            partition,
            stratum,
            truth,
            decision,
            family_truth,
            family_decision,
            capture,
            data_kind,
            observation_sha256,
            background_sha256,
            instrument_profile_sha256,
            scenario_profile_sha256,
            detector_profile_sha256,
        });
    }

    if cases.is_empty() {
        return Err("manifest must contain at least one case".to_owned());
    }

    Ok(Manifest {
        sha256: digest_hex(bytes),
        cases,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}

pub fn wilson_one_sided_95(successes: u64, trials: u64) -> Option<Bounds> {
    if trials == 0 || successes > trials {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = ONE_SIDED_95_Z * ONE_SIDED_95_Z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = ONE_SIDED_95_Z
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    Some(Bounds {
        lower: (center - half).max(0.0),
        upper: (center + half).min(1.0),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gate {
    Pass,
    Fail,
    Inconclusive,
}

impl Gate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataKindGates {
    pub data_kind: DataKind,
    pub positive_strata: u64,
    pub negative_strata: u64,
    pub detection: Gate,
    pub false_positive: Gate,
    pub open_set: Gate,
}

#[derive(Clone, Debug)]
pub struct StratumMetrics {
    pub stratum: String,
    pub data_kind: DataKind,
    pub cases: u64,
    pub independent_signal_groups: u64,
    pub independent_negative_groups: u64,
    pub true_positive: u64,
    pub false_negative: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub unknown_signal: u64,
    pub unknown_negative: u64,
    pub held_out_signal_candidates: u64,
    pub held_out_unknown_labels: u64,
    pub forced_known_labels: u64,
    pub recall: Option<f64>,
    pub recall_bounds: Option<Bounds>,
    pub false_positive_rate: Option<f64>,
    pub fpr_bounds: Option<Bounds>,
}

impl StratumMetrics {
    fn from_cases(stratum: String, data_kind: DataKind, cases: &[&Case]) -> Self {
        let mut metrics = Self {
            stratum,
            data_kind,
            cases: cases.len() as u64,
            independent_signal_groups: 0,
            independent_negative_groups: 0,
            true_positive: 0,
            false_negative: 0,
            false_positive: 0,
            true_negative: 0,
            unknown_signal: 0,
            unknown_negative: 0,
            held_out_signal_candidates: 0,
            held_out_unknown_labels: 0,
            forced_known_labels: 0,
            recall: None,
            recall_bounds: None,
            false_positive_rate: None,
            fpr_bounds: None,
        };
        let mut signal_groups = BTreeSet::new();
        let mut negative_groups = BTreeSet::new();

        for case in cases {
            match case.truth {
                Truth::Signal => {
                    signal_groups.insert(case.background_group.as_str());
                    match case.decision {
                        Decision::Candidate => metrics.true_positive += 1,
                        Decision::Reject => metrics.false_negative += 1,
                        Decision::Unknown => {
                            metrics.false_negative += 1;
                            metrics.unknown_signal += 1;
                        }
                    }
                    if case.family_truth == FamilyTruth::HeldOut
                        && case.decision == Decision::Candidate
                    {
                        metrics.held_out_signal_candidates += 1;
                        match case.family_decision {
                            FamilyDecision::Unknown => metrics.held_out_unknown_labels += 1,
                            FamilyDecision::Known => metrics.forced_known_labels += 1,
                            FamilyDecision::NotAssigned => {}
                        }
                    }
                }
                Truth::NoSignal => {
                    negative_groups.insert(case.background_group.as_str());
                    match case.decision {
                        Decision::Candidate => metrics.false_positive += 1,
                        Decision::Reject => metrics.true_negative += 1,
                        Decision::Unknown => metrics.unknown_negative += 1,
                    }
                }
            }
        }

        metrics.independent_signal_groups = signal_groups.len() as u64;
        metrics.independent_negative_groups = negative_groups.len() as u64;
        let positive_trials = metrics.true_positive + metrics.false_negative;
        let negative_trials =
            metrics.false_positive + metrics.true_negative + metrics.unknown_negative;
        metrics.recall = rate(metrics.true_positive, positive_trials);
        metrics.recall_bounds =
            wilson_one_sided_95(metrics.true_positive, positive_trials);
        metrics.false_positive_rate = rate(metrics.false_positive, negative_trials);
        metrics.fpr_bounds =
            wilson_one_sided_95(metrics.false_positive, negative_trials);
        metrics
    }
}

fn rate(successes: u64, trials: u64) -> Option<f64> {
    if trials == 0 {
        None
    } else {
        Some(successes as f64 / trials as f64)
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub manifest_sha256: String,
    pub source_revision: String,
    pub manifest_rows: u64,
    pub holdout_rows: u64,
    pub scored_holdout_rows: u64,
    pub blocked_incomplete_holdout_rows: u64,
    pub detection_gate: Gate,
    pub false_positive_gate: Gate,
    pub open_set_gate: Gate,
    pub data_kind_gates: Vec<DataKindGates>,
    pub overall_gate: Gate,
    pub held_out_signal_trials: u64,
    pub held_out_signal_candidates: u64,
    pub forced_known_labels: u64,
    pub metrics: Vec<StratumMetrics>,
    pub case_lines: Vec<String>,
}

fn compare_recall(metrics: &[StratumMetrics], minimum: f64) -> Gate {
    let mut seen = false;
    let mut insufficient = false;
    for row in metrics {
        let trials = row.true_positive + row.false_negative;
        if trials == 0 {
            continue;
        }
        seen = true;
        let point = row.recall.unwrap_or(0.0);
        if point < minimum {
            return Gate::Fail;
        }
        if row.recall_bounds.map(|b| b.lower < minimum).unwrap_or(true) {
            insufficient = true;
        }
    }
    if !seen || insufficient {
        Gate::Inconclusive
    } else {
        Gate::Pass
    }
}

fn compare_fpr(metrics: &[StratumMetrics], maximum: f64) -> Gate {
    let mut seen = false;
    let mut insufficient = false;
    for row in metrics {
        let trials = row.false_positive + row.true_negative + row.unknown_negative;
        if trials == 0 {
            continue;
        }
        seen = true;
        let point = row.false_positive_rate.unwrap_or(0.0);
        if point > maximum {
            return Gate::Fail;
        }
        if row.fpr_bounds.map(|b| b.upper > maximum).unwrap_or(true) {
            insufficient = true;
        }
    }
    if !seen || insufficient {
        Gate::Inconclusive
    } else {
        Gate::Pass
    }
}

fn combine_gates(gates: &[Gate]) -> Gate {
    if gates.iter().any(|gate| *gate == Gate::Fail) {
        Gate::Fail
    } else if !gates.is_empty() && gates.iter().all(|gate| *gate == Gate::Pass) {
        Gate::Pass
    } else {
        Gate::Inconclusive
    }
}

fn evaluate_open_set(held_out_signal_cases: &[&Case]) -> Gate {
    if held_out_signal_cases.is_empty() {
        return Gate::Inconclusive;
    }
    let forced_known_labels = held_out_signal_cases
        .iter()
        .filter(|case| {
            case.decision == Decision::Candidate
                && case.family_decision == FamilyDecision::Known
        })
        .count();
    if forced_known_labels > 0 {
        return Gate::Fail;
    }
    let held_out_unknown_labels = held_out_signal_cases
        .iter()
        .filter(|case| {
            case.decision == Decision::Candidate
                && case.family_decision == FamilyDecision::Unknown
        })
        .count();
    if held_out_unknown_labels > 0 {
        Gate::Pass
    } else {
        Gate::Inconclusive
    }
}

pub fn evaluate(manifest: &Manifest, source_revision: &str) -> Report {
    let holdout_rows: Vec<&Case> = manifest
        .cases
        .iter()
        .filter(|case| case.partition == Partition::Holdout)
        .collect();
    let incomplete = holdout_rows
        .iter()
        .filter(|case| case.capture == Capture::Incomplete)
        .count() as u64;
    let scored: Vec<&Case> = holdout_rows
        .iter()
        .copied()
        .filter(|case| case.capture == Capture::Complete)
        .collect();

    let mut grouped: BTreeMap<(String, DataKind), Vec<&Case>> = BTreeMap::new();
    for case in &scored {
        grouped
            .entry((case.stratum.clone(), case.data_kind))
            .or_default()
            .push(case);
    }

    let metrics: Vec<StratumMetrics> = grouped
        .into_iter()
        .map(|((stratum, data_kind), cases)| {
            StratumMetrics::from_cases(stratum, data_kind, &cases)
        })
        .collect();

    // Never let simulated detection results compensate for a lack of measured
    // controls (or vice versa). Every data-kind slice must qualify independently.
    let kinds: BTreeSet<DataKind> = scored.iter().map(|case| case.data_kind).collect();
    let mut data_kind_gates = Vec::new();
    for data_kind in kinds {
        let kind_metrics: Vec<StratumMetrics> = metrics
            .iter()
            .filter(|row| row.data_kind == data_kind)
            .cloned()
            .collect();
        let kind_cases: Vec<&Case> = scored
            .iter()
            .copied()
            .filter(|case| case.data_kind == data_kind)
            .collect();
        let held_out: Vec<&Case> = kind_cases
            .iter()
            .copied()
            .filter(|case| {
                case.truth == Truth::Signal && case.family_truth == FamilyTruth::HeldOut
            })
            .collect();
        data_kind_gates.push(DataKindGates {
            data_kind,
            positive_strata: kind_metrics
                .iter()
                .filter(|row| row.true_positive + row.false_negative > 0)
                .count() as u64,
            negative_strata: kind_metrics
                .iter()
                .filter(|row| row.false_positive + row.true_negative + row.unknown_negative > 0)
                .count() as u64,
            detection: compare_recall(&kind_metrics, 0.90),
            false_positive: compare_fpr(&kind_metrics, 0.01),
            open_set: evaluate_open_set(&held_out),
        });
    }

    let detection_gate = combine_gates(
        &data_kind_gates.iter().map(|g| g.detection).collect::<Vec<_>>()
    );
    let false_positive_gate = combine_gates(
        &data_kind_gates.iter().map(|g| g.false_positive).collect::<Vec<_>>()
    );
    let open_set_gate = combine_gates(
        &data_kind_gates.iter().map(|g| g.open_set).collect::<Vec<_>>()
    );

    let held_out_signal_cases: Vec<&Case> = scored
        .iter()
        .copied()
        .filter(|case| case.truth == Truth::Signal && case.family_truth == FamilyTruth::HeldOut)
        .collect();
    let held_out_signal_trials = held_out_signal_cases.len() as u64;
    let held_out_signal_candidates = held_out_signal_cases
        .iter()
        .filter(|case| case.decision == Decision::Candidate)
        .count() as u64;
    let forced_known_labels = held_out_signal_cases
        .iter()
        .filter(|case| {
            case.decision == Decision::Candidate
                && case.family_decision == FamilyDecision::Known
        })
        .count() as u64;

    let gates = [detection_gate, false_positive_gate, open_set_gate];
    let overall_gate = combine_gates(&gates);

    let mut case_lines = Vec::new();
    let mut ordered: Vec<&Case> = holdout_rows;
    ordered.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    for case in ordered {
        let disposition = if case.capture == Capture::Incomplete {
            "BLOCKED_INCOMPLETE_CAPTURE"
        } else if case.truth == Truth::Signal
            && case.decision != Decision::Candidate
        {
            "MISSED_SIGNAL"
        } else if case.truth == Truth::NoSignal && case.decision == Decision::Candidate {
            "FALSE_ALARM"
        } else {
            "OBSERVED"
        };
        case_lines.push(format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.case_id,
            case.background_group,
            case.partition.as_str(),
            case.stratum,
            case.truth.as_str(),
            case.decision.as_str(),
            case.family_truth.as_str(),
            case.family_decision.as_str(),
            case.capture.as_str(),
            case.data_kind.as_str(),
            case.observation_sha256,
            case.background_sha256,
            case.instrument_profile_sha256,
            case.scenario_profile_sha256,
            case.detector_profile_sha256,
            disposition
        ));
    }

    Report {
        manifest_sha256: manifest.sha256.clone(),
        source_revision: source_revision.to_owned(),
        manifest_rows: manifest.cases.len() as u64,
        holdout_rows: (manifest.cases.len()
            - manifest.cases.iter().filter(|c| c.partition != Partition::Holdout).count())
            as u64,
        scored_holdout_rows: scored.len() as u64,
        blocked_incomplete_holdout_rows: incomplete,
        detection_gate,
        false_positive_gate,
        open_set_gate,
        data_kind_gates,
        overall_gate,
        held_out_signal_trials,
        held_out_signal_candidates,
        forced_known_labels,
        metrics,
        case_lines,
    }
}

fn option_number(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "NA".to_owned())
}

fn option_bounds(value: Option<Bounds>) -> String {
    value
        .map(|b| format!("{:.6}..{:.6}", b.lower, b.upper))
        .unwrap_or_else(|| "NA".to_owned())
}

impl Report {
    pub fn render_tsv(&self) -> String {
        let mut out = String::new();
        out.push_str("OPEN_OBS_REPORT\t1\n");
        out.push_str(&format!("protocol\t{}\t{}\n", PROTOCOL_ID, PROTOCOL_VERSION));
        out.push_str(&format!("source_revision\t{}\n", self.source_revision));
        out.push_str(&format!("manifest_sha256\t{}\n", self.manifest_sha256));
        out.push_str(&format!("manifest_rows\t{}\n", self.manifest_rows));
        out.push_str(&format!("holdout_rows\t{}\n", self.holdout_rows));
        out.push_str(&format!("scored_holdout_rows\t{}\n", self.scored_holdout_rows));
        out.push_str(&format!(
            "blocked_incomplete_holdout_rows\t{}\n",
            self.blocked_incomplete_holdout_rows
        ));
        out.push_str(&format!(
            "held_out_signal_trials\t{}\n",
            self.held_out_signal_trials
        ));
        out.push_str(&format!(
            "held_out_signal_candidates\t{}\n",
            self.held_out_signal_candidates
        ));
        out.push_str(&format!("forced_known_labels\t{}\n", self.forced_known_labels));
        out.push_str(&format!("gate_detection\t{}\n", self.detection_gate.as_str()));
        out.push_str(&format!("gate_false_positive\t{}\n", self.false_positive_gate.as_str()));
        out.push_str(&format!("gate_open_set\t{}\n", self.open_set_gate.as_str()));
        out.push_str(&format!("qualification\t{}\n", self.overall_gate.as_str()));
        out.push_str("data_kind_gate\tkind\tpositive_strata\tnegative_strata\tdetection\tfalse_positive\topen_set\n");
        for gate in &self.data_kind_gates {
            out.push_str(&format!(
                "data_kind_gate\t{}\t{}\t{}\t{}\t{}\t{}\n",
                gate.data_kind.as_str(),
                gate.positive_strata,
                gate.negative_strata,
                gate.detection.as_str(),
                gate.false_positive.as_str(),
                gate.open_set.as_str()
            ));
        }
        out.push_str("metric\tstratum\tdata_kind\tcases\tsignal_groups\tnegative_groups\tTP\tFN\tFP\tTN\tunknown_signal\tunknown_negative\trecall\trecall_95_one_sided_bounds\tfpr\tfpr_95_one_sided_bounds\tforced_known_labels\n");

        for row in &self.metrics {
            out.push_str(&format!(
                "metrics\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                row.stratum,
                row.data_kind.as_str(),
                row.cases,
                row.independent_signal_groups,
                row.independent_negative_groups,
                row.true_positive,
                row.false_negative,
                row.false_positive,
                row.true_negative,
                row.unknown_signal,
                row.unknown_negative,
                option_number(row.recall),
                option_bounds(row.recall_bounds),
                option_number(row.false_positive_rate),
                option_bounds(row.fpr_bounds),
                row.forced_known_labels
            ));
        }

        out.push_str("case\tcase_id\tbackground_group\tpartition\tstratum\ttruth\tdecision\tfamily_truth\tfamily_decision\tcapture\tdata_kind\tobservation_sha256\tbackground_sha256\tinstrument_profile_sha256\tscenario_profile_sha256\tdetector_profile_sha256\tdisposition\n");
        for line in &self.case_lines {
            out.push_str("case\t");
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_test_digests(content: &str) -> Result<Manifest, String> {
        let mut lines = content.split_terminator('\n');
        let _legacy_or_current_header = lines
            .next()
            .ok_or_else(|| "test manifest is empty".to_owned())?;
        let mut canonical = format!("{MANIFEST_HEADER}\n");
        let instrument = digest_hex(b"test-instrument-profile-v1");
        let detector = digest_hex(b"test-detector-profile-v1");
        for line in lines {
            if line.is_empty() {
                canonical.push('\n');
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 10 {
                return Err(format!("test helper expected legacy 10-column row, found {}", fields.len()));
            }
            let observation = digest_hex(line.as_bytes());
            let background = digest_hex(format!("background:{line}").as_bytes());
            let scenario =
                digest_hex(format!("scenario:{}:{}", fields[3], fields[9]).as_bytes());
            canonical.push_str(line);
            canonical.push_str(&format!(
                "\t{observation}\t{background}\t{instrument}\t{scenario}\t{detector}\n"
            ));
        }
        parse_manifest(canonical.as_bytes())
    }

    fn parse_test_manifest(content: &str) -> Result<Manifest, String> {
        with_test_digests(content)
    }

    fn manifest(rows: &[&str]) -> Manifest {
        let mut content = format!("{MANIFEST_HEADER}\n");
        for row in rows {
            content.push_str(row);
            content.push('\n');
        }
        parse_test_manifest(&content).expect("test manifest should parse")
    }

    fn rewrite_tsv_field(bytes: &[u8], line_index: usize, field_index: usize, value: &str) -> Vec<u8> {
        let text = std::str::from_utf8(bytes).expect("generated corpus utf8");
        let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
        let fields: Vec<&str> = lines[line_index].split('\t').collect();
        assert!(field_index < fields.len());
        let mut owned: Vec<String> = fields.into_iter().map(str::to_owned).collect();
        owned[field_index] = value.to_owned();
        lines[line_index] = owned.join("\t");
        format!("{}\n", lines.join("\n")).into_bytes()
    }

    #[test]
    fn rejects_malformed_observation_digest() {
        let valid = synthetic::generate_manifest().expect("generated v0.2 manifest");
        let malformed = rewrite_tsv_field(&valid, 1, 10, "not-a-sha256");
        let error = parse_manifest(&malformed).unwrap_err();
        assert!(error.contains("observation_sha256"));
        assert!(error.contains("64 lowercase hexadecimal"));
    }

    #[test]
    fn rejects_duplicate_observation_content_digest() {
        let valid = synthetic::generate_manifest().expect("generated v0.2 manifest");
        let text = std::str::from_utf8(&valid).expect("generated corpus utf8");
        let first_digest = text.lines().nth(1).unwrap().split('\t').nth(10).unwrap();
        let duplicate = rewrite_tsv_field(&valid, 2, 10, first_digest);
        let error = parse_manifest(&duplicate).unwrap_err();
        assert!(error.contains("duplicate observation bytes digest"));
    }

    #[test]
    fn rejects_instrument_or_detector_profile_changes_within_manifest() {
        let valid = synthetic::generate_manifest().expect("generated v0.2 manifest");
        let changed_instrument =
            rewrite_tsv_field(&valid, 2, 12, &digest_hex(b"different instrument profile"));
        let error = parse_manifest(&changed_instrument).unwrap_err();
        assert!(error.contains("instrument profile changed"));

        let changed_detector =
            rewrite_tsv_field(&valid, 2, 14, &digest_hex(b"different detector profile"));
        let error = parse_manifest(&changed_detector).unwrap_err();
        assert!(error.contains("detector profile changed"));
    }

    #[test]
    fn rejects_scenario_profile_changes_inside_one_stratum() {
        let valid = synthetic::generate_manifest().expect("generated v0.2 manifest");
        // Rows 1 and 3 are both the same train/known-linear scenario stratum.
        let changed =
            rewrite_tsv_field(&valid, 3, 13, &digest_hex(b"wrong scenario profile"));
        let error = parse_manifest(&changed).unwrap_err();
        assert!(error.contains("scenario profile changed"));
    }

    #[test]
    fn render_receipt_preserves_all_five_digests_per_case() {
        let parsed = synthetic::generate_manifest()
            .and_then(|bytes| parse_manifest(&bytes))
            .expect("generated v0.2 manifest");
        let report = evaluate(&parsed, &"e".repeat(40));
        let rendered = report.render_tsv();
        assert!(rendered.contains("observation_sha256"));
        assert!(rendered.contains("background_sha256"));
        assert!(rendered.contains("instrument_profile_sha256"));
        assert!(rendered.contains("scenario_profile_sha256"));
        assert!(rendered.contains("detector_profile_sha256"));
        assert!(rendered.contains("obs-00080-"));
    }



    fn valid_rows() -> Vec<&'static str> {
        vec![
            "train-1\tbg-train-1\ttrain\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "cal-1\tbg-cal-1\tcalibration\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-pos-1\tbg-pos-1\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-neg-1\tbg-neg-1\tholdout\tquiet_control\tno_signal\treject\tnot_applicable\tnot_assigned\tcomplete\tsimulated",
            "hold-new-1\tbg-new-1\tholdout\theld_out_signal\tsignal\tcandidate\theld_out\tunknown\tcomplete\tsimulated",
        ]
    }

    #[test]
    fn rejects_background_group_crossing_partitions() {
        let rows = [
            "train-1\tbg-shared\ttrain\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-1\tbg-shared\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n{}\n", rows[0], rows[1]);
        let err = parse_test_manifest(&content).unwrap_err();
        assert!(err.contains("crosses"));
    }

    #[test]
    fn rejects_pseudoreplicated_trials_from_one_background() {
        let rows = [
            "hold-1\tbg-same\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-2\tbg-same\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n{}\n", rows[0], rows[1]);
        let err = parse_test_manifest(&content).unwrap_err();
        assert!(err.contains("reused across cases"));
    }

    #[test]
    fn rejects_same_background_reused_across_different_strata() {
        let rows = [
            "hold-1\tbg-same\tholdout\tlow_snr\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-2\tbg-same\tholdout\thigh_snr\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n{}\n", rows[0], rows[1]);
        let err = parse_test_manifest(&content).unwrap_err();
        assert!(err.contains("reused across cases"));
    }

    #[test]
    fn rejects_promotion_from_incomplete_capture() {
        let rows = [
            "hold-1\tbg-same\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tincomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n", rows[0]);
        let err = parse_test_manifest(&content).unwrap_err();
        assert!(err.contains("incomplete capture"));
    }

    #[test]
    fn computes_recall_and_false_alarm_counts_separately() {
        let parsed = manifest(&valid_rows());
        let report = evaluate(&parsed, &"a".repeat(40));
        let signal = report
            .metrics
            .iter()
            .find(|m| m.stratum == "known_signal")
            .expect("signal metrics");
        let control = report
            .metrics
            .iter()
            .find(|m| m.stratum == "quiet_control")
            .expect("control metrics");
        assert_eq!(signal.true_positive, 1);
        assert_eq!(signal.false_negative, 0);
        assert_eq!(control.false_positive, 0);
        assert_eq!(control.true_negative, 1);
        assert_eq!(report.forced_known_labels, 0);
    }

    #[test]
    fn fails_open_set_gate_when_held_out_signal_is_forced_into_known_family() {
        let rows = [
            "train-1\tbg-train\ttrain\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "hold-new\tbg-new\tholdout\theld_out_signal\tsignal\tcandidate\theld_out\tknown\tcomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n{}\n", rows[0], rows[1]);
        let parsed = parse_test_manifest(&content).expect("valid evaluated failure");
        let report = evaluate(&parsed, &"b".repeat(40));
        assert_eq!(report.open_set_gate, Gate::Fail);
        assert_eq!(report.forced_known_labels, 1);
    }

    #[test]
    fn confidence_bound_respects_independent_trial_count() {
        let one_percent = wilson_one_sided_95(0, 300).expect("nonzero trials");
        let one_tenth_percent = wilson_one_sided_95(0, 3_000).expect("nonzero trials");
        assert!(one_percent.upper < 0.01);
        assert!(one_tenth_percent.upper < 0.001);
        assert!(wilson_one_sided_95(0, 0).is_none());
    }

    #[test]
    fn rejects_duplicate_case_ids() {
        let rows = [
            "same-id\tbg-1\ttrain\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
            "same-id\tbg-2\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated",
        ];
        let content = format!("{MANIFEST_HEADER}\n{}\n{}\n", rows[0], rows[1]);
        let err = parse_test_manifest(&content).unwrap_err();
        assert!(err.contains("duplicate case_id"));
    }

    #[test]
    fn does_not_pool_simulated_success_with_a_missing_raw_detection_arm() {
        let mut content = format!("{MANIFEST_HEADER}\n");
        for row in valid_rows() {
            content.push_str(row);
            content.push('\n');
        }

        // A synthetic-only candidate path with enough independent cases to pass
        // the point-estimate and one-sided-bound gates.
        for i in 0..40 {
            content.push_str(&format!(
                "sim-pos-{i}\tbg-sim-pos-{i}\tholdout\tknown_signal\tsignal\tcandidate\tknown\tknown\tcomplete\tsimulated\n"
            ));
        }
        for i in 0..40 {
            content.push_str(&format!(
                "sim-new-{i}\tbg-sim-new-{i}\tholdout\theld_out_signal\tsignal\tcandidate\theld_out\tunknown\tcomplete\tsimulated\n"
            ));
        }
        for i in 0..399 {
            content.push_str(&format!(
                "sim-neg-{i}\tbg-sim-neg-{i}\tholdout\tquiet_control\tno_signal\treject\tnot_applicable\tnot_assigned\tcomplete\tsimulated\n"
            ));
        }

        // Raw controls alone cannot stand in for a raw positive / held-out arm.
        for i in 0..400 {
            content.push_str(&format!(
                "raw-neg-{i}\tbg-raw-neg-{i}\tholdout\tquiet_control\tno_signal\treject\tnot_applicable\tnot_assigned\tcomplete\traw_observation\n"
            ));
        }

        let parsed = parse_test_manifest(&content).expect("mixed evidence manifest");
        let report = evaluate(&parsed, &"c".repeat(40));
        let simulated = report
            .data_kind_gates
            .iter()
            .find(|g| g.data_kind == DataKind::Simulated)
            .expect("simulated gate");
        let raw = report
            .data_kind_gates
            .iter()
            .find(|g| g.data_kind == DataKind::RawObservation)
            .expect("raw observation gate");

        assert_eq!(simulated.detection, Gate::Pass);
        assert_eq!(simulated.false_positive, Gate::Pass);
        assert_eq!(simulated.open_set, Gate::Pass);
        assert_eq!(raw.detection, Gate::Inconclusive);
        assert_eq!(raw.open_set, Gate::Inconclusive);
        assert_eq!(report.overall_gate, Gate::Inconclusive);
    }

    #[test]
    fn incomplete_holdout_is_recorded_but_never_scored_as_a_detection() {
        let rows = [
            "hold-incomplete\tbg-incomplete\tholdout\theld_out_signal\tsignal\tunknown\theld_out\tnot_assigned\tincomplete\tsimulated",
        ];
        let parsed = manifest(&rows);
        let report = evaluate(&parsed, &"d".repeat(40));
        assert_eq!(report.blocked_incomplete_holdout_rows, 1);
        assert_eq!(report.scored_holdout_rows, 0);
        assert!(report.case_lines[0].ends_with("BLOCKED_INCOMPLETE_CAPTURE"));
        let rendered = report.render_tsv();
        assert!(rendered.contains("incomplete\tsimulated"));
        assert!(rendered.contains("BLOCKED_INCOMPLETE_CAPTURE"));
    }
}
