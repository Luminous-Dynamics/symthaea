use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub mod transform_qualification;

const SCHEMA_VERSION: &str = "mel-judge-grounding-probe-panel-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbePanelV1 {
    schema_version: String,
    experiment_id: String,
    frozen_at_utc: String,
    judge: JudgeIdentityV1,
    policy: ProbePolicyV1,
    cases: Vec<ProbeCaseV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JudgeIdentityV1 {
    judge_id: String,
    judge_version: String,
    adapter_sha256: String,
    protocol_sha256: String,
    rubric_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbePolicyV1 {
    minimum_baseline_margin: f64,
    maximum_prompt_only_margin_ratio: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeCaseV1 {
    case_id: String,
    candidate_a_id: String,
    candidate_b_id: String,
    candidate_a_audio_sha256: String,
    candidate_b_audio_sha256: String,
    human_reference_winner: String,
    human_reference_directional: bool,
    baseline: BaselineOutcomeV1,
    probes: Vec<ProbeV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineOutcomeV1 {
    score_a: f64,
    score_b: f64,
    prompt_sha256: String,
    instruction_sha256: String,
    audio_a_sha256: String,
    audio_b_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeV1 {
    probe_id: String,
    kind: String,
    score_a: f64,
    score_b: f64,
    prompt_sha256: String,
    instruction_sha256: String,
    audio_a_sha256: Option<String>,
    audio_b_sha256: Option<String>,
    expected_candidate: Option<String>,
    maximum_margin_shift: Option<f64>,
    maximum_score_drift: Option<f64>,
    minimum_score_drop: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeSummary {
    pub probe_count: usize,
    pub evaluable_count: usize,
    pub passed_count: usize,
    pub pass_rate: Option<f64>,
    pub mean_absolute_margin_shift: Option<f64>,
    pub mean_maximum_absolute_score_drift: Option<f64>,
    pub mean_margin_ratio: Option<f64>,
    pub mean_expected_candidate_score_drop: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeFinding {
    pub case_id: String,
    pub probe_id: String,
    pub kind: String,
    pub evaluable: bool,
    pub passed: Option<bool>,
    pub observed_value: Option<f64>,
    pub threshold: Option<f64>,
    pub candidate_a_absolute_score_drift: Option<f64>,
    pub candidate_b_absolute_score_drift: Option<f64>,
    pub maximum_absolute_score_drift: Option<f64>,
    pub pair_margin_absolute_drift: Option<f64>,
    pub maximum_margin_shift_threshold: Option<f64>,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct GroundingReport {
    pub schema_version: String,
    pub experiment_id: String,
    pub judge_id: String,
    pub judge_version: String,
    pub case_count: usize,
    pub total_probe_count: usize,
    pub human_directional_case_count: usize,
    pub baseline_correct_count: usize,
    pub baseline_human_reference_accuracy: Option<f64>,
    pub prompt_only: ProbeSummary,
    pub audio_swap: ProbeSummary,
    pub instruction_paraphrase: ProbeSummary,
    pub nuisance_transform: ProbeSummary,
    pub semantic_mismatch: ProbeSummary,
    pub findings: Vec<ProbeFinding>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Default)]
struct SummaryAccumulator {
    probes: usize,
    evaluable: usize,
    passed: usize,
    absolute_margin_shift_sum: f64,
    margin_shift_count: usize,
    margin_ratio_sum: f64,
    margin_ratio_count: usize,
    score_drop_sum: f64,
    score_drop_count: usize,
    maximum_absolute_score_drift_sum: f64,
    maximum_absolute_score_drift_count: usize,
}

impl SummaryAccumulator {
    fn add(&mut self, passed: bool, observed: Option<f64>, metric_kind: MetricKind) {
        self.probes += 1;
        self.evaluable += 1;
        if passed {
            self.passed += 1;
        }
        if let Some(value) = observed {
            match metric_kind {
                MetricKind::MarginShift => {
                    self.absolute_margin_shift_sum += value;
                    self.margin_shift_count += 1;
                }
                MetricKind::MarginRatio => {
                    self.margin_ratio_sum += value;
                    self.margin_ratio_count += 1;
                }
                MetricKind::ScoreDrop => {
                    self.score_drop_sum += value;
                    self.score_drop_count += 1;
                }
            }
        }
    }

    fn add_maximum_absolute_score_drift(&mut self, value: f64) {
        self.maximum_absolute_score_drift_sum += value;
        self.maximum_absolute_score_drift_count += 1;
    }

    fn skip(&mut self) {
        self.probes += 1;
    }

    fn finish(self) -> ProbeSummary {
        ProbeSummary {
            probe_count: self.probes,
            evaluable_count: self.evaluable,
            passed_count: self.passed,
            pass_rate: rate(self.passed, self.evaluable),
            mean_absolute_margin_shift: mean(
                self.absolute_margin_shift_sum,
                self.margin_shift_count,
            ),
            mean_maximum_absolute_score_drift: mean(
                self.maximum_absolute_score_drift_sum,
                self.maximum_absolute_score_drift_count,
            ),
            mean_margin_ratio: mean(self.margin_ratio_sum, self.margin_ratio_count),
            mean_expected_candidate_score_drop: mean(self.score_drop_sum, self.score_drop_count),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MetricKind {
    MarginShift,
    MarginRatio,
    ScoreDrop,
}

pub fn parse_and_audit(input: &str) -> Result<GroundingReport, String> {
    let panel: ProbePanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid probe panel JSON: {error}"))?;
    audit_panel(&panel)
}

fn audit_panel(panel: &ProbePanelV1) -> Result<GroundingReport, String> {
    validate_header(panel)?;

    let mut case_ids = BTreeSet::new();
    let mut findings = Vec::new();
    let mut prompt_only = SummaryAccumulator::default();
    let mut audio_swap = SummaryAccumulator::default();
    let mut paraphrase = SummaryAccumulator::default();
    let mut nuisance = SummaryAccumulator::default();
    let mut semantic_mismatch = SummaryAccumulator::default();

    let mut human_directional_case_count = 0usize;
    let mut baseline_correct_count = 0usize;
    let mut total_probe_count = 0usize;

    for case in &panel.cases {
        validate_case(case)?;
        if !case_ids.insert(case.case_id.as_str()) {
            return Err(format!("duplicate case_id: {}", case.case_id));
        }

        let baseline_margin = case.baseline.score_a - case.baseline.score_b;
        if case.human_reference_directional {
            human_directional_case_count += 1;
            let human_winner = case.human_reference_winner.as_str();
            if (human_winner == "a" && baseline_margin > 0.0)
                || (human_winner == "b" && baseline_margin < 0.0)
            {
                baseline_correct_count += 1;
            }
        }

        let mut probe_ids = BTreeSet::new();
        let mut probe_kinds = BTreeSet::new();
        for probe in &case.probes {
            probe_kinds.insert(probe.kind.as_str());
            total_probe_count += 1;
            if probe.probe_id.trim().is_empty() || !probe_ids.insert(probe.probe_id.as_str()) {
                return Err(format!(
                    "case {} has an empty or duplicate probe_id",
                    case.case_id
                ));
            }
            validate_probe_shape(case, probe)?;
            let probe_margin = probe.score_a - probe.score_b;

            match probe.kind.as_str() {
                "prompt_only" => {
                    if baseline_margin.abs() < panel.policy.minimum_baseline_margin {
                        prompt_only.skip();
                        findings.push(ProbeFinding {
                            case_id: case.case_id.clone(),
                            probe_id: probe.probe_id.clone(),
                            kind: probe.kind.clone(),
                            evaluable: false,
                            passed: None,
                            observed_value: None,
                            threshold: Some(panel.policy.maximum_prompt_only_margin_ratio),
                            candidate_a_absolute_score_drift: None,
                            candidate_b_absolute_score_drift: None,
                            maximum_absolute_score_drift: None,
                            pair_margin_absolute_drift: None,
                            maximum_margin_shift_threshold: None,
                            reason:
                                "baseline pair margin is below the configured evaluability floor"
                                    .to_owned(),
                        });
                    } else {
                        let ratio = probe_margin.abs() / baseline_margin.abs();
                        let passed = ratio <= panel.policy.maximum_prompt_only_margin_ratio;
                        prompt_only.add(passed, Some(ratio), MetricKind::MarginRatio);
                        findings.push(finding(
                            case,
                            probe,
                            passed,
                            Some(ratio),
                            Some(panel.policy.maximum_prompt_only_margin_ratio),
                            "prompt-only margin ratio",
                        ));
                    }
                }
                "audio_swap" => {
                    if baseline_margin.abs() < panel.policy.minimum_baseline_margin {
                        audio_swap.skip();
                        findings.push(ProbeFinding {
                            case_id: case.case_id.clone(),
                            probe_id: probe.probe_id.clone(),
                            kind: probe.kind.clone(),
                            evaluable: false,
                            passed: None,
                            observed_value: None,
                            threshold: None,
                            candidate_a_absolute_score_drift: None,
                            candidate_b_absolute_score_drift: None,
                            maximum_absolute_score_drift: None,
                            pair_margin_absolute_drift: None,
                            maximum_margin_shift_threshold: None,
                            reason:
                                "baseline pair margin is below the configured evaluability floor"
                                    .to_owned(),
                        });
                    } else {
                        let passed = baseline_margin.signum() != probe_margin.signum()
                            && probe_margin.abs() >= panel.policy.minimum_baseline_margin;
                        let shift = (baseline_margin + probe_margin).abs();
                        audio_swap.add(passed, Some(shift), MetricKind::MarginShift);
                        findings.push(finding(
                            case,
                            probe,
                            passed,
                            Some(shift),
                            Some(0.0),
                            "absolute residual after expected margin reversal",
                        ));
                    }
                }
                "instruction_paraphrase" => {
                    if baseline_margin.abs() < panel.policy.minimum_baseline_margin {
                        paraphrase.skip();
                        findings.push(ProbeFinding {
                            case_id: case.case_id.clone(),
                            probe_id: probe.probe_id.clone(),
                            kind: probe.kind.clone(),
                            evaluable: false,
                            passed: None,
                            observed_value: None,
                            threshold: None,
                            reason:
                                "baseline pair margin is below the configured evaluability floor"
                                    .to_owned(),
                        });
                    } else {
                        let passed = baseline_margin.signum() == probe_margin.signum();
                        let shift = (baseline_margin - probe_margin).abs();
                        paraphrase.add(passed, Some(shift), MetricKind::MarginShift);
                        findings.push(finding(
                            case,
                            probe,
                            passed,
                            Some(shift),
                            None,
                            "absolute pair-margin drift; sign must remain stable",
                        ));
                    }
                }
                "nuisance_transform" => {
                    let margin_tolerance = probe
                        .maximum_margin_shift
                        .expect("validated nuisance transform requires margin tolerance");
                    let score_tolerance = probe
                        .maximum_score_drift
                        .expect("validated nuisance transform requires score tolerance");
                    let shift = (baseline_margin - probe_margin).abs();
                    let candidate_a_drift = (case.baseline.score_a - probe.score_a).abs();
                    let candidate_b_drift = (case.baseline.score_b - probe.score_b).abs();
                    let maximum_score_drift = candidate_a_drift.max(candidate_b_drift);
                    let passed =
                        shift <= margin_tolerance && maximum_score_drift <= score_tolerance;
                    nuisance.add(passed, Some(shift), MetricKind::MarginShift);
                    nuisance.add_maximum_absolute_score_drift(maximum_score_drift);
                    let mut probe_finding = finding(
                        case,
                        probe,
                        passed,
                        Some(maximum_score_drift),
                        Some(score_tolerance),
                        "both absolute candidate-score drift and pair-margin drift",
                    );
                    probe_finding.candidate_a_absolute_score_drift = Some(candidate_a_drift);
                    probe_finding.candidate_b_absolute_score_drift = Some(candidate_b_drift);
                    probe_finding.maximum_absolute_score_drift = Some(maximum_score_drift);
                    probe_finding.pair_margin_absolute_drift = Some(shift);
                    probe_finding.maximum_margin_shift_threshold = Some(margin_tolerance);
                    findings.push(probe_finding);
                }
                "semantic_mismatch" => {
                    let expected_candidate = probe
                        .expected_candidate
                        .as_deref()
                        .expect("validated semantic mismatch has candidate");
                    let base_score = if expected_candidate == "a" {
                        case.baseline.score_a
                    } else {
                        case.baseline.score_b
                    };
                    let variant_score = if expected_candidate == "a" {
                        probe.score_a
                    } else {
                        probe.score_b
                    };
                    let drop = base_score - variant_score;
                    let minimum_drop = probe
                        .minimum_score_drop
                        .expect("validated semantic mismatch has minimum score drop");
                    let passed = drop >= minimum_drop;
                    semantic_mismatch.add(passed, Some(drop), MetricKind::ScoreDrop);
                    findings.push(finding(
                        case,
                        probe,
                        passed,
                        Some(drop),
                        Some(minimum_drop),
                        "score drop for the declared mismatched candidate",
                    ));
                }
                other => return Err(format!("unsupported probe kind {other}")),
            }
        }
        for required_kind in [
            "prompt_only",
            "audio_swap",
            "instruction_paraphrase",
            "nuisance_transform",
            "semantic_mismatch",
        ] {
            if !probe_kinds.contains(required_kind) {
                return Err(format!(
                    "case {} is missing required probe kind {required_kind}",
                    case.case_id
                ));
            }
        }
    }

    if total_probe_count == 0 {
        return Err("probe panel must contain at least one probe outcome".to_owned());
    }

    Ok(GroundingReport {
        schema_version: "mel-judge-grounding-report-v1".to_owned(),
        experiment_id: panel.experiment_id.clone(),
        judge_id: panel.judge.judge_id.clone(),
        judge_version: panel.judge.judge_version.clone(),
        case_count: panel.cases.len(),
        total_probe_count,
        human_directional_case_count,
        baseline_correct_count,
        baseline_human_reference_accuracy: rate(
            baseline_correct_count,
            human_directional_case_count,
        ),
        prompt_only: prompt_only.finish(),
        audio_swap: audio_swap.finish(),
        instruction_paraphrase: paraphrase.finish(),
        nuisance_transform: nuisance.finish(),
        semantic_mismatch: semantic_mismatch.finish(),
        findings,
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &ProbePanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION
        || panel.experiment_id.trim().is_empty()
        || !is_calendar_valid_utc(&panel.frozen_at_utc)
    {
        return Err("probe schema, experiment_id or frozen_at_utc is invalid".to_owned());
    }
    if panel.judge.judge_id.trim().is_empty() || panel.judge.judge_version.trim().is_empty() {
        return Err("judge identity must be non-empty".to_owned());
    }
    require_digest(&panel.judge.adapter_sha256, "judge adapter digest")?;
    require_digest(&panel.judge.protocol_sha256, "judge protocol digest")?;
    require_digest(&panel.judge.rubric_sha256, "judge rubric digest")?;

    if !panel.policy.minimum_baseline_margin.is_finite()
        || panel.policy.minimum_baseline_margin < 0.0
        || !panel.policy.maximum_prompt_only_margin_ratio.is_finite()
        || panel.policy.maximum_prompt_only_margin_ratio < 0.0
        || panel.policy.maximum_prompt_only_margin_ratio > 1.0
    {
        return Err("invalid probe policy thresholds".to_owned());
    }
    if panel.cases.is_empty() {
        return Err("probe panel needs at least one case".to_owned());
    }
    validate_claims(&panel.claim_ceiling)
}

fn validate_case(case: &ProbeCaseV1) -> Result<(), String> {
    if case.case_id.trim().is_empty()
        || case.candidate_a_id.trim().is_empty()
        || case.candidate_b_id.trim().is_empty()
        || case.candidate_a_id == case.candidate_b_id
    {
        return Err("case and candidate identities must be non-empty and distinct".to_owned());
    }
    require_digest(&case.candidate_a_audio_sha256, "candidate A audio digest")?;
    require_digest(&case.candidate_b_audio_sha256, "candidate B audio digest")?;
    if !matches!(
        case.human_reference_winner.as_str(),
        "a" | "b" | "tie" | "abstain"
    ) {
        return Err(format!(
            "case {} has unsupported human_reference_winner",
            case.case_id
        ));
    }
    if case.human_reference_directional
        && !matches!(case.human_reference_winner.as_str(), "a" | "b")
    {
        return Err(format!(
            "case {} marks a non-directional human reference as directional",
            case.case_id
        ));
    }
    if case.probes.is_empty() {
        return Err(format!("case {} has no probes", case.case_id));
    }
    validate_scores(case.baseline.score_a, case.baseline.score_b, "baseline")?;
    require_digest(&case.baseline.prompt_sha256, "baseline prompt digest")?;
    require_digest(
        &case.baseline.instruction_sha256,
        "baseline instruction digest",
    )?;
    if case.baseline.audio_a_sha256 != case.candidate_a_audio_sha256
        || case.baseline.audio_b_sha256 != case.candidate_b_audio_sha256
    {
        return Err(format!(
            "case {} baseline audio digests do not bind candidate identities",
            case.case_id
        ));
    }
    Ok(())
}

fn validate_probe_shape(case: &ProbeCaseV1, probe: &ProbeV1) -> Result<(), String> {
    validate_scores(probe.score_a, probe.score_b, &probe.kind)?;
    require_digest(&probe.prompt_sha256, "probe prompt digest")?;
    require_digest(&probe.instruction_sha256, "probe instruction digest")?;

    match probe.kind.as_str() {
        "prompt_only" => {
            if probe.audio_a_sha256.is_some() || probe.audio_b_sha256.is_some() {
                return Err(format!(
                    "prompt-only probe {} must not expose audio digests",
                    probe.probe_id
                ));
            }
            require_same_text_context(case, probe)?;
            if probe.expected_candidate.is_some()
                || probe.maximum_margin_shift.is_some()
                || probe.maximum_score_drift.is_some()
                || probe.minimum_score_drop.is_some()
            {
                return Err(format!(
                    "prompt-only probe {} has irrelevant threshold fields",
                    probe.probe_id
                ));
            }
        }
        "audio_swap" => {
            require_audio_hashes(case, probe)?;
            if probe.audio_a_sha256.as_deref() != Some(case.candidate_b_audio_sha256.as_str())
                || probe.audio_b_sha256.as_deref() != Some(case.candidate_a_audio_sha256.as_str())
            {
                return Err(format!(
                    "audio-swap probe {} must exchange the exact baseline audio digests",
                    probe.probe_id
                ));
            }
            require_same_text_context(case, probe)?;
            if probe.expected_candidate.is_some()
                || probe.maximum_margin_shift.is_some()
                || probe.maximum_score_drift.is_some()
                || probe.minimum_score_drop.is_some()
            {
                return Err(format!(
                    "audio-swap probe {} has irrelevant threshold fields",
                    probe.probe_id
                ));
            }
        }
        "instruction_paraphrase" => {
            require_audio_hashes(case, probe)?;
            if probe.audio_a_sha256.as_deref() != Some(case.candidate_a_audio_sha256.as_str())
                || probe.audio_b_sha256.as_deref() != Some(case.candidate_b_audio_sha256.as_str())
                || probe.prompt_sha256 != case.baseline.prompt_sha256
                || probe.instruction_sha256 == case.baseline.instruction_sha256
            {
                return Err(format!(
                    "instruction-paraphrase probe {} must keep audio/prompt and change instruction digest",
                    probe.probe_id
                ));
            }
            if probe.expected_candidate.is_some()
                || probe.maximum_margin_shift.is_some()
                || probe.maximum_score_drift.is_some()
                || probe.minimum_score_drop.is_some()
            {
                return Err(format!(
                    "instruction-paraphrase probe {} has irrelevant threshold fields",
                    probe.probe_id
                ));
            }
        }
        "nuisance_transform" => {
            require_audio_hashes(case, probe)?;
            if probe.prompt_sha256 != case.baseline.prompt_sha256
                || probe.instruction_sha256 != case.baseline.instruction_sha256
                || (probe.audio_a_sha256.as_deref() == Some(case.candidate_a_audio_sha256.as_str())
                    && probe.audio_b_sha256.as_deref()
                        == Some(case.candidate_b_audio_sha256.as_str()))
            {
                return Err(format!(
                    "nuisance probe {} must keep text fixed and identify transformed audio bytes",
                    probe.probe_id
                ));
            }
            let tolerance = probe.maximum_margin_shift.ok_or_else(|| {
                format!(
                    "nuisance probe {} needs maximum_margin_shift",
                    probe.probe_id
                )
            })?;
            let score_tolerance = probe.maximum_score_drift.ok_or_else(|| {
                format!(
                    "nuisance probe {} needs maximum_score_drift",
                    probe.probe_id
                )
            })?;
            if !tolerance.is_finite()
                || tolerance < 0.0
                || !score_tolerance.is_finite()
                || score_tolerance < 0.0
                || probe.minimum_score_drop.is_some()
                || probe.expected_candidate.is_some()
            {
                return Err(format!(
                    "nuisance probe {} has invalid tolerance/fields",
                    probe.probe_id
                ));
            }
        }
        "semantic_mismatch" => {
            require_audio_hashes(case, probe)?;
            if probe.prompt_sha256 != case.baseline.prompt_sha256
                || probe.instruction_sha256 != case.baseline.instruction_sha256
            {
                return Err(format!(
                    "semantic-mismatch probe {} must preserve prompt and instruction",
                    probe.probe_id
                ));
            }
            let expected = probe.expected_candidate.as_deref().ok_or_else(|| {
                format!(
                    "semantic-mismatch probe {} needs expected_candidate",
                    probe.probe_id
                )
            })?;
            if expected != "a" && expected != "b" {
                return Err(format!(
                    "semantic-mismatch probe {} expected_candidate must be a or b",
                    probe.probe_id
                ));
            }
            let audio_a_changed =
                probe.audio_a_sha256.as_deref() != Some(case.candidate_a_audio_sha256.as_str());
            let audio_b_changed =
                probe.audio_b_sha256.as_deref() != Some(case.candidate_b_audio_sha256.as_str());
            if (expected == "a" && (!audio_a_changed || audio_b_changed))
                || (expected == "b" && (!audio_b_changed || audio_a_changed))
            {
                return Err(format!(
                    "semantic-mismatch probe {} must change only its expected candidate audio",
                    probe.probe_id
                ));
            }
            let minimum = probe.minimum_score_drop.ok_or_else(|| {
                format!(
                    "semantic-mismatch probe {} needs minimum_score_drop",
                    probe.probe_id
                )
            })?;
            if !minimum.is_finite()
                || minimum < 0.0
                || probe.maximum_margin_shift.is_some()
                || probe.maximum_score_drift.is_some()
            {
                return Err(format!(
                    "semantic-mismatch probe {} has invalid threshold/fields",
                    probe.probe_id
                ));
            }
        }
        other => {
            return Err(format!(
                "unsupported probe kind {other} in case {}",
                case.case_id
            ));
        }
    }
    Ok(())
}

fn require_same_text_context(case: &ProbeCaseV1, probe: &ProbeV1) -> Result<(), String> {
    if probe.prompt_sha256 != case.baseline.prompt_sha256
        || probe.instruction_sha256 != case.baseline.instruction_sha256
    {
        return Err(format!(
            "probe {} must preserve baseline prompt and instruction digests",
            probe.probe_id
        ));
    }
    Ok(())
}

fn require_audio_hashes(case: &ProbeCaseV1, probe: &ProbeV1) -> Result<(), String> {
    let audio_a = probe
        .audio_a_sha256
        .as_deref()
        .ok_or_else(|| format!("probe {} is missing audio A digest", probe.probe_id))?;
    let audio_b = probe
        .audio_b_sha256
        .as_deref()
        .ok_or_else(|| format!("probe {} is missing audio B digest", probe.probe_id))?;
    require_digest(audio_a, "probe audio A digest")?;
    require_digest(audio_b, "probe audio B digest")?;
    if audio_a == audio_b {
        return Err(format!(
            "probe {} has identical A/B audio digests",
            probe.probe_id
        ));
    }
    if case.candidate_a_audio_sha256 == case.candidate_b_audio_sha256 {
        return Err(format!(
            "case {} has identical candidate audio digests",
            case.case_id
        ));
    }
    Ok(())
}

fn validate_scores(score_a: f64, score_b: f64, name: &str) -> Result<(), String> {
    if !score_a.is_finite()
        || !score_b.is_finite()
        || !(0.0..=10.0).contains(&score_a)
        || !(0.0..=10.0).contains(&score_b)
    {
        return Err(format!("{name} scores must be finite and in [0, 10]"));
    }
    Ok(())
}

fn finding(
    case: &ProbeCaseV1,
    probe: &ProbeV1,
    passed: bool,
    observed: Option<f64>,
    threshold: Option<f64>,
    reason: &str,
) -> ProbeFinding {
    ProbeFinding {
        case_id: case.case_id.clone(),
        probe_id: probe.probe_id.clone(),
        kind: probe.kind.clone(),
        evaluable: true,
        passed: Some(passed),
        observed_value: observed,
        threshold,
        candidate_a_absolute_score_drift: None,
        candidate_b_absolute_score_drift: None,
        maximum_absolute_score_drift: None,
        pair_margin_absolute_drift: None,
        maximum_margin_shift_threshold: None,
        reason: reason.to_owned(),
    }
}

fn validate_claims(claims: &[String]) -> Result<(), String> {
    if claims.is_empty() || claims.iter().any(|claim| claim.trim().is_empty()) {
        Err("claim_ceiling must contain non-empty limits".to_owned())
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

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn mean(sum: f64, count: usize) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/valid-probe-panel-v1.json");

    fn report() -> GroundingReport {
        parse_and_audit(FIXTURE).expect("known-answer panel should audit")
    }

    fn value() -> serde_json::Value {
        serde_json::from_str(FIXTURE).expect("fixture JSON")
    }

    #[test]
    fn valid_probe_panel_reports_each_axis_separately() {
        let report = report();
        assert_eq!(report.case_count, 1);
        assert_eq!(report.total_probe_count, 5);
        assert_eq!(report.human_directional_case_count, 1);
        assert_eq!(report.baseline_human_reference_accuracy, Some(1.0));
        assert_eq!(report.prompt_only.pass_rate, Some(1.0));
        assert_eq!(report.audio_swap.pass_rate, Some(1.0));
        assert_eq!(report.instruction_paraphrase.pass_rate, Some(1.0));
        assert_eq!(report.nuisance_transform.pass_rate, Some(1.0));
        assert_eq!(report.semantic_mismatch.pass_rate, Some(1.0));
    }

    #[test]
    fn prompt_only_judge_leakage_is_exposed() {
        let mut panel = value();
        panel["cases"][0]["probes"][0]["score_a"] = serde_json::json!(8.0);
        panel["cases"][0]["probes"][0]["score_b"] = serde_json::json!(3.0);
        let report = parse_and_audit(&panel.to_string()).expect("prompt-only probe is valid");
        assert_eq!(report.prompt_only.pass_rate, Some(0.0));
        assert!(report.prompt_only.mean_margin_ratio.unwrap() > 0.99);
    }

    #[test]
    fn prompt_only_policy_ratio_must_be_between_zero_and_one() {
        let mut panel = value();
        panel["policy"]["maximum_prompt_only_margin_ratio"] = serde_json::json!(1.1);
        let error = parse_and_audit(&panel.to_string())
            .expect_err("prompt-only ratio above one should be rejected");
        assert!(error.contains("invalid probe policy thresholds"));
    }

    #[test]
    fn audio_swap_without_reversal_is_rejected_as_a_probe_failure() {
        let mut panel = value();
        panel["cases"][0]["probes"][1]["score_a"] = serde_json::json!(7.0);
        panel["cases"][0]["probes"][1]["score_b"] = serde_json::json!(3.0);
        let report = parse_and_audit(&panel.to_string()).expect("probe structure is valid");
        assert_eq!(report.audio_swap.pass_rate, Some(0.0));
    }

    #[test]
    fn instruction_paraphrase_rank_flip_is_reported() {
        let mut panel = value();
        panel["cases"][0]["probes"][2]["score_a"] = serde_json::json!(2.0);
        panel["cases"][0]["probes"][2]["score_b"] = serde_json::json!(7.0);
        let report = parse_and_audit(&panel.to_string()).expect("probe structure is valid");
        assert_eq!(report.instruction_paraphrase.pass_rate, Some(0.0));
    }

    #[test]
    fn nuisance_transform_margin_drift_is_reported() {
        let mut panel = value();
        panel["cases"][0]["probes"][3]["score_a"] = serde_json::json!(4.0);
        panel["cases"][0]["probes"][3]["score_b"] = serde_json::json!(3.0);
        let report = parse_and_audit(&panel.to_string()).expect("probe structure is valid");
        assert_eq!(report.nuisance_transform.pass_rate, Some(0.0));
    }

    #[test]
    fn nuisance_transform_rejects_common_mode_score_shift_even_when_margin_is_unchanged() {
        let mut panel = value();
        panel["cases"][0]["probes"][3]["score_a"] = serde_json::json!(8.8);
        panel["cases"][0]["probes"][3]["score_b"] = serde_json::json!(3.8);
        let report = parse_and_audit(&panel.to_string()).expect("probe structure is valid");
        assert_eq!(report.nuisance_transform.pass_rate, Some(0.0));
        assert!(
            (report.nuisance_transform.mean_maximum_absolute_score_drift.unwrap() - 0.8).abs()
                < 1e-9
        );
        let nuisance_finding = report
            .findings
            .iter()
            .find(|finding| finding.kind == "nuisance_transform")
            .expect("nuisance finding is present");
        assert!((nuisance_finding.candidate_a_absolute_score_drift.unwrap() - 0.8).abs() < 1e-9);
        assert!((nuisance_finding.candidate_b_absolute_score_drift.unwrap() - 0.8).abs() < 1e-9);
        assert!((nuisance_finding.maximum_absolute_score_drift.unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn nuisance_transform_reports_both_candidate_drifts_separately() {
        let report = report();
        let nuisance_finding = report
            .findings
            .iter()
            .find(|finding| finding.kind == "nuisance_transform")
            .expect("nuisance finding is present");
        assert!((nuisance_finding.candidate_a_absolute_score_drift.unwrap() - 0.05).abs() < 1e-9);
        assert!((nuisance_finding.candidate_b_absolute_score_drift.unwrap() - 0.05).abs() < 1e-9);
        assert!(
            (report.nuisance_transform.mean_maximum_absolute_score_drift.unwrap() - 0.05).abs()
                < 1e-9
        );
    }

    #[test]
    fn nuisance_finding_preserves_both_margin_and_absolute_score_drift() {
        let report = report();
        let nuisance_finding = report
            .findings
            .iter()
            .find(|finding| finding.kind == "nuisance_transform")
            .expect("nuisance finding is present");
        assert!(nuisance_finding.pair_margin_absolute_drift.unwrap().abs() < 1e-9);
        assert_eq!(nuisance_finding.maximum_margin_shift_threshold, Some(0.2));
    }

    #[test]
    fn semantic_mismatch_without_a_score_drop_is_reported() {
        let mut panel = value();
        panel["cases"][0]["probes"][4]["score_a"] = serde_json::json!(7.0);
        let report = parse_and_audit(&panel.to_string()).expect("probe structure is valid");
        assert_eq!(report.semantic_mismatch.pass_rate, Some(0.0));
    }

    #[test]
    fn missing_probe_kind_is_rejected_not_counted_as_a_pass() {
        let mut panel = value();
        panel["cases"][0]["probes"].as_array_mut().unwrap().pop();
        let error = parse_and_audit(&panel.to_string())
            .expect_err("incomplete probe panel must be rejected");
        assert!(error.contains("missing required probe kind semantic_mismatch"));
    }

    #[test]
    fn duplicate_probe_ids_are_rejected() {
        let mut panel = value();
        panel["cases"][0]["probes"][1]["probe_id"] =
            panel["cases"][0]["probes"][0]["probe_id"].clone();
        let error = parse_and_audit(&panel.to_string()).expect_err("duplicate probe should fail");
        assert!(error.contains("duplicate probe_id"));
    }

    #[test]
    fn invalid_digest_is_rejected() {
        let mut panel = value();
        panel["cases"][0]["candidate_a_audio_sha256"] = serde_json::json!("not-a-digest");
        let error = parse_and_audit(&panel.to_string()).expect_err("bad digest should fail");
        assert!(error.contains("64-character lowercase SHA-256 digest"));
    }
}
