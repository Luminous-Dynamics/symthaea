//! Task-aware scoring for lyric and audio edits.
//!
//! The evaluator consumes independently produced metric intervals and
//! content-addressed receipts. It does not analyze audio, transcribe lyrics,
//! estimate F0, calculate speaker embeddings, or judge naturalness.
//!
//! Instruction realization and preservation are reported separately. A case
//! is an exact match only when every required rubric passes. Indeterminate,
//! unsupported, failed, and abstained cases never receive exact-match credit.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const EDIT_ASSURANCE_SCHEMA_V1: &str = "lyric-edit-assurance/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditAssuranceError {
    pub message: String,
}

impl EditAssuranceError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl Display for EditAssuranceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for EditAssuranceError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditOperationV1 {
    PartialSubstitution,
    FullSubstitution,
    Deletion,
    Insertion,
    Translation,
    CodeMixing,
    GeneralAudioEdit,
}

impl EditOperationV1 {
    const fn requires_lyric_payload(self) -> bool {
        !matches!(self, Self::GeneralAudioEdit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TimbreModeV1 {
    SameTimbre,
    CrossTimbre,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RubricRoleV1 {
    Instruction,
    Preservation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionStatusV1 {
    Produced,
    Unsupported,
    Failed,
    Abstained,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RubricDecisionV1 {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseOutcomeV1 {
    ExactMatch,
    Failed,
    Indeterminate,
    Unsupported,
    ExecutionFailed,
    Abstained,
}

/// The complete group matrix must be declared before evaluation.
/// A missing or unexpected group rejects the run rather than silently changing
/// macro weights.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EditStratumKeyV1 {
    pub operation: EditOperationV1,
    pub language_id: String,
    pub timbre_mode: TimbreModeV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditBenchmarkSpecV1 {
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub evaluation_split_id: String,
    pub evaluation_manifest_sha256: String,
    pub calibration_split_id: String,
    pub calibration_manifest_sha256: String,
    pub calibration_receipt_sha256: String,
    pub system_id: String,
    pub system_version: String,
    pub system_artifact_sha256: String,
    pub expected_groups: Vec<EditStratumKeyV1>,
}

/// One required instruction or preservation rubric.
///
/// Accepted bounds are metric-specific and must be frozen using an independent
/// calibration/development split. For an error metric such as PER, accepted
/// bounds may be [0, max_error]. For a similarity metric, they may be
/// [min_similarity, max_similarity]. Units are never coerced.
#[derive(Clone, Debug, PartialEq)]
pub struct RubricRequirementV1 {
    pub rubric_id: String,
    pub metric_id: String,
    pub role: RubricRoleV1,
    pub unit: String,
    pub accepted_min: f64,
    pub accepted_max: f64,
}

/// One observation from an independently versioned analyzer or human panel.
#[derive(Clone, Debug, PartialEq)]
pub struct RubricObservationV1 {
    pub rubric_id: String,
    pub metric_id: String,
    pub unit: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub evaluator_id: String,
    pub evaluator_version: String,
    pub evaluator_artifact_sha256: String,
    pub calibration_receipt_sha256: String,
}

/// Frozen task identity and the exact rubric universe for one case.
#[derive(Clone, Debug, PartialEq)]
pub struct EditCaseSpecV1 {
    pub sample_id: String,
    pub operation: EditOperationV1,
    pub language_id: String,
    pub timbre_mode: TimbreModeV1,
    pub singer_id: String,
    pub song_id: String,
    pub recording_session_id: String,
    pub technique_id: String,
    pub source_audio_sha256: String,
    pub melody_reference_sha256: String,
    pub timbre_reference_sha256: Option<String>,
    pub original_lyrics_sha256: Option<String>,
    pub requested_lyrics_sha256: Option<String>,
    pub task_manifest_sha256: String,
    pub rubrics: Vec<RubricRequirementV1>,
}

/// Every manifest case must have exactly one execution record. Unsupported,
/// failed and abstained executions are retained in IFR/CR/EMR denominators.
#[derive(Clone, Debug, PartialEq)]
pub struct EditCaseExecutionV1 {
    pub sample_id: String,
    pub status: ExecutionStatusV1,
    pub output_audio_sha256: Option<String>,
    pub failure_receipt_sha256: Option<String>,
    pub observations: Vec<RubricObservationV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RubricResultV1 {
    pub rubric_id: String,
    pub metric_id: String,
    pub role: RubricRoleV1,
    pub unit: String,
    pub accepted_min: f64,
    pub accepted_max: f64,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    /// None means no metric observation exists because the system did not
    /// produce an output. Such a rubric remains in the denominator.
    pub decision: Option<RubricDecisionV1>,
    pub evaluator_id: Option<String>,
    pub evaluator_version: Option<String>,
    pub evaluator_artifact_sha256: Option<String>,
    pub calibration_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RubricRateV1 {
    pub passed: usize,
    pub total: usize,
    pub indeterminate: usize,
    pub unscored: usize,
    pub rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaseResultV1 {
    pub sample_id: String,
    pub operation: EditOperationV1,
    pub language_id: String,
    pub timbre_mode: TimbreModeV1,
    pub execution_status: ExecutionStatusV1,
    pub output_audio_sha256: Option<String>,
    pub outcome: CaseOutcomeV1,
    pub exact_match: bool,
    pub instruction_rubrics: RubricRateV1,
    pub preservation_rubrics: RubricRateV1,
    pub rubrics: Vec<RubricResultV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditGroupResultV1 {
    pub group: EditStratumKeyV1,
    pub case_count: usize,
    pub exact_match_cases: usize,
    pub exact_match_rate: f64,
    pub instruction_following: RubricRateV1,
    pub consistency: RubricRateV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditBenchmarkReportV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub system_id: String,
    pub system_version: String,
    pub evaluation_split_id: String,
    pub case_count: usize,
    pub exact_match_cases: usize,
    pub exact_match_rate: f64,
    pub instruction_following: RubricRateV1,
    pub consistency: RubricRateV1,
    pub indeterminate_cases: usize,
    pub unsupported_cases: usize,
    pub failed_cases: usize,
    pub abstained_cases: usize,
    /// Unweighted mean of the predeclared group-level rates. This does not
    /// replace the micro rates or per-group results.
    pub macro_group_exact_match_rate: f64,
    pub macro_group_instruction_following_rate: f64,
    pub macro_group_consistency_rate: f64,
    pub groups: Vec<EditGroupResultV1>,
    pub cases: Vec<CaseResultV1>,
}

#[derive(Clone, Debug, Default)]
struct MutableRubricCounts {
    passed: usize,
    total: usize,
    indeterminate: usize,
    unscored: usize,
}

impl MutableRubricCounts {
    fn record(&mut self, decision: Option<RubricDecisionV1>) {
        self.total += 1;
        match decision {
            Some(RubricDecisionV1::Pass) => self.passed += 1,
            Some(RubricDecisionV1::Indeterminate) => self.indeterminate += 1,
            Some(RubricDecisionV1::Fail) => {}
            None => self.unscored += 1,
        }
    }

    fn finish(&self) -> RubricRateV1 {
        RubricRateV1 {
            passed: self.passed,
            total: self.total,
            indeterminate: self.indeterminate,
            unscored: self.unscored,
            rate: self.passed as f64 / self.total.max(1) as f64,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct MutableGroupCounts {
    case_count: usize,
    exact_match_cases: usize,
    instruction: MutableRubricCounts,
    preservation: MutableRubricCounts,
}

impl MutableGroupCounts {
    fn add_case(&mut self, case: &CaseResultV1) {
        self.case_count += 1;
        if case.exact_match {
            self.exact_match_cases += 1;
        }
        for rubric in &case.rubrics {
            match rubric.role {
                RubricRoleV1::Instruction => self.instruction.record(rubric.decision),
                RubricRoleV1::Preservation => self.preservation.record(rubric.decision),
            }
        }
    }
}

/// Evaluate a complete run against its frozen manifest and rubric profile.
///
/// Missing or duplicate case executions, missing/extra/duplicate rubric
/// observations, malformed provenance, or incomplete predeclared group
/// coverage reject the run.
///
/// IFR, CR and EMR are computed from this local rubric contract. This is not a
/// reproduction of the official MMAE/Qwen3-Omni evaluator unless its exact
/// rubric mapping and official judge outputs are separately supplied.
pub fn evaluate_edit_benchmark(
    spec: &EditBenchmarkSpecV1,
    cases: &[EditCaseSpecV1],
    executions: &[EditCaseExecutionV1],
) -> Result<EditBenchmarkReportV1, EditAssuranceError> {
    validate_benchmark_spec(spec)?;

    if cases.is_empty() {
        return Err(EditAssuranceError::new(
            "at least one case from the frozen evaluation manifest is required",
        ));
    }
    if executions.len() != cases.len() {
        return Err(EditAssuranceError::new(format!(
            "expected exactly {} execution records, received {}",
            cases.len(),
            executions.len()
        )));
    }

    let expected_groups: BTreeSet<EditStratumKeyV1> =
        spec.expected_groups.iter().cloned().collect();
    let mut seen_samples = BTreeSet::new();
    let mut actual_groups = BTreeSet::new();
    let mut execution_by_sample = BTreeMap::new();

    for execution in executions {
        require_non_empty(&execution.sample_id, "execution sample_id")?;
        if execution_by_sample
            .insert(execution.sample_id.as_str(), execution)
            .is_some()
        {
            return Err(EditAssuranceError::new(format!(
                "duplicate execution record for sample {}",
                execution.sample_id
            )));
        }
    }

    let mut case_results = Vec::with_capacity(cases.len());
    let mut group_counts: BTreeMap<EditStratumKeyV1, MutableGroupCounts> = BTreeMap::new();

    for case in cases {
        validate_case_spec(case)?;
        if !seen_samples.insert(case.sample_id.as_str()) {
            return Err(EditAssuranceError::new(format!(
                "duplicate case sample_id: {}",
                case.sample_id
            )));
        }

        let group = EditStratumKeyV1 {
            operation: case.operation,
            language_id: case.language_id.clone(),
            timbre_mode: case.timbre_mode,
        };
        if !expected_groups.contains(&group) {
            return Err(EditAssuranceError::new(format!(
                "case {} belongs to an unexpected operation/language/timbre group",
                case.sample_id
            )));
        }
        actual_groups.insert(group.clone());

        let execution = execution_by_sample
            .get(case.sample_id.as_str())
            .copied()
            .ok_or_else(|| {
                EditAssuranceError::new(format!(
                    "missing execution record for sample {}",
                    case.sample_id
                ))
            })?;
        let case_result = evaluate_case(case, execution)?;
        group_counts
            .entry(group)
            .or_default()
            .add_case(&case_result);
        case_results.push(case_result);
    }

    for execution_id in execution_by_sample.keys() {
        if !seen_samples.contains(*execution_id) {
            return Err(EditAssuranceError::new(format!(
                "execution record has no matching case manifest entry: {execution_id}"
            )));
        }
    }

    if actual_groups != expected_groups {
        let missing: Vec<String> = expected_groups
            .difference(&actual_groups)
            .map(|group| {
                format!(
                    "{:?}/{}/{}",
                    group.operation, group.language_id, group.timbre_mode as u8
                )
            })
            .collect();
        return Err(EditAssuranceError::new(format!(
            "missing predeclared operation/language/timbre groups: {}",
            missing.join(", ")
        )));
    }

    let mut total_instruction = MutableRubricCounts::default();
    let mut total_preservation = MutableRubricCounts::default();
    let mut exact_match_cases = 0usize;
    let mut indeterminate_cases = 0usize;
    let mut unsupported_cases = 0usize;
    let mut failed_cases = 0usize;
    let mut abstained_cases = 0usize;

    for case in &case_results {
        if case.exact_match {
            exact_match_cases += 1;
        }
        match case.outcome {
            CaseOutcomeV1::Indeterminate => indeterminate_cases += 1,
            CaseOutcomeV1::Unsupported => unsupported_cases += 1,
            CaseOutcomeV1::Failed | CaseOutcomeV1::ExecutionFailed => failed_cases += 1,
            CaseOutcomeV1::Abstained => abstained_cases += 1,
            CaseOutcomeV1::ExactMatch => {}
        }
        for rubric in &case.rubrics {
            match rubric.role {
                RubricRoleV1::Instruction => total_instruction.record(rubric.decision),
                RubricRoleV1::Preservation => total_preservation.record(rubric.decision),
            }
        }
    }

    let groups: Vec<EditGroupResultV1> = group_counts
        .into_iter()
        .map(|(group, counts)| EditGroupResultV1 {
            group,
            case_count: counts.case_count,
            exact_match_cases: counts.exact_match_cases,
            exact_match_rate: counts.exact_match_cases as f64
                / counts.case_count.max(1) as f64,
            instruction_following: counts.instruction.finish(),
            consistency: counts.preservation.finish(),
        })
        .collect();

    let macro_group_exact_match_rate =
        mean(groups.iter().map(|group| group.exact_match_rate));
    let macro_group_instruction_following_rate = mean(
        groups.iter().map(|group| group.instruction_following.rate),
    );
    let macro_group_consistency_rate =
        mean(groups.iter().map(|group| group.consistency.rate));

    Ok(EditBenchmarkReportV1 {
        schema: EDIT_ASSURANCE_SCHEMA_V1.to_owned(),
        benchmark_id: spec.benchmark_id.clone(),
        benchmark_version: spec.benchmark_version.clone(),
        system_id: spec.system_id.clone(),
        system_version: spec.system_version.clone(),
        evaluation_split_id: spec.evaluation_split_id.clone(),
        case_count: case_results.len(),
        exact_match_cases,
        exact_match_rate: exact_match_cases as f64 / case_results.len() as f64,
        instruction_following: total_instruction.finish(),
        consistency: total_preservation.finish(),
        indeterminate_cases,
        unsupported_cases,
        failed_cases,
        abstained_cases,
        macro_group_exact_match_rate,
        macro_group_instruction_following_rate,
        macro_group_consistency_rate,
        groups,
        cases: case_results,
    })
}

fn validate_benchmark_spec(spec: &EditBenchmarkSpecV1) -> Result<(), EditAssuranceError> {
    require_non_empty(&spec.benchmark_id, "benchmark_id")?;
    require_non_empty(&spec.benchmark_version, "benchmark_version")?;
    require_non_empty(&spec.evaluation_split_id, "evaluation_split_id")?;
    require_non_empty(&spec.calibration_split_id, "calibration_split_id")?;
    require_non_empty(&spec.system_id, "system_id")?;
    require_non_empty(&spec.system_version, "system_version")?;
    validate_digest(&spec.evaluation_manifest_sha256, "evaluation_manifest_sha256")?;
    validate_digest(&spec.calibration_manifest_sha256, "calibration_manifest_sha256")?;
    validate_digest(&spec.calibration_receipt_sha256, "calibration_receipt_sha256")?;
    validate_digest(&spec.system_artifact_sha256, "system_artifact_sha256")?;

    if spec.evaluation_split_id == spec.calibration_split_id {
        return Err(EditAssuranceError::new(
            "evaluation and calibration split IDs must differ",
        ));
    }
    if spec.evaluation_manifest_sha256 == spec.calibration_manifest_sha256 {
        return Err(EditAssuranceError::new(
            "evaluation and calibration manifest digests must differ",
        ));
    }
    if spec.expected_groups.is_empty() {
        return Err(EditAssuranceError::new(
            "expected_groups must declare the complete task/language/timbre matrix",
        ));
    }

    let mut seen_groups = BTreeSet::new();
    for group in &spec.expected_groups {
        require_non_empty(&group.language_id, "expected group language_id")?;
        if !seen_groups.insert(group.clone()) {
            return Err(EditAssuranceError::new(
                "duplicate expected operation/language/timbre group",
            ));
        }
    }
    Ok(())
}

fn validate_case_spec(case: &EditCaseSpecV1) -> Result<(), EditAssuranceError> {
    require_non_empty(&case.sample_id, "sample_id")?;
    require_non_empty(&case.language_id, "language_id")?;
    require_non_empty(&case.singer_id, "singer_id")?;
    require_non_empty(&case.song_id, "song_id")?;
    require_non_empty(&case.recording_session_id, "recording_session_id")?;
    require_non_empty(&case.technique_id, "technique_id")?;
    validate_digest(&case.source_audio_sha256, "source_audio_sha256")?;
    validate_digest(&case.melody_reference_sha256, "melody_reference_sha256")?;
    validate_digest(&case.task_manifest_sha256, "task_manifest_sha256")?;

    match case.timbre_mode {
        TimbreModeV1::NotApplicable => {
            if case.timbre_reference_sha256.is_some() {
                return Err(EditAssuranceError::new(format!(
                    "case {} declares a timbre reference but timbre mode is not applicable",
                    case.sample_id
                )));
            }
        }
        TimbreModeV1::SameTimbre | TimbreModeV1::CrossTimbre => {
            let digest = case.timbre_reference_sha256.as_deref().ok_or_else(|| {
                EditAssuranceError::new(format!(
                    "case {} requires an explicit timbre reference",
                    case.sample_id
                ))
            })?;
            validate_digest(digest, "timbre_reference_sha256")?;
        }
    }

    if case.operation.requires_lyric_payload() {
        let original = case.original_lyrics_sha256.as_deref().ok_or_else(|| {
            EditAssuranceError::new(format!(
                "lyric operation in case {} is missing original_lyrics_sha256",
                case.sample_id
            ))
        })?;
        let requested = case.requested_lyrics_sha256.as_deref().ok_or_else(|| {
            EditAssuranceError::new(format!(
                "lyric operation in case {} is missing requested_lyrics_sha256",
                case.sample_id
            ))
        })?;
        validate_digest(original, "original_lyrics_sha256")?;
        validate_digest(requested, "requested_lyrics_sha256")?;
    } else if case.original_lyrics_sha256.is_some() || case.requested_lyrics_sha256.is_some() {
        return Err(EditAssuranceError::new(format!(
            "general audio edit case {} must not declare lyric payload digests in v1",
            case.sample_id
        )));
    }

    if case.rubrics.is_empty() {
        return Err(EditAssuranceError::new(format!(
            "case {} has no required rubrics",
            case.sample_id
        )));
    }

    let mut rubric_ids = BTreeSet::new();
    let mut has_instruction = false;
    let mut has_preservation = false;
    for rubric in &case.rubrics {
        require_non_empty(&rubric.rubric_id, "rubric_id")?;
        require_non_empty(&rubric.metric_id, "rubric metric_id")?;
        require_non_empty(&rubric.unit, "rubric unit")?;
        if !rubric_ids.insert(rubric.rubric_id.as_str()) {
            return Err(EditAssuranceError::new(format!(
                "duplicate required rubric ID {} in case {}",
                rubric.rubric_id, case.sample_id
            )));
        }
        if !rubric.accepted_min.is_finite()
            || !rubric.accepted_max.is_finite()
            || rubric.accepted_min >= rubric.accepted_max
        {
            return Err(EditAssuranceError::new(format!(
                "rubric {} in case {} has invalid accepted bounds",
                rubric.rubric_id, case.sample_id
            )));
        }
        match rubric.role {
            RubricRoleV1::Instruction => has_instruction = true,
            RubricRoleV1::Preservation => has_preservation = true,
        }
    }

    if !has_instruction || !has_preservation {
        return Err(EditAssuranceError::new(format!(
            "case {} must declare at least one instruction and one preservation rubric",
            case.sample_id
        )));
    }
    Ok(())
}

fn evaluate_case(
    case: &EditCaseSpecV1,
    execution: &EditCaseExecutionV1,
) -> Result<CaseResultV1, EditAssuranceError> {
    if execution.sample_id != case.sample_id {
        return Err(EditAssuranceError::new(format!(
            "execution/sample identity mismatch for case {}",
            case.sample_id
        )));
    }

    let mut rubric_results = Vec::with_capacity(case.rubrics.len());
    let outcome;

    match execution.status {
        ExecutionStatusV1::Produced => {
            let output_digest = execution.output_audio_sha256.as_deref().ok_or_else(|| {
                EditAssuranceError::new(format!(
                    "produced case {} is missing output_audio_sha256",
                    case.sample_id
                ))
            })?;
            validate_digest(output_digest, "output_audio_sha256")?;
            if execution.failure_receipt_sha256.is_some() {
                return Err(EditAssuranceError::new(format!(
                    "produced case {} must not contain a failure receipt",
                    case.sample_id
                )));
            }

            let observations = index_observations(case, &execution.observations)?;
            for requirement in &case.rubrics {
                let observation = observations
                    .get(requirement.rubric_id.as_str())
                    .copied()
                    .ok_or_else(|| {
                        EditAssuranceError::new(format!(
                            "missing observation for rubric {} in case {}",
                            requirement.rubric_id, case.sample_id
                        ))
                    })?;
                rubric_results.push(evaluate_rubric(requirement, observation));
            }

            let any_fail = rubric_results
                .iter()
                .any(|result| result.decision == Some(RubricDecisionV1::Fail));
            let any_indeterminate = rubric_results
                .iter()
                .any(|result| result.decision == Some(RubricDecisionV1::Indeterminate));
            outcome = if any_fail {
                CaseOutcomeV1::Failed
            } else if any_indeterminate {
                CaseOutcomeV1::Indeterminate
            } else {
                CaseOutcomeV1::ExactMatch
            };
        }
        ExecutionStatusV1::Unsupported
        | ExecutionStatusV1::Failed
        | ExecutionStatusV1::Abstained => {
            if execution.output_audio_sha256.is_some() {
                return Err(EditAssuranceError::new(format!(
                    "non-produced case {} must not declare output_audio_sha256",
                    case.sample_id
                )));
            }
            let receipt = execution.failure_receipt_sha256.as_deref().ok_or_else(|| {
                EditAssuranceError::new(format!(
                    "non-produced case {} is missing its failure/unsupported/abstention receipt",
                    case.sample_id
                ))
            })?;
            validate_digest(receipt, "failure_receipt_sha256")?;
            if !execution.observations.is_empty() {
                return Err(EditAssuranceError::new(format!(
                    "non-produced case {} must not contain metric observations",
                    case.sample_id
                )));
            }

            for requirement in &case.rubrics {
                rubric_results.push(RubricResultV1 {
                    rubric_id: requirement.rubric_id.clone(),
                    metric_id: requirement.metric_id.clone(),
                    role: requirement.role,
                    unit: requirement.unit.clone(),
                    accepted_min: requirement.accepted_min,
                    accepted_max: requirement.accepted_max,
                    lower_bound: None,
                    upper_bound: None,
                    decision: None,
                    evaluator_id: None,
                    evaluator_version: None,
                    evaluator_artifact_sha256: None,
                    calibration_receipt_sha256: None,
                });
            }
            outcome = match execution.status {
                ExecutionStatusV1::Unsupported => CaseOutcomeV1::Unsupported,
                ExecutionStatusV1::Failed => CaseOutcomeV1::ExecutionFailed,
                ExecutionStatusV1::Abstained => CaseOutcomeV1::Abstained,
                ExecutionStatusV1::Produced => unreachable!(),
            };
        }
    }

    let mut instruction = MutableRubricCounts::default();
    let mut preservation = MutableRubricCounts::default();
    for rubric in &rubric_results {
        match rubric.role {
            RubricRoleV1::Instruction => instruction.record(rubric.decision),
            RubricRoleV1::Preservation => preservation.record(rubric.decision),
        }
    }

    Ok(CaseResultV1 {
        sample_id: case.sample_id.clone(),
        operation: case.operation,
        language_id: case.language_id.clone(),
        timbre_mode: case.timbre_mode,
        execution_status: execution.status,
        output_audio_sha256: execution.output_audio_sha256.clone(),
        exact_match: outcome == CaseOutcomeV1::ExactMatch,
        instruction_rubrics: instruction.finish(),
        preservation_rubrics: preservation.finish(),
        outcome,
        rubrics: rubric_results,
    })
}

fn index_observations<'a>(
    case: &EditCaseSpecV1,
    observations: &'a [RubricObservationV1],
) -> Result<BTreeMap<&'a str, &'a RubricObservationV1>, EditAssuranceError> {
    if observations.len() != case.rubrics.len() {
        return Err(EditAssuranceError::new(format!(
            "case {} requires exactly {} rubric observations, received {}",
            case.sample_id,
            case.rubrics.len(),
            observations.len()
        )));
    }

    let expected: BTreeMap<&str, (&str, &str)> = case
        .rubrics
        .iter()
        .map(|rubric| {
            (
                rubric.rubric_id.as_str(),
                (rubric.metric_id.as_str(), rubric.unit.as_str()),
            )
        })
        .collect();
    let mut indexed = BTreeMap::new();

    for observation in observations {
        require_non_empty(&observation.rubric_id, "observation rubric_id")?;
        require_non_empty(&observation.metric_id, "observation metric_id")?;
        require_non_empty(&observation.unit, "observation unit")?;
        require_non_empty(&observation.evaluator_id, "evaluator_id")?;
        require_non_empty(&observation.evaluator_version, "evaluator_version")?;
        validate_digest(&observation.evaluator_artifact_sha256, "evaluator_artifact_sha256")?;
        validate_digest(
            &observation.calibration_receipt_sha256,
            "rubric calibration_receipt_sha256",
        )?;

        if !observation.lower_bound.is_finite()
            || !observation.upper_bound.is_finite()
            || observation.lower_bound > observation.upper_bound
        {
            return Err(EditAssuranceError::new(format!(
                "observation interval for rubric {} must be finite and ordered",
                observation.rubric_id
            )));
        }

        let (expected_metric, expected_unit) = expected
            .get(observation.rubric_id.as_str())
            .copied()
            .ok_or_else(|| {
                EditAssuranceError::new(format!(
                    "unexpected rubric observation {} in case {}",
                    observation.rubric_id, case.sample_id
                ))
            })?;
        if observation.metric_id != expected_metric || observation.unit != expected_unit {
            return Err(EditAssuranceError::new(format!(
                "metric ID or unit mismatch for rubric {} in case {}",
                observation.rubric_id, case.sample_id
            )));
        }
        if indexed
            .insert(observation.rubric_id.as_str(), observation)
            .is_some()
        {
            return Err(EditAssuranceError::new(format!(
                "duplicate observation for rubric {} in case {}",
                observation.rubric_id, case.sample_id
            )));
        }
    }

    for rubric in &case.rubrics {
        if !indexed.contains_key(rubric.rubric_id.as_str()) {
            return Err(EditAssuranceError::new(format!(
                "missing observation for rubric {} in case {}",
                rubric.rubric_id, case.sample_id
            )));
        }
    }
    Ok(indexed)
}

fn evaluate_rubric(
    requirement: &RubricRequirementV1,
    observation: &RubricObservationV1,
) -> RubricResultV1 {
    let decision = classify_interval(
        observation.lower_bound,
        observation.upper_bound,
        requirement.accepted_min,
        requirement.accepted_max,
    );

    RubricResultV1 {
        rubric_id: requirement.rubric_id.clone(),
        metric_id: requirement.metric_id.clone(),
        role: requirement.role,
        unit: requirement.unit.clone(),
        accepted_min: requirement.accepted_min,
        accepted_max: requirement.accepted_max,
        lower_bound: Some(observation.lower_bound),
        upper_bound: Some(observation.upper_bound),
        decision: Some(decision),
        evaluator_id: Some(observation.evaluator_id.clone()),
        evaluator_version: Some(observation.evaluator_version.clone()),
        evaluator_artifact_sha256: Some(observation.evaluator_artifact_sha256.clone()),
        calibration_receipt_sha256: Some(observation.calibration_receipt_sha256.clone()),
    }
}

fn classify_interval(
    lower: f64,
    upper: f64,
    accepted_min: f64,
    accepted_max: f64,
) -> RubricDecisionV1 {
    if lower >= accepted_min && upper <= accepted_max {
        RubricDecisionV1::Pass
    } else if upper < accepted_min || lower > accepted_max {
        RubricDecisionV1::Fail
    } else {
        RubricDecisionV1::Indeterminate
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn validate_digest(value: &str, field: &str) -> Result<(), EditAssuranceError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
    });
    if !valid {
        return Err(EditAssuranceError::new(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), EditAssuranceError> {
    if value.trim().is_empty() {
        return Err(EditAssuranceError::new(format!("{field} must not be empty")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn benchmark_spec() -> EditBenchmarkSpecV1 {
        EditBenchmarkSpecV1 {
            benchmark_id: "lyric-editbench-adapter".to_owned(),
            benchmark_version: "v1".to_owned(),
            evaluation_split_id: "heldout-singers".to_owned(),
            evaluation_manifest_sha256: digest('a'),
            calibration_split_id: "calibration-singers".to_owned(),
            calibration_manifest_sha256: digest('b'),
            calibration_receipt_sha256: digest('c'),
            system_id: "muse".to_owned(),
            system_version: "dev-v1".to_owned(),
            system_artifact_sha256: digest('d'),
            expected_groups: vec![EditStratumKeyV1 {
                operation: EditOperationV1::PartialSubstitution,
                language_id: "en".to_owned(),
                timbre_mode: TimbreModeV1::SameTimbre,
            }],
        }
    }

    fn case_spec() -> EditCaseSpecV1 {
        EditCaseSpecV1 {
            sample_id: "sample-001".to_owned(),
            operation: EditOperationV1::PartialSubstitution,
            language_id: "en".to_owned(),
            timbre_mode: TimbreModeV1::SameTimbre,
            singer_id: "singer-001".to_owned(),
            song_id: "song-001".to_owned(),
            recording_session_id: "session-001".to_owned(),
            technique_id: "legato".to_owned(),
            source_audio_sha256: digest('e'),
            melody_reference_sha256: digest('f'),
            timbre_reference_sha256: Some(digest('1')),
            original_lyrics_sha256: Some(digest('2')),
            requested_lyrics_sha256: Some(digest('3')),
            task_manifest_sha256: digest('4'),
            rubrics: vec![
                RubricRequirementV1 {
                    rubric_id: "lyric-realization".to_owned(),
                    metric_id: "phoneme-error-rate".to_owned(),
                    role: RubricRoleV1::Instruction,
                    unit: "ratio".to_owned(),
                    accepted_min: 0.0,
                    accepted_max: 0.15,
                },
                RubricRequirementV1 {
                    rubric_id: "melody-preservation".to_owned(),
                    metric_id: "f0-correlation".to_owned(),
                    role: RubricRoleV1::Preservation,
                    unit: "pearson_r".to_owned(),
                    accepted_min: 0.90,
                    accepted_max: 1.0,
                },
                RubricRequirementV1 {
                    rubric_id: "timbre-preservation".to_owned(),
                    metric_id: "speaker-similarity".to_owned(),
                    role: RubricRoleV1::Preservation,
                    unit: "cosine".to_owned(),
                    accepted_min: 0.80,
                    accepted_max: 1.0,
                },
            ],
        }
    }

    fn observation(
        rubric_id: &str,
        metric_id: &str,
        unit: &str,
        lower: f64,
        upper: f64,
        evaluator_id: &str,
    ) -> RubricObservationV1 {
        RubricObservationV1 {
            rubric_id: rubric_id.to_owned(),
            metric_id: metric_id.to_owned(),
            unit: unit.to_owned(),
            lower_bound: lower,
            upper_bound: upper,
            evaluator_id: evaluator_id.to_owned(),
            evaluator_version: "pinned-v1".to_owned(),
            evaluator_artifact_sha256: digest('5'),
            calibration_receipt_sha256: digest('6'),
        }
    }

    fn produced_execution() -> EditCaseExecutionV1 {
        EditCaseExecutionV1 {
            sample_id: "sample-001".to_owned(),
            status: ExecutionStatusV1::Produced,
            output_audio_sha256: Some(digest('7')),
            failure_receipt_sha256: None,
            observations: vec![
                observation(
                    "lyric-realization",
                    "phoneme-error-rate",
                    "ratio",
                    0.05,
                    0.10,
                    "singing-asr",
                ),
                observation(
                    "melody-preservation",
                    "f0-correlation",
                    "pearson_r",
                    0.94,
                    0.97,
                    "rmvpe-correlation",
                ),
                observation(
                    "timbre-preservation",
                    "speaker-similarity",
                    "cosine",
                    0.86,
                    0.90,
                    "wavlm-speaker",
                ),
            ],
        }
    }

    fn evaluate_one(
        case: EditCaseSpecV1,
        execution: EditCaseExecutionV1,
    ) -> Result<EditBenchmarkReportV1, EditAssuranceError> {
        evaluate_edit_benchmark(&benchmark_spec(), &[case], &[execution])
    }

    #[test]
    fn all_instruction_and_preservation_rubrics_pass_exact_match() {
        let report = evaluate_one(case_spec(), produced_execution()).unwrap();
        assert_eq!(report.exact_match_rate, 1.0);
        assert_eq!(report.instruction_following.rate, 1.0);
        assert_eq!(report.consistency.rate, 1.0);
        assert_eq!(report.cases[0].outcome, CaseOutcomeV1::ExactMatch);
    }

    #[test]
    fn correct_lyrics_with_broken_melody_fails_exact_match() {
        let mut execution = produced_execution();
        execution.observations[1].lower_bound = 0.70;
        execution.observations[1].upper_bound = 0.78;
        let report = evaluate_one(case_spec(), execution).unwrap();
        assert_eq!(report.instruction_following.rate, 1.0);
        assert_eq!(report.consistency.rate, 0.5);
        assert_eq!(report.exact_match_rate, 0.0);
        assert_eq!(report.cases[0].outcome, CaseOutcomeV1::Failed);
    }

    #[test]
    fn preservation_success_cannot_hide_wrong_lyrics() {
        let mut execution = produced_execution();
        execution.observations[0].lower_bound = 0.30;
        execution.observations[0].upper_bound = 0.40;
        let report = evaluate_one(case_spec(), execution).unwrap();
        assert_eq!(report.instruction_following.rate, 0.0);
        assert_eq!(report.consistency.rate, 1.0);
        assert_eq!(report.exact_match_rate, 0.0);
    }

    #[test]
    fn no_op_or_wrong_operation_does_not_get_exact_match_credit() {
        let mut execution = produced_execution();
        execution.observations[0].lower_bound = 0.95;
        execution.observations[0].upper_bound = 1.0;
        let report = evaluate_one(case_spec(), execution).unwrap();
        assert_eq!(report.exact_match_cases, 0);
        assert_eq!(report.exact_match_rate, 0.0);
    }

    #[test]
    fn boundary_overlap_is_indeterminate_and_never_passes() {
        let mut execution = produced_execution();
        execution.observations[0].lower_bound = 0.10;
        execution.observations[0].upper_bound = 0.20;
        let report = evaluate_one(case_spec(), execution).unwrap();
        assert_eq!(report.instruction_following.rate, 0.0);
        assert_eq!(report.cases[0].outcome, CaseOutcomeV1::Indeterminate);
        assert_eq!(
            report.cases[0].rubrics[0].decision,
            Some(RubricDecisionV1::Indeterminate)
        );
        assert_eq!(report.indeterminate_cases, 1);
    }

    #[test]
    fn unsupported_execution_stays_in_every_denominator() {
        let execution = EditCaseExecutionV1 {
            sample_id: "sample-001".to_owned(),
            status: ExecutionStatusV1::Unsupported,
            output_audio_sha256: None,
            failure_receipt_sha256: Some(digest('8')),
            observations: Vec::new(),
        };
        let report = evaluate_one(case_spec(), execution).unwrap();
        assert_eq!(report.exact_match_rate, 0.0);
        assert_eq!(report.instruction_following.total, 1);
        assert_eq!(report.consistency.total, 2);
        assert_eq!(report.instruction_following.unscored, 1);
        assert_eq!(report.consistency.unscored, 2);
        assert_eq!(report.unsupported_cases, 1);
    }

    #[test]
    fn failed_and_abstained_runs_are_not_dropped() {
        for (status, expected) in [
            (ExecutionStatusV1::Failed, CaseOutcomeV1::ExecutionFailed),
            (ExecutionStatusV1::Abstained, CaseOutcomeV1::Abstained),
        ] {
            let execution = EditCaseExecutionV1 {
                sample_id: "sample-001".to_owned(),
                status,
                output_audio_sha256: None,
                failure_receipt_sha256: Some(digest('9')),
                observations: Vec::new(),
            };
            let report = evaluate_one(case_spec(), execution).unwrap();
            assert_eq!(report.cases[0].outcome, expected);
            assert_eq!(report.exact_match_rate, 0.0);
            assert_eq!(report.instruction_following.total, 1);
            assert_eq!(report.consistency.total, 2);
        }
    }

    #[test]
    fn missing_rubric_observation_is_rejected() {
        let mut execution = produced_execution();
        execution.observations.pop();
        assert!(evaluate_one(case_spec(), execution).is_err());
    }

    #[test]
    fn duplicate_rubric_observation_is_rejected() {
        let mut execution = produced_execution();
        execution.observations.push(execution.observations[0].clone());
        assert!(evaluate_one(case_spec(), execution).is_err());
    }

    #[test]
    fn unexpected_rubric_observation_is_rejected() {
        let mut execution = produced_execution();
        execution.observations[0].rubric_id = "unrequested".to_owned();
        assert!(evaluate_one(case_spec(), execution).is_err());
    }

    #[test]
    fn metric_id_and_unit_must_match_the_rubric_contract() {
        let mut wrong_metric = produced_execution();
        wrong_metric.observations[0].metric_id = "different-metric".to_owned();
        assert!(evaluate_one(case_spec(), wrong_metric).is_err());

        let mut wrong_unit = produced_execution();
        wrong_unit.observations[0].unit = "seconds".to_owned();
        assert!(evaluate_one(case_spec(), wrong_unit).is_err());
    }

    #[test]
    fn malformed_intervals_and_evaluator_digests_are_rejected() {
        let mut reversed = produced_execution();
        reversed.observations[0].lower_bound = 0.9;
        reversed.observations[0].upper_bound = 0.1;
        assert!(evaluate_one(case_spec(), reversed).is_err());

        let mut invalid_digest = produced_execution();
        invalid_digest.observations[0].evaluator_artifact_sha256 = "bad".to_owned();
        assert!(evaluate_one(case_spec(), invalid_digest).is_err());
    }

    #[test]
    fn duplicate_case_ids_and_missing_executions_are_rejected() {
        let case = case_spec();
        let execution = produced_execution();
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[case.clone(), case],
            &[execution.clone(), execution.clone()],
        )
        .is_err());

        assert!(evaluate_edit_benchmark(&benchmark_spec(), &[case_spec()], &[]).is_err());
    }

    #[test]
    fn evaluation_and_calibration_manifests_must_differ() {
        let mut spec = benchmark_spec();
        spec.evaluation_manifest_sha256 = spec.calibration_manifest_sha256.clone();
        assert!(evaluate_edit_benchmark(
            &spec,
            &[case_spec()],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn missing_predeclared_group_is_rejected() {
        let mut spec = benchmark_spec();
        spec.expected_groups.push(EditStratumKeyV1 {
            operation: EditOperationV1::Deletion,
            language_id: "en".to_owned(),
            timbre_mode: TimbreModeV1::SameTimbre,
        });
        assert!(evaluate_edit_benchmark(
            &spec,
            &[case_spec()],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn unexpected_group_is_rejected() {
        let mut case = case_spec();
        case.language_id = "zh".to_owned();
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[case],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn lyric_tasks_require_both_original_and_requested_lyric_digests() {
        let mut case = case_spec();
        case.requested_lyrics_sha256 = None;
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[case],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn timbre_reference_mode_must_match_reference_presence() {
        let mut case = case_spec();
        case.timbre_reference_sha256 = None;
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[case],
            &[produced_execution()],
        )
        .is_err());

        let mut not_applicable = case_spec();
        not_applicable.timbre_mode = TimbreModeV1::NotApplicable;
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[not_applicable],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn rubric_profile_requires_both_instruction_and_preservation_roles() {
        let mut case = case_spec();
        case.rubrics
            .retain(|rubric| rubric.role == RubricRoleV1::Instruction);
        assert!(evaluate_edit_benchmark(
            &benchmark_spec(),
            &[case],
            &[produced_execution()],
        )
        .is_err());
    }

    #[test]
    fn per_group_metrics_are_reported_without_collapsing_task_types() {
        let mut spec = benchmark_spec();
        spec.expected_groups.push(EditStratumKeyV1 {
            operation: EditOperationV1::Deletion,
            language_id: "en".to_owned(),
            timbre_mode: TimbreModeV1::SameTimbre,
        });

        let mut deletion_case = case_spec();
        deletion_case.sample_id = "sample-002".to_owned();
        deletion_case.operation = EditOperationV1::Deletion;

        let mut deletion_execution = produced_execution();
        deletion_execution.sample_id = "sample-002".to_owned();

        let report = evaluate_edit_benchmark(
            &spec,
            &[case_spec(), deletion_case],
            &[produced_execution(), deletion_execution],
        )
        .unwrap();
        assert_eq!(report.groups.len(), 2);
        assert_eq!(report.macro_group_exact_match_rate, 1.0);
        assert_eq!(report.case_count, 2);
    }
}
