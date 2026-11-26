//! MusicListenBench-style FLIP/STAY scoring for task-relative musical
//! change sensitivity and sound-only nuisance invariance.
//!
//! The evaluator consumes already-produced model answers. It does not invoke
//! an audio model, authenticate audio bytes, or establish that an intervention
//! was correctly materialized.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const RUN_SCHEMA_V1: &str = "mel-auditory-flip-stay-run-v1";
const REPORT_SCHEMA_V1: &str = "mel-auditory-flip-stay-report-v1";
const CONFIRMATORY_SOURCE_FLOOR: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalTaskV1 {
    Melody,
    Harmony,
    Timbre,
    Rhythm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalRelationV1 {
    Same,
    Different,
    ClipAFaster,
    ClipBFaster,
}

impl MusicalRelationV1 {
    fn opposite(self) -> Self {
        match self {
            Self::Same => Self::Different,
            Self::Different => Self::Same,
            Self::ClipAFaster => Self::ClipBFaster,
            Self::ClipBFaster => Self::ClipAFaster,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAnswerV1 {
    A,
    B,
    Unparseable,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlipFactorV1 {
    NoteIdentity,
    ThirdQuality,
    InstrumentFamily,
    PulseRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StayFactorV1 {
    Room,
    Eq,
    Hiss,
    Transposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditoryVariantV1 {
    pub variant_id: String,
    pub audio_sha256: String,
    pub question: String,
    pub question_sha256: String,
    pub option_a_relation: MusicalRelationV1,
    pub option_b_relation: MusicalRelationV1,
    pub expected_relation: MusicalRelationV1,
    pub model_answer: ModelAnswerV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditoryPairV1 {
    pub item_id: String,
    pub pair_id: String,
    /// Digest of the generating MIDI/work/recording lineage, not an annotator ID.
    pub source_lineage_sha256: String,
    pub task: MusicalTaskV1,
    pub flip_factor: FlipFactorV1,
    pub stay_factor: StayFactorV1,
    pub clean: AuditoryVariantV1,
    pub flip: AuditoryVariantV1,
    pub stay: AuditoryVariantV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditoryInvarianceRunV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub dataset_revision_sha256: String,
    pub split: DatasetSplitV1,
    pub model_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub pairs: Vec<AuditoryPairV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSplitV1 {
    RenderedHeldOut,
    RealRecordingHeldOut,
    SyntheticFixture,
    LocalExtension,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VariantOutcomeV1 {
    pub variant_id: String,
    pub audio_sha256: String,
    pub question_sha256: String,
    pub option_a_relation: MusicalRelationV1,
    pub option_b_relation: MusicalRelationV1,
    pub expected_relation: MusicalRelationV1,
    pub predicted_relation: Option<MusicalRelationV1>,
    pub model_answer: ModelAnswerV1,
    pub correct: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditoryPairResultV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_lineage_sha256: String,
    pub task: MusicalTaskV1,
    pub flip_factor: FlipFactorV1,
    pub stay_factor: StayFactorV1,
    pub clean: VariantOutcomeV1,
    pub flip: VariantOutcomeV1,
    pub stay: VariantOutcomeV1,
    pub joint_clean_flip_stay_correct: bool,
    pub conditional_flip_eligible: bool,
    pub conditional_stay_eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditoryMetricSummaryV1 {
    pub pairs: usize,
    pub clean_items: usize,
    pub clean_correct: usize,
    pub clean_accuracy: Option<f64>,
    pub flip_items: usize,
    pub flip_correct: usize,
    pub flip_miss_count: usize,
    pub miss_rate: Option<f64>,
    pub stay_items: usize,
    pub stay_correct: usize,
    pub stay_false_flip_count: usize,
    pub false_flip_rate: Option<f64>,
    pub conditional_flip_items: usize,
    pub conditional_flip_misses: usize,
    pub conditional_miss_rate: Option<f64>,
    pub conditional_stay_items: usize,
    pub conditional_stay_false_flips: usize,
    pub conditional_false_flip_rate: Option<f64>,
    pub accuracy_flip_stay: Option<f64>,
    pub joint_clean_flip_stay_correct: usize,
    pub joint_clean_flip_stay_rate: Option<f64>,
    pub answer_a_count: usize,
    pub answer_b_count: usize,
    pub unparseable_count: usize,
    pub abstain_count: usize,
    pub parsed_answer_rate: Option<f64>,
    pub answer_a_rate: Option<f64>,
    pub always_a_miss_rate: Option<f64>,
    pub always_a_false_flip_rate: Option<f64>,
    pub always_a_accuracy_flip_stay: Option<f64>,
    pub always_b_miss_rate: Option<f64>,
    pub always_b_false_flip_rate: Option<f64>,
    pub always_b_accuracy_flip_stay: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskMetricV1 {
    pub task: MusicalTaskV1,
    pub metrics: AuditoryMetricSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StayFactorMetricV1 {
    pub factor: StayFactorV1,
    pub metrics: AuditoryMetricSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditoryInvarianceReportV1 {
    pub schema: &'static str,
    pub dataset_id: String,
    pub dataset_version: String,
    pub dataset_revision_sha256: String,
    pub split: DatasetSplitV1,
    pub model_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub pair_count: usize,
    pub source_lineage_clusters: usize,
    pub recommended_confirmatory_source_floor: usize,
    pub confirmatory_source_floor_met: bool,
    pub overall: AuditoryMetricSummaryV1,
    pub by_task: Vec<TaskMetricV1>,
    pub by_stay_factor: Vec<StayFactorMetricV1>,
    pub pair_results: Vec<AuditoryPairResultV1>,
    /// The input includes digests, but this scorer does not re-read audio bytes.
    pub audio_bytes_recomputed: bool,
    /// Runtime and inference identity are declarations, not signed run receipts.
    pub inference_receipt_authenticated: bool,
    pub benchmark_manifest_authenticated: bool,
    pub population_confidence_interval_computed: bool,
}

pub fn audit_auditory_invariance_json(input: &str) -> Result<AuditoryInvarianceReportV1, String> {
    let run: AuditoryInvarianceRunV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid auditory FLIP/STAY run JSON: {error}"))?;
    audit_auditory_invariance(&run)
}

pub fn audit_auditory_invariance(
    run: &AuditoryInvarianceRunV1,
) -> Result<AuditoryInvarianceReportV1, String> {
    validate_run_identity(run)?;
    if run.pairs.is_empty() {
        return Err("auditory FLIP/STAY run must contain at least one complete pair".into());
    }

    let mut item_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut variant_ids = BTreeSet::new();
    let mut source_lineages = BTreeSet::new();
    let mut results = Vec::with_capacity(run.pairs.len());

    for (index, pair) in run.pairs.iter().enumerate() {
        validate_pair(pair, index)?;
        if !item_ids.insert(pair.item_id.as_str()) {
            return Err(format!("duplicate item_id at pair index {index}"));
        }
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(format!("duplicate pair_id at pair index {index}"));
        }
        for variant in [&pair.clean, &pair.flip, &pair.stay] {
            if !variant_ids.insert(variant.variant_id.as_str()) {
                return Err(format!("duplicate variant_id: {}", variant.variant_id));
            }
        }
        source_lineages.insert(pair.source_lineage_sha256.as_str());
        results.push(score_pair(pair));
    }

    let overall = summarize(&results);
    let mut task_groups: BTreeMap<MusicalTaskV1, Vec<AuditoryPairResultV1>> = BTreeMap::new();
    let mut factor_groups: BTreeMap<StayFactorV1, Vec<AuditoryPairResultV1>> = BTreeMap::new();
    for result in &results {
        task_groups
            .entry(result.task)
            .or_default()
            .push(result.clone());
        factor_groups
            .entry(result.stay_factor)
            .or_default()
            .push(result.clone());
    }
    let by_task = task_groups
        .into_iter()
        .map(|(task, items)| TaskMetricV1 {
            task,
            metrics: summarize(&items),
        })
        .collect();
    let by_stay_factor = factor_groups
        .into_iter()
        .map(|(factor, items)| StayFactorMetricV1 {
            factor,
            metrics: summarize(&items),
        })
        .collect();

    Ok(AuditoryInvarianceReportV1 {
        schema: REPORT_SCHEMA_V1,
        dataset_id: run.dataset_id.clone(),
        dataset_version: run.dataset_version.clone(),
        dataset_revision_sha256: run.dataset_revision_sha256.clone(),
        split: run.split,
        model_identity_sha256: run.model_identity_sha256.clone(),
        system_prompt_sha256: run.system_prompt_sha256.clone(),
        runtime_identity_sha256: run.runtime_identity_sha256.clone(),
        decoding_config_sha256: run.decoding_config_sha256.clone(),
        pair_count: results.len(),
        source_lineage_clusters: source_lineages.len(),
        recommended_confirmatory_source_floor: CONFIRMATORY_SOURCE_FLOOR,
        confirmatory_source_floor_met: source_lineages.len() >= CONFIRMATORY_SOURCE_FLOOR,
        overall,
        by_task,
        by_stay_factor,
        pair_results: results,
        audio_bytes_recomputed: false,
        inference_receipt_authenticated: false,
        benchmark_manifest_authenticated: false,
        population_confidence_interval_computed: false,
    })
}

fn validate_run_identity(run: &AuditoryInvarianceRunV1) -> Result<(), String> {
    if run.schema != RUN_SCHEMA_V1 {
        return Err(format!("unsupported auditory run schema: {}", run.schema));
    }
    require_text(&run.dataset_id, "dataset_id")?;
    require_text(&run.dataset_version, "dataset_version")?;
    validate_digest(&run.dataset_revision_sha256, "dataset_revision_sha256")?;
    validate_digest(&run.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&run.system_prompt_sha256, "system_prompt_sha256")?;
    validate_digest(&run.runtime_identity_sha256, "runtime_identity_sha256")?;
    validate_digest(&run.decoding_config_sha256, "decoding_config_sha256")
}

fn validate_pair(pair: &AuditoryPairV1, index: usize) -> Result<(), String> {
    for (label, value) in [
        ("item_id", pair.item_id.as_str()),
        ("pair_id", pair.pair_id.as_str()),
    ] {
        require_text(value, &format!("pairs[{index}].{label}"))?;
    }
    validate_digest(
        &pair.source_lineage_sha256,
        &format!("pairs[{index}].source_lineage_sha256"),
    )?;

    validate_factor_scope(pair.task, pair.flip_factor, pair.stay_factor)?;
    validate_variant(&pair.clean, pair.task, "clean")?;
    validate_variant(&pair.flip, pair.task, "flip")?;
    validate_variant(&pair.stay, pair.task, "stay")?;

    let variants = [&pair.clean, &pair.flip, &pair.stay];
    let mut ids = BTreeSet::new();
    let mut audio_digests = BTreeSet::new();
    for (variant_index, variant) in variants.iter().enumerate() {
        if !ids.insert(variant.variant_id.as_str()) {
            return Err(format!("pair {} repeats a variant_id", pair.pair_id));
        }
        if !audio_digests.insert(variant.audio_sha256.as_str()) {
            return Err(format!(
                "pair {} reuses an audio digest across variants",
                pair.pair_id
            ));
        }
        if variant_index > 0 && variant.question.trim().is_empty() {
            return Err(format!("pair {} contains an empty question", pair.pair_id));
        }
    }

    if pair.stay.expected_relation != pair.clean.expected_relation {
        return Err(format!(
            "pair {} STAY expected relation must equal CLEAN",
            pair.pair_id
        ));
    }
    if pair.flip.expected_relation != pair.clean.expected_relation.opposite() {
        return Err(format!(
            "pair {} FLIP expected relation must oppose CLEAN",
            pair.pair_id
        ));
    }
    Ok(())
}

fn validate_variant(
    variant: &AuditoryVariantV1,
    task: MusicalTaskV1,
    label: &str,
) -> Result<(), String> {
    require_text(&variant.variant_id, &format!("{label}.variant_id"))?;
    validate_digest(&variant.audio_sha256, &format!("{label}.audio_sha256"))?;
    require_text(&variant.question, &format!("{label}.question"))?;
    validate_digest(
        &variant.question_sha256,
        &format!("{label}.question_sha256"),
    )?;
    let observed_question_digest = format!("{:x}", Sha256::digest(variant.question.as_bytes()));
    if observed_question_digest != variant.question_sha256 {
        return Err(format!(
            "{label}.question_sha256 does not match the supplied question text"
        ));
    }
    let pair_is_valid = match task {
        MusicalTaskV1::Melody | MusicalTaskV1::Harmony | MusicalTaskV1::Timbre => {
            matches!(
                (variant.option_a_relation, variant.option_b_relation),
                (MusicalRelationV1::Same, MusicalRelationV1::Different)
                    | (MusicalRelationV1::Different, MusicalRelationV1::Same)
            )
        }
        MusicalTaskV1::Rhythm => {
            matches!(
                (variant.option_a_relation, variant.option_b_relation),
                (
                    MusicalRelationV1::ClipAFaster,
                    MusicalRelationV1::ClipBFaster
                ) | (
                    MusicalRelationV1::ClipBFaster,
                    MusicalRelationV1::ClipAFaster
                )
            )
        }
    };
    if !pair_is_valid {
        return Err(format!("{label} option meanings are invalid for its task"));
    }
    if variant.expected_relation != variant.option_a_relation
        && variant.expected_relation != variant.option_b_relation
    {
        return Err(format!(
            "{label} expected relation is not represented by A or B"
        ));
    }
    Ok(())
}

fn validate_factor_scope(
    task: MusicalTaskV1,
    flip: FlipFactorV1,
    stay: StayFactorV1,
) -> Result<(), String> {
    let flip_matches = matches!(
        (task, flip),
        (MusicalTaskV1::Melody, FlipFactorV1::NoteIdentity)
            | (MusicalTaskV1::Harmony, FlipFactorV1::ThirdQuality)
            | (MusicalTaskV1::Timbre, FlipFactorV1::InstrumentFamily)
            | (MusicalTaskV1::Rhythm, FlipFactorV1::PulseRate)
    );
    if !flip_matches {
        return Err(format!(
            "FLIP factor {flip:?} is incompatible with task {task:?}"
        ));
    }
    if stay == StayFactorV1::Transposition
        && !matches!(task, MusicalTaskV1::Melody | MusicalTaskV1::Harmony)
    {
        return Err(format!("transposition STAY is not valid for task {task:?}"));
    }
    Ok(())
}

fn score_pair(pair: &AuditoryPairV1) -> AuditoryPairResultV1 {
    let clean = score_variant(&pair.clean);
    let flip = score_variant(&pair.flip);
    let stay = score_variant(&pair.stay);
    AuditoryPairResultV1 {
        item_id: pair.item_id.clone(),
        pair_id: pair.pair_id.clone(),
        source_lineage_sha256: pair.source_lineage_sha256.clone(),
        task: pair.task,
        flip_factor: pair.flip_factor,
        stay_factor: pair.stay_factor,
        joint_clean_flip_stay_correct: clean.correct && flip.correct && stay.correct,
        conditional_flip_eligible: clean.correct,
        conditional_stay_eligible: clean.correct,
        clean,
        flip,
        stay,
    }
}

fn score_variant(variant: &AuditoryVariantV1) -> VariantOutcomeV1 {
    let predicted_relation = match variant.model_answer {
        ModelAnswerV1::A => Some(variant.option_a_relation),
        ModelAnswerV1::B => Some(variant.option_b_relation),
        ModelAnswerV1::Unparseable | ModelAnswerV1::Abstain => None,
    };
    VariantOutcomeV1 {
        variant_id: variant.variant_id.clone(),
        audio_sha256: variant.audio_sha256.clone(),
        question_sha256: variant.question_sha256.clone(),
        option_a_relation: variant.option_a_relation,
        option_b_relation: variant.option_b_relation,
        expected_relation: variant.expected_relation,
        predicted_relation,
        model_answer: variant.model_answer,
        correct: predicted_relation == Some(variant.expected_relation),
    }
}

fn summarize(results: &[AuditoryPairResultV1]) -> AuditoryMetricSummaryV1 {
    let pairs = results.len();
    let clean_correct = results.iter().filter(|result| result.clean.correct).count();
    let flip_correct = results.iter().filter(|result| result.flip.correct).count();
    let flip_miss_count = pairs.saturating_sub(flip_correct);
    let stay_correct = results.iter().filter(|result| result.stay.correct).count();
    let stay_false_flip_count = pairs.saturating_sub(stay_correct);

    let conditional: Vec<_> = results
        .iter()
        .filter(|result| result.conditional_stay_eligible)
        .collect();
    let conditional_flip_eligible = results
        .iter()
        .filter(|result| result.conditional_flip_eligible)
        .count();
    let conditional_flip_misses = results
        .iter()
        .filter(|result| result.conditional_flip_eligible && !result.flip.correct)
        .count();
    let conditional_stay_false_flips = conditional
        .iter()
        .filter(|result| !result.stay.correct)
        .count();

    let all_answers = results
        .iter()
        .flat_map(|result| [&result.clean, &result.flip, &result.stay]);
    let (answer_a_count, answer_b_count, unparseable_count, abstain_count) =
        all_answers.fold((0, 0, 0, 0), |mut counts, answer| {
            match answer.model_answer {
                ModelAnswerV1::A => counts.0 += 1,
                ModelAnswerV1::B => counts.1 += 1,
                ModelAnswerV1::Unparseable => counts.2 += 1,
                ModelAnswerV1::Abstain => counts.3 += 1,
            }
            counts
        });
    let answer_count = pairs.saturating_mul(3);
    let parsed_count = answer_a_count + answer_b_count;
    let joint = results
        .iter()
        .filter(|result| result.joint_clean_flip_stay_correct)
        .count();

    let miss_rate = rate(flip_miss_count, pairs);
    let false_flip_rate = rate(stay_false_flip_count, pairs);
    let always_a_misses = results
        .iter()
        .filter(|result| result.flip.option_a_relation != result.flip.expected_relation)
        .count();
    let always_a_false_flips = results
        .iter()
        .filter(|result| result.stay.option_a_relation != result.stay.expected_relation)
        .count();
    let always_b_misses = results
        .iter()
        .filter(|result| result.flip.option_b_relation != result.flip.expected_relation)
        .count();
    let always_b_false_flips = results
        .iter()
        .filter(|result| result.stay.option_b_relation != result.stay.expected_relation)
        .count();
    let always_a_miss_rate = rate(always_a_misses, pairs);
    let always_a_false_flip_rate = rate(always_a_false_flips, pairs);
    let always_b_miss_rate = rate(always_b_misses, pairs);
    let always_b_false_flip_rate = rate(always_b_false_flips, pairs);

    AuditoryMetricSummaryV1 {
        pairs,
        clean_items: pairs,
        clean_correct,
        clean_accuracy: rate(clean_correct, pairs),
        flip_items: pairs,
        flip_correct,
        flip_miss_count,
        miss_rate,
        stay_items: pairs,
        stay_correct,
        stay_false_flip_count,
        false_flip_rate,
        conditional_flip_items: conditional_flip_eligible,
        conditional_flip_misses,
        conditional_miss_rate: rate(conditional_flip_misses, conditional_flip_eligible),
        conditional_stay_items: conditional.len(),
        conditional_stay_false_flips,
        conditional_false_flip_rate: rate(conditional_stay_false_flips, conditional.len()),
        accuracy_flip_stay: match (miss_rate, false_flip_rate) {
            (Some(mr), Some(ffr)) => Some(1.0 - (mr + ffr) / 2.0),
            _ => None,
        },
        joint_clean_flip_stay_correct: joint,
        joint_clean_flip_stay_rate: rate(joint, pairs),
        answer_a_count,
        answer_b_count,
        unparseable_count,
        abstain_count,
        parsed_answer_rate: rate(parsed_count, answer_count),
        answer_a_rate: rate(answer_a_count, parsed_count),
        always_a_miss_rate,
        always_a_false_flip_rate,
        always_a_accuracy_flip_stay: match (always_a_miss_rate, always_a_false_flip_rate) {
            (Some(mr), Some(ffr)) => Some(1.0 - (mr + ffr) / 2.0),
            _ => None,
        },
        always_b_miss_rate,
        always_b_false_flip_rate,
        always_b_accuracy_flip_stay: match (always_b_miss_rate, always_b_false_flip_rate) {
            (Some(mr), Some(ffr)) => Some(1.0 - (mr + ffr) / 2.0),
            _ => None,
        },
    }
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATASET: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PROMPT: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const RUNTIME: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DECODING: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const LINEAGE: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn digest(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    fn variant(
        id: &str,
        audio_seed: &str,
        question: &str,
        expected: MusicalRelationV1,
        answer: ModelAnswerV1,
    ) -> AuditoryVariantV1 {
        AuditoryVariantV1 {
            variant_id: id.into(),
            audio_sha256: digest(audio_seed),
            question: question.into(),
            question_sha256: digest(question),
            option_a_relation: MusicalRelationV1::Same,
            option_b_relation: MusicalRelationV1::Different,
            expected_relation: expected,
            model_answer: answer,
        }
    }

    fn pair(id: &str) -> AuditoryPairV1 {
        AuditoryPairV1 {
            item_id: format!("item-{id}"),
            pair_id: format!("pair-{id}"),
            source_lineage_sha256: LINEAGE.into(),
            task: MusicalTaskV1::Melody,
            flip_factor: FlipFactorV1::NoteIdentity,
            stay_factor: StayFactorV1::Hiss,
            clean: variant(
                &format!("{id}-clean"),
                &format!("{id}-clean-audio"),
                "Are these melodies the same? A=same, B=different.",
                MusicalRelationV1::Same,
                ModelAnswerV1::A,
            ),
            flip: variant(
                &format!("{id}-flip"),
                &format!("{id}-flip-audio"),
                "Are these melodies the same? A=same, B=different.",
                MusicalRelationV1::Different,
                ModelAnswerV1::B,
            ),
            stay: variant(
                &format!("{id}-stay"),
                &format!("{id}-stay-audio"),
                "Are these melodies the same? A=same, B=different.",
                MusicalRelationV1::Same,
                ModelAnswerV1::A,
            ),
        }
    }

    fn run(pairs: Vec<AuditoryPairV1>) -> AuditoryInvarianceRunV1 {
        AuditoryInvarianceRunV1 {
            schema: RUN_SCHEMA_V1.into(),
            dataset_id: "amaai-lab/MusicListenBench".into(),
            dataset_version: "1.0".into(),
            dataset_revision_sha256: DATASET.into(),
            split: DatasetSplitV1::SyntheticFixture,
            model_identity_sha256: MODEL.into(),
            system_prompt_sha256: PROMPT.into(),
            runtime_identity_sha256: RUNTIME.into(),
            decoding_config_sha256: DECODING.into(),
            pairs,
        }
    }

    #[test]
    fn fully_correct_triad_scores_one_on_clean_flip_stay_and_joint() {
        let report = audit_auditory_invariance(&run(vec![pair("one")])).unwrap();
        assert_eq!(report.overall.clean_accuracy, Some(1.0));
        assert_eq!(report.overall.miss_rate, Some(0.0));
        assert_eq!(report.overall.false_flip_rate, Some(0.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(1.0));
        assert_eq!(report.overall.joint_clean_flip_stay_rate, Some(1.0));
    }

    #[test]
    fn detects_both_over_sensitivity_and_under_sensitivity_separately() {
        let mut example = pair("one");
        example.stay.model_answer = ModelAnswerV1::B;
        let report = audit_auditory_invariance(&run(vec![example])).unwrap();
        assert_eq!(report.overall.miss_rate, Some(0.0));
        assert_eq!(report.overall.false_flip_rate, Some(1.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(0.5));
        assert_eq!(report.overall.conditional_false_flip_rate, Some(1.0));

        let mut example = pair("two");
        example.flip.model_answer = ModelAnswerV1::A;
        let report = audit_auditory_invariance(&run(vec![example])).unwrap();
        assert_eq!(report.overall.miss_rate, Some(1.0));
        assert_eq!(report.overall.false_flip_rate, Some(0.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(0.5));
    }

    #[test]
    fn reports_actual_constant_letter_baselines_under_question_specific_mapping() {
        let report = audit_auditory_invariance(&run(vec![pair("letter-bias")])).unwrap();
        assert_eq!(report.overall.always_a_miss_rate, Some(1.0));
        assert_eq!(report.overall.always_a_false_flip_rate, Some(0.0));
        assert_eq!(report.overall.always_a_accuracy_flip_stay, Some(0.5));
        assert_eq!(report.overall.always_b_miss_rate, Some(0.0));
        assert_eq!(report.overall.always_b_false_flip_rate, Some(1.0));
        assert_eq!(report.overall.always_b_accuracy_flip_stay, Some(0.5));
    }

    #[test]
    fn conditional_false_flip_uses_only_clean_correct_denominator() {
        let mut example = pair("one");
        example.clean.model_answer = ModelAnswerV1::B;
        example.stay.model_answer = ModelAnswerV1::B;
        let report = audit_auditory_invariance(&run(vec![example])).unwrap();
        assert_eq!(report.overall.clean_accuracy, Some(0.0));
        assert_eq!(report.overall.false_flip_rate, Some(1.0));
        assert_eq!(report.overall.conditional_stay_items, 0);
        assert_eq!(report.overall.conditional_false_flip_rate, None);
    }

    #[test]
    fn rhythm_letter_mapping_is_decoded_to_semantic_order() {
        let mut example = pair("rhythm");
        example.task = MusicalTaskV1::Rhythm;
        example.flip_factor = FlipFactorV1::PulseRate;
        example.clean.option_a_relation = MusicalRelationV1::ClipAFaster;
        example.clean.option_b_relation = MusicalRelationV1::ClipBFaster;
        example.clean.expected_relation = MusicalRelationV1::ClipAFaster;
        example.clean.model_answer = ModelAnswerV1::A;
        example.flip.option_a_relation = MusicalRelationV1::ClipBFaster;
        example.flip.option_b_relation = MusicalRelationV1::ClipAFaster;
        example.flip.expected_relation = MusicalRelationV1::ClipBFaster;
        example.flip.model_answer = ModelAnswerV1::A;
        example.stay.option_a_relation = MusicalRelationV1::ClipAFaster;
        example.stay.option_b_relation = MusicalRelationV1::ClipBFaster;
        example.stay.expected_relation = MusicalRelationV1::ClipAFaster;
        example.stay.model_answer = ModelAnswerV1::A;
        let report = audit_auditory_invariance(&run(vec![example])).unwrap();
        assert_eq!(report.overall.miss_rate, Some(0.0));
        assert_eq!(report.overall.false_flip_rate, Some(0.0));
        assert_eq!(report.overall.answer_a_rate, Some(1.0));
    }

    #[test]
    fn invalid_and_abstaining_outputs_remain_errors_and_reduce_coverage() {
        let mut example = pair("invalid");
        example.flip.model_answer = ModelAnswerV1::Unparseable;
        example.stay.model_answer = ModelAnswerV1::Abstain;
        let report = audit_auditory_invariance(&run(vec![example])).unwrap();
        assert_eq!(report.overall.flip_miss_count, 1);
        assert_eq!(report.overall.stay_false_flip_count, 1);
        assert_eq!(report.overall.unparseable_count, 1);
        assert_eq!(report.overall.abstain_count, 1);
        assert!(report.overall.parsed_answer_rate.unwrap() < 1.0);
    }

    #[test]
    fn task_incompatible_factors_and_invalid_pair_relations_are_rejected() {
        let mut example = pair("invalid-factor");
        example.task = MusicalTaskV1::Timbre;
        example.flip_factor = FlipFactorV1::InstrumentFamily;
        example.stay_factor = StayFactorV1::Transposition;
        assert!(audit_auditory_invariance(&run(vec![example])).is_err());

        let mut example = pair("invalid-relation");
        example.stay.expected_relation = MusicalRelationV1::Different;
        assert!(audit_auditory_invariance(&run(vec![example])).is_err());
    }

    #[test]
    fn duplicate_pairs_and_variant_audio_hashes_are_rejected() {
        let one = pair("one");
        assert!(audit_auditory_invariance(&run(vec![one.clone(), one])).is_err());

        let mut example = pair("duplicate-audio");
        example.stay.audio_sha256 = example.clean.audio_sha256.clone();
        assert!(audit_auditory_invariance(&run(vec![example])).is_err());
    }

    #[test]
    fn question_digest_mismatch_and_task_option_mapping_mismatch_are_rejected() {
        let mut example = pair("bad-question");
        example.clean.question_sha256 = DATASET.into();
        assert!(audit_auditory_invariance(&run(vec![example])).is_err());

        let mut example = pair("bad-options");
        example.task = MusicalTaskV1::Rhythm;
        example.flip_factor = FlipFactorV1::PulseRate;
        assert!(audit_auditory_invariance(&run(vec![example])).is_err());
    }

    #[test]
    fn task_and_stay_factor_denominators_reconcile_to_the_run() {
        let one = pair("one");
        let mut two = pair("two");
        two.task = MusicalTaskV1::Harmony;
        two.flip_factor = FlipFactorV1::ThirdQuality;
        two.stay_factor = StayFactorV1::Room;
        let report = audit_auditory_invariance(&run(vec![one, two])).unwrap();
        assert_eq!(report.by_task.len(), 2);
        assert_eq!(report.by_stay_factor.len(), 2);
        assert_eq!(
            report
                .by_task
                .iter()
                .map(|item| item.metrics.pairs)
                .sum::<usize>(),
            report.pair_count
        );
        assert_eq!(
            report
                .by_stay_factor
                .iter()
                .map(|item| item.metrics.pairs)
                .sum::<usize>(),
            report.pair_count
        );
    }
}
