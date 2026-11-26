//! Rust adapter for MusicListenBench v1.0's paired CLEAN / FLIP / STAY protocol.
//!
//! It scores supplied model outcomes only. It does not run a model, download
//! benchmark audio, authenticate audio bytes, or establish that a FLIP/STAY
//! transformation was applied. Task-specific transform qualification remains
//! with the independent acoustic/renderer owners.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CASE_SCHEMA_V1: &str = "mel-music-listen-bench-case-v1";
const REPORT_SCHEMA_V1: &str = "mel-music-listen-bench-report-v1";
const EVALUATOR_VERSION_V1: &str = "mel-music-listen-bench-evaluator-v1";
const CONFIRMATORY_LINEAGE_FLOOR: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicListenTaskV1 {
    Melody,
    Harmony,
    Timbre,
    Rhythm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicListenSplitV1 {
    RenderedTest,
    RealRecordingTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicalRelationV1 {
    Same,
    Different,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicListenAnswerV1 {
    ChoiceA,
    ChoiceB,
    Unparseable,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicListenFlipFactorV1 {
    NoteIdentity,
    ChordThirdQuality,
    InstrumentFamily,
    PulseRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicListenStayFactorV1 {
    Room,
    Eq,
    Hiss,
    RelativePitchPreservingTransposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicListenVariantV1 {
    pub variant_id: String,
    pub audio_sha256: String,
    pub question: String,
    pub question_sha256: String,
    /// Semantic meaning of the answer letter, not the letter itself.
    pub option_a_relation: MusicalRelationV1,
    pub option_b_relation: MusicalRelationV1,
    pub expected_relation: MusicalRelationV1,
    pub model_answer: MusicListenAnswerV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicListenTriadV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_lineage_sha256: String,
    pub task: MusicListenTaskV1,
    pub flip_factor: MusicListenFlipFactorV1,
    pub stay_factor: MusicListenStayFactorV1,
    pub model_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub clean: MusicListenVariantV1,
    pub flip: MusicListenVariantV1,
    pub stay: MusicListenVariantV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicListenBenchCaseV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    /// Immutable dataset/manifest revision used for the evaluated items.
    pub dataset_revision_sha256: String,
    /// Git tree or commit identity for the official benchmark scorer.
    pub official_scorer_revision_sha256: String,
    pub split: MusicListenSplitV1,
    pub triads: Vec<MusicListenTriadV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VariantMetricV1 {
    pub items: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub answered: usize,
    pub choice_a: usize,
    pub choice_b: usize,
    pub choice_a_rate_among_answered: Option<f64>,
    pub unparseable: usize,
    pub abstained: usize,
    pub accuracy: Option<f64>,
    pub coverage: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedMetricsV1 {
    pub triads: usize,
    pub clean: VariantMetricV1,
    pub flip: VariantMetricV1,
    pub stay: VariantMetricV1,
    pub flip_miss_rate: Option<f64>,
    pub stay_false_flip_rate: Option<f64>,
    /// Conditional denominators contain only triads with a correct CLEAN answer.
    pub clean_correct_triads: usize,
    pub conditional_flip_miss_rate: Option<f64>,
    pub conditional_stay_false_flip_rate: Option<f64>,
    pub accuracy_flip_stay: Option<f64>,
    pub joint_clean_flip_stay_exact_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicListenTaskSliceV1 {
    pub task: MusicListenTaskV1,
    pub metrics: PairedMetricsV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicListenFlipFactorSliceV1 {
    pub task: MusicListenTaskV1,
    pub factor: MusicListenFlipFactorV1,
    pub items: usize,
    pub miss_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicListenStayFactorSliceV1 {
    pub task: MusicListenTaskV1,
    pub factor: MusicListenStayFactorV1,
    pub items: usize,
    pub false_flip_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicListenTriadResultV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_lineage_sha256: String,
    pub task: MusicListenTaskV1,
    pub clean_predicted_relation: Option<MusicalRelationV1>,
    pub flip_predicted_relation: Option<MusicalRelationV1>,
    pub stay_predicted_relation: Option<MusicalRelationV1>,
    pub clean_correct: bool,
    pub flip_correct: bool,
    pub stay_correct: bool,
    pub joint_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicListenBenchReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub dataset_revision_sha256: String,
    pub official_scorer_revision_sha256: String,
    pub split: MusicListenSplitV1,
    pub overall: PairedMetricsV1,
    pub by_task: Vec<MusicListenTaskSliceV1>,
    pub by_flip_factor: Vec<MusicListenFlipFactorSliceV1>,
    pub by_stay_factor: Vec<MusicListenStayFactorSliceV1>,
    pub triad_results: Vec<MusicListenTriadResultV1>,
    pub source_lineage_count: usize,
    pub recommended_confirmatory_lineage_floor: usize,
    pub confirmatory_lineage_floor_met: bool,
    pub audio_bytes_recomputed: bool,
    pub transformations_authenticated: bool,
    pub model_invoked_by_evaluator: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScoredVariantV1 {
    predicted: Option<MusicalRelationV1>,
    correct: bool,
    answered: bool,
    choice_a: bool,
    choice_b: bool,
    unparseable: bool,
    abstained: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScoredTriadV1 {
    clean: ScoredVariantV1,
    flip: ScoredVariantV1,
    stay: ScoredVariantV1,
    joint_exact: bool,
}

pub fn audit_music_listen_case_json(input: &str) -> Result<MusicListenBenchReportV1, String> {
    let case: MusicListenBenchCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid MusicListenBench case JSON: {error}"))?;
    audit_music_listen_case(&case)
}

pub fn audit_music_listen_case(
    case: &MusicListenBenchCaseV1,
) -> Result<MusicListenBenchReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!("unsupported MusicListenBench case schema: {}", case.schema));
    }
    require_text(&case.benchmark_id, "benchmark_id")?;
    if case.benchmark_id != "MusicListenBench" {
        return Err("benchmark_id must be MusicListenBench".into());
    }
    require_text(&case.benchmark_version, "benchmark_version")?;
    validate_digest(&case.dataset_revision_sha256, "dataset_revision_sha256")?;
    validate_digest(
        &case.official_scorer_revision_sha256,
        "official_scorer_revision_sha256",
    )?;
    if case.benchmark_version != "1.0" {
        return Err("this adapter only admits the pinned MusicListenBench v1.0 protocol".into());
    }
    if case.triads.is_empty() {
        return Err("MusicListenBench case must contain at least one complete triad".into());
    }

    let mut item_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut lineage_ids = BTreeSet::new();
    let first_identity = &case.triads[0];
    for (index, triad) in case.triads.iter().enumerate().skip(1) {
        if triad.model_identity_sha256 != first_identity.model_identity_sha256
            || triad.system_prompt_sha256 != first_identity.system_prompt_sha256
            || triad.runtime_identity_sha256 != first_identity.runtime_identity_sha256
            || triad.decoding_config_sha256 != first_identity.decoding_config_sha256
        {
            return Err(format!(
                "triads[{index}] changes model/prompt/runtime/decoding identity within one evaluation run"
            ));
        }
    }
    let mut triad_results = Vec::with_capacity(case.triads.len());
    let mut scored = Vec::with_capacity(case.triads.len());

    for (index, triad) in case.triads.iter().enumerate() {
        validate_triad(triad, index)?;
        if !item_ids.insert(triad.item_id.as_str()) {
            return Err(format!("duplicate MusicListenBench item_id: {}", triad.item_id));
        }
        if !pair_ids.insert(triad.pair_id.as_str()) {
            return Err(format!("duplicate MusicListenBench pair_id: {}", triad.pair_id));
        }
        lineage_ids.insert(triad.source_lineage_sha256.as_str());

        let clean = score_variant(&triad.clean);
        let flip = score_variant(&triad.flip);
        let stay = score_variant(&triad.stay);
        let joint_exact = clean.correct && flip.correct && stay.correct;

        triad_results.push(MusicListenTriadResultV1 {
            item_id: triad.item_id.clone(),
            pair_id: triad.pair_id.clone(),
            source_lineage_sha256: triad.source_lineage_sha256.clone(),
            task: triad.task,
            clean_predicted_relation: clean.predicted,
            flip_predicted_relation: flip.predicted,
            stay_predicted_relation: stay.predicted,
            clean_correct: clean.correct,
            flip_correct: flip.correct,
            stay_correct: stay.correct,
            joint_exact,
        });
        scored.push(ScoredTriadV1 {
            clean,
            flip,
            stay,
            joint_exact,
        });
    }

    let overall = summarize(&scored);
    let mut by_task = Vec::new();
    for task in [
        MusicListenTaskV1::Melody,
        MusicListenTaskV1::Harmony,
        MusicListenTaskV1::Timbre,
        MusicListenTaskV1::Rhythm,
    ] {
        let indices: Vec<_> = case
            .triads
            .iter()
            .enumerate()
            .filter_map(|(index, triad)| (triad.task == task).then_some(index))
            .collect();
        if !indices.is_empty() {
            by_task.push(MusicListenTaskSliceV1 {
                task,
                metrics: summarize_indices(&scored, &indices),
            });
        }
    }

    let mut by_flip_factor = Vec::new();
    let mut flip_groups: BTreeMap<(MusicListenTaskV1, MusicListenFlipFactorV1), Vec<usize>> =
        BTreeMap::new();
    let mut by_stay_factor = Vec::new();
    let mut stay_groups: BTreeMap<(MusicListenTaskV1, MusicListenStayFactorV1), Vec<usize>> =
        BTreeMap::new();
    for (index, triad) in case.triads.iter().enumerate() {
        flip_groups
            .entry((triad.task, triad.flip_factor))
            .or_default()
            .push(index);
        stay_groups
            .entry((triad.task, triad.stay_factor))
            .or_default()
            .push(index);
    }
    for ((task, factor), indices) in flip_groups {
        let misses = indices
            .iter()
            .filter(|index| !scored[**index].flip.correct)
            .count();
        by_flip_factor.push(MusicListenFlipFactorSliceV1 {
            task,
            factor,
            items: indices.len(),
            miss_rate: rate(misses, indices.len()),
        });
    }
    for ((task, factor), indices) in stay_groups {
        let false_flips = indices
            .iter()
            .filter(|index| !scored[**index].stay.correct)
            .count();
        by_stay_factor.push(MusicListenStayFactorSliceV1 {
            task,
            factor,
            items: indices.len(),
            false_flip_rate: rate(false_flips, indices.len()),
        });
    }

    let lineage_count = lineage_ids.len();
    Ok(MusicListenBenchReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: EVALUATOR_VERSION_V1,
        benchmark_id: case.benchmark_id.clone(),
        benchmark_version: case.benchmark_version.clone(),
        dataset_revision_sha256: case.dataset_revision_sha256.clone(),
        official_scorer_revision_sha256: case.official_scorer_revision_sha256.clone(),
        split: case.split,
        overall,
        by_task,
        by_flip_factor,
        by_stay_factor,
        triad_results,
        source_lineage_count: lineage_count,
        recommended_confirmatory_lineage_floor: CONFIRMATORY_LINEAGE_FLOOR,
        confirmatory_lineage_floor_met: lineage_count >= CONFIRMATORY_LINEAGE_FLOOR,
        audio_bytes_recomputed: false,
        transformations_authenticated: false,
        model_invoked_by_evaluator: false,
    })
}

fn validate_triad(triad: &MusicListenTriadV1, index: usize) -> Result<(), String> {
    for (value, label) in [
        (triad.item_id.as_str(), "item_id"),
        (triad.pair_id.as_str(), "pair_id"),
    ] {
        require_text(value, &format!("triads[{index}].{label}"))?;
    }
    for (value, label) in [
        (&triad.source_lineage_sha256, "source_lineage_sha256"),
        (&triad.model_identity_sha256, "model_identity_sha256"),
        (&triad.system_prompt_sha256, "system_prompt_sha256"),
        (&triad.runtime_identity_sha256, "runtime_identity_sha256"),
        (&triad.decoding_config_sha256, "decoding_config_sha256"),
    ] {
        validate_digest(value, &format!("triads[{index}].{label}"))?;
    }

    if !flip_factor_compatible(triad.task, triad.flip_factor) {
        return Err(format!(
            "triads[{index}] FLIP factor {:?} is incompatible with task {:?}",
            triad.flip_factor, triad.task
        ));
    }
    if triad.stay_factor == MusicListenStayFactorV1::RelativePitchPreservingTransposition
        && !matches!(triad.task, MusicListenTaskV1::Melody | MusicListenTaskV1::Harmony)
    {
        return Err(format!(
            "triads[{index}] relative-pitch-preserving transposition is not an admitted STAY factor for {:?}",
            triad.task
        ));
    }

    let variants = [&triad.clean, &triad.flip, &triad.stay];
    let mut variant_ids = BTreeSet::new();
    let mut audio_digests = BTreeSet::new();
    for (variant_index, variant) in variants.iter().enumerate() {
        require_text(
            &variant.variant_id,
            &format!("triads[{index}].variants[{variant_index}].variant_id"),
        )?;
        require_text(
            &variant.question,
            &format!("triads[{index}].variants[{variant_index}].question"),
        )?;
        validate_digest(
            &variant.audio_sha256,
            &format!("triads[{index}].variants[{variant_index}].audio_sha256"),
        )?;
        validate_digest(
            &variant.question_sha256,
            &format!("triads[{index}].variants[{variant_index}].question_sha256"),
        )?;
        let question_digest = sha256_hex(variant.question.as_bytes());
        if variant.question_sha256 != question_digest {
            return Err(format!(
                "triads[{index}] variant {} question digest does not match exact UTF-8 question bytes",
                variant.variant_id
            ));
        }
        if !variant_ids.insert(variant.variant_id.as_str()) {
            return Err(format!("triads[{index}] repeats a variant_id"));
        }
        if !audio_digests.insert(variant.audio_sha256.as_str()) {
            return Err(format!("triads[{index}] variants must use distinct audio artifacts"));
        }
        if variant.option_a_relation == variant.option_b_relation {
            return Err(format!(
                "triads[{index}] variant {} has non-contrasting A/B semantic options",
                variant.variant_id
            ));
        }
        if ![variant.option_a_relation, variant.option_b_relation]
            .contains(&variant.expected_relation)
        {
            return Err(format!(
                "triads[{index}] variant {} expected relation is not represented by either option",
                variant.variant_id
            ));
        }
    }

    if triad.clean.expected_relation != triad.stay.expected_relation {
        return Err(format!(
            "triads[{index}] CLEAN and STAY expected musical relations must match"
        ));
    }
    if triad.flip.expected_relation == triad.clean.expected_relation {
        return Err(format!(
            "triads[{index}] FLIP expected musical relation must oppose CLEAN"
        ));
    }
    Ok(())
}

fn flip_factor_compatible(
    task: MusicListenTaskV1,
    factor: MusicListenFlipFactorV1,
) -> bool {
    matches!(
        (task, factor),
        (MusicListenTaskV1::Melody, MusicListenFlipFactorV1::NoteIdentity)
            | (MusicListenTaskV1::Harmony, MusicListenFlipFactorV1::ChordThirdQuality)
            | (MusicListenTaskV1::Timbre, MusicListenFlipFactorV1::InstrumentFamily)
            | (MusicListenTaskV1::Rhythm, MusicListenFlipFactorV1::PulseRate)
    )
}

fn score_variant(variant: &MusicListenVariantV1) -> ScoredVariantV1 {
    let predicted = match variant.model_answer {
        MusicListenAnswerV1::ChoiceA => Some(variant.option_a_relation),
        MusicListenAnswerV1::ChoiceB => Some(variant.option_b_relation),
        MusicListenAnswerV1::Unparseable | MusicListenAnswerV1::Abstain => None,
    };
    ScoredVariantV1 {
        predicted,
        correct: predicted == Some(variant.expected_relation),
        answered: predicted.is_some(),
        choice_a: variant.model_answer == MusicListenAnswerV1::ChoiceA,
        choice_b: variant.model_answer == MusicListenAnswerV1::ChoiceB,
        unparseable: variant.model_answer == MusicListenAnswerV1::Unparseable,
        abstained: variant.model_answer == MusicListenAnswerV1::Abstain,
    }
}

fn summarize(scored: &[ScoredTriadV1]) -> PairedMetricsV1 {
    let indices: Vec<_> = (0..scored.len()).collect();
    summarize_indices(scored, &indices)
}

fn summarize_indices(scored: &[ScoredTriadV1], indices: &[usize]) -> PairedMetricsV1 {
    let clean = variant_metrics(indices.iter().map(|index| scored[*index].clean));
    let flip = variant_metrics(indices.iter().map(|index| scored[*index].flip));
    let stay = variant_metrics(indices.iter().map(|index| scored[*index].stay));
    let clean_correct_indices: Vec<_> = indices
        .iter()
        .filter(|index| scored[**index].clean.correct)
        .copied()
        .collect();
    let conditional_flip_errors = clean_correct_indices
        .iter()
        .filter(|index| !scored[**index].flip.correct)
        .count();
    let conditional_stay_errors = clean_correct_indices
        .iter()
        .filter(|index| !scored[**index].stay.correct)
        .count();
    let triads = indices.len();
    let joint_correct = indices
        .iter()
        .filter(|index| scored[**index].joint_exact)
        .count();

    PairedMetricsV1 {
        triads,
        flip_miss_rate: rate(flip.incorrect, flip.items),
        stay_false_flip_rate: rate(stay.incorrect, stay.items),
        clean_correct_triads: clean_correct_indices.len(),
        conditional_flip_miss_rate: rate(
            conditional_flip_errors,
            clean_correct_indices.len(),
        ),
        conditional_stay_false_flip_rate: rate(
            conditional_stay_errors,
            clean_correct_indices.len(),
        ),
        accuracy_flip_stay: match (
            rate(flip.incorrect, flip.items),
            rate(stay.incorrect, stay.items),
        ) {
            (Some(miss_rate), Some(false_flip_rate)) => {
                Some(1.0 - (miss_rate + false_flip_rate) / 2.0)
            }
            _ => None,
        },
        joint_clean_flip_stay_exact_rate: rate(joint_correct, triads),
        clean,
        flip,
        stay,
    }
}

fn variant_metrics(
    values: impl Iterator<Item = ScoredVariantV1>,
) -> VariantMetricV1 {
    let values: Vec<_> = values.collect();
    let items = values.len();
    let correct = values.iter().filter(|value| value.correct).count();
    let answered = values.iter().filter(|value| value.answered).count();
    let choice_a = values.iter().filter(|value| value.choice_a).count();
    let choice_b = values.iter().filter(|value| value.choice_b).count();
    let unparseable = values.iter().filter(|value| value.unparseable).count();
    let abstained = values.iter().filter(|value| value.abstained).count();
    VariantMetricV1 {
        items,
        correct,
        incorrect: items - correct,
        answered,
        choice_a,
        choice_b,
        choice_a_rate_among_answered: rate(choice_a, answered),
        unparseable,
        abstained,
        accuracy: rate(correct, items),
        coverage: rate(answered, items),
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

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINEAGE: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PROMPT: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const RUNTIME: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DECODING: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const DATASET: &str = "9999999999999999999999999999999999999999999999999999999999999999";
    const OFFICIAL_SCORER: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn variant(
        variant_id: &str,
        audio_sha256: &str,
        question: &str,
        option_a_relation: MusicalRelationV1,
        option_b_relation: MusicalRelationV1,
        expected_relation: MusicalRelationV1,
        model_answer: MusicListenAnswerV1,
    ) -> MusicListenVariantV1 {
        MusicListenVariantV1 {
            variant_id: variant_id.into(),
            audio_sha256: audio_sha256.into(),
            question: question.into(),
            question_sha256: sha256_hex(question.as_bytes()),
            option_a_relation,
            option_b_relation,
            expected_relation,
            model_answer,
        }
    }

    fn triad(
        task: MusicListenTaskV1,
        flip_factor: MusicListenFlipFactorV1,
        stay_factor: MusicListenStayFactorV1,
    ) -> MusicListenTriadV1 {
        triad_with_answers(
            task,
            flip_factor,
            stay_factor,
            MusicListenAnswerV1::ChoiceA,
            MusicListenAnswerV1::ChoiceB,
            MusicListenAnswerV1::ChoiceA,
        )
    }

    fn triad_with_answers(
        task: MusicListenTaskV1,
        flip_factor: MusicListenFlipFactorV1,
        stay_factor: MusicListenStayFactorV1,
        clean_answer: MusicListenAnswerV1,
        flip_answer: MusicListenAnswerV1,
        stay_answer: MusicListenAnswerV1,
    ) -> MusicListenTriadV1 {
        let (clean_expected, flip_expected, flip_choice_a, flip_choice_b) = (
            MusicalRelationV1::Same,
            MusicalRelationV1::Different,
            MusicalRelationV1::Same,
            MusicalRelationV1::Different,
        );
        MusicListenTriadV1 {
            item_id: format!("item-{task:?}"),
            pair_id: format!("pair-{task:?}"),
            source_lineage_sha256: LINEAGE.into(),
            task,
            flip_factor,
            stay_factor,
            model_identity_sha256: MODEL.into(),
            system_prompt_sha256: PROMPT.into(),
            runtime_identity_sha256: RUNTIME.into(),
            decoding_config_sha256: DECODING.into(),
            clean: variant(
                "clean", "1111111111111111111111111111111111111111111111111111111111111111",
                "Do these clips have the same musical property?",
                MusicalRelationV1::Same, MusicalRelationV1::Different, clean_expected,
                clean_answer,
            ),
            flip: variant(
                "flip", "2222222222222222222222222222222222222222222222222222222222222222",
                "Did the queried musical property change?",
                flip_choice_a, flip_choice_b, flip_expected, flip_answer,
            ),
            stay: variant(
                "stay", "3333333333333333333333333333333333333333333333333333333333333333",
                "Does the queried musical property remain the same?",
                MusicalRelationV1::Same, MusicalRelationV1::Different, clean_expected,
                stay_answer,
            ),
        }
    }

    fn case(triads: Vec<MusicListenTriadV1>) -> MusicListenBenchCaseV1 {
        MusicListenBenchCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            benchmark_id: "MusicListenBench".into(),
            benchmark_version: "1.0".into(),
            dataset_revision_sha256: DATASET.into(),
            official_scorer_revision_sha256: OFFICIAL_SCORER.into(),
            split: MusicListenSplitV1::RenderedTest,
            triads,
        }
    }

    #[test]
    fn correct_semantic_answers_score_perfect_even_when_flip_letter_mapping_reverses() {
        let mut input = case(vec![triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Hiss,
        )]);
        input.triads[0].flip.option_a_relation = MusicalRelationV1::Different;
        input.triads[0].flip.option_b_relation = MusicalRelationV1::Same;
        input.triads[0].flip.model_answer = MusicListenAnswerV1::ChoiceA;
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(1.0));
        assert_eq!(report.overall.flip.accuracy, Some(1.0));
        assert_eq!(report.overall.stay.accuracy, Some(1.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(1.0));
        assert_eq!(report.overall.joint_clean_flip_stay_exact_rate, Some(1.0));
    }

    #[test]
    fn fixed_a_letter_is_not_mistaken_for_a_musically_correct_response() {
        let mut input = case(vec![triad(
            MusicListenTaskV1::Harmony,
            MusicListenFlipFactorV1::ChordThirdQuality,
            MusicListenStayFactorV1::Room,
        )]);
        input.triads[0].flip.model_answer = MusicListenAnswerV1::ChoiceA;
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(1.0));
        assert_eq!(report.overall.flip.accuracy, Some(0.0));
        assert_eq!(report.overall.stay.accuracy, Some(1.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(0.5));
    }

    #[test]
    fn wrong_clean_answer_is_excluded_from_clean_correct_conditional_denominator() {
        let input = case(vec![triad_with_answers(
            MusicListenTaskV1::Timbre,
            MusicListenFlipFactorV1::InstrumentFamily,
            MusicListenStayFactorV1::Eq,
            MusicListenAnswerV1::ChoiceB,
            MusicListenAnswerV1::ChoiceA,
            MusicListenAnswerV1::ChoiceB,
        )]);
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.overall.clean_correct_triads, 0);
        assert_eq!(report.overall.conditional_flip_miss_rate, None);
        assert_eq!(report.overall.conditional_stay_false_flip_rate, None);
    }

    #[test]
    fn abstain_and_unparseable_are_errors_and_reduce_coverage() {
        let input = case(vec![triad_with_answers(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Hiss,
            MusicListenAnswerV1::Abstain,
            MusicListenAnswerV1::Unparseable,
            MusicListenAnswerV1::ChoiceA,
        )]);
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.overall.clean.incorrect, 1);
        assert_eq!(report.overall.clean.abstained, 1);
        assert_eq!(report.overall.clean.coverage, Some(0.0));
        assert_eq!(report.overall.flip.unparseable, 1);
        assert_eq!(report.overall.flip.coverage, Some(0.0));
        assert_eq!(report.overall.accuracy_flip_stay, Some(0.5));
    }

    #[test]
    fn mixed_model_runtime_or_prompt_identities_cannot_be_aggregated() {
        let mut input = case(vec![
            triad(
                MusicListenTaskV1::Melody,
                MusicListenFlipFactorV1::NoteIdentity,
                MusicListenStayFactorV1::Room,
            ),
            triad(
                MusicListenTaskV1::Harmony,
                MusicListenFlipFactorV1::ChordThirdQuality,
                MusicListenStayFactorV1::Hiss,
            ),
        ]);
        input.triads[1].runtime_identity_sha256 =
            "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn report_exposes_letter_bias_without_using_letters_as_semantics() {
        let input = case(vec![triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Hiss,
        )]);
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.overall.clean.choice_a_rate_among_answered, Some(1.0));
        assert_eq!(report.overall.flip.choice_a_rate_among_answered, Some(0.0));
        assert_eq!(report.overall.stay.choice_a_rate_among_answered, Some(1.0));
    }

    #[test]
    fn incompatible_flip_factor_is_rejected() {
        let input = case(vec![triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::ChordThirdQuality,
            MusicListenStayFactorV1::Hiss,
        )]);
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn relative_pitch_transposition_is_not_a_timbre_stay_control() {
        let input = case(vec![triad(
            MusicListenTaskV1::Timbre,
            MusicListenFlipFactorV1::InstrumentFamily,
            MusicListenStayFactorV1::RelativePitchPreservingTransposition,
        )]);
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn clean_stay_flip_semantic_relation_contract_is_enforced() {
        let mut input = case(vec![triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Hiss,
        )]);
        input.triads[0].stay.expected_relation = MusicalRelationV1::Different;
        assert!(audit_music_listen_case(&input).is_err());
        input.triads[0].stay.expected_relation = MusicalRelationV1::Same;
        input.triads[0].flip.expected_relation = MusicalRelationV1::Same;
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn duplicated_artifacts_and_question_digest_substitution_are_rejected() {
        let mut input = case(vec![triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Room,
        )]);
        input.triads[0].stay.audio_sha256 = input.triads[0].clean.audio_sha256.clone();
        assert!(audit_music_listen_case(&input).is_err());
        input.triads[0].stay.audio_sha256 =
            "3333333333333333333333333333333333333333333333333333333333333333".into();
        input.triads[0].clean.question_sha256 =
            "9999999999999999999999999999999999999999999999999999999999999999".into();
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn duplicate_item_and_pair_ids_are_rejected() {
        let mut first = triad(
            MusicListenTaskV1::Melody,
            MusicListenFlipFactorV1::NoteIdentity,
            MusicListenStayFactorV1::Room,
        );
        let mut second = triad(
            MusicListenTaskV1::Harmony,
            MusicListenFlipFactorV1::ChordThirdQuality,
            MusicListenStayFactorV1::Hiss,
        );
        second.item_id = first.item_id.clone();
        let mut input = case(vec![first.clone(), second]);
        assert!(audit_music_listen_case(&input).is_err());
        first.item_id = "item-one".into();
        first.pair_id = "pair-one".into();
        let mut second = triad(
            MusicListenTaskV1::Harmony,
            MusicListenFlipFactorV1::ChordThirdQuality,
            MusicListenStayFactorV1::Hiss,
        );
        second.pair_id = "pair-one".into();
        input = case(vec![first, second]);
        assert!(audit_music_listen_case(&input).is_err());
    }

    #[test]
    fn task_and_factor_slices_retain_exact_denominators() {
        let input = case(vec![
            triad(
                MusicListenTaskV1::Melody,
                MusicListenFlipFactorV1::NoteIdentity,
                MusicListenStayFactorV1::Hiss,
            ),
            triad(
                MusicListenTaskV1::Harmony,
                MusicListenFlipFactorV1::ChordThirdQuality,
                MusicListenStayFactorV1::Room,
            ),
        ]);
        let report = audit_music_listen_case(&input).unwrap();
        assert_eq!(report.by_task.len(), 2);
        assert_eq!(report.by_flip_factor.len(), 2);
        assert_eq!(report.by_stay_factor.len(), 2);
        assert_eq!(
            report.by_task.iter().map(|slice| slice.metrics.triads).sum::<usize>(),
            report.overall.triads
        );
        assert_eq!(
            report.confirmatory_lineage_floor_met,
            false
        );
    }
}
