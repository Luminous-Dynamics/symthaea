//! Property-scoped musical identity across genuinely different performances.
//!
//! This extension consumes supplied answers and evidence metadata. It reuses
//! MusicListenBench's semantic A/B normalization, but leaves the official
//! MusicListenBench v1.0 triad schema and score untouched. It distinguishes
//! performance-preserving variation from deliberate changes to the queried
//! musical property. It does not authenticate audio bytes, performers, takes,
//! or human annotations.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::music_listen_bench::{
    MusicListenAnswerV1, MusicListenVariantV1, MusicalRelationV1, predicted_relation_for_variant,
    sha256_hex,
};

const CASE_SCHEMA_V1: &str = "mel-performance-invariance-case-v1";
const REPORT_SCHEMA_V1: &str = "mel-performance-invariance-report-v1";
const EVALUATOR_VERSION_V1: &str = "mel-performance-invariance-evaluator-v1";
const CONFIRMATORY_LINEAGE_FLOOR: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerformanceInvarianceSplitV1 {
    SyntheticRendered,
    PublicRealRecordingEvaluationOnly,
    MusicianRecordedConfirmatory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalTargetPropertyV1 {
    ChordQuality,
    RootPitchClass,
    ChordVoicing,
    RelativeIntervalSequence,
    MelodicContour,
    RhythmicRatioPattern,
    PulseRate,
    InstrumentFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerformanceVariationFactorV1 {
    DifferentPerformer,
    DifferentTake,
    StrumOrArticulation,
    VelocityOrDynamics,
    ExpressiveRubato,
    GlobalTransposition,
    ChordInversion,
    ChordVoicing,
    UniformTempoScale,
    CrossRenderer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalChangeFactorV1 {
    ChordQuality,
    RootPitchClass,
    ChordVoicing,
    MelodicInterval,
    MelodicContour,
    RhythmicRatio,
    PulseRate,
    InstrumentFamily,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceVariantV1 {
    pub performance_condition_sha256: String,
    pub answer: MusicListenVariantV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceTriadV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_lineage_sha256: String,
    pub composition_sha256: String,
    pub target_property: MusicalTargetPropertyV1,
    pub performance_stay_factor: PerformanceVariationFactorV1,
    pub musical_flip_factor: MusicalChangeFactorV1,
    pub model_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub clean: PerformanceVariantV1,
    pub performance_stay: PerformanceVariantV1,
    pub musical_flip: PerformanceVariantV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceInvarianceCaseV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub split: PerformanceInvarianceSplitV1,
    pub dataset_revision_sha256: String,
    pub annotation_policy_sha256: String,
    /// Digest of the rights/eligibility snapshot consumed from the owning registry.
    pub source_permissions_manifest_sha256: String,
    pub triads: Vec<PerformanceTriadV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VariantMetricV1 {
    pub items: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub answered: usize,
    pub choice_a: usize,
    pub choice_b: usize,
    pub unparseable: usize,
    pub abstained: usize,
    pub accuracy: Option<f64>,
    pub coverage: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TargetPropertyMetricsV1 {
    /// None for the mixed-target descriptive overall summary.
    pub target_property: Option<MusicalTargetPropertyV1>,
    pub triads: usize,
    pub clean: VariantMetricV1,
    pub performance_stay: VariantMetricV1,
    pub musical_flip: VariantMetricV1,
    pub performance_false_difference_rate: Option<f64>,
    pub musical_flip_miss_rate: Option<f64>,
    pub clean_correct_triads: usize,
    pub conditional_performance_false_difference_rate: Option<f64>,
    pub conditional_musical_flip_miss_rate: Option<f64>,
    pub joint_exact_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceFactorMetricsV1 {
    pub target_property: MusicalTargetPropertyV1,
    pub factor: PerformanceVariationFactorV1,
    pub items: usize,
    pub false_difference_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalFlipFactorMetricsV1 {
    pub target_property: MusicalTargetPropertyV1,
    pub factor: MusicalChangeFactorV1,
    pub items: usize,
    pub miss_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceTriadResultV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_lineage_sha256: String,
    pub target_property: MusicalTargetPropertyV1,
    pub performance_stay_factor: PerformanceVariationFactorV1,
    pub musical_flip_factor: MusicalChangeFactorV1,
    pub clean_predicted_relation: Option<MusicalRelationV1>,
    pub performance_stay_predicted_relation: Option<MusicalRelationV1>,
    pub musical_flip_predicted_relation: Option<MusicalRelationV1>,
    pub clean_correct: bool,
    pub performance_stay_correct: bool,
    pub musical_flip_correct: bool,
    pub joint_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceInvarianceReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub split: PerformanceInvarianceSplitV1,
    pub dataset_revision_sha256: String,
    pub annotation_policy_sha256: String,
    pub source_permissions_manifest_sha256: String,
    pub overall: TargetPropertyMetricsV1,
    pub by_target_property: Vec<TargetPropertyMetricsV1>,
    pub by_performance_factor: Vec<PerformanceFactorMetricsV1>,
    pub by_musical_flip_factor: Vec<MusicalFlipFactorMetricsV1>,
    pub triad_results: Vec<PerformanceTriadResultV1>,
    pub source_lineage_count: usize,
    pub recommended_confirmatory_lineage_floor: usize,
    pub confirmatory_lineage_floor_met: bool,
    /// The input hashes are self-reported; this adapter does not open the audio files.
    pub audio_bytes_recomputed: bool,
    /// Performance/take and annotation references are metadata, not authenticated receipts.
    pub performance_and_annotation_receipts_authenticated: bool,
    /// The rights/eligibility manifest is a digest reference, not an authenticated receipt here.
    pub source_permission_receipts_authenticated: bool,
    /// No model is invoked by this deterministic answer-scoring adapter.
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
    performance_stay: ScoredVariantV1,
    musical_flip: ScoredVariantV1,
    joint_exact: bool,
}

pub fn audit_performance_invariance_case_json(
    input: &str,
) -> Result<PerformanceInvarianceReportV1, String> {
    let case: PerformanceInvarianceCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid performance-invariance case JSON: {error}"))?;
    audit_performance_invariance_case(&case)
}

pub fn audit_performance_invariance_case(
    case: &PerformanceInvarianceCaseV1,
) -> Result<PerformanceInvarianceReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!(
            "unsupported performance-invariance schema: {}",
            case.schema
        ));
    }
    require_text(&case.benchmark_id, "benchmark_id")?;
    require_text(&case.benchmark_version, "benchmark_version")?;
    if case.benchmark_id != "Melothaea-Performance-Identity" {
        return Err("benchmark_id must be Melothaea-Performance-Identity".into());
    }
    if case.benchmark_version != "1.0" {
        return Err("this evaluator only admits performance-invariance benchmark v1.0".into());
    }
    validate_digest(&case.dataset_revision_sha256, "dataset_revision_sha256")?;
    validate_digest(&case.annotation_policy_sha256, "annotation_policy_sha256")?;
    validate_digest(
        &case.source_permissions_manifest_sha256,
        "source_permissions_manifest_sha256",
    )?;
    if case.triads.is_empty() {
        return Err("performance-invariance case must contain at least one complete triad".into());
    }

    let first = &case.triads[0];
    let mut item_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut source_lineages = BTreeSet::new();
    let mut variant_ids = BTreeSet::new();
    let mut composition_lineages = BTreeMap::<&str, &str>::new();
    let mut audio_lineages = BTreeMap::<&str, &str>::new();
    let mut scored = Vec::with_capacity(case.triads.len());
    let mut results = Vec::with_capacity(case.triads.len());

    for (index, triad) in case.triads.iter().enumerate() {
        validate_triad(triad, index)?;
        for variant_id in [
            triad.clean.answer.variant_id.as_str(),
            triad.performance_stay.answer.variant_id.as_str(),
            triad.musical_flip.answer.variant_id.as_str(),
        ] {
            if !variant_ids.insert(variant_id) {
                return Err(format!(
                    "triads[{index}] repeats a variant ID across the evaluation run"
                ));
            }
        }
        if triad.model_identity_sha256 != first.model_identity_sha256
            || triad.system_prompt_sha256 != first.system_prompt_sha256
            || triad.runtime_identity_sha256 != first.runtime_identity_sha256
            || triad.decoding_config_sha256 != first.decoding_config_sha256
        {
            return Err(format!(
                "triads[{index}] changes model/prompt/runtime/decoding identity within one run"
            ));
        }
        if !item_ids.insert(triad.item_id.as_str()) {
            return Err(format!("duplicate item_id in triads[{index}]"));
        }
        if !pair_ids.insert(triad.pair_id.as_str()) {
            return Err(format!("duplicate pair_id in triads[{index}]"));
        }
        let lineage = triad.source_lineage_sha256.as_str();
        let composition = triad.composition_sha256.as_str();
        source_lineages.insert(lineage);
        if let Some(existing_lineage) = composition_lineages.insert(composition, lineage)
            && existing_lineage != lineage
        {
            return Err(format!(
                "triads[{index}] reuses a composition digest across different source-work lineages"
            ));
        }
        for variant in [
            &triad.clean.answer,
            &triad.performance_stay.answer,
            &triad.musical_flip.answer,
        ] {
            if let Some(existing_lineage) =
                audio_lineages.insert(variant.audio_sha256.as_str(), lineage)
                && existing_lineage != lineage
            {
                return Err(format!(
                    "triads[{index}] reuses an audio digest across different source-work lineages"
                ));
            }
        }

        let clean = score_variant(&triad.clean.answer);
        let performance_stay = score_variant(&triad.performance_stay.answer);
        let musical_flip = score_variant(&triad.musical_flip.answer);
        let joint_exact = clean.correct && performance_stay.correct && musical_flip.correct;
        scored.push(ScoredTriadV1 {
            clean,
            performance_stay,
            musical_flip,
            joint_exact,
        });
        results.push(PerformanceTriadResultV1 {
            item_id: triad.item_id.clone(),
            pair_id: triad.pair_id.clone(),
            source_lineage_sha256: triad.source_lineage_sha256.clone(),
            target_property: triad.target_property,
            performance_stay_factor: triad.performance_stay_factor,
            musical_flip_factor: triad.musical_flip_factor,
            clean_predicted_relation: clean.predicted,
            performance_stay_predicted_relation: performance_stay.predicted,
            musical_flip_predicted_relation: musical_flip.predicted,
            clean_correct: clean.correct,
            performance_stay_correct: performance_stay.correct,
            musical_flip_correct: musical_flip.correct,
            joint_exact,
        });
    }

    let overall = summarize_indices(None, &scored, &(0..scored.len()).collect::<Vec<_>>());
    let mut by_target_property = Vec::new();
    for target_property in target_properties() {
        let indices: Vec<_> = case
            .triads
            .iter()
            .enumerate()
            .filter_map(|(index, triad)| {
                (triad.target_property == target_property).then_some(index)
            })
            .collect();
        if !indices.is_empty() {
            by_target_property.push(summarize_indices(Some(target_property), &scored, &indices));
        }
    }

    let mut by_performance_factor = Vec::new();
    for target_property in target_properties() {
        for factor in performance_factors() {
            let indices: Vec<_> = case
                .triads
                .iter()
                .enumerate()
                .filter_map(|(index, triad)| {
                    (triad.target_property == target_property
                        && triad.performance_stay_factor == factor)
                        .then_some(index)
                })
                .collect();
            if !indices.is_empty() {
                let metric =
                    variant_metrics(indices.iter().map(|index| scored[*index].performance_stay));
                by_performance_factor.push(PerformanceFactorMetricsV1 {
                    target_property,
                    factor,
                    items: metric.items,
                    false_difference_rate: rate(metric.incorrect, metric.items),
                });
            }
        }
    }

    let mut by_musical_flip_factor = Vec::new();
    for target_property in target_properties() {
        for factor in musical_flip_factors() {
            let indices: Vec<_> = case
                .triads
                .iter()
                .enumerate()
                .filter_map(|(index, triad)| {
                    (triad.target_property == target_property
                        && triad.musical_flip_factor == factor)
                        .then_some(index)
                })
                .collect();
            if !indices.is_empty() {
                let metric =
                    variant_metrics(indices.iter().map(|index| scored[*index].musical_flip));
                by_musical_flip_factor.push(MusicalFlipFactorMetricsV1 {
                    target_property,
                    factor,
                    items: metric.items,
                    miss_rate: rate(metric.incorrect, metric.items),
                });
            }
        }
    }

    Ok(PerformanceInvarianceReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: EVALUATOR_VERSION_V1,
        benchmark_id: case.benchmark_id.clone(),
        benchmark_version: case.benchmark_version.clone(),
        split: case.split,
        dataset_revision_sha256: case.dataset_revision_sha256.clone(),
        annotation_policy_sha256: case.annotation_policy_sha256.clone(),
        source_permissions_manifest_sha256: case.source_permissions_manifest_sha256.clone(),
        overall,
        by_target_property,
        by_performance_factor,
        by_musical_flip_factor,
        triad_results: results,
        source_lineage_count: source_lineages.len(),
        recommended_confirmatory_lineage_floor: CONFIRMATORY_LINEAGE_FLOOR,
        confirmatory_lineage_floor_met: source_lineages.len() >= CONFIRMATORY_LINEAGE_FLOOR,
        audio_bytes_recomputed: false,
        performance_and_annotation_receipts_authenticated: false,
        source_permission_receipts_authenticated: false,
        model_invoked_by_evaluator: false,
    })
}

fn validate_triad(triad: &PerformanceTriadV1, index: usize) -> Result<(), String> {
    for (value, label) in [
        (triad.item_id.as_str(), "item_id"),
        (triad.pair_id.as_str(), "pair_id"),
    ] {
        require_text(value, &format!("triads[{index}].{label}"))?;
    }
    for (value, label) in [
        (&triad.source_lineage_sha256, "source_lineage_sha256"),
        (&triad.composition_sha256, "composition_sha256"),
        (&triad.model_identity_sha256, "model_identity_sha256"),
        (&triad.system_prompt_sha256, "system_prompt_sha256"),
        (&triad.runtime_identity_sha256, "runtime_identity_sha256"),
        (&triad.decoding_config_sha256, "decoding_config_sha256"),
        (
            &triad.clean.performance_condition_sha256,
            "clean.performance_condition_sha256",
        ),
        (
            &triad.performance_stay.performance_condition_sha256,
            "performance_stay.performance_condition_sha256",
        ),
        (
            &triad.musical_flip.performance_condition_sha256,
            "musical_flip.performance_condition_sha256",
        ),
    ] {
        validate_digest(value, &format!("triads[{index}].{label}"))?;
    }

    if triad.clean.performance_condition_sha256 != triad.musical_flip.performance_condition_sha256 {
        return Err(format!(
            "triads[{index}] musical FLIP must hold the performance condition constant"
        ));
    }
    if triad.clean.performance_condition_sha256
        == triad.performance_stay.performance_condition_sha256
    {
        return Err(format!(
            "triads[{index}] performance STAY must bind a distinct performance condition"
        ));
    }
    if !performance_stay_factor_compatible(triad.target_property, triad.performance_stay_factor) {
        return Err(format!(
            "triads[{index}] performance-STAY factor {:?} is not valid for target property {:?}",
            triad.performance_stay_factor, triad.target_property
        ));
    }
    if !musical_flip_factor_compatible(triad.target_property, triad.musical_flip_factor) {
        return Err(format!(
            "triads[{index}] musical-FLIP factor {:?} is not valid for target property {:?}",
            triad.musical_flip_factor, triad.target_property
        ));
    }

    let variants = [
        ("clean", &triad.clean.answer, MusicalRelationV1::Same),
        (
            "performance_stay",
            &triad.performance_stay.answer,
            MusicalRelationV1::Same,
        ),
        (
            "musical_flip",
            &triad.musical_flip.answer,
            MusicalRelationV1::Different,
        ),
    ];
    let mut variant_ids = BTreeSet::new();
    let mut audio_digests = BTreeSet::new();
    let canonical_question = triad.clean.answer.question.as_str();
    for (variant_name, variant, expected_relation) in variants {
        require_text(
            &variant.variant_id,
            &format!("triads[{index}].{variant_name}.variant_id"),
        )?;
        require_text(
            &variant.question,
            &format!("triads[{index}].{variant_name}.question"),
        )?;
        validate_digest(
            &variant.audio_sha256,
            &format!("triads[{index}].{variant_name}.audio_sha256"),
        )?;
        validate_digest(
            &variant.question_sha256,
            &format!("triads[{index}].{variant_name}.question_sha256"),
        )?;
        if sha256_hex(variant.question.as_bytes()) != variant.question_sha256 {
            return Err(format!(
                "triads[{index}] {variant_name} question digest does not match its UTF-8 bytes"
            ));
        }
        if variant.option_a_relation == variant.option_b_relation {
            return Err(format!(
                "triads[{index}] {variant_name} options must encode distinct semantic relations"
            ));
        }
        if variant.expected_relation != expected_relation {
            return Err(format!(
                "triads[{index}] {variant_name} expected relation must be {:?}",
                expected_relation
            ));
        }
        if !variant_ids.insert(variant.variant_id.as_str()) {
            return Err(format!("triads[{index}] repeats a variant ID"));
        }
        if !audio_digests.insert(variant.audio_sha256.as_str()) {
            return Err(format!(
                "triads[{index}] reuses audio bytes across variants"
            ));
        }
        if variant.question != canonical_question {
            return Err(format!(
                "triads[{index}] {variant_name} changes the target-property question within one matched triad"
            ));
        }
    }
    Ok(())
}

fn performance_stay_factor_compatible(
    target: MusicalTargetPropertyV1,
    factor: PerformanceVariationFactorV1,
) -> bool {
    use MusicalTargetPropertyV1 as Target;
    use PerformanceVariationFactorV1 as Factor;

    match target {
        Target::ChordQuality => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::ExpressiveRubato
                | Factor::GlobalTransposition
                | Factor::ChordInversion
                | Factor::ChordVoicing
                | Factor::UniformTempoScale
                | Factor::CrossRenderer
        ),
        Target::RootPitchClass => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::ExpressiveRubato
                | Factor::ChordInversion
                | Factor::ChordVoicing
                | Factor::UniformTempoScale
                | Factor::CrossRenderer
        ),
        Target::ChordVoicing => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::ExpressiveRubato
                | Factor::UniformTempoScale
                | Factor::CrossRenderer
        ),
        Target::RelativeIntervalSequence | Target::MelodicContour => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::ExpressiveRubato
                | Factor::GlobalTransposition
                | Factor::UniformTempoScale
                | Factor::CrossRenderer
        ),
        Target::RhythmicRatioPattern => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::UniformTempoScale
                | Factor::CrossRenderer
        ),
        Target::PulseRate => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::CrossRenderer
        ),
        Target::InstrumentFamily => matches!(
            factor,
            Factor::DifferentPerformer
                | Factor::DifferentTake
                | Factor::StrumOrArticulation
                | Factor::VelocityOrDynamics
                | Factor::ExpressiveRubato
                | Factor::GlobalTransposition
                | Factor::ChordInversion
                | Factor::ChordVoicing
                | Factor::UniformTempoScale
        ),
    }
}

fn musical_flip_factor_compatible(
    target: MusicalTargetPropertyV1,
    factor: MusicalChangeFactorV1,
) -> bool {
    matches!(
        (target, factor),
        (
            MusicalTargetPropertyV1::ChordQuality,
            MusicalChangeFactorV1::ChordQuality
        ) | (
            MusicalTargetPropertyV1::RootPitchClass,
            MusicalChangeFactorV1::RootPitchClass
        ) | (
            MusicalTargetPropertyV1::ChordVoicing,
            MusicalChangeFactorV1::ChordVoicing
        ) | (
            MusicalTargetPropertyV1::RelativeIntervalSequence,
            MusicalChangeFactorV1::MelodicInterval
        ) | (
            MusicalTargetPropertyV1::MelodicContour,
            MusicalChangeFactorV1::MelodicContour
        ) | (
            MusicalTargetPropertyV1::RhythmicRatioPattern,
            MusicalChangeFactorV1::RhythmicRatio
        ) | (
            MusicalTargetPropertyV1::PulseRate,
            MusicalChangeFactorV1::PulseRate
        ) | (
            MusicalTargetPropertyV1::InstrumentFamily,
            MusicalChangeFactorV1::InstrumentFamily
        )
    )
}

fn score_variant(variant: &MusicListenVariantV1) -> ScoredVariantV1 {
    let predicted = predicted_relation_for_variant(variant);
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

fn summarize_indices(
    target_property: Option<MusicalTargetPropertyV1>,
    scored: &[ScoredTriadV1],
    indices: &[usize],
) -> TargetPropertyMetricsV1 {
    let clean = variant_metrics(indices.iter().map(|index| scored[*index].clean));
    let performance_stay =
        variant_metrics(indices.iter().map(|index| scored[*index].performance_stay));
    let musical_flip = variant_metrics(indices.iter().map(|index| scored[*index].musical_flip));
    let clean_correct_indices: Vec<_> = indices
        .iter()
        .filter(|index| scored[**index].clean.correct)
        .copied()
        .collect();
    let conditional_performance_errors = clean_correct_indices
        .iter()
        .filter(|index| !scored[**index].performance_stay.correct)
        .count();
    let conditional_flip_errors = clean_correct_indices
        .iter()
        .filter(|index| !scored[**index].musical_flip.correct)
        .count();
    let joint_exact = indices
        .iter()
        .filter(|index| scored[**index].joint_exact)
        .count();

    TargetPropertyMetricsV1 {
        target_property,
        triads: indices.len(),
        performance_false_difference_rate: rate(performance_stay.incorrect, performance_stay.items),
        musical_flip_miss_rate: rate(musical_flip.incorrect, musical_flip.items),
        clean_correct_triads: clean_correct_indices.len(),
        conditional_performance_false_difference_rate: rate(
            conditional_performance_errors,
            clean_correct_indices.len(),
        ),
        conditional_musical_flip_miss_rate: rate(
            conditional_flip_errors,
            clean_correct_indices.len(),
        ),
        joint_exact_rate: rate(joint_exact, indices.len()),
        clean,
        performance_stay,
        musical_flip,
    }
}

fn variant_metrics(values: impl Iterator<Item = ScoredVariantV1>) -> VariantMetricV1 {
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
        unparseable,
        abstained,
        accuracy: rate(correct, items),
        coverage: rate(answered, items),
    }
}

fn target_properties() -> [MusicalTargetPropertyV1; 8] {
    [
        MusicalTargetPropertyV1::ChordQuality,
        MusicalTargetPropertyV1::RootPitchClass,
        MusicalTargetPropertyV1::ChordVoicing,
        MusicalTargetPropertyV1::RelativeIntervalSequence,
        MusicalTargetPropertyV1::MelodicContour,
        MusicalTargetPropertyV1::RhythmicRatioPattern,
        MusicalTargetPropertyV1::PulseRate,
        MusicalTargetPropertyV1::InstrumentFamily,
    ]
}

fn performance_factors() -> [PerformanceVariationFactorV1; 10] {
    [
        PerformanceVariationFactorV1::DifferentPerformer,
        PerformanceVariationFactorV1::DifferentTake,
        PerformanceVariationFactorV1::StrumOrArticulation,
        PerformanceVariationFactorV1::VelocityOrDynamics,
        PerformanceVariationFactorV1::ExpressiveRubato,
        PerformanceVariationFactorV1::GlobalTransposition,
        PerformanceVariationFactorV1::ChordInversion,
        PerformanceVariationFactorV1::ChordVoicing,
        PerformanceVariationFactorV1::UniformTempoScale,
        PerformanceVariationFactorV1::CrossRenderer,
    ]
}

fn musical_flip_factors() -> [MusicalChangeFactorV1; 8] {
    [
        MusicalChangeFactorV1::ChordQuality,
        MusicalChangeFactorV1::RootPitchClass,
        MusicalChangeFactorV1::ChordVoicing,
        MusicalChangeFactorV1::MelodicInterval,
        MusicalChangeFactorV1::MelodicContour,
        MusicalChangeFactorV1::RhythmicRatio,
        MusicalChangeFactorV1::PulseRate,
        MusicalChangeFactorV1::InstrumentFamily,
    ]
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

    const LINEAGE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMPOSITION: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const MODEL: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const PROMPT: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const RUNTIME: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const DECODING: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn variant(
        id: &str,
        audio: &str,
        question: &str,
        expected: MusicalRelationV1,
        answer: MusicListenAnswerV1,
    ) -> MusicListenVariantV1 {
        MusicListenVariantV1 {
            variant_id: id.into(),
            audio_sha256: audio.into(),
            question: question.into(),
            question_sha256: sha256_hex(question.as_bytes()),
            option_a_relation: MusicalRelationV1::Same,
            option_b_relation: MusicalRelationV1::Different,
            expected_relation: expected,
            model_answer: answer,
        }
    }

    fn question_for(target_property: MusicalTargetPropertyV1) -> &'static str {
        match target_property {
            MusicalTargetPropertyV1::ChordQuality => "Are these chord qualities the same?",
            MusicalTargetPropertyV1::RootPitchClass => {
                "Do these clips share the same chord root pitch class?"
            }
            MusicalTargetPropertyV1::ChordVoicing => "Do these chords have the same voicing?",
            MusicalTargetPropertyV1::RelativeIntervalSequence => {
                "Is the ordered melodic interval sequence the same?"
            }
            MusicalTargetPropertyV1::MelodicContour => "Is the melodic contour the same?",
            MusicalTargetPropertyV1::RhythmicRatioPattern => {
                "Is the rhythmic ratio pattern the same?"
            }
            MusicalTargetPropertyV1::PulseRate => "Is the pulse rate the same?",
            MusicalTargetPropertyV1::InstrumentFamily => {
                "Are these performances from the same instrument family?"
            }
        }
    }

    fn triad(
        item_id: &str,
        target_property: MusicalTargetPropertyV1,
        stay_factor: PerformanceVariationFactorV1,
        flip_factor: MusicalChangeFactorV1,
        stay_correct: bool,
        flip_correct: bool,
    ) -> PerformanceTriadV1 {
        let clean_condition = digest('1');
        let stay_condition = digest('2');
        let question = question_for(target_property);
        let clean = variant(
            &format!("{item_id}-clean"),
            &digest('a'),
            question,
            MusicalRelationV1::Same,
            MusicListenAnswerV1::ChoiceA,
        );
        let performance_stay = variant(
            &format!("{item_id}-performance-stay"),
            &digest('b'),
            question,
            MusicalRelationV1::Same,
            if stay_correct {
                MusicListenAnswerV1::ChoiceA
            } else {
                MusicListenAnswerV1::ChoiceB
            },
        );
        let musical_flip = variant(
            &format!("{item_id}-musical-flip"),
            &digest('c'),
            question,
            MusicalRelationV1::Different,
            if flip_correct {
                MusicListenAnswerV1::ChoiceB
            } else {
                MusicListenAnswerV1::ChoiceA
            },
        );
        PerformanceTriadV1 {
            item_id: item_id.into(),
            pair_id: format!("pair-{item_id}"),
            source_lineage_sha256: LINEAGE.into(),
            composition_sha256: COMPOSITION.into(),
            target_property,
            performance_stay_factor: stay_factor,
            musical_flip_factor: flip_factor,
            model_identity_sha256: MODEL.into(),
            system_prompt_sha256: PROMPT.into(),
            runtime_identity_sha256: RUNTIME.into(),
            decoding_config_sha256: DECODING.into(),
            clean: PerformanceVariantV1 {
                performance_condition_sha256: clean_condition.clone(),
                answer: clean,
            },
            performance_stay: PerformanceVariantV1 {
                performance_condition_sha256: stay_condition,
                answer: performance_stay,
            },
            musical_flip: PerformanceVariantV1 {
                performance_condition_sha256: clean_condition,
                answer: musical_flip,
            },
        }
    }

    fn case(triads: Vec<PerformanceTriadV1>) -> PerformanceInvarianceCaseV1 {
        PerformanceInvarianceCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            benchmark_id: "Melothaea-Performance-Identity".into(),
            benchmark_version: "1.0".into(),
            split: PerformanceInvarianceSplitV1::SyntheticRendered,
            dataset_revision_sha256: digest('3'),
            annotation_policy_sha256: digest('4'),
            source_permissions_manifest_sha256: digest('5'),
            triads,
        }
    }

    #[test]
    fn correct_performance_stay_and_musical_flip_are_distinguished() {
        let input = case(vec![triad(
            "chord-quality-1",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        let report = audit_performance_invariance_case(&input).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(1.0));
        assert_eq!(report.overall.performance_stay.accuracy, Some(1.0));
        assert_eq!(report.overall.musical_flip.accuracy, Some(1.0));
        assert_eq!(report.overall.performance_false_difference_rate, Some(0.0));
        assert_eq!(report.overall.musical_flip_miss_rate, Some(0.0));
        assert!(!report.audio_bytes_recomputed);
        assert!(!report.performance_and_annotation_receipts_authenticated);
    }

    #[test]
    fn same_chord_false_difference_is_reported_as_performance_stay_error() {
        let input = case(vec![triad(
            "strum-variation",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::StrumOrArticulation,
            MusicalChangeFactorV1::ChordQuality,
            false,
            true,
        )]);
        let report = audit_performance_invariance_case(&input).unwrap();
        assert_eq!(report.overall.performance_false_difference_rate, Some(1.0));
        assert_eq!(report.overall.musical_flip_miss_rate, Some(0.0));
    }

    #[test]
    fn root_pitch_class_rejects_global_transposition_as_a_stay_factor() {
        let mut input = case(vec![triad(
            "root-change",
            MusicalTargetPropertyV1::RootPitchClass,
            PerformanceVariationFactorV1::GlobalTransposition,
            MusicalChangeFactorV1::RootPitchClass,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&input).is_err());
        input.triads[0].performance_stay_factor = PerformanceVariationFactorV1::DifferentTake;
        assert!(audit_performance_invariance_case(&input).is_ok());
    }

    #[test]
    fn relative_interval_sequence_allows_global_transposition_as_a_stay_factor() {
        let input = case(vec![triad(
            "relative-intervals",
            MusicalTargetPropertyV1::RelativeIntervalSequence,
            PerformanceVariationFactorV1::GlobalTransposition,
            MusicalChangeFactorV1::MelodicInterval,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&input).is_ok());
    }

    #[test]
    fn chord_quality_can_survive_inversion_but_voicing_cannot() {
        let quality = case(vec![triad(
            "inversion-quality",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::ChordInversion,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&quality).is_ok());

        let voicing = case(vec![triad(
            "inversion-voicing",
            MusicalTargetPropertyV1::ChordVoicing,
            PerformanceVariationFactorV1::ChordInversion,
            MusicalChangeFactorV1::ChordVoicing,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&voicing).is_err());
    }

    #[test]
    fn rhythmic_ratio_allows_uniform_tempo_scale_but_pulse_rate_does_not() {
        let ratio = case(vec![triad(
            "rhythm-ratio",
            MusicalTargetPropertyV1::RhythmicRatioPattern,
            PerformanceVariationFactorV1::UniformTempoScale,
            MusicalChangeFactorV1::RhythmicRatio,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&ratio).is_ok());

        let pulse = case(vec![triad(
            "pulse-rate",
            MusicalTargetPropertyV1::PulseRate,
            PerformanceVariationFactorV1::UniformTempoScale,
            MusicalChangeFactorV1::PulseRate,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&pulse).is_err());
    }

    #[test]
    fn musical_flip_must_preserve_the_clean_performance_condition() {
        let mut input = case(vec![triad(
            "flip-control",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0].musical_flip.performance_condition_sha256 = digest('9');
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn performance_stay_must_have_a_distinct_performance_condition() {
        let mut input = case(vec![triad(
            "stay-control",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0]
            .performance_stay
            .performance_condition_sha256 =
            input.triads[0].clean.performance_condition_sha256.clone();
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn instrument_family_target_does_not_admit_cross_renderer_as_performance_stay() {
        let input = case(vec![triad(
            "instrument-family",
            MusicalTargetPropertyV1::InstrumentFamily,
            PerformanceVariationFactorV1::CrossRenderer,
            MusicalChangeFactorV1::InstrumentFamily,
            true,
            true,
        )]);
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn identical_audio_hashes_across_variants_are_rejected() {
        let mut input = case(vec![triad(
            "duplicate-audio",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0].performance_stay.answer.audio_sha256 =
            input.triads[0].clean.answer.audio_sha256.clone();
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn model_runtime_and_variant_identities_are_run_consistent() {
        let first = triad(
            "run-first",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        let mut second = triad(
            "run-second",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        second.source_lineage_sha256 = digest('9');
        second.composition_sha256 = digest('8');
        second.clean.answer.audio_sha256 = digest('1');
        second.performance_stay.answer.audio_sha256 = digest('2');
        second.musical_flip.answer.audio_sha256 = digest('3');
        second.runtime_identity_sha256 = digest('7');
        assert!(
            audit_performance_invariance_case(&case(vec![first.clone(), second.clone()])).is_err()
        );

        second.runtime_identity_sha256 = first.runtime_identity_sha256.clone();
        second.clean.answer.variant_id = first.clean.answer.variant_id.clone();
        assert!(audit_performance_invariance_case(&case(vec![first, second])).is_err());
    }

    #[test]
    fn question_text_must_remain_identical_across_the_matched_triad() {
        let mut input = case(vec![triad(
            "question-scope",
            MusicalTargetPropertyV1::RelativeIntervalSequence,
            PerformanceVariationFactorV1::GlobalTransposition,
            MusicalChangeFactorV1::MelodicInterval,
            true,
            true,
        )]);
        input.triads[0].musical_flip.answer.question = "Is the chord quality the same?".into();
        input.triads[0].musical_flip.answer.question_sha256 =
            sha256_hex(input.triads[0].musical_flip.answer.question.as_bytes());
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn audio_and_composition_digests_cannot_cross_source_work_lineages() {
        let first = triad(
            "source-a",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        let mut second = triad(
            "source-b",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        second.source_lineage_sha256 = digest('9');
        assert!(
            audit_performance_invariance_case(&case(vec![first.clone(), second.clone()])).is_err()
        );

        second.composition_sha256 = digest('8');
        second.clean.answer.audio_sha256 = digest('8');
        second.performance_stay.answer.audio_sha256 = digest('9');
        second.musical_flip.answer.audio_sha256 = digest('a');
        assert!(audit_performance_invariance_case(&case(vec![first, second])).is_err());
    }

    #[test]
    fn duplicate_item_and_pair_ids_are_rejected() {
        let first = triad(
            "duplicate-id",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        let mut second = first.clone();
        second.source_lineage_sha256 = digest('5');
        let input = case(vec![first, second]);
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn option_letter_reversal_is_normalized_to_semantic_relation() {
        let mut item = triad(
            "letter-reversal",
            MusicalTargetPropertyV1::RelativeIntervalSequence,
            PerformanceVariationFactorV1::GlobalTransposition,
            MusicalChangeFactorV1::MelodicInterval,
            true,
            true,
        );
        item.performance_stay.answer.option_a_relation = MusicalRelationV1::Different;
        item.performance_stay.answer.option_b_relation = MusicalRelationV1::Same;
        item.performance_stay.answer.model_answer = MusicListenAnswerV1::ChoiceB;
        let report = audit_performance_invariance_case(&case(vec![item])).unwrap();
        assert_eq!(
            report.triad_results[0].performance_stay_predicted_relation,
            Some(MusicalRelationV1::Same)
        );
        assert!(report.triad_results[0].performance_stay_correct);
    }

    #[test]
    fn unparseable_and_abstaining_answers_remain_in_denominators() {
        let mut item = triad(
            "coverage",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        );
        item.performance_stay.answer.model_answer = MusicListenAnswerV1::Unparseable;
        item.musical_flip.answer.model_answer = MusicListenAnswerV1::Abstain;
        let report = audit_performance_invariance_case(&case(vec![item])).unwrap();
        assert_eq!(report.overall.performance_stay.incorrect, 1);
        assert_eq!(report.overall.performance_stay.unparseable, 1);
        assert_eq!(report.overall.musical_flip.incorrect, 1);
        assert_eq!(report.overall.musical_flip.abstained, 1);
        assert_eq!(report.overall.performance_stay.coverage, Some(0.0));
    }

    #[test]
    fn question_digest_and_target_specific_expected_relations_are_enforced() {
        let mut input = case(vec![triad(
            "question-digest",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0].clean.answer.question_sha256 = digest('7');
        assert!(audit_performance_invariance_case(&input).is_err());

        input = case(vec![triad(
            "wrong-relation",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0].performance_stay.answer.expected_relation = MusicalRelationV1::Different;
        assert!(audit_performance_invariance_case(&input).is_err());
    }

    #[test]
    fn target_property_and_factor_denominators_remain_separate() {
        let items = vec![
            triad(
                "quality",
                MusicalTargetPropertyV1::ChordQuality,
                PerformanceVariationFactorV1::DifferentTake,
                MusicalChangeFactorV1::ChordQuality,
                true,
                true,
            ),
            triad(
                "root",
                MusicalTargetPropertyV1::RootPitchClass,
                PerformanceVariationFactorV1::DifferentPerformer,
                MusicalChangeFactorV1::RootPitchClass,
                false,
                true,
            ),
            triad(
                "intervals",
                MusicalTargetPropertyV1::RelativeIntervalSequence,
                PerformanceVariationFactorV1::GlobalTransposition,
                MusicalChangeFactorV1::MelodicInterval,
                true,
                false,
            ),
        ];
        let report = audit_performance_invariance_case(&case(items)).unwrap();
        assert_eq!(report.overall.triads, 3);
        assert_eq!(report.by_target_property.len(), 3);
        assert_eq!(report.by_performance_factor.len(), 3);
        assert_eq!(report.by_musical_flip_factor.len(), 3);
        assert_eq!(
            report.by_target_property[0].target_property,
            Some(MusicalTargetPropertyV1::ChordQuality)
        );
        assert_eq!(
            report.by_target_property[1].target_property,
            Some(MusicalTargetPropertyV1::RootPitchClass)
        );
        assert_eq!(
            report.by_target_property[2].target_property,
            Some(MusicalTargetPropertyV1::RelativeIntervalSequence)
        );
        assert_eq!(report.source_lineage_count, 1);
        assert!(!report.confirmatory_lineage_floor_met);
    }

    #[test]
    fn invalid_identity_and_incomplete_case_are_rejected() {
        let mut input = case(vec![triad(
            "identity",
            MusicalTargetPropertyV1::ChordQuality,
            PerformanceVariationFactorV1::DifferentTake,
            MusicalChangeFactorV1::ChordQuality,
            true,
            true,
        )]);
        input.triads[0].runtime_identity_sha256 = "bad".into();
        assert!(audit_performance_invariance_case(&input).is_err());

        input = case(Vec::new());
        assert!(audit_performance_invariance_case(&input).is_err());
    }
}
