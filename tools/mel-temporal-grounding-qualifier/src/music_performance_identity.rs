//! Property-relative musical identity across different performance realizations.
//!
//! This is a separate extension to MusicListenBench CLEAN/FLIP/STAY. It does
//! not modify the official benchmark schema or scorer. It adds a third,
//! musically distinct comparison arm: performance-STAY, where the waveform and
//! performance realization change but the declared musical property remains
//! the same. All property labels and performance metadata are external
//! evidence references, not authenticated ground truth in this evaluator.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::music_listen_bench::{
    decode_semantic_relation, sha256_hex, validate_digest, MusicListenAnswerV1,
    MusicListenVariantV1, MusicalRelationV1,
};

const CASE_SCHEMA_V1: &str = "mel-music-performance-identity-case-v1";
const REPORT_SCHEMA_V1: &str = "mel-music-performance-identity-report-v1";
const EVALUATOR_VERSION_V1: &str = "mel-music-performance-identity-v1";
const CONFIRMATORY_SOURCE_WORK_FLOOR: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalIdentityPropertyV1 {
    ChordQuality,
    RootPitchClass,
    Voicing,
    RelativeIntervalSequence,
    MelodicContour,
    RhythmicRatio,
    PulseRate,
    InstrumentFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PerformanceStayFactorV1 {
    DifferentTakeSamePerformer,
    DifferentPerformerSameInstrument,
    StrumOrArticulationChange,
    ExpressiveTimingChange,
    DifferentTempoSameRhythmicRatio,
    DifferentRendererSamePerformance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MusicalFlipFactorV1 {
    ChordQualityChange,
    RootPitchClassChange,
    VoicingChange,
    RelativeIntervalSequenceChange,
    MelodicContourChange,
    RhythmicRatioChange,
    PulseRateChange,
    InstrumentFamilyChange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalIdentityVariantV1 {
    pub response: MusicListenVariantV1,
    /// Opaque identity of the performance take; same across renderer-only views.
    pub take_identity_sha256: String,
    /// Opaque identity for one performance realization.
    pub performance_identity_sha256: String,
    /// Pseudonymous performer identity commitment, not a personal identifier.
    pub performer_identity_sha256: String,
    pub instrument_family_sha256: String,
    pub renderer_identity_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalPerformanceIdentityTriadV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_work_lineage_sha256: String,
    pub target_property: MusicalIdentityPropertyV1,
    pub performance_stay_factor: PerformanceStayFactorV1,
    pub musical_flip_factor: MusicalFlipFactorV1,
    /// Reference manifest containing the independent target-property labels.
    pub target_property_reference_sha256: String,
    /// Pairing manifest binds the intended single-factor counterfactual design.
    pub pairing_manifest_sha256: String,
    pub model_identity_sha256: String,
    pub system_prompt_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub clean: MusicalIdentityVariantV1,
    pub performance_stay: MusicalIdentityVariantV1,
    pub musical_flip: MusicalIdentityVariantV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalPerformanceIdentityCaseV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub dataset_revision_sha256: String,
    pub evaluator_protocol_sha256: String,
    pub triads: Vec<MusicalPerformanceIdentityTriadV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IdentityVariantMetricsV1 {
    pub items: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub answered: usize,
    pub unparseable: usize,
    pub abstained: usize,
    pub accuracy: Option<f64>,
    pub coverage: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalIdentityMetricsV1 {
    pub triads: usize,
    pub clean: IdentityVariantMetricsV1,
    pub performance_stay: IdentityVariantMetricsV1,
    pub musical_flip: IdentityVariantMetricsV1,
    pub performance_stay_false_difference_rate: Option<f64>,
    pub musical_flip_miss_rate: Option<f64>,
    /// Conditional error rates only over triads whose CLEAN answer is correct.
    pub clean_correct_triads: usize,
    pub conditional_performance_stay_false_difference_rate: Option<f64>,
    pub conditional_musical_flip_miss_rate: Option<f64>,
    pub joint_clean_performance_stay_musical_flip_exact_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalIdentityPropertySliceV1 {
    pub property: MusicalIdentityPropertyV1,
    pub metrics: MusicalIdentityMetricsV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceStayFactorSliceV1 {
    pub property: MusicalIdentityPropertyV1,
    pub factor: PerformanceStayFactorV1,
    pub items: usize,
    pub false_difference_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalFlipFactorSliceV1 {
    pub property: MusicalIdentityPropertyV1,
    pub factor: MusicalFlipFactorV1,
    pub items: usize,
    pub miss_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceStayIdentitySliceV1 {
    pub identity_sha256: String,
    pub items: usize,
    pub false_difference_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalIdentityTriadResultV1 {
    pub item_id: String,
    pub pair_id: String,
    pub source_work_lineage_sha256: String,
    pub target_property: MusicalIdentityPropertyV1,
    pub clean_performer_identity_sha256: String,
    pub performance_stay_performer_identity_sha256: String,
    pub performance_stay_take_identity_sha256: String,
    pub performance_stay_renderer_identity_sha256: String,
    pub clean_predicted_relation: Option<MusicalRelationV1>,
    pub performance_stay_predicted_relation: Option<MusicalRelationV1>,
    pub musical_flip_predicted_relation: Option<MusicalRelationV1>,
    pub clean_correct: bool,
    pub performance_stay_correct: bool,
    pub musical_flip_correct: bool,
    pub joint_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MusicalPerformanceIdentityReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub dataset_revision_sha256: String,
    pub evaluator_protocol_sha256: String,
    pub overall: MusicalIdentityMetricsV1,
    pub by_property: Vec<MusicalIdentityPropertySliceV1>,
    pub by_performance_stay_factor: Vec<PerformanceStayFactorSliceV1>,
    pub by_musical_flip_factor: Vec<MusicalFlipFactorSliceV1>,
    pub by_performance_stay_performer: Vec<PerformanceStayIdentitySliceV1>,
    pub by_performance_stay_take: Vec<PerformanceStayIdentitySliceV1>,
    pub by_performance_stay_renderer: Vec<PerformanceStayIdentitySliceV1>,
    pub source_work_clusters: usize,
    pub performer_clusters: usize,
    pub recommended_confirmatory_source_work_floor: usize,
    pub confirmatory_source_work_floor_met: bool,
    pub triad_results: Vec<MusicalIdentityTriadResultV1>,
    /// Performance/property labels are digest-bound references, not signatures.
    pub target_property_reference_authenticated: bool,
    pub pairing_manifests_authenticated: bool,
    /// The evaluator receives digest metadata; it does not possess source audio.
    pub audio_bytes_recomputed: bool,
    /// No population interval or inferential statement is made by this v1.
    pub population_confidence_interval_computed: bool,
}

#[derive(Debug, Clone, Copy)]
struct ScoredIdentityTriadV1 {
    clean_correct: bool,
    performance_stay_correct: bool,
    musical_flip_correct: bool,
    clean_answer: MusicListenAnswerV1,
    performance_stay_answer: MusicListenAnswerV1,
    musical_flip_answer: MusicListenAnswerV1,
}

pub fn audit_musical_performance_identity_json(
    input: &str,
) -> Result<MusicalPerformanceIdentityReportV1, String> {
    let case: MusicalPerformanceIdentityCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid musical performance identity JSON: {error}"))?;
    audit_musical_performance_identity(&case)
}

pub fn audit_musical_performance_identity(
    case: &MusicalPerformanceIdentityCaseV1,
) -> Result<MusicalPerformanceIdentityReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!("unsupported performance identity schema: {}", case.schema));
    }
    require_text(&case.benchmark_id, "benchmark_id")?;
    require_text(&case.benchmark_version, "benchmark_version")?;
    if case.benchmark_id != "MEL-PERFORMANCE-IDENTITY" {
        return Err("benchmark_id must be MEL-PERFORMANCE-IDENTITY".into());
    }
    validate_digest(&case.dataset_revision_sha256, "dataset_revision_sha256")?;
    validate_digest(&case.evaluator_protocol_sha256, "evaluator_protocol_sha256")?;
    if case.triads.is_empty() {
        return Err("performance identity case must contain at least one complete triad".into());
    }

    let first = &case.triads[0];
    for (index, triad) in case.triads.iter().enumerate() {
        validate_triad(triad, index)?;
        if triad.model_identity_sha256 != first.model_identity_sha256
            || triad.system_prompt_sha256 != first.system_prompt_sha256
            || triad.runtime_identity_sha256 != first.runtime_identity_sha256
            || triad.decoding_config_sha256 != first.decoding_config_sha256
        {
            return Err(format!(
                "triads[{index}] changes model/prompt/runtime/decoding identity in one evaluation run"
            ));
        }
    }

    let mut item_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut lineages = BTreeSet::new();
    let mut performers = BTreeSet::new();
    let mut scored = Vec::with_capacity(case.triads.len());
    let mut results = Vec::with_capacity(case.triads.len());

    for triad in &case.triads {
        if !item_ids.insert(triad.item_id.as_str()) {
            return Err(format!("duplicate performance identity item_id: {}", triad.item_id));
        }
        if !pair_ids.insert(triad.pair_id.as_str()) {
            return Err(format!("duplicate performance identity pair_id: {}", triad.pair_id));
        }
        lineages.insert(triad.source_work_lineage_sha256.as_str());
        performers.insert(triad.clean.performer_identity_sha256.as_str());
        performers.insert(triad.performance_stay.performer_identity_sha256.as_str());
        performers.insert(triad.musical_flip.performer_identity_sha256.as_str());

        let clean = decode_semantic_relation(&triad.clean.response);
        let performance_stay = decode_semantic_relation(&triad.performance_stay.response);
        let musical_flip = decode_semantic_relation(&triad.musical_flip.response);
        let clean_correct = clean == Some(MusicalRelationV1::Same);
        let performance_stay_correct = performance_stay == Some(MusicalRelationV1::Same);
        let musical_flip_correct = musical_flip == Some(MusicalRelationV1::Different);
        let joint_exact = clean_correct && performance_stay_correct && musical_flip_correct;

        scored.push(ScoredIdentityTriadV1 {
            clean_correct,
            performance_stay_correct,
            musical_flip_correct,
            clean_answer: triad.clean.response.model_answer,
            performance_stay_answer: triad.performance_stay.response.model_answer,
            musical_flip_answer: triad.musical_flip.response.model_answer,
        });
        results.push(MusicalIdentityTriadResultV1 {
            item_id: triad.item_id.clone(),
            pair_id: triad.pair_id.clone(),
            source_work_lineage_sha256: triad.source_work_lineage_sha256.clone(),
            target_property: triad.target_property,
            clean_performer_identity_sha256: triad.clean.performer_identity_sha256.clone(),
            performance_stay_performer_identity_sha256: triad.performance_stay.performer_identity_sha256.clone(),
            performance_stay_take_identity_sha256: triad.performance_stay.take_identity_sha256.clone(),
            performance_stay_renderer_identity_sha256: triad.performance_stay.renderer_identity_sha256.clone(),
            clean_predicted_relation: clean,
            performance_stay_predicted_relation: performance_stay,
            musical_flip_predicted_relation: musical_flip,
            clean_correct,
            performance_stay_correct,
            musical_flip_correct,
            joint_exact,
        });
    }

    let overall = summarize(&scored);
    let property_keys: BTreeSet<_> = case
        .triads
        .iter()
        .map(|triad| triad.target_property)
        .collect();
    let by_property = property_keys
        .into_iter()
        .map(|property| {
            let indices: Vec<_> = case
                .triads
                .iter()
                .enumerate()
                .filter_map(|(index, triad)| (triad.target_property == property).then_some(index))
                .collect();
            MusicalIdentityPropertySliceV1 {
                property,
                metrics: summarize_indices(&scored, &indices),
            }
        })
        .collect();

    let stay_keys: BTreeSet<_> = case
        .triads
        .iter()
        .map(|triad| (triad.target_property, triad.performance_stay_factor))
        .collect();
    let by_performance_stay_factor = stay_keys
        .into_iter()
        .map(|(property, factor)| {
            let indices: Vec<_> = case
                .triads
                .iter()
                .enumerate()
                .filter_map(|(index, triad)| {
                    (triad.target_property == property && triad.performance_stay_factor == factor)
                        .then_some(index)
                })
                .collect();
            PerformanceStayFactorSliceV1 {
                property,
                factor,
                items: indices.len(),
                false_difference_rate: rate(
                    indices.iter().filter(|index| !scored[**index].performance_stay_correct).count(),
                    indices.len(),
                ),
            }
        })
        .collect();

    let flip_keys: BTreeSet<_> = case
        .triads
        .iter()
        .map(|triad| (triad.target_property, triad.musical_flip_factor))
        .collect();
    let by_musical_flip_factor = flip_keys
        .into_iter()
        .map(|(property, factor)| {
            let indices: Vec<_> = case
                .triads
                .iter()
                .enumerate()
                .filter_map(|(index, triad)| {
                    (triad.target_property == property && triad.musical_flip_factor == factor)
                        .then_some(index)
                })
                .collect();
            MusicalFlipFactorSliceV1 {
                property,
                factor,
                items: indices.len(),
                miss_rate: rate(
                    indices.iter().filter(|index| !scored[**index].musical_flip_correct).count(),
                    indices.len(),
                ),
            }
        })
        .collect();

    Ok(MusicalPerformanceIdentityReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: EVALUATOR_VERSION_V1,
        benchmark_id: case.benchmark_id.clone(),
        benchmark_version: case.benchmark_version.clone(),
        dataset_revision_sha256: case.dataset_revision_sha256.clone(),
        evaluator_protocol_sha256: case.evaluator_protocol_sha256.clone(),
        overall,
        by_property,
        by_performance_stay_factor,
        by_musical_flip_factor,
        by_performance_stay_performer: build_identity_slices(
            &case.triads,
            &scored,
            PerformanceStayIdentityKeyV1::Performer,
        ),
        by_performance_stay_take: build_identity_slices(
            &case.triads,
            &scored,
            PerformanceStayIdentityKeyV1::Take,
        ),
        by_performance_stay_renderer: build_identity_slices(
            &case.triads,
            &scored,
            PerformanceStayIdentityKeyV1::Renderer,
        ),
        source_work_clusters: lineages.len(),
        performer_clusters: performers.len(),
        recommended_confirmatory_source_work_floor: CONFIRMATORY_SOURCE_WORK_FLOOR,
        confirmatory_source_work_floor_met: lineages.len() >= CONFIRMATORY_SOURCE_WORK_FLOOR,
        triad_results: results,
        target_property_reference_authenticated: false,
        pairing_manifests_authenticated: false,
        audio_bytes_recomputed: false,
        population_confidence_interval_computed: false,
    })
}

fn validate_triad(triad: &MusicalPerformanceIdentityTriadV1, index: usize) -> Result<(), String> {
    require_text(&triad.item_id, &format!("triads[{index}].item_id"))?;
    require_text(&triad.pair_id, &format!("triads[{index}].pair_id"))?;
    for (value, label) in [
        (&triad.source_work_lineage_sha256, "source_work_lineage_sha256"),
        (&triad.target_property_reference_sha256, "target_property_reference_sha256"),
        (&triad.pairing_manifest_sha256, "pairing_manifest_sha256"),
        (&triad.model_identity_sha256, "model_identity_sha256"),
        (&triad.system_prompt_sha256, "system_prompt_sha256"),
        (&triad.runtime_identity_sha256, "runtime_identity_sha256"),
        (&triad.decoding_config_sha256, "decoding_config_sha256"),
    ] {
        validate_digest(value, &format!("triads[{index}].{label}"))?;
    }
    if !flip_compatible(triad.target_property, triad.musical_flip_factor) {
        return Err(format!(
            "triads[{index}] musical FLIP factor {:?} is incompatible with target property {:?}",
            triad.musical_flip_factor, triad.target_property
        ));
    }
    if !stay_compatible(triad.target_property, triad.performance_stay_factor) {
        return Err(format!(
            "triads[{index}] performance-STAY factor {:?} is incompatible with target property {:?}",
            triad.performance_stay_factor, triad.target_property
        ));
    }

    let variants = [
        ("clean", &triad.clean, MusicalRelationV1::Same),
        ("performance_stay", &triad.performance_stay, MusicalRelationV1::Same),
        ("musical_flip", &triad.musical_flip, MusicalRelationV1::Different),
    ];
    let mut variant_ids = BTreeSet::new();
    let mut audio_digests = BTreeSet::new();
    for (label, variant, expected) in variants {
        validate_variant(variant, label, index, expected)?;
        if !variant_ids.insert(variant.response.variant_id.as_str()) {
            return Err(format!("triads[{index}] duplicate variant_id"));
        }
        if !audio_digests.insert(variant.response.audio_sha256.as_str()) {
            return Err(format!("triads[{index}] variants must have distinct audio digests"));
        }
    }

    let clean = &triad.clean;
    let stay = &triad.performance_stay;
    let flip = &triad.musical_flip;
    if clean.take_identity_sha256 == stay.take_identity_sha256
        && triad.performance_stay_factor != PerformanceStayFactorV1::DifferentRendererSamePerformance
    {
        return Err(format!(
            "triads[{index}] performance-STAY must use a distinct take for the declared performance variation"
        ));
    }
    if clean.performance_identity_sha256 == stay.performance_identity_sha256
        && triad.performance_stay_factor != PerformanceStayFactorV1::DifferentRendererSamePerformance
    {
        return Err(format!(
            "triads[{index}] performance-STAY must identify a different performance realization"
        ));
    }
    if clean.take_identity_sha256 != stay.take_identity_sha256
        && triad.performance_stay_factor == PerformanceStayFactorV1::DifferentRendererSamePerformance
    {
        return Err(format!(
            "triads[{index}] renderer-only STAY must retain the same take identity"
        ));
    }
    if clean.performance_identity_sha256 != stay.performance_identity_sha256
        && triad.performance_stay_factor == PerformanceStayFactorV1::DifferentRendererSamePerformance
    {
        return Err(format!(
            "triads[{index}] renderer-only STAY must retain the same performance identity"
        ));
    }

    validate_performance_stay_metadata(triad, index)?;

    if triad.target_property == MusicalIdentityPropertyV1::InstrumentFamily {
        if clean.instrument_family_sha256 == flip.instrument_family_sha256 {
            return Err(format!(
                "triads[{index}] instrument-family FLIP must change instrument-family identity"
            ));
        }
    } else if clean.instrument_family_sha256 != flip.instrument_family_sha256 {
        return Err(format!(
            "triads[{index}] non-timbre musical FLIP must preserve instrument-family identity"
        ));
    }
    if clean.renderer_identity_sha256 != flip.renderer_identity_sha256
        && triad.target_property != MusicalIdentityPropertyV1::InstrumentFamily
    {
        return Err(format!(
            "triads[{index}] non-timbre musical FLIP must keep the renderer fixed"
        ));
    }

    Ok(())
}

fn validate_variant(
    variant: &MusicalIdentityVariantV1,
    label: &str,
    triad_index: usize,
    expected: MusicalRelationV1,
) -> Result<(), String> {
    let response = &variant.response;
    require_text(&response.variant_id, &format!("triads[{triad_index}].{label}.variant_id"))?;
    require_text(&response.question, &format!("triads[{triad_index}].{label}.question"))?;
    for (value, name) in [
        (&response.audio_sha256, "audio_sha256"),
        (&response.question_sha256, "question_sha256"),
        (&variant.take_identity_sha256, "take_identity_sha256"),
        (&variant.performance_identity_sha256, "performance_identity_sha256"),
        (&variant.performer_identity_sha256, "performer_identity_sha256"),
        (&variant.instrument_family_sha256, "instrument_family_sha256"),
        (&variant.renderer_identity_sha256, "renderer_identity_sha256"),
    ] {
        validate_digest(value, &format!("triads[{triad_index}].{label}.{name}"))?;
    }
    if response.question_sha256 != sha256_hex(response.question.as_bytes()) {
        return Err(format!(
            "triads[{triad_index}] {label} question digest mismatch"
        ));
    }
    if response.expected_relation != expected {
        return Err(format!(
            "triads[{triad_index}] {label} expected relation must be {:?}",
            expected
        ));
    }
    if response.option_a_relation == response.option_b_relation {
        return Err(format!(
            "triads[{triad_index}] {label} A/B options must decode to opposite musical relations"
        ));
    }
    Ok(())
}

fn validate_performance_stay_metadata(
    triad: &MusicalPerformanceIdentityTriadV1,
    index: usize,
) -> Result<(), String> {
    let clean = &triad.clean;
    let stay = &triad.performance_stay;
    let same_performer = clean.performer_identity_sha256 == stay.performer_identity_sha256;
    let same_instrument = clean.instrument_family_sha256 == stay.instrument_family_sha256;
    let same_renderer = clean.renderer_identity_sha256 == stay.renderer_identity_sha256;
    let changed_take = clean.take_identity_sha256 != stay.take_identity_sha256;
    let changed_performance =
        clean.performance_identity_sha256 != stay.performance_identity_sha256;

    let valid = match triad.performance_stay_factor {
        PerformanceStayFactorV1::DifferentTakeSamePerformer => {
            same_performer && same_instrument && same_renderer && changed_take && changed_performance
        }
        PerformanceStayFactorV1::DifferentPerformerSameInstrument => {
            !same_performer && same_instrument && same_renderer && changed_take && changed_performance
        }
        PerformanceStayFactorV1::StrumOrArticulationChange => {
            matches!(
                triad.target_property,
                MusicalIdentityPropertyV1::ChordQuality
                    | MusicalIdentityPropertyV1::RootPitchClass
                    | MusicalIdentityPropertyV1::RelativeIntervalSequence
                    | MusicalIdentityPropertyV1::MelodicContour
            ) && same_performer
                && same_instrument
                && same_renderer
                && changed_take
                && changed_performance
        }
        PerformanceStayFactorV1::ExpressiveTimingChange => {
            !matches!(
                triad.target_property,
                MusicalIdentityPropertyV1::RhythmicRatio
                    | MusicalIdentityPropertyV1::PulseRate
            ) && same_performer
                && same_instrument
                && same_renderer
                && changed_take
                && changed_performance
        }
        PerformanceStayFactorV1::DifferentTempoSameRhythmicRatio => {
            triad.target_property == MusicalIdentityPropertyV1::RhythmicRatio
                && same_performer
                && same_instrument
                && same_renderer
                && changed_take
                && changed_performance
        }
        PerformanceStayFactorV1::DifferentRendererSamePerformance => {
            triad.target_property != MusicalIdentityPropertyV1::InstrumentFamily
                && same_performer
                && same_instrument
                && !same_renderer
                && !changed_take
                && !changed_performance
        }
    };
    if !valid {
        return Err(format!(
            "triads[{index}] performance-STAY identity metadata does not match declared factor {:?}",
            triad.performance_stay_factor
        ));
    }
    Ok(())
}

fn flip_compatible(
    property: MusicalIdentityPropertyV1,
    factor: MusicalFlipFactorV1,
) -> bool {
    matches!(
        (property, factor),
        (MusicalIdentityPropertyV1::ChordQuality, MusicalFlipFactorV1::ChordQualityChange)
            | (MusicalIdentityPropertyV1::RootPitchClass, MusicalFlipFactorV1::RootPitchClassChange)
            | (MusicalIdentityPropertyV1::Voicing, MusicalFlipFactorV1::VoicingChange)
            | (
                MusicalIdentityPropertyV1::RelativeIntervalSequence,
                MusicalFlipFactorV1::RelativeIntervalSequenceChange
            )
            | (
                MusicalIdentityPropertyV1::MelodicContour,
                MusicalFlipFactorV1::MelodicContourChange
            )
            | (MusicalIdentityPropertyV1::RhythmicRatio, MusicalFlipFactorV1::RhythmicRatioChange)
            | (MusicalIdentityPropertyV1::PulseRate, MusicalFlipFactorV1::PulseRateChange)
            | (
                MusicalIdentityPropertyV1::InstrumentFamily,
                MusicalFlipFactorV1::InstrumentFamilyChange
            )
    )
}

fn stay_compatible(
    property: MusicalIdentityPropertyV1,
    factor: PerformanceStayFactorV1,
) -> bool {
    match factor {
        PerformanceStayFactorV1::DifferentTakeSamePerformer
        | PerformanceStayFactorV1::DifferentPerformerSameInstrument => true,
        PerformanceStayFactorV1::StrumOrArticulationChange => matches!(
            property,
            MusicalIdentityPropertyV1::ChordQuality
                | MusicalIdentityPropertyV1::RootPitchClass
                | MusicalIdentityPropertyV1::RelativeIntervalSequence
                | MusicalIdentityPropertyV1::MelodicContour
        ),
        PerformanceStayFactorV1::ExpressiveTimingChange => !matches!(
            property,
            MusicalIdentityPropertyV1::RhythmicRatio | MusicalIdentityPropertyV1::PulseRate
        ),
        PerformanceStayFactorV1::DifferentTempoSameRhythmicRatio => {
            property == MusicalIdentityPropertyV1::RhythmicRatio
        }
        PerformanceStayFactorV1::DifferentRendererSamePerformance => {
            property != MusicalIdentityPropertyV1::InstrumentFamily
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PerformanceStayIdentityKeyV1 {
    Performer,
    Take,
    Renderer,
}

fn build_identity_slices(
    triads: &[MusicalPerformanceIdentityTriadV1],
    scored: &[ScoredIdentityTriadV1],
    key: PerformanceStayIdentityKeyV1,
) -> Vec<PerformanceStayIdentitySliceV1> {
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, triad) in triads.iter().enumerate() {
        let identity = match key {
            PerformanceStayIdentityKeyV1::Performer => {
                triad.performance_stay.performer_identity_sha256.as_str()
            }
            PerformanceStayIdentityKeyV1::Take => {
                triad.performance_stay.take_identity_sha256.as_str()
            }
            PerformanceStayIdentityKeyV1::Renderer => {
                triad.performance_stay.renderer_identity_sha256.as_str()
            }
        };
        grouped.entry(identity).or_default().push(index);
    }
    grouped
        .into_iter()
        .map(|(identity, indices)| PerformanceStayIdentitySliceV1 {
            identity_sha256: identity.to_owned(),
            items: indices.len(),
            false_difference_rate: rate(
                indices
                    .iter()
                    .filter(|index| !scored[**index].performance_stay_correct)
                    .count(),
                indices.len(),
            ),
        })
        .collect()
}

fn summarize(scored: &[ScoredIdentityTriadV1]) -> MusicalIdentityMetricsV1 {
    let indices: Vec<_> = (0..scored.len()).collect();
    summarize_indices(scored, &indices)
}

fn summarize_indices(
    scored: &[ScoredIdentityTriadV1],
    indices: &[usize],
) -> MusicalIdentityMetricsV1 {
    let clean = variant_metrics(indices.iter().map(|index| {
        (scored[*index].clean_correct, scored[*index].clean_answer)
    }));
    let performance_stay = variant_metrics(indices.iter().map(|index| {
        (
            scored[*index].performance_stay_correct,
            scored[*index].performance_stay_answer,
        )
    }));
    let musical_flip = variant_metrics(indices.iter().map(|index| {
        (scored[*index].musical_flip_correct, scored[*index].musical_flip_answer)
    }));
    let clean_correct: Vec<_> = indices
        .iter()
        .filter(|index| scored[**index].clean_correct)
        .copied()
        .collect();
    let conditional_performance_stay_errors = clean_correct
        .iter()
        .filter(|index| !scored[**index].performance_stay_correct)
        .count();
    let conditional_musical_flip_errors = clean_correct
        .iter()
        .filter(|index| !scored[**index].musical_flip_correct)
        .count();
    let joint = indices
        .iter()
        .filter(|index| {
            scored[**index].clean_correct
                && scored[**index].performance_stay_correct
                && scored[**index].musical_flip_correct
        })
        .count();
    MusicalIdentityMetricsV1 {
        triads: indices.len(),
        performance_stay_false_difference_rate: rate(
            performance_stay.incorrect,
            performance_stay.items,
        ),
        musical_flip_miss_rate: rate(musical_flip.incorrect, musical_flip.items),
        clean_correct_triads: clean_correct.len(),
        conditional_performance_stay_false_difference_rate: rate(
            conditional_performance_stay_errors,
            clean_correct.len(),
        ),
        conditional_musical_flip_miss_rate: rate(
            conditional_musical_flip_errors,
            clean_correct.len(),
        ),
        joint_clean_performance_stay_musical_flip_exact_rate: rate(joint, indices.len()),
        clean,
        performance_stay,
        musical_flip,
    }
}

fn variant_metrics<'a>(
    values: impl Iterator<Item = (bool, MusicListenAnswerV1)> + 'a,
) -> IdentityVariantMetricsV1 {
    let mut result = IdentityVariantMetricsV1 {
        items: 0,
        correct: 0,
        incorrect: 0,
        answered: 0,
        unparseable: 0,
        abstained: 0,
        accuracy: None,
        coverage: None,
    };
    for (correct, answer) in values {
        result.items += 1;
        if correct {
            result.correct += 1;
        } else {
            result.incorrect += 1;
        }
        match answer {
            MusicListenAnswerV1::ChoiceA | MusicListenAnswerV1::ChoiceB => result.answered += 1,
            MusicListenAnswerV1::Unparseable => result.unparseable += 1,
            MusicListenAnswerV1::Abstain => result.abstained += 1,
        }
    }
    result.accuracy = rate(result.correct, result.items);
    result.coverage = rate(result.answered, result.items);
    result
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
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

    fn digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn response(
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

    fn variant(
        id: &str,
        audio: char,
        take: char,
        performance: char,
        performer: char,
        instrument: char,
        renderer: char,
        expected: MusicalRelationV1,
        answer: MusicListenAnswerV1,
    ) -> MusicalIdentityVariantV1 {
        MusicalIdentityVariantV1 {
            response: response(id, &digest(audio), "Is the target musical property the same?", expected, answer),
            take_identity_sha256: digest(take),
            performance_identity_sha256: digest(performance),
            performer_identity_sha256: digest(performer),
            instrument_family_sha256: digest(instrument),
            renderer_identity_sha256: digest(renderer),
        }
    }

    fn triad(
        property: MusicalIdentityPropertyV1,
        stay_factor: PerformanceStayFactorV1,
        flip_factor: MusicalFlipFactorV1,
    ) -> MusicalPerformanceIdentityTriadV1 {
        let clean = variant(
            "clean",
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            MusicalRelationV1::Same,
            MusicListenAnswerV1::ChoiceA,
        );
        let (stay_take, stay_performance, stay_performer, stay_renderer) = match stay_factor {
            PerformanceStayFactorV1::DifferentPerformerSameInstrument => ('7', '8', '9', '6'),
            PerformanceStayFactorV1::DifferentRendererSamePerformance => ('2', '3', '4', 'a'),
            _ => ('7', '8', '4', '6'),
        };
        let performance_stay = variant(
            "performance-stay",
            'b',
            stay_take,
            stay_performance,
            stay_performer,
            '5',
            stay_renderer,
            MusicalRelationV1::Same,
            MusicListenAnswerV1::ChoiceA,
        );
        let flip_instrument = if property == MusicalIdentityPropertyV1::InstrumentFamily { 'c' } else { '5' };
        let musical_flip = variant(
            "musical-flip",
            'd',
            '2',
            '3',
            '4',
            flip_instrument,
            '6',
            MusicalRelationV1::Different,
            MusicListenAnswerV1::ChoiceB,
        );
        MusicalPerformanceIdentityTriadV1 {
            item_id: format!("item-{property:?}-{stay_factor:?}"),
            pair_id: format!("pair-{property:?}-{stay_factor:?}"),
            source_work_lineage_sha256: digest('e'),
            target_property: property,
            performance_stay_factor: stay_factor,
            musical_flip_factor: flip_factor,
            target_property_reference_sha256: digest('f'),
            pairing_manifest_sha256: digest('0'),
            model_identity_sha256: digest('a'),
            system_prompt_sha256: digest('b'),
            runtime_identity_sha256: digest('c'),
            decoding_config_sha256: digest('d'),
            clean,
            performance_stay,
            musical_flip,
        }
    }

    fn case(triads: Vec<MusicalPerformanceIdentityTriadV1>) -> MusicalPerformanceIdentityCaseV1 {
        MusicalPerformanceIdentityCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            benchmark_id: "MEL-PERFORMANCE-IDENTITY".into(),
            benchmark_version: "1.0".into(),
            dataset_revision_sha256: digest('1'),
            evaluator_protocol_sha256: digest('2'),
            triads,
        }
    }

    #[test]
    fn property_relative_reference_triad_scores_clean_stay_and_flip_separately() {
        let input = case(vec![triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::ChordQualityChange,
        )]);
        let report = audit_musical_performance_identity(&input).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(1.0));
        assert_eq!(report.overall.performance_stay.accuracy, Some(1.0));
        assert_eq!(report.overall.musical_flip.accuracy, Some(1.0));
        assert_eq!(report.overall.performance_stay_false_difference_rate, Some(0.0));
        assert_eq!(report.overall.musical_flip_miss_rate, Some(0.0));
        assert_eq!(report.overall.joint_clean_performance_stay_musical_flip_exact_rate, Some(1.0));
        assert_eq!(report.source_work_clusters, 1);
        assert!(!report.confirmatory_source_work_floor_met);
        assert!(!report.target_property_reference_authenticated);
        assert!(!report.audio_bytes_recomputed);
    }

    #[test]
    fn always_same_fails_musical_flip() {
        let mut item = triad(
            MusicalIdentityPropertyV1::RelativeIntervalSequence,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::RelativeIntervalSequenceChange,
        );
        item.musical_flip.response.model_answer = MusicListenAnswerV1::ChoiceA;
        let report = audit_musical_performance_identity(&case(vec![item])).unwrap();
        assert_eq!(report.overall.performance_stay_false_difference_rate, Some(0.0));
        assert_eq!(report.overall.musical_flip_miss_rate, Some(1.0));
        assert_eq!(report.overall.joint_clean_performance_stay_musical_flip_exact_rate, Some(0.0));
    }

    #[test]
    fn always_different_fails_clean_and_performance_stay() {
        let mut item = triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::ChordQualityChange,
        );
        item.clean.response.model_answer = MusicListenAnswerV1::ChoiceB;
        item.performance_stay.response.model_answer = MusicListenAnswerV1::ChoiceB;
        let report = audit_musical_performance_identity(&case(vec![item])).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(0.0));
        assert_eq!(report.overall.performance_stay_false_difference_rate, Some(1.0));
        assert_eq!(report.overall.musical_flip_miss_rate, Some(0.0));
    }

    #[test]
    fn reversed_a_b_mapping_is_decoded_to_semantic_meaning() {
        let mut item = triad(
            MusicalIdentityPropertyV1::RootPitchClass,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::RootPitchClassChange,
        );
        item.clean.response.option_a_relation = MusicalRelationV1::Different;
        item.clean.response.option_b_relation = MusicalRelationV1::Same;
        item.clean.response.model_answer = MusicListenAnswerV1::ChoiceB;
        let report = audit_musical_performance_identity(&case(vec![item])).unwrap();
        assert_eq!(report.overall.clean.accuracy, Some(1.0));
    }

    #[test]
    fn target_property_and_flip_factor_must_match() {
        let mut item = triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::RootPitchClassChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![item.clone()])).is_err());
        item.musical_flip_factor = MusicalFlipFactorV1::ChordQualityChange;
        assert!(audit_musical_performance_identity(&case(vec![item])).is_ok());
    }

    #[test]
    fn different_performer_factor_requires_distinct_performer_but_same_instrument() {
        let item = triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentPerformerSameInstrument,
            MusicalFlipFactorV1::ChordQualityChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![item.clone()])).is_ok());
        let mut bad = item;
        bad.performance_stay.performer_identity_sha256 = bad.clean.performer_identity_sha256.clone();
        assert!(audit_musical_performance_identity(&case(vec![bad])).is_err());
    }

    #[test]
    fn tempo_stay_is_admissible_for_rhythm_ratio_but_not_pulse_rate() {
        let rhythm = triad(
            MusicalIdentityPropertyV1::RhythmicRatio,
            PerformanceStayFactorV1::DifferentTempoSameRhythmicRatio,
            MusicalFlipFactorV1::RhythmicRatioChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![rhythm])).is_ok());
        let pulse = triad(
            MusicalIdentityPropertyV1::PulseRate,
            PerformanceStayFactorV1::DifferentTempoSameRhythmicRatio,
            MusicalFlipFactorV1::PulseRateChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![pulse])).is_err());
    }

    #[test]
    fn strum_change_is_not_a_rhythm_ratio_or_voicing_stay() {
        let ratio = triad(
            MusicalIdentityPropertyV1::RhythmicRatio,
            PerformanceStayFactorV1::StrumOrArticulationChange,
            MusicalFlipFactorV1::RhythmicRatioChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![ratio])).is_err());
        let voicing = triad(
            MusicalIdentityPropertyV1::Voicing,
            PerformanceStayFactorV1::StrumOrArticulationChange,
            MusicalFlipFactorV1::VoicingChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![voicing])).is_err());
    }

    #[test]
    fn renderer_stay_requires_same_take_performance_and_non_timbre_property() {
        let renderer = triad(
            MusicalIdentityPropertyV1::RelativeIntervalSequence,
            PerformanceStayFactorV1::DifferentRendererSamePerformance,
            MusicalFlipFactorV1::RelativeIntervalSequenceChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![renderer.clone()])).is_ok());

        let mut bad = renderer;
        bad.performance_stay.take_identity_sha256 = digest('7');
        bad.performance_stay.performance_identity_sha256 = digest('8');
        assert!(audit_musical_performance_identity(&case(vec![bad])).is_err());
    }

    #[test]
    fn instrument_family_flip_requires_instrument_identity_change() {
        let item = triad(
            MusicalIdentityPropertyV1::InstrumentFamily,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::InstrumentFamilyChange,
        );
        assert!(audit_musical_performance_identity(&case(vec![item.clone()])).is_ok());
        let mut bad = item;
        bad.musical_flip.instrument_family_sha256 = bad.clean.instrument_family_sha256.clone();
        assert!(audit_musical_performance_identity(&case(vec![bad])).is_err());
    }

    #[test]
    fn performer_take_and_renderer_results_remain_separate_slices() {
        let first = triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::ChordQualityChange,
        );
        let second = triad(
            MusicalIdentityPropertyV1::RootPitchClass,
            PerformanceStayFactorV1::DifferentPerformerSameInstrument,
            MusicalFlipFactorV1::RootPitchClassChange,
        );
        let mut second = second;
        second.item_id = "second-item".into();
        second.pair_id = "second-pair".into();
        second.clean.performer_identity_sha256 = digest('1');
        second.performance_stay.performer_identity_sha256 = digest('2');
        second.musical_flip.performer_identity_sha256 = digest('1');
        second.source_work_lineage_sha256 = digest('3');

        let report = audit_musical_performance_identity(&case(vec![first, second])).unwrap();
        assert_eq!(report.by_performance_stay_performer.len(), 2);
        assert_eq!(report.by_performance_stay_take.len(), 2);
        assert_eq!(report.by_performance_stay_renderer.len(), 1);
        assert_eq!(report.by_performance_stay_renderer[0].items, 2);
    }

    #[test]
    fn unknown_audio_reuse_duplicate_pair_and_model_identity_drift_are_rejected() {
        let item = triad(
            MusicalIdentityPropertyV1::ChordQuality,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::ChordQualityChange,
        );
        let mut reused_audio = item.clone();
        reused_audio.performance_stay.response.audio_sha256 = reused_audio.clean.response.audio_sha256.clone();
        assert!(audit_musical_performance_identity(&case(vec![reused_audio])).is_err());

        let mut duplicate_pair = triad(
            MusicalIdentityPropertyV1::RootPitchClass,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::RootPitchClassChange,
        );
        duplicate_pair.pair_id = item.pair_id.clone();
        assert!(audit_musical_performance_identity(&case(vec![item.clone(), duplicate_pair])).is_err());

        let mut model_drift = triad(
            MusicalIdentityPropertyV1::RootPitchClass,
            PerformanceStayFactorV1::DifferentTakeSamePerformer,
            MusicalFlipFactorV1::RootPitchClassChange,
        );
        model_drift.model_identity_sha256 = digest('9');
        assert!(audit_musical_performance_identity(&case(vec![item, model_drift])).is_err());
    }
}
