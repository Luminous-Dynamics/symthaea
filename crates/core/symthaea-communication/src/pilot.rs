//! Small, real-data human speech pilot runner.

use crate::human::{HumanCommunicationProvider, LocalJsonlProvider};
use crate::metrics::{character_error_rate, chrf, exact_preservation, word_error_rate};
use crate::{Modality, SensorCalibration, SignalObservation, TimeSpan};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PilotSample {
    pub id: String,
    pub language: String,
    pub audio_path: String,
    pub reference_transcript: String,
    #[serde(default)]
    pub translation_reference_en: Option<String>,
    #[serde(default)]
    pub required_entities: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PilotOutcome {
    pub id: String,
    pub language: String,
    pub detected_language: Option<String>,
    pub language_confidence: f32,
    pub transcript: Option<String>,
    pub word_error_rate: Option<f64>,
    pub character_error_rate: Option<f64>,
    pub entity_preservation: Option<f64>,
    pub translation_en: Option<String>,
    pub translation_chrf: Option<f64>,
    pub response_language_correct: Option<bool>,
    pub latency_ms: f64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PilotLanguageSummary {
    pub language: String,
    pub attempted: u64,
    pub successful: u64,
    pub language_id_accuracy: f64,
    pub mean_word_error_rate: f64,
    pub mean_character_error_rate: f64,
    pub mean_entity_preservation: f64,
    pub mean_latency_ms: f64,
    pub expected_calibration_error: f64,
    pub mean_translation_chrf: Option<f64>,
    pub response_language_accuracy: Option<f64>,
}

pub fn summarize(outcomes: &[PilotOutcome]) -> Vec<PilotLanguageSummary> {
    let mut grouped: BTreeMap<&str, Vec<&PilotOutcome>> = BTreeMap::new();
    for outcome in outcomes {
        grouped.entry(&outcome.language).or_default().push(outcome);
    }
    grouped
        .into_iter()
        .map(|(language, values)| {
            let successful: Vec<_> = values
                .iter()
                .copied()
                .filter(|value| value.error.is_none())
                .collect();
            let mean = |selector: fn(&PilotOutcome) -> Option<f64>| {
                let selected: Vec<_> = successful
                    .iter()
                    .filter_map(|value| selector(value))
                    .collect();
                if selected.is_empty() {
                    f64::NAN
                } else {
                    selected.iter().sum::<f64>() / selected.len() as f64
                }
            };
            let accuracy = values
                .iter()
                .filter(|value| value.detected_language.as_deref() == Some(language))
                .count() as f64
                / values.len().max(1) as f64;
            let calibration = values
                .iter()
                .map(|value| {
                    let correct = if value.detected_language.as_deref() == Some(language) {
                        1.0
                    } else {
                        0.0
                    };
                    (value.language_confidence as f64 - correct).abs()
                })
                .sum::<f64>()
                / values.len().max(1) as f64;
            PilotLanguageSummary {
                language: language.into(),
                attempted: values.len() as u64,
                successful: successful.len() as u64,
                language_id_accuracy: accuracy,
                mean_word_error_rate: mean(|value| value.word_error_rate),
                mean_character_error_rate: mean(|value| value.character_error_rate),
                mean_entity_preservation: mean(|value| value.entity_preservation),
                mean_translation_chrf: {
                    let value = mean(|outcome| outcome.translation_chrf);
                    value.is_finite().then_some(value)
                },
                response_language_accuracy: {
                    let selected: Vec<_> = successful
                        .iter()
                        .filter_map(|value| value.response_language_correct)
                        .collect();
                    (!selected.is_empty()).then(|| {
                        selected.iter().filter(|value| **value).count() as f64
                            / selected.len() as f64
                    })
                },
                mean_latency_ms: values.iter().map(|value| value.latency_ms).sum::<f64>()
                    / values.len().max(1) as f64,
                expected_calibration_error: calibration,
            }
        })
        .collect()
}

pub fn evaluate_sample(provider: &mut LocalJsonlProvider, sample: &PilotSample) -> PilotOutcome {
    let start = Instant::now();
    match observation(sample).and_then(|observation| {
        let languages = provider.identify_language(&observation)?;
        let best = languages.first().cloned();
        let transcript = provider.transcribe(&observation)?;
        let translation = if sample.translation_reference_en.is_some() && sample.language != "en" {
            Some(provider.translate_speech(&observation, "en")?)
        } else {
            None
        };
        Ok((best, transcript, translation))
    }) {
        Ok((best, transcript, translation)) => PilotOutcome {
            id: sample.id.clone(),
            language: sample.language.clone(),
            detected_language: best.as_ref().map(|value| value.language.clone()),
            language_confidence: best.as_ref().map_or(0.0, |value| value.confidence),
            word_error_rate: Some(word_error_rate(
                &sample.reference_transcript,
                &transcript.original,
            )),
            character_error_rate: Some(character_error_rate(
                &sample.reference_transcript,
                &transcript.original,
            )),
            entity_preservation: Some(exact_preservation(
                &sample.required_entities,
                &transcript.original,
            )),
            translation_chrf: sample
                .translation_reference_en
                .as_ref()
                .zip(translation.as_ref())
                .map(|(reference, actual)| chrf(reference, &actual.original, 6)),
            response_language_correct: translation
                .as_ref()
                .map(|actual| actual.primary_language.as_deref() == Some("en")),
            translation_en: translation.map(|value| value.original),
            transcript: Some(transcript.original),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            error: None,
        },
        Err(error) => PilotOutcome {
            id: sample.id.clone(),
            language: sample.language.clone(),
            detected_language: None,
            language_confidence: 0.0,
            transcript: None,
            word_error_rate: None,
            character_error_rate: None,
            entity_preservation: None,
            translation_en: None,
            translation_chrf: None,
            response_language_correct: None,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            error: Some(error),
        },
    }
}

/// Recompute WER/CER for existing outcomes against their reference
/// transcripts, without re-invoking the provider. Lets a metrics-only fix
/// (e.g. text normalization) be applied to already-collected transcripts.
pub fn rescore(samples: &[PilotSample], outcomes: &[PilotOutcome]) -> Vec<PilotOutcome> {
    let references: BTreeMap<&str, &str> = samples
        .iter()
        .map(|sample| (sample.id.as_str(), sample.reference_transcript.as_str()))
        .collect();
    outcomes
        .iter()
        .cloned()
        .map(|mut outcome| {
            if let (Some(reference), Some(transcript)) =
                (references.get(outcome.id.as_str()), &outcome.transcript)
            {
                outcome.word_error_rate = Some(word_error_rate(reference, transcript));
                outcome.character_error_rate = Some(character_error_rate(reference, transcript));
            }
            outcome
        })
        .collect()
}

fn observation(sample: &PilotSample) -> Result<SignalObservation, String> {
    let mut reader =
        hound::WavReader::open(Path::new(&sample.audio_path)).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("pilot requires mono WAV input".into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                return Err("unsupported integer WAV bit depth".into());
            }
            let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|value| value.map(|sample| sample as f32 / scale))
                .collect()
        }
    }
    .map_err(|error| error.to_string())?;
    let duration = samples.len() as f64 / spec.sample_rate as f64;
    let mut observation = SignalObservation {
        id: String::new(),
        modality: Modality::Audio {
            sample_rate_hz: spec.sample_rate,
            channels: 1,
        },
        samples,
        features: BTreeMap::new(),
        original_text: None,
        normalized_text: None,
        uncertain_spans: vec![],
        timing: TimeSpan {
            start_s: 0.0,
            end_s: duration,
        },
        location: None,
        calibration: SensorCalibration::default(),
        source_identity: None,
        environment: BTreeMap::from([
            ("external_id".into(), sample.id.clone()),
            ("expected_language".into(), sample.language.clone()),
        ]),
    };
    observation
        .refresh_id()
        .map_err(|error| error.to_string())?;
    Ok(observation)
}
