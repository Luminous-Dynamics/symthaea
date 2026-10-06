// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time streaming voice: text → phonemes → synthesis → speaker output.
//!
//! Combines [`SimpleG2P`] + [`StreamingVocalTract`] + [`AudioOutput`] into a single
//! `speak()` call that streams audio incrementally (first audio within ~25ms).
//!
//! # Modes
//!
//! - **`speak()`** — synchronous, blocks caller until utterance is buffered
//! - **`speak_async()`** — spawns a background thread, returns a [`SpeakHandle`]
//! - **`speak_to_file()`** — writes WAV to disk (no audio device needed)
//!
//! # Prosody
//!
//! The cognitive state can be updated mid-utterance via the shared
//! [`Arc<parking_lot::Mutex<VoiceCognitiveState>>`] returned by [`LiveVoice::cognitive_state_handle()`].
//! Changes take effect on the next motor frame (~5ms latency).
//!
//! Feature-gated under `live-voice`.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
#[cfg(feature = "ssm_language")]
use symthaea_broca::{
    ContentBindingStatus, LexicalMorphosyntacticBinding, LexicalPhonologicalWitness,
    LinguisticFrame, MorphophonologicalCompilationWitness,
    MorphophonologicalDerivationWitness, MorphophonologicalRuleSet, PhonologicalPlan,
};
use symthaea_vocal_tract::pipeline::{
    Intonation, MannerClass, PitchAccent, ProsodyContext, phoneme_manner_class, predict_duration,
};

use super::audio_out::AudioOutput;
use super::formant_targets::FormantDatabase;
use super::repl_voice::{PronunciationLexiconEvidence, SimpleG2P};
use super::vocal_tract_controller::train_controller_on_phoneme_db;
use super::vocal_tract_encoder::VoiceCognitiveState;
use super::vocal_tract_fep::StreamingVocalTract;

/// Motor frame rate (Hz). Each frame produces `sample_rate / FRAME_RATE` audio samples.
const FRAME_RATE: u32 = 200;

/// Motor frame timestep (seconds).
const DT: f32 = 1.0 / FRAME_RATE as f32;

/// Base phoneme duration (seconds) for G2P timing.
const BASE_PHONEME_DURATION: f32 = 0.06;

/// Evidence emitted by the explicit phonological-plan realization path.
#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonologicalPlanRealizationReceipt {
    /// Receipt schema, independent from the phonological-plan contract version.
    pub schema_version: u32,
    pub plan_version: String,
    pub plan_grounding_blake3: String,
    pub realization_authorized: bool,
    pub segment_count: usize,
    /// Exact per-segment motor-frame schedule derived from phoneme identity, stress,
    /// utterance-final position, rate, and explicitly encoded silence pause weight.
    pub segment_frame_counts: Vec<usize>,
    pub scheduler_frames: usize,
    pub sample_count: usize,
    pub sample_rate: u32,
    pub rate: f32,
    pub pitch_range: f32,
    pub prominence: f32,
    pub pause_weight: f32,
    pub audio_blake3: String,
}

#[cfg(feature = "ssm_language")]
impl PhonologicalPlanRealizationReceipt {
    /// Independently verify the receipt's plan-bound fields and internal sample accounting.
    pub fn verify_against_plan(&self, plan: &PhonologicalPlan) -> Result<()> {
        self.verify_against_plan_internal(plan, false)
    }

    fn verify_against_plan_internal(
        &self,
        plan: &PhonologicalPlan,
        allow_verified_lexical: bool,
    ) -> Result<()> {
        plan.validate()
            .map_err(|error| anyhow::anyhow!("invalid phonological plan: {error}"))?;

        if self.schema_version != 3 {
            anyhow::bail!("unsupported realization receipt schema: {}", self.schema_version);
        }
        if self.plan_version != plan.version {
            anyhow::bail!("realization receipt plan version does not match plan");
        }
        if !self.realization_authorized || !plan.realization_authorized {
            anyhow::bail!("realization receipt or plan is not authorized");
        }
        if matches!(plan.content_binding, ContentBindingStatus::LexicallyBound) && !allow_verified_lexical {
            anyhow::bail!(
                "lexically bound phonological plans require validated lexical-binding realization"
            );
        }

        let expected_grounding =
            blake3::hash(plan.grounding_surface().as_bytes()).to_hex().to_string();
        if self.plan_grounding_blake3 != expected_grounding {
            anyhow::bail!("realization receipt plan grounding does not match plan");
        }

        if self.segment_count != plan.segments.len()
            || self.rate != plan.rate
            || self.pitch_range != plan.pitch_range
            || self.prominence != plan.prominence
            || self.pause_weight != plan.pause_weight
        {
            anyhow::bail!("realization receipt plan fields do not match plan");
        }

        let expected_segment_frame_counts = plan
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| {
                let mut frames = predict_duration(
                    &segment.symbol,
                    segment.stress.ordinal(),
                    false,
                    index + 1 == plan.segments.len(),
                    plan.rate,
                );
                if segment.symbol.eq_ignore_ascii_case("SIL") {
                    frames = ((frames as f32) * (1.0 + plan.pause_weight)).round() as usize;
                }
                frames
            })
            .collect::<Vec<_>>();

        if self.segment_frame_counts != expected_segment_frame_counts {
            anyhow::bail!("realization receipt segment frame schedule does not match plan");
        }

        let expected_scheduler_frames = expected_segment_frame_counts.iter().copied().sum::<usize>();
        if self.scheduler_frames != expected_scheduler_frames {
            anyhow::bail!("realization receipt scheduler frame total does not match plan");
        }

        let samples_per_frame = (self.sample_rate / FRAME_RATE) as usize;
        let expected_samples = self.scheduler_frames.saturating_mul(samples_per_frame);
        if self.sample_rate < FRAME_RATE {
            anyhow::bail!(
                "realization receipt sample rate is below the motor-frame rate: {} < {}",
                self.sample_rate,
                FRAME_RATE
            );
        }
        if !self.sample_rate.is_multiple_of(FRAME_RATE) {
            anyhow::bail!(
                "realization receipt sample rate is not divisible by the motor-frame rate: {} % {} != 0",
                self.sample_rate,
                FRAME_RATE
            );
        }
        if self.sample_count != expected_samples || self.sample_count == 0 {
            anyhow::bail!("realization receipt sample accounting is inconsistent");
        }

        if !is_hex_digest(&self.audio_blake3) || !is_hex_digest(&self.plan_grounding_blake3) {
            anyhow::bail!("realization receipt hashes are malformed");
        }

        Ok(())
    }

    /// Verify the audio digest independently from the plan receipt.
    ///
    /// The digest is domain-separated and binds the authoritative sample-rate metadata
    /// together with the exact f32 sample buffer, so metadata-only rate tampering cannot
    /// preserve a previously valid audio receipt.
    pub fn verify_samples(&self, samples: &[f32]) -> bool {
        self.sample_rate >= FRAME_RATE
            && self.sample_count > 0
            && samples.len() == self.sample_count
            && is_hex_digest(&self.audio_blake3)
            && hash_audio_binding(self.sample_rate, samples) == self.audio_blake3
    }
}

#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedLexicalPhonologicalRealizationReceipt {
    /// Existing v3 realization receipt, preserving its independent plan/audio accounting.
    pub realization: PhonologicalPlanRealizationReceipt,
    /// Complete provenance for each embedded pronunciation resource used to derive the witness.
    ///
    /// An empty vector is intentional for a caller-supplied witness that was not derived from
    /// Symthaea's embedded pronunciation resources.
    pub pronunciation_lexicon_evidence: Vec<PronunciationLexiconEvidence>,
    /// Canonical digest binding the exact pronunciation-resource evidence list to this receipt.
    pub pronunciation_lexicon_evidence_blake3: String,
    /// Exact witness contract version.
    pub witness_version: String,
    /// Exact lexical-binding provenance carried by the witness.
    pub lexical_binding_provenance: String,
    /// Canonical witness identity used to authorize the lexical realization.
    pub witness_blake3: String,
}

/// A stronger realization receipt that records the morphophonological derivation witness
/// alongside the existing lexical-to-phonological realization receipt.
///
/// This is additive: ordinary lexical verification does not acquire a stronger claim merely
/// because this type exists. Callers must explicitly choose this receipt and admission path.
#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt {
    pub schema_version: u32,
    pub realization: VerifiedLexicalPhonologicalRealizationReceipt,
    pub morphophonological_witness: MorphophonologicalDerivationWitness,
    pub morphophonological_witness_version: String,
    pub morphophonological_witness_blake3: String,
    pub morphophonological_rule_set_blake3: String,
    /// Optional source-to-rule compilation provenance. Present only for the stronger
    /// compilation-backed admission path.
    pub compilation_witness: Option<MorphophonologicalCompilationWitness>,
}

#[cfg(feature = "ssm_language")]
impl MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt {
    pub const SCHEMA_VERSION: u32 = 2;

    pub fn verify_against_plan_and_rule_set(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
    ) -> Result<()> {
        if self.schema_version != Self::SCHEMA_VERSION {
            anyhow::bail!("morphophonological realization receipt schema version is unsupported");
        }
        self.realization
            .verify_against_plan(plan, frame, binding, lexical_witness)?;

        morphophonological_witness
            .validate_against_binding_and_rule_set(binding, rule_set)
            .map_err(|error| anyhow::anyhow!("invalid morphophonological witness: {error}"))?;

        if self.morphophonological_witness_version != morphophonological_witness.version {
            anyhow::bail!(
                "morphophonological realization receipt witness version does not match witness"
            );
        }
        if self.morphophonological_witness != *morphophonological_witness {
            anyhow::bail!(
                "morphophonological realization receipt witness does not match supplied witness"
            );
        }
        let expected_witness =
            blake3::hash(morphophonological_witness.grounding_surface().as_bytes())
                .to_hex()
                .to_string();
        if self.morphophonological_witness_blake3 != expected_witness {
            anyhow::bail!(
                "morphophonological realization receipt witness hash does not match witness"
            );
        }
        if self.morphophonological_rule_set_blake3 != rule_set.resource_blake3() {
            anyhow::bail!(
                "morphophonological realization receipt rule-set hash does not match executable rule set"
            );
        }

        if let Some(compilation_witness) = &self.compilation_witness {
            if compilation_witness.output_rule_set_blake3 != rule_set.resource_blake3() {
                anyhow::bail!(
                    "receipt compilation witness does not match executable rule-set identity"
                );
            }
            compilation_witness
                .validate_shape()
                .map_err(|error| {
                    anyhow::anyhow!("invalid optional morphophonological compilation witness: {error}")
                })?;
        }

        Ok(())
    }

    pub fn verify_against_plan_and_rule_set_and_current_resources(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        g2p: &SimpleG2P,
    ) -> Result<()> {
        self.realization
            .verify_against_plan_and_current_resources(
                plan,
                frame,
                binding,
                lexical_witness,
                g2p,
            )?;
        self.verify_against_plan_and_rule_set(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )
    }

    pub fn verify_against_plan_and_rule_set_with_compilation_witness(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        compilation_witness: &MorphophonologicalCompilationWitness,
    ) -> Result<()> {
        self.verify_against_plan_and_rule_set(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )?;
        if self.compilation_witness.as_ref() != Some(compilation_witness) {
            anyhow::bail!(
                "morphophonological realization receipt compilation witness does not match supplied witness"
            );
        }
        compilation_witness
            .validate_shape()
            .map_err(|error| anyhow::anyhow!("invalid morphophonological compilation witness: {error}"))?;
        if compilation_witness.output_rule_set_blake3 != rule_set.resource_blake3() {
            anyhow::bail!(
                "morphophonological compilation witness output does not match executable rule set"
            );
        }
        Ok(())
    }

    pub fn verify_against_plan_and_rule_set_with_source_artifact(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        source_artifact: &[u8],
    ) -> Result<()> {
        self.verify_against_plan_and_rule_set(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )?;
        let compilation_witness = self.compilation_witness.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "source-artifact verification requires a persisted compilation witness"
            )
        })?;
        if compilation_witness.output_rule_set_blake3 != rule_set.resource_blake3() {
            anyhow::bail!(
                "source-artifact compilation witness output does not match executable rule set"
            );
        }
        morphophonological_witness
            .validate_against_binding_and_rule_set_with_source_artifact(
                binding,
                rule_set,
                source_artifact,
            )
            .map_err(|error| {
                anyhow::anyhow!("invalid morphophonological source artifact: {error}")
            })?;
        Ok(())
    }
}

#[cfg(feature = "ssm_language")]
impl VerifiedLexicalPhonologicalRealizationReceipt {
    pub fn verify_against_plan(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        witness: &LexicalPhonologicalWitness,
    ) -> Result<()> {
        binding
            .validate_against_frame(frame)
            .map_err(|_| anyhow::anyhow!("invalid lexical binding"))?;
        plan.validate_against_lexical_binding_and_witness(frame, binding, witness)
            .map_err(|error| anyhow::anyhow!("invalid lexical realization witness: {error}"))?;

        if self.witness_version != witness.version {
            anyhow::bail!("verified realization receipt witness version does not match witness");
        }
        for evidence in &self.pronunciation_lexicon_evidence {
            if !evidence.is_well_formed() {
                anyhow::bail!(
                    "verified realization receipt contains malformed pronunciation evidence"
                );
            }

            match evidence.source_id.as_str() {
                "symthaea-hand-lexicon-v1" => {
                    if evidence.dialect_scope != "en-unspecified"
                        || evidence.variant_policy != "single-curated-entry"
                        || evidence.selected_variant != "only-entry"
                        || evidence.available_variants != 1
                    {
                        anyhow::bail!(
                            "hand-lexicon pronunciation evidence has unsupported scope or variant semantics"
                        );
                    }
                }
                "cmudict-embedded-v1" => {
                    if evidence.dialect_scope != "en-US"
                        || evidence.variant_policy != "primary-un-suffixed-entry"
                        || evidence.selected_variant != "primary"
                        || evidence.available_variants == 0
                    {
                        anyhow::bail!(
                            "CMUdict pronunciation evidence has unsupported scope or variant semantics"
                        );
                    }
                }
                _ => anyhow::bail!(
                    "verified realization receipt contains an unsupported pronunciation-lexicon source"
                ),
            }
        }

        let expected_lexicon_evidence =
            hash_pronunciation_lexicon_evidence(&self.pronunciation_lexicon_evidence);
        if self.pronunciation_lexicon_evidence_blake3 != expected_lexicon_evidence {
            anyhow::bail!(
                "verified realization receipt pronunciation-resource evidence digest does not match evidence"
            );
        }

        if self.lexical_binding_provenance != binding.provenance_token() {
            anyhow::bail!(
                "verified realization receipt lexical-binding provenance does not match binding"
            );
        }

        let expected_witness =
            blake3::hash(witness.grounding_surface().as_bytes()).to_hex().to_string();
        if self.witness_blake3 != expected_witness {
            anyhow::bail!("verified realization receipt witness hash does not match witness");
        }

        self.realization.verify_against_plan_internal(plan, true)
    }

    pub fn verify_samples(&self, samples: &[f32]) -> bool {
        self.realization.verify_samples(samples)
    }

    /// Verify this receipt against the pronunciation resources currently embedded in a
    /// verifier process.
    ///
    /// The ordinary receipt verifier is deliberately detached: it preserves historical
    /// evidence inspectability even after an embedded resource update. This method adds
    /// current-resource identity and re-derives the exact strict lexical realization when
    /// resource evidence is present.
    pub fn verify_against_plan_and_current_resources(
        &self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        witness: &LexicalPhonologicalWitness,
        g2p: &SimpleG2P,
    ) -> Result<()> {
        self.verify_against_plan(plan, frame, binding, witness)?;

        if self.pronunciation_lexicon_evidence.is_empty() {
            return Ok(());
        }

        let (derived_witness, derived_evidence) =
            derive_english_lexical_phonological_witness_from_g2p(g2p, binding, &plan.segments)?;
        if derived_witness != *witness {
            anyhow::bail!(
                "current-resource pronunciation derivation does not reproduce the retained witness"
            );
        }
        if derived_evidence != self.pronunciation_lexicon_evidence {
            anyhow::bail!(
                "current-resource pronunciation derivation does not reproduce the retained resource evidence"
            );
        }

        for evidence in &self.pronunciation_lexicon_evidence {
            g2p.verify_pronunciation_lexicon_evidence(evidence)?;
        }

        Ok(())
    }
}

#[cfg(feature = "ssm_language")]
fn hash_pronunciation_lexicon_evidence(
    evidence: &[PronunciationLexiconEvidence],
) -> String {
    let mut canonical = evidence.to_vec();
    canonical.sort_by(|left, right| {
        (
            &left.source_id,
            &left.dialect_scope,
            &left.variant_policy,
            &left.selected_variant,
            left.available_variants,
            &left.resource_blake3,
        )
            .cmp(&(
                &right.source_id,
                &right.dialect_scope,
                &right.variant_policy,
                &right.selected_variant,
                right.available_variants,
                &right.resource_blake3,
            ))
    });

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-pronunciation-lexicon-evidence-v1\\0");
    for item in canonical {
        hasher.update(&(item.source_id.len() as u64).to_le_bytes());
        hasher.update(item.source_id.as_bytes());
        hasher.update(&(item.dialect_scope.len() as u64).to_le_bytes());
        hasher.update(item.dialect_scope.as_bytes());
        hasher.update(&(item.variant_policy.len() as u64).to_le_bytes());
        hasher.update(item.variant_policy.as_bytes());
        hasher.update(&(item.selected_variant.len() as u64).to_le_bytes());
        hasher.update(item.selected_variant.as_bytes());
        hasher.update(&(item.available_variants as u64).to_le_bytes());
        hasher.update(&(item.resource_blake3.len() as u64).to_le_bytes());
        hasher.update(item.resource_blake3.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(feature = "ssm_language")]
fn hash_audio_binding(sample_rate: u32, samples: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-phonological-plan-audio-v1\\0");
    hasher.update(&sample_rate.to_le_bytes());
    for sample in samples {
        hasher.update(&sample.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(feature = "ssm_language")]
fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Handle to a background `speak_async()` call.
///
/// Dropping the handle does NOT stop playback — call [`SpeakHandle::stop()`] explicitly,
/// or use [`SpeakHandle::join()`] to wait for completion.
pub struct SpeakHandle {
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    speaking: Arc<AtomicBool>,
}

impl SpeakHandle {
    /// Stop the background utterance. The ring buffer drains naturally to silence.
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
    }

    /// Whether the background thread is still synthesizing.
    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    /// Block until the utterance finishes (or is stopped).
    pub fn join(mut self) -> Result<()> {
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("speak thread panicked"))?
        } else {
            Ok(())
        }
    }
}

/// Real-time streaming voice: text → phonemes → synthesis → speaker output.
///
/// # Example
///
/// ```no_run
/// # use symthaea_core::genesis::GenesisSeed;
/// # use symthaea::voice::live_voice::LiveVoice;
/// let genesis = GenesisSeed::from_phrase("my-voice");
/// let mut voice = LiveVoice::new(&genesis).unwrap();
/// voice.speak("hello world").unwrap();
/// ```
pub struct LiveVoice {
    streaming: StreamingVocalTract,
    audio: AudioOutput,
    g2p: SimpleG2P,
    formant_db: FormantDatabase,
    /// Shared cognitive state — can be updated from another thread mid-utterance.
    cognitive_state: Arc<parking_lot::Mutex<VoiceCognitiveState>>,
    speaking: Arc<AtomicBool>,
    genesis: GenesisSeed,
}

#[cfg(feature = "ssm_language")]
fn derive_english_lexical_phonological_witness_from_g2p(
    g2p: &SimpleG2P,
    binding: &LexicalMorphosyntacticBinding,
    segments: &[symthaea_broca::PhonemeSlot],
) -> Result<(LexicalPhonologicalWitness, Vec<PronunciationLexiconEvidence>)> {

    binding
        .validate()
        .map_err(|error| anyhow::anyhow!("invalid lexical binding: {error}"))?;

    if !binding.language.language_tag.eq_ignore_ascii_case("en") {
        anyhow::bail!(
            "strict embedded pronunciation witness currently supports only language tag en"
        );
    }
    if segments.is_empty() {
        anyhow::bail!("strict lexical pronunciation witness requires phonological segments");
    }

    let mut mappings = Vec::with_capacity(binding.constituents.len());
    let mut cursor = 0usize;
    let mut pronunciation_lexicon_evidence = Vec::new();

    for (lexical_position, constituent) in binding.constituents.iter().enumerate() {
        let form = constituent
            .morphophonological_form
            .as_deref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "lexical position {lexical_position} has no morphophonological form"
                )
            })?;

        let (expected_phones, evidence) = g2p
            .word_to_phonemes_from_lexicon_with_evidence(form)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "lexical position {lexical_position} form {form:?} has no embedded pronunciation"
                )
            })?;
        pronunciation_lexicon_evidence.push(evidence);

        while segments
            .get(cursor)
            .is_some_and(|segment| segment.symbol.eq_ignore_ascii_case("SIL"))
        {
            cursor += 1;
        }

        let start = cursor;
        let mut symbols = Vec::with_capacity(expected_phones.len());
        for (offset, expected_phone) in expected_phones.iter().enumerate() {
            let segment = segments.get(cursor).ok_or_else(|| {
                anyhow::anyhow!(
                    "lexical position {lexical_position} requires pronunciation segment offset {offset}, but the stream ended"
                )
            })?;
            if segment.symbol.eq_ignore_ascii_case("SIL") {
                anyhow::bail!(
                    "explicit SIL occurred inside lexical position {lexical_position}"
                );
            }

            let expected_base = expected_phone
                .trim_end_matches(|c: char| c.is_ascii_digit());
            let actual_base = segment
                .symbol
                .trim_end_matches(|c: char| c.is_ascii_digit());
            if expected_base != actual_base {
                anyhow::bail!(
                    "lexical position {lexical_position} pronunciation mismatch at segment {cursor}: expected {expected_phone}, got {}",
                    segment.symbol
                );
            }

            if let Some(stress_digit) =
                expected_phone.chars().last().filter(|c| c.is_ascii_digit())
            {
                let expected_stress = match stress_digit {
                    '0' => 0,
                    '1' => 1,
                    '2' => 2,
                    _ => unreachable!("digit filter limits pronunciation stress to ASCII digits"),
                };
                if segment.stress.ordinal() != expected_stress {
                    anyhow::bail!(
                        "lexical position {lexical_position} stress mismatch at segment {cursor}: expected {expected_stress}, got {}",
                        segment.stress.ordinal()
                    );
                }
            }

            symbols.push(segment.symbol.clone());
            cursor += 1;
        }

        mappings.push(
            LexicalPhonologicalMapping::from_lexical_binding(
                binding,
                lexical_position,
                (start..cursor).collect(),
                symbols,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "lexical position {lexical_position} disappeared during witness derivation"
                )
            })?,
        );
    }

    if segments[cursor..]
        .iter()
        .any(|segment| !segment.symbol.eq_ignore_ascii_case("SIL"))
    {
        anyhow::bail!(
            "phonological stream contains non-silence segments not accounted for by the embedded pronunciation witness"
        );
    }

    let witness = LexicalPhonologicalWitness::new(binding, mappings)
        .map_err(|error| anyhow::anyhow!("invalid derived lexical witness: {error}"))?;
    pronunciation_lexicon_evidence.sort_by(|left, right| {
        (
            &left.source_id,
            &left.dialect_scope,
            &left.variant_policy,
            &left.selected_variant,
            left.available_variants,
            &left.resource_blake3,
        )
            .cmp(&(
                &right.source_id,
                &right.dialect_scope,
                &right.variant_policy,
                &right.selected_variant,
                right.available_variants,
                &right.resource_blake3,
            ))
    });
    pronunciation_lexicon_evidence.dedup();

    Ok((witness, pronunciation_lexicon_evidence))
}
impl LiveVoice {
    /// Create all components, train the controller, and open audio output.
    ///
    /// Training runs 30 epochs on the full FormantDatabase (~43 phonemes).
    pub fn new(genesis: &GenesisSeed) -> Result<Self> {
        let audio = AudioOutput::new()?;
        let sample_rate = audio.sample_rate();

        let mut streaming = StreamingVocalTract::new(genesis, sample_rate, FRAME_RATE);

        let db = FormantDatabase::new();
        train_controller_on_phoneme_db(&mut streaming.pipeline.controller, genesis, &db, 30);

        Ok(Self {
            streaming,
            audio,
            g2p: SimpleG2P::new(),
            formant_db: db,
            cognitive_state: Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default())),
            speaking: Arc::new(AtomicBool::new(false)),
            genesis: genesis.clone(),
        })
    }

    /// Create a LiveVoice without an audio device (for `speak_to_file()` only).
    ///
    /// Uses a default sample rate of 24000 Hz.
    pub fn new_headless(genesis: &GenesisSeed) -> Self {
        Self::new_headless_with_rate(genesis, 24000)
    }

    /// Create a headless LiveVoice with a specific sample rate.
    pub fn new_headless_with_rate(genesis: &GenesisSeed, sample_rate: u32) -> Self {
        let mut streaming = StreamingVocalTract::new(genesis, sample_rate, FRAME_RATE);

        let db = FormantDatabase::new();
        train_controller_on_phoneme_db(&mut streaming.pipeline.controller, genesis, &db, 30);

        // AudioOutput::new() would fail headless, so we create a dummy.
        // speak() and speak_async() will fail if called, but speak_to_file() works.
        // We use a separate struct field to track this.
        Self {
            streaming,
            audio: AudioOutput::new_dummy(sample_rate),
            g2p: SimpleG2P::new(),
            formant_db: db,
            cognitive_state: Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default())),
            speaking: Arc::new(AtomicBool::new(false)),
            genesis: genesis.clone(),
        }
    }

    /// Derive an auditable English lexical-to-phonological witness using only embedded
    /// pronunciation lexicons.
    ///
    /// The general SimpleG2P spelling-rule fallback is deliberately excluded. Every lexical
    /// constituent must have a pronunciation entry, and the supplied phonological stream must
    /// match it in order. CMU stress digits are checked against the plan's explicit stress field.
    #[cfg(feature = "ssm_language")]
    pub fn derive_english_lexical_phonological_witness(
        &self,
        binding: &LexicalMorphosyntacticBinding,
        segments: &[symthaea_broca::PhonemeSlot],
    ) -> Result<(LexicalPhonologicalWitness, Vec<PronunciationLexiconEvidence>)> {
        derive_english_lexical_phonological_witness_from_g2p(&self.g2p, binding, segments)
    }
    /// Realize a plan by deriving its lexical-to-phonological witness strictly from the
    /// embedded English pronunciation resources. Unlisted forms fail closed instead of
    /// falling back to spelling heuristics.
    #[cfg(feature = "ssm_language")]
    pub fn speak_english_lexicon_verified_lexical_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
    ) -> Result<VerifiedLexicalPhonologicalRealizationReceipt> {
        plan.validate_against_frame(frame)
            .map_err(|error| anyhow::anyhow!("invalid phonological plan lineage: {error}"))?;
        let (witness, pronunciation_lexicon_evidence) =
            self.derive_english_lexical_phonological_witness(binding, &plan.segments)?;
        let mut receipt = self
            .speak_verified_lexical_phonological_plan_with_receipt(
                plan, frame, binding, &witness,
            )?;
        receipt.pronunciation_lexicon_evidence = pronunciation_lexicon_evidence;
        receipt.pronunciation_lexicon_evidence_blake3 =
            hash_pronunciation_lexicon_evidence(&receipt.pronunciation_lexicon_evidence);
        receipt.verify_against_plan_and_current_resources(
            plan, frame, binding, &witness, &self.g2p
        )?;
        Ok(receipt)
    }

    /// Synthesize an explicit phonological plan through the live audio output path.
    ///
    /// This is intentionally plan-native: no text, G2P reconstruction, or lexical inference
    /// occurs here. The validated plan supplies explicit phoneme identity, stress, rate,
    /// phrase boundaries, and prosodic intent. The vocal-tract controller supplies
    /// speaker/anatomical parameters. Word-final timing remains unasserted because the current
    /// phonological contract does not encode word boundaries.
    ///
    /// The current implementation pre-synthesizes the validated plan before enqueueing it to
    /// the audio sink, so this method does not claim first-audio latency.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan(&mut self, plan: &PhonologicalPlan) -> Result<usize> {
        Ok(self.speak_phonological_plan_with_receipt(plan)?.sample_count)
    }

    /// Realize an explicit phonological plan and return an evidence receipt that binds
    /// the validated plan grounding to the generated audio buffer.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
    ) -> Result<PhonologicalPlanRealizationReceipt> {
        let (samples, receipt) = self.synthesize_phonological_plan(plan)?;
        self.push_with_backpressure(&samples);
        Ok(receipt)
    }

    /// Synthesize an explicit phonological plan directly to WAV without an audio device.
    ///
    /// The same scheduler-owned duration calculation is used by the real-time path.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan_to_file(
        &mut self,
        plan: &PhonologicalPlan,
        path: &Path,
    ) -> Result<usize> {
        let (samples, receipt) = self.synthesize_phonological_plan(plan)?;
        write_wav(path, &samples, receipt.sample_rate)?;
        Ok(receipt.sample_count)
    }

    /// Stronger admission path: also requires an explicit source-to-rule compilation witness.
    #[cfg(feature = "ssm_language")]
    pub fn speak_morphophonology_compilation_verified_lexical_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        compilation_witness: &MorphophonologicalCompilationWitness,
    ) -> Result<MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt> {
        compilation_witness
            .validate_against_source_artifact_and_rule_set(
                &[],
                rule_set,
            )
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "source-to-rule compilation witness requires exact source artifact bytes"
                )
            })?;

        let receipt = self.speak_morphophonology_verified_lexical_phonological_plan_with_receipt(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )?;
        let mut receipt = receipt;
        receipt.compilation_witness = Some(compilation_witness.clone());
        receipt
            .verify_against_plan_and_rule_set_with_compilation_witness(
                plan,
                frame,
                binding,
                lexical_witness,
                morphophonological_witness,
                rule_set,
                compilation_witness,
            )?;
        Ok(receipt)
    }

    /// Source-artifact-backed compilation admission. Exact source bytes are checked before
    /// the underlying morphophonological realization is allowed to synthesize/enqueue audio.
    #[cfg(feature = "ssm_language")]
    pub fn speak_morphophonology_compilation_verified_lexical_phonological_plan_with_source_artifact_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        compilation_witness: &MorphophonologicalCompilationWitness,
        source_artifact: &[u8],
    ) -> Result<MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt> {
        compilation_witness
            .validate_against_source_artifact_and_rule_set(source_artifact, rule_set)
            .map_err(|error| {
                anyhow::anyhow!("invalid morphophonological compilation admission: {error}")
            })?;

        let mut receipt =
            self.speak_morphophonology_verified_lexical_phonological_plan_with_receipt(
                plan,
                frame,
                binding,
                lexical_witness,
                morphophonological_witness,
                rule_set,
            )?;
        receipt.compilation_witness = Some(compilation_witness.clone());
        receipt
            .verify_against_plan_and_rule_set_with_compilation_witness(
                plan,
                frame,
                binding,
                lexical_witness,
                morphophonological_witness,
                rule_set,
                compilation_witness,
            )?;
        Ok(receipt)
    }

    /// Stronger admission path: requires a morphophonological witness backed by an
    /// executable rule set before synthesis/audio enqueue occurs.
    #[cfg(feature = "ssm_language")]
    pub fn speak_morphophonology_verified_lexical_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
    ) -> Result<MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt> {
        morphophonological_witness
            .validate_against_binding_and_rule_set(binding, rule_set)
            .map_err(|error| {
                anyhow::anyhow!("invalid morphophonological admission witness: {error}")
            })?;

        let realization = self.speak_verified_lexical_phonological_plan_with_receipt(
            plan,
            frame,
            binding,
            lexical_witness,
        )?;

        let receipt = MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt {
            schema_version: MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt::SCHEMA_VERSION,
            realization,
            morphophonological_witness: morphophonological_witness.clone(),
            morphophonological_witness_version: morphophonological_witness.version.clone(),
            morphophonological_witness_blake3: morphophonological_witness.provenance_token(),
            morphophonological_rule_set_blake3: rule_set.resource_blake3(),
            compilation_witness: None,
        };

        receipt.verify_against_plan_and_rule_set(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )?;
        Ok(receipt)
    }

    /// Source-artifact-backed variant: the exact source bytes are checked before synthesis.
    #[cfg(feature = "ssm_language")]
    pub fn speak_morphophonology_verified_lexical_phonological_plan_with_source_artifact_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        lexical_witness: &LexicalPhonologicalWitness,
        morphophonological_witness: &MorphophonologicalDerivationWitness,
        rule_set: &MorphophonologicalRuleSet,
        source_artifact: &[u8],
    ) -> Result<MorphophonologicalVerifiedLexicalPhonologicalRealizationReceipt> {
        morphophonological_witness
            .validate_against_binding_and_rule_set_with_source_artifact(
                binding,
                rule_set,
                source_artifact,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "invalid morphophonological source-artifact admission: {error}"
                )
            })?;

        let receipt = self.speak_morphophonology_verified_lexical_phonological_plan_with_receipt(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
        )?;

        receipt.verify_against_plan_and_rule_set_with_source_artifact(
            plan,
            frame,
            binding,
            lexical_witness,
            morphophonological_witness,
            rule_set,
            source_artifact,
        )?;
        Ok(receipt)
    }

    /// Realize a lexicalized plan only after the exact lexical binding and realization witness
    /// have been independently validated.
    ///
    /// The witness is retained by the caller and its canonical identity is included in the
    /// returned evidence wrapper; the underlying v3 realization receipt remains unchanged.
    #[cfg(feature = "ssm_language")]
    pub fn speak_verified_lexical_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
        frame: &LinguisticFrame,
        binding: &LexicalMorphosyntacticBinding,
        witness: &LexicalPhonologicalWitness,
    ) -> Result<VerifiedLexicalPhonologicalRealizationReceipt> {
        binding
            .validate_against_frame(frame)
            .map_err(|_| anyhow::anyhow!("invalid lexical binding"))?;
        plan.validate_against_lexical_binding_and_witness(frame, binding, witness)
            .map_err(|error| anyhow::anyhow!("invalid lexical realization witness: {error}"))?;

        let (samples, realization) =
            self.synthesize_phonological_plan_with_admission(plan, true)?;
        self.push_with_backpressure(&samples);

        let pronunciation_lexicon_evidence = Vec::new();
        Ok(VerifiedLexicalPhonologicalRealizationReceipt {
            realization,
            pronunciation_lexicon_evidence_blake3 =
                hash_pronunciation_lexicon_evidence(&pronunciation_lexicon_evidence),
            pronunciation_lexicon_evidence,
            witness_version: witness.version.clone(),
            lexical_binding_provenance: binding.provenance_token(),
            witness_blake3: blake3::hash(witness.grounding_surface().as_bytes())
                .to_hex()
                .to_string(),
        })
    }

    #[cfg(feature = "ssm_language")]
    fn pitch_accent_for_plan(segment_is_focus: bool, prominence: f32) -> PitchAccent {
        if segment_is_focus && prominence >= 0.82 {
            PitchAccent::RiseHigh
        } else if prominence >= 0.62 {
            PitchAccent::High
        } else {
            PitchAccent::None
        }
    }

#[cfg(feature = "ssm_language")]
fn progress_within_frame_span(global_frame: usize, start_frame: usize, end_frame: usize) -> f32 {
    let span = end_frame.saturating_sub(start_frame);
    if span <= 1 {
        0.0
    } else {
        global_frame
            .saturating_sub(start_frame)
            .min(span.saturating_sub(1)) as f32
            / (span.saturating_sub(1)) as f32
    }
}

    #[cfg(feature = "ssm_language")]
    fn synthesize_phonological_plan(
        &mut self,
        plan: &PhonologicalPlan,
    ) -> Result<(Vec<f32>, PhonologicalPlanRealizationReceipt)> {
        self.synthesize_phonological_plan_with_admission(plan, false)
    }

    #[cfg(feature = "ssm_language")]
    fn synthesize_phonological_plan_with_admission(
        &mut self,
        plan: &PhonologicalPlan,
        allow_verified_lexical: bool,
    ) -> Result<(Vec<f32>, PhonologicalPlanRealizationReceipt)> {
        plan.validate()
            .map_err(|error| anyhow::anyhow!("invalid phonological plan: {error}"))?;
        if matches!(plan.content_binding, ContentBindingStatus::LexicallyBound)
            && !allow_verified_lexical
        {
            anyhow::bail!(
                "lexically bound phonological plans require validated lexical-binding realization"
            );
        }
        if !plan.ready_for_realization() {
            anyhow::bail!("phonological plan is not ready for realization");
        }
        let sample_rate = self.sample_rate();
        if sample_rate < FRAME_RATE {
            anyhow::bail!(
                "phonological plan realization requires sample rate >= {} Hz; got {}",
                FRAME_RATE,
                sample_rate
            );
        }
        if !sample_rate.is_multiple_of(FRAME_RATE) {
            anyhow::bail!(
                "phonological plan realization requires sample rate divisible by {} Hz; got {}",
                FRAME_RATE,
                sample_rate
            );
        }

        // The current vocal-tract realization surface consumes canonical ARPABET symbols
        // (plus explicit SIL). Never silently route an unsupported plan symbol to the
        // vocal-tract's generic silence fallback: that would change the requested
        // phonological content while still emitting a successful receipt.
        for segment in &plan.segments {
            if !segment.symbol.eq_ignore_ascii_case("SIL")
                && matches!(phoneme_manner_class(&segment.symbol), MannerClass::Silence)
            {
                anyhow::bail!(
                    "phonological plan contains unsupported realization symbol: {}",
                    segment.symbol
                );
            }
        }

        let scheduled_frames_for = |index: usize, segment: &symthaea_broca::PhonemeSlot| {
            let mut frames = predict_duration(
                &segment.symbol,
                segment.stress.ordinal(),
                false,
                index + 1 == plan.segments.len(),
                plan.rate,
            );
            if segment.symbol.eq_ignore_ascii_case("SIL") {
                frames = ((frames as f32) * (1.0 + plan.pause_weight)).round() as usize;
            }
            frames
        };

        let segment_frame_counts = plan
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| scheduled_frames_for(index, segment))
            .collect::<Vec<_>>();

        let mut segment_frame_offsets = Vec::with_capacity(segment_frame_counts.len() + 1);
        segment_frame_offsets.push(0);
        for &frames in &segment_frame_counts {
            let next = segment_frame_offsets.last().copied().unwrap_or(0).saturating_add(frames);
            segment_frame_offsets.push(next);
        }
        let total_frames = *segment_frame_offsets.last().unwrap_or(&0);

        let samples_per_frame = (self.sample_rate() / FRAME_RATE.max(1)) as usize;
        let mut all_samples =
            Vec::with_capacity(total_frames.saturating_mul(samples_per_frame.max(1)));

        let last_index = plan.segments.len().saturating_sub(1);
        for (index, segment) in plan.segments.iter().enumerate() {
            // Pause weight is only realized when the phonological plan explicitly encodes
            // a silence segment; this prevents inventing pause locations from an abstract
            // scalar alone.
            let frames = segment_frame_counts[index];

            let phoneme = if segment.symbol.eq_ignore_ascii_case("SIL") {
                None
            } else {
                Some(segment.symbol.as_str())
            };
            let state = self.cognitive_state.lock().clone();
            let segment_count = plan.segments.len();
            let phrase_index = plan.segments[..index]
                .iter()
                .filter(|slot| slot.phrase_boundary_after)
                .count()
                .min(u8::MAX as usize) as u8;
            let phrase_start = plan.segments[..index]
                .iter()
                .rposition(|slot| slot.phrase_boundary_after)
                .map(|boundary| boundary + 1)
                .unwrap_or(0);
            let phrase_end = plan.segments[index..]
                .iter()
                .position(|slot| slot.phrase_boundary_after)
                .map(|offset| index + offset)
                .unwrap_or(segment_count.saturating_sub(1));
            let phrase_start_frame = segment_frame_offsets
                .get(phrase_start)
                .copied()
                .unwrap_or(0);
            let phrase_end_frame = segment_frame_offsets
                .get(phrase_end.saturating_add(1))
                .copied()
                .unwrap_or(total_frames);
            let phrase_span_frames = phrase_end_frame
                .saturating_sub(phrase_start_frame)
                .max(1);
            let segment_start_frame = segment_frame_offsets
                .get(index)
                .copied()
                .unwrap_or(0);
            let syllable_start = plan.segments[..=index]
                .iter()
                .rposition(|slot| slot.syllable_index != segment.syllable_index)
                .map(|previous| previous + 1)
                .unwrap_or(0);
            let syllable_end = plan.segments[index..]
                .iter()
                .position(|slot| slot.syllable_index != segment.syllable_index)
                .map(|offset| index + offset)
                .unwrap_or(segment_count);
            let syllable_start_frame = segment_frame_offsets
                .get(syllable_start)
                .copied()
                .unwrap_or(0);
            let syllable_end_frame = segment_frame_offsets
                .get(syllable_end)
                .copied()
                .unwrap_or(total_frames);
            let intonation = match plan.intonation {
                symthaea_broca::IntonationIntent::Statement => Intonation::Statement,
                symthaea_broca::IntonationIntent::Question => Intonation::Question,
                symthaea_broca::IntonationIntent::Exclamation => Intonation::Exclamation,
            };

            for frame_index in 0..frames {
                let progress = if frames > 1 {
                    frame_index as f32 / (frames - 1) as f32
                } else {
                    0.0
                };
                let global_frame = segment_start_frame.saturating_add(frame_index);
                let utterance_progress = if total_frames > 1 {
                    global_frame as f32 / (total_frames - 1) as f32
                } else {
                    0.0
                };
                let phrase_progress = if phrase_span_frames > 1 {
                    global_frame.saturating_sub(phrase_start_frame) as f32
                        / (phrase_span_frames - 1) as f32
                } else {
                    0.0
                };

                let syllable_progress = progress_within_frame_span(
                    global_frame,
                    syllable_start_frame,
                    syllable_end_frame,
                );
                let prosody = ProsodyContext {
                    utterance_progress: utterance_progress.clamp(0.0, 1.0),
                    phoneme_progress: progress,
                    stress: segment.stress.ordinal(),
                    // Keep planned pitch range and speaker-state arousal as separate
                    // prosody controls. The pitch range never overwrites authoritative base F0.
                    base_f0: self.streaming.base_f0(),
                    arousal: state.emotional_arousal.clamp(0.0, 1.0),
                    pitch_range: plan.pitch_range,
                    intonation,
                    phrase_index,
                    phrase_progress: phrase_progress.clamp(0.0, 1.0),
                    is_focus: segment.is_focus && plan.focus_role.is_some(),
                    pitch_accent: Self::pitch_accent_for_plan(segment.is_focus, plan.prominence),
                    is_syllable_onset: segment.is_syllable_onset,
                    syllable_progress,
                    prev_source_type: None,
                    next_source_type: None,
                };
                let mut chunk = self
                    .streaming
                    .tick_with_prosody(&state, None, DT, phoneme, &prosody);
                if phoneme.is_none() {
                    // An explicit SIL slot is a phonological timing decision, not merely an
                    // instruction to advance the motor clock. Force its emitted PCM to zero so
                    // pause-weight evidence cannot hide residual voiced/noise energy.
                    chunk.fill(0.0);
                }
                all_samples.extend_from_slice(&chunk);
            }
        }

        let plan_grounding = plan.grounding_surface();
        let sample_rate = self.streaming.vocoder.sample_rate();
        let audio_blake3 = hash_audio_binding(sample_rate, &all_samples);
        let receipt = PhonologicalPlanRealizationReceipt {
            schema_version: 3,
            plan_version: plan.version.clone(),
            plan_grounding_blake3: blake3::hash(plan_grounding.as_bytes())
                .to_hex()
                .to_string(),
            realization_authorized: plan.realization_authorized,
            segment_count: plan.segments.len(),
            segment_frame_counts: segment_frame_counts.clone(),
            scheduler_frames: total_frames,
            sample_count: all_samples.len(),
            sample_rate: self.streaming.vocoder.sample_rate(),
            rate: plan.rate,
            pitch_range: plan.pitch_range,
            prominence: plan.prominence,
            pause_weight: plan.pause_weight,
            audio_blake3,
        };

        Ok((all_samples, receipt))
    }

    /// Speak text in real time with enhanced prosody control
    pub fn speak(&mut self, text: &str) -> Result<()> {
        self.speaking.store(true, Ordering::SeqCst);

        // Enhanced text analysis
        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let prosody = self.analyze_prosody(text);

        for timed in &phonemes {
            if !self.speaking.load(Ordering::SeqCst) {
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            // Apply prosody modulation
            let mut state = self.cognitive_state.lock().clone();
            self.apply_prosody(&mut state, &prosody);

            for _ in 0..n_frames {
                if !self.speaking.load(Ordering::SeqCst) {
                    break;
                }

                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                self.push_with_backpressure(&chunk);
            }
        }

        self.speaking.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn analyze_prosody(&self, text: &str) -> ProsodyAnalysis {
        // Analyze sentence structure, emphasis, etc.
        ProsodyAnalysis {
            pitch_range: 1.0,
            speaking_rate: 1.0,
            emphasis: Vec::new(),
        }
    }

    fn apply_prosody(&mut self, state: &mut VoiceCognitiveState, prosody: &ProsodyAnalysis) {
        state.emotional_arousal = prosody.pitch_range.clamp(0.0, 1.0);
        self.modulate_tau(1.0 / prosody.speaking_rate);
    }

    /// Speak text on a background thread. Returns a [`SpeakHandle`] for control.
    ///
    /// The cognitive loop can continue running while speech plays. Use
    /// [`cognitive_state_handle()`](Self::cognitive_state_handle) to modulate prosody mid-utterance.
    ///
    /// # Note
    /// This takes `&mut self` to ensure exclusive synthesis access, then moves
    /// the necessary state into the thread. Only one `speak_async` at a time.
    pub fn speak_async(&mut self, text: &str) -> SpeakHandle {
        self.speaking.store(true, Ordering::SeqCst);

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let speaking = Arc::clone(&self.speaking);
        let cog_state = Arc::clone(&self.cognitive_state);

        // Synthesize frames into a buffer on a dedicated thread.
        // We can't move `self` into the thread, so we pre-synthesize all audio.
        let mut all_samples = Vec::new();
        for timed in &phonemes {
            if !speaking.load(Ordering::SeqCst) {
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            for _ in 0..n_frames {
                if !speaking.load(Ordering::SeqCst) {
                    break;
                }

                let state = cog_state.lock().clone();
                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                all_samples.extend_from_slice(&chunk);
            }
        }

        // Push synthesized audio to the ring buffer on a background thread
        // (backpressure may block, so we don't want to block the caller)
        let speaking_bg = Arc::clone(&self.speaking);
        let mut audio = self.audio.take_producer();

        let thread = std::thread::Builder::new()
            .name("live-voice-push".into())
            .spawn(move || {
                let mut offset = 0;
                while offset < all_samples.len() {
                    if !speaking_bg.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Some(ref mut producer) = audio {
                        let written = push_samples_to_producer(producer, &all_samples[offset..]);
                        offset += written;
                        if offset < all_samples.len() {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    } else {
                        break;
                    }
                }
                speaking_bg.store(false, Ordering::SeqCst);
                Ok(())
            })
            .expect("failed to spawn speak thread");

        SpeakHandle {
            thread: Some(thread),
            speaking: Arc::clone(&self.speaking),
        }
    }

    /// Synthesize text to a WAV file. No audio device needed.
    ///
    /// Uses the same G2P → frame-by-frame synthesis pipeline as `speak()`,
    /// but collects all samples and writes them to disk via `hound`.
    pub fn speak_to_file(&mut self, text: &str, path: &Path) -> Result<usize> {
        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let mut all_samples = Vec::new();

        for timed in &phonemes {
            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            let state = self.cognitive_state.lock().clone();
            for _ in 0..n_frames {
                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                all_samples.extend_from_slice(&chunk);
            }
        }

        let sample_rate = self.streaming.vocoder.sample_rate();
        write_wav(path, &all_samples, sample_rate)?;

        Ok(all_samples.len())
    }

    /// Push samples to the ring buffer with simple backpressure.
    fn push_with_backpressure(&mut self, samples: &[f32]) {
        let mut offset = 0;
        while offset < samples.len() {
            let written = self.audio.push_samples(&samples[offset..]);
            offset += written;
            if offset < samples.len() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Stop speaking immediately. The ring buffer drains naturally to silence.
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
    }

    /// Whether `speak()` or `speak_async()` is currently running.
    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    /// Get a clone of the stop flag for cross-thread interruption.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Get a handle to the shared cognitive state for real-time prosody modulation.
    ///
    /// Lock the mutex and modify the state from any thread; changes take effect
    /// on the next motor frame (~5ms).
    pub fn cognitive_state_handle(&self) -> Arc<parking_lot::Mutex<VoiceCognitiveState>> {
        Arc::clone(&self.cognitive_state)
    }

    /// Set the cognitive state (convenience wrapper — locks internally).
    pub fn set_cognitive_state(&self, state: VoiceCognitiveState) {
        *self.cognitive_state.lock() = state;
    }

    /// Modulate the LTC controller's time constant for speech rate control.
    ///
    /// `factor > 1.0` → slower, more deliberate formant transitions (max 3.0).
    /// `factor < 1.0` → faster, more agile transitions.
    /// `factor = 1.0` → default rate.
    pub fn modulate_tau(&mut self, factor: f32) {
        self.streaming.pipeline.controller.modulate_tau(factor);
    }

    /// Run additional training epochs on the formant database.
    pub fn train(&mut self, epochs: usize) {
        train_controller_on_phoneme_db(
            &mut self.streaming.pipeline.controller,
            &self.genesis,
            &self.formant_db,
            epochs,
        );
    }

    /// Audio sample rate from the output device (or headless default).
    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    /// Reset the vocal tract pipeline state.
    pub fn reset(&mut self) {
        self.streaming.reset();
    }
}

/// Write 16-bit PCM mono WAV via hound.
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let amplitude = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(amplitude)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Push samples to a ring buffer producer (used by background thread).
fn push_samples_to_producer(producer: &mut ringbuf::HeapProd<f32>, samples: &[f32]) -> usize {
    use ringbuf::traits::Producer;
    let mut written = 0;
    for &s in samples {
        if producer.try_push(s).is_ok() {
            written += 1;
        } else {
            break;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_sequence_generation() {
        let g2p = SimpleG2P::new();
        let phonemes = g2p.text_to_phonemes("hello world", BASE_PHONEME_DURATION);
        assert!(
            !phonemes.is_empty(),
            "Should produce phonemes for 'hello world'"
        );

        let non_silence: Vec<_> = phonemes.iter().filter(|p| p.phoneme != "SIL").collect();
        assert!(
            non_silence.len() >= 4,
            "Should have at least 4 non-silence phonemes, got {}",
            non_silence.len()
        );
    }

    #[test]
    fn test_stop_flag_works() {
        let flag = Arc::new(AtomicBool::new(true));
        assert!(flag.load(Ordering::SeqCst));

        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_plan_prominence_selects_typed_pitch_accent() {
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(true, 0.90),
            PitchAccent::RiseHigh
        );
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(false, 0.70),
            PitchAccent::High
        );
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(false, 0.20),
            PitchAccent::None
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_rate_reaches_real_scheduler() {
        fn make_plan(rate: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-rate-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(4);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.rate = rate;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![PhonemeSlot::new(
                    "AH",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                )],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-rate-test");
        let mut voice = LiveVoice::new_headless(&genesis);

        let slow = make_plan(0.70);
        let fast = make_plan(1.30);

        let dir = std::env::temp_dir().join(format!(
            "symthaea-plan-native-rate-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temporary output directory");
        let slow_path = dir.join("slow.wav");
        let fast_path = dir.join("fast.wav");

        let slow_samples = voice
            .speak_phonological_plan_to_file(&slow, &slow_path)
            .expect("slow plan should synthesize");

        voice.reset();

        let fast_samples = voice
            .speak_phonological_plan_to_file(&fast, &fast_path)
            .expect("fast plan should synthesize");

        assert!(
            slow_samples > fast_samples,
            "scheduler must consume plan rate: slow={slow_samples}, fast={fast_samples}"
        );
        let slow_frames = predict_duration("AH", 1, false, true, slow.rate);
        let samples_per_frame = (voice.sample_rate() / FRAME_RATE) as usize;
        assert_eq!(
            slow_samples,
            slow_frames * samples_per_frame,
            "scheduler sample count must equal deterministic frame count"
        );

        std::fs::remove_dir_all(&dir).expect("remove temporary output directory");
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_consumes_pitch_range() {
        fn make_plan(pitch_range: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-pitch-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(2);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.pitch_range = pitch_range;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![PhonemeSlot::new(
                    "AH",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                )],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-pitch-test");
        let mut voice = LiveVoice::new_headless(&genesis);

        for arousal in [0.0_f32, 1.0_f32] {
            voice.set_cognitive_state(VoiceCognitiveState {
                emotional_arousal: arousal,
                ..Default::default()
            });

            let narrow = make_plan(0.65);
            let (narrow_samples, _) = voice
                .synthesize_phonological_plan(&narrow)
                .expect("narrow pitch plan should synthesize");

            voice.reset();

            voice.set_cognitive_state(VoiceCognitiveState {
                emotional_arousal: arousal,
                ..Default::default()
            });

            let wide = make_plan(1.45);
            let (wide_samples, _) = voice
                .synthesize_phonological_plan(&wide)
                .expect("wide pitch plan should synthesize");

            assert_eq!(
                narrow_samples.len(),
                wide_samples.len(),
                "pitch range should change realization, not deterministic duration"
            );

            let difference = narrow_samples
                .iter()
                .zip(&wide_samples)
                .map(|(left, right)| (left - right).abs())
                .sum::<f32>();
            assert!(
                difference > 1e-3,
                "plan pitch range must remain observable at arousal {arousal}: absolute sample difference={difference}"
            );
        }
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_syllable_progress_spans_multiple_phonemes() {
        let start = 10;
        let end = 16;

        assert!((progress_within_frame_span(start, start, end) - 0.0).abs() < f32::EPSILON);
        assert!((progress_within_frame_span(12, start, end) - 0.4).abs() < f32::EPSILON);
        assert!((progress_within_frame_span(15, start, end) - 1.0).abs() < f32::EPSILON);

        let next_start = 16;
        let next_end = 20;
        assert!((progress_within_frame_span(next_start, next_start, next_end) - 0.0).abs() < f32::EPSILON);
        assert!((progress_within_frame_span(19, next_start, next_end) - 1.0).abs() < f32::EPSILON);
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_consumes_explicit_pause_weight() {
        fn make_plan(pause_weight: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-pause-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(2);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.pause_weight = pause_weight;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![
                    PhonemeSlot::new(
                        "AH",
                        0,
                        SyllableStress::Primary,
                        true,
                        false,
                        true,
                    ),
                    PhonemeSlot::new(
                        "SIL",
                        1,
                        SyllableStress::None,
                        true,
                        false,
                        true,
                    ),
                ],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-pause-test");
        let mut voice = LiveVoice::new_headless(&genesis);

        let no_pause = make_plan(0.0);
        let explicit_pause = make_plan(1.0);

        let (no_pause_samples, _) = voice
            .synthesize_phonological_plan(&no_pause)
            .expect("no-pause plan should synthesize");
        voice.reset();
        let (explicit_pause_samples, _) = voice
            .synthesize_phonological_plan(&explicit_pause)
            .expect("explicit pause plan should synthesize");

        assert!(
            explicit_pause_samples.len() > no_pause_samples.len(),
            "pause weight must extend only the explicitly encoded silence segment"
        );

        let samples_per_frame = (voice.sample_rate() / FRAME_RATE) as usize;
        let first_segment_frames = predict_duration("AH", 1, false, false, 1.0);
        let silence_start = first_segment_frames * samples_per_frame;
        let silence_frames = ((predict_duration("SIL", 0, false, true, 1.0) as f32)
            * 2.0)
            .round() as usize;
        let silence_end = silence_start + silence_frames * samples_per_frame;
        assert!(
            explicit_pause_samples[silence_start..silence_end]
                .iter()
                .all(|sample| *sample == 0.0),
            "explicit SIL must emit zero PCM across its scheduled pause window"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_receipt_binds_plan_and_audio() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-receipt-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
        speech_plan.prosody.pause_weight = 0.5;
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![
                PhonemeSlot::new(
                    "AH",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                ),
                PhonemeSlot::new(
                    "SIL",
                    1,
                    SyllableStress::None,
                    true,
                    false,
                    true,
                ),
            ],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit receipt fixture");

        let mut voice = LiveVoice::new_headless(&genesis);
        let receipt = voice
            .speak_phonological_plan_with_receipt(&plan)
            .expect("plan-native receipt should be emitted");

        assert_eq!(receipt.schema_version, 3);
        for legacy_schema in [1, 2] {
            let mut legacy_receipt = receipt.clone();
            legacy_receipt.schema_version = legacy_schema;
            assert!(
                legacy_receipt.verify_against_plan(&plan).is_err(),
                "legacy realization receipt schema must fail closed"
            );
        }
        assert_eq!(receipt.plan_version, plan.version);
        assert!(receipt.realization_authorized);
        assert_eq!(
            receipt.plan_grounding_blake3,
            blake3::hash(plan.grounding_surface().as_bytes())
                .to_hex()
                .to_string()
        );
        assert_eq!(receipt.segment_count, 2);
        assert_eq!(receipt.segment_frame_counts.len(), 2);
        assert_eq!(
            receipt.sample_count,
            receipt.scheduler_frames * (receipt.sample_rate / FRAME_RATE) as usize
        );
        assert_eq!(receipt.sample_rate, voice.sample_rate());
        assert_eq!(receipt.sample_rate, 24_000);
        assert_eq!(receipt.rate, plan.rate);
        assert_eq!(receipt.pitch_range, plan.pitch_range);
        assert_eq!(receipt.prominence, plan.prominence);
        assert_eq!(receipt.pause_weight, plan.pause_weight);
        assert_eq!(receipt.audio_blake3.len(), 64);
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_receipt_rejects_tampering() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-receipt-verify-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit receipt fixture");

        let mut voice = LiveVoice::new_headless(&genesis);
        let (samples, receipt) = voice
            .synthesize_phonological_plan(&plan)
            .expect("plan-native receipt should be emitted");
        assert!(receipt.verify_against_plan(&plan).is_ok());
        let mut unauthorized_receipt = receipt.clone();
        unauthorized_receipt.realization_authorized = false;
        assert!(unauthorized_receipt.verify_against_plan(&plan).is_err());
        assert!(receipt.verify_samples(&samples));

        let mut tampered_rate = receipt.clone();
        tampered_rate.sample_rate = tampered_rate.sample_rate.saturating_add(1);
        assert!(
            !tampered_rate.verify_samples(&samples),
            "audio receipt must bind sample-rate metadata to the sample digest"
        );

        let mut impossible_rate = receipt.clone();
        impossible_rate.sample_rate = FRAME_RATE - 1;
        assert!(
            impossible_rate.verify_against_plan(&plan).is_err(),
            "receipt verification must reject a sample rate that cannot produce a motor-frame sample"
        );
        assert!(!impossible_rate.verify_samples(&samples));

        let mut empty = receipt.clone();
        empty.sample_count = 0;
        empty.audio_blake3 = hash_audio_binding(empty.sample_rate, &[]);
        assert!(!empty.verify_samples(&[]), "empty audio receipts must not self-validate");

        let mut tampered_schedule = receipt.clone();
        tampered_schedule.segment_frame_counts[0] =
            tampered_schedule.segment_frame_counts[0].saturating_add(1);
        assert!(
            tampered_schedule.verify_against_plan(&plan).is_err(),
            "receipt verification must recompute the exact segment schedule"
        );

        let mut tampered_total = receipt.clone();
        tampered_total.scheduler_frames = tampered_total.scheduler_frames.saturating_add(1);
        assert!(
            tampered_total.verify_against_plan(&plan).is_err(),
            "receipt verification must bind the total scheduler-frame count"
        );

        let mut tampered = plan.clone();
        tampered.rate = if tampered.rate < 1.0 { 1.2 } else { 0.8 };
        assert!(receipt.verify_against_plan(&tampered).is_err());
        assert!(!receipt.verify_samples(&samples[..samples.len().saturating_sub(1)]));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_unsupported_plan_phoneme_is_rejected() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-unsupported-phoneme-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "ZZ",
                0,
                SyllableStress::None,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("phonological binding itself should allow caller-defined identity");

        let genesis = GenesisSeed::from_phrase("plan-native-unsupported-phoneme-test");
        let mut voice = LiveVoice::new_headless(&genesis);
        let error = voice
            .synthesize_phonological_plan(&plan)
            .expect_err("unsupported realization symbol must fail closed");
        assert!(
            error
                .to_string()
                .contains("unsupported realization symbol"),
            "unexpected rejection: {error:#}"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_rejects_zero_frame_sample_rate() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-invalid-sample-rate-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let frame = LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout));
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit phonological fixture");

        let mut voice = LiveVoice::new_headless_with_rate(&genesis, FRAME_RATE - 1);
        let error = voice
            .synthesize_phonological_plan(&plan)
            .expect_err("a motor frame with zero output samples must fail closed");
        assert!(error.to_string().contains("sample rate >= 200 Hz"));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_rejects_non_divisible_sample_rate() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-nondivisible-sample-rate-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let frame = LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout));
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit phonological fixture");

        let mut voice = LiveVoice::new_headless_with_rate(&genesis, 22_050);
        let error = voice
            .synthesize_phonological_plan(&plan)
            .expect_err("non-divisible sample rate must fail closed");
        assert!(error.to_string().contains("sample rate divisible by 200 Hz"));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_strict_english_lexicon_derivation_uses_only_embedded_pronunciation() {
        use symthaea_broca::{
            ContentBindingStatus, GrammaticalFunction, LanguageRuleBinding, LanguageRuleStatus,
            LexemeBinding, LexicalSource, LinguisticFrame, PhonemeSlot, SpeechPlan,
            StructuredDecoder, SyllableStress, ThoughtChannels, LexicalMorphosyntacticBinding,
        };

        let genesis = GenesisSeed::from_phrase("strict-lexicon-witness-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let frame = LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout));
        let mut voice = LiveVoice::new_headless(&genesis);

        let make_binding = |form: &str| {
            let constituents = frame
                .constituents
                .iter()
                .map(|slot| LexemeBinding {
                    position: slot.position,
                    source: LexicalSource::SemanticConstituent {
                        role: slot.role.clone(),
                        prime: slot.prime.clone(),
                    },
                    lemma: form.into(),
                    lexeme_id: format!("en:fixture:{}", slot.position),
                    grammatical_function: GrammaticalFunction::Other("fixture".into()),
                    morphology: Vec::new(),
                    morphophonological_form: Some(form.into()),
                    provenance: "fixture:v1".into(),
                    semantic_payload: true,
                })
                .collect::<Vec<_>>();

            LexicalMorphosyntacticBinding::new(
                &frame,
                LanguageRuleBinding {
                    language_tag: "en".into(),
                    status: LanguageRuleStatus::Bound,
                    rule_id: Some("fixture:rules:v1".into()),
                    provenance: Some("fixture:rules:v1".into()),
                    unbound_reason: None,
                },
                constituents,
                Vec::new(),
                Vec::new(),
            )
            .expect("fixture lexical binding")
        };

        let binding = make_binding("hello");
        let (hello_phones, _hello_evidence) = voice
            .g2p
            .word_to_phonemes_from_lexicon_with_evidence("hello")
            .expect("hello must be in an embedded pronunciation lexicon");

        let mut segments = Vec::new();
        for constituent in &binding.constituents {
            let (phones, _) = voice
                .g2p
                .word_to_phonemes_from_lexicon(
                    constituent
                        .morphophonological_form
                        .as_deref()
                        .expect("fixture form"),
                )
                .expect("fixture form must have an embedded pronunciation");

            for (index, phone) in phones.iter().enumerate() {
                let base = phone.trim_end_matches(|c: char| c.is_ascii_digit());
                let stress = match phone.chars().last() {
                    Some('1') => SyllableStress::Primary,
                    Some('2') => SyllableStress::Secondary,
                    _ => SyllableStress::None,
                };
                segments.push(PhonemeSlot::new(
                    base,
                    segments.len(),
                    stress,
                    true,
                    false,
                    constituent.position + 1 == binding.constituents.len() && index + 1 == phones.len(),
                ));
            }
        }

        let (witness, pronunciation_lexicon_evidence) = voice
            .derive_english_lexical_phonological_witness(&binding, &segments)
            .expect("embedded lexicon should derive a witness");
        assert_eq!(pronunciation_lexicon_evidence.len(), 1);
        assert_eq!(
            pronunciation_lexicon_evidence[0].source_id,
            "symthaea-hand-lexicon-v1"
        );
        assert_eq!(
            pronunciation_lexicon_evidence[0].dialect_scope,
            "en-unspecified"
        );
        assert_eq!(
            pronunciation_lexicon_evidence[0].selected_variant,
            "only-entry"
        );
        assert_eq!(pronunciation_lexicon_evidence[0].available_variants, 1);
        assert!(witness.validate_against_segments(&binding, &segments).is_ok());
        assert_eq!(
            witness.mappings[0].symbols[0],
            hello_phones[0].trim_end_matches(|c: char| c.is_ascii_digit())
        );

        let mut plan = symthaea_broca::PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_lexical_segments_from_binding_with_witness(
            &frame,
            &binding,
            &witness,
            segments.clone(),
        )
        .expect("strict witness should bind into a realization-ready plan");
        assert_eq!(plan.content_binding, ContentBindingStatus::LexicallyBound);

        let receipt = voice
            .speak_english_lexicon_verified_lexical_phonological_plan_with_receipt(
                &plan, &frame, &binding,
            )
            .expect("strict lexicon-backed plan should realize");
        receipt
            .verify_against_plan(&plan, &frame, &binding, &witness)
            .expect("strict receipt should independently revalidate");

        assert_eq!(receipt.pronunciation_lexicon_evidence.len(), 1);
        assert_eq!(
            receipt.pronunciation_lexicon_evidence[0].source_id,
            "symthaea-hand-lexicon-v1"
        );
        assert_eq!(
            receipt.pronunciation_lexicon_evidence[0].dialect_scope,
            "en-unspecified"
        );
        assert_eq!(
            receipt.pronunciation_lexicon_evidence[0].variant_policy,
            "single-curated-entry"
        );
        assert_eq!(
            receipt.pronunciation_lexicon_evidence[0].selected_variant,
            "only-entry"
        );
        assert_eq!(receipt.pronunciation_lexicon_evidence[0].available_variants, 1);
        assert!(receipt.pronunciation_lexicon_evidence[0].is_well_formed());
        assert_eq!(
            receipt.pronunciation_lexicon_evidence_blake3,
            super::hash_pronunciation_lexicon_evidence(
                &receipt.pronunciation_lexicon_evidence
            )
        );
        receipt
            .verify_against_plan_and_current_resources(
                &plan,
                &frame,
                &binding,
                &witness,
                &voice.g2p,
            )
            .expect("strict receipt must reproduce its current resource derivation");

        let mut detached_resource_substitution = receipt.clone();
        detached_resource_substitution.pronunciation_lexicon_evidence[0].resource_blake3 =
            "f".repeat(64);
        detached_resource_substitution.pronunciation_lexicon_evidence_blake3 =
            super::hash_pronunciation_lexicon_evidence(
                &detached_resource_substitution.pronunciation_lexicon_evidence,
            );
        detached_resource_substitution
            .verify_against_plan(&plan, &frame, &binding, &witness)
            .expect("detached verification should preserve historical evidence inspectability");
        assert!(
            detached_resource_substitution
                .verify_against_plan_and_current_resources(
                    &plan,
                    &frame,
                    &binding,
                    &witness,
                    &voice.g2p,
                )
                .is_err(),
            "substituted resource evidence must fail current-resource re-derivation"
        );

        let unlisted = make_binding("zzzxxyq");
        assert!(
            voice
                .derive_english_lexical_phonological_witness(&unlisted, &segments)
                .is_err(),
            "unlisted forms must fail rather than using spelling-rule fallback"
        );
    }


    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_lexically_bound_plan_requires_verified_binding() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-lexical-boundary-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("fixture phonology should bind");
        plan.content_binding = ContentBindingStatus::LexicallyBound;
        plan.lexical_provenance = Some(
            blake3::hash(b"self-attested-lexical-binding")
                .to_hex()
                .to_string(),
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rejected-lexical-bound.wav");
        let mut voice = LiveVoice::new_headless(&genesis);

        let error = voice
            .speak_phonological_plan_to_file(&plan, &path)
            .expect_err("unverified lexical plans must not reach scheduler realization");

        assert!(error.to_string().contains("validated lexical-binding realization"));
        assert!(!path.exists());
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_receipt_rejects_unverified_lexical_bound_plan_after_hash_update() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-lexical-receipt-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("fixture phonology should bind");

        let mut voice = LiveVoice::new_headless(&genesis);
        let mut receipt = voice
            .speak_phonological_plan_with_receipt(&plan)
            .expect("phonological receipt should be emitted");

        plan.content_binding = ContentBindingStatus::LexicallyBound;
        plan.lexical_provenance = Some(
            blake3::hash(b"self-attested-lexical-binding")
                .to_hex()
                .to_string(),
        );
        receipt.plan_grounding_blake3 =
            blake3::hash(plan.grounding_surface().as_bytes()).to_hex().to_string();

        let error = receipt
            .verify_against_plan(&plan)
            .expect_err("receipt verification must reject an unverified lexical plan");

        assert!(error.to_string().contains("validated lexical-binding realization"));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_verified_lexical_phonological_path_emits_witness_bound_receipt() {
        use symthaea_broca::{
            ContentBindingStatus, GrammaticalFunction, LanguageRuleBinding, LanguageRuleStatus,
            LexemeBinding, LexicalPhonologicalMapping, LexicalPhonologicalWitness, LexicalSource,
            LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder, SyllableStress,
            ThoughtChannels, LexicalMorphosyntacticBinding,
        };

        let genesis = GenesisSeed::from_phrase("verified-lexical-voice-path");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);

        let constituents = frame
            .constituents
            .iter()
            .map(|slot| LexemeBinding {
                position: slot.position,
                source: LexicalSource::SemanticConstituent {
                    role: slot.role.clone(),
                    prime: slot.prime.clone(),
                },
                lemma: slot.prime.to_ascii_lowercase(),
                lexeme_id: format!("fixture:lexeme:{}", slot.position),
                grammatical_function: GrammaticalFunction::Other("fixture".into()),
                morphology: Vec::new(),
                morphophonological_form: Some(slot.prime.to_ascii_lowercase()),
                provenance: "fixture:lexicon:v1".into(),
                semantic_payload: true,
            })
            .collect::<Vec<_>>();

        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            LanguageRuleBinding {
                language_tag: "en".into(),
                status: LanguageRuleStatus::Bound,
                rule_id: Some("fixture:rules:v1".into()),
                provenance: Some("fixture:rules:v1".into()),
                unbound_reason: None,
            },
            constituents,
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture lexical binding");

        let phoneme_symbols = ["AH", "B", "K", "D", "EH", "F", "G", "M", "N", "P", "R", "S"];
        let segments = binding
            .constituents
            .iter()
            .enumerate()
            .map(|(index, _)| {
                PhonemeSlot::new(
                    phoneme_symbols[index % phoneme_symbols.len()],
                    index,
                    SyllableStress::Primary,
                    true,
                    false,
                    index + 1 == binding.constituents.len(),
                )
            })
            .collect::<Vec<_>>();

        let witness = LexicalPhonologicalWitness::new(
            &binding,
            segments
                .iter()
                .enumerate()
                .map(|(index, segment)| {
                    LexicalPhonologicalMapping::from_lexical_binding(
                        &binding,
                        index,
                        vec![index],
                        vec![segment.symbol.clone()],
                    )
                    .expect("lexical position exists")
                })
                .collect(),
        )
        .expect("fixture realization witness");

        let mut plan = symthaea_broca::PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_lexical_segments_from_binding_with_witness(
            &frame,
            &binding,
            &witness,
            segments,
        )
        .expect("witness-backed lexical plan");

        assert_eq!(plan.content_binding, ContentBindingStatus::LexicallyBound);

        let mut voice = LiveVoice::new_headless(&genesis);
        let receipt = voice
            .speak_verified_lexical_phonological_plan_with_receipt(
                &plan,
                &frame,
                &binding,
                &witness,
            )
            .expect("verified lexical plan should realize");

        receipt
            .verify_against_plan(&plan, &frame, &binding, &witness)
            .expect("verified receipt should independently revalidate");
        assert_eq!(
            receipt.witness_version,
            witness.version,
            "receipt must retain the exact witness version"
        );
        assert_eq!(
            receipt.lexical_binding_provenance,
            binding.provenance_token(),
            "receipt must retain the exact lexical-binding provenance"
        );
        assert_eq!(
            receipt.witness_blake3,
            witness.provenance_token(),
            "receipt must retain the exact witness identity"
        );
        assert!(receipt.realization.sample_count > 0);

        let mut tampered_witness = witness.clone();
        tampered_witness.mappings[0].symbols[0] = "TAMPERED".into();
        assert!(
            receipt
                .verify_against_plan(&plan, &frame, &binding, &tampered_witness)
                .is_err(),
            "changing the retained witness must invalidate the verified receipt"
        );

        let mut tampered_receipt = receipt.clone();
        tampered_receipt.witness_blake3 =
            blake3::hash(b"different-witness").to_hex().to_string();
        assert!(
            tampered_receipt
                .verify_against_plan(&plan, &frame, &binding, &witness)
                .is_err(),
            "changing only the receipt witness digest must fail verification"
        );

        let mut unsupported_source_receipt = receipt.clone();
        unsupported_source_receipt.pronunciation_lexicon_evidence = vec![
            PronunciationLexiconEvidence {
                source_id: "invented-v1".into(),
                dialect_scope: "en-US".into(),
                variant_policy: "primary-un-suffixed-entry".into(),
                selected_variant: "primary".into(),
                available_variants: 1,
                resource_blake3: "0".repeat(64),
            },
        ];
        unsupported_source_receipt.pronunciation_lexicon_evidence_blake3 =
            hash_pronunciation_lexicon_evidence(
                &unsupported_source_receipt.pronunciation_lexicon_evidence
            );
        assert!(
            unsupported_source_receipt
                .verify_against_plan(&plan, &frame, &binding, &witness)
                .is_err(),
            "unsupported lexicon source identifiers must fail verification"
        );

        let mut tampered_resource_receipt = receipt.clone();
        tampered_resource_receipt.pronunciation_lexicon_evidence[0].resource_blake3 =
            "1".repeat(64);
        assert!(
            tampered_resource_receipt
                .verify_against_plan(&plan, &frame, &binding, &witness)
                .is_err(),
            "resource digest tampering must fail without recomputing the receipt evidence binding"
        );

        let mut tampered_scope_receipt = receipt.clone();
        tampered_scope_receipt.pronunciation_lexicon_evidence[0].dialect_scope = "en-GB".into();
        assert!(
            tampered_scope_receipt
                .verify_against_plan(&plan, &frame, &binding, &witness)
                .is_err(),
            "dialect scope tampering must fail closed"
        );

        let mut tampered_variant_receipt = receipt.clone();
        tampered_variant_receipt.pronunciation_lexicon_evidence[0].selected_variant =
            "alternate-1".into();
        assert!(
            tampered_variant_receipt
                .verify_against_plan(&plan, &frame, &binding, &witness)
                .is_err(),
            "selected pronunciation variant tampering must fail closed"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_morphophonological_verified_path_binds_rule_set_before_realization() {
        use symthaea_broca::{
            ContentBindingStatus, GrammaticalFunction, LanguageRuleBinding, LanguageRuleStatus,
            LexemeBinding, LexicalPhonologicalMapping, LexicalPhonologicalWitness, LexicalSource,
            LinguisticFrame, MorphophonologicalDerivationWitness, MorphophonologicalResourceEvidence,
            MorphophonologicalRule, MorphophonologicalRuleOperation, MorphophonologicalRuleSet,
            PhonemeSlot, SpeechPlan, StructuredDecoder, SyllableStress, ThoughtChannels,
            LexicalMorphosyntacticBinding,
        };

        let genesis = GenesisSeed::from_phrase("morphophonological-admission-path");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);

        let constituents = frame
            .constituents
            .iter()
            .map(|slot| LexemeBinding {
                position: slot.position,
                source: LexicalSource::SemanticConstituent {
                    role: slot.role.clone(),
                    prime: slot.prime.clone(),
                },
                lemma: slot.prime.to_ascii_lowercase(),
                lexeme_id: format!("fixture:lexeme:{}", slot.position),
                grammatical_function: GrammaticalFunction::Other("fixture".into()),
                morphology: Vec::new(),
                morphophonological_form: Some(slot.prime.to_ascii_lowercase()),
                provenance: "fixture:lexicon:v1".into(),
                semantic_payload: true,
            })
            .collect::<Vec<_>>();

        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            LanguageRuleBinding {
                language_tag: "en".into(),
                status: LanguageRuleStatus::Bound,
                rule_id: Some("fixture:rules:v1".into()),
                provenance: Some("fixture:rules:v1".into()),
                unbound_reason: None,
            },
            constituents,
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture lexical binding");

        let segments = binding
            .constituents
            .iter()
            .enumerate()
            .map(|(index, _)| {
                PhonemeSlot::new(
                    "AH",
                    index,
                    SyllableStress::Primary,
                    true,
                    false,
                    index + 1 == binding.constituents.len(),
                )
            })
            .collect::<Vec<_>>();

        let lexical_witness = LexicalPhonologicalWitness::new(
            &binding,
            segments
                .iter()
                .enumerate()
                .map(|(index, segment)| {
                    LexicalPhonologicalMapping::from_lexical_binding(
                        &binding,
                        index,
                        vec![index],
                        vec![segment.symbol.clone()],
                    )
                    .expect("lexical position exists")
                })
                .collect(),
        )
        .expect("fixture lexical witness");

        let rule_set = MorphophonologicalRuleSet::new(
            "en",
            "fixture:rules:v1",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:rules:v1",
                "fixture-v1",
            )
            .expect("fixture resource evidence"),
            "fixture:rules:v1",
            "fixture:rules:v1",
            vec![MorphophonologicalRule {
                rule_id: "fixture:identity".into(),
                morphology: Vec::new(),
                operation: MorphophonologicalRuleOperation::Identity,
            }],
        )
        .expect("identity rule set");

        let morph_witness =
            MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
                .expect("morphophonological witness");

        let mut plan = symthaea_broca::PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_lexical_segments_from_binding_with_witness(
            &frame,
            &binding,
            &lexical_witness,
            segments,
        )
        .expect("witness-backed lexical plan");
        assert_eq!(plan.content_binding, ContentBindingStatus::LexicallyBound);

        let mut voice = LiveVoice::new_headless(&genesis);
        let receipt = voice
            .speak_morphophonology_verified_lexical_phonological_plan_with_receipt(
                &plan,
                &frame,
                &binding,
                &lexical_witness,
                &morph_witness,
                &rule_set,
            )
            .expect("morphophonology-backed realization should succeed");

        receipt
            .verify_against_plan_and_rule_set(
                &plan,
                &frame,
                &binding,
                &lexical_witness,
                &morph_witness,
                &rule_set,
            )
            .expect("combined receipt should independently revalidate");

        let mut tampered_witness = morph_witness.clone();
        tampered_witness.steps[0].output_form.push('!');
        assert!(
            receipt
                .verify_against_plan_and_rule_set(
                    &plan,
                    &frame,
                    &binding,
                    &lexical_witness,
                    &tampered_witness,
                    &rule_set,
                )
                .is_err(),
            "morphophonological witness tampering must invalidate the combined receipt"
        );

        let mut tampered_rules = rule_set.clone();
        tampered_rules.rules[0].operation =
            MorphophonologicalRuleOperation::AppendSuffix { suffix: "x".into() };
        assert!(
            receipt
                .verify_against_plan_and_rule_set(
                    &plan,
                    &frame,
                    &binding,
                    &lexical_witness,
                    &morph_witness,
                    &tampered_rules,
                )
                .is_err(),
            "executable rule-set tampering must invalidate the combined receipt"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_role_only_phonological_plan_is_rejected() {
        use symthaea_broca::{LinguisticFrame, SpeechPlan, StructuredDecoder, ThoughtChannels};

        let genesis = GenesisSeed::from_phrase("plan-native-rejection-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        let frame = LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout));
        let plan = PhonologicalPlan::from_linguistic_frame(&frame);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rejected.wav");
        let mut voice = LiveVoice::new_headless(&genesis);

        let error = voice
            .speak_phonological_plan_to_file(&plan, &path)
            .expect_err("role-only plans must not reach synthesis");

        assert!(error.to_string().contains("not ready for realization"));
        assert!(!path.exists());
    }

    #[test]
    fn test_speak_to_file_headless() {
        let genesis = GenesisSeed::from_phrase("test-headless");
        let mut voice = LiveVoice::new_headless(&genesis);

        let dir = tempfile::tempdir().expect("tempdir");
        let wav_path = dir.path().join("test.wav");

        let n_samples = voice
            .speak_to_file("hello", &wav_path)
            .expect("speak_to_file should succeed");

        assert!(n_samples > 0, "Should produce audio samples");
        assert!(wav_path.exists(), "WAV file should be created");

        // Verify WAV is readable
        let reader = hound::WavReader::open(&wav_path).expect("Should read WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 24000);
        assert!(reader.len() > 0);
    }

    #[test]
    fn test_cognitive_state_handle() {
        let state = Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default()));
        let handle = Arc::clone(&state);

        // Modify from "another thread" (simulated)
        {
            let mut s = handle.lock();
            s.emotional_arousal = 0.9;
        }

        let current = state.lock().clone();
        assert!((current.emotional_arousal - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_speak_to_file_cognitive_modulation() {
        let genesis = GenesisSeed::from_phrase("test-prosody");
        let mut voice = LiveVoice::new_headless(&genesis);

        let dir = tempfile::tempdir().expect("tempdir");

        // Calm state
        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.1,
            ..Default::default()
        });
        let calm_path = dir.path().join("calm.wav");
        let calm_n = voice.speak_to_file("hello", &calm_path).unwrap();

        // Reset pipeline state between utterances
        voice.reset();

        // Excited state
        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.9,
            emotional_valence: 0.8,
            consciousness_level: 0.9,
            ..Default::default()
        });
        let excited_path = dir.path().join("excited.wav");
        let excited_n = voice.speak_to_file("hello", &excited_path).unwrap();

        // Both should produce audio
        assert!(calm_n > 0);
        assert!(excited_n > 0);

        // Read both WAVs and compare RMS — different cognitive states should
        // produce different audio content (even if same phonemes)
        let calm_reader = hound::WavReader::open(&calm_path).unwrap();
        let excited_reader = hound::WavReader::open(&excited_path).unwrap();

        let calm_samples: Vec<f32> = calm_reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32767.0)
            .collect();
        let excited_samples: Vec<f32> = excited_reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32767.0)
            .collect();

        let calm_rms = rms(&calm_samples);
        let excited_rms = rms(&excited_samples);

        // Both should have non-trivial content
        assert!(
            calm_rms > 1e-6,
            "Calm audio should have content: rms={calm_rms}"
        );
        assert!(
            excited_rms > 1e-6,
            "Excited audio should have content: rms={excited_rms}"
        );
    }

    #[test]
    #[ignore] // Requires audio device
    fn test_live_voice_speak_produces_audio() {
        let genesis = GenesisSeed::from_phrase("test-live-voice");
        let mut voice = LiveVoice::new(&genesis).expect("Should create LiveVoice");
        assert!(voice.sample_rate() > 0);

        voice.speak("hello").expect("Should speak without error");
        assert!(!voice.is_speaking());
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }
}
