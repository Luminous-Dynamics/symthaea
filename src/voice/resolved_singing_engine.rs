// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Renderer-neutral contract for selected pronunciation and explicit score
//! alignment. The legacy SingingVoiceEngine remains the v2 compatibility
//! interface; v3-capable renderers implement this separate contract.

#![forbid(unsafe_code)]

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use symthaea_pronunciation_plan::{
    BeatDomainScore, PronunciationPlan, SyllableNoteBinding, VocalPhraseSpan,
    VocalScoreAlignment,
};

use super::singing_engine::{
    SingingBackendCapabilities, SingingRenderPolicy, VocalStem,
};

const RECEIPT_SCHEMA: &str = "resolved-vocal-render-receipt/v1";

/// A score source for a resolved pronunciation. The original beat-domain
/// score is retained instead of reducing every request to seconds prematurely.
#[derive(Debug, Clone, Copy)]
pub enum ResolvedScoreInput<'a> {
    /// Explicit seconds-domain note/rest events and syllable interval bindings.
    SecondsAlignment(&'a VocalScoreAlignment),
    /// Musical beat events plus tempo map; materialization occurs at the
    /// renderer boundary, while the receipt hashes this exact source object.
    BeatDomain {
        score: &'a BeatDomainScore,
        syllable_bindings: &'a [SyllableNoteBinding],
        phrases: &'a [VocalPhraseSpan],
    },
}

/// A complete request for a resolved-plan renderer.
///
/// The pronunciation must already be selected and renderable. The score
/// alignment must cover all sung intervals exactly once and preserve rests.
/// The policy is checked against the selected renderer's resolved-plan
/// capability manifest before synthesis begins.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedSingingRequest<'a> {
    pub pronunciation: &'a PronunciationPlan,
    pub score: ResolvedScoreInput<'a>,
    pub policy: &'a SingingRenderPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedRenderReceipt {
    pub schema: String,
    pub backend_id: String,
    pub capability_manifest_blake3: String,
    pub pronunciation_plan_blake3: String,
    pub score_input_blake3: String,
    pub render_policy_blake3: String,
    pub sample_rate_hz: u32,
    pub sample_count: u64,
    /// Domain-separated digest over sample-rate metadata and little-endian f32
    /// samples. This is not a WAV-file digest; callers may also hash the WAV.
    pub audio_buffer_blake3: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedSingingOutput {
    pub stem: VocalStem,
    pub receipt: ResolvedRenderReceipt,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ScoreDigestPayload<'a> {
    SecondsAlignment {
        alignment: &'a VocalScoreAlignment,
    },
    BeatDomain {
        score: &'a BeatDomainScore,
        syllable_bindings: &'a [SyllableNoteBinding],
        phrases: &'a [VocalPhraseSpan],
    },
}

fn json_digest<T: Serialize>(domain: &[u8], value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn score_digest(score: ResolvedScoreInput<'_>) -> Result<String> {
    let payload = match score {
        ResolvedScoreInput::SecondsAlignment(alignment) => {
            ScoreDigestPayload::SecondsAlignment { alignment }
        }
        ResolvedScoreInput::BeatDomain {
            score,
            syllable_bindings,
            phrases,
        } => ScoreDigestPayload::BeatDomain {
            score,
            syllable_bindings,
            phrases,
        },
    };
    json_digest(b"symthaea-resolved-score-input-v1\0", &payload)
}

fn audio_buffer_digest(stem: &VocalStem) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-resolved-vocal-audio-buffer-v1\0");
    hasher.update(&stem.sample_rate.to_le_bytes());
    hasher.update(&(stem.samples.len() as u64).to_le_bytes());
    for sample in &stem.samples {
        hasher.update(&sample.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn build_receipt(
    capabilities: &SingingBackendCapabilities,
    pronunciation: &PronunciationPlan,
    score: ResolvedScoreInput<'_>,
    policy: &SingingRenderPolicy,
    stem: &VocalStem,
) -> Result<ResolvedRenderReceipt> {
    if stem.backend != capabilities.backend_id {
        bail!(
            "resolved stem backend {:?} does not match capability ID {:?}",
            stem.backend,
            capabilities.backend_id
        );
    }
    Ok(ResolvedRenderReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        backend_id: capabilities.backend_id.clone(),
        capability_manifest_blake3: json_digest(
            b"symthaea-singing-capabilities-v1\0",
            capabilities,
        )?,
        pronunciation_plan_blake3: json_digest(
            b"symthaea-pronunciation-plan-v1\0",
            pronunciation,
        )?,
        score_input_blake3: score_digest(score)?,
        render_policy_blake3: json_digest(
            b"symthaea-singing-render-policy-v1\0",
            policy,
        )?,
        sample_rate_hz: stem.sample_rate,
        sample_count: stem.samples.len() as u64,
        audio_buffer_blake3: audio_buffer_digest(stem),
    })
}

impl ResolvedRenderReceipt {
    /// Recompute every input and output binding. This verifies receipt
    /// integrity, not acoustic correctness or that the selected pronunciation
    /// is linguistically true.
    pub fn verify_against(
        &self,
        capabilities: &SingingBackendCapabilities,
        pronunciation: &PronunciationPlan,
        score: ResolvedScoreInput<'_>,
        policy: &SingingRenderPolicy,
        stem: &VocalStem,
    ) -> Result<()> {
        capabilities.validate()?;
        stem.validate()?;
        policy.validate_against(capabilities)?;
        let expected = build_receipt(capabilities, pronunciation, score, policy, stem)?;
        if self != &expected {
            bail!("resolved vocal render receipt does not match inputs or audio");
        }
        Ok(())
    }
}

/// A renderer that can consume an already-selected pronunciation and
/// validated score/alignment graph without running G2P or reconstructing
/// syllable-to-note bindings.
///
/// Implementors provide their resolved-route capabilities and low-level
/// synthesis operation. Callers should use render_request, which verifies
/// backend identity, applies the render policy before synthesis, validates the
/// returned stem, and creates an independently re-checkable receipt.
pub trait ResolvedSingingVoiceEngine {
    /// Backend ID for this resolved-plan route. It may differ from the legacy
    /// v2 backend ID when the route has different capabilities.
    fn resolved_backend_id(&self) -> &str;

    fn resolved_capabilities(&self) -> Result<SingingBackendCapabilities>;

    /// Low-level renderer hook. Qualified callers should use render_request.
    fn render_resolved_unchecked(
        &mut self,
        pronunciation: &PronunciationPlan,
        score: ResolvedScoreInput<'_>,
    ) -> Result<VocalStem>;

    fn render_request(
        &mut self,
        request: ResolvedSingingRequest<'_>,
    ) -> Result<ResolvedSingingOutput> {
        let capabilities = self.resolved_capabilities()?;
        capabilities.validate()?;
        if capabilities.backend_id != self.resolved_backend_id() {
            bail!(
                "resolved capability ID {:?} does not match renderer ID {:?}",
                capabilities.backend_id,
                self.resolved_backend_id()
            );
        }
        request.policy.validate_against(&capabilities)?;

        // Validate semantic inputs before allocating or synthesizing audio.
        request
            .pronunciation
            .validate_renderable()
            .map_err(anyhow::Error::msg)?;
        match request.score {
            ResolvedScoreInput::SecondsAlignment(alignment) => alignment
                .validate_against(request.pronunciation)
                .map_err(anyhow::Error::msg)?,
            ResolvedScoreInput::BeatDomain {
                score,
                syllable_bindings,
                phrases,
            } => {
                let alignment = VocalScoreAlignment {
                    schema_version: 1,
                    events: score.materialize_seconds().map_err(anyhow::Error::msg)?,
                    syllable_bindings: syllable_bindings.to_vec(),
                    phrases: phrases.to_vec(),
                };
                alignment
                    .validate_against(request.pronunciation)
                    .map_err(anyhow::Error::msg)?;
            }
        }

        let stem = self.render_resolved_unchecked(request.pronunciation, request.score)?;
        stem.validate()?;
        let receipt = build_receipt(
            &capabilities,
            request.pronunciation,
            request.score,
            request.policy,
            &stem,
        )?;
        receipt.verify_against(
            &capabilities,
            request.pronunciation,
            request.score,
            request.policy,
            &stem,
        )?;
        Ok(ResolvedSingingOutput { stem, receipt })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_digest_binds_rate_and_exact_float_samples() {
        let left = VocalStem {
            samples: vec![0.0, 0.25, -0.5],
            sample_rate: 24_000,
            backend: "fixture".to_string(),
        };
        let mut changed_samples = left.clone();
        changed_samples.samples[1] = 0.250_001;
        let mut changed_rate = left.clone();
        changed_rate.sample_rate = 48_000;

        assert_ne!(audio_buffer_digest(&left), audio_buffer_digest(&changed_samples));
        assert_ne!(audio_buffer_digest(&left), audio_buffer_digest(&changed_rate));
    }
}
