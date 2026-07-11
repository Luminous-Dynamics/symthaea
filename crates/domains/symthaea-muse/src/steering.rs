// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Text-prompt steering: compose N candidate pieces, rank them by CLAP
//! similarity to a natural-language prompt, return the best.
//!
//! This is the industry's headline UX ("describe the music you want") done
//! the provenance-clean way: the *composer* stays the symbolic theory engine
//! (every note explainable, no scraped training data); the prompt only
//! *selects among* its outputs, using the pretrained CLAP towers
//! ([`crate::clap_embed`]) as a perceptual judge. Honest framing: this is
//! generate-and-rank, not conditioning — a prompt the composer can't reach
//! (e.g. "death metal") will simply pick the least-far candidate, and the
//! returned similarity scores say exactly how good the match actually is.
//!
//! Ranking core ([`rank_by_prompt`]) is deliberately independent of the ORT
//! sessions — it takes an embedding closure — so its selection logic is unit
//! tested without any model download; the [`steer`] convenience wrapper
//! wires the real towers.
//!
//! Feature-gated behind `theory` + `clap-fad`.

use crate::clap_embed::{ClapEmbedder, ClapTextEmbedder, cosine_similarity};
use crate::theory_realize::compose_and_realize_styled;
use crate::{AudioData, Composition, MusicalState};
use anyhow::Result;
use symthaea_music_theory::{MusicalIntent, Style};

/// CLAP's native input rate — candidates must be rendered at this rate
/// because [`ClapEmbedder::embed`] refuses to resample (see clap_embed.rs).
pub const STEERING_SAMPLE_RATE: u32 = 48_000;

/// One ranked candidate.
#[derive(Debug, Clone, Copy)]
pub struct CandidateScore {
    /// The seed this candidate was composed with.
    pub seed: u64,
    /// Cosine similarity between the candidate's audio embedding and the
    /// prompt's text embedding, in CLAP's shared space.
    pub similarity: f32,
}

/// Downmix a stereo render to the mono f64 waveform CLAP expects.
fn downmix(audio: &AudioData) -> Vec<f64> {
    match audio {
        AudioData::StereoF32(frames) => {
            frames.iter().map(|[l, r]| ((l + r) * 0.5) as f64).collect()
        }
        AudioData::F32(mono) => mono.iter().map(|&s| s as f64).collect(),
        AudioData::I16(mono) => mono.iter().map(|&s| s as f64 / 32768.0).collect(),
    }
}

/// Rank pre-rendered candidates against a target (prompt) embedding using
/// the provided audio-embedding function. Returns indices-with-scores sorted
/// best-first. Separated from the ORT sessions so the selection logic is
/// testable without models: `embed_audio` can be a stub.
pub fn rank_by_prompt<E>(
    candidates: &[(u64, Vec<f64>)],
    target_embedding: &[f32],
    mut embed_audio: E,
) -> Result<Vec<CandidateScore>>
where
    E: FnMut(&[f64]) -> Result<Vec<f32>>,
{
    let mut scores = Vec::with_capacity(candidates.len());
    for (seed, waveform) in candidates {
        let emb = embed_audio(waveform)?;
        scores.push(CandidateScore {
            seed: *seed,
            similarity: cosine_similarity(&emb, target_embedding),
        });
    }
    // Best first; ties broken by lower seed for determinism.
    scores.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.seed.cmp(&b.seed))
    });
    Ok(scores)
}

/// Compose `n_candidates` pieces (seeds `base_seed..base_seed+n`), rank them
/// against `prompt`, and return the best composition plus the full ranking
/// (so callers can report how close the match actually was, not just assert
/// success). The candidates differ genuinely — seed drives motif choice,
/// orientation, form (ternary/rondo), accompaniment pattern, and instrument
/// ensemble.
pub fn steer(
    intent_base: &MusicalIntent,
    style: Style,
    state: &MusicalState,
    prompt: &str,
    n_candidates: u64,
    audio_tower: &mut ClapEmbedder,
    text_tower: &mut ClapTextEmbedder,
) -> Result<(Composition, Vec<CandidateScore>)> {
    anyhow::ensure!(n_candidates > 0, "need at least one candidate");
    let target = text_tower.embed(prompt)?;

    let mut rendered: Vec<(u64, Composition)> = Vec::with_capacity(n_candidates as usize);
    let mut waveforms: Vec<(u64, Vec<f64>)> = Vec::with_capacity(n_candidates as usize);
    for i in 0..n_candidates {
        let seed = intent_base.seed.wrapping_add(i);
        let intent = MusicalIntent {
            seed,
            ..*intent_base
        };
        let comp = compose_and_realize_styled(&intent, style, state, STEERING_SAMPLE_RATE);
        waveforms.push((seed, downmix(&comp.audio)));
        rendered.push((seed, comp));
    }

    let scores = rank_by_prompt(&waveforms, &target, |w| audio_tower.embed(w))?;
    let best_seed = scores[0].seed;
    let best = rendered
        .into_iter()
        .find(|(seed, _)| *seed == best_seed)
        .map(|(_, c)| c)
        .expect("best seed came from the same candidate set");
    Ok((best, scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_picks_the_closest_embedding_and_orders_the_rest() {
        // Stub embedder: maps each waveform to a fixed vector by its first
        // sample. Target is closest to candidate 2, then 0, then 1.
        let candidates = vec![
            (10u64, vec![0.0f64]),
            (11u64, vec![1.0f64]),
            (12u64, vec![2.0f64]),
        ];
        let target = [1.0f32, 0.2];
        let scores = rank_by_prompt(&candidates, &target, |w| {
            Ok(match w[0] as i64 {
                0 => vec![1.0, 0.0],  // sim ~0.98
                1 => vec![0.0, 1.0],  // sim ~0.20
                _ => vec![1.0, 0.21], // sim ~1.00
            })
        })
        .unwrap();
        assert_eq!(scores[0].seed, 12);
        assert_eq!(scores[1].seed, 10);
        assert_eq!(scores[2].seed, 11);
        assert!(scores[0].similarity >= scores[1].similarity);
        assert!(scores[1].similarity >= scores[2].similarity);
    }

    #[test]
    fn ranking_ties_break_deterministically_by_seed() {
        let candidates = vec![(7u64, vec![0.0f64]), (3u64, vec![0.0f64])];
        let target = [1.0f32];
        let scores = rank_by_prompt(&candidates, &target, |_| Ok(vec![1.0])).unwrap();
        assert_eq!(scores[0].seed, 3, "equal similarity → lower seed first");
    }

    #[test]
    fn downmix_averages_stereo() {
        let audio = AudioData::StereoF32(vec![[1.0, 0.0], [0.5, 0.5]]);
        let mono = downmix(&audio);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-9);
        assert!((mono[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn embedding_errors_propagate_not_panic() {
        let candidates = vec![(1u64, vec![0.0f64])];
        let err = rank_by_prompt(&candidates, &[1.0], |_| anyhow::bail!("ORT exploded"));
        assert!(err.is_err());
    }
}
