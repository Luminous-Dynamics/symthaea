// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neural singing built from Kokoro's intelligible voice and Muse melodies.
//!
//! Each word is rendered by Kokoro, divided across its lyric syllables, then
//! moved onto the corresponding notes with an STFT phase vocoder. Pitch and
//! duration are controlled independently, avoiding the speed/pitch coupling
//! of the original resampling spike.

use anyhow::{Result, anyhow};
use rustfft::{FftPlanner, num_complex::Complex};
use symthaea_muse::Note;

use super::singing_engine::{SingingVoiceEngine, VocalPerformance, VocalStem, VocalSyllable};
use super::{KokoroConfig, KokoroEngine};

const FFT_SIZE: usize = 1024;
const ANALYSIS_HOP: usize = 256;

pub struct KokoroSingingEngine {
    engine: KokoroEngine,
    voice: String,
}

impl KokoroSingingEngine {
    pub fn load(voice: &str) -> Result<Self> {
        let config = KokoroConfig {
            voices_filename: format!("voices/{voice}.bin"),
            ..KokoroConfig::default()
        };
        let mut engine =
            KokoroEngine::load(config).ok_or_else(|| anyhow!("Kokoro failed to load"))?;
        engine.speed = Some(0.92);
        Ok(Self {
            engine,
            voice: voice.to_string(),
        })
    }
}

impl SingingVoiceEngine for KokoroSingingEngine {
    fn id(&self) -> &str {
        "kokoro-v2"
    }

    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem> {
        performance.validate()?;
        let sample_rate = self.engine.sample_rate();
        // One phrase-level synthesis preserves coarticulation and breath placement.
        let raw = self
            .engine
            .synthesize(&performance.lyrics, None)
            .ok_or_else(|| anyhow!("Kokoro could not synthesize the vocal phrase"))?;
        let voiced = trim_outer_silence(&raw, sample_rate);
        let syllable_weights: Vec<f32> = performance
            .syllables
            .iter()
            .map(|s| {
                s.phonemes
                    .iter()
                    .map(|p| p.natural_duration_s.max(0.02))
                    .sum()
            })
            .collect();
        let source_boundaries = acoustic_boundaries(voiced, &syllable_weights, sample_rate);
        let mut rendered: Vec<(usize, Vec<f32>)> = Vec::new();
        for (syllable_index, syllable) in performance.syllables.iter().enumerate() {
            let source =
                &voiced[source_boundaries[syllable_index]..source_boundaries[syllable_index + 1]];
            let sung = render_syllable(source, syllable, sample_rate);
            rendered.push((
                ((syllable.note.start_time - syllable.consonant_advance_s).max(0.0)
                    * sample_rate as f32) as usize,
                sung,
            ));
        }

        let output_len = rendered
            .iter()
            .map(|(start, audio)| start + audio.len())
            .max()
            .unwrap_or(0);
        let mut output = vec![0.0f32; output_len];
        for (start, audio) in rendered {
            for (dst, src) in output[start..].iter_mut().zip(audio) {
                *dst += src;
            }
        }
        normalize_peak(&mut output, 0.92);
        let stem = VocalStem {
            samples: output,
            sample_rate,
            backend: format!("kokoro-v2:{}", self.voice),
        };
        stem.validate()?;
        Ok(stem)
    }
}

/// Compatibility entry point used by the REPL and existing examples.
pub fn sing_with_kokoro(lyrics: &str, melody: &[Note], voice: &str) -> Result<(Vec<f32>, u32)> {
    if lyrics.trim().is_empty() || melody.is_empty() {
        return Ok((Vec::new(), 24_000));
    }
    // Normalize the melody's timeline to start at t=0. Composers may
    // position the first USABLE note well into a piece (an intro/count-in
    // section before the melodic content proper) -- fine for a full
    // composition, but callers here truncate the melody down to just the
    // notes actually being sung (the REPL's /sing handler and every eval
    // in this investigation both do), and without re-basing the timeline
    // that leaves a dead silent lead-in exactly as long as whatever the
    // first surviving note's original start_time was. Found 2026-07-22
    // from direct listening ("most of the audio is silent, some parts
    // very sped up or slowed down"): every composed test melody's first
    // note started at ~5.0s regardless of phrase, producing a ~5s silent
    // lead-in on every clip and compressing the actual sung content into
    // whatever fraction of the nominal duration remained.
    let offset = melody
        .iter()
        .map(|note| note.start_time)
        .fold(f32::MAX, f32::min)
        .max(0.0);
    let melody: Vec<Note> = if offset > 0.0 {
        melody
            .iter()
            .map(|note| Note {
                start_time: note.start_time - offset,
                ..*note
            })
            .collect()
    } else {
        melody.to_vec()
    };
    let performance = VocalPerformance::from_melody(lyrics, &melody, "en")?;
    let mut engine = KokoroSingingEngine::load(voice)?;
    let stem = engine.render(&performance)?;
    Ok((stem.samples, stem.sample_rate))
}

fn trim_outer_silence(samples: &[f32], sample_rate: u32) -> &[f32] {
    let peak = samples.iter().fold(0.0f32, |p, &x| p.max(x.abs()));
    let threshold = (peak * 0.015).max(1e-4);
    let start = samples
        .iter()
        .position(|x| x.abs() >= threshold)
        .unwrap_or(0);
    let end = samples
        .iter()
        .rposition(|x| x.abs() >= threshold)
        .map(|i| i + 1)
        .unwrap_or(samples.len());
    // Preserve 35ms around detected speech so breaths and pickups survive.
    let padding = (sample_rate as f32 * 0.035) as usize;
    &samples[start.saturating_sub(padding)..(end + padding).min(samples.len()).max(start)]
}

fn render_syllable(source: &[f32], syllable: &VocalSyllable, sample_rate: u32) -> Vec<f32> {
    let target_duration = (syllable.end_time_s() - syllable.note.start_time).max(0.06);
    let target_len = (target_duration * sample_rate as f32) as usize;
    if source.is_empty() {
        return vec![0.0; target_len];
    }
    let vowel_count = syllable
        .phonemes
        .iter()
        .filter(|p| p.is_vowel)
        .count()
        .max(1);
    let raw_consonant_total: usize = syllable
        .phonemes
        .iter()
        .filter(|p| !p.is_vowel)
        .map(|p| (p.natural_duration_s.clamp(0.025, 0.12) * sample_rate as f32) as usize)
        .sum();
    let consonant_total =
        raw_consonant_total.min(target_len.saturating_sub((0.055 * sample_rate as f32) as usize));
    let consonant_scale = consonant_total as f32 / raw_consonant_total.max(1) as f32;
    let vowel_total = target_len.saturating_sub(consonant_total);
    let phoneme_weights: Vec<f32> = syllable
        .phonemes
        .iter()
        .map(|p| p.natural_duration_s.max(0.02))
        .collect();
    let source_boundaries = acoustic_boundaries(source, &phoneme_weights, sample_rate);
    let mut pieces: Vec<Vec<f32>> = Vec::with_capacity(syllable.phonemes.len());

    for (i, phoneme) in syllable.phonemes.iter().enumerate() {
        let chunk = &source[source_boundaries[i]..source_boundaries[i + 1]];
        let destination_len = if phoneme.is_vowel {
            vowel_total / vowel_count
        } else {
            (phoneme.natural_duration_s.clamp(0.025, 0.12) * sample_rate as f32 * consonant_scale)
                as usize
        };
        let mut piece = if phoneme.is_vowel {
            render_vowel_notes(chunk, syllable, sample_rate, destination_len)
        } else {
            // Consonants carry identity and intelligibility in their transients;
            // change duration gently but never force them onto the musical F0.
            resample_exact(chunk, destination_len)
        };
        fade_edges(&mut piece, (sample_rate as f32 * 0.003) as usize);
        pieces.push(piece);
    }
    let mut output = Vec::with_capacity(target_len);
    for piece in pieces {
        output.extend(piece);
    }
    output.resize(target_len, 0.0);
    output.truncate(target_len);
    let gain = 0.58 + syllable.energy.clamp(0.0, 1.0) * 0.42;
    output.iter_mut().for_each(|x| *x *= gain);
    fade_edges(&mut output, (sample_rate as f32 * 0.008) as usize);
    output
}

/// Use natural-duration predictions as anchors, then snap cuts to nearby
/// low-energy regions. This is a deterministic acoustic aligner: it avoids
/// slicing through plosives and vowel centers without requiring a second
/// neural model in the zero-provision Kokoro fallback.
fn acoustic_boundaries(samples: &[f32], weights: &[f32], sample_rate: u32) -> Vec<usize> {
    if weights.is_empty() {
        return vec![0, samples.len()];
    }
    let total = weights.iter().sum::<f32>().max(1e-6);
    let mut boundaries = Vec::with_capacity(weights.len() + 1);
    boundaries.push(0);
    let mut cumulative = 0.0;
    let min_gap = (sample_rate as f32 * 0.012) as usize;
    for weight in weights.iter().take(weights.len() - 1) {
        cumulative += *weight;
        let expected = (samples.len() as f32 * cumulative / total) as usize;
        let radius = (samples.len() / 12).min((sample_rate as f32 * 0.09) as usize);
        let previous = *boundaries.last().unwrap_or(&0);
        let cut = snap_to_low_energy(samples, expected, radius, sample_rate)
            .clamp((previous + min_gap).min(samples.len()), samples.len());
        boundaries.push(cut);
    }
    boundaries.push(samples.len());
    boundaries
}

fn snap_to_low_energy(samples: &[f32], expected: usize, radius: usize, sample_rate: u32) -> usize {
    let window = ((sample_rate as f32 * 0.008) as usize).max(8);
    let from = expected.saturating_sub(radius).max(window);
    let to = (expected + radius).min(samples.len().saturating_sub(window));
    if from >= to {
        return expected.min(samples.len());
    }
    (from..=to)
        .step_by((window / 4).max(1))
        .min_by(|&a, &b| {
            let energy = |center: usize| {
                samples[center - window..center + window]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
            };
            energy(a).total_cmp(&energy(b))
        })
        .unwrap_or(expected)
}

fn render_vowel_notes(
    source: &[f32],
    syllable: &VocalSyllable,
    sample_rate: u32,
    destination_len: usize,
) -> Vec<f32> {
    let notes: Vec<_> = syllable.notes().collect();
    let total_duration = notes.iter().map(|n| n.duration.max(0.01)).sum::<f32>();
    let note_weights: Vec<f32> = notes.iter().map(|n| n.duration.max(0.01)).collect();
    let source_boundaries = acoustic_boundaries(source, &note_weights, sample_rate);
    let mut output = Vec::with_capacity(destination_len);
    let mut assigned = 0usize;
    for (index, note) in notes.iter().enumerate() {
        let length = if index + 1 == notes.len() {
            destination_len.saturating_sub(assigned)
        } else {
            (destination_len as f32 * note.duration.max(0.01) / total_duration) as usize
        };
        let chunk = &source[source_boundaries[index]..source_boundaries[index + 1]];
        let source_f0 = estimate_f0_track(chunk, sample_rate)
            .or_else(|| estimate_f0_track(source, sample_rate))
            .unwrap_or(220.0);
        let ratio = (note.frequency / source_f0).clamp(0.5, 2.0);
        let mut segment = retune_to_length(chunk, ratio, length);
        add_expression_vibrato(&mut segment, sample_rate, syllable);
        fade_edges(&mut segment, (sample_rate as f32 * 0.002) as usize);
        output.extend(segment);
        assigned += length;
    }
    output.resize(destination_len, 0.0);
    output
}

fn normalize_peak(samples: &mut [f32], ceiling: f32) {
    let peak = samples.iter().fold(0.0f32, |p, &x| p.max(x.abs()));
    if peak > ceiling && peak > 0.0 {
        let gain = ceiling / peak;
        samples.iter_mut().for_each(|x| *x *= gain);
    }
}

/// Autocorrelation F0 estimate, weighted toward the stable middle of a word.
fn estimate_f0(samples: &[f32], sample_rate: u32) -> Option<f32> {
    if samples.len() < 1024 {
        return None;
    }
    let span = samples.len().min(sample_rate as usize / 2);
    let start = samples.len().saturating_sub(span) / 2;
    let x = &samples[start..start + span];
    let min_lag = sample_rate as usize / 420;
    let max_lag = (sample_rate as usize / 85).min(x.len() / 2);
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in min_lag..=max_lag {
        let (dot, ea, eb) = x[..x.len() - lag].iter().zip(&x[lag..]).fold(
            (0.0f64, 0.0f64, 0.0f64),
            |(dot, ea, eb), (&a, &b)| {
                (
                    dot + (a * b) as f64,
                    ea + (a * a) as f64,
                    eb + (b * b) as f64,
                )
            },
        );
        let score = dot / (ea * eb).sqrt().max(1e-12);
        if score > best.1 {
            best = (lag, score);
        }
    }
    (best.0 > 0 && best.1 > 0.25).then(|| sample_rate as f32 / best.0 as f32)
}

fn estimate_f0_track(samples: &[f32], sample_rate: u32) -> Option<f32> {
    let frame = (sample_rate as f32 * 0.05) as usize;
    let hop = (sample_rate as f32 * 0.02) as usize;
    if samples.len() < frame {
        return estimate_f0(samples, sample_rate);
    }
    let mut pitches: Vec<f32> = (0..=samples.len() - frame)
        .step_by(hop.max(1))
        .filter_map(|start| estimate_f0(&samples[start..start + frame], sample_rate))
        .collect();
    if pitches.is_empty() {
        return None;
    }
    pitches.sort_by(f32::total_cmp);
    Some(pitches[pitches.len() / 2])
}

fn retune_to_length(input: &[f32], pitch_ratio: f32, target_len: usize) -> Vec<f32> {
    if input.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    // The phase-vocoder stretch is followed by resampling. Stretching by
    // target/input * pitch_ratio leaves exactly target_len samples after the
    // resampler changes pitch by pitch_ratio.
    let stretch = (target_len as f32 * pitch_ratio / input.len() as f32).clamp(0.3, 4.0);
    let stretched = phase_vocoder(input, stretch);
    resample_exact(&stretched, target_len)
}

/// Local-maxima peaks in `magnitudes[1..half]` (DC and Nyquist excluded)
/// for identity phase locking. Falls back to a single peak at DC when the
/// frame has no interior local maximum (near-silent/flat spectrum), which
/// degenerates every bin to locking against DC -- effectively disabling
/// locking for that frame, the same as the old unlocked behavior.
fn find_spectral_peaks(magnitudes: &[f32], half: usize) -> Vec<usize> {
    let mut peaks: Vec<usize> = (1..half)
        .filter(|&k| magnitudes[k] > magnitudes[k - 1] && magnitudes[k] >= magnitudes[k + 1])
        .collect();
    if peaks.is_empty() {
        peaks.push(0);
    }
    peaks
}

/// Assigns every bin in `0..=half` to its nearest peak (region of
/// influence split at the midpoint between adjacent peaks) -- the
/// standard Laroche & Dolson region-of-influence rule. `peaks` must be
/// sorted ascending (guaranteed by `find_spectral_peaks`'s scan order).
fn assign_regions_of_influence(peaks: &[usize], half: usize) -> Vec<usize> {
    let mut region = vec![0usize; half + 1];
    let mut peak_idx = 0;
    for (k, slot) in region.iter_mut().enumerate() {
        while peak_idx + 1 < peaks.len() && k >= (peaks[peak_idx] + peaks[peak_idx + 1]).div_ceil(2)
        {
            peak_idx += 1;
        }
        *slot = peaks[peak_idx];
    }
    region
}

fn phase_vocoder(input: &[f32], stretch: f32) -> Vec<f32> {
    let synthesis_hop = (ANALYSIS_HOP as f32 * stretch).round().max(1.0) as usize;
    let frames = input.len().div_ceil(ANALYSIS_HOP) + 1;
    let mut output = vec![0.0f32; frames * synthesis_hop + FFT_SIZE];
    let mut norm = vec![0.0f32; output.len()];
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT_SIZE as f32).cos())
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let ifft = planner.plan_fft_inverse(FFT_SIZE);
    let mut previous = vec![0.0f32; FFT_SIZE];
    let mut accumulated = vec![0.0f32; FFT_SIZE];

    for frame in 0..frames {
        let in_pos = frame * ANALYSIS_HOP;
        let out_pos = frame * synthesis_hop;
        let mut spectrum: Vec<Complex<f32>> = (0..FFT_SIZE)
            .map(|i| {
                Complex::new(
                    input.get(in_pos + i).copied().unwrap_or(0.0) * window[i],
                    0.0,
                )
            })
            .collect();
        fft.process(&mut spectrum);

        // Pass 1: raw analysis phase/magnitude, and the standard per-bin
        // phase-vocoder accumulation -- kept for EVERY bin (not just peaks)
        // so each bin's phase-increment history stays fresh for whichever
        // frame it happens to be a peak in; locked bins simply don't use
        // their own accumulated value as the OUTPUT phase (pass 2).
        let magnitudes: Vec<f32> = spectrum.iter().map(|bin| bin.norm()).collect();
        let phases: Vec<f32> = spectrum.iter().map(|bin| bin.arg()).collect();
        for k in 0..FFT_SIZE {
            let phase = phases[k];
            let expected = std::f32::consts::TAU * k as f32 * ANALYSIS_HOP as f32 / FFT_SIZE as f32;
            let mut delta = phase - previous[k] - expected;
            delta -= std::f32::consts::TAU * (delta / std::f32::consts::TAU).round();
            accumulated[k] += (expected + delta) * synthesis_hop as f32 / ANALYSIS_HOP as f32;
            previous[k] = phase;
        }

        // Pass 2: identity phase locking (Laroche & Dolson 1999). Naive
        // per-bin accumulation lets every bin's phase drift independently,
        // which destroys the phase COHERENCE between what were
        // harmonically-related overtones once the fundamental shifts --
        // the classic phase-vocoder "phasiness"/buzzy-robotic artifact on
        // pitched content, root-caused 2026-07-21 chasing a "random fart
        // noises" report (a click-based investigation turned out to be
        // chasing a false signal -- see SYMTHAEA_SINGING_PLAN_2026-07-18.md
        // for the full trail). Bins are grouped into regions of influence
        // around each frame's spectral peaks; every bin in a region is
        // locked to move RIGIDLY with its peak (same absolute phase
        // advance as the peak, offset by their fixed analysis-frame phase
        // difference) instead of accumulating its own independent history.
        let half = FFT_SIZE / 2;
        let peaks = find_spectral_peaks(&magnitudes, half);
        let region = assign_regions_of_influence(&peaks, half);
        for k in 0..=half {
            let p = region[k];
            let locked_phase = accumulated[p] + (phases[k] - phases[p]);
            spectrum[k] = Complex::from_polar(magnitudes[k], locked_phase);
        }
        // Real input -> conjugate-symmetric spectrum required for a real
        // (imaginary-free) ifft output; mirror rather than let the upper
        // half accumulate its own independent (and asymmetric) phase.
        for k in (half + 1)..FFT_SIZE {
            spectrum[k] = spectrum[FFT_SIZE - k].conj();
        }

        ifft.process(&mut spectrum);
        for i in 0..FFT_SIZE {
            let weight = window[i];
            output[out_pos + i] += spectrum[i].re * weight / FFT_SIZE as f32;
            norm[out_pos + i] += weight * weight;
        }
    }
    // Floor RELATIVE to the peak coverage this specific call actually
    // reaches, not a fixed absolute value calibrated for steady-state 4x
    // overlap. Most inputs here are short (a single phoneme or a note
    // within a multi-note syllable, both routinely well under FFT_SIZE),
    // so a short call may never reach steady-state coverage ANYWHERE in
    // its output -- a fixed absolute floor (the previous 0.05, tuned for
    // long/fully-covered audio) zeroed out large, genuinely-real portions
    // of short segments, not just their true edges. Found 2026-07-22 from
    // direct listening ("most of the audio is silent, some parts very
    // sped up or slowed down") -- exactly what zeroing real content
    // inside a fixed-duration slot, then linearly resampling that whole
    // buffer to fill the slot (`resample_exact` in `retune_to_length`),
    // produces: silence where content was zeroed, and the surviving
    // audible fragments compressed into whatever fraction of the slot
    // wasn't. A small absolute floor is still needed to guard the
    // original numerically-unstable near-zero division this code exists
    // to prevent, but it must scale down for calls that never reach much
    // coverage at all.
    let max_norm = norm.iter().cloned().fold(0.0f32, f32::max);
    let norm_floor = (max_norm * 0.02).max(1e-4);
    for (x, n) in output.iter_mut().zip(norm) {
        if n > norm_floor {
            *x /= n;
        } else {
            *x = 0.0;
        }
    }
    output
}

fn resample_exact(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![0.0; output_len];
    }
    let scale = (input.len() - 1) as f32 / output_len.saturating_sub(1).max(1) as f32;
    (0..output_len)
        .map(|i| {
            let pos = i as f32 * scale;
            let a = pos.floor() as usize;
            let b = (a + 1).min(input.len() - 1);
            input[a] + (input[b] - input[a]) * pos.fract()
        })
        .collect()
}

fn add_expression_vibrato(samples: &mut [f32], sample_rate: u32, syllable: &VocalSyllable) {
    if syllable.vibrato_depth_cents <= 0.0 || syllable.vibrato_onset >= 1.0 {
        return;
    }
    let depth_ratio = 2.0f32.powf(syllable.vibrato_depth_cents / 1200.0) - 1.0;
    let depth = depth_ratio * sample_rate as f32
        / (std::f32::consts::TAU * syllable.vibrato_rate_hz.max(1.0));
    let dry = samples.to_vec();
    let delay = depth.ceil() + 2.0;
    let onset = (samples.len() as f32 * syllable.vibrato_onset.clamp(0.0, 1.0)) as usize;
    for (i, sample) in samples.iter_mut().enumerate() {
        if i < onset {
            continue;
        }
        let ramp = ((i - onset) as f32 / (sample_rate as f32 * 0.12)).clamp(0.0, 1.0);
        let phase =
            std::f32::consts::TAU * syllable.vibrato_rate_hz * i as f32 / sample_rate as f32;
        // `delay` must be ramped too, not just the modulation depth: at
        // i == onset, ramp == 0, but the un-ramped `- delay` term still
        // yanked the read position back by a full `delay` samples in one
        // step -- a discontinuity every single vibrato onset (found
        // 2026-07-21 chasing the "random fart noises" report; this
        // survived the phase_vocoder norm-floor fix untouched, which is
        // why that fix alone didn't reduce click counts). Ramping delay
        // alongside depth makes pos == i exactly at onset (a true
        // continuation of the dry signal), matching depth's existing
        // fade-in.
        let pos = i as f32 - delay * ramp - depth * ramp * phase.sin();
        if pos >= 0.0 {
            let a = pos.floor() as usize;
            let b = (a + 1).min(dry.len() - 1);
            *sample = dry[a] + (dry[b] - dry[a]) * pos.fract();
        }
    }
}

fn fade_edges(samples: &mut [f32], fade: usize) {
    let len = samples.len();
    if len < 2 {
        // `n = fade.min(len / 2)` is exactly 0 whenever len <= 1, so the
        // loop below never runs and a lone sample here is left completely
        // unfaded -- found 2026-07-21 chasing the "random fart noises"
        // report: degenerate 0-or-1-sample pieces are a real, reachable
        // case (render_vowel_notes divides destination_len per note, and
        // a short ornamental note in a melisma can round down to 0 or 1
        // sample), and a raw, unfaded sample sitting at a piece boundary
        // next to otherwise-zeroed neighbors is exactly what
        // singing_quality's contextual-click detector flags: a lone,
        // large sample-to-sample delta. Too short to apply any real fade
        // shape anyway, so zero it outright.
        samples.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    let n = fade.min(len / 2).max(1);
    for i in 0..n {
        let gain = i as f32 / n as f32;
        samples[i] *= gain;
        samples[len - 1 - i] *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retune_preserves_requested_duration() {
        let input: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.04).sin()).collect();
        for ratio in [0.75, 1.0, 1.5] {
            let output = retune_to_length(&input, ratio, 6000);
            assert_eq!(output.len(), 6000);
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }
}
