// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Web Audio API bridge for real-time spectral analysis.
//!
//! Provides both a catalog-player bridge (AudioBridge) and shared analysis
//! math (compute_analysis) reusable by the synthesis path.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{AnalyserNode, AudioContext, HtmlAudioElement, MediaElementAudioSourceNode};

/// Number of FFT frequency bins (must be power of 2).
pub const FFT_SIZE: u32 = 256;
/// Number of usable frequency bins (FFT_SIZE / 2).
pub const BIN_COUNT: usize = (FFT_SIZE / 2) as usize;

/// Live audio analysis data extracted each frame.
#[derive(Clone, Debug)]
pub struct AudioAnalysis {
    pub frequency_data: Vec<u8>,
    pub waveform_data: Vec<u8>,
    pub spectral_centroid: f32,
    pub rms_energy: f32,
    pub spectral_flux: f32,
}

impl Default for AudioAnalysis {
    fn default() -> Self {
        Self {
            frequency_data: vec![0u8; BIN_COUNT],
            waveform_data: vec![0u8; FFT_SIZE as usize],
            spectral_centroid: 0.0,
            rms_energy: 0.0,
            spectral_flux: 0.0,
        }
    }
}

/// Compute spectral analysis from raw FFT data.
/// Shared between AudioBridge (catalog player) and LiveSynthRunner (synthesis).
pub fn compute_analysis(
    analyser: &AnalyserNode,
    prev_frequency: &mut Vec<u8>,
) -> AudioAnalysis {
    let mut freq = vec![0u8; BIN_COUNT];
    let mut wave = vec![0u8; FFT_SIZE as usize];

    analyser.get_byte_frequency_data(&mut freq);
    analyser.get_byte_time_domain_data(&mut wave);

    // Spectral centroid: weighted average of frequency bin indices
    let mut weighted_sum = 0.0_f32;
    let mut total_energy = 0.0_f32;
    for (i, &mag) in freq.iter().enumerate() {
        let m = mag as f32;
        weighted_sum += i as f32 * m;
        total_energy += m;
    }
    let spectral_centroid = if total_energy > 0.0 {
        (weighted_sum / total_energy / BIN_COUNT as f32).min(1.0)
    } else {
        0.0
    };

    // RMS energy from waveform (centered around 128)
    let rms_energy = {
        let sum_sq: f32 = wave
            .iter()
            .map(|&s| {
                let centered = (s as f32 - 128.0) / 128.0;
                centered * centered
            })
            .sum();
        (sum_sq / wave.len() as f32).sqrt().min(1.0)
    };

    // Spectral flux: half-wave rectified L1 change from previous frame
    let spectral_flux = {
        let flux: f32 = freq
            .iter()
            .zip(prev_frequency.iter())
            .map(|(&curr, &prev)| {
                let diff = curr as f32 - prev as f32;
                if diff > 0.0 { diff } else { 0.0 }
            })
            .sum();
        (flux / (BIN_COUNT as f32 * 255.0) * 4.0).min(1.0)
    };

    prev_frequency.copy_from_slice(&freq);

    AudioAnalysis {
        frequency_data: freq,
        waveform_data: wave,
        spectral_centroid,
        rms_energy,
        spectral_flux,
    }
}

/// Bridge between the HTML5 audio element and the Web Audio API analyser.
pub struct AudioBridge {
    _context: AudioContext,
    analyser: AnalyserNode,
    _source: MediaElementAudioSourceNode,
    prev_frequency: Vec<u8>,
}

impl AudioBridge {
    pub fn new(audio_element: &HtmlAudioElement) -> Result<Self, JsValue> {
        let context = AudioContext::new()?;

        let analyser = context.create_analyser()?;
        analyser.set_fft_size(FFT_SIZE);
        analyser.set_smoothing_time_constant(0.5);

        let source = context.create_media_element_source(audio_element)?;
        source.connect_with_audio_node(&analyser)?;
        analyser.connect_with_audio_node(&context.destination())?;

        Ok(Self {
            _context: context,
            analyser,
            _source: source,
            prev_frequency: vec![0u8; BIN_COUNT],
        })
    }

    pub fn analyse(&mut self) -> AudioAnalysis {
        compute_analysis(&self.analyser, &mut self.prev_frequency)
    }
}

pub type SharedAudioBridge = Rc<RefCell<Option<AudioBridge>>>;

pub fn create_shared_bridge() -> SharedAudioBridge {
    Rc::new(RefCell::new(None))
}
