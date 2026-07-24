// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One Web Audio analysis graph for Muse's one persistent `<audio>` element.
//! It is deliberately derived presentation state: playback remains owned by
//! the browser element and the pure playback reducer.

use std::cell::RefCell;
use web_sys::{AnalyserNode, AudioContext, HtmlAudioElement, MediaElementAudioSourceNode};

struct AudioGraph {
    context: AudioContext,
    analyser: AnalyserNode,
    // Retain the source node for the lifetime of the graph. Browsers permit
    // only one MediaElementSource for a particular media element.
    _source: MediaElementAudioSourceNode,
}

thread_local! {
    static GRAPH: RefCell<Option<AudioGraph>> = const { RefCell::new(None) };
}

/// Connect the root media element once. Safe to call for every load/play.
pub fn ensure_connected(audio: &HtmlAudioElement) -> Result<(), String> {
    GRAPH.with(|slot| {
        if let Some(graph) = slot.borrow().as_ref() {
            let _ = graph.context.resume();
            return Ok(());
        }

        let context = AudioContext::new().map_err(|_| "Web Audio is unavailable".to_string())?;
        let analyser = context
            .create_analyser()
            .map_err(|_| "could not create audio analyser".to_string())?;
        analyser.set_fft_size(512);
        analyser.set_smoothing_time_constant(0.72);
        analyser.set_min_decibels(-92.0);
        analyser.set_max_decibels(-18.0);

        let source = context
            .create_media_element_source(audio)
            .map_err(|_| "could not connect Muse's audio element".to_string())?;
        source
            .connect_with_audio_node(&analyser)
            .map_err(|_| "could not connect audio to analyser".to_string())?;
        analyser
            .connect_with_audio_node(&context.destination())
            .map_err(|_| "could not connect analysed audio to output".to_string())?;
        let _ = context.resume();
        *slot.borrow_mut() = Some(AudioGraph {
            context,
            analyser,
            _source: source,
        });
        Ok(())
    })
}

/// A compact spectrum for visual motion. Values are measured from the actual
/// rendered music, never guessed from the score or wall clock.
pub fn spectrum() -> Option<[u8; 64]> {
    GRAPH.with(|slot| {
        let borrow = slot.borrow();
        let analyser = &borrow.as_ref()?.analyser;
        let mut bins = [0_u8; 64];
        analyser.get_byte_frequency_data(&mut bins);
        Some(bins)
    })
}

/// The instantaneous pressure waveform used by Wave mode. Unlike the
/// spectrum bars, this preserves polarity and zero crossings, so the drawn
/// line visibly follows attacks, releases, and silence in the music.
pub fn waveform() -> Option<[u8; 256]> {
    GRAPH.with(|slot| {
        let borrow = slot.borrow();
        let analyser = &borrow.as_ref()?.analyser;
        let mut samples = [128_u8; 256];
        analyser.get_byte_time_domain_data(&mut samples);
        Some(samples)
    })
}
