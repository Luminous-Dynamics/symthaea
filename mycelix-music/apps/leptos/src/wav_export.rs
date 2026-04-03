// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! WAV export for consciousness compositions.
//!
//! Offline-renders a Journey as stereo WAV audio using the SynthEngine,
//! then triggers a browser download via Blob URL.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::journey::Journey;
use crate::synth_engine::{SynthEngine, SAMPLE_RATE};

/// Render a Journey to a WAV byte vector (offline, not real-time).
pub fn render_journey_to_wav(journey: &Journey) -> Vec<u8> {
    let mut engine = SynthEngine::new();
    engine.running = true;

    let chunk_size = engine.chunk_samples();
    let total_samples = (journey.duration_secs * SAMPLE_RATE) as usize;
    let total_chunks = (total_samples + chunk_size - 1) / chunk_size;

    let mut all_samples: Vec<f32> = Vec::with_capacity(total_samples * 2);
    let mut time_offset = 0.0_f32;

    for _ in 0..total_chunks {
        let (v, a, p) = journey.sample(time_offset);
        engine.set_state(a, v, p);

        let chunk = engine.render();
        time_offset += chunk.len() as f32 / 2.0 / SAMPLE_RATE;
        all_samples.extend_from_slice(&chunk);
    }

    let exact_len = total_samples * 2;
    all_samples.truncate(exact_len);

    encode_wav_16bit(&all_samples, SAMPLE_RATE as u32, 2)
}

/// Encode interleaved f32 stereo samples as 16-bit PCM WAV.
fn encode_wav_16bit(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let num_samples = samples.len();
    let bytes_per_sample = 2u16; // 16-bit
    let data_size = (num_samples * bytes_per_sample as usize) as u32;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * bytes_per_sample;
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 [-1, 1] to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i16_val.to_le_bytes());
    }

    buf
}

/// Trigger a browser download of WAV data.
pub fn download_wav(data: &[u8], filename: &str) {
    let uint8_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
    uint8_array.copy_from(data);

    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());

    let mut options = web_sys::BlobPropertyBag::new();
    options.type_("audio/wav");

    let blob = match web_sys::Blob::new_with_buffer_source_sequence_and_options(&array, &options) {
        Ok(b) => b,
        Err(_) => return,
    };

    let url = match web_sys::Url::create_object_url_with_blob(&blob) {
        Ok(u) => u,
        Err(_) => return,
    };

    // Create a temporary <a> element and click it
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Ok(el) = doc.create_element("a") {
            let a = el.dyn_ref::<web_sys::HtmlElement>().unwrap();
            let _ = el.set_attribute("href", &url);
            let _ = el.set_attribute("download", filename);
            a.style().set_property("display", "none").ok();
            if let Some(body) = doc.body() {
                let _ = body.append_child(&el);
                a.click();
                let _ = body.remove_child(&el);
            }
        }
        let _ = web_sys::Url::revoke_object_url(&url);
    }
}
