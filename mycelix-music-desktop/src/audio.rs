// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native audio: pre-filled ring buffer + dedicated synthesis thread.

use bevy::prelude::*;
use ringbuf::traits::{Split, Producer, Consumer, Observer};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use symthaea_muse::streaming::StreamingSynth;
use symthaea_muse::{MuseConfig, MusicalState};

struct SharedState {
    musical_state: Mutex<MusicalState>,
    running: AtomicBool,
}

#[derive(Resource)]
pub struct AudioEngine {
    shared: Arc<SharedState>,
    _synth_thread: Option<thread::JoinHandle<()>>,
    _stream: Option<cpal::Stream>,
}

#[derive(Resource, Default)]
pub struct AudioPlaying(pub bool);

pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AudioPlaying>()
            .add_systems(Startup, setup_audio)
            .add_systems(Update, sync_consciousness_to_synth);
    }
}

fn setup_audio(mut commands: Commands) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => { eprintln!("[audio] No output device"); return; }
    };
    let supported = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => { eprintln!("[audio] No config: {e}"); return; }
    };

    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    eprintln!("[audio] Output: {}Hz, {} channels", sample_rate, channels);

    // Ring buffer: 2 seconds
    let ring_size = sample_rate as usize * 4;
    let rb = ringbuf::HeapRb::<[f32; 2]>::new(ring_size);
    let (mut producer, mut consumer) = rb.split();

    // Create synth on main thread and pre-fill BEFORE starting the stream
    let config = MuseConfig {
        sample_rate,
        duration_secs: 86400.0,
        max_notes: 10,
        num_partials: 6,
        enable_antialiasing: true,
        enable_sub_bass: true,
        ..Default::default()
    };

    let mut synth = StreamingSynth::new(config, sample_rate);
    synth.enable_binaural = false;
    synth.enable_sidechain = true;
    synth.enable_fep = false;
    synth.feedback_strength = 0.2;

    // Pre-fill 1 second of audio BEFORE starting the cpal stream
    for _ in 0..30 {
        let chunk = synth.render_chunk();
        for sample in &chunk {
            let _ = producer.try_push(*sample);
        }
    }
    eprintln!("[audio] Pre-filled {} samples before stream start", producer.occupied_len());

    // NOW start the cpal stream (buffer is full, no underruns)
    let stream = device.build_output_stream(
        &supported.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / channels;
            for frame in 0..frames {
                let sample: [f32; 2] = consumer.try_pop().unwrap_or([0.0, 0.0]);
                let base = frame * channels;
                if base < data.len() {
                    data[base] = sample[0];
                    if channels > 1 && base + 1 < data.len() {
                        data[base + 1] = sample[1];
                    }
                }
            }
        },
        |err| eprintln!("[audio] Stream error: {err}"),
        None,
    );

    let shared = Arc::new(SharedState {
        musical_state: Mutex::new(MusicalState::default()),
        running: AtomicBool::new(true),
    });

    match stream {
        Ok(s) => {
            if let Err(e) = s.play() {
                eprintln!("[audio] Failed to start: {e}");
                return;
            }
            eprintln!("[audio] Stream started — ring buffer pre-filled");

            // Hand off synth to dedicated thread for ongoing rendering
            let shared_synth = shared.clone();
            let synth_thread = thread::spawn(move || {
                eprintln!("[audio] Synth thread started — 6 partials, real-time");
                while shared_synth.running.load(Ordering::Relaxed) {
                    if let Ok(state) = shared_synth.musical_state.lock() {
                        synth.update_state(&state);
                    }

                    let chunk = synth.render_chunk();
                    for sample in &chunk {
                        let _ = producer.try_push(*sample);
                    }

                    // Pace: don't overproduce
                    if producer.vacant_len() < ring_size / 4 {
                        thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            });

            commands.insert_resource(AudioEngine {
                shared,
                _synth_thread: Some(synth_thread),
                _stream: Some(s),
            });
            commands.insert_resource(AudioPlaying(true));
        }
        Err(e) => eprintln!("[audio] Failed to build stream: {e}"),
    }
}

fn sync_consciousness_to_synth(
    consciousness: Res<crate::consciousness::ConsciousnessState>,
    engine: Option<Res<AudioEngine>>,
) {
    if let Some(engine) = engine {
        if let Ok(mut state) = engine.shared.musical_state.try_lock() {
            *state = consciousness.musical_state.clone();
        }
    }
}
