// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Web Audio playback bridge for live consciousness synthesis.
//! Renders audio via requestAnimationFrame with generous lookahead.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{AnalyserNode, AudioContext, GainNode};

use crate::audio_bridge::{compute_analysis, AudioAnalysis, BIN_COUNT, FFT_SIZE};
use crate::synth_engine::{SynthEngine, SAMPLE_RATE};

const SCHEDULE_AHEAD: f64 = 0.5;

pub struct LiveSynth {
    context: AudioContext,
    engine: Rc<RefCell<SynthEngine>>,
    gain: GainNode,
    analyser: AnalyserNode,
}

impl LiveSynth {
    pub fn new() -> Result<Self, JsValue> {
        let context = AudioContext::new()?;
        let _ = context.resume();

        let gain = context.create_gain()?;
        let analyser = context.create_analyser()?;
        analyser.set_fft_size(FFT_SIZE);
        analyser.set_smoothing_time_constant(0.5);

        gain.connect_with_audio_node(&analyser)?;
        gain.connect_with_audio_node(&context.destination())?;

        let engine = Rc::new(RefCell::new(SynthEngine::new()));

        Ok(Self { context, engine, gain, analyser })
    }

    pub fn engine(&self) -> Rc<RefCell<SynthEngine>> {
        self.engine.clone()
    }

    pub fn start(self) -> Rc<RefCell<LiveSynthRunner>> {
        let runner = Rc::new(RefCell::new(LiveSynthRunner {
            context: self.context,
            engine: self.engine,
            gain: self.gain,
            analyser: self.analyser,
            prev_frequency: vec![0u8; BIN_COUNT],
            next_buffer_time: 0.0,
            running: true,
        }));

        runner.borrow().engine.borrow_mut().running = true;

        {
            let mut r = runner.borrow_mut();
            r.next_buffer_time = r.context.current_time() + 0.05;
        }

        let runner_clone = runner.clone();
        schedule_loop(runner_clone);

        runner
    }
}

pub struct LiveSynthRunner {
    context: AudioContext,
    engine: Rc<RefCell<SynthEngine>>,
    gain: GainNode,
    analyser: AnalyserNode,
    prev_frequency: Vec<u8>,
    next_buffer_time: f64,
    pub running: bool,
}

impl LiveSynthRunner {
    pub fn engine(&self) -> Rc<RefCell<SynthEngine>> {
        self.engine.clone()
    }

    pub fn stop(&mut self) {
        self.running = false;
        self.engine.borrow_mut().running = false;
    }

    pub fn analyse(&mut self) -> AudioAnalysis {
        compute_analysis(&self.analyser, &mut self.prev_frequency)
    }

    fn schedule_buffers(&mut self) {
        if !self.running { return; }

        let current_time = self.context.current_time();
        let mut chunks = 0u32;

        while self.next_buffer_time < current_time + SCHEDULE_AHEAD {
            let samples = self.engine.borrow_mut().render();
            let chunk_frames = samples.len() / 2;
            if chunk_frames == 0 { break; }
            let chunk_duration = chunk_frames as f64 / SAMPLE_RATE as f64;

            let buffer = match self.context.create_buffer(2, chunk_frames as u32, SAMPLE_RATE as f32) {
                Ok(b) => b,
                Err(_) => return,
            };

            let mut left = vec![0.0f32; chunk_frames];
            let mut right = vec![0.0f32; chunk_frames];
            for i in 0..chunk_frames {
                left[i] = samples[i * 2];
                right[i] = samples[i * 2 + 1];
            }

            if buffer.copy_to_channel(&left, 0).is_err() { return; }
            if buffer.copy_to_channel(&right, 1).is_err() { return; }

            let source = match self.context.create_buffer_source() {
                Ok(s) => s,
                Err(_) => return,
            };
            source.set_buffer(Some(&buffer));
            if source.connect_with_audio_node(&self.gain).is_err() { return; }

            let start_time = self.next_buffer_time.max(current_time);
            if source.start_with_when(start_time).is_err() { return; }

            self.next_buffer_time = start_time + chunk_duration;
            chunks += 1;
            if chunks >= 10 { break; } // Don't block main thread too long
        }
    }
}

fn schedule_loop(runner: Rc<RefCell<LiveSynthRunner>>) {
    let runner_clone = runner.clone();

    let closure = Closure::once(Box::new(move || {
        {
            let mut r = runner_clone.borrow_mut();
            if !r.running { return; }
            r.schedule_buffers();
        }
        if runner_clone.borrow().running {
            schedule_loop(runner_clone);
        }
    }) as Box<dyn FnOnce()>);

    if let Some(window) = web_sys::window() {
        let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
    }
    closure.forget();
}
