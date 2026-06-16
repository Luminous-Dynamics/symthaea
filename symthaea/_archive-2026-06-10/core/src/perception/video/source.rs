// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Video Source - Abstraction over video input devices
//!
//! Provides a unified interface for webcams, video files, and test data.

use super::frame::{Frame, FrameFormat};
use anyhow::Result;
use std::time::{Duration, Instant};

/// Configuration for video sources
#[derive(Debug, Clone)]
pub struct VideoSourceConfig {
    /// Target frame width (will downsample if source is larger)
    pub width: u32,
    /// Target frame height
    pub height: u32,
    /// Target framerate (fps)
    pub fps: u32,
    /// Camera index (for webcam)
    pub camera_index: u32,
}

impl Default for VideoSourceConfig {
    fn default() -> Self {
        Self {
            width: 64, // Small for LTC processing
            height: 64,
            fps: 30,
            camera_index: 0,
        }
    }
}

/// Trait for video input sources
pub trait VideoSource: Send {
    /// Start capturing video
    fn start(&mut self) -> Result<()>;

    /// Stop capturing
    fn stop(&mut self) -> Result<()>;

    /// Get the next frame (blocks until available or timeout)
    fn next_frame(&mut self) -> Result<Option<Frame>>;

    /// Check if source is currently active
    fn is_active(&self) -> bool;

    /// Get current configuration
    fn config(&self) -> &VideoSourceConfig;
}

// ============================================================================
// WEBCAM SOURCE (using nokhwa when available)
// ============================================================================

/// Webcam video source
///
/// When the `webcam` feature is enabled, provides the WebcamSource struct.
/// Real camera access requires the `nokhwa` crate at runtime; this stub
/// compiles without it and returns an error on `start()` if no camera
/// backend is linked.
#[cfg(feature = "webcam")]
pub struct WebcamSource {
    config: VideoSourceConfig,
    sequence: u64,
    active: bool,
}

#[cfg(feature = "webcam")]
impl WebcamSource {
    pub fn new(config: VideoSourceConfig) -> Result<Self> {
        Ok(Self {
            config,
            sequence: 0,
            active: false,
        })
    }
}

#[cfg(feature = "webcam")]
impl VideoSource for WebcamSource {
    fn start(&mut self) -> Result<()> {
        anyhow::bail!(
            "Webcam feature enabled but no camera backend linked. \
             Add the `nokhwa` crate to Cargo.toml for real camera access."
        )
    }

    fn stop(&mut self) -> Result<()> {
        self.active = false;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        Ok(None)
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &VideoSourceConfig {
        &self.config
    }
}

// Stub for when webcam feature is not enabled
#[cfg(not(feature = "webcam"))]
pub struct WebcamSource {
    config: VideoSourceConfig,
    sequence: u64,
    active: bool,
    last_frame_time: Option<Instant>,
}

#[cfg(not(feature = "webcam"))]
impl WebcamSource {
    pub fn new(config: VideoSourceConfig) -> Result<Self> {
        tracing::warn!("Webcam feature not enabled - using mock video source");
        Ok(Self {
            config,
            sequence: 0,
            active: false,
            last_frame_time: None,
        })
    }
}

#[cfg(not(feature = "webcam"))]
impl VideoSource for WebcamSource {
    fn start(&mut self) -> Result<()> {
        self.active = true;
        self.last_frame_time = Some(Instant::now());
        tracing::info!(
            "Mock webcam started ({}x{} @ {} fps)",
            self.config.width,
            self.config.height,
            self.config.fps
        );
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.active = false;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if !self.active {
            return Ok(None);
        }

        // Simulate frame rate
        let frame_duration = Duration::from_secs_f32(1.0 / self.config.fps as f32);
        if let Some(last) = self.last_frame_time {
            let elapsed = last.elapsed();
            if elapsed < frame_duration {
                std::thread::sleep(frame_duration - elapsed);
            }
        }
        self.last_frame_time = Some(Instant::now());

        // Generate synthetic frame data with some temporal variation
        let width = self.config.width;
        let height = self.config.height;
        let size = (width as usize) * (height as usize) * 3;

        // Create a pattern that changes over time (for testing temporal detection)
        let phase = (self.sequence as f32 * 0.1).sin();
        let brightness = ((phase + 1.0) / 2.0 * 200.0) as u8 + 55;

        let mut data = Vec::with_capacity(size);
        for y in 0..height {
            for x in 0..width {
                // Create a moving pattern
                let cx = (x as f32 / width as f32) - 0.5;
                let cy = (y as f32 / height as f32) - 0.5;
                let dist = (cx * cx + cy * cy).sqrt();
                let wave = ((dist * 10.0 + self.sequence as f32 * 0.2).sin() + 1.0) / 2.0;

                let r = (brightness as f32 * wave) as u8;
                let g = (brightness as f32 * (1.0 - wave * 0.5)) as u8;
                let b = (brightness as f32 * 0.5) as u8;

                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        self.sequence += 1;
        Ok(Some(Frame::new(
            data,
            width,
            height,
            FrameFormat::Rgb24,
            self.sequence,
        )))
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &VideoSourceConfig {
        &self.config
    }
}

// ============================================================================
// MOCK VIDEO SOURCE (for testing)
// ============================================================================

/// Mock video source that generates synthetic frames
///
/// Useful for testing temporal perception without a real camera.
pub struct MockVideoSource {
    config: VideoSourceConfig,
    sequence: u64,
    active: bool,
    last_frame_time: Option<Instant>,
    /// Pattern type for generated frames
    pub pattern: MockPattern,
    /// Custom brightness function (sequence -> brightness 0-255)
    brightness_fn: Option<Box<dyn Fn(u64) -> u8 + Send>>,
}

/// Pattern types for mock video source
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MockPattern {
    /// Solid color that pulses smoothly
    Pulse,
    /// Moving gradient
    MovingGradient,
    /// Blinking at specified BPM (sharp on/off)
    Blink { bpm: u32 },
    /// Random noise
    Noise,
    /// Static solid color
    Solid,

    // === Complex Patterns for Testing ===
    /// Polyrhythm: two overlapping rhythms (e.g., 3:2)
    /// Creates a complex pattern where two BPMs interfere
    Polyrhythm { bpm_a: u32, bpm_b: u32 },

    /// Heartbeat pattern: LUB-dub ... pause ... LUB-dub
    /// Mimics biological cardiac rhythm (irregular spacing)
    Heartbeat { bpm: u32 },

    /// Accelerando: gradually speeding up
    /// BPM increases from start_bpm over cycle_frames
    Accelerando {
        start_bpm: u32,
        end_bpm: u32,
        cycle_frames: u32,
    },

    /// Triplet feel: 3 beats in the time of 2
    /// Creates swing/shuffle feeling
    Triplet { bpm: u32 },

    /// Burst pattern: random short bursts of activity
    /// Irregular timing - tests robustness
    Burst { avg_gap_frames: u32 },

    /// Morse code style: short-long patterns
    /// Tests pattern within pattern recognition
    Morse { unit_frames: u32 },
}

impl MockVideoSource {
    pub fn new(config: VideoSourceConfig) -> Self {
        Self {
            config,
            sequence: 0,
            active: false,
            last_frame_time: None,
            pattern: MockPattern::Pulse,
            brightness_fn: None,
        }
    }

    /// Create a mock source that blinks at a specific BPM (for rhythm testing)
    pub fn with_bpm(config: VideoSourceConfig, bpm: u32) -> Self {
        let mut source = Self::new(config);
        source.pattern = MockPattern::Blink { bpm };
        source
    }

    /// Set a custom brightness function
    pub fn with_brightness_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(u64) -> u8 + Send + 'static,
    {
        self.brightness_fn = Some(Box::new(f));
        self
    }

    fn generate_frame(&mut self) -> Frame {
        let width = self.config.width;
        let height = self.config.height;
        let size = (width as usize) * (height as usize) * 3;
        let seq = self.sequence;

        let fps = self.config.fps as f32;
        let brightness = if let Some(ref f) = self.brightness_fn {
            f(seq)
        } else {
            match self.pattern {
                MockPattern::Pulse => {
                    let phase = (seq as f32 * 0.1).sin();
                    ((phase + 1.0) / 2.0 * 200.0) as u8 + 55
                }
                MockPattern::Blink { bpm } => {
                    // Convert BPM to frame period
                    let frames_per_beat = (fps * 60.0) / bpm as f32;
                    let beat_position = (seq as f32 % frames_per_beat) / frames_per_beat;
                    // Sharp on/off at 50% duty cycle
                    if beat_position < 0.5 { 255 } else { 50 }
                }
                MockPattern::MovingGradient => (seq % 256) as u8,
                MockPattern::Noise => {
                    // Simple PRNG
                    let hash = seq
                        .wrapping_mul(0x5851F42D4C957F2D)
                        .wrapping_add(0x14057B7EF767814F);
                    (hash % 256) as u8
                }
                MockPattern::Solid => 128,

                // === Complex Patterns ===
                MockPattern::Polyrhythm { bpm_a, bpm_b } => {
                    // Two overlapping rhythms - brightness is sum of both
                    let frames_per_beat_a = (fps * 60.0) / bpm_a as f32;
                    let frames_per_beat_b = (fps * 60.0) / bpm_b as f32;

                    let phase_a = (seq as f32 % frames_per_beat_a) / frames_per_beat_a;
                    let phase_b = (seq as f32 % frames_per_beat_b) / frames_per_beat_b;

                    // Each rhythm contributes a pulse
                    let pulse_a = if phase_a < 0.3 { 127 } else { 0 };
                    let pulse_b = if phase_b < 0.3 { 127 } else { 0 };

                    (50 + pulse_a + pulse_b) as u8
                }

                MockPattern::Heartbeat { bpm } => {
                    // LUB-dub pattern: two quick beats, then pause
                    // Real heart: ~60% systole (LUB-dub), 40% diastole (pause)
                    let frames_per_beat = (fps * 60.0) / bpm as f32;
                    let beat_position = (seq as f32 % frames_per_beat) / frames_per_beat;

                    // LUB at 0%, dub at 20%, rest until 100%
                    if beat_position < 0.08 {
                        255 // LUB (strong)
                    } else if beat_position < 0.15 {
                        80 // brief pause
                    } else if beat_position < 0.22 {
                        200 // dub (weaker)
                    } else {
                        50 // diastole rest
                    }
                }

                MockPattern::Accelerando {
                    start_bpm,
                    end_bpm,
                    cycle_frames,
                } => {
                    // BPM increases linearly over cycle_frames, then resets
                    let cycle_pos = (seq % cycle_frames as u64) as f32 / cycle_frames as f32;
                    let current_bpm = start_bpm as f32 + (end_bpm - start_bpm) as f32 * cycle_pos;

                    let frames_per_beat = (fps * 60.0) / current_bpm;
                    let beat_position = (seq as f32 % frames_per_beat) / frames_per_beat;

                    if beat_position < 0.5 { 255 } else { 50 }
                }

                MockPattern::Triplet { bpm } => {
                    // Triplet feel: 3 beats in time of 2
                    // Creates a "swing" or "shuffle" pattern
                    let frames_per_beat = (fps * 60.0) / bpm as f32;
                    let beat_position = (seq as f32 % frames_per_beat) / frames_per_beat;

                    // Triplet subdivisions: 0%, 33%, 67% of beat
                    if beat_position < 0.11
                        || (beat_position > 0.33 && beat_position < 0.44)
                        || (beat_position > 0.67 && beat_position < 0.78)
                    {
                        255
                    } else {
                        50
                    }
                }

                MockPattern::Burst { avg_gap_frames } => {
                    // Pseudo-random bursts using deterministic hash
                    let hash = seq
                        .wrapping_mul(0x5851F42D4C957F2D)
                        .wrapping_add(0x14057B7EF767814F);
                    let gap = hash % (avg_gap_frames as u64 * 2);

                    // Burst when sequence aligns with hash-based timing
                    if seq % (gap.max(1)) < 3 { 255 } else { 50 }
                }

                MockPattern::Morse { unit_frames } => {
                    // SOS pattern: ... --- ... (short short short, long long long, short short short)
                    // Pattern repeats every 27 units (3+1+3+1+3+3+7+1+7+1+7+3+3+1+3+1+3 = ~27 with gaps)
                    let unit = unit_frames as u64;
                    let pattern_length = 30 * unit; // Approximate full SOS cycle
                    let pos = seq % pattern_length;

                    // Simplified: dots at 0-1, 2-3, 4-5 units; dashes at 8-11, 13-16, 18-21; dots at 24-25, 26-27, 28-29
                    let in_dot = |p: u64| -> bool {
                        (p < unit)
                            || (p >= 2 * unit && p < 3 * unit)
                            || (p >= 4 * unit && p < 5 * unit)
                            || (p >= 24 * unit && p < 25 * unit)
                            || (p >= 26 * unit && p < 27 * unit)
                            || (p >= 28 * unit && p < 29 * unit)
                    };
                    let in_dash = |p: u64| -> bool {
                        (p >= 8 * unit && p < 11 * unit)
                            || (p >= 13 * unit && p < 16 * unit)
                            || (p >= 18 * unit && p < 21 * unit)
                    };

                    if in_dot(pos) || in_dash(pos) { 255 } else { 50 }
                }
            }
        };

        let mut data = Vec::with_capacity(size);
        for _ in 0..(width * height) {
            data.push(brightness);
            data.push(brightness);
            data.push(brightness);
        }

        Frame::new(data, width, height, FrameFormat::Rgb24, seq)
    }
}

impl VideoSource for MockVideoSource {
    fn start(&mut self) -> Result<()> {
        self.active = true;
        self.last_frame_time = Some(Instant::now());
        self.sequence = 0;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.active = false;
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        if !self.active {
            return Ok(None);
        }

        // Simulate frame rate
        let frame_duration = Duration::from_secs_f32(1.0 / self.config.fps as f32);
        if let Some(last) = self.last_frame_time {
            let elapsed = last.elapsed();
            if elapsed < frame_duration {
                std::thread::sleep(frame_duration - elapsed);
            }
        }
        self.last_frame_time = Some(Instant::now());

        let frame = self.generate_frame();
        self.sequence += 1;

        Ok(Some(frame))
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn config(&self) -> &VideoSourceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_source_pulse() {
        let config = VideoSourceConfig {
            width: 32,
            height: 32,
            fps: 60,
            camera_index: 0,
        };
        let mut source = MockVideoSource::new(config);
        source.start().unwrap();

        // Get a few frames
        for _ in 0..10 {
            let frame = source.next_frame().unwrap();
            assert!(frame.is_some());
            let f = frame.unwrap();
            assert_eq!(f.width, 32);
            assert_eq!(f.height, 32);
        }

        source.stop().unwrap();
        assert!(!source.is_active());
    }

    #[test]
    fn test_mock_source_bpm() {
        let config = VideoSourceConfig::default();
        let mut source = MockVideoSource::with_bpm(config, 60);
        source.start().unwrap();

        // At 60 BPM with 30 fps, we should see alternating bright/dark every 15 frames
        let frame1 = source.next_frame().unwrap().unwrap();
        let brightness1 = frame1.average_brightness();

        // Skip ahead
        for _ in 0..16 {
            source.next_frame().unwrap();
        }

        let frame2 = source.next_frame().unwrap().unwrap();
        let brightness2 = frame2.average_brightness();

        // They should be different (one bright, one dark)
        assert!((brightness1 - brightness2).abs() > 0.3);
    }
}
