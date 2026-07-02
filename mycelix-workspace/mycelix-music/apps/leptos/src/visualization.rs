// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Audio visualization using Bevy ECS.
//!
//! Defines the component/resource/system architecture for rendering real-time
//! audio visualizations: waveform, spectrum, and consciousness state trajectory.
//!
//! NOTE: Bevy is not yet wired as a dependency. This module defines the types
//! and system signatures so the architecture is concrete and ready to activate
//! once `bevy = "0.15"` is added to Cargo.toml with WASM features.

// --------------------------------------------------------------------------
// When bevy is added as a dependency, uncomment the `use bevy::prelude::*;`
// and remove the stub type aliases below.
// --------------------------------------------------------------------------

// use bevy::prelude::*;

// --- Stub types until Bevy is a real dependency ---
// These mirror Bevy's trait/derive API so the module compiles standalone.

/// Marker for Bevy Resource derive (stub).
pub trait Resource {}

/// Marker for Bevy Component derive (stub).
pub trait Component {}

/// Placeholder for Bevy's Color type.
#[derive(Clone, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

// --- Constants ---

/// Default waveform color (matches --accent: #8b5cf6).
pub const WAVEFORM_COLOR: Color = Color::rgba(0.545, 0.361, 0.965, 1.0);

/// Default spectrum bar color (green accent).
pub const SPECTRUM_COLOR: Color = Color::rgba(0.133, 0.773, 0.369, 1.0);

/// Consciousness orbit trail color (fading purple).
pub const ORBIT_COLOR: Color = Color::rgba(0.655, 0.545, 0.980, 0.8);

/// Number of FFT bins for spectrum display.
pub const SPECTRUM_BINS: usize = 64;

/// Maximum orbit trail history points.
pub const ORBIT_MAX_POINTS: usize = 256;

/// Canvas width in logical pixels.
pub const CANVAS_WIDTH: f32 = 800.0;

/// Canvas height in logical pixels.
pub const CANVAS_HEIGHT: f32 = 400.0;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Audio buffer shared between the synthesis pipeline and visualization.
///
/// Updated every 32ms by the consciousness audio bridge callback.
/// Bevy systems read from this each frame to render visualizations.
#[derive(Debug)]
pub struct AudioBuffer {
    /// Latest PCM chunk (stereo interleaved: [L0, R0, L1, R1, ...]).
    pub samples: Vec<f32>,
    /// Audio sample rate (typically 44100).
    pub sample_rate: u32,
    /// FFT magnitude bins (length = SPECTRUM_BINS).
    pub spectrum: Vec<f32>,
    /// Current consciousness level (Phi), range 0.0-1.0.
    pub consciousness_level: f32,
    /// Current arousal, range 0.0-1.0.
    pub arousal: f32,
    /// Current valence, range -1.0 to 1.0.
    pub valence: f32,
    /// Whether synthesis is actively running.
    pub is_active: bool,
}

impl Default for AudioBuffer {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            sample_rate: 44100,
            spectrum: vec![0.0; SPECTRUM_BINS],
            consciousness_level: 0.0,
            arousal: 0.5,
            valence: 0.0,
            is_active: false,
        }
    }
}

impl Resource for AudioBuffer {}

/// Visualization configuration resource.
#[derive(Debug)]
pub struct VisualizationConfig {
    /// Show waveform oscilloscope.
    pub show_waveform: bool,
    /// Show spectrum bars.
    pub show_spectrum: bool,
    /// Show consciousness orbit trail.
    pub show_orbit: bool,
    /// Show phi meter.
    pub show_phi_meter: bool,
    /// Background color.
    pub background: Color,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_waveform: true,
            show_spectrum: true,
            show_orbit: true,
            show_phi_meter: true,
            background: Color::rgba(0.039, 0.039, 0.059, 1.0), // matches --bg: #0a0a0f
        }
    }
}

impl Resource for VisualizationConfig {}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Waveform oscilloscope display entity.
#[derive(Debug)]
pub struct WaveformDisplay {
    /// Display width in pixels.
    pub width: f32,
    /// Display height in pixels.
    pub height: f32,
    /// Trace color.
    pub color: Color,
    /// Downsampling factor (1 = every sample, 2 = every other, etc.).
    pub downsample: usize,
}

impl Default for WaveformDisplay {
    fn default() -> Self {
        Self {
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT * 0.3,
            color: WAVEFORM_COLOR,
            downsample: 4,
        }
    }
}

impl Component for WaveformDisplay {}

/// Spectrum analyzer bar display entity.
#[derive(Debug)]
pub struct SpectrumDisplay {
    /// Number of frequency bars.
    pub num_bars: usize,
    /// Width of each bar in pixels.
    pub bar_width: f32,
    /// Maximum bar height in pixels.
    pub max_height: f32,
    /// Smoothing factor for bar movement (0.0 = instant, 1.0 = frozen).
    pub smoothing: f32,
    /// Previous bar heights for smoothing.
    pub prev_heights: Vec<f32>,
}

impl Default for SpectrumDisplay {
    fn default() -> Self {
        let num_bars = SPECTRUM_BINS;
        Self {
            num_bars,
            bar_width: (CANVAS_WIDTH * 0.4) / num_bars as f32,
            max_height: CANVAS_HEIGHT * 0.3,
            smoothing: 0.3,
            prev_heights: vec![0.0; num_bars],
        }
    }
}

impl Component for SpectrumDisplay {}

/// Consciousness state orbit trail entity.
///
/// Plots the (valence, arousal) trajectory over time as a fading trail.
#[derive(Debug)]
pub struct ConsciousnessOrbit {
    /// History of (valence, arousal) points.
    pub history: Vec<(f32, f32)>,
    /// Maximum number of trail points before oldest are dropped.
    pub max_points: usize,
    /// Display radius in pixels.
    pub radius: f32,
}

impl Default for ConsciousnessOrbit {
    fn default() -> Self {
        Self {
            history: Vec::with_capacity(ORBIT_MAX_POINTS),
            max_points: ORBIT_MAX_POINTS,
            radius: CANVAS_HEIGHT * 0.18,
        }
    }
}

impl Component for ConsciousnessOrbit {}

/// Phi consciousness level meter entity.
#[derive(Debug)]
pub struct PhiMeter {
    /// Display width in pixels.
    pub width: f32,
    /// Display height in pixels.
    pub height: f32,
    /// Current smoothed phi value (for animation).
    pub display_phi: f32,
    /// Smoothing factor.
    pub smoothing: f32,
}

impl Default for PhiMeter {
    fn default() -> Self {
        Self {
            width: 30.0,
            height: CANVAS_HEIGHT * 0.8,
            display_phi: 0.0,
            smoothing: 0.15,
        }
    }
}

impl Component for PhiMeter {}

// ---------------------------------------------------------------------------
// Plugin definition (activates when Bevy is wired)
// ---------------------------------------------------------------------------

/// Bevy plugin for audio visualization.
///
/// When Bevy is added as a dependency, this implements `bevy::app::Plugin`:
/// ```ignore
/// impl Plugin for AudioVisualizationPlugin {
///     fn build(&self, app: &mut App) {
///         app.init_resource::<AudioBuffer>()
///            .init_resource::<VisualizationConfig>()
///            .add_systems(Update, (
///                update_waveform,
///                update_spectrum,
///                update_consciousness_orbit,
///                update_phi_meter,
///            ));
///     }
/// }
/// ```
pub struct AudioVisualizationPlugin;

// ---------------------------------------------------------------------------
// System signatures (implementations will use Bevy queries + gizmos)
// ---------------------------------------------------------------------------

/// Update waveform display from latest PCM samples.
///
/// Reads `AudioBuffer.samples`, downsamples by `WaveformDisplay.downsample`,
/// and draws a polyline trace. Left channel only (even indices) for clarity.
pub fn update_waveform(buffer: &AudioBuffer, display: &mut WaveformDisplay) {
    if !buffer.is_active || buffer.samples.is_empty() {
        return;
    }

    let _step = display.downsample.max(1);
    let _points_count = buffer.samples.len() / 2 / _step; // mono from stereo

    // In Bevy: iterate samples, compute screen-space (x, y) for each point,
    // draw line segments using Gizmos or a custom mesh.
    // x = i * (display.width / points_count)
    // y = display.height/2 + sample * display.height/2
}

/// Update spectrum bars from FFT magnitude data.
///
/// Reads `AudioBuffer.spectrum`, applies smoothing against previous frame,
/// and updates bar entity transforms/scales.
pub fn update_spectrum(buffer: &AudioBuffer, display: &mut SpectrumDisplay) {
    if !buffer.is_active {
        return;
    }

    let alpha = display.smoothing;
    for (i, mag) in buffer.spectrum.iter().enumerate().take(display.num_bars) {
        let target = mag * display.max_height;
        if i < display.prev_heights.len() {
            display.prev_heights[i] = display.prev_heights[i] * alpha + target * (1.0 - alpha);
        }
    }

    // In Bevy: set Transform.scale.y of each bar sprite to prev_heights[i].
}

/// Update consciousness orbit trail from current V-A state.
///
/// Appends (valence, arousal) to history ring buffer, draws trail with
/// fading alpha from oldest to newest.
pub fn update_consciousness_orbit(buffer: &AudioBuffer, orbit: &mut ConsciousnessOrbit) {
    if !buffer.is_active {
        return;
    }

    // Append current point
    orbit.history.push((buffer.valence, buffer.arousal));
    if orbit.history.len() > orbit.max_points {
        orbit.history.remove(0);
    }

    // In Bevy: draw circles/lines for each point in history.
    // Alpha = i / history.len() (older points more transparent).
    // Map valence [-1,1] to x, arousal [0,1] to y within orbit.radius.
}

/// Update phi meter bar from current consciousness level.
///
/// Smoothly animates the meter fill height toward `AudioBuffer.consciousness_level`.
/// Draws tier threshold lines at 0.2 (Observer), 0.4 (Participant),
/// 0.6 (Contributor), 0.8 (Guardian).
pub fn update_phi_meter(buffer: &AudioBuffer, meter: &mut PhiMeter) {
    let target = buffer.consciousness_level;
    meter.display_phi = meter.display_phi * meter.smoothing + target * (1.0 - meter.smoothing);

    // In Bevy: set bar sprite height to display_phi * meter.height.
    // Draw horizontal lines at tier thresholds.
    // Color gradient: red (0.0) -> yellow (0.5) -> green (1.0).
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple 512-point FFT magnitude computation for visualization.
///
/// Takes mono PCM samples (f32), returns SPECTRUM_BINS magnitude values.
/// Uses a basic DFT (not optimized -- for visualization only, not analysis).
pub fn compute_spectrum_magnitudes(mono_samples: &[f32]) -> Vec<f32> {
    let n = 512.min(mono_samples.len());
    let mut magnitudes = Vec::with_capacity(SPECTRUM_BINS);

    for bin in 0..SPECTRUM_BINS {
        let freq = bin as f32 * (0.5 / SPECTRUM_BINS as f32);
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;

        for (k, &sample) in mono_samples.iter().take(n).enumerate() {
            let angle = 2.0 * std::f32::consts::PI * freq * k as f32;
            real += sample * angle.cos();
            imag += sample * (-angle).sin();
        }

        let mag = (real * real + imag * imag).sqrt() / n as f32;
        magnitudes.push(mag);
    }

    magnitudes
}

/// Extract mono samples from stereo interleaved buffer (take left channel).
pub fn stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    stereo.chunks_exact(2).map(|pair| pair[0]).collect()
}

/// Map (valence, arousal) to a descriptive emotion label.
pub fn emotion_label(valence: f32, arousal: f32) -> &'static str {
    match (arousal > 0.6, arousal < 0.4, valence > 0.3, valence < -0.3) {
        (true, _, true, _) => "Excitement / Joy",
        (true, _, _, true) => "Panic / Anger",
        (_, true, true, _) => "Contentment / Peace",
        (_, true, _, true) => "Sadness / Grief",
        _ => "Neutral / Exploratory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_default() {
        let buf = AudioBuffer::default();
        assert_eq!(buf.sample_rate, 44100);
        assert_eq!(buf.spectrum.len(), SPECTRUM_BINS);
        assert!(!buf.is_active);
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![0.5, 0.3, -0.2, 0.1, 0.8, -0.4];
        let mono = stereo_to_mono(&stereo);
        assert_eq!(mono, vec![0.5, -0.2, 0.8]);
    }

    #[test]
    fn test_emotion_labels() {
        assert_eq!(emotion_label(0.5, 0.8), "Excitement / Joy");
        assert_eq!(emotion_label(-0.5, 0.8), "Panic / Anger");
        assert_eq!(emotion_label(0.5, 0.2), "Contentment / Peace");
        assert_eq!(emotion_label(-0.5, 0.2), "Sadness / Grief");
        assert_eq!(emotion_label(0.0, 0.5), "Neutral / Exploratory");
    }

    #[test]
    fn test_spectrum_magnitudes_length() {
        let samples = vec![0.0; 512];
        let mags = compute_spectrum_magnitudes(&samples);
        assert_eq!(mags.len(), SPECTRUM_BINS);
    }

    #[test]
    fn test_spectrum_silence_is_zero() {
        let samples = vec![0.0; 512];
        let mags = compute_spectrum_magnitudes(&samples);
        for m in &mags {
            assert!(m.abs() < 1e-6, "silence should produce zero magnitudes");
        }
    }

    #[test]
    fn test_orbit_ring_buffer() {
        let mut orbit = ConsciousnessOrbit {
            history: Vec::new(),
            max_points: 3,
            radius: 100.0,
        };
        let buf = AudioBuffer { is_active: true, valence: 0.1, arousal: 0.2, ..Default::default() };
        update_consciousness_orbit(&buf, &mut orbit);
        update_consciousness_orbit(&buf, &mut orbit);
        update_consciousness_orbit(&buf, &mut orbit);
        update_consciousness_orbit(&buf, &mut orbit);
        assert_eq!(orbit.history.len(), 3, "should cap at max_points");
    }

    #[test]
    fn test_spectrum_smoothing() {
        let mut display = SpectrumDisplay::default();
        let buf = AudioBuffer {
            is_active: true,
            spectrum: vec![1.0; SPECTRUM_BINS],
            ..Default::default()
        };
        // First update from zero
        update_spectrum(&buf, &mut display);
        assert!(display.prev_heights[0] > 0.0, "should move toward target");
        assert!(display.prev_heights[0] < display.max_height, "should not reach target instantly");
    }
}
