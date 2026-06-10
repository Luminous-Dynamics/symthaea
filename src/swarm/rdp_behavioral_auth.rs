// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Behavioral Biometric Authentication for Sovereign RDP.
//!
//! Encodes keystroke and mouse dynamics into HDC vectors for continuous
//! authentication during remote sessions. After a 5-minute enrollment phase,
//! the system continuously compares the user's behavioral pattern against
//! the enrolled profile via cosine similarity in 16,384D HDC space.
//!
//! ## HDC Advantage
//!
//! Profile comparison is a single `ContinuousHV::similarity()` call — nanoseconds,
//! not ML inference. This enables per-second continuous auth with zero latency impact.
//!
//! ## Science
//!
//! - Monrose & Rubin (2000): Keystroke dynamics as biometric
//! - Ahmed & Traore (2007): Mouse dynamics authentication
//! - Poduval et al. (2021): HDC for temporal biological signal classification

use std::collections::VecDeque;
use std::time::Instant;

use super::rdp_protocol::InputEvent;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// HDC dimension for behavioral profiles.
const BEHAVIOR_HDC_DIM: usize = 16_384;

/// Number of histogram bins for keystroke inter-key intervals.
const KEYSTROKE_BINS: usize = 20;

/// Number of histogram bins for mouse velocity.
const MOUSE_VELOCITY_BINS: usize = 10;

/// Maximum inter-key interval tracked (ms).
const MAX_INTERKEY_MS: u64 = 2000;

/// Maximum mouse velocity tracked (normalized units/s).
const MAX_MOUSE_VELOCITY: f32 = 5.0;

/// Enrollment duration (seconds).
const ENROLLMENT_DURATION_SECS: u64 = 300; // 5 minutes

/// Minimum events needed for enrollment.
const MIN_ENROLLMENT_EVENTS: usize = 100;

/// Anomaly threshold: cosine similarity below this triggers alert.
const ANOMALY_THRESHOLD: f32 = 0.5;

/// Match threshold: cosine similarity above this confirms identity.
const MATCH_THRESHOLD: f32 = 0.7;

/// Rolling window size for continuous checking.
const CHECK_WINDOW_SIZE: usize = 200;

/// Behavioral authentication state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthState {
    /// Collecting baseline behavioral data.
    Enrolling,
    /// Enrolled and continuously authenticating.
    Active,
    /// Anomaly detected — input should be paused.
    Anomaly,
}

/// Behavioral biometric authenticator.
pub struct BehavioralAuthenticator {
    /// Current state.
    pub state: AuthState,
    /// Enrolled behavioral profile (HDC vector).
    enrolled_profile: Option<ContinuousHV>,
    /// Rolling window of recent events for continuous checking.
    recent_events: VecDeque<TimedInput>,
    /// Enrollment start time.
    enrollment_start: Instant,
    /// Last check result (cosine similarity).
    pub last_similarity: f32,
    /// Consecutive anomaly checks.
    anomaly_streak: u32,
    /// Basis vectors for histogram bin encoding.
    keystroke_basis: Vec<ContinuousHV>,
    mouse_basis: Vec<ContinuousHV>,
}

/// An input event with timestamp for temporal feature extraction.
#[derive(Clone)]
struct TimedInput {
    event: InputEvent,
    timestamp_ms: u64,
}

impl BehavioralAuthenticator {
    /// Create a new authenticator in Enrolling state.
    pub fn new() -> Self {
        // Generate deterministic basis vectors for histogram encoding.
        let keystroke_basis = (0..KEYSTROKE_BINS)
            .map(|i| generate_basis(i as u64, 0x4B4559))
            .collect();
        let mouse_basis = (0..MOUSE_VELOCITY_BINS)
            .map(|i| generate_basis(i as u64, 0x4D4F55))
            .collect();

        Self {
            state: AuthState::Enrolling,
            enrolled_profile: None,
            recent_events: VecDeque::with_capacity(CHECK_WINDOW_SIZE + 64),
            enrollment_start: Instant::now(),
            last_similarity: 0.0,
            anomaly_streak: 0,
            keystroke_basis,
            mouse_basis,
        }
    }

    /// Process new input events. Updates enrollment or checks authentication.
    pub fn process(&mut self, events: &[InputEvent], timestamp_ms: u64) {
        // Add to rolling window.
        for event in events {
            self.recent_events.push_back(TimedInput {
                event: event.clone(),
                timestamp_ms,
            });
        }

        // Trim rolling window.
        while self.recent_events.len() > CHECK_WINDOW_SIZE + 64 {
            self.recent_events.pop_front();
        }

        match self.state {
            AuthState::Enrolling => {
                let elapsed = self.enrollment_start.elapsed().as_secs();
                if elapsed >= ENROLLMENT_DURATION_SECS
                    && self.recent_events.len() >= MIN_ENROLLMENT_EVENTS
                {
                    // Enrollment complete — build profile from collected events.
                    let profile =
                        self.build_profile(&self.recent_events.iter().cloned().collect::<Vec<_>>());
                    self.enrolled_profile = Some(profile);
                    self.state = AuthState::Active;
                }
            }
            AuthState::Active | AuthState::Anomaly => {
                if self.recent_events.len() >= CHECK_WINDOW_SIZE / 2 {
                    self.check_continuous();
                }
            }
        }
    }

    /// Build an HDC behavioral profile from a set of timed input events.
    fn build_profile(&self, events: &[TimedInput]) -> ContinuousHV {
        let keystroke_hv = self.encode_keystroke_dynamics(events);
        let mouse_hv = self.encode_mouse_dynamics(events);

        // Bundle keystroke and mouse profiles.
        let mut combined = vec![0.0f32; BEHAVIOR_HDC_DIM];
        for i in 0..BEHAVIOR_HDC_DIM {
            combined[i] = keystroke_hv.values[i] + mouse_hv.values[i];
        }

        // Normalize.
        let norm: f32 = combined
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        for v in &mut combined {
            *v /= norm;
        }

        ContinuousHV::from_values(combined)
    }

    /// Encode keystroke dynamics as HDC vector.
    ///
    /// Features: inter-key interval histogram (how fast/slow between keypresses).
    fn encode_keystroke_dynamics(&self, events: &[TimedInput]) -> ContinuousHV {
        let mut histogram = vec![0.0f32; KEYSTROKE_BINS];

        // Extract key events and compute inter-key intervals.
        let key_times: Vec<u64> = events
            .iter()
            .filter(|e| matches!(e.event, InputEvent::Key { pressed: true, .. }))
            .map(|e| e.timestamp_ms)
            .collect();

        for window in key_times.windows(2) {
            let interval = window[1].saturating_sub(window[0]);
            let bin = ((interval as f32 / MAX_INTERKEY_MS as f32) * KEYSTROKE_BINS as f32)
                .floor()
                .min((KEYSTROKE_BINS - 1) as f32) as usize;
            histogram[bin] += 1.0;
        }

        // Normalize histogram.
        let total: f32 = histogram.iter().sum::<f32>().max(1.0);
        for h in &mut histogram {
            *h /= total;
        }

        // Encode: weighted sum of basis vectors.
        let mut values = vec![0.0f32; BEHAVIOR_HDC_DIM];
        for (bin, weight) in histogram.iter().enumerate() {
            if *weight > 0.0 && bin < self.keystroke_basis.len() {
                for i in 0..BEHAVIOR_HDC_DIM {
                    values[i] += self.keystroke_basis[bin].values[i] * weight;
                }
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Encode mouse dynamics as HDC vector.
    ///
    /// Features: velocity distribution histogram.
    fn encode_mouse_dynamics(&self, events: &[TimedInput]) -> ContinuousHV {
        let mut histogram = vec![0.0f32; MOUSE_VELOCITY_BINS];

        // Extract pointer events and compute velocities.
        let pointer_events: Vec<(f32, f32, u64)> = events
            .iter()
            .filter_map(|e| match &e.event {
                InputEvent::Pointer { x, y, .. } => Some((*x, *y, e.timestamp_ms)),
                _ => None,
            })
            .collect();

        for window in pointer_events.windows(2) {
            let (x1, y1, t1) = window[0];
            let (x2, y2, t2) = window[1];
            let dt = (t2.saturating_sub(t1)) as f32 / 1000.0; // seconds
            if dt > 0.001 {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let velocity = (dx * dx + dy * dy).sqrt() / dt;
                let bin = ((velocity / MAX_MOUSE_VELOCITY) * MOUSE_VELOCITY_BINS as f32)
                    .floor()
                    .min((MOUSE_VELOCITY_BINS - 1) as f32) as usize;
                histogram[bin] += 1.0;
            }
        }

        // Normalize.
        let total: f32 = histogram.iter().sum::<f32>().max(1.0);
        for h in &mut histogram {
            *h /= total;
        }

        // Encode.
        let mut values = vec![0.0f32; BEHAVIOR_HDC_DIM];
        for (bin, weight) in histogram.iter().enumerate() {
            if *weight > 0.0 && bin < self.mouse_basis.len() {
                for i in 0..BEHAVIOR_HDC_DIM {
                    values[i] += self.mouse_basis[bin].values[i] * weight;
                }
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Continuous authentication check against enrolled profile.
    fn check_continuous(&mut self) {
        let Some(ref enrolled) = self.enrolled_profile else {
            return;
        };

        let recent: Vec<TimedInput> = self
            .recent_events
            .iter()
            .rev()
            .take(CHECK_WINDOW_SIZE)
            .cloned()
            .collect();

        let current_profile = self.build_profile(&recent);
        let similarity = enrolled.similarity(&current_profile);
        let similarity = if similarity.is_finite() {
            similarity
        } else {
            0.0
        };

        self.last_similarity = similarity;

        if similarity < ANOMALY_THRESHOLD {
            self.anomaly_streak += 1;
            if self.anomaly_streak >= 3 {
                self.state = AuthState::Anomaly;
            }
        } else {
            self.anomaly_streak = 0;
            if self.state == AuthState::Anomaly && similarity > MATCH_THRESHOLD {
                self.state = AuthState::Active;
            }
        }
    }

    /// Whether input forwarding should be paused.
    pub fn should_block_input(&self) -> bool {
        self.state == AuthState::Anomaly
    }

    /// Force re-enrollment (e.g., after re-attestation).
    pub fn reset_enrollment(&mut self) {
        self.state = AuthState::Enrolling;
        self.enrolled_profile = None;
        self.recent_events.clear();
        self.enrollment_start = Instant::now();
        self.anomaly_streak = 0;
    }
}

impl Default for BehavioralAuthenticator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a deterministic basis vector from an index and domain seed.
fn generate_basis(index: u64, domain: u64) -> ContinuousHV {
    let combined = index.wrapping_mul(0x517cc1b727220a95).wrapping_add(domain);
    let mut values = vec![0.0f32; BEHAVIOR_HDC_DIM];
    let mut state = combined;
    for v in values.iter_mut() {
        let hash = blake3::hash(&state.to_le_bytes());
        let bytes = hash.as_bytes();
        let u =
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32 / u32::MAX as f32;
        *v = (u - 0.5) * 2.0;
        state = state.wrapping_add(1);
    }
    ContinuousHV::from_values(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_events(count: usize, interval_ms: u64) -> Vec<(InputEvent, u64)> {
        (0..count)
            .map(|i| {
                (
                    InputEvent::Key {
                        code: 65 + (i % 26) as u32,
                        pressed: true,
                        modifiers: 0,
                    },
                    i as u64 * interval_ms,
                )
            })
            .collect()
    }

    #[test]
    fn test_enrollment_requires_min_events() {
        let mut auth = BehavioralAuthenticator::new();
        // Not enough events.
        for i in 0..10 {
            auth.process(
                &[InputEvent::Key {
                    code: 65,
                    pressed: true,
                    modifiers: 0,
                }],
                i * 100,
            );
        }
        assert_eq!(auth.state, AuthState::Enrolling);
    }

    #[test]
    fn test_same_pattern_high_similarity() {
        let auth = BehavioralAuthenticator::new();

        // Build two profiles from the same typing pattern.
        let events: Vec<TimedInput> = make_key_events(200, 100)
            .into_iter()
            .map(|(event, ts)| TimedInput {
                event,
                timestamp_ms: ts,
            })
            .collect();

        let profile1 = auth.build_profile(&events);
        let profile2 = auth.build_profile(&events);

        let sim = profile1.similarity(&profile2);
        assert!(
            sim > 0.99,
            "Same pattern should have near-perfect similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_different_patterns_lower_similarity() {
        let auth = BehavioralAuthenticator::new();

        // Fast typer: 50ms intervals.
        let fast: Vec<TimedInput> = make_key_events(200, 50)
            .into_iter()
            .map(|(event, ts)| TimedInput {
                event,
                timestamp_ms: ts,
            })
            .collect();

        // Slow typer: 500ms intervals.
        let slow: Vec<TimedInput> = make_key_events(200, 500)
            .into_iter()
            .map(|(event, ts)| TimedInput {
                event,
                timestamp_ms: ts,
            })
            .collect();

        let fast_profile = auth.build_profile(&fast);
        let slow_profile = auth.build_profile(&slow);

        let sim = fast_profile.similarity(&slow_profile);
        assert!(
            sim < 0.9,
            "Different typing speeds should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_anomaly_blocks_input() {
        let mut auth = BehavioralAuthenticator::new();
        auth.state = AuthState::Anomaly;
        assert!(auth.should_block_input());

        auth.state = AuthState::Active;
        assert!(!auth.should_block_input());
    }
}
