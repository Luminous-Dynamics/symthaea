// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Touch-as-proprioception: maps Android touch events to consciousness-relevant signals.
//!
//! Touch is Soma's proprioceptive sense — her body IS the touchscreen. Each touch
//! event produces surprise (unexpected location), attention focus (where the finger
//! points), scroll velocity (motion state), and multi-touch count (integration/binding).
//!
//! The module maintains a simple velocity-based prediction of the next touch location.
//! Surprise is the Euclidean distance between the predicted and actual touch position,
//! grounding the Free Energy Principle in physical interaction: unexpected touches
//! produce prediction error that drives learning.
//!
//! Neuromodulator nudges translate touch patterns into biologically-grounded signals:
//! - Rapid tapping → NE spike (environmental change / arousal)
//! - Long press → 5-HT boost (focused calm attention)
//! - Fast scroll → NE boost (scanning / vigilance)
//! - Multi-touch → OT boost (integration / binding)

use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────────────────

/// Default long-press threshold in milliseconds.
const DEFAULT_LONG_PRESS_THRESHOLD_MS: u64 = 500;

/// EMA smoothing factor for scroll velocity (lower = smoother).
const SCROLL_EMA_ALPHA: f32 = 0.3;

/// NE nudge from rapid tapping (inter-tap < 300ms).
const TAP_NE_NUDGE: f32 = 0.04;

/// 5-HT nudge from a long press (sustained focused attention).
const LONG_PRESS_5HT_NUDGE: f32 = 0.03;

/// NE nudge from fast scrolling (velocity > 0.5 normalized/sec).
const SCROLL_NE_NUDGE: f32 = 0.03;

/// DA nudge from fast scrolling (novelty-seeking scan).
const SCROLL_DA_NUDGE: f32 = 0.02;

/// OT nudge per additional simultaneous touch (integration/binding).
const MULTI_TOUCH_OT_NUDGE: f32 = 0.02;

/// Maximum OT nudge from multi-touch (cap at 5 fingers).
const MAX_MULTI_TOUCH_OT: f32 = 0.10;

/// Velocity threshold (normalized units per second) for "fast" scroll.
const FAST_SCROLL_THRESHOLD: f32 = 0.5;

/// Inter-tap threshold in milliseconds for "rapid" tapping.
const RAPID_TAP_THRESHOLD_MS: u64 = 300;

/// Maximum number of tracked active touches.
const MAX_ACTIVE_TOUCHES: usize = 10;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Touch action type, mirroring Android MotionEvent actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TouchAction {
    /// Finger touched screen.
    Down,
    /// Finger moved on screen.
    Move,
    /// Finger lifted from screen.
    Up,
    /// Touch cancelled by system.
    Cancel,
}

/// A single touch event from the platform layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchEvent {
    /// X position normalized to 0.0–1.0.
    pub x: f32,
    /// Y position normalized to 0.0–1.0.
    pub y: f32,
    /// Type of touch action.
    pub action: TouchAction,
    /// Pressure normalized to 0.0–1.0.
    pub pressure: f32,
    /// Event timestamp in milliseconds since epoch (or boot).
    pub timestamp_ms: u64,
}

/// An active finger currently on the screen.
#[derive(Debug, Clone)]
struct ActiveTouch {
    x: f32,
    y: f32,
    down_timestamp_ms: u64,
    last_x: f32,
    last_y: f32,
    last_timestamp_ms: u64,
}

/// Recognized gesture type from touch sequence analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gesture {
    /// No gesture detected.
    None,
    /// Single tap (quick down+up).
    Tap,
    /// Double tap (two taps within 300ms).
    DoubleTap,
    /// Long press (sustained > 500ms).
    LongPress,
    /// Swipe in a cardinal direction.
    SwipeLeft,
    SwipeRight,
    SwipeUp,
    SwipeDown,
    /// Pinch (two fingers moving apart or together).
    PinchIn,
    PinchOut,
}

/// Consciousness-relevant output from touch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchBodyState {
    /// Prediction error: how unexpected this touch location was (0.0–1.0).
    pub surprise: f32,
    /// Where touch is directing attention (normalized x, y), if active.
    pub attention_focus: Option<(f32, f32)>,
    /// EMA-smoothed scroll velocity in normalized units per second.
    pub scroll_velocity: f32,
    /// Whether a long press is currently active (focused attention).
    pub is_long_press: bool,
    /// Number of simultaneous fingers on screen (integration/binding).
    pub multi_touch_count: u8,
    /// Whether any finger is currently touching the screen.
    pub touch_active: bool,
    /// Recognized gesture from touch sequence.
    pub gesture: Gesture,
}

impl Default for TouchBodyState {
    fn default() -> Self {
        Self {
            surprise: 0.0,
            attention_focus: None,
            scroll_velocity: 0.0,
            is_long_press: false,
            multi_touch_count: 0,
            touch_active: false,
            gesture: Gesture::None,
        }
    }
}

/// Neuromodulator nudges derived from touch patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TouchNudges {
    /// Norepinephrine delta (arousal, environmental change).
    pub ne_delta: f32,
    /// Serotonin delta (calm focused attention).
    pub serotonin_delta: f32,
    /// Dopamine delta (novelty-seeking, reward anticipation).
    pub da_delta: f32,
    /// Oxytocin delta (integration, binding, social touch).
    pub ot_delta: f32,
}

impl Default for TouchNudges {
    fn default() -> Self {
        Self {
            ne_delta: 0.0,
            serotonin_delta: 0.0,
            da_delta: 0.0,
            ot_delta: 0.0,
        }
    }
}

// ── TouchBody ──────────────────────────────────────────────────────────────────

/// Touch-as-proprioception processor.
///
/// Maintains active finger state, velocity prediction, and scroll EMA.
/// Each call to [`on_touch`] produces a [`TouchBodyState`] reflecting the
/// consciousness-relevant interpretation of the touch event.
pub struct TouchBody {
    /// Currently active fingers on screen.
    active_touches: Vec<ActiveTouch>,
    /// Timestamp of last processed event (ms).
    last_event_ms: u64,
    /// Predicted next touch location (simple velocity extrapolation).
    touch_prediction: Option<(f32, f32)>,
    /// Long-press detection threshold in milliseconds.
    long_press_threshold_ms: u64,
    /// EMA-smoothed scroll velocity (normalized units per second).
    scroll_ema: f32,
    /// Total events processed (lifetime counter).
    total_events: u64,
    /// Current output state (updated on each event).
    current_state: TouchBodyState,
    /// Timestamp of last Down event, for rapid-tap detection.
    last_down_ms: u64,
    /// Count of rapid taps (resets when inter-tap > threshold).
    rapid_tap_count: u32,
    /// Position of the initial Down event (for swipe detection).
    down_position: Option<(f32, f32)>,
    /// Previous two-finger distance (for pinch detection).
    prev_pinch_distance: Option<f32>,
}

impl TouchBody {
    /// Create a new TouchBody with default configuration.
    pub fn new() -> Self {
        Self {
            active_touches: Vec::with_capacity(4),
            last_event_ms: 0,
            touch_prediction: None,
            long_press_threshold_ms: DEFAULT_LONG_PRESS_THRESHOLD_MS,
            scroll_ema: 0.0,
            total_events: 0,
            current_state: TouchBodyState::default(),
            last_down_ms: 0,
            rapid_tap_count: 0,
            down_position: None,
            prev_pinch_distance: None,
        }
    }

    /// Process a touch event and return the updated consciousness state.
    ///
    /// This is the main entry point called from the platform layer each time
    /// a touch event arrives.
    pub fn on_touch(&mut self, event: TouchEvent) -> TouchBodyState {
        self.total_events += 1;

        let surprise = match event.action {
            TouchAction::Down => self.handle_down(&event),
            TouchAction::Move => self.handle_move(&event),
            TouchAction::Up => self.handle_up(&event),
            TouchAction::Cancel => self.handle_cancel(&event),
        };

        self.last_event_ms = event.timestamp_ms;

        // Determine long-press status: any active touch held longer than threshold.
        let is_long_press = self.active_touches.iter().any(|t| {
            event.timestamp_ms.saturating_sub(t.down_timestamp_ms) >= self.long_press_threshold_ms
        });

        let attention_focus = if !self.active_touches.is_empty() {
            // Attention centroid of all active touches.
            let (sx, sy) = self
                .active_touches
                .iter()
                .fold((0.0f32, 0.0f32), |(ax, ay), t| (ax + t.x, ay + t.y));
            let n = self.active_touches.len() as f32;
            Some((sx / n, sy / n))
        } else {
            None
        };

        let multi_touch_count = self.active_touches.len().min(255) as u8;
        let touch_active = !self.active_touches.is_empty();

        // Gesture detection
        let gesture = self.detect_gesture(&event, is_long_press, multi_touch_count);

        self.current_state = TouchBodyState {
            surprise,
            attention_focus,
            scroll_velocity: self.scroll_ema,
            is_long_press,
            multi_touch_count,
            touch_active,
            gesture,
        };

        self.current_state.clone()
    }

    /// Convert current touch state to neuromodulator nudges.
    ///
    /// Call after [`on_touch`] to get the deltas to apply to the neuromod bath.
    pub fn neuromod_nudges(&self) -> TouchNudges {
        let mut nudges = TouchNudges::default();

        // Rapid tapping → NE spike (environmental change detection).
        if self.rapid_tap_count >= 2 {
            nudges.ne_delta += TAP_NE_NUDGE * (self.rapid_tap_count as f32).min(5.0);
        }

        // Long press → 5-HT boost (focused calm attention).
        if self.current_state.is_long_press {
            nudges.serotonin_delta += LONG_PRESS_5HT_NUDGE;
        }

        // Fast scroll → NE + DA boost (scanning, novelty-seeking arousal).
        if self.scroll_ema > FAST_SCROLL_THRESHOLD {
            let intensity =
                ((self.scroll_ema - FAST_SCROLL_THRESHOLD) / FAST_SCROLL_THRESHOLD).min(1.0);
            nudges.ne_delta += SCROLL_NE_NUDGE * intensity;
            nudges.da_delta += SCROLL_DA_NUDGE * intensity;
        }

        // Multi-touch → OT boost (integration / binding).
        if self.current_state.multi_touch_count > 1 {
            let extra_fingers = (self.current_state.multi_touch_count - 1) as f32;
            nudges.ot_delta += (extra_fingers * MULTI_TOUCH_OT_NUDGE).min(MAX_MULTI_TOUCH_OT);
        }

        nudges
    }

    /// Detect gesture from current touch state and event sequence.
    fn detect_gesture(
        &mut self,
        event: &TouchEvent,
        is_long_press: bool,
        multi_touch_count: u8,
    ) -> Gesture {
        match event.action {
            TouchAction::Down => {
                self.down_position = Some((event.x, event.y));
                // Track pinch distance for two-finger gestures
                if self.active_touches.len() == 2 {
                    let t0 = &self.active_touches[0];
                    let t1 = &self.active_touches[1];
                    let dx = t0.x - t1.x;
                    let dy = t0.y - t1.y;
                    self.prev_pinch_distance = Some((dx * dx + dy * dy).sqrt());
                }
                Gesture::None
            }
            TouchAction::Move => {
                // Pinch detection (two fingers)
                if multi_touch_count == 2 && self.active_touches.len() == 2 {
                    let t0 = &self.active_touches[0];
                    let t1 = &self.active_touches[1];
                    let dx = t0.x - t1.x;
                    let dy = t0.y - t1.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if let Some(prev) = self.prev_pinch_distance {
                        let delta = dist - prev;
                        self.prev_pinch_distance = Some(dist);
                        if delta > 0.05 {
                            return Gesture::PinchOut;
                        } else if delta < -0.05 {
                            return Gesture::PinchIn;
                        }
                    }
                }
                Gesture::None
            }
            TouchAction::Up => {
                self.prev_pinch_distance = None;
                if let Some((dx, dy)) = self.down_position.take() {
                    let delta_x = event.x - dx;
                    let delta_y = event.y - dy;
                    let dist = (delta_x * delta_x + delta_y * delta_y).sqrt();
                    let duration = event.timestamp_ms.saturating_sub(self.last_down_ms);

                    if is_long_press {
                        return Gesture::LongPress;
                    }

                    // Swipe: moved > 0.15 normalized distance within 500ms
                    if dist > 0.15 && duration < 500 {
                        if delta_x.abs() > delta_y.abs() {
                            return if delta_x > 0.0 {
                                Gesture::SwipeRight
                            } else {
                                Gesture::SwipeLeft
                            };
                        } else {
                            return if delta_y > 0.0 {
                                Gesture::SwipeDown
                            } else {
                                Gesture::SwipeUp
                            };
                        }
                    }

                    // Tap: small movement, short duration
                    if dist < 0.05 && duration < 300 {
                        if self.rapid_tap_count >= 2 {
                            return Gesture::DoubleTap;
                        }
                        return Gesture::Tap;
                    }
                }
                Gesture::None
            }
            TouchAction::Cancel => {
                self.down_position = None;
                self.prev_pinch_distance = None;
                Gesture::None
            }
        }
    }

    /// Update the internal velocity-based prediction for the next touch location.
    ///
    /// Uses the most recent active touch's velocity to extrapolate forward.
    pub fn predict_next_touch(&mut self) {
        if let Some(touch) = self.active_touches.last() {
            let dt_ms = touch
                .last_timestamp_ms
                .saturating_sub(touch.down_timestamp_ms);
            if dt_ms > 0 {
                let dt_s = dt_ms as f32 / 1000.0;
                let vx = (touch.x - touch.last_x) / dt_s;
                let vy = (touch.y - touch.last_y) / dt_s;
                // Extrapolate 100ms forward, clamped to [0, 1].
                let pred_x = (touch.x + vx * 0.1).clamp(0.0, 1.0);
                let pred_y = (touch.y + vy * 0.1).clamp(0.0, 1.0);
                self.touch_prediction = Some((pred_x, pred_y));
            } else {
                // No motion yet — predict same location.
                self.touch_prediction = Some((touch.x, touch.y));
            }
        } else {
            self.touch_prediction = None;
        }
    }

    /// Current touch body state (read-only).
    pub fn state(&self) -> &TouchBodyState {
        &self.current_state
    }

    /// Total events processed over the lifetime of this TouchBody.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Set the long-press detection threshold in milliseconds.
    pub fn set_long_press_threshold_ms(&mut self, ms: u64) {
        self.long_press_threshold_ms = ms;
    }

    // ── Private handlers ───────────────────────────────────────────────────

    fn handle_down(&mut self, event: &TouchEvent) -> f32 {
        // Compute surprise as distance from prediction.
        let surprise = self.compute_surprise(event.x, event.y);

        // Track rapid tapping.
        if self.last_down_ms > 0
            && event.timestamp_ms.saturating_sub(self.last_down_ms) < RAPID_TAP_THRESHOLD_MS
        {
            self.rapid_tap_count += 1;
        } else {
            self.rapid_tap_count = 1;
        }
        self.last_down_ms = event.timestamp_ms;

        // Register new active touch (cap at MAX_ACTIVE_TOUCHES).
        if self.active_touches.len() < MAX_ACTIVE_TOUCHES {
            self.active_touches.push(ActiveTouch {
                x: event.x,
                y: event.y,
                down_timestamp_ms: event.timestamp_ms,
                last_x: event.x,
                last_y: event.y,
                last_timestamp_ms: event.timestamp_ms,
            });
        }

        surprise
    }

    fn handle_move(&mut self, event: &TouchEvent) -> f32 {
        // Update the most recent active touch position and compute velocity.
        if let Some(touch) = self.active_touches.last_mut() {
            let dt_ms = event.timestamp_ms.saturating_sub(touch.last_timestamp_ms);
            if dt_ms > 0 {
                let dt_s = dt_ms as f32 / 1000.0;
                let dx = event.x - touch.x;
                let dy = event.y - touch.y;
                let velocity = (dx * dx + dy * dy).sqrt() / dt_s;

                // EMA-smooth the scroll velocity.
                self.scroll_ema =
                    SCROLL_EMA_ALPHA * velocity + (1.0 - SCROLL_EMA_ALPHA) * self.scroll_ema;
            }

            touch.last_x = touch.x;
            touch.last_y = touch.y;
            touch.x = event.x;
            touch.y = event.y;
            touch.last_timestamp_ms = event.timestamp_ms;
        }

        // Update prediction after processing move.
        self.predict_next_touch();

        // Move events produce minimal surprise (finger is tracked).
        0.0
    }

    fn handle_up(&mut self, event: &TouchEvent) -> f32 {
        // Check if this was a long press before removing.
        let was_long_press = self.active_touches.iter().any(|t| {
            event.timestamp_ms.saturating_sub(t.down_timestamp_ms) >= self.long_press_threshold_ms
        });

        // Remove the last active touch (LIFO — matches multi-touch semantics).
        self.active_touches.pop();

        // Clear prediction on finger lift.
        if self.active_touches.is_empty() {
            self.touch_prediction = None;
            // Decay scroll velocity faster when finger lifts.
            self.scroll_ema *= 0.5;
        }

        // Lifting after a long press produces no surprise (expected).
        // Quick tap-and-lift produces mild surprise from the release.
        if was_long_press { 0.0 } else { 0.1 }
    }

    fn handle_cancel(&mut self, _event: &TouchEvent) -> f32 {
        // System cancelled all touches — clear state.
        self.active_touches.clear();
        self.touch_prediction = None;
        self.scroll_ema *= 0.3;

        // Cancellation is mildly surprising (unexpected interruption).
        0.3
    }

    /// Compute surprise as Euclidean distance from predicted location.
    ///
    /// If no prediction exists, surprise is maximum (1.0) — the touch is
    /// completely unexpected, which is exactly what the FEP would predict
    /// for a system with no prior about where the next touch will land.
    fn compute_surprise(&self, x: f32, y: f32) -> f32 {
        match self.touch_prediction {
            Some((px, py)) => {
                let dx = x - px;
                let dy = y - py;
                // Max possible distance on unit square is sqrt(2) ≈ 1.414.
                // Normalize to 0.0–1.0.
                let dist = (dx * dx + dy * dy).sqrt();
                (dist / std::f32::consts::SQRT_2).min(1.0)
            }
            None => 1.0,
        }
    }
}

impl Default for TouchBody {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(action: TouchAction, x: f32, y: f32, ts: u64) -> TouchEvent {
        TouchEvent {
            x,
            y,
            action,
            pressure: 0.5,
            timestamp_ms: ts,
        }
    }

    #[test]
    fn single_tap_produces_surprise() {
        let mut body = TouchBody::new();

        // First touch with no prediction → maximum surprise.
        let state = body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        assert_eq!(
            state.surprise, 1.0,
            "first touch should be maximally surprising"
        );
        assert!(state.touch_active);
        assert_eq!(state.multi_touch_count, 1);
    }

    #[test]
    fn predicted_touch_reduces_surprise() {
        let mut body = TouchBody::new();

        // First touch.
        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        // Move to establish velocity.
        body.on_touch(make_event(TouchAction::Move, 0.6, 0.5, 200));
        // Lift.
        body.on_touch(make_event(TouchAction::Up, 0.6, 0.5, 300));

        // Set prediction manually at the expected location.
        body.touch_prediction = Some((0.6, 0.5));

        // Touch near predicted location → low surprise.
        let state = body.on_touch(make_event(TouchAction::Down, 0.61, 0.5, 400));
        assert!(
            state.surprise < 0.1,
            "touch near prediction should have low surprise, got {}",
            state.surprise
        );
    }

    #[test]
    fn long_press_detection() {
        let mut body = TouchBody::new();

        // Touch down.
        let state = body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        assert!(!state.is_long_press, "should not be long press immediately");

        // Move event at 600ms (> 500ms threshold), same position.
        let state = body.on_touch(make_event(TouchAction::Move, 0.5, 0.5, 700));
        assert!(state.is_long_press, "should be long press after 600ms");

        // Verify neuromod nudge.
        let nudges = body.neuromod_nudges();
        assert!(
            nudges.serotonin_delta > 0.0,
            "long press should produce 5-HT nudge"
        );
    }

    #[test]
    fn scroll_velocity_tracking() {
        let mut body = TouchBody::new();

        // Touch down.
        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 0));

        // Fast vertical scroll: 0.5 units in 100ms = 5.0 units/sec.
        body.on_touch(make_event(TouchAction::Move, 0.5, 0.0, 100));

        let state = body.state();
        assert!(
            state.scroll_velocity > 0.0,
            "scroll velocity should be positive after move, got {}",
            state.scroll_velocity
        );
        assert!(
            state.scroll_velocity > FAST_SCROLL_THRESHOLD,
            "fast scroll should exceed threshold, got {}",
            state.scroll_velocity
        );

        // Neuromod should reflect fast scroll.
        let nudges = body.neuromod_nudges();
        assert!(nudges.ne_delta > 0.0, "fast scroll should produce NE nudge");
        assert!(nudges.da_delta > 0.0, "fast scroll should produce DA nudge");
    }

    #[test]
    fn multi_touch_count() {
        let mut body = TouchBody::new();

        // Two fingers down.
        body.on_touch(make_event(TouchAction::Down, 0.3, 0.3, 100));
        let state = body.on_touch(make_event(TouchAction::Down, 0.7, 0.7, 100));

        assert_eq!(state.multi_touch_count, 2);
        assert!(state.touch_active);

        // Attention focus should be centroid of both touches.
        let (fx, fy) = state.attention_focus.unwrap();
        assert!(
            (fx - 0.5).abs() < 0.01,
            "centroid x should be ~0.5, got {}",
            fx
        );
        assert!(
            (fy - 0.5).abs() < 0.01,
            "centroid y should be ~0.5, got {}",
            fy
        );

        // OT nudge from multi-touch.
        let nudges = body.neuromod_nudges();
        assert!(nudges.ot_delta > 0.0, "multi-touch should produce OT nudge");
        assert!(
            (nudges.ot_delta - MULTI_TOUCH_OT_NUDGE).abs() < f32::EPSILON,
            "2 fingers = 1 extra = 1x OT nudge"
        );
    }

    #[test]
    fn neuromod_nudges_rapid_tapping() {
        let mut body = TouchBody::new();

        // Rapid taps: 3 taps within 300ms intervals.
        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        body.on_touch(make_event(TouchAction::Up, 0.5, 0.5, 150));
        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 300));
        body.on_touch(make_event(TouchAction::Up, 0.5, 0.5, 350));
        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 500));

        let nudges = body.neuromod_nudges();
        assert!(
            nudges.ne_delta > 0.0,
            "rapid tapping should produce NE nudge, got {}",
            nudges.ne_delta
        );
        // 3 rapid taps → NE = TAP_NE_NUDGE * 3.
        assert!(
            (nudges.ne_delta - TAP_NE_NUDGE * 3.0).abs() < f32::EPSILON,
            "3 rapid taps should produce 3x NE nudge, got {}",
            nudges.ne_delta
        );
    }

    #[test]
    fn cancel_clears_state() {
        let mut body = TouchBody::new();

        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        assert!(body.state().touch_active);

        let state = body.on_touch(make_event(TouchAction::Cancel, 0.5, 0.5, 200));
        assert!(!state.touch_active);
        assert_eq!(state.multi_touch_count, 0);
        assert!(state.attention_focus.is_none());
        assert!(
            (state.surprise - 0.3).abs() < f32::EPSILON,
            "cancel surprise should be 0.3"
        );
    }

    #[test]
    fn up_without_long_press_gives_mild_surprise() {
        let mut body = TouchBody::new();

        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        // Quick lift at 200ms (< 500ms threshold).
        let state = body.on_touch(make_event(TouchAction::Up, 0.5, 0.5, 200));

        assert!(!state.touch_active);
        assert!(
            (state.surprise - 0.1).abs() < f32::EPSILON,
            "quick tap-and-lift should produce mild surprise (0.1), got {}",
            state.surprise
        );
    }

    #[test]
    fn scroll_ema_decays_on_lift() {
        let mut body = TouchBody::new();

        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 0));
        body.on_touch(make_event(TouchAction::Move, 0.5, 0.0, 100));

        let vel_before = body.scroll_ema;
        assert!(vel_before > 0.0);

        body.on_touch(make_event(TouchAction::Up, 0.5, 0.0, 200));

        assert!(
            body.scroll_ema < vel_before,
            "scroll EMA should decay on finger lift"
        );
        assert!(
            (body.scroll_ema - vel_before * 0.5).abs() < f32::EPSILON,
            "scroll EMA should halve on lift"
        );
    }

    #[test]
    fn total_events_counter() {
        let mut body = TouchBody::new();
        assert_eq!(body.total_events(), 0);

        body.on_touch(make_event(TouchAction::Down, 0.5, 0.5, 100));
        body.on_touch(make_event(TouchAction::Move, 0.6, 0.5, 200));
        body.on_touch(make_event(TouchAction::Up, 0.6, 0.5, 300));

        assert_eq!(body.total_events(), 3);
    }

    #[test]
    fn default_impl_matches_new() {
        let a = TouchBody::new();
        let b = TouchBody::default();
        assert_eq!(a.total_events(), b.total_events());
        assert_eq!(a.scroll_ema, b.scroll_ema);
        assert_eq!(a.long_press_threshold_ms, b.long_press_threshold_ms);
    }

    #[test]
    fn max_active_touches_capped() {
        let mut body = TouchBody::new();

        // Add MAX_ACTIVE_TOUCHES + 2 fingers.
        for i in 0..(MAX_ACTIVE_TOUCHES + 2) {
            let x = (i as f32) / (MAX_ACTIVE_TOUCHES as f32);
            body.on_touch(make_event(
                TouchAction::Down,
                x.min(1.0),
                0.5,
                100 + i as u64,
            ));
        }

        assert_eq!(
            body.active_touches.len(),
            MAX_ACTIVE_TOUCHES,
            "should cap at MAX_ACTIVE_TOUCHES"
        );
    }
}
