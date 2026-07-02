// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Consciousness Journey — a path through V/A/Phi space that unfolds over time.
//!
//! Users plot waypoints on a timeline. The system interpolates between them
//! using Catmull-Rom splines, generating a unique composition from the
//! consciousness arc.

use serde::{Deserialize, Serialize};

/// A single point on the consciousness journey.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Waypoint {
    /// Position on the timeline (seconds from start).
    pub time_secs: f32,
    /// Emotional valence [-1, 1].
    pub valence: f32,
    /// Emotional arousal [0, 1].
    pub arousal: f32,
    /// Consciousness level [0, 1].
    pub phi: f32,
}

/// A complete consciousness journey.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Journey {
    pub waypoints: Vec<Waypoint>,
    pub duration_secs: f32,
}

impl Journey {
    pub fn new(duration_secs: f32) -> Self {
        Self {
            waypoints: vec![
                Waypoint { time_secs: 0.0, valence: 0.0, arousal: 0.3, phi: 0.3 },
                Waypoint { time_secs: duration_secs, valence: 0.0, arousal: 0.3, phi: 0.3 },
            ],
            duration_secs,
        }
    }

    /// Sample the journey at a given time using Catmull-Rom interpolation.
    /// Returns (valence, arousal, phi).
    pub fn sample(&self, time: f32) -> (f32, f32, f32) {
        let t = time.clamp(0.0, self.duration_secs);
        let pts = &self.waypoints;

        if pts.is_empty() {
            return (0.0, 0.5, 0.5);
        }
        if pts.len() == 1 {
            return (pts[0].valence, pts[0].arousal, pts[0].phi);
        }

        // Find the segment containing t
        let mut i = 0;
        for (idx, w) in pts.iter().enumerate() {
            if w.time_secs > t {
                break;
            }
            i = idx;
        }

        // Clamp segment indices
        let i1 = i.min(pts.len() - 2);
        let i2 = i1 + 1;

        let t1 = pts[i1].time_secs;
        let t2 = pts[i2].time_secs;
        let segment_t = if (t2 - t1).abs() > 0.001 {
            ((t - t1) / (t2 - t1)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Catmull-Rom control points (clamp at boundaries)
        let i0 = if i1 > 0 { i1 - 1 } else { i1 };
        let i3 = (i2 + 1).min(pts.len() - 1);

        let v = catmull_rom(
            pts[i0].valence, pts[i1].valence,
            pts[i2].valence, pts[i3].valence,
            segment_t,
        ).clamp(-1.0, 1.0);

        let a = catmull_rom(
            pts[i0].arousal, pts[i1].arousal,
            pts[i2].arousal, pts[i3].arousal,
            segment_t,
        ).clamp(0.0, 1.0);

        let p = catmull_rom(
            pts[i0].phi, pts[i1].phi,
            pts[i2].phi, pts[i3].phi,
            segment_t,
        ).clamp(0.0, 1.0);

        (v, a, p)
    }

    /// Add a waypoint at the given time, inserting in sorted order.
    pub fn add_waypoint(&mut self, wp: Waypoint) {
        let pos = self.waypoints
            .iter()
            .position(|w| w.time_secs > wp.time_secs)
            .unwrap_or(self.waypoints.len());
        self.waypoints.insert(pos, wp);
    }

    /// Remove the waypoint nearest to the given time (but not first/last).
    pub fn remove_nearest(&mut self, time: f32) {
        if self.waypoints.len() <= 2 {
            return; // Keep at least start + end
        }
        let mut best_idx = 1;
        let mut best_dist = f32::MAX;
        for (i, w) in self.waypoints.iter().enumerate() {
            if i == 0 || i == self.waypoints.len() - 1 {
                continue; // Don't remove endpoints
            }
            let d = (w.time_secs - time).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        self.waypoints.remove(best_idx);
    }

    /// Generate N evenly-spaced samples for visualization preview.
    pub fn preview_samples(&self, n: usize) -> Vec<(f32, f32, f32, f32)> {
        (0..n)
            .map(|i| {
                let t = self.duration_secs * i as f32 / (n - 1).max(1) as f32;
                let (v, a, p) = self.sample(t);
                (t, v, a, p)
            })
            .collect()
    }
}

/// Catmull-Rom spline interpolation between p1 and p2.
/// p0 and p3 are the neighboring control points for tangent estimation.
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

// --- localStorage persistence ---

const STORAGE_KEY: &str = "mycelix-music-journeys";

/// Saved journey with a name.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedJourney {
    pub name: String,
    pub journey: Journey,
}

/// Load saved journeys from localStorage.
pub fn load_saved_journeys() -> Vec<SavedJourney> {
    let storage = match web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
    {
        Some(s) => s,
        None => return Vec::new(),
    };
    let json = match storage.get_item(STORAGE_KEY).ok().flatten() {
        Some(j) => j,
        None => return Vec::new(),
    };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Save a journey to localStorage.
pub fn save_journey(name: &str, journey: &Journey) {
    let mut saved = load_saved_journeys();
    // Replace if name exists, else append
    if let Some(existing) = saved.iter_mut().find(|s| s.name == name) {
        existing.journey = journey.clone();
    } else {
        saved.push(SavedJourney {
            name: name.to_string(),
            journey: journey.clone(),
        });
    }
    if let Ok(json) = serde_json::to_string(&saved) {
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item(STORAGE_KEY, &json);
        }
    }
}

/// Delete a saved journey by name.
pub fn delete_saved_journey(name: &str) {
    let mut saved = load_saved_journeys();
    saved.retain(|s| s.name != name);
    if let Ok(json) = serde_json::to_string(&saved) {
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item(STORAGE_KEY, &json);
        }
    }
}

/// Preset journeys for the gallery / quick-start.
pub fn preset_journeys() -> Vec<(&'static str, Journey)> {
    vec![
        ("Awakening", Journey {
            duration_secs: 60.0,
            waypoints: vec![
                Waypoint { time_secs: 0.0, valence: -0.2, arousal: 0.1, phi: 0.1 },
                Waypoint { time_secs: 15.0, valence: 0.0, arousal: 0.2, phi: 0.3 },
                Waypoint { time_secs: 30.0, valence: 0.3, arousal: 0.4, phi: 0.5 },
                Waypoint { time_secs: 45.0, valence: 0.6, arousal: 0.6, phi: 0.7 },
                Waypoint { time_secs: 60.0, valence: 0.8, arousal: 0.5, phi: 0.9 },
            ],
        }),
        ("Storm & Calm", Journey {
            duration_secs: 90.0,
            waypoints: vec![
                Waypoint { time_secs: 0.0, valence: 0.2, arousal: 0.3, phi: 0.5 },
                Waypoint { time_secs: 20.0, valence: -0.5, arousal: 0.8, phi: 0.3 },
                Waypoint { time_secs: 40.0, valence: -0.8, arousal: 0.95, phi: 0.2 },
                Waypoint { time_secs: 60.0, valence: -0.2, arousal: 0.5, phi: 0.4 },
                Waypoint { time_secs: 75.0, valence: 0.4, arousal: 0.2, phi: 0.7 },
                Waypoint { time_secs: 90.0, valence: 0.6, arousal: 0.15, phi: 0.85 },
            ],
        }),
        ("Contemplation", Journey {
            duration_secs: 120.0,
            waypoints: vec![
                Waypoint { time_secs: 0.0, valence: 0.1, arousal: 0.2, phi: 0.4 },
                Waypoint { time_secs: 40.0, valence: 0.0, arousal: 0.15, phi: 0.6 },
                Waypoint { time_secs: 80.0, valence: 0.2, arousal: 0.1, phi: 0.8 },
                Waypoint { time_secs: 120.0, valence: 0.3, arousal: 0.1, phi: 0.95 },
            ],
        }),
    ]
}
