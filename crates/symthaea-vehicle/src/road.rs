// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Road model: segments, projection, curvature, and presets.
//!
//! Provides geometric road descriptions so the vehicle's `lateral_offset` and
//! `curvature_ahead` channels carry meaningful signal during training.

use serde::{Deserialize, Serialize};

use crate::types::VehicleState;

/// A single road segment before resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoadSegment {
    /// A straight segment of given length (m).
    Straight { length: f64 },
    /// A circular arc. `radius` in meters, `sweep_angle` in radians.
    /// Positive sweep = curving left (counter-clockwise).
    Arc { radius: f64, sweep_angle: f64 },
}

/// A resolved segment with precomputed start pose and cumulative arc-length.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedSegment {
    pub kind: RoadSegment,
    /// World-frame start pose: (x, y, heading).
    pub start_x: f64,
    pub start_y: f64,
    pub start_heading: f64,
    /// Cumulative arc-length at the start of this segment.
    pub s_start: f64,
    /// Length of this segment.
    pub length: f64,
}

/// Projection result from `Road::project`.
#[derive(Debug, Clone, Copy)]
pub struct RoadProjection {
    /// Arc-length along the road centerline (m).
    pub s: f64,
    /// Signed lateral offset from centerline (m). Positive = left of road direction.
    pub lateral_offset: f64,
    /// Road heading at the projected point (rad).
    pub road_heading: f64,
}

/// A road composed of sequential segments with precomputed geometry.
#[derive(Debug, Clone)]
pub struct Road {
    segments: Vec<ResolvedSegment>,
    /// Total arc-length of the road (m).
    pub total_length: f64,
    /// Look-ahead distance for curvature query (m).
    pub lookahead: f64,
    /// Whether the road forms a closed loop.
    pub closed: bool,
}

impl Road {
    /// Build a road from a sequence of segments, starting at origin heading east.
    pub fn from_segments(segs: &[RoadSegment]) -> Self {
        Self::from_segments_at(segs, 0.0, 0.0, 0.0, false)
    }

    /// Build a closed-loop road (s wraps modulo total_length).
    pub fn from_segments_closed(segs: &[RoadSegment]) -> Self {
        Self::from_segments_at(segs, 0.0, 0.0, 0.0, true)
    }

    fn from_segments_at(
        segs: &[RoadSegment],
        start_x: f64,
        start_y: f64,
        start_heading: f64,
        closed: bool,
    ) -> Self {
        let mut resolved = Vec::with_capacity(segs.len());
        let mut x = start_x;
        let mut y = start_y;
        let mut h = start_heading;
        let mut s_accum = 0.0;

        for seg in segs {
            let seg_len = match seg {
                RoadSegment::Straight { length } => *length,
                RoadSegment::Arc {
                    radius,
                    sweep_angle,
                } => radius * sweep_angle.abs(),
            };

            resolved.push(ResolvedSegment {
                kind: seg.clone(),
                start_x: x,
                start_y: y,
                start_heading: h,
                s_start: s_accum,
                length: seg_len,
            });

            // Advance pose to end of segment
            match seg {
                RoadSegment::Straight { length } => {
                    x += h.cos() * length;
                    y += h.sin() * length;
                }
                RoadSegment::Arc {
                    radius,
                    sweep_angle,
                } => {
                    // Center of the turning circle
                    let sign = sweep_angle.signum();
                    let cx = x - sign * h.sin() * radius;
                    let cy = y + sign * h.cos() * radius;
                    let new_h = h + sweep_angle;
                    x = cx + sign * new_h.sin() * radius;
                    y = cy - sign * new_h.cos() * radius;
                    h = new_h;
                }
            }
            s_accum += seg_len;
        }

        Self {
            segments: resolved,
            total_length: s_accum,
            lookahead: 20.0,
            closed,
        }
    }

    /// Project a world point onto the road centerline.
    /// Returns the closest (s, lateral_offset, road_heading).
    pub fn project(&self, x: f64, y: f64) -> RoadProjection {
        let mut best = RoadProjection {
            s: 0.0,
            lateral_offset: f64::MAX,
            road_heading: 0.0,
        };
        let mut best_abs_lat = f64::MAX;

        for seg in &self.segments {
            let proj = Self::project_onto_segment(seg, x, y);
            let abs_lat = proj.lateral_offset.abs();
            if abs_lat < best_abs_lat {
                best = proj;
                best_abs_lat = abs_lat;
            }
        }

        best
    }

    fn project_onto_segment(seg: &ResolvedSegment, px: f64, py: f64) -> RoadProjection {
        match &seg.kind {
            RoadSegment::Straight { .. } => {
                // Direction vector of the segment
                let dx = seg.start_heading.cos();
                let dy = seg.start_heading.sin();
                // Vector from segment start to point
                let vx = px - seg.start_x;
                let vy = py - seg.start_y;
                // Longitudinal projection (dot product)
                let along = (vx * dx + vy * dy).clamp(0.0, seg.length);
                // Lateral offset (cross product): positive = left of direction
                let lateral = vx * dy - vy * dx;
                RoadProjection {
                    s: seg.s_start + along,
                    lateral_offset: -lateral,
                    road_heading: seg.start_heading,
                }
            }
            RoadSegment::Arc {
                radius,
                sweep_angle,
            } => {
                let sign = sweep_angle.signum();
                // Center of turning circle
                let cx = seg.start_x - sign * seg.start_heading.sin() * radius;
                let cy = seg.start_y + sign * seg.start_heading.cos() * radius;

                // Angle from center to point
                let to_px = px - cx;
                let to_py = py - cy;
                let angle_to_point = to_py.atan2(to_px);

                // Angle from center to segment start
                let angle_start = (seg.start_y - cy).atan2(seg.start_x - cx);

                // Angular position along the arc
                let mut delta_angle = angle_to_point - angle_start;
                // Normalize to the direction of the sweep
                if sign > 0.0 {
                    while delta_angle < 0.0 {
                        delta_angle += 2.0 * std::f64::consts::PI;
                    }
                    while delta_angle > 2.0 * std::f64::consts::PI {
                        delta_angle -= 2.0 * std::f64::consts::PI;
                    }
                } else {
                    while delta_angle > 0.0 {
                        delta_angle -= 2.0 * std::f64::consts::PI;
                    }
                    while delta_angle < -2.0 * std::f64::consts::PI {
                        delta_angle += 2.0 * std::f64::consts::PI;
                    }
                }

                let arc_frac = delta_angle / sweep_angle;
                let arc_frac_clamped = arc_frac.clamp(0.0, 1.0);
                let s_along = arc_frac_clamped * seg.length;

                // Road heading at this arc position
                let road_heading = seg.start_heading + sweep_angle * arc_frac_clamped;

                // Lateral offset: distance from center minus radius, signed
                let dist = (to_px * to_px + to_py * to_py).sqrt();
                let lateral = sign * (dist - *radius);

                RoadProjection {
                    s: seg.s_start + s_along,
                    lateral_offset: lateral,
                    road_heading,
                }
            }
        }
    }

    /// Curvature at arc-length s. Straight=0, Arc=sign/radius.
    pub fn curvature_at(&self, s: f64) -> f64 {
        let s = if self.closed && self.total_length > 0.0 {
            s.rem_euclid(self.total_length)
        } else {
            s.clamp(0.0, self.total_length)
        };

        for seg in &self.segments {
            let s_end = seg.s_start + seg.length;
            if s >= seg.s_start && s <= s_end {
                return match &seg.kind {
                    RoadSegment::Straight { .. } => 0.0,
                    RoadSegment::Arc {
                        radius,
                        sweep_angle,
                    } => sweep_angle.signum() / radius,
                };
            }
        }
        0.0
    }

    /// Road heading at arc-length s.
    pub fn heading_at(&self, s: f64) -> f64 {
        let s = if self.closed && self.total_length > 0.0 {
            s.rem_euclid(self.total_length)
        } else {
            s.clamp(0.0, self.total_length)
        };

        for seg in &self.segments {
            let s_end = seg.s_start + seg.length;
            if s >= seg.s_start && s <= s_end {
                let frac = if seg.length > 0.0 {
                    (s - seg.s_start) / seg.length
                } else {
                    0.0
                };
                return match &seg.kind {
                    RoadSegment::Straight { .. } => seg.start_heading,
                    RoadSegment::Arc { sweep_angle, .. } => seg.start_heading + sweep_angle * frac,
                };
            }
        }
        0.0
    }

    /// Exact world position and heading at arc-length s.
    ///
    /// Returns (x, y, heading) using the same closed-form geometry as construction.
    pub fn position_at(&self, s: f64) -> (f64, f64, f64) {
        let s = if self.closed && self.total_length > 0.0 {
            s.rem_euclid(self.total_length)
        } else {
            s.clamp(0.0, self.total_length)
        };

        for seg in &self.segments {
            let s_end = seg.s_start + seg.length;
            if s >= seg.s_start && s <= s_end {
                let ds = s - seg.s_start;
                return match &seg.kind {
                    RoadSegment::Straight { .. } => {
                        let x = seg.start_x + seg.start_heading.cos() * ds;
                        let y = seg.start_y + seg.start_heading.sin() * ds;
                        (x, y, seg.start_heading)
                    }
                    RoadSegment::Arc {
                        radius,
                        sweep_angle,
                    } => {
                        let sign = sweep_angle.signum();
                        let cx = seg.start_x - sign * seg.start_heading.sin() * radius;
                        let cy = seg.start_y + sign * seg.start_heading.cos() * radius;
                        let frac = if seg.length > 0.0 {
                            ds / seg.length
                        } else {
                            0.0
                        };
                        let partial_sweep = sweep_angle * frac;
                        let h = seg.start_heading + partial_sweep;
                        let x = cx + sign * h.sin() * radius;
                        let y = cy - sign * h.cos() * radius;
                        (x, y, h)
                    }
                };
            }
        }
        (0.0, 0.0, 0.0)
    }

    /// Apply road context to a vehicle state: sets lateral_offset and curvature_ahead.
    pub fn apply(&self, state: &mut VehicleState) {
        let proj = self.project(state.position_x, state.position_y);
        state.lateral_offset = proj.lateral_offset;
        state.curvature_ahead = self.curvature_at(proj.s + self.lookahead);
    }

    // ── Preset roads ──

    /// A 2000m straight road along the x-axis.
    pub fn straight() -> Self {
        Self::from_segments(&[RoadSegment::Straight { length: 2000.0 }])
    }

    /// An oval: two straights connected by two 180° arcs.
    pub fn oval(straight_len: f64, radius: f64) -> Self {
        Self::from_segments_closed(&[
            RoadSegment::Straight {
                length: straight_len,
            },
            RoadSegment::Arc {
                radius,
                sweep_angle: std::f64::consts::PI,
            },
            RoadSegment::Straight {
                length: straight_len,
            },
            RoadSegment::Arc {
                radius,
                sweep_angle: std::f64::consts::PI,
            },
        ])
    }

    /// S-curve: straight → left arc → right arc → straight.
    pub fn s_curve() -> Self {
        Self::from_segments(&[
            RoadSegment::Straight { length: 200.0 },
            RoadSegment::Arc {
                radius: 100.0,
                sweep_angle: std::f64::consts::FRAC_PI_4,
            },
            RoadSegment::Arc {
                radius: 100.0,
                sweep_angle: -std::f64::consts::FRAC_PI_4,
            },
            RoadSegment::Straight { length: 200.0 },
        ])
    }

    /// Gentle highway: long straights with R=200m bends.
    pub fn gentle_highway() -> Self {
        Self::from_segments(&[
            RoadSegment::Straight { length: 500.0 },
            RoadSegment::Arc {
                radius: 200.0,
                sweep_angle: std::f64::consts::FRAC_PI_6,
            },
            RoadSegment::Straight { length: 400.0 },
            RoadSegment::Arc {
                radius: 200.0,
                sweep_angle: -std::f64::consts::FRAC_PI_8,
            },
            RoadSegment::Straight { length: 500.0 },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_straight_projection() {
        let road = Road::straight();
        let proj = road.project(100.0, 2.0);
        assert!((proj.s - 100.0).abs() < 1.0, "s={}", proj.s);
        assert!(
            (proj.lateral_offset - 2.0).abs() < 0.1,
            "lat={}",
            proj.lateral_offset
        );
        assert!(
            proj.road_heading.abs() < 0.01,
            "heading={}",
            proj.road_heading
        );
    }

    #[test]
    fn test_straight_curvature() {
        let road = Road::straight();
        assert!(
            road.curvature_at(50.0).abs() < 1e-10,
            "Straight curvature should be 0"
        );
        assert!(
            road.curvature_at(500.0).abs() < 1e-10,
            "Straight curvature should be 0"
        );
    }

    #[test]
    fn test_arc_curvature() {
        let road = Road::from_segments(&[RoadSegment::Arc {
            radius: 100.0,
            sweep_angle: std::f64::consts::FRAC_PI_2,
        }]);
        let k = road.curvature_at(10.0);
        assert!(
            (k - 0.01).abs() < 1e-6,
            "Arc curvature should be 1/R = 0.01: {k}"
        );
    }

    #[test]
    fn test_oval_total_length() {
        let road = Road::oval(200.0, 50.0);
        let expected = 2.0 * 200.0 + 2.0 * std::f64::consts::PI * 50.0;
        assert!(
            (road.total_length - expected).abs() < 1.0,
            "Oval length: {} vs expected {}",
            road.total_length,
            expected
        );
        assert!(road.closed);
    }

    #[test]
    fn test_s_curve_sign_flip() {
        let road = Road::s_curve();
        // First arc is left (positive curvature)
        let k1 = road.curvature_at(250.0);
        assert!(k1 > 0.0, "First arc should curve left: {k1}");
        // Second arc is right (negative curvature)
        // First arc length = 100 * pi/4 ≈ 78.5, starts at s=200
        // Second arc starts at s ≈ 278.5
        let k2 = road.curvature_at(300.0);
        assert!(k2 < 0.0, "Second arc should curve right: {k2}");
    }

    #[test]
    fn test_road_apply() {
        let road = Road::straight();
        let mut state = VehicleState::cruising(13.4);
        state.position_x = 50.0;
        state.position_y = 1.5;
        road.apply(&mut state);
        assert!(
            (state.lateral_offset - 1.5).abs() < 0.1,
            "lat={}",
            state.lateral_offset
        );
        assert!(
            state.curvature_ahead.abs() < 1e-6,
            "curvature={}",
            state.curvature_ahead
        );
    }

    #[test]
    fn test_presets_construct() {
        let straight = Road::straight();
        assert!(straight.total_length > 100.0);

        let oval = Road::oval(300.0, 80.0);
        assert!(oval.total_length > 600.0);
        assert!(oval.closed);

        let s = Road::s_curve();
        assert!(s.total_length > 400.0);

        let hw = Road::gentle_highway();
        assert!(hw.total_length > 1000.0);
    }

    #[test]
    fn test_heading_at_straight() {
        let road = Road::straight();
        let h = road.heading_at(500.0);
        assert!(h.abs() < 1e-10, "Straight heading should be 0: {h}");
    }

    #[test]
    fn test_heading_at_arc() {
        let road = Road::from_segments(&[RoadSegment::Arc {
            radius: 100.0,
            sweep_angle: std::f64::consts::FRAC_PI_2,
        }]);
        let h = road.heading_at(road.total_length);
        assert!(
            (h - std::f64::consts::FRAC_PI_2).abs() < 0.01,
            "End heading should be pi/2: {h}"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let seg = RoadSegment::Arc {
            radius: 50.0,
            sweep_angle: 1.0,
        };
        let json = serde_json::to_string(&seg).unwrap();
        let restored: RoadSegment = serde_json::from_str(&json).unwrap();
        match restored {
            RoadSegment::Arc {
                radius,
                sweep_angle,
            } => {
                assert!((radius - 50.0).abs() < 1e-10);
                assert!((sweep_angle - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Arc"),
        }
    }

    #[test]
    fn test_segment_boundary() {
        // Road: 100m straight + 100m straight
        let road = Road::from_segments(&[
            RoadSegment::Straight { length: 100.0 },
            RoadSegment::Straight { length: 100.0 },
        ]);
        // Point at exactly the boundary
        let proj = road.project(100.0, 0.5);
        assert!(
            (proj.s - 100.0).abs() < 1.0,
            "Boundary s should be ~100: {}",
            proj.s
        );
        assert!(
            (proj.lateral_offset - 0.5).abs() < 0.1,
            "Boundary lat should be ~0.5: {}",
            proj.lateral_offset
        );
    }

    #[test]
    fn test_position_at_straight() {
        let road = Road::straight();
        let (x, y, h) = road.position_at(500.0);
        assert!((x - 500.0).abs() < 1e-6, "x={x}");
        assert!(y.abs() < 1e-6, "y={y}");
        assert!(h.abs() < 1e-6, "h={h}");
    }

    #[test]
    fn test_position_at_arc_quarter_turn() {
        let r = 100.0;
        let road = Road::from_segments(&[RoadSegment::Arc {
            radius: r,
            sweep_angle: std::f64::consts::FRAC_PI_2,
        }]);
        // End of 90° left arc: should be at (R, R) heading pi/2
        let (x, y, h) = road.position_at(road.total_length);
        assert!((x - r).abs() < 1.0, "Quarter turn end x should be ~R: {x}");
        assert!((y - r).abs() < 1.0, "Quarter turn end y should be ~R: {y}");
        assert!(
            (h - std::f64::consts::FRAC_PI_2).abs() < 0.01,
            "End heading should be pi/2: {h}"
        );
    }

    #[test]
    fn test_position_at_oval_wraps() {
        let road = Road::oval(200.0, 50.0);
        // s=0 and s=total_length should give the same position (closed loop)
        let (x0, y0, _h0) = road.position_at(0.0);
        let (xn, yn, _hn) = road.position_at(road.total_length);
        assert!((x0 - xn).abs() < 1.0, "Oval wrap x: {x0} vs {xn}");
        assert!((y0 - yn).abs() < 1.0, "Oval wrap y: {y0} vs {yn}");
    }

    #[test]
    fn test_position_at_midpoint_straight() {
        let road = Road::from_segments(&[
            RoadSegment::Straight { length: 100.0 },
            RoadSegment::Straight { length: 100.0 },
        ]);
        let (x, y, _) = road.position_at(150.0);
        assert!((x - 150.0).abs() < 1e-6, "x={x}");
        assert!(y.abs() < 1e-6, "y={y}");
    }
}
