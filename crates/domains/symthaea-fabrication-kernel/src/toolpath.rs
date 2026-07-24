// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! G-code generation from sliced layers.
//!
//! Converts `SliceLayer` contour data into a sequence of G-code commands
//! suitable for FDM 3D printers. Handles preamble/postamble, extrusion
//! calculation, retraction on long travel moves, and temperature control.

use crate::slicer::{Contour, Point2, Segment2, SliceConfig, SliceError, SliceLayer};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Configuration ────────────────────────────────────────────────────────

/// Configuration for G-code generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolpathConfig {
    /// Print speed in mm/s.
    pub print_speed_mm_s: f32,
    /// Travel (non-extruding) speed in mm/s.
    pub travel_speed_mm_s: f32,
    /// Retraction distance in mm.
    pub retract_distance_mm: f32,
    /// Retraction speed in mm/s.
    pub retract_speed_mm_s: f32,
    /// Extrusion width in mm.
    pub extrusion_width_mm: f32,
    /// Filament diameter in mm.
    pub filament_diameter_mm: f32,
    /// Heated bed temperature in degrees Celsius.
    pub bed_temp_c: u16,
    /// Nozzle temperature in degrees Celsius.
    pub nozzle_temp_c: u16,
}

impl Default for ToolpathConfig {
    fn default() -> Self {
        Self {
            print_speed_mm_s: 60.0,
            travel_speed_mm_s: 120.0,
            retract_distance_mm: 1.0,
            retract_speed_mm_s: 40.0,
            extrusion_width_mm: 0.4,
            filament_diameter_mm: 1.75,
            bed_temp_c: 60,
            nozzle_temp_c: 210,
        }
    }
}

/// Invalid toolpath configuration or layer geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum GCodeGenerationError {
    InvalidSliceConfig(SliceError),
    InvalidConfig(&'static str),
    EmptyJob,
    InvalidLayer {
        layer_index: usize,
        reason: &'static str,
    },
}

impl fmt::Display for GCodeGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSliceConfig(error) => write!(f, "invalid slice configuration: {error}"),
            Self::InvalidConfig(field) => write!(f, "invalid toolpath configuration: {field}"),
            Self::EmptyJob => write!(f, "toolpath job contains no printable geometry"),
            Self::InvalidLayer {
                layer_index,
                reason,
            } => {
                write!(f, "invalid layer {layer_index}: {reason}")
            }
        }
    }
}

impl std::error::Error for GCodeGenerationError {}

impl ToolpathConfig {
    /// Validate all scalar inputs without coercion.
    pub fn validate_strict(&self) -> Result<(), GCodeGenerationError> {
        for (name, value) in [
            ("print_speed_mm_s", self.print_speed_mm_s),
            ("travel_speed_mm_s", self.travel_speed_mm_s),
            ("retract_speed_mm_s", self.retract_speed_mm_s),
            ("extrusion_width_mm", self.extrusion_width_mm),
            ("filament_diameter_mm", self.filament_diameter_mm),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(GCodeGenerationError::InvalidConfig(name));
            }
        }
        if !self.retract_distance_mm.is_finite() || self.retract_distance_mm < 0.0 {
            return Err(GCodeGenerationError::InvalidConfig("retract_distance_mm"));
        }
        if self.nozzle_temp_c == 0 {
            return Err(GCodeGenerationError::InvalidConfig("nozzle_temp_c"));
        }
        Ok(())
    }
}

// ── G-code commands ──────────────────────────────────────────────────────

/// A single G-code command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCodeCommand {
    /// Rapid move (non-extruding travel).
    G0 {
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
        f: Option<f32>,
    },
    /// Linear move (extruding or controlled motion).
    G1 {
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
        e: Option<f32>,
        f: Option<f32>,
    },
    /// Home all axes.
    G28,
    /// Set nozzle temperature (no wait).
    M104 { s: u16 },
    /// Set nozzle temperature and wait.
    M109 { s: u16 },
    /// Set bed temperature (no wait).
    M140 { s: u16 },
    /// Set bed temperature and wait.
    M190 { s: u16 },
    /// Comment line.
    Comment(String),
}

impl fmt::Display for GCodeCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GCodeCommand::G0 { x, y, z, f: feed } => {
                write!(f, "G0")?;
                if let Some(v) = x {
                    write!(f, " X{:.3}", v)?;
                }
                if let Some(v) = y {
                    write!(f, " Y{:.3}", v)?;
                }
                if let Some(v) = z {
                    write!(f, " Z{:.3}", v)?;
                }
                if let Some(v) = feed {
                    write!(f, " F{:.0}", v)?;
                }
                Ok(())
            }
            GCodeCommand::G1 {
                x,
                y,
                z,
                e,
                f: feed,
            } => {
                write!(f, "G1")?;
                if let Some(v) = x {
                    write!(f, " X{:.3}", v)?;
                }
                if let Some(v) = y {
                    write!(f, " Y{:.3}", v)?;
                }
                if let Some(v) = z {
                    write!(f, " Z{:.3}", v)?;
                }
                if let Some(v) = e {
                    write!(f, " E{:.5}", v)?;
                }
                if let Some(v) = feed {
                    write!(f, " F{:.0}", v)?;
                }
                Ok(())
            }
            GCodeCommand::G28 => write!(f, "G28"),
            GCodeCommand::M104 { s } => write!(f, "M104 S{}", s),
            GCodeCommand::M109 { s } => write!(f, "M109 S{}", s),
            GCodeCommand::M140 { s } => write!(f, "M140 S{}", s),
            GCodeCommand::M190 { s } => write!(f, "M190 S{}", s),
            GCodeCommand::Comment(text) => write!(f, "; {}", text),
        }
    }
}

// ── G-code program ───────────────────────────────────────────────────────

/// A complete G-code program: ordered commands plus accumulated extrusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCodeProgram {
    /// Ordered list of G-code commands.
    pub commands: Vec<GCodeCommand>,
    /// Total filament extruded in mm.
    pub total_extrusion_mm: f64,
}

impl GCodeProgram {
    /// Render the entire program as a newline-separated G-code string.
    pub fn to_gcode_string(&self) -> String {
        let mut out = String::with_capacity(self.commands.len() * 32);
        for cmd in &self.commands {
            out.push_str(&cmd.to_string());
            out.push('\n');
        }
        out
    }

    /// Number of commands in the program.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Alias for `len()`.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Whether the program contains no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl fmt::Display for GCodeProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_gcode_string())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Distance between two 2D points.
fn distance(a: &Point2, b: &Point2) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt()
}

/// Extrusion amount (mm of filament) for a given move length.
///
/// Based on volumetric equivalence:
///   cross_section_deposited = extrusion_width * layer_height
///   cross_section_filament  = PI/4 * filament_diameter^2
///   extrusion_per_mm = deposited / filament
fn extrusion_per_mm(config: &ToolpathConfig, layer_height: f32) -> f64 {
    let deposited = config.extrusion_width_mm as f64 * layer_height as f64;
    let filament_area = std::f64::consts::FRAC_PI_4 * (config.filament_diameter_mm as f64).powi(2);
    deposited / filament_area
}

/// Emit G-code to trace a contour with extrusion, updating cumulative E.
fn trace_contour(
    contour: &Contour,
    commands: &mut Vec<GCodeCommand>,
    cumulative_e: &mut f64,
    current_pos: &mut Option<Point2>,
    e_per_mm: f64,
    config: &ToolpathConfig,
    retracted: &mut bool,
) {
    if contour.points.len() < 2 {
        return;
    }

    let first = &contour.points[0];

    // Travel to the start of the contour if needed.
    let needs_travel = match current_pos {
        Some(pos) => distance(pos, first) > 1.0,
        None => true,
    };

    if needs_travel {
        // Retract before travel (if not already retracted).
        if !*retracted {
            *cumulative_e -= config.retract_distance_mm as f64;
            commands.push(GCodeCommand::G1 {
                x: None,
                y: None,
                z: None,
                e: Some(*cumulative_e as f32),
                f: Some(config.retract_speed_mm_s * 60.0),
            });
            *retracted = true;
        }

        // Rapid travel to start of contour.
        commands.push(GCodeCommand::G0 {
            x: Some(first.x),
            y: Some(first.y),
            z: None,
            f: Some(config.travel_speed_mm_s * 60.0),
        });

        // Unretract.
        *cumulative_e += config.retract_distance_mm as f64;
        commands.push(GCodeCommand::G1 {
            x: None,
            y: None,
            z: None,
            e: Some(*cumulative_e as f32),
            f: Some(config.retract_speed_mm_s * 60.0),
        });
        *retracted = false;
    }

    // Move to first point (if we didn't travel, we still need to be there).
    if !needs_travel {
        // Short move — just extrude to the first point.
        if let Some(pos) = current_pos {
            let d = distance(pos, first) as f64;
            if d > 1e-4 {
                *cumulative_e += d * e_per_mm;
                commands.push(GCodeCommand::G1 {
                    x: Some(first.x),
                    y: Some(first.y),
                    z: None,
                    e: Some(*cumulative_e as f32),
                    f: Some(config.print_speed_mm_s * 60.0),
                });
            }
        }
    }

    // Trace the remaining points.
    let mut prev = *first;
    for pt in &contour.points[1..] {
        let d = distance(&prev, pt) as f64;
        *cumulative_e += d * e_per_mm;
        commands.push(GCodeCommand::G1 {
            x: Some(pt.x),
            y: Some(pt.y),
            z: None,
            e: Some(*cumulative_e as f32),
            f: Some(config.print_speed_mm_s * 60.0),
        });
        prev = *pt;
    }

    // Close the contour: extrude back to the first point.
    let d = distance(&prev, first) as f64;
    if d > 1e-4 {
        *cumulative_e += d * e_per_mm;
        commands.push(GCodeCommand::G1 {
            x: Some(first.x),
            y: Some(first.y),
            z: None,
            e: Some(*cumulative_e as f32),
            f: Some(config.print_speed_mm_s * 60.0),
        });
    }

    *current_pos = Some(*first);
}

// ── Public API ───────────────────────────────────────────────────────────

/// Generate a complete G-code program from sliced layers.
///
/// # Arguments
/// * `layers` — Sliced layers (from `slice_mesh` or similar).
/// * `slice_config` — Slice configuration (provides `layer_height`).
/// * `config` — Toolpath/printer configuration.
pub fn try_generate_gcode(
    layers: &[SliceLayer],
    slice_config: &SliceConfig,
    config: &ToolpathConfig,
) -> Result<GCodeProgram, GCodeGenerationError> {
    slice_config
        .validate_strict()
        .map_err(GCodeGenerationError::InvalidSliceConfig)?;
    config.validate_strict()?;
    if layers.is_empty()
        || !layers.iter().any(|layer| {
            layer
                .outer_contours
                .iter()
                .any(|contour| contour.points.len() >= 2)
                || layer
                    .inner_contours
                    .iter()
                    .any(|contour| contour.points.len() >= 2)
                || layer
                    .infill_lines
                    .iter()
                    .any(|segment| segment.length() > 0.0)
        })
    {
        return Err(GCodeGenerationError::EmptyJob);
    }

    for (layer_index, layer) in layers.iter().enumerate() {
        if !layer.z.is_finite() || layer.z < 0.0 {
            return Err(GCodeGenerationError::InvalidLayer {
                layer_index,
                reason: "z must be finite and non-negative",
            });
        }
        let invalid_point = layer
            .outer_contours
            .iter()
            .chain(layer.inner_contours.iter())
            .flat_map(|contour| contour.points.iter())
            .chain(
                layer
                    .infill_lines
                    .iter()
                    .flat_map(|segment| [&segment.start, &segment.end]),
            )
            .any(|point| !point.x.is_finite() || !point.y.is_finite());
        if invalid_point {
            return Err(GCodeGenerationError::InvalidLayer {
                layer_index,
                reason: "coordinates must be finite",
            });
        }
    }

    Ok(generate_gcode(layers, slice_config, config))
}

pub fn generate_gcode(
    layers: &[SliceLayer],
    slice_config: &SliceConfig,
    config: &ToolpathConfig,
) -> GCodeProgram {
    let mut commands = Vec::new();
    let mut cumulative_e: f64 = 0.0;

    let e_per_mm = extrusion_per_mm(config, slice_config.layer_height);

    // ── Preamble ─────────────────────────────────────────────────────
    commands.push(GCodeCommand::Comment(
        "Generated by symthaea-fabrication-kernel".into(),
    ));
    commands.push(GCodeCommand::G28);
    commands.push(GCodeCommand::M190 {
        s: config.bed_temp_c,
    });
    commands.push(GCodeCommand::M109 {
        s: config.nozzle_temp_c,
    });
    commands.push(GCodeCommand::G1 {
        x: None,
        y: None,
        z: None,
        e: None,
        f: Some(config.print_speed_mm_s * 60.0),
    });

    // ── Layer loop ───────────────────────────────────────────────────
    let mut current_pos: Option<Point2> = None;
    let mut retracted = false;

    for (layer_idx, layer) in layers.iter().enumerate() {
        // Comment for layer start.
        commands.push(GCodeCommand::Comment(format!(
            "Layer {} z={:.3}",
            layer_idx, layer.z
        )));

        // Move to layer height.
        commands.push(GCodeCommand::G0 {
            x: None,
            y: None,
            z: Some(layer.z),
            f: None,
        });

        // Trace outer contours first (perimeters).
        for contour in &layer.outer_contours {
            trace_contour(
                contour,
                &mut commands,
                &mut cumulative_e,
                &mut current_pos,
                e_per_mm,
                config,
                &mut retracted,
            );
        }

        // Then inner contours (holes/infill boundaries).
        for contour in &layer.inner_contours {
            trace_contour(
                contour,
                &mut commands,
                &mut cumulative_e,
                &mut current_pos,
                e_per_mm,
                config,
                &mut retracted,
            );
        }

        // Then infill lines — sorted by nearest-neighbor to minimize travel.
        #[allow(unused_assignments)]
        if !layer.infill_lines.is_empty() {
            commands.push(GCodeCommand::Comment("Infill".into()));
            // Nearest-neighbor sort: greedily pick the closest segment start/end
            // to the current head position to reduce travel moves.
            let mut remaining: Vec<Segment2> = layer.infill_lines.clone();
            let mut ordered: Vec<Segment2> = Vec::with_capacity(remaining.len());
            let mut nn_pos: Option<Point2> = current_pos;
            while !remaining.is_empty() {
                let (best_idx, reversed) = match nn_pos {
                    Some(pos) => {
                        let mut best_i = 0usize;
                        let mut best_dist = f32::MAX;
                        let mut best_rev = false;
                        for (i, seg) in remaining.iter().enumerate() {
                            let ds = distance(&pos, &seg.start);
                            let de = distance(&pos, &seg.end);
                            if ds < best_dist {
                                best_dist = ds;
                                best_i = i;
                                best_rev = false;
                            }
                            if de < best_dist {
                                best_dist = de;
                                best_i = i;
                                best_rev = true;
                            }
                        }
                        (best_i, best_rev)
                    }
                    None => (0, false),
                };
                let seg = remaining.remove(best_idx);
                let seg = if reversed {
                    Segment2 {
                        start: seg.end,
                        end: seg.start,
                    }
                } else {
                    seg
                };
                nn_pos = Some(seg.end);
                ordered.push(seg);
            }
            for seg in &ordered {
                // Travel to start of infill segment.
                let needs_travel = match current_pos {
                    Some(pos) => distance(&pos, &seg.start) > 1.0,
                    None => true,
                };

                if needs_travel {
                    if !retracted {
                        cumulative_e -= config.retract_distance_mm as f64;
                        commands.push(GCodeCommand::G1 {
                            x: None,
                            y: None,
                            z: None,
                            e: Some(cumulative_e as f32),
                            f: Some(config.retract_speed_mm_s * 60.0),
                        });
                        retracted = true;
                    }
                    commands.push(GCodeCommand::G0 {
                        x: Some(seg.start.x),
                        y: Some(seg.start.y),
                        z: None,
                        f: Some(config.travel_speed_mm_s * 60.0),
                    });
                    cumulative_e += config.retract_distance_mm as f64;
                    commands.push(GCodeCommand::G1 {
                        x: None,
                        y: None,
                        z: None,
                        e: Some(cumulative_e as f32),
                        f: Some(config.retract_speed_mm_s * 60.0),
                    });
                    retracted = false;
                }

                // Extrude to end of infill segment.
                let d = distance(&seg.start, &seg.end) as f64;
                cumulative_e += d * e_per_mm;
                commands.push(GCodeCommand::G1 {
                    x: Some(seg.end.x),
                    y: Some(seg.end.y),
                    z: None,
                    e: Some(cumulative_e as f32),
                    f: Some(config.print_speed_mm_s * 60.0),
                });
                current_pos = Some(seg.end);
            }
        }
    }
    // Retraction state is loop-internal; silence clippy for the final write.
    let _ = retracted;

    // ── Postamble ────────────────────────────────────────────────────
    commands.push(GCodeCommand::Comment("Cooldown".into()));
    commands.push(GCodeCommand::M104 { s: 0 });
    commands.push(GCodeCommand::M140 { s: 0 });
    commands.push(GCodeCommand::G28);

    GCodeProgram {
        total_extrusion_mm: cumulative_e,
        commands,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slicer::{Segment2, SliceConfig};

    #[test]
    fn strict_gcode_rejects_empty_job() {
        assert!(matches!(
            try_generate_gcode(&[], &SliceConfig::default(), &ToolpathConfig::default()),
            Err(GCodeGenerationError::EmptyJob)
        ));

        let empty_layer = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![],
            inner_contours: vec![],
            infill_lines: vec![],
        }];
        assert!(matches!(
            try_generate_gcode(
                &empty_layer,
                &SliceConfig::default(),
                &ToolpathConfig::default(),
            ),
            Err(GCodeGenerationError::EmptyJob)
        ));
    }

    #[test]
    fn strict_gcode_rejects_zero_filament_diameter() {
        let config = ToolpathConfig {
            filament_diameter_mm: 0.0,
            ..Default::default()
        };
        let error = try_generate_gcode(&single_layer(), &SliceConfig::default(), &config)
            .err()
            .expect("invalid filament diameter must fail");
        assert_eq!(
            error,
            GCodeGenerationError::InvalidConfig("filament_diameter_mm")
        );
    }

    #[test]
    fn strict_gcode_rejects_non_finite_layer_coordinates() {
        let mut layers = single_layer();
        layers[0].outer_contours[0].points[0].x = f32::NAN;
        let error =
            try_generate_gcode(&layers, &SliceConfig::default(), &ToolpathConfig::default())
                .err()
                .expect("non-finite layer coordinates must fail");
        assert_eq!(
            error,
            GCodeGenerationError::InvalidLayer {
                layer_index: 0,
                reason: "coordinates must be finite",
            }
        );
    }

    fn square_contour() -> Contour {
        Contour {
            points: vec![
                Point2::new(0.0, 0.0),
                Point2::new(10.0, 0.0),
                Point2::new(10.0, 10.0),
                Point2::new(0.0, 10.0),
            ],
        }
    }

    fn single_layer() -> Vec<SliceLayer> {
        vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour()],
            inner_contours: vec![],
            infill_lines: vec![],
        }]
    }

    fn two_layers_with_travel() -> Vec<SliceLayer> {
        vec![
            SliceLayer {
                z: 0.2,
                outer_contours: vec![square_contour()],
                inner_contours: vec![],
                infill_lines: vec![],
            },
            SliceLayer {
                z: 0.4,
                outer_contours: vec![Contour {
                    points: vec![
                        Point2::new(50.0, 50.0),
                        Point2::new(60.0, 50.0),
                        Point2::new(60.0, 60.0),
                        Point2::new(50.0, 60.0),
                    ],
                }],
                inner_contours: vec![],
                infill_lines: vec![],
            },
        ]
    }

    fn default_configs() -> (SliceConfig, ToolpathConfig) {
        (SliceConfig::default(), ToolpathConfig::default())
    }

    fn has_command<F: Fn(&GCodeCommand) -> bool>(program: &GCodeProgram, pred: F) -> bool {
        program.commands.iter().any(pred)
    }

    fn gcode_string_contains(program: &GCodeProgram, needle: &str) -> bool {
        let s = program.to_gcode_string();
        s.contains(needle)
    }

    #[test]
    fn preamble_has_homing() {
        let (sc, tc) = default_configs();
        let program = generate_gcode(&[], &sc, &tc);
        assert!(
            has_command(&program, |c| matches!(c, GCodeCommand::G28)),
            "preamble must contain G28 homing command"
        );
    }

    #[test]
    fn preamble_has_temperatures() {
        let (sc, tc) = default_configs();
        let program = generate_gcode(&[], &sc, &tc);
        assert!(
            has_command(&program, |c| matches!(c, GCodeCommand::M190 { s: 60 })),
            "preamble must wait for bed temperature"
        );
        assert!(
            has_command(&program, |c| matches!(c, GCodeCommand::M109 { s: 210 })),
            "preamble must wait for nozzle temperature"
        );
    }

    #[test]
    fn empty_layers_still_has_preamble() {
        let (sc, tc) = default_configs();
        let program = generate_gcode(&[], &sc, &tc);
        // Preamble (comment, G28, M190, M109, G1 feed) + postamble (comment, M104, M140, G28)
        assert!(
            program.len() >= 5,
            "empty layers should still produce preamble + postamble, got {} commands",
            program.len()
        );
    }

    #[test]
    fn single_layer_has_z_move() {
        let (sc, tc) = default_configs();
        let layers = single_layer();
        let program = generate_gcode(&layers, &sc, &tc);
        assert!(
            has_command(
                &program,
                |c| matches!(c, GCodeCommand::G0 { z: Some(z), .. } if (*z - 0.2).abs() < 1e-4)
            ),
            "must have G0 Z move to layer height 0.2"
        );
    }

    #[test]
    fn extrusion_is_positive() {
        let (sc, tc) = default_configs();
        let layers = single_layer();
        let program = generate_gcode(&layers, &sc, &tc);
        assert!(
            program.total_extrusion_mm > 0.0,
            "total extrusion must be positive for a layer with a contour, got {}",
            program.total_extrusion_mm
        );
    }

    #[test]
    fn retraction_on_travel() {
        let (sc, tc) = default_configs();
        let layers = two_layers_with_travel();
        let program = generate_gcode(&layers, &sc, &tc);
        let gcode = program.to_gcode_string();
        // There should be a G0 travel move (rapid) to the distant contour on layer 2.
        let has_travel_g0 = program.commands.iter().any(|c| {
            matches!(c, GCodeCommand::G0 { x: Some(x), y: Some(y), .. }
                if *x > 40.0 && *y > 40.0)
        });
        assert!(
            has_travel_g0,
            "should have a G0 travel move to the distant contour on layer 2\n{}",
            gcode
        );
    }

    #[test]
    fn postamble_cools_down() {
        let (sc, tc) = default_configs();
        let program = generate_gcode(&[], &sc, &tc);
        assert!(
            has_command(&program, |c| matches!(c, GCodeCommand::M104 { s: 0 })),
            "postamble must turn off nozzle heater (M104 S0)"
        );
        assert!(
            has_command(&program, |c| matches!(c, GCodeCommand::M140 { s: 0 })),
            "postamble must turn off bed heater (M140 S0)"
        );
    }

    #[test]
    fn gcode_command_display_format() {
        let cmd = GCodeCommand::G0 {
            x: Some(10.0),
            y: Some(20.0),
            z: None,
            f: Some(7200.0),
        };
        let s = cmd.to_string();
        assert_eq!(s, "G0 X10.000 Y20.000 F7200");

        let cmd2 = GCodeCommand::G1 {
            x: Some(5.5),
            y: Some(3.2),
            z: None,
            e: Some(1.23456),
            f: None,
        };
        let s2 = cmd2.to_string();
        assert_eq!(s2, "G1 X5.500 Y3.200 E1.23456");

        assert_eq!(GCodeCommand::G28.to_string(), "G28");
        assert_eq!(GCodeCommand::M104 { s: 200 }.to_string(), "M104 S200");
        assert_eq!(GCodeCommand::M109 { s: 210 }.to_string(), "M109 S210");
        assert_eq!(GCodeCommand::M140 { s: 60 }.to_string(), "M140 S60");
        assert_eq!(GCodeCommand::M190 { s: 60 }.to_string(), "M190 S60");
        assert_eq!(GCodeCommand::Comment("hello".into()).to_string(), "; hello");
    }

    #[test]
    fn extrusion_per_mm_sanity() {
        let config = ToolpathConfig::default();
        let e = extrusion_per_mm(&config, 0.2);
        // 0.4 * 0.2 / (PI/4 * 1.75^2) = 0.08 / 2.405 ≈ 0.0333
        assert!(e > 0.01 && e < 0.1, "extrusion_per_mm={} out of range", e);
    }

    #[test]
    fn gcode_program_display() {
        let (sc, tc) = default_configs();
        let program = generate_gcode(&[], &sc, &tc);
        let display = format!("{}", program);
        assert!(display.contains("G28"), "Display output must contain G28");
    }

    #[test]
    fn gcode_with_infill() {
        let (sc, tc) = default_configs();
        let layers = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour()],
            inner_contours: vec![],
            infill_lines: vec![
                Segment2 {
                    start: Point2::new(1.0, 2.0),
                    end: Point2::new(9.0, 2.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 5.0),
                    end: Point2::new(9.0, 5.0),
                },
            ],
        }];
        let program = generate_gcode(&layers, &sc, &tc);
        // Should have the infill comment.
        assert!(
            gcode_string_contains(&program, "Infill"),
            "should contain Infill comment"
        );
        // Infill should add extrusion beyond just the contour.
        let contour_only = generate_gcode(&single_layer(), &sc, &tc);
        assert!(
            program.total_extrusion_mm > contour_only.total_extrusion_mm,
            "infill should add extrusion: {} vs {}",
            program.total_extrusion_mm,
            contour_only.total_extrusion_mm
        );
    }

    #[test]
    fn multiple_contours_per_layer() {
        let (sc, tc) = default_configs();
        let layers = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour()],
            inner_contours: vec![Contour {
                points: vec![
                    Point2::new(3.0, 3.0),
                    Point2::new(7.0, 3.0),
                    Point2::new(7.0, 7.0),
                    Point2::new(3.0, 7.0),
                ],
            }],
            infill_lines: vec![],
        }];
        let program = generate_gcode(&layers, &sc, &tc);
        // Both contours should produce extrusion moves.
        let g1_count = program
            .commands
            .iter()
            .filter(|c| matches!(c, GCodeCommand::G1 { e: Some(_), .. }))
            .count();
        // At minimum: outer (4 edges + close) + inner (4 edges + close) = ~10 G1 moves
        assert!(
            g1_count >= 8,
            "multiple contours should produce many G1 moves, got {}",
            g1_count
        );
    }

    #[test]
    fn nearest_neighbor_reduces_travel() {
        let (sc, tc) = default_configs();
        // Create infill segments in worst-case (far-apart alternating) order.
        // Segments at Y=1,3,5,7,9 but listed in non-sequential order to maximize travel.
        let worst_order_layers = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour()],
            inner_contours: vec![],
            infill_lines: vec![
                // Far end first, then near, alternating — maximizes naive travel
                Segment2 {
                    start: Point2::new(1.0, 9.0),
                    end: Point2::new(9.0, 9.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 1.0),
                    end: Point2::new(9.0, 1.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 8.0),
                    end: Point2::new(9.0, 8.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 2.0),
                    end: Point2::new(9.0, 2.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 7.0),
                    end: Point2::new(9.0, 7.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 3.0),
                    end: Point2::new(9.0, 3.0),
                },
            ],
        }];
        // Sequential (already-optimal) order for comparison.
        let sequential_layers = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![square_contour()],
            inner_contours: vec![],
            infill_lines: vec![
                Segment2 {
                    start: Point2::new(1.0, 1.0),
                    end: Point2::new(9.0, 1.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 2.0),
                    end: Point2::new(9.0, 2.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 3.0),
                    end: Point2::new(9.0, 3.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 7.0),
                    end: Point2::new(9.0, 7.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 8.0),
                    end: Point2::new(9.0, 8.0),
                },
                Segment2 {
                    start: Point2::new(1.0, 9.0),
                    end: Point2::new(9.0, 9.0),
                },
            ],
        }];

        let worst_prog = generate_gcode(&worst_order_layers, &sc, &tc);
        let seq_prog = generate_gcode(&sequential_layers, &sc, &tc);

        // Count G0 travel moves in infill section (after "Infill" comment).
        fn count_infill_travel(prog: &GCodeProgram) -> usize {
            let mut in_infill = false;
            let mut count = 0usize;
            for cmd in &prog.commands {
                if let GCodeCommand::Comment(s) = cmd {
                    if s == "Infill" {
                        in_infill = true;
                    }
                }
                if in_infill {
                    if let GCodeCommand::G0 {
                        x: Some(_),
                        y: Some(_),
                        ..
                    } = cmd
                    {
                        count += 1;
                    }
                }
            }
            count
        }

        let nn_travel = count_infill_travel(&worst_prog);
        let seq_travel = count_infill_travel(&seq_prog);
        // Nearest-neighbor reordering should produce travel count <= the sequential (already-ordered) case.
        assert!(
            nn_travel <= seq_travel + 1,
            "nearest-neighbor ({} travels) should be close to sequential ({} travels)",
            nn_travel,
            seq_travel
        );
    }
}
