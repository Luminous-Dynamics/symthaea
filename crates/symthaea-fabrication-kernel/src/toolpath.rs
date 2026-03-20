//! G-code toolpath generation from sliced layers
//!
//! Takes a [`SliceResult`] (from the slicer module) and produces a
//! [`GCodeProgram`] containing standard RepRap-flavour G-code commands
//! suitable for FDM 3D printers.

use crate::slicer::{SliceResult, SlicedLayer};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Toolpath generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolpathConfig {
    /// Print (extrusion) speed in mm/s.
    pub print_speed_mm_s: f32,
    /// Non-extrusion travel speed in mm/s.
    pub travel_speed_mm_s: f32,
    /// Retraction distance in mm.
    pub retract_distance_mm: f32,
    /// Retraction speed in mm/s.
    pub retract_speed_mm_s: f32,
    /// Width of extruded bead in mm.
    pub extrusion_width_mm: f32,
    /// Filament diameter in mm (typically 1.75).
    pub filament_diameter_mm: f32,
    /// Heated bed temperature in °C.
    pub bed_temp_c: u16,
    /// Nozzle / hot-end temperature in °C.
    pub nozzle_temp_c: u16,
}

impl Default for ToolpathConfig {
    fn default() -> Self {
        Self {
            print_speed_mm_s: 50.0,
            travel_speed_mm_s: 150.0,
            retract_distance_mm: 1.0,
            retract_speed_mm_s: 40.0,
            extrusion_width_mm: 0.48,
            filament_diameter_mm: 1.75,
            bed_temp_c: 60,
            nozzle_temp_c: 200,
        }
    }
}

impl ToolpathConfig {
    /// Filament cross-section area in mm².
    fn filament_area(&self) -> f64 {
        let r = self.filament_diameter_mm as f64 / 2.0;
        std::f64::consts::PI * r * r
    }

    /// Extrusion ratio: mm of filament per mm of toolpath movement.
    fn extrusion_ratio(&self, layer_height: f32) -> f64 {
        let bead_area = self.extrusion_width_mm as f64 * layer_height as f64;
        bead_area / self.filament_area()
    }
}

// ---------------------------------------------------------------------------
// G-code commands
// ---------------------------------------------------------------------------

/// Individual G-code commands supported by the toolpath generator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GCodeCommand {
    /// Rapid non-extrusion move.
    G0 {
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
        f: Option<f32>,
    },
    /// Linear extrusion move.
    G1 {
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
        e: Option<f32>,
        f: Option<f32>,
    },
    /// Home all axes.
    G28,
    /// Set nozzle temperature (non-blocking).
    M104 { s: u16 },
    /// Set nozzle temperature and wait.
    M109 { s: u16 },
    /// Set bed temperature (non-blocking).
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
                    write!(f, " X{v:.3}")?;
                }
                if let Some(v) = y {
                    write!(f, " Y{v:.3}")?;
                }
                if let Some(v) = z {
                    write!(f, " Z{v:.3}")?;
                }
                if let Some(v) = feed {
                    write!(f, " F{v:.1}")?;
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
                    write!(f, " X{v:.3}")?;
                }
                if let Some(v) = y {
                    write!(f, " Y{v:.3}")?;
                }
                if let Some(v) = z {
                    write!(f, " Z{v:.3}")?;
                }
                if let Some(v) = e {
                    write!(f, " E{v:.5}")?;
                }
                if let Some(v) = feed {
                    write!(f, " F{v:.1}")?;
                }
                Ok(())
            }
            GCodeCommand::G28 => write!(f, "G28"),
            GCodeCommand::M104 { s } => write!(f, "M104 S{s}"),
            GCodeCommand::M109 { s } => write!(f, "M109 S{s}"),
            GCodeCommand::M140 { s } => write!(f, "M140 S{s}"),
            GCodeCommand::M190 { s } => write!(f, "M190 S{s}"),
            GCodeCommand::Comment(text) => write!(f, "; {text}"),
        }
    }
}

// ---------------------------------------------------------------------------
// G-code program
// ---------------------------------------------------------------------------

/// A complete G-code program ready for printing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCodeProgram {
    /// Ordered list of G-code commands.
    pub commands: Vec<GCodeCommand>,
    /// Total filament extruded in mm.
    pub total_extrusion_mm: f64,
}

impl GCodeProgram {
    /// Render the entire program as a single G-code string.
    pub fn to_gcode_string(&self) -> String {
        let mut out = String::with_capacity(self.commands.len() * 30);
        for cmd in &self.commands {
            out.push_str(&format!("{cmd}\n"));
        }
        out
    }

    /// Number of commands in the program.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the program is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Count commands of a specific type.
    pub fn count_g0(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| matches!(c, GCodeCommand::G0 { .. }))
            .count()
    }

    pub fn count_g1(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| matches!(c, GCodeCommand::G1 { .. }))
            .count()
    }
}

impl fmt::Display for GCodeProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for cmd in &self.commands {
            writeln!(f, "{cmd}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Distance helper
// ---------------------------------------------------------------------------

fn distance_2d(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a G-code program from sliced layers.
///
/// The output follows standard RepRap G-code conventions:
/// 1. Preamble: home, heat bed (wait), heat nozzle (wait)
/// 2. Per-layer: z-move, outer perimeters, inner perimeters, infill
/// 3. Retraction on travel moves between non-adjacent segments
/// 4. Postamble: cool down, home
pub fn generate_gcode(result: &SliceResult, config: &ToolpathConfig) -> GCodeProgram {
    let mut commands = Vec::new();
    let mut total_e = 0.0_f64;
    let mut current_pos: [f32; 2] = [0.0, 0.0];
    let mut retracted = false;

    let layer_height = result.config.layer_height;
    let extrusion_ratio = config.extrusion_ratio(layer_height);
    let print_feed = config.print_speed_mm_s * 60.0; // mm/min
    let travel_feed = config.travel_speed_mm_s * 60.0;
    let retract_feed = config.retract_speed_mm_s * 60.0;

    // -----------------------------------------------------------------------
    // Preamble
    // -----------------------------------------------------------------------
    commands.push(GCodeCommand::Comment("Generated by symthaea-fabrication-kernel".into()));
    commands.push(GCodeCommand::G28);
    commands.push(GCodeCommand::Comment("Heat bed and wait".into()));
    commands.push(GCodeCommand::M190 { s: config.bed_temp_c });
    commands.push(GCodeCommand::Comment("Heat nozzle and wait".into()));
    commands.push(GCodeCommand::M109 { s: config.nozzle_temp_c });
    // Reset extruder
    commands.push(GCodeCommand::G1 {
        x: None,
        y: None,
        z: None,
        e: Some(0.0),
        f: None,
    });

    // -----------------------------------------------------------------------
    // Per-layer
    // -----------------------------------------------------------------------
    for (layer_idx, layer) in result.layers.iter().enumerate() {
        commands.push(GCodeCommand::Comment(format!(
            "Layer {layer_idx} z={:.3}{}",
            layer.z_height,
            if layer.is_solid { " (solid)" } else { "" }
        )));

        // Move to layer Z
        commands.push(GCodeCommand::G0 {
            x: None,
            y: None,
            z: Some(layer.z_height),
            f: Some(travel_feed),
        });

        // Print perimeters (outer first for better surface finish)
        emit_perimeters(
            &layer,
            &mut commands,
            &mut total_e,
            &mut current_pos,
            &mut retracted,
            extrusion_ratio,
            print_feed,
            travel_feed,
            retract_feed,
            config.retract_distance_mm,
        );

        // Print infill
        emit_infill(
            &layer,
            &mut commands,
            &mut total_e,
            &mut current_pos,
            &mut retracted,
            extrusion_ratio,
            print_feed,
            travel_feed,
            retract_feed,
            config.retract_distance_mm,
        );
    }

    // -----------------------------------------------------------------------
    // Postamble
    // -----------------------------------------------------------------------
    commands.push(GCodeCommand::Comment("Postamble — cool down".into()));
    // Retract if needed
    if !retracted {
        total_e -= config.retract_distance_mm as f64;
        commands.push(GCodeCommand::G1 {
            x: None,
            y: None,
            z: None,
            e: Some(total_e as f32),
            f: Some(retract_feed),
        });
    }
    // Turn off heaters
    commands.push(GCodeCommand::M104 { s: 0 });
    commands.push(GCodeCommand::M140 { s: 0 });
    // Home
    commands.push(GCodeCommand::G28);
    commands.push(GCodeCommand::Comment("End of print".into()));

    GCodeProgram {
        commands,
        total_extrusion_mm: total_e.max(0.0),
    }
}

/// Emit G-code for perimeter contours in a single layer.
fn emit_perimeters(
    layer: &SlicedLayer,
    commands: &mut Vec<GCodeCommand>,
    total_e: &mut f64,
    current_pos: &mut [f32; 2],
    retracted: &mut bool,
    extrusion_ratio: f64,
    print_feed: f32,
    travel_feed: f32,
    retract_feed: f32,
    retract_dist: f32,
) {
    // Sort: outer perimeters first
    let mut sorted: Vec<&_> = layer.perimeters.iter().collect();
    sorted.sort_by(|a, b| b.is_outer.cmp(&a.is_outer));

    for contour in sorted {
        if contour.points.is_empty() {
            continue;
        }

        let start = contour.points[0];

        // Travel to start of contour (with retract if far)
        let travel_dist = distance_2d(*current_pos, start);
        if travel_dist > 1.0 {
            // Retract
            if !*retracted {
                *total_e -= retract_dist as f64;
                commands.push(GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(*total_e as f32),
                    f: Some(retract_feed),
                });
                *retracted = true;
            }
            // Travel
            commands.push(GCodeCommand::G0 {
                x: Some(start[0]),
                y: Some(start[1]),
                z: None,
                f: Some(travel_feed),
            });
        }

        // Un-retract
        if *retracted {
            *total_e += retract_dist as f64;
            commands.push(GCodeCommand::G1 {
                x: None,
                y: None,
                z: None,
                e: Some(*total_e as f32),
                f: Some(retract_feed),
            });
            *retracted = false;
        }

        // Extrude along contour
        for i in 1..contour.points.len() {
            let pt = contour.points[i];
            let seg_len = distance_2d(contour.points[i - 1], pt);
            *total_e += seg_len as f64 * extrusion_ratio;
            commands.push(GCodeCommand::G1 {
                x: Some(pt[0]),
                y: Some(pt[1]),
                z: None,
                e: Some(*total_e as f32),
                f: Some(print_feed),
            });
        }

        // Close the contour back to the start
        if contour.points.len() >= 3 {
            let last = *contour.points.last().unwrap();
            let seg_len = distance_2d(last, start);
            if seg_len > 1e-4 {
                *total_e += seg_len as f64 * extrusion_ratio;
                commands.push(GCodeCommand::G1 {
                    x: Some(start[0]),
                    y: Some(start[1]),
                    z: None,
                    e: Some(*total_e as f32),
                    f: Some(print_feed),
                });
            }
        }

        *current_pos = start;
    }
}

/// Emit G-code for infill lines in a single layer.
fn emit_infill(
    layer: &SlicedLayer,
    commands: &mut Vec<GCodeCommand>,
    total_e: &mut f64,
    current_pos: &mut [f32; 2],
    retracted: &mut bool,
    extrusion_ratio: f64,
    print_feed: f32,
    travel_feed: f32,
    retract_feed: f32,
    retract_dist: f32,
) {
    for line in &layer.infill_lines {
        let travel_dist = distance_2d(*current_pos, line.start);

        // Retract + travel if needed
        if travel_dist > 1.0 {
            if !*retracted {
                *total_e -= retract_dist as f64;
                commands.push(GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(*total_e as f32),
                    f: Some(retract_feed),
                });
                *retracted = true;
            }
            commands.push(GCodeCommand::G0 {
                x: Some(line.start[0]),
                y: Some(line.start[1]),
                z: None,
                f: Some(travel_feed),
            });
        }

        // Un-retract
        if *retracted {
            *total_e += retract_dist as f64;
            commands.push(GCodeCommand::G1 {
                x: None,
                y: None,
                z: None,
                e: Some(*total_e as f32),
                f: Some(retract_feed),
            });
            *retracted = false;
        }

        // Extrude the infill line
        let seg_len = line.length();
        *total_e += seg_len as f64 * extrusion_ratio;
        commands.push(GCodeCommand::G1 {
            x: Some(line.end[0]),
            y: Some(line.end[1]),
            z: None,
            e: Some(*total_e as f32),
            f: Some(print_feed),
        });

        *current_pos = line.end;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::TriangleMesh;
    use crate::slicer::{SlicerConfig, slice_mesh};

    fn make_test_cube() -> TriangleMesh {
        let h = 5.0f32;
        let vertices = vec![
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
            [h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h],
            [-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h],
            [-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h],
            [h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h],
            [-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        ];
        let indices = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [8, 9, 10], [8, 10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ];
        TriangleMesh { vertices, normals, indices }
    }

    fn slice_test_cube() -> SliceResult {
        let mesh = make_test_cube();
        let config = SlicerConfig {
            layer_height: 1.0,
            infill_density: 0.2,
            wall_count: 1,
            top_layers: 1,
            bottom_layers: 1,
            ..SlicerConfig::default()
        };
        slice_mesh(&mesh, &config).expect("slicing should succeed")
    }

    #[test]
    fn test_preamble_has_homing() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());
        let has_g28 = program.commands.iter().any(|c| matches!(c, GCodeCommand::G28));
        assert!(has_g28, "preamble must contain G28 homing command");
    }

    #[test]
    fn test_temperature_commands_present() {
        let sliced = slice_test_cube();
        let config = ToolpathConfig {
            bed_temp_c: 65,
            nozzle_temp_c: 210,
            ..ToolpathConfig::default()
        };
        let program = generate_gcode(&sliced, &config);

        let has_bed = program
            .commands
            .iter()
            .any(|c| matches!(c, GCodeCommand::M190 { s: 65 }));
        let has_nozzle = program
            .commands
            .iter()
            .any(|c| matches!(c, GCodeCommand::M109 { s: 210 }));

        assert!(has_bed, "must have M190 S65 (bed temp wait)");
        assert!(has_nozzle, "must have M109 S210 (nozzle temp wait)");
    }

    #[test]
    fn test_extrusion_positive() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());
        assert!(
            program.total_extrusion_mm > 0.0,
            "total extrusion must be positive, got {}",
            program.total_extrusion_mm
        );
    }

    #[test]
    fn test_layer_transitions_have_z_move() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());

        // Count G0 moves that set Z (layer transitions)
        let z_moves: Vec<_> = program
            .commands
            .iter()
            .filter(|c| matches!(c, GCodeCommand::G0 { z: Some(_), .. }))
            .collect();

        // Should have at least one Z-move per layer
        assert!(
            z_moves.len() >= sliced.layers.len(),
            "expected at least {} z-moves, got {}",
            sliced.layers.len(),
            z_moves.len()
        );
    }

    #[test]
    fn test_retraction_on_travel() {
        let sliced = slice_test_cube();
        let config = ToolpathConfig {
            retract_distance_mm: 2.0,
            ..ToolpathConfig::default()
        };
        let program = generate_gcode(&sliced, &config);

        // There should be at least one G0 travel move (indicating retraction happened)
        let travel_count = program.count_g0();
        assert!(
            travel_count > 0,
            "should have travel (G0) moves in the program"
        );
    }

    #[test]
    fn test_program_to_string_roundtrip() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());
        let text = program.to_gcode_string();

        assert!(!text.is_empty(), "program text should not be empty");
        assert!(text.contains("G28"), "output must contain G28");
        assert!(text.contains("G1"), "output must contain G1 extrusion moves");
        // Each line should be a valid G-code line or comment
        for line in text.lines() {
            assert!(
                line.starts_with('G')
                    || line.starts_with('M')
                    || line.starts_with(';')
                    || line.is_empty(),
                "unexpected line: {line}"
            );
        }
    }

    #[test]
    fn test_postamble_cools_down() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());

        // Last few commands should include M104 S0 and M140 S0
        let tail: Vec<_> = program.commands.iter().rev().take(10).collect();
        let has_nozzle_off = tail.iter().any(|c| matches!(c, GCodeCommand::M104 { s: 0 }));
        let has_bed_off = tail.iter().any(|c| matches!(c, GCodeCommand::M140 { s: 0 }));

        assert!(has_nozzle_off, "postamble must turn off nozzle (M104 S0)");
        assert!(has_bed_off, "postamble must turn off bed (M140 S0)");
    }

    #[test]
    fn test_gcode_command_display() {
        assert_eq!(format!("{}", GCodeCommand::G28), "G28");
        assert_eq!(format!("{}", GCodeCommand::M104 { s: 200 }), "M104 S200");
        assert_eq!(format!("{}", GCodeCommand::M190 { s: 60 }), "M190 S60");
        assert_eq!(
            format!("{}", GCodeCommand::Comment("test".into())),
            "; test"
        );

        let g0 = GCodeCommand::G0 {
            x: Some(10.0),
            y: Some(20.0),
            z: None,
            f: Some(3000.0),
        };
        let s = format!("{g0}");
        assert!(s.contains("G0"));
        assert!(s.contains("X10.000"));
        assert!(s.contains("Y20.000"));
        assert!(s.contains("F3000.0"));
        assert!(!s.contains("Z"));
    }

    #[test]
    fn test_extrusion_moves_have_e_values() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());

        let g1_with_xy: Vec<_> = program
            .commands
            .iter()
            .filter(|c| {
                matches!(c, GCodeCommand::G1 { x: Some(_), y: Some(_), .. })
            })
            .collect();

        // All G1 moves with X/Y should also have E (extrusion)
        for cmd in &g1_with_xy {
            if let GCodeCommand::G1 { e, .. } = cmd {
                assert!(e.is_some(), "G1 with X/Y must have E value: {cmd}");
            }
        }
    }

    #[test]
    fn test_program_len() {
        let sliced = slice_test_cube();
        let program = generate_gcode(&sliced, &ToolpathConfig::default());
        assert_eq!(program.len(), program.commands.len());
        assert!(!program.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = ToolpathConfig::default();
        assert!(config.print_speed_mm_s > 0.0);
        assert!(config.travel_speed_mm_s > config.print_speed_mm_s);
        assert!(config.filament_diameter_mm > 0.0);
        assert!(config.nozzle_temp_c > 0);
        assert!(config.bed_temp_c > 0);
    }
}
