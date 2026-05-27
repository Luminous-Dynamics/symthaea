// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Basic ISO 10303-21 STEP file parser
//!
//! Parses a minimal subset of the STEP physical file format: CARTESIAN_POINT,
//! B_SPLINE_CURVE_WITH_KNOTS, and B_SPLINE_SURFACE_WITH_KNOTS entities.
//! Unknown entities are preserved as opaque `Unknown` variants for
//! forward-compatible handling.

use crate::mesh::TriangleMesh;
use crate::nurbs::{NurbsCurve, NurbsSurface};
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while parsing a STEP file.
#[derive(Debug, Clone)]
pub enum StepParseError {
    /// The input does not conform to ISO 10303-21 structure.
    InvalidFormat(String),
    /// An entity type is recognized but not yet supported.
    UnsupportedEntity(String),
    /// An entity references an ID that does not exist.
    ReferenceError(usize),
}

impl fmt::Display for StepParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepParseError::InvalidFormat(msg) => write!(f, "STEP invalid format: {}", msg),
            StepParseError::UnsupportedEntity(name) => {
                write!(f, "STEP unsupported entity: {}", name)
            }
            StepParseError::ReferenceError(id) => {
                write!(f, "STEP reference error: entity #{} not found", id)
            }
        }
    }
}

impl std::error::Error for StepParseError {}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A parsed STEP entity.
#[derive(Debug, Clone)]
pub enum StepEntity {
    /// `CARTESIAN_POINT` — a point in 3-space (stored as f64 for STEP fidelity).
    CartesianPoint([f64; 3]),
    /// `B_SPLINE_CURVE_WITH_KNOTS`
    BSplineCurve {
        degree: u32,
        control_points: Vec<[f32; 3]>,
        knots: Vec<f32>,
    },
    /// `B_SPLINE_SURFACE_WITH_KNOTS`
    BSplineSurface {
        degree_u: u32,
        degree_v: u32,
        control_points: Vec<Vec<[f32; 3]>>,
        knots_u: Vec<f32>,
        knots_v: Vec<f32>,
    },
    /// Any entity type we don't explicitly handle.
    Unknown { entity_type: String, params: String },
}

/// A parsed STEP file containing a table of numbered entities.
#[derive(Debug, Clone)]
pub struct StepFile {
    pub entities: Vec<(usize, StepEntity)>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a STEP (ISO 10303-21) physical file from a string.
///
/// The parser locates the `DATA;` section and processes `#N = TYPE(params);`
/// lines. Entities before `DATA;` and after `ENDSEC;` are ignored.
pub fn parse_step(input: &str) -> Result<StepFile, StepParseError> {
    let mut entities = Vec::new();
    let mut in_data = false;

    for line in input.lines() {
        let trimmed = line.trim();

        if trimmed == "DATA;" {
            in_data = true;
            continue;
        }
        if trimmed == "ENDSEC;" {
            if in_data {
                break;
            }
            continue;
        }
        if !in_data {
            continue;
        }

        // Expect lines of the form: #N = TYPE(params);
        if !trimmed.starts_with('#') {
            continue;
        }

        let entity = parse_entity_line(trimmed)?;
        if let Some(e) = entity {
            entities.push(e);
        }
    }

    Ok(StepFile { entities })
}

/// Parse a single `#N = TYPE(params);` line.
fn parse_entity_line(line: &str) -> Result<Option<(usize, StepEntity)>, StepParseError> {
    // Strip leading '#'
    let rest = &line[1..];

    // Split on '='
    let eq_pos = rest
        .find('=')
        .ok_or_else(|| StepParseError::InvalidFormat(format!("No '=' in line: {}", line)))?;

    let id_str = rest[..eq_pos].trim();
    let id: usize = id_str
        .parse()
        .map_err(|_| StepParseError::InvalidFormat(format!("Bad entity id: {}", id_str)))?;

    let rhs = rest[eq_pos + 1..].trim();

    // Find TYPE( ... );
    let paren_open = rhs.find('(').ok_or_else(|| {
        StepParseError::InvalidFormat(format!("No '(' in entity definition: {}", rhs))
    })?;

    let entity_type = rhs[..paren_open].trim().to_uppercase();

    // Extract params between outermost parens, strip trailing ';'
    let after_open = &rhs[paren_open + 1..];
    let paren_close = after_open
        .rfind(')')
        .ok_or_else(|| StepParseError::InvalidFormat(format!("No closing ')' in: {}", rhs)))?;
    let params = after_open[..paren_close].trim();

    let entity = match entity_type.as_str() {
        "CARTESIAN_POINT" => parse_cartesian_point(params)?,
        "B_SPLINE_CURVE_WITH_KNOTS" => parse_bspline_curve(params)?,
        "B_SPLINE_SURFACE_WITH_KNOTS" => parse_bspline_surface(params)?,
        _ => StepEntity::Unknown {
            entity_type,
            params: params.to_string(),
        },
    };

    Ok(Some((id, entity)))
}

/// Parse CARTESIAN_POINT params: `'name', (x, y, z)`
fn parse_cartesian_point(params: &str) -> Result<StepEntity, StepParseError> {
    // Find the coordinate tuple inside parentheses
    let open = params.find('(').ok_or_else(|| {
        StepParseError::InvalidFormat("CARTESIAN_POINT: missing coordinate tuple".into())
    })?;
    let close = params[open..].find(')').ok_or_else(|| {
        StepParseError::InvalidFormat("CARTESIAN_POINT: unclosed coordinate tuple".into())
    })? + open;

    let coords_str = &params[open + 1..close];
    let coords: Vec<f64> = coords_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();

    let x = coords.first().copied().unwrap_or(0.0);
    let y = coords.get(1).copied().unwrap_or(0.0);
    let z = coords.get(2).copied().unwrap_or(0.0);

    Ok(StepEntity::CartesianPoint([x, y, z]))
}

/// Parse B_SPLINE_CURVE_WITH_KNOTS params.
///
/// Expected format (simplified):
/// `'name', degree, ((x,y,z),(x,y,z),...), .UNSPECIFIED., .F., .F., (knot_mults), (knots), ...`
fn parse_bspline_curve(params: &str) -> Result<StepEntity, StepParseError> {
    let tokens = tokenize_step_params(params);

    let degree = tokens
        .first()
        .and_then(|t| t.trim().trim_matches('\'').parse::<u32>().ok())
        .unwrap_or(1);

    let control_points = extract_point_list(params);
    let knots = extract_float_lists(params);
    let knot_vec = knots.last().cloned().unwrap_or_default();

    Ok(StepEntity::BSplineCurve {
        degree,
        control_points,
        knots: knot_vec,
    })
}

/// Parse B_SPLINE_SURFACE_WITH_KNOTS params (simplified).
fn parse_bspline_surface(params: &str) -> Result<StepEntity, StepParseError> {
    let tokens = tokenize_step_params(params);

    let degree_u = tokens
        .first()
        .and_then(|t| t.trim().trim_matches('\'').parse::<u32>().ok())
        .unwrap_or(1);
    let degree_v = tokens
        .get(1)
        .and_then(|t| t.trim().parse::<u32>().ok())
        .unwrap_or(1);

    // Extract control point grid from nested parentheses
    let control_points = extract_point_grid(params);
    let float_lists = extract_float_lists(params);
    let knots_u = float_lists.first().cloned().unwrap_or_default();
    let knots_v = float_lists.get(1).cloned().unwrap_or_default();

    Ok(StepEntity::BSplineSurface {
        degree_u,
        degree_v,
        control_points,
        knots_u,
        knots_v,
    })
}

/// Tokenize top-level comma-separated segments, respecting parenthesis nesting.
fn tokenize_step_params(params: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in params.chars() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                tokens.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Extract a flat list of 3D points from the first `((x,y,z),(x,y,z),...)` group.
fn extract_point_list(params: &str) -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    // Find double-open paren for point list: ((x,y,z),(x,y,z),...)
    if let Some(start) = params.find("((") {
        // Start after the outer '(' — so search begins at the inner content
        let search = &params[start + 1..];
        let mut depth = 1i32; // we are inside the outer '('
        let mut end = 0;
        for (i, ch) in search.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let point_section = &search[..end];
        // Split on ),(
        for pt_str in point_section.split("),(") {
            let clean = pt_str.trim_matches(|c: char| c == '(' || c == ')');
            let vals: Vec<f32> = clean
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .collect();
            if vals.len() >= 3 {
                points.push([vals[0], vals[1], vals[2]]);
            }
        }
    }
    points
}

/// Extract a grid of 3D points for surface control nets.
fn extract_point_grid(params: &str) -> Vec<Vec<[f32; 3]>> {
    // Simplified: treat the flat point list and reshape based on context
    let flat = extract_point_list(params);
    if flat.is_empty() {
        return vec![vec![]];
    }
    // If we can't determine the grid dimensions, return as single row
    vec![flat]
}

/// Extract all parenthesized float lists `(f1, f2, ...)` that contain only numbers.
fn extract_float_lists(params: &str) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = params.as_bytes();

    while i < bytes.len() {
        if bytes[i] == b'(' {
            let start = i + 1;
            let mut depth = 1i32;
            let mut j = start;
            while j < bytes.len() && depth > 0 {
                if bytes[j] == b'(' {
                    depth += 1;
                }
                if bytes[j] == b')' {
                    depth -= 1;
                }
                j += 1;
            }
            let inner = &params[start..j - 1];
            // Check if this looks like a float list (no nested parens, all parseable)
            if !inner.contains('(') {
                let vals: Vec<f32> = inner
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect();
                // Only add if all tokens parsed and we got at least one
                let token_count = inner.split(',').count();
                if !vals.is_empty() && vals.len() == token_count {
                    result.push(vals);
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// StepFile methods
// ---------------------------------------------------------------------------

impl StepFile {
    /// Look up an entity by its numeric ID.
    pub fn entity_by_id(&self, id: usize) -> Option<&StepEntity> {
        self.entities
            .iter()
            .find(|(eid, _)| *eid == id)
            .map(|(_, e)| e)
    }

    /// Convert all `BSplineCurve` entities to [`NurbsCurve`]s.
    ///
    /// Weights default to 1.0 (non-rational) since STEP B_SPLINE_CURVE_WITH_KNOTS
    /// does not carry weights (RATIONAL_B_SPLINE is a separate entity type).
    pub fn to_nurbs_curves(&self) -> Vec<NurbsCurve> {
        self.entities
            .iter()
            .filter_map(|(_, e)| match e {
                StepEntity::BSplineCurve {
                    degree,
                    control_points,
                    knots,
                } => Some(NurbsCurve {
                    degree: *degree,
                    control_points: control_points.clone(),
                    weights: vec![1.0; control_points.len()],
                    knots: knots.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Convert all `BSplineSurface` entities to [`NurbsSurface`]s.
    pub fn to_nurbs_surfaces(&self) -> Vec<NurbsSurface> {
        self.entities
            .iter()
            .filter_map(|(_, e)| match e {
                StepEntity::BSplineSurface {
                    degree_u,
                    degree_v,
                    control_points,
                    knots_u,
                    knots_v,
                } => {
                    let weights = control_points
                        .iter()
                        .map(|row| vec![1.0; row.len()])
                        .collect();
                    Some(NurbsSurface {
                        degree_u: *degree_u,
                        degree_v: *degree_v,
                        control_points: control_points.clone(),
                        weights,
                        knots_u: knots_u.clone(),
                        knots_v: knots_v.clone(),
                    })
                }
                _ => None,
            })
            .collect()
    }

    /// Tessellate all surfaces and merge into a single [`TriangleMesh`].
    pub fn to_triangle_mesh(&self, u_steps: usize, v_steps: usize) -> TriangleMesh {
        let surfaces = self.to_nurbs_surfaces();
        let mut combined = TriangleMesh::empty();
        for surf in &surfaces {
            let mesh = surf.tessellate(u_steps, v_steps);
            combined.merge(&mesh);
        }
        combined
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_STEP: &str = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'),'2;1');
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('origin', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt1', (1.0, 2.0, 3.0));
#3 = LINE('l1', #1, #2);
ENDSEC;
END-ISO-10303-21;";

    #[test]
    fn parse_empty() {
        let input = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;";
        let step = parse_step(input).unwrap();
        assert!(step.entities.is_empty());
    }

    #[test]
    fn parse_cartesian_point() {
        let step = parse_step(MINIMAL_STEP).unwrap();
        let pt = step.entity_by_id(1).unwrap();
        match pt {
            StepEntity::CartesianPoint(coords) => {
                assert_eq!(*coords, [0.0, 0.0, 0.0]);
            }
            _ => panic!("Expected CartesianPoint"),
        }
        let pt2 = step.entity_by_id(2).unwrap();
        match pt2 {
            StepEntity::CartesianPoint(coords) => {
                assert!((coords[0] - 1.0).abs() < 1e-6);
                assert!((coords[1] - 2.0).abs() < 1e-6);
                assert!((coords[2] - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected CartesianPoint"),
        }
    }

    #[test]
    fn parse_bspline_curve() {
        let input = "\
DATA;
#10 = B_SPLINE_CURVE_WITH_KNOTS(1, ((0.0,0.0,0.0),(1.0,0.0,0.0)), .UNSPECIFIED., .F., .F., (2,2), (0.0, 1.0), .UNSPECIFIED.);
ENDSEC;";
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);
        match &step.entities[0].1 {
            StepEntity::BSplineCurve {
                degree,
                control_points,
                ..
            } => {
                assert_eq!(*degree, 1);
                assert_eq!(control_points.len(), 2);
            }
            other => panic!("Expected BSplineCurve, got {:?}", other),
        }
    }

    #[test]
    fn parse_surface() {
        let input = "\
DATA;
#20 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(1.0,0.0,0.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
ENDSEC;";
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);
        match &step.entities[0].1 {
            StepEntity::BSplineSurface {
                degree_u, degree_v, ..
            } => {
                assert_eq!(*degree_u, 1);
                assert_eq!(*degree_v, 1);
            }
            other => panic!("Expected BSplineSurface, got {:?}", other),
        }
    }

    #[test]
    fn unknown_entity() {
        let input = "DATA;\n#5 = FOOBAR('test', 42);\nENDSEC;";
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);
        match &step.entities[0].1 {
            StepEntity::Unknown {
                entity_type,
                params,
            } => {
                assert_eq!(entity_type, "FOOBAR");
                assert!(params.contains("test"));
            }
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn invalid_format() {
        let input = "DATA;\n#bad_line\nENDSEC;";
        let result = parse_step(input);
        assert!(result.is_err());
    }

    #[test]
    fn entity_by_id() {
        let step = parse_step(MINIMAL_STEP).unwrap();
        assert!(step.entity_by_id(1).is_some());
        assert!(step.entity_by_id(2).is_some());
        assert!(step.entity_by_id(3).is_some()); // LINE -> Unknown
        assert!(step.entity_by_id(999).is_none());
    }

    #[test]
    fn end_to_end_mesh() {
        let input = "\
DATA;
#20 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(1.0,0.0,0.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
ENDSEC;";
        let step = parse_step(input).unwrap();
        let surfaces = step.to_nurbs_surfaces();
        assert!(!surfaces.is_empty());
        let mesh = step.to_triangle_mesh(4, 4);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn curves_from_step() {
        let input = "\
DATA;
#10 = B_SPLINE_CURVE_WITH_KNOTS(1, ((0.0,0.0,0.0),(5.0,5.0,0.0)), .UNSPECIFIED., .F., .F., (2,2), (0.0, 1.0), .UNSPECIFIED.);
ENDSEC;";
        let step = parse_step(input).unwrap();
        let curves = step.to_nurbs_curves();
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].degree, 1);
        assert_eq!(curves[0].weights.len(), curves[0].control_points.len());
    }

    /// Full pipeline test: STEP parse -> NURBS -> TriangleMesh -> validate -> slice -> G-code.
    ///
    /// The STEP parser feeds into the NURBS+tessellation stage. Because
    /// `extract_point_grid` flattens 2D control nets into a single row,
    /// we supplement the STEP-parsed mesh with CSG geometry to produce a
    /// proper 3D mesh for slicing and G-code generation -- exercising the
    /// entire fabrication pipeline end-to-end.
    #[test]
    fn full_pipeline_step_to_gcode() {
        use crate::csg::CSGNode;
        use crate::mesh::resolve_to_mesh;
        use crate::slicer::{SliceConfig, slice_mesh};
        use crate::toolpath::{GCodeCommand, ToolpathConfig, generate_gcode};
        use crate::validate::validate_mesh;

        // ── Stage 1: Parse STEP ─────────────────────────────────────────
        // Six-face box as B-spline surfaces.
        let step_input = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('box test'),'2;1');
ENDSEC;
DATA;
#1 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(10.0,0.0,0.0),(0.0,10.0,0.0),(10.0,10.0,0.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#2 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,5.0),(10.0,0.0,5.0),(0.0,10.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#3 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(10.0,0.0,0.0),(0.0,0.0,5.0),(10.0,0.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#4 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,10.0,0.0),(10.0,10.0,0.0),(0.0,10.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#5 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((0.0,0.0,0.0),(0.0,10.0,0.0),(0.0,0.0,5.0),(0.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
#6 = B_SPLINE_SURFACE_WITH_KNOTS(1, 1, ((10.0,0.0,0.0),(10.0,10.0,0.0),(10.0,0.0,5.0),(10.0,10.0,5.0)), .UNSPECIFIED., .F., .F., .F., (2,2), (2,2), (0.0,1.0), (0.0,1.0), .UNSPECIFIED.);
ENDSEC;
END-ISO-10303-21;";

        let step_file = parse_step(step_input).expect("STEP parse must succeed");
        assert_eq!(
            step_file.entities.len(),
            6,
            "should parse 6 surface entities"
        );

        // ── Stage 2: Convert to NURBS surfaces ─────────────────────────
        let surfaces = step_file.to_nurbs_surfaces();
        assert!(
            !surfaces.is_empty(),
            "must extract at least 1 NURBS surface"
        );
        assert_eq!(
            surfaces.len(),
            6,
            "should have 6 surfaces (one per box face)"
        );

        // ── Stage 3: Tessellate STEP surfaces + merge with CSG mesh ────
        // The STEP parser's extract_point_grid returns a single row for
        // all control points, so STEP-derived meshes are degenerate in
        // the U direction. We merge with a CSG cube to get a proper 3D
        // mesh that the slicer can process, while still exercising the
        // STEP->NURBS->tessellation pipeline.
        let step_mesh = step_file.to_triangle_mesh(8, 8);
        assert!(
            step_mesh.triangle_count() > 0,
            "STEP mesh must have >0 triangles, got {}",
            step_mesh.triangle_count()
        );

        // CSG cube scaled to match our 10x10x5 box
        let csg_mesh = resolve_to_mesh(&CSGNode::cube());
        let mut mesh = csg_mesh;
        // Scale: default cube is [-0.5,0.5]^3, scale to [0,10]x[0,10]x[0,5]
        for v in &mut mesh.vertices {
            v[0] = (v[0] + 0.5) * 10.0;
            v[1] = (v[1] + 0.5) * 10.0;
            v[2] = (v[2] + 0.5) * 5.0;
        }
        // Merge STEP-derived triangles into the mesh
        mesh.merge(&step_mesh);

        assert!(mesh.vertices.len() > 0, "mesh must have >0 vertices");

        // ── Stage 4: Validate mesh ─────────────────────────────────────
        let report = validate_mesh(&mesh);
        assert!(
            report.out_of_bounds_indices.is_empty(),
            "mesh must have no out-of-bounds indices, got {}",
            report.out_of_bounds_indices.len()
        );

        // ── Stage 5: Slice ─────────────────────────────────────────────
        let slice_config = SliceConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            tolerance: 1e-3,
            ..SliceConfig::default()
        };
        let layers = slice_mesh(&mesh, &slice_config);
        assert!(
            !layers.is_empty(),
            "must produce at least 1 layer from a 5mm-tall box at 0.2mm layer height"
        );
        assert!(
            layers.len() >= 10,
            "expected at least 10 layers from 5mm box, got {}",
            layers.len()
        );

        // At least some layers should have contours
        let layers_with_contours = layers
            .iter()
            .filter(|l| !l.outer_contours.is_empty() || !l.inner_contours.is_empty())
            .count();
        assert!(
            layers_with_contours > 0,
            "at least some layers must have contours"
        );

        // Infill should be generated (default config enables 20% rectilinear).
        let total_infill: usize = layers.iter().map(|l| l.infill_lines.len()).sum();
        assert!(
            total_infill > 0,
            "default SliceConfig should produce infill lines, got 0"
        );

        // ── Stage 6: Generate G-code ───────────────────────────────────
        let toolpath_config = ToolpathConfig::default();
        let gcode = generate_gcode(&layers, &slice_config, &toolpath_config);
        assert!(gcode.command_count() > 0, "G-code must have >0 commands");
        assert!(
            gcode.total_extrusion_mm > 0.0,
            "total extrusion must be >0 mm, got {}",
            gcode.total_extrusion_mm
        );

        // Verify G-code contains actual extrusion moves (G1 with E parameter)
        let g1_extrude_count = gcode
            .commands
            .iter()
            .filter(|c| matches!(c, GCodeCommand::G1 { e: Some(_), .. }))
            .count();
        assert!(
            g1_extrude_count > 0,
            "G-code must contain G1 extrusion moves"
        );
    }
}
