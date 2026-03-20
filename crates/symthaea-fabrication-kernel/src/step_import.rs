//! ISO 10303-21 (STEP Part 21) parser
//!
//! Parses a subset of STEP physical files into an entity table, then
//! converts recognized geometric entities (B-spline curves/surfaces,
//! Cartesian points, advanced faces) into NURBS and triangle mesh
//! representations via [`crate::nurbs`].

use crate::mesh::TriangleMesh;
use crate::nurbs::{NurbsCurve, NurbsSurface};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while parsing a STEP file.
#[derive(Debug, Clone, PartialEq)]
pub enum StepParseError {
    /// The input does not conform to ISO 10303-21 structure.
    InvalidFormat,
    /// An entity type is recognized but not supported by this parser.
    UnsupportedEntity(String),
    /// An entity references another entity by ID that does not exist.
    ReferenceError(usize),
}

impl std::fmt::Display for StepParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "Invalid STEP file format"),
            Self::UnsupportedEntity(name) => write!(f, "Unsupported entity: {}", name),
            Self::ReferenceError(id) => write!(f, "Unresolved reference: #{}", id),
        }
    }
}

impl std::error::Error for StepParseError {}

// ---------------------------------------------------------------------------
// Entity model
// ---------------------------------------------------------------------------

/// A parsed STEP entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepEntity {
    /// `B_SPLINE_CURVE_WITH_KNOTS` or `B_SPLINE_CURVE`
    BSplineCurve {
        degree: u32,
        control_points: Vec<[f64; 3]>,
        knots: Vec<f64>,
    },
    /// `B_SPLINE_SURFACE_WITH_KNOTS` or `B_SPLINE_SURFACE`
    BSplineSurface {
        degree_u: u32,
        degree_v: u32,
        control_points: Vec<Vec<[f64; 3]>>,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
    },
    /// `CARTESIAN_POINT`
    CartesianPoint([f64; 3]),
    /// `ADVANCED_FACE`
    AdvancedFace {
        surface_ref: usize,
        bounds: Vec<usize>,
    },
    /// Any entity type not specifically handled.
    Unknown {
        entity_type: String,
        params: String,
    },
}

// ---------------------------------------------------------------------------
// StepFile
// ---------------------------------------------------------------------------

/// A parsed STEP file containing an ordered list of entities.
///
/// Entity IDs from the file (`#1`, `#2`, ...) are stored as keys in an
/// internal map for cross-reference resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepFile {
    /// Ordered entity list (index matches insertion order, *not* entity ID).
    pub entities: Vec<StepEntity>,
    /// Map from STEP entity ID to index in `entities`.
    #[serde(skip)]
    id_map: HashMap<usize, usize>,
}

impl StepFile {
    /// Convert all `BSplineSurface` entities into [`NurbsSurface`] values.
    pub fn to_nurbs_surfaces(&self) -> Vec<NurbsSurface> {
        let mut surfaces = Vec::new();
        for entity in &self.entities {
            if let StepEntity::BSplineSurface {
                degree_u,
                degree_v,
                control_points,
                knots_u,
                knots_v,
            } = entity
            {
                let cp: Vec<Vec<[f32; 3]>> = control_points
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                            .collect()
                    })
                    .collect();

                let rows = cp.len();
                let cols = if rows > 0 { cp[0].len() } else { 0 };

                // Unit weights (STEP B_SPLINE_SURFACE doesn't carry NURBS weights
                // unless it's RATIONAL_B_SPLINE_SURFACE)
                let weights: Vec<Vec<f32>> = (0..rows).map(|_| vec![1.0; cols]).collect();

                surfaces.push(NurbsSurface {
                    degree_u: *degree_u,
                    degree_v: *degree_v,
                    control_points: cp,
                    weights,
                    knots_u: knots_u.iter().map(|k| *k as f32).collect(),
                    knots_v: knots_v.iter().map(|k| *k as f32).collect(),
                });
            }
        }
        surfaces
    }

    /// Convert all `BSplineCurve` entities into [`NurbsCurve`] values.
    pub fn to_nurbs_curves(&self) -> Vec<NurbsCurve> {
        let mut curves = Vec::new();
        for entity in &self.entities {
            if let StepEntity::BSplineCurve {
                degree,
                control_points,
                knots,
            } = entity
            {
                let cp: Vec<[f32; 3]> = control_points
                    .iter()
                    .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                    .collect();
                let weights = vec![1.0f32; cp.len()];

                curves.push(NurbsCurve {
                    degree: *degree,
                    control_points: cp,
                    weights,
                    knots: knots.iter().map(|k| *k as f32).collect(),
                });
            }
        }
        curves
    }

    /// Tessellate all surfaces and merge into a single [`TriangleMesh`].
    pub fn to_triangle_mesh(&self, tolerance: f32) -> TriangleMesh {
        let surfaces = self.to_nurbs_surfaces();
        let mut mesh = TriangleMesh::empty();
        for surface in &surfaces {
            let sub = surface.tessellate(tolerance);
            mesh.merge(&sub);
        }
        mesh
    }

    /// Look up an entity by its STEP file ID (e.g., the `N` in `#N`).
    pub fn entity_by_id(&self, id: usize) -> Option<&StepEntity> {
        self.id_map.get(&id).and_then(|idx| self.entities.get(*idx))
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a STEP Part 21 file from its text content.
///
/// Recognizes the following structure:
/// ```text
/// ISO-10303-21;
/// HEADER;
/// ...
/// ENDSEC;
/// DATA;
/// #1 = ENTITY_TYPE(...);
/// ...
/// ENDSEC;
/// END-ISO-10303-21;
/// ```
///
/// Entity types currently parsed:
/// - `CARTESIAN_POINT`
/// - `B_SPLINE_CURVE_WITH_KNOTS`
/// - `B_SPLINE_SURFACE_WITH_KNOTS`
/// - `ADVANCED_FACE`
///
/// All other entity types are stored as [`StepEntity::Unknown`].
pub fn parse_step(input: &str) -> Result<StepFile, StepParseError> {
    let mut entities = Vec::new();
    let mut id_map = HashMap::new();

    let mut in_data = false;
    let mut found_header = false;

    // Accumulate multi-line entity text
    let mut current_entity_line = String::new();

    for line in input.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with("/*") {
            continue;
        }

        // Detect structure markers
        let upper = trimmed.to_uppercase();
        if upper.starts_with("ISO-10303-21") {
            found_header = true;
            continue;
        }
        if upper == "END-ISO-10303-21;" {
            break;
        }
        if upper == "DATA;" {
            in_data = true;
            continue;
        }
        if upper == "ENDSEC;" {
            if in_data {
                in_data = false;
            }
            continue;
        }
        if upper == "HEADER;" {
            continue;
        }

        if !in_data {
            continue;
        }

        // Accumulate entity lines (entities may span multiple lines)
        current_entity_line.push_str(trimmed);
        if !trimmed.ends_with(';') {
            current_entity_line.push(' ');
            continue;
        }

        // Parse the complete entity line
        let entity_line = current_entity_line.clone();
        current_entity_line.clear();

        if let Some((id, entity)) = parse_entity_line(&entity_line)? {
            let idx = entities.len();
            entities.push(entity);
            id_map.insert(id, idx);
        }
    }

    // An empty file (no ISO header at all) is considered invalid
    if !found_header && !input.trim().is_empty() {
        return Err(StepParseError::InvalidFormat);
    }

    Ok(StepFile { entities, id_map })
}

/// Parse a single entity line like `#5 = CARTESIAN_POINT((1.0, 2.0, 3.0));`
fn parse_entity_line(line: &str) -> Result<Option<(usize, StepEntity)>, StepParseError> {
    let line = line.trim().trim_end_matches(';');

    // Match `#N = TYPE(...)`
    let hash_pos = match line.find('#') {
        Some(p) => p,
        None => return Ok(None),
    };
    let eq_pos = match line.find('=') {
        Some(p) => p,
        None => return Ok(None),
    };

    let id_str = line[hash_pos + 1..eq_pos].trim();
    let id: usize = id_str.parse().map_err(|_| StepParseError::InvalidFormat)?;

    let rest = line[eq_pos + 1..].trim();

    // Extract entity type and params
    let paren_pos = rest.find('(');
    let (entity_type, params_raw) = match paren_pos {
        Some(pos) => {
            let etype = rest[..pos].trim().to_uppercase();
            // Everything between the outer parens
            let params = rest[pos + 1..].trim();
            let params = params.trim_end_matches(')');
            (etype, params.to_string())
        }
        None => {
            let etype = rest.trim().to_uppercase();
            (etype, String::new())
        }
    };

    let entity = match entity_type.as_str() {
        "CARTESIAN_POINT" => parse_cartesian_point(&params_raw)?,
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" => parse_bspline_curve(&params_raw)?,
        "B_SPLINE_SURFACE_WITH_KNOTS" | "B_SPLINE_SURFACE" => parse_bspline_surface(&params_raw)?,
        "ADVANCED_FACE" => parse_advanced_face(&params_raw)?,
        _ => StepEntity::Unknown {
            entity_type,
            params: params_raw,
        },
    };

    Ok(Some((id, entity)))
}

/// Parse `CARTESIAN_POINT` params: `'name', (x, y, z)`
fn parse_cartesian_point(params: &str) -> Result<StepEntity, StepParseError> {
    let coords = extract_float_list(params);
    if coords.len() >= 3 {
        Ok(StepEntity::CartesianPoint([coords[0], coords[1], coords[2]]))
    } else if coords.len() == 2 {
        Ok(StepEntity::CartesianPoint([coords[0], coords[1], 0.0]))
    } else {
        Err(StepParseError::InvalidFormat)
    }
}

/// Parse `B_SPLINE_CURVE_WITH_KNOTS` params.
///
/// Simplified format: `'name', degree, (cp_refs...), ...knots...`
/// We extract degree and inline coordinate lists.
fn parse_bspline_curve(params: &str) -> Result<StepEntity, StepParseError> {
    let numbers = extract_float_list(params);
    // Heuristic: first number is degree, then groups of 3 for control points,
    // remaining are knots.
    if numbers.is_empty() {
        return Err(StepParseError::InvalidFormat);
    }

    let degree = numbers[0] as u32;
    let rest = &numbers[1..];

    // Try to find a reasonable split between control points and knots.
    // A valid knot vector for n CPs at degree p has n + p + 1 entries.
    // We try increasing n until the count works out.
    let total = rest.len();
    let mut n_cps = 0;
    for candidate_n in 1..=(total / 3) {
        let cp_floats = candidate_n * 3;
        let knot_count = total - cp_floats;
        if knot_count == candidate_n + degree as usize + 1 {
            n_cps = candidate_n;
            break;
        }
    }

    if n_cps == 0 {
        // Fall back: treat all as control points (3-tuples), no knots
        let cp_count = total / 3;
        let control_points: Vec<[f64; 3]> = (0..cp_count)
            .map(|i| [rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]])
            .collect();
        let n = control_points.len();
        // Generate a clamped uniform knot vector
        let knots = make_clamped_knots(n, degree);
        return Ok(StepEntity::BSplineCurve {
            degree,
            control_points,
            knots,
        });
    }

    let control_points: Vec<[f64; 3]> = (0..n_cps)
        .map(|i| [rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2]])
        .collect();
    let knots: Vec<f64> = rest[n_cps * 3..].to_vec();

    Ok(StepEntity::BSplineCurve {
        degree,
        control_points,
        knots,
    })
}

/// Parse `B_SPLINE_SURFACE_WITH_KNOTS` params.
///
/// Simplified: `'name', degree_u, degree_v, ((cp_row), ...), ...knots_u..., ...knots_v...`
fn parse_bspline_surface(params: &str) -> Result<StepEntity, StepParseError> {
    let numbers = extract_float_list(params);
    if numbers.len() < 2 {
        return Err(StepParseError::InvalidFormat);
    }

    let degree_u = numbers[0] as u32;
    let degree_v = numbers[1] as u32;
    let rest = &numbers[2..];

    // Heuristic: try to figure out rows x cols from remaining float count.
    // Total = rows * cols * 3  +  (rows + degree_u + 1)  +  (cols + degree_v + 1)
    // We try small grid sizes.
    let total = rest.len();
    let mut best_rows = 0usize;
    let mut best_cols = 0usize;

    'outer: for rows in 2..=20 {
        for cols in 2..=20 {
            let cp_count = rows * cols * 3;
            let ku = rows + degree_u as usize + 1;
            let kv = cols + degree_v as usize + 1;
            if cp_count + ku + kv == total {
                best_rows = rows;
                best_cols = cols;
                break 'outer;
            }
        }
    }

    if best_rows == 0 {
        // Fallback: minimal 2x2 surface from whatever we have
        let cp_count = (total / 3).max(4);
        let side = (cp_count as f64).sqrt().ceil() as usize;
        let rows = side.min(total / 3 / 2).max(2);
        let cols = ((total / 3) / rows).max(2);
        let _actual_cps = rows * cols;
        let control_points: Vec<Vec<[f64; 3]>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let idx = r * cols + c;
                        if idx * 3 + 2 < rest.len() {
                            [rest[idx * 3], rest[idx * 3 + 1], rest[idx * 3 + 2]]
                        } else {
                            [0.0, 0.0, 0.0]
                        }
                    })
                    .collect()
            })
            .collect();
        let knots_u = make_clamped_knots(rows, degree_u);
        let knots_v = make_clamped_knots(cols, degree_v);

        return Ok(StepEntity::BSplineSurface {
            degree_u,
            degree_v,
            control_points,
            knots_u,
            knots_v,
        });
    }

    let cp_end = best_rows * best_cols * 3;
    let ku_end = cp_end + best_rows + degree_u as usize + 1;

    let control_points: Vec<Vec<[f64; 3]>> = (0..best_rows)
        .map(|r| {
            (0..best_cols)
                .map(|c| {
                    let idx = r * best_cols + c;
                    [rest[idx * 3], rest[idx * 3 + 1], rest[idx * 3 + 2]]
                })
                .collect()
        })
        .collect();

    let knots_u: Vec<f64> = rest[cp_end..ku_end].to_vec();
    let knots_v: Vec<f64> = rest[ku_end..].to_vec();

    Ok(StepEntity::BSplineSurface {
        degree_u,
        degree_v,
        control_points,
        knots_u,
        knots_v,
    })
}

/// Parse `ADVANCED_FACE` params: `'name', (#bound_refs...), #surface_ref, .SAME_SENSE.`
fn parse_advanced_face(params: &str) -> Result<StepEntity, StepParseError> {
    let refs = extract_refs(params);
    if refs.is_empty() {
        return Err(StepParseError::InvalidFormat);
    }
    // Last ref is typically the surface, preceding ones are bounds
    let surface_ref = *refs.last().unwrap();
    let bounds = refs[..refs.len() - 1].to_vec();
    Ok(StepEntity::AdvancedFace {
        surface_ref,
        bounds,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract all floating-point numbers from a string (ignores non-numeric tokens).
fn extract_float_list(s: &str) -> Vec<f64> {
    let mut result = Vec::new();
    // Split by common delimiters
    for token in s.split(|c: char| c == ',' || c == '(' || c == ')' || c == ' ' || c == ';') {
        let t = token.trim().trim_matches('\'');
        if let Ok(v) = t.parse::<f64>() {
            result.push(v);
        }
    }
    result
}

/// Extract entity references (`#N`) from a string.
fn extract_refs(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                if let Ok(id) = s[start..end].parse::<usize>() {
                    result.push(id);
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    result
}

/// Generate a clamped uniform knot vector for `n` control points at `degree`.
fn make_clamped_knots(n: usize, degree: u32) -> Vec<f64> {
    let p = degree as usize;
    let m = n + p + 1;
    let mut knots = Vec::with_capacity(m);
    for _ in 0..=p {
        knots.push(0.0);
    }
    let interior = m - 2 * (p + 1);
    for i in 1..=interior {
        knots.push(i as f64 / (interior + 1) as f64);
    }
    for _ in 0..=p {
        knots.push(1.0);
    }
    knots
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_step() -> String {
        r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'), '2;1');
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
"#
        .to_string()
    }

    #[test]
    fn test_parse_empty_file() {
        let step = parse_step(&minimal_step()).unwrap();
        assert!(step.entities.is_empty());
    }

    #[test]
    fn test_parse_cartesian_point() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('origin', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('pt', (1.5, 2.5, 3.5));
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 2);

        match &step.entities[0] {
            StepEntity::CartesianPoint(p) => {
                assert!((p[0]).abs() < 1e-10);
                assert!((p[1]).abs() < 1e-10);
                assert!((p[2]).abs() < 1e-10);
            }
            other => panic!("Expected CartesianPoint, got {:?}", other),
        }

        match &step.entities[1] {
            StepEntity::CartesianPoint(p) => {
                assert!((p[0] - 1.5).abs() < 1e-10);
                assert!((p[1] - 2.5).abs() < 1e-10);
                assert!((p[2] - 3.5).abs() < 1e-10);
            }
            other => panic!("Expected CartesianPoint, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_bspline_curve() {
        // degree=1, 2 control points, 4 knots (0,0,1,1)
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = B_SPLINE_CURVE_WITH_KNOTS('line', 1, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);

        match &step.entities[0] {
            StepEntity::BSplineCurve {
                degree,
                control_points,
                knots,
            } => {
                assert_eq!(*degree, 1);
                assert_eq!(control_points.len(), 2);
                assert_eq!(knots.len(), 4);
                assert!((control_points[1][0] - 10.0).abs() < 1e-10);
            }
            other => panic!("Expected BSplineCurve, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_bspline_surface() {
        // degree_u=1, degree_v=1, 2x2 control points (12 floats), knots_u (4), knots_v (4)
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = B_SPLINE_SURFACE_WITH_KNOTS('patch', 1, 1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);

        match &step.entities[0] {
            StepEntity::BSplineSurface {
                degree_u,
                degree_v,
                control_points,
                knots_u,
                knots_v,
            } => {
                assert_eq!(*degree_u, 1);
                assert_eq!(*degree_v, 1);
                assert_eq!(control_points.len(), 2);
                assert_eq!(control_points[0].len(), 2);
                assert_eq!(knots_u.len(), 4);
                assert_eq!(knots_v.len(), 4);
            }
            other => panic!("Expected BSplineSurface, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_entity_handling() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = DIRECTION('dir', (1.0, 0.0, 0.0));
#2 = VECTOR('vec', #1, 5.0);
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 2);

        match &step.entities[0] {
            StepEntity::Unknown {
                entity_type,
                params: _,
            } => {
                assert_eq!(entity_type, "DIRECTION");
            }
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn test_reference_resolution() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('p1', (1.0, 2.0, 3.0));
#20 = CARTESIAN_POINT('p2', (4.0, 5.0, 6.0));
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert!(step.entity_by_id(10).is_some());
        assert!(step.entity_by_id(20).is_some());
        assert!(step.entity_by_id(99).is_none());
    }

    #[test]
    fn test_advanced_face_parsing() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#5 = ADVANCED_FACE('face1', (#10, #11), #20, .T.);
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 2);

        match &step.entities[1] {
            StepEntity::AdvancedFace {
                surface_ref,
                bounds,
            } => {
                assert_eq!(*surface_ref, 20);
                assert_eq!(bounds, &[10, 11]);
            }
            other => panic!("Expected AdvancedFace, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_format() {
        let result = parse_step("This is not a STEP file at all.");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), StepParseError::InvalidFormat);
    }

    #[test]
    fn test_end_to_end_parse_to_mesh() {
        // Build a STEP file with a bilinear surface patch
        let input = r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'), '2;1');
ENDSEC;
DATA;
#1 = B_SPLINE_SURFACE_WITH_KNOTS('flat', 1, 1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0);
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        let surfaces = step.to_nurbs_surfaces();
        assert_eq!(surfaces.len(), 1);

        let mesh = step.to_triangle_mesh(0.01);
        assert!(
            mesh.triangle_count() >= 2,
            "Expected at least 2 triangles, got {}",
            mesh.triangle_count()
        );
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
    }

    #[test]
    fn test_multiline_entity() {
        let input = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('long',
    (100.0, 200.0,
     300.0));
ENDSEC;
END-ISO-10303-21;
"#;
        let step = parse_step(input).unwrap();
        assert_eq!(step.entities.len(), 1);
        match &step.entities[0] {
            StepEntity::CartesianPoint(p) => {
                assert!((p[0] - 100.0).abs() < 1e-10);
                assert!((p[1] - 200.0).abs() < 1e-10);
                assert!((p[2] - 300.0).abs() < 1e-10);
            }
            other => panic!("Expected CartesianPoint, got {:?}", other),
        }
    }
}
