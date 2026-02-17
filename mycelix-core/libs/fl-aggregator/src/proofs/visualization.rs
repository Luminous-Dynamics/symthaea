//! Proof Visualization
//!
//! Provides visual representations of proof structure and verification:
//! - SVG rendering of circuit structure
//! - Constraint visualization
//! - Verification flow diagrams
//! - Proof size breakdown charts

use std::collections::HashMap;
use std::fmt::Write;

use crate::proofs::{ProofType, SecurityLevel};

// ============================================================================
// Visualization Configuration
// ============================================================================

/// Configuration for visualization output
#[derive(Clone, Debug)]
pub struct VisualizationConfig {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Show labels
    pub show_labels: bool,
    /// Show legend
    pub show_legend: bool,
    /// Font size
    pub font_size: u32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            color_scheme: ColorScheme::default(),
            show_labels: true,
            show_legend: true,
            font_size: 12,
        }
    }
}

/// Color scheme for visualizations
#[derive(Clone, Debug)]
pub struct ColorScheme {
    pub background: String,
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub success: String,
    pub error: String,
    pub text: String,
    pub grid: String,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            background: "#ffffff".to_string(),
            primary: "#3b82f6".to_string(),
            secondary: "#64748b".to_string(),
            accent: "#8b5cf6".to_string(),
            success: "#22c55e".to_string(),
            error: "#ef4444".to_string(),
            text: "#1f2937".to_string(),
            grid: "#e5e7eb".to_string(),
        }
    }
}

impl ColorScheme {
    /// Dark theme
    pub fn dark() -> Self {
        Self {
            background: "#1f2937".to_string(),
            primary: "#60a5fa".to_string(),
            secondary: "#9ca3af".to_string(),
            accent: "#a78bfa".to_string(),
            success: "#4ade80".to_string(),
            error: "#f87171".to_string(),
            text: "#f3f4f6".to_string(),
            grid: "#374151".to_string(),
        }
    }
}

// ============================================================================
// Circuit Visualization
// ============================================================================

/// Visualize circuit structure
pub fn visualize_circuit(proof_type: ProofType, config: &VisualizationConfig) -> String {
    let circuit_info = get_circuit_info(proof_type);
    render_circuit_svg(&circuit_info, config)
}

/// Circuit information for visualization
#[derive(Clone, Debug)]
pub struct CircuitInfo {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
    pub rows: usize,
    pub constraints: Vec<ConstraintInfo>,
}

/// Column information
#[derive(Clone, Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub purpose: String,
    pub column_type: ColumnType,
}

/// Type of column
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Main,
    Auxiliary,
    PublicInput,
}

/// Constraint information
#[derive(Clone, Debug)]
pub struct ConstraintInfo {
    pub name: String,
    pub degree: usize,
    pub columns_used: Vec<usize>,
}

fn get_circuit_info(proof_type: ProofType) -> CircuitInfo {
    match proof_type {
        ProofType::Range => CircuitInfo {
            name: "RangeProof".to_string(),
            columns: vec![
                ColumnInfo {
                    name: "bit".to_string(),
                    purpose: "Binary decomposition".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "range_flag".to_string(),
                    purpose: "In-range indicator".to_string(),
                    column_type: ColumnType::Auxiliary,
                },
                ColumnInfo {
                    name: "accumulator".to_string(),
                    purpose: "Running sum".to_string(),
                    column_type: ColumnType::Main,
                },
            ],
            rows: 64,
            constraints: vec![
                ConstraintInfo {
                    name: "bit_binary".to_string(),
                    degree: 2,
                    columns_used: vec![0],
                },
                ConstraintInfo {
                    name: "accumulator_correct".to_string(),
                    degree: 2,
                    columns_used: vec![0, 2],
                },
            ],
        },
        ProofType::GradientIntegrity => CircuitInfo {
            name: "GradientIntegrityProof".to_string(),
            columns: vec![
                ColumnInfo {
                    name: "membership[0-7]".to_string(),
                    purpose: "Merkle membership".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "commitment[8-11]".to_string(),
                    purpose: "Gradient commitment".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "gradient".to_string(),
                    purpose: "Gradient value".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "squared".to_string(),
                    purpose: "Squared for norm".to_string(),
                    column_type: ColumnType::Auxiliary,
                },
                ColumnInfo {
                    name: "running_norm".to_string(),
                    purpose: "L2 norm accumulator".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "round_id".to_string(),
                    purpose: "Round binding".to_string(),
                    column_type: ColumnType::PublicInput,
                },
            ],
            rows: 256,
            constraints: vec![
                ConstraintInfo {
                    name: "merkle_hash".to_string(),
                    degree: 2,
                    columns_used: vec![0, 1],
                },
                ConstraintInfo {
                    name: "commitment_valid".to_string(),
                    degree: 2,
                    columns_used: vec![1, 2],
                },
                ConstraintInfo {
                    name: "norm_accumulation".to_string(),
                    degree: 2,
                    columns_used: vec![2, 3, 4],
                },
            ],
        },
        ProofType::IdentityAssurance => CircuitInfo {
            name: "IdentityAssuranceProof".to_string(),
            columns: vec![
                ColumnInfo {
                    name: "factor_contrib".to_string(),
                    purpose: "Factor contribution".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "category".to_string(),
                    purpose: "Factor category".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "verified".to_string(),
                    purpose: "Verification flag".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "running_score".to_string(),
                    purpose: "Accumulated score".to_string(),
                    column_type: ColumnType::Auxiliary,
                },
            ],
            rows: 9,
            constraints: vec![
                ConstraintInfo {
                    name: "verified_binary".to_string(),
                    degree: 2,
                    columns_used: vec![2],
                },
                ConstraintInfo {
                    name: "score_accumulation".to_string(),
                    degree: 2,
                    columns_used: vec![0, 2, 3],
                },
            ],
        },
        ProofType::VoteEligibility => CircuitInfo {
            name: "VoteEligibilityProof".to_string(),
            columns: vec![
                ColumnInfo {
                    name: "requirement".to_string(),
                    purpose: "Requirement type".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "value".to_string(),
                    purpose: "User value".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "threshold".to_string(),
                    purpose: "Required threshold".to_string(),
                    column_type: ColumnType::PublicInput,
                },
                ColumnInfo {
                    name: "met".to_string(),
                    purpose: "Requirement met flag".to_string(),
                    column_type: ColumnType::Auxiliary,
                },
            ],
            rows: 7,
            constraints: vec![
                ConstraintInfo {
                    name: "met_binary".to_string(),
                    degree: 2,
                    columns_used: vec![3],
                },
                ConstraintInfo {
                    name: "comparison".to_string(),
                    degree: 2,
                    columns_used: vec![1, 2, 3],
                },
            ],
        },
        ProofType::Membership => CircuitInfo {
            name: "MembershipProof".to_string(),
            columns: vec![
                ColumnInfo {
                    name: "current_hash[0-3]".to_string(),
                    purpose: "Current node hash".to_string(),
                    column_type: ColumnType::Main,
                },
                ColumnInfo {
                    name: "sibling[4-7]".to_string(),
                    purpose: "Sibling hash".to_string(),
                    column_type: ColumnType::Main,
                },
            ],
            rows: 20,
            constraints: vec![ConstraintInfo {
                name: "hash_chain".to_string(),
                degree: 2,
                columns_used: vec![0, 1],
            }],
        },
    }
}

fn render_circuit_svg(info: &CircuitInfo, config: &VisualizationConfig) -> String {
    let mut svg = String::new();

    // SVG header
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        config.width, config.height, config.width, config.height
    )
    .unwrap();

    // Background
    writeln!(
        svg,
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        config.color_scheme.background
    )
    .unwrap();

    // Title
    if config.show_labels {
        writeln!(
            svg,
            r#"<text x="{}" y="30" font-size="{}" fill="{}" text-anchor="middle" font-weight="bold">{}</text>"#,
            config.width / 2,
            config.font_size + 4,
            config.color_scheme.text,
            info.name
        )
        .unwrap();
    }

    // Draw circuit grid
    let grid_x = 60;
    let grid_y = 60;
    let grid_width = config.width - 120;
    let grid_height = config.height - 150;

    let col_width = grid_width / info.columns.len().max(1) as u32;
    let row_height = (grid_height / info.rows.min(20).max(1) as u32).min(30);

    // Column headers
    for (i, col) in info.columns.iter().enumerate() {
        let x = grid_x + (i as u32 * col_width) + col_width / 2;
        let color = match col.column_type {
            ColumnType::Main => &config.color_scheme.primary,
            ColumnType::Auxiliary => &config.color_scheme.secondary,
            ColumnType::PublicInput => &config.color_scheme.accent,
        };

        if config.show_labels {
            writeln!(
                svg,
                r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="middle" transform="rotate(-45 {} {})">{}</text>"#,
                x, grid_y - 10, config.font_size - 2, color, x, grid_y - 10, col.name
            )
            .unwrap();
        }

        // Draw column cells
        for row in 0..info.rows.min(20) {
            let y = grid_y + (row as u32 * row_height);
            writeln!(
                svg,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" fill-opacity="0.3" stroke="{}" stroke-width="1"/>"#,
                grid_x + (i as u32 * col_width),
                y,
                col_width - 2,
                row_height - 2,
                color,
                config.color_scheme.grid
            )
            .unwrap();
        }
    }

    // Legend
    if config.show_legend {
        let legend_y = config.height - 60;
        let legend_items = [
            ("Main", &config.color_scheme.primary),
            ("Auxiliary", &config.color_scheme.secondary),
            ("Public Input", &config.color_scheme.accent),
        ];

        for (i, (label, color)) in legend_items.iter().enumerate() {
            let x = 60 + (i as u32 * 150);
            writeln!(
                svg,
                r#"<rect x="{}" y="{}" width="15" height="15" fill="{}" fill-opacity="0.5"/>"#,
                x, legend_y, color
            )
            .unwrap();
            writeln!(
                svg,
                r#"<text x="{}" y="{}" font-size="{}" fill="{}">{}</text>"#,
                x + 20,
                legend_y + 12,
                config.font_size,
                config.color_scheme.text,
                label
            )
            .unwrap();
        }

        // Circuit stats
        writeln!(
            svg,
            r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="end">Columns: {} | Rows: {} | Constraints: {}</text>"#,
            config.width - 20,
            legend_y + 12,
            config.font_size,
            config.color_scheme.secondary,
            info.columns.len(),
            info.rows,
            info.constraints.len()
        )
        .unwrap();
    }

    // Close SVG
    writeln!(svg, "</svg>").unwrap();

    svg
}

// ============================================================================
// Proof Size Visualization
// ============================================================================

/// Visualize proof size breakdown
pub fn visualize_proof_size(sizes: &ProofSizeBreakdown, config: &VisualizationConfig) -> String {
    let total: u64 = sizes.components.values().sum();

    let mut svg = String::new();

    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
        config.width, config.height
    )
    .unwrap();

    writeln!(
        svg,
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        config.color_scheme.background
    )
    .unwrap();

    // Title
    writeln!(
        svg,
        r#"<text x="{}" y="30" font-size="{}" fill="{}" text-anchor="middle" font-weight="bold">Proof Size Breakdown ({:.1} KB)</text>"#,
        config.width / 2,
        config.font_size + 4,
        config.color_scheme.text,
        total as f64 / 1024.0
    )
    .unwrap();

    // Pie chart
    let cx = config.width / 3;
    let cy = config.height / 2;
    let radius = config.height.min(config.width) / 3;

    let colors = [
        "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4",
    ];

    let mut start_angle: f64 = 0.0;

    for (i, (_name, &size)) in sizes.components.iter().enumerate() {
        let percentage = size as f64 / total as f64;
        let end_angle = start_angle + percentage * 360.0;

        let start_rad = start_angle.to_radians();
        let end_rad = end_angle.to_radians();

        let x1 = cx as f64 + (radius as f64 * start_rad.cos());
        let y1 = cy as f64 + (radius as f64 * start_rad.sin());
        let x2 = cx as f64 + (radius as f64 * end_rad.cos());
        let y2 = cy as f64 + (radius as f64 * end_rad.sin());

        let large_arc = if percentage > 0.5 { 1 } else { 0 };

        writeln!(
            svg,
            r#"<path d="M {} {} L {} {} A {} {} 0 {} 1 {} {} Z" fill="{}" stroke="{}" stroke-width="2"/>"#,
            cx, cy, x1, y1, radius, radius, large_arc, x2, y2,
            colors[i % colors.len()],
            config.color_scheme.background
        )
        .unwrap();

        start_angle = end_angle;
    }

    // Legend
    let legend_x = config.width * 2 / 3;
    let mut legend_y = 80u32;

    for (i, (name, &size)) in sizes.components.iter().enumerate() {
        let percentage = (size as f64 / total as f64) * 100.0;

        writeln!(
            svg,
            r#"<rect x="{}" y="{}" width="15" height="15" fill="{}"/>"#,
            legend_x,
            legend_y,
            colors[i % colors.len()]
        )
        .unwrap();

        writeln!(
            svg,
            r#"<text x="{}" y="{}" font-size="{}" fill="{}">{}: {:.1}% ({} bytes)</text>"#,
            legend_x + 20,
            legend_y + 12,
            config.font_size,
            config.color_scheme.text,
            name,
            percentage,
            size
        )
        .unwrap();

        legend_y += 25;
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

/// Proof size breakdown
#[derive(Clone, Debug, Default)]
pub struct ProofSizeBreakdown {
    pub components: HashMap<String, u64>,
}

impl ProofSizeBreakdown {
    /// Create from proof type estimate
    pub fn for_proof_type(proof_type: ProofType, security_level: SecurityLevel) -> Self {
        let base_multiplier = match security_level {
            SecurityLevel::Standard96 => 0.7,
            SecurityLevel::Standard128 => 1.0,
            SecurityLevel::High256 => 2.0,
        };

        let (trace, fri, queries, commitments) = match proof_type {
            ProofType::Range => (2000, 5000, 3000, 1500),
            ProofType::Membership => (3000, 7000, 4000, 2000),
            ProofType::GradientIntegrity => (5000, 10000, 6000, 3000),
            ProofType::IdentityAssurance => (2500, 6000, 3500, 1800),
            ProofType::VoteEligibility => (2200, 5500, 3200, 1600),
        };

        let mut components = HashMap::new();
        components.insert(
            "Trace Commitments".to_string(),
            (trace as f64 * base_multiplier) as u64,
        );
        components.insert(
            "FRI Proof".to_string(),
            (fri as f64 * base_multiplier) as u64,
        );
        components.insert(
            "Query Responses".to_string(),
            (queries as f64 * base_multiplier) as u64,
        );
        components.insert(
            "Constraint Commitments".to_string(),
            (commitments as f64 * base_multiplier) as u64,
        );
        components.insert("Public Inputs".to_string(), 256);

        Self { components }
    }
}

// ============================================================================
// Verification Flow Visualization
// ============================================================================

/// Visualize verification flow
pub fn visualize_verification_flow(config: &VisualizationConfig) -> String {
    let steps = [
        ("Parse Proof", "Deserialize proof bytes"),
        ("Verify Commitments", "Check polynomial commitments"),
        ("Verify FRI", "Validate FRI layers"),
        ("Check Constraints", "Verify constraint polynomial"),
        ("Verify Public Inputs", "Match public inputs"),
        ("Result", "Accept/Reject"),
    ];

    let mut svg = String::new();

    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
        config.width, config.height
    )
    .unwrap();

    writeln!(
        svg,
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        config.color_scheme.background
    )
    .unwrap();

    // Title
    writeln!(
        svg,
        r#"<text x="{}" y="30" font-size="{}" fill="{}" text-anchor="middle" font-weight="bold">Verification Flow</text>"#,
        config.width / 2,
        config.font_size + 4,
        config.color_scheme.text
    )
    .unwrap();

    // Flow boxes
    let box_width = 150u32;
    let box_height = 60u32;
    let start_x = (config.width - (box_width * 3 + 40)) / 2;
    let start_y = 70u32;

    for (i, (name, desc)) in steps.iter().enumerate() {
        let row = i / 3;
        let col = i % 3;

        let x = start_x + col as u32 * (box_width + 20);
        let y = start_y + row as u32 * (box_height + 40);

        // Box
        writeln!(
            svg,
            r#"<rect x="{}" y="{}" width="{}" height="{}" rx="5" fill="{}" stroke="{}" stroke-width="2"/>"#,
            x, y, box_width, box_height,
            config.color_scheme.primary,
            config.color_scheme.secondary
        )
        .unwrap();

        // Step name
        writeln!(
            svg,
            r#"<text x="{}" y="{}" font-size="{}" fill="white" text-anchor="middle" font-weight="bold">{}</text>"#,
            x + box_width / 2,
            y + 25,
            config.font_size,
            name
        )
        .unwrap();

        // Description
        writeln!(
            svg,
            r#"<text x="{}" y="{}" font-size="{}" fill="white" fill-opacity="0.8" text-anchor="middle">{}</text>"#,
            x + box_width / 2,
            y + 45,
            config.font_size - 2,
            desc
        )
        .unwrap();

        // Arrow to next
        if i < steps.len() - 1 {
            let next_row = (i + 1) / 3;
            #[allow(unused_variables)]
            let next_col = (i + 1) % 3;

            let arrow_x1 = x + box_width;
            let arrow_y1 = y + box_height / 2;

            if next_row == row as usize {
                // Same row - horizontal arrow
                writeln!(
                    svg,
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2" marker-end="url(#arrow)"/>"#,
                    arrow_x1, arrow_y1,
                    arrow_x1 + 15, arrow_y1,
                    config.color_scheme.secondary
                )
                .unwrap();
            } else {
                // New row - wrap arrow
                let next_x = start_x;
                let next_y = start_y + (next_row as u32) * (box_height + 40) + box_height / 2;

                writeln!(
                    svg,
                    r#"<path d="M {} {} Q {} {} {} {}" fill="none" stroke="{}" stroke-width="2"/>"#,
                    arrow_x1, arrow_y1,
                    config.width / 2, (arrow_y1 + next_y) / 2,
                    next_x - 10, next_y,
                    config.color_scheme.secondary
                )
                .unwrap();
            }
        }
    }

    // Arrow marker definition
    writeln!(
        svg,
        r#"<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{}"/></marker></defs>"#,
        config.color_scheme.secondary
    )
    .unwrap();

    writeln!(svg, "</svg>").unwrap();
    svg
}

// ============================================================================
// ASCII Visualization
// ============================================================================

/// Generate ASCII representation of circuit
pub fn ascii_circuit(proof_type: ProofType) -> String {
    let info = get_circuit_info(proof_type);
    let mut output = String::new();

    writeln!(output, "\n{}", "=".repeat(60)).unwrap();
    writeln!(output, "  {} Circuit", info.name).unwrap();
    writeln!(output, "{}", "=".repeat(60)).unwrap();

    writeln!(output, "\nColumns ({}): ", info.columns.len()).unwrap();
    for col in &info.columns {
        let type_char = match col.column_type {
            ColumnType::Main => 'M',
            ColumnType::Auxiliary => 'A',
            ColumnType::PublicInput => 'P',
        };
        writeln!(output, "  [{}] {} - {}", type_char, col.name, col.purpose).unwrap();
    }

    writeln!(output, "\nRows: {}", info.rows).unwrap();

    writeln!(output, "\nConstraints ({}): ", info.constraints.len()).unwrap();
    for constraint in &info.constraints {
        writeln!(
            output,
            "  {} (degree {}) - uses columns {:?}",
            constraint.name, constraint.degree, constraint.columns_used
        )
        .unwrap();
    }

    // Simple visual trace representation
    writeln!(output, "\nTrace Layout:").unwrap();
    writeln!(output, "+{}+", "-".repeat(info.columns.len() * 8)).unwrap();

    for row in 0..info.rows.min(5) {
        write!(output, "|").unwrap();
        for _ in &info.columns {
            write!(output, "  [{:02}] ", row).unwrap();
        }
        writeln!(output, "|").unwrap();
    }

    if info.rows > 5 {
        writeln!(output, "|{}|", "  ...  ".repeat(info.columns.len())).unwrap();
    }

    writeln!(output, "+{}+", "-".repeat(info.columns.len() * 8)).unwrap();

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_info() {
        let info = get_circuit_info(ProofType::Range);
        assert_eq!(info.name, "RangeProof");
        assert_eq!(info.columns.len(), 3);
        assert_eq!(info.rows, 64);
    }

    #[test]
    fn test_visualize_circuit() {
        let config = VisualizationConfig::default();
        let svg = visualize_circuit(ProofType::Range, &config);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("RangeProof"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_proof_size_breakdown() {
        let breakdown =
            ProofSizeBreakdown::for_proof_type(ProofType::Range, SecurityLevel::Standard128);

        assert!(!breakdown.components.is_empty());
        assert!(breakdown.components.contains_key("FRI Proof"));
    }

    #[test]
    fn test_visualize_proof_size() {
        let breakdown =
            ProofSizeBreakdown::for_proof_type(ProofType::Range, SecurityLevel::Standard128);
        let config = VisualizationConfig::default();
        let svg = visualize_proof_size(&breakdown, &config);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("FRI Proof"));
    }

    #[test]
    fn test_verification_flow() {
        let config = VisualizationConfig::default();
        let svg = visualize_verification_flow(&config);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Verification Flow"));
        assert!(svg.contains("Parse Proof"));
    }

    #[test]
    fn test_ascii_circuit() {
        let ascii = ascii_circuit(ProofType::Range);

        assert!(ascii.contains("RangeProof"));
        assert!(ascii.contains("Columns"));
        assert!(ascii.contains("Rows"));
    }

    #[test]
    fn test_color_scheme_dark() {
        let dark = ColorScheme::dark();
        assert!(dark.background.contains("1f2937"));
    }

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.show_labels);
    }

    #[test]
    fn test_all_proof_types() {
        let config = VisualizationConfig::default();

        for proof_type in [
            ProofType::Range,
            ProofType::Membership,
            ProofType::GradientIntegrity,
            ProofType::IdentityAssurance,
            ProofType::VoteEligibility,
        ] {
            let svg = visualize_circuit(proof_type, &config);
            assert!(svg.contains("<svg"));

            let ascii = ascii_circuit(proof_type);
            assert!(!ascii.is_empty());
        }
    }
}
