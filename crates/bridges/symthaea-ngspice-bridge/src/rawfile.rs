// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict parser for ngspice ASCII rawfiles.
//!
//! This module parses numeric output only. It does not infer solver convergence,
//! physical validity, or qualification; callers must obtain those from separate
//! solver-specific evidence and independent checks.

use std::error::Error;
use std::fmt;

/// Stable parser identity for evidence manifests and regression fixtures.
pub const PARSER_VERSION: &str = "ngspice-ascii-raw-v1";

// Bounds are intentional: rawfiles may be solver-generated or externally supplied.
const MAX_RAWFILE_BYTES: usize = 64 * 1024 * 1024;
const MAX_RAWFILE_LINES: usize = 3_000_000;
const MAX_VARIABLES: usize = 10_000;
const MAX_POINTS: usize = 1_000_000;
const MAX_SCALARS: usize = 5_000_000;

/// A vector declared in an ngspice rawfile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawVariable {
    /// Zero-based variable index declared by the file.
    pub index: usize,
    /// Exact solver vector name, for example time or v(out).
    pub name: String,
    /// Exact solver variable kind, for example time or voltage.
    pub kind: String,
}

impl RawVariable {
    /// Map a small, explicit set of common ngspice kinds to SI display units.
    ///
    /// Unknown kinds deliberately return None; callers must not guess units.
    pub fn si_unit(&self) -> Option<&'static str> {
        match self.kind.as_str() {
            "time" => Some("s"),
            "voltage" => Some("V"),
            "current" => Some("A"),
            "frequency" => Some("Hz"),
            _ => None,
        }
    }
}

/// One successfully parsed, single-plot ASCII ngspice rawfile.
#[derive(Debug, Clone, PartialEq)]
pub struct AsciiRawfile {
    /// Title declared by ngspice.
    pub title: String,
    /// Analysis name, such as Transient Analysis or Operating Point.
    pub plotname: String,
    /// Raw flags, retained rather than normalized away.
    pub flags: Vec<String>,
    /// Variables in solver-declared order.
    pub variables: Vec<RawVariable>,
    /// Each row contains one finite scalar per declared variable.
    pub points: Vec<Vec<f64>>,
}

/// A parse or lookup failure. Errors are descriptive but never recover by
/// fabricating missing values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawfileError(pub String);

impl fmt::Display for RawfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RawfileError {}

impl AsciiRawfile {
    /// Parse a single-plot, real-valued ngspice ASCII rawfile.
    ///
    /// Multi-plot, complex, malformed, truncated, non-finite, or count-
    /// inconsistent files are rejected. The parser intentionally supports a
    /// strict subset rather than silently accepting formats it cannot verify.
    pub fn parse(input: &str) -> Result<Self, RawfileError> {
        if input.len() > MAX_RAWFILE_BYTES {
            return Err(RawfileError(format!(
                "rawfile is {} bytes; maximum supported size is {MAX_RAWFILE_BYTES}",
                input.len()
            )));
        }
        let line_count = input.lines().count();
        if line_count > MAX_RAWFILE_LINES {
            return Err(RawfileError(format!(
                "rawfile has {line_count} lines; maximum supported count is {MAX_RAWFILE_LINES}"
            )));
        }
        let lines: Vec<&str> = input.lines().collect();

        let plotname_count = lines
            .iter()
            .filter(|line| line.trim_start().starts_with("Plotname:"))
            .count();
        if plotname_count != 1 {
            return Err(RawfileError(format!(
                "expected exactly one Plotname header, found {plotname_count}"
            )));
        }

        let variables_headers: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (line.trim() == "Variables:").then_some(index))
            .collect();
        let values_headers: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (line.trim() == "Values:").then_some(index))
            .collect();
        if variables_headers.len() != 1 || values_headers.len() != 1 {
            return Err(RawfileError(
                "expected exactly one Variables: and one Values: section".into(),
            ));
        }

        let variables_header = variables_headers[0];
        let values_header = values_headers[0];
        if values_header <= variables_header {
            return Err(RawfileError(
                "Values: section appears before Variables: section".into(),
            ));
        }

        let title = required_header(&lines[..variables_header], "Title:")?;
        let plotname = required_header(&lines[..variables_header], "Plotname:")?;
        let flags_header = required_header(&lines[..variables_header], "Flags:")?;
        let flags: Vec<String> = flags_header
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect();
        if !flags.iter().any(|flag| flag == "real") || flags.iter().any(|flag| flag == "complex") {
            return Err(RawfileError(format!(
                "only real-valued rawfiles are supported (flags: {})",
                flags.join(" ")
            )));
        }

        let variable_count = parse_positive_count(
            &lines[..variables_header],
            "No. Variables:",
        )?;
        let point_count = parse_positive_count(&lines[..variables_header], "No. Points:")?;
        if variable_count > MAX_VARIABLES {
            return Err(RawfileError(format!(
                "declared variable count {variable_count} exceeds limit {MAX_VARIABLES}"
            )));
        }
        if point_count > MAX_POINTS {
            return Err(RawfileError(format!(
                "declared point count {point_count} exceeds limit {MAX_POINTS}"
            )));
        }
        let scalar_count = variable_count.checked_mul(point_count).ok_or_else(|| {
            RawfileError("declared variable/point product overflows".into())
        })?;
        if scalar_count > MAX_SCALARS {
            return Err(RawfileError(format!(
                "declared scalar count {scalar_count} exceeds limit {MAX_SCALARS}"
            )));
        }

        let variable_lines: Vec<&str> = lines[variables_header + 1..values_header]
            .iter()
            .copied()
            .filter(|line| !line.trim().is_empty())
            .collect();
        if variable_lines.len() != variable_count {
            return Err(RawfileError(format!(
                "declared {variable_count} variables but found {} variable rows",
                variable_lines.len()
            )));
        }

        let mut variables = Vec::with_capacity(variable_count);
        for (expected_index, line) in variable_lines.into_iter().enumerate() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 3 {
                return Err(RawfileError(format!(
                    "variable row {expected_index} must contain index, name, and kind"
                )));
            }
            let index = fields[0].parse::<usize>().map_err(|_| {
                RawfileError(format!(
                    "invalid variable index {:?} on row {expected_index}",
                    fields[0]
                ))
            })?;
            if index != expected_index {
                return Err(RawfileError(format!(
                    "variable index {index} is out of order; expected {expected_index}"
                )));
            }
            let name = fields[1].to_string();
            if variables.iter().any(|variable: &RawVariable| variable.name == name) {
                return Err(RawfileError(format!("duplicate variable name {name:?}")));
            }
            variables.push(RawVariable {
                index,
                name,
                kind: fields[2].to_string(),
            });
        }

        let mut points: Vec<Vec<f64>> = Vec::with_capacity(point_count);
        let mut current_point: Option<Vec<f64>> = None;
        for (offset, raw_line) in lines[values_header + 1..].iter().enumerate() {
            if raw_line.trim().is_empty() {
                continue;
            }

            let line_number = values_header + 2 + offset;
            let fields: Vec<&str> = raw_line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }

            // ngspice's native writer prefixes point indices with whitespace
            // too, then emits one scientific-notation value per line. Detect
            // a new point by the declared value count, not indentation.
            let starts_new_point = current_point
                .as_ref()
                .is_none_or(|point| point.len() == variable_count);

            if starts_new_point {
                finish_point(
                    &mut points,
                    &mut current_point,
                    variable_count,
                    point_count,
                )?;
                if points.len() >= point_count {
                    return Err(RawfileError(format!(
                        "unexpected extra data point at line {line_number}"
                    )));
                }

                let point_index = fields[0].parse::<usize>().map_err(|_| {
                    RawfileError(format!(
                        "expected a point index at line {line_number}"
                    ))
                })?;
                if point_index != points.len() {
                    return Err(RawfileError(format!(
                        "point index {point_index} is out of order; expected {}",
                        points.len()
                    )));
                }
                current_point = Some(Vec::with_capacity(variable_count));
                for field in fields.iter().skip(1) {
                    push_finite_value(
                        current_point.as_mut().expect("point was just initialized"),
                        field,
                        line_number,
                    )?;
                }
            } else {
                let point = current_point.as_mut().expect("incomplete point exists");
                for field in &fields {
                    push_finite_value(point, field, line_number)?;
                }
            }

            if current_point
                .as_ref()
                .is_some_and(|point| point.len() > variable_count)
            {
                return Err(RawfileError(format!(
                    "point {} contains more than {variable_count} values",
                    points.len()
                )));
            }
        }

        finish_point(
            &mut points,
            &mut current_point,
            variable_count,
            point_count,
        )?;
        if points.len() != point_count {
            return Err(RawfileError(format!(
                "declared {point_count} points but parsed {}",
                points.len()
            )));
        }

        Ok(Self {
            title,
            plotname,
            flags,
            variables,
            points,
        })
    }

    /// Read the last finite sample for a vector by its exact solver name.
    pub fn final_value(&self, variable_name: &str) -> Result<f64, RawfileError> {
        let index = self.variable_index(variable_name)?;
        self.points
            .last()
            .and_then(|point| point.get(index))
            .copied()
            .ok_or_else(|| RawfileError("rawfile has no final sample".into()))
    }

    /// Return the sample of variable_name whose axis_name value is nearest to
    /// target. This does not interpolate between samples.
    pub fn value_nearest_to(
        &self,
        axis_name: &str,
        target: f64,
        variable_name: &str,
    ) -> Result<f64, RawfileError> {
        if !target.is_finite() {
            return Err(RawfileError(
                "nearest-sample target must be finite".into(),
            ));
        }
        let axis_index = self.variable_index(axis_name)?;
        let value_index = self.variable_index(variable_name)?;
        let point = self
            .points
            .iter()
            .min_by(|left, right| {
                (left[axis_index] - target)
                    .abs()
                    .total_cmp(&(right[axis_index] - target).abs())
            })
            .ok_or_else(|| RawfileError("rawfile has no samples".into()))?;
        Ok(point[value_index])
    }

    /// Return the largest absolute sample for a vector.
    pub fn peak_abs_value(&self, variable_name: &str) -> Result<f64, RawfileError> {
        let index = self.variable_index(variable_name)?;
        self.points
            .iter()
            .map(|point| point[index].abs())
            .max_by(f64::total_cmp)
            .ok_or_else(|| RawfileError("rawfile has no samples".into()))
    }

    /// Return a known SI unit for a vector; unknown kinds fail closed.
    pub fn si_unit(&self, variable_name: &str) -> Result<&'static str, RawfileError> {
        let variable = self
            .variables
            .iter()
            .find(|variable| variable.name == variable_name)
            .ok_or_else(|| RawfileError(format!("requested vector {variable_name:?} is absent")))?;
        variable.si_unit().ok_or_else(|| {
            RawfileError(format!(
                "no verified unit mapping for vector {:?} of kind {:?}",
                variable.name, variable.kind
            ))
        })
    }

    fn variable_index(&self, variable_name: &str) -> Result<usize, RawfileError> {
        self.variables
            .iter()
            .find(|variable| variable.name == variable_name)
            .map(|variable| variable.index)
            .ok_or_else(|| RawfileError(format!("requested vector {variable_name:?} is absent")))
    }
}

fn required_header(lines: &[&str], prefix: &str) -> Result<String, RawfileError> {
    let matches: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.trim_start().strip_prefix(prefix))
        .map(str::trim)
        .collect();
    if matches.len() != 1 || matches[0].is_empty() {
        return Err(RawfileError(format!(
            "expected one non-empty {prefix} header"
        )));
    }
    Ok(matches[0].to_string())
}

fn parse_positive_count(lines: &[&str], prefix: &str) -> Result<usize, RawfileError> {
    let value = required_header(lines, prefix)?;
    let count = value
        .parse::<usize>()
        .map_err(|_| RawfileError(format!("invalid numeric value for {prefix}")))?;
    if count == 0 {
        return Err(RawfileError(format!("{prefix} must be greater than zero")));
    }
    Ok(count)
}

fn push_finite_value(
    point: &mut Vec<f64>,
    text: &str,
    line_number: usize,
) -> Result<(), RawfileError> {
    let value = text.parse::<f64>().map_err(|_| {
        RawfileError(format!(
            "invalid numeric value {text:?} on rawfile line {line_number}"
        ))
    })?;
    if !value.is_finite() {
        return Err(RawfileError(format!(
            "non-finite numeric value on rawfile line {line_number}"
        )));
    }
    point.push(value);
    Ok(())
}

fn finish_point(
    points: &mut Vec<Vec<f64>>,
    current_point: &mut Option<Vec<f64>>,
    variable_count: usize,
    point_count: usize,
) -> Result<(), RawfileError> {
    if let Some(point) = current_point.take() {
        if point.len() != variable_count {
            return Err(RawfileError(format!(
                "point {} has {} values; expected {variable_count}",
                points.len(),
                point.len()
            )));
        }
        if points.len() >= point_count {
            return Err(RawfileError(
                "rawfile contains more points than declared".into(),
            ));
        }
        points.push(point);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RC_FIXTURE: &str = include_str!("../tests/fixtures/rc_step_ascii.raw");
    const RLC_FIXTURE: &str =
        include_str!("../tests/fixtures/rlc_step_analytic_ascii.raw");
    const DC_FIXTURE: &str =
        include_str!("../tests/fixtures/dc_operating_point_analytic_ascii.raw");

    #[test]
    fn parses_critically_damped_rlc_analytic_reference() {
        let raw = AsciiRawfile::parse(RLC_FIXTURE).expect("valid RLC ASCII fixture");
        assert_eq!(raw.title, "Critically damped RLC step analytical reference");
        assert_eq!(raw.plotname, "Transient Analysis");
        assert_eq!(raw.points.len(), 5);
        assert_eq!(raw.si_unit("time").unwrap(), "s");
        assert_eq!(raw.si_unit("v(out)").unwrap(), "V");

        // Series RLC: R=200 ohm, L=10 mH, C=1 uF gives zeta=1 and
        // omega_n=1/sqrt(LC)=10,000 rad/s. The capacitor step response is
        // 1 - (1 + omega_n*t)*exp(-omega_n*t) V.
        let omega_n = 1.0 / (10e-3_f64 * 1e-6_f64).sqrt();
        assert!((omega_n - 10_000.0).abs() < 1e-9);
        for time in [0.0, 1e-4, 2e-4, 5e-4, 1e-3] {
            let omega_t = omega_n * time;
            let expected = 1.0 - (1.0 + omega_t) * (-omega_t).exp();
            let actual = raw.value_nearest_to("time", time, "v(out)").unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "RLC mismatch at t={time}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn parses_resistive_divider_dc_operating_point_reference() {
        let raw = AsciiRawfile::parse(DC_FIXTURE).expect("valid DC ASCII fixture");
        assert_eq!(raw.title, "Resistive divider operating point analytical reference");
        assert_eq!(raw.plotname, "Operating Point");
        assert_eq!(raw.points.len(), 1);
        assert_eq!(raw.si_unit("v(in)").unwrap(), "V");
        assert_eq!(raw.si_unit("v(out)").unwrap(), "V");

        // Ideal divider: 12 V * 1 kOhm / (3 kOhm + 1 kOhm) = 3 V.
        let supply = 12.0_f64;
        let top_resistance = 3_000.0_f64;
        let bottom_resistance = 1_000.0_f64;
        let expected_vout = supply * bottom_resistance / (top_resistance + bottom_resistance);
        assert!((raw.final_value("v(in)").unwrap() - supply).abs() < 1e-12);
        assert!((expected_vout - 3.0).abs() < 1e-12);
        assert!((raw.final_value("v(out)").unwrap() - expected_vout).abs() < 1e-12);
    }

    #[test]
    fn parses_rc_transient_golden_fixture_and_units() {
        let raw = AsciiRawfile::parse(RC_FIXTURE).expect("valid ASCII fixture");
        assert_eq!(raw.title, "RC step reference");
        assert_eq!(raw.plotname, "Transient Analysis");
        assert_eq!(raw.variables.len(), 2);
        assert_eq!(raw.points.len(), 3);
        assert_eq!(raw.si_unit("time").unwrap(), "s");
        assert_eq!(raw.si_unit("v(out)").unwrap(), "V");
        let tau = 1_000.0_f64 * 1e-6_f64;
        for time in [0.0, 0.001, 0.002] {
            let expected = 1.0 - (-time / tau).exp();
            let actual = raw.value_nearest_to("time", time, "v(out)").unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "RC mismatch at t={time}: actual={actual}, expected={expected}"
            );
        }
        assert!((raw.peak_abs_value("v(out)").unwrap() - (1.0 - (-0.002 / tau).exp())).abs() < 1e-9);
    }

    #[test]
    fn rejects_missing_requested_vector() {
        let raw = AsciiRawfile::parse(RC_FIXTURE).unwrap();
        assert!(raw.final_value("v(missing)").is_err());
    }

    #[test]
    fn rejects_declared_point_count_mismatch() {
        let malformed = RC_FIXTURE.replace("No. Points: 3", "No. Points: 4");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_declared_variable_count_mismatch() {
        let malformed = RC_FIXTURE.replace("No. Variables: 2", "No. Variables: 3");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_truncated_last_point() {
        let malformed = RC_FIXTURE.replace("    8.646647167633873e-01\n", "");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_hostile_oversized_counts_before_allocating_point_storage() {
        let malformed = RC_FIXTURE.replace("No. Points: 3", "No. Points: 18446744073709551615");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_non_finite_samples() {
        let malformed = RC_FIXTURE.replace("6.321205588285577e-01", "NaN");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_complex_rawfiles() {
        let malformed = RC_FIXTURE.replace("Flags: real", "Flags: complex");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_non_sequential_point_indices() {
        let malformed = RC_FIXTURE.replace(
            " 1  1.000000000000000e-03",
            " 2  1.000000000000000e-03",
        );
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn unknown_variable_kinds_do_not_get_guessed_units() {
        let malformed = RC_FIXTURE.replace("v(out)  voltage", "v(out)  unknown");
        let raw = AsciiRawfile::parse(&malformed).unwrap();
        assert!(raw.si_unit("v(out)").is_err());
    }

    #[test]
    fn rejects_multiple_plot_sections() {
        let malformed = format!("{RC_FIXTURE}\nPlotname: AC Analysis\n");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_duplicate_variable_names() {
        let malformed = RC_FIXTURE.replace("v(out)  voltage", "time  voltage");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_duplicate_required_headers() {
        let malformed = RC_FIXTURE.replace(
            "Flags: real",
            "Flags: real\nFlags: real",
        );
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_indented_duplicate_plotname_header() {
        let malformed = RC_FIXTURE.replace(
            "Plotname: Transient Analysis",
            "Plotname: Transient Analysis\n  Plotname: conflicting analysis",
        );
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_indented_duplicate_required_header() {
        let malformed = RC_FIXTURE.replace(
            "Flags: real",
            "Flags: real\n  Flags: complex",
        );
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn rejects_non_numeric_trailing_data_after_declared_points() {
        let malformed = format!("{RC_FIXTURE}\nsolver warning: converged?\n");
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }

    #[test]
    fn nearest_sample_rejects_non_finite_target() {
        let raw = AsciiRawfile::parse(RC_FIXTURE).unwrap();
        assert!(raw.value_nearest_to("time", f64::NAN, "v(out)").is_err());
        assert!(raw.value_nearest_to("time", f64::INFINITY, "v(out)").is_err());
    }

    #[test]
    fn rejects_extra_scalar_after_a_complete_point() {
        let malformed = RC_FIXTURE.replace(
            "    0.000000000000000e+00\n 1  1.000000000000000e-03",
            "    0.000000000000000e+00\n    1.000000000000000e-01\n 1  1.000000000000000e-03",
        );
        assert!(AsciiRawfile::parse(&malformed).is_err());
    }
}
