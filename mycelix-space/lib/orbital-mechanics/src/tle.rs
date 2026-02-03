//! Two-Line Element (TLE) Parsing
//!
//! TLE is the standard format for orbital elements used by NORAD/Space-Track.
//! This module parses TLE strings into structured data for SGP4 propagation.
//!
//! # TLE Format
//! ```text
//! Line 0: ISS (ZARYA)
//! Line 1: 1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993
//! Line 2: 2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424573
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc, NaiveDate, Duration};

#[derive(Error, Debug)]
pub enum TleParseError {
    #[error("Invalid line length: expected 69 characters, got {0}")]
    InvalidLineLength(usize),

    #[error("Invalid line number: expected {expected}, got {actual}")]
    InvalidLineNumber { expected: u8, actual: u8 },

    #[error("Checksum mismatch on line {line}: expected {expected}, got {actual}")]
    ChecksumMismatch { line: u8, expected: u8, actual: u8 },

    #[error("Parse error in field '{field}': {message}")]
    FieldParseError { field: String, message: String },

    #[error("Invalid NORAD catalog number: {0}")]
    InvalidNoradId(String),
}

/// Two-Line Element set representing orbital parameters
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TwoLineElement {
    /// Satellite name (from line 0, if present)
    pub name: Option<String>,

    /// NORAD Catalog Number (5 digits)
    pub norad_id: u32,

    /// International Designator (launch year, launch number, piece)
    pub intl_designator: String,

    /// Classification (U = Unclassified, C = Classified, S = Secret)
    pub classification: char,

    /// Epoch time (when these elements are valid)
    pub epoch: DateTime<Utc>,

    /// First derivative of mean motion (rev/day²)
    pub mean_motion_dot: f64,

    /// Second derivative of mean motion (rev/day³)
    pub mean_motion_ddot: f64,

    /// BSTAR drag coefficient (1/earth radii)
    pub bstar: f64,

    /// Element set number
    pub element_set_number: u16,

    /// Inclination (degrees)
    pub inclination_deg: f64,

    /// Right Ascension of Ascending Node (degrees)
    pub raan_deg: f64,

    /// Eccentricity (dimensionless, 0-1)
    pub eccentricity: f64,

    /// Argument of Perigee (degrees)
    pub arg_of_perigee_deg: f64,

    /// Mean Anomaly (degrees)
    pub mean_anomaly_deg: f64,

    /// Mean Motion (revolutions per day)
    pub mean_motion: f64,

    /// Revolution number at epoch
    pub rev_at_epoch: u32,

    /// Raw TLE lines for passthrough to SGP4
    pub line1: String,
    pub line2: String,
}

impl TwoLineElement {
    /// Parse a TLE from two or three lines
    pub fn parse(input: &str) -> Result<Self, TleParseError> {
        let lines: Vec<&str> = input.lines().collect();

        let (name, line1, line2) = match lines.len() {
            2 => (None, lines[0], lines[1]),
            3 => (Some(lines[0].trim().to_string()), lines[1], lines[2]),
            _ => return Err(TleParseError::FieldParseError {
                field: "input".to_string(),
                message: format!("Expected 2 or 3 lines, got {}", lines.len()),
            }),
        };

        Self::parse_lines(name, line1, line2)
    }

    /// Parse from separate line strings
    pub fn parse_lines(name: Option<String>, line1: &str, line2: &str) -> Result<Self, TleParseError> {
        // Validate line lengths
        if line1.len() != 69 {
            return Err(TleParseError::InvalidLineLength(line1.len()));
        }
        if line2.len() != 69 {
            return Err(TleParseError::InvalidLineLength(line2.len()));
        }

        // Validate line numbers
        let line1_num = line1.chars().next().unwrap_or('0');
        let line2_num = line2.chars().next().unwrap_or('0');

        if line1_num != '1' {
            return Err(TleParseError::InvalidLineNumber {
                expected: 1,
                actual: line1_num as u8 - b'0',
            });
        }
        if line2_num != '2' {
            return Err(TleParseError::InvalidLineNumber {
                expected: 2,
                actual: line2_num as u8 - b'0',
            });
        }

        // Validate checksums
        Self::validate_checksum(line1, 1)?;
        Self::validate_checksum(line2, 2)?;

        // Parse Line 1 fields
        let norad_id = line1[2..7].trim().parse::<u32>()
            .map_err(|_| TleParseError::InvalidNoradId(line1[2..7].to_string()))?;

        let classification = line1.chars().nth(7).unwrap_or('U');
        let intl_designator = line1[9..17].trim().to_string();

        // Parse epoch (YYDDD.DDDDDDDD format)
        let epoch = Self::parse_epoch(&line1[18..32])?;

        // Parse mean motion derivatives
        let mean_motion_dot = Self::parse_float(&line1[33..43], "mean_motion_dot")?;
        let mean_motion_ddot = Self::parse_exponential(&line1[44..52], "mean_motion_ddot")?;
        let bstar = Self::parse_exponential(&line1[53..61], "bstar")?;

        let element_set_number = line1[64..68].trim().parse::<u16>()
            .unwrap_or(0);

        // Parse Line 2 fields
        let inclination_deg = Self::parse_float(&line2[8..16], "inclination")?;
        let raan_deg = Self::parse_float(&line2[17..25], "raan")?;

        // Eccentricity has implied decimal point
        let ecc_str = format!("0.{}", line2[26..33].trim());
        let eccentricity = ecc_str.parse::<f64>()
            .map_err(|_| TleParseError::FieldParseError {
                field: "eccentricity".to_string(),
                message: format!("Cannot parse '{}'", ecc_str),
            })?;

        let arg_of_perigee_deg = Self::parse_float(&line2[34..42], "arg_of_perigee")?;
        let mean_anomaly_deg = Self::parse_float(&line2[43..51], "mean_anomaly")?;
        let mean_motion = Self::parse_float(&line2[52..63], "mean_motion")?;

        let rev_at_epoch = line2[63..68].trim().parse::<u32>()
            .unwrap_or(0);

        Ok(TwoLineElement {
            name,
            norad_id,
            intl_designator,
            classification,
            epoch,
            mean_motion_dot,
            mean_motion_ddot,
            bstar,
            element_set_number,
            inclination_deg,
            raan_deg,
            eccentricity,
            arg_of_perigee_deg,
            mean_anomaly_deg,
            mean_motion,
            rev_at_epoch,
            line1: line1.to_string(),
            line2: line2.to_string(),
        })
    }

    /// Validate TLE checksum (modulo 10 of sum of digits, with '-' counting as 1)
    fn validate_checksum(line: &str, line_num: u8) -> Result<(), TleParseError> {
        let expected = line.chars().last()
            .and_then(|c| c.to_digit(10))
            .unwrap_or(0) as u8;

        let mut sum: u32 = 0;
        for c in line[..68].chars() {
            if let Some(d) = c.to_digit(10) {
                sum += d;
            } else if c == '-' {
                sum += 1;
            }
        }

        let actual = (sum % 10) as u8;

        if actual != expected {
            return Err(TleParseError::ChecksumMismatch {
                line: line_num,
                expected,
                actual,
            });
        }

        Ok(())
    }

    /// Parse TLE epoch format (YYDDD.DDDDDDDD)
    fn parse_epoch(s: &str) -> Result<DateTime<Utc>, TleParseError> {
        let s = s.trim();

        let year_2digit: i32 = s[0..2].parse()
            .map_err(|_| TleParseError::FieldParseError {
                field: "epoch_year".to_string(),
                message: format!("Cannot parse '{}'", &s[0..2]),
            })?;

        // Y2K handling: 00-56 = 2000-2056, 57-99 = 1957-1999
        let year = if year_2digit < 57 { 2000 + year_2digit } else { 1900 + year_2digit };

        let day_of_year: f64 = s[2..].parse()
            .map_err(|_| TleParseError::FieldParseError {
                field: "epoch_day".to_string(),
                message: format!("Cannot parse '{}'", &s[2..]),
            })?;

        let base_date = NaiveDate::from_ymd_opt(year, 1, 1)
            .ok_or_else(|| TleParseError::FieldParseError {
                field: "epoch".to_string(),
                message: format!("Invalid year: {}", year),
            })?;

        let days = (day_of_year - 1.0).floor() as i64;
        let fraction = day_of_year - (day_of_year.floor());
        let seconds = (fraction * 86400.0) as i64;

        let date = base_date + Duration::days(days);
        let datetime = date.and_hms_opt(0, 0, 0)
            .ok_or_else(|| TleParseError::FieldParseError {
                field: "epoch".to_string(),
                message: "Invalid time".to_string(),
            })?;

        Ok(DateTime::from_naive_utc_and_offset(datetime, Utc) + Duration::seconds(seconds))
    }

    /// Parse a simple float field
    fn parse_float(s: &str, field: &str) -> Result<f64, TleParseError> {
        s.trim().parse::<f64>()
            .map_err(|_| TleParseError::FieldParseError {
                field: field.to_string(),
                message: format!("Cannot parse '{}'", s.trim()),
            })
    }

    /// Parse TLE exponential format (e.g., " 12345-4" = 0.12345e-4)
    fn parse_exponential(s: &str, field: &str) -> Result<f64, TleParseError> {
        let s = s.trim();
        if s.is_empty() || s == "00000-0" || s == "00000+0" {
            return Ok(0.0);
        }

        // Find the sign of the exponent
        let exp_sign_pos = s.rfind(|c| c == '-' || c == '+');

        if let Some(pos) = exp_sign_pos {
            if pos == 0 {
                // Sign at start is mantissa sign, not exponent
                return s.parse::<f64>().map_err(|_| TleParseError::FieldParseError {
                    field: field.to_string(),
                    message: format!("Cannot parse '{}'", s),
                });
            }

            let mantissa_str = &s[..pos];
            let exp_str = &s[pos..];

            // TLE format has implied decimal point at start of mantissa
            let mantissa: f64 = format!("0.{}", mantissa_str.trim_start_matches(&['-', '+', ' '][..]))
                .parse()
                .map_err(|_| TleParseError::FieldParseError {
                    field: field.to_string(),
                    message: format!("Cannot parse mantissa '{}'", mantissa_str),
                })?;

            let exp: i32 = exp_str.parse()
                .map_err(|_| TleParseError::FieldParseError {
                    field: field.to_string(),
                    message: format!("Cannot parse exponent '{}'", exp_str),
                })?;

            let sign = if mantissa_str.starts_with('-') { -1.0 } else { 1.0 };

            Ok(sign * mantissa * 10.0_f64.powi(exp))
        } else {
            // No exponent
            Ok(0.0)
        }
    }

    /// Get orbital period in minutes
    pub fn period_minutes(&self) -> f64 {
        1440.0 / self.mean_motion  // 1440 minutes per day
    }

    /// Get semi-major axis in km (approximate)
    pub fn semi_major_axis_km(&self) -> f64 {
        const MU_EARTH: f64 = 398600.4418; // km³/s²
        let n = self.mean_motion * 2.0 * std::f64::consts::PI / 86400.0; // rad/s
        (MU_EARTH / (n * n)).powf(1.0 / 3.0)
    }

    /// Get apogee altitude in km (above Earth surface)
    pub fn apogee_km(&self) -> f64 {
        const EARTH_RADIUS: f64 = 6378.137;
        let a = self.semi_major_axis_km();
        a * (1.0 + self.eccentricity) - EARTH_RADIUS
    }

    /// Get perigee altitude in km (above Earth surface)
    pub fn perigee_km(&self) -> f64 {
        const EARTH_RADIUS: f64 = 6378.137;
        let a = self.semi_major_axis_km();
        a * (1.0 - self.eccentricity) - EARTH_RADIUS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

    #[test]
    fn test_parse_iss_tle() {
        let tle = TwoLineElement::parse(ISS_TLE).expect("Failed to parse ISS TLE");

        assert_eq!(tle.norad_id, 25544);
        assert_eq!(tle.name, Some("ISS (ZARYA)".to_string()));
        assert!((tle.inclination_deg - 51.6416).abs() < 0.0001);
        assert!((tle.eccentricity - 0.0006703).abs() < 0.0000001);
        assert!((tle.mean_motion - 15.72125391).abs() < 0.00000001);
    }

    #[test]
    fn test_orbital_parameters() {
        let tle = TwoLineElement::parse(ISS_TLE).unwrap();

        // ISS orbital period should be ~92 minutes
        let period = tle.period_minutes();
        assert!(period > 90.0 && period < 94.0, "Period: {} minutes", period);

        // ISS altitude should be in LEO range (note: synthetic TLE may vary)
        let sma = tle.semi_major_axis_km();
        let perigee = tle.perigee_km();
        let apogee = tle.apogee_km();
        // Allow 340-460 km range for variations in synthetic TLE
        assert!(perigee > 340.0 && perigee < 460.0,
                "Perigee: {} km, SMA: {} km, ecc: {}", perigee, sma, tle.eccentricity);
        assert!(apogee > 340.0 && apogee < 460.0,
                "Apogee: {} km", apogee);
    }
}
