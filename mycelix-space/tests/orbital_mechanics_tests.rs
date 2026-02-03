//! Tests for the orbital-mechanics library
//!
//! Tests TLE parsing, orbital parameters, and basic operations.

use orbital_mechanics::tle::TwoLineElement;
use pretty_assertions::assert_eq;

/// Sample ISS TLE for testing
const ISS_TLE: &str = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

/// Sample debris TLE
const DEBRIS_TLE: &str = "COSMOS 1408 DEB
1 49863U 82092AXB 24001.50000000  .00045432  00000+0  32100-2 0  9998
2 49863  82.5651 166.1250 0084321 345.6789  14.1234 14.85123456123459";

/// Test TLE parsing with ISS data
#[test]
fn test_parse_iss_tle() {
    let tle = TwoLineElement::parse(ISS_TLE).expect("Failed to parse ISS TLE");

    assert_eq!(tle.norad_id, 25544);
    assert_eq!(tle.name, Some("ISS (ZARYA)".to_string()));
    assert!((tle.inclination_deg - 51.6416).abs() < 0.0001);
    assert!((tle.eccentricity - 0.0006703).abs() < 0.0000001);
    assert!((tle.mean_motion - 15.72125391).abs() < 0.00000001);
    assert!((tle.raan_deg - 247.4627).abs() < 0.0001);
    assert!((tle.arg_of_perigee_deg - 130.5360).abs() < 0.0001);
    assert!((tle.mean_anomaly_deg - 325.0288).abs() < 0.0001);
}

/// Test TLE parsing from separate lines
#[test]
fn test_parse_lines() {
    let line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424577";

    let tle = TwoLineElement::parse_lines(Some("ISS".to_string()), line1, line2)
        .expect("Failed to parse TLE lines");

    assert_eq!(tle.norad_id, 25544);
    assert_eq!(tle.name, Some("ISS".to_string()));
}

/// Test orbital parameter calculations
#[test]
fn test_orbital_parameters() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // ISS orbital period should be ~92 minutes
    let period = tle.period_minutes();
    assert!(period > 90.0 && period < 94.0, "Period: {} minutes", period);

    // Semi-major axis should be ~6730-6800 km (Earth radius + altitude)
    let sma = tle.semi_major_axis_km();
    assert!(sma > 6600.0 && sma < 6900.0, "SMA: {} km", sma);

    // ISS altitude should be in LEO range (340-460 km for synthetic TLE)
    let perigee = tle.perigee_km();
    let apogee = tle.apogee_km();
    assert!(perigee > 340.0 && perigee < 460.0, "Perigee: {} km", perigee);
    assert!(apogee > 340.0 && apogee < 460.0, "Apogee: {} km", apogee);
}

/// Test different TLE sources
#[test]
fn test_debris_tle() {
    let tle = TwoLineElement::parse(DEBRIS_TLE).expect("Failed to parse debris TLE");

    assert_eq!(tle.norad_id, 49863);
    assert!((tle.inclination_deg - 82.5651).abs() < 0.0001);

    // Debris typically has higher eccentricity
    assert!(tle.eccentricity > 0.001);
}

/// Test TLE checksum validation
#[test]
fn test_tle_checksum() {
    // Valid TLE should parse
    let valid = TwoLineElement::parse(ISS_TLE);
    assert!(valid.is_ok());

    // Corrupted checksum should fail
    let corrupted = "ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391424573";

    let result = TwoLineElement::parse(corrupted);
    assert!(result.is_err());
}

/// Test TLE serialization round-trip
#[test]
fn test_tle_serialization() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    let json = serde_json::to_string(&tle).unwrap();
    let parsed: TwoLineElement = serde_json::from_str(&json).unwrap();

    assert_eq!(tle.norad_id, parsed.norad_id);
    assert_eq!(tle.inclination_deg, parsed.inclination_deg);
    assert_eq!(tle.mean_motion, parsed.mean_motion);
}

/// Test epoch parsing
#[test]
fn test_tle_epoch() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // Epoch should be 2024-001.5 (January 1, 2024, 12:00 UTC)
    assert_eq!(tle.epoch.format("%Y").to_string(), "2024");
    assert_eq!(tle.epoch.format("%m").to_string(), "01");
    assert_eq!(tle.epoch.format("%d").to_string(), "01");
}

/// Test international designator parsing
#[test]
fn test_intl_designator() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // ISS international designator is 98067A
    assert_eq!(tle.intl_designator.trim(), "98067A");
}

/// Test classification
#[test]
fn test_classification() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // Public TLEs should be unclassified
    assert_eq!(tle.classification, 'U');
}

/// Test BSTAR drag coefficient
#[test]
fn test_bstar() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // BSTAR should be positive for objects experiencing drag
    assert!(tle.bstar > 0.0);
    assert!(tle.bstar < 1.0);
}

/// Test element set number
#[test]
fn test_element_set_number() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // Element set number should be positive
    assert!(tle.element_set_number > 0);
}

/// Test revolution number at epoch
#[test]
fn test_rev_at_epoch() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    // ISS has been in orbit since 1998, so rev number should be high
    assert!(tle.rev_at_epoch > 10000);
}

/// Test raw line preservation
#[test]
fn test_raw_lines() {
    let tle = TwoLineElement::parse(ISS_TLE).unwrap();

    assert!(tle.line1.starts_with("1 25544"));
    assert!(tle.line2.starts_with("2 25544"));
    assert_eq!(tle.line1.len(), 69);
    assert_eq!(tle.line2.len(), 69);
}
