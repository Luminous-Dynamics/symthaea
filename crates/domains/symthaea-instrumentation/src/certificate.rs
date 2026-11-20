// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical calibration-certificate claims and binding to resolved metadata.
//!
//! Parsing and claim binding establish what the verified bytes say. They do not
//! establish issuer authority or traceability-chain validity; deployments must
//! authenticate those inputs and verify referenced reference-standard evidence.

use std::fmt;

use crate::{ArtifactReference, InstrumentIdentity, Quantity, Unit};

pub const CALIBRATION_CERTIFICATE_SCHEMA: &str = "symthaea.calibration-certificate.v1";
pub const MAX_CERTIFICATE_RECORD_ID_BYTES: usize = 256;
pub const MAX_CERTIFICATE_PROVIDER_ID_BYTES: usize = 256;
pub const MAX_CERTIFICATE_PROCEDURE_ID_BYTES: usize = 256;

/// Canonical certificate claims. All fields are covered by the artifact digest;
/// the signed review receipt separately approves this exact artifact digest.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationCertificate {
    record_id: String,
    instrument: InstrumentIdentity,
    quantity: Quantity,
    unit: Unit,
    range_min_value: f64,
    range_max_value: f64,
    valid_from_ns: u64,
    valid_until_ns: u64,
    standard_uncertainty: f64,
    calibration_provider_id: String,
    calibration_procedure_id: String,
    reference_standard: ArtifactReference,
}

impl CalibrationCertificate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        record_id: impl Into<String>,
        instrument: InstrumentIdentity,
        quantity: Quantity,
        unit: Unit,
        range_min_value: f64,
        range_max_value: f64,
        valid_from_ns: u64,
        valid_until_ns: u64,
        standard_uncertainty: f64,
        calibration_provider_id: impl Into<String>,
        calibration_procedure_id: impl Into<String>,
        reference_standard: ArtifactReference,
    ) -> Result<Self, CalibrationCertificateError> {
        let record_id = canonical_token(
            record_id.into(),
            "record_id",
            MAX_CERTIFICATE_RECORD_ID_BYTES,
        )?;
        let calibration_provider_id = canonical_token(
            calibration_provider_id.into(),
            "calibration_provider_id",
            MAX_CERTIFICATE_PROVIDER_ID_BYTES,
        )?;
        let calibration_procedure_id = canonical_token(
            calibration_procedure_id.into(),
            "calibration_procedure_id",
            MAX_CERTIFICATE_PROCEDURE_ID_BYTES,
        )?;
        canonical_token(
            instrument.instrument_id().to_owned(),
            "instrument_id",
            MAX_CERTIFICATE_RECORD_ID_BYTES,
        )?;
        canonical_token(
            instrument.channel_id().to_owned(),
            "channel_id",
            MAX_CERTIFICATE_RECORD_ID_BYTES,
        )?;
        canonical_token(
            reference_standard.artifact_id().to_owned(),
            "reference_standard_artifact_id",
            MAX_CERTIFICATE_RECORD_ID_BYTES,
        )?;

        if quantity != unit.quantity() {
            return Err(CalibrationCertificateError::QuantityUnitMismatch);
        }
        if !range_min_value.is_finite() || !range_max_value.is_finite() {
            return Err(CalibrationCertificateError::NonFiniteRange);
        }
        if range_min_value > range_max_value {
            return Err(CalibrationCertificateError::InvalidRange);
        }
        if valid_from_ns >= valid_until_ns {
            return Err(CalibrationCertificateError::InvalidValidityWindow);
        }
        if !standard_uncertainty.is_finite() || standard_uncertainty < 0.0 {
            return Err(CalibrationCertificateError::InvalidUncertainty);
        }

        Ok(Self {
            record_id,
            instrument,
            quantity,
            unit,
            range_min_value: normalize_zero(range_min_value),
            range_max_value: normalize_zero(range_max_value),
            valid_from_ns,
            valid_until_ns,
            standard_uncertainty: normalize_zero(standard_uncertainty),
            calibration_provider_id,
            calibration_procedure_id,
            reference_standard,
        })
    }

    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    pub fn instrument(&self) -> &InstrumentIdentity {
        &self.instrument
    }

    pub fn quantity(&self) -> Quantity {
        self.quantity
    }

    pub fn unit(&self) -> Unit {
        self.unit
    }

    pub fn range(&self) -> (f64, f64) {
        (self.range_min_value, self.range_max_value)
    }

    pub fn validity_window_ns(&self) -> (u64, u64) {
        (self.valid_from_ns, self.valid_until_ns)
    }

    pub fn standard_uncertainty(&self) -> f64 {
        self.standard_uncertainty
    }

    pub fn calibration_provider_id(&self) -> &str {
        &self.calibration_provider_id
    }

    pub fn calibration_procedure_id(&self) -> &str {
        &self.calibration_procedure_id
    }

    pub fn reference_standard(&self) -> &ArtifactReference {
        &self.reference_standard
    }

    /// Identify the first calibration metadata claim that differs from the record
    /// constructed by a resolver. Extra traceability claims remain accessible above.
    pub fn first_metadata_mismatch(
        &self,
        record_id: &str,
        instrument: &InstrumentIdentity,
        quantity: Quantity,
        unit: Unit,
        range_min_value: f64,
        range_max_value: f64,
        valid_from_ns: u64,
        valid_until_ns: u64,
        standard_uncertainty: f64,
    ) -> Option<&'static str> {
        if self.record_id != record_id {
            return Some("record_id");
        }
        if &self.instrument != instrument {
            return Some("instrument_or_channel");
        }
        if self.quantity != quantity {
            return Some("quantity");
        }
        if self.unit != unit {
            return Some("unit");
        }
        if self.range_min_value != range_min_value {
            return Some("range_min_value");
        }
        if self.range_max_value != range_max_value {
            return Some("range_max_value");
        }
        if self.valid_from_ns != valid_from_ns {
            return Some("valid_from_ns");
        }
        if self.valid_until_ns != valid_until_ns {
            return Some("valid_until_ns");
        }
        if self.standard_uncertainty != standard_uncertainty {
            return Some("standard_uncertainty");
        }
        None
    }

    /// Canonical UTF-8 envelope with fixed field order and a final LF.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let fields = [
            CALIBRATION_CERTIFICATE_SCHEMA.to_owned(),
            format!("record_id={}", self.record_id),
            format!("instrument_id={}", self.instrument.instrument_id()),
            format!("channel_id={}", self.instrument.channel_id()),
            format!("quantity={}", quantity_token(self.quantity)),
            format!("unit={}", unit_token(self.unit)),
            format!("range_min_value={}", canonical_float(self.range_min_value)),
            format!("range_max_value={}", canonical_float(self.range_max_value)),
            format!("valid_from_ns={}", self.valid_from_ns),
            format!("valid_until_ns={}", self.valid_until_ns),
            format!(
                "standard_uncertainty={}",
                canonical_float(self.standard_uncertainty)
            ),
            format!("calibration_provider_id={}", self.calibration_provider_id),
            format!("calibration_procedure_id={}", self.calibration_procedure_id),
            format!(
                "reference_standard_artifact_id={}",
                self.reference_standard.artifact_id()
            ),
            format!(
                "reference_standard_sha256={}",
                self.reference_standard.sha256_hex()
            ),
        ];
        format!("{}\n", fields.join("\n")).into_bytes()
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CalibrationCertificateError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| CalibrationCertificateError::InvalidEncoding)?;
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() != 15 || lines[0] != CALIBRATION_CERTIFICATE_SCHEMA {
            return Err(CalibrationCertificateError::UnsupportedSchemaOrLayout);
        }

        let record_id = field(lines[1], "record_id")?;
        let instrument_id = field(lines[2], "instrument_id")?;
        let channel_id = field(lines[3], "channel_id")?;
        let quantity = parse_quantity(field(lines[4], "quantity")?)?;
        let unit = parse_unit(field(lines[5], "unit")?)?;
        let range_min_value = parse_float(field(lines[6], "range_min_value")?)?;
        let range_max_value = parse_float(field(lines[7], "range_max_value")?)?;
        let valid_from_ns = parse_u64(field(lines[8], "valid_from_ns")?)?;
        let valid_until_ns = parse_u64(field(lines[9], "valid_until_ns")?)?;
        let standard_uncertainty = parse_float(field(lines[10], "standard_uncertainty")?)?;
        let calibration_provider_id = field(lines[11], "calibration_provider_id")?;
        let calibration_procedure_id = field(lines[12], "calibration_procedure_id")?;
        let reference_standard_id = field(lines[13], "reference_standard_artifact_id")?;
        let reference_standard_digest = field(lines[14], "reference_standard_sha256")?;

        let instrument = InstrumentIdentity::new(instrument_id, channel_id)
            .map_err(|_| CalibrationCertificateError::InvalidInstrumentIdentity)?;
        let reference_standard =
            ArtifactReference::new(reference_standard_id, reference_standard_digest)
                .map_err(|_| CalibrationCertificateError::InvalidReferenceStandard)?;

        let certificate = Self::new(
            record_id,
            instrument,
            quantity,
            unit,
            range_min_value,
            range_max_value,
            valid_from_ns,
            valid_until_ns,
            standard_uncertainty,
            calibration_provider_id,
            calibration_procedure_id,
            reference_standard,
        )?;
        if certificate.to_canonical_bytes() != bytes {
            return Err(CalibrationCertificateError::NonCanonicalEncoding);
        }
        Ok(certificate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationCertificateError {
    InvalidEncoding,
    UnsupportedSchemaOrLayout,
    NonCanonicalEncoding,
    InvalidField(&'static str),
    InvalidNumber(&'static str),
    InvalidInstrumentIdentity,
    InvalidReferenceStandard,
    QuantityUnitMismatch,
    NonFiniteRange,
    InvalidRange,
    InvalidValidityWindow,
    InvalidUncertainty,
}

impl fmt::Display for CalibrationCertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CalibrationCertificateError {}

fn field<'a>(
    line: &'a str,
    name: &'static str,
) -> Result<&'a str, CalibrationCertificateError> {
    line.strip_prefix(&format!("{name}="))
        .ok_or(CalibrationCertificateError::UnsupportedSchemaOrLayout)
}

fn canonical_token(
    value: String,
    field_name: &'static str,
    maximum_bytes: usize,
) -> Result<String, CalibrationCertificateError> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > maximum_bytes
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(CalibrationCertificateError::InvalidField(field_name));
    }
    Ok(value)
}

fn parse_float(value: &str) -> Result<f64, CalibrationCertificateError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| CalibrationCertificateError::InvalidNumber("float"))?;
    if !parsed.is_finite() || canonical_float(parsed) != value {
        return Err(CalibrationCertificateError::InvalidNumber("float"));
    }
    Ok(normalize_zero(parsed))
}

fn parse_u64(value: &str) -> Result<u64, CalibrationCertificateError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| CalibrationCertificateError::InvalidNumber("u64"))?;
    if parsed.to_string() != value {
        return Err(CalibrationCertificateError::InvalidNumber("u64"));
    }
    Ok(parsed)
}

fn canonical_float(value: f64) -> String {
    normalize_zero(value).to_string()
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn quantity_token(quantity: Quantity) -> &'static str {
    match quantity {
        Quantity::Length => "Length",
        Quantity::Time => "Time",
        Quantity::Frequency => "Frequency",
        Quantity::AcousticPressure => "AcousticPressure",
        Quantity::Temperature => "Temperature",
        Quantity::ElectricalPotential => "ElectricalPotential",
        Quantity::ElectricalCurrent => "ElectricalCurrent",
        Quantity::HeartRate => "HeartRate",
        Quantity::RespiratoryRate => "RespiratoryRate",
        Quantity::OxygenSaturation => "OxygenSaturation",
        Quantity::Speed => "Speed",
        Quantity::Acceleration => "Acceleration",
        Quantity::AngularVelocity => "AngularVelocity",
        Quantity::SoundPressureLevel => "SoundPressureLevel",
        Quantity::Dimensionless => "Dimensionless",
    }
}

fn parse_quantity(value: &str) -> Result<Quantity, CalibrationCertificateError> {
    match value {
        "Length" => Ok(Quantity::Length),
        "Time" => Ok(Quantity::Time),
        "Frequency" => Ok(Quantity::Frequency),
        "AcousticPressure" => Ok(Quantity::AcousticPressure),
        "Temperature" => Ok(Quantity::Temperature),
        "ElectricalPotential" => Ok(Quantity::ElectricalPotential),
        "ElectricalCurrent" => Ok(Quantity::ElectricalCurrent),
        "HeartRate" => Ok(Quantity::HeartRate),
        "RespiratoryRate" => Ok(Quantity::RespiratoryRate),
        "OxygenSaturation" => Ok(Quantity::OxygenSaturation),
        "Speed" => Ok(Quantity::Speed),
        "Acceleration" => Ok(Quantity::Acceleration),
        "AngularVelocity" => Ok(Quantity::AngularVelocity),
        "SoundPressureLevel" => Ok(Quantity::SoundPressureLevel),
        "Dimensionless" => Ok(Quantity::Dimensionless),
        _ => Err(CalibrationCertificateError::InvalidField("quantity")),
    }
}

fn unit_token(unit: Unit) -> &'static str {
    match unit {
        Unit::Meter => "Meter",
        Unit::Centimeter => "Centimeter",
        Unit::Millimeter => "Millimeter",
        Unit::Micrometer => "Micrometer",
        Unit::Second => "Second",
        Unit::Millisecond => "Millisecond",
        Unit::Microsecond => "Microsecond",
        Unit::Hertz => "Hertz",
        Unit::Kilohertz => "Kilohertz",
        Unit::Megahertz => "Megahertz",
        Unit::Pascal => "Pascal",
        Unit::Kilopascal => "Kilopascal",
        Unit::DegreeCelsius => "DegreeCelsius",
        Unit::Kelvin => "Kelvin",
        Unit::Volt => "Volt",
        Unit::Millivolt => "Millivolt",
        Unit::Microvolt => "Microvolt",
        Unit::Ampere => "Ampere",
        Unit::Milliampere => "Milliampere",
        Unit::BeatsPerMinute => "BeatsPerMinute",
        Unit::BreathsPerMinute => "BreathsPerMinute",
        Unit::Percent => "Percent",
        Unit::OxygenSaturationPercent => "OxygenSaturationPercent",
        Unit::MeterPerSecond => "MeterPerSecond",
        Unit::MeterPerSecondSquared => "MeterPerSecondSquared",
        Unit::RadianPerSecond => "RadianPerSecond",
        Unit::DecibelRe20Micropascal => "DecibelRe20Micropascal",
        Unit::One => "One",
    }
}

fn parse_unit(value: &str) -> Result<Unit, CalibrationCertificateError> {
    match value {
        "Meter" => Ok(Unit::Meter),
        "Centimeter" => Ok(Unit::Centimeter),
        "Millimeter" => Ok(Unit::Millimeter),
        "Micrometer" => Ok(Unit::Micrometer),
        "Second" => Ok(Unit::Second),
        "Millisecond" => Ok(Unit::Millisecond),
        "Microsecond" => Ok(Unit::Microsecond),
        "Hertz" => Ok(Unit::Hertz),
        "Kilohertz" => Ok(Unit::Kilohertz),
        "Megahertz" => Ok(Unit::Megahertz),
        "Pascal" => Ok(Unit::Pascal),
        "Kilopascal" => Ok(Unit::Kilopascal),
        "DegreeCelsius" => Ok(Unit::DegreeCelsius),
        "Kelvin" => Ok(Unit::Kelvin),
        "Volt" => Ok(Unit::Volt),
        "Millivolt" => Ok(Unit::Millivolt),
        "Microvolt" => Ok(Unit::Microvolt),
        "Ampere" => Ok(Unit::Ampere),
        "Milliampere" => Ok(Unit::Milliampere),
        "BeatsPerMinute" => Ok(Unit::BeatsPerMinute),
        "BreathsPerMinute" => Ok(Unit::BreathsPerMinute),
        "Percent" => Ok(Unit::Percent),
        "OxygenSaturationPercent" => Ok(Unit::OxygenSaturationPercent),
        "MeterPerSecond" => Ok(Unit::MeterPerSecond),
        "MeterPerSecondSquared" => Ok(Unit::MeterPerSecondSquared),
        "RadianPerSecond" => Ok(Unit::RadianPerSecond),
        "DecibelRe20Micropascal" => Ok(Unit::DecibelRe20Micropascal),
        "One" => Ok(Unit::One),
        _ => Err(CalibrationCertificateError::InvalidField("unit")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certificate() -> CalibrationCertificate {
        CalibrationCertificate::new(
            "calibration-17",
            InstrumentIdentity::new("instrument-1", "channel-a").unwrap(),
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            10,
            100,
            900.0,
            "calibration-lab-01",
            "frequency-procedure-v1",
            ArtifactReference::new(
                "reference-standard-01",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn canonical_certificate_roundtrips_every_bound_claim() {
        let original = certificate();
        let bytes = original.to_canonical_bytes();
        let parsed = CalibrationCertificate::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed, original);
        assert_eq!(parsed.instrument().instrument_id(), "instrument-1");
        assert_eq!(parsed.reference_standard().artifact_id(), "reference-standard-01");
        assert_eq!(parsed.calibration_provider_id(), "calibration-lab-01");
    }

    #[test]
    fn noncanonical_field_order_and_extra_fields_are_rejected() {
        let bytes = certificate().to_canonical_bytes();
        let mut lines: Vec<&str> = std::str::from_utf8(&bytes).unwrap().lines().collect();
        lines.swap(1, 2);
        let reordered = format!("{}\n", lines.join("\n")).into_bytes();
        assert_eq!(
            CalibrationCertificate::from_canonical_bytes(&reordered).unwrap_err(),
            CalibrationCertificateError::UnsupportedSchemaOrLayout
        );
        let mut extra = bytes;
        extra.extend_from_slice(b"extra=field\n");
        assert_eq!(
            CalibrationCertificate::from_canonical_bytes(&extra).unwrap_err(),
            CalibrationCertificateError::UnsupportedSchemaOrLayout
        );
    }

    #[test]
    fn alternate_number_format_is_not_canonical() {
        let mut bytes = certificate().to_canonical_bytes();
        let original = b"range_min_value=4000000";
        let replacement = b"range_min_value=4e6";
        let offset = bytes
            .windows(original.len())
            .position(|window| window == original)
            .unwrap();
        bytes.splice(offset..offset + original.len(), replacement.iter().copied());
        assert_eq!(
            CalibrationCertificate::from_canonical_bytes(&bytes).unwrap_err(),
            CalibrationCertificateError::InvalidNumber("float")
        );
    }

    #[test]
    fn metadata_binding_names_each_changed_critical_claim() {
        let cert = certificate();
        let (minimum, maximum) = cert.range();
        let (valid_from, valid_until) = cert.validity_window_ns();
        let base = |record_id: &str,
                    instrument: &InstrumentIdentity,
                    quantity: Quantity,
                    unit: Unit,
                    range_min: f64,
                    range_max: f64,
                    start: u64,
                    end: u64,
                    uncertainty: f64| {
            cert.first_metadata_mismatch(
                record_id,
                instrument,
                quantity,
                unit,
                range_min,
                range_max,
                start,
                end,
                uncertainty,
            )
        };

        assert_eq!(
            base(
                "another-record",
                cert.instrument(),
                cert.quantity(),
                cert.unit(),
                minimum,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("record_id")
        );
        let other_instrument =
            InstrumentIdentity::new("instrument-2", "channel-a").unwrap();
        assert_eq!(
            base(
                cert.record_id(),
                &other_instrument,
                cert.quantity(),
                cert.unit(),
                minimum,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("instrument_or_channel")
        );
        assert_eq!(
            base(
                cert.record_id(),
                cert.instrument(),
                Quantity::AcousticPressure,
                Unit::Pascal,
                minimum,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("quantity")
        );
        assert_eq!(
            base(
                cert.record_id(),
                cert.instrument(),
                cert.quantity(),
                Unit::Megahertz,
                minimum,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("unit")
        );
        assert_eq!(
            base(
                cert.record_id(),
                cert.instrument(),
                cert.quantity(),
                cert.unit(),
                minimum + 1.0,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("range_min_value")
        );
        assert_eq!(
            base(
                cert.record_id(),
                cert.instrument(),
                cert.quantity(),
                cert.unit(),
                minimum,
                maximum,
                valid_from + 1,
                valid_until,
                cert.standard_uncertainty(),
            ),
            Some("valid_from_ns")
        );
        assert_eq!(
            base(
                cert.record_id(),
                cert.instrument(),
                cert.quantity(),
                cert.unit(),
                minimum,
                maximum,
                valid_from,
                valid_until,
                cert.standard_uncertainty() + 1.0,
            ),
            Some("standard_uncertainty")
        );
    }

    #[test]
    fn control_characters_in_instrument_and_reference_ids_are_rejected() {
        assert_eq!(
            CalibrationCertificate::new(
                "calibration-17",
                InstrumentIdentity::new("instrument\\nforged", "channel-a").unwrap(),
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                6_000_000.0,
                10,
                100,
                900.0,
                "lab",
                "procedure",
                certificate().reference_standard().clone(),
            )
            .unwrap_err(),
            CalibrationCertificateError::InvalidField("instrument_id")
        );

        assert_eq!(
            CalibrationCertificate::new(
                "calibration-17",
                InstrumentIdentity::new("instrument-1", "channel-a").unwrap(),
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                6_000_000.0,
                10,
                100,
                900.0,
                "lab",
                "procedure",
                ArtifactReference::new(
                    "reference\\nforged",
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                )
                .unwrap(),
            )
            .unwrap_err(),
            CalibrationCertificateError::InvalidField("reference_standard_artifact_id")
        );
    }

    #[test]
    fn incompatible_units_and_invalid_ranges_fail_closed() {
        assert_eq!(
            CalibrationCertificate::new(
                "calibration-17",
                InstrumentIdentity::new("instrument-1", "channel-a").unwrap(),
                Quantity::AcousticPressure,
                Unit::Hertz,
                0.0,
                1.0,
                10,
                100,
                0.1,
                "lab",
                "procedure",
                certificate().reference_standard().clone(),
            )
            .unwrap_err(),
            CalibrationCertificateError::QuantityUnitMismatch
        );
        assert_eq!(
            CalibrationCertificate::new(
                "calibration-17",
                InstrumentIdentity::new("instrument-1", "channel-a").unwrap(),
                Quantity::Frequency,
                Unit::Hertz,
                10.0,
                1.0,
                10,
                100,
                0.1,
                "lab",
                "procedure",
                certificate().reference_standard().clone(),
            )
            .unwrap_err(),
            CalibrationCertificateError::InvalidRange
        );
    }
}
