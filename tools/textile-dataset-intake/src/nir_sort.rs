// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict parser for the NIST NIR-SORT v2 filename convention.
//!
//! Parsed fibre ratios are nominal filename metadata. They are not spectral
//! measurements, ground-truth chemistry, or a substitute for SampleInformation.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceClass {
    KnownProvenanceDyed,
    KnownProvenanceUndyed,
    PostConsumerUnknownProvenance,
    PreConsumerUnknownProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FibreKind {
    Cotton,
    Polyester,
    SyntheticPolyamide,
    Acrylic,
    Elastane,
    OtherPolyurethane,
    RegeneratedCellulosic,
    OtherNaturalCellulosic,
    NaturalPolyamide,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NominalFibreSlot {
    /// One-letter NIR-SORT filename code.
    pub code: char,
    /// None is the filename's explicit empty-slot marker x.
    pub fibre: Option<FibreKind>,
    /// Nominal percentage encoded in the filename, not a measured result.
    pub percentage: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricForm {
    Woven,
    Knit,
    Nonwoven,
    Yarn,
    ShreddedFibres,
    Fasteners,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Instrument {
    HandheldFabritell,
    TimeGatedRaman,
    BenchtopNir,
    HandheldRaman,
    BenchtopNirStandoff,
    HandheldSortile,
    StereoOpticalMicroscope,
    GenericPlaceholder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMode {
    Absorbance,
    Colour,
    Mono,
    Polarized,
    Reflectance,
    Scattering,
    Transmittance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandoffDistance {
    HalfCentimetre,
    OneCentimetre,
    TwoCentimetres,
    ThreeCentimetres,
    FourCentimetres,
    EightCentimetres,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NirSortFileIdentity {
    pub filename: String,
    pub sample_id: String,
    pub source_class: SourceClass,
    /// Three four-character slots, including zero-valued empty slots.
    pub nominal_fibre_slots: Vec<NominalFibreSlot>,
    pub batch_lot: u8,
    pub fabric_form: FabricForm,
    pub instrument: Instrument,
    /// For standoff Q, the filename allows a single decimal replicate digit;
    /// other instruments encode a two-digit replicate number.
    pub replicate: u8,
    pub measurement_mode: MeasurementMode,
    pub standoff_distance: Option<StandoffDistance>,
    /// Stable grouping key excluding instrument, replicate, mode and distance.
    /// Keep every acquisition with this key in exactly one data partition.
    pub specimen_group_key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilenameError {
    NotUtf8,
    MissingExtension,
    UnsupportedExtension(String),
    WrongSegmentCount(usize),
    InvalidSampleId(String),
    InvalidSourceCode(char),
    InvalidFibreSlot(String),
    DuplicateFibreCode(char),
    InvalidNominalBlendTotal(u16),
    InvalidBatchLot(String),
    InvalidFabricForm(String),
    InvalidInstrument(char),
    InvalidAcquisitionCode(String),
    InvalidMeasurementMode(char),
    InvalidStandoffDistance(char),
    InvalidStandoffMode(char),
    ExtensionInstrumentMismatch { extension: String, instrument: char },
}

impl std::fmt::Display for FilenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FilenameError {}

fn digits(value: &str) -> Option<u8> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse::<u8>().ok()
}

fn source_class(code: char) -> Result<SourceClass, FilenameError> {
    match code {
        'A' => Ok(SourceClass::KnownProvenanceDyed),
        'B' => Ok(SourceClass::KnownProvenanceUndyed),
        'C' => Ok(SourceClass::PostConsumerUnknownProvenance),
        'D' => Ok(SourceClass::PreConsumerUnknownProvenance),
        other => Err(FilenameError::InvalidSourceCode(other)),
    }
}

fn fibre_kind(code: char) -> Option<FibreKind> {
    match code {
        'a' => Some(FibreKind::Cotton),
        'b' => Some(FibreKind::Polyester),
        'c' => Some(FibreKind::SyntheticPolyamide),
        'd' => Some(FibreKind::Acrylic),
        'e' => Some(FibreKind::Elastane),
        'f' => Some(FibreKind::OtherPolyurethane),
        'g' => Some(FibreKind::RegeneratedCellulosic),
        'h' => Some(FibreKind::OtherNaturalCellulosic),
        'i' => Some(FibreKind::NaturalPolyamide),
        _ => None,
    }
}

fn parse_fibre_slots(value: &str) -> Result<Vec<NominalFibreSlot>, FilenameError> {
    if value.len() != 12 || !value.is_ascii() {
        return Err(FilenameError::InvalidFibreSlot(value.to_owned()));
    }
    let mut slots = Vec::with_capacity(3);
    let mut seen = std::collections::BTreeSet::new();
    let mut total = 0_u16;

    for chunk in value.as_bytes().chunks_exact(4) {
        let code = chunk[0] as char;
        let percentage_text = std::str::from_utf8(&chunk[1..])
            .map_err(|_| FilenameError::InvalidFibreSlot(value.to_owned()))?;
        let percentage = digits(percentage_text)
            .filter(|number| *number <= 100)
            .ok_or_else(|| FilenameError::InvalidFibreSlot(value.to_owned()))?;

        let fibre = if code == 'x' {
            if percentage != 0 {
                return Err(FilenameError::InvalidFibreSlot(value.to_owned()));
            }
            None
        } else {
            let kind = fibre_kind(code)
                .ok_or_else(|| FilenameError::InvalidFibreSlot(value.to_owned()))?;
            if !seen.insert(code) {
                return Err(FilenameError::DuplicateFibreCode(code));
            }
            total += u16::from(percentage);
            Some(kind)
        };

        slots.push(NominalFibreSlot {
            code,
            fibre,
            percentage,
        });
    }

    if total != 100 {
        return Err(FilenameError::InvalidNominalBlendTotal(total));
    }
    Ok(slots)
}

fn fabric_form(value: &str) -> Result<FabricForm, FilenameError> {
    match value {
        "00" => Ok(FabricForm::Woven),
        "01" => Ok(FabricForm::Knit),
        "02" => Ok(FabricForm::Nonwoven),
        "03" => Ok(FabricForm::Yarn),
        "04" => Ok(FabricForm::ShreddedFibres),
        "05" => Ok(FabricForm::Fasteners),
        _ => Err(FilenameError::InvalidFabricForm(value.to_owned())),
    }
}

fn instrument(code: char) -> Result<Instrument, FilenameError> {
    match code {
        'F' => Ok(Instrument::HandheldFabritell),
        'G' => Ok(Instrument::TimeGatedRaman),
        'N' => Ok(Instrument::BenchtopNir),
        'P' => Ok(Instrument::HandheldRaman),
        'Q' => Ok(Instrument::BenchtopNirStandoff),
        'S' => Ok(Instrument::HandheldSortile),
        'T' => Ok(Instrument::StereoOpticalMicroscope),
        'X' => Ok(Instrument::GenericPlaceholder),
        other => Err(FilenameError::InvalidInstrument(other)),
    }
}

fn measurement_mode(code: char) -> Result<MeasurementMode, FilenameError> {
    match code {
        'A' => Ok(MeasurementMode::Absorbance),
        'C' => Ok(MeasurementMode::Colour),
        'M' => Ok(MeasurementMode::Mono),
        'P' => Ok(MeasurementMode::Polarized),
        'R' => Ok(MeasurementMode::Reflectance),
        'S' => Ok(MeasurementMode::Scattering),
        'T' => Ok(MeasurementMode::Transmittance),
        other => Err(FilenameError::InvalidMeasurementMode(other)),
    }
}

fn standoff_distance(code: char) -> Result<StandoffDistance, FilenameError> {
    match code {
        '3' => Ok(StandoffDistance::HalfCentimetre),
        '4' => Ok(StandoffDistance::OneCentimetre),
        '5' => Ok(StandoffDistance::TwoCentimetres),
        '6' => Ok(StandoffDistance::ThreeCentimetres),
        '7' => Ok(StandoffDistance::FourCentimetres),
        '8' => Ok(StandoffDistance::EightCentimetres),
        other => Err(FilenameError::InvalidStandoffDistance(other)),
    }
}

/// Parses a single NIR-SORT filename. The supplied value may be a full path.
pub fn parse_nir_sort_filename(path_or_name: &str) -> Result<NirSortFileIdentity, FilenameError> {
    let path = Path::new(path_or_name);
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(FilenameError::NotUtf8)?;
    let (stem, extension) = filename
        .rsplit_once('.')
        .ok_or(FilenameError::MissingExtension)?;
    let extension = extension.to_ascii_lowercase();
    if extension != "csv" && extension != "bmp" {
        return Err(FilenameError::UnsupportedExtension(extension));
    }

    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() != 6 {
        return Err(FilenameError::WrongSegmentCount(parts.len()));
    }

    let sample = parts[0];
    if sample.len() != 3 || !sample.is_ascii() {
        return Err(FilenameError::InvalidSampleId(sample.to_owned()));
    }
    let mut sample_chars = sample.chars();
    let source_code = sample_chars
        .next()
        .ok_or_else(|| FilenameError::InvalidSampleId(sample.to_owned()))?;
    let sample_number = sample_chars.as_str();
    if digits(sample_number).is_none() {
        return Err(FilenameError::InvalidSampleId(sample.to_owned()));
    }
    let class = source_class(source_code)?;
    let nominal_fibre_slots = parse_fibre_slots(parts[1])?;

    let batch_lot = digits(parts[2])
        .filter(|_| parts[2].len() == 2)
        .ok_or_else(|| FilenameError::InvalidBatchLot(parts[2].to_owned()))?;
    let fabric_form = fabric_form(parts[3])?;
    let instrument_code = parts[4]
        .chars()
        .next()
        .filter(|_| parts[4].len() == 1)
        .ok_or(FilenameError::InvalidInstrument('?'))?;
    let instrument = instrument(instrument_code)?;
    let acquisition = parts[5];

    let (replicate, measurement_mode, standoff_distance) =
        if instrument == Instrument::BenchtopNirStandoff {
            if acquisition.len() != 3 || !acquisition.is_ascii() {
                return Err(FilenameError::InvalidAcquisitionCode(acquisition.to_owned()));
            }
            let mut chars = acquisition.chars();
            let distance_code = chars.next().expect("three-character acquisition");
            let replicate_code = chars.next().expect("three-character acquisition");
            let mode_code = chars.next().expect("three-character acquisition");
            let distance = standoff_distance(distance_code)?;
            let replicate = digits(&replicate_code.to_string()).ok_or_else(|| {
                FilenameError::InvalidAcquisitionCode(acquisition.to_owned())
            })?;
            let mode = match mode_code {
                'A' => MeasurementMode::Absorbance,
                'R' => MeasurementMode::Reflectance,
                other => return Err(FilenameError::InvalidStandoffMode(other)),
            };
            (replicate, mode, Some(distance))
        } else {
            if acquisition.len() != 3 || !acquisition.is_ascii() {
                return Err(FilenameError::InvalidAcquisitionCode(acquisition.to_owned()));
            }
            let replicate = digits(&acquisition[..2])
                .ok_or_else(|| FilenameError::InvalidAcquisitionCode(acquisition.to_owned()))?;
            let mode_code = acquisition.chars().nth(2).expect("three-character acquisition");
            (replicate, measurement_mode(mode_code)?, None)
        };

    if instrument == Instrument::StereoOpticalMicroscope && extension != "bmp" {
        return Err(FilenameError::ExtensionInstrumentMismatch {
            extension,
            instrument: instrument_code,
        });
    }
    if instrument != Instrument::StereoOpticalMicroscope
        && instrument != Instrument::GenericPlaceholder
        && extension != "csv"
    {
        return Err(FilenameError::ExtensionInstrumentMismatch {
            extension,
            instrument: instrument_code,
        });
    }

    let specimen_group_key =
        format!("{}:{}:{}:{}", sample, parts[1], parts[2], parts[3]);

    Ok(NirSortFileIdentity {
        filename: filename.to_owned(),
        sample_id: sample.to_owned(),
        source_class: class,
        nominal_fibre_slots,
        batch_lot,
        fabric_form,
        instrument,
        replicate,
        measurement_mode,
        standoff_distance,
        specimen_group_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_benchtop_nir_and_preserves_nominal_slots() {
        let parsed = parse_nir_sort_filename(
            "Spectroscopy/NearInfrared/Benchtop_N/Absorbance/Source_A/A01_a100x000x000_00_00_N_01A.csv",
        )
        .unwrap();
        assert_eq!(parsed.sample_id, "A01");
        assert_eq!(parsed.source_class, SourceClass::KnownProvenanceDyed);
        assert_eq!(parsed.instrument, Instrument::BenchtopNir);
        assert_eq!(parsed.replicate, 1);
        assert_eq!(parsed.measurement_mode, MeasurementMode::Absorbance);
        assert_eq!(parsed.standoff_distance, None);
        assert_eq!(parsed.nominal_fibre_slots[0].fibre, Some(FibreKind::Cotton));
        assert_eq!(parsed.nominal_fibre_slots[0].percentage, 100);
        assert_eq!(parsed.nominal_fibre_slots[1].fibre, None);
        assert_eq!(parsed.specimen_group_key, "A01:a100x000x000:00:00");
    }

    #[test]
    fn parses_postconsumer_optical_sample_without_confusing_it_for_nir() {
        let parsed = parse_nir_sort_filename("C11_a052b048x000_00_01_T_04M.bmp").unwrap();
        assert_eq!(parsed.source_class, SourceClass::PostConsumerUnknownProvenance);
        assert_eq!(parsed.fabric_form, FabricForm::Knit);
        assert_eq!(parsed.instrument, Instrument::StereoOpticalMicroscope);
        assert_eq!(parsed.replicate, 4);
        assert_eq!(parsed.measurement_mode, MeasurementMode::Mono);
        assert_eq!(parsed.nominal_fibre_slots[0].percentage, 52);
        assert_eq!(parsed.nominal_fibre_slots[1].fibre, Some(FibreKind::Polyester));
    }

    #[test]
    fn parses_standoff_q_distance_as_a_separate_field() {
        let parsed = parse_nir_sort_filename("A01_a100x000x000_00_00_Q_43R.csv").unwrap();
        assert_eq!(parsed.instrument, Instrument::BenchtopNirStandoff);
        assert_eq!(parsed.replicate, 3);
        assert_eq!(parsed.measurement_mode, MeasurementMode::Reflectance);
        assert_eq!(parsed.standoff_distance, Some(StandoffDistance::OneCentimetre));
    }

    #[test]
    fn rejects_malformed_and_misleading_identity_strings() {
        assert!(matches!(
            parse_nir_sort_filename("E01_a100x000x000_00_00_N_01A.csv"),
            Err(FilenameError::InvalidSourceCode('E'))
        ));
        assert!(matches!(
            parse_nir_sort_filename("A01_a090x000x000_00_00_N_01A.csv"),
            Err(FilenameError::InvalidNominalBlendTotal(90))
        ));
        assert!(matches!(
            parse_nir_sort_filename("A01_a050a050x000_00_00_N_01A.csv"),
            Err(FilenameError::DuplicateFibreCode('a'))
        ));
        assert!(parse_nir_sort_filename("A01_a100x000x000_00_06_N_01A.csv").is_err());
        assert!(parse_nir_sort_filename("A01_a100x000x000_00_00_Q_93A.csv").is_err());
    }

    #[test]
    fn replicate_instrument_mode_and_distance_do_not_change_group_key() {
        let first = parse_nir_sort_filename("A01_a100x000x000_00_00_N_01A.csv").unwrap();
        let second = parse_nir_sort_filename("A01_a100x000x000_00_00_N_02R.csv").unwrap();
        let third = parse_nir_sort_filename("A01_a100x000x000_00_00_Q_43A.csv").unwrap();
        assert_eq!(first.specimen_group_key, second.specimen_group_key);
        assert_eq!(second.specimen_group_key, third.specimen_group_key);
    }
}
