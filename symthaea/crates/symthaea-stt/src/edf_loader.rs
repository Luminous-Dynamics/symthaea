// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EDF File Loader for Physiological Signals
//!
//! Parses European Data Format (EDF/EDF+) files commonly used for:
//! - Polysomnography (PSG) - sleep studies
//! - EEG/EOG/EMG recordings
//! - Other clinical physiological data
//!
//! Reference: https://www.edfplus.info/specs/edf.html

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// EDF file header information
#[derive(Debug, Clone)]
pub struct EdfHeader {
    /// Version (always "0")
    pub version: String,
    /// Patient identification
    pub patient_id: String,
    /// Recording identification
    pub recording_id: String,
    /// Start date (dd.mm.yy)
    pub start_date: String,
    /// Start time (hh.mm.ss)
    pub start_time: String,
    /// Number of bytes in header
    pub header_bytes: usize,
    /// Reserved field (EDF+ annotations marker)
    pub reserved: String,
    /// Number of data records
    pub num_records: usize,
    /// Duration of each record in seconds
    pub record_duration: f64,
    /// Number of signals
    pub num_signals: usize,
}

/// Signal header information
#[derive(Debug, Clone)]
pub struct SignalHeader {
    /// Signal label (e.g., "EEG Fpz-Cz")
    pub label: String,
    /// Transducer type
    pub transducer: String,
    /// Physical dimension (units)
    pub dimension: String,
    /// Physical minimum
    pub physical_min: f64,
    /// Physical maximum
    pub physical_max: f64,
    /// Digital minimum
    pub digital_min: i32,
    /// Digital maximum
    pub digital_max: i32,
    /// Prefiltering
    pub prefilter: String,
    /// Number of samples per record
    pub samples_per_record: usize,
    /// Reserved
    pub reserved: String,
}

/// A single signal extracted from the EDF file
#[derive(Debug, Clone)]
pub struct EdfSignal {
    /// Signal header info
    pub header: SignalHeader,
    /// Sample data (converted to physical units)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: f32,
}

/// Parsed EDF file
pub struct EdfFile {
    /// File header
    pub header: EdfHeader,
    /// Signal headers
    pub signal_headers: Vec<SignalHeader>,
    /// Signals by label
    signals: HashMap<String, EdfSignal>,
}

impl EdfFile {
    /// Open and parse an EDF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file =
            File::open(path.as_ref()).map_err(|e| format!("Failed to open EDF file: {e}"))?;
        let mut reader = BufReader::new(file);

        // Parse main header (256 bytes)
        let header = Self::parse_header(&mut reader)?;

        // Parse signal headers (256 bytes per signal)
        let signal_headers = Self::parse_signal_headers(&mut reader, header.num_signals)?;

        // Calculate data start position
        let data_start = header.header_bytes;

        // Read all signal data
        let signals = Self::read_signals(&mut reader, &header, &signal_headers, data_start)?;

        Ok(Self {
            header,
            signal_headers,
            signals,
        })
    }

    /// Get a signal by label
    pub fn get_signal(&self, label: &str) -> Option<&EdfSignal> {
        self.signals.get(label)
    }

    /// Get all available signal labels
    pub fn signal_labels(&self) -> Vec<&str> {
        self.signals.keys().map(|s| s.as_str()).collect()
    }

    /// Get recording duration in seconds
    pub fn duration(&self) -> f64 {
        self.header.num_records as f64 * self.header.record_duration
    }

    fn parse_header(reader: &mut BufReader<File>) -> Result<EdfHeader, String> {
        let mut buf = [0u8; 256];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read EDF header: {e}"))?;

        Ok(EdfHeader {
            version: String::from_utf8_lossy(&buf[0..8]).trim().to_string(),
            patient_id: String::from_utf8_lossy(&buf[8..88]).trim().to_string(),
            recording_id: String::from_utf8_lossy(&buf[88..168]).trim().to_string(),
            start_date: String::from_utf8_lossy(&buf[168..176]).trim().to_string(),
            start_time: String::from_utf8_lossy(&buf[176..184]).trim().to_string(),
            header_bytes: String::from_utf8_lossy(&buf[184..192])
                .trim()
                .parse()
                .unwrap_or(256),
            reserved: String::from_utf8_lossy(&buf[192..236]).trim().to_string(),
            num_records: String::from_utf8_lossy(&buf[236..244])
                .trim()
                .parse()
                .unwrap_or(0),
            record_duration: String::from_utf8_lossy(&buf[244..252])
                .trim()
                .parse()
                .unwrap_or(1.0),
            num_signals: String::from_utf8_lossy(&buf[252..256])
                .trim()
                .parse()
                .unwrap_or(0),
        })
    }

    fn parse_signal_headers(
        reader: &mut BufReader<File>,
        num_signals: usize,
    ) -> Result<Vec<SignalHeader>, String> {
        // Signal headers are stored as parallel arrays
        // Each field is num_signals * field_width bytes

        let read_field =
            |reader: &mut BufReader<File>, width: usize, n: usize| -> Result<Vec<String>, String> {
                let mut buf = vec![0u8; width * n];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| format!("Failed to read signal header field: {e}"))?;
                Ok((0..n)
                    .map(|i| {
                        String::from_utf8_lossy(&buf[i * width..(i + 1) * width])
                            .trim()
                            .to_string()
                    })
                    .collect())
            };

        let labels = read_field(reader, 16, num_signals)?;
        let transducers = read_field(reader, 80, num_signals)?;
        let dimensions = read_field(reader, 8, num_signals)?;
        let physical_mins = read_field(reader, 8, num_signals)?;
        let physical_maxs = read_field(reader, 8, num_signals)?;
        let digital_mins = read_field(reader, 8, num_signals)?;
        let digital_maxs = read_field(reader, 8, num_signals)?;
        let prefilters = read_field(reader, 80, num_signals)?;
        let samples_per_record = read_field(reader, 8, num_signals)?;
        let reserveds = read_field(reader, 32, num_signals)?;

        let mut headers = Vec::with_capacity(num_signals);
        for i in 0..num_signals {
            headers.push(SignalHeader {
                label: labels[i].clone(),
                transducer: transducers[i].clone(),
                dimension: dimensions[i].clone(),
                physical_min: physical_mins[i].parse().unwrap_or(0.0),
                physical_max: physical_maxs[i].parse().unwrap_or(0.0),
                digital_min: digital_mins[i].parse().unwrap_or(0),
                digital_max: digital_maxs[i].parse().unwrap_or(0),
                prefilter: prefilters[i].clone(),
                samples_per_record: samples_per_record[i].parse().unwrap_or(0),
                reserved: reserveds[i].clone(),
            });
        }

        Ok(headers)
    }

    fn read_signals(
        reader: &mut BufReader<File>,
        header: &EdfHeader,
        signal_headers: &[SignalHeader],
        data_start: usize,
    ) -> Result<HashMap<String, EdfSignal>, String> {
        // Seek to data section
        reader
            .seek(SeekFrom::Start(data_start as u64))
            .map_err(|e| format!("Failed to seek to data: {e}"))?;

        let _num_signals = signal_headers.len();
        let num_records = header.num_records;

        // Initialize sample buffers
        let mut all_samples: Vec<Vec<f32>> = signal_headers
            .iter()
            .map(|sh| Vec::with_capacity(sh.samples_per_record * num_records))
            .collect();

        // Read data record by record
        for _record in 0..num_records {
            for (sig_idx, sig_header) in signal_headers.iter().enumerate() {
                let n_samples = sig_header.samples_per_record;

                // Read raw 16-bit samples
                let mut buf = vec![0u8; n_samples * 2];
                if reader.read_exact(&mut buf).is_err() {
                    // End of file - this is OK, just stop reading
                    break;
                }

                // Convert to physical units
                let gain = if sig_header.digital_max != sig_header.digital_min {
                    (sig_header.physical_max - sig_header.physical_min)
                        / (sig_header.digital_max - sig_header.digital_min) as f64
                } else {
                    1.0
                };
                let offset = sig_header.physical_min - gain * sig_header.digital_min as f64;

                for i in 0..n_samples {
                    let raw = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]) as f64;
                    let physical = (raw * gain + offset) as f32;
                    all_samples[sig_idx].push(physical);
                }
            }
        }

        // Build signal map
        let mut signals = HashMap::new();
        for (i, sh) in signal_headers.iter().enumerate() {
            let sample_rate = sh.samples_per_record as f32 / header.record_duration as f32;
            signals.insert(
                sh.label.clone(),
                EdfSignal {
                    header: sh.clone(),
                    samples: all_samples[i].clone(),
                    sample_rate,
                },
            );
        }

        Ok(signals)
    }
}

/// Sleep stage annotation from hypnogram
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepStage {
    /// Awake
    Wake,
    /// Stage N1 - drowsy / light sleep
    N1,
    /// Stage N2 - light sleep with spindles
    N2,
    /// Stage N3 - deep sleep / slow wave sleep
    N3,
    /// REM sleep - dreaming
    REM,
    /// Movement artifact
    Movement,
    /// Unknown / unscored
    Unknown,
}

impl SleepStage {
    /// Parse from EDF+ annotation string
    pub fn from_annotation(s: &str) -> Self {
        let s = s.trim().to_uppercase();
        match s.as_str() {
            "W" | "WAKE" | "SLEEP STAGE W" => SleepStage::Wake,
            "1" | "N1" | "SLEEP STAGE 1" => SleepStage::N1,
            "2" | "N2" | "SLEEP STAGE 2" => SleepStage::N2,
            "3" | "4" | "N3" | "SLEEP STAGE 3" | "SLEEP STAGE 4" => SleepStage::N3,
            "R" | "REM" | "SLEEP STAGE R" => SleepStage::REM,
            "M" | "MOVEMENT" | "SLEEP STAGE M" => SleepStage::Movement,
            _ => SleepStage::Unknown,
        }
    }

    /// Is this a sleep stage (not wake or artifact)?
    pub fn is_sleep(&self) -> bool {
        matches!(
            self,
            SleepStage::N1 | SleepStage::N2 | SleepStage::N3 | SleepStage::REM
        )
    }

    /// Is this deep sleep (N3)?
    pub fn is_deep_sleep(&self) -> bool {
        matches!(self, SleepStage::N3)
    }

    /// Is this REM sleep?
    pub fn is_rem(&self) -> bool {
        matches!(self, SleepStage::REM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_stage_parsing() {
        assert_eq!(SleepStage::from_annotation("W"), SleepStage::Wake);
        assert_eq!(SleepStage::from_annotation("Sleep stage 3"), SleepStage::N3);
        assert_eq!(SleepStage::from_annotation("R"), SleepStage::REM);
        assert!(SleepStage::N3.is_deep_sleep());
        assert!(SleepStage::REM.is_rem());
    }
}
