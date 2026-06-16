// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EDF/EDF+ File Loader for Sleep-EDF Database
//!
//! Parses European Data Format files used in PhysioNet's Sleep-EDF database.
//!
//! ## Sleep-EDF File Structure
//!
//! Each recording has two files:
//! - `SC4001E0-PSG.edf` - Polysomnography signals (EEG, EOG, EMG)
//! - `SC4001EC-Hypnogram.edf` - Sleep stage annotations
//!
//! ## Sleep Stages (AASM Standard)
//!
//! - **W** (Wake): Alert, eyes open or drowsy
//! - **N1** (Stage 1): Light sleep, theta waves
//! - **N2** (Stage 2): Sleep spindles, K-complexes
//! - **N3** (Stage 3): Deep sleep, delta waves (>20%)
//! - **R** (REM): Rapid eye movement, dreaming
//!
//! ## Key Insight for Symthaea
//!
//! Different sleep stages have distinct EEG dynamics:
//! - Wake: High-frequency, low-amplitude, desynchronized
//! - Deep Sleep: Low-frequency delta waves, high synchronization
//! - REM: Wake-like activity but with rapid eye movements
//!
//! The LTC's time constants (τ) should naturally lock onto these
//! different frequency regimes, and Φ integration should capture
//! the synchronization/desynchronization patterns.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Sleep stage classification (AASM standard)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SleepStage {
    /// Wakefulness
    Wake,
    /// NREM Stage 1 - Light sleep, theta waves
    N1,
    /// NREM Stage 2 - Sleep spindles, K-complexes
    N2,
    /// NREM Stage 3 - Deep/slow wave sleep, delta waves
    N3,
    /// REM sleep - Rapid eye movement, dreaming
    REM,
    /// Movement time (artifact)
    Movement,
    /// Unknown or unscored
    Unknown,
}

impl SleepStage {
    /// Parse from Sleep-EDF annotation string
    pub fn from_annotation(s: &str) -> Self {
        let s = s.trim().to_uppercase();
        match s.as_str() {
            "SLEEP STAGE W" | "SLEEP-S0" | "W" | "STAGE W" => SleepStage::Wake,
            "SLEEP STAGE 1" | "SLEEP-S1" | "N1" | "STAGE 1" => SleepStage::N1,
            "SLEEP STAGE 2" | "SLEEP-S2" | "N2" | "STAGE 2" => SleepStage::N2,
            "SLEEP STAGE 3" | "SLEEP STAGE 4" | "SLEEP-S3" | "SLEEP-S4" | "N3" | "STAGE 3"
            | "STAGE 4" => SleepStage::N3,
            "SLEEP STAGE R" | "SLEEP-REM" | "R" | "REM" | "STAGE R" => SleepStage::REM,
            "MOVEMENT TIME" | "MOVEMENT" | "MT" => SleepStage::Movement,
            _ => SleepStage::Unknown,
        }
    }

    /// Get numeric label (0-5 for ML training)
    pub fn as_label(&self) -> u8 {
        match self {
            SleepStage::Wake => 0,
            SleepStage::N1 => 1,
            SleepStage::N2 => 2,
            SleepStage::N3 => 3,
            SleepStage::REM => 4,
            SleepStage::Movement => 5,
            SleepStage::Unknown => 6,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SleepStage::Wake => "Wake",
            SleepStage::N1 => "N1 (Light)",
            SleepStage::N2 => "N2 (Spindles)",
            SleepStage::N3 => "N3 (Deep)",
            SleepStage::REM => "REM",
            SleepStage::Movement => "Movement",
            SleepStage::Unknown => "Unknown",
        }
    }

    /// Is this a conscious state? (for binary classification)
    pub fn is_conscious(&self) -> bool {
        match self {
            SleepStage::Wake => true,
            SleepStage::REM => false, // Interesting: high activity but disconnected
            _ => false,
        }
    }

    /// Expected EEG characteristics
    pub fn expected_dynamics(&self) -> &'static str {
        match self {
            SleepStage::Wake => "High-freq alpha/beta, desynchronized, low Φ",
            SleepStage::N1 => "Theta (4-7 Hz), transitioning, moderate Φ",
            SleepStage::N2 => "Sleep spindles (12-14 Hz), K-complexes, variable Φ",
            SleepStage::N3 => "Delta (<4 Hz), high synchronization, HIGH Φ",
            SleepStage::REM => "Wake-like activity, LOW Φ (paradox!)",
            SleepStage::Movement => "Artifact-dominated",
            SleepStage::Unknown => "Unknown",
        }
    }
}

/// Sleep annotation (stage + timing)
#[derive(Debug, Clone)]
pub struct SleepAnnotation {
    /// Start time in seconds from recording start
    pub onset_sec: f64,
    /// Duration of this epoch in seconds (typically 30s)
    pub duration_sec: f64,
    /// Sleep stage
    pub stage: SleepStage,
    /// Raw annotation text
    pub raw_text: String,
}

/// Single EDF signal (channel)
#[derive(Debug, Clone)]
pub struct EdfSignal {
    /// Channel label (e.g., "EEG Fpz-Cz", "EOG horizontal")
    pub label: String,
    /// Signal type (EEG, EOG, EMG, etc.)
    pub transducer_type: String,
    /// Physical dimension (e.g., "uV")
    pub physical_dim: String,
    /// Physical minimum
    pub physical_min: f64,
    /// Physical maximum
    pub physical_max: f64,
    /// Digital minimum
    pub digital_min: i32,
    /// Digital maximum
    pub digital_max: i32,
    /// Prefiltering info
    pub prefiltering: String,
    /// Number of samples per data record
    pub samples_per_record: usize,
    /// Sample rate (computed)
    pub sample_rate: f64,
    /// Signal data in physical units (microvolts for EEG)
    pub data: Vec<f64>,
}

impl EdfSignal {
    /// Get normalized data in range [-1, 1]
    pub fn normalized(&self) -> Vec<f64> {
        if self.data.is_empty() {
            return vec![];
        }

        let min_val = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return vec![0.0; self.data.len()];
        }

        self.data
            .iter()
            .map(|&v| 2.0 * (v - min_val) / range - 1.0)
            .collect()
    }

    /// Get 30-second epoch starting at given sample index
    pub fn get_epoch(&self, start_sample: usize) -> Option<Vec<f64>> {
        let epoch_samples = (30.0 * self.sample_rate) as usize;
        if start_sample + epoch_samples > self.data.len() {
            return None;
        }
        Some(self.data[start_sample..start_sample + epoch_samples].to_vec())
    }

    /// Is this an EEG channel?
    pub fn is_eeg(&self) -> bool {
        let label = self.label.to_uppercase();
        label.contains("EEG")
            || label.contains("FPZ")
            || label.contains("PZ")
            || label.contains("CZ")
            || label.contains("OZ")
    }

    /// Is this the frontal EEG channel (Fpz-Cz)?
    pub fn is_frontal(&self) -> bool {
        let label = self.label.to_uppercase();
        label.contains("FPZ") || (label.contains("F") && label.contains("CZ"))
    }

    /// Is this the occipital EEG channel (Pz-Oz)?
    pub fn is_occipital(&self) -> bool {
        let label = self.label.to_uppercase();
        label.contains("PZ") || (label.contains("P") && label.contains("OZ"))
    }
}

/// EDF file header and data
#[derive(Debug)]
pub struct EdfFile {
    /// File version
    pub version: String,
    /// Patient identification
    pub patient_id: String,
    /// Recording identification
    pub recording_id: String,
    /// Start date (DD.MM.YY)
    pub start_date: String,
    /// Start time (hh.mm.ss)
    pub start_time: String,
    /// Number of bytes in header
    pub header_bytes: usize,
    /// Reserved field (contains EDF+ indicator)
    pub reserved: String,
    /// Number of data records
    pub num_records: i64,
    /// Duration of each data record in seconds
    pub record_duration: f64,
    /// Number of signals (channels)
    pub num_signals: usize,
    /// Signal data
    pub signals: Vec<EdfSignal>,
    /// Sleep annotations (if loaded from hypnogram)
    pub annotations: Vec<SleepAnnotation>,
}

impl EdfFile {
    /// Load EDF file from path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, crate::errors::SymthaeaError> {
        let file = File::open(path.as_ref()).map_err(|e| {
            crate::errors::SymthaeaError::Perception(format!("Cannot open file: {e}"))
        })?;
        let mut reader = BufReader::new(file);

        // Read fixed header (256 bytes)
        let mut header = [0u8; 256];
        reader.read_exact(&mut header).map_err(|e| {
            crate::errors::SymthaeaError::Perception(format!("Cannot read header: {e}"))
        })?;

        // Parse header fields
        let version = String::from_utf8_lossy(&header[0..8]).trim().to_string();
        let patient_id = String::from_utf8_lossy(&header[8..88]).trim().to_string();
        let recording_id = String::from_utf8_lossy(&header[88..168]).trim().to_string();
        let start_date = String::from_utf8_lossy(&header[168..176])
            .trim()
            .to_string();
        let start_time = String::from_utf8_lossy(&header[176..184])
            .trim()
            .to_string();
        let header_bytes: usize = String::from_utf8_lossy(&header[184..192])
            .trim()
            .parse()
            .unwrap_or(256);
        let reserved = String::from_utf8_lossy(&header[192..236])
            .trim()
            .to_string();
        let num_records: i64 = String::from_utf8_lossy(&header[236..244])
            .trim()
            .parse()
            .unwrap_or(-1);
        let record_duration: f64 = String::from_utf8_lossy(&header[244..252])
            .trim()
            .parse()
            .unwrap_or(1.0);
        let num_signals: usize = String::from_utf8_lossy(&header[252..256])
            .trim()
            .parse()
            .unwrap_or(0);

        if num_signals == 0 {
            return Err(crate::errors::SymthaeaError::Perception(
                "No signals in file".into(),
            ));
        }

        // Read signal headers (256 bytes per signal)
        let signal_header_size = 256 * num_signals;
        let mut signal_header = vec![0u8; signal_header_size];
        reader.read_exact(&mut signal_header).map_err(|e| {
            crate::errors::SymthaeaError::Perception(format!("Cannot read signal headers: {e}"))
        })?;

        // Parse each signal's header fields
        // Fields are stored in sequence: all labels, then all transducers, etc.
        let mut signals = Vec::with_capacity(num_signals);

        for i in 0..num_signals {
            // Each field is at: field_offset + i * field_size
            let label = String::from_utf8_lossy(&signal_header[i * 16..(i + 1) * 16])
                .trim()
                .to_string();
            let transducer = String::from_utf8_lossy(
                &signal_header[16 * num_signals + i * 80..16 * num_signals + (i + 1) * 80],
            )
            .trim()
            .to_string();
            let physical_dim = String::from_utf8_lossy(
                &signal_header[96 * num_signals + i * 8..96 * num_signals + (i + 1) * 8],
            )
            .trim()
            .to_string();
            let physical_min: f64 = String::from_utf8_lossy(
                &signal_header[104 * num_signals + i * 8..104 * num_signals + (i + 1) * 8],
            )
            .trim()
            .parse()
            .unwrap_or(-3200.0);
            let physical_max: f64 = String::from_utf8_lossy(
                &signal_header[112 * num_signals + i * 8..112 * num_signals + (i + 1) * 8],
            )
            .trim()
            .parse()
            .unwrap_or(3200.0);
            let digital_min: i32 = String::from_utf8_lossy(
                &signal_header[120 * num_signals + i * 8..120 * num_signals + (i + 1) * 8],
            )
            .trim()
            .parse()
            .unwrap_or(-32768);
            let digital_max: i32 = String::from_utf8_lossy(
                &signal_header[128 * num_signals + i * 8..128 * num_signals + (i + 1) * 8],
            )
            .trim()
            .parse()
            .unwrap_or(32767);
            let prefiltering = String::from_utf8_lossy(
                &signal_header[136 * num_signals + i * 80..136 * num_signals + (i + 1) * 80],
            )
            .trim()
            .to_string();
            let samples_per_record: usize = String::from_utf8_lossy(
                &signal_header[216 * num_signals + i * 8..216 * num_signals + (i + 1) * 8],
            )
            .trim()
            .parse()
            .unwrap_or(0);

            let sample_rate = if record_duration > 0.0 {
                samples_per_record as f64 / record_duration
            } else {
                100.0
            };

            signals.push(EdfSignal {
                label,
                transducer_type: transducer,
                physical_dim,
                physical_min,
                physical_max,
                digital_min,
                digital_max,
                prefiltering,
                samples_per_record,
                sample_rate,
                data: Vec::new(),
            });
        }

        // Seek to data start
        reader
            .seek(SeekFrom::Start(header_bytes as u64))
            .map_err(|e| {
                crate::errors::SymthaeaError::Perception(format!("Cannot seek to data: {e}"))
            })?;

        // Read data records
        let actual_records = if num_records < 0 {
            // Calculate from file size
            let metadata = std::fs::metadata(path.as_ref()).map_err(|e| {
                crate::errors::SymthaeaError::Perception(format!("Cannot get file size: {e}"))
            })?;
            let data_bytes = metadata.len() as i64 - header_bytes as i64;
            let bytes_per_record: i64 = signals
                .iter()
                .map(|s| s.samples_per_record as i64 * 2)
                .sum();
            if data_bytes <= 0 || bytes_per_record <= 0 {
                0
            } else {
                (data_bytes / bytes_per_record) as usize
            }
        } else {
            num_records.max(0) as usize
        };

        // Pre-allocate signal data
        for signal in &mut signals {
            signal
                .data
                .reserve(signal.samples_per_record * actual_records);
        }

        // Read and convert data
        for _record in 0..actual_records {
            for signal in &mut signals {
                let mut sample_buf = vec![0u8; signal.samples_per_record * 2];
                if reader.read_exact(&mut sample_buf).is_err() {
                    break;
                }

                // Convert 16-bit signed integers to physical values
                let scale = (signal.physical_max - signal.physical_min)
                    / (signal.digital_max - signal.digital_min) as f64;
                let offset = signal.physical_min - signal.digital_min as f64 * scale;

                for chunk in sample_buf.chunks_exact(2) {
                    let raw = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let physical = raw as f64 * scale + offset;
                    signal.data.push(physical);
                }
            }
        }

        Ok(EdfFile {
            version,
            patient_id,
            recording_id,
            start_date,
            start_time,
            header_bytes,
            reserved,
            num_records: actual_records as i64,
            record_duration,
            num_signals,
            signals,
            annotations: Vec::new(),
        })
    }

    /// Load hypnogram annotations from EDF+ annotation file
    pub fn load_hypnogram<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), crate::errors::SymthaeaError> {
        let file = File::open(path.as_ref()).map_err(|e| {
            crate::errors::SymthaeaError::Perception(format!("Cannot open hypnogram: {e}"))
        })?;
        let mut reader = BufReader::new(file);

        // Read entire file (hypnograms are small)
        let mut contents = Vec::new();
        reader.read_to_end(&mut contents).map_err(|e| {
            crate::errors::SymthaeaError::Perception(format!("Cannot read hypnogram: {e}"))
        })?;

        // Parse EDF+ annotations
        // Format: +onset\x14duration\x14annotation\x14\x00 (repeated)
        self.annotations.clear();

        // Find annotation signal (usually labeled "EDF Annotations")
        let content_str = String::from_utf8_lossy(&contents);

        // Simple regex-free parsing of Sleep-EDF format
        let mut current_time: f64 = 0.0;
        let epoch_duration = 30.0; // Standard 30-second epochs

        for line in content_str.split('\x14') {
            let line = line.trim_matches(char::from(0));
            if line.is_empty() {
                continue;
            }

            // Check for time offset (+seconds)
            if let Some(stripped) = line.strip_prefix('+') {
                if let Ok(time) = stripped.parse::<f64>() {
                    current_time = time;
                }
                continue;
            }

            // Check for sleep stage annotation
            let stage = SleepStage::from_annotation(line);
            if stage != SleepStage::Unknown {
                self.annotations.push(SleepAnnotation {
                    onset_sec: current_time,
                    duration_sec: epoch_duration,
                    stage,
                    raw_text: line.to_string(),
                });
                current_time += epoch_duration;
            }
        }

        Ok(())
    }

    /// Get frontal EEG channel (Fpz-Cz)
    pub fn frontal_eeg(&self) -> Option<&EdfSignal> {
        self.signals.iter().find(|s| s.is_frontal())
    }

    /// Get occipital EEG channel (Pz-Oz)
    pub fn occipital_eeg(&self) -> Option<&EdfSignal> {
        self.signals.iter().find(|s| s.is_occipital())
    }

    /// Get all EEG channels
    pub fn eeg_channels(&self) -> Vec<&EdfSignal> {
        self.signals.iter().filter(|s| s.is_eeg()).collect()
    }

    /// Get total duration in seconds
    pub fn duration_sec(&self) -> f64 {
        self.num_records as f64 * self.record_duration
    }

    /// Get epoch with label at given index
    pub fn get_labeled_epoch(&self, epoch_idx: usize) -> Option<(Vec<f64>, Vec<f64>, SleepStage)> {
        let frontal = self.frontal_eeg()?;
        let occipital = self.occipital_eeg()?;
        let annotation = self.annotations.get(epoch_idx)?;

        let start_sample = (annotation.onset_sec * frontal.sample_rate).max(0.0) as usize;
        let frontal_data = frontal.get_epoch(start_sample)?;
        let occipital_data = occipital.get_epoch(start_sample)?;

        Some((frontal_data, occipital_data, annotation.stage))
    }

    /// Get number of labeled epochs
    pub fn num_epochs(&self) -> usize {
        self.annotations.len()
    }

    /// Print summary
    pub fn summary(&self) -> String {
        let mut s = format!(
            "EDF File Summary:\n\
             Patient: {}\n\
             Recording: {}\n\
             Date: {} {}\n\
             Duration: {:.1} hours\n\
             Signals: {}\n",
            self.patient_id,
            self.recording_id,
            self.start_date,
            self.start_time,
            self.duration_sec() / 3600.0,
            self.num_signals
        );

        for signal in &self.signals {
            s.push_str(&format!(
                "  - {} ({:.1} Hz, {} samples)\n",
                signal.label,
                signal.sample_rate,
                signal.data.len()
            ));
        }

        if !self.annotations.is_empty() {
            s.push_str(&format!(
                "\nAnnotations: {} epochs\n",
                self.annotations.len()
            ));

            // Count stages
            let mut counts = [0usize; 7];
            for ann in &self.annotations {
                counts[ann.stage.as_label() as usize] += 1;
            }

            for (i, &count) in counts.iter().enumerate() {
                if count > 0 {
                    let stage = match i {
                        0 => SleepStage::Wake,
                        1 => SleepStage::N1,
                        2 => SleepStage::N2,
                        3 => SleepStage::N3,
                        4 => SleepStage::REM,
                        5 => SleepStage::Movement,
                        _ => SleepStage::Unknown,
                    };
                    s.push_str(&format!(
                        "  {}: {} ({:.1}%)\n",
                        stage.name(),
                        count,
                        100.0 * count as f64 / self.annotations.len() as f64
                    ));
                }
            }
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_stage_parsing() {
        assert_eq!(
            SleepStage::from_annotation("Sleep stage W"),
            SleepStage::Wake
        );
        assert_eq!(SleepStage::from_annotation("Sleep stage 1"), SleepStage::N1);
        assert_eq!(SleepStage::from_annotation("Sleep stage 2"), SleepStage::N2);
        assert_eq!(SleepStage::from_annotation("Sleep stage 3"), SleepStage::N3);
        assert_eq!(SleepStage::from_annotation("Sleep stage 4"), SleepStage::N3); // 3&4 merged
        assert_eq!(
            SleepStage::from_annotation("Sleep stage R"),
            SleepStage::REM
        );
        assert_eq!(
            SleepStage::from_annotation("Movement time"),
            SleepStage::Movement
        );
    }

    #[test]
    fn test_stage_labels() {
        assert_eq!(SleepStage::Wake.as_label(), 0);
        assert_eq!(SleepStage::N1.as_label(), 1);
        assert_eq!(SleepStage::N2.as_label(), 2);
        assert_eq!(SleepStage::N3.as_label(), 3);
        assert_eq!(SleepStage::REM.as_label(), 4);
    }

    #[test]
    fn test_consciousness_classification() {
        assert!(SleepStage::Wake.is_conscious());
        assert!(!SleepStage::N1.is_conscious());
        assert!(!SleepStage::N2.is_conscious());
        assert!(!SleepStage::N3.is_conscious());
        // REM is the interesting case: high activity but NOT conscious
        assert!(!SleepStage::REM.is_conscious());
    }
}
