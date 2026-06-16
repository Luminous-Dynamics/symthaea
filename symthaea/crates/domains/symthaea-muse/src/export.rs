// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio export: write compositions to WAV files.

use crate::{AudioData, Composition};

/// Write a composition to a WAV file at the given path.
pub fn write_wav(path: &str, comp: &Composition) -> Result<(), String> {
    match &comp.audio {
        AudioData::I16(samples) => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: comp.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
            for &s in samples {
                w.write_sample(s).map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
        }
        AudioData::F32(samples) => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: comp.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
            for &s in samples {
                w.write_sample(s).map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
        }
        AudioData::StereoF32(samples) => {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: comp.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
            for p in samples {
                w.write_sample(p[0]).map_err(|e| e.to_string())?;
                w.write_sample(p[1]).map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
