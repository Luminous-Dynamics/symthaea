// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio export: write compositions to WAV files or in-memory WAV bytes.

use crate::{AudioData, Composition};

/// Write a composition to a WAV file at the given path.
pub fn write_wav(path: &str, comp: &Composition) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    write_wav_to(std::io::BufWriter::new(file), comp)
}

/// Encode a composition as an in-memory WAV file (same formats as `write_wav`).
pub fn wav_bytes(comp: &Composition) -> Result<Vec<u8>, String> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    write_wav_to(&mut cursor, comp)?;
    Ok(cursor.into_inner())
}

/// Write a composition as WAV to any seekable writer.
fn write_wav_to<W: std::io::Write + std::io::Seek>(
    writer: W,
    comp: &Composition,
) -> Result<(), String> {
    match &comp.audio {
        AudioData::I16(samples) => {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: comp.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::new(writer, spec).map_err(|e| e.to_string())?;
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
            let mut w = hound::WavWriter::new(writer, spec).map_err(|e| e.to_string())?;
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
            let mut w = hound::WavWriter::new(writer, spec).map_err(|e| e.to_string())?;
            for p in samples {
                w.write_sample(p[0]).map_err(|e| e.to_string())?;
                w.write_sample(p[1]).map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_bytes_produces_valid_riff_header() {
        let comp = Composition {
            audio: AudioData::F32(vec![0.0, 0.5, -0.5, 0.25]),
            sample_rate: 44100,
            notes: Vec::new(),
            duration_secs: 4.0 / 44100.0,
            section: crate::structure::SectionType::Developmental,
        };
        let bytes = wav_bytes(&comp).expect("in-memory WAV encoding should succeed");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // 44-byte header + 4 samples * 4 bytes
        assert!(bytes.len() > 44);
    }
}
