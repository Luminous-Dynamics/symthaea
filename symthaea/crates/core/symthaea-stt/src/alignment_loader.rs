// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parquet-based alignment loader for LibriSpeech
//!
//! Loads phoneme alignments from the gilkeyio/librispeech-alignments
//! Hugging Face dataset in Parquet format.
//!
//! The dataset uses a nested schema with lists of phoneme segments per utterance.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, StructArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// A single phoneme segment with timing information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhonemeSegment {
    /// The phoneme label (e.g., "AH0", "T", "SIL")
    pub phoneme: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

/// Alignment data for a single utterance
#[derive(Debug, Clone)]
pub struct UtteranceAlignment {
    /// Utterance ID (e.g., "1272-128104-0000")
    pub id: String,
    /// Path to audio file
    pub audio_path: Option<String>,
    /// Phoneme segments with timing
    pub phonemes: Vec<PhonemeSegment>,
    /// Word-level alignments (if available)
    pub words: Option<Vec<WordSegment>>,
}

/// A single word segment
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordSegment {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// Load alignments from a Parquet file using Arrow
///
/// The gilkeyio/librispeech-alignments dataset has a nested schema:
/// - id: string (utterance ID)
/// - phonemes: list of structs with {phoneme, start, end}
///
/// Returns a HashMap mapping utterance IDs to their alignment data
pub fn load_alignments<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, UtteranceAlignment>, String> {
    let file =
        File::open(path.as_ref()).map_err(|e| format!("Failed to open parquet file: {e}"))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to create parquet reader: {e}"))?;

    // Print schema for debugging
    let schema = builder.schema();
    eprintln!("Arrow schema:");
    for field in schema.fields() {
        eprintln!("  {}: {:?}", field.name(), field.data_type());
    }

    let reader = builder
        .build()
        .map_err(|e| format!("Failed to build reader: {e}"))?;

    let mut alignments = HashMap::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("Failed to read batch: {e}"))?;

        // Get the ID column
        let id_col = batch.column_by_name("id").ok_or("Missing 'id' column")?;
        let ids = id_col.as_string::<i32>();

        // Try to find phonemes column (might be nested)
        let phonemes_col = batch.column_by_name("phonemes");

        // Debug: print available columns
        if alignments.is_empty() {
            eprintln!(
                "Batch columns: {:?}",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            );
        }

        for row_idx in 0..batch.num_rows() {
            let id = match ids.value(row_idx) {
                s if !s.is_empty() => s.to_string(),
                _ => continue,
            };

            let mut phonemes = Vec::new();

            // Try to extract phonemes from nested structure
            if let Some(col) = phonemes_col {
                if let Some(list_array) = col.as_list_opt::<i32>() {
                    // It's a list column - extract the nested phonemes
                    if !list_array.is_null(row_idx) {
                        let phoneme_list = list_array.value(row_idx);

                        // The list elements should be structs with phoneme, start, end
                        if let Some(struct_array) =
                            phoneme_list.as_any().downcast_ref::<StructArray>()
                        {
                            phonemes = extract_phonemes_from_struct(struct_array);
                        }
                    }
                }
            }

            // If phonemes column didn't work, try individual nested columns
            // The schema might have top-level phoneme/start/end columns that are lists
            if phonemes.is_empty() {
                if let (Some(ph_col), Some(start_col), Some(end_col)) = (
                    batch.column_by_name("phoneme").cloned(),
                    get_nested_column(&batch, &["phonemes", "start"])
                        .or_else(|| batch.column_by_name("start").cloned()),
                    get_nested_column(&batch, &["phonemes", "end"])
                        .or_else(|| batch.column_by_name("end").cloned()),
                ) {
                    // Try as list arrays (one list per utterance)
                    if let (Some(ph_list), Some(start_list), Some(end_list)) = (
                        ph_col.as_list_opt::<i32>(),
                        start_col.as_list_opt::<i32>(),
                        end_col.as_list_opt::<i32>(),
                    ) {
                        if !ph_list.is_null(row_idx) {
                            let ph_values = ph_list.value(row_idx);
                            let start_values = start_list.value(row_idx);
                            let end_values = end_list.value(row_idx);

                            phonemes = extract_phonemes_from_arrays(
                                &ph_values,
                                &start_values,
                                &end_values,
                            );
                        }
                    }
                    // Try as scalar arrays (flat format, same row has single phoneme)
                    else if let (Some(ph_arr), Some(start_arr), Some(end_arr)) = (
                        ph_col.as_string_opt::<i32>(),
                        start_col.as_primitive_opt::<arrow::datatypes::Float64Type>(),
                        end_col.as_primitive_opt::<arrow::datatypes::Float64Type>(),
                    ) {
                        let phoneme = ph_arr.value(row_idx).to_string();
                        let start = start_arr.value(row_idx);
                        let end = end_arr.value(row_idx);
                        phonemes.push(PhonemeSegment {
                            phoneme,
                            start,
                            end,
                        });
                    }
                }
            }

            // Sort phonemes by start time
            phonemes.sort_by(|a, b| {
                a.start
                    .partial_cmp(&b.start)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Merge with existing or insert new
            let entry = alignments
                .entry(id.clone())
                .or_insert_with(|| UtteranceAlignment {
                    id: id.clone(),
                    audio_path: None,
                    phonemes: Vec::new(),
                    words: None,
                });
            entry.phonemes.extend(phonemes);
        }
    }

    // Sort all phonemes by start time (in case of multiple batches)
    for alignment in alignments.values_mut() {
        alignment.phonemes.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    eprintln!(
        "Loaded {} utterance alignments from parquet",
        alignments.len()
    );

    // Print first alignment for debugging
    if let Some(first) = alignments.values().next() {
        eprintln!(
            "First alignment: {} has {} phonemes",
            first.id,
            first.phonemes.len()
        );
        if !first.phonemes.is_empty() {
            eprintln!("  First phoneme: {:?}", first.phonemes[0]);
            eprintln!("  Last phoneme: {:?}", first.phonemes.last().unwrap());
        }
    }

    Ok(alignments)
}

/// Helper to get a nested column by path
fn get_nested_column(batch: &arrow::record_batch::RecordBatch, path: &[&str]) -> Option<ArrayRef> {
    if path.is_empty() {
        return None;
    }
    let mut current: Option<ArrayRef> = batch.column_by_name(path[0]).map(Arc::clone);
    for name in &path[1..] {
        current = current.and_then(|arr| {
            arr.as_any()
                .downcast_ref::<StructArray>()
                .and_then(|struct_arr| struct_arr.column_by_name(name).map(Arc::clone))
        });
    }
    current
}

/// Extract phonemes from a StructArray
fn extract_phonemes_from_struct(struct_array: &StructArray) -> Vec<PhonemeSegment> {
    let mut phonemes = Vec::new();

    let phoneme_col = struct_array.column_by_name("phoneme");
    let start_col = struct_array.column_by_name("start");
    let end_col = struct_array.column_by_name("end");

    if let (Some(ph), Some(st), Some(en)) = (phoneme_col, start_col, end_col) {
        if let (Some(ph_arr), Some(st_arr), Some(en_arr)) = (
            ph.as_string_opt::<i32>(),
            st.as_primitive_opt::<arrow::datatypes::Float64Type>(),
            en.as_primitive_opt::<arrow::datatypes::Float64Type>(),
        ) {
            for i in 0..struct_array.len() {
                if !ph_arr.is_null(i) {
                    phonemes.push(PhonemeSegment {
                        phoneme: ph_arr.value(i).to_string(),
                        start: st_arr.value(i),
                        end: en_arr.value(i),
                    });
                }
            }
        }
    }

    phonemes
}

/// Extract phonemes from parallel arrays
fn extract_phonemes_from_arrays(
    ph_arr: &dyn Array,
    start_arr: &dyn Array,
    end_arr: &dyn Array,
) -> Vec<PhonemeSegment> {
    let mut phonemes = Vec::new();

    if let (Some(ph), Some(st), Some(en)) = (
        ph_arr.as_string_opt::<i32>(),
        start_arr.as_primitive_opt::<arrow::datatypes::Float64Type>(),
        end_arr.as_primitive_opt::<arrow::datatypes::Float64Type>(),
    ) {
        let len = ph.len().min(st.len()).min(en.len());
        for i in 0..len {
            if !ph.is_null(i) {
                phonemes.push(PhonemeSegment {
                    phoneme: ph.value(i).to_string(),
                    start: st.value(i),
                    end: en.value(i),
                });
            }
        }
    }

    phonemes
}

/// Convert utterance ID to LibriSpeech audio file path
///
/// LibriSpeech IDs are formatted as "speaker-chapter-utterance" (e.g., "1272-128104-0000")
/// The audio file is at: {base}/{speaker}/{chapter}/{speaker}-{chapter}-{utterance}.flac
pub fn id_to_audio_path(id: &str, base_dir: &Path) -> Option<std::path::PathBuf> {
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() != 3 {
        return None;
    }

    let speaker = parts[0];
    let chapter = parts[1];

    Some(
        base_dir
            .join(speaker)
            .join(chapter)
            .join(format!("{id}.flac")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_to_audio_path() {
        let base = Path::new("/data/LibriSpeech/dev-clean");
        let path = id_to_audio_path("1272-128104-0000", base).unwrap();
        assert_eq!(
            path.to_str().unwrap(),
            "/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
        );
    }
}
