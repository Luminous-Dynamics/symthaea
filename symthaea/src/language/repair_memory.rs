// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight repair-memory retrieval for coding generation.
//!
//! The memory file is JSONL using the repair lesson shape emitted by
//! `benchmark_coding_backends --repair-lessons-jsonl`. This module keeps the
//! first integration intentionally simple: retrieve similar successful repairs
//! by name/signature/category tokens and inject their hints into `CodeContext`.

use std::path::Path;

use serde::Deserialize;
use symthaea_core::synthesis_trait::SynthesisRequest;

pub const REPAIR_MEMORY_JSONL_ENV: &str = "SYMTHAEA_REPAIR_MEMORY_JSONL";

#[derive(Debug, Clone, Deserialize)]
pub struct RepairMemoryRecord {
    pub task_name: String,
    pub signature: String,
    pub category: String,
    pub diagnostic: String,
    pub hint: String,
    #[serde(default)]
    pub fixed_source_preview: Option<String>,
    #[serde(default)]
    pub final_backend: String,
    #[serde(default)]
    pub broca_training_record: bool,
}

pub fn repair_priors_for_request(
    request: &SynthesisRequest,
    limit: usize,
) -> Vec<(String, String)> {
    let Some(path) = std::env::var_os(REPAIR_MEMORY_JSONL_ENV) else {
        return Vec::new();
    };
    let Ok(records) = load_repair_memory(Path::new(&path)) else {
        return Vec::new();
    };
    let mut scored = records
        .into_iter()
        .filter(|record| record.broca_training_record || record.fixed_source_preview.is_some())
        .filter_map(|record| {
            let score = score_record(request, &record);
            (score > 0).then_some((score, record))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|(a, _), (b, _)| b.cmp(a));
    scored
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(idx, (_, record))| {
            (
                format!("repair_memory_{}_{}", idx, sanitize_label(&record.category)),
                format!(
                    "Past successful repair `{}` for `{}` via `{}`: {} Diagnostic: {}",
                    record.category,
                    record.signature,
                    record.final_backend,
                    record.hint,
                    record.diagnostic
                ),
            )
        })
        .collect()
}

pub fn load_repair_memory(path: &Path) -> Result<Vec<RepairMemoryRecord>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read repair memory {}: {error}", path.display()))?;
    let mut records = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record = serde_json::from_str::<RepairMemoryRecord>(trimmed).map_err(|error| {
            format!(
                "failed to parse repair memory {}:{}: {error}",
                path.display(),
                idx + 1
            )
        })?;
        records.push(record);
    }
    Ok(records)
}

fn score_record(request: &SynthesisRequest, record: &RepairMemoryRecord) -> usize {
    let request_signature = request.signature.as_deref().unwrap_or_default();
    let request_haystack = format!(
        "{} {} {}",
        request.name.to_ascii_lowercase(),
        request.purpose.to_ascii_lowercase(),
        request_signature.to_ascii_lowercase()
    );
    let record_haystack = format!(
        "{} {} {} {}",
        record.task_name.to_ascii_lowercase(),
        record.signature.to_ascii_lowercase(),
        record.category.to_ascii_lowercase(),
        record.hint.to_ascii_lowercase()
    );

    let mut score = 0;
    if !request.name.is_empty() && record.task_name == request.name {
        score += 6;
    }
    if !request_signature.is_empty() && record.signature == request_signature {
        score += 8;
    }
    if let (Some(request_return), Some(record_return)) = (
        signature_return_type(request_signature),
        signature_return_type(&record.signature),
    ) {
        if request_return == record_return {
            score += 3;
        }
    }
    if let (Some(request_params), Some(record_params)) = (
        signature_parameter_shape(request_signature),
        signature_parameter_shape(&record.signature),
    ) {
        if request_params == record_params {
            score += 2;
        }
    }
    for token in request_haystack
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 4)
    {
        if record_haystack.contains(token) {
            score += 1;
        }
    }
    score
}

fn signature_return_type(signature: &str) -> Option<String> {
    signature
        .split_once("->")
        .map(|(_, return_type)| {
            return_type
                .trim()
                .trim_end_matches('{')
                .trim()
                .to_ascii_lowercase()
        })
        .filter(|return_type| !return_type.is_empty())
}

fn signature_parameter_shape(signature: &str) -> Option<Vec<String>> {
    let params = signature
        .split_once('(')?
        .1
        .split_once(')')?
        .0
        .split(',')
        .filter_map(|param| {
            let param = param.trim();
            if param.is_empty() {
                return None;
            }
            let type_part = param
                .split_once(':')
                .map(|(_, ty)| ty)
                .unwrap_or(param)
                .trim()
                .to_ascii_lowercase();
            (!type_part.is_empty()).then_some(type_part)
        })
        .collect::<Vec<_>>();
    Some(params)
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_and_retrieves_repair_memory() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"{"task_name":"sum","signature":"fn sum(items: &[i32]) -> i32","category":"stub","diagnostic":"todo remains","hint":"replace placeholders with accumulator statements","fixed_source_preview":"pub fn sum(items: &[i32]) -> i32 { items.iter().sum() }","final_backend":"CodeGenerator","broca_training_record":true}"#,
        )
        .unwrap();
        unsafe {
            std::env::set_var(REPAIR_MEMORY_JSONL_ENV, file.path());
        }

        let request = SynthesisRequest::new("rust", "sum", "Sum integers in a slice")
            .with_signature("fn sum(items: &[i32]) -> i32");
        let priors = repair_priors_for_request(&request, 2);

        unsafe {
            std::env::remove_var(REPAIR_MEMORY_JSONL_ENV);
        }

        assert_eq!(priors.len(), 1);
        assert!(priors[0].0.contains("repair_memory"));
        assert!(priors[0].1.contains("replace placeholders"));
    }

    #[test]
    fn scores_signature_shape_and_return_type() {
        let request = SynthesisRequest::new("rust", "total", "Sum integers")
            .with_signature("fn total(values: &[i32]) -> i32");
        let close = RepairMemoryRecord {
            task_name: "sum".to_string(),
            signature: "fn sum(items: &[i32]) -> i32".to_string(),
            category: "stub".to_string(),
            diagnostic: "todo".to_string(),
            hint: "use an accumulator".to_string(),
            fixed_source_preview: Some(
                "fn sum(items: &[i32]) -> i32 { items.iter().sum() }".to_string(),
            ),
            final_backend: "CodeGenerator".to_string(),
            broca_training_record: true,
        };
        let distant = RepairMemoryRecord {
            task_name: "parse_i32".to_string(),
            signature: "fn parse_i32(raw: &str) -> Result<i32, std::num::ParseIntError>"
                .to_string(),
            category: "result".to_string(),
            diagnostic: "wrapped result".to_string(),
            hint: "return the parse result directly".to_string(),
            fixed_source_preview: Some(
                "fn parse_i32(raw: &str) -> Result<i32, std::num::ParseIntError> { raw.parse() }"
                    .to_string(),
            ),
            final_backend: "CodeGenerator".to_string(),
            broca_training_record: true,
        };

        assert!(score_record(&request, &close) > score_record(&request, &distant));
    }
}
