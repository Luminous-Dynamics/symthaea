// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible projection probe for the existing Glyph/HDC substrate.
//!
//! This binary retrieves the nearest existing modality and glyph vectors for
//! each JUNG-001C synthetic input. It does not interpret the user, diagnose,
//! validate Jungian theory, or produce psychological claims.

#[cfg(feature = "glyph_codex")]
use serde_json::{Value, json};
#[cfg(feature = "glyph_codex")]
use sha2::{Digest, Sha256};
#[cfg(feature = "glyph_codex")]
use std::path::PathBuf;
#[cfg(feature = "glyph_codex")]
use symthaea::hdc::glyph_basis::{
    FieldModality, GlyphBasis, GlyphRegistry, N_FIELD_MODALITIES,
};
#[cfg(feature = "glyph_codex")]
use symthaea::hdc::moral_text_encoder::TextHdcEncoder;

#[cfg(feature = "glyph_codex")]
const DIMENSION: usize = 16_384;
#[cfg(feature = "glyph_codex")]
const EXPECTED_CASES: usize = 14;

#[cfg(feature = "glyph_codex")]
#[derive(Debug)]
struct Args {
    corpus: PathBuf,
    output: PathBuf,
    source_revision: String,
    glyph_source_blob: String,
    registry_source_blob: String,
    encoder_source_blob: String,
}

#[cfg(feature = "glyph_codex")]
fn valid_git_id(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(feature = "glyph_codex")]
fn parse_args() -> Result<Args, String> {
    let mut corpus = None;
    let mut output = None;
    let mut source_revision = None;
    let mut glyph_source_blob = None;
    let mut registry_source_blob = None;
    let mut encoder_source_blob = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            return Err(
                "usage: jungian_glyph_projection_probe --corpus PATH --output PATH \
                 --source-revision GIT_ID --glyph-source-blob GIT_BLOB \
                 --registry-source-blob GIT_BLOB --encoder-source-blob GIT_BLOB".to_string(),
            );
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--source-revision" => source_revision = Some(value),
            "--glyph-source-blob" => glyph_source_blob = Some(value),
            "--registry-source-blob" => registry_source_blob = Some(value),
            "--encoder-source-blob" => encoder_source_blob = Some(value),
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    let args = Args {
        corpus: corpus.ok_or("--corpus is required")?,
        output: output.ok_or("--output is required")?,
        source_revision: source_revision.ok_or("--source-revision is required")?,
        glyph_source_blob: glyph_source_blob.ok_or("--glyph-source-blob is required")?,
        registry_source_blob: registry_source_blob.ok_or("--registry-source-blob is required")?,
        encoder_source_blob: encoder_source_blob.ok_or("--encoder-source-blob is required")?,
    };
    for (name, id) in [
        ("source revision", args.source_revision.as_str()),
        ("GlyphBasis blob", args.glyph_source_blob.as_str()),
        ("glyph registry blob", args.registry_source_blob.as_str()),
        ("text encoder blob", args.encoder_source_blob.as_str()),
    ] {
        if !valid_git_id(id) {
            return Err(format!("{name} must be a full lowercase 40- or 64-character Git object ID"));
        }
    }
    Ok(args)
}

#[cfg(feature = "glyph_codex")]
fn required_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing/empty string field: {field}"))
}

#[cfg(feature = "glyph_codex")]
fn run() -> Result<(), String> {
    let args = parse_args()?;
    let corpus_bytes = std::fs::read(&args.corpus)
        .map_err(|error| format!("read corpus {}: {error}", args.corpus.display()))?;
    let corpus: Value = serde_json::from_slice(&corpus_bytes)
        .map_err(|error| format!("parse corpus JSON: {error}"))?;

    if required_str(&corpus, "schema")? != "symthaea-jungian-symbolic-probe-v1" {
        return Err("unexpected corpus schema".into());
    }
    if required_str(&corpus, "corpus_id")? != "JUNG-001C-DEV-001" {
        return Err("unexpected corpus ID".into());
    }
    if corpus.get("version").and_then(Value::as_u64) != Some(1) {
        return Err("only the frozen corpus version 1 is supported".into());
    }
    if required_str(&corpus, "status")? != "DevelopmentProbe"
        || required_str(&corpus, "split")? != "ExposedDevelopment"
    {
        return Err("refusing to treat non-development data as this probe".into());
    }
    let cases = corpus
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("corpus cases must be an array")?;
    if cases.len() != EXPECTED_CASES {
        return Err(format!("expected {EXPECTED_CASES} cases, found {}", cases.len()));
    }

    let mut case_ids = std::collections::HashSet::new();
    for case in cases {
        let id = required_str(case, "id")?;
        required_str(case, "input_text")?;
        if !case_ids.insert(id.to_string()) {
            return Err(format!("duplicate case ID: {id}"));
        }
    }

    let corpus_blake3 = blake3::hash(&corpus_bytes).to_hex().to_string();
    let corpus_sha256 = format!("{:x}", Sha256::digest(&corpus_bytes));
    let basis = GlyphBasis::new(DIMENSION);
    let registry = GlyphRegistry::new(DIMENSION);
    if registry.len() != 70 {
        return Err(format!("expected 70 glyphs; got {}", registry.len()));
    }
    let encoder = TextHdcEncoder::with_sentiment(DIMENSION, 3, 0.5, 0.2);

    let mut rows = Vec::with_capacity(cases.len());
    let mut encoded_inputs: Vec<(String, symthaea::hdc::ContinuousHV)> = Vec::with_capacity(cases.len());
    for case in cases {
        let case_id = required_str(case, "id")?;
        let text = required_str(case, "input_text")?;
        let hv = encoder.encode(text);
        let coords = basis.project(&hv);
        if coords.iter().any(|score| !score.is_finite() || !(-1.0..=1.0).contains(score)) {
            return Err(format!("non-finite/out-of-range modality score for {case_id}"));
        }

        let mut modality_order: Vec<usize> = (0..N_FIELD_MODALITIES).collect();
        modality_order.sort_by(|left, right| {
            coords[*right]
                .partial_cmp(&coords[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        let modality_scores: Vec<Value> = FieldModality::ALL
            .iter()
            .enumerate()
            .map(|(index, modality)| {
                json!({
                    "index": index,
                    "modality": modality.name(),
                    "cosine_similarity": coords[index],
                })
            })
            .collect();
        let top_modalities: Vec<Value> = modality_order
            .iter()
            .take(5)
            .map(|index| {
                json!({
                    "rank": modality_order.iter().position(|candidate| candidate == index).unwrap_or(0) + 1,
                    "index": index,
                    "modality": FieldModality::ALL[*index].name(),
                    "cosine_similarity": coords[*index],
                })
            })
            .collect();

        let mut glyph_rankings: Vec<(&symthaea::hdc::glyph_basis::GlyphEntry, f32)> = registry
            .entries
            .iter()
            .map(|entry| (entry, hv.similarity(&entry.encoding)))
            .collect();
        if glyph_rankings.iter().any(|(_, score)| !score.is_finite()) {
            return Err(format!("non-finite glyph retrieval score for {case_id}"));
        }
        glyph_rankings.sort_by(|(left_entry, left_score), (right_entry, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_entry.id.cmp(&right_entry.id))
        });
        let glyph_scores: Vec<Value> = glyph_rankings
            .iter()
            .enumerate()
            .map(|(rank, (entry, score))| {
                json!({
                    "rank": rank + 1,
                    "glyph_id": entry.id,
                    "name": entry.name,
                    "class": format!("{:?}", entry.class),
                    "field_modality": entry.field_modality.name(),
                    "cosine_similarity": score,
                })
            })
            .collect();
        let top_glyphs: Vec<Value> = glyph_rankings
            .iter()
            .take(5)
            .enumerate()
            .map(|(rank, (entry, score))| {
                json!({
                    "rank": rank + 1,
                    "glyph_id": entry.id,
                    "name": entry.name,
                    "class": format!("{:?}", entry.class),
                    "field_modality": entry.field_modality.name(),
                    "cosine_similarity": score,
                    "core_function": entry.core_function,
                    "echo_phrase": entry.echo_phrase,
                })
            })
            .collect();

        rows.push(json!({
            "case_id": case_id,
            "input_blake3": blake3::hash(text.as_bytes()).to_hex().to_string(),
            "input_sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
            "modality_scores": modality_scores,
            "top_modalities": top_modalities,
            "glyph_scores": glyph_scores,
            "top_glyphs": top_glyphs,
        }));
        encoded_inputs.push((case_id.to_string(), hv));
    }

    let mut pairwise_input_similarity = Vec::with_capacity(encoded_inputs.len() * (encoded_inputs.len() - 1) / 2);
    for left_index in 0..encoded_inputs.len() {
        for right_index in (left_index + 1)..encoded_inputs.len() {
            let (left_id, left_hv) = &encoded_inputs[left_index];
            let (right_id, right_hv) = &encoded_inputs[right_index];
            let similarity = left_hv.similarity(right_hv) as f64;
            if !similarity.is_finite() || !(-1.0..=1.0).contains(&similarity) {
                return Err(format!("invalid whole-text input similarity for {left_id} vs {right_id}"));
            }
            pairwise_input_similarity.push(json!({
                "left_case_id": left_id,
                "right_case_id": right_id,
                "cosine_similarity": similarity,
            }));
        }
    }
    if pairwise_input_similarity.len() != EXPECTED_CASES * (EXPECTED_CASES - 1) / 2 {
        return Err("pairwise input similarity matrix is incomplete".into());
    }

    let output = json!({
        "schema": "symthaea-jungian-glyph-projection-v1",
        "corpus_id": "JUNG-001C-DEV-001",
        "corpus_version": 1,
        "corpus_status": "DevelopmentProbe",
        "corpus_split": "ExposedDevelopment",
        "corpus_blake3": corpus_blake3,
        "corpus_sha256": corpus_sha256,
        "source_revision": args.source_revision,
        "glyph_source_blob": args.glyph_source_blob,
        "glyph_registry_source_blob": args.registry_source_blob,
        "encoder_source_blob": args.encoder_source_blob,
        "dimension": DIMENSION,
        "registry_size": registry.len(),
        "encoder_profile": {
            "type": "TextHdcEncoder",
            "ngram_size": 3,
            "trigram_weight": 0.5,
            "sentiment_weight": 0.2,
            "seed_profile": "the exact committed TextHdcEncoder implementation",
        },
        "model_scope": "Existing general-purpose Glyph Codex substrate; not a dedicated Jungian model or validated archetype detector.",
        "basis_profile": "GlyphBasis::new over the 11 committed MODALITY_KEYWORDS; vectors use TextHdcEncoder with 0.5 trigram and 0.2 sentiment weights.",
        "registry_profile": "GlyphRegistry::new encodes only name + core_function + echo_phrase; relational_archetype, somatic_practice, sonic_signature, and field_modality are not part of the embedding input text.",
        "retrieval": "cosine similarity only; nearest-neighbor association is not an interpretation or truth score",
        "claim_ceiling": "Representation-level projection/retrieval only. No validated archetype, unconscious-content, clinical, subjective-experience, or consciousness claim.",
        "pairwise_input_similarity_profile": "Cosine between whole-text TextHdcEncoder outputs; reflects this committed lexical/morphological/positional encoding only, not semantic truth.",
        "pairwise_input_similarity": pairwise_input_similarity,
        "cases": rows,
    });

    let mut bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("serialize projection result: {error}"))?;
    bytes.push(b'\n');
    if let Some(parent) = args.output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    }
    std::fs::write(&args.output, &bytes)
        .map_err(|error| format!("write output {}: {error}", args.output.display()))?;

    println!("source_revision={}", args.source_revision);
    println!("glyph_source_blob={}", args.glyph_source_blob);
    println!("glyph_registry_source_blob={}", args.registry_source_blob);
    println!("corpus_blake3={}", corpus_blake3);
    println!("corpus_sha256={}", corpus_sha256);
    println!("encoder_source_blob={}", args.encoder_source_blob);
    println!("dimension={DIMENSION}");
    println!("cases={}", rows.len());
    println!("glyph_registry_entries={}", registry.len());
    println!("projection_output={}", args.output.display());
    println!("projection_output_blake3={}", blake3::hash(&bytes).to_hex());
    Ok(())
}

#[cfg(feature = "glyph_codex")]
fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "glyph_codex"))]
fn main() {
    eprintln!("This probe requires --features glyph_codex");
    std::process::exit(2);
}
