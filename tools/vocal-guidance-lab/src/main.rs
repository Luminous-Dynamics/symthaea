// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-agnostic two-objective gradient conflict auditor.
//!
//! This is a mathematical probe, not a diffusion implementation and not a
//! reproduction of any paper. It accepts gradients exported by an external
//! experiment, measures their interaction, and applies symmetric
//! conflict-only projection. Exact source and checkpoint identifiers travel
//! with the receipt so later acoustic experiments remain traceable.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;

const INPUT_SCHEMA: &str = "vocal-guidance-gradients/v1";
const REPORT_SCHEMA: &str = "vocal-guidance-conflict-report/v1";
const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GradientInput {
    schema: String,
    source_revision: String,
    model_id: String,
    checkpoint_digest: String,
    phoneme_gradient: Vec<f64>,
    pitch_gradient: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ConflictClass {
    Cooperative,
    ConflictingButProjectable,
    InfeasibleAntiparallel,
    MissingBothGradients,
    MissingPhonemeGradient,
    MissingPitchGradient,
}

#[derive(Debug, Serialize)]
struct GuidanceReceipt {
    schema: &'static str,
    source_revision: String,
    model_id: String,
    checkpoint_digest: String,
    projection: &'static str,
    conflict_class: ConflictClass,
    original_dot_product: f64,
    original_cosine: Option<f64>,
    phoneme_original_l2_norm: f64,
    pitch_original_l2_norm: f64,
    phoneme_projected: Vec<f64>,
    pitch_projected: Vec<f64>,
    projected_dot_product: f64,
    projected_cosine: Option<f64>,
    combined_guidance: Vec<f64>,
    combined_l2_norm: f64,
    two_objective_guidance_direction_available: bool,
    numerical_epsilon: f64,
}

fn fail(message: impl Into<String>) -> String {
    message.into()
}

fn valid_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn squared_norm(vector: &[f64]) -> f64 {
    dot(vector, vector)
}

fn norm(vector: &[f64]) -> f64 {
    squared_norm(vector).sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> Option<f64> {
    let denominator = norm(a) * norm(b);
    (denominator > EPSILON).then(|| (dot(a, b) / denominator).clamp(-1.0, 1.0))
}

fn validate(input: &GradientInput) -> Result<(), String> {
    if input.schema != INPUT_SCHEMA {
        return Err(fail(format!(
            "unsupported input schema {:?}; expected {INPUT_SCHEMA:?}",
            input.schema
        )));
    }
    if !valid_hex(&input.source_revision, 40) {
        return Err(fail(
            "source_revision must be a 40-character lowercase Git commit SHA",
        ));
    }
    if input.model_id.trim().is_empty() {
        return Err(fail("model_id must be non-empty"));
    }
    if !valid_hex(&input.checkpoint_digest, 64) {
        return Err(fail(
            "checkpoint_digest must be a 64-character lowercase BLAKE3 digest",
        ));
    }
    if input.phoneme_gradient.len() != input.pitch_gradient.len() {
        return Err(fail(format!(
            "gradient dimensions differ: phoneme={} pitch={}",
            input.phoneme_gradient.len(),
            input.pitch_gradient.len()
        )));
    }
    if input.phoneme_gradient.is_empty() {
        return Err(fail("gradient vectors must not be empty"));
    }
    if input
        .phoneme_gradient
        .iter()
        .chain(&input.pitch_gradient)
        .any(|value| !value.is_finite())
    {
        return Err(fail("gradient vectors must contain only finite values"));
    }
    Ok(())
}

fn subtract_projection(vector: &[f64], onto: &[f64], scale: f64) -> Vec<f64> {
    vector
        .iter()
        .zip(onto)
        .map(|(value, basis)| value - scale * basis)
        .collect()
}

fn evaluate(input: GradientInput) -> Result<GuidanceReceipt, String> {
    validate(&input)?;

    let phoneme = &input.phoneme_gradient;
    let pitch = &input.pitch_gradient;
    let phoneme_norm = norm(phoneme);
    let pitch_norm = norm(pitch);
    let original_dot = dot(phoneme, pitch);
    let original_cosine = cosine(phoneme, pitch);

    let (phoneme_projected, pitch_projected, conflict_class) =
        if phoneme_norm <= EPSILON && pitch_norm <= EPSILON {
            (
                vec![0.0; phoneme.len()],
                vec![0.0; pitch.len()],
                ConflictClass::MissingBothGradients,
            )
        } else if phoneme_norm <= EPSILON {
            (
                vec![0.0; phoneme.len()],
                pitch.clone(),
                ConflictClass::MissingPhonemeGradient,
            )
        } else if pitch_norm <= EPSILON {
            (
                phoneme.clone(),
                vec![0.0; pitch.len()],
                ConflictClass::MissingPitchGradient,
            )
        } else if original_dot >= 0.0 {
            (
                phoneme.clone(),
                pitch.clone(),
                ConflictClass::Cooperative,
            )
        } else {
            // Symmetric two-objective conflict projection. Each guidance vector
            // loses only the component that directly opposes the other vector.
            let phoneme_scale = original_dot / squared_norm(pitch);
            let pitch_scale = original_dot / squared_norm(phoneme);
            let phoneme_projected = subtract_projection(phoneme, pitch, phoneme_scale);
            let pitch_projected = subtract_projection(pitch, phoneme, pitch_scale);
            let projected_norm_sum = norm(&phoneme_projected) + norm(&pitch_projected);
            let conflict_class = if projected_norm_sum <= EPSILON {
                ConflictClass::InfeasibleAntiparallel
            } else {
                ConflictClass::ConflictingButProjectable
            };
            (phoneme_projected, pitch_projected, conflict_class)
        };

    let projected_dot = dot(&phoneme_projected, &pitch_projected);
    let projected_cosine = cosine(&phoneme_projected, &pitch_projected);
    let combined_guidance: Vec<f64> = phoneme_projected
        .iter()
        .zip(&pitch_projected)
        .map(|(left, right)| left + right)
        .collect();
    let combined_norm = norm(&combined_guidance);
    let both_original_objectives_present =
        phoneme_norm > EPSILON && pitch_norm > EPSILON;
    let two_objective_guidance_direction_available =
        both_original_objectives_present && combined_norm > EPSILON;

    Ok(GuidanceReceipt {
        schema: REPORT_SCHEMA,
        source_revision: input.source_revision,
        model_id: input.model_id,
        checkpoint_digest: input.checkpoint_digest,
        projection: "symmetric_conflict_only_two_objective_projection/v1",
        conflict_class,
        original_dot_product: original_dot,
        original_cosine,
        phoneme_original_l2_norm: phoneme_norm,
        pitch_original_l2_norm: pitch_norm,
        phoneme_projected,
        pitch_projected,
        projected_dot_product: projected_dot,
        projected_cosine,
        combined_guidance,
        combined_l2_norm: combined_norm,
        two_objective_guidance_direction_available,
        numerical_epsilon: EPSILON,
    })
}

fn argument(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let index = args
        .iter()
        .position(|argument| argument == flag)
        .ok_or_else(|| fail(format!("missing required argument {flag}")))?;
    args.get(index + 1)
        .map(PathBuf::from)
        .ok_or_else(|| fail(format!("missing path after {flag}")))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let input_path = argument(&args, "--input")?;
    let output_path = argument(&args, "--output")?;
    let bytes = fs::read(&input_path)
        .map_err(|error| fail(format!("cannot read {}: {error}", input_path.display())))?;
    let input: GradientInput = serde_json::from_slice(&bytes)
        .map_err(|error| fail(format!("invalid gradient input: {error}")))?;
    let report = evaluate(input)?;
    let output = serde_json::to_vec_pretty(&report)
        .map_err(|error| fail(format!("cannot serialize report: {error}")))?;
    fs::write(&output_path, output)
        .map_err(|error| fail(format!("cannot write {}: {error}", output_path.display())))?;
    let summary = serde_json::to_string_pretty(&report)
        .map_err(|error| fail(format!("cannot serialize summary: {error}")))?;
    println!("{summary}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vocal-guidance-lab: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(phoneme_gradient: Vec<f64>, pitch_gradient: Vec<f64>) -> GradientInput {
        GradientInput {
            schema: INPUT_SCHEMA.into(),
            source_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            model_id: "fixture-guidance-model".into(),
            checkpoint_digest: "a".repeat(64),
            phoneme_gradient,
            pitch_gradient,
        }
    }

    #[test]
    fn cooperative_objectives_are_left_unchanged() {
        let report = evaluate(input(vec![1.0, 0.0], vec![1.0, 1.0])).unwrap();
        assert!(matches!(report.conflict_class, ConflictClass::Cooperative));
        assert_eq!(report.phoneme_projected, vec![1.0, 0.0]);
        assert_eq!(report.pitch_projected, vec![1.0, 1.0]);
        assert!((report.original_cosine.unwrap() - 2.0_f64.sqrt() / 2.0).abs() < 1e-12);
        assert!(report.two_objective_guidance_direction_available);
    }

    #[test]
    fn conflicting_objectives_remove_opposing_components_without_claiming_orthogonality() {
        let report = evaluate(input(vec![1.0, 0.0], vec![-1.0, 1.0])).unwrap();
        assert!(matches!(
            report.conflict_class,
            ConflictClass::ConflictingButProjectable
        ));
        assert_eq!(report.phoneme_projected, vec![0.5, 0.5]);
        assert_eq!(report.pitch_projected, vec![0.0, 1.0]);
        assert!((report.projected_dot_product - 0.5).abs() < 1e-12);
        assert!(report.projected_cosine.unwrap() > 0.0);
        assert!(report.two_objective_guidance_direction_available);
    }

    #[test]
    fn combined_projected_direction_is_not_antagonistic_to_either_original_objective() {
        let cases = [
            (vec![1.0, 0.0], vec![-1.0, 1.0]),
            (vec![2.0, 1.0], vec![-1.0, 2.0]),
            (vec![0.5, -2.0, 1.0], vec![-1.0, 0.25, -0.5]),
        ];
        for (phoneme, pitch) in cases {
            let report = evaluate(input(phoneme.clone(), pitch.clone())).unwrap();
            if report.two_objective_guidance_direction_available {
                assert!(
                    dot(&phoneme, &report.combined_guidance) >= -EPSILON,
                    "combined direction opposes phoneme objective: {report:?}"
                );
                assert!(
                    dot(&pitch, &report.combined_guidance) >= -EPSILON,
                    "combined direction opposes pitch objective: {report:?}"
                );
            }
        }
    }

    #[test]
    fn antiparallel_objectives_report_no_joint_direction_instead_of_hiding_conflict() {
        let report = evaluate(input(vec![1.0, 0.0], vec![-2.0, 0.0])).unwrap();
        assert!(matches!(
            report.conflict_class,
            ConflictClass::InfeasibleAntiparallel
        ));
        assert!(!report.two_objective_guidance_direction_available);
        assert_eq!(report.combined_l2_norm, 0.0);
    }

    #[test]
    fn missing_objective_gradient_is_reported_not_fabricated() {
        let report = evaluate(input(vec![0.0, 0.0], vec![1.0, 0.0])).unwrap();
        assert!(matches!(
            report.conflict_class,
            ConflictClass::MissingPhonemeGradient
        ));
        assert_eq!(report.phoneme_original_l2_norm, 0.0);
        assert!(!report.two_objective_guidance_direction_available);

        let both_missing = evaluate(input(vec![0.0], vec![0.0])).unwrap();
        assert!(matches!(
            both_missing.conflict_class,
            ConflictClass::MissingBothGradients
        ));
    }

    #[test]
    fn inconsistent_dimensions_fail_closed() {
        assert!(evaluate(input(vec![1.0, 0.0], vec![1.0])).is_err());
    }

    #[test]
    fn non_finite_gradients_fail_closed() {
        assert!(evaluate(input(vec![1.0, f64::NAN], vec![1.0, 0.0])).is_err());
        assert!(evaluate(input(vec![1.0, 0.0], vec![1.0, f64::INFINITY])).is_err());
    }

    #[test]
    fn provenance_must_be_exactly_pinned() {
        let mut bad_revision = input(vec![1.0], vec![1.0]);
        bad_revision.source_revision = "latest".into();
        assert!(validate(&bad_revision).is_err());

        let mut bad_digest = input(vec![1.0], vec![1.0]);
        bad_digest.checkpoint_digest = "not-a-digest".into();
        assert!(validate(&bad_digest).is_err());
    }
}
