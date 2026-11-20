// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Deterministic D4 finite-reference regression corpus.
// Exact labels are checked by the separate structural/polynomial oracle crate.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};
use symmetry_discovery_benchmark_oracle::Manifest;

const MAX_DEGREE: u32 = 4;
const COORDINATE_SCALE: i128 = 10;
const COMMON_GROUP_DENOMINATOR: i128 = 8;
const COMMON_TARGET_DENOMINATOR: i128 = 80_000;
const TRAINING_SAMPLES: usize = 32;
const CALIBRATION_SAMPLES: usize = 16;
const HOLDOUT_SAMPLES: usize = 32;
const PRIME: i128 = 1_000_000_007;
const GENERATOR_VERSION: &str = "d4-reynolds-regression-v1";
const FEATURE_BASIS_ID: &str = "xy-monomials-total-degree-le-4-graded-lex-v1";

type Matrix2 = [[i128; 2]; 2];
type Exponent = (u32, u32);
type Polynomial = BTreeMap<Exponent, i128>;

#[derive(Clone, Copy)]
struct Family {
    label: &'static str,
    members: &'static [&'static str],
    base_seed: u32,
}

const FAMILIES: [Family; 10] = [
    Family { label: "D4", members: &["e", "r90", "r180", "r270", "fx", "fy", "fdiag", "fanti"], base_seed: 0xC017_0000 },
    Family { label: "C4", members: &["e", "r90", "r180", "r270"], base_seed: 0xC017_1000 },
    Family { label: "C2_R180", members: &["e", "r180"], base_seed: 0xC017_2000 },
    Family { label: "C2_FX", members: &["e", "fx"], base_seed: 0xC017_3000 },
    Family { label: "C2_FY", members: &["e", "fy"], base_seed: 0xC017_3800 },
    Family { label: "C2_FDIAG", members: &["e", "fdiag"], base_seed: 0xC017_4000 },
    Family { label: "C2_FANTI", members: &["e", "fanti"], base_seed: 0xC017_4800 },
    Family { label: "V4_AXIS", members: &["e", "r180", "fx", "fy"], base_seed: 0xC017_5000 },
    Family { label: "V4_DIAG", members: &["e", "r180", "fdiag", "fanti"], base_seed: 0xC017_6000 },
    Family { label: "C1", members: &["e"], base_seed: 0xC017_7000 },
];

fn d4_matrix(element_id: &str) -> Result<Matrix2, String> {
    match element_id {
        "e" => Ok([[1, 0], [0, 1]]),
        "r90" => Ok([[0, -1], [1, 0]]),
        "r180" => Ok([[-1, 0], [0, -1]]),
        "r270" => Ok([[0, 1], [-1, 0]]),
        "fx" => Ok([[1, 0], [0, -1]]),
        "fy" => Ok([[-1, 0], [0, 1]]),
        "fdiag" => Ok([[0, 1], [1, 0]]),
        "fanti" => Ok([[0, -1], [-1, 0]]),
        _ => Err(format!("unknown D4 element {element_id}")),
    }
}

fn transpose(matrix: Matrix2) -> Matrix2 {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn binomial(n: u32, k: u32) -> Result<i128, String> {
    if k > n { return Ok(0); }
    let k = k.min(n - k);
    let mut result = 1_i128;
    for index in 0..k {
        result = result.checked_mul(i128::from(n - index))
            .ok_or_else(|| "binomial coefficient overflow".to_owned())?;
        result /= i128::from(index + 1);
    }
    Ok(result)
}

fn integer_power(base: i128, exponent: u32) -> Result<i128, String> {
    base.checked_pow(exponent).ok_or_else(|| "integer power overflow".to_owned())
}

/// Exact coefficient transformation for f(M x), represented as a sparse map.
fn transform_polynomial(polynomial: &Polynomial, matrix: Matrix2) -> Result<Polynomial, String> {
    let mut transformed = Polynomial::new();
    for ((power_x, power_y), coefficient) in polynomial {
        for choose_x in 0..=*power_x {
            for choose_y in 0..=*power_y {
                let factors = [
                    *coefficient,
                    binomial(*power_x, choose_x)?,
                    integer_power(matrix[0][0], choose_x)?,
                    integer_power(matrix[0][1], *power_x - choose_x)?,
                    binomial(*power_y, choose_y)?,
                    integer_power(matrix[1][0], choose_y)?,
                    integer_power(matrix[1][1], *power_y - choose_y)?,
                ];
                let term = factors.iter().try_fold(1_i128, |accumulator, factor| {
                    accumulator.checked_mul(*factor)
                        .ok_or_else(|| "polynomial transform multiplication overflow".to_owned())
                })?;
                let exponents = (
                    choose_x + choose_y,
                    (*power_x - choose_x) + (*power_y - choose_y),
                );
                let entry = transformed.entry(exponents).or_insert(0_i128);
                *entry = entry.checked_add(term)
                    .ok_or_else(|| "polynomial transform accumulation overflow".to_owned())?;
            }
        }
    }
    transformed.retain(|_, coefficient| *coefficient != 0);
    Ok(transformed)
}

fn add_polynomial(target: &mut Polynomial, source: &Polynomial) -> Result<(), String> {
    for (exponents, coefficient) in source {
        let entry = target.entry(*exponents).or_insert(0_i128);
        *entry = entry.checked_add(*coefficient)
            .ok_or_else(|| "Reynolds projection coefficient overflow".to_owned())?;
    }
    target.retain(|_, coefficient| *coefficient != 0);
    Ok(())
}

fn feature_basis() -> Vec<Exponent> {
    let mut basis = Vec::new();
    for degree in 0..=MAX_DEGREE {
        for power_x in 0..=degree {
            basis.push((power_x, degree - power_x));
        }
    }
    basis
}

fn xorshift32(state: &mut u32) -> u32 {
    let mut value = *state;
    value ^= value.wrapping_shl(13);
    value ^= value >> 17;
    value ^= value.wrapping_shl(5);
    *state = value;
    value
}

fn base_polynomial(seed: u32) -> Polynomial {
    let mut state = seed.max(1);
    let mut polynomial = Polynomial::new();
    for exponents in feature_basis() {
        let raw = (xorshift32(&mut state) % 11) as i128 - 5;
        let coefficient = if raw == 0 { 1 } else { raw };
        polynomial.insert(exponents, coefficient);
    }
    polynomial
}

/// Reynolds projection P_H f = sum_{h in H} f(h^-1 x).
/// The returned polynomial stores integer numerators; its common denominator
/// is |H|. Orthogonality of D4 matrices makes h^-1 = h^T.
fn reynolds_projection(base: &Polynomial, family: Family) -> Result<Polynomial, String> {
    let mut projected = Polynomial::new();
    for element_id in family.members {
        let inverse_action = transpose(d4_matrix(element_id)?);
        add_polynomial(&mut projected, &transform_polynomial(base, inverse_action)?)?;
    }
    Ok(projected)
}

fn case_id(polynomial: &Polynomial, denominator: usize) -> String {
    let mut canonical = format!("reynolds-denominator={denominator};");
    for ((power_x, power_y), coefficient) in polynomial {
        canonical.push_str(&format!("{power_x},{power_y}:{coefficient};"));
    }
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in canonical.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("REG-{hash:016x}")
}

fn polynomial_terms(polynomial: &Polynomial) -> Vec<Value> {
    polynomial.iter().map(|((power_x, power_y), coefficient)| {
        json!({"exponents": [power_x, power_y], "coefficient": coefficient})
    }).collect()
}

fn oracle_case(case_id: &str, family: Family, polynomial: &Polynomial) -> Value {
    json!({
        "case_id": case_id,
        "track": "finite_reference_group_regression_discovery",
        "expected_subgroup_label": family.label,
        "target_subgroup": family.members,
        "polynomial": {
            "variables": ["x", "y"],
            "exact_rational": true,
            "terms": polynomial_terms(polynomial)
        }
    })
}

fn exact_oracle_accepts(
    manifest: &Manifest,
    all_element_ids: &[String],
    family: Family,
    case_id: &str,
    polynomial: &Polynomial,
) -> Result<bool, String> {
    let case = oracle_case(case_id, family, polynomial);
    let all_elements = manifest.audit_proposed_subgroup(&case, all_element_ids)?;
    let expected: BTreeSet<String> = family.members.iter().map(|id| (*id).to_owned()).collect();
    let actual: BTreeSet<String> = all_elements.exact_stabilizer.iter().cloned().collect();
    if actual != expected { return Ok(false); }

    let subgroup_members = expected.iter().cloned().collect::<Vec<_>>();
    let subgroup_audit = manifest.audit_proposed_subgroup(&case, &subgroup_members)?;
    if !subgroup_audit.group_axioms_satisfied
        || !subgroup_audit.false_promotion_elements.is_empty()
        || !subgroup_audit.missed_symmetry_elements.is_empty()
    {
        return Err(format!("exact oracle produced an invalid subgroup certificate for {case_id}"));
    }
    manifest.verify_proposal_audit(&case, &subgroup_audit)?;
    Ok(true)
}

#[derive(Debug, Clone)]
struct GeneratedFunction {
    case_id: String,
    family: Family,
    polynomial: Polynomial,
    coefficient_denominator: usize,
    base_seed: u32,
    accepted_attempt: u32,
}

fn generate_function(
    manifest: &Manifest,
    all_element_ids: &[String],
    family: Family,
    index: u32,
) -> Result<GeneratedFunction, String> {
    for attempt in 0..2_000_u32 {
        let seed = family.base_seed
            .wrapping_add(index.wrapping_mul(7_919))
            .wrapping_add(attempt.wrapping_mul(104_729));
        let base = base_polynomial(seed);
        let projected = reynolds_projection(&base, family)?;
        if projected.is_empty() { continue; }
        let id = case_id(&projected, family.members.len());
        if exact_oracle_accepts(manifest, all_element_ids, family, &id, &projected)? {
            return Ok(GeneratedFunction {
                case_id: id,
                family,
                polynomial: projected,
                coefficient_denominator: family.members.len(),
                base_seed: seed,
                accepted_attempt: attempt,
            });
        }
    }
    Err(format!("could not generate exact {} regression function {}", family.label, index + 1))
}

fn mod_pow(mut base: i128, mut exponent: i128) -> i128 {
    base = base.rem_euclid(PRIME);
    let mut result = 1_i128;
    while exponent > 0 {
        if exponent & 1 == 1 { result = (result * base).rem_euclid(PRIME); }
        base = (base * base).rem_euclid(PRIME);
        exponent >>= 1;
    }
    result
}

fn modular_feature_matrix_rank(samples: &[SampleV1], basis: &[Exponent]) -> usize {
    let mut matrix = samples.iter().map(|sample| {
        basis.iter().map(|(power_x, power_y)| {
            (mod_pow(i128::from(sample.x_scaled), i128::from(*power_x))
                * mod_pow(i128::from(sample.y_scaled), i128::from(*power_y)))
                .rem_euclid(PRIME)
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    let mut rank = 0_usize;
    let columns = basis.len();
    for column in 0..columns {
        let Some(pivot) = (rank..matrix.len()).find(|row| matrix[*row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = mod_pow(matrix[rank][column], PRIME - 2);
        for entry in column..columns {
            matrix[rank][entry] = (matrix[rank][entry] * inverse).rem_euclid(PRIME);
        }
        for row in 0..matrix.len() {
            if row == rank { continue; }
            let factor = matrix[row][column];
            if factor == 0 { continue; }
            for entry in column..columns {
                matrix[row][entry] =
                    (matrix[row][entry] - factor * matrix[rank][entry]).rem_euclid(PRIME);
            }
        }
        rank += 1;
        if rank == matrix.len() { break; }
    }
    rank
}

#[derive(Debug, Clone)]
struct SampleV1 {
    x_scaled: i32,
    y_scaled: i32,
    target_numerator: i128,
    target_denominator: i128,
}

fn evaluate_target(
    polynomial: &Polynomial,
    subgroup_order: usize,
    x_scaled: i32,
    y_scaled: i32,
) -> Result<(i128, i128), String> {
    let mut numerator = 0_i128;
    for ((power_x, power_y), coefficient) in polynomial {
        let degree = power_x + power_y;
        let scale_multiplier = integer_power(COORDINATE_SCALE, MAX_DEGREE - degree)?;
        let monomial = integer_power(i128::from(x_scaled), *power_x)?
            .checked_mul(integer_power(i128::from(y_scaled), *power_y)?)
            .ok_or_else(|| "sample monomial overflow".to_owned())?;
        let common_group_multiplier = COMMON_GROUP_DENOMINATOR
            / i128::try_from(subgroup_order).map_err(|_| "subgroup order conversion overflow".to_owned())?;
        let term = coefficient
            .checked_mul(common_group_multiplier)
            .and_then(|value| value.checked_mul(scale_multiplier))
            .and_then(|value| value.checked_mul(monomial))
            .ok_or_else(|| "sample target numerator overflow".to_owned())?;
        numerator = numerator.checked_add(term)
            .ok_or_else(|| "sample target accumulation overflow".to_owned())?;
    }
    Ok((numerator, COMMON_TARGET_DENOMINATOR))
}

fn generate_sample_batch(
    polynomial: &Polynomial,
    subgroup_order: usize,
    count: usize,
    seed: u32,
    used_points: &mut BTreeSet<(i32, i32)>,
) -> Result<Vec<SampleV1>, String> {
    let mut state = seed.max(1);
    let mut samples = Vec::with_capacity(count);
    let mut attempts = 0_u32;
    while samples.len() < count {
        attempts = attempts.checked_add(1).ok_or_else(|| "sample attempt counter overflow".to_owned())?;
        if attempts > 500_000 {
            return Err("could not draw a disjoint sample batch from the frozen grid".to_owned());
        }
        let x_scaled = (xorshift32(&mut state) % 41) as i32 - 20;
        let y_scaled = (xorshift32(&mut state) % 41) as i32 - 20;
        if !used_points.insert((x_scaled, y_scaled)) { continue; }
        let (target_numerator, target_denominator) =
            evaluate_target(polynomial, subgroup_order, x_scaled, y_scaled)?;
        samples.push(SampleV1 { x_scaled, y_scaled, target_numerator, target_denominator });
    }
    Ok(samples)
}

fn sample_sets(
    function: &GeneratedFunction,
    function_index: u32,
    basis: &[Exponent],
) -> Result<(Vec<SampleV1>, Vec<SampleV1>, Vec<SampleV1>), String> {
    for retry in 0..128_u32 {
        let mut used_points = BTreeSet::new();
        let root = function.base_seed
            .wrapping_add(function_index.wrapping_mul(65_537))
            .wrapping_add(retry.wrapping_mul(1_000_003));
        let training = generate_sample_batch(
            &function.polynomial, function.coefficient_denominator,
            TRAINING_SAMPLES, root ^ 0x1357_9BDF, &mut used_points,
        )?;
        if modular_feature_matrix_rank(&training, basis) != basis.len() { continue; }
        let calibration = generate_sample_batch(
            &function.polynomial, function.coefficient_denominator,
            CALIBRATION_SAMPLES, root ^ 0x2468_ACE0, &mut used_points,
        )?;
        let holdout = generate_sample_batch(
            &function.polynomial, function.coefficient_denominator,
            HOLDOUT_SAMPLES, root ^ 0xA5A5_5A5A, &mut used_points,
        )?;
        return Ok((training, calibration, holdout));
    }
    Err(format!("could not construct a full-rank sample design for {}", function.case_id))
}

fn samples_json(samples: &[SampleV1]) -> Vec<Value> {
    samples.iter().map(|sample| {
        json!({
            "x_scaled": sample.x_scaled,
            "y_scaled": sample.y_scaled,
            "coordinate_scale": COORDINATE_SCALE,
            "target_numerator": sample.target_numerator,
            "target_denominator": sample.target_denominator
        })
    }).collect()
}

fn ground_truth_json(function: &GeneratedFunction, split: &str) -> Value {
    let terms = polynomial_terms(&function.polynomial).into_iter().map(|term| {
        json!({
            "exponents": term["exponents"].clone(),
            "coefficient_numerator": term["coefficient"].clone(),
            "coefficient_denominator": function.coefficient_denominator
        })
    }).collect::<Vec<_>>();
    json!({
        "case_id": function.case_id,
        "split": split,
        "target_subgroup_label": function.family.label,
        "target_subgroup_element_ids": function.family.members,
        "target_polynomial": {
            "variables": ["x", "y"],
            "coefficient_denominator": function.coefficient_denominator,
            "terms": terms
        },
        "generation_provenance": {
            "generator_version": GENERATOR_VERSION,
            "base_seed": function.base_seed,
            "accepted_attempt": function.accepted_attempt,
            "reynolds_projection": "sum_{h in H} f(h^-1 x) / |H|",
            "exact_stabilizer_oracle": "symmetry-discovery-benchmark-oracle 0.1.0"
        }
    })
}

fn public_function_json(
    function: &GeneratedFunction,
    training: &[SampleV1],
    calibration: &[SampleV1],
    holdout: &[SampleV1],
) -> Value {
    json!({
        "case_id": function.case_id,
        "feature_basis_id": FEATURE_BASIS_ID,
        "training_samples": samples_json(training),
        "calibration_samples": samples_json(calibration),
        "holdout_samples": samples_json(holdout)
    })
}

#[derive(Debug)]
pub struct GeneratedCorpusV1 {
    pub public_splits: BTreeMap<String, Value>,
    pub ground_truth: Value,
    pub receipt: Value,
}

/// Generate the 10 × 90 exact D4 regression corpus and its label-free sample
/// exports. Target labels and coefficients remain in the separate ground_truth value.
pub fn generate_corpus() -> Result<GeneratedCorpusV1, String> {
    let manifest = Manifest::parse()?;
    manifest.verify_group_axioms()?;
    let all_element_ids = manifest.element_ids();
    let basis = feature_basis();

    let mut public_cases: BTreeMap<String, Vec<Value>> = [
        ("discovery".to_owned(), Vec::new()),
        ("calibration".to_owned(), Vec::new()),
        ("final_holdout".to_owned(), Vec::new()),
    ].into_iter().collect();
    let mut ground_truth_cases = Vec::new();
    let mut seen_case_ids = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut split_family_counts = BTreeMap::<(String, String), usize>::new();

    for family in FAMILIES {
        for index in 0..90_u32 {
            let function = generate_function(&manifest, &all_element_ids, family, index)?;
            if !seen_case_ids.insert(function.case_id.clone()) {
                return Err(format!("duplicate generated regression case {}", function.case_id));
            }
            let split_rank = (index * 37) % 90;
            let split = if split_rank < 30 { "discovery" }
                else if split_rank < 60 { "calibration" }
                else { "final_holdout" };
            let (training, calibration, holdout) = sample_sets(&function, index, &basis)?;
            let public_case = public_function_json(&function, &training, &calibration, &holdout);
            public_cases.get_mut(split)
                .ok_or_else(|| format!("unknown generated split {split}"))?
                .push(public_case);
            ground_truth_cases.push(ground_truth_json(&function, split));
            *family_counts.entry(family.label.to_owned()).or_insert(0) += 1;
            *split_family_counts.entry((family.label.to_owned(), split.to_owned())).or_insert(0) += 1;
        }
    }

    let mut public_splits = BTreeMap::new();
    for split in ["discovery", "calibration", "final_holdout"] {
        let cases = public_cases.remove(split)
            .ok_or_else(|| format!("missing generated split {split}"))?;
        public_splits.insert(split.to_owned(), json!({
            "schema_version": "0.1.0",
            "benchmark_id": "SYMM-COLT-D4-REGRESSION-001",
            "split": split,
            "ambient_group_id": "D4",
            "feature_basis_id": FEATURE_BASIS_ID,
            "feature_basis": basis,
            "coordinate_distribution": {
                "family": "product-uniform discrete grid",
                "coordinate_scale": COORDINATE_SCALE,
                "integer_grid_min": -20,
                "integer_grid_max": 20,
                "sampling_rule": "independent deterministic substreams; no repeated point per function across train/calibration/holdout"
            },
            "sample_counts_per_function": {
                "training": TRAINING_SAMPLES,
                "calibration": CALIBRATION_SAMPLES,
                "holdout": HOLDOUT_SAMPLES
            },
            "target_encoding": {
                "numerator_denominator": "exact signed integer rational",
                "common_denominator": COMMON_TARGET_DENOMINATOR,
                "note": "the common denominator is independent of the target subgroup order"
            },
            "cases": cases
        }));
    }

    let ground_truth = json!({
        "schema_version": "0.1.0",
        "benchmark_id": "SYMM-COLT-D4-REGRESSION-001",
        "disclosure": "Ground truth labels and exact coefficients are intentionally separate from public_inputs. This public repository is reproducible, not a sealed leaderboard.",
        "ambient_group_id": "D4",
        "feature_basis_id": FEATURE_BASIS_ID,
        "cases": ground_truth_cases
    });
    let receipt = json!({
        "schema_version": "0.1.0",
        "generator_version": GENERATOR_VERSION,
        "ambient_group_id": "D4",
        "distinct_subgroups": FAMILIES.len(),
        "instances_per_subgroup": 90,
        "instances_per_split_per_subgroup": 30,
        "total_instances": seen_case_ids.len(),
        "family_counts": family_counts,
        "split_family_counts": split_family_counts,
        "public_split_counts": public_splits.iter().map(|(split, value)| {
            (split.clone(), value["cases"].as_array().map_or(0, Vec::len))
        }).collect::<BTreeMap<_, _>>(),
        "all_cases_exact_oracle_verified": true,
        "sample_design": {
            "feature_dimension": basis.len(),
            "training_rows": TRAINING_SAMPLES,
            "calibration_rows": CALIBRATION_SAMPLES,
            "holdout_rows": HOLDOUT_SAMPLES,
            "training_design_rank": "full column rank modulo 1_000_000_007"
        },
        "claim_boundary": "This is a deterministic exact D4 regression corpus and oracle. It is not a faithful COLT 2026 algorithm/theorem reproduction or a benchmark result."
    });
    Ok(GeneratedCorpusV1 { public_splits, ground_truth, receipt })
}

fn write_json(path: &std::path::Path, value: &Value) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("could not create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|error| format!("could not serialize {}: {error}", path.display()))
}

pub fn write_corpus(output_dir: &std::path::Path, include_ground_truth: bool) -> Result<(), String> {
    let corpus = generate_corpus()?;
    let public_dir = output_dir.join("public_inputs");
    std::fs::create_dir_all(&public_dir)
        .map_err(|error| format!("could not create {}: {error}", public_dir.display()))?;
    for (split, value) in &corpus.public_splits {
        write_json(&public_dir.join(format!("{split}.json")), value)?;
    }
    write_json(&output_dir.join("generation_receipt.json"), &corpus.receipt)?;
    if include_ground_truth {
        write_json(&output_dir.join("ground_truth.json"), &corpus.ground_truth)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d4_actions_are_orthogonal_and_have_exact_inverse_transposes() {
        for family in FAMILIES {
            for element_id in family.members {
                let matrix = d4_matrix(element_id).unwrap();
                let inverse = transpose(matrix);
                let mut product = [[0_i128; 2]; 2];
                for row in 0..2 {
                    for column in 0..2 {
                        product[row][column] =
                            matrix[row][0] * inverse[0][column] + matrix[row][1] * inverse[1][column];
                    }
                }
                assert_eq!(product, [[1, 0], [0, 1]]);
            }
        }
    }

    #[test]
    fn Reynolds_projection_is_exactly_invariant_under_each_declared_subgroup() {
        for family in FAMILIES {
            let base = base_polynomial(family.base_seed);
            let projected = reynolds_projection(&base, family).unwrap();
            assert!(!projected.is_empty());
            for element_id in family.members {
                let transformed =
                    transform_polynomial(&projected, transpose(d4_matrix(element_id).unwrap())).unwrap();
                assert_eq!(transformed, projected, "{} failed", family.label);
            }
        }
    }

    #[test]
    fn training_feature_matrix_has_full_rank_on_a_known_grid() {
        let basis = feature_basis();
        let mut used = BTreeSet::new();
        let samples = generate_sample_batch(&base_polynomial(17), 8, TRAINING_SAMPLES, 123_456, &mut used).unwrap();
        assert_eq!(modular_feature_matrix_rank(&samples, &basis), basis.len());
    }

    #[test]
    fn full_900_instance_corpus_has_exact_oracle_labels_and_label_free_inputs() {
        let corpus = generate_corpus().expect("generate exact D4 regression corpus");
        assert_eq!(corpus.ground_truth["cases"].as_array().unwrap().len(), 900);
        assert_eq!(corpus.receipt["total_instances"], 900);
        for (split, public) in &corpus.public_splits {
            assert_eq!(public["cases"].as_array().unwrap().len(), 300);
            for case in public["cases"].as_array().unwrap() {
                assert!(case.get("target_subgroup_label").is_none());
                assert!(case.get("target_subgroup_element_ids").is_none());
                assert!(case.get("target_polynomial").is_none());
                assert!(case.get("generation_seed").is_none());
                let training = case["training_samples"].as_array().unwrap();
                let calibration = case["calibration_samples"].as_array().unwrap();
                let holdout = case["holdout_samples"].as_array().unwrap();
                let unique: BTreeSet<(i64, i64)> = training.iter().chain(calibration).chain(holdout)
                    .map(|sample| (
                        sample["x_scaled"].as_i64().unwrap(),
                        sample["y_scaled"].as_i64().unwrap()
                    )).collect();
                assert_eq!(unique.len(), TRAINING_SAMPLES + CALIBRATION_SAMPLES + HOLDOUT_SAMPLES);
                assert_eq!(training.len(), TRAINING_SAMPLES);
                assert_eq!(calibration.len(), CALIBRATION_SAMPLES);
                assert_eq!(holdout.len(), HOLDOUT_SAMPLES);
                let training_struct = training.iter().map(|sample| SampleV1 {
                    x_scaled: sample["x_scaled"].as_i64().unwrap() as i32,
                    y_scaled: sample["y_scaled"].as_i64().unwrap() as i32,
                    target_numerator: sample["target_numerator"].as_i64().unwrap() as i128,
                    target_denominator: sample["target_denominator"].as_i64().unwrap() as i128,
                }).collect::<Vec<_>>();
                assert_eq!(
                    modular_feature_matrix_rank(&training_struct, &feature_basis()),
                    feature_basis().len(),
                    "training design must identify the 15-dimensional polynomial feature vector in {split}"
                );
            }
        }
    }
}
