// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1C released-Cobaya fixed-point contract checker.
//!
//! This program executes no Python and no cosmology. It proves that the frozen
//! A1C contract is internally consistent with the already-frozen A1R subject.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAX_JSON_BYTES: u64 = 1024 * 1024;
const PROTOCOL: &str = "DE-001A1C-CONTRACT-CONSISTENCY-v1";
const AUTHORITY: &str = "contract-consistency-only";

const COBAYA_VERSION: &str = "3.6.2";
const COBAYA_COMMIT: &str = "899f30a49f85de610dac321e91a1af50018e56aa";
const COBAYA_SDIST_SHA256: &str =
    "8f1061d6347427f08380e1e0c0b766d695d3978b5439fb0b1cc1a7002152d9c8";
const BAO_DATA_COMMIT: &str = "bb0c1c9009dc76d1391300e169e8df38fd1096db";
const LIKELIHOOD_SHA256: &str =
    "fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa";
const BESTFIT_SHA256: &str =
    "bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358";
const MEAN_SHA256: &str =
    "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585";
const COVARIANCE_SHA256: &str =
    "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    authority: String,
    status: String,
    source: Source,
    subject: Subject,
    provider: Provider,
    execution_policy: ExecutionPolicy,
    independence: Independence,
    execution_prerequisites: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    cobaya_version: String,
    cobaya_source_commit: String,
    cobaya_sdist_sha256: String,
    likelihood_alias: String,
    likelihood_class: String,
    likelihood_class_path: String,
    likelihood_base_path: String,
    likelihood_definition_role: String,
    likelihood_definition_sha256: String,
    bao_data_commit: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Subject {
    model: String,
    bestfit_role: String,
    bestfit_sha256: String,
    mean_role: String,
    mean_size: u64,
    mean_sha256: String,
    covariance_role: String,
    covariance_size: u64,
    covariance_sha256: String,
    row_count: u64,
    omega_m: f64,
    h_r_d_mpc: f64,
    reference_chi2_bao: f64,
    absolute_tolerance: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provider {
    implementation: String,
    rdrag_gauge_mpc: f64,
    h0_formula: String,
    speed_of_light_km_s: f64,
    required_provider_methods: Vec<String>,
    gauge_statement: String,
    camb_forbidden: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionPolicy {
    likelihood_calls: u64,
    sampler_forbidden: bool,
    minimizer_forbidden: bool,
    parameter_mutation_forbidden: bool,
    network_forbidden: bool,
    data_install_forbidden: bool,
    packages_path_mutation_forbidden: bool,
    optimization_forbidden: bool,
    required_outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Independence {
    measurement_data_independent: bool,
    covariance_independent: bool,
    model_family_independent: bool,
    background_code_independent_from_a1r: bool,
    gaussian_likelihood_code_independent_from_a1r: bool,
    scope: String,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    execution_authorized: bool,
    a1c_manifest_sha256: String,
    a1r_manifest_sha256: String,
    cobaya_version: String,
    cobaya_source_commit: String,
    subject: Subject,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    execution_authorized: bool,
    error: String,
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > MAX_JSON_BYTES {
        return Err(format!("{}: file too large", path.display()));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exact_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn ordered_strings_equal(actual: &[String], expected: &[&str]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| actual == expected)
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, String> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("missing A1R field {}", path.join(".")))?;
    }
    Ok(current)
}

fn value_str<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, String> {
    value_at(value, path)?
        .as_str()
        .ok_or_else(|| format!("A1R field {} is not a string", path.join(".")))
}

fn value_u64(value: &Value, path: &[&str]) -> Result<u64, String> {
    value_at(value, path)?
        .as_u64()
        .ok_or_else(|| format!("A1R field {} is not an unsigned integer", path.join(".")))
}

fn value_f64(value: &Value, path: &[&str]) -> Result<f64, String> {
    value_at(value, path)?
        .as_f64()
        .ok_or_else(|| format!("A1R field {} is not numeric", path.join(".")))
}

fn validate_source(source: &Source) -> Result<(), String> {
    if source.cobaya_version != COBAYA_VERSION
        || source.cobaya_source_commit != COBAYA_COMMIT
        || source.cobaya_sdist_sha256 != COBAYA_SDIST_SHA256
        || source.likelihood_alias != "bao.desi_dr2"
        || source.likelihood_class
            != "cobaya.likelihoods.bao.desi_dr2.desi_bao_all.desi_bao_all"
        || source.likelihood_class_path != "cobaya/likelihoods/bao/desi_dr2/desi_bao_all.py"
        || source.likelihood_base_path != "cobaya/likelihoods/base_classes/bao.py"
        || source.likelihood_definition_role != "likelihood-definition"
        || source.likelihood_definition_sha256 != LIKELIHOOD_SHA256
        || source.bao_data_commit != BAO_DATA_COMMIT
    {
        return Err("A1C source identity differs from the frozen released-Cobaya subject".into());
    }
    if !is_lower_hex_sha256(&source.cobaya_sdist_sha256)
        || !is_lower_hex_sha256(&source.likelihood_definition_sha256)
    {
        return Err("A1C source SHA-256 identity is malformed".into());
    }
    Ok(())
}

fn validate_subject(subject: &Subject) -> Result<(), String> {
    if subject.model != "flat-lambda-cdm-bao-only-late-background"
        || subject.bestfit_role != "reference-bestfit-text"
        || subject.bestfit_sha256 != BESTFIT_SHA256
        || subject.mean_role != "dataset-mean"
        || subject.mean_size != 472
        || subject.mean_sha256 != MEAN_SHA256
        || subject.covariance_role != "dataset-covariance"
        || subject.covariance_size != 2547
        || subject.covariance_sha256 != COVARIANCE_SHA256
        || subject.row_count != 13
        || !exact_f64(subject.omega_m, 0.297_177_87)
        || !exact_f64(subject.h_r_d_mpc, 101.547_86)
        || !exact_f64(subject.reference_chi2_bao, 10.282_299)
        || !exact_f64(subject.absolute_tolerance, 0.01)
    {
        return Err("A1C subject differs from the frozen DE-001A fixed point".into());
    }
    for digest in [
        subject.bestfit_sha256.as_str(),
        subject.mean_sha256.as_str(),
        subject.covariance_sha256.as_str(),
    ] {
        if !is_lower_hex_sha256(digest) {
            return Err("A1C subject contains malformed SHA-256 identity".into());
        }
    }
    Ok(())
}

fn validate_provider(provider: &Provider) -> Result<(), String> {
    let expected_methods = [
        "get_angular_diameter_distance",
        "get_Hubble",
        "get_param:rdrag",
    ];
    if provider.implementation != "python-analytic-flat-lcdm-hrdrag-gauge-v1"
        || !exact_f64(provider.rdrag_gauge_mpc, 100.0)
        || provider.h0_formula != "100*h_r_d_mpc/rdrag_gauge_mpc"
        || !exact_f64(provider.speed_of_light_km_s, 299_792.458)
        || !ordered_strings_equal(&provider.required_provider_methods, &expected_methods)
        || provider.gauge_statement.trim().is_empty()
        || !provider.camb_forbidden
    {
        return Err("A1C provider contract is not the frozen two-parameter gauge provider".into());
    }
    Ok(())
}

fn validate_execution(policy: &ExecutionPolicy) -> Result<(), String> {
    let expected_outputs = [
        "logp_bao",
        "chi2_bao",
        "point_manifest_sha256",
        "a0_receipt_sha256",
        "environment_receipt_sha256",
        "cobaya_version",
    ];
    if policy.likelihood_calls != 1
        || !policy.sampler_forbidden
        || !policy.minimizer_forbidden
        || !policy.parameter_mutation_forbidden
        || !policy.network_forbidden
        || !policy.data_install_forbidden
        || !policy.packages_path_mutation_forbidden
        || !policy.optimization_forbidden
        || !ordered_strings_equal(&policy.required_outputs, &expected_outputs)
    {
        return Err(
            "A1C execution policy permits an operation forbidden by the fixed-point contract".into(),
        );
    }
    Ok(())
}

fn validate_independence(independence: &Independence) -> Result<(), String> {
    if independence.measurement_data_independent
        || independence.covariance_independent
        || independence.model_family_independent
        || !independence.background_code_independent_from_a1r
        || !independence.gaussian_likelihood_code_independent_from_a1r
        || independence.scope.trim().is_empty()
    {
        return Err("A1C independence declaration overstates or understates the frozen scope".into());
    }
    Ok(())
}

fn validate_subject_against_a1r(subject: &Subject, a1r: &Value) -> Result<(), String> {
    if value_str(a1r, &["protocol"])? != "DE-001A1R-RUST-ORACLE-v1"
        || value_str(a1r, &["scientific_claim"])? != "NONE"
        || value_str(a1r, &["model"])? != subject.model.as_str()
        || value_str(a1r, &["source", "bestfit_role"])? != subject.bestfit_role.as_str()
        || value_str(a1r, &["source", "bestfit_sha256"])? != subject.bestfit_sha256.as_str()
        || value_str(a1r, &["data", "mean_role"])? != subject.mean_role.as_str()
        || value_u64(a1r, &["data", "mean_size"])? != subject.mean_size
        || value_str(a1r, &["data", "mean_sha256"])? != subject.mean_sha256.as_str()
        || value_str(a1r, &["data", "covariance_role"])? != subject.covariance_role.as_str()
        || value_u64(a1r, &["data", "covariance_size"])? != subject.covariance_size
        || value_str(a1r, &["data", "covariance_sha256"])?
            != subject.covariance_sha256.as_str()
        || value_u64(a1r, &["data", "row_count"])? != subject.row_count
        || !exact_f64(value_f64(a1r, &["parameters", "omega_m"])?, subject.omega_m)
        || !exact_f64(
            value_f64(a1r, &["parameters", "h_r_d_mpc"])?,
            subject.h_r_d_mpc,
        )
        || !exact_f64(
            value_f64(a1r, &["reference", "chi2_bao"])?,
            subject.reference_chi2_bao,
        )
        || !exact_f64(
            value_f64(a1r, &["reference", "absolute_tolerance"])?,
            subject.absolute_tolerance,
        )
    {
        return Err("A1C and A1R do not describe the exact same fixed-point subject".into());
    }
    Ok(())
}

fn execute(a1c_path: &Path, a1r_path: &Path) -> Result<Receipt, String> {
    let a1c_bytes = read_regular_file(a1c_path)?;
    let a1r_bytes = read_regular_file(a1r_path)?;
    let manifest: Manifest = serde_json::from_slice(&a1c_bytes)
        .map_err(|error| format!("invalid A1C manifest JSON: {error}"))?;
    let a1r: Value = serde_json::from_slice(&a1r_bytes)
        .map_err(|error| format!("invalid A1R manifest JSON: {error}"))?;

    if manifest.schema_version != 1
        || manifest.protocol != "DE-001A1C-COBAYA-FIXED-POINT-v1"
        || manifest.scientific_claim != "NONE"
        || manifest.authority != "released-likelihood-fixed-point-reproduction-only"
        || manifest.status != "preregistered-contract-only"
    {
        return Err("invalid A1C manifest identity".into());
    }

    validate_source(&manifest.source)?;
    validate_subject(&manifest.subject)?;
    validate_provider(&manifest.provider)?;
    validate_execution(&manifest.execution_policy)?;
    validate_independence(&manifest.independence)?;
    validate_subject_against_a1r(&manifest.subject, &a1r)?;

    let expected_prerequisites = [
        "qualified DE-001A1Q PASS",
        "qualified DE-001A numerical environment PASS",
    ];
    if !ordered_strings_equal(&manifest.execution_prerequisites, &expected_prerequisites) {
        return Err("A1C execution prerequisites are not frozen as expected".into());
    }

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        execution_authorized: false,
        a1c_manifest_sha256: sha256_hex(&a1c_bytes),
        a1r_manifest_sha256: sha256_hex(&a1r_bytes),
        cobaya_version: manifest.source.cobaya_version,
        cobaya_source_commit: manifest.source.cobaya_source_commit,
        subject: manifest.subject,
    })
}

fn write_json<T: Serialize>(value: &T) -> io::Result<()> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    writeln!(&mut lock)?;
    Ok(())
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: de001a-a1c-contract A1C_CONTRACT.json A1R_MANIFEST.json".into());
    }
    let receipt = execute(&PathBuf::from(&args[0]), &PathBuf::from(&args[1]))?;
    write_json(&receipt).map_err(|error| format!("failed to write A1C contract receipt: {error}"))?;
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                execution_authorized: false,
                error,
            };
            let _ = write_json(&receipt);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ExecutionPolicy {
        ExecutionPolicy {
            likelihood_calls: 1,
            sampler_forbidden: true,
            minimizer_forbidden: true,
            parameter_mutation_forbidden: true,
            network_forbidden: true,
            data_install_forbidden: true,
            packages_path_mutation_forbidden: true,
            optimization_forbidden: true,
            required_outputs: vec![
                "logp_bao".into(),
                "chi2_bao".into(),
                "point_manifest_sha256".into(),
                "a0_receipt_sha256".into(),
                "environment_receipt_sha256".into(),
                "cobaya_version".into(),
            ],
        }
    }

    #[test]
    fn fixed_point_policy_rejects_sampler_or_extra_calls() {
        let mut value = policy();
        assert!(validate_execution(&value).is_ok());
        value.sampler_forbidden = false;
        assert!(validate_execution(&value).is_err());

        let mut value = policy();
        value.likelihood_calls = 2;
        assert!(validate_execution(&value).is_err());
    }

    #[test]
    fn float_identity_is_bit_exact() {
        assert!(exact_f64(0.297_177_87, 0.297_177_87));
        assert!(!exact_f64(0.297_177_87, 0.297_177_88));
    }
}
