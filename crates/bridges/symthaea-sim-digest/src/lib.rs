// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stable, language-neutral digest encoding for the public simulation ABI.
//!
//! This crate deliberately does not use serde, JSON, bincode, generated WIT
//! memory layout, or Rust enum discriminants as a commitment format. The v1
//! encoding is a small protocol that can be implemented independently by other
//! languages, remote workers, and replay/verifier tooling.
//!
//! ## Encoding rules
//!
//! - request domain: `symthaea.simulation.request.v1\0`
//! - output domain: `symthaea.simulation.output.v1\0`
//! - strings: UTF-8 bytes prefixed by little-endian `u64` byte length
//! - lists: little-endian `u64` element count followed by elements in ABI order
//! - booleans: one byte (`0` or `1`)
//! - enum variants: explicit one-byte v1 tags defined below
//! - `option<T>`: one-byte tag (`0` none, `1` some) followed by `T` when present
//! - finite `f64`: exact IEEE-754 bit pattern as little-endian `u64`
//!
//! List order and the sign bit of zero are intentionally preserved. NaN and
//! infinities fail closed rather than requiring a cross-language NaN policy.

use sha2::{Digest, Sha256};
use thiserror::Error;
use symthaea_sim_bridge::{
    EngineeringDomain, Interval, ModelParameter, SimulationMetric, SimulationRequest,
    SimulationResult, SolverKind, UncertaintyEstimate,
};

/// Stable label recorded in extension execution provenance for this encoding.
pub const SIMULATION_DIGEST_PROFILE_V1: &str = "simulation-provider-v1-canonical-v1";

const REQUEST_DOMAIN_V1: &[u8] = b"symthaea.simulation.request.v1\0";
const OUTPUT_DOMAIN_V1: &[u8] = b"symthaea.simulation.output.v1\0";

/// Canonical simulation encoding failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CanonicalDigestError {
    /// The native request failed its own validation before commitment.
    #[error("invalid simulation request: {0}")]
    InvalidRequest(String),
    /// The provider output failed normalized result validation before commitment.
    #[error("invalid simulation output: {0}")]
    InvalidOutput(String),
    /// A finite-only numeric field contained NaN or infinity.
    #[error("non-finite value in canonical simulation field {0}")]
    NonFinite(&'static str),
    /// A collection/string length could not be represented by the v1 u64 length prefix.
    #[error("length overflow in canonical simulation field {0}")]
    LengthOverflow(&'static str),
}

/// Encode one validated native request into the exact v1 commitment bytes.
pub fn canonical_request_bytes_v1(
    request: &SimulationRequest,
) -> Result<Vec<u8>, CanonicalDigestError> {
    request
        .validate()
        .map_err(|error| CanonicalDigestError::InvalidRequest(error.to_string()))?;

    let mut out = Vec::new();
    out.extend_from_slice(REQUEST_DOMAIN_V1);
    put_string(&mut out, &request.id, "request.id")?;
    out.push(domain_tag(request.domain));
    out.push(solver_tag(request.solver));
    put_string(&mut out, &request.objective, "request.objective")?;
    put_len(&mut out, request.parameters.len(), "request.parameters")?;
    for parameter in &request.parameters {
        put_parameter(&mut out, parameter)?;
    }
    put_len(
        &mut out,
        request.requested_metrics.len(),
        "request.requested_metrics",
    )?;
    for metric in &request.requested_metrics {
        put_string(&mut out, metric, "request.requested_metrics[]")?;
    }
    Ok(out)
}

/// SHA-256 of [`canonical_request_bytes_v1`].
pub fn canonical_request_sha256_v1(
    request: &SimulationRequest,
) -> Result<[u8; 32], CanonicalDigestError> {
    Ok(Sha256::digest(canonical_request_bytes_v1(request)?).into())
}

/// Encode provider-owned output fields into the exact v1 commitment bytes.
///
/// `SimulationResult::evidence` is deliberately excluded: the output digest
/// commits to the typed guest/provider result *before* the host constructs
/// execution provenance. This avoids circular self-attestation.
pub fn canonical_output_bytes_v1(
    result: &SimulationResult,
) -> Result<Vec<u8>, CanonicalDigestError> {
    result
        .validate()
        .map_err(|error| CanonicalDigestError::InvalidOutput(error.to_string()))?;

    let mut out = Vec::new();
    out.extend_from_slice(OUTPUT_DOMAIN_V1);
    put_string(&mut out, &result.request_id, "output.request_id")?;
    put_bool(&mut out, result.converged);
    put_f64(&mut out, result.confidence, "output.confidence")?;
    put_uncertainty(&mut out, result.uncertainty, "output.uncertainty")?;
    put_len(&mut out, result.metrics.len(), "output.metrics")?;
    for metric in &result.metrics {
        put_metric(&mut out, metric)?;
    }
    put_len(&mut out, result.warnings.len(), "output.warnings")?;
    for warning in &result.warnings {
        put_string(&mut out, warning, "output.warnings[]")?;
    }
    Ok(out)
}

/// SHA-256 of [`canonical_output_bytes_v1`].
pub fn canonical_output_sha256_v1(
    result: &SimulationResult,
) -> Result<[u8; 32], CanonicalDigestError> {
    Ok(Sha256::digest(canonical_output_bytes_v1(result)?).into())
}

fn put_parameter(out: &mut Vec<u8>, parameter: &ModelParameter) -> Result<(), CanonicalDigestError> {
    put_string(out, &parameter.name, "request.parameters[].name")?;
    put_f64(out, parameter.value, "request.parameters[].value")?;
    put_string(out, &parameter.unit, "request.parameters[].unit")?;
    put_string(
        out,
        &parameter.provenance,
        "request.parameters[].provenance",
    )?;
    put_optional_uncertainty(
        out,
        parameter.uncertainty,
        "request.parameters[].uncertainty",
    )
}

fn put_metric(out: &mut Vec<u8>, metric: &SimulationMetric) -> Result<(), CanonicalDigestError> {
    put_string(out, &metric.name, "output.metrics[].name")?;
    put_f64(out, metric.value, "output.metrics[].value")?;
    put_string(out, &metric.unit, "output.metrics[].unit")?;
    put_optional_uncertainty(out, metric.uncertainty, "output.metrics[].uncertainty")
}

fn put_optional_uncertainty(
    out: &mut Vec<u8>,
    uncertainty: Option<UncertaintyEstimate>,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    match uncertainty {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            put_uncertainty(out, value, field)?;
        }
    }
    Ok(())
}

fn put_uncertainty(
    out: &mut Vec<u8>,
    uncertainty: UncertaintyEstimate,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    put_f64(out, uncertainty.epistemic, field)?;
    put_f64(out, uncertainty.aleatoric, field)?;
    match uncertainty.interval {
        None => out.push(0),
        Some(interval) => {
            out.push(1);
            put_interval(out, interval, field)?;
        }
    }
    Ok(())
}

fn put_interval(
    out: &mut Vec<u8>,
    interval: Interval,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    put_f64(out, interval.lower, field)?;
    put_f64(out, interval.upper, field)
}

fn put_string(
    out: &mut Vec<u8>,
    value: &str,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    put_len(out, value.len(), field)?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_len(
    out: &mut Vec<u8>,
    len: usize,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    let len = u64::try_from(len).map_err(|_| CanonicalDigestError::LengthOverflow(field))?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn put_bool(out: &mut Vec<u8>, value: bool) {
    out.push(u8::from(value));
}

fn put_f64(
    out: &mut Vec<u8>,
    value: f64,
    field: &'static str,
) -> Result<(), CanonicalDigestError> {
    if !value.is_finite() {
        return Err(CanonicalDigestError::NonFinite(field));
    }
    out.extend_from_slice(&value.to_bits().to_le_bytes());
    Ok(())
}

/// Explicit v1 tags matching the declared WIT variant order without relying on
/// Rust's in-memory enum representation.
const fn domain_tag(domain: EngineeringDomain) -> u8 {
    match domain {
        EngineeringDomain::Civil => 0,
        EngineeringDomain::Mechanical => 1,
        EngineeringDomain::Electrical => 2,
        EngineeringDomain::Aerospace => 3,
        EngineeringDomain::ChemicalProcess => 4,
        EngineeringDomain::Robotics => 5,
        EngineeringDomain::Nuclear => 6,
        EngineeringDomain::Materials => 7,
        EngineeringDomain::Environmental => 8,
        EngineeringDomain::Systems => 9,
    }
}

/// Explicit v1 tags matching the declared WIT variant order without relying on
/// Rust's in-memory enum representation.
const fn solver_tag(solver: SolverKind) -> u8 {
    match solver {
        SolverKind::FiniteElement => 0,
        SolverKind::ComputationalFluidDynamics => 1,
        SolverKind::MultibodyDynamics => 2,
        SolverKind::Circuit => 3,
        SolverKind::Process => 4,
        SolverKind::CadGeometry => 5,
        SolverKind::MultiPhysics => 6,
        SolverKind::Custom => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_sim_bridge::{ExecutionMode, SimulationEvidence};

    fn fixture_request() -> SimulationRequest {
        SimulationRequest::new(
            "req-1",
            EngineeringDomain::Systems,
            SolverKind::Custom,
            "canonical digest fixture",
        )
        .with_parameter("alpha", 1.5, "1", "fixture")
        .with_last_parameter_uncertainty(
            UncertaintyEstimate::new(0.25, 0.125)
                .with_interval(Interval::new(-2.0, 3.0)),
        )
    }

    fn fixture_request_with_metric() -> SimulationRequest {
        let mut request = fixture_request();
        request.requested_metrics.push("fixture.metric".into());
        request
    }

    fn fixture_output() -> SimulationResult {
        SimulationResult {
            request_id: "req-1".into(),
            converged: true,
            confidence: 0.7,
            uncertainty: UncertaintyEstimate::new(0.2, 0.1)
                .with_interval(Interval::new(1.0, 2.0)),
            metrics: vec![SimulationMetric {
                name: "fixture.metric".into(),
                value: -0.0,
                unit: "1".into(),
                uncertainty: Some(UncertaintyEstimate::new(0.05, 0.0)),
            }],
            warnings: vec!["not engineering evidence".into()],
            evidence: SimulationEvidence::default(),
        }
    }

    #[test]
    fn request_digest_v1_has_frozen_cross_language_vector() {
        let request = fixture_request_with_metric();
        let bytes = canonical_request_bytes_v1(&request).unwrap();
        assert_eq!(bytes.len(), 195);
        assert_eq!(
            hex(&canonical_request_sha256_v1(&request).unwrap()),
            "86b7bf55d8c2d10f24c9ad2c923e722eb3d1431f1b20fde9cb724359dfb2cbdc"
        );
    }

    #[test]
    fn output_digest_v1_has_frozen_cross_language_vector() {
        let output = fixture_output();
        let bytes = canonical_output_bytes_v1(&output).unwrap();
        assert_eq!(bytes.len(), 190);
        assert_eq!(
            hex(&canonical_output_sha256_v1(&output).unwrap()),
            "b1eccaa53d3e5876e0e0f3ead5b56820cc38a4cc9ec456e9f284755999cec481"
        );
    }

    #[test]
    fn ordered_wit_lists_are_not_silently_sorted() {
        let mut first = fixture_request_with_metric();
        first.requested_metrics.push("second".into());
        let mut second = first.clone();
        second.requested_metrics.swap(0, 1);
        assert_ne!(
            canonical_request_sha256_v1(&first).unwrap(),
            canonical_request_sha256_v1(&second).unwrap()
        );
    }

    #[test]
    fn signed_zero_is_preserved_by_the_commitment() {
        let negative = fixture_output();
        let mut positive = negative.clone();
        positive.metrics[0].value = 0.0;
        assert_ne!(
            canonical_output_sha256_v1(&negative).unwrap(),
            canonical_output_sha256_v1(&positive).unwrap()
        );
    }

    #[test]
    fn non_finite_values_fail_closed() {
        let mut request = fixture_request();
        request.parameters[0].value = f64::NAN;
        assert!(canonical_request_sha256_v1(&request).is_err());

        let mut output = fixture_output();
        output.metrics[0].value = f64::INFINITY;
        assert!(canonical_output_sha256_v1(&output).is_err());
    }

    #[test]
    fn host_evidence_is_excluded_from_provider_output_digest() {
        let plain = fixture_output();
        let mut host_annotated = plain.clone();
        host_annotated.evidence = SimulationEvidence {
            mode: ExecutionMode::DryRun,
            backend: Some("host-owned-lineage".into()),
            ..SimulationEvidence::default()
        };
        assert_eq!(
            canonical_output_sha256_v1(&plain).unwrap(),
            canonical_output_sha256_v1(&host_annotated).unwrap()
        );
    }

    fn hex(bytes: &[u8]) -> String {
        const TABLE: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            out.push(TABLE[(byte >> 4) as usize] as char);
            out.push(TABLE[(byte & 0x0f) as usize] as char);
        }
        out
    }
}
