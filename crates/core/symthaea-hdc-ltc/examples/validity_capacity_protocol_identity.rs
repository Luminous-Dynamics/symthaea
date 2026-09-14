// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Non-executing protocol identity for the frozen validity-capacity research plan.
//!
//! This executable constructs the declared `research_v0` capacity and control
//! plans, canonicalizes them exactly like evidence-v1, and checks their
//! domain-separated SHA-256 commitments against the frozen preregistration. It
//! runs no capacity observations and therefore can be used as a pre-execution
//! launch gate.

#[path = "support/evidence_sha256.rs"]
mod evidence_sha256;

use serde_json::{Map, Value, json};
use std::io;
use symthaea_hdc_ltc::{
    ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityControlAxis,
    ValidityCapacityControlCase, ValidityCapacityControlPlan, ValidityCapacityPlan,
};

const PLAN_DOMAIN: &str = "symthaea:hdc-ltc:validity-capacity:plan:v1";
const FROZEN_CAPACITY_PLAN_SHA256: &str =
    "023cb6eed3e0bfa487f23218af225889d5358d8e9b3ec6fb5db8ed12b97c3459";
const FROZEN_CONTROL_PLAN_SHA256: &str =
    "d0d1912bff351817d12d1cc140bd481a6d291c0bb84a31b1dbf43b1edb9bf414";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (capacity_digest, control_digest) = current_research_v0_digests()?;
    require_frozen_identity(&capacity_digest, &control_digest)?;
    println!("CAPACITY_PLAN_SHA256={capacity_digest}");
    println!("CONTROL_PLAN_SHA256={control_digest}");
    Ok(())
}

fn current_research_v0_digests() -> Result<(String, String), serde_json::Error> {
    let capacity = capacity_plan_value(&ValidityCapacityPlan::research_v0());
    let controls = control_plan_value(&ValidityCapacityControlPlan::research_v0());
    Ok((digest_value(&capacity)?, digest_value(&controls)?))
}

fn require_frozen_identity(capacity: &str, controls: &str) -> Result<(), io::Error> {
    if capacity != FROZEN_CAPACITY_PLAN_SHA256 {
        return Err(io::Error::other(format!(
            "research_v0 capacity plan drift: frozen={FROZEN_CAPACITY_PLAN_SHA256}, current={capacity}; define a new protocol version instead of mutating research_v0"
        )));
    }
    if controls != FROZEN_CONTROL_PLAN_SHA256 {
        return Err(io::Error::other(format!(
            "research_v0 control plan drift: frozen={FROZEN_CONTROL_PLAN_SHA256}, current={controls}; define a new protocol version instead of mutating research_v0"
        )));
    }
    Ok(())
}

fn digest_value(value: &Value) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_vec(&canonicalize(value))?;
    let mut bytes = Vec::with_capacity(PLAN_DOMAIN.len() + 1 + canonical.len());
    bytes.extend_from_slice(PLAN_DOMAIN.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&canonical);
    Ok(evidence_sha256::sha256_hex(&bytes))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                let nested = map
                    .get(key.as_str())
                    .expect("canonicalization key came from the same map");
                sorted.insert(key.to_string(), canonicalize(nested));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

fn capacity_axis_label(axis: ValidityCapacityAxis) -> &'static str {
    match axis {
        ValidityCapacityAxis::Smoke => "smoke",
        ValidityCapacityAxis::Dimension => "dimension",
        ValidityCapacityAxis::KeyCount => "key_count",
        ValidityCapacityAxis::CandidateCount => "candidate_count",
        ValidityCapacityAxis::Horizon => "horizon",
        ValidityCapacityAxis::SpanLength => "span_length",
    }
}

fn control_axis_label(axis: ValidityCapacityControlAxis) -> &'static str {
    match axis {
        ValidityCapacityControlAxis::Smoke => "smoke",
        ValidityCapacityControlAxis::SemanticRunLength => "semantic_run_length",
        ValidityCapacityControlAxis::WriteSegmentation => "write_segmentation",
    }
}

fn capacity_case_value(case: ValidityCapacityCase) -> Value {
    json!({
        "axis": capacity_axis_label(case.axis),
        "dim": case.dim,
        "key_count": case.key_count,
        "candidate_count": case.candidate_count,
        "horizon": case.horizon,
        "span_length": case.span_length,
    })
}

fn control_case_value(case: ValidityCapacityControlCase) -> Value {
    json!({
        "axis": control_axis_label(case.axis),
        "dim": case.dim,
        "key_count": case.key_count,
        "candidate_count": case.candidate_count,
        "horizon": case.horizon,
        "semantic_run_length": case.semantic_run_length,
        "write_segment_length": case.write_segment_length,
    })
}

fn capacity_plan_value(plan: &ValidityCapacityPlan) -> Value {
    json!({
        "plan": "validity_capacity",
        "cases": plan.cases.iter().copied().map(capacity_case_value).collect::<Vec<_>>(),
        "replicate_seeds": plan.replicate_seeds.clone(),
    })
}

fn control_plan_value(plan: &ValidityCapacityControlPlan) -> Value {
    json!({
        "plan": "validity_capacity_controls",
        "cases": plan.cases.iter().copied().map(control_case_value).collect::<Vec<_>>(),
        "replicate_seeds": plan.replicate_seeds.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalization_is_key_order_independent() {
        let left = json!({"b": 2, "a": {"d": 4, "c": 3}});
        let right = json!({"a": {"c": 3, "d": 4}, "b": 2});
        assert_eq!(
            serde_json::to_vec(&canonicalize(&left)).unwrap(),
            serde_json::to_vec(&canonicalize(&right)).unwrap()
        );
    }

    #[test]
    fn research_v0_matches_frozen_preregistration_identity() {
        let (capacity, controls) = current_research_v0_digests().unwrap();
        require_frozen_identity(&capacity, &controls).unwrap();
        assert_eq!(capacity, FROZEN_CAPACITY_PLAN_SHA256);
        assert_eq!(controls, FROZEN_CONTROL_PLAN_SHA256);
    }
}
