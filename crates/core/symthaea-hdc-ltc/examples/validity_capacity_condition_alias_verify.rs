// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verify that repeated validity-capacity axis anchors are condition aliases,
//! never additional independent replication.
//!
//! This verifier consumes retained v2 JSONL only. It does not rerun the capacity
//! experiment, does not interpret outcomes, and does not define a success
//! threshold. It is intended to run only after the v1/v2 artifact-pair verifier
//! has established the parent evidence chain.

use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use symthaea_hdc_ltc::{ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityPlan};

type AnyError = Box<dyn std::error::Error + Send + Sync + 'static>;

const CONTRACT_VERSION: &str = "hls-validity-condition-alias-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Protocol {
    Smoke,
    ResearchV0,
}

impl Protocol {
    fn parse(value: &str) -> Result<Self, io::Error> {
        match value {
            "smoke" => Ok(Self::Smoke),
            "research-v0" => Ok(Self::ResearchV0),
            other => Err(invalid(format!(
                "unknown protocol {other:?}; expected smoke or research-v0"
            ))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::ResearchV0 => "research-v0",
        }
    }

    fn plan(self) -> ValidityCapacityPlan {
        match self {
            Self::Smoke => ValidityCapacityPlan::smoke(),
            Self::ResearchV0 => ValidityCapacityPlan::research_v0(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NumericCondition {
    dim: usize,
    key_count: usize,
    candidate_count: usize,
    horizon: u64,
    span_length: u64,
}

impl From<ValidityCapacityCase> for NumericCondition {
    fn from(case: ValidityCapacityCase) -> Self {
        Self {
            dim: case.dim,
            key_count: case.key_count,
            candidate_count: case.candidate_count,
            horizon: case.horizon,
            span_length: case.span_length,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LabeledCaseKey {
    axis: String,
    condition: NumericCondition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConditionSeedKey {
    condition: NumericCondition,
    seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AliasGroup {
    condition: NumericCondition,
    axes: Vec<ValidityCapacityAxis>,
}

fn main() -> Result<(), AnyError> {
    let (protocol, v2_path) = parse_args()?;
    let plan = protocol.plan();
    let alias_groups = validate_plan_contract(protocol, &plan)?;
    verify_v2_alias_identity(&v2_path, protocol, &plan)?;

    let unique_condition_count = plan
        .cases
        .iter()
        .copied()
        .map(NumericCondition::from)
        .collect::<HashSet<_>>()
        .len();

    let manifest = json!({
        "contract_version": CONTRACT_VERSION,
        "protocol": protocol.label(),
        "labeled_case_count": plan.cases.len(),
        "unique_numeric_condition_count": unique_condition_count,
        "alias_group_count": alias_groups.len(),
        "alias_groups": alias_groups.iter().map(alias_group_value).collect::<Vec<_>>(),
        "same_seed_alias_payload_identity_verified": true,
        "replication_unit": "preregistered_seed_within_unique_numeric_condition",
        "global_pooling_counts_each_numeric_condition_once": true,
        "per_axis_presentations_preserve_labeled_anchor_copies": true,
        "scientific_claim": "none",
        "interpretation": "not_performed_by_verifier",
    });

    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}

fn parse_args() -> Result<(Protocol, PathBuf), io::Error> {
    let mut args = env::args().skip(1);
    let mut protocol = None;
    let mut v2_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--protocol" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--protocol requires a value"))?;
                protocol = Some(Protocol::parse(&value)?);
            }
            "--v2-evidence" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--v2-evidence requires a value"))?;
                v2_path = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!(
                    "usage: validity_capacity_condition_alias_verify \\\n                     --v2-evidence <v2-jsonl> [--protocol smoke|research-v0]"
                );
                std::process::exit(0);
            }
            other => return Err(invalid(format!("unexpected argument {other:?}"))),
        }
    }

    Ok((
        protocol.unwrap_or(Protocol::Smoke),
        v2_path.ok_or_else(|| invalid("--v2-evidence is required"))?,
    ))
}

fn validate_plan_contract(
    protocol: Protocol,
    plan: &ValidityCapacityPlan,
) -> Result<Vec<AliasGroup>, io::Error> {
    let groups = alias_groups(plan);
    let unique_condition_count = plan
        .cases
        .iter()
        .copied()
        .map(NumericCondition::from)
        .collect::<HashSet<_>>()
        .len();

    match protocol {
        Protocol::Smoke => {
            if plan.cases.len() != 2 || unique_condition_count != 2 || !groups.is_empty() {
                return Err(other(format!(
                    "smoke alias contract drifted: labeled={}, unique={}, aliases={}",
                    plan.cases.len(),
                    unique_condition_count,
                    groups.len()
                )));
            }
        }
        Protocol::ResearchV0 => {
            if plan.cases.len() != 27 || unique_condition_count != 23 || groups.len() != 2 {
                return Err(other(format!(
                    "research_v0 alias contract drifted: labeled={}, unique={}, aliases={}",
                    plan.cases.len(),
                    unique_condition_count,
                    groups.len()
                )));
            }

            let central = NumericCondition {
                dim: 4_096,
                key_count: 8,
                candidate_count: 8,
                horizon: 128,
                span_length: 8,
            };
            let crossover = NumericCondition {
                dim: 4_096,
                key_count: 8,
                candidate_count: 8,
                horizon: 256,
                span_length: 8,
            };

            require_group(
                &groups,
                central,
                &[
                    ValidityCapacityAxis::Dimension,
                    ValidityCapacityAxis::KeyCount,
                    ValidityCapacityAxis::CandidateCount,
                    ValidityCapacityAxis::Horizon,
                ],
            )?;
            require_group(
                &groups,
                crossover,
                &[ValidityCapacityAxis::Horizon, ValidityCapacityAxis::SpanLength],
            )?;
        }
    }

    Ok(groups)
}

fn alias_groups(plan: &ValidityCapacityPlan) -> Vec<AliasGroup> {
    let mut groups: Vec<AliasGroup> = Vec::new();
    for &case in &plan.cases {
        let condition = NumericCondition::from(case);
        if let Some(group) = groups.iter_mut().find(|group| group.condition == condition) {
            group.axes.push(case.axis);
        } else {
            groups.push(AliasGroup {
                condition,
                axes: vec![case.axis],
            });
        }
    }
    groups.retain(|group| group.axes.len() > 1);
    groups
}

fn require_group(
    groups: &[AliasGroup],
    condition: NumericCondition,
    expected_axes: &[ValidityCapacityAxis],
) -> Result<(), io::Error> {
    let group = groups
        .iter()
        .find(|group| group.condition == condition)
        .ok_or_else(|| other(format!("missing frozen alias group {condition:?}")))?;
    if group.axes != expected_axes {
        return Err(other(format!(
            "alias axes drifted for {condition:?}: actual={:?}, expected={expected_axes:?}",
            group.axes
        )));
    }
    Ok(())
}

fn verify_v2_alias_identity(
    path: &PathBuf,
    protocol: Protocol,
    plan: &ValidityCapacityPlan,
) -> Result<(), AnyError> {
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes)?;

    let expected_cases = plan
        .cases
        .iter()
        .copied()
        .map(|case| LabeledCaseKey {
            axis: axis_label(case.axis).to_owned(),
            condition: NumericCondition::from(case),
        })
        .collect::<HashSet<_>>();
    let expected_seeds = plan.replicate_seeds.iter().copied().collect::<HashSet<_>>();

    let mut saw_header = false;
    let mut saw_footer = false;
    let mut footer_count = None;
    let mut seen = HashSet::new();
    let mut normalized_by_condition_seed: HashMap<ConditionSeedKey, Value> = HashMap::new();
    let mut observation_count = 0_usize;

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let kind = required_str(&record, "kind")?;
        let payload = record
            .get("payload")
            .ok_or_else(|| other("v2 record missing payload"))?;

        match kind {
            "header" => {
                if saw_header
                    || payload["evidence_version"].as_str()
                        != Some("hls-validity-capacity-evidence-v2")
                    || payload["protocol"].as_str() != Some(protocol.label())
                {
                    return Err(other("v2 alias verifier received an invalid header").into());
                }
                saw_header = true;
            }
            "falsification_observation" => {
                let case_value = payload
                    .get("case")
                    .ok_or_else(|| other("falsification observation missing case"))?;
                let condition = NumericCondition {
                    dim: required_usize(case_value, "dim")?,
                    key_count: required_usize(case_value, "key_count")?,
                    candidate_count: required_usize(case_value, "candidate_count")?,
                    horizon: required_u64(case_value, "horizon")?,
                    span_length: required_u64(case_value, "span_length")?,
                };
                let labeled_case = LabeledCaseKey {
                    axis: required_str(case_value, "axis")?.to_owned(),
                    condition,
                };
                let seed = required_u64(payload, "seed")?;

                if !expected_cases.contains(&labeled_case) || !expected_seeds.contains(&seed) {
                    return Err(other(format!(
                        "observation is outside frozen {} plan: case={labeled_case:?}, seed={seed}",
                        protocol.label()
                    ))
                    .into());
                }
                if !seen.insert((labeled_case.clone(), seed)) {
                    return Err(other(format!(
                        "duplicate v2 observation: case={labeled_case:?}, seed={seed}"
                    ))
                    .into());
                }

                let normalized = normalize_alias_payload(payload)?;
                let alias_key = ConditionSeedKey { condition, seed };
                if let Some(reference) = normalized_by_condition_seed.get(&alias_key) {
                    if reference != &normalized {
                        return Err(other(format!(
                            "same-seed axis aliases disagree bit-for-bit for condition={condition:?}, seed={seed}"
                        ))
                        .into());
                    }
                } else {
                    normalized_by_condition_seed.insert(alias_key, normalized);
                }
                observation_count += 1;
            }
            "footer" => {
                if saw_footer
                    || payload["complete"].as_bool() != Some(true)
                    || payload["scientific_claim"].as_str() != Some("none")
                {
                    return Err(other("v2 alias verifier received an invalid footer").into());
                }
                footer_count = payload["falsification_observation_count"].as_u64();
                saw_footer = true;
            }
            _ => {}
        }
    }

    if !saw_header || !saw_footer {
        return Err(other("v2 alias verifier requires a complete header/footer boundary").into());
    }

    let expected_count = plan
        .cases
        .len()
        .checked_mul(plan.replicate_seeds.len())
        .ok_or_else(|| other("expected observation count overflow"))?;
    if observation_count != expected_count || footer_count != Some(expected_count as u64) {
        return Err(other(format!(
            "v2 alias observation count mismatch: records={observation_count}, footer={footer_count:?}, expected={expected_count}"
        ))
        .into());
    }

    for expected_case in &expected_cases {
        for &seed in &plan.replicate_seeds {
            if !seen.contains(&(expected_case.clone(), seed)) {
                return Err(other(format!(
                    "missing frozen v2 observation: case={expected_case:?}, seed={seed}"
                ))
                .into());
            }
        }
    }

    Ok(())
}

fn normalize_alias_payload(payload: &Value) -> Result<Value, io::Error> {
    let mut normalized = payload.clone();
    let case = normalized
        .get_mut("case")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| other("falsification observation case is not an object"))?;
    case.remove("axis")
        .ok_or_else(|| other("falsification observation case axis missing"))?;
    Ok(normalized)
}

fn alias_group_value(group: &AliasGroup) -> Value {
    json!({
        "condition": condition_value(group.condition),
        "axes": group.axes.iter().copied().map(axis_label).collect::<Vec<_>>(),
        "independent_replication_count_contribution": 1,
    })
}

fn condition_value(condition: NumericCondition) -> Value {
    json!({
        "dim": condition.dim,
        "key_count": condition.key_count,
        "candidate_count": condition.candidate_count,
        "horizon": condition.horizon,
        "span_length": condition.span_length,
    })
}

fn axis_label(axis: ValidityCapacityAxis) -> &'static str {
    match axis {
        ValidityCapacityAxis::Smoke => "smoke",
        ValidityCapacityAxis::Dimension => "dimension",
        ValidityCapacityAxis::KeyCount => "key_count",
        ValidityCapacityAxis::CandidateCount => "candidate_count",
        ValidityCapacityAxis::Horizon => "horizon",
        ValidityCapacityAxis::SpanLength => "span_length",
    }
}

fn required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, io::Error> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| other(format!("missing or invalid string field {key:?}")))
}

fn required_u64(value: &Value, key: &str) -> Result<u64, io::Error> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| other(format!("missing or invalid u64 field {key:?}")))
}

fn required_usize(value: &Value, key: &str) -> Result<usize, io::Error> {
    usize::try_from(required_u64(value, key)?)
        .map_err(|_| other(format!("field {key:?} does not fit usize")))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn other(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_v0_freezes_27_labels_as_23_unique_conditions() {
        let plan = ValidityCapacityPlan::research_v0();
        let groups = validate_plan_contract(Protocol::ResearchV0, &plan).unwrap();
        assert_eq!(plan.cases.len(), 27);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            plan.cases
                .iter()
                .copied()
                .map(NumericCondition::from)
                .collect::<HashSet<_>>()
                .len(),
            23
        );
    }

    #[test]
    fn smoke_has_no_numeric_condition_aliases() {
        let plan = ValidityCapacityPlan::smoke();
        let groups = validate_plan_contract(Protocol::Smoke, &plan).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn normalization_ignores_only_axis_label() {
        let first = json!({
            "case": {"axis": "dimension", "dim": 64, "key_count": 2, "candidate_count": 2, "horizon": 8, "span_length": 2},
            "seed": 7,
            "accuracy": {"empirical": {"bits": "0x3ff0000000000000"}},
        });
        let second = json!({
            "case": {"axis": "horizon", "dim": 64, "key_count": 2, "candidate_count": 2, "horizon": 8, "span_length": 2},
            "seed": 7,
            "accuracy": {"empirical": {"bits": "0x3ff0000000000000"}},
        });
        assert_eq!(
            normalize_alias_payload(&first).unwrap(),
            normalize_alias_payload(&second).unwrap()
        );

        let changed = json!({
            "case": {"axis": "horizon", "dim": 64, "key_count": 2, "candidate_count": 2, "horizon": 8, "span_length": 2},
            "seed": 7,
            "accuracy": {"empirical": {"bits": "0x3fefffffffffffff"}},
        });
        assert_ne!(
            normalize_alias_payload(&first).unwrap(),
            normalize_alias_payload(&changed).unwrap()
        );
    }
}
