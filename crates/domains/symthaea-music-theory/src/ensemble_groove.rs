// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-independent analysis of interacting rhythmic layers.
//!
//! This module complements the cycle-owned composition module: that module
//! makes long-form sections and phrase boundaries obey a named cycle; this
//! module describes and measures the relationships BETWEEN independent
//! layers over their complete shared supercycle. It does not classify genres
//! or claim cultural authenticity. It provides exact, integer-time mechanics
//! that a tradition-specific, source-reviewed grammar can use and test.
//!
//! The key design choice is to represent a groove as a set of independently
//! repeating role layers rather than one flattened drum pattern. Different
//! layers may have different cycle lengths. Their joint horizon is the least
//! common multiple (LCM), so cross-cycle alignment and phase resets are
//! observable instead of guessed from a one-bar loop.
//!
//! All time is integer ticks. No floating-point rounding is used to decide
//! whether two attacks coincide or satisfy a separation constraint.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Wire schema identifier for EnsembleGrooveV1.
pub const ENSEMBLE_GROOVE_SCHEMA_V1: &str = "ensemble-groove-v1";

/// Hard cap on the least-common-multiple horizon, in ticks.
pub const MAX_SUPERCYCLE_TICKS: u64 = 1_048_576;

/// Hard cap on materialized attacks across the complete supercycle.
pub const MAX_MATERIALIZED_ATTACKS: usize = 20_000;

/// Hard cap on pairwise attack comparisons during one analysis.
pub const MAX_PAIRWISE_COMPARISONS: usize = 10_000_000;

/// One attack in a repeating rhythmic layer.
///
/// accent is an ordinal, role-local level from 0 (unaccented) to 3
/// (strongest). It is not interpreted as loudness in dB or as a universal
/// cross-cultural accent vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveEventV1 {
    /// Attack position within the owning layer's cycle.
    pub onset_tick: u32,
    /// Relative, ordinal accent level in [0, 3].
    pub accent: u8,
}

/// An independently repeating ensemble part.
///
/// role is descriptive (for example bell, bass, lead_guitar, log_drum, or
/// lead_voice); the analyzer does not infer a tradition from the role name.
/// cycle_ticks may differ between layers, allowing a shared ensemble to
/// express cross-cycle relationships.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveLayerV1 {
    /// Stable machine ID, preferably snake_case.
    pub id: String,
    /// Musical function of this layer, not an instrument identity claim.
    pub role: String,
    /// Number of ticks in one repetition of this layer.
    pub cycle_ticks: u32,
    /// Attacks within one cycle. An empty list represents an intentional rest.
    pub events: Vec<GrooveEventV1>,
}

/// Pairwise relationship that a candidate ensemble may be required to obey.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrooveConstraintKindV1 {
    /// The two layers may never attack on the same tick in the full
    /// supercycle. Use only when the arrangement contract really requires
    /// separation; intentional unisons are valid in many traditions.
    NoCoincidentAttacks,
    /// Every attack from one layer must be at least this many ticks away from
    /// every attack from the other layer, measured circularly over the
    /// supercycle.
    MinimumCircularSeparation {
        minimum_ticks: u32,
    },
}

/// A named, directed pair of layer IDs and their intended relationship.
///
/// The analysis is symmetric for the current constraint set, but the source
/// and target remain explicit so the wire contract can grow to include
/// directed call/response and leader/follower relationships without changing
/// layer identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveConstraintV1 {
    pub left_layer_id: String,
    pub right_layer_id: String,
    pub rule: GrooveConstraintKindV1,
}

/// Versioned ensemble groove contract.
///
/// ticks_per_beat declares the quantization resolution used by the author.
/// It does not force every note onto the beat: offbeat and cross-rhythm
/// positions are ordinary integer offsets inside the declared cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleGrooveV1 {
    pub schema: String,
    pub ticks_per_beat: u32,
    pub layers: Vec<GrooveLayerV1>,
    #[serde(default)]
    pub constraints: Vec<GrooveConstraintV1>,
}

/// A source/contract validation issue. Validation fails closed: an invalid
/// pattern cannot be measured as if it were a valid groove.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveValidationIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

/// Per-layer facts after repeating the layer over the complete supercycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrooveLayerAnalysisV1 {
    pub layer_id: String,
    pub role: String,
    pub cycle_ticks: u32,
    pub materialized_attack_count: u64,
    pub attacks_per_beat: f64,
    /// Counts for ordinal accent levels 0, 1, 2 and 3.
    pub accent_histogram: [u64; 4],
    /// Sorted unique attack positions modulo the shared supercycle.
    pub unique_phase_positions: Vec<u64>,
}

/// Pairwise timing facts for two layers over the same complete supercycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroovePairAnalysisV1 {
    pub left_layer_id: String,
    pub right_layer_id: String,
    pub left_attack_count: u64,
    pub right_attack_count: u64,
    pub possible_attack_pairs: u64,
    pub coincident_attack_pairs: u64,
    pub coincidence_rate: f64,
    /// None when at least one layer has no attacks.
    pub minimum_circular_separation_ticks: Option<u64>,
}

/// Result of checking one declared relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveConstraintResultV1 {
    pub left_layer_id: String,
    pub right_layer_id: String,
    pub rule: GrooveConstraintKindV1,
    pub observed_coincident_pairs: u64,
    pub observed_minimum_separation_ticks: Option<u64>,
    pub passed: bool,
}

/// Deterministic analysis of an ensemble groove contract.
///
/// This is a mechanical description, not a musicality or authenticity score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleGrooveAnalysisV1 {
    pub schema: String,
    pub supercycle_ticks: u64,
    pub supercycle_beats: f64,
    pub layer_analyses: Vec<GrooveLayerAnalysisV1>,
    pub pair_analyses: Vec<GroovePairAnalysisV1>,
    pub constraint_results: Vec<GrooveConstraintResultV1>,
    pub all_constraints_passed: bool,
}

/// Validate the contract and return the exact shared supercycle length.
pub fn validate_ensemble_groove(
    groove: &EnsembleGrooveV1,
) -> Result<u64, Vec<GrooveValidationIssue>> {
    let mut issues = Vec::new();

    if groove.schema != ENSEMBLE_GROOVE_SCHEMA_V1 {
        issues.push(issue(
            "unsupported_schema",
            "schema",
            format!(
                "expected {ENSEMBLE_GROOVE_SCHEMA_V1}, received {}",
                groove.schema
            ),
        ));
    }
    if groove.ticks_per_beat == 0 {
        issues.push(issue(
            "zero_ticks_per_beat",
            "ticks_per_beat",
            "ticks_per_beat must be positive",
        ));
    }
    if groove.layers.is_empty() {
        issues.push(issue(
            "empty_layers",
            "layers",
            "at least one rhythmic layer is required",
        ));
    }

    let mut ids = HashSet::new();
    let mut supercycle = 1u64;
    for (layer_index, layer) in groove.layers.iter().enumerate() {
        let path = format!("layers[{layer_index}]");
        if layer.id.trim().is_empty() {
            issues.push(issue(
                "empty_layer_id",
                format!("{path}.id"),
                "layer ID must not be empty",
            ));
        } else if !ids.insert(layer.id.as_str()) {
            issues.push(issue(
                "duplicate_layer_id",
                format!("{path}.id"),
                format!("duplicate layer ID {}", layer.id),
            ));
        }
        if layer.role.trim().is_empty() {
            issues.push(issue(
                "empty_layer_role",
                format!("{path}.role"),
                "layer role must not be empty",
            ));
        }
        if layer.cycle_ticks == 0 {
            issues.push(issue(
                "zero_cycle",
                format!("{path}.cycle_ticks"),
                "layer cycle must be positive",
            ));
            continue;
        }
        for (event_index, event) in layer.events.iter().enumerate() {
            if event.onset_tick >= layer.cycle_ticks {
                issues.push(issue(
                    "onset_outside_cycle",
                    format!("{path}.events[{event_index}].onset_tick"),
                    format!(
                        "onset {} must be smaller than cycle length {}",
                        event.onset_tick, layer.cycle_ticks
                    ),
                ));
            }
            if event.accent > 3 {
                issues.push(issue(
                    "accent_out_of_range",
                    format!("{path}.events[{event_index}].accent"),
                    format!("accent {} is outside [0, 3]", event.accent),
                ));
            }
        }

        match checked_lcm(supercycle, layer.cycle_ticks as u64) {
            Some(next) if next <= MAX_SUPERCYCLE_TICKS => supercycle = next,
            Some(next) => issues.push(issue(
                "supercycle_too_large",
                format!("{path}.cycle_ticks"),
                format!(
                    "LCM horizon {next} exceeds the {MAX_SUPERCYCLE_TICKS}-tick safety limit"
                ),
            )),
            None => issues.push(issue(
                "supercycle_overflow",
                format!("{path}.cycle_ticks"),
                "LCM computation overflowed",
            )),
        }
    }

    if groove.ticks_per_beat > 0
        && supercycle % groove.ticks_per_beat as u64 != 0
    {
        issues.push(issue(
            "supercycle_fractional_beats",
            "layers",
            "shared supercycle is not an integer number of declared beats",
        ));
    }

    let total_attacks = groove.layers.iter().try_fold(0u64, |total, layer| {
        if layer.cycle_ticks == 0 {
            return Some(total);
        }
        let repetitions = supercycle / layer.cycle_ticks as u64;
        total.checked_add(repetitions.checked_mul(layer.events.len() as u64)?)
    });
    match total_attacks {
        Some(total) if total <= MAX_MATERIALIZED_ATTACKS as u64 => {}
        Some(total) => issues.push(issue(
            "too_many_materialized_attacks",
            "layers",
            format!(
                "supercycle materializes {total} attacks; limit is {MAX_MATERIALIZED_ATTACKS}"
            ),
        )),
        None => issues.push(issue(
            "attack_count_overflow",
            "layers",
            "materialized attack count overflowed",
        )),
    }

    for (constraint_index, constraint) in groove.constraints.iter().enumerate() {
        let path = format!("constraints[{constraint_index}]");
        if constraint.left_layer_id == constraint.right_layer_id {
            issues.push(issue(
                "self_constraint",
                &path,
                "a layer cannot constrain itself",
            ));
        }
        if !ids.contains(constraint.left_layer_id.as_str()) {
            issues.push(issue(
                "unknown_constraint_layer",
                format!("{path}.left_layer_id"),
                format!("unknown layer {}", constraint.left_layer_id),
            ));
        }
        if !ids.contains(constraint.right_layer_id.as_str()) {
            issues.push(issue(
                "unknown_constraint_layer",
                format!("{path}.right_layer_id"),
                format!("unknown layer {}", constraint.right_layer_id),
            ));
        }
        if let GrooveConstraintKindV1::MinimumCircularSeparation { minimum_ticks } =
            constraint.rule
        {
            if minimum_ticks == 0 || minimum_ticks as u64 > supercycle / 2 {
                issues.push(issue(
                    "invalid_minimum_separation",
                    format!("{path}.rule.minimum_ticks"),
                    "minimum separation must be positive and no greater than half the supercycle",
                ));
            }
        }
    }

    if issues.is_empty() {
        Ok(supercycle)
    } else {
        Err(issues)
    }
}

/// Analyze all rhythmic layers and their pairwise timing relationships.
///
/// Attack positions are materialized over the least-common-multiple horizon.
/// Constraints are checked against that complete horizon, not against a
/// single bar or an arbitrary excerpt.
pub fn analyze_ensemble_groove(
    groove: &EnsembleGrooveV1,
) -> Result<EnsembleGrooveAnalysisV1, Vec<GrooveValidationIssue>> {
    let supercycle_ticks = validate_ensemble_groove(groove)?;
    let beats = supercycle_ticks as f64 / groove.ticks_per_beat as f64;

    let materialized = groove
        .layers
        .iter()
        .map(|layer| materialize_layer(layer, supercycle_ticks))
        .collect::<Vec<_>>();

    let mut layer_analyses = Vec::with_capacity(groove.layers.len());
    for (layer, attacks) in groove.layers.iter().zip(&materialized) {
        let mut accent_histogram = [0u64; 4];
        let mut positions = Vec::with_capacity(attacks.len());
        for &(onset, accent) in attacks {
            accent_histogram[accent as usize] += 1;
            positions.push(onset);
        }
        positions.sort_unstable();
        positions.dedup();
        layer_analyses.push(GrooveLayerAnalysisV1 {
            layer_id: layer.id.clone(),
            role: layer.role.clone(),
            cycle_ticks: layer.cycle_ticks,
            materialized_attack_count: attacks.len() as u64,
            attacks_per_beat: attacks.len() as f64 / beats,
            accent_histogram,
            unique_phase_positions: positions,
        });
    }

    let pair_count = groove.layers.len().saturating_mul(groove.layers.len().saturating_sub(1)) / 2;
    let mut pair_analyses = Vec::with_capacity(pair_count);
    let mut total_pairwise_comparisons = 0usize;
    for left_index in 0..groove.layers.len() {
        for right_index in (left_index + 1)..groove.layers.len() {
            let left = &materialized[left_index];
            let right = &materialized[right_index];
            let comparisons = left.len().saturating_mul(right.len());
            total_pairwise_comparisons =
                total_pairwise_comparisons.saturating_add(comparisons);
            if total_pairwise_comparisons > MAX_PAIRWISE_COMPARISONS {
                return Err(vec![issue(
                    "pairwise_work_limit",
                    "layers",
                    format!(
                        "ensemble requires {total_pairwise_comparisons} total attack comparisons; limit is {MAX_PAIRWISE_COMPARISONS}"
                    ),
                )]);
            }
            pair_analyses.push(analyze_pair(
                &groove.layers[left_index].id,
                &groove.layers[right_index].id,
                left,
                right,
                supercycle_ticks,
            ));
        }
    }

    let mut constraint_results = Vec::with_capacity(groove.constraints.len());
    for constraint in &groove.constraints {
        let pair = pair_analyses
            .iter()
            .find(|pair| {
                (pair.left_layer_id == constraint.left_layer_id
                    && pair.right_layer_id == constraint.right_layer_id)
                    || (pair.left_layer_id == constraint.right_layer_id
                        && pair.right_layer_id == constraint.left_layer_id)
            })
            .expect("constraint layers were validated and all layer pairs are analyzed");
        let passed = match constraint.rule {
            GrooveConstraintKindV1::NoCoincidentAttacks => {
                pair.coincident_attack_pairs == 0
            }
            GrooveConstraintKindV1::MinimumCircularSeparation { minimum_ticks } => pair
                .minimum_circular_separation_ticks
                .is_some_and(|observed| observed >= minimum_ticks as u64),
        };
        constraint_results.push(GrooveConstraintResultV1 {
            left_layer_id: constraint.left_layer_id.clone(),
            right_layer_id: constraint.right_layer_id.clone(),
            rule: constraint.rule,
            observed_coincident_pairs: pair.coincident_attack_pairs,
            observed_minimum_separation_ticks: pair.minimum_circular_separation_ticks,
            passed,
        });
    }

    let all_constraints_passed = constraint_results.iter().all(|result| result.passed);
    Ok(EnsembleGrooveAnalysisV1 {
        schema: "ensemble-groove-analysis-v1".into(),
        supercycle_ticks,
        supercycle_beats: beats,
        layer_analyses,
        pair_analyses,
        constraint_results,
        all_constraints_passed,
    })
}

fn materialize_layer(layer: &GrooveLayerV1, horizon: u64) -> Vec<(u64, u8)> {
    let mut attacks = Vec::new();
    let mut base = 0u64;
    while base < horizon {
        for event in &layer.events {
            attacks.push((base + event.onset_tick as u64, event.accent));
        }
        base += layer.cycle_ticks as u64;
    }
    attacks.sort_unstable_by_key(|(onset, accent)| (*onset, *accent));
    attacks
}

fn analyze_pair(
    left_id: &str,
    right_id: &str,
    left: &[(u64, u8)],
    right: &[(u64, u8)],
    horizon: u64,
) -> GroovePairAnalysisV1 {
    let possible_attack_pairs = (left.len() as u64).saturating_mul(right.len() as u64);
    let mut coincident_attack_pairs = 0u64;
    let mut minimum_separation = None::<u64>;

    for &(left_onset, _) in left {
        for &(right_onset, _) in right {
            let forward = left_onset.abs_diff(right_onset);
            let circular = forward.min(horizon - forward.min(horizon));
            if circular == 0 {
                coincident_attack_pairs += 1;
            }
            minimum_separation = Some(
                minimum_separation
                    .map(|current| current.min(circular))
                    .unwrap_or(circular),
            );
        }
    }

    GroovePairAnalysisV1 {
        left_layer_id: left_id.into(),
        right_layer_id: right_id.into(),
        left_attack_count: left.len() as u64,
        right_attack_count: right.len() as u64,
        possible_attack_pairs,
        coincident_attack_pairs,
        coincidence_rate: if possible_attack_pairs == 0 {
            0.0
        } else {
            coincident_attack_pairs as f64 / possible_attack_pairs as f64
        },
        minimum_circular_separation_ticks: minimum_separation,
    }
}

fn checked_lcm(left: u64, right: u64) -> Option<u64> {
    if left == 0 || right == 0 {
        return None;
    }
    (left / gcd(left, right)).checked_mul(right)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn issue(
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) -> GrooveValidationIssue {
    GrooveValidationIssue {
        code: code.into(),
        path: path.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross_cycle_fixture() -> EnsembleGrooveV1 {
        EnsembleGrooveV1 {
            schema: ENSEMBLE_GROOVE_SCHEMA_V1.into(),
            ticks_per_beat: 24,
            layers: vec![
                GrooveLayerV1 {
                    id: "timekeeper".into(),
                    role: "reference_pulse".into(),
                    cycle_ticks: 48,
                    events: [0, 12, 24, 36]
                        .into_iter()
                        .map(|onset_tick| GrooveEventV1 {
                            onset_tick,
                            accent: if onset_tick == 0 { 3 } else { 1 },
                        })
                        .collect(),
                },
                GrooveLayerV1 {
                    id: "independent_line".into(),
                    role: "independent_counter_rhythm".into(),
                    cycle_ticks: 32,
                    events: vec![
                        GrooveEventV1 {
                            onset_tick: 10,
                            accent: 2,
                        },
                        GrooveEventV1 {
                            onset_tick: 26,
                            accent: 0,
                        },
                    ],
                },
            ],
            constraints: vec![
                GrooveConstraintV1 {
                    left_layer_id: "timekeeper".into(),
                    right_layer_id: "independent_line".into(),
                    rule: GrooveConstraintKindV1::NoCoincidentAttacks,
                },
                GrooveConstraintV1 {
                    left_layer_id: "timekeeper".into(),
                    right_layer_id: "independent_line".into(),
                    rule: GrooveConstraintKindV1::MinimumCircularSeparation {
                        minimum_ticks: 2,
                    },
                },
            ],
        }
    }

    #[test]
    fn computes_full_lcm_supercycle_and_role_density() {
        let report = analyze_ensemble_groove(&cross_cycle_fixture()).unwrap();
        assert_eq!(report.supercycle_ticks, 96);
        assert_eq!(report.supercycle_beats, 4.0);
        assert_eq!(report.layer_analyses[0].materialized_attack_count, 8);
        assert_eq!(report.layer_analyses[1].materialized_attack_count, 6);
        assert_eq!(report.layer_analyses[0].attacks_per_beat, 2.0);
        assert_eq!(report.layer_analyses[1].attacks_per_beat, 1.5);
        assert!(report.all_constraints_passed);
    }

    #[test]
    fn detects_cross_cycle_collisions_that_a_single_bar_probe_misses() {
        let mut groove = cross_cycle_fixture();
        groove.layers[1].events = vec![
            GrooveEventV1 {
                onset_tick: 8,
                accent: 2,
            },
            GrooveEventV1 {
                onset_tick: 24,
                accent: 0,
            },
        ];
        let report = analyze_ensemble_groove(&groove).unwrap();
        let pair = &report.pair_analyses[0];
        assert_eq!(pair.coincident_attack_pairs, 2);
        assert!(!report.all_constraints_passed);
        assert!(report
            .constraint_results
            .iter()
            .any(|result| !result.passed));
    }

    #[test]
    fn reports_accent_distribution_without_equating_accent_to_loudness() {
        let report = analyze_ensemble_groove(&cross_cycle_fixture()).unwrap();
        assert_eq!(report.layer_analyses[0].accent_histogram, [0, 6, 0, 2]);
        assert_eq!(report.layer_analyses[1].accent_histogram, [3, 0, 3, 0]);
    }

    #[test]
    fn rejects_unknown_layers_and_unrepresentable_accent_levels() {
        let mut groove = cross_cycle_fixture();
        groove.layers[0].events[0].accent = 4;
        groove.constraints[0].right_layer_id = "missing".into();
        let issues = validate_ensemble_groove(&groove).unwrap_err();
        assert!(issues.iter().any(|item| item.code == "accent_out_of_range"));
        assert!(issues.iter().any(|item| item.code == "unknown_constraint_layer"));
    }

    #[test]
    fn rejects_lcm_expansion_above_the_declared_bound() {
        let groove = EnsembleGrooveV1 {
            schema: ENSEMBLE_GROOVE_SCHEMA_V1.into(),
            ticks_per_beat: 1,
            layers: vec![
                GrooveLayerV1 {
                    id: "a".into(),
                    role: "layer_a".into(),
                    cycle_ticks: 65_521,
                    events: vec![],
                },
                GrooveLayerV1 {
                    id: "b".into(),
                    role: "layer_b".into(),
                    cycle_ticks: 65_519,
                    events: vec![],
                },
            ],
            constraints: vec![],
        };
        let issues = validate_ensemble_groove(&groove).unwrap_err();
        assert!(issues
            .iter()
            .any(|item| item.code == "supercycle_too_large"));
    }

    #[test]
    fn empty_layer_is_a_valid_explicit_rest() {
        let mut groove = cross_cycle_fixture();
        groove.layers[1].events.clear();
        groove.constraints.clear();
        let report = analyze_ensemble_groove(&groove).unwrap();
        assert_eq!(report.layer_analyses[1].materialized_attack_count, 0);
        assert_eq!(
            report.pair_analyses[0].minimum_circular_separation_ticks,
            None
        );
    }
}
