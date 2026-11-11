// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent strict reference for context-bound motor efficiency maps.
//!
//! The map is a single frozen configuration/context slice over motor torque
//! and motor speed. It performs bilinear interpolation only inside the
//! characterized rectangle. It never extrapolates, clamps, or substitutes a
//! different configuration or context.
//!
//! Node uncertainty bounds are combined as a weighted absolute bound plus a
//! map-level systematic bound. This is not a probabilistic confidence interval.

use std::fmt;

pub const MODEL_VERSION: &str = "electric-propulsion-map-oracle-v0.1.0";

/// One measured or otherwise evidence-qualified map node.
///
/// Efficiency is a dimensionless fraction in (0, 1]. The uncertainty is an
/// absolute bound in efficiency-fraction units, not a relative percentage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapNode {
    pub efficiency: f64,
    pub absolute_uncertainty_bound: f64,
}

/// A validated two-dimensional efficiency map bound to one configuration and
/// one operating-context slice.
///
/// Nodes are stored row-major: speed is the outer index and torque varies
/// fastest, so node index = speed_index * torque_axis_len + torque_index.
#[derive(Debug, Clone, PartialEq)]
pub struct EfficiencyMap2D {
    map_id: String,
    configuration_digest: String,
    context_digest: String,
    motor_torque_axis_nm: Vec<f64>,
    motor_speed_axis_rpm: Vec<f64>,
    nodes: Vec<MapNode>,
    systematic_uncertainty_bound: f64,
}

/// Query for one exact map configuration/context and one motor operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapQuery<'a> {
    pub configuration_digest: &'a str,
    pub context_digest: &'a str,
    pub motor_torque_nm: f64,
    pub motor_speed_rpm: f64,
}

/// The interpolation cell uses corner order:
/// lower-speed/lower-torque, lower-speed/upper-torque,
/// upper-speed/lower-torque, upper-speed/upper-torque.
#[derive(Debug, Clone, PartialEq)]
pub struct MapLookup {
    pub model_version: &'static str,
    pub map_id: String,
    pub configuration_digest: String,
    pub context_digest: String,
    pub motor_torque_nm: f64,
    pub motor_speed_rpm: f64,
    pub efficiency: f64,
    pub absolute_uncertainty_bound: f64,
    pub efficiency_lower_bound: f64,
    pub efficiency_upper_bound: f64,
    pub torque_cell_lower_index: usize,
    pub speed_cell_lower_index: usize,
    pub interpolation_weights: [f64; 4],
}

/// Typed non-PASS outcomes for invalid maps and unsupported queries.
#[derive(Debug, Clone, PartialEq)]
pub enum MapError {
    InvalidMap(&'static str),
    InvalidQuery(&'static str),
    ConfigurationMismatch,
    ContextMismatch,
    OutOfDomain {
        axis: &'static str,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    NumericalFailure,
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MapError {}

impl EfficiencyMap2D {
    /// Construct and validate one immutable map slice.
    ///
    /// No map digest is inferred here. The caller must provide identity values
    /// from the upstream evidence/configuration manifest.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        map_id: impl Into<String>,
        configuration_digest: impl Into<String>,
        context_digest: impl Into<String>,
        motor_torque_axis_nm: Vec<f64>,
        motor_speed_axis_rpm: Vec<f64>,
        nodes: Vec<MapNode>,
        systematic_uncertainty_bound: f64,
    ) -> Result<Self, MapError> {
        let map_id = map_id.into();
        let configuration_digest = configuration_digest.into();
        let context_digest = context_digest.into();

        if map_id.trim().is_empty() {
            return Err(MapError::InvalidMap("map identity must not be empty"));
        }
        if configuration_digest.trim().is_empty() {
            return Err(MapError::InvalidMap(
                "configuration digest must not be empty",
            ));
        }
        if context_digest.trim().is_empty() {
            return Err(MapError::InvalidMap("context digest must not be empty"));
        }
        validate_axis(&motor_torque_axis_nm, "torque")?;
        validate_axis(&motor_speed_axis_rpm, "speed")?;

        if !systematic_uncertainty_bound.is_finite()
            || !(0.0..=1.0).contains(&systematic_uncertainty_bound)
        {
            return Err(MapError::InvalidMap(
                "systematic uncertainty bound must be finite and in [0, 1]",
            ));
        }

        let expected_nodes = motor_torque_axis_nm
            .len()
            .checked_mul(motor_speed_axis_rpm.len())
            .ok_or(MapError::InvalidMap("map dimensions overflow"))?;
        if nodes.len() != expected_nodes {
            return Err(MapError::InvalidMap(
                "node count must equal torque-axis length times speed-axis length",
            ));
        }

        for node in &nodes {
            if !node.efficiency.is_finite() || !(0.0 < node.efficiency && node.efficiency <= 1.0)
            {
                return Err(MapError::InvalidMap(
                    "node efficiency must be finite and in (0, 1]",
                ));
            }
            if !node.absolute_uncertainty_bound.is_finite()
                || !(0.0..=1.0).contains(&node.absolute_uncertainty_bound)
            {
                return Err(MapError::InvalidMap(
                    "node uncertainty bound must be finite and in [0, 1]",
                ));
            }
        }

        Ok(Self {
            map_id,
            configuration_digest,
            context_digest,
            motor_torque_axis_nm,
            motor_speed_axis_rpm,
            nodes,
            systematic_uncertainty_bound,
        })
    }

    /// Interpolate one operating point, rejecting any unsupported context or
    /// coordinate rather than extrapolating or substituting another map.
    pub fn lookup(&self, query: MapQuery<'_>) -> Result<MapLookup, MapError> {
        if query.configuration_digest != self.configuration_digest {
            return Err(MapError::ConfigurationMismatch);
        }
        if query.context_digest != self.context_digest {
            return Err(MapError::ContextMismatch);
        }
        if !query.motor_torque_nm.is_finite() || !query.motor_speed_rpm.is_finite() {
            return Err(MapError::InvalidQuery(
                "motor torque and speed must be finite",
            ));
        }

        let (torque_lower, torque_upper, torque_fraction) =
            bracket(&self.motor_torque_axis_nm, query.motor_torque_nm, "torque")?;
        let (speed_lower, speed_upper, speed_fraction) =
            bracket(&self.motor_speed_axis_rpm, query.motor_speed_rpm, "speed")?;

        let one_minus_torque = 1.0 - torque_fraction;
        let one_minus_speed = 1.0 - speed_fraction;
        let weights = [
            one_minus_torque * one_minus_speed,
            torque_fraction * one_minus_speed,
            one_minus_torque * speed_fraction,
            torque_fraction * speed_fraction,
        ];

        let torque_len = self.motor_torque_axis_nm.len();
        let node_indices = [
            speed_lower * torque_len + torque_lower,
            speed_lower * torque_len + torque_upper,
            speed_upper * torque_len + torque_lower,
            speed_upper * torque_len + torque_upper,
        ];

        let mut efficiency = 0.0;
        let mut nodal_uncertainty = 0.0;
        for (weight, index) in weights.iter().zip(node_indices) {
            let node = self.nodes[index];
            efficiency += *weight * node.efficiency;
            nodal_uncertainty += *weight * node.absolute_uncertainty_bound;
        }
        let absolute_uncertainty_bound =
            nodal_uncertainty + self.systematic_uncertainty_bound;

        if !efficiency.is_finite()
            || !absolute_uncertainty_bound.is_finite()
            || !(0.0 < efficiency && efficiency <= 1.0)
        {
            return Err(MapError::NumericalFailure);
        }

        Ok(MapLookup {
            model_version: MODEL_VERSION,
            map_id: self.map_id.clone(),
            configuration_digest: self.configuration_digest.clone(),
            context_digest: self.context_digest.clone(),
            motor_torque_nm: query.motor_torque_nm,
            motor_speed_rpm: query.motor_speed_rpm,
            efficiency,
            absolute_uncertainty_bound,
            efficiency_lower_bound: (efficiency - absolute_uncertainty_bound).max(0.0),
            efficiency_upper_bound: (efficiency + absolute_uncertainty_bound).min(1.0),
            torque_cell_lower_index: torque_lower,
            speed_cell_lower_index: speed_lower,
            interpolation_weights: weights,
        })
    }
}

fn validate_axis(axis: &[f64], name: &'static str) -> Result<(), MapError> {
    if axis.len() < 2 {
        return Err(MapError::InvalidMap(
            if name == "torque" {
                "torque axis must contain at least two points"
            } else {
                "speed axis must contain at least two points"
            },
        ));
    }
    if axis.iter().any(|value| !value.is_finite()) {
        return Err(MapError::InvalidMap(
            if name == "torque" {
                "torque axis must contain only finite values"
            } else {
                "speed axis must contain only finite values"
            },
        ));
    }
    if axis.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(MapError::InvalidMap(
            if name == "torque" {
                "torque axis must be strictly increasing"
            } else {
                "speed axis must be strictly increasing"
            },
        ));
    }
    if axis
        .windows(2)
        .any(|pair| !(pair[1] - pair[0]).is_finite())
    {
        return Err(MapError::InvalidMap(
            if name == "torque" {
                "torque axis intervals must be finite"
            } else {
                "speed axis intervals must be finite"
            },
        ));
    }
    Ok(())
}

fn bracket(
    axis: &[f64],
    value: f64,
    name: &'static str,
) -> Result<(usize, usize, f64), MapError> {
    let minimum = axis[0];
    let maximum = axis[axis.len() - 1];
    if value < minimum || value > maximum {
        return Err(MapError::OutOfDomain {
            axis: name,
            value,
            minimum,
            maximum,
        });
    }

    let upper_bound = axis.partition_point(|coordinate| *coordinate <= value);
    let lower_index = if upper_bound == 0 {
        0
    } else if upper_bound >= axis.len() {
        axis.len() - 2
    } else {
        upper_bound - 1
    };
    let upper_index = lower_index + 1;
    let lower = axis[lower_index];
    let upper = axis[upper_index];
    let fraction = (value - lower) / (upper - lower);
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
        return Err(MapError::NumericalFailure);
    }
    Ok((lower_index, upper_index, fraction))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> EfficiencyMap2D {
        EfficiencyMap2D::new(
            "motor-map-test",
            "config-sha-001",
            "context-sha-001",
            vec![0.0, 10.0],
            vec![1_000.0, 2_000.0],
            vec![
                MapNode {
                    efficiency: 0.70,
                    absolute_uncertainty_bound: 0.01,
                },
                MapNode {
                    efficiency: 0.80,
                    absolute_uncertainty_bound: 0.02,
                },
                MapNode {
                    efficiency: 0.90,
                    absolute_uncertainty_bound: 0.03,
                },
                MapNode {
                    efficiency: 0.95,
                    absolute_uncertainty_bound: 0.04,
                },
            ],
            0.005,
        )
        .unwrap()
    }

    fn query(torque_nm: f64, speed_rpm: f64) -> MapQuery<'static> {
        MapQuery {
            configuration_digest: "config-sha-001",
            context_digest: "context-sha-001",
            motor_torque_nm: torque_nm,
            motor_speed_rpm: speed_rpm,
        }
    }

    #[test]
    fn bilinear_lookup_matches_independent_asymmetric_reference() {
        let result = fixture().lookup(query(2.5, 1_500.0)).unwrap();
        assert!((result.efficiency - 0.81875).abs() < 1.0e-12);
        assert_eq!(result.torque_cell_lower_index, 0);
        assert_eq!(result.speed_cell_lower_index, 0);
        assert_eq!(result.interpolation_weights, [0.375, 0.125, 0.375, 0.125]);
    }

    #[test]
    fn exact_grid_corners_return_their_node_values() {
        let map = fixture();
        let cases = [
            (0.0, 1_000.0, 0.70),
            (10.0, 1_000.0, 0.80),
            (0.0, 2_000.0, 0.90),
            (10.0, 2_000.0, 0.95),
        ];
        for (torque, speed, expected) in cases {
            let result = map.lookup(query(torque, speed)).unwrap();
            assert!((result.efficiency - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn constant_map_is_constant_throughout_its_domain() {
        let map = EfficiencyMap2D::new(
            "constant",
            "config",
            "context",
            vec![0.0, 5.0, 10.0],
            vec![1_000.0, 1_500.0, 2_000.0],
            vec![
                MapNode { efficiency: 0.9, absolute_uncertainty_bound: 0.0 };
                9
            ],
            0.0,
        )
        .unwrap();
        for (torque, speed) in [(0.0, 1_000.0), (2.5, 1_250.0), (7.0, 1_900.0), (10.0, 2_000.0)] {
            let result = map.lookup(MapQuery {
                configuration_digest: "config",
                context_digest: "context",
                motor_torque_nm: torque,
                motor_speed_rpm: speed,
            }).unwrap();
            assert!((result.efficiency - 0.9).abs() < 1.0e-12);
        }
    }

    #[test]
    fn out_of_domain_torque_and_speed_are_rejected_without_clamping() {
        let map = fixture();
        for torque in [-0.001, 10.001] {
            assert!(matches!(
                map.lookup(query(torque, 1_500.0)),
                Err(MapError::OutOfDomain { axis: "torque", .. })
            ));
        }
        for speed in [999.0, 2_001.0] {
            assert!(matches!(
                map.lookup(query(5.0, speed)),
                Err(MapError::OutOfDomain { axis: "speed", .. })
            ));
        }
    }

    #[test]
    fn configuration_and_context_mismatches_are_independent_non_passes() {
        let map = fixture();
        assert_eq!(
            map.lookup(MapQuery {
                configuration_digest: "other-config",
                context_digest: "context-sha-001",
                motor_torque_nm: 5.0,
                motor_speed_rpm: 1_500.0,
            }),
            Err(MapError::ConfigurationMismatch)
        );
        assert_eq!(
            map.lookup(MapQuery {
                configuration_digest: "config-sha-001",
                context_digest: "other-context",
                motor_torque_nm: 5.0,
                motor_speed_rpm: 1_500.0,
            }),
            Err(MapError::ContextMismatch)
        );
    }

    #[test]
    fn non_finite_queries_are_rejected() {
        let map = fixture();
        for query_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                map.lookup(query(query_value, 1_500.0)),
                Err(MapError::InvalidQuery(_))
            ));
            assert!(matches!(
                map.lookup(query(5.0, query_value)),
                Err(MapError::InvalidQuery(_))
            ));
        }
    }

    #[test]
    fn conservative_uncertainty_adds_weighted_node_and_systematic_bounds() {
        let result = fixture().lookup(query(2.5, 1_500.0)).unwrap();
        assert!((result.absolute_uncertainty_bound - 0.0275).abs() < 1.0e-12);
        assert!((result.efficiency_lower_bound - 0.79125).abs() < 1.0e-12);
        assert!((result.efficiency_upper_bound - 0.84625).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_axes_and_nodes_fail_closed() {
        let base_nodes = vec![
            MapNode { efficiency: 0.9, absolute_uncertainty_bound: 0.01 };
            4
        ];
        let invalid = [
            EfficiencyMap2D::new(
                "m",
                "c",
                "x",
                vec![0.0],
                vec![1.0, 2.0],
                base_nodes.clone(),
                0.0,
            ),
            EfficiencyMap2D::new(
                "m",
                "c",
                "x",
                vec![0.0, 0.0],
                vec![1.0, 2.0],
                base_nodes.clone(),
                0.0,
            ),
            EfficiencyMap2D::new(
                "m",
                "c",
                "x",
                vec![0.0, f64::NAN],
                vec![1.0, 2.0],
                base_nodes.clone(),
                0.0,
            ),
            EfficiencyMap2D::new(
                "m",
                "c",
                "x",
                vec![0.0, 1.0],
                vec![2.0, 1.0],
                base_nodes.clone(),
                0.0,
            ),
            EfficiencyMap2D::new(
                "m",
                "c",
                "x",
                vec![0.0, 1.0],
                vec![1.0, 2.0],
                base_nodes[..3].to_vec(),
                0.0,
            ),
        ];
        assert!(invalid.into_iter().all(|result| result.is_err()));
    }

    #[test]
    fn invalid_efficiency_and_uncertainty_nodes_are_rejected() {
        for node in [
            MapNode { efficiency: 0.0, absolute_uncertainty_bound: 0.0 },
            MapNode { efficiency: 1.01, absolute_uncertainty_bound: 0.0 },
            MapNode { efficiency: f64::NAN, absolute_uncertainty_bound: 0.0 },
            MapNode { efficiency: 0.9, absolute_uncertainty_bound: -0.01 },
            MapNode { efficiency: 0.9, absolute_uncertainty_bound: f64::INFINITY },
            MapNode { efficiency: 0.9, absolute_uncertainty_bound: 1.01 },
        ] {
            assert!(
                EfficiencyMap2D::new(
                    "m",
                    "c",
                    "x",
                    vec![0.0, 1.0],
                    vec![1_000.0, 2_000.0],
                    vec![node; 4],
                    0.0,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn blank_identity_and_invalid_systematic_uncertainty_are_rejected() {
        let nodes = vec![
            MapNode { efficiency: 0.9, absolute_uncertainty_bound: 0.01 };
            4
        ];
        for (map_id, config, context) in [
            ("", "c", "x"),
            ("m", "", "x"),
            ("m", "c", ""),
            (" ", "c", "x"),
        ] {
            assert!(
                EfficiencyMap2D::new(
                    map_id,
                    config,
                    context,
                    vec![0.0, 1.0],
                    vec![1_000.0, 2_000.0],
                    nodes.clone(),
                    0.0,
                )
                .is_err()
            );
        }
        for uncertainty in [-0.01, 1.01, f64::NAN] {
            assert!(
                EfficiencyMap2D::new(
                    "m",
                    "c",
                    "x",
                    vec![0.0, 1.0],
                    vec![1_000.0, 2_000.0],
                    nodes.clone(),
                    uncertainty,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn exact_internal_grid_line_uses_deterministic_upper_adjacent_cell() {
        let map = EfficiencyMap2D::new(
            "grid-lines",
            "config",
            "context",
            vec![0.0, 5.0, 10.0],
            vec![1_000.0, 1_500.0, 2_000.0],
            vec![
                MapNode {
                    efficiency: 0.8,
                    absolute_uncertainty_bound: 0.0,
                };
                9
            ],
            0.0,
        )
        .unwrap();
        let result = map
            .lookup(MapQuery {
                configuration_digest: "config",
                context_digest: "context",
                motor_torque_nm: 5.0,
                motor_speed_rpm: 1_500.0,
            })
            .unwrap();
        assert_eq!(result.torque_cell_lower_index, 1);
        assert_eq!(result.speed_cell_lower_index, 1);
        assert_eq!(result.interpolation_weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn identical_frozen_inputs_replay_identically() {
        let map = fixture();
        let input = query(2.5, 1_500.0);
        assert_eq!(map.lookup(input), map.lookup(input));
    }
}
