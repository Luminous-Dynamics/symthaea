//! Event-triggered spatially structured transient nuisance for RCAF.
//!
//! This is a stylized motion-like observation artifact family. It combines an optional broad
//! transient component with localized graph-distance fields triggered at declared ROI seeds.
//! It does not simulate rigid-body motion, scanner acquisition physics, or real BOLD effects.

use crate::spatial_nuisance::SpatialGraph;
use std::fmt;

/// One transient event. The contribution begins at onset and decays exponentially.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionEvent {
    pub onset: usize,
    /// Signed event amplitude. Sign is part of the declared synthetic challenge.
    pub amplitude: f64,
    /// Exponential decay time constant, in samples.
    pub decay_samples: f64,
}

/// A localized event with a spatial origin in the explicit ROI graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizedMotionEvent {
    pub event: MotionEvent,
    pub seed_roi_index: usize,
}

/// Parameters for broad and localized transient fields.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionTransientConfig {
    pub time_points: usize,
    pub dose: f64,
    /// Uniform per-ROI contribution of the broad event train.
    pub broad_loading: f64,
    /// Contribution of the localized graph-distance field.
    pub localized_loading: f64,
    /// Graph-distance decay length for the localized spatial field. Edge length is the
    /// inverse adjacency weight; this is a declared synthetic graph-geometry convention.
    pub spatial_decay_length: f64,
    pub broad_events: Vec<MotionEvent>,
    pub localized_events: Vec<LocalizedMotionEvent>,
}

/// Generated motion-like nuisance. Layout: time x ROI.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionTransientField {
    pub roi_labels: Vec<String>,
    pub values: Vec<Vec<f64>>,
    pub broad_event_train: Vec<f64>,
    pub localized_event_trains: Vec<Vec<f64>>,
    /// Per-localized-event spatial loading; event order matches localized_events.
    pub localized_spatial_loadings: Vec<Vec<f64>>,
    pub localized_seed_roi_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotionTransientError {
    InvalidGraph(&'static str),
    InvalidConfig(&'static str),
    InvalidEvent { index: usize, reason: &'static str },
    InvalidSeedRoi { index: usize },
    NonFiniteGeneratedField,
}

impl fmt::Display for MotionTransientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(reason) => write!(f, "invalid motion transient graph: {reason}"),
            Self::InvalidConfig(reason) => write!(f, "invalid motion transient config: {reason}"),
            Self::InvalidEvent { index, reason } => {
                write!(f, "invalid motion event {index}: {reason}")
            }
            Self::InvalidSeedRoi { index } => {
                write!(
                    f,
                    "localized motion event seed ROI index {index} is outside graph"
                )
            }
            Self::NonFiniteGeneratedField => {
                write!(f, "motion transient generation produced non-finite values")
            }
        }
    }
}

impl std::error::Error for MotionTransientError {}

/// Generate deterministic broad and localized event-triggered nuisance fields.
///
/// The broad component is spatially constant. The localized component is a graph-distance
/// exponential field from each event's seed ROI. For a node r and seed s:
///
///     q_r = exp(-d_graph(s, r) / spatial_decay_length)
///
/// Edge traversal cost is the inverse adjacency weight; disconnected nodes have zero localized
/// loading. Each event train is amplitude * exp(-(t-onset)/decay_samples) for t >= onset.
/// Event times and amplitudes are explicit inputs, so one generated field can be reused exactly
/// in nuisance-only and signal-plus-nuisance counterfactuals.
pub fn generate_motion_transient_field(
    graph: &SpatialGraph,
    config: &MotionTransientConfig,
) -> Result<MotionTransientField, MotionTransientError> {
    graph
        .validate()
        .map_err(|_| MotionTransientError::InvalidGraph("graph validation failed"))?;
    if graph
        .adjacency
        .iter()
        .flatten()
        .any(|weight| *weight > 0.0 && !(1.0 / *weight).is_finite())
    {
        return Err(MotionTransientError::InvalidGraph(
            "inverse edge lengths must remain finite",
        ));
    }
    validate_config(config)?;

    let roi_count = graph.roi_labels.len();
    let mut broad_event_train = vec![0.0; config.time_points];
    for (event_index, event) in config.broad_events.iter().enumerate() {
        validate_event(event_index, event, config.time_points)?;
        accumulate_event_train(&mut broad_event_train, event);
    }

    if broad_event_train.iter().any(|value| !value.is_finite()) {
        return Err(MotionTransientError::NonFiniteGeneratedField);
    }

    let mut localized_event_trains = Vec::with_capacity(config.localized_events.len());
    let mut localized_spatial_loadings = Vec::with_capacity(config.localized_events.len());
    let mut localized_seed_roi_indices = Vec::with_capacity(config.localized_events.len());

    for (event_index, localized) in config.localized_events.iter().enumerate() {
        validate_event(event_index, &localized.event, config.time_points)?;
        if localized.seed_roi_index >= roi_count {
            return Err(MotionTransientError::InvalidSeedRoi {
                index: localized.seed_roi_index,
            });
        }
        let train = event_train(localized.event, config.time_points);
        if train.iter().any(|value| !value.is_finite()) {
            return Err(MotionTransientError::NonFiniteGeneratedField);
        }
        let spatial_loading =
            graph_distance_loading(graph, localized.seed_roi_index, config.spatial_decay_length);
        if spatial_loading.iter().any(|value| !value.is_finite()) {
            return Err(MotionTransientError::NonFiniteGeneratedField);
        }
        localized_event_trains.push(train);
        localized_spatial_loadings.push(spatial_loading);
        localized_seed_roi_indices.push(localized.seed_roi_index);
    }

    let mut values = vec![vec![0.0; roi_count]; config.time_points];
    for (time, row) in values.iter_mut().enumerate() {
        let broad_value = config.dose * config.broad_loading * broad_event_train[time];
        for (roi, value) in row.iter_mut().enumerate() {
            *value += broad_value;
            for (event_index, train) in localized_event_trains.iter().enumerate() {
                *value += config.dose
                    * config.localized_loading
                    * train[time]
                    * localized_spatial_loadings[event_index][roi];
            }
        }
    }

    if values.iter().flatten().any(|value| !value.is_finite()) {
        return Err(MotionTransientError::NonFiniteGeneratedField);
    }

    Ok(MotionTransientField {
        roi_labels: graph.roi_labels.clone(),
        values,
        broad_event_train,
        localized_event_trains,
        localized_spatial_loadings,
        localized_seed_roi_indices,
    })
}

fn validate_config(config: &MotionTransientConfig) -> Result<(), MotionTransientError> {
    if config.time_points < 2 {
        return Err(MotionTransientError::InvalidConfig(
            "at least two time points are required",
        ));
    }
    if !config.dose.is_finite() || config.dose < 0.0 {
        return Err(MotionTransientError::InvalidConfig(
            "dose must be finite and non-negative",
        ));
    }
    if !config.broad_loading.is_finite() || config.broad_loading < 0.0 {
        return Err(MotionTransientError::InvalidConfig(
            "broad loading must be finite and non-negative",
        ));
    }
    if !config.localized_loading.is_finite() || config.localized_loading < 0.0 {
        return Err(MotionTransientError::InvalidConfig(
            "localized loading must be finite and non-negative",
        ));
    }
    if !config.spatial_decay_length.is_finite() || config.spatial_decay_length <= 0.0 {
        return Err(MotionTransientError::InvalidConfig(
            "spatial decay length must be finite and positive",
        ));
    }
    if config.broad_events.is_empty() && config.localized_events.is_empty() {
        return Err(MotionTransientError::InvalidConfig(
            "at least one broad or localized motion event is required",
        ));
    }
    Ok(())
}

fn validate_event(
    index: usize,
    event: &MotionEvent,
    time_points: usize,
) -> Result<(), MotionTransientError> {
    if event.onset >= time_points {
        return Err(MotionTransientError::InvalidEvent {
            index,
            reason: "event onset must be inside the time series",
        });
    }
    if !event.amplitude.is_finite() {
        return Err(MotionTransientError::InvalidEvent {
            index,
            reason: "event amplitude must be finite",
        });
    }
    if !event.decay_samples.is_finite() || event.decay_samples <= 0.0 {
        return Err(MotionTransientError::InvalidEvent {
            index,
            reason: "decay time constant must be finite and positive",
        });
    }
    Ok(())
}

fn event_train(event: MotionEvent, time_points: usize) -> Vec<f64> {
    (0..time_points)
        .map(|time| {
            if time < event.onset {
                0.0
            } else {
                event.amplitude * (-((time - event.onset) as f64) / event.decay_samples).exp()
            }
        })
        .collect()
}

fn accumulate_event_train(train: &mut [f64], event: MotionEvent) {
    for (time, value) in train.iter_mut().enumerate().skip(event.onset) {
        *value += event.amplitude * (-((time - event.onset) as f64) / event.decay_samples).exp();
    }
}

fn graph_distance_loading(graph: &SpatialGraph, seed: usize, decay_length: f64) -> Vec<f64> {
    let n = graph.roi_labels.len();
    let mut distances = vec![f64::INFINITY; n];
    let mut visited = vec![false; n];
    distances[seed] = 0.0;

    for _ in 0..n {
        let current = (0..n)
            .filter(|index| !visited[*index] && distances[*index].is_finite())
            .min_by(|left, right| distances[*left].total_cmp(&distances[*right]));
        let Some(current) = current else {
            break;
        };
        visited[current] = true;

        for (neighbor, weight) in graph.adjacency[current].iter().enumerate() {
            if *weight <= 0.0 || visited[neighbor] {
                continue;
            }
            let edge_length = 1.0 / *weight;
            let candidate = distances[current] + edge_length;
            if candidate < distances[neighbor] {
                distances[neighbor] = candidate;
            }
        }
    }

    distances
        .iter()
        .map(|distance| {
            if distance.is_finite() {
                (-distance / decay_length).exp()
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph() -> SpatialGraph {
        let labels = (0..6).map(|index| format!("roi-{index}")).collect();
        let mut adjacency = vec![vec![0.0; 6]; 6];
        for index in 0..5 {
            adjacency[index][index + 1] = 1.0;
            adjacency[index + 1][index] = 1.0;
        }
        SpatialGraph::new(labels, adjacency).unwrap()
    }

    fn config() -> MotionTransientConfig {
        MotionTransientConfig {
            time_points: 24,
            dose: 0.3,
            broad_loading: 0.2,
            localized_loading: 0.8,
            spatial_decay_length: 2.0,
            broad_events: vec![MotionEvent {
                onset: 4,
                amplitude: 1.0,
                decay_samples: 3.0,
            }],
            localized_events: vec![LocalizedMotionEvent {
                event: MotionEvent {
                    onset: 8,
                    amplitude: 0.7,
                    decay_samples: 2.0,
                },
                seed_roi_index: 2,
            }],
        }
    }

    #[test]
    fn broad_event_is_zero_before_onset_and_decays_after_onset() {
        let field = generate_motion_transient_field(&path_graph(), &config()).unwrap();
        assert!(field.broad_event_train[..4]
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(field.broad_event_train[4], 1.0);
        assert!(field.broad_event_train[5] < field.broad_event_train[4]);
        assert!(field.broad_event_train[6] < field.broad_event_train[5]);
    }

    #[test]
    fn localized_field_is_strongest_at_seed_and_decays_with_graph_distance() {
        let field = generate_motion_transient_field(&path_graph(), &config()).unwrap();
        let loading = &field.localized_spatial_loadings[0];
        assert_eq!(loading[2], 1.0);
        assert!(loading[1] > loading[0]);
        assert!(loading[3] > loading[4]);
        assert!(loading[0] > 0.0);
    }

    #[test]
    fn localized_event_is_exactly_zero_before_its_onset() {
        let field = generate_motion_transient_field(&path_graph(), &config()).unwrap();
        assert!(field.localized_event_trains[0][..8]
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(field.localized_event_trains[0][8], 0.7);
    }

    #[test]
    fn disconnected_rois_receive_no_localized_component() {
        let labels = vec!["a".to_string(), "b".to_string(), "isolated".to_string()];
        let adjacency = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        let graph = SpatialGraph::new(labels, adjacency).unwrap();
        let mut config = config();
        config.time_points = 24;
        config.localized_events = vec![LocalizedMotionEvent {
            event: MotionEvent {
                onset: 3,
                amplitude: 1.0,
                decay_samples: 2.0,
            },
            seed_roi_index: 0,
        }];
        let field = generate_motion_transient_field(&graph, &config).unwrap();
        assert_eq!(field.localized_spatial_loadings[0][2], 0.0);
    }

    #[test]
    fn replay_is_deterministic_for_fixed_graph_and_events() {
        let graph = path_graph();
        let config = config();
        let first = generate_motion_transient_field(&graph, &config).unwrap();
        let second = generate_motion_transient_field(&graph, &config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_events_and_configurations_fail_closed() {
        let graph = path_graph();
        let mut invalid_onset = config();
        invalid_onset.broad_events[0].onset = 24;
        assert!(matches!(
            generate_motion_transient_field(&graph, &invalid_onset),
            Err(MotionTransientError::InvalidEvent { .. })
        ));

        let mut invalid_scale = config();
        invalid_scale.spatial_decay_length = 0.0;
        assert!(matches!(
            generate_motion_transient_field(&graph, &invalid_scale),
            Err(MotionTransientError::InvalidConfig(_))
        ));

        let mut invalid_seed = config();
        invalid_seed.localized_events[0].seed_roi_index = 6;
        assert!(matches!(
            generate_motion_transient_field(&graph, &invalid_seed),
            Err(MotionTransientError::InvalidSeedRoi { .. })
        ));
    }

    #[test]
    fn joint_roi_permutation_permutes_the_spatial_field() {
        let graph = path_graph();
        let config = config();
        let original = generate_motion_transient_field(&graph, &config).unwrap();
        let permutation = [3usize, 0, 5, 1, 4, 2];
        let labels = permutation
            .iter()
            .map(|&i| graph.roi_labels[i].clone())
            .collect();
        let adjacency = permutation
            .iter()
            .map(|&i| permutation.iter().map(|&j| graph.adjacency[i][j]).collect())
            .collect();
        let permuted_graph = SpatialGraph::new(labels, adjacency).unwrap();
        let permuted_config = MotionTransientConfig {
            localized_events: vec![LocalizedMotionEvent {
                event: config.localized_events[0].event,
                seed_roi_index: permutation.iter().position(|&i| i == 2).unwrap(),
            }],
            ..config
        };
        let permuted = generate_motion_transient_field(&permuted_graph, &permuted_config).unwrap();
        for time in 0..24 {
            for roi in 0..6 {
                assert!(
                    (original.values[time][permutation[roi]] - permuted.values[time][roi]).abs()
                        < 1e-12
                );
            }
        }
    }
}
