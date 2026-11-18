//! Independent circle-layout replay for CROP-INTEL-001.
//!
//! The input records are already digitized circles. This module does not detect
//! circles in photographs, calibrate image pixels to field metres, infer an
//! author, or establish a construction mechanism.

use crate::geometry::Point2;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleRecord {
    pub id: u64,
    pub center: Point2,
    pub radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleContact {
    pub first_index: usize,
    pub second_index: usize,
    pub first_id: u64,
    pub second_id: u64,
    /// Centre distance minus the sum of the two radii, in the input coordinate unit.
    pub tangent_gap: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircleLayout {
    pub circle_count: usize,
    pub centroid: Point2,
    /// One radial distance per input circle, preserving input order.
    pub radial_distances: Vec<f64>,
    pub contacts: Vec<CircleContact>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircleClassing {
    /// One-based radius class labels in ascending class-centre order.
    pub radius_labels: Vec<usize>,
    pub radius_centres: Vec<f64>,
    /// One-based radial-band labels in ascending radial-centre order.
    pub radial_labels: Vec<usize>,
    pub radial_centres: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircleLayoutError {
    EmptyInput,
    InvalidCsvHeader,
    InvalidCsvRow { line: usize, field: &'static str },
    DuplicateId { id: u64 },
    NonFiniteCoordinate { id: u64 },
    InvalidRadius { id: u64 },
    InvalidTolerance,
    InvalidClassCount { requested: usize, available: usize },
    ClassingLengthMismatch { circles: usize, labels: usize },
    InvalidContactIndex,
}

impl std::fmt::Display for CircleLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "at least one circle record is required"),
            Self::InvalidCsvHeader => write!(f, "expected CSV header: id,x,y,radius"),
            Self::InvalidCsvRow { line, field } => {
                write!(f, "invalid CSV {field} on line {line}")
            }
            Self::DuplicateId { id } => write!(f, "duplicate circle id {id}"),
            Self::NonFiniteCoordinate { id } => {
                write!(f, "circle {id} has a non-finite centre coordinate")
            }
            Self::InvalidRadius { id } => {
                write!(f, "circle {id} must have a finite positive radius")
            }
            Self::InvalidTolerance => {
                write!(f, "contact tolerance must be finite and non-negative")
            }
            Self::InvalidClassCount {
                requested,
                available,
            } => write!(
                f,
                "class count must be in 1..={available}; requested {requested}"
            ),
            Self::ClassingLengthMismatch { circles, labels } => write!(
                f,
                "received {labels} class labels for {circles} circles"
            ),
            Self::InvalidContactIndex => {
                write!(f, "contact edge references an invalid or identical circle index")
            }
        }
    }
}

impl std::error::Error for CircleLayoutError {}

/// Parse the strict, four-column CCAT-style circle CSV.
///
/// This is intentionally a narrow CSV reader: no quoted fields, embedded
/// commas, or implicit unit conversion. It preserves source order and IDs.
pub fn parse_circle_csv(input: &str) -> Result<Vec<CircleRecord>, CircleLayoutError> {
    let mut lines = input.lines();
    let header = lines.next().unwrap_or_default().trim_start_matches('\u{feff}').trim();
    if header != "id,x,y,radius" {
        return Err(CircleLayoutError::InvalidCsvHeader);
    }

    let mut circles = Vec::new();
    for (line_index, raw_line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split(',').map(str::trim).collect();
        if columns.len() != 4 {
            return Err(CircleLayoutError::InvalidCsvRow {
                line: line_number,
                field: "column count",
            });
        }

        let id = columns[0]
            .parse::<u64>()
            .map_err(|_| CircleLayoutError::InvalidCsvRow {
                line: line_number,
                field: "id",
            })?;
        let x = columns[1]
            .parse::<f64>()
            .map_err(|_| CircleLayoutError::InvalidCsvRow {
                line: line_number,
                field: "x",
            })?;
        let y = columns[2]
            .parse::<f64>()
            .map_err(|_| CircleLayoutError::InvalidCsvRow {
                line: line_number,
                field: "y",
            })?;
        let radius = columns[3]
            .parse::<f64>()
            .map_err(|_| CircleLayoutError::InvalidCsvRow {
                line: line_number,
                field: "radius",
            })?;

        circles.push(CircleRecord {
            id,
            center: Point2::new(x, y),
            radius,
        });
    }

    if circles.is_empty() {
        return Err(CircleLayoutError::EmptyInput);
    }
    validate_circles(&circles)?;
    Ok(circles)
}

/// Validate circle identifiers, coordinates and radii.
pub fn validate_circles(circles: &[CircleRecord]) -> Result<(), CircleLayoutError> {
    if circles.is_empty() {
        return Err(CircleLayoutError::EmptyInput);
    }

    let mut identifiers = HashSet::with_capacity(circles.len());
    for circle in circles {
        if !identifiers.insert(circle.id) {
            return Err(CircleLayoutError::DuplicateId { id: circle.id });
        }
        if !circle.center.x.is_finite() || !circle.center.y.is_finite() {
            return Err(CircleLayoutError::NonFiniteCoordinate { id: circle.id });
        }
        if !circle.radius.is_finite() || circle.radius <= 0.0 {
            return Err(CircleLayoutError::InvalidRadius { id: circle.id });
        }
    }
    Ok(())
}

/// Compute the arithmetic centre, radial distances and a proximity graph.
///
/// A contact edge is included whenever the absolute tangent gap is within the
/// caller-declared tolerance. This is a geometric proximity relation, not
/// proof that two crop features touched during construction.
pub fn describe_circle_layout(
    circles: &[CircleRecord],
    absolute_tangent_gap_tolerance: f64,
) -> Result<CircleLayout, CircleLayoutError> {
    validate_circles(circles)?;
    if !absolute_tangent_gap_tolerance.is_finite()
        || absolute_tangent_gap_tolerance < 0.0
    {
        return Err(CircleLayoutError::InvalidTolerance);
    }

    let count = circles.len() as f64;
    let centroid = Point2::new(
        circles
            .iter()
            .map(|circle| circle.center.x / count)
            .sum::<f64>(),
        circles
            .iter()
            .map(|circle| circle.center.y / count)
            .sum::<f64>(),
    );
    let radial_distances: Vec<f64> = circles
        .iter()
        .map(|circle| {
            let dx = circle.center.x - centroid.x;
            let dy = circle.center.y - centroid.y;
            dx.hypot(dy)
        })
        .collect();

    let mut contacts = Vec::new();
    for first_index in 0..circles.len() {
        for second_index in (first_index + 1)..circles.len() {
            let first = circles[first_index];
            let second = circles[second_index];
            let centre_distance = (first.center.x - second.center.x)
                .hypot(first.center.y - second.center.y);
            let tangent_gap = centre_distance - (first.radius + second.radius);
            if tangent_gap.abs() <= absolute_tangent_gap_tolerance {
                contacts.push(CircleContact {
                    first_index,
                    second_index,
                    first_id: first.id,
                    second_id: second.id,
                    tangent_gap,
                });
            }
        }
    }

    Ok(CircleLayout {
        circle_count: circles.len(),
        centroid,
        radial_distances,
        contacts,
    })
}

/// Reproduce the transparent 1D quantile-initialized k-means used as a
/// descriptive radius/radial-band baseline in the CCAT scripts.
///
/// Labels are one-based and ordered by the final cluster centres. The result
/// is sensitive to the requested class count and is not a claim that the
/// resulting classes are natural, intentional, or semantically meaningful.
pub fn class_circle_layout(
    circles: &[CircleRecord],
    layout: &CircleLayout,
    class_count: usize,
) -> Result<CircleClassing, CircleLayoutError> {
    validate_circles(circles)?;
    if circles.len() != layout.radial_distances.len() {
        return Err(CircleLayoutError::ClassingLengthMismatch {
            circles: circles.len(),
            labels: layout.radial_distances.len(),
        });
    }
    if class_count == 0 || class_count > circles.len() {
        return Err(CircleLayoutError::InvalidClassCount {
            requested: class_count,
            available: circles.len(),
        });
    }

    let radius_values: Vec<f64> = circles.iter().map(|circle| circle.radius).collect();
    let (radius_labels, radius_centres) =
        quantile_kmeans_1d(&radius_values, class_count);
    let (radial_labels, radial_centres) =
        quantile_kmeans_1d(&layout.radial_distances, class_count);

    Ok(CircleClassing {
        radius_labels,
        radius_centres,
        radial_labels,
        radial_centres,
    })
}

fn quantile_kmeans_1d(values: &[f64], class_count: usize) -> (Vec<usize>, Vec<f64>) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let quantile = |probability: f64| {
        let position = probability * (sorted.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    };

    let mut centres: Vec<f64> = (1..=class_count)
        .map(|index| quantile(index as f64 / (class_count + 1) as f64))
        .collect();
    let mut labels = vec![0; values.len()];

    for _ in 0..100 {
        for (value_index, value) in values.iter().copied().enumerate() {
            labels[value_index] = centres
                .iter()
                .enumerate()
                .min_by(|(_, first), (_, second)| {
                    (value - **first)
                        .abs()
                        .total_cmp(&(value - **second).abs())
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
        }

        let mut sums = vec![0.0; class_count];
        let mut counts = vec![0_usize; class_count];
        for (value, label) in values.iter().copied().zip(labels.iter().copied()) {
            sums[label] += value;
            counts[label] += 1;
        }

        let mut updated = centres.clone();
        for class_index in 0..class_count {
            if counts[class_index] > 0 {
                updated[class_index] = sums[class_index] / counts[class_index] as f64;
            }
        }
        let converged = updated
            .iter()
            .zip(centres.iter())
            .all(|(new, old)| (new - old).abs() <= 1.0e-10);
        centres = updated;
        if converged {
            break;
        }
    }

    let mut order: Vec<usize> = (0..class_count).collect();
    order.sort_by(|first, second| centres[*first].total_cmp(&centres[*second]));
    let mut remap = vec![0; class_count];
    let ordered_centres: Vec<f64> = order
        .iter()
        .enumerate()
        .map(|(new_label, old_label)| {
            remap[*old_label] = new_label + 1;
            centres[*old_label]
        })
        .collect();
    let ordered_labels = labels
        .iter()
        .map(|label| remap[*label])
        .collect();

    (ordered_labels, ordered_centres)
}

/// Count unordered pairs of neighbouring radius classes around each centre.
///
/// The output key is (smaller neighbour class, centre class, larger neighbour
/// class). This captures A-E-B as (A,E,B) and A-E-A as (A,E,A).
pub fn count_centred_motifs(
    class_labels: &[usize],
    contacts: &[CircleContact],
) -> Result<BTreeMap<(usize, usize, usize), usize>, CircleLayoutError> {
    if class_labels.is_empty() {
        return Err(CircleLayoutError::EmptyInput);
    }

    let mut adjacency = vec![Vec::new(); class_labels.len()];
    for contact in contacts {
        if contact.first_index >= class_labels.len()
            || contact.second_index >= class_labels.len()
            || contact.first_index == contact.second_index
        {
            return Err(CircleLayoutError::InvalidContactIndex);
        }
        adjacency[contact.first_index].push(contact.second_index);
        adjacency[contact.second_index].push(contact.first_index);
    }

    let mut motifs = BTreeMap::new();
    for (centre_index, neighbours) in adjacency.iter().enumerate() {
        for first_neighbour in 0..neighbours.len() {
            for second_neighbour in (first_neighbour + 1)..neighbours.len() {
                let first_class = class_labels[neighbours[first_neighbour]];
                let second_class = class_labels[neighbours[second_neighbour]];
                let key = (
                    first_class.min(second_class),
                    class_labels[centre_index],
                    first_class.max(second_class),
                );
                *motifs.entry(key).or_insert(0) += 1;
            }
        }
    }

    Ok(motifs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CCAT_ZEALS_KNOLL: &str =
        include_str!("../../datasets/ccat-zeals-knoll-2026/circles_drag_60.csv");

    #[test]
    fn parses_attributed_ccat_fixture_with_all_60_records() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        assert_eq!(circles.len(), 60);
        assert_eq!(circles[0].id, 1);
        assert_eq!(circles[59].id, 60);
        assert!((circles[0].radius - 86.09605).abs() < 1.0e-10);
    }

    #[test]
    fn reproduces_ccat_45_pixel_contact_and_ae_counts() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        let layout = describe_circle_layout(&circles, 45.0).expect("valid layout");
        let classes =
            class_circle_layout(&circles, &layout, 5).expect("five descriptive classes");

        assert_eq!(layout.circle_count, 60);
        assert_eq!(layout.contacts.len(), 66);
        let ae_edges = layout
            .contacts
            .iter()
            .filter(|edge| {
                let first = classes.radius_labels[edge.first_index];
                let second = classes.radius_labels[edge.second_index];
                (first == 1 && second == 5) || (first == 5 && second == 1)
            })
            .count();
        assert_eq!(ae_edges, 12);
    }

    #[test]
    fn reproduces_reported_centred_motif_counts() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        let layout = describe_circle_layout(&circles, 45.0).expect("valid layout");
        let classes =
            class_circle_layout(&circles, &layout, 5).expect("five descriptive classes");
        let motifs = count_centred_motifs(&classes.radius_labels, &layout.contacts)
            .expect("valid adjacency graph");

        assert_eq!(motifs.get(&(1, 5, 2)), Some(&10)); // A-E-B
        assert_eq!(motifs.get(&(2, 1, 5)), Some(&10)); // B-A-E
        assert_eq!(motifs.get(&(1, 5, 1)), Some(&8)); // A-E-A
    }

    #[test]
    fn rejects_bad_csv_headers_and_non_positive_radii() {
        assert_eq!(
            parse_circle_csv("x,y,radius\n1,2,3"),
            Err(CircleLayoutError::InvalidCsvHeader)
        );
        assert_eq!(
            parse_circle_csv("id,x,y,radius\n1,2,0"),
            Err(CircleLayoutError::InvalidRadius { id: 1 })
        );
    }

    #[test]
    fn rejects_duplicate_ids_and_non_finite_coordinates() {
        assert_eq!(
            parse_circle_csv("id,x,y,radius\n1,0,0,1\n1,1,1,1"),
            Err(CircleLayoutError::DuplicateId { id: 1 })
        );
        assert_eq!(
            parse_circle_csv("id,x,y,radius\n1,NaN,0,1"),
            Err(CircleLayoutError::NonFiniteCoordinate { id: 1 })
        );
    }

    #[test]
    fn contact_tolerance_must_be_finite_and_non_negative() {
        let circles = [
            CircleRecord {
                id: 1,
                center: Point2::new(0.0, 0.0),
                radius: 1.0,
            },
            CircleRecord {
                id: 2,
                center: Point2::new(2.0, 0.0),
                radius: 1.0,
            },
        ];
        assert_eq!(
            describe_circle_layout(&circles, -0.1),
            Err(CircleLayoutError::InvalidTolerance)
        );
        assert_eq!(
            describe_circle_layout(&circles, f64::NAN),
            Err(CircleLayoutError::InvalidTolerance)
        );
    }

    #[test]
    fn descriptive_classing_is_invariant_to_translation_and_uniform_scale() {
        let circles = [
            CircleRecord {
                id: 1,
                center: Point2::new(0.0, 0.0),
                radius: 1.0,
            },
            CircleRecord {
                id: 2,
                center: Point2::new(4.0, 0.0),
                radius: 2.0,
            },
            CircleRecord {
                id: 3,
                center: Point2::new(0.0, 3.0),
                radius: 3.0,
            },
            CircleRecord {
                id: 4,
                center: Point2::new(4.0, 3.0),
                radius: 4.0,
            },
            CircleRecord {
                id: 5,
                center: Point2::new(2.0, 1.5),
                radius: 5.0,
            },
        ];
        let original_layout = describe_circle_layout(&circles, 0.5).expect("valid layout");
        let original = class_circle_layout(&circles, &original_layout, 3).expect("classing");

        let transformed: Vec<CircleRecord> = circles
            .iter()
            .map(|circle| CircleRecord {
                id: circle.id,
                center: Point2::new(
                    circle.center.x * 7.0 + 91.0,
                    circle.center.y * 7.0 - 33.0,
                ),
                radius: circle.radius * 7.0,
            })
            .collect();
        let transformed_layout =
            describe_circle_layout(&transformed, 3.5).expect("scaled layout");
        let transformed_classing =
            class_circle_layout(&transformed, &transformed_layout, 3).expect("scaled classing");

        assert_eq!(original.radius_labels, transformed_classing.radius_labels);
        assert_eq!(original.radial_labels, transformed_classing.radial_labels);
    }

    #[test]
    fn rejects_invalid_class_count_and_motif_contact_indices() {
        let circles = [
            CircleRecord {
                id: 1,
                center: Point2::new(0.0, 0.0),
                radius: 1.0,
            },
            CircleRecord {
                id: 2,
                center: Point2::new(2.0, 0.0),
                radius: 1.0,
            },
        ];
        let layout = describe_circle_layout(&circles, 0.0).expect("valid layout");
        assert_eq!(
            class_circle_layout(&circles, &layout, 3),
            Err(CircleLayoutError::InvalidClassCount {
                requested: 3,
                available: 2
            })
        );
        let bad_edge = CircleContact {
            first_index: 0,
            second_index: 2,
            first_id: 1,
            second_id: 3,
            tangent_gap: 0.0,
        };
        assert_eq!(
            count_centred_motifs(&[1, 2], &[bad_edge]),
            Err(CircleLayoutError::InvalidContactIndex)
        );
    }
}
