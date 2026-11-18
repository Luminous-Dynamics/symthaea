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
    InvalidSensitivityConfig,
    RadiusPerturbationExceedsMinimumRadius,
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
            Self::InvalidSensitivityConfig => {
                write!(f, "sensitivity configuration must have positive replicates and finite non-negative bounds")
            }
            Self::RadiusPerturbationExceedsMinimumRadius => {
                write!(f, "maximum radius perturbation must be smaller than the minimum source radius")
            }
        }
    }
}

impl std::error::Error for CircleLayoutError {}

/// Configuration for an explicitly hypothetical image-space measurement stress test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleSensitivityConfig {
    pub replicates: usize,
    pub seed: u64,
    pub max_center_offset_px: f64,
    pub max_radius_offset_px: f64,
    pub class_count: usize,
    pub tangent_gap_tolerance_px: f64,
}

/// Stability summary from rerunning the complete descriptive circle pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct CircleSensitivitySummary {
    pub seed: u64,
    pub replicates: usize,
    pub baseline_contact_edges: usize,
    pub baseline_first_last_class_edges: usize,
    pub baseline_aeb_motifs: usize,
    pub baseline_bae_motifs: usize,
    pub baseline_aea_motifs: usize,
    pub mean_contact_edges: f64,
    pub contact_edges_sample_std_dev: f64,
    pub mean_first_last_class_edges: f64,
    pub first_last_class_edges_sample_std_dev: f64,
    pub mean_aeb_motifs: f64,
    pub aeb_motifs_sample_std_dev: f64,
    pub mean_bae_motifs: f64,
    pub bae_motifs_sample_std_dev: f64,
    pub mean_aea_motifs: f64,
    pub aea_motifs_sample_std_dev: f64,
    pub mean_radius_label_agreement: f64,
    /// Fraction of replicates that preserve all five baseline counts exactly.
    pub exact_baseline_signature_fraction: f64,
}

#[derive(Debug, Clone, Copy)]
struct RunningStats {
    count: usize,
    mean: f64,
    sum_squared_deviations: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            sum_squared_deviations: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta_after = value - self.mean;
        self.sum_squared_deviations += delta * delta_after;
    }

    fn sample_std_dev(self) -> f64 {
        if self.count <= 1 {
            0.0
        } else {
            (self.sum_squared_deviations / (self.count - 1) as f64)
                .max(0.0)
                .sqrt()
        }
    }
}

struct SensitivityRng {
    state: u64,
}

impl SensitivityRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    /// Uniform value in [0, 1), using the high 53 random bits.
    fn unit_interval(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Uniform value in [-1, 1).
    fn signed_unit(&mut self) -> f64 {
        self.unit_interval() * 2.0 - 1.0
    }
}

/// Rerun radius classes, radial classes, tangent-gap contacts and centred
/// motifs after independent uniform perturbations of circle centres and radii.
///
/// The caller supplies assumed pixel-error bounds. This function does not
/// estimate digitizer uncertainty. Outputs are sensitivity diagnostics only;
/// do not treat them as confidence intervals or p-values.
pub fn audit_circle_measurement_sensitivity(
    circles: &[CircleRecord],
    config: CircleSensitivityConfig,
) -> Result<CircleSensitivitySummary, CircleLayoutError> {
    validate_circles(circles)?;
    if config.replicates == 0
        || !config.max_center_offset_px.is_finite()
        || config.max_center_offset_px < 0.0
        || !config.max_radius_offset_px.is_finite()
        || config.max_radius_offset_px < 0.0
        || !config.tangent_gap_tolerance_px.is_finite()
        || config.tangent_gap_tolerance_px < 0.0
        || config.class_count == 0
        || config.class_count > circles.len()
    {
        return Err(CircleLayoutError::InvalidSensitivityConfig);
    }
    let minimum_radius = circles
        .iter()
        .map(|circle| circle.radius)
        .fold(f64::INFINITY, f64::min);
    if config.max_radius_offset_px >= minimum_radius {
        return Err(CircleLayoutError::RadiusPerturbationExceedsMinimumRadius);
    }

    let baseline_layout =
        describe_circle_layout(circles, config.tangent_gap_tolerance_px)?;
    let baseline_classes = class_circle_layout(circles, &baseline_layout, config.class_count)?;
    let baseline_motifs =
        count_centred_motifs(&baseline_classes.radius_labels, &baseline_layout.contacts)?;
    let baseline_first_last = count_first_last_class_edges(
        &baseline_classes.radius_labels,
        &baseline_layout.contacts,
        config.class_count,
    );
    let baseline_aeb = *baseline_motifs
        .get(&(1, config.class_count, 2))
        .unwrap_or(&0);
    let baseline_bae = *baseline_motifs
        .get(&(2, 1, config.class_count))
        .unwrap_or(&0);
    let baseline_aea = *baseline_motifs
        .get(&(1, config.class_count, 1))
        .unwrap_or(&0);

    let mut rng = SensitivityRng::new(config.seed);
    let mut contacts_stats = RunningStats::new();
    let mut first_last_stats = RunningStats::new();
    let mut aeb_stats = RunningStats::new();
    let mut bae_stats = RunningStats::new();
    let mut aea_stats = RunningStats::new();
    let mut label_agreement_stats = RunningStats::new();
    let mut exact_signatures = 0_usize;

    for _ in 0..config.replicates {
        let perturbed: Vec<CircleRecord> = circles
            .iter()
            .map(|circle| CircleRecord {
                id: circle.id,
                center: Point2::new(
                    circle.center.x
                        + rng.signed_unit() * config.max_center_offset_px,
                    circle.center.y
                        + rng.signed_unit() * config.max_center_offset_px,
                ),
                radius: circle.radius
                    + rng.signed_unit() * config.max_radius_offset_px,
            })
            .collect();

        let layout =
            describe_circle_layout(&perturbed, config.tangent_gap_tolerance_px)?;
        let classes = class_circle_layout(&perturbed, &layout, config.class_count)?;
        let motifs = count_centred_motifs(&classes.radius_labels, &layout.contacts)?;
        let first_last = count_first_last_class_edges(
            &classes.radius_labels,
            &layout.contacts,
            config.class_count,
        );
        let aeb = *motifs
            .get(&(1, config.class_count, 2))
            .unwrap_or(&0);
        let bae = *motifs
            .get(&(2, 1, config.class_count))
            .unwrap_or(&0);
        let aea = *motifs
            .get(&(1, config.class_count, 1))
            .unwrap_or(&0);

        contacts_stats.push(layout.contacts.len() as f64);
        first_last_stats.push(first_last as f64);
        aeb_stats.push(aeb as f64);
        bae_stats.push(bae as f64);
        aea_stats.push(aea as f64);

        let matching_labels = classes
            .radius_labels
            .iter()
            .zip(baseline_classes.radius_labels.iter())
            .filter(|(candidate, baseline)| candidate == baseline)
            .count();
        label_agreement_stats.push(matching_labels as f64 / circles.len() as f64);

        if layout.contacts.len() == baseline_layout.contacts.len()
            && first_last == baseline_first_last
            && aeb == baseline_aeb
            && bae == baseline_bae
            && aea == baseline_aea
        {
            exact_signatures += 1;
        }
    }

    Ok(CircleSensitivitySummary {
        seed: config.seed,
        replicates: config.replicates,
        baseline_contact_edges: baseline_layout.contacts.len(),
        baseline_first_last_class_edges: baseline_first_last,
        baseline_aeb_motifs: baseline_aeb,
        baseline_bae_motifs: baseline_bae,
        baseline_aea_motifs: baseline_aea,
        mean_contact_edges: contacts_stats.mean,
        contact_edges_sample_std_dev: contacts_stats.sample_std_dev(),
        mean_first_last_class_edges: first_last_stats.mean,
        first_last_class_edges_sample_std_dev: first_last_stats.sample_std_dev(),
        mean_aeb_motifs: aeb_stats.mean,
        aeb_motifs_sample_std_dev: aeb_stats.sample_std_dev(),
        mean_bae_motifs: bae_stats.mean,
        bae_motifs_sample_std_dev: bae_stats.sample_std_dev(),
        mean_aea_motifs: aea_stats.mean,
        aea_motifs_sample_std_dev: aea_stats.sample_std_dev(),
        mean_radius_label_agreement: label_agreement_stats.mean,
        exact_baseline_signature_fraction: exact_signatures as f64 / config.replicates as f64,
    })
}

fn count_first_last_class_edges(
    labels: &[usize],
    contacts: &[CircleContact],
    class_count: usize,
) -> usize {
    contacts
        .iter()
        .filter(|edge| {
            let first = labels[edge.first_index];
            let second = labels[edge.second_index];
            (first == 1 && second == class_count)
                || (first == class_count && second == 1)
        })
        .count()
}

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

    #[test]
    fn zero_perturbation_preserves_baseline_signature_exactly() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        let result = audit_circle_measurement_sensitivity(
            &circles,
            CircleSensitivityConfig {
                replicates: 5,
                seed: 123,
                max_center_offset_px: 0.0,
                max_radius_offset_px: 0.0,
                class_count: 5,
                tangent_gap_tolerance_px: 45.0,
            },
        )
        .expect("valid zero-noise stress test");

        assert_eq!(result.baseline_contact_edges, 66);
        assert_eq!(result.baseline_first_last_class_edges, 12);
        assert_eq!(result.baseline_aeb_motifs, 10);
        assert_eq!(result.baseline_bae_motifs, 10);
        assert_eq!(result.baseline_aea_motifs, 8);
        assert_eq!(result.exact_baseline_signature_fraction, 1.0);
        assert_eq!(result.mean_radius_label_agreement, 1.0);
    }

    #[test]
    fn sensitivity_audit_is_deterministic_for_a_frozen_seed() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        let config = CircleSensitivityConfig {
            replicates: 20,
            seed: 20261118,
            max_center_offset_px: 1.0,
            max_radius_offset_px: 1.0,
            class_count: 5,
            tangent_gap_tolerance_px: 45.0,
        };
        let first =
            audit_circle_measurement_sensitivity(&circles, config).expect("first replay");
        let second =
            audit_circle_measurement_sensitivity(&circles, config).expect("second replay");
        assert_eq!(first, second);
    }

    #[test]
    fn sensitivity_audit_rejects_invalid_bounds_and_replica_counts() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("valid CCAT fixture");
        let base = CircleSensitivityConfig {
            replicates: 10,
            seed: 7,
            max_center_offset_px: 1.0,
            max_radius_offset_px: 1.0,
            class_count: 5,
            tangent_gap_tolerance_px: 45.0,
        };
        assert_eq!(
            audit_circle_measurement_sensitivity(
                &circles,
                CircleSensitivityConfig { replicates: 0, ..base }
            ),
            Err(CircleLayoutError::InvalidSensitivityConfig)
        );
        assert_eq!(
            audit_circle_measurement_sensitivity(
                &circles,
                CircleSensitivityConfig {
                    max_radius_offset_px: 100.0,
                    ..base
                }
            ),
            Err(CircleLayoutError::RadiusPerturbationExceedsMinimumRadius)
        );
        assert_eq!(
            audit_circle_measurement_sensitivity(
                &circles,
                CircleSensitivityConfig {
                    max_center_offset_px: f64::NAN,
                    ..base
                }
            ),
            Err(CircleLayoutError::InvalidSensitivityConfig)
        );
    }
}
