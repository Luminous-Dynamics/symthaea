// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Qualified local charts and finite-atlas evidence over implicit manifolds.
//!
//! A local chart is a bounded numerical coordinate patch. A finite collection of
//! charts is partial exploration by default. Global coverage is never inferred
//! from chart count or connectivity; a complete-coverage claim requires an
//! explicit separately identified witness bound to the exact atlas structure.

use blake3::Hasher;
use thiserror::Error;

use crate::implicit_manifold::{
    ConstraintValidity, EqualityConstraint, ImplicitManifold, ImplicitManifoldError,
    ProjectionOutcome, TangentBasis,
};
use crate::state_space::MetricSpace;

/// Numerical/domain policy for one local tangent chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalChartPolicy {
    radius: f64,
    inverse_roundtrip_tolerance: f64,
}

impl LocalChartPolicy {
    /// Construct a chart policy with explicit tangent-domain radius and inverse tolerance.
    pub fn new(
        radius: f64,
        inverse_roundtrip_tolerance: f64,
    ) -> Result<Self, AtlasError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(AtlasError::InvalidPolicy {
                reason: format!("chart radius must be finite and > 0, got {radius}"),
            });
        }
        if !inverse_roundtrip_tolerance.is_finite() || inverse_roundtrip_tolerance <= 0.0 {
            return Err(AtlasError::InvalidPolicy {
                reason: format!(
                    "inverse roundtrip tolerance must be finite and > 0, got {inverse_roundtrip_tolerance}"
                ),
            });
        }
        Ok(Self {
            radius,
            inverse_roundtrip_tolerance,
        })
    }

    /// Conservative reference policy for analytic fixtures.
    pub fn reference_v1() -> Self {
        Self::new(0.1, 1e-3).expect("reference local-chart policy is statically valid")
    }

    /// Maximum Euclidean norm of local tangent coordinates accepted by the chart.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Maximum ambient round-trip error for a qualified inverse/localization result.
    pub fn inverse_roundtrip_tolerance(&self) -> f64 {
        self.inverse_roundtrip_tolerance
    }

    fn update_hash(&self, hasher: &mut Hasher) {
        hasher.update(&self.radius.to_bits().to_le_bytes());
        hasher.update(&self.inverse_roundtrip_tolerance.to_bits().to_le_bytes());
    }
}

/// Typed chart/atlas failure that does not imply global manifold failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum AtlasError {
    /// Chart/atlas numerical policy is invalid.
    #[error("invalid atlas policy: {reason}")]
    InvalidPolicy { reason: String },
    /// A chart was used with a different exact manifold profile.
    #[error("chart manifold identity does not match the supplied manifold")]
    ManifoldMismatch,
    /// A chart center or candidate point is not locally qualified.
    #[error("point is not qualified for local chart use: {reason}")]
    PointNotQualified { reason: String },
    /// Local coordinate dimension differs from the chart tangent dimension.
    #[error("local-coordinate dimension mismatch: expected {expected}, got {actual}")]
    LocalDimensionMismatch { expected: usize, actual: usize },
    /// Local coordinate is outside the chart's declared tangent-domain ball.
    #[error("local coordinate norm {norm} exceeds chart radius {radius}")]
    OutsideLocalDomain { norm: f64, radius: f64 },
    /// Local projection failed under an explicit projection outcome.
    #[error("chart retraction rejected by projection: {outcome:?}")]
    ProjectionRejected { outcome: ProjectionOutcome },
    /// Candidate inverse/localization did not round-trip under the declared tolerance.
    #[error("chart inverse round-trip error {error} exceeds tolerance {tolerance}")]
    InverseNotQualified { error: f64, tolerance: f64 },
    /// Non-finite local coordinate or derived value.
    #[error("non-finite value during {stage}")]
    NonFinite { stage: &'static str },
    /// An overlap witness references a chart not present in the atlas.
    #[error("overlap witness references a chart outside this atlas")]
    UnknownOverlapChart,
    /// Duplicate chart identity was supplied.
    #[error("duplicate chart identity")]
    DuplicateChart,
    /// Coverage witness is bound to another atlas structure.
    #[error("coverage witness is not bound to this exact atlas structure")]
    CoverageWitnessMismatch,
    /// Underlying implicit-manifold operation failed.
    #[error(transparent)]
    Implicit(#[from] ImplicitManifoldError),
}

/// Qualified state returned by a local chart retraction.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartPoint {
    /// Ambient manifold state.
    pub state: Vec<f64>,
    /// Exact projection outcome supporting the local retraction.
    pub projection: ProjectionOutcome,
}

/// One bounded tangent chart around an exact regular center point.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalChart {
    manifold_identity: [u8; 32],
    center: Vec<f64>,
    tangent_basis: Vec<Vec<f64>>,
    policy: LocalChartPolicy,
    identity: [u8; 32],
}

impl LocalChart {
    /// Build a local tangent chart at a constraint-satisfied regular point.
    pub fn new<C>(
        manifold: &ImplicitManifold<C>,
        center: &[f64],
        policy: LocalChartPolicy,
    ) -> Result<Self, AtlasError>
    where
        C: EqualityConstraint,
    {
        match manifold.validate_state(center) {
            ConstraintValidity::Valid { .. } => {}
            other => {
                return Err(AtlasError::PointNotQualified {
                    reason: format!("center constraint validity: {other:?}"),
                });
            }
        }
        let TangentBasis { vectors, .. } = manifold.tangent_basis(center)?;
        let manifold_identity = manifold.profile().identity();

        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-local-chart-v1\0");
        hasher.update(&manifold_identity);
        hash_f64_slice(&mut hasher, center);
        hasher.update(&(vectors.len() as u64).to_le_bytes());
        for vector in &vectors {
            hash_f64_slice(&mut hasher, vector);
        }
        policy.update_hash(&mut hasher);
        let identity = *hasher.finalize().as_bytes();

        Ok(Self {
            manifold_identity,
            center: center.to_vec(),
            tangent_basis: vectors,
            policy,
            identity,
        })
    }

    /// Exact chart identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Exact manifold identity to which this chart is bound.
    pub fn manifold_identity(&self) -> [u8; 32] {
        self.manifold_identity
    }

    /// Chart center in ambient coordinates.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Orthonormal tangent basis at the center.
    pub fn tangent_basis(&self) -> &[Vec<f64>] {
        &self.tangent_basis
    }

    /// Local chart policy.
    pub fn policy(&self) -> LocalChartPolicy {
        self.policy
    }

    /// Map bounded tangent coordinates back to the manifold by local projection.
    pub fn retract<C>(
        &self,
        manifold: &ImplicitManifold<C>,
        local: &[f64],
    ) -> Result<ChartPoint, AtlasError>
    where
        C: EqualityConstraint,
    {
        self.require_manifold(manifold)?;
        if local.len() != self.tangent_basis.len() {
            return Err(AtlasError::LocalDimensionMismatch {
                expected: self.tangent_basis.len(),
                actual: local.len(),
            });
        }
        if local.iter().any(|value| !value.is_finite()) {
            return Err(AtlasError::NonFinite {
                stage: "local coordinates",
            });
        }
        let norm = stable_norm(local);
        if norm > self.policy.radius {
            return Err(AtlasError::OutsideLocalDomain {
                norm,
                radius: self.policy.radius,
            });
        }

        let mut candidate = self.center.clone();
        for (coordinate, basis) in local.iter().zip(&self.tangent_basis) {
            for (value, direction) in candidate.iter_mut().zip(basis) {
                *value += coordinate * direction;
            }
        }
        if candidate.iter().any(|value| !value.is_finite()) {
            return Err(AtlasError::NonFinite {
                stage: "chart ambient candidate",
            });
        }

        let projection = manifold.project(&candidate)?;
        if matches!(
            &projection.outcome,
            ProjectionOutcome::AlreadySatisfied { .. } | ProjectionOutcome::Projected { .. }
        ) {
            Ok(ChartPoint {
                state: projection.state,
                projection: projection.outcome,
            })
        } else {
            Err(AtlasError::ProjectionRejected {
                outcome: projection.outcome,
            })
        }
    }

    /// Localize a qualified manifold point into this tangent chart.
    ///
    /// The inverse is accepted only when retraction of the inferred coordinates
    /// returns to the supplied point within the chart's explicit ambient tolerance.
    pub fn localize<C>(
        &self,
        manifold: &ImplicitManifold<C>,
        point: &[f64],
    ) -> Result<Vec<f64>, AtlasError>
    where
        C: EqualityConstraint,
    {
        self.require_manifold(manifold)?;
        match manifold.validate_state(point) {
            ConstraintValidity::Valid { .. } => {}
            other => {
                return Err(AtlasError::PointNotQualified {
                    reason: format!("point constraint validity: {other:?}"),
                });
            }
        }
        // Constraint satisfaction is not enough: the target must be regular too.
        manifold.tangent_basis(point)?;

        let difference: Vec<f64> = point
            .iter()
            .zip(&self.center)
            .map(|(point_value, center_value)| point_value - center_value)
            .collect();
        if difference.iter().any(|value| !value.is_finite()) {
            return Err(AtlasError::NonFinite {
                stage: "chart localization difference",
            });
        }
        let local: Vec<f64> = self
            .tangent_basis
            .iter()
            .map(|basis| dot(&difference, basis))
            .collect();
        let norm = stable_norm(&local);
        if norm > self.policy.radius {
            return Err(AtlasError::OutsideLocalDomain {
                norm,
                radius: self.policy.radius,
            });
        }

        let roundtrip = self.retract(manifold, &local)?;
        let point_owned = point.to_vec();
        let error = manifold
            .ambient()
            .distance(&point_owned, &roundtrip.state)
            .map_err(|state_error| AtlasError::PointNotQualified {
                reason: state_error.to_string(),
            })?;
        if error > self.policy.inverse_roundtrip_tolerance {
            return Err(AtlasError::InverseNotQualified {
                error,
                tolerance: self.policy.inverse_roundtrip_tolerance,
            });
        }
        Ok(local)
    }

    /// Produce an explicit witness that one manifold point lies in both charts.
    pub fn overlap_witness<C>(
        &self,
        other: &LocalChart,
        manifold: &ImplicitManifold<C>,
        point: &[f64],
    ) -> Result<ChartOverlapWitness, AtlasError>
    where
        C: EqualityConstraint,
    {
        self.require_manifold(manifold)?;
        other.require_manifold(manifold)?;
        let first_local = self.localize(manifold, point)?;
        let second_local = other.localize(manifold, point)?;
        ChartOverlapWitness::new(
            self.identity,
            other.identity,
            point.to_vec(),
            first_local,
            second_local,
        )
    }

    fn require_manifold<C>(&self, manifold: &ImplicitManifold<C>) -> Result<(), AtlasError>
    where
        C: EqualityConstraint,
    {
        if self.manifold_identity == manifold.profile().identity() {
            Ok(())
        } else {
            Err(AtlasError::ManifoldMismatch)
        }
    }
}

/// Explicit shared-point evidence for adjacency/overlap between two charts.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartOverlapWitness {
    first_chart: [u8; 32],
    second_chart: [u8; 32],
    point: Vec<f64>,
    first_local: Vec<f64>,
    second_local: Vec<f64>,
    identity: [u8; 32],
}

impl ChartOverlapWitness {
    fn new(
        first_chart: [u8; 32],
        second_chart: [u8; 32],
        point: Vec<f64>,
        first_local: Vec<f64>,
        second_local: Vec<f64>,
    ) -> Result<Self, AtlasError> {
        if point
            .iter()
            .chain(&first_local)
            .chain(&second_local)
            .any(|value| !value.is_finite())
        {
            return Err(AtlasError::NonFinite {
                stage: "chart overlap witness",
            });
        }
        let (ordered_first, ordered_second, ordered_first_local, ordered_second_local) =
            if first_chart <= second_chart {
                (first_chart, second_chart, first_local, second_local)
            } else {
                (second_chart, first_chart, second_local, first_local)
            };

        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-chart-overlap-witness-v1\0");
        hasher.update(&ordered_first);
        hasher.update(&ordered_second);
        hash_f64_slice(&mut hasher, &point);
        hash_f64_slice(&mut hasher, &ordered_first_local);
        hash_f64_slice(&mut hasher, &ordered_second_local);
        let identity = *hasher.finalize().as_bytes();

        Ok(Self {
            first_chart: ordered_first,
            second_chart: ordered_second,
            point,
            first_local: ordered_first_local,
            second_local: ordered_second_local,
            identity,
        })
    }

    /// First chart identity in deterministic lexical order.
    pub fn first_chart(&self) -> [u8; 32] {
        self.first_chart
    }

    /// Second chart identity in deterministic lexical order.
    pub fn second_chart(&self) -> [u8; 32] {
        self.second_chart
    }

    /// Shared manifold point used to qualify the overlap.
    pub fn point(&self) -> &[f64] {
        &self.point
    }

    /// Local coordinates in the first chart.
    pub fn first_local(&self) -> &[f64] {
        &self.first_local
    }

    /// Local coordinates in the second chart.
    pub fn second_local(&self) -> &[f64] {
        &self.second_local
    }

    /// Exact overlap-witness identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// External evidence binding for a complete-coverage claim.
///
/// This type records identity/provenance only. Constructing one does not make the
/// completeness theorem true; an evidence layer must independently qualify the
/// referenced certificate/method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasCoverageWitness {
    atlas_structure_identity: [u8; 32],
    method: String,
    evidence_digest: [u8; 32],
    identity: [u8; 32],
}

impl AtlasCoverageWitness {
    /// Bind an external coverage certificate to one exact atlas structure.
    pub fn new(
        atlas_structure_identity: [u8; 32],
        method: impl Into<String>,
        evidence_digest: [u8; 32],
    ) -> Result<Self, AtlasError> {
        let method = method.into();
        if method.trim().is_empty() {
            return Err(AtlasError::InvalidPolicy {
                reason: "coverage witness method must not be empty".to_string(),
            });
        }
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-atlas-coverage-witness-v1\0");
        hasher.update(&atlas_structure_identity);
        update_string_hash(&mut hasher, &method);
        hasher.update(&evidence_digest);
        let identity = *hasher.finalize().as_bytes();
        Ok(Self {
            atlas_structure_identity,
            method,
            evidence_digest,
            identity,
        })
    }

    /// Exact atlas structure to which this evidence is bound.
    pub fn atlas_structure_identity(&self) -> [u8; 32] {
        self.atlas_structure_identity
    }

    /// External proof/certificate method identifier.
    pub fn method(&self) -> &str {
        &self.method
    }

    /// External evidence digest.
    pub fn evidence_digest(&self) -> [u8; 32] {
        self.evidence_digest
    }

    /// Exact coverage-witness identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Coverage state of a finite atlas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtlasCoverage {
    /// Default: finite charts/overlaps are only partial exploration evidence.
    PartialExploration,
    /// A complete-coverage claim is explicitly bound to external evidence.
    ///
    /// The atlas validates identity binding only; it does not prove the external
    /// coverage theorem internally.
    ExternallyWitnessedCompleteClaim { witness_identity: [u8; 32] },
}

/// Finite chart collection with explicit overlap evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteAtlas {
    manifold_identity: [u8; 32],
    charts: Vec<LocalChart>,
    overlaps: Vec<ChartOverlapWitness>,
    coverage: AtlasCoverage,
}

impl FiniteAtlas {
    /// Construct an empty partial atlas for one exact manifold profile.
    pub fn new(manifold_identity: [u8; 32]) -> Self {
        Self {
            manifold_identity,
            charts: Vec::new(),
            overlaps: Vec::new(),
            coverage: AtlasCoverage::PartialExploration,
        }
    }

    /// Exact manifold identity.
    pub fn manifold_identity(&self) -> [u8; 32] {
        self.manifold_identity
    }

    /// Ordered charts retained by this atlas.
    pub fn charts(&self) -> &[LocalChart] {
        &self.charts
    }

    /// Explicit overlap witnesses retained by this atlas.
    pub fn overlaps(&self) -> &[ChartOverlapWitness] {
        &self.overlaps
    }

    /// Current coverage claim state.
    pub fn coverage(&self) -> &AtlasCoverage {
        &self.coverage
    }

    /// Add a chart bound to the same exact manifold.
    pub fn add_chart(&mut self, chart: LocalChart) -> Result<(), AtlasError> {
        if chart.manifold_identity != self.manifold_identity {
            return Err(AtlasError::ManifoldMismatch);
        }
        if self
            .charts
            .iter()
            .any(|existing| existing.identity == chart.identity)
        {
            return Err(AtlasError::DuplicateChart);
        }
        self.charts.push(chart);
        // Structural mutation invalidates any prior complete-coverage evidence.
        self.coverage = AtlasCoverage::PartialExploration;
        Ok(())
    }

    /// Add explicit shared-point evidence for two charts already in the atlas.
    pub fn add_overlap(&mut self, overlap: ChartOverlapWitness) -> Result<(), AtlasError> {
        let has_first = self
            .charts
            .iter()
            .any(|chart| chart.identity == overlap.first_chart);
        let has_second = self
            .charts
            .iter()
            .any(|chart| chart.identity == overlap.second_chart);
        if !has_first || !has_second {
            return Err(AtlasError::UnknownOverlapChart);
        }
        if !self
            .overlaps
            .iter()
            .any(|existing| existing.identity == overlap.identity)
        {
            self.overlaps.push(overlap);
            self.coverage = AtlasCoverage::PartialExploration;
        }
        Ok(())
    }

    /// Deterministic identity of the finite chart/overlap structure.
    ///
    /// Chart and overlap IDs are sorted, so insertion order is not evidence.
    pub fn structure_identity(&self) -> [u8; 32] {
        let mut chart_ids: Vec<[u8; 32]> = self.charts.iter().map(LocalChart::identity).collect();
        chart_ids.sort();
        let mut overlap_ids: Vec<[u8; 32]> = self
            .overlaps
            .iter()
            .map(ChartOverlapWitness::identity)
            .collect();
        overlap_ids.sort();

        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-finite-atlas-structure-v1\0");
        hasher.update(&self.manifold_identity);
        hasher.update(&(chart_ids.len() as u64).to_le_bytes());
        for identity in chart_ids {
            hasher.update(&identity);
        }
        hasher.update(&(overlap_ids.len() as u64).to_le_bytes());
        for identity in overlap_ids {
            hasher.update(&identity);
        }
        *hasher.finalize().as_bytes()
    }

    /// Attach a complete-coverage claim only through separately identified evidence.
    pub fn attach_coverage_witness(
        &mut self,
        witness: &AtlasCoverageWitness,
    ) -> Result<(), AtlasError> {
        if witness.atlas_structure_identity != self.structure_identity() {
            return Err(AtlasError::CoverageWitnessMismatch);
        }
        self.coverage = AtlasCoverage::ExternallyWitnessedCompleteClaim {
            witness_identity: witness.identity,
        };
        Ok(())
    }
}

fn stable_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn hash_f64_slice(hasher: &mut Hasher, values: &[f64]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn update_string_hash(hasher: &mut Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::implicit_manifold::{ImplicitManifold, ProjectionPolicy, SphereConstraint};

    fn sphere() -> ImplicitManifold<SphereConstraint> {
        ImplicitManifold::new(
            SphereConstraint::new(3, 1.0).unwrap(),
            ProjectionPolicy::reference_v1(),
        )
        .unwrap()
    }

    fn point_on_equator(angle: f64) -> [f64; 3] {
        [angle.cos(), angle.sin(), 0.0]
    }

    #[test]
    fn local_chart_round_trip_is_qualified_inside_small_patch() {
        let manifold = sphere();
        let chart = LocalChart::new(
            &manifold,
            &[1.0, 0.0, 0.0],
            LocalChartPolicy::reference_v1(),
        )
        .unwrap();
        let point = chart.retract(&manifold, &[0.01, -0.015]).unwrap();
        let local = chart.localize(&manifold, &point.state).unwrap();
        assert_eq!(local.len(), 2);
        let roundtrip = chart.retract(&manifold, &local).unwrap();
        let error = manifold
            .ambient()
            .distance(&point.state, &roundtrip.state)
            .unwrap();
        assert!(error <= chart.policy().inverse_roundtrip_tolerance());
    }

    #[test]
    fn chart_rejects_coordinates_outside_declared_radius() {
        let manifold = sphere();
        let chart = LocalChart::new(
            &manifold,
            &[1.0, 0.0, 0.0],
            LocalChartPolicy::reference_v1(),
        )
        .unwrap();
        let error = chart.retract(&manifold, &[0.11, 0.0]).unwrap_err();
        assert!(matches!(error, AtlasError::OutsideLocalDomain { .. }));
    }

    #[test]
    fn overlap_requires_one_point_qualified_in_both_charts() {
        let manifold = sphere();
        let policy = LocalChartPolicy::reference_v1();
        let first = LocalChart::new(&manifold, &point_on_equator(0.0), policy).unwrap();
        let second = LocalChart::new(&manifold, &point_on_equator(0.08), policy).unwrap();
        let witness_point = point_on_equator(0.04);
        let witness = first
            .overlap_witness(&second, &manifold, &witness_point)
            .unwrap();
        assert_eq!(witness.point().len(), 3);
        assert_eq!(witness.first_local().len(), 2);
        assert_eq!(witness.second_local().len(), 2);
    }

    #[test]
    fn finite_atlas_is_partial_by_default_and_after_mutation() {
        let manifold = sphere();
        let policy = LocalChartPolicy::reference_v1();
        let first = LocalChart::new(&manifold, &point_on_equator(0.0), policy).unwrap();
        let second = LocalChart::new(&manifold, &point_on_equator(0.08), policy).unwrap();
        let overlap = first
            .overlap_witness(&second, &manifold, &point_on_equator(0.04))
            .unwrap();

        let mut atlas = FiniteAtlas::new(manifold.profile().identity());
        assert_eq!(atlas.coverage(), &AtlasCoverage::PartialExploration);
        atlas.add_chart(first).unwrap();
        atlas.add_chart(second).unwrap();
        atlas.add_overlap(overlap).unwrap();
        assert_eq!(atlas.coverage(), &AtlasCoverage::PartialExploration);
    }

    #[test]
    fn complete_coverage_claim_requires_exact_structure_witness() {
        let manifold = sphere();
        let policy = LocalChartPolicy::reference_v1();
        let first = LocalChart::new(&manifold, &point_on_equator(0.0), policy).unwrap();
        let second = LocalChart::new(&manifold, &point_on_equator(0.08), policy).unwrap();
        let overlap = first
            .overlap_witness(&second, &manifold, &point_on_equator(0.04))
            .unwrap();

        let mut atlas = FiniteAtlas::new(manifold.profile().identity());
        atlas.add_chart(first).unwrap();
        atlas.add_chart(second).unwrap();
        atlas.add_overlap(overlap).unwrap();

        let wrong = AtlasCoverageWitness::new(
            [7; 32],
            "analytic-sphere-cover-fixture-v1",
            [9; 32],
        )
        .unwrap();
        assert_eq!(
            atlas.attach_coverage_witness(&wrong),
            Err(AtlasError::CoverageWitnessMismatch)
        );

        let witness = AtlasCoverageWitness::new(
            atlas.structure_identity(),
            "analytic-sphere-cover-fixture-v1",
            [9; 32],
        )
        .unwrap();
        atlas.attach_coverage_witness(&witness).unwrap();
        assert_eq!(
            atlas.coverage(),
            &AtlasCoverage::ExternallyWitnessedCompleteClaim {
                witness_identity: witness.identity()
            }
        );
    }
}
