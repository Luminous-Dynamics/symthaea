//! Plural, distribution-aware consequence vectors for counterfactual scenarios.
//!
//! This crate intentionally does not provide an aggregate utility score or
//! stakeholder-independent ranking. It represents consequence dimensions,
//! uncertainty intervals, time horizons, distributional slices, and source
//! references so humans or explicitly governed processes can inspect tradeoffs
//! without hidden normative weights.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, OutcomeError>;

#[derive(Debug, Clone, PartialEq)]
pub enum OutcomeError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    InvalidInterval { lower: f64, point: f64, upper: f64 },
    InvalidTimeHorizon { start_unix_ms: i64, end_unix_ms: i64 },
    MissingSource,
    MissingDimensions,
    DuplicateDimension(String),
}

impl Display for OutcomeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::InvalidInterval { lower, point, upper } => write!(
                f,
                "outcome interval must satisfy lower <= point <= upper, got {lower} <= {point} <= {upper}"
            ),
            Self::InvalidTimeHorizon {
                start_unix_ms,
                end_unix_ms,
            } => write!(
                f,
                "outcome horizon end {end_unix_ms} precedes start {start_unix_ms}"
            ),
            Self::MissingSource => write!(f, "outcome dimensions require at least one source reference"),
            Self::MissingDimensions => write!(f, "scenario outcome vector requires at least one dimension"),
            Self::DuplicateDimension(id) => write!(f, "duplicate scenario outcome dimension {id}"),
        }
    }
}

impl Error for OutcomeError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(OutcomeError::EmptyField(field));
    }
    Ok(())
}

fn finite(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() {
        return Err(OutcomeError::NonFinite { field, value });
    }
    Ok(())
}

/// Numeric consequence estimate with an optional uncertainty interval.
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeEstimate {
    pub point: f64,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub unit: String,
}

impl OutcomeEstimate {
    pub fn point(point: f64, unit: impl Into<String>) -> Result<Self> {
        finite(point, "outcome point estimate")?;
        let unit = unit.into();
        non_empty(&unit, "outcome unit")?;
        Ok(Self {
            point,
            lower: None,
            upper: None,
            unit,
        })
    }

    pub fn interval(
        point: f64,
        lower: f64,
        upper: f64,
        unit: impl Into<String>,
    ) -> Result<Self> {
        finite(point, "outcome point estimate")?;
        finite(lower, "outcome lower bound")?;
        finite(upper, "outcome upper bound")?;
        if lower > point || point > upper {
            return Err(OutcomeError::InvalidInterval {
                lower,
                point,
                upper,
            });
        }
        let unit = unit.into();
        non_empty(&unit, "outcome unit")?;
        Ok(Self {
            point,
            lower: Some(lower),
            upper: Some(upper),
            unit,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutcomeTimeHorizon {
    pub start_unix_ms: i64,
    pub end_unix_ms: i64,
}

impl OutcomeTimeHorizon {
    pub fn new(start_unix_ms: i64, end_unix_ms: i64) -> Result<Self> {
        if end_unix_ms < start_unix_ms {
            return Err(OutcomeError::InvalidTimeHorizon {
                start_unix_ms,
                end_unix_ms,
            });
        }
        Ok(Self {
            start_unix_ms,
            end_unix_ms,
        })
    }
}

/// Where an outcome number came from. These are references, not claims that the
/// source is necessarily correct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutcomeSourceRef {
    BaselineEvidence(String),
    ScenarioModelOutput { model_id: String, run_id: String },
    CausalEstimate(String),
    HumanAssessment(String),
    ExternalStudy(String),
}

impl OutcomeSourceRef {
    fn validate(&self) -> Result<()> {
        match self {
            Self::BaselineEvidence(id)
            | Self::CausalEstimate(id)
            | Self::HumanAssessment(id)
            | Self::ExternalStudy(id) => non_empty(id, "outcome source id"),
            Self::ScenarioModelOutput { model_id, run_id } => {
                non_empty(model_id, "scenario model id")?;
                non_empty(run_id, "scenario model run id")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeScopeKind {
    SystemWide,
    GeographicRegion,
    PopulationGroup,
    Facility,
    Ecosystem,
    StakeholderGroup,
    Other,
}

/// A distributional slice makes winners, losers, and heterogeneous effects
/// visible instead of hiding them inside a system-wide average.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionSlice {
    pub scope_kind: OutcomeScopeKind,
    pub scope_label: String,
    pub estimate: OutcomeEstimate,
}

impl DistributionSlice {
    pub fn new(
        scope_kind: OutcomeScopeKind,
        scope_label: impl Into<String>,
        estimate: OutcomeEstimate,
    ) -> Result<Self> {
        let scope_label = scope_label.into();
        non_empty(&scope_label, "distribution scope label")?;
        Ok(Self {
            scope_kind,
            scope_label,
            estimate,
        })
    }
}

/// One consequence dimension. It carries no hidden desirability weight.
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeDimension {
    pub id: String,
    pub label: String,
    pub estimate: OutcomeEstimate,
    pub horizon: Option<OutcomeTimeHorizon>,
    pub sources: Vec<OutcomeSourceRef>,
    pub distributional_slices: Vec<DistributionSlice>,
    pub uncertainty_note: Option<String>,
}

impl OutcomeDimension {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        estimate: OutcomeEstimate,
        sources: Vec<OutcomeSourceRef>,
    ) -> Result<Self> {
        let id = id.into();
        let label = label.into();
        non_empty(&id, "outcome dimension id")?;
        non_empty(&label, "outcome dimension label")?;
        if sources.is_empty() {
            return Err(OutcomeError::MissingSource);
        }
        for source in &sources {
            source.validate()?;
        }
        Ok(Self {
            id,
            label,
            estimate,
            horizon: None,
            sources,
            distributional_slices: Vec::new(),
            uncertainty_note: None,
        })
    }

    pub fn with_horizon(mut self, horizon: OutcomeTimeHorizon) -> Self {
        self.horizon = Some(horizon);
        self
    }

    pub fn with_distributional_slice(mut self, slice: DistributionSlice) -> Self {
        self.distributional_slices.push(slice);
        self
    }

    pub fn with_uncertainty_note(mut self, note: impl Into<String>) -> Result<Self> {
        let note = note.into();
        non_empty(&note, "outcome uncertainty note")?;
        self.uncertainty_note = Some(note);
        Ok(self)
    }
}

/// Multi-dimensional consequences for one scenario.
///
/// This type has no `total_score`, `utility`, `fitness`, or implicit ranking.
/// Consumers that want a preference ordering must supply an explicit, auditable
/// value model outside this crate.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioOutcomeVector {
    pub scenario_id: String,
    pub dimensions: Vec<OutcomeDimension>,
}

impl ScenarioOutcomeVector {
    pub fn new(
        scenario_id: impl Into<String>,
        dimensions: Vec<OutcomeDimension>,
    ) -> Result<Self> {
        let scenario_id = scenario_id.into();
        non_empty(&scenario_id, "scenario id")?;
        if dimensions.is_empty() {
            return Err(OutcomeError::MissingDimensions);
        }

        let mut ids = HashSet::new();
        for dimension in &dimensions {
            if !ids.insert(dimension.id.clone()) {
                return Err(OutcomeError::DuplicateDimension(dimension.id.clone()));
            }
        }

        Ok(Self {
            scenario_id,
            dimensions,
        })
    }

    pub fn dimension(&self, id: &str) -> Option<&OutcomeDimension> {
        self.dimensions.iter().find(|dimension| dimension.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> OutcomeSourceRef {
        OutcomeSourceRef::ScenarioModelOutput {
            model_id: "watershed-twin".into(),
            run_id: "run-001".into(),
        }
    }

    #[test]
    fn interval_requires_ordered_bounds() {
        assert!(matches!(
            OutcomeEstimate::interval(10.0, 12.0, 14.0, "%"),
            Err(OutcomeError::InvalidInterval { .. })
        ));
        assert!(OutcomeEstimate::interval(10.0, 8.0, 14.0, "%").is_ok());
    }

    #[test]
    fn vector_rejects_duplicate_dimensions() {
        let dimension = OutcomeDimension::new(
            "water-reliability",
            "Water reliability",
            OutcomeEstimate::point(0.9, "fraction").unwrap(),
            vec![source()],
        )
        .unwrap();

        assert_eq!(
            ScenarioOutcomeVector::new(
                "scenario-1",
                vec![dimension.clone(), dimension],
            )
            .unwrap_err(),
            OutcomeError::DuplicateDimension("water-reliability".into())
        );
    }

    #[test]
    fn distributional_slices_preserve_heterogeneous_effects() {
        let dimension = OutcomeDimension::new(
            "household-disruption",
            "Household service disruption",
            OutcomeEstimate::interval(1200.0, 900.0, 1600.0, "households").unwrap(),
            vec![source()],
        )
        .unwrap()
        .with_distributional_slice(
            DistributionSlice::new(
                OutcomeScopeKind::GeographicRegion,
                "upstream district",
                OutcomeEstimate::point(200.0, "households").unwrap(),
            )
            .unwrap(),
        )
        .with_distributional_slice(
            DistributionSlice::new(
                OutcomeScopeKind::GeographicRegion,
                "downstream district",
                OutcomeEstimate::point(1000.0, "households").unwrap(),
            )
            .unwrap(),
        );

        assert_eq!(dimension.distributional_slices.len(), 2);
        assert_ne!(
            dimension.distributional_slices[0].estimate.point,
            dimension.distributional_slices[1].estimate.point
        );
    }

    #[test]
    fn each_dimension_requires_provenance() {
        assert_eq!(
            OutcomeDimension::new(
                "wetland-area",
                "Wetland area",
                OutcomeEstimate::point(42.0, "ha").unwrap(),
                vec![],
            )
            .unwrap_err(),
            OutcomeError::MissingSource
        );
    }

    #[test]
    fn consequence_vector_preserves_multiple_incommensurate_dimensions() {
        let water = OutcomeDimension::new(
            "water-reliability",
            "Water reliability",
            OutcomeEstimate::point(0.92, "fraction").unwrap(),
            vec![source()],
        )
        .unwrap();
        let habitat = OutcomeDimension::new(
            "wetland-area",
            "Wetland area",
            OutcomeEstimate::point(812.0, "ha").unwrap(),
            vec![source()],
        )
        .unwrap();
        let cost = OutcomeDimension::new(
            "capital-cost",
            "Capital cost",
            OutcomeEstimate::interval(80.0, 70.0, 100.0, "million-ZAR").unwrap(),
            vec![source()],
        )
        .unwrap();

        let vector = ScenarioOutcomeVector::new("scenario-2", vec![water, habitat, cost]).unwrap();
        assert_eq!(vector.dimensions.len(), 3);
        assert_eq!(vector.dimension("wetland-area").unwrap().estimate.unit, "ha");
    }
}
