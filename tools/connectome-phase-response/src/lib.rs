//! Fixed-point circular phase-response transfer algebra.
//!
//! This crate is a measurement/benchmark primitive, not a fly simulator or
//! causal-mechanism estimator. Phase is represented in microturns (1,000,000
//! units per cycle), and all wrapping/distance operations are exact integers.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

/// Number of legs in the registered hexapod response tensor.
pub const LEG_COUNT: usize = 6;
/// One complete oscillator cycle in the fixed-point representation.
pub const PHASE_SCALE: i64 = 1_000_000;
/// Canonical phase interval is [-0.5, 0.5) cycles.
pub const PHASE_HALF_TURN: i64 = PHASE_SCALE / 2;
/// V1 phase-grid size. Biological studies may preregister a different grid.
pub const DEFAULT_PHASE_BINS: u8 = 8;

/// Canonical fixed-point oscillator phase.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct Phase(i32);

impl Phase {
    /// Construct a phase from any signed microturn count, wrapping to
    /// [-500_000, 500_000). No floating-point conversion is performed.
    pub fn from_microturns(value: i64) -> Self {
        Self(wrap_microturns(value) as i32)
    }

    /// Return the canonical signed microturn count.
    pub const fn microturns(self) -> i32 {
        self.0
    }

    /// Exact circular difference, left minus right, wrapped to [-0.5, 0.5).
    pub fn difference(left: Self, right: Self) -> Self {
        Self::from_microturns(i64::from(left.0) - i64::from(right.0))
    }

    /// Exact shortest circular distance in microturns.
    pub fn circular_distance(self, other: Self) -> u32 {
        Self::difference(self, other).0.unsigned_abs()
    }

    /// Whether this value is already in canonical form.
    pub const fn is_canonical(self) -> bool {
        self.0 >= -(PHASE_HALF_TURN as i32) && self.0 < PHASE_HALF_TURN as i32
    }
}

fn wrap_microturns(value: i64) -> i64 {
    let wrapped = value.rem_euclid(PHASE_SCALE);
    if wrapped >= PHASE_HALF_TURN {
        wrapped - PHASE_SCALE
    } else {
        wrapped
    }
}

/// Stable zero-based leg identity. An axis position is not accepted unless it
/// is explicitly mapped to one of these six canonical identities.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct LegId(u8);

impl LegId {
    pub fn new(index: u8) -> Result<Self, PhaseError> {
        if usize::from(index) < LEG_COUNT {
            Ok(Self(index))
        } else {
            Err(PhaseError::InvalidLeg(index))
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn value(self) -> u8 {
        self.0
    }
}

/// Context labels correspond to declared preparation/condition strata. They
/// are not interchangeable and must be source-bound in biological runs.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum Context {
    BallContact,
    SlipperySurface,
    AirStepping,
    AirFeCoSilenced,
}

/// Registered perturbation class. This vocabulary does not authorize a
/// biological procedure.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum ProbeClass {
    LocalPhaseKick,
    LoadOrContact,
    ProprioceptiveChannel,
    DescendingCommand,
    MechanicalBodyOrSubstrate,
}

/// A typed design cell in the phase-response transfer tensor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ProbeKey {
    pub context: Context,
    pub class: ProbeClass,
    pub source_leg: Option<LegId>,
    pub phase_bin: u8,
    pub delay_index: u8,
}

impl ProbeKey {
    pub fn validate(self) -> Result<(), PhaseError> {
        if self.phase_bin >= DEFAULT_PHASE_BINS {
            return Err(PhaseError::InvalidPhaseBin(self.phase_bin));
        }
        let source_required = self.class != ProbeClass::DescendingCommand;
        if source_required && self.source_leg.is_none() {
            return Err(PhaseError::SourceLegRequired(self.class));
        }
        if !source_required && self.source_leg.is_some() {
            return Err(PhaseError::SourceLegForbidden(self.class));
        }
        Ok(())
    }
}

/// An explicit reason why a phase-response observation is unavailable.
/// None of these states may be silently encoded as a zero response.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum MeasurementStatus {
    NoResponse,
    FailedAssay,
    MissingTrack,
    OutOfSupport,
    NotAssessed,
}

/// One paired control/perturbed response row or an explicit unavailable state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RowOutcome {
    Measured {
        control: [PhaseSample; LEG_COUNT],
        perturbed: [PhaseSample; LEG_COUNT],
    },
    Unavailable(MeasurementStatus),
}

/// Source-bound row within a phase-response transfer surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferRow {
    pub key: ProbeKey,
    pub outcome: RowOutcome,
}

/// Six-leg phase-reset vector in canonical leg-axis order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseReset {
    pub source_leg: Option<LegId>,
    /// Wrapped phase displacement, in canonical microturns.
    pub by_leg: [Phase; LEG_COUNT],
    /// Integer changes in cycle winding. Kept separate from wrapped phase.
    pub cycle_slips_by_leg: [i64; LEG_COUNT],
}

impl PhaseReset {
    /// Sum the absolute phase reset outside the stimulated leg. For a global
    /// descending command (no source leg), all six axes are included.
    pub fn off_diagonal_l1_microturns(self) -> u64 {
        self.by_leg
            .iter()
            .enumerate()
            .filter(|(axis, _)| self.source_leg.map(LegId::index) != Some(*axis))
            .map(|(_, value)| u64::from(value.microturns().unsigned_abs()))
            .sum()
    }

    pub fn local_reset(self) -> Option<Phase> {
        self.source_leg.map(|leg| self.by_leg[leg.index()])
    }

    pub fn off_diagonal_cycle_slips_l1(self) -> u64 {
        self.cycle_slips_by_leg
            .iter()
            .enumerate()
            .filter(|(axis, _)| self.source_leg.map(LegId::index) != Some(*axis))
            .map(|(_, slips)| slips.unsigned_abs())
            .sum()
    }

    pub fn local_cycle_slips(self) -> Option<i64> {
        self.source_leg.map(|leg| self.cycle_slips_by_leg[leg.index()])
    }
}

/// Validated set of phase-response rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseResponseSurface {
    rows: BTreeMap<ProbeKey, TransferRow>,
}

impl PhaseResponseSurface {
    /// Reject duplicate design cells and invalid axis/source declarations.
    pub fn new(rows: Vec<TransferRow>) -> Result<Self, PhaseError> {
        let mut indexed = BTreeMap::new();
        for row in rows {
            row.key.validate()?;
            if indexed.insert(row.key, row).is_some() {
                return Err(PhaseError::DuplicateProbeKey);
            }
        }
        Ok(Self { rows: indexed })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Require exactly the declared design grid: no omissions, duplicates or
    /// unregistered extra probes.
    pub fn validate_complete(&self, expected: &[ProbeKey]) -> Result<(), PhaseError> {
        let mut expected_set = BTreeSet::new();
        for key in expected {
            key.validate()?;
            if !expected_set.insert(*key) {
                return Err(PhaseError::DuplicateExpectedKey);
            }
        }
        let actual: BTreeSet<_> = self.rows.keys().copied().collect();
        if actual != expected_set {
            let missing = expected_set.difference(&actual).count();
            let unexpected = actual.difference(&expected_set).count();
            return Err(PhaseError::GridMismatch { missing, unexpected });
        }
        Ok(())
    }

    pub fn row(&self, key: &ProbeKey) -> Option<&TransferRow> {
        self.rows.get(key)
    }

    /// Return the intervention-conditioned phase reset across all six legs.
    /// Unavailable observations produce a typed error and are never treated
    /// as six zero-valued responses.
    pub fn phase_reset(&self, key: &ProbeKey) -> Result<PhaseReset, PhaseError> {
        let row = self.rows.get(key).ok_or(PhaseError::MissingProbeKey)?;
        match &row.outcome {
            RowOutcome::Measured { control, perturbed } => {
                let by_leg = std::array::from_fn(|axis| {
                    Phase::difference(
                        perturbed[axis].fractional_phase(),
                        control[axis].fractional_phase(),
                    )
                });
                let cycle_slips_by_leg = std::array::from_fn(|axis| {
                    perturbed[axis].cycle_index() - control[axis].cycle_index()
                });
                Ok(PhaseReset {
                    source_leg: key.source_leg,
                    by_leg,
                    cycle_slips_by_leg,
                })
            }
            RowOutcome::Unavailable(status) => Err(PhaseError::Unavailable(*status)),
        }
    }

    pub fn measured_keys(&self) -> impl Iterator<Item = ProbeKey> + '_ {
        self.rows.iter().filter_map(|(key, row)| {
            matches!(&row.outcome, RowOutcome::Measured { .. }).then_some(*key)
        })
    }
}

/// Predicted response for one model and one registered probe cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbePrediction {
    pub key: ProbeKey,
    pub by_leg: [Phase; LEG_COUNT],
    pub cycle_slips_by_leg: [i64; LEG_COUNT],
}

/// A model signature contains steady-state summaries and the full phase-reset
/// predictions. The summaries are retained to demonstrate when frequency/PLV
/// rankings are insufficient; they are not used as substitutes for transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSignature {
    pub model_id: String,
    pub steady_state_frequency_millihz: u32,
    pub steady_state_plv_milli: u16,
    pub predictions: Vec<ProbePrediction>,
}

impl ModelSignature {
    fn prediction_map(
        &self,
    ) -> Result<BTreeMap<ProbeKey, ([Phase; LEG_COUNT], [i64; LEG_COUNT])>, PhaseError> {
        if self.model_id.is_empty() {
            return Err(PhaseError::EmptyModelId);
        }
        if self.steady_state_plv_milli > 1_000 {
            return Err(PhaseError::InvalidPlv(self.steady_state_plv_milli));
        }
        let mut map = BTreeMap::new();
        for prediction in &self.predictions {
            prediction.key.validate()?;
            if map
                .insert(
                    prediction.key,
                    (prediction.by_leg, prediction.cycle_slips_by_leg),
                )
                .is_some()
            {
                return Err(PhaseError::DuplicatePredictionKey(self.model_id.clone()));
            }
        }
        if map.is_empty() {
            return Err(PhaseError::EmptyPredictionSet(self.model_id.clone()));
        }
        Ok(map)
    }
}

/// Exact circular L1 separation in microturns between two model-predicted
/// six-leg response vectors. This is a deterministic synthetic diagnostic,
/// not a likelihood, probability or biological effect size.
pub fn response_vector_distance(
    left: &[Phase; LEG_COUNT],
    right: &[Phase; LEG_COUNT],
) -> u64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| u64::from(a.circular_distance(*b)))
        .sum()
}

/// Compute the minimum pairwise response separation for every candidate
/// probe. Every model must predict every same probe; incomplete model
/// signatures fail closed instead of improving a score by dropping a pair.
pub fn minimum_pairwise_probe_separation(
    models: &[ModelSignature],
) -> Result<BTreeMap<ProbeKey, u64>, PhaseError> {
    if models.len() < 2 {
        return Err(PhaseError::InsufficientModels);
    }
    let mut ids = BTreeSet::new();
    let maps = models
        .iter()
        .map(|model| {
            if !ids.insert(model.model_id.clone()) {
                return Err(PhaseError::DuplicateModelId(model.model_id.clone()));
            }
            model.prediction_map()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let reference_keys: BTreeSet<_> = maps[0].keys().copied().collect();
    for (index, map) in maps.iter().enumerate().skip(1) {
        let keys: BTreeSet<_> = map.keys().copied().collect();
        if keys != reference_keys {
            return Err(PhaseError::ModelProbeGridMismatch {
                model_id: models[index].model_id.clone(),
            });
        }
    }

    let mut result = BTreeMap::new();
    for key in reference_keys {
        let mut minimum = u64::MAX;
        for left in 0..maps.len() {
            for right in (left + 1)..maps.len() {
                let distance = response_vector_distance(
                    &maps[left][&key].0,
                    &maps[right][&key].0,
                );
                minimum = minimum.min(distance);
            }
        }
        result.insert(key, minimum);
    }
    Ok(result)
}

/// Compute the minimum pairwise cycle-slip separation for every probe cell.
/// This remains a separate quantity from fractional phase distance.
pub fn minimum_pairwise_cycle_slip_separation(
    models: &[ModelSignature],
) -> Result<BTreeMap<ProbeKey, u64>, PhaseError> {
    if models.len() < 2 {
        return Err(PhaseError::InsufficientModels);
    }
    let mut ids = BTreeSet::new();
    let maps = models
        .iter()
        .map(|model| {
            if !ids.insert(model.model_id.clone()) {
                return Err(PhaseError::DuplicateModelId(model.model_id.clone()));
            }
            model.prediction_map()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let reference_keys: BTreeSet<_> = maps[0].keys().copied().collect();
    for (index, map) in maps.iter().enumerate().skip(1) {
        let keys: BTreeSet<_> = map.keys().copied().collect();
        if keys != reference_keys {
            return Err(PhaseError::ModelProbeGridMismatch {
                model_id: models[index].model_id.clone(),
            });
        }
    }

    let mut result = BTreeMap::new();
    for key in reference_keys {
        let mut minimum = u64::MAX;
        for left in 0..maps.len() {
            for right in (left + 1)..maps.len() {
                let distance = maps[left][&key]
                    .1
                    .iter()
                    .zip(maps[right][&key].1.iter())
                    .map(|(a, b)| a.abs_diff(*b))
                    .sum::<u64>();
                minimum = minimum.min(distance);
            }
        }
        result.insert(key, minimum);
    }
    Ok(result)
}

/// Minimum separation across all model pairs and supplied probe cells, for
/// the fractional phase component only.
pub fn minimum_campaign_separation(models: &[ModelSignature]) -> Result<u64, PhaseError> {
    let by_probe = minimum_pairwise_probe_separation(models)?;
    by_probe.values().copied().min().ok_or(PhaseError::EmptyProbeGrid)
}

/// Minimum separation across all model pairs and supplied probe cells, for
/// integer cycle slips only.
pub fn minimum_campaign_cycle_slip_separation(
    models: &[ModelSignature],
) -> Result<u64, PhaseError> {
    let by_probe = minimum_pairwise_cycle_slip_separation(models)?;
    by_probe.values().copied().min().ok_or(PhaseError::EmptyProbeGrid)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhaseError {
    InvalidLeg(u8),
    InvalidPhaseBin(u8),
    SourceLegRequired(ProbeClass),
    SourceLegForbidden(ProbeClass),
    DuplicateProbeKey,
    DuplicateExpectedKey,
    MissingProbeKey,
    GridMismatch { missing: usize, unexpected: usize },
    Unavailable(MeasurementStatus),
    EmptyModelId,
    InvalidPlv(u16),
    DuplicatePredictionKey(String),
    EmptyPredictionSet(String),
    InsufficientModels,
    DuplicateModelId(String),
    ModelProbeGridMismatch { model_id: String },
    EmptyProbeGrid,
}

impl fmt::Display for PhaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLeg(value) => write!(f, "invalid leg ID {value}; expected 0..6"),
            Self::InvalidPhaseBin(value) => write!(f, "invalid phase bin {value}; expected 0..8"),
            Self::SourceLegRequired(class) => write!(f, "probe class {class:?} requires a source leg"),
            Self::SourceLegForbidden(class) => write!(f, "probe class {class:?} is a global input and forbids a source leg"),
            Self::DuplicateProbeKey => write!(f, "duplicate phase-response probe key"),
            Self::DuplicateExpectedKey => write!(f, "duplicate key in expected phase-response grid"),
            Self::MissingProbeKey => write!(f, "phase-response probe key is absent"),
            Self::GridMismatch { missing, unexpected } => write!(f, "phase-response grid mismatch: {missing} missing, {unexpected} unexpected"),
            Self::Unavailable(status) => write!(f, "phase-response measurement unavailable: {status:?}"),
            Self::EmptyModelId => write!(f, "model ID is empty"),
            Self::InvalidPlv(value) => write!(f, "PLV milli value {value} exceeds 1000"),
            Self::DuplicatePredictionKey(model) => write!(f, "model {model} has duplicate prediction keys"),
            Self::EmptyPredictionSet(model) => write!(f, "model {model} has no predictions"),
            Self::InsufficientModels => write!(f, "at least two models are required for pairwise separation"),
            Self::DuplicateModelId(model) => write!(f, "duplicate model ID {model}"),
            Self::ModelProbeGridMismatch { model_id } => write!(f, "model {model_id} does not predict the identical probe grid"),
            Self::EmptyProbeGrid => write!(f, "phase-response probe grid is empty"),
        }
    }
}

impl Error for PhaseError {}

/// One phase sample with both its integer cycle winding and fractional
/// position on the cycle. Keeping both prevents a full step/cycle slip from
/// being aliased to zero by circular wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PhaseSample {
    cycle_index: i64,
    fractional_phase: Phase,
}

impl PhaseSample {
    /// Split an unwrapped microturn count into integer cycle winding and
    /// canonical fractional phase.
    pub fn from_unwrapped_microturns(value: i64) -> Self {
        let whole = value.div_euclid(PHASE_SCALE);
        let remainder = value.rem_euclid(PHASE_SCALE);
        if remainder >= PHASE_HALF_TURN {
            Self {
                cycle_index: whole + 1,
                fractional_phase: Phase::from_microturns(remainder - PHASE_SCALE),
            }
        } else {
            Self {
                cycle_index: whole,
                fractional_phase: Phase::from_microturns(remainder),
            }
        }
    }

    pub const fn cycle_index(self) -> i64 {
        self.cycle_index
    }

    pub const fn fractional_phase(self) -> Phase {
        self.fractional_phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(context: Context, source: u8, phase_bin: u8, delay_index: u8) -> ProbeKey {
        ProbeKey {
            context,
            class: ProbeClass::LocalPhaseKick,
            source_leg: Some(LegId::new(source).unwrap()),
            phase_bin,
            delay_index,
        }
    }

    fn zeros() -> [Phase; LEG_COUNT] {
        [Phase::from_microturns(0); LEG_COUNT]
    }

    fn zero_samples() -> [PhaseSample; LEG_COUNT] {
        [PhaseSample::from_unwrapped_microturns(0); LEG_COUNT]
    }

    #[test]
    fn phase_difference_wraps_across_cycle_boundary() {
        let left = Phase::from_microturns(490_000);
        let right = Phase::from_microturns(-490_000);
        assert_eq!(Phase::difference(left, right).microturns(), -20_000);
        assert_eq!(left.circular_distance(right), 20_000);
        assert_eq!(Phase::from_microturns(1_490_000).microturns(), 490_000);
    }

    #[test]
    fn phase_reset_keeps_sender_recipient_axes_and_off_diagonal_transfer() {
        let probe = key(Context::BallContact, 0, 2, 1);
        let control = zero_samples();
        let mut perturbed = zero_samples();
        perturbed[0] = PhaseSample::from_unwrapped_microturns(100_000);
        perturbed[2] = PhaseSample::from_unwrapped_microturns(25_000);
        let surface = PhaseResponseSurface::new(vec![TransferRow {
            key: probe,
            outcome: RowOutcome::Measured { control, perturbed },
        }]).unwrap();

        let response = surface.phase_reset(&probe).unwrap();
        assert_eq!(response.local_reset().unwrap().microturns(), 100_000);
        assert_eq!(response.by_leg[2].microturns(), 25_000);
        assert_eq!(response.off_diagonal_l1_microturns(), 25_000);
    }

    #[test]
    fn local_only_response_is_not_confused_with_cross_leg_transfer() {
        let probe = key(Context::AirStepping, 3, 0, 1);
        let control = zero_samples();
        let mut perturbed = zero_samples();
        perturbed[3] = PhaseSample::from_unwrapped_microturns(-80_000);
        let surface = PhaseResponseSurface::new(vec![TransferRow {
            key: probe,
            outcome: RowOutcome::Measured { control, perturbed },
        }]).unwrap();
        assert_eq!(surface.phase_reset(&probe).unwrap().off_diagonal_l1_microturns(), 0);
    }

    #[test]
    fn unavailable_measurements_are_never_silently_zero_filled() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let surface = PhaseResponseSurface::new(vec![TransferRow {
            key: probe,
            outcome: RowOutcome::Unavailable(MeasurementStatus::NoResponse),
        }]).unwrap();
        assert_eq!(surface.phase_reset(&probe), Err(PhaseError::Unavailable(MeasurementStatus::NoResponse)));
    }

    #[test]
    fn source_identity_is_required_by_leg_local_probe_classes() {
        let invalid = ProbeKey {
            context: Context::BallContact,
            class: ProbeClass::LocalPhaseKick,
            source_leg: None,
            phase_bin: 0,
            delay_index: 0,
        };
        assert_eq!(invalid.validate(), Err(PhaseError::SourceLegRequired(ProbeClass::LocalPhaseKick)));

        let invalid_global = ProbeKey {
            context: Context::AirStepping,
            class: ProbeClass::DescendingCommand,
            source_leg: Some(LegId::new(0).unwrap()),
            phase_bin: 0,
            delay_index: 0,
        };
        assert_eq!(invalid_global.validate(), Err(PhaseError::SourceLegForbidden(ProbeClass::DescendingCommand)));
    }

    #[test]
    fn phase_grid_rejects_duplicate_and_missing_rows() {
        let first = key(Context::BallContact, 0, 0, 1);
        let second = key(Context::BallContact, 0, 1, 1);
        let surface = PhaseResponseSurface::new(vec![
            TransferRow { key: first, outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed) },
            TransferRow { key: second, outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed) },
        ]).unwrap();
        assert_eq!(surface.validate_complete(&[first]), Err(PhaseError::GridMismatch { missing: 0, unexpected: 1 }));
        assert!(matches!(
            PhaseResponseSurface::new(vec![
                TransferRow { key: first, outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed) },
                TransferRow { key: first, outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed) },
            ]),
            Err(PhaseError::DuplicateProbeKey)
        ));
    }

    #[test]
    fn steady_state_ties_can_hide_phase_response_transfer_difference() {
        let probe = key(Context::BallContact, 0, 0, 1);
        let mut local_only = zeros();
        local_only[0] = Phase::from_microturns(80_000);
        let mut sensory_transfer = local_only;
        sensory_transfer[1] = Phase::from_microturns(30_000);

        let central = ModelSignature {
            model_id: "central-only".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction { key: probe, by_leg: local_only, cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        let proprioceptive = ModelSignature {
            model_id: "central-plus-proprioception".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction { key: probe, by_leg: sensory_transfer, cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        let separation = minimum_pairwise_probe_separation(&[central, proprioceptive]).unwrap();
        assert_eq!(separation[&probe], 30_000);
    }

    #[test]
    fn contact_and_air_contexts_remain_distinct_design_cells() {
        let ball = key(Context::BallContact, 0, 3, 1);
        let air = key(Context::AirStepping, 0, 3, 1);
        assert_ne!(ball, air);
        assert!(ball.validate().is_ok());
        assert!(air.validate().is_ok());
    }

    #[test]
    fn models_must_predict_the_same_probe_grid() {
        let probe_a = key(Context::BallContact, 0, 0, 1);
        let probe_b = key(Context::AirStepping, 0, 0, 1);
        let first = ModelSignature {
            model_id: "a".to_string(),
            steady_state_frequency_millihz: 10_000,
            steady_state_plv_milli: 500,
            predictions: vec![ProbePrediction { key: probe_a, by_leg: zeros(), cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        let second = ModelSignature {
            model_id: "b".to_string(),
            steady_state_frequency_millihz: 10_000,
            steady_state_plv_milli: 500,
            predictions: vec![ProbePrediction { key: probe_b, by_leg: zeros(), cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        assert!(matches!(
            minimum_pairwise_probe_separation(&[first, second]),
            Err(PhaseError::ModelProbeGridMismatch { .. })
        ));
    }

    #[test]
    fn separation_is_circular_l1_microturns_not_a_probability() {
        let left = [Phase::from_microturns(490_000); LEG_COUNT];
        let right = [Phase::from_microturns(-490_000); LEG_COUNT];
        assert_eq!(response_vector_distance(&left, &right), 6 * 20_000);
    }
    #[test]
    fn phase_bin_can_be_uninformative_while_another_bin_separates_models() {
        let phase_zero = key(Context::BallContact, 0, 0, 1);
        let phase_half = key(Context::BallContact, 0, 4, 1);
        let mut common = zeros();
        common[0] = Phase::from_microturns(50_000);
        let mut separated = common;
        separated[2] = Phase::from_microturns(40_000);
        let model_a = ModelSignature {
            model_id: "central".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![
                ProbePrediction { key: phase_zero, by_leg: common, cycle_slips_by_leg: [0; LEG_COUNT] },
                ProbePrediction { key: phase_half, by_leg: common, cycle_slips_by_leg: [0; LEG_COUNT] },
            ],
        };
        let model_b = ModelSignature {
            model_id: "sensory".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![
                ProbePrediction { key: phase_zero, by_leg: common },
                ProbePrediction { key: phase_half, by_leg: separated, cycle_slips_by_leg: [0; LEG_COUNT] },
            ],
        };
        let separation = minimum_pairwise_probe_separation(&[model_a, model_b]).unwrap();
        assert_eq!(separation[&phase_zero], 0);
        assert_eq!(separation[&phase_half], 40_000);
    }

    #[test]
    fn shared_input_challenger_can_mimic_off_diagonal_transfer() {
        let probe = key(Context::BallContact, 0, 2, 1);
        let mut apparent_transfer = zeros();
        apparent_transfer[0] = Phase::from_microturns(70_000);
        apparent_transfer[3] = Phase::from_microturns(20_000);
        let central_path = ModelSignature {
            model_id: "direct-central-coupling".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction { key: probe, by_leg: apparent_transfer, cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        let common_drive = ModelSignature {
            model_id: "latent-shared-input".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction { key: probe, by_leg: apparent_transfer, cycle_slips_by_leg: [0; LEG_COUNT] }],
        };
        assert_eq!(minimum_campaign_separation(&[central_path, common_drive]).unwrap(), 0);
    }

    #[test]
    fn all_unavailable_states_remain_typed() {
        let statuses = [
            MeasurementStatus::NoResponse,
            MeasurementStatus::FailedAssay,
            MeasurementStatus::MissingTrack,
            MeasurementStatus::OutOfSupport,
            MeasurementStatus::NotAssessed,
        ];
        let probe = key(Context::AirStepping, 1, 2, 1);
        for status in statuses {
            let surface = PhaseResponseSurface::new(vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Unavailable(status),
            }])
            .unwrap();
            assert_eq!(surface.phase_reset(&probe), Err(PhaseError::Unavailable(status)));
        }
    }

    #[test]
    fn full_cycle_slip_is_not_aliased_to_zero_phase() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let control = zero_samples();
        let mut perturbed = zero_samples();
        perturbed[0] = PhaseSample::from_unwrapped_microturns(1_000_000);
        perturbed[2] = PhaseSample::from_unwrapped_microturns(1_100_000);
        let surface = PhaseResponseSurface::new(vec![TransferRow {
            key: probe,
            outcome: RowOutcome::Measured { control, perturbed },
        }]).unwrap();

        let response = surface.phase_reset(&probe).unwrap();
        assert_eq!(response.by_leg[0].microturns(), 0);
        assert_eq!(response.cycle_slips_by_leg[0], 1);
        assert_eq!(response.by_leg[2].microturns(), 100_000);
        assert_eq!(response.cycle_slips_by_leg[2], 1);
        assert_eq!(response.off_diagonal_cycle_slips_l1(), 1);
    }

    #[test]
    fn model_separation_keeps_cycle_slips_separate_from_fractional_phase() {
        let probe = key(Context::BallContact, 0, 0, 1);
        let fractional = zeros();
        let central = ModelSignature {
            model_id: "central".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: fractional,
                cycle_slips_by_leg: [0; LEG_COUNT],
            }],
        };
        let slipped = ModelSignature {
            model_id: "cycle-slip".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: fractional,
                cycle_slips_by_leg: [0, 1, 0, 0, 0, 0],
            }],
        };
        assert_eq!(
            minimum_pairwise_probe_separation(&[central.clone(), slipped.clone()]).unwrap()[&probe],
            0
        );
        assert_eq!(
            minimum_pairwise_cycle_slip_separation(&[central, slipped]).unwrap()[&probe],
            1
        );
    }

}
