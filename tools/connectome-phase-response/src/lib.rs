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
/// Maximum registered perturbation magnitude levels in V1.
pub const MAX_MAGNITUDE_BINS: u8 = 16;

/// Canonical fixed-point oscillator phase.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct Phase(i32);

impl Phase {
    /// Construct a phase from any signed microturn count, wrapping to
    /// [-500_000, 500_000). No floating-point conversion is performed.
    pub fn from_microturns(value: i64) -> Self {
        Self(wrap_microturns(value) as i32)
    }

    /// Construct a phase sample from an unwrapped phase in radians.
    ///
    /// The conversion quantizes to the nearest microturn using Rust's
    /// round-half-away-from-zero rule. The quantization error is less than
    /// half a microturn; upstream phase-estimator uncertainty is not included.
    pub fn from_unwrapped_radians(value: f64) -> Result<Self, PhaseError> {
        if !value.is_finite() {
            return Err(PhaseError::NonFinitePhase);
        }
        let microturns = value * (PHASE_SCALE as f64) / std::f64::consts::TAU;
        if !microturns.is_finite()
            || microturns < i64::MIN as f64
            || microturns >= i64::MAX as f64
        {
            return Err(PhaseError::PhaseOutOfRange);
        }
        let rounded = microturns.round();
        if rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
            return Err(PhaseError::PhaseOutOfRange);
        }
        Ok(Self::from_unwrapped_microturns(rounded as i64))
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

/// Identity for descending command inputs. Named neurons remain distinct.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum DescendingNeuron {
    DNg100,
    DNg97,
    DNb08,
}

/// A typed design cell in the phase-response transfer tensor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ProbeKey {
    pub context: Context,
    pub class: ProbeClass,
    pub source_leg: Option<LegId>,
    pub descending_neuron: Option<DescendingNeuron>,
    pub phase_bin: u8,
    pub magnitude_bin: u8,
    pub delay_index: u8,
}

impl ProbeKey {
    pub fn validate(self) -> Result<(), PhaseError> {
        if self.phase_bin >= DEFAULT_PHASE_BINS {
            return Err(PhaseError::InvalidPhaseBin(self.phase_bin));
        }
        if self.magnitude_bin >= MAX_MAGNITUDE_BINS {
            return Err(PhaseError::InvalidMagnitudeBin(self.magnitude_bin));
        }
        match self.class {
            ProbeClass::LocalPhaseKick => {
                if self.source_leg.is_none() {
                    return Err(PhaseError::SourceLegRequired(self.class));
                }
                if self.descending_neuron.is_some() {
                    return Err(PhaseError::DescendingNeuronForbidden);
                }
            }
            ProbeClass::DescendingCommand => {
                if self.source_leg.is_some() {
                    return Err(PhaseError::SourceLegForbidden(self.class));
                }
                if self.descending_neuron.is_none() {
                    return Err(PhaseError::DescendingNeuronRequired);
                }
            }
            _ => {
                if self.descending_neuron.is_some() {
                    return Err(PhaseError::DescendingNeuronForbidden);
                }
            }
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

/// Immutable source and evaluation-unit commitments for one response surface.
/// These digests bind lineage strings; they are not signature verification or
/// proof that the upstream bytes have been authenticated by a trusted party.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceBinding {
    pub dataset_id: String,
    pub dataset_revision_sha256: String,
    pub phase_estimator_sha256: String,
    pub axis_identity_receipt_sha256: String,
    pub raw_trajectory_sha256: String,
    pub perturbation_receipt_sha256: String,
    pub context_definition_sha256: String,
    pub animal_unit_token: String,
    pub session_id: String,
    pub trial_unit_token: String,
}

impl SourceBinding {
    pub fn validate(&self) -> Result<(), PhaseError> {
        for (field, value) in [
            ("dataset_id", self.dataset_id.as_str()),
            ("animal_unit_token", self.animal_unit_token.as_str()),
            ("session_id", self.session_id.as_str()),
            ("trial_unit_token", self.trial_unit_token.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(PhaseError::MissingSourceBindingField(field));
            }
        }
        for (field, digest) in [
            ("dataset_revision_sha256", self.dataset_revision_sha256.as_str()),
            ("phase_estimator_sha256", self.phase_estimator_sha256.as_str()),
            ("axis_identity_receipt_sha256", self.axis_identity_receipt_sha256.as_str()),
            ("raw_trajectory_sha256", self.raw_trajectory_sha256.as_str()),
            ("perturbation_receipt_sha256", self.perturbation_receipt_sha256.as_str()),
            ("context_definition_sha256", self.context_definition_sha256.as_str()),
        ] {
            if !valid_sha256(digest) {
                return Err(PhaseError::InvalidSourceDigest(field));
            }
        }
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

    pub fn off_diagonal_cycle_slips_l1(self) -> Result<u64, PhaseError> {
        self.cycle_slips_by_leg
            .iter()
            .enumerate()
            .filter(|(axis, _)| self.source_leg.map(LegId::index) != Some(*axis))
            .try_fold(0_u64, |total, (_, slips)| {
                total.checked_add(slips.unsigned_abs())
            })
            .ok_or(PhaseError::ArithmeticOverflow)
    }

    pub fn local_cycle_slips(self) -> Option<i64> {
        self.source_leg.map(|leg| self.cycle_slips_by_leg[leg.index()])
    }
}

/// Validated set of phase-response rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseResponseSurface {
    source_binding: SourceBinding,
    rows: BTreeMap<ProbeKey, TransferRow>,
}

impl PhaseResponseSurface {
    /// Require source lineage, reject duplicate design cells, and validate
    /// every probe key before admitting the response surface.
    pub fn new(
        source_binding: SourceBinding,
        rows: Vec<TransferRow>,
    ) -> Result<Self, PhaseError> {
        source_binding.validate()?;
        let mut indexed = BTreeMap::new();
        for row in rows {
            row.key.validate()?;
            if indexed.insert(row.key, row).is_some() {
                return Err(PhaseError::DuplicateProbeKey);
            }
        }
        Ok(Self {
            source_binding,
            rows: indexed,
        })
    }

    pub fn source_binding(&self) -> &SourceBinding {
        &self.source_binding
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
                    cycle_slip_delta(perturbed[axis], control[axis], by_leg[axis])
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

/// Recover the integer full-cycle slip after removing the wrapped
/// fractional phase displacement. This avoids counting a small phase advance
/// across the fractional branch boundary as a full-cycle slip.
fn cycle_slip_delta(perturbed: PhaseSample, control: PhaseSample, wrapped: Phase) -> i64 {
    let winding_delta = i128::from(perturbed.cycle_index())
        - i128::from(control.cycle_index());
    let fractional_delta = i128::from(perturbed.fractional_phase().microturns())
        - i128::from(control.fractional_phase().microturns());
    let unwrapped_delta = winding_delta * i128::from(PHASE_SCALE) + fractional_delta;
    let residual = unwrapped_delta - i128::from(wrapped.microturns());
    debug_assert_eq!(residual.rem_euclid(i128::from(PHASE_SCALE)), 0);
    (residual / i128::from(PHASE_SCALE)) as i64
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

fn checked_cycle_slip_vector_distance(
    left: &[i64; LEG_COUNT],
    right: &[i64; LEG_COUNT],
) -> Option<u64> {
    left.iter()
        .zip(right.iter())
        .try_fold(0_u64, |total, (a, b)| total.checked_add(a.abs_diff(*b)))
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
                let distance = checked_cycle_slip_vector_distance(
                    &maps[left][&key].1,
                    &maps[right][&key].1,
                )
                .ok_or(PhaseError::ArithmeticOverflow)?;
                minimum = minimum.min(distance);
            }
        }
        result.insert(key, minimum);
    }
    Ok(result)
}

/// Minimum pairwise separation for the complete registered probe portfolio.
/// Responses from all phase bins, contexts, delays and perturbation classes are
/// concatenated conceptually before distance is computed. This lets a portfolio
/// distinguish models when individual probe cells are individually blind.
pub fn minimum_joint_campaign_phase_separation(
    models: &[ModelSignature],
) -> Result<u64, PhaseError> {
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
    if reference_keys.is_empty() {
        return Err(PhaseError::EmptyProbeGrid);
    }
    for (index, map) in maps.iter().enumerate().skip(1) {
        let keys: BTreeSet<_> = map.keys().copied().collect();
        if keys != reference_keys {
            return Err(PhaseError::ModelProbeGridMismatch {
                model_id: models[index].model_id.clone(),
            });
        }
    }

    let mut minimum = u64::MAX;
    for left in 0..maps.len() {
        for right in (left + 1)..maps.len() {
            let portfolio_distance = reference_keys
                .iter()
                .try_fold(0_u64, |total, key| {
                    total.checked_add(response_vector_distance(
                        &maps[left][key].0,
                        &maps[right][key].0,
                    ))
                })
                .ok_or(PhaseError::ArithmeticOverflow)?;
            minimum = minimum.min(portfolio_distance);
        }
    }
    Ok(minimum)
}

/// Minimum pairwise cycle-slip separation for the complete registered probe
/// portfolio. This remains distinct from fractional phase displacement.
pub fn minimum_joint_campaign_cycle_slip_separation(
    models: &[ModelSignature],
) -> Result<u64, PhaseError> {
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
    if reference_keys.is_empty() {
        return Err(PhaseError::EmptyProbeGrid);
    }
    for (index, map) in maps.iter().enumerate().skip(1) {
        let keys: BTreeSet<_> = map.keys().copied().collect();
        if keys != reference_keys {
            return Err(PhaseError::ModelProbeGridMismatch {
                model_id: models[index].model_id.clone(),
            });
        }
    }

    let mut minimum = u64::MAX;
    for left in 0..maps.len() {
        for right in (left + 1)..maps.len() {
            let portfolio_distance = reference_keys
                .iter()
                .try_fold(0_u64, |total, key| {
                    let cell_distance = checked_cycle_slip_vector_distance(
                        &maps[left][key].1,
                        &maps[right][key].1,
                    )?;
                    total.checked_add(cell_distance)
                })
                .ok_or(PhaseError::ArithmeticOverflow)?;
            minimum = minimum.min(portfolio_distance);
        }
    }
    Ok(minimum)
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
    InvalidMagnitudeBin(u8),
    NonFinitePhase,
    PhaseOutOfRange,
    SourceLegRequired(ProbeClass),
    SourceLegForbidden(ProbeClass),
    DescendingNeuronRequired,
    DescendingNeuronForbidden,
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
    ArithmeticOverflow,
    MissingSourceBindingField(&'static str),
    InvalidSourceDigest(&'static str),
}

impl fmt::Display for PhaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLeg(value) => write!(f, "invalid leg ID {value}; expected 0..6"),
            Self::InvalidPhaseBin(value) => write!(f, "invalid phase bin {value}; expected 0..8"),
            Self::InvalidMagnitudeBin(value) => write!(f, "invalid magnitude bin {value}; expected 0..16"),
            Self::NonFinitePhase => write!(f, "phase input is non-finite"),
            Self::PhaseOutOfRange => write!(f, "unwrapped phase is outside the representable microturn range"),
            Self::SourceLegRequired(class) => write!(f, "probe class {class:?} requires a source leg"),
            Self::SourceLegForbidden(class) => write!(f, "probe class {class:?} is a global input and forbids a source leg"),
            Self::DescendingNeuronRequired => write!(f, "descending command probe requires an exact DNg100/DNg97/DNb08 identity"),
            Self::DescendingNeuronForbidden => write!(f, "non-descending probe cannot carry a descending-neuron identity"),
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
            Self::ArithmeticOverflow => write!(f, "phase-response separation overflowed its exact integer range"),
            Self::MissingSourceBindingField(field) => write!(f, "source binding field {field} is empty"),
            Self::InvalidSourceDigest(field) => write!(f, "source binding digest {field} is not a lowercase SHA-256 hex string"),
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
            descending_neuron: None,
            phase_bin,
            magnitude_bin: 0,
            delay_index,
        }
    }

    fn zeros() -> [Phase; LEG_COUNT] {
        [Phase::from_microturns(0); LEG_COUNT]
    }

    fn test_source_binding() -> SourceBinding {
        SourceBinding {
            dataset_id: "synthetic-six-leg-phase-response".to_string(),
            dataset_revision_sha256: "a".repeat(64),
            phase_estimator_sha256: "b".repeat(64),
            axis_identity_receipt_sha256: "c".repeat(64),
            raw_trajectory_sha256: "d".repeat(64),
            perturbation_receipt_sha256: "e".repeat(64),
            context_definition_sha256: "f".repeat(64),
            animal_unit_token: "synthetic-animal-0001".to_string(),
            session_id: "synthetic-session-0001".to_string(),
            trial_unit_token: "synthetic-trial-0001".to_string(),
        }
    }

    fn zero_samples() -> [PhaseSample; LEG_COUNT] {
        [PhaseSample::from_unwrapped_microturns(0); LEG_COUNT]
    }

    #[test]
    fn radians_ingestion_is_checked_and_preserves_cycle_winding() {
        let one_cycle = PhaseSample::from_unwrapped_radians(std::f64::consts::TAU).unwrap();
        assert_eq!(one_cycle.cycle_index(), 1);
        assert_eq!(one_cycle.fractional_phase().microturns(), 0);

        let before = PhaseSample::from_unwrapped_radians(0.99 * std::f64::consts::TAU).unwrap();
        let after = PhaseSample::from_unwrapped_radians(1.01 * std::f64::consts::TAU).unwrap();
        let fractional = Phase::difference(after.fractional_phase(), before.fractional_phase());
        assert_eq!(fractional.microturns(), 20_000);
        assert_eq!(cycle_slip_delta(after, before, fractional), 0);

        assert_eq!(
            PhaseSample::from_unwrapped_radians(f64::NAN),
            Err(PhaseError::NonFinitePhase)
        );
        assert_eq!(
            PhaseSample::from_unwrapped_radians(f64::INFINITY),
            Err(PhaseError::NonFinitePhase)
        );
        assert_eq!(
            PhaseSample::from_unwrapped_radians(1.0e300),
            Err(PhaseError::PhaseOutOfRange)
        );
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
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Measured { control, perturbed },
            }],
        )
        .unwrap();

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
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Measured { control, perturbed },
            }],
        )
        .unwrap();
        assert_eq!(
            surface
                .phase_reset(&probe)
                .unwrap()
                .off_diagonal_l1_microturns(),
            0
        );
    }

    #[test]
    fn phase_response_surface_requires_valid_source_binding() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let row = TransferRow {
            key: probe,
            outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed),
        };
        let mut bad_digest = test_source_binding();
        bad_digest.raw_trajectory_sha256 = "not-a-digest".to_string();
        assert_eq!(
            PhaseResponseSurface::new(bad_digest, vec![row.clone()]),
            Err(PhaseError::InvalidSourceDigest("raw_trajectory_sha256"))
        );

        let mut missing_trial = test_source_binding();
        missing_trial.trial_unit_token.clear();
        assert_eq!(
            PhaseResponseSurface::new(missing_trial, vec![row.clone()]),
            Err(PhaseError::MissingSourceBindingField("trial_unit_token"))
        );

        let surface = PhaseResponseSurface::new(test_source_binding(), vec![row]).unwrap();
        assert_eq!(
            surface.source_binding().dataset_id,
            "synthetic-six-leg-phase-response"
        );
    }

    #[test]
    fn unavailable_measurements_are_never_silently_zero_filled() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Unavailable(MeasurementStatus::NoResponse),
            }],
        )
        .unwrap();
        assert_eq!(
            surface.phase_reset(&probe),
            Err(PhaseError::Unavailable(MeasurementStatus::NoResponse))
        );
    }

    #[test]
    fn source_identity_is_required_by_leg_local_probe_classes() {
        let invalid = ProbeKey {
            context: Context::BallContact,
            class: ProbeClass::LocalPhaseKick,
            source_leg: None,
            descending_neuron: None,
            phase_bin: 0,
            magnitude_bin: 0,
            delay_index: 0,
        };
        assert_eq!(
            invalid.validate(),
            Err(PhaseError::SourceLegRequired(ProbeClass::LocalPhaseKick))
        );

        let invalid_global = ProbeKey {
            context: Context::AirStepping,
            class: ProbeClass::DescendingCommand,
            source_leg: Some(LegId::new(0).unwrap()),
            descending_neuron: Some(DescendingNeuron::DNg100),
            phase_bin: 0,
            magnitude_bin: 0,
            delay_index: 0,
        };
        assert_eq!(
            invalid_global.validate(),
            Err(PhaseError::SourceLegForbidden(
                ProbeClass::DescendingCommand
            ))
        );
    }

    #[test]
    fn perturbation_magnitude_is_a_distinct_design_coordinate() {
        let lower = key(Context::BallContact, 0, 2, 1);
        let stronger = ProbeKey {
            magnitude_bin: 1,
            ..lower
        };
        assert!(lower.validate().is_ok());
        assert!(stronger.validate().is_ok());
        assert_ne!(lower, stronger);

        let invalid = ProbeKey {
            magnitude_bin: MAX_MAGNITUDE_BINS,
            ..lower
        };
        assert_eq!(
            invalid.validate(),
            Err(PhaseError::InvalidMagnitudeBin(MAX_MAGNITUDE_BINS))
        );
    }

    #[test]
    fn descending_command_sources_remain_distinct() {
        let dng100 = ProbeKey {
            context: Context::BallContact,
            class: ProbeClass::DescendingCommand,
            source_leg: None,
            descending_neuron: Some(DescendingNeuron::DNg100),
            phase_bin: 2,
            magnitude_bin: 0,
            delay_index: 1,
        };
        let dng97 = ProbeKey {
            descending_neuron: Some(DescendingNeuron::DNg97),
            ..dng100
        };
        let dnb08 = ProbeKey {
            descending_neuron: Some(DescendingNeuron::DNb08),
            ..dng100
        };
        assert!(dng100.validate().is_ok());
        assert!(dng97.validate().is_ok());
        assert!(dnb08.validate().is_ok());
        assert_ne!(dng100, dng97);
        assert_ne!(dng100, dnb08);
    }

    #[test]
    fn descending_command_without_neuron_identity_is_rejected() {
        let invalid = ProbeKey {
            context: Context::BallContact,
            class: ProbeClass::DescendingCommand,
            source_leg: None,
            descending_neuron: None,
            phase_bin: 0,
            magnitude_bin: 0,
            delay_index: 0,
        };
        assert_eq!(
            invalid.validate(),
            Err(PhaseError::DescendingNeuronRequired)
        );
    }

    #[test]
    fn phase_grid_rejects_duplicate_and_missing_rows() {
        let first = key(Context::BallContact, 0, 0, 1);
        let second = key(Context::BallContact, 0, 1, 1);
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![
                TransferRow {
                    key: first,
                    outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed),
                },
                TransferRow {
                    key: second,
                    outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed),
                },
            ],
        )
        .unwrap();
        assert_eq!(
            surface.validate_complete(&[first]),
            Err(PhaseError::GridMismatch {
                missing: 0,
                unexpected: 1
            })
        );
        assert!(matches!(
            PhaseResponseSurface::new(
                test_source_binding(),
                vec![
                    TransferRow {
                        key: first,
                        outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed)
                    },
                    TransferRow {
                        key: first,
                        outcome: RowOutcome::Unavailable(MeasurementStatus::NotAssessed)
                    },
                ]
            ),
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
                ProbePrediction {
                    key: phase_zero,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
                ProbePrediction {
                    key: phase_half,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
            ],
        };
        let model_b = ModelSignature {
            model_id: "sensory".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![
                ProbePrediction {
                    key: phase_zero,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
                ProbePrediction {
                    key: phase_half,
                    by_leg: separated,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
            ],
        };
        let separation = minimum_pairwise_probe_separation(&[model_a, model_b]).unwrap();
        assert_eq!(separation[&phase_zero], 0);
        assert_eq!(separation[&phase_half], 40_000);
    }

    #[test]
    fn individually_blind_phase_bins_can_form_a_discriminating_portfolio() {
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
                ProbePrediction {
                    key: phase_zero,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
                ProbePrediction {
                    key: phase_half,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
            ],
        };
        let model_b = ModelSignature {
            model_id: "sensory".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![
                ProbePrediction {
                    key: phase_zero,
                    by_leg: common,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
                ProbePrediction {
                    key: phase_half,
                    by_leg: separated,
                    cycle_slips_by_leg: [0; LEG_COUNT],
                },
            ],
        };
        assert_eq!(
            minimum_campaign_separation(&[model_a.clone(), model_b.clone()]).unwrap(),
            0
        );
        assert_eq!(
            minimum_joint_campaign_phase_separation(&[model_a, model_b]).unwrap(),
            40_000
        );
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
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: apparent_transfer,
                cycle_slips_by_leg: [0; LEG_COUNT],
            }],
        };
        let common_drive = ModelSignature {
            model_id: "latent-shared-input".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 920,
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: apparent_transfer,
                cycle_slips_by_leg: [0; LEG_COUNT],
            }],
        };
        assert_eq!(
            minimum_campaign_separation(&[central_path, common_drive]).unwrap(),
            0
        );
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
            let surface = PhaseResponseSurface::new(
                test_source_binding(),
                vec![TransferRow {
                    key: probe,
                    outcome: RowOutcome::Unavailable(status),
                }],
            )
            .unwrap();
            assert_eq!(
                surface.phase_reset(&probe),
                Err(PhaseError::Unavailable(status))
            );
        }
    }

    #[test]
    fn raw_phase_reset_cycle_slip_summary_rejects_overflow() {
        let reset = PhaseReset {
            source_leg: None,
            by_leg: zeros(),
            cycle_slips_by_leg: [i64::MIN; LEG_COUNT],
        };
        assert_eq!(
            reset.off_diagonal_cycle_slips_l1(),
            Err(PhaseError::ArithmeticOverflow)
        );
    }

    #[test]
    fn cycle_slip_distance_overflow_is_rejected() {
        let probe = key(Context::BallContact, 0, 0, 1);
        let left = ModelSignature {
            model_id: "left-extreme".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: zeros(),
                cycle_slips_by_leg: [i64::MIN, i64::MIN, 0, 0, 0, 0],
            }],
        };
        let right = ModelSignature {
            model_id: "right-extreme".to_string(),
            steady_state_frequency_millihz: 11_000,
            steady_state_plv_milli: 900,
            predictions: vec![ProbePrediction {
                key: probe,
                by_leg: zeros(),
                cycle_slips_by_leg: [i64::MAX, i64::MAX, 0, 0, 0, 0],
            }],
        };
        assert_eq!(
            minimum_pairwise_cycle_slip_separation(&[left.clone(), right.clone()]),
            Err(PhaseError::ArithmeticOverflow)
        );
        assert_eq!(
            minimum_joint_campaign_cycle_slip_separation(&[left, right]),
            Err(PhaseError::ArithmeticOverflow)
        );
    }

    #[test]
    fn full_cycle_slip_is_not_aliased_to_zero_phase() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let control = zero_samples();
        let mut perturbed = zero_samples();
        perturbed[0] = PhaseSample::from_unwrapped_microturns(1_000_000);
        perturbed[2] = PhaseSample::from_unwrapped_microturns(1_100_000);
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Measured { control, perturbed },
            }],
        )
        .unwrap();

        let response = surface.phase_reset(&probe).unwrap();
        assert_eq!(response.by_leg[0].microturns(), 0);
        assert_eq!(response.cycle_slips_by_leg[0], 1);
        assert_eq!(response.by_leg[2].microturns(), 100_000);
        assert_eq!(response.cycle_slips_by_leg[2], 1);
        assert_eq!(response.off_diagonal_cycle_slips_l1().unwrap(), 1);
    }

    #[test]
    fn fractional_branch_crossing_is_not_a_full_cycle_slip() {
        let probe = key(Context::BallContact, 0, 1, 1);
        let mut control = zero_samples();
        let mut perturbed = zero_samples();
        control[0] = PhaseSample::from_unwrapped_microturns(990_000);
        perturbed[0] = PhaseSample::from_unwrapped_microturns(1_010_000);
        let surface = PhaseResponseSurface::new(
            test_source_binding(),
            vec![TransferRow {
                key: probe,
                outcome: RowOutcome::Measured { control, perturbed },
            }],
        )
        .unwrap();

        let response = surface.phase_reset(&probe).unwrap();
        assert_eq!(response.by_leg[0].microturns(), 20_000);
        assert_eq!(response.cycle_slips_by_leg[0], 0);
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
