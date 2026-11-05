//! Dependency-free, auditable storage-feasibility reference kernel.
//!
//! This is a synthetic engineering oracle for physical and arithmetic invariants.
//! It is not a SAWEM market-clearing implementation, a legal interpretation of
//! the Market Code, a dispatch controller, or evidence of market eligibility.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatingMode {
    Standstill,
    Charging,
    Discharging,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReserveProduct {
    Regulating,
    Instantaneous,
    TenMinute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReserveDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReserveCommitment {
    pub product: ReserveProduct,
    pub direction: ReserveDirection,
    pub mw: f64,
    /// Asset response evidence measured from the scheduled operating point.
    pub start_response_seconds: Option<f64>,
    pub full_activation_seconds: Option<f64>,
    /// Sustained activation duration required by the contract / system profile.
    /// SAWEM Rev 2.3 does not define this for every reserve product.
    pub sustain_duration_seconds: Option<f64>,
    /// Independently demonstrated asset sustain capability, not a nameplate claim.
    pub proven_sustain_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketReserveRequirement {
    pub maximum_start_response_seconds: Option<f64>,
    pub maximum_full_activation_seconds: Option<f64>,
    pub minimum_sustain_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveQualification {
    Qualified,
    Infeasible,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReserveAssessment {
    pub qualification: ReserveQualification,
    pub committed_up_mw: f64,
    pub committed_down_mw: f64,
    pub available_up_mw: f64,
    pub available_down_mw: f64,
    pub indeterminate_reasons: Vec<&'static str>,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionTimes {
    pub standstill_to_charging_seconds: f64,
    pub standstill_to_discharging_seconds: f64,
    pub charging_to_standstill_seconds: f64,
    pub discharging_to_standstill_seconds: f64,
    pub charging_to_discharging_seconds: f64,
    pub discharging_to_charging_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssetLimits {
    pub max_charge_mw: f64,
    pub min_stable_charge_mw: f64,
    pub max_discharge_mw: f64,
    pub min_energy_mwh: f64,
    pub max_energy_mwh: f64,
    /// Symmetric efficiency assumption: charge and discharge efficiencies are
    /// each sqrt(round_trip_efficiency). Replace with separate measured values
    /// when the source contract provides them.
    pub round_trip_efficiency: f64,
    pub transitions: TransitionTimes,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DispatchStep {
    pub duration_hours: f64,
    pub charge_mw: f64,
    pub discharge_mw: f64,
    /// Declared downtime at the start of this interval for a mode transition.
    pub transition_seconds_before_action: f64,
    pub reserves: &'static [ReserveCommitment],
    pub network_export_limit_mw: f64,
    pub network_import_limit_mw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    SimultaneousChargeAndDischarge,
    ChargePowerExceeded,
    DischargePowerExceeded,
    BelowMinimumStableCharging,
    TransitionTimeInsufficient,
    TransitionTimeExceedsInterval,
    ExportLimitExceeded,
    ImportLimitExceeded,
    ReservePowerHeadroomExceeded,
    ReserveEnergyHeadroomExceeded,
    ReserveTargetBelowMinimumStableCharge,
    ReserveResponseDeadlineExceeded,
    ReserveSustainInsufficient,
    ReserveTransitionEvidenceContradiction,
    ReserveTransitionDeadlineExceeded,
    UnsupportedReserveProductDirection,
    InvalidReserveValue,
    DuplicateReserveProductDirection,
    EnergyBelowMinimum,
    EnergyAboveMaximum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub index: usize,
    pub requested_mode: OperatingMode,
    pub achieved_mode: OperatingMode,
    pub energy_before_mwh: f64,
    /// Feasible state carried into the next interval. Hard-invalid actions do not commit.
    pub energy_after_mwh: f64,
    /// Candidate state before hard-constraint commit/rejection.
    pub candidate_energy_after_mwh: f64,
    pub net_export_mw: f64,
    pub transition_seconds_used: f64,
    pub reserve_up_mw: f64,
    pub reserve_down_mw: f64,
    pub reserve_assessment: ReserveAssessment,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchResult {
    pub feasible: bool,
    pub initial_energy_mwh: f64,
    pub final_energy_mwh: f64,
    pub steps: Vec<StepResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    NonFiniteValue(&'static str),
    NegativeValue(&'static str),
    InvalidEnergyBounds,
    InvalidChargingBounds,
    InvalidEfficiency,
    InvalidInitialEnergy,
    InvalidSevenDayDeclarationCount,
    IntegerOverflow,
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue(field) => write!(f, "{field} must be finite"),
            Self::NegativeValue(field) => write!(f, "{field} must be nonnegative"),
            Self::InvalidEnergyBounds => write!(f, "minimum energy must be below maximum energy"),
            Self::InvalidChargingBounds => write!(
                f,
                "minimum stable charge must not exceed maximum charge power"
            ),
            Self::InvalidEfficiency => write!(f, "round-trip efficiency must be in (0, 1]"),
            Self::InvalidInitialEnergy => write!(f, "initial energy is outside asset limits"),
            Self::InvalidSevenDayDeclarationCount => {
                write!(f, "exactly seven daily energy declarations are required")
            }
            Self::IntegerOverflow => write!(f, "integer arithmetic overflow"),
        }
    }
}

impl Error for InputError {}

impl AssetLimits {
    pub fn validate(self) -> Result<(), InputError> {
        let positive = [
            ("max_charge_mw", self.max_charge_mw),
            ("max_discharge_mw", self.max_discharge_mw),
            ("max_energy_mwh", self.max_energy_mwh),
        ];
        for (field, value) in positive {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
            if value <= 0.0 {
                return Err(InputError::NegativeValue(field));
            }
        }

        let nonnegative = [
            ("min_stable_charge_mw", self.min_stable_charge_mw),
            ("min_energy_mwh", self.min_energy_mwh),
            (
                "standstill_to_charging_seconds",
                self.transitions.standstill_to_charging_seconds,
            ),
            (
                "standstill_to_discharging_seconds",
                self.transitions.standstill_to_discharging_seconds,
            ),
            (
                "charging_to_standstill_seconds",
                self.transitions.charging_to_standstill_seconds,
            ),
            (
                "discharging_to_standstill_seconds",
                self.transitions.discharging_to_standstill_seconds,
            ),
            (
                "charging_to_discharging_seconds",
                self.transitions.charging_to_discharging_seconds,
            ),
            (
                "discharging_to_charging_seconds",
                self.transitions.discharging_to_charging_seconds,
            ),
        ];
        for (field, value) in nonnegative {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
            if value < 0.0 {
                return Err(InputError::NegativeValue(field));
            }
        }

        if self.min_stable_charge_mw > self.max_charge_mw + EPSILON {
            return Err(InputError::InvalidChargingBounds);
        }
        if self.min_energy_mwh < 0.0 || self.min_energy_mwh >= self.max_energy_mwh {
            return Err(InputError::InvalidEnergyBounds);
        }
        if !self.round_trip_efficiency.is_finite()
            || self.round_trip_efficiency <= 0.0
            || self.round_trip_efficiency > 1.0
        {
            return Err(InputError::InvalidEfficiency);
        }
        Ok(())
    }
}

fn mode_for(charge_mw: f64, discharge_mw: f64) -> OperatingMode {
    if charge_mw > EPSILON {
        OperatingMode::Charging
    } else if discharge_mw > EPSILON {
        OperatingMode::Discharging
    } else {
        OperatingMode::Standstill
    }
}

fn required_transition_seconds(
    previous: OperatingMode,
    next: OperatingMode,
    times: TransitionTimes,
) -> f64 {
    match (previous, next) {
        (OperatingMode::Standstill, OperatingMode::Charging) => {
            times.standstill_to_charging_seconds
        }
        (OperatingMode::Standstill, OperatingMode::Discharging) => {
            times.standstill_to_discharging_seconds
        }
        (OperatingMode::Charging, OperatingMode::Standstill) => {
            times.charging_to_standstill_seconds
        }
        (OperatingMode::Discharging, OperatingMode::Standstill) => {
            times.discharging_to_standstill_seconds
        }
        (OperatingMode::Charging, OperatingMode::Discharging) => {
            times.charging_to_discharging_seconds
        }
        (OperatingMode::Discharging, OperatingMode::Charging) => {
            times.discharging_to_charging_seconds
        }
        _ => 0.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceIncrement {
    Zero,
    One,
    Two,
    Three,
    EmergencyLevelOne,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneratingPriceCurve {
    pub minimum_stable_generation_mw: f64,
    pub elbow_one_mw: f64,
    pub elbow_two_mw: f64,
    pub maximum_continuous_rating_mw: f64,
    pub emergency_level_one_mw: f64,
    /// Price values use signed integer minor currency units per MWh.
    pub increment_zero_minor_per_mwh: i64,
    pub increment_one_minor_per_mwh: i64,
    pub increment_two_minor_per_mwh: i64,
    pub increment_three_minor_per_mwh: i64,
    pub emergency_level_one_minor_per_mwh: i64,
}

/// Demand-side Trading Unit consumption price curve from Market Code Rev 2.3 §9.3(1)(d).
///
/// Unlike the generating curve, its volume axis descends from Declared Consumption
/// through three elbow points. This type only resolves the synthetic unit price at
/// a supplied contracted-consumption operating point; it does not establish DSU
/// eligibility, qualifying reserve capacity, or settlement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemandSideConsumptionPriceCurve {
    pub declared_consumption_mw: f64,
    pub first_elbow_mw: f64,
    pub second_elbow_mw: f64,
    pub third_elbow_mw: f64,
    /// Cost increments use the same explicit minor-currency-per-MWh basis as
    /// the generating-price oracle. They are not reserve payments.
    pub increment_one_minor_per_mwh: i64,
    pub increment_two_minor_per_mwh: i64,
    pub increment_three_minor_per_mwh: i64,
}

/// DSU reserve products with explicit qualifying-capacity clauses in Code §§9.8.4(3), 9.8.6(4) and 9.8.7(3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsuUpwardReserveProduct {
    RegulatingUp,
    Instantaneous,
    TenMinute,
}

/// Inputs to the bounded DSU qualifying-capacity arithmetic.
///
/// Values are assumed to have been identity-, authority-, time- and provenance-qualified
/// upstream. This pure kernel does not authenticate the underlying contract or meter data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DsuReserveQualificationInputs {
    pub product: DsuUpwardReserveProduct,
    pub contracted_capacity_mw: Option<f64>,
    pub official_energy_consumption_mw: Option<f64>,
    pub effective_available_capacity_mw: Option<f64>,
    pub was_utilized_for_reserve_or_balancing: Option<bool>,
    /// Whether a not-utilized unit was nevertheless instructed to deliver energy
    /// different from its contracted amount (Code §9.8.3(4)).
    pub energy_instruction_differed_from_contract: Option<bool>,
    /// SO acceptability assessment. If no criteria were established, §9.8.3(2)
    /// deems the resource acceptable; callers must represent that rule explicitly as true.
    pub acceptable_performance: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsuQualificationError {
    InvalidConsumptionCurve,
    NonFiniteInput(&'static str),
    NegativeInput(&'static str),
}

/// A missing required input or an unimplemented utilization branch is not a zero-capacity result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DsuQualificationResult {
    Determined { qualifying_capacity_mw: f64 },
    Indeterminate { reason: &'static str },
}

/// Reserve-capacity categories with separate payment clauses in Code §§9.8.4–9.8.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveCapacityPaymentProduct {
    RegulatingUp,
    RegulatingDown,
    Instantaneous,
    TenMinute,
}

/// Reserve-category market price supplied by the Market Operator / authoritative
/// settlement data, in minor ZAR per MW per Trading Period. This is deliberately
/// distinct from an SMP or a unit-level lost-opportunity price differential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReserveCategoryMarketPrice {
    pub minor_zar_per_mw_per_trading_period: u64,
}

/// System Operator performance-criteria state for reserve capacity payments.
///
/// Code §9.8.3(2) deems all contracted resources acceptable when no performance
/// criteria have been established. When criteria do exist, the actual SO factor
/// must be provided; unknown/missing values remain indeterminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceAdjustmentEvidence {
    Unknown,
    NoCriteriaEstablished,
    CriteriaEstablished {
        factor_numerator: Option<u32>,
        factor_denominator: Option<u32>,
    },
}

/// Inputs to the formula-only reserve capacity payment calculation.
///
/// Capacities use integer milli-MW. Upstream systems are responsible for binding
/// each value to the exact Trading Unit, Trading Period, contract and source receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReserveCapacityPaymentInputs {
    pub product: ReserveCapacityPaymentProduct,
    pub market_price: Option<ReserveCategoryMarketPrice>,
    pub contracted_capacity_milli_mw: Option<u32>,
    pub qualifying_capacity_milli_mw: Option<u32>,
    /// Ten-Minute has no performance-adjustment term in §9.8.7 and requires None.
    /// Other products require explicit criteria state.
    pub performance_adjustment: Option<PerformanceAdjustmentEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveCapacityPaymentError {
    QualifyingCapacityExceedsContracted,
    UnexpectedPerformanceAdjustmentForTenMinute,
    MissingPerformanceAdjustmentForProduct,
    ZeroPerformanceFactorDenominator,
}

/// Exact rational payment amount in minor ZAR per Trading Period.
/// No rounding mode or settlement rounding convention is inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactReserveCapacityPayment {
    pub numerator_minor_zar: u128,
    pub denominator: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveCapacityPaymentResult {
    Determined(ExactReserveCapacityPayment),
    Indeterminate { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservePriceError {
    NonFiniteOperatingPoint,
    ContractedPointOutsideCurve,
    InvalidPriceCurve,
    InvalidConsumptionPriceCurve,
    NoReducibleConsumptionBelowThirdElbow,
    NegativeIncrementalPrice,
    NonMonotonicIncrementalPrice,
    AmbiguousCoincidentElbows,
    UnsupportedProductDirection,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReserveOpportunityPrice {
    pub selected_increment: PriceIncrement,
    /// Difference in the same minor-currency-per-MWh price basis as SMP and IP.
    /// This is not a cash payment and is not multiplied by reserve MW here.
    pub price_difference_minor_per_mwh: i64,
}

impl GeneratingPriceCurve {
    pub fn validate(self) -> Result<(), ReservePriceError> {
        let points = [
            self.minimum_stable_generation_mw,
            self.elbow_one_mw,
            self.elbow_two_mw,
            self.maximum_continuous_rating_mw,
            self.emergency_level_one_mw,
        ];
        if points.iter().any(|value| !value.is_finite()) {
            return Err(ReservePriceError::NonFiniteOperatingPoint);
        }
        if self.minimum_stable_generation_mw < 0.0
            || self.elbow_one_mw < self.minimum_stable_generation_mw
            || self.elbow_two_mw < self.elbow_one_mw
            || self.maximum_continuous_rating_mw < self.elbow_two_mw
            || self.emergency_level_one_mw < self.maximum_continuous_rating_mw
        {
            return Err(ReservePriceError::InvalidPriceCurve);
        }
        if (self.elbow_one_mw - self.elbow_two_mw).abs() <= EPSILON {
            return Err(ReservePriceError::AmbiguousCoincidentElbows);
        }

        let prices = [
            self.increment_zero_minor_per_mwh,
            self.increment_one_minor_per_mwh,
            self.increment_two_minor_per_mwh,
            self.increment_three_minor_per_mwh,
            self.emergency_level_one_minor_per_mwh,
        ];
        if prices.iter().any(|value| *value < 0) {
            return Err(ReservePriceError::NegativeIncrementalPrice);
        }
        if self.increment_two_minor_per_mwh < self.increment_one_minor_per_mwh
            || self.increment_three_minor_per_mwh < self.increment_two_minor_per_mwh
            || self.emergency_level_one_minor_per_mwh < self.increment_three_minor_per_mwh
        {
            return Err(ReservePriceError::NonMonotonicIncrementalPrice);
        }
        Ok(())
    }

    fn price_at(
        self,
        product: ReserveProduct,
        direction: ReserveDirection,
        contracted_operating_point_mw: f64,
    ) -> Result<(PriceIncrement, i64), ReservePriceError> {
        self.validate()?;
        if !contracted_operating_point_mw.is_finite() {
            return Err(ReservePriceError::NonFiniteOperatingPoint);
        }
        if contracted_operating_point_mw < -EPSILON
            || contracted_operating_point_mw > self.emergency_level_one_mw + EPSILON
        {
            return Err(ReservePriceError::ContractedPointOutsideCurve);
        }

        // A single contracted operating point exactly on an elbow is subject to
        // the product-specific selection convention in §§9.6.1–9.6.4.
        let is_regulating_down =
            product == ReserveProduct::Regulating && direction == ReserveDirection::Down;

        if (contracted_operating_point_mw - self.maximum_continuous_rating_mw).abs() <= EPSILON {
            return Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh));
        }
        if (contracted_operating_point_mw - self.elbow_two_mw).abs() <= EPSILON {
            return if is_regulating_down {
                Ok((PriceIncrement::Two, self.increment_two_minor_per_mwh))
            } else {
                Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh))
            };
        }
        if (contracted_operating_point_mw - self.elbow_one_mw).abs() <= EPSILON {
            return if is_regulating_down {
                Ok((PriceIncrement::One, self.increment_one_minor_per_mwh))
            } else {
                Ok((PriceIncrement::Two, self.increment_two_minor_per_mwh))
            };
        }
        if (contracted_operating_point_mw - self.minimum_stable_generation_mw).abs() <= EPSILON {
            return Ok((PriceIncrement::One, self.increment_one_minor_per_mwh));
        }

        if contracted_operating_point_mw < self.minimum_stable_generation_mw {
            return Ok((PriceIncrement::Zero, self.increment_zero_minor_per_mwh));
        }
        if contracted_operating_point_mw < self.elbow_one_mw {
            return Ok((PriceIncrement::One, self.increment_one_minor_per_mwh));
        }
        if contracted_operating_point_mw < self.elbow_two_mw {
            return Ok((PriceIncrement::Two, self.increment_two_minor_per_mwh));
        }
        if contracted_operating_point_mw <= self.maximum_continuous_rating_mw {
            return Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh));
        }
        Ok((
            PriceIncrement::EmergencyLevelOne,
            self.emergency_level_one_minor_per_mwh,
        ))
    }
}

impl DemandSideConsumptionPriceCurve {
    pub fn validate(self) -> Result<(), ReservePriceError> {
        let points = [
            self.declared_consumption_mw,
            self.first_elbow_mw,
            self.second_elbow_mw,
            self.third_elbow_mw,
        ];
        if points.iter().any(|value| !value.is_finite()) {
            return Err(ReservePriceError::NonFiniteOperatingPoint);
        }
        if points.iter().any(|value| *value < 0.0)
            || self.first_elbow_mw > self.declared_consumption_mw
            || self.second_elbow_mw > self.first_elbow_mw
            || self.third_elbow_mw > self.second_elbow_mw
        {
            return Err(ReservePriceError::InvalidConsumptionPriceCurve);
        }

        let prices = [
            self.increment_one_minor_per_mwh,
            self.increment_two_minor_per_mwh,
            self.increment_three_minor_per_mwh,
        ];
        if prices.iter().any(|value| *value < 0) {
            return Err(ReservePriceError::NegativeIncrementalPrice);
        }
        if self.increment_two_minor_per_mwh < self.increment_one_minor_per_mwh
            || self.increment_three_minor_per_mwh < self.increment_two_minor_per_mwh
        {
            return Err(ReservePriceError::NonMonotonicIncrementalPrice);
        }
        Ok(())
    }

    /// Select the price increment for an absolute contracted consumption point.
    /// At an elbow, use the next non-empty lower-consumption block. The third
    /// elbow is the bottom of the declared curve and has no further shed-able block.
    pub fn price_at(
        self,
        contracted_consumption_mw: f64,
    ) -> Result<(PriceIncrement, i64), ReservePriceError> {
        self.validate()?;
        if !contracted_consumption_mw.is_finite() {
            return Err(ReservePriceError::NonFiniteOperatingPoint);
        }
        if contracted_consumption_mw < self.third_elbow_mw - EPSILON
            || contracted_consumption_mw > self.declared_consumption_mw + EPSILON
        {
            return Err(ReservePriceError::ContractedPointOutsideCurve);
        }
        if (contracted_consumption_mw - self.third_elbow_mw).abs() <= EPSILON {
            return Err(ReservePriceError::NoReducibleConsumptionBelowThirdElbow);
        }

        if (contracted_consumption_mw - self.first_elbow_mw).abs() <= EPSILON {
            if self.first_elbow_mw - self.second_elbow_mw > EPSILON {
                return Ok((PriceIncrement::Two, self.increment_two_minor_per_mwh));
            }
            if self.second_elbow_mw - self.third_elbow_mw > EPSILON {
                return Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh));
            }
            return Err(ReservePriceError::NoReducibleConsumptionBelowThirdElbow);
        }
        if (contracted_consumption_mw - self.second_elbow_mw).abs() <= EPSILON {
            if self.second_elbow_mw - self.third_elbow_mw > EPSILON {
                return Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh));
            }
            return Err(ReservePriceError::NoReducibleConsumptionBelowThirdElbow);
        }

        if contracted_consumption_mw > self.first_elbow_mw {
            return Ok((PriceIncrement::One, self.increment_one_minor_per_mwh));
        }
        if contracted_consumption_mw > self.second_elbow_mw {
            return Ok((PriceIncrement::Two, self.increment_two_minor_per_mwh));
        }
        if contracted_consumption_mw > self.third_elbow_mw {
            return Ok((PriceIncrement::Three, self.increment_three_minor_per_mwh));
        }
        Err(ReservePriceError::NoReducibleConsumptionBelowThirdElbow)
    }
}

/// Calculate the §9.6 Trading Unit opportunity-price differential for a
/// generating price curve. This does not calculate contracted MW, qualifying
/// capacity, reserve-capacity payments, performance adjustment or settlement.
pub fn reserve_opportunity_price(
    product: ReserveProduct,
    direction: ReserveDirection,
    system_marginal_price_minor_per_mwh: i64,
    contracted_operating_point_mw: f64,
    curve: GeneratingPriceCurve,
) -> Result<ReserveOpportunityPrice, ReservePriceError> {
    sawem_rev_2_3_reserve_requirement(product, direction)
        .map_err(|_| ReservePriceError::UnsupportedProductDirection)?;
    let (selected_increment, unit_incremental_price) =
        curve.price_at(product, direction, contracted_operating_point_mw)?;
    let difference = match (product, direction) {
        (ReserveProduct::Regulating, ReserveDirection::Down) => {
            unit_incremental_price.checked_sub(system_marginal_price_minor_per_mwh)
        }
        (ReserveProduct::Regulating, ReserveDirection::Up)
        | (ReserveProduct::Instantaneous, ReserveDirection::Up)
        | (ReserveProduct::TenMinute, ReserveDirection::Up) => {
            system_marginal_price_minor_per_mwh.checked_sub(unit_incremental_price)
        }
        _ => return Err(ReservePriceError::UnsupportedProductDirection),
    }
    .ok_or(ReservePriceError::ArithmeticOverflow)?;

    Ok(ReserveOpportunityPrice {
        selected_increment,
        price_difference_minor_per_mwh: difference.max(0),
    })
}

/// Calculate an upward-reserve opportunity-price differential for a Demand Side Unit
/// using its §9.3(1)(d) consumption curve. This is a price-curve component only:
/// eligibility, contracted MW, qualifying capacity, performance adjustment and
/// settlement are separate evidence-bound calculations.
pub fn dsu_upward_reserve_opportunity_price(
    product: ReserveProduct,
    system_marginal_price_minor_per_mwh: i64,
    contracted_consumption_mw: f64,
    curve: DemandSideConsumptionPriceCurve,
) -> Result<ReserveOpportunityPrice, ReservePriceError> {
    let (selected_increment, unit_incremental_price) = curve.price_at(contracted_consumption_mw)?;
    let difference = system_marginal_price_minor_per_mwh
        .checked_sub(unit_incremental_price)
        .ok_or(ReservePriceError::ArithmeticOverflow)?;

    // This helper is upward-only. The Code defines DSU qualifying-capacity
    // clauses for these product families, but eligibility and qualifying MW
    // remain separate from this price-curve calculation.
    sawem_rev_2_3_reserve_requirement(product, ReserveDirection::Up)
        .map_err(|_| ReservePriceError::UnsupportedProductDirection)?;

    Ok(ReserveOpportunityPrice {
        selected_increment,
        price_difference_minor_per_mwh: difference.max(0),
    })
}

/// Calculate DSU qualifying-for-payment capacity for the explicit §9.8.4(3),
/// §9.8.6(4) and §9.8.7(3) arithmetic branches.
///
/// Regulating Up / Instantaneous use official metered consumption minus the third
/// consumption elbow, and are only determined here when the unit was not used for
/// reserve/balancing. Ten-Minute uses effective available capacity minus that elbow.
/// Payment prices, performance factors, source authenticity and utilized-service
/// branches are intentionally outside this function.
pub fn qualifying_dsu_capacity_for_payment(
    inputs: DsuReserveQualificationInputs,
    curve: DemandSideConsumptionPriceCurve,
) -> Result<DsuQualificationResult, DsuQualificationError> {
    curve
        .validate()
        .map_err(|_| DsuQualificationError::InvalidConsumptionCurve)?;

    let Some(contracted_mw) = inputs.contracted_capacity_mw else {
        return Ok(DsuQualificationResult::Indeterminate {
            reason: "missing contracted reserve capacity",
        });
    };
    if !contracted_mw.is_finite() {
        return Err(DsuQualificationError::NonFiniteInput(
            "contracted_capacity_mw",
        ));
    }
    if contracted_mw < 0.0 {
        return Err(DsuQualificationError::NegativeInput(
            "contracted_capacity_mw",
        ));
    }
    if contracted_mw == 0.0 {
        return Ok(DsuQualificationResult::Determined {
            qualifying_capacity_mw: 0.0,
        });
    }

    let Some(effective_available_mw) = inputs.effective_available_capacity_mw else {
        return Ok(DsuQualificationResult::Indeterminate {
            reason: "missing effective available capacity",
        });
    };
    if !effective_available_mw.is_finite() {
        return Err(DsuQualificationError::NonFiniteInput(
            "effective_available_capacity_mw",
        ));
    }
    if effective_available_mw < 0.0 {
        return Err(DsuQualificationError::NegativeInput(
            "effective_available_capacity_mw",
        ));
    }

    match inputs.product {
        DsuUpwardReserveProduct::RegulatingUp | DsuUpwardReserveProduct::Instantaneous => {
            match inputs.was_utilized_for_reserve_or_balancing {
                None => {
                    return Ok(DsuQualificationResult::Indeterminate {
                        reason: "missing reserve/balancing utilization state",
                    });
                }
                Some(true) => {
                    if inputs.acceptable_performance != Some(true) {
                        return Ok(DsuQualificationResult::Indeterminate {
                            reason: "utilized reserve lacks acceptable-performance evidence",
                        });
                    }
                    // §9.8.3(3), subject to the effective-availability ceiling in (6).
                    return Ok(DsuQualificationResult::Determined {
                        qualifying_capacity_mw: contracted_mw.min(effective_available_mw),
                    });
                }
                Some(false) => {}
            }

            match inputs.energy_instruction_differed_from_contract {
                None => {
                    return Ok(DsuQualificationResult::Indeterminate {
                        reason: "missing not-utilized instruction comparison",
                    });
                }
                Some(true) => {
                    if inputs.acceptable_performance != Some(true) {
                        return Ok(DsuQualificationResult::Indeterminate {
                            reason: "changed-instruction branch lacks acceptable-performance evidence",
                        });
                    }
                    // §9.8.3(4), subject to the effective-availability ceiling in (6).
                    return Ok(DsuQualificationResult::Determined {
                        qualifying_capacity_mw: contracted_mw.min(effective_available_mw),
                    });
                }
                Some(false) => {
                    if inputs.acceptable_performance != Some(true) {
                        return Ok(DsuQualificationResult::Indeterminate {
                            reason: "non-utilized unchanged-instruction branch lacks SO capability assessment",
                        });
                    }
                }
            }

            let Some(consumption_mw) = inputs.official_energy_consumption_mw else {
                return Ok(DsuQualificationResult::Indeterminate {
                    reason: "missing official energy consumption",
                });
            };
            if !consumption_mw.is_finite() {
                return Err(DsuQualificationError::NonFiniteInput(
                    "official_energy_consumption_mw",
                ));
            }
            if consumption_mw < 0.0 {
                return Err(DsuQualificationError::NegativeInput(
                    "official_energy_consumption_mw",
                ));
            }
            if consumption_mw > curve.declared_consumption_mw {
                return Ok(DsuQualificationResult::Indeterminate {
                    reason: "official consumption exceeds declared consumption",
                });
            }

            // §§9.8.4(3) / 9.8.6(4), with §9.8.3(6)'s availability ceiling.
            let reducible_mw = (consumption_mw - curve.third_elbow_mw).max(0.0);
            Ok(DsuQualificationResult::Determined {
                qualifying_capacity_mw: contracted_mw.min(reducible_mw).min(effective_available_mw),
            })
        }
        DsuUpwardReserveProduct::TenMinute => {
            // §9.8.7(3): DSU Ten-Minute uses Effective Available Capacity less
            // the third consumption elbow; the clause does not condition this on
            // reserve utilization or apply a performance-adjustment factor.
            let reducible_mw = (effective_available_mw - curve.third_elbow_mw).max(0.0);
            Ok(DsuQualificationResult::Determined {
                qualifying_capacity_mw: contracted_mw.min(reducible_mw),
            })
        }
    }
}

/// Calculate the product-specific reserve-capacity payment formula in Code §§9.8.4–9.8.7.
///
/// The input market price is the reserve-category price (Rand/MW/Trading Period)
/// from the Market Operator, not the energy SMP and not a unit's lost-opportunity
/// differential. Qualifying capacity must already have been established upstream.
/// Returns exact rational minor-ZAR per Trading Period and does not round.
pub fn calculate_reserve_capacity_payment(
    inputs: ReserveCapacityPaymentInputs,
) -> Result<ReserveCapacityPaymentResult, ReserveCapacityPaymentError> {
    let Some(contracted_milli_mw) = inputs.contracted_capacity_milli_mw else {
        return Ok(ReserveCapacityPaymentResult::Indeterminate {
            reason: "missing contracted reserve capacity",
        });
    };

    let Some(qualifying_milli_mw) = inputs.qualifying_capacity_milli_mw else {
        if contracted_milli_mw == 0 {
            return Ok(ReserveCapacityPaymentResult::Determined(
                ExactReserveCapacityPayment {
                    numerator_minor_zar: 0,
                    denominator: 1,
                },
            ));
        }
        return Ok(ReserveCapacityPaymentResult::Indeterminate {
            reason: "missing capacity qualifying for payment",
        });
    };

    if qualifying_milli_mw > contracted_milli_mw {
        return Err(ReserveCapacityPaymentError::QualifyingCapacityExceedsContracted);
    }

    // §9.8.3(1): uncontracted capacity cannot receive a reserve-capacity payment.
    if contracted_milli_mw == 0 || qualifying_milli_mw == 0 {
        return Ok(ReserveCapacityPaymentResult::Determined(
            ExactReserveCapacityPayment {
                numerator_minor_zar: 0,
                denominator: 1,
            },
        ));
    }

    let Some(market_price) = inputs.market_price else {
        return Ok(ReserveCapacityPaymentResult::Indeterminate {
            reason: "missing reserve-category market price",
        });
    };

    let (factor_numerator, factor_denominator) = match inputs.product {
        ReserveCapacityPaymentProduct::TenMinute => {
            if inputs.performance_adjustment.is_some() {
                return Err(
                    ReserveCapacityPaymentError::UnexpectedPerformanceAdjustmentForTenMinute,
                );
            }
            (1_u32, 1_u32)
        }
        ReserveCapacityPaymentProduct::RegulatingUp
        | ReserveCapacityPaymentProduct::RegulatingDown
        | ReserveCapacityPaymentProduct::Instantaneous => {
            let Some(performance) = inputs.performance_adjustment else {
                return Ok(ReserveCapacityPaymentResult::Indeterminate {
                    reason: "missing System Operator performance-criteria state",
                });
            };
            match performance {
                PerformanceAdjustmentEvidence::Unknown => {
                    return Ok(ReserveCapacityPaymentResult::Indeterminate {
                        reason: "System Operator performance-criteria state is unknown",
                    });
                }
                PerformanceAdjustmentEvidence::NoCriteriaEstablished => (1, 1),
                PerformanceAdjustmentEvidence::CriteriaEstablished {
                    factor_numerator: Some(numerator),
                    factor_denominator: Some(denominator),
                } => {
                    if denominator == 0 {
                        return Err(ReserveCapacityPaymentError::ZeroPerformanceFactorDenominator);
                    }
                    (numerator, denominator)
                }
                PerformanceAdjustmentEvidence::CriteriaEstablished {
                    factor_numerator: _,
                    factor_denominator: _,
                } => {
                    return Ok(ReserveCapacityPaymentResult::Indeterminate {
                        reason: "published performance criteria lack an authoritative SO factor",
                    });
                }
            }
        }
    };

    // price [minor ZAR / MW / period] * capacity [milli-MW / 1000]
    // * performance factor [numerator / denominator].
    // u64 * u32 * u32 fits u128 without floating-point conversion.
    let numerator_minor_zar = u128::from(market_price.minor_zar_per_mw_per_trading_period)
        * u128::from(qualifying_milli_mw)
        * u128::from(factor_numerator);
    let denominator = 1_000_u128 * u128::from(factor_denominator);

    Ok(ReserveCapacityPaymentResult::Determined(
        ExactReserveCapacityPayment {
            numerator_minor_zar,
            denominator,
        },
    ))
}

/// SAWEM Market Code Rev 2.3 product response requirements.
///
/// The Code specifies no minimum sustain duration for regulating or ten-minute
/// reserve in these definitions. That remains an external operator/contract
/// input; it is deliberately not inferred here.
pub fn sawem_rev_2_3_reserve_requirement(
    product: ReserveProduct,
    direction: ReserveDirection,
) -> Result<MarketReserveRequirement, Violation> {
    match (product, direction) {
        (ReserveProduct::Regulating, ReserveDirection::Up)
        | (ReserveProduct::Regulating, ReserveDirection::Down) => Ok(MarketReserveRequirement {
            maximum_start_response_seconds: Some(10.0),
            maximum_full_activation_seconds: Some(600.0),
            minimum_sustain_seconds: None,
        }),
        (ReserveProduct::Instantaneous, ReserveDirection::Up) => Ok(MarketReserveRequirement {
            maximum_start_response_seconds: None,
            maximum_full_activation_seconds: Some(10.0),
            minimum_sustain_seconds: Some(600.0),
        }),
        (ReserveProduct::TenMinute, ReserveDirection::Up) => Ok(MarketReserveRequirement {
            maximum_start_response_seconds: None,
            maximum_full_activation_seconds: Some(600.0),
            minimum_sustain_seconds: None,
        }),
        _ => Err(Violation::UnsupportedReserveProductDirection),
    }
}

fn operating_point_for_net_power(
    net_export_mw: f64,
    min_stable_charge_mw: f64,
) -> Result<(f64, f64), Violation> {
    if net_export_mw < -EPSILON {
        let charge_mw = -net_export_mw;
        if charge_mw + EPSILON < min_stable_charge_mw {
            return Err(Violation::ReserveTargetBelowMinimumStableCharge);
        }
        Ok((charge_mw, 0.0))
    } else {
        Ok((0.0, net_export_mw.max(0.0)))
    }
}

fn validate_reserve_response(
    reserve: ReserveCommitment,
    requirement: MarketReserveRequirement,
    indeterminate_reasons: &mut Vec<&'static str>,
    violations: &mut Vec<Violation>,
) {
    for value in [
        reserve.mw,
        reserve.start_response_seconds.unwrap_or(0.0),
        reserve.full_activation_seconds.unwrap_or(0.0),
        reserve.sustain_duration_seconds.unwrap_or(0.0),
        reserve.proven_sustain_seconds.unwrap_or(0.0),
    ] {
        if !value.is_finite() || value < 0.0 {
            violations.push(Violation::InvalidReserveValue);
            return;
        }
    }
    if reserve.mw <= EPSILON {
        violations.push(Violation::InvalidReserveValue);
        return;
    }

    if let Some(deadline) = requirement.maximum_start_response_seconds {
        match reserve.start_response_seconds {
            Some(actual) if actual <= deadline + EPSILON => {}
            Some(_) => violations.push(Violation::ReserveResponseDeadlineExceeded),
            None => indeterminate_reasons.push("missing_start_response_evidence"),
        }
    }

    if let Some(deadline) = requirement.maximum_full_activation_seconds {
        match reserve.full_activation_seconds {
            Some(actual) if actual <= deadline + EPSILON => {}
            Some(_) => violations.push(Violation::ReserveResponseDeadlineExceeded),
            None => indeterminate_reasons.push("missing_full_activation_evidence"),
        }
    }

    let required_minimum = requirement.minimum_sustain_seconds.unwrap_or(0.0);
    let energy_assessment_duration = match reserve.sustain_duration_seconds {
        Some(duration) => duration.max(required_minimum),
        None if required_minimum > 0.0 => required_minimum,
        None => {
            indeterminate_reasons.push("sustain_duration_not_defined_by_market_profile");
            return;
        }
    };
    if energy_assessment_duration <= EPSILON {
        violations.push(Violation::InvalidReserveValue);
        return;
    }

    match reserve.proven_sustain_seconds {
        Some(proven) if proven + EPSILON >= energy_assessment_duration => {}
        Some(_) => violations.push(Violation::ReserveSustainInsufficient),
        None => indeterminate_reasons.push("missing_asset_sustain_evidence"),
    }
}

fn evaluate_reserve_energy_scenario(
    reserves: &[ReserveCommitment],
    direction: ReserveDirection,
    base_charge_mw: f64,
    base_discharge_mw: f64,
    starting_energy_mwh: f64,
    limits: AssetLimits,
    charge_efficiency: f64,
    discharge_efficiency: f64,
) -> Result<(), Violation> {
    let mut timed_reserves = Vec::new();
    for reserve in reserves.iter().filter(|item| item.direction == direction) {
        let requirement = sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction)?;
        let minimum = requirement.minimum_sustain_seconds.unwrap_or(0.0);
        let duration = match reserve.sustain_duration_seconds {
            Some(value) => value.max(minimum),
            None if minimum > 0.0 => minimum,
            None => return Err(Violation::InvalidReserveValue),
        };
        if reserve
            .proven_sustain_seconds
            .is_none_or(|proven| proven + EPSILON < duration)
        {
            return Err(Violation::ReserveSustainInsufficient);
        }
        timed_reserves.push((reserve.mw, duration));
    }
    if timed_reserves.is_empty() {
        return Ok(());
    }

    let mut boundaries: Vec<f64> = timed_reserves
        .iter()
        .map(|(_, duration)| *duration)
        .collect();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| (*left - *right).abs() <= EPSILON);

    let baseline_net = base_discharge_mw - base_charge_mw;
    let mut energy = starting_energy_mwh;
    let mut previous_boundary = 0.0;

    for boundary in boundaries {
        let active_reserve_mw: f64 = timed_reserves
            .iter()
            .filter(|(_, duration)| *duration > previous_boundary + EPSILON)
            .map(|(mw, _)| *mw)
            .sum();
        let target_net = match direction {
            ReserveDirection::Up => baseline_net + active_reserve_mw,
            ReserveDirection::Down => baseline_net - active_reserve_mw,
        };
        let (target_charge_mw, target_discharge_mw) =
            operating_point_for_net_power(target_net, limits.min_stable_charge_mw)?;

        if target_charge_mw > limits.max_charge_mw + EPSILON
            || target_discharge_mw > limits.max_discharge_mw + EPSILON
        {
            return Err(Violation::ReservePowerHeadroomExceeded);
        }

        // Integrate only the incremental energy effect of reserve activation
        // relative to the scheduled operating point. This correctly treats
        // stopping an existing charge/discharge as reserve response.
        let incremental_energy_rate = (target_charge_mw - base_charge_mw) * charge_efficiency
            - (target_discharge_mw - base_discharge_mw) / discharge_efficiency;
        let elapsed_hours = (boundary - previous_boundary) / 3_600.0;
        energy += incremental_energy_rate * elapsed_hours;

        if energy < limits.min_energy_mwh - EPSILON {
            return Err(Violation::ReserveEnergyHeadroomExceeded);
        }
        if energy > limits.max_energy_mwh + EPSILON {
            return Err(Violation::ReserveEnergyHeadroomExceeded);
        }
        previous_boundary = boundary;
    }
    Ok(())
}

fn assess_reserve_mode_transition(
    baseline_net_mw: f64,
    target_net_mw: f64,
    reserves: &[ReserveCommitment],
    limits: AssetLimits,
    indeterminate_reasons: &mut Vec<&'static str>,
    violations: &mut Vec<Violation>,
) {
    let baseline_mode = mode_for((-baseline_net_mw).max(0.0), baseline_net_mw.max(0.0));
    let (target_charge, target_discharge) =
        match operating_point_for_net_power(target_net_mw, limits.min_stable_charge_mw) {
            Ok(value) => value,
            Err(violation) => {
                violations.push(violation);
                return;
            }
        };
    let target_mode = mode_for(target_charge, target_discharge);
    let required_transition =
        required_transition_seconds(baseline_mode, target_mode, limits.transitions);

    if required_transition <= EPSILON {
        return;
    }

    for reserve in reserves {
        let requirement =
            match sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction) {
                Ok(value) => value,
                Err(_) => continue,
            };
        if let Some(deadline) = requirement.maximum_full_activation_seconds {
            if required_transition > deadline + EPSILON {
                violations.push(Violation::ReserveTransitionDeadlineExceeded);
            }
        }
        match reserve.full_activation_seconds {
            Some(actual) if actual + EPSILON >= required_transition => {}
            Some(_) => violations.push(Violation::ReserveTransitionEvidenceContradiction),
            None => {
                indeterminate_reasons.push("missing_full_activation_evidence_for_mode_transition")
            }
        }
    }
}

fn assess_reserves(
    limits: AssetLimits,
    charge_mw: f64,
    discharge_mw: f64,
    energy_after_schedule_mwh: f64,
    charge_efficiency: f64,
    discharge_efficiency: f64,
    reserves: &[ReserveCommitment],
) -> ReserveAssessment {
    let baseline_net = discharge_mw - charge_mw;
    let available_up_mw = (limits.max_discharge_mw - baseline_net).max(0.0);
    let available_down_mw = (baseline_net + limits.max_charge_mw).max(0.0);
    let mut committed_up_mw = 0.0;
    let mut committed_down_mw = 0.0;
    let mut seen = BTreeSet::new();
    let mut indeterminate_reasons = Vec::new();
    let mut violations = Vec::new();

    for reserve in reserves {
        if !seen.insert((reserve.product, reserve.direction)) {
            violations.push(Violation::DuplicateReserveProductDirection);
            continue;
        }
        let requirement =
            match sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction) {
                Ok(value) => value,
                Err(violation) => {
                    violations.push(violation);
                    continue;
                }
            };
        validate_reserve_response(
            *reserve,
            requirement,
            &mut indeterminate_reasons,
            &mut violations,
        );
        match reserve.direction {
            ReserveDirection::Up => committed_up_mw += reserve.mw,
            ReserveDirection::Down => committed_down_mw += reserve.mw,
        }
    }

    if committed_up_mw > available_up_mw + EPSILON {
        violations.push(Violation::ReservePowerHeadroomExceeded);
    }
    if committed_down_mw > available_down_mw + EPSILON {
        violations.push(Violation::ReservePowerHeadroomExceeded);
    }

    if committed_up_mw > EPSILON {
        let target = baseline_net + committed_up_mw;
        let directional_reserves: Vec<_> = reserves
            .iter()
            .filter(|reserve| reserve.direction == ReserveDirection::Up)
            .copied()
            .collect();
        assess_reserve_mode_transition(
            baseline_net,
            target,
            &directional_reserves,
            limits,
            &mut indeterminate_reasons,
            &mut violations,
        );
    }
    if committed_down_mw > EPSILON {
        let target = baseline_net - committed_down_mw;
        let directional_reserves: Vec<_> = reserves
            .iter()
            .filter(|reserve| reserve.direction == ReserveDirection::Down)
            .copied()
            .collect();
        assess_reserve_mode_transition(
            baseline_net,
            target,
            &directional_reserves,
            limits,
            &mut indeterminate_reasons,
            &mut violations,
        );
    }

    if !reserves.is_empty() {
        if reserves.iter().any(|reserve| {
            let requirement = sawem_rev_2_3_reserve_requirement(reserve.product, reserve.direction);
            requirement.is_err()
                || (reserve.sustain_duration_seconds.is_none()
                    && requirement
                        .ok()
                        .is_some_and(|value| value.minimum_sustain_seconds.is_none()))
        }) {
            indeterminate_reasons.push("reserve_energy_duration_incomplete");
        } else {
            for direction in [ReserveDirection::Up, ReserveDirection::Down] {
                match evaluate_reserve_energy_scenario(
                    reserves,
                    direction,
                    charge_mw,
                    discharge_mw,
                    energy_after_schedule_mwh,
                    limits,
                    charge_efficiency,
                    discharge_efficiency,
                ) {
                    Ok(()) => {}
                    Err(violation) => violations.push(violation),
                }
            }
        }
    }

    violations.sort_by_key(|value| format!("{value:?}"));
    violations.dedup();
    indeterminate_reasons.sort_unstable();
    indeterminate_reasons.dedup();

    let qualification = if !violations.is_empty() {
        ReserveQualification::Infeasible
    } else if !indeterminate_reasons.is_empty() {
        ReserveQualification::Indeterminate
    } else {
        ReserveQualification::Qualified
    };
    ReserveAssessment {
        qualification,
        committed_up_mw,
        committed_down_mw,
        available_up_mw,
        available_down_mw,
        indeterminate_reasons,
        violations,
    }
}

pub fn evaluate_dispatch(
    limits: AssetLimits,
    initial_energy_mwh: f64,
    steps: &[DispatchStep],
) -> Result<DispatchResult, InputError> {
    limits.validate()?;
    if !initial_energy_mwh.is_finite() {
        return Err(InputError::NonFiniteValue("initial_energy_mwh"));
    }
    if initial_energy_mwh < limits.min_energy_mwh - EPSILON
        || initial_energy_mwh > limits.max_energy_mwh + EPSILON
    {
        return Err(InputError::InvalidInitialEnergy);
    }

    let charge_efficiency = limits.round_trip_efficiency.sqrt();
    let discharge_efficiency = charge_efficiency;
    let mut energy = initial_energy_mwh;
    let mut previous_mode = OperatingMode::Standstill;
    let mut results = Vec::with_capacity(steps.len());

    for (index, step) in steps.iter().enumerate() {
        let mut violations = Vec::new();
        let finite_values = [
            ("duration_hours", step.duration_hours),
            ("charge_mw", step.charge_mw),
            ("discharge_mw", step.discharge_mw),
            (
                "transition_seconds_before_action",
                step.transition_seconds_before_action,
            ),
            ("network_export_limit_mw", step.network_export_limit_mw),
            ("network_import_limit_mw", step.network_import_limit_mw),
        ];
        for (field, value) in finite_values {
            if !value.is_finite() {
                return Err(InputError::NonFiniteValue(field));
            }
        }
        if step.duration_hours <= 0.0
            || step.charge_mw < 0.0
            || step.discharge_mw < 0.0
            || step.transition_seconds_before_action < 0.0
            || step.network_export_limit_mw < 0.0
            || step.network_import_limit_mw < 0.0
        {
            return Err(InputError::NegativeValue("dispatch_step"));
        }

        let requested_mode = mode_for(step.charge_mw, step.discharge_mw);
        let simultaneous_charge_and_discharge =
            step.charge_mw > EPSILON && step.discharge_mw > EPSILON;
        if simultaneous_charge_and_discharge {
            violations.push(Violation::SimultaneousChargeAndDischarge);
        }
        let charge_power_valid = step.charge_mw <= limits.max_charge_mw + EPSILON;
        if !charge_power_valid {
            violations.push(Violation::ChargePowerExceeded);
        }
        let stable_charge_valid =
            step.charge_mw <= EPSILON || step.charge_mw + EPSILON >= limits.min_stable_charge_mw;
        if !stable_charge_valid {
            violations.push(Violation::BelowMinimumStableCharging);
        }
        let discharge_power_valid = step.discharge_mw <= limits.max_discharge_mw + EPSILON;
        if !discharge_power_valid {
            violations.push(Violation::DischargePowerExceeded);
        }
        let mode_request_valid = !simultaneous_charge_and_discharge
            && charge_power_valid
            && stable_charge_valid
            && discharge_power_valid;

        let required_transition =
            required_transition_seconds(previous_mode, requested_mode, limits.transitions);
        if step.transition_seconds_before_action + EPSILON < required_transition {
            violations.push(Violation::TransitionTimeInsufficient);
        }
        let interval_seconds = step.duration_hours * 3_600.0;
        if step.transition_seconds_before_action > interval_seconds + EPSILON {
            violations.push(Violation::TransitionTimeExceedsInterval);
        }

        let transition_valid = step.transition_seconds_before_action + EPSILON
            >= required_transition
            && step.transition_seconds_before_action <= interval_seconds + EPSILON;
        let net_export = step.discharge_mw - step.charge_mw;
        let export_valid = net_export <= step.network_export_limit_mw + EPSILON;
        if !export_valid {
            violations.push(Violation::ExportLimitExceeded);
        }
        let import_valid = -net_export <= step.network_import_limit_mw + EPSILON;
        if !import_valid {
            violations.push(Violation::ImportLimitExceeded);
        }
        let network_valid = export_valid && import_valid;

        let action_valid = mode_request_valid && transition_valid && network_valid;
        let active_hours = if action_valid {
            step.duration_hours - step.transition_seconds_before_action / 3_600.0
        } else {
            0.0
        };

        let energy_before = energy;
        let mut candidate_energy = energy_before;
        if action_valid {
            candidate_energy += step.charge_mw * charge_efficiency * active_hours;
            candidate_energy -= step.discharge_mw / discharge_efficiency * active_hours;
        }

        let energy_bounds_valid = candidate_energy >= limits.min_energy_mwh - EPSILON
            && candidate_energy <= limits.max_energy_mwh + EPSILON;
        if candidate_energy < limits.min_energy_mwh - EPSILON {
            violations.push(Violation::EnergyBelowMinimum);
        }
        if candidate_energy > limits.max_energy_mwh + EPSILON {
            violations.push(Violation::EnergyAboveMaximum);
        }

        // Commit the scheduled operating point only when its own hard constraints
        // pass. Reserve feasibility is assessed independently around that committed
        // point and cannot silently alter the base dispatch trace.
        let action_committed = action_valid && energy_bounds_valid;
        if action_committed {
            energy = candidate_energy;
        }
        let achieved_mode = if action_committed {
            requested_mode
        } else {
            previous_mode
        };
        let (reserve_base_charge, reserve_base_discharge) = if action_committed {
            (step.charge_mw, step.discharge_mw)
        } else {
            (0.0, 0.0)
        };
        let reserve_assessment = assess_reserves(
            limits,
            reserve_base_charge,
            reserve_base_discharge,
            energy,
            charge_efficiency,
            discharge_efficiency,
            step.reserves,
        );
        violations.extend(reserve_assessment.violations.iter().copied());

        results.push(StepResult {
            index,
            requested_mode,
            achieved_mode,
            energy_before_mwh: energy_before,
            energy_after_mwh: energy,
            candidate_energy_after_mwh: candidate_energy,
            net_export_mw: net_export,
            transition_seconds_used: step.transition_seconds_before_action,
            reserve_up_mw: reserve_assessment.committed_up_mw,
            reserve_down_mw: reserve_assessment.committed_down_mw,
            reserve_assessment,
            violations,
        });
        previous_mode = achieved_mode;
    }

    let feasible = results.iter().all(|step| {
        step.violations.is_empty()
            && step.reserve_assessment.qualification == ReserveQualification::Qualified
    });
    Ok(DispatchResult {
        feasible,
        initial_energy_mwh,
        final_energy_mwh: energy,
        steps: results,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SevenDayEnergyLimits {
    pub daily_limit_mwh: f64,
    pub next_six_days_limit_mwh: f64,
    pub seven_day_total_limit_mwh: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergyLimitResult {
    pub feasible: bool,
    pub daily_total_mwh: f64,
    pub next_six_days_total_mwh: f64,
    pub seven_day_total_mwh: f64,
    pub violations: Vec<&'static str>,
}

pub fn validate_seven_day_energy_declarations(
    declarations_mwh: &[f64],
    limits: SevenDayEnergyLimits,
) -> Result<EnergyLimitResult, InputError> {
    if declarations_mwh.len() != 7 {
        return Err(InputError::InvalidSevenDayDeclarationCount);
    }
    for value in declarations_mwh {
        if !value.is_finite() {
            return Err(InputError::NonFiniteValue("daily_declaration_mwh"));
        }
        if *value < 0.0 {
            return Err(InputError::NegativeValue("daily_declaration_mwh"));
        }
    }
    let caps = [
        ("daily_limit_mwh", limits.daily_limit_mwh),
        ("next_six_days_limit_mwh", limits.next_six_days_limit_mwh),
        (
            "seven_day_total_limit_mwh",
            limits.seven_day_total_limit_mwh,
        ),
    ];
    for (name, value) in caps {
        if !value.is_finite() {
            return Err(InputError::NonFiniteValue(name));
        }
        if value < 0.0 {
            return Err(InputError::NegativeValue(name));
        }
    }

    let daily_total = declarations_mwh[0];
    let next_six_total: f64 = declarations_mwh[1..].iter().sum();
    let seven_day_total: f64 = declarations_mwh.iter().sum();
    let mut violations = Vec::new();
    if declarations_mwh
        .iter()
        .any(|value| *value > limits.daily_limit_mwh + EPSILON)
    {
        violations.push("daily_limit_exceeded");
    }
    if next_six_total > limits.next_six_days_limit_mwh + EPSILON {
        violations.push("next_six_days_limit_exceeded");
    }
    if seven_day_total > limits.seven_day_total_limit_mwh + EPSILON {
        violations.push("seven_day_total_limit_exceeded");
    }

    Ok(EnergyLimitResult {
        feasible: violations.is_empty(),
        daily_total_mwh: daily_total,
        next_six_days_total_mwh: next_six_total,
        seven_day_total_mwh: seven_day_total,
        violations,
    })
}

pub fn required_credit_cover_minor_units(
    actual_exposure_minor_units: u64,
    undefined_potential_exposure_minor_units: u64,
) -> Result<u64, InputError> {
    actual_exposure_minor_units
        .checked_add(undefined_potential_exposure_minor_units)
        .ok_or(InputError::IntegerOverflow)
}

pub fn sum_balancing_payment_components_minor_units(components: &[i64]) -> Result<i64, InputError> {
    components.iter().try_fold(0_i64, |total, component| {
        total
            .checked_add(*component)
            .ok_or(InputError::IntegerOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> AssetLimits {
        AssetLimits {
            max_charge_mw: 50.0,
            min_stable_charge_mw: 5.0,
            max_discharge_mw: 60.0,
            min_energy_mwh: 20.0,
            max_energy_mwh: 200.0,
            round_trip_efficiency: 0.9,
            transitions: TransitionTimes {
                standstill_to_charging_seconds: 300.0,
                standstill_to_discharging_seconds: 900.0,
                charging_to_standstill_seconds: 300.0,
                discharging_to_standstill_seconds: 300.0,
                charging_to_discharging_seconds: 600.0,
                discharging_to_charging_seconds: 600.0,
            },
        }
    }

    const fn reserve(
        product: ReserveProduct,
        direction: ReserveDirection,
        mw: f64,
        start_response_seconds: Option<f64>,
        full_activation_seconds: Option<f64>,
        sustain_duration_seconds: Option<f64>,
        proven_sustain_seconds: Option<f64>,
    ) -> ReserveCommitment {
        ReserveCommitment {
            product,
            direction,
            mw,
            start_response_seconds,
            full_activation_seconds,
            sustain_duration_seconds,
            proven_sustain_seconds,
        }
    }

    fn empty_reserves() -> &'static [ReserveCommitment] {
        &[]
    }

    fn generating_price_curve() -> GeneratingPriceCurve {
        GeneratingPriceCurve {
            minimum_stable_generation_mw: 20.0,
            elbow_one_mw: 40.0,
            elbow_two_mw: 70.0,
            maximum_continuous_rating_mw: 100.0,
            emergency_level_one_mw: 110.0,
            increment_zero_minor_per_mwh: 10,
            increment_one_minor_per_mwh: 20,
            increment_two_minor_per_mwh: 30,
            increment_three_minor_per_mwh: 40,
            emergency_level_one_minor_per_mwh: 50,
        }
    }

    fn step(
        charge_mw: f64,
        discharge_mw: f64,
        transition_seconds: f64,
        reserves: &'static [ReserveCommitment],
    ) -> DispatchStep {
        DispatchStep {
            duration_hours: 1.0,
            charge_mw,
            discharge_mw,
            transition_seconds_before_action: transition_seconds,
            reserves,
            network_export_limit_mw: 60.0,
            network_import_limit_mw: 50.0,
        }
    }

    fn demand_side_consumption_price_curve() -> DemandSideConsumptionPriceCurve {
        DemandSideConsumptionPriceCurve {
            declared_consumption_mw: 100.0,
            first_elbow_mw: 80.0,
            second_elbow_mw: 50.0,
            third_elbow_mw: 20.0,
            increment_one_minor_per_mwh: 10,
            increment_two_minor_per_mwh: 20,
            increment_three_minor_per_mwh: 30,
        }
    }

    #[test]
    fn demand_side_consumption_price_curve_selects_next_increment_at_elbows() {
        let curve = demand_side_consumption_price_curve();
        assert_eq!(curve.price_at(90.0), Ok((PriceIncrement::One, 10)));
        assert_eq!(curve.price_at(80.0), Ok((PriceIncrement::Two, 20)));
        assert_eq!(curve.price_at(65.0), Ok((PriceIncrement::Two, 20)));
        assert_eq!(curve.price_at(50.0), Ok((PriceIncrement::Three, 30)));
        assert_eq!(curve.price_at(35.0), Ok((PriceIncrement::Three, 30)));
        assert_eq!(
            curve.price_at(20.0),
            Err(ReservePriceError::NoReducibleConsumptionBelowThirdElbow)
        );
    }

    #[test]
    fn dsu_reserve_opportunity_price_uses_consumption_curve_and_upward_products() {
        let curve = demand_side_consumption_price_curve();

        let regulating =
            dsu_upward_reserve_opportunity_price(ReserveProduct::Regulating, 50, 80.0, curve)
                .expect("DSU regulating-up price curve point is valid");
        assert_eq!(regulating.selected_increment, PriceIncrement::Two);
        assert_eq!(regulating.price_difference_minor_per_mwh, 30);

        let instantaneous =
            dsu_upward_reserve_opportunity_price(ReserveProduct::Instantaneous, 25, 50.0, curve)
                .expect("DSU instantaneous-up price curve point is valid");
        assert_eq!(instantaneous.selected_increment, PriceIncrement::Three);
        assert_eq!(instantaneous.price_difference_minor_per_mwh, 0);

        let ten_minute =
            dsu_upward_reserve_opportunity_price(ReserveProduct::TenMinute, 35, 90.0, curve)
                .expect("DSU ten-minute-up price curve point is valid");
        assert_eq!(ten_minute.selected_increment, PriceIncrement::One);
        assert_eq!(ten_minute.price_difference_minor_per_mwh, 25);
    }

    #[test]
    fn demand_side_consumption_price_curve_rejects_invalid_and_unavailable_points() {
        let curve = demand_side_consumption_price_curve();
        assert_eq!(
            curve.price_at(110.0),
            Err(ReservePriceError::ContractedPointOutsideCurve)
        );

        let mut invalid_order = curve;
        invalid_order.second_elbow_mw = 85.0;
        assert_eq!(
            invalid_order.validate(),
            Err(ReservePriceError::InvalidConsumptionPriceCurve)
        );

        let mut negative_price = curve;
        negative_price.increment_one_minor_per_mwh = -1;
        assert_eq!(
            negative_price.validate(),
            Err(ReservePriceError::NegativeIncrementalPrice)
        );

        let mut non_monotonic = curve;
        non_monotonic.increment_three_minor_per_mwh = 15;
        assert_eq!(
            non_monotonic.validate(),
            Err(ReservePriceError::NonMonotonicIncrementalPrice)
        );

        let mut coincident_empty_blocks = curve;
        coincident_empty_blocks.first_elbow_mw = 50.0;
        assert_eq!(
            coincident_empty_blocks.price_at(50.0),
            Ok((PriceIncrement::Three, 30))
        );
    }

    #[test]
    fn dsu_regulating_up_and_instantaneous_capacity_use_metered_consumption() {
        let curve = demand_side_consumption_price_curve();

        let capped_by_contract = qualifying_dsu_capacity_for_payment(
            DsuReserveQualificationInputs {
                product: DsuUpwardReserveProduct::RegulatingUp,
                contracted_capacity_mw: Some(25.0),
                official_energy_consumption_mw: Some(90.0),
                effective_available_capacity_mw: Some(80.0),
                was_utilized_for_reserve_or_balancing: Some(false),
                energy_instruction_differed_from_contract: Some(false),
                acceptable_performance: Some(true),
            },
            curve,
        )
        .expect("synthetic Regulating Up inputs are valid");
        assert_eq!(
            capped_by_contract,
            DsuQualificationResult::Determined {
                qualifying_capacity_mw: 25.0
            }
        );

        let capped_by_metered_consumption = qualifying_dsu_capacity_for_payment(
            DsuReserveQualificationInputs {
                product: DsuUpwardReserveProduct::Instantaneous,
                contracted_capacity_mw: Some(60.0),
                official_energy_consumption_mw: Some(35.0),
                effective_available_capacity_mw: Some(80.0),
                was_utilized_for_reserve_or_balancing: Some(false),
                energy_instruction_differed_from_contract: Some(false),
                acceptable_performance: Some(true),
            },
            curve,
        )
        .expect("synthetic Instantaneous inputs are valid");
        assert_eq!(
            capped_by_metered_consumption,
            DsuQualificationResult::Determined {
                qualifying_capacity_mw: 15.0
            }
        );
    }

    #[test]
    fn dsu_regulating_capacity_is_capped_by_effective_availability_and_instruction_state() {
        let curve = demand_side_consumption_price_curve();

        let mut base = DsuReserveQualificationInputs {
            product: DsuUpwardReserveProduct::RegulatingUp,
            contracted_capacity_mw: Some(80.0),
            official_energy_consumption_mw: Some(90.0),
            effective_available_capacity_mw: Some(12.0),
            was_utilized_for_reserve_or_balancing: Some(false),
            energy_instruction_differed_from_contract: Some(false),
            acceptable_performance: Some(true),
        };

        // Clause-specific consumption headroom is 70 MW, but §9.8.3(6)
        // caps qualifying capacity at 12 MW effective availability.
        assert_eq!(
            qualifying_dsu_capacity_for_payment(base, curve),
            Ok(DsuQualificationResult::Determined {
                qualifying_capacity_mw: 12.0
            })
        );

        // Under §9.8.3(4), a not-utilized unit with a different energy
        // instruction and acceptable performance can qualify up to contract
        // and effective availability, rather than the idle consumption curve.
        base.energy_instruction_differed_from_contract = Some(true);
        assert_eq!(
            qualifying_dsu_capacity_for_payment(base, curve),
            Ok(DsuQualificationResult::Determined {
                qualifying_capacity_mw: 12.0
            })
        );

        // Missing evidence of the instruction comparison cannot select a path.
        base.energy_instruction_differed_from_contract = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(base, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "missing not-utilized instruction comparison"
            })
        );

        // An utilized reserve branch is governed by §9.8.3(3) and requires
        // acceptable performance evidence before the contracted amount is qualified.
        base.was_utilized_for_reserve_or_balancing = Some(true);
        base.energy_instruction_differed_from_contract = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(base, curve),
            Ok(DsuQualificationResult::Determined {
                qualifying_capacity_mw: 12.0
            })
        );

        base.acceptable_performance = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(base, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "utilized reserve lacks acceptable-performance evidence"
            })
        );
    }

    #[test]
    fn dsu_ten_minute_capacity_uses_effective_available_capacity() {
        let curve = demand_side_consumption_price_curve();

        let qualified = qualifying_dsu_capacity_for_payment(
            DsuReserveQualificationInputs {
                product: DsuUpwardReserveProduct::TenMinute,
                contracted_capacity_mw: Some(70.0),
                official_energy_consumption_mw: None,
                effective_available_capacity_mw: Some(75.0),
                was_utilized_for_reserve_or_balancing: None,
                energy_instruction_differed_from_contract: None,
                acceptable_performance: None,
            },
            curve,
        )
        .expect("synthetic Ten-Minute inputs are valid");
        assert_eq!(
            qualified,
            DsuQualificationResult::Determined {
                qualifying_capacity_mw: 55.0
            }
        );

        let below_third_elbow = qualifying_dsu_capacity_for_payment(
            DsuReserveQualificationInputs {
                product: DsuUpwardReserveProduct::TenMinute,
                contracted_capacity_mw: Some(70.0),
                official_energy_consumption_mw: None,
                effective_available_capacity_mw: Some(10.0),
                was_utilized_for_reserve_or_balancing: None,
                energy_instruction_differed_from_contract: None,
                acceptable_performance: None,
            },
            curve,
        )
        .expect("below-elbow case remains a valid input");
        assert_eq!(
            below_third_elbow,
            DsuQualificationResult::Determined {
                qualifying_capacity_mw: 0.0
            }
        );
    }

    #[test]
    fn dsu_capacity_qualification_keeps_missing_and_utilized_cases_indeterminate() {
        let curve = demand_side_consumption_price_curve();
        let base = DsuReserveQualificationInputs {
            product: DsuUpwardReserveProduct::RegulatingUp,
            contracted_capacity_mw: Some(50.0),
            official_energy_consumption_mw: Some(90.0),
            effective_available_capacity_mw: Some(90.0),
            was_utilized_for_reserve_or_balancing: Some(false),
            energy_instruction_differed_from_contract: Some(false),
            acceptable_performance: Some(true),
        };

        let mut missing_meter = base;
        missing_meter.official_energy_consumption_mw = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(missing_meter, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "missing official energy consumption"
            })
        );

        let mut utilized = base;
        utilized.was_utilized_for_reserve_or_balancing = Some(true);
        utilized.acceptable_performance = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(utilized, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "utilized reserve lacks acceptable-performance evidence"
            })
        );

        let mut missing_utilization = base;
        missing_utilization.was_utilized_for_reserve_or_balancing = None;
        assert_eq!(
            qualifying_dsu_capacity_for_payment(missing_utilization, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "missing reserve/balancing utilization state"
            })
        );

        let mut above_declared = base;
        above_declared.official_energy_consumption_mw = Some(101.0);
        assert_eq!(
            qualifying_dsu_capacity_for_payment(above_declared, curve),
            Ok(DsuQualificationResult::Indeterminate {
                reason: "official consumption exceeds declared consumption"
            })
        );
    }

    #[test]
    fn dsu_capacity_qualification_rejects_negative_and_nonfinite_inputs() {
        let curve = demand_side_consumption_price_curve();
        let mut negative_contract = DsuReserveQualificationInputs {
            product: DsuUpwardReserveProduct::TenMinute,
            contracted_capacity_mw: Some(-1.0),
            official_energy_consumption_mw: None,
            effective_available_capacity_mw: Some(75.0),
            was_utilized_for_reserve_or_balancing: None,
            energy_instruction_differed_from_contract: None,
            acceptable_performance: None,
        };
        assert_eq!(
            qualifying_dsu_capacity_for_payment(negative_contract, curve),
            Err(DsuQualificationError::NegativeInput(
                "contracted_capacity_mw"
            ))
        );

        negative_contract.contracted_capacity_mw = Some(20.0);
        negative_contract.effective_available_capacity_mw = Some(f64::NAN);
        assert_eq!(
            qualifying_dsu_capacity_for_payment(negative_contract, curve),
            Err(DsuQualificationError::NonFiniteInput(
                "effective_available_capacity_mw"
            ))
        );
    }

    fn reserve_payment_inputs(
        product: ReserveCapacityPaymentProduct,
        performance_adjustment: Option<PerformanceAdjustmentEvidence>,
    ) -> ReserveCapacityPaymentInputs {
        ReserveCapacityPaymentInputs {
            product,
            market_price: Some(ReserveCategoryMarketPrice {
                minor_zar_per_mw_per_trading_period: 10_000,
            }),
            contracted_capacity_milli_mw: Some(10_000),
            qualifying_capacity_milli_mw: Some(5_500),
            performance_adjustment,
        }
    }

    #[test]
    fn reserve_capacity_payment_uses_reserve_category_price_and_exact_performance_factor() {
        let inputs = reserve_payment_inputs(
            ReserveCapacityPaymentProduct::RegulatingUp,
            Some(PerformanceAdjustmentEvidence::CriteriaEstablished {
                factor_numerator: Some(4),
                factor_denominator: Some(5),
            }),
        );
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Determined(
                ExactReserveCapacityPayment {
                    // 10,000 minor ZAR/MW/period × 5.5 MW × 4/5 = 44,000 minor ZAR.
                    numerator_minor_zar: 220_000_000,
                    denominator: 5_000,
                }
            ))
        );
    }

    #[test]
    fn reserve_capacity_payment_uses_unit_factor_when_no_performance_criteria_exist() {
        for product in [
            ReserveCapacityPaymentProduct::RegulatingUp,
            ReserveCapacityPaymentProduct::RegulatingDown,
            ReserveCapacityPaymentProduct::Instantaneous,
        ] {
            let inputs = reserve_payment_inputs(
                product,
                Some(PerformanceAdjustmentEvidence::NoCriteriaEstablished),
            );
            assert_eq!(
                calculate_reserve_capacity_payment(inputs),
                Ok(ReserveCapacityPaymentResult::Determined(
                    ExactReserveCapacityPayment {
                        numerator_minor_zar: 55_000_000,
                        denominator: 1_000,
                    }
                ))
            );
        }
    }

    #[test]
    fn ten_minute_payment_has_no_performance_adjustment_term_and_retains_fractional_minor_units() {
        let mut inputs = reserve_payment_inputs(ReserveCapacityPaymentProduct::TenMinute, None);
        inputs.market_price = Some(ReserveCategoryMarketPrice {
            minor_zar_per_mw_per_trading_period: 12_345,
        });
        inputs.qualifying_capacity_milli_mw = Some(1_250);

        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Determined(
                ExactReserveCapacityPayment {
                    numerator_minor_zar: 15_431_250,
                    denominator: 1_000,
                }
            ))
        );

        inputs.performance_adjustment = Some(PerformanceAdjustmentEvidence::NoCriteriaEstablished);
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Err(ReserveCapacityPaymentError::UnexpectedPerformanceAdjustmentForTenMinute)
        );
    }

    #[test]
    fn reserve_capacity_payment_fails_closed_on_missing_criteria_and_missing_market_data() {
        let mut inputs = reserve_payment_inputs(
            ReserveCapacityPaymentProduct::Instantaneous,
            Some(PerformanceAdjustmentEvidence::Unknown),
        );
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Indeterminate {
                reason: "System Operator performance-criteria state is unknown"
            })
        );

        inputs.performance_adjustment = Some(PerformanceAdjustmentEvidence::CriteriaEstablished {
            factor_numerator: Some(1),
            factor_denominator: None,
        });
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Indeterminate {
                reason: "published performance criteria lack an authoritative SO factor"
            })
        );

        inputs.performance_adjustment = None;
        inputs.market_price = None;
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Indeterminate {
                reason: "missing reserve-category market price"
            })
        );
    }

    #[test]
    fn reserve_capacity_payment_rejects_uncontracted_overclaim_and_bad_factor() {
        let mut inputs = reserve_payment_inputs(
            ReserveCapacityPaymentProduct::RegulatingDown,
            Some(PerformanceAdjustmentEvidence::NoCriteriaEstablished),
        );
        inputs.contracted_capacity_milli_mw = Some(0);
        inputs.qualifying_capacity_milli_mw = Some(0);
        inputs.market_price = None;
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Ok(ReserveCapacityPaymentResult::Determined(
                ExactReserveCapacityPayment {
                    numerator_minor_zar: 0,
                    denominator: 1
                }
            ))
        );

        inputs.contracted_capacity_milli_mw = Some(1_000);
        inputs.qualifying_capacity_milli_mw = Some(1_001);
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Err(ReserveCapacityPaymentError::QualifyingCapacityExceedsContracted)
        );

        inputs.qualifying_capacity_milli_mw = Some(500);
        inputs.performance_adjustment = Some(PerformanceAdjustmentEvidence::CriteriaEstablished {
            factor_numerator: Some(1),
            factor_denominator: Some(0),
        });
        assert_eq!(
            calculate_reserve_capacity_payment(inputs),
            Err(ReserveCapacityPaymentError::ZeroPerformanceFactorDenominator)
        );
    }

    #[test]
    fn reserve_opportunity_price_uses_product_specific_elbow_rules_and_signs() {
        let curve = generating_price_curve();

        let regulating_up = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            50,
            40.0,
            curve,
        )
        .expect("regulating up price is supported");
        assert_eq!(regulating_up.selected_increment, PriceIncrement::Two);
        assert_eq!(regulating_up.price_difference_minor_per_mwh, 20);

        let regulating_down = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            15,
            40.0,
            curve,
        )
        .expect("regulating down price is supported");
        assert_eq!(regulating_down.selected_increment, PriceIncrement::One);
        assert_eq!(regulating_down.price_difference_minor_per_mwh, 5);

        let instantaneous = reserve_opportunity_price(
            ReserveProduct::Instantaneous,
            ReserveDirection::Up,
            50,
            40.0,
            curve,
        )
        .expect("instantaneous up price is supported");
        assert_eq!(instantaneous.selected_increment, PriceIncrement::Two);
        assert_eq!(instantaneous.price_difference_minor_per_mwh, 20);

        let ten_minute = reserve_opportunity_price(
            ReserveProduct::TenMinute,
            ReserveDirection::Up,
            20,
            40.0,
            curve,
        )
        .expect("ten-minute up price is supported");
        assert_eq!(ten_minute.selected_increment, PriceIncrement::Two);
        assert_eq!(ten_minute.price_difference_minor_per_mwh, 0);
    }

    #[test]
    fn reserve_opportunity_price_uses_mingen_mcr_and_emergency_boundaries() {
        let curve = generating_price_curve();

        let at_mingen = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            15,
            20.0,
            curve,
        )
        .expect("mingen boundary is valid");
        assert_eq!(at_mingen.selected_increment, PriceIncrement::One);
        assert_eq!(at_mingen.price_difference_minor_per_mwh, 5);

        let at_mcr = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            50,
            100.0,
            curve,
        )
        .expect("MCR boundary is valid");
        assert_eq!(at_mcr.selected_increment, PriceIncrement::Three);
        assert_eq!(at_mcr.price_difference_minor_per_mwh, 10);

        let above_mcr = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            50,
            105.0,
            curve,
        )
        .expect("emergency curve segment is valid");
        assert_eq!(
            above_mcr.selected_increment,
            PriceIncrement::EmergencyLevelOne
        );
        assert_eq!(above_mcr.price_difference_minor_per_mwh, 0);
    }

    #[test]
    fn reserve_opportunity_price_selects_lower_regulating_down_elbow_two() {
        let curve = generating_price_curve();
        let down = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            25,
            70.0,
            curve,
        )
        .expect("regulating down is supported");
        assert_eq!(down.selected_increment, PriceIncrement::Two);
        assert_eq!(down.price_difference_minor_per_mwh, 5);

        // When the selected incremental price is below SMP, the Code's
        // max(IP - SMP, 0) rule must clamp the opportunity price to zero.
        let down_clamped = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            35,
            70.0,
            curve,
        )
        .expect("regulating down zero-clamp case is supported");
        assert_eq!(down_clamped.selected_increment, PriceIncrement::Two);
        assert_eq!(down_clamped.price_difference_minor_per_mwh, 0);

        let up = reserve_opportunity_price(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            50,
            70.0,
            curve,
        )
        .expect("regulating up is supported");
        assert_eq!(up.selected_increment, PriceIncrement::Three);
        assert_eq!(up.price_difference_minor_per_mwh, 10);
    }

    #[test]
    fn reserve_opportunity_price_rejects_bad_curves_and_unsupported_directions() {
        let mut invalid = generating_price_curve();
        invalid.increment_two_minor_per_mwh = 15;
        assert_eq!(
            reserve_opportunity_price(
                ReserveProduct::Regulating,
                ReserveDirection::Up,
                50,
                40.0,
                invalid,
            ),
            Err(ReservePriceError::NonMonotonicIncrementalPrice)
        );

        assert_eq!(
            reserve_opportunity_price(
                ReserveProduct::Instantaneous,
                ReserveDirection::Down,
                50,
                40.0,
                generating_price_curve(),
            ),
            Err(ReservePriceError::UnsupportedProductDirection)
        );

        assert_eq!(
            reserve_opportunity_price(
                ReserveProduct::Regulating,
                ReserveDirection::Down,
                i64::MIN,
                40.0,
                generating_price_curve(),
            ),
            Err(ReservePriceError::ArithmeticOverflow)
        );
    }

    #[test]
    fn feasible_discharge_reserve_and_energy_balance() {
        static RESERVES: [ReserveCommitment; 2] = [
            reserve(
                ReserveProduct::Regulating,
                ReserveDirection::Up,
                5.0,
                Some(5.0),
                Some(580.0),
                Some(600.0),
                Some(600.0),
            ),
            reserve(
                ReserveProduct::TenMinute,
                ReserveDirection::Up,
                5.0,
                None,
                Some(600.0),
                Some(600.0),
                Some(600.0),
            ),
        ];
        let result = evaluate_dispatch(limits(), 150.0, &[step(0.0, 30.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(result.feasible);
        assert!((result.final_energy_mwh - (150.0 - 30.0 / 0.9_f64.sqrt() * 0.75)).abs() < 1.0e-8);
        assert_eq!(result.steps[0].reserve_up_mw, 10.0);
    }

    #[test]
    fn upward_reserve_includes_reducing_existing_charge() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            80.0,
            Some(5.0),
            Some(600.0),
            Some(600.0),
            Some(600.0),
        )];
        let result = evaluate_dispatch(limits(), 150.0, &[step(30.0, 0.0, 300.0, &RESERVES)])
            .expect("fixture inputs are valid");
        let assessment = &result.steps[0].reserve_assessment;
        assert_eq!(assessment.available_up_mw, 90.0);
        assert_eq!(assessment.committed_up_mw, 80.0);
        assert_eq!(assessment.qualification, ReserveQualification::Qualified);
        assert!(result.feasible);
    }

    #[test]
    fn downward_reserve_includes_reducing_discharge_and_increasing_charge() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            80.0,
            Some(5.0),
            Some(600.0),
            Some(600.0),
            Some(600.0),
        )];
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 40.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        let assessment = &result.steps[0].reserve_assessment;
        assert_eq!(assessment.available_down_mw, 90.0);
        assert_eq!(assessment.committed_down_mw, 80.0);
        assert_eq!(assessment.qualification, ReserveQualification::Qualified);
        assert!(result.feasible);
    }

    #[test]
    fn reserve_target_cannot_land_inside_minimum_charging_deadband() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Regulating,
            ReserveDirection::Down,
            11.0,
            Some(5.0),
            Some(600.0),
            Some(600.0),
            Some(600.0),
        )];
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 10.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert_eq!(
            result.steps[0].reserve_assessment.qualification,
            ReserveQualification::Infeasible
        );
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReserveTargetBelowMinimumStableCharge)
        );
        assert!(!result.feasible);
    }

    #[test]
    fn sawem_rev_2_3_reserve_response_and_sustain_requirements_are_product_specific() {
        let regulating =
            sawem_rev_2_3_reserve_requirement(ReserveProduct::Regulating, ReserveDirection::Up)
                .expect("regulating up is supported");
        assert_eq!(regulating.maximum_start_response_seconds, Some(10.0));
        assert_eq!(regulating.maximum_full_activation_seconds, Some(600.0));
        assert_eq!(regulating.minimum_sustain_seconds, None);

        let instantaneous =
            sawem_rev_2_3_reserve_requirement(ReserveProduct::Instantaneous, ReserveDirection::Up)
                .expect("instantaneous up is supported");
        assert_eq!(instantaneous.maximum_full_activation_seconds, Some(10.0));
        assert_eq!(instantaneous.minimum_sustain_seconds, Some(600.0));

        let ten_minute =
            sawem_rev_2_3_reserve_requirement(ReserveProduct::TenMinute, ReserveDirection::Up)
                .expect("ten-minute up is supported");
        assert_eq!(ten_minute.maximum_full_activation_seconds, Some(600.0));
        assert_eq!(ten_minute.minimum_sustain_seconds, None);

        assert_eq!(
            sawem_rev_2_3_reserve_requirement(
                ReserveProduct::Instantaneous,
                ReserveDirection::Down,
            ),
            Err(Violation::UnsupportedReserveProductDirection)
        );
    }

    #[test]
    fn reserve_activation_cannot_be_faster_than_registered_mode_transition() {
        let mut asset = limits();
        asset.transitions.charging_to_discharging_seconds = 900.0;
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            80.0,
            Some(5.0),
            Some(600.0),
            Some(600.0),
            Some(600.0),
        )];
        let result = evaluate_dispatch(asset, 150.0, &[step(30.0, 0.0, 300.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReserveTransitionDeadlineExceeded)
        );
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReserveTransitionEvidenceContradiction)
        );
        assert!(!result.feasible);
    }

    #[test]
    fn ten_minute_reserve_without_operational_sustain_profile_is_indeterminate() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::TenMinute,
            ReserveDirection::Up,
            10.0,
            None,
            Some(600.0),
            None,
            None,
        )];
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 10.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        let assessment = &result.steps[0].reserve_assessment;
        assert_eq!(
            assessment.qualification,
            ReserveQualification::Indeterminate
        );
        assert!(
            assessment
                .indeterminate_reasons
                .contains(&"sustain_duration_not_defined_by_market_profile")
        );
        assert!(!result.feasible);
        assert!(result.steps[0].violations.is_empty());
    }

    #[test]
    fn instantaneous_reserve_enforces_ten_second_activation_and_ten_minute_sustain() {
        static TOO_SLOW: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Instantaneous,
            ReserveDirection::Up,
            5.0,
            None,
            Some(11.0),
            None,
            Some(600.0),
        )];
        let slow = evaluate_dispatch(limits(), 100.0, &[step(0.0, 0.0, 0.0, &TOO_SLOW)])
            .expect("fixture inputs are valid");
        assert!(
            slow.steps[0]
                .violations
                .contains(&Violation::ReserveResponseDeadlineExceeded)
        );

        static TOO_SHORT: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Instantaneous,
            ReserveDirection::Up,
            5.0,
            None,
            Some(10.0),
            None,
            Some(500.0),
        )];
        let short = evaluate_dispatch(limits(), 100.0, &[step(0.0, 0.0, 0.0, &TOO_SHORT)])
            .expect("fixture inputs are valid");
        assert!(
            short.steps[0]
                .violations
                .contains(&Violation::ReserveSustainInsufficient)
        );
    }

    #[test]
    fn rejects_energy_plus_reserve_double_counting() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Instantaneous,
            ReserveDirection::Up,
            10.0,
            None,
            Some(8.0),
            None,
            Some(600.0),
        )];
        let result = evaluate_dispatch(limits(), 150.0, &[step(0.0, 55.0, 900.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReservePowerHeadroomExceeded)
        );
    }

    #[test]
    fn rejects_reserve_without_soc_energy_headroom() {
        static RESERVES: [ReserveCommitment; 1] = [reserve(
            ReserveProduct::Regulating,
            ReserveDirection::Up,
            20.0,
            Some(5.0),
            Some(600.0),
            Some(600.0),
            Some(600.0),
        )];
        let result = evaluate_dispatch(limits(), 20.5, &[step(0.0, 0.0, 0.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ReserveEnergyHeadroomExceeded)
        );
    }

    #[test]
    fn rejects_minimum_stable_charge_violation() {
        let result = evaluate_dispatch(limits(), 100.0, &[step(2.0, 0.0, 300.0, empty_reserves())])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::BelowMinimumStableCharging)
        );
    }

    #[test]
    fn rejects_insufficient_mode_transition_time() {
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 10.0, 60.0, empty_reserves())])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::TransitionTimeInsufficient)
        );
    }

    #[test]
    fn failed_transition_does_not_promote_internal_mode() {
        let result = evaluate_dispatch(
            limits(),
            100.0,
            &[
                step(0.0, 10.0, 60.0, empty_reserves()),
                step(0.0, 10.0, 0.0, empty_reserves()),
            ],
        )
        .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert_eq!(result.steps[0].achieved_mode, OperatingMode::Standstill);
        assert_eq!(result.steps[1].achieved_mode, OperatingMode::Standstill);
        assert!(
            result.steps[1]
                .violations
                .contains(&Violation::TransitionTimeInsufficient)
        );
        assert_eq!(result.final_energy_mwh, 100.0);
    }

    #[test]
    fn rejects_network_export_violation() {
        let mut s = step(0.0, 40.0, 900.0, empty_reserves());
        s.network_export_limit_mw = 30.0;
        let result = evaluate_dispatch(limits(), 100.0, &[s]).expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::ExportLimitExceeded)
        );
        assert_eq!(result.steps[0].energy_after_mwh, 100.0);
        assert_eq!(result.final_energy_mwh, 100.0);
        assert_eq!(result.steps[0].achieved_mode, OperatingMode::Standstill);
    }

    #[test]
    fn rejects_soc_floor_and_ceiling_violations() {
        let discharge =
            evaluate_dispatch(limits(), 25.0, &[step(0.0, 10.0, 900.0, empty_reserves())])
                .expect("fixture inputs are valid");
        assert!(!discharge.feasible);
        assert!(
            discharge.steps[0]
                .violations
                .contains(&Violation::EnergyBelowMinimum)
        );
        assert_eq!(discharge.steps[0].energy_after_mwh, 25.0);
        assert!(discharge.steps[0].candidate_energy_after_mwh < 20.0);

        let charge =
            evaluate_dispatch(limits(), 195.0, &[step(10.0, 0.0, 300.0, empty_reserves())])
                .expect("fixture inputs are valid");
        assert!(!charge.feasible);
        assert!(
            charge.steps[0]
                .violations
                .contains(&Violation::EnergyAboveMaximum)
        );
        assert_eq!(charge.steps[0].energy_after_mwh, 195.0);
        assert!(charge.steps[0].candidate_energy_after_mwh > 200.0);
    }

    #[test]
    fn rejects_duplicate_reserve_product_direction() {
        static RESERVES: [ReserveCommitment; 2] = [
            reserve(
                ReserveProduct::Regulating,
                ReserveDirection::Up,
                2.0,
                Some(5.0),
                Some(500.0),
                Some(600.0),
                Some(600.0),
            ),
            reserve(
                ReserveProduct::Regulating,
                ReserveDirection::Up,
                3.0,
                Some(5.0),
                Some(500.0),
                Some(600.0),
                Some(600.0),
            ),
        ];
        let result = evaluate_dispatch(limits(), 100.0, &[step(0.0, 0.0, 0.0, &RESERVES)])
            .expect("fixture inputs are valid");
        assert!(!result.feasible);
        assert!(
            result.steps[0]
                .violations
                .contains(&Violation::DuplicateReserveProductDirection)
        );
    }

    #[test]
    fn validates_daily_and_rolling_week_energy_caps() {
        let limits = SevenDayEnergyLimits {
            daily_limit_mwh: 250.0,
            next_six_days_limit_mwh: 1_200.0,
            seven_day_total_limit_mwh: 1_450.0,
        };
        let declarations = [180.0, 190.0, 210.0, 175.0, 200.0, 190.0, 180.0];
        let result = validate_seven_day_energy_declarations(&declarations, limits)
            .expect("fixture inputs are valid");
        assert!(result.feasible);
        assert_eq!(result.next_six_days_total_mwh, 1_145.0);
        assert_eq!(result.seven_day_total_mwh, 1_325.0);

        let invalid = [180.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0];
        let rejected = validate_seven_day_energy_declarations(&invalid, limits)
            .expect("fixture inputs are valid");
        assert!(!rejected.feasible);
        assert!(
            rejected
                .violations
                .contains(&"next_six_days_limit_exceeded")
        );
        assert!(
            rejected
                .violations
                .contains(&"seven_day_total_limit_exceeded")
        );
    }

    #[test]
    fn credit_cover_and_multiple_balancing_components_use_checked_arithmetic() {
        assert_eq!(
            required_credit_cover_minor_units(25_000_000, 10_000_000),
            Ok(35_000_000)
        );
        assert_eq!(
            sum_balancing_payment_components_minor_units(&[-4_500_000, 1_000_000]),
            Ok(-3_500_000)
        );
        assert_eq!(
            required_credit_cover_minor_units(u64::MAX, 1),
            Err(InputError::IntegerOverflow)
        );
    }
}
