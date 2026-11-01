pub mod vector_conformal;
pub use vector_conformal::*;

pub mod forecast_journal;
pub mod survival;
pub use forecast_journal::*;
pub use survival::*;

use std::collections::BTreeMap;

/// Integer-valued resource units. Spend dimensions are never scalarized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceVector {
    pub tokens: u128,
    pub tool_calls: u128,
    pub cpu_ms: u128,
    pub gpu_ms: u128,
    pub wall_ms: u128,
    pub micro_usd: u128,
}

impl ResourceVector {
    pub const ZERO: Self = Self::new(0, 0, 0, 0, 0, 0);

    pub const fn new(
        tokens: u128,
        tool_calls: u128,
        cpu_ms: u128,
        gpu_ms: u128,
        wall_ms: u128,
        micro_usd: u128,
    ) -> Self {
        Self {
            tokens,
            tool_calls,
            cpu_ms,
            gpu_ms,
            wall_ms,
            micro_usd,
        }
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            tokens: self.tokens.checked_add(rhs.tokens)?,
            tool_calls: self.tool_calls.checked_add(rhs.tool_calls)?,
            cpu_ms: self.cpu_ms.checked_add(rhs.cpu_ms)?,
            gpu_ms: self.gpu_ms.checked_add(rhs.gpu_ms)?,
            wall_ms: self.wall_ms.checked_add(rhs.wall_ms)?,
            micro_usd: self.micro_usd.checked_add(rhs.micro_usd)?,
        })
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(Self {
            tokens: self.tokens.checked_sub(rhs.tokens)?,
            tool_calls: self.tool_calls.checked_sub(rhs.tool_calls)?,
            cpu_ms: self.cpu_ms.checked_sub(rhs.cpu_ms)?,
            gpu_ms: self.gpu_ms.checked_sub(rhs.gpu_ms)?,
            wall_ms: self.wall_ms.checked_sub(rhs.wall_ms)?,
            micro_usd: self.micro_usd.checked_sub(rhs.micro_usd)?,
        })
    }

    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_sub(rhs.tokens),
            tool_calls: self.tool_calls.saturating_sub(rhs.tool_calls),
            cpu_ms: self.cpu_ms.saturating_sub(rhs.cpu_ms),
            gpu_ms: self.gpu_ms.saturating_sub(rhs.gpu_ms),
            wall_ms: self.wall_ms.saturating_sub(rhs.wall_ms),
            micro_usd: self.micro_usd.saturating_sub(rhs.micro_usd),
        }
    }

    /// Returns true when every component of self is no greater than rhs.
    pub fn fits_within(self, rhs: Self) -> bool {
        self.tokens <= rhs.tokens
            && self.tool_calls <= rhs.tool_calls
            && self.cpu_ms <= rhs.cpu_ms
            && self.gpu_ms <= rhs.gpu_ms
            && self.wall_ms <= rhs.wall_ms
            && self.micro_usd <= rhs.micro_usd
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReservationId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationStatus {
    Reserved,
    OutcomeUnknown,
    Settled,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reservation {
    upper_bound: ResourceVector,
    status: ReservationStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerError {
    DuplicateReservation,
    UnknownReservation,
    InvalidTransition,
    HardBudgetExceeded,
    ArithmeticOverflow,
    NonAuthoritativeEvidence,
    EvidenceIdentityMismatch,
    MissingStateDigest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SettlementOutcome {
    pub measured_usage: ResourceVector,
    pub reservation_exceeded: bool,
    pub hard_cap_exceeded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectDisposition {
    Applied { measured_usage: ResourceVector },
    NotApplied,
    Unknown,
}

/// A typed stand-in for an authoritative post-action read-back / meter receipt.
/// The digest is an identity binding in this simulator, not a digital signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateEvidence {
    pub action_id: u64,
    pub authoritative: bool,
    pub state_digest: [u8; 32],
    pub effect: EffectDisposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconciliationOutcome {
    Settled(SettlementOutcome),
    Released,
    StillUnknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetLedger {
    pub cap: ResourceVector,
    pub spent: ResourceVector,
    pub reserved: ResourceVector,
    pub reservation_overruns: u64,
    pub hard_budget_violations: u64,
    reservations: BTreeMap<ReservationId, Reservation>,
}

impl BudgetLedger {
    pub fn new(cap: ResourceVector) -> Self {
        Self {
            cap,
            spent: ResourceVector::ZERO,
            reserved: ResourceVector::ZERO,
            reservation_overruns: 0,
            hard_budget_violations: 0,
            reservations: BTreeMap::new(),
        }
    }

    /// Atomically checks all resource dimensions before creating a reservation.
    pub fn reserve(
        &mut self,
        id: ReservationId,
        upper_bound: ResourceVector,
    ) -> Result<(), LedgerError> {
        if self.reservations.contains_key(&id) {
            return Err(LedgerError::DuplicateReservation);
        }
        let next_reserved = self
            .reserved
            .checked_add(upper_bound)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let committed = self
            .spent
            .checked_add(next_reserved)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        if !committed.fits_within(self.cap) {
            return Err(LedgerError::HardBudgetExceeded);
        }

        self.reserved = next_reserved;
        self.reservations.insert(
            id,
            Reservation {
                upper_bound,
                status: ReservationStatus::Reserved,
            },
        );
        Ok(())
    }

    /// Ambiguous completion holds the reservation; it does not imply non-application.
    pub fn mark_outcome_unknown(&mut self, id: ReservationId) -> Result<(), LedgerError> {
        let reservation = self
            .reservations
            .get_mut(&id)
            .ok_or(LedgerError::UnknownReservation)?;
        if reservation.status != ReservationStatus::Reserved {
            return Err(LedgerError::InvalidTransition);
        }
        reservation.status = ReservationStatus::OutcomeUnknown;
        Ok(())
    }

    /// Reconcile only from authoritative, identity-bound evidence.
    pub fn reconcile(
        &mut self,
        id: ReservationId,
        evidence: StateEvidence,
    ) -> Result<ReconciliationOutcome, LedgerError> {
        if !evidence.authoritative {
            return Err(LedgerError::NonAuthoritativeEvidence);
        }
        if evidence.action_id != id.0 {
            return Err(LedgerError::EvidenceIdentityMismatch);
        }
        if evidence.state_digest.iter().all(|byte| *byte == 0) {
            return Err(LedgerError::MissingStateDigest);
        }
        let reservation = self
            .reservations
            .get(&id)
            .copied()
            .ok_or(LedgerError::UnknownReservation)?;
        if matches!(
            reservation.status,
            ReservationStatus::Settled | ReservationStatus::Released
        ) {
            return Err(LedgerError::InvalidTransition);
        }

        match evidence.effect {
            EffectDisposition::Applied { measured_usage } => self
                .settle(id, measured_usage)
                .map(ReconciliationOutcome::Settled),
            EffectDisposition::NotApplied => {
                self.release(id)?;
                Ok(ReconciliationOutcome::Released)
            }
            EffectDisposition::Unknown => {
                self.reservations
                    .get_mut(&id)
                    .ok_or(LedgerError::UnknownReservation)?
                    .status = ReservationStatus::OutcomeUnknown;
                Ok(ReconciliationOutcome::StillUnknown)
            }
        }
    }

    fn settle(
        &mut self,
        id: ReservationId,
        measured_usage: ResourceVector,
    ) -> Result<SettlementOutcome, LedgerError> {
        let reservation = self
            .reservations
            .get(&id)
            .copied()
            .ok_or(LedgerError::UnknownReservation)?;
        let next_reserved = self
            .reserved
            .checked_sub(reservation.upper_bound)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let next_spent = self
            .spent
            .checked_add(measured_usage)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let committed = next_spent
            .checked_add(next_reserved)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        let hard_cap_exceeded = !committed.fits_within(self.cap);
        let reservation_exceeded = !measured_usage.fits_within(reservation.upper_bound);

        self.reserved = next_reserved;
        self.spent = next_spent;
        self.reservations
            .get_mut(&id)
            .ok_or(LedgerError::UnknownReservation)?
            .status = ReservationStatus::Settled;
        if reservation_exceeded {
            self.reservation_overruns = self.reservation_overruns.saturating_add(1);
        }
        if hard_cap_exceeded {
            self.hard_budget_violations = self.hard_budget_violations.saturating_add(1);
        }

        Ok(SettlementOutcome {
            measured_usage,
            reservation_exceeded,
            hard_cap_exceeded,
        })
    }

    fn release(&mut self, id: ReservationId) -> Result<(), LedgerError> {
        let reservation = self
            .reservations
            .get(&id)
            .copied()
            .ok_or(LedgerError::UnknownReservation)?;
        if matches!(
            reservation.status,
            ReservationStatus::Settled | ReservationStatus::Released
        ) {
            return Err(LedgerError::InvalidTransition);
        }
        let next_reserved = self
            .reserved
            .checked_sub(reservation.upper_bound)
            .ok_or(LedgerError::ArithmeticOverflow)?;
        self.reserved = next_reserved;
        self.reservations
            .get_mut(&id)
            .ok_or(LedgerError::UnknownReservation)?
            .status = ReservationStatus::Released;
        Ok(())
    }

    pub fn reservation_status(&self, id: ReservationId) -> Result<ReservationStatus, LedgerError> {
        self.reservations
            .get(&id)
            .map(|reservation| reservation.status)
            .ok_or(LedgerError::UnknownReservation)
    }

    pub fn remaining_capacity(&self) -> ResourceVector {
        self.cap
            .saturating_sub(self.spent)
            .saturating_sub(self.reserved)
    }

    pub fn hard_budget_invariant_holds(&self) -> bool {
        self.spent
            .checked_add(self.reserved)
            .is_some_and(|committed| committed.fits_within(self.cap))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftStatus {
    InDomain,
    SuspectedShift,
    UnsupportedShift,
}

/// Immutable pre-execution forecast. Quantiles are marginal per resource;
/// this type does not imply a joint vector-coverage guarantee.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RemainingCostForecast {
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub calibration_digest: [u8; 32],
    pub completion_probability: f64,
    pub remaining_q50: ResourceVector,
    pub remaining_q90: ResourceVector,
    pub remaining_q95: ResourceVector,
    pub mandatory_verification_reserve: ResourceVector,
    pub support_count: u64,
    pub shift_status: ShiftStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastError {
    InvalidProbability,
    MissingDigest,
    MissingSupport,
    NonMonotoneQuantiles,
}

impl RemainingCostForecast {
    pub fn validate(self) -> Result<(), ForecastError> {
        if !self.completion_probability.is_finite()
            || !(0.0..=1.0).contains(&self.completion_probability)
        {
            return Err(ForecastError::InvalidProbability);
        }
        let digests = [
            self.pre_state_digest,
            self.frontier_digest,
            self.model_digest,
            self.calibration_digest,
        ];
        if digests
            .iter()
            .any(|digest| digest.iter().all(|byte| *byte == 0))
        {
            return Err(ForecastError::MissingDigest);
        }
        if self.support_count == 0 {
            return Err(ForecastError::MissingSupport);
        }
        if !self.remaining_q50.fits_within(self.remaining_q90)
            || !self.remaining_q90.fits_within(self.remaining_q95)
        {
            return Err(ForecastError::NonMonotoneQuantiles);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstentionReason {
    InvalidForecast,
    ShiftedContext,
    InsufficientSupport,
    CompletionProbabilityTooLow,
    InsufficientRemainingBudget,
    ExistingBudgetViolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Abstain(AbstentionReason),
}

/// Conservative feasibility gate. It is a decision aid, never the resource
/// enforcement mechanism; reserve() must still be called before each dispatch.
pub fn assess_completion_feasibility(
    forecast: RemainingCostForecast,
    ledger: &BudgetLedger,
    minimum_completion_probability: f64,
    minimum_support: u64,
) -> AdmissionDecision {
    if forecast.validate().is_err()
        || !minimum_completion_probability.is_finite()
        || !(0.0..=1.0).contains(&minimum_completion_probability)
    {
        return AdmissionDecision::Abstain(AbstentionReason::InvalidForecast);
    }
    if !ledger.hard_budget_invariant_holds() {
        return AdmissionDecision::Abstain(AbstentionReason::ExistingBudgetViolation);
    }
    if forecast.shift_status != ShiftStatus::InDomain {
        return AdmissionDecision::Abstain(AbstentionReason::ShiftedContext);
    }
    if forecast.support_count < minimum_support {
        return AdmissionDecision::Abstain(AbstentionReason::InsufficientSupport);
    }
    if forecast.completion_probability < minimum_completion_probability {
        return AdmissionDecision::Abstain(AbstentionReason::CompletionProbabilityTooLow);
    }

    let Some(required) = forecast
        .remaining_q90
        .checked_add(forecast.mandatory_verification_reserve)
    else {
        return AdmissionDecision::Abstain(AbstentionReason::InvalidForecast);
    };
    if !required.fits_within(ledger.remaining_capacity()) {
        return AdmissionDecision::Abstain(AbstentionReason::InsufficientRemainingBudget);
    }
    AdmissionDecision::Admit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(value: u128) -> ResourceVector {
        ResourceVector::new(value, 0, 0, 0, 0, 0)
    }

    fn evidence(id: ReservationId, effect: EffectDisposition) -> StateEvidence {
        StateEvidence {
            action_id: id.0,
            authoritative: true,
            state_digest: [1; 32],
            effect,
        }
    }

    fn forecast() -> RemainingCostForecast {
        RemainingCostForecast {
            pre_state_digest: [1; 32],
            frontier_digest: [2; 32],
            model_digest: [3; 32],
            calibration_digest: [4; 32],
            completion_probability: 0.95,
            remaining_q50: tokens(30),
            remaining_q90: tokens(60),
            remaining_q95: tokens(75),
            mandatory_verification_reserve: tokens(20),
            support_count: 128,
            shift_status: ShiftStatus::InDomain,
        }
    }

    #[test]
    fn reservations_are_atomic_across_resource_dimensions() {
        let cap = ResourceVector::new(100, 4, 10_000, 5_000, 20_000, 1_000_000);
        let mut ledger = BudgetLedger::new(cap);
        let first = ResourceVector::new(70, 2, 4_000, 1_000, 5_000, 200_000);
        let second = ResourceVector::new(31, 1, 1_000, 1_000, 1_000, 100_000);

        assert_eq!(ledger.reserve(ReservationId(1), first), Ok(()));
        assert_eq!(
            ledger.reserve(ReservationId(2), second),
            Err(LedgerError::HardBudgetExceeded)
        );
        assert_eq!(ledger.reserved, first);
        assert!(ledger.hard_budget_invariant_holds());
    }

    #[test]
    fn unknown_outcome_retains_reservation_until_authoritative_reconciliation() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(7);
        ledger.reserve(id, tokens(40)).unwrap();
        ledger.mark_outcome_unknown(id).unwrap();

        assert_eq!(ledger.reserved, tokens(40));
        assert_eq!(
            ledger.reconcile(
                id,
                StateEvidence {
                    authoritative: false,
                    ..evidence(id, EffectDisposition::NotApplied)
                }
            ),
            Err(LedgerError::NonAuthoritativeEvidence)
        );
        assert_eq!(ledger.reserved, tokens(40));

        let result = ledger
            .reconcile(id, evidence(id, EffectDisposition::Unknown))
            .unwrap();
        assert_eq!(result, ReconciliationOutcome::StillUnknown);
        assert_eq!(ledger.reserved, tokens(40));
        assert_eq!(
            ledger.reservation_status(id),
            Ok(ReservationStatus::OutcomeUnknown)
        );
    }

    #[test]
    fn authoritative_non_application_releases_reservation() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(8);
        ledger.reserve(id, tokens(40)).unwrap();
        ledger.mark_outcome_unknown(id).unwrap();

        assert_eq!(
            ledger
                .reconcile(id, evidence(id, EffectDisposition::NotApplied))
                .unwrap(),
            ReconciliationOutcome::Released
        );
        assert_eq!(ledger.reserved, ResourceVector::ZERO);
        assert_eq!(
            ledger.reservation_status(id),
            Ok(ReservationStatus::Released)
        );
        assert!(ledger.hard_budget_invariant_holds());
    }

    #[test]
    fn settlement_charges_measured_usage_not_the_reservation() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(9);
        ledger.reserve(id, tokens(30)).unwrap();

        let result = ledger
            .reconcile(
                id,
                evidence(
                    id,
                    EffectDisposition::Applied {
                        measured_usage: tokens(25),
                    },
                ),
            )
            .unwrap();
        assert_eq!(
            result,
            ReconciliationOutcome::Settled(SettlementOutcome {
                measured_usage: tokens(25),
                reservation_exceeded: false,
                hard_cap_exceeded: false,
            })
        );
        assert_eq!(ledger.spent, tokens(25));
        assert_eq!(ledger.reserved, ResourceVector::ZERO);
        assert_eq!(ledger.hard_budget_violations, 0);
    }

    #[test]
    fn reservation_overrun_is_visible_even_when_global_cap_holds() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(10);
        ledger.reserve(id, tokens(30)).unwrap();

        let result = ledger
            .reconcile(
                id,
                evidence(
                    id,
                    EffectDisposition::Applied {
                        measured_usage: tokens(40),
                    },
                ),
            )
            .unwrap();
        assert_eq!(
            result,
            ReconciliationOutcome::Settled(SettlementOutcome {
                measured_usage: tokens(40),
                reservation_exceeded: true,
                hard_cap_exceeded: false,
            })
        );
        assert_eq!(ledger.reservation_overruns, 1);
        assert_eq!(ledger.hard_budget_violations, 0);
    }

    #[test]
    fn hard_cap_overrun_is_recorded_not_clamped_away() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(11);
        ledger.reserve(id, tokens(90)).unwrap();

        let result = ledger
            .reconcile(
                id,
                evidence(
                    id,
                    EffectDisposition::Applied {
                        measured_usage: tokens(110),
                    },
                ),
            )
            .unwrap();
        assert!(matches!(
            result,
            ReconciliationOutcome::Settled(SettlementOutcome {
                hard_cap_exceeded: true,
                ..
            })
        ));
        assert_eq!(ledger.spent, tokens(110));
        assert_eq!(ledger.hard_budget_violations, 1);
        assert!(!ledger.hard_budget_invariant_holds());
        assert_eq!(ledger.remaining_capacity(), ResourceVector::ZERO);
    }

    #[test]
    fn feasibility_requires_room_for_mandatory_verification() {
        let ledger = BudgetLedger::new(tokens(100));
        assert_eq!(
            assess_completion_feasibility(forecast(), &ledger, 0.90, 32),
            AdmissionDecision::Admit
        );

        let mut nearly_spent = BudgetLedger::new(tokens(100));
        nearly_spent.reserve(ReservationId(1), tokens(30)).unwrap();
        nearly_spent
            .reconcile(
                ReservationId(1),
                evidence(
                    ReservationId(1),
                    EffectDisposition::Applied {
                        measured_usage: tokens(30),
                    },
                ),
            )
            .unwrap();
        assert_eq!(
            assess_completion_feasibility(forecast(), &nearly_spent, 0.90, 32),
            AdmissionDecision::Abstain(AbstentionReason::InsufficientRemainingBudget)
        );
    }

    #[test]
    fn shift_and_insufficient_support_force_abstention() {
        let ledger = BudgetLedger::new(tokens(100));
        let mut shifted = forecast();
        shifted.shift_status = ShiftStatus::SuspectedShift;
        assert_eq!(
            assess_completion_feasibility(shifted, &ledger, 0.90, 32),
            AdmissionDecision::Abstain(AbstentionReason::ShiftedContext)
        );

        let mut under_supported = forecast();
        under_supported.support_count = 12;
        assert_eq!(
            assess_completion_feasibility(under_supported, &ledger, 0.90, 32),
            AdmissionDecision::Abstain(AbstentionReason::InsufficientSupport)
        );
    }

    #[test]
    fn invalid_or_non_monotone_forecast_is_rejected() {
        let ledger = BudgetLedger::new(tokens(100));
        let mut invalid = forecast();
        invalid.remaining_q50 = tokens(80);
        invalid.remaining_q90 = tokens(60);
        assert_eq!(
            assess_completion_feasibility(invalid, &ledger, 0.90, 32),
            AdmissionDecision::Abstain(AbstentionReason::InvalidForecast)
        );

        let mut missing_digest = forecast();
        missing_digest.model_digest = [0; 32];
        assert_eq!(missing_digest.validate(), Err(ForecastError::MissingDigest));
    }

    #[test]
    fn duplicate_reservation_ids_are_rejected() {
        let mut ledger = BudgetLedger::new(tokens(100));
        let id = ReservationId(12);
        ledger.reserve(id, tokens(10)).unwrap();
        assert_eq!(
            ledger.reserve(id, tokens(10)),
            Err(LedgerError::DuplicateReservation)
        );
        assert_eq!(ledger.reserved, tokens(10));
    }
}
