//! Dependency-free reliability outcome reference model.
//! Research model only; not a device simulator.
//! Test: rustc --edition=2021 --test reliability_contract_v1.rs -o /tmp/rc && /tmp/rc

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cell { Healthy, StuckOff, StuckOn }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Domain { Power, Clock, Reset, Controller, Interconnect }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error { Empty, Overflow, AllocationFailed, OutOfRange, InvalidRate, InvalidRectangle }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Infrastructure {
    pub power: bool,
    pub clock: bool,
    pub reset: bool,
    pub controller: bool,
    pub interconnect: bool,
}

impl Default for Infrastructure {
    fn default() -> Self {
        Self { power: true, clock: true, reset: true, controller: true, interconnect: true }
    }
}

impl Infrastructure {
    pub fn fail(&mut self, domain: Domain) {
        match domain {
            Domain::Power => self.power = false,
            Domain::Clock => self.clock = false,
            Domain::Reset => self.reset = false,
            Domain::Controller => self.controller = false,
            Domain::Interconnect => self.interconnect = false,
        }
    }
    pub fn available(self) -> bool {
        self.power && self.clock && self.reset && self.controller && self.interconnect
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FaultMap {
    rows: usize,
    cols: usize,
    cells: Vec<Cell>,
    failed_rows: Vec<bool>,
    failed_cols: Vec<bool>,
    pub infrastructure: Infrastructure,
}

impl FaultMap {
    pub fn new(rows: usize, cols: usize) -> Result<Self, Error> {
        if rows == 0 || cols == 0 { return Err(Error::Empty); }
        let n = rows.checked_mul(cols).ok_or(Error::Overflow)?;
        // Fallible reservations avoid an unconditional allocator panic for a
        // non-overflowing but unreasonable synthetic map.
        let mut cells = Vec::new();
        cells.try_reserve_exact(n).map_err(|_| Error::AllocationFailed)?;
        cells.resize(n, Cell::Healthy);
        let mut failed_rows = Vec::new();
        failed_rows.try_reserve_exact(rows).map_err(|_| Error::AllocationFailed)?;
        failed_rows.resize(rows, false);
        let mut failed_cols = Vec::new();
        failed_cols.try_reserve_exact(cols).map_err(|_| Error::AllocationFailed)?;
        failed_cols.resize(cols, false);
        Ok(Self { rows, cols, cells, failed_rows, failed_cols,
            infrastructure: Infrastructure::default() })
    }

    fn index(&self, row: usize, col: usize) -> Result<usize, Error> {
        if row >= self.rows || col >= self.cols { return Err(Error::OutOfRange); }
        Ok(row * self.cols + col)
    }

    pub fn cell(&self, row: usize, col: usize) -> Result<Cell, Error> {
        Ok(self.cells[self.index(row, col)?])
    }

    pub fn set_cell(&mut self, row: usize, col: usize, state: Cell) -> Result<(), Error> {
        let i = self.index(row, col)?;
        self.cells[i] = state;
        Ok(())
    }

    pub fn fail_row(&mut self, row: usize) -> Result<(), Error> {
        *self.failed_rows.get_mut(row).ok_or(Error::OutOfRange)? = true;
        Ok(())
    }

    pub fn fail_column(&mut self, col: usize) -> Result<(), Error> {
        *self.failed_cols.get_mut(col).ok_or(Error::OutOfRange)? = true;
        Ok(())
    }

    pub fn fail_rectangle(
        &mut self, row: usize, col: usize, height: usize, width: usize, state: Cell,
    ) -> Result<(), Error> {
        if height == 0 || width == 0 { return Err(Error::InvalidRectangle); }
        let r_end = row.checked_add(height).ok_or(Error::InvalidRectangle)?;
        let c_end = col.checked_add(width).ok_or(Error::InvalidRectangle)?;
        if r_end > self.rows || c_end > self.cols { return Err(Error::InvalidRectangle); }
        for r in row..r_end {
            for c in col..c_end { self.set_cell(r, c, state)?; }
        }
        Ok(())
    }

    /// Reproducible synthetic injection. Rates are per-mille; their sum <= 1000.
    /// The xorshift generator is for deterministic fixtures, not scientific RNG.
    pub fn inject_independent(
        &mut self, stuck_off: u16, stuck_on: u16, seed: u64,
    ) -> Result<(), Error> {
        if stuck_off > 1000 || stuck_on > 1000
            || u32::from(stuck_off) + u32::from(stuck_on) > 1000 {
            return Err(Error::InvalidRate);
        }
        let mut rng = Rng::new(seed);
        for cell in &mut self.cells {
            let draw = (rng.next() % 1000) as u16;
            *cell = if draw < stuck_off { Cell::StuckOff }
                else if draw < stuck_off + stuck_on { Cell::StuckOn }
                else { Cell::Healthy };
        }
        Ok(())
    }

    /// Stuck-on cells remain physically present but faulty; mapping/checking
    /// must handle them. Row/column loss and stuck-off remove cell availability.
    pub fn present(&self, row: usize, col: usize) -> Result<bool, Error> {
        let i = self.index(row, col)?;
        Ok(!self.failed_rows[row] && !self.failed_cols[col] && self.cells[i] != Cell::StuckOff)
    }

    pub fn healthy_cells(&self) -> usize {
        let mut total = 0;
        for r in 0..self.rows {
            if self.failed_rows[r] { continue; }
            for c in 0..self.cols {
                if !self.failed_cols[c] && self.cells[r * self.cols + c] == Cell::Healthy {
                    total += 1;
                }
            }
        }
        total
    }

    pub fn rows_intact(&self) -> bool { !self.failed_rows.iter().any(|x| *x) }
    pub fn columns_intact(&self) -> bool { !self.failed_cols.iter().any(|x| *x) }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Contract {
    pub deadline_ticks: u64,
    pub minimum_healthy_cells: usize,
    pub require_all_rows: bool,
    pub require_all_columns: bool,
}

impl Contract {
    /// Resource-only preflight. This does not prove that a workload can be mapped.
    pub fn resources_sufficient(self, map: &FaultMap) -> bool {
        map.healthy_cells() >= self.minimum_healthy_cells
            && (!self.require_all_rows || map.rows_intact())
            && (!self.require_all_columns || map.columns_intact())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verification { Passed, Rejected, Unavailable }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbstentionReason {
    NoQualifiedMapping,
    EvidenceStale,
    RiskBudgetExceeded,
    UnsupportedWorkload,
    OperatorRequested,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttemptDisposition {
    Attempted,
    Abstained(AbstentionReason),
}

/// Evaluator-only label. It is deliberately separate from verifier-facing data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluatorLabel {
    pub ground_truth_correct: bool,
}

/// Verifier-facing output and timing. No evaluator ground truth is carried here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Observation {
    pub disposition: AttemptDisposition,
    pub verification: Verification,
    pub elapsed_ticks: u64,
    pub recovery: Recovery,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Recovery { NotAttempted, Succeeded, Failed }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    CorrectAcceptedOnTime,
    SilentCorruptionAccepted,
    DetectedRejected,
    VerificationUnavailable,
    DeadlineMiss,
    HardwareUnavailable,
    ResourceInfeasible,
    Abstained,
    Unverifiable,
}

pub fn classify(
    contract: Contract,
    map: &FaultMap,
    o: Observation,
    evaluator_label: Option<EvaluatorLabel>,
) -> Outcome {
    if matches!(o.disposition, AttemptDisposition::Abstained(_)) {
        return Outcome::Abstained;
    }
    if !map.infrastructure.available() { return Outcome::HardwareUnavailable; }
    if !contract.resources_sufficient(map) { return Outcome::ResourceInfeasible; }
    if o.elapsed_ticks > contract.deadline_ticks { return Outcome::DeadlineMiss; }
    match o.verification {
        Verification::Rejected => Outcome::DetectedRejected,
        Verification::Unavailable => Outcome::VerificationUnavailable,
        Verification::Passed => match evaluator_label {
            Some(label) if label.ground_truth_correct => Outcome::CorrectAcceptedOnTime,
            Some(_) => Outcome::SilentCorruptionAccepted,
            None => Outcome::Unverifiable,
        },
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Counts {
    pub correct_accepted: u64,
    pub silent_corruption: u64,
    pub detected_rejected: u64,
    pub verification_unavailable: u64,
    pub deadline_miss: u64,
    pub hardware_unavailable: u64,
    pub resource_infeasible: u64,
    pub abstained: u64,
    pub unverifiable: u64,
    pub recovery_succeeded: u64,
    pub recovery_failed: u64,
    pub recovery_not_attempted: u64,
}

impl Counts {
    pub fn record(&mut self, outcome: Outcome, recovery: Recovery) {
        match outcome {
            Outcome::CorrectAcceptedOnTime => self.correct_accepted += 1,
            Outcome::SilentCorruptionAccepted => self.silent_corruption += 1,
            Outcome::DetectedRejected => self.detected_rejected += 1,
            Outcome::VerificationUnavailable => self.verification_unavailable += 1,
            Outcome::DeadlineMiss => self.deadline_miss += 1,
            Outcome::HardwareUnavailable => self.hardware_unavailable += 1,
            Outcome::ResourceInfeasible => self.resource_infeasible += 1,
            Outcome::Abstained => self.abstained += 1,
            Outcome::Unverifiable => self.unverifiable += 1,
        }
        match recovery {
            Recovery::Succeeded => self.recovery_succeeded += 1,
            Recovery::Failed => self.recovery_failed += 1,
            Recovery::NotAttempted => self.recovery_not_attempted += 1,
        }
    }

    pub fn total(&self) -> u64 {
        self.correct_accepted + self.silent_corruption + self.detected_rejected
            + self.verification_unavailable + self.deadline_miss
            + self.hardware_unavailable + self.resource_infeasible
            + self.abstained + self.unverifiable
    }

    /// Correct accepted results per recorded scheduled trial. Callers must
    /// record every scheduled trial exactly once, including abstentions.
    pub fn correct_accepted_per_scheduled_trial(&self) -> Option<f64> {
        let n = self.total();
        if n == 0 { None } else { Some(self.correct_accepted as f64 / n as f64) }
    }
}

/// Exact one-sided upper failure bound for zero failures in n independent
/// Bernoulli experimental units at confidence 1-alpha.
pub fn zero_failure_upper_bound(n: u64, alpha: f64) -> Option<f64> {
    if n == 0 || !(alpha > 0.0 && alpha < 1.0) { return None; }
    // exp_m1 avoids catastrophic cancellation when the true bound is tiny.
    Some(-((alpha.ln() / n as f64).exp_m1()))
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> Contract {
        Contract { deadline_ticks: 100, minimum_healthy_cells: 1,
            require_all_rows: true, require_all_columns: true }
    }

    fn observation(v: Verification, elapsed: u64) -> Observation {
        Observation { disposition: AttemptDisposition::Attempted,
            verification: v, elapsed_ticks: elapsed, recovery: Recovery::NotAttempted }
    }

    fn label(correct: bool) -> Option<EvaluatorLabel> {
        Some(EvaluatorLabel { ground_truth_correct: correct })
    }

    #[test]
    fn dimensions_and_rate_inputs_are_checked() {
        assert_eq!(FaultMap::new(0, 1), Err(Error::Empty));
        assert_eq!(FaultMap::new(usize::MAX, 2), Err(Error::Overflow));
        let mut m = FaultMap::new(2, 2).unwrap();
        assert_eq!(m.inject_independent(800, 300, 1), Err(Error::InvalidRate));
    }

    #[test]
    fn fixed_seed_produces_identical_fault_maps() {
        let mut a = FaultMap::new(16, 16).unwrap();
        let mut b = FaultMap::new(16, 16).unwrap();
        a.inject_independent(390, 50, 0xC0FFEE).unwrap();
        b.inject_independent(390, 50, 0xC0FFEE).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn stuck_on_cells_are_present_but_not_healthy() {
        let mut m = FaultMap::new(4, 4).unwrap();
        m.set_cell(0, 0, Cell::StuckOn).unwrap();
        assert_eq!(m.present(0, 0), Ok(true));
        assert_eq!(m.healthy_cells(), 15);
    }

    #[test]
    fn cluster_and_output_failures_have_distinct_geometry() {
        let mut m = FaultMap::new(8, 8).unwrap();
        m.fail_rectangle(2, 3, 2, 3, Cell::StuckOff).unwrap();
        assert_eq!(m.healthy_cells(), 58);
        assert_eq!(m.cell(2, 3).unwrap(), Cell::StuckOff);
        m.fail_row(0).unwrap();
        m.fail_column(7).unwrap();
        assert!(!m.rows_intact());
        assert!(!m.columns_intact());
        assert_eq!(m.fail_row(8), Err(Error::OutOfRange));
    }

    #[test]
    fn shared_domain_failure_cannot_be_fixed_by_more_cells() {
        let mut m = FaultMap::new(8, 8).unwrap();
        m.infrastructure.fail(Domain::Power);
        let o = classify(contract(), &m, observation(Verification::Passed, 1), label(true));
        assert_eq!(o, Outcome::HardwareUnavailable);
    }

    #[test]
    fn rejection_and_silent_corruption_are_not_conflated() {
        let m = FaultMap::new(4, 4).unwrap();
        assert_eq!(classify(contract(), &m, observation(Verification::Rejected, 5), label(false)),
            Outcome::DetectedRejected);
        assert_eq!(classify(contract(), &m, observation(Verification::Passed, 5), label(false)),
            Outcome::SilentCorruptionAccepted);
    }

    #[test]
    fn verifier_unavailable_and_deadline_miss_fail_the_contract() {
        let m = FaultMap::new(4, 4).unwrap();
        assert_eq!(classify(contract(), &m, observation(Verification::Unavailable, 5), label(true)),
            Outcome::VerificationUnavailable);
        assert_eq!(classify(contract(), &m, observation(Verification::Passed, 101), label(true)),
            Outcome::DeadlineMiss);
        // The deadline boundary itself is accepted.
        assert_eq!(classify(contract(), &m, observation(Verification::Passed, 100), label(true)),
            Outcome::CorrectAcceptedOnTime);
    }

    #[test]
    fn outcome_counts_keep_recovery_orthogonal() {
        let mut c = Counts::default();
        c.record(Outcome::CorrectAcceptedOnTime, Recovery::Succeeded);
        c.record(Outcome::SilentCorruptionAccepted, Recovery::Failed);
        assert_eq!(c.total(), 2);
        assert_eq!(c.correct_accepted_per_scheduled_trial(), Some(0.5));
        assert_eq!(c.recovery_succeeded, 1);
        assert_eq!(c.recovery_failed, 1);
    }

    #[test]
    fn zero_failure_bound_uses_exact_closed_form() {
        let p299 = zero_failure_upper_bound(299, 0.05).unwrap();
        let p2995 = zero_failure_upper_bound(2995, 0.05).unwrap();
        assert!((p299 - 0.00997).abs() < 0.00002);
        assert!((p2995 - 0.001).abs() < 0.00001);
        assert!(zero_failure_upper_bound(0, 0.05).is_none());
        assert!(zero_failure_upper_bound(10, 1.0).is_none());
        assert!(zero_failure_upper_bound(10, 0.0).is_none());
        assert!(zero_failure_upper_bound(u64::MAX, 0.05).unwrap() > 0.0);
    }

    #[test]
    fn zero_seed_is_non_degenerate_and_reproducible() {
        let mut a = FaultMap::new(16, 16).unwrap();
        let mut b = FaultMap::new(16, 16).unwrap();
        a.inject_independent(500, 0, 0).unwrap();
        b.inject_independent(500, 0, 0).unwrap();
        assert_eq!(a, b);
        assert!(a.healthy_cells() > 0);
        assert!(a.healthy_cells() < 256);
    }

    #[test]
    fn abstention_and_resource_infeasibility_are_not_hardware_failure() {
        let m = FaultMap::new(4, 4).unwrap();
        let abstained = Observation {
            disposition: AttemptDisposition::Abstained(AbstentionReason::EvidenceStale),
            verification: Verification::Unavailable, elapsed_ticks: 0,
            recovery: Recovery::NotAttempted,
        };
        assert_eq!(classify(contract(), &m, abstained, None), Outcome::Abstained);

        let mut c = contract();
        c.minimum_healthy_cells = 17;
        assert_eq!(classify(c, &m, observation(Verification::Passed, 1), label(true)),
            Outcome::ResourceInfeasible);
    }
}
