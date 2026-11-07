//! Evidence-aware acquisition and lifecycle cost comparisons for marine concepts.
//!
//! This crate performs transparent arithmetic over declared estimates. It is not a
//! naval-architecture model, market-price database, approval mechanism, or proof that
//! a vessel is safe or physically validated. Synthetic values are suitable only for
//! testing the accounting contract.
//!
//! Input cost bands are conservative bounds, not probability distributions. Their
//! summation does not assume statistical independence or cancellation.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceOrigin {
    SupplierQuote,
    Invoice,
    Measured,
    Modelled,
    Assumption,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostBand {
    pub lower: f64,
    pub nominal: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostEstimate {
    /// Stable reference into a source/quote/invoice/measurement/assumption ledger.
    pub amount: CostBand,
    pub evidence_ref: Option<String>,
    pub origin: Option<EvidenceOrigin>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostLine {
    /// Stable line identity, e.g. hull_structure, propulsion, electrical or outfitting.
    pub line_id: String,
    pub estimate: CostEstimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplacementEvent {
    /// One-based year in which the cost is incurred.
    pub year: u32,
    pub component_id: String,
    pub estimate: CostEstimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostCase {
    pub case_id: String,
    pub vessel_family: String,
    /// Identifier for the same useful service contract (payload, route, duty cycle,
    /// speed/arrival policy, availability and operating envelope).
    pub service_contract_id: String,
    /// Must identify currency, price year and region/basis; e.g. ZAR-2026-SA.
    pub currency_basis_id: String,
    /// Defines included cost boundary and delivery scope; must match to compare.
    pub cost_scope_id: String,
    /// Design/engineering/qualification/tooling cost incurred once for the programme.
    pub non_recurring_program_cost: Option<CostEstimate>,
    /// Per-vessel build costs under the declared common acquisition scope.
    /// Missing is unknown, not zero. Lines should include all in-scope components.
    pub unit_build_costs: Option<Vec<CostLine>>,
    /// Annual vessel-owner operating cost per vessel under this service contract.
    pub annual_operating_cost: Option<CostEstimate>,
    /// None means the replacement plan is unknown; Some(vec![]) means explicitly none.
    pub replacement_plan: Option<Vec<ReplacementEvent>>,
    /// Net residual/re-sale value at the end of the service life. Zero must be explicit.
    pub terminal_residual_value: Option<CostEstimate>,
    pub service_life_years: u32,
    /// Useful service units delivered per year, for the declared service contract.
    pub annual_service_units: f64,
    /// Real annual discount rate as a fraction (e.g. 0.05 for 5%).
    pub discount_rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostInterval {
    pub lower: f64,
    pub nominal: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Fully loaded construction cost for the first vessel (all programme NRE on unit 1).
    pub first_unit_build_cost: CostInterval,
    /// Construction cost per vessel with programme NRE amortized over production volume.
    pub amortized_build_cost_per_unit: CostInterval,
    /// Discounted lifetime cost per vessel, including amortized build cost and residual value.
    pub lifetime_cost_per_unit: CostInterval,
    /// Discounted programme cost for all vessels at the requested production volume.
    pub program_lifetime_cost: CostInterval,
    /// Present value of the useful service units expected across the service life.
    pub discounted_service_units: f64,
    /// Lifetime cost divided by discounted service units.
    pub cost_per_service_unit: CostInterval,
    /// Exposed so break-even analysis can show its assumptions transparently.
    pub non_recurring_program_cost: CostInterval,
    pub nominal_variable_lifecycle_cost_per_unit: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonDisposition {
    CandidateRobustlyCheaperWithinDeclaredBounds,
    BaselineRobustlyCheaperWithinDeclaredBounds,
    Indeterminate,
    Incomparable,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakEven {
    /// Continuous production-volume crossover; round upward for an integer build decision.
    pub vessel_units: f64,
    /// True when the candidate becomes cheaper at volumes above the crossover.
    pub candidate_cheaper_above_crossover: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub disposition: ComparisonDisposition,
    pub baseline: Option<Evaluation>,
    pub candidate: Option<Evaluation>,
    /// Candidate minus baseline cost per discounted service unit. Bounds are conservative.
    pub candidate_minus_baseline_per_service_unit: Option<CostInterval>,
    /// Nominal crossover only; it is not a robust break-even guarantee.
    pub nominal_break_even: Option<BreakEven>,
    pub findings: Vec<String>,
}

impl CostBand {
    pub const fn exact(value: f64) -> Self {
        Self { lower: value, nominal: value, upper: value }
    }

    fn valid(self) -> bool {
        self.lower.is_finite()
            && self.nominal.is_finite()
            && self.upper.is_finite()
            && self.lower >= 0.0
            && self.lower <= self.nominal
            && self.nominal <= self.upper
    }
}

fn interval_from_band(band: CostBand) -> CostInterval {
    CostInterval { lower: band.lower, nominal: band.nominal, upper: band.upper }
}

fn add_bands(left: CostBand, right: CostBand) -> CostBand {
    CostBand {
        lower: left.lower + right.lower,
        nominal: left.nominal + right.nominal,
        upper: left.upper + right.upper,
    }
}

fn scale_band(band: CostBand, factor: f64) -> CostBand {
    CostBand {
        lower: band.lower * factor,
        nominal: band.nominal * factor,
        upper: band.upper * factor,
    }
}

fn subtract_terminal_value(cost: CostBand, residual: CostBand, discount: f64) -> CostBand {
    CostBand {
        lower: cost.lower - residual.upper * discount,
        nominal: cost.nominal - residual.nominal * discount,
        upper: cost.upper - residual.lower * discount,
    }
}

fn present_value_factor(rate: f64, year: u32) -> f64 {
    1.0 / (1.0 + rate).powf(f64::from(year))
}

fn annuity_factor(rate: f64, years: u32) -> f64 {
    (1..=years).map(|year| present_value_factor(rate, year)).sum()
}

fn check_estimate(case_id: &str, field: &str, estimate: &CostEstimate, findings: &mut Vec<String>) {
    if !estimate.amount.valid() {
        findings.push(format!("{case_id}: {field} must be finite, non-negative, and ordered lower <= nominal <= upper"));
    }
    if estimate.evidence_ref.as_deref().is_none_or(|reference| reference.trim().is_empty()) {
        findings.push(format!("{case_id}: missing evidence reference for {field}"));
    }
    if estimate.origin.is_none() {
        findings.push(format!("{case_id}: missing evidence origin for {field}"));
    }
}

fn validate_case(case: &CostCase, volume: u32) -> Vec<String> {
    let mut findings = Vec::new();
    if case.case_id.trim().is_empty() { findings.push("missing case_id".to_owned()); }
    if case.vessel_family.trim().is_empty() { findings.push(format!("{}: missing vessel_family", case.case_id)); }
    if case.service_contract_id.trim().is_empty() { findings.push(format!("{}: missing service_contract_id", case.case_id)); }
    if case.currency_basis_id.trim().is_empty() { findings.push(format!("{}: missing currency_basis_id", case.case_id)); }
    if case.cost_scope_id.trim().is_empty() { findings.push(format!("{}: missing cost_scope_id", case.case_id)); }
    if volume == 0 { findings.push("production volume must be positive".to_owned()); }
    if case.service_life_years == 0 { findings.push(format!("{}: service_life_years must be positive", case.case_id)); }
    if !case.annual_service_units.is_finite() || case.annual_service_units <= 0.0 {
        findings.push(format!("{}: annual_service_units must be finite and positive", case.case_id));
    }
    if !case.discount_rate.is_finite() || !(0.0..1.0).contains(&case.discount_rate) {
        findings.push(format!("{}: discount_rate must be finite and in [0, 1)", case.case_id));
    }

    match &case.non_recurring_program_cost {
        Some(estimate) => check_estimate(&case.case_id, "non_recurring_program_cost", estimate, &mut findings),
        None => findings.push(format!("{}: non_recurring_program_cost is unknown", case.case_id)),
    }
    match &case.unit_build_costs {
        Some(lines) if !lines.is_empty() => {
            let mut ids: Vec<&str> = Vec::new();
            for line in lines {
                if line.line_id.trim().is_empty() {
                    findings.push(format!("{}: build cost line has an empty line_id", case.case_id));
                }
                if ids.contains(&line.line_id.as_str()) {
                    findings.push(format!("{}: duplicate build cost line_id {}", case.case_id, line.line_id));
                }
                ids.push(line.line_id.as_str());
                check_estimate(&case.case_id, &format!("unit_build_costs.{}", line.line_id), &line.estimate, &mut findings);
            }
        }
        Some(_) => findings.push(format!("{}: unit_build_costs is empty; missing costs are not zero", case.case_id)),
        None => findings.push(format!("{}: unit_build_costs is unknown", case.case_id)),
    }
    match &case.annual_operating_cost {
        Some(estimate) => check_estimate(&case.case_id, "annual_operating_cost", estimate, &mut findings),
        None => findings.push(format!("{}: annual_operating_cost is unknown", case.case_id)),
    }
    match &case.replacement_plan {
        Some(events) => {
            let mut keys: Vec<(u32, &str)> = Vec::new();
            for event in events {
                if event.year == 0 || event.year > case.service_life_years {
                    findings.push(format!("{}: replacement event {} year {} is outside 1..={}", case.case_id, event.component_id, event.year, case.service_life_years));
                }
                if event.component_id.trim().is_empty() {
                    findings.push(format!("{}: replacement event has an empty component_id", case.case_id));
                }
                if keys.contains(&(event.year, event.component_id.as_str())) {
                    findings.push(format!("{}: duplicate replacement event for {} in year {}", case.case_id, event.component_id, event.year));
                }
                keys.push((event.year, event.component_id.as_str()));
                check_estimate(&case.case_id, &format!("replacement_plan.{}.year{}", event.component_id, event.year), &event.estimate, &mut findings);
            }
        }
        None => findings.push(format!("{}: replacement_plan is unknown; declare events or an explicit empty plan", case.case_id)),
    }
    match &case.terminal_residual_value {
        Some(estimate) => check_estimate(&case.case_id, "terminal_residual_value", estimate, &mut findings),
        None => findings.push(format!("{}: terminal_residual_value is unknown; zero must be explicit", case.case_id)),
    }
    findings.sort();
    findings.dedup();
    findings
}

fn ensure_finite_interval(interval: CostInterval) -> bool {
    interval.lower.is_finite()
        && interval.nominal.is_finite()
        && interval.upper.is_finite()
        && interval.lower <= interval.nominal
        && interval.nominal <= interval.upper
}

/// Evaluate a single design at a declared programme production volume.
///
/// All currency inputs must already be converted to the identical currency_basis_id.
/// Cost bands are interval bounds, not probability/confidence intervals. Annual operating
/// cost is paid at year end; replacement events are discounted at their declared year;
/// residual value is discounted at the end of service life.
/// Returns validation findings instead of substituting unknown inputs with zero.
pub fn evaluate(case: &CostCase, production_volume: u32) -> Result<Evaluation, Vec<String>> {
    let findings = validate_case(case, production_volume);
    if !findings.is_empty() {
        return Err(findings);
    }

    let nre = case.non_recurring_program_cost.as_ref().expect("validated NRE").amount;
    let lines = case.unit_build_costs.as_ref().expect("validated build lines");
    let annual_opex = case.annual_operating_cost.as_ref().expect("validated annual opex").amount;
    let replacements = case.replacement_plan.as_ref().expect("validated replacement plan");
    let residual = case.terminal_residual_value.as_ref().expect("validated residual").amount;

    let mut unit_build = CostBand::exact(0.0);
    for line in lines {
        unit_build = add_bands(unit_build, line.estimate.amount);
    }
    let first_build = add_bands(nre, unit_build);
    let amortized_build = add_bands(scale_band(nre, 1.0 / f64::from(production_volume)), unit_build);

    let annual_factor = annuity_factor(case.discount_rate, case.service_life_years);
    let opex_pv = scale_band(annual_opex, annual_factor);
    let mut replacement_pv = CostBand::exact(0.0);
    for event in replacements {
        replacement_pv = add_bands(
            replacement_pv,
            scale_band(event.estimate.amount, present_value_factor(case.discount_rate, event.year)),
        );
    }

    let operating_and_replacement = add_bands(opex_pv, replacement_pv);
    let lifecycle_before_residual = add_bands(amortized_build, operating_and_replacement);
    let residual_factor = present_value_factor(case.discount_rate, case.service_life_years);
    let lifetime = subtract_terminal_value(lifecycle_before_residual, residual, residual_factor);
    let variable_lifetime = subtract_terminal_value(
        add_bands(unit_build, operating_and_replacement),
        residual,
        residual_factor,
    );
    let program = add_bands(nre, scale_band(variable_lifetime, f64::from(production_volume)));
    let discounted_service_units = case.annual_service_units * annual_factor;
    let cost_per_service = CostInterval {
        lower: lifetime.lower / discounted_service_units,
        nominal: lifetime.nominal / discounted_service_units,
        upper: lifetime.upper / discounted_service_units,
    };

    let evaluation = Evaluation {
        first_unit_build_cost: interval_from_band(first_build),
        amortized_build_cost_per_unit: interval_from_band(amortized_build),
        lifetime_cost_per_unit: interval_from_band(lifetime),
        program_lifetime_cost: interval_from_band(program),
        discounted_service_units,
        cost_per_service_unit: cost_per_service,
        non_recurring_program_cost: interval_from_band(nre),
        nominal_variable_lifecycle_cost_per_unit: variable_lifetime.nominal,
    };

    let intervals = [
        evaluation.first_unit_build_cost,
        evaluation.amortized_build_cost_per_unit,
        evaluation.lifetime_cost_per_unit,
        evaluation.program_lifetime_cost,
        evaluation.cost_per_service_unit,
    ];
    if !evaluation.discounted_service_units.is_finite()
        || evaluation.discounted_service_units <= 0.0
        || intervals.into_iter().any(|interval| !ensure_finite_interval(interval))
        || !evaluation.nominal_variable_lifecycle_cost_per_unit.is_finite()
    {
        return Err(vec![format!("{}: derived arithmetic overflow or invalid interval", case.case_id)]);
    }
    Ok(evaluation)
}

fn compatibility_findings(baseline: &CostCase, candidate: &CostCase) -> Vec<String> {
    let mut findings = Vec::new();
    if baseline.service_contract_id != candidate.service_contract_id {
        findings.push("service_contract_id mismatch: useful service is not held constant".to_owned());
    }
    if baseline.currency_basis_id != candidate.currency_basis_id {
        findings.push("currency_basis_id mismatch: currency, price year, or region differs".to_owned());
    }
    if baseline.cost_scope_id != candidate.cost_scope_id {
        findings.push("cost_scope_id mismatch: included acquisition/lifecycle costs differ".to_owned());
    }
    if baseline.service_life_years != candidate.service_life_years {
        findings.push("service_life_years mismatch".to_owned());
    }
    if baseline.annual_service_units != candidate.annual_service_units {
        findings.push("annual_service_units mismatch".to_owned());
    }
    if baseline.discount_rate != candidate.discount_rate {
        findings.push("discount_rate mismatch".to_owned());
    }
    findings
}

fn nominal_break_even(baseline: &Evaluation, candidate: &Evaluation) -> Option<BreakEven> {
    let fixed_delta = candidate.non_recurring_program_cost.nominal - baseline.non_recurring_program_cost.nominal;
    let variable_delta_denominator =
        baseline.nominal_variable_lifecycle_cost_per_unit - candidate.nominal_variable_lifecycle_cost_per_unit;
    if !fixed_delta.is_finite() || !variable_delta_denominator.is_finite() || variable_delta_denominator == 0.0 {
        return None;
    }
    let units = fixed_delta / variable_delta_denominator;
    if !units.is_finite() || units <= 0.0 {
        return None;
    }
    Some(BreakEven {
        vessel_units: units,
        candidate_cheaper_above_crossover: variable_delta_denominator > 0.0,
    })
}

/// Compare two candidates only when the declared service, money basis, cost boundary,
/// service life and discounting assumptions match. A robust winner requires non-overlapping
/// worst-case bounds; overlapping intervals remain indeterminate.
pub fn compare(
    baseline: &CostCase,
    candidate: &CostCase,
    production_volume: u32,
) -> ComparisonReport {
    let baseline_eval = evaluate(baseline, production_volume);
    let candidate_eval = evaluate(candidate, production_volume);
    let mut findings = Vec::new();

    let baseline = match baseline_eval {
        Ok(value) => Some(value),
        Err(errors) => {
            findings.extend(errors.into_iter().map(|e| format!("baseline: {e}")));
            None
        }
    };
    let candidate = match candidate_eval {
        Ok(value) => Some(value),
        Err(errors) => {
            findings.extend(errors.into_iter().map(|e| format!("candidate: {e}")));
            None
        }
    };

    if baseline.is_none() || candidate.is_none() {
        findings.sort();
        findings.dedup();
        return ComparisonReport {
            disposition: ComparisonDisposition::Invalid,
            baseline,
            candidate,
            candidate_minus_baseline_per_service_unit: None,
            nominal_break_even: None,
            findings,
        };
    }

    let mut mismatch = compatibility_findings(baseline.as_ref().unwrap(), candidate.as_ref().unwrap());
    if !mismatch.is_empty() {
        findings.append(&mut mismatch);
        findings.sort();
        findings.dedup();
        return ComparisonReport {
            disposition: ComparisonDisposition::Incomparable,
            baseline,
            candidate,
            candidate_minus_baseline_per_service_unit: None,
            nominal_break_even: None,
            findings,
        };
    }

    let b = baseline.as_ref().unwrap();
    let c = candidate.as_ref().unwrap();
    let difference = CostInterval {
        lower: c.cost_per_service_unit.lower - b.cost_per_service_unit.upper,
        nominal: c.cost_per_service_unit.nominal - b.cost_per_service_unit.nominal,
        upper: c.cost_per_service_unit.upper - b.cost_per_service_unit.lower,
    };
    let disposition = if difference.upper < 0.0 {
        ComparisonDisposition::CandidateRobustlyCheaperWithinDeclaredBounds
    } else if difference.lower > 0.0 {
        ComparisonDisposition::BaselineRobustlyCheaperWithinDeclaredBounds
    } else {
        ComparisonDisposition::Indeterminate
    };
    let crossover = nominal_break_even(b, c);
    ComparisonReport {
        disposition,
        baseline,
        candidate,
        candidate_minus_baseline_per_service_unit: Some(difference),
        nominal_break_even: crossover,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(value: f64, id: &str) -> CostEstimate {
        CostEstimate {
            amount: CostBand::exact(value),
            evidence_ref: Some(format!("synthetic:{id}")),
            origin: Some(EvidenceOrigin::Synthetic),
        }
    }

    fn case(id: &str, nre: f64, build: f64, opex: f64, replacement: f64, residual: f64) -> CostCase {
        CostCase {
            case_id: id.to_owned(),
            vessel_family: id.to_owned(),
            service_contract_id: "same-mission-v1".to_owned(),
            currency_basis_id: "ZAR-2026-SA".to_owned(),
            cost_scope_id: "yard-gate-plus-service-v1".to_owned(),
            non_recurring_program_cost: Some(estimate(nre, &format!("{id}-nre"))),
            unit_build_costs: Some(vec![
                CostLine { line_id: "hull-and-fabrication".to_owned(), estimate: estimate(build * 0.7, &format!("{id}-hull")) },
                CostLine { line_id: "propulsion-and-outfit".to_owned(), estimate: estimate(build * 0.3, &format!("{id}-equipment")) },
            ]),
            annual_operating_cost: Some(estimate(opex, &format!("{id}-opex"))),
            replacement_plan: Some(if replacement == 0.0 {
                Vec::new()
            } else {
                vec![ReplacementEvent {
                    year: 3,
                    component_id: "battery-module".to_owned(),
                    estimate: estimate(replacement, &format!("{id}-replacement")),
                }]
            }),
            terminal_residual_value: Some(estimate(residual, &format!("{id}-residual"))),
            service_life_years: 5,
            annual_service_units: 200.0,
            discount_rate: 0.0,
        }
    }

    #[test]
    fn computes_first_unit_amortized_build_and_lifecycle_cost_separately() {
        let baseline = case("baseline", 100_000.0, 70_000.0, 10_000.0, 4_000.0, 10_000.0);
        let result = evaluate(&baseline, 10).unwrap();
        assert_eq!(result.first_unit_build_cost.nominal, 170_000.0);
        assert_eq!(result.amortized_build_cost_per_unit.nominal, 80_000.0);
        assert_eq!(result.lifetime_cost_per_unit.nominal, 124_000.0);
        assert_eq!(result.program_lifetime_cost.nominal, 1_240_000.0);
        assert_eq!(result.discounted_service_units, 1_000.0);
        assert_eq!(result.cost_per_service_unit.nominal, 124.0);
    }

    #[test]
    fn cheaper_build_can_lose_on_whole_life_cost() {
        let baseline = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        let candidate = case("candidate", 100.0, 90.0, 13.0, 0.0, 0.0);
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::BaselineRobustlyCheaperWithinDeclaredBounds);
        let delta = result.candidate_minus_baseline_per_service_unit.unwrap().nominal;
        assert!((delta - 0.005).abs() < 1e-12);
    }

    #[test]
    fn reports_robust_candidate_saving_only_for_non_overlapping_bounds() {
        let mut baseline = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        baseline.unit_build_costs.as_mut().unwrap()[0].estimate.amount =
            CostBand { lower: 68.0, nominal: 70.0, upper: 72.0 };
        let mut candidate = case("candidate", 100.0, 70.0, 8.0, 0.0, 0.0);
        candidate.unit_build_costs.as_mut().unwrap()[0].estimate.amount =
            CostBand { lower: 48.0, nominal: 49.0, upper: 50.0 };
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::CandidateRobustlyCheaperWithinDeclaredBounds);
    }

    #[test]
    fn overlapping_ranges_are_indeterminate_not_a_win() {
        let mut baseline = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        baseline.unit_build_costs.as_mut().unwrap()[0].estimate.amount =
            CostBand { lower: 68.0, nominal: 70.0, upper: 85.0 };
        let mut candidate = case("candidate", 100.0, 90.0, 10.0, 0.0, 0.0);
        candidate.unit_build_costs.as_mut().unwrap()[0].estimate.amount =
            CostBand { lower: 62.0, nominal: 63.0, upper: 80.0 };
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::Indeterminate);
    }

    #[test]
    fn missing_residual_is_not_silently_treated_as_zero() {
        let mut value = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        value.terminal_residual_value = None;
        let errors = evaluate(&value, 10).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("terminal_residual_value is unknown")));
    }

    #[test]
    fn mismatched_service_contract_is_incomparable() {
        let baseline = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        let mut candidate = case("candidate", 100.0, 90.0, 10.0, 0.0, 0.0);
        candidate.service_contract_id = "faster-and-heavier-service-v2".to_owned();
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
    }

    #[test]
    fn exposes_break_even_when_nre_and_recurring_cost_trade_off() {
        let baseline = case("baseline", 100.0, 100.0, 0.0, 0.0, 0.0);
        let candidate = case("candidate", 250.0, 50.0, 0.0, 0.0, 0.0);
        let result = compare(&baseline, &candidate, 10);
        let break_even = result.nominal_break_even.unwrap();
        assert_eq!(break_even.vessel_units, 3.0);
        assert!(break_even.candidate_cheaper_above_crossover);
    }
    
    #[test]
    fn lower_nre_but_higher_recurring_cost_is_cheaper_only_below_crossover() {
        let baseline = case("baseline", 250.0, 50.0, 0.0, 0.0, 0.0);
        let candidate = case("candidate", 100.0, 100.0, 0.0, 0.0, 0.0);
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::BaselineRobustlyCheaperWithinDeclaredBounds);
        let break_even = result.nominal_break_even.unwrap();
        assert_eq!(break_even.vessel_units, 3.0);
        assert!(!break_even.candidate_cheaper_above_crossover);
    }

    #[test]
    fn zero_production_volume_is_invalid() {
        let value = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        let errors = evaluate(&value, 0).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("production volume must be positive")));
    }

    #[test]
    fn reversed_cost_interval_is_invalid() {
        let mut value = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        value.annual_operating_cost.as_mut().unwrap().amount =
            CostBand { lower: 12.0, nominal: 10.0, upper: 14.0 };
        let errors = evaluate(&value, 10).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("ordered lower <= nominal <= upper")));
    }

    #[test]
    fn replacement_outside_service_life_is_invalid() {
        let mut value = case("baseline", 100.0, 100.0, 10.0, 5.0, 0.0);
        value.replacement_plan.as_mut().unwrap()[0].year = 6;
        let errors = evaluate(&value, 10).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("outside 1..=5")));
    }

    #[test]
    fn missing_source_reference_and_origin_are_not_accepted() {
        let mut value = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        let operating = value.annual_operating_cost.as_mut().unwrap();
        operating.evidence_ref = None;
        operating.origin = None;
        let errors = evaluate(&value, 10).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("missing evidence reference for annual_operating_cost")));
        assert!(errors.iter().any(|error| error.contains("missing evidence origin for annual_operating_cost")));
    }

    #[test]
    fn mismatched_currency_basis_is_incomparable() {
        let baseline = case("baseline", 100.0, 100.0, 10.0, 0.0, 0.0);
        let mut candidate = case("candidate", 100.0, 90.0, 10.0, 0.0, 0.0);
        candidate.currency_basis_id = "CAD-2024-CA".to_owned();
        let result = compare(&baseline, &candidate, 10);
        assert_eq!(result.disposition, ComparisonDisposition::Incomparable);
        assert!(result.findings.iter().any(|finding| finding.contains("currency_basis_id mismatch")));
    }

    #[test]
    fn derived_overflow_is_rejected() {
        let mut value = case("baseline", 0.0, 1.0, 0.0, 0.0, 0.0);
        value.unit_build_costs = Some(vec![
            CostLine { line_id: "hull".to_owned(), estimate: estimate(f64::MAX, "hull") },
            CostLine { line_id: "equipment".to_owned(), estimate: estimate(f64::MAX, "equipment") },
        ]);
        let errors = evaluate(&value, 10).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("derived arithmetic overflow or invalid interval")));
    }

}