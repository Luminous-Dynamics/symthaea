//! Offline, synthetic-first evaluator for the HEAVY-OIL-RESCUE-001 research track.
//!
//! This module is an economic-screening and admissibility prototype, not a chemistry,
//! refinery, process-safety, lifecycle-assessment, or commercial-validation model.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub const fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub const fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    fn is_finite_ordered(self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }

    fn is_nonnegative(self) -> bool {
        self.lower >= 0.0
    }

    fn is_point(self) -> bool {
        self.lower == self.upper
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalanceTolerance {
    pub absolute_kg: f64,
    pub relative_fraction: f64,
}

impl Default for BalanceTolerance {
    fn default() -> Self {
        Self {
            absolute_kg: 0.01,
            relative_fraction: 1.0e-6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Pass,
    Fail,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassBalanceStatus {
    /// Both sides are point values and close within the declared tolerance.
    ExactBalanced,
    /// Both sides are point values and differ beyond the declared tolerance.
    ExactUnbalanced,
    /// Uncertain input/output ranges overlap; closure is possible, not established.
    PossibleWithUncertainty,
    /// Input/output intervals are disjoint beyond the declared tolerance.
    ImpossibleWithinBounds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceTier {
    /// Software fixtures only. This tier can never establish physical or commercial viability.
    Synthetic,
    /// Literature/simulator reproduction with applicability still bounded to that evidence.
    LiteratureReproduced,
    /// Laboratory/bench measurement, not yet independently reproduced.
    BenchMeasured,
    /// Independently replicated within a named, versioned experimental envelope.
    IndependentlyReplicated,
    /// Observed commercial operation with its declared operating boundary and time window.
    CommercialObserved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EconomicsStatus {
    /// Conservative netback lower bound is strictly above the hurdle.
    AboveHurdle,
    /// Optimistic netback upper bound is at or below the hurdle.
    BelowHurdle,
    /// Netback interval crosses the hurdle.
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admissibility {
    Eligible,
    Blocked,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockerDisposition {
    Fail,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blocker {
    pub name: String,
    pub disposition: BlockerDisposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conclusion {
    /// One or more hard gates failed.
    Blocked,
    /// At least one hard gate or the mass balance remains unknown.
    Indeterminate,
    /// Hard gates pass, but even the optimistic economics miss the hurdle.
    NotEconomicallyViableWithinBounds,
    /// Hard gates pass, but the netback interval crosses the hurdle.
    EconomicallyIndeterminate,
    /// Positive arithmetic on synthetic fixtures only; not a physical result.
    SyntheticOnly,
    /// Positive screen based on literature/simulator assumptions only.
    LiteratureScreenOnly,
    /// Bench evidence exists, but independent replication is still required.
    BenchMeasuredNeedsIndependentReplication,
    /// Independent result within its stated experimental and model boundary.
    IndependentlyReplicatedWithinDeclaredBoundary,
    /// Commercial evidence within the declared asset, process and time boundary.
    CommercialEvidenceWithinDeclaredBoundary,
}

#[derive(Clone, Debug)]
pub struct MaterialFlow {
    pub name: String,
    /// Kilograms crossing the process boundary for this case's declared feed basis.
    pub mass_kg: Interval,
}

#[derive(Clone, Debug)]
pub struct OutputStream {
    pub name: String,
    /// Kilograms crossing the process boundary for this case's declared feed basis.
    pub mass_kg: Interval,
    /// Sale price in USD per metric tonne of this output. None means no sale revenue
    /// is credited; disposal/recovery costs belong in the cost ledger.
    pub price_usd_per_tonne: Option<Interval>,
}

#[derive(Clone, Debug)]
pub struct CostLine {
    pub name: String,
    /// Cost in USD per metric tonne of feedstock, already normalized to that basis.
    pub usd_per_feed_tonne: Interval,
}

#[derive(Clone, Debug)]
pub struct RouteCase {
    pub id: String,
    /// Kilograms of feedstock represented by this case. Must match the input flow
    /// named "feedstock". Economic outputs are normalized per metric tonne of feed.
    pub feed_basis_kg: f64,
    pub inputs: Vec<MaterialFlow>,
    pub outputs: Vec<OutputStream>,
    pub costs: Vec<CostLine>,
    pub minimum_netback_usd_per_feed_tonne: f64,
    pub energy_balance_gate: Gate,
    pub product_quality_gate: Gate,
    pub environmental_gate: Gate,
    pub process_safety_gate: Gate,
    pub model_applicability_gate: Gate,
    pub evidence_tier: EvidenceTier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalErrorCode {
    MissingRouteId,
    InvalidRouteId,
    EmptyInputs,
    EmptyOutputs,
    EmptyCosts,
    InvalidInterval,
    NegativeMass,
    InvalidFeedBasis,
    MissingFeedstockFlow,
    FeedBasisMismatch,
    DuplicateName,
    InvalidCost,
    InvalidHurdle,
    InvalidTolerance,
    NonFiniteDerivedValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalError {
    pub code: EvalErrorCode,
    pub field: String,
}

impl EvalError {
    fn new(code: EvalErrorCode, field: impl Into<String>) -> Self {
        Self {
            code,
            field: field.into(),
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.field)
    }
}

impl Error for EvalError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Assessment {
    pub route_id: String,
    pub input_mass_kg: Interval,
    pub output_mass_kg: Interval,
    pub mass_balance: MassBalanceStatus,
    pub revenue_usd_per_feed_tonne: Interval,
    pub cost_usd_per_feed_tonne: Interval,
    pub netback_usd_per_feed_tonne: Interval,
    pub economics: EconomicsStatus,
    pub admissibility: Admissibility,
    pub blockers: Vec<Blocker>,
    pub conclusion: Conclusion,
}

pub fn evaluate(
    route: &RouteCase,
    tolerance: BalanceTolerance,
) -> Result<Assessment, EvalError> {
    validate_route(route)?;
    validate_tolerance(tolerance)?;

    let input_mass_kg = sum_input_mass(&route.inputs)?;
    let output_mass_kg = sum_output_mass(&route.outputs)?;
    let mass_balance = classify_mass_balance(input_mass_kg, output_mass_kg, tolerance)?;

    let revenue_usd_per_feed_tonne = sum_revenue(&route.outputs, route.feed_basis_kg)?;
    let cost_usd_per_feed_tonne = sum_costs(&route.costs)?;
    let netback_usd_per_feed_tonne = Interval::new(
        revenue_usd_per_feed_tonne.lower - cost_usd_per_feed_tonne.upper,
        revenue_usd_per_feed_tonne.upper - cost_usd_per_feed_tonne.lower,
    );

    if !netback_usd_per_feed_tonne.is_finite_ordered() {
        return Err(EvalError::new(
            EvalErrorCode::NonFiniteDerivedValue,
            "netback_usd_per_feed_tonne",
        ));
    }

    let economics = if netback_usd_per_feed_tonne.lower
        > route.minimum_netback_usd_per_feed_tonne
    {
        EconomicsStatus::AboveHurdle
    } else if netback_usd_per_feed_tonne.upper
        <= route.minimum_netback_usd_per_feed_tonne
    {
        EconomicsStatus::BelowHurdle
    } else {
        EconomicsStatus::Indeterminate
    };

    let blockers = collect_blockers(route, mass_balance);
    let admissibility = classify_admissibility(&blockers);
    let conclusion = classify_conclusion(admissibility, economics, route.evidence_tier);

    Ok(Assessment {
        route_id: route.id.clone(),
        input_mass_kg,
        output_mass_kg,
        mass_balance,
        revenue_usd_per_feed_tonne,
        cost_usd_per_feed_tonne,
        netback_usd_per_feed_tonne,
        economics,
        admissibility,
        blockers,
        conclusion,
    })
}

fn validate_route(route: &RouteCase) -> Result<(), EvalError> {
    if route.id.trim().is_empty() {
        return Err(EvalError::new(EvalErrorCode::MissingRouteId, "id"));
    }
    if !route
        .id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err(EvalError::new(EvalErrorCode::InvalidRouteId, "id"));
    }
    if route.inputs.is_empty() {
        return Err(EvalError::new(EvalErrorCode::EmptyInputs, "inputs"));
    }
    if route.outputs.is_empty() {
        return Err(EvalError::new(EvalErrorCode::EmptyOutputs, "outputs"));
    }
    if route.costs.is_empty() {
        return Err(EvalError::new(EvalErrorCode::EmptyCosts, "costs"));
    }
    if !route.feed_basis_kg.is_finite() || route.feed_basis_kg <= 0.0 {
        return Err(EvalError::new(
            EvalErrorCode::InvalidFeedBasis,
            "feed_basis_kg",
        ));
    }
    if !route.minimum_netback_usd_per_feed_tonne.is_finite() {
        return Err(EvalError::new(
            EvalErrorCode::InvalidHurdle,
            "minimum_netback_usd_per_feed_tonne",
        ));
    }

    validate_unique_names(route.inputs.iter().map(|flow| flow.name.as_str()), "inputs")?;
    validate_unique_names(
        route.outputs.iter().map(|flow| flow.name.as_str()),
        "outputs",
    )?;
    validate_unique_names(route.costs.iter().map(|line| line.name.as_str()), "costs")?;

    for flow in &route.inputs {
        validate_name(&flow.name, "inputs.name")?;
        validate_mass_interval(flow.mass_kg, &format!("inputs.{}.mass_kg", flow.name))?;
    }

    let feedstock_flows: Vec<&MaterialFlow> = route
        .inputs
        .iter()
        .filter(|flow| flow.name == "feedstock")
        .collect();
    if feedstock_flows.len() != 1 {
        return Err(EvalError::new(
            EvalErrorCode::MissingFeedstockFlow,
            "inputs must contain exactly one flow named feedstock",
        ));
    }
    let feed_mass = feedstock_flows[0].mass_kg;
    if !feed_mass.is_point() || feed_mass.lower != route.feed_basis_kg {
        return Err(EvalError::new(
            EvalErrorCode::FeedBasisMismatch,
            "feedstock input must be a point value equal to feed_basis_kg",
        ));
    }

    for flow in &route.outputs {
        validate_name(&flow.name, "outputs.name")?;
        validate_mass_interval(flow.mass_kg, &format!("outputs.{}.mass_kg", flow.name))?;
        if let Some(price) = flow.price_usd_per_tonne {
            validate_nonnegative_interval(price, &format!("outputs.{}.price", flow.name))?;
        }
    }

    for line in &route.costs {
        validate_name(&line.name, "costs.name")?;
        validate_nonnegative_interval(
            line.usd_per_feed_tonne,
            &format!("costs.{}", line.name),
        )
        .map_err(|_| EvalError::new(EvalErrorCode::InvalidCost, format!("costs.{}", line.name)))?;
    }

    Ok(())
}

fn validate_tolerance(tolerance: BalanceTolerance) -> Result<(), EvalError> {
    if !tolerance.absolute_kg.is_finite()
        || !tolerance.relative_fraction.is_finite()
        || tolerance.absolute_kg < 0.0
        || tolerance.relative_fraction < 0.0
    {
        return Err(EvalError::new(
            EvalErrorCode::InvalidTolerance,
            "balance tolerance must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn validate_name(name: &str, field: &str) -> Result<(), EvalError> {
    if name.trim().is_empty() {
        return Err(EvalError::new(
            EvalErrorCode::InvalidInterval,
            format!("{field}: empty name"),
        ));
    }
    Ok(())
}

fn validate_unique_names<'a>(
    names: impl Iterator<Item = &'a str>,
    field: &str,
) -> Result<(), EvalError> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name.to_owned()) {
            return Err(EvalError::new(
                EvalErrorCode::DuplicateName,
                format!("{field}.{name}"),
            ));
        }
    }
    Ok(())
}

fn validate_mass_interval(value: Interval, field: &str) -> Result<(), EvalError> {
    if !value.is_finite_ordered() {
        return Err(EvalError::new(
            EvalErrorCode::InvalidInterval,
            field.to_owned(),
        ));
    }
    if !value.is_nonnegative() {
        return Err(EvalError::new(
            EvalErrorCode::NegativeMass,
            field.to_owned(),
        ));
    }
    Ok(())
}

fn validate_nonnegative_interval(value: Interval, field: &str) -> Result<(), EvalError> {
    if !value.is_finite_ordered() || !value.is_nonnegative() {
        return Err(EvalError::new(
            EvalErrorCode::InvalidInterval,
            field.to_owned(),
        ));
    }
    Ok(())
}

fn sum_input_mass(inputs: &[MaterialFlow]) -> Result<Interval, EvalError> {
    let mut ordered: Vec<&MaterialFlow> = inputs.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    let mut lower = 0.0;
    let mut upper = 0.0;
    for flow in ordered {
        lower += flow.mass_kg.lower;
        upper += flow.mass_kg.upper;
    }
    checked_interval(lower, upper, "input_mass_kg")
}

fn sum_output_mass(outputs: &[OutputStream]) -> Result<Interval, EvalError> {
    let mut ordered: Vec<&OutputStream> = outputs.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    let mut lower = 0.0;
    let mut upper = 0.0;
    for flow in ordered {
        lower += flow.mass_kg.lower;
        upper += flow.mass_kg.upper;
    }
    checked_interval(lower, upper, "output_mass_kg")
}

fn classify_mass_balance(
    input: Interval,
    output: Interval,
    tolerance: BalanceTolerance,
) -> Result<MassBalanceStatus, EvalError> {
    let scale = input.upper.abs().max(output.upper.abs());
    let allowed = tolerance.absolute_kg + tolerance.relative_fraction * scale;
    if !allowed.is_finite() {
        return Err(EvalError::new(
            EvalErrorCode::NonFiniteDerivedValue,
            "mass_balance_tolerance",
        ));
    }

    if input.is_point() && output.is_point() {
        if (input.lower - output.lower).abs() <= allowed {
            return Ok(MassBalanceStatus::ExactBalanced);
        }
        return Ok(MassBalanceStatus::ExactUnbalanced);
    }

    if input.upper + allowed < output.lower || output.upper + allowed < input.lower {
        return Ok(MassBalanceStatus::ImpossibleWithinBounds);
    }

    Ok(MassBalanceStatus::PossibleWithUncertainty)
}

fn sum_revenue(
    outputs: &[OutputStream],
    feed_basis_kg: f64,
) -> Result<Interval, EvalError> {
    let mut ordered: Vec<&OutputStream> = outputs
        .iter()
        .filter(|output| output.price_usd_per_tonne.is_some())
        .collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));

    let mut lower = 0.0;
    let mut upper = 0.0;
    for output in ordered {
        if let Some(price) = output.price_usd_per_tonne {
            // kg output / kg feed is numerically equal to tonnes output / tonne feed.
            lower += (output.mass_kg.lower / feed_basis_kg) * price.lower;
            upper += (output.mass_kg.upper / feed_basis_kg) * price.upper;
        }
    }
    checked_interval(lower, upper, "revenue_usd_per_feed_tonne")
}

fn sum_costs(costs: &[CostLine]) -> Result<Interval, EvalError> {
    let mut ordered: Vec<&CostLine> = costs.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    let mut lower = 0.0;
    let mut upper = 0.0;
    for line in ordered {
        lower += line.usd_per_feed_tonne.lower;
        upper += line.usd_per_feed_tonne.upper;
    }
    checked_interval(lower, upper, "cost_usd_per_feed_tonne")
}

fn checked_interval(lower: f64, upper: f64, field: &str) -> Result<Interval, EvalError> {
    let interval = Interval::new(lower, upper);
    if !interval.is_finite_ordered() {
        return Err(EvalError::new(
            EvalErrorCode::NonFiniteDerivedValue,
            field.to_owned(),
        ));
    }
    Ok(interval)
}

fn collect_blockers(route: &RouteCase, mass_balance: MassBalanceStatus) -> Vec<Blocker> {
    let mut blockers = Vec::new();

    match mass_balance {
        MassBalanceStatus::ExactBalanced => {}
        MassBalanceStatus::PossibleWithUncertainty => blockers.push(Blocker {
            name: "mass_balance:PossibleWithUncertainty".to_owned(),
            disposition: BlockerDisposition::Unknown,
        }),
        MassBalanceStatus::ExactUnbalanced => blockers.push(Blocker {
            name: "mass_balance:ExactUnbalanced".to_owned(),
            disposition: BlockerDisposition::Fail,
        }),
        MassBalanceStatus::ImpossibleWithinBounds => blockers.push(Blocker {
            name: "mass_balance:ImpossibleWithinBounds".to_owned(),
            disposition: BlockerDisposition::Fail,
        }),
    }

    push_gate_blocker("energy_balance", route.energy_balance_gate, &mut blockers);
    push_gate_blocker("product_quality", route.product_quality_gate, &mut blockers);
    push_gate_blocker("environmental", route.environmental_gate, &mut blockers);
    push_gate_blocker("process_safety", route.process_safety_gate, &mut blockers);
    push_gate_blocker(
        "model_applicability",
        route.model_applicability_gate,
        &mut blockers,
    );

    blockers.sort_by(|left, right| left.name.cmp(&right.name));
    blockers
}

fn push_gate_blocker(name: &str, gate: Gate, blockers: &mut Vec<Blocker>) {
    match gate {
        Gate::Pass => {}
        Gate::Fail => blockers.push(Blocker {
            name: name.to_owned(),
            disposition: BlockerDisposition::Fail,
        }),
        Gate::Unknown => blockers.push(Blocker {
            name: name.to_owned(),
            disposition: BlockerDisposition::Unknown,
        }),
    }
}

fn classify_admissibility(blockers: &[Blocker]) -> Admissibility {
    if blockers
        .iter()
        .any(|blocker| blocker.disposition == BlockerDisposition::Fail)
    {
        Admissibility::Blocked
    } else if !blockers.is_empty() {
        Admissibility::Indeterminate
    } else {
        Admissibility::Eligible
    }
}

fn classify_conclusion(
    admissibility: Admissibility,
    economics: EconomicsStatus,
    evidence_tier: EvidenceTier,
) -> Conclusion {
    match admissibility {
        Admissibility::Blocked => return Conclusion::Blocked,
        Admissibility::Indeterminate => return Conclusion::Indeterminate,
        Admissibility::Eligible => {}
    }

    match economics {
        EconomicsStatus::BelowHurdle => Conclusion::NotEconomicallyViableWithinBounds,
        EconomicsStatus::Indeterminate => Conclusion::EconomicallyIndeterminate,
        EconomicsStatus::AboveHurdle => match evidence_tier {
            EvidenceTier::Synthetic => Conclusion::SyntheticOnly,
            EvidenceTier::LiteratureReproduced => Conclusion::LiteratureScreenOnly,
            EvidenceTier::BenchMeasured => Conclusion::BenchMeasuredNeedsIndependentReplication,
            EvidenceTier::IndependentlyReplicated => {
                Conclusion::IndependentlyReplicatedWithinDeclaredBoundary
            }
            EvidenceTier::CommercialObserved => {
                Conclusion::CommercialEvidenceWithinDeclaredBoundary
            }
        },
    }
}

/// Produces a stable CSV record. The appended FNV-1a value is a non-cryptographic
/// regression checksum only; it is not an evidence signature or provenance attestation.
pub fn deterministic_receipt(assessment: &Assessment) -> String {
    let record = format!(
        "v0.1,{},{:?},{:?},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:?}",
        assessment.route_id,
        assessment.mass_balance,
        assessment.economics,
        assessment.admissibility,
        assessment.revenue_usd_per_feed_tonne.lower,
        assessment.revenue_usd_per_feed_tonne.upper,
        assessment.cost_usd_per_feed_tonne.lower,
        assessment.cost_usd_per_feed_tonne.upper,
        assessment.netback_usd_per_feed_tonne.lower,
        assessment.netback_usd_per_feed_tonne.upper,
        assessment.conclusion,
    );
    let checksum = fnv1a64_hex(record.as_bytes());
    format!("{record},fnv1a64-non-cryptographic,{checksum}")
}

pub fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub mod fixtures {
    use super::*;

    fn base_case(id: &str) -> RouteCase {
        RouteCase {
            id: id.to_owned(),
            feed_basis_kg: 1_000.0,
            inputs: vec![
                MaterialFlow {
                    name: "feedstock".to_owned(),
                    mass_kg: Interval::point(1_000.0),
                },
                MaterialFlow {
                    name: "hydrogen".to_owned(),
                    mass_kg: Interval::point(5.0),
                },
                MaterialFlow {
                    name: "solvent_makeup".to_owned(),
                    mass_kg: Interval::point(20.0),
                },
            ],
            outputs: vec![
                OutputStream {
                    name: "light_distillate".to_owned(),
                    mass_kg: Interval::point(550.0),
                    price_usd_per_tonne: Some(Interval::new(500.0, 650.0)),
                },
                OutputStream {
                    name: "middle_distillate".to_owned(),
                    mass_kg: Interval::point(300.0),
                    price_usd_per_tonne: Some(Interval::new(350.0, 480.0)),
                },
                OutputStream {
                    name: "fuel_gas".to_owned(),
                    mass_kg: Interval::point(25.0),
                    price_usd_per_tonne: Some(Interval::new(90.0, 150.0)),
                },
                OutputStream {
                    name: "aqueous_waste".to_owned(),
                    mass_kg: Interval::point(30.0),
                    price_usd_per_tonne: None,
                },
                OutputStream {
                    name: "carbonaceous_residue".to_owned(),
                    mass_kg: Interval::point(120.0),
                    price_usd_per_tonne: None,
                },
            ],
            costs: vec![
                CostLine {
                    name: "feedstock".to_owned(),
                    usd_per_feed_tonne: Interval::new(80.0, 110.0),
                },
                CostLine {
                    name: "hydrogen".to_owned(),
                    usd_per_feed_tonne: Interval::new(35.0, 60.0),
                },
                CostLine {
                    name: "solvent_makeup".to_owned(),
                    usd_per_feed_tonne: Interval::new(5.0, 15.0),
                },
                CostLine {
                    name: "energy_and_utilities".to_owned(),
                    usd_per_feed_tonne: Interval::new(20.0, 40.0),
                },
                CostLine {
                    name: "catalyst_capital_operations".to_owned(),
                    usd_per_feed_tonne: Interval::new(25.0, 60.0),
                },
                CostLine {
                    name: "waste_treatment".to_owned(),
                    usd_per_feed_tonne: Interval::new(6.0, 20.0),
                },
                CostLine {
                    name: "logistics".to_owned(),
                    usd_per_feed_tonne: Interval::new(15.0, 35.0),
                },
            ],
            minimum_netback_usd_per_feed_tonne: 0.0,
            energy_balance_gate: Gate::Pass,
            product_quality_gate: Gate::Pass,
            environmental_gate: Gate::Pass,
            process_safety_gate: Gate::Pass,
            model_applicability_gate: Gate::Pass,
            evidence_tier: EvidenceTier::Synthetic,
        }
    }

    /// All fixture quantities/prices are fabricated arithmetic test values.
    /// They do not represent a real feedstock, refinery process, market, or yield.
    pub fn synthetic_positive_case() -> RouteCase {
        base_case("synthetic-positive-screen")
    }

    pub fn synthetic_lossmaking_case() -> RouteCase {
        let mut route = base_case("synthetic-lossmaking-screen");
        for output in &mut route.outputs {
            match output.name.as_str() {
                "light_distillate" => {
                    output.price_usd_per_tonne = Some(Interval::new(200.0, 220.0));
                }
                "middle_distillate" => {
                    output.price_usd_per_tonne = Some(Interval::new(120.0, 140.0));
                }
                "fuel_gas" => {
                    output.price_usd_per_tonne = Some(Interval::new(10.0, 20.0));
                }
                _ => {}
            }
        }
        route
    }

    pub fn overlap_only_uncertainty_case() -> RouteCase {
        let mut route = base_case("synthetic-overlap-only");
        route.inputs[1].mass_kg = Interval::new(4.0, 6.0);
        route
    }

    pub fn unbalanced_case() -> RouteCase {
        let mut route = base_case("synthetic-unbalanced");
        for output in &mut route.outputs {
            if output.name == "carbonaceous_residue" {
                output.mass_kg = Interval::point(110.0);
            }
        }
        route
    }

    pub fn unknown_environment_case() -> RouteCase {
        let mut route = base_case("synthetic-unknown-environment");
        route.environmental_gate = Gate::Unknown;
        route
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assess(route: &RouteCase) -> Assessment {
        evaluate(route, BalanceTolerance::default()).expect("fixture should be numerically valid")
    }

    #[test]
    fn point_mass_balance_is_checked_and_economics_are_interval_bounded() {
        let result = assess(&fixtures::synthetic_positive_case());
        assert_eq!(result.mass_balance, MassBalanceStatus::ExactBalanced);
        assert_eq!(result.input_mass_kg, Interval::point(1_025.0));
        assert_eq!(result.output_mass_kg, Interval::point(1_025.0));
        assert_eq!(
            result.revenue_usd_per_feed_tonne,
            Interval::new(382.25, 505.25)
        );
        assert_eq!(
            result.cost_usd_per_feed_tonne,
            Interval::new(186.0, 340.0)
        );
        assert_eq!(
            result.netback_usd_per_feed_tonne,
            Interval::new(42.25, 319.25)
        );
        assert_eq!(result.economics, EconomicsStatus::AboveHurdle);
        assert_eq!(result.conclusion, Conclusion::SyntheticOnly);
    }

    #[test]
    fn interval_overlap_never_counts_as_exact_mass_closure() {
        let result = assess(&fixtures::overlap_only_uncertainty_case());
        assert_eq!(
            result.mass_balance,
            MassBalanceStatus::PossibleWithUncertainty
        );
        assert_eq!(result.admissibility, Admissibility::Indeterminate);
        assert_eq!(result.conclusion, Conclusion::Indeterminate);
    }

    #[test]
    fn point_mass_imbalance_blocks_an_otherwise_positive_economic_screen() {
        let result = assess(&fixtures::unbalanced_case());
        assert_eq!(result.mass_balance, MassBalanceStatus::ExactUnbalanced);
        assert_eq!(result.admissibility, Admissibility::Blocked);
        assert_eq!(result.conclusion, Conclusion::Blocked);
    }

    #[test]
    fn lossmaking_case_is_below_hurdle_even_at_optimistic_bounds() {
        let result = assess(&fixtures::synthetic_lossmaking_case());
        assert_eq!(result.economics, EconomicsStatus::BelowHurdle);
        assert!(result.netback_usd_per_feed_tonne.upper < 0.0);
        assert_eq!(
            result.conclusion,
            Conclusion::NotEconomicallyViableWithinBounds
        );
    }

    #[test]
    fn unknown_environmental_gate_prevents_positive_promotion() {
        let result = assess(&fixtures::unknown_environment_case());
        assert_eq!(result.admissibility, Admissibility::Indeterminate);
        assert_eq!(result.conclusion, Conclusion::Indeterminate);
        assert!(result.blockers.iter().any(|blocker| {
            blocker.name == "environmental"
                && blocker.disposition == BlockerDisposition::Unknown
        }));
    }

    #[test]
    fn summation_is_invariant_to_input_output_and_cost_order() {
        let mut forward = fixtures::synthetic_positive_case();
        let mut reversed = fixtures::synthetic_positive_case();
        reversed.inputs.reverse();
        reversed.outputs.reverse();
        reversed.costs.reverse();

        let a = assess(&forward);
        let b = assess(&reversed);
        assert_eq!(a.input_mass_kg, b.input_mass_kg);
        assert_eq!(a.output_mass_kg, b.output_mass_kg);
        assert_eq!(a.revenue_usd_per_feed_tonne, b.revenue_usd_per_feed_tonne);
        assert_eq!(a.cost_usd_per_feed_tonne, b.cost_usd_per_feed_tonne);
        assert_eq!(a.netback_usd_per_feed_tonne, b.netback_usd_per_feed_tonne);

        // Keep this binding mutable in neither API nor output: the test also guards
        // against an accidental API that would mutate the caller's route definition.
        forward.inputs.reverse();
        assert_eq!(forward.inputs.len(), reversed.inputs.len());
    }

    #[test]
    fn synthetic_positive_margin_is_never_labelled_physically_or_commercially_valid() {
        let result = assess(&fixtures::synthetic_positive_case());
        assert_eq!(result.conclusion, Conclusion::SyntheticOnly);
        assert_ne!(
            result.conclusion,
            Conclusion::IndependentlyReplicatedWithinDeclaredBoundary
        );
        assert_ne!(
            result.conclusion,
            Conclusion::CommercialEvidenceWithinDeclaredBoundary
        );
    }

    #[test]
    fn non_finite_and_negative_mass_inputs_fail_closed() {
        let mut non_finite = fixtures::synthetic_positive_case();
        non_finite.costs[0].usd_per_feed_tonne = Interval::new(0.0, f64::INFINITY);
        assert!(evaluate(&non_finite, BalanceTolerance::default()).is_err());

        let mut negative = fixtures::synthetic_positive_case();
        negative.outputs[0].mass_kg = Interval::point(-1.0);
        let error = evaluate(&negative, BalanceTolerance::default()).unwrap_err();
        assert_eq!(error.code, EvalErrorCode::NegativeMass);
    }

    #[test]
    fn invalid_tolerance_fails_closed() {
        let error = evaluate(
            &fixtures::synthetic_positive_case(),
            BalanceTolerance {
                absolute_kg: -0.1,
                relative_fraction: 0.0,
            },
        )
        .unwrap_err();
        assert_eq!(error.code, EvalErrorCode::InvalidTolerance);
    }

    #[test]
    fn deterministic_receipt_is_repeatable() {
        let result = assess(&fixtures::synthetic_positive_case());
        assert_eq!(
            deterministic_receipt(&result),
            deterministic_receipt(&result)
        );
        assert!(deterministic_receipt(&result).contains("fnv1a64-non-cryptographic"));
    }
}
