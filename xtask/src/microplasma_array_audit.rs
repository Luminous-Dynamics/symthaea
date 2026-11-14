use anyhow::{bail, Context};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    generated_date: String,
    claim_ceiling: String,
    maximum_cell_output_cv_percent: f64,
    aggregate_output_absolute_tolerance: f64,
    counts_expected: ExpectedCounts,
    cases: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedCounts {
    #[serde(rename = "PASS")]
    pass: usize,
    #[serde(rename = "BLOCKED")]
    blocked: usize,
    #[serde(rename = "REJECTED")]
    rejected: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    claim: String,
    topology_id: String,
    service_contract_id: String,
    declared_cell_count: usize,
    cells: Vec<Cell>,
    diagnostics_complete: bool,
    shared_utilities: SharedUtilities,
    neighbor_response: String,
    uniformity_required: bool,
    minimum_useful_output: f64,
    observed_useful_output: f64,
    quality_endpoint_pass: bool,
    fault: Fault,
    expected_disposition: String,
    rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cell {
    cell_id: String,
    state: String,
    normalized_output: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedUtilities {
    power: String,
    gas: String,
    cooling: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fault {
    kind: String,
    affected_cell_ids: Vec<String>,
    injection_confirmed: bool,
    containment_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Pass,
    Blocked,
    Rejected,
}

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

struct Outcome {
    id: String,
    disposition: Disposition,
    reason: &'static str,
}

fn outcome(case: &Scenario, disposition: Disposition, reason: &'static str) -> Outcome {
    Outcome {
        id: case.id.clone(),
        disposition,
        reason,
    }
}

fn utility_status(status: &str) -> Option<Disposition> {
    match status {
        "IN_RANGE" => None,
        "OUT_OF_RANGE" => Some(Disposition::Rejected),
        "UNKNOWN" => Some(Disposition::Blocked),
        _ => Some(Disposition::Blocked),
    }
}

fn decide(case: &Scenario, fixture: &Fixture) -> Outcome {
    let reject = |reason| outcome(case, Disposition::Rejected, reason);
    let blocked = |reason| outcome(case, Disposition::Blocked, reason);
    let pass = |reason| outcome(case, Disposition::Pass, reason);

    if case.topology_id.trim().is_empty()
        || case.service_contract_id.trim().is_empty()
        || case.declared_cell_count == 0
        || case.cells.len() != case.declared_cell_count
    {
        return reject("declared cell manifest is empty or does not match the observed cell set");
    }

    let mut cell_ids = HashSet::new();
    for cell in &case.cells {
        if cell.cell_id.trim().is_empty() || !cell_ids.insert(cell.cell_id.as_str()) {
            return reject("cell identity is empty or duplicated within the declared topology");
        }
    }

    let mut affected_ids = HashSet::new();
    for affected in &case.fault.affected_cell_ids {
        if !affected_ids.insert(affected.as_str()) || !cell_ids.contains(affected.as_str()) {
            return reject("fault receipt contains a duplicate or foreign cell identity");
        }
    }

    if !case.diagnostics_complete {
        return blocked("cell-level diagnostic coverage is incomplete; unobserved cells cannot count as healthy");
    }

    if case.cells.iter().any(|cell| cell.state == "UNKNOWN") {
        return blocked("at least one cell state is unknown");
    }

    if case.cells.iter().any(|cell| !["ACTIVE", "ISOLATED"].contains(&cell.state.as_str())) {
        return blocked("cell state is not in the S0 state vocabulary");
    }

    for cell in &case.cells {
        if !cell.normalized_output.is_finite() || cell.normalized_output < 0.0 {
            return reject("cell useful-output observation is non-finite or negative");
        }
        match cell.state.as_str() {
            "ACTIVE" if cell.normalized_output <= 0.0 => {
                return reject("an active cell has no positive useful-output observation");
            }
            "ISOLATED" if cell.normalized_output != 0.0 => {
                return reject("an isolated cell contributes non-zero useful output");
            }
            _ => {}
        }
    }

    for status in [
        case.shared_utilities.power.as_str(),
        case.shared_utilities.gas.as_str(),
        case.shared_utilities.cooling.as_str(),
    ] {
        if let Some(disposition) = utility_status(status) {
            return match disposition {
                Disposition::Rejected => reject("a shared power, gas or cooling boundary is out of range"),
                Disposition::Blocked => blocked("a shared power, gas or cooling state is unknown or unrecognized"),
                Disposition::Pass => unreachable!(),
            };
        }
    }

    match case.neighbor_response.as_str() {
        "WITHIN_ENVELOPE" => {}
        "OUTSIDE_ENVELOPE" => return reject("observed neighbouring-cell response is outside its declared envelope"),
        _ => return blocked("neighbour response is unknown or unrecognized"),
    }

    if !case.observed_useful_output.is_finite()
        || !case.minimum_useful_output.is_finite()
        || case.minimum_useful_output <= 0.0
    {
        return reject("aggregate or minimum useful output is non-finite or invalid");
    }

    let cell_output_sum: f64 = case.cells.iter().map(|cell| cell.normalized_output).sum();
    if (cell_output_sum - case.observed_useful_output).abs()
        > fixture.aggregate_output_absolute_tolerance
    {
        return reject("aggregate useful output does not reconcile to the sum of cell observations");
    }

    if case.observed_useful_output < case.minimum_useful_output {
        return reject("observed useful output is below the predeclared application service floor");
    }

    if !case.quality_endpoint_pass {
        return reject("the application-specific product-quality endpoint failed");
    }

    let active_outputs: Vec<f64> = case
        .cells
        .iter()
        .filter(|cell| cell.state == "ACTIVE")
        .map(|cell| cell.normalized_output)
        .collect();
    if active_outputs.is_empty() {
        return reject("no active cell contributes useful output");
    }

    let mean = active_outputs.iter().sum::<f64>() / active_outputs.len() as f64;
    let variance = active_outputs
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / active_outputs.len() as f64;
    let cv_percent = if mean > 0.0 {
        variance.sqrt() / mean * 100.0
    } else {
        f64::INFINITY
    };

    if case.uniformity_required && active_outputs.len() < 2 {
        return blocked("fewer than two active cells are available to evaluate the declared uniformity claim");
    }
    if case.uniformity_required && cv_percent > fixture.maximum_cell_output_cv_percent {
        return reject("recomputed cell-output coefficient of variation exceeds the declared fixture ceiling");
    }

    match case.claim.as_str() {
        "NORMAL_SERVICE" => {
            if case.fault.kind != "NONE"
                || !case.fault.affected_cell_ids.is_empty()
                || case.fault.injection_confirmed
                || case.fault.containment_verified
                || case.cells.iter().any(|cell| cell.state != "ACTIVE")
            {
                return reject("normal-service claim conflicts with a declared fault or isolated cell");
            }
        }
        "DEGRADED_SERVICE" => {
            if case.fault.kind == "NONE" {
                return reject("degraded-service claim has no declared non-nominal fault");
            }
        }
        _ => return blocked("claim type is not registered in the S0 contract"),
    }

    match case.fault.kind.as_str() {
        "NONE" => pass("all declared cells are active and the nominal synthetic service contract is satisfied"),
        "INDEPENDENT_CELL_FAILURE" => {
            if case.fault.affected_cell_ids.is_empty() {
                return reject("independent-cell failure has no affected cell identity");
            }
            if case.fault.affected_cell_ids.iter().any(|id| {
                case.cells
                    .iter()
                    .find(|cell| cell.cell_id == *id)
                    .is_some_and(|cell| cell.state != "ISOLATED")
            }) {
                return reject("an independently faulted cell is not recorded as isolated");
            }
            if !case.fault.injection_confirmed || !case.fault.containment_verified {
                return blocked("fault injection or cell isolation/containment evidence is missing");
            }
            pass("independent cell failure is explicitly isolated and the declared degraded service floor is met")
        }
        "SHARED_POWER_FAILURE" | "SHARED_GAS_FAILURE" | "SHARED_COOLING_FAILURE" => {
            if !case.fault.affected_cell_ids.is_empty() {
                return reject("shared-utility fault incorrectly names individual affected cells");
            }
            if !case.fault.injection_confirmed || !case.fault.containment_verified {
                return blocked("shared-utility fault confirmation or failover/containment evidence is missing");
            }
            pass("shared-utility fault is confirmed, post-failover utility states are in range, and degraded service is retained")
        }
        "THERMAL_CROSS_TALK" => {
            if !case.fault.injection_confirmed || !case.fault.containment_verified {
                return blocked("cross-talk fault confirmation or containment evidence is missing");
            }
            pass("neighbour response remains inside the declared envelope after the synthetic cross-talk challenge")
        }
        "SENSOR_LOSS" => blocked("sensor loss prevents a complete cell-state and service claim"),
        _ => blocked("fault class has no registered S0 containment semantics"),
    }
}

fn evaluate(fixture: &Fixture) -> anyhow::Result<Vec<Outcome>> {
    if fixture.schema_version != "0.1.0"
        || fixture.benchmark_id != "MICROPLASMA-ARRAY-RESILIENCE-001"
        || fixture.claim_ceiling
            != "SYNTHETIC_CONTRACT_AUDIT_ONLY_NO_PHYSICAL_ARRAY_QUALIFICATION_OR_ACTUATION_AUTHORITY"
    {
        bail!("fixture identity, version or claim ceiling is not recognized");
    }
    if fixture.generated_date.trim().is_empty()
        || !fixture.maximum_cell_output_cv_percent.is_finite()
        || fixture.maximum_cell_output_cv_percent <= 0.0
        || !fixture.aggregate_output_absolute_tolerance.is_finite()
        || fixture.aggregate_output_absolute_tolerance <= 0.0
        || fixture.cases.is_empty()
    {
        bail!("fixture metadata or numeric audit tolerances are invalid");
    }

    let mut case_ids = HashSet::new();
    let mut outcomes = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        if case.id.trim().is_empty() || !case_ids.insert(case.id.as_str()) {
            bail!("fixture case identity is empty or duplicated: {}", case.id);
        }
        let result = decide(case, fixture);
        if result.disposition.as_str() != case.expected_disposition {
            bail!(
                "known-answer mismatch for {}: expected {}, got {} ({})",
                result.id,
                case.expected_disposition,
                result.disposition.as_str(),
                result.reason
            );
        }
        outcomes.push(result);
    }

    let pass_count = outcomes.iter().filter(|x| x.disposition == Disposition::Pass).count();
    let blocked_count = outcomes.iter().filter(|x| x.disposition == Disposition::Blocked).count();
    let rejected_count = outcomes.iter().filter(|x| x.disposition == Disposition::Rejected).count();
    if pass_count != fixture.counts_expected.pass
        || blocked_count != fixture.counts_expected.blocked
        || rejected_count != fixture.counts_expected.rejected
    {
        bail!("fixture disposition crosswalk does not match its declared expected counts");
    }
    Ok(outcomes)
}

pub fn run(root: &Path) -> anyhow::Result<()> {
    let fixture_path = root.join("data/research/microplasma_array_fault_fixture_v0_1.json");
    let source = fs::read_to_string(&fixture_path)
        .with_context(|| format!("reading {}", fixture_path.display()))?;
    let fixture: Fixture = serde_json::from_str(&source)
        .with_context(|| format!("parsing {}", fixture_path.display()))?;
    let outcomes = evaluate(&fixture)?;

    let pass_count = outcomes.iter().filter(|x| x.disposition == Disposition::Pass).count();
    let blocked_count = outcomes.iter().filter(|x| x.disposition == Disposition::Blocked).count();
    let rejected_count = outcomes.iter().filter(|x| x.disposition == Disposition::Rejected).count();

    println!(
        "{}: {}/{} known-answer cases matched (PASS {}, BLOCKED {}, REJECTED {})",
        fixture.benchmark_id,
        outcomes.len(),
        fixture.cases.len(),
        pass_count,
        blocked_count,
        rejected_count
    );
    for result in outcomes {
        println!("{} {} — {}", result.disposition.as_str(), result.id, result.reason);
    }
    println!("Synthetic contract audit only; no physical array qualification or actuation authority.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!(
            "../../data/research/microplasma_array_fault_fixture_v0_1.json"
        ))
        .expect("frozen synthetic fixture must parse")
    }

    #[test]
    fn fixture_replays_all_known_answers_and_disposition_counts() {
        let fixture = fixture();
        let outcomes = evaluate(&fixture).expect("known-answer replay");
        assert_eq!(outcomes.len(), fixture.cases.len());
        assert_eq!(fixture.counts_expected.pass, 6);
        assert_eq!(fixture.counts_expected.blocked, 7);
        assert_eq!(fixture.counts_expected.rejected, 15);
    }

    #[test]
    fn unknown_diagnostics_never_become_a_pass() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|x| x.id == "DIAGNOSTIC_COVERAGE_LOST").unwrap();
        assert_eq!(decide(case, &fixture).disposition, Disposition::Blocked);
    }

    #[test]
    fn isolated_cell_must_contribute_zero_output() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|x| x.id == "ISOLATED_CELL_HAS_NONZERO_OUTPUT").unwrap();
        assert_eq!(decide(case, &fixture).disposition, Disposition::Rejected);
    }

    #[test]
    fn a_faulted_active_cell_cannot_be_counted_as_healthy() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|x| x.id == "FAILED_CELL_STILL_COUNTED_ACTIVE").unwrap();
        assert_eq!(decide(case, &fixture).disposition, Disposition::Rejected);
    }

    #[test]
    fn array_output_must_reconcile_to_cell_observations() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|x| x.id == "AGGREGATE_OUTPUT_DOES_NOT_RECONCILE").unwrap();
        assert_eq!(decide(case, &fixture).disposition, Disposition::Rejected);
    }

    #[test]
    fn service_quality_is_not_inferred_from_throughput_alone() {
        let fixture = fixture();
        let case = fixture.cases.iter().find(|x| x.id == "PRODUCT_QUALITY_ENDPOINT_FAILS").unwrap();
        assert_eq!(decide(case, &fixture).disposition, Disposition::Rejected);
    }
}
