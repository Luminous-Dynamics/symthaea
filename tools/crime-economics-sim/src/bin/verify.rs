use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

const EXPECTED_HEADER: &str = "schema_version\tscenario_id\tseed\tpolicy_id\tpopulation\tmodel_weeks\tneed_property_events_local\topportunity_property_events_local\tnon_economic_harm_events\tdisplaced_property_events\tsimulated_harm_events_total\treported_harm_events\tfalse_alarms\trobot_compromise_weeks\trobot_safety_incidents\tannualized_cost_units\tessential_access_gain_bps\tessential_access_gap_bps\tprivacy_burden_points\tbenefit_concentration_bps";
const EXPECTED_SCENARIOS: [&str; 8] = [
    "reference",
    "high_displacement",
    "reporting_suppression",
    "concentrated_automation",
    "high_robot_compromise",
    "unequal_essential_access",
    "low_robot_effectiveness",
    "zero_quality_work_effect",
];
const EXPECTED_POLICIES: [&str; 5] = [
    "baseline",
    "essentials",
    "quality_work",
    "robot_protection",
    "combined",
];
const REPLICATIONS: u64 = 32;
const POPULATION: u64 = 1_000;
const WEEKS: u64 = 52;

#[derive(Clone, Debug)]
struct Row {
    scenario: String,
    seed: u64,
    policy: String,
    population: u64,
    weeks: u64,
    need: u64,
    opportunity: u64,
    non_economic: u64,
    displaced: u64,
    simulated_harm: u64,
    reported_harm: u64,
    false_alarms: u64,
    compromise_weeks: u64,
    safety_incidents: u64,
    cost: u64,
    access_gain_bps: u64,
    access_gap_bps: u64,
    privacy_points: u64,
    concentration_bps: i64,
}

impl Row {
    fn parse(line: &str, line_number: usize) -> Result<Self, String> {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 20 {
            return Err(format!(
                "line {line_number}: expected 20 fields, got {}",
                f.len()
            ));
        }
        if f[0] != "1" {
            return Err(format!(
                "line {line_number}: unsupported schema version {}",
                f[0]
            ));
        }
        let parse_u64 = |index: usize, name: &str| -> Result<u64, String> {
            f[index]
                .parse::<u64>()
                .map_err(|_| format!("line {line_number}: invalid {name}"))
        };
        Ok(Self {
            scenario: f[1].to_string(),
            seed: parse_u64(2, "seed")?,
            policy: f[3].to_string(),
            population: parse_u64(4, "population")?,
            weeks: parse_u64(5, "weeks")?,
            need: parse_u64(6, "need count")?,
            opportunity: parse_u64(7, "opportunity count")?,
            non_economic: parse_u64(8, "non-economic count")?,
            displaced: parse_u64(9, "displacement count")?,
            simulated_harm: parse_u64(10, "true crime count")?,
            reported_harm: parse_u64(11, "report count")?,
            false_alarms: parse_u64(12, "false-alarm count")?,
            compromise_weeks: parse_u64(13, "compromise-week count")?,
            safety_incidents: parse_u64(14, "robot-safety incident count")?,
            cost: parse_u64(15, "cost units")?,
            access_gain_bps: parse_u64(16, "access gain")?,
            access_gap_bps: parse_u64(17, "access gap")?,
            privacy_points: parse_u64(18, "privacy burden")?,
            concentration_bps: f[19]
                .parse::<i64>()
                .map_err(|_| format!("line {line_number}: invalid benefit concentration"))?,
        })
    }

    fn identity_key(&self) -> (String, u64, String) {
        (self.scenario.clone(), self.seed, self.policy.clone())
    }

    fn same_non_reporting_fields(&self, other: &Row) -> bool {
        self.need == other.need
            && self.opportunity == other.opportunity
            && self.non_economic == other.non_economic
            && self.displaced == other.displaced
            && self.simulated_harm == other.simulated_harm
            && self.false_alarms == other.false_alarms
            && self.compromise_weeks == other.compromise_weeks
            && self.safety_incidents == other.safety_incidents
            && self.cost == other.cost
            && self.access_gain_bps == other.access_gain_bps
            && self.access_gap_bps == other.access_gap_bps
            && self.privacy_points == other.privacy_points
            && self.concentration_bps == other.concentration_bps
    }

    fn same_modeled_outcomes_except_cost(&self, other: &Row) -> bool {
        self.population == other.population
            && self.weeks == other.weeks
            && self.need == other.need
            && self.opportunity == other.opportunity
            && self.non_economic == other.non_economic
            && self.displaced == other.displaced
            && self.simulated_harm == other.simulated_harm
            && self.reported_harm == other.reported_harm
            && self.false_alarms == other.false_alarms
            && self.compromise_weeks == other.compromise_weeks
            && self.safety_incidents == other.safety_incidents
            && self.access_gain_bps == other.access_gain_bps
            && self.access_gap_bps == other.access_gap_bps
            && self.privacy_points == other.privacy_points
            && self.concentration_bps == other.concentration_bps
    }
}

fn get<'a>(
    map: &'a BTreeMap<(String, u64, String), Row>,
    scenario: &str,
    seed: u64,
    policy: &str,
) -> Result<&'a Row, String> {
    map.get(&(scenario.to_string(), seed, policy.to_string()))
        .ok_or_else(|| format!("missing required row {scenario}/{seed}/{policy}"))
}

fn run(path: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read artifact {path}: {e}"))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| "artifact is empty".to_string())?;
    if header != EXPECTED_HEADER {
        return Err("artifact header differs from the frozen verifier schema".to_string());
    }

    let mut rows: BTreeMap<(String, u64, String), Row> = BTreeMap::new();
    for (offset, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank row at line {}", offset + 2));
        }
        let row = Row::parse(line, offset + 2)?;
        if !EXPECTED_SCENARIOS.contains(&row.scenario.as_str()) {
            return Err(format!("unknown scenario on line {}", offset + 2));
        }
        if !EXPECTED_POLICIES.contains(&row.policy.as_str()) {
            return Err(format!("unknown policy on line {}", offset + 2));
        }
        if row.seed >= REPLICATIONS {
            return Err(format!(
                "seed outside frozen replication range on line {}",
                offset + 2
            ));
        }
        if row.population != POPULATION || row.weeks != WEEKS {
            return Err(format!(
                "unexpected population/window on line {}",
                offset + 2
            ));
        }
        let expected_total = row.need + row.opportunity + row.non_economic + row.displaced;
        if expected_total != row.simulated_harm {
            return Err(format!(
                "crime event accounting identity failed on line {}",
                offset + 2
            ));
        }
        if row.reported_harm > row.simulated_harm {
            return Err(format!(
                "reported incidents exceed underlying simulated incidents on line {}",
                offset + 2
            ));
        }
        if row.access_gain_bps > 10_000 || row.access_gap_bps > 10_000 {
            return Err(format!(
                "access metric outside basis-point bounds on line {}",
                offset + 2
            ));
        }
        if row.compromise_weeks > WEEKS || row.safety_incidents > row.compromise_weeks {
            return Err(format!(
                "robot safety/compromise bounds failed on line {}",
                offset + 2
            ));
        }

        let robot_policy = row.policy == "robot_protection" || row.policy == "combined";
        let essentials_policy = row.policy == "essentials" || row.policy == "combined";
        let expected_cost = match row.policy.as_str() {
            "baseline" => 0,
            "essentials" => 40_000,
            "quality_work" => 60_000,
            "robot_protection" => 100_000,
            "combined" => 190_000,
            _ => unreachable!(),
        };
        if row.cost != expected_cost {
            return Err(format!("cost ledger mismatch on line {}", offset + 2));
        }
        if !robot_policy
            && (row.false_alarms != 0
                || row.compromise_weeks != 0
                || row.safety_incidents != 0
                || row.privacy_points != 0
                || row.concentration_bps != -1
                || row.displaced != 0)
        {
            return Err(format!(
                "non-robot policy contains robot-only outcomes on line {}",
                offset + 2
            ));
        }
        if robot_policy
            && (row.privacy_points == 0 || !(0..=10_000).contains(&row.concentration_bps))
        {
            return Err(format!(
                "robot-policy governance fields invalid on line {}",
                offset + 2
            ));
        }
        if !essentials_policy && row.access_gain_bps != 0 {
            return Err(format!(
                "policy without an essentials program reported_harm access gains on line {}",
                offset + 2
            ));
        }
        let key = row.identity_key();
        if rows.insert(key.clone(), row).is_some() {
            return Err(format!("duplicate scenario/seed/policy key {key:?}"));
        }
    }

    let expected_rows = EXPECTED_SCENARIOS.len() * EXPECTED_POLICIES.len() * REPLICATIONS as usize;
    if rows.len() != expected_rows {
        return Err(format!(
            "expected {expected_rows} unique rows, found {}",
            rows.len()
        ));
    }
    for scenario in EXPECTED_SCENARIOS {
        for seed in 0..REPLICATIONS {
            for policy in EXPECTED_POLICIES {
                let _ = get(&rows, scenario, seed, policy)?;
            }
        }
    }

    // Negative control: changing report propensity must not change underlying harm.
    let mut observed_strict_reporting_reduction = false;
    for seed in 0..REPLICATIONS {
        for policy in EXPECTED_POLICIES {
            let reference = get(&rows, "reference", seed, policy)?;
            let suppressed = get(&rows, "reporting_suppression", seed, policy)?;
            if !reference.same_non_reporting_fields(suppressed) {
                return Err(format!(
                    "reporting-only scenario changed non-report outcomes: seed={seed}, policy={policy}"
                ));
            }
            if suppressed.reported_harm > reference.reported_harm {
                return Err(format!(
                    "suppressed reporting unexpectedly created more reported_harm: seed={seed}, policy={policy}"
                ));
            }
            if suppressed.reported_harm < reference.reported_harm {
                observed_strict_reporting_reduction = true;
            }
        }
    }
    if !observed_strict_reporting_reduction {
        return Err(
            "reporting_suppression never strictly reduced reported_harm for any matched seed/policy"
                .to_string(),
        );
    }

    // Negative control: higher displacement leaves focal-area prevention unchanged.
    for seed in 0..REPLICATIONS {
        for policy in EXPECTED_POLICIES {
            let reference = get(&rows, "reference", seed, policy)?;
            let high = get(&rows, "high_displacement", seed, policy)?;
            if reference.need != high.need
                || reference.opportunity != high.opportunity
                || reference.non_economic != high.non_economic
            {
                return Err(format!(
                    "displacement scenario changed focal-area outcomes: seed={seed}, policy={policy}"
                ));
            }
            if (policy == "robot_protection" || policy == "combined")
                && high.displaced < reference.displaced
            {
                return Err(format!(
                    "higher displacement did not increase displacement count: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    // Negative control: non-economic harm remains visible across all interventions.
    for seed in 0..REPLICATIONS {
        let expected = get(&rows, "reference", seed, "baseline")?.non_economic;
        for scenario in EXPECTED_SCENARIOS {
            for policy in EXPECTED_POLICIES {
                if get(&rows, scenario, seed, policy)?.non_economic != expected {
                    return Err(format!(
                        "non-economic harm was silently changed: seed={seed}, scenario={scenario}, policy={policy}"
                    ));
                }
            }
        }
    }

    // Adversarial scenario: bypassing the lowest-access cohorts worsens the access gap.
    for seed in 0..REPLICATIONS {
        for policy in ["essentials", "combined"] {
            let reference = get(&rows, "reference", seed, policy)?;
            let unequal = get(&rows, "unequal_essential_access", seed, policy)?;
            if unequal.access_gap_bps <= reference.access_gap_bps
                || unequal.access_gain_bps >= reference.access_gain_bps
            {
                return Err(format!(
                    "unequal access was not surfaced: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    // Adversarial scenario: concentrated automation worsens the need-linked channel
    // for robot-enabled arms, while the non-robot comparison arms remain unchanged.
    for seed in 0..REPLICATIONS {
        for policy in EXPECTED_POLICIES {
            let reference = get(&rows, "reference", seed, policy)?;
            let concentrated = get(&rows, "concentrated_automation", seed, policy)?;
            if policy == "robot_protection" || policy == "combined" {
                if concentrated.need < reference.need
                    || concentrated.concentration_bps <= reference.concentration_bps
                {
                    return Err(format!(
                        "concentrated automation shock not surfaced: seed={seed}, policy={policy}"
                    ));
                }
            } else if !reference.same_non_reporting_fields(concentrated)
                || reference.reported_harm != concentrated.reported_harm
            {
                return Err(format!(
                    "automation shock leaked into a non-robot arm: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    // Adversarial scenario: lower robot effectiveness cannot improve a robot-enabled
    // policy; non-robot arms must remain bit-for-bit equivalent to the reference.
    for seed in 0..REPLICATIONS {
        for policy in EXPECTED_POLICIES {
            let reference = get(&rows, "reference", seed, policy)?;
            let low = get(&rows, "low_robot_effectiveness", seed, policy)?;
            if policy == "robot_protection" || policy == "combined" {
                if low.need < reference.need
                    || low.opportunity < reference.opportunity
                    || low.displaced > reference.displaced
                    || low.simulated_harm < reference.simulated_harm
                {
                    return Err(format!(
                        "lower robot effectiveness improved outcomes: seed={seed}, policy={policy}"
                    ));
                }
            } else if !reference.same_non_reporting_fields(low)
                || reference.reported_harm != low.reported_harm
            {
                return Err(format!(
                    "robot effectiveness sensitivity changed a non-robot arm: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    // Negative control: setting the assumed quality-work effect to zero must
    // leave the quality-work-only arm's modeled outcomes equal to baseline (cost
    // aside), and must not perturb policies that do not include quality work.
    for seed in 0..REPLICATIONS {
        let null_baseline = get(&rows, "zero_quality_work_effect", seed, "baseline")?;
        let null_work = get(&rows, "zero_quality_work_effect", seed, "quality_work")?;
        if !null_work.same_modeled_outcomes_except_cost(null_baseline)
            || null_baseline.cost != 0
            || null_work.cost != 60_000
        {
            return Err(format!(
                "zero quality-work effect did not reduce to null outcomes: seed={seed}"
            ));
        }

        for policy in ["baseline", "essentials", "robot_protection"] {
            let reference = get(&rows, "reference", seed, policy)?;
            let null = get(&rows, "zero_quality_work_effect", seed, policy)?;
            if !reference.same_non_reporting_fields(null)
                || reference.reported_harm != null.reported_harm
            {
                return Err(format!(
                    "zero quality-work effect changed non-work policy: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    // Adversarial scenario: higher robot compromise probability increases
    // compromise exposure and removes protection only from robot-enabled arms.
    for seed in 0..REPLICATIONS {
        for policy in EXPECTED_POLICIES {
            let reference = get(&rows, "reference", seed, policy)?;
            let high = get(&rows, "high_robot_compromise", seed, policy)?;
            if policy == "robot_protection" || policy == "combined" {
                if high.compromise_weeks < reference.compromise_weeks
                    || high.safety_incidents < reference.safety_incidents
                    || high.need < reference.need
                    || high.opportunity < reference.opportunity
                {
                    return Err(format!(
                        "robot compromise exposure not surfaced: seed={seed}, policy={policy}"
                    ));
                }
            } else if !reference.same_non_reporting_fields(high)
                || reference.reported_harm != high.reported_harm
            {
                return Err(format!(
                    "robot compromise changed a non-robot policy: seed={seed}, policy={policy}"
                ));
            }
        }
    }

    println!("VERIFICATION=PASS");
    println!(
        "CLAIM_CEILING=synthetic artifact invariants only; no causal, legal, safety, or real-world claim"
    );
    println!("ROWS={}", rows.len());
    println!("SCENARIOS={}", EXPECTED_SCENARIOS.len());
    println!("POLICIES={}", EXPECTED_POLICIES.len());
    println!("REPLICATIONS_PER_CELL={REPLICATIONS}");
    println!(
        "NEGATIVE_CONTROLS=reporting_suppression,high_displacement,non_economic_harm,unequal_access,concentrated_automation,high_robot_compromise,low_robot_effectiveness,zero_quality_work_effect"
    );
    println!(
        "SUMMARY_FIELDS=scenario,policy,mean_simulated_harm_events,mean_reported_harm_events,mean_displaced_property_events,mean_robot_safety_incidents,cost_units,access_gap_bps"
    );
    for scenario in EXPECTED_SCENARIOS {
        for policy in EXPECTED_POLICIES {
            let group: Vec<&Row> = (0..REPLICATIONS)
                .map(|seed| get(&rows, scenario, seed, policy))
                .collect::<Result<Vec<_>, _>>()?;
            let mean = |selector: fn(&Row) -> u64| -> f64 {
                group.iter().map(|row| selector(row) as f64).sum::<f64>() / group.len() as f64
            };
            let first = group[0];
            println!(
                "SYNTHETIC_SUMMARY scenario={scenario} policy={policy} mean_simulated_harm_events={:.3} mean_reported_harm_events={:.3} mean_displaced_property_events={:.3} mean_robot_safety_incidents={:.3} annualized_cost_units={} access_gap_bps={} note=ASSUMPTION_DRIVEN_NOT_CAUSAL",
                mean(|row| row.simulated_harm),
                mean(|row| row.reported_harm),
                mean(|row| row.displaced),
                mean(|row| row.safety_incidents),
                first.cost,
                first.access_gap_bps
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(value) => value,
        None => {
            eprintln!("usage: verify <simulation.tsv>");
            return ExitCode::from(2);
        }
    };
    match run(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VERIFICATION=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
