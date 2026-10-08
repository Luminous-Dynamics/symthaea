//! CIV-ECON smoke laboratory.
//!
//! Deterministic, dependency-free positive controls for the Civilization OS
//! economic boundary. This is deliberately not a macroeconomic simulator.
//!
//! It verifies:
//! 1. closed-loop bridge conservation;
//! 2. equal balances can coexist with concentrated structural power;
//! 3. stock-flow accounting reconciles physical state.
//!
//! Run:
//!   cargo run -p symthaea-lab --bin civ_os_smoke

const PERSONS: usize = 100;
const COMMUNITIES: usize = 4;
const RESOURCES: usize = 8;

#[derive(Debug, Clone, Copy)]
struct Person {
    pc: i64,
    community: usize,
}

#[derive(Debug, Clone, Copy)]
struct Stock {
    opening: i64,
    inflow: i64,
    outflow: i64,
    closing: i64,
}

impl Stock {
    fn new(opening: i64) -> Self {
        Self {
            opening,
            inflow: 0,
            outflow: 0,
            closing: opening,
        }
    }

    fn receive(&mut self, quantity: i64) {
        assert!(quantity >= 0);
        self.inflow += quantity;
        self.closing += quantity;
    }

    fn consume(&mut self, quantity: i64) {
        assert!(quantity >= 0);
        assert!(self.closing >= quantity, "physical stock overdraft");
        self.outflow += quantity;
        self.closing -= quantity;
    }

    fn reconciles(&self) -> bool {
        self.opening + self.inflow - self.outflow == self.closing
    }
}

#[derive(Debug, Clone, Copy)]
struct Bridge {
    cross_community_volume: [u64; COMMUNITIES],
}

impl Bridge {
    fn total(&self) -> u64 {
        self.cross_community_volume.iter().sum()
    }

    fn dominant_share(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let dominant = *self.cross_community_volume.iter().max().unwrap();
        dominant as f64 / total as f64
    }
}

#[derive(Debug, Clone, Copy)]
struct ClaimLedger {
    balances: [i64; 2],
}

impl ClaimLedger {
    fn total(&self) -> i64 {
        self.balances.iter().sum()
    }

    fn convert_closed_loop(&mut self, quantity: i64) {
        assert!(quantity > 0);
        assert!(self.balances[0] >= quantity);
        self.balances[0] -= quantity;
        self.balances[1] += quantity;

        assert!(self.balances[1] >= quantity);
        self.balances[1] -= quantity;
        self.balances[0] += quantity;
    }
}

fn gini(values: &[i64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len() as f64;
    let total: i64 = sorted.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let weighted_sum: i64 = sorted
        .iter()
        .enumerate()
        .map(|(index, value)| (index as i64 + 1) * *value)
        .sum();

    (2.0 * weighted_sum as f64) / (n * total as f64) - (n + 1.0) / n
}

fn build_people() -> Vec<Person> {
    (0..PERSONS)
        .map(|id| Person {
            pc: 100,
            community: id % COMMUNITIES,
        })
        .collect()
}

fn run_closed_loop_bridge_control() {
    let mut ledger = ClaimLedger {
        balances: [10_000, 0],
    };

    let before = ledger.total();
    ledger.convert_closed_loop(7_500);
    let after = ledger.total();

    assert_eq!(before, after, "closed exchange created or destroyed claims");

    println!("CONTROL:CLOSED_LOOP_CLAIM_CONSERVATION:PASS");
    println!("  claim_total_before={before}");
    println!("  claim_total_after={after}");
    println!("  net_claim_creation={}", after - before);
}

fn run_equal_balance_power_control() {
    let people = build_people();
    let pc: Vec<i64> = people.iter().map(|person| person.pc).collect();
    let pc_gini = gini(&pc);

    let bridge = Bridge {
        // One structural bridge carries most cross-community traffic.
        cross_community_volume: [800, 100, 50, 50],
    };

    assert!(
        pc_gini.abs() < 1e-12,
        "positive control requires equal purchasing balances"
    );

    let bridge_share = bridge.dominant_share();
    assert!(
        bridge_share > 0.70,
        "positive control requires concentrated bridge power"
    );

    println!("CONTROL:EQUAL_BALANCES_UNEQUAL_STRUCTURAL_POWER:PASS");
    println!("  purchasing_power_gini={pc_gini:.6}");
    println!("  dominant_bridge_share={bridge_share:.6}");
    println!(
        "  interpretation=balance equality does not imply structural power equality"
    );
}

fn run_stock_flow_control() {
    let mut stocks = [Stock::new(1_000); RESOURCES];

    for tick in 0..12 {
        for (resource_id, stock) in stocks.iter_mut().enumerate() {
            let renewable = if resource_id % 2 == 0 { 12 } else { 0 };
            let demand = 5 + (tick as i64 % 3) + (resource_id as i64 % 2);

            stock.receive(renewable);
            stock.consume(demand);

            assert!(
                stock.reconciles(),
                "stock-flow reconciliation failed for resource {resource_id}"
            );
            assert!(stock.closing >= 0, "resource stock became negative");
        }
    }

    let total_opening: i64 = stocks.iter().map(|stock| stock.opening).sum();
    let total_inflow: i64 = stocks.iter().map(|stock| stock.inflow).sum();
    let total_outflow: i64 = stocks.iter().map(|stock| stock.outflow).sum();
    let total_closing: i64 = stocks.iter().map(|stock| stock.closing).sum();

    assert_eq!(
        total_opening + total_inflow - total_outflow,
        total_closing
    );

    println!("CONTROL:PHYSICAL_STOCK_FLOW_RECONCILIATION:PASS");
    println!("  resources={RESOURCES}");
    println!("  opening_total={total_opening}");
    println!("  inflow_total={total_inflow}");
    println!("  outflow_total={total_outflow}");
    println!("  closing_total={total_closing}");
}

fn run_non_equivalence_control() {
    // Same numeric quantity, different semantic objects.
    let pc_units = 100_i64;
    let mut resource_units = 100_i64;

    // A financial entitlement does not create physical inventory.
    let before = resource_units;
    let _economic_balance = pc_units;
    assert_eq!(resource_units, before);

    // Only a declared physical inflow changes physical stock.
    resource_units += 25;
    assert_eq!(resource_units, 125);

    println!("CONTROL:CLAIM_RESOURCE_NON_EQUIVALENCE:PASS");
    println!("  claim_units={pc_units}");
    println!("  physical_units_before=100");
    println!("  physical_units_after={resource_units}");
}

fn main() {
    println!("CIV-ECON-SMOKE-V1");
    println!("persons={PERSONS} communities={COMMUNITIES} resources={RESOURCES}");

    run_closed_loop_bridge_control();
    run_equal_balance_power_control();
    run_stock_flow_control();
    run_non_equivalence_control();

    println!("RESULT:PASS");
}
