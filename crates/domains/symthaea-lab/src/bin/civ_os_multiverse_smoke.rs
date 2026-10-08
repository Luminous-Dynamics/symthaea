//! CIV-ECON behavioral + metric-gaming smoke laboratory.
//!
//! These are deterministic positive/negative controls for Civilization OS
//! model neutrality. They are not economic forecasts.
//!
//! Controls:
//! B1 same institution + different behavioral rule -> different trajectories.
//! B2 same behavior + different institution label -> institution remains explicit.
//! G1 known metric target -> proxy improves while hidden maintenance declines.
//! G2 diversified evaluation -> proxy and maintenance are both visible.
//!
//! Run:
//!   cargo run -p symthaea-lab --bin civ_os_multiverse_smoke

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behavior {
    Satisficing,
    TrendFollowing,
    Precautionary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Institution {
    Creditism,
    Market,
}

#[derive(Debug, Clone, Copy)]
struct Agent {
    behavior: Behavior,
    institution: Institution,
    demand: i64,
}

#[derive(Debug, Clone, Copy)]
struct Trajectory {
    total_consumed: i64,
    final_stock: i64,
}

fn next_demand(agent: Agent, observed_stock: i64) -> i64 {
    match agent.behavior {
        Behavior::Satisficing => agent.demand.min(6),
        Behavior::TrendFollowing => {
            if observed_stock > 700 {
                agent.demand + 2
            } else {
                agent.demand.max(3)
            }
        }
        Behavior::Precautionary => {
            if observed_stock < 700 {
                agent.demand.max(2)
            } else {
                agent.demand.min(5)
            }
        }
    }
}

fn simulate(behavior: Behavior, institution: Institution) -> Trajectory {
    let mut agents = vec![Agent {
        behavior,
        institution,
        demand: 5,
    }; 100];

    // The institution is deliberately held constant for the B1 comparison.
    assert!(agents.iter().all(|agent| agent.institution == institution));

    let mut stock = 1_000_i64;
    let mut total_consumed = 0_i64;

    for tick in 0..20 {
        // Same physical shock schedule for every behavioral profile.
        if tick == 10 {
            stock -= 300;
            assert!(stock >= 0);
        }

        let per_agent = next_demand(agents[0], stock);
        let demand = per_agent * agents.len() as i64;
        let actual = demand.min(stock);
        stock -= actual;
        total_consumed += actual;

        for agent in &mut agents {
            // A simple experience update creates path dependence but no
            // institutional change.
            agent.demand = per_agent;
        }
    }

    Trajectory {
        total_consumed,
        final_stock: stock,
    }
}

fn run_behavior_multiverse_control() {
    let satisficing = simulate(Behavior::Satisficing, Institution::Creditism);
    let trend = simulate(Behavior::TrendFollowing, Institution::Creditism);
    let precautionary = simulate(Behavior::Precautionary, Institution::Creditism);

    assert_ne!(
        satisficing.final_stock, trend.final_stock,
        "behavior profiles must generate distinguishable trajectories"
    );
    assert_ne!(
        trend.final_stock, precautionary.final_stock,
        "behavior profiles must generate distinguishable trajectories"
    );

    println!("CONTROL:SAME_INSTITUTION_DIFFERENT_BEHAVIOR:PASS");
    println!(
        "  satisficing={{consumed:{}, final_stock:{}}}",
        satisficing.total_consumed, satisficing.final_stock
    );
    println!(
        "  trend_following={{consumed:{}, final_stock:{}}}",
        trend.total_consumed, trend.final_stock
    );
    println!(
        "  precautionary={{consumed:{}, final_stock:{}}}",
        precautionary.total_consumed, precautionary.final_stock
    );
}

fn run_institution_separation_control() {
    let creditism = simulate(Behavior::Satisficing, Institution::Creditism);
    let market = simulate(Behavior::Satisficing, Institution::Market);

    // The physical behavior model is identical. The experiment labels differ.
    assert_eq!(creditism.final_stock, market.final_stock);
    assert_eq!(creditism.total_consumed, market.total_consumed);

    println!("CONTROL:SAME_BEHAVIOR_DIFFERENT_INSTITUTION_LABEL:PASS");
    println!(
        "  note=behavioral trajectory is unchanged because the institution label is not secretly coupled to the agent rule"
    );
}

#[derive(Debug, Clone, Copy)]
struct Production {
    throughput: i64,
    maintenance_stock: i64,
}

fn optimize_for_throughput_only() -> Production {
    // The metric-aware agent learns that deferring maintenance boosts the
    // measured throughput during the evaluation window.
    Production {
        throughput: 1_200,
        maintenance_stock: 400,
    }
}

fn optimize_for_diversified_dashboard() -> Production {
    Production {
        throughput: 1_000,
        maintenance_stock: 900,
    }
}

fn run_metric_gaming_control() {
    let target = optimize_for_throughput_only();
    let diversified = optimize_for_diversified_dashboard();

    assert!(target.throughput > diversified.throughput);
    assert!(
        target.maintenance_stock < diversified.maintenance_stock,
        "metric gaming fixture must expose the hidden trade-off"
    );

    println!("CONTROL:KNOWN_METRIC_GAMING:PASS");
    println!(
        "  throughput_target={{throughput:{}, maintenance_stock:{}}}",
        target.throughput, target.maintenance_stock
    );
    println!(
        "  diversified={{throughput:{}, maintenance_stock:{}}}",
        diversified.throughput, diversified.maintenance_stock
    );
    println!(
        "  divergence=target_metric_improves_while_independent_outcome_degrades"
    );
}

fn run_no_hidden_objective_control() {
    let diversified = optimize_for_diversified_dashboard();

    assert_eq!(diversified.throughput, 1_000);
    assert_eq!(diversified.maintenance_stock, 900);

    println!("CONTROL:DIVERSIFIED_EVALUATION_EXPOSES_TRADEOFF:PASS");
}

fn main() {
    println!("CIV-ECON-MULTIVERSE-SMOKE-V1");

    run_behavior_multiverse_control();
    run_institution_separation_control();
    run_metric_gaming_control();
    run_no_hidden_objective_control();

    println!("RESULT:PASS");
}
