use heavy_oil_rescue_bench::{
    deterministic_receipt, evaluate, fixtures, BalanceTolerance,
};

fn main() {
    println!(
        "schema_version,route_id,mass_balance,economics,admissibility,revenue_lower_usd_per_t_feed,revenue_upper_usd_per_t_feed,cost_lower_usd_per_t_feed,cost_upper_usd_per_t_feed,netback_lower_usd_per_t_feed,netback_upper_usd_per_t_feed,conclusion,checksum_algorithm,checksum_hex"
    );

    let cases = vec![
        fixtures::synthetic_positive_case(),
        fixtures::synthetic_lossmaking_case(),
        fixtures::overlap_only_uncertainty_case(),
        fixtures::unbalanced_case(),
        fixtures::unknown_environment_case(),
    ];

    for route in cases {
        match evaluate(&route, BalanceTolerance::default()) {
            Ok(assessment) => println!("{}", deterministic_receipt(&assessment)),
            Err(error) => {
                eprintln!("fixture {} failed closed: {}", route.id, error);
                std::process::exit(2);
            }
        }
    }

    eprintln!(
        "Synthetic fixtures only. No row is a measured petroleum result or commercial claim."
    );
}
