// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emit the LCF discriminating-experiment proposal as markdown.
//!
//! Usage: `cargo run --example optimal_experiment_proposal [budget_usd]`

use spark_engine::bayesian::HypothesisBelief;
use spark_engine::optimal_experiment::generate_proposal_markdown;

fn main() {
    let budget_usd: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000.0);
    print!(
        "{}",
        generate_proposal_markdown(&HypothesisBelief::default_priors(), budget_usd)
    );
}
