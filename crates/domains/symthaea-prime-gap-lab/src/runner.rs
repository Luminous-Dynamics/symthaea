pub fn run_demo() {
    let mut ledger = crate::claim_ledger::ClaimLedger::new();

    // Add claims
    let tpc = crate::claim_ledger::ClaimLedger::open_conjecture(
        "Twin Prime Conjecture",
        vec!["Requires analytic number theory beyond current scope.".to_string()],
    );
    ledger.add_claim(tpc).unwrap();

    let hl = crate::claim_ledger::ClaimLedger::heuristic_claim(
        "Hardy-Littlewood k-tuple conjecture",
        vec!["Heuristic model only.".to_string()],
    );
    ledger.add_claim(hl).unwrap();

    // Generate reports
    let md = crate::reports::generate_markdown_report(&ledger);
    let json = crate::reports::generate_json_report(&ledger);

    std::fs::create_dir_all("docs/experiments/prime-gap-lab").unwrap();
    std::fs::write("docs/experiments/prime-gap-lab/latest-claim-ledger.md", md).unwrap();
    std::fs::write(
        "docs/experiments/prime-gap-lab/latest-claim-ledger.json",
        json,
    )
    .unwrap();
    println!("Reports generated in docs/experiments/prime-gap-lab/.");
}
