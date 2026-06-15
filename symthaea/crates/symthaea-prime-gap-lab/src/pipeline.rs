pub fn run_research_pipeline(max_k: usize, max_width: u64) {
    println!("--- Initializing Prime Gap Research Workbench v1.0 ---");
    let mut ledger = crate::claim_ledger::ClaimLedger::new();
    let search = crate::search_engine::SearchEngine::new(max_k, max_width);

    // Execute integrated pipeline
    search.run_proof_search(&mut ledger);

    // Finalize report
    let md = crate::reports::generate_markdown_report(&ledger);
    std::fs::write("docs/experiments/prime-gap-lab/v1_0_final_ledger.md", md).unwrap();
    println!("--- Research Run Complete: Report generated ---");
}
