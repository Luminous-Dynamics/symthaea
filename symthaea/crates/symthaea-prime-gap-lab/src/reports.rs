pub fn generate_json_report(ledger: &crate::claim_ledger::ClaimLedger) -> String {
    serde_json::to_string_pretty(ledger).unwrap()
}

pub fn generate_markdown_report(ledger: &crate::claim_ledger::ClaimLedger) -> String {
    let mut report = String::from("# Claim Ledger Report\n\n");
    for claim in &ledger.claims {
        report.push_str(&format!("## Claim: {}\n", claim.name));
        report.push_str(&format!("- Status: {:?}\n", claim.status));
        report.push_str("\n");
    }
    report.push_str("## Non-Claims\n\n");
    for non_claim in &ledger.non_claims {
        report.push_str(&format!("- {}\n", non_claim));
    }
    report
}
