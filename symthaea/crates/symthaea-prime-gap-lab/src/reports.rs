pub fn generate_json_report(ledger: &crate::claim_ledger::ClaimLedger) -> String {
    serde_json::to_string_pretty(ledger).unwrap()
}

pub fn generate_markdown_report(ledger: &crate::claim_ledger::ClaimLedger) -> String {
    let mut report = String::from("# Claim Ledger Report\n\n");
    for claim in &ledger.claims {
        report.push_str(&format!("## Claim: {}\n", claim.name));
        report.push_str(&format!("- Status: {:?}\n", claim.status));
        report.push_str(&format!("- Evidence: {:?}\n", claim.evidence));
        report.push_str(&format!("- Kind: {:?}\n", claim.kind));
        report.push_str(&format!("- Scope: {:?}\n", claim.scope));
        report.push_str(&format!("- Caveats: {:?}\n", claim.caveats));
        report.push('\n');
    }
    report.push_str("## Non-Claims\n\n");
    for non_claim in &ledger.non_claims {
        report.push_str(&format!("- {}\n", non_claim));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim_ledger::ClaimLedger;

    #[test]
    fn test_report_content() {
        let ledger = ClaimLedger::new();
        let md = generate_markdown_report(&ledger);
        assert!(md.contains("## Non-Claims"));
        assert!(md.contains("Does not prove twin primes."));

        let json = generate_json_report(&ledger);
        assert!(serde_json::from_str::<ClaimLedger>(&json).is_ok());
    }
}
