//! Integration tests verifying SDK compatibility with Holochain zome.
//!
//! These tests ensure:
//! 1. Commitment hashes match between SDK and zome
//! 2. Proof formats are compatible
//! 3. Jurisdiction validation works correctly

use mycelix_zk_tax::brackets::compute_commitment;
use mycelix_zk_tax::{FilingStatus, Jurisdiction};

/// Compute FNV-1a commitment (same algorithm as zome).
/// This is a direct port to verify compatibility.
fn compute_commitment_zome_style(lower: u64, upper: u64, tax_year: u32) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;

    for byte in lower.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    for byte in upper.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    for byte in tax_year.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Expand to 32 bytes
    let mut commitment = [0u8; 32];
    let mut h = hash;
    for chunk in commitment.chunks_mut(8) {
        chunk.copy_from_slice(&h.to_le_bytes());
        h = h.wrapping_mul(prime);
    }

    hex::encode(commitment)
}

#[test]
fn test_commitment_matches_zome_algorithm() {
    // US 2024 Bracket 2 bounds
    let lower = 47_150_u64;
    let upper = 100_525_u64;
    let year = 2024_u32;

    // Compute using SDK
    let sdk_commitment = compute_commitment(lower, upper, year);

    // Compute using zome algorithm
    let zome_commitment = compute_commitment_zome_style(lower, upper, year);

    // They must match!
    assert_eq!(
        sdk_commitment.to_hex(),
        zome_commitment,
        "SDK and zome commitment algorithms must produce identical results"
    );
}

#[test]
fn test_commitment_matches_all_g20_jurisdictions() {
    use mycelix_zk_tax::brackets::get_brackets;

    let jurisdictions = [
        Jurisdiction::US,
        Jurisdiction::CA,
        Jurisdiction::MX,
        Jurisdiction::BR,
        Jurisdiction::AR,
        Jurisdiction::UK,
        Jurisdiction::DE,
        Jurisdiction::FR,
        Jurisdiction::IT,
        Jurisdiction::RU,
        Jurisdiction::TR,
        Jurisdiction::JP,
        Jurisdiction::CN,
        Jurisdiction::IN,
        Jurisdiction::KR,
        Jurisdiction::ID,
        Jurisdiction::AU,
        Jurisdiction::SA,
        Jurisdiction::ZA,
    ];

    for j in jurisdictions {
        let brackets = get_brackets(j, 2024, FilingStatus::Single).unwrap();

        for bracket in brackets {
            let sdk_commitment = compute_commitment(bracket.lower, bracket.upper, 2024);
            let zome_commitment =
                compute_commitment_zome_style(bracket.lower, bracket.upper, 2024);

            assert_eq!(
                sdk_commitment.to_hex(),
                zome_commitment,
                "Commitment mismatch for {:?} bracket {} ({}% rate)",
                j,
                bracket.index,
                bracket.rate_bps as f64 / 100.0
            );
        }
    }
}

#[test]
fn test_jurisdiction_codes_match_zome_validation() {
    // The zome validates these specific jurisdiction codes
    let zome_valid_jurisdictions = [
        "US", "CA", "MX", "BR", "AR", // Americas
        "UK", "DE", "FR", "IT", "RU", "TR", // Europe
        "JP", "CN", "IN", "KR", "ID", "AU", // Asia-Pacific
        "SA", "ZA", // Middle East & Africa
    ];

    // Verify SDK jurisdictions match
    let sdk_jurisdictions = [
        ("US", Jurisdiction::US),
        ("CA", Jurisdiction::CA),
        ("MX", Jurisdiction::MX),
        ("BR", Jurisdiction::BR),
        ("AR", Jurisdiction::AR),
        ("UK", Jurisdiction::UK),
        ("DE", Jurisdiction::DE),
        ("FR", Jurisdiction::FR),
        ("IT", Jurisdiction::IT),
        ("RU", Jurisdiction::RU),
        ("TR", Jurisdiction::TR),
        ("JP", Jurisdiction::JP),
        ("CN", Jurisdiction::CN),
        ("IN", Jurisdiction::IN),
        ("KR", Jurisdiction::KR),
        ("ID", Jurisdiction::ID),
        ("AU", Jurisdiction::AU),
        ("SA", Jurisdiction::SA),
        ("ZA", Jurisdiction::ZA),
    ];

    // Check all zome codes parse in SDK
    for code in zome_valid_jurisdictions {
        let result: Result<Jurisdiction, _> = code.parse();
        assert!(
            result.is_ok(),
            "Zome code '{}' should parse in SDK",
            code
        );
    }

    // Check SDK jurisdictions format to zome-compatible strings
    for (expected_code, jurisdiction) in sdk_jurisdictions {
        // The debug format gives us the enum variant name
        let formatted = format!("{:?}", jurisdiction);
        assert_eq!(
            formatted, expected_code,
            "SDK jurisdiction {:?} should format as '{}'",
            jurisdiction, expected_code
        );
    }
}

#[test]
fn test_bracket_bounds_validation() {
    // Zome requires lower < upper (except for u64::MAX)
    use mycelix_zk_tax::brackets::get_brackets;

    let jurisdictions = [
        Jurisdiction::US,
        Jurisdiction::UK,
        Jurisdiction::CA,
        Jurisdiction::DE,
        Jurisdiction::JP,
        Jurisdiction::CN,
        Jurisdiction::SA, // 0% tax, special case
    ];

    for j in jurisdictions {
        let brackets = get_brackets(j, 2024, FilingStatus::Single).unwrap();

        for bracket in brackets {
            // Lower must be less than upper (unless upper is MAX)
            if bracket.upper != u64::MAX {
                assert!(
                    bracket.lower < bracket.upper,
                    "Invalid bracket bounds for {:?}: {} >= {}",
                    j,
                    bracket.lower,
                    bracket.upper
                );
            }
        }
    }
}

#[test]
fn test_tax_year_in_valid_range() {
    // Zome validates year is between 2020 and 2030
    use mycelix_zk_tax::brackets::get_brackets;

    // 2020-2025 should work
    for year in [2020, 2021, 2022, 2023, 2024, 2025] {
        let result = get_brackets(Jurisdiction::US, year, FilingStatus::Single);
        assert!(
            result.is_ok(),
            "Year {} should be valid",
            year
        );
    }

    // Year outside range should fail
    for year in [2019, 2026] {
        let result = get_brackets(Jurisdiction::US, year, FilingStatus::Single);
        assert!(
            result.is_err(),
            "Year {} should be invalid",
            year
        );
    }
}

#[test]
fn test_commitment_hex_format() {
    // Zome expects 64 character hex string (32 bytes)
    let commitment = compute_commitment(47_150, 100_525, 2024);
    let hex = commitment.to_hex();

    assert_eq!(
        hex.len(),
        64,
        "Commitment hex should be 64 characters, got {}",
        hex.len()
    );

    // Should be valid hex
    assert!(
        hex::decode(&hex).is_ok(),
        "Commitment should be valid hex"
    );
}

#[test]
fn test_proof_receipt_structure() {
    // Verify the proof structure matches what the zome expects
    use mycelix_zk_tax::brackets::find_bracket;

    let income = 85_000_u64;
    let jurisdiction = Jurisdiction::US;
    let filing_status = FilingStatus::Single;
    let tax_year = 2024_u32;

    let bracket = find_bracket(income, jurisdiction, tax_year, filing_status).unwrap();
    let commitment = compute_commitment(bracket.lower, bracket.upper, tax_year);

    // These fields match TaxProofReceipt in the zome
    let proof_id = format!("proof_{}", hex::encode(&commitment.to_hex()[..16]));
    let jurisdiction_str = format!("{:?}", jurisdiction);
    let filing_status_str = format!("{:?}", filing_status);

    // Verify all fields are populated correctly
    assert!(!proof_id.is_empty());
    assert_eq!(jurisdiction_str, "US");
    assert_eq!(filing_status_str, "Single");
    assert_eq!(bracket.index, 2);
    assert_eq!(bracket.rate_bps, 2200);
    assert!(bracket.lower < bracket.upper);
    assert_eq!(commitment.to_hex().len(), 64);
}
