// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test Fixtures
//!
//! Reusable test data and helper functions for testing

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance, Verification},
    hash,
};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Create a sample claim with default values
pub fn create_sample_claim() -> DesciClaim {
    create_claim_with_tier(EpistemicTier::E0)
}

/// Create a claim with specific epistemic tier
pub fn create_claim_with_tier(tier: EpistemicTier) -> DesciClaim {
    let content = ClaimContent {
        dataset_hash: "blake3:abc123def456...".to_string(),
        description: "Sample research dataset for testing".to_string(),
        category: "genomics".to_string(),
        keywords: vec!["test".to_string(), "sample".to_string()],
        storage_ref: Some("ipfs://QmTest123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("MIT".to_string()),
    };

    DesciClaim::new(tier, content, "test_creator".to_string())
}

/// Create a claim with provenance chain
pub fn create_claim_with_provenance() -> DesciClaim {
    let mut claim = create_sample_claim();

    let prov1 = Provenance::new("Lab Notebook:TEST-001".to_string(), "laboratory_record".to_string())
        .with_url("https://example.com/notebook/001".to_string());

    let prov2 = Provenance::new("Database:TEST-DB".to_string(), "database".to_string())
        .with_url("https://example.com/db/record/123".to_string());

    claim.add_provenance(prov1);
    claim.add_provenance(prov2);

    claim
}

/// Create a claim with verifications
pub fn create_verified_claim(verification_count: usize) -> DesciClaim {
    let tier = match verification_count {
        0 => EpistemicTier::E0,
        1 => EpistemicTier::E1,
        2..=4 => EpistemicTier::E2,
        5..=9 => EpistemicTier::E3,
        _ => EpistemicTier::E4,
    };

    let mut claim = create_claim_with_tier(tier);

    for i in 0..verification_count {
        let verification = Verification {
            verifier: format!("verifier_{}", i),
            timestamp: Utc::now(),
            signature: vec![i as u8; 64],
            notes: Some(format!("Verification {}", i)),
        };
        claim.add_verification(verification);
    }

    claim
}

/// Create a longevity research claim
pub fn create_longevity_claim() -> DesciClaim {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"longevity_data").to_string(),
        description: "NAD+ supplementation effects on aging mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec!["NAD+".to_string(), "aging".to_string(), "mice".to_string()],
        storage_ref: Some("ipfs://QmLongevity123".to_string()),
        reproducibility_score: Some(0.92),
        license: Some("CC-BY-4.0".to_string()),
    };

    DesciClaim::new(EpistemicTier::E3, content, "longevity_researcher".to_string())
}

/// Create a climate research claim
pub fn create_climate_claim() -> DesciClaim {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"climate_data").to_string(),
        description: "Ocean temperature measurements 2020-2025".to_string(),
        category: "climate".to_string(),
        keywords: vec!["ocean".to_string(), "temperature".to_string(), "climate".to_string()],
        storage_ref: Some("ipfs://QmClimate123".to_string()),
        reproducibility_score: Some(0.95),
        license: Some("CC0-1.0".to_string()),
    };

    DesciClaim::new(EpistemicTier::E2, content, "climate_researcher".to_string())
}

/// Create multiple claims with different characteristics
pub fn create_test_claim_set() -> Vec<DesciClaim> {
    vec![
        create_sample_claim(),
        create_longevity_claim(),
        create_climate_claim(),
        create_verified_claim(3),
        create_claim_with_provenance(),
    ]
}

/// Create a claim with invalid data for error testing
pub fn create_invalid_claim() -> DesciClaim {
    let content = ClaimContent {
        dataset_hash: "".to_string(),  // Invalid: empty hash
        description: "".to_string(),    // Invalid: empty description
        category: "".to_string(),       // Invalid: empty category
        keywords: vec![],
        storage_ref: None,
        reproducibility_score: Some(1.5),  // Invalid: > 1.0
        license: None,
    };

    DesciClaim::new(EpistemicTier::E0, content, "".to_string())  // Invalid: empty creator
}

/// Create test CSV data
pub fn create_test_csv_data() -> &'static str {
    "id,name,value\n\
     1,test1,100\n\
     2,test2,200\n\
     3,test3,300\n"
}

/// Create large test data for streaming tests
pub fn create_large_test_data(size_kb: usize) -> Vec<u8> {
    let chunk = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let chunk_size = chunk.len();
    let total_size = size_kb * 1024;
    let repeats = total_size / chunk_size;

    chunk.repeat(repeats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sample_claim() {
        let claim = create_sample_claim();
        assert_eq!(claim.epistemic_tier, EpistemicTier::E0);
        assert!(!claim.content.description.is_empty());
    }

    #[test]
    fn test_create_verified_claim() {
        let claim = create_verified_claim(3);
        assert_eq!(claim.verifications.len(), 3);
        assert!(claim.epistemic_tier >= EpistemicTier::E2);
    }

    #[test]
    fn test_create_claim_set() {
        let claims = create_test_claim_set();
        assert_eq!(claims.len(), 5);

        // Check diversity
        let categories: Vec<_> = claims.iter().map(|c| &c.content.category).collect();
        assert!(categories.contains(&&"genomics".to_string()));
        assert!(categories.contains(&&"longevity".to_string()));
    }

    #[test]
    fn test_large_data_creation() {
        let data = create_large_test_data(10);  // 10 KB
        assert!(data.len() >= 10 * 1024);
        assert!(data.len() < 11 * 1024);  // Allow some overhead
    }
}
