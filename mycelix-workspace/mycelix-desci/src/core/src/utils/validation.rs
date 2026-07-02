// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validation utilities
//!
//! Functions for validating claims, data formats, and business logic

use crate::claims::{DesciClaim, EpistemicTier, Provenance};
use crate::error::{Error, Result};
use regex::Regex;
use std::sync::OnceLock;

/// Maximum length for description fields
pub const MAX_DESCRIPTION_LENGTH: usize = 10_000;

/// Maximum length for category names
pub const MAX_CATEGORY_LENGTH: usize = 100;

/// Maximum number of keywords per claim
pub const MAX_KEYWORDS: usize = 50;

/// Maximum length for a single keyword
pub const MAX_KEYWORD_LENGTH: usize = 50;

/// Maximum length for URLs
pub const MAX_URL_LENGTH: usize = 2048;

/// Minimum number of verifications required for tier upgrades
pub const MIN_VERIFICATIONS_E2: usize = 1;
pub const MIN_VERIFICATIONS_E3: usize = 3;
pub const MIN_VERIFICATIONS_E4: usize = 5;

// Regex patterns stored as static OnceLock for efficiency
static HASH_REGEX: OnceLock<Regex> = OnceLock::new();
static URL_REGEX: OnceLock<Regex> = OnceLock::new();
static SPDX_LICENSE_REGEX: OnceLock<Regex> = OnceLock::new();

/// Validate a complete claim
///
/// Checks all claim fields for correctness and consistency
pub fn validate_claim(claim: &DesciClaim) -> Result<()> {
    // Validate content
    validate_description(&claim.content.description)?;
    validate_category(&claim.content.category)?;
    validate_keywords(&claim.content.keywords)?;
    validate_hash_format(&claim.content.dataset_hash)?;

    // Validate optional fields
    if let Some(ref storage_ref) = claim.content.storage_ref {
        validate_url(storage_ref)?;
    }

    if let Some(ref license) = claim.content.license {
        validate_license(license)?;
    }

    if let Some(score) = claim.content.reproducibility_score {
        validate_reproducibility_score(score)?;
    }

    // Validate provenance
    for prov in &claim.provenance {
        validate_provenance(prov)?;
    }

    // Validate creator (non-empty)
    if claim.creator.trim().is_empty() {
        return Err(Error::Validation("Creator cannot be empty".to_string()));
    }

    // Validate tier requirements
    validate_tier_requirements(claim)?;

    Ok(())
}

/// Validate that a claim meets requirements for its epistemic tier
pub fn validate_tier_requirements(claim: &DesciClaim) -> Result<()> {
    let verification_count = claim.verifications.len();

    match claim.epistemic_tier {
        EpistemicTier::E0 => {
            // No requirements for E0
            Ok(())
        }
        EpistemicTier::E1 => {
            // Requires at least basic provenance
            if claim.provenance.is_empty() {
                return Err(Error::Validation(
                    "E1 tier requires at least one provenance entry".to_string(),
                ));
            }
            Ok(())
        }
        EpistemicTier::E2 => {
            // Requires verification
            if verification_count < MIN_VERIFICATIONS_E2 {
                return Err(Error::Validation(format!(
                    "E2 tier requires at least {} verification(s), found {}",
                    MIN_VERIFICATIONS_E2, verification_count
                )));
            }
            Ok(())
        }
        EpistemicTier::E3 => {
            // Requires multiple verifications
            if verification_count < MIN_VERIFICATIONS_E3 {
                return Err(Error::Validation(format!(
                    "E3 tier requires at least {} verifications, found {}",
                    MIN_VERIFICATIONS_E3, verification_count
                )));
            }
            Ok(())
        }
        EpistemicTier::E4 => {
            // Requires peer review level verification
            if verification_count < MIN_VERIFICATIONS_E4 {
                return Err(Error::Validation(format!(
                    "E4 tier requires at least {} verifications (peer review), found {}",
                    MIN_VERIFICATIONS_E4, verification_count
                )));
            }
            Ok(())
        }
    }
}

/// Validate provenance entry
pub fn validate_provenance(prov: &Provenance) -> Result<()> {
    // Validate source (non-empty)
    if prov.source.trim().is_empty() {
        return Err(Error::Validation("Provenance source cannot be empty".to_string()));
    }

    // Validate source_type (non-empty)
    if prov.source_type.trim().is_empty() {
        return Err(Error::Validation("Provenance source_type cannot be empty".to_string()));
    }

    // Validate URL if present
    if let Some(ref url) = prov.url {
        validate_url(url)?;
    }

    // Validate metadata (no empty keys if it's an object)
    if let Some(obj) = prov.metadata.as_object() {
        for key in obj.keys() {
            if key.trim().is_empty() {
                return Err(Error::Validation(
                    "Provenance metadata cannot have empty keys".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Validate description length and content
pub fn validate_description(desc: &str) -> Result<()> {
    if desc.trim().is_empty() {
        return Err(Error::Validation("Description cannot be empty".to_string()));
    }

    if desc.len() > MAX_DESCRIPTION_LENGTH {
        return Err(Error::Validation(format!(
            "Description too long: {} chars (max {})",
            desc.len(),
            MAX_DESCRIPTION_LENGTH
        )));
    }

    Ok(())
}

/// Validate category name
pub fn validate_category(category: &str) -> Result<()> {
    if category.trim().is_empty() {
        return Err(Error::Validation("Category cannot be empty".to_string()));
    }

    if category.len() > MAX_CATEGORY_LENGTH {
        return Err(Error::Validation(format!(
            "Category too long: {} chars (max {})",
            category.len(),
            MAX_CATEGORY_LENGTH
        )));
    }

    Ok(())
}

/// Validate keywords list
pub fn validate_keywords(keywords: &[String]) -> Result<()> {
    if keywords.len() > MAX_KEYWORDS {
        return Err(Error::Validation(format!(
            "Too many keywords: {} (max {})",
            keywords.len(),
            MAX_KEYWORDS
        )));
    }

    for keyword in keywords {
        if keyword.trim().is_empty() {
            return Err(Error::Validation("Keywords cannot be empty".to_string()));
        }

        if keyword.len() > MAX_KEYWORD_LENGTH {
            return Err(Error::Validation(format!(
                "Keyword too long: '{}' ({} chars, max {})",
                keyword,
                keyword.len(),
                MAX_KEYWORD_LENGTH
            )));
        }
    }

    Ok(())
}

/// Validate hash format (hexadecimal string)
pub fn validate_hash_format(hash: &str) -> Result<()> {
    let regex = HASH_REGEX.get_or_init(|| {
        Regex::new(r"^[a-fA-F0-9]+$")
            .expect("Static regex pattern '^[a-fA-F0-9]+$' must be valid")
    });

    if hash.is_empty() {
        return Err(Error::Validation("Hash cannot be empty".to_string()));
    }

    if !regex.is_match(hash) {
        return Err(Error::Validation(format!(
            "Invalid hash format: '{}' (must be hexadecimal)",
            hash
        )));
    }

    // Common hash lengths: SHA-256 (64), SHA-512 (128), BLAKE3 (64)
    let valid_lengths = [32, 64, 128];
    if !valid_lengths.contains(&hash.len()) {
        return Err(Error::Validation(format!(
            "Unusual hash length: {} chars (expected 32, 64, or 128)",
            hash.len()
        )));
    }

    Ok(())
}

/// Validate reproducibility score (0.0 to 1.0)
pub fn validate_reproducibility_score(score: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&score) {
        return Err(Error::Validation(format!(
            "Reproducibility score out of range: {} (must be 0.0-1.0)",
            score
        )));
    }

    Ok(())
}

/// Validate URL format
pub fn validate_url(url: &str) -> Result<()> {
    if url.len() > MAX_URL_LENGTH {
        return Err(Error::Validation(format!(
            "URL too long: {} chars (max {})",
            url.len(),
            MAX_URL_LENGTH
        )));
    }

    let regex = URL_REGEX.get_or_init(|| {
        Regex::new(r"^(https?|ipfs|ipns)://[^\s]+$")
            .expect("Static URL regex pattern must be valid")
    });

    if !regex.is_match(url) {
        return Err(Error::Validation(format!(
            "Invalid URL format: '{}' (must start with http://, https://, ipfs://, or ipns://)",
            url
        )));
    }

    Ok(())
}

/// Validate license identifier (SPDX or custom)
pub fn validate_license(license: &str) -> Result<()> {
    if license.trim().is_empty() {
        return Err(Error::Validation("License cannot be empty".to_string()));
    }

    // Check if it's a valid SPDX identifier or "CUSTOM"
    if license == "CUSTOM" || is_valid_spdx_license(license) {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "Invalid license: '{}' (use SPDX identifier or 'CUSTOM')",
            license
        )))
    }
}

/// Check if a string is a valid SPDX license identifier
pub fn is_valid_spdx_license(license: &str) -> bool {
    let regex = SPDX_LICENSE_REGEX.get_or_init(|| {
        Regex::new(r"^[A-Za-z0-9\.\-]+(\+)?$")
            .expect("Static SPDX license regex pattern must be valid")
    });

    // Common SPDX licenses (subset for validation)
    let common_licenses = [
        "MIT", "Apache-2.0", "GPL-3.0", "GPL-2.0", "LGPL-3.0", "BSD-3-Clause",
        "BSD-2-Clause", "MPL-2.0", "AGPL-3.0", "CC0-1.0", "CC-BY-4.0",
        "CC-BY-SA-4.0", "Unlicense",
    ];

    regex.is_match(license) && common_licenses.contains(&license)
}

/// Check if a URL is valid (convenience wrapper)
pub fn is_valid_url(url: &str) -> bool {
    validate_url(url).is_ok()
}

/// Sanitize description (remove control characters, trim whitespace)
pub fn sanitize_description(desc: &str) -> String {
    desc.chars()
        .filter(|c| !c.is_control() || c.is_whitespace())
        .collect::<String>()
        .trim()
        .to_string()
}

/// Sanitize filename (remove unsafe characters)
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

/// Normalize category (lowercase, trim)
pub fn normalize_category(cat: &str) -> String {
    cat.trim().to_lowercase()
}

/// Normalize keyword (lowercase, trim)
pub fn normalize_keyword(keyword: &str) -> String {
    keyword.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::ClaimContent;

    fn create_valid_content() -> ClaimContent {
        ClaimContent {
            dataset_hash: "a".repeat(64), // Valid SHA-256 length
            description: "Test description".to_string(),
            category: "genomics".to_string(),
            keywords: vec!["CRISPR".to_string()],
            storage_ref: None,
            reproducibility_score: None,
            license: None,
        }
    }

    #[test]
    fn test_validate_valid_claim() {
        let claim = DesciClaim::new(
            EpistemicTier::E0,
            create_valid_content(),
            "creator".to_string(),
        );
        assert!(validate_claim(&claim).is_ok());
    }

    #[test]
    fn test_validate_description_empty() {
        assert!(validate_description("").is_err());
        assert!(validate_description("   ").is_err());
    }

    #[test]
    fn test_validate_description_too_long() {
        let long_desc = "a".repeat(MAX_DESCRIPTION_LENGTH + 1);
        assert!(validate_description(&long_desc).is_err());
    }

    #[test]
    fn test_validate_description_valid() {
        assert!(validate_description("Valid description").is_ok());
        assert!(validate_description(&"a".repeat(MAX_DESCRIPTION_LENGTH)).is_ok());
    }

    #[test]
    fn test_validate_category_empty() {
        assert!(validate_category("").is_err());
        assert!(validate_category("  ").is_err());
    }

    #[test]
    fn test_validate_category_too_long() {
        let long_cat = "a".repeat(MAX_CATEGORY_LENGTH + 1);
        assert!(validate_category(&long_cat).is_err());
    }

    #[test]
    fn test_validate_category_valid() {
        assert!(validate_category("genomics").is_ok());
        assert!(validate_category("climate science").is_ok());
    }

    #[test]
    fn test_validate_keywords_empty() {
        assert!(validate_keywords(&[]).is_ok()); // Empty list is ok
    }

    #[test]
    fn test_validate_keywords_too_many() {
        let many_keywords: Vec<String> = (0..MAX_KEYWORDS + 1)
            .map(|i| format!("keyword{}", i))
            .collect();
        assert!(validate_keywords(&many_keywords).is_err());
    }

    #[test]
    fn test_validate_keywords_empty_keyword() {
        assert!(validate_keywords(&["".to_string()]).is_err());
        assert!(validate_keywords(&["  ".to_string()]).is_err());
    }

    #[test]
    fn test_validate_keywords_too_long() {
        let long_kw = "a".repeat(MAX_KEYWORD_LENGTH + 1);
        assert!(validate_keywords(&[long_kw]).is_err());
    }

    #[test]
    fn test_validate_keywords_valid() {
        assert!(validate_keywords(&["CRISPR".to_string(), "genomics".to_string()]).is_ok());
    }

    #[test]
    fn test_validate_hash_format_valid() {
        assert!(validate_hash_format(&"a".repeat(64)).is_ok()); // SHA-256
        assert!(validate_hash_format(&"A".repeat(64)).is_ok()); // Uppercase
        assert!(validate_hash_format(&"1234567890abcdef".repeat(4)).is_ok()); // Mixed
    }

    #[test]
    fn test_validate_hash_format_invalid() {
        assert!(validate_hash_format("").is_err());
        assert!(validate_hash_format("not_hex").is_err());
        assert!(validate_hash_format("12345g").is_err()); // Invalid char
        assert!(validate_hash_format(&"a".repeat(63)).is_err()); // Wrong length
    }

    #[test]
    fn test_validate_reproducibility_score_valid() {
        assert!(validate_reproducibility_score(0.0).is_ok());
        assert!(validate_reproducibility_score(0.5).is_ok());
        assert!(validate_reproducibility_score(1.0).is_ok());
    }

    #[test]
    fn test_validate_reproducibility_score_invalid() {
        assert!(validate_reproducibility_score(-0.1).is_err());
        assert!(validate_reproducibility_score(1.1).is_err());
    }

    #[test]
    fn test_validate_url_valid() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://localhost:8080").is_ok());
        assert!(validate_url("ipfs://QmHash123").is_ok());
        assert!(validate_url("ipns://example.com").is_ok());
    }

    #[test]
    fn test_validate_url_invalid() {
        assert!(validate_url("").is_err());
        assert!(validate_url("not a url").is_err());
        assert!(validate_url("ftp://example.com").is_err()); // Unsupported protocol

        let long_url = format!("https://{}", "a".repeat(MAX_URL_LENGTH));
        assert!(validate_url(&long_url).is_err());
    }

    #[test]
    fn test_validate_license_valid() {
        assert!(validate_license("MIT").is_ok());
        assert!(validate_license("Apache-2.0").is_ok());
        assert!(validate_license("GPL-3.0").is_ok());
        assert!(validate_license("CUSTOM").is_ok());
    }

    #[test]
    fn test_validate_license_invalid() {
        assert!(validate_license("").is_err());
        assert!(validate_license("Unknown-License").is_err());
    }

    #[test]
    fn test_is_valid_spdx_license() {
        assert!(is_valid_spdx_license("MIT"));
        assert!(is_valid_spdx_license("Apache-2.0"));
        assert!(!is_valid_spdx_license("CUSTOM"));
        assert!(!is_valid_spdx_license("InvalidLicense"));
    }

    #[test]
    fn test_validate_tier_requirements_e0() {
        let claim = DesciClaim::new(
            EpistemicTier::E0,
            create_valid_content(),
            "creator".to_string(),
        );
        assert!(validate_tier_requirements(&claim).is_ok());
    }

    #[test]
    fn test_validate_tier_requirements_e1_no_provenance() {
        let mut claim = DesciClaim::new(
            EpistemicTier::E1,
            create_valid_content(),
            "creator".to_string(),
        );
        claim.provenance.clear(); // Remove auto-added provenance
        assert!(validate_tier_requirements(&claim).is_err());
    }

    #[test]
    fn test_validate_tier_requirements_e2_insufficient_verifications() {
        let claim = DesciClaim::new(
            EpistemicTier::E2,
            create_valid_content(),
            "creator".to_string(),
        );
        assert!(validate_tier_requirements(&claim).is_err()); // No verifications yet
    }

    #[test]
    fn test_sanitize_description() {
        assert_eq!(sanitize_description("  hello  "), "hello");
        assert_eq!(sanitize_description("hello\nworld"), "hello\nworld");
        assert_eq!(sanitize_description("hello\x00world"), "helloworld"); // Control char removed
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file/name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file*?.txt"), "file__.txt");
        assert_eq!(sanitize_filename("file<>|name"), "file___name");
    }

    #[test]
    fn test_normalize_category() {
        assert_eq!(normalize_category("Genomics"), "genomics");
        assert_eq!(normalize_category("  CLIMATE  "), "climate");
    }

    #[test]
    fn test_normalize_keyword() {
        assert_eq!(normalize_keyword("CRISPR"), "crispr");
        assert_eq!(normalize_keyword("  Gene-Editing  "), "gene-editing");
    }

    #[test]
    fn test_validate_provenance_valid() {
        let prov = Provenance::new("DOI:10.1234/example".to_string(), "publication".to_string());
        assert!(validate_provenance(&prov).is_ok());
    }

    #[test]
    fn test_validate_provenance_empty_source() {
        let mut prov = Provenance::new("DOI:10.1234/example".to_string(), "publication".to_string());
        prov.source = "".to_string();
        assert!(validate_provenance(&prov).is_err());
    }

    #[test]
    fn test_validate_provenance_empty_source_type() {
        let mut prov = Provenance::new("DOI:10.1234/example".to_string(), "publication".to_string());
        prov.source_type = "".to_string();
        assert!(validate_provenance(&prov).is_err());
    }
}
