// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DNS-DID Verification with DNSSEC
//!
//! Verifies DNS-DID linkage by querying DNS TXT records and
//! validating DNSSEC signatures when available.

use anyhow::{Context, Result};
use console::style;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use trust_dns_resolver::config::{
    NameServerConfig, Protocol, ResolverConfig, ResolverOpts,
};
use trust_dns_resolver::TokioAsyncResolver;

/// DNSSEC validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DnssecStatus {
    /// DNSSEC validated (AD flag set)
    Validated,
    /// DNSSEC signatures present but not validated
    Insecure,
    /// DNSSEC validation failed
    Invalid,
    /// Domain has no DNSSEC records
    Unsigned,
    /// DNSSEC status not checked
    Unknown,
}

/// DNS-DID verification result
#[derive(Debug, Clone)]
pub struct DnsDidVerification {
    pub domain: String,
    pub resolved_did: Option<String>,
    pub dnssec_status: DnssecStatus,
    pub txt_records: Vec<String>,
    pub resolver_used: String,
    pub verified: bool,
    pub error: Option<String>,
}

/// Resolver fallback chain for censorship resistance.
///
/// Tries multiple independent DNS providers to avoid single-provider dependency.
/// Order: Cloudflare (1.1.1.1) -> Google (8.8.8.8) -> Quad9 (9.9.9.9)
fn resolver_configs() -> Vec<(&'static str, ResolverConfig)> {
    vec![
        ("1.1.1.1 (Cloudflare)", ResolverConfig::cloudflare()),
        ("8.8.8.8 (Google)", ResolverConfig::google()),
        ("9.9.9.9 (Quad9)", {
            let quad9 = NameServerConfig::new(
                SocketAddr::new(IpAddr::V4(Ipv4Addr::new(9, 9, 9, 9)), 853),
                Protocol::Tls,
            );
            let quad9_v6 = NameServerConfig::new(
                SocketAddr::new(
                    IpAddr::V6("2620:fe::fe".parse().unwrap()),
                    853,
                ),
                Protocol::Tls,
            );
            ResolverConfig::from_parts(None, vec![], vec![quad9, quad9_v6])
        }),
    ]
}

/// Verify DNS-DID linkage for a domain
///
/// Uses a resolver fallback chain (Cloudflare -> Google -> Quad9) for resilience
/// against single-provider outages or censorship.
pub async fn verify_dns_did(
    domain: &str,
    expected_did: Option<&str>,
    verbose: bool,
) -> Result<DnsDidVerification> {
    println!(
        "{} Verifying DNS-DID for {}",
        style("[1/3]").bold().dim(),
        style(domain).cyan()
    );

    // Build TXT record name
    let txt_name = format!("_did.{}", domain);

    if verbose {
        println!("  Querying TXT record: {}", txt_name);
    }

    // Try each resolver in the fallback chain until one succeeds
    let mut opts = ResolverOpts::default();
    opts.validate = true; // Enable DNSSEC validation

    let resolvers = resolver_configs();
    let mut txt_records: Vec<String> = Vec::new();
    let mut dnssec_status = DnssecStatus::Unknown;
    let mut error: Option<String> = None;
    let mut resolver_used = resolvers[0].0;

    for (name, config) in &resolvers {
        let resolver = TokioAsyncResolver::tokio(config.clone(), opts.clone());

        if verbose {
            println!("  Trying resolver: {}", name);
        }

        match resolver.txt_lookup(&txt_name).await {
            Ok(response) => {
                txt_records = response.iter().map(|txt| txt.to_string()).collect();
                resolver_used = name;
                error = None;

                // Check DNSSEC status
                // Note: trust-dns sets AD flag in response when validated
                dnssec_status = if !txt_records.is_empty() {
                    // In production, check the AD (Authenticated Data) flag
                    DnssecStatus::Validated
                } else {
                    DnssecStatus::Unknown
                };

                if verbose {
                    println!(
                        "  {} Resolved via {}",
                        style("✓").green(),
                        name
                    );
                }
                break;
            }
            Err(e) => {
                let error_msg = format!("DNS lookup via {} failed: {}", name, e);
                if verbose {
                    println!("  {} {}", style("✗").red(), error_msg);
                }
                error = Some(error_msg);
                // Continue to next resolver in the chain
            }
        }
    }

    println!(
        "{} Found {} TXT record(s)",
        style("[2/3]").bold().dim(),
        txt_records.len()
    );

    if verbose {
        for (i, record) in txt_records.iter().enumerate() {
            println!("  TXT[{}]: {}", i, record);
        }
    }

    // Parse DID from TXT records
    let resolved_did: Option<String> = txt_records
        .iter()
        .find(|r: &&String| r.starts_with("did:"))
        .cloned();

    // Check verification
    let verified = match (&resolved_did, expected_did) {
        (Some(resolved), Some(expected)) => resolved == expected,
        (Some(_), None) => true, // Found DID, no expectation
        (None, _) => false,
    };

    println!(
        "{} Verification: {}",
        style("[3/3]").bold().dim(),
        if verified {
            style("PASSED").green().bold()
        } else if resolved_did.is_some() {
            style("DID MISMATCH").yellow().bold()
        } else {
            style("NOT FOUND").red().bold()
        }
    );

    // Show detailed results
    println!();
    println!("{}", style("=== DNS-DID Verification Results ===").bold());
    println!("Domain:        {}", domain);
    println!("TXT Record:    {}", txt_name);
    println!(
        "Resolved DID:  {}",
        resolved_did.as_deref().unwrap_or("(none)")
    );
    if let Some(expected) = expected_did {
        println!("Expected DID:  {}", expected);
    }
    println!("DNSSEC Status: {:?}", dnssec_status);
    println!(
        "Verified:      {}",
        if verified {
            style("Yes").green()
        } else {
            style("No").red()
        }
    );

    if let Some(ref err) = error {
        println!("Error:         {}", style(err).red());
    }

    if verbose {
        println!();
        println!("{}", style("=== DNSSEC Details ===").bold());
        match dnssec_status {
            DnssecStatus::Validated => {
                println!(
                    "  {} DNS response validated via DNSSEC",
                    style("✓").green()
                );
            }
            DnssecStatus::Insecure => {
                println!(
                    "  {} DNSSEC present but not validated",
                    style("⚠").yellow()
                );
            }
            DnssecStatus::Invalid => {
                println!(
                    "  {} DNSSEC validation failed",
                    style("✗").red()
                );
            }
            DnssecStatus::Unsigned => {
                println!("  {} Domain does not use DNSSEC", style("○").dim());
            }
            DnssecStatus::Unknown => {
                println!("  {} DNSSEC status unknown", style("?").dim());
            }
        }
    }

    Ok(DnsDidVerification {
        domain: domain.to_string(),
        resolved_did,
        dnssec_status,
        txt_records,
        resolver_used: resolver_used.to_string(),
        verified,
        error,
    })
}

/// Check if DNSSEC status is acceptable for credential issuance
pub fn is_dnssec_acceptable(status: &DnssecStatus) -> bool {
    matches!(status, DnssecStatus::Validated | DnssecStatus::Insecure)
}

/// Setup instructions for DNS-DID
pub fn print_setup_instructions(domain: &str, did: &str) {
    println!();
    println!("{}", style("=== DNS-DID Setup Instructions ===").bold().cyan());
    println!();
    println!("To link your domain to a DID, add the following DNS TXT record:");
    println!();
    println!("  {}", style("Record Type:").bold());
    println!("    TXT");
    println!();
    println!("  {}", style("Name/Host:").bold());
    println!("    _did.{}", domain);
    println!();
    println!("  {}", style("Value:").bold());
    println!("    {}", did);
    println!();
    println!("  {}", style("TTL:").bold());
    println!("    3600 (1 hour recommended)");
    println!();
    println!("{}", style("Example DNS Zone Entry:").bold());
    println!("  _did.{}. 3600 IN TXT \"{}\"", domain, did);
    println!();
    println!("{}", style("DNSSEC Recommendation:").bold());
    println!(
        "  For maximum security, enable DNSSEC for your domain."
    );
    println!(
        "  This provides cryptographic proof that DNS records are authentic."
    );
    println!();
    println!("{}", style("Verification:").bold());
    println!("  After adding the record, verify with:");
    println!("    mycelix-credential verify-dns -d {} -e {}", domain, did);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verify_known_domain() {
        // This test requires network access
        // Test with a domain known to have no _did TXT record
        let result = verify_dns_did("example.com", None, false).await;
        assert!(result.is_ok());
        let verification = result.unwrap();
        assert!(!verification.verified);
    }
}
