// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::client::MycellixClient;
use anyhow::{Context, Result};

/// Register a new DID
pub async fn handle_register(
    client: &MycellixClient,
    did: String,
    agent_key: String,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                   DID REGISTRATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🆔 DID:        {}", did);
    println!("🔑 Agent Key:  {}", truncate_key(&agent_key));
    println!();

    // Validate DID format
    if !did.starts_with("did:mycelix:") {
        println!("❌ Error: DID must start with 'did:mycelix:'");
        println!();
        println!("💡 Example: did:mycelix:abc123");
        return Ok(());
    }

    // Register with DID registry
    println!("📡 Registering with DID registry...");

    client
        .register_did(did.clone(), agent_key.clone())
        .await
        .context("Failed to register DID with registry")?;

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                 REGISTRATION COMPLETE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("✅ DID successfully registered!");
    println!();
    println!("🆔 DID:        {}", did);
    println!("🔑 Agent Key:  {}", truncate_key(&agent_key));
    println!();
    println!("💡 Your DID is now publicly resolvable on the network.");
    println!("   Use 'mycelix-mail did resolve {}' to verify.", did);

    Ok(())
}

/// Resolve a DID to agent key
pub async fn handle_resolve(client: &MycellixClient, did: String) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                    DID RESOLUTION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🔍 Resolving DID: {}", did);
    println!();

    // Resolve DID
    println!("📡 Querying DID registry...");

    match client
        .resolve_did(did.clone())
        .await
        .context("Failed to resolve DID")?
    {
        Some(resolution) => {
            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("                  RESOLUTION RESULT");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("🆔 DID:         {}", resolution.did);
            println!("🔑 Agent Key:   {}", resolution.agent_pub_key);
            println!(
                "📅 Created:     {}",
                format_timestamp(resolution.created_at)
            );
            println!(
                "🔄 Updated:     {}",
                format_timestamp(resolution.updated_at)
            );
            println!();

            let age = format_age(resolution.created_at);
            println!("💡 This DID was registered {}", age);
        }
        None => {
            println!();
            println!("⚠️  DID not found on registry: {}", did);
            println!("   Use 'mycelix-mail did register' to add it.");
        }
    }

    Ok(())
}

/// List all known DIDs
pub async fn handle_list(client: &MycellixClient, filter: Option<String>) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                    KNOWN DIDs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if let Some(ref pattern) = filter {
        println!("🔍 Filter:  Contains '{}'", pattern);
        println!();
    }

    // Fetch all DIDs
    println!("📡 Querying DID registry...");

    let mut dids = client.list_dids().await.context("Failed to list DIDs")?;

    // Apply filter if specified
    if let Some(ref pattern) = filter {
        let pattern_lower = pattern.to_lowercase();
        dids.retain(|d| d.did.to_lowercase().contains(&pattern_lower));
    }

    println!();

    // Display results
    if dids.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("No DIDs found.");
        println!();
        println!("💡 Tips:");
        println!("   • Use 'mycelix-mail did register <did> <agent-key>' to register a DID");
        if filter.is_some() {
            println!("   • Try removing the --filter option");
        }
        return Ok(());
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Found {} DID(s)", dids.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Table header
    println!("{:<40} {:<30} {:<12}", "DID", "Agent Key", "Registered");
    println!("{}", "─".repeat(82));

    // Table rows
    for did_resolution in dids {
        let did_short = truncate_string(&did_resolution.did, 38);
        let key_short = truncate_key(&did_resolution.agent_pub_key);
        let age = format_relative_time(did_resolution.created_at);

        println!("{:<40} {:<30} {:<12}", did_short, key_short, age);
    }

    println!();
    println!("💡 Use 'mycelix-mail did resolve <did>' to view full details");

    Ok(())
}

/// Show current user's DID
pub async fn handle_whoami(client: &MycellixClient) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                    YOUR IDENTITY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match client.get_my_did() {
        Ok(did) => {
            println!("🆔 Your DID:");
            println!("   {}", did);
            println!();

            match client.get_my_agent_key() {
                Ok(agent_key) => {
                    println!("🔑 Your Agent Key:");
                    println!("   {}", agent_key);
                    println!();
                }
                Err(_) => {
                    println!("🔑 Agent Key: Not configured");
                    println!("   Run 'mycelix-mail init' to generate keys");
                    println!();
                }
            }

            match client.resolve_did(did.clone()).await {
                Ok(Some(resolution)) => {
                    println!("📡 Registry Status:");
                    println!("   ✅ Registered on network");
                    println!("   📅 Created: {}", format_timestamp(resolution.created_at));
                    println!();

                    let age = format_age(resolution.created_at);
                    println!("💡 Your DID was registered {}", age);
                }
                Ok(None) => {
                    println!("📡 Registry Status:");
                    println!("   ⚠️  Not yet registered on network");
                    println!();
                    println!("💡 Run 'mycelix-mail did register' to publish your DID");
                }
                Err(err) => {
                    println!("📡 Registry Status:");
                    println!("   ⚠️  Could not query registry: {}", err);
                }
            }
        }
        Err(_) => {
            println!("❌ No DID configured.");
            println!();
            println!("💡 Run 'mycelix-mail init' to set up your identity");
        }
    }

    Ok(())
}

// ========== Helper Functions ==========

/// Format timestamp as human-readable string
fn format_timestamp(ts: i64) -> String {
    use chrono::{DateTime, Utc};
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Format timestamp as relative time (e.g., "2d ago")
fn format_relative_time(ts: i64) -> String {
    use chrono::{DateTime, Utc};
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    let now = Utc::now();
    let duration = now.signed_duration_since(dt);

    if duration.num_days() > 365 {
        format!("{}y ago", duration.num_days() / 365)
    } else if duration.num_days() > 0 {
        format!("{}d ago", duration.num_days())
    } else if duration.num_hours() > 0 {
        format!("{}h ago", duration.num_hours())
    } else if duration.num_minutes() > 0 {
        format!("{}m ago", duration.num_minutes())
    } else {
        "Just now".to_string()
    }
}

/// Format age (e.g., "2 days ago")
fn format_age(ts: i64) -> String {
    use chrono::{DateTime, Utc};
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());
    let now = Utc::now();
    let duration = now.signed_duration_since(dt);

    if duration.num_days() > 365 {
        let years = duration.num_days() / 365;
        if years == 1 {
            "1 year ago".to_string()
        } else {
            format!("{} years ago", years)
        }
    } else if duration.num_days() > 30 {
        let months = duration.num_days() / 30;
        if months == 1 {
            "1 month ago".to_string()
        } else {
            format!("{} months ago", months)
        }
    } else if duration.num_days() > 0 {
        if duration.num_days() == 1 {
            "1 day ago".to_string()
        } else {
            format!("{} days ago", duration.num_days())
        }
    } else if duration.num_hours() > 0 {
        if duration.num_hours() == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{} hours ago", duration.num_hours())
        }
    } else {
        "less than an hour ago".to_string()
    }
}

/// Truncate string for display
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Truncate agent key for display (show first and last parts)
fn truncate_key(key: &str) -> String {
    if key.len() <= 28 {
        key.to_string()
    } else {
        format!("{}...{}", &key[..12], &key[key.len() - 12..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_string() {
        let short = "did:mycelix:abc";
        assert_eq!(truncate_string(short, 20), "did:mycelix:abc");

        let long = "did:mycelix:very_long_identifier_here";
        let truncated = truncate_string(long, 20);
        assert_eq!(truncated.len(), 20);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_key() {
        let short_key = "abc123";
        assert_eq!(truncate_key(short_key), "abc123");

        let long_key = "uhC0kPEGdJSwVsHRpGgghJMWRJZ7RDmFmjd_Qw3cDwPyN8mH9B4K";
        let truncated = truncate_key(long_key);
        assert!(truncated.contains("..."));
        assert!(truncated.starts_with("uhC0kPEGdJSw"));
        assert!(truncated.ends_with("yN8mH9B4K"));
    }

    #[test]
    fn test_format_timestamp() {
        // Test with known timestamp (2021-01-01 00:00:00 UTC)
        let ts = 1609459200;
        let formatted = format_timestamp(ts);
        assert!(formatted.contains("2021"));
        assert!(formatted.contains("UTC"));
    }

    #[test]
    fn test_format_relative_time() {
        use chrono::Utc;

        // Test "just now"
        let now = Utc::now().timestamp();
        let relative = format_relative_time(now);
        assert_eq!(relative, "Just now");

        // Test "days ago"
        let three_days_ago = Utc::now().timestamp() - (3 * 24 * 60 * 60);
        let relative = format_relative_time(three_days_ago);
        assert!(relative.contains("3d ago"));
    }

    #[test]
    fn test_format_age() {
        use chrono::Utc;

        // Test "less than an hour ago"
        let recent = Utc::now().timestamp() - 30;
        let age = format_age(recent);
        assert_eq!(age, "less than an hour ago");

        // Test "days ago"
        let five_days_ago = Utc::now().timestamp() - (5 * 24 * 60 * 60);
        let age = format_age(five_days_ago);
        assert_eq!(age, "5 days ago");

        // Test singular
        let one_day_ago = Utc::now().timestamp() - (24 * 60 * 60);
        let age = format_age(one_day_ago);
        assert_eq!(age, "1 day ago");
    }
}
