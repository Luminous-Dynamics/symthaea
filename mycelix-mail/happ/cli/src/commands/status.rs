// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::client::MycellixClient;
use anyhow::{Context, Result};

/// Show comprehensive system status and configuration
pub async fn handle_status(client: &MycellixClient, detailed: bool) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                     MYCELIX MAIL STATUS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // 1. Identity Information
    println!("📋 Identity");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if let Ok(did) = client.get_my_did() {
        println!("   DID: {}", did);
    } else {
        println!("   DID: Not configured");
        println!("   Run 'mycelix-mail init' to set up your identity");
    }

    if let Ok(agent_key) = client.get_my_agent_key() {
        println!("   Agent Key: {}", truncate_key(&agent_key, 64));
    } else {
        println!("   Agent Key: Not configured");
    }

    println!();

    // 2. Service Endpoints
    println!("🌐 Service Endpoints");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Holochain Conductor: {}", client.get_conductor_url());
    println!("   DID Registry:        {}", client.get_did_registry_url());
    println!("   MATL Bridge:         {}", client.get_matl_bridge_url());
    println!();

    // 3. System Health
    println!("💚 System Health");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let health_status = client
        .health_check()
        .await
        .context("Failed to check system health")?;

    if health_status {
        println!("   Status: ✅ All systems operational");
        println!("   Conductor: Connected");
    } else {
        println!("   Status: ⚠️  Backend not connected");
        println!("   Conductor: Not connected (stub mode)");
        println!();
        println!("   Note: Currently running with stub backend.");
        println!("   Start Holochain conductor to enable full functionality.");
    }
    println!();

    // 4. Mailbox Statistics
    println!("📊 Mailbox Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let stats = client
        .get_stats()
        .await
        .context("Failed to get mailbox statistics")?;

    println!("   Total Messages:  {}", stats.total_messages);
    println!("   Unread Messages: {}", stats.unread_messages);
    println!("   Contacts:        {}", stats.total_contacts);
    println!("   Trust Scores:    {}", stats.total_trust_scores);

    if let Some(last_sync) = stats.last_sync {
        println!("   Last Sync:       {}", format_timestamp(last_sync));
    } else {
        println!("   Last Sync:       Never");
    }

    // 5. Detailed Information (optional)
    if detailed {
        println!();
        println!("⚙️  Configuration Details");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let config = client.get_config();

        println!(
            "   Default Tier:    Tier {} ({})",
            config.preferences.default_tier,
            tier_name(config.preferences.default_tier)
        );
        println!(
            "   Auto Sync:       {}",
            if config.preferences.auto_sync {
                "Enabled"
            } else {
                "Disabled"
            }
        );
        println!(
            "   Cache TTL:       {} seconds",
            config.preferences.cache_ttl
        );
        println!("   Display Format:  {}", config.preferences.display_format);
        println!("   Timeout:         {} seconds", config.conductor.timeout);

        if let Some(ref email) = config.identity.email {
            println!("   Email:           {}", email);
        }

        println!(
            "   Private Key:     {}",
            config.identity.private_key_path.display()
        );
        println!(
            "   Public Key:      {}",
            config.identity.public_key_path.display()
        );
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if !detailed {
        println!();
        println!("💡 Run 'mycelix-mail status --detailed' for more information");
    }

    Ok(())
}

/// Truncate a key for display
fn truncate_key(key: &str, max_len: usize) -> String {
    if key.len() <= max_len {
        key.to_string()
    } else {
        format!("{}...{}", &key[..max_len - 10], &key[key.len() - 7..])
    }
}

/// Format timestamp as human-readable string
fn format_timestamp(ts: i64) -> String {
    use chrono::{DateTime, Utc};

    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());

    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Get tier name from tier number
fn tier_name(tier: u8) -> &'static str {
    match tier {
        0 => "Null",
        1 => "Testimonial",
        2 => "Privately Verifiable",
        3 => "Cryptographically Proven",
        4 => "Publicly Reproducible",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_key() {
        let short_key = "abc123";
        assert_eq!(truncate_key(short_key, 64), "abc123");

        let long_key = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let truncated = truncate_key(long_key, 64);
        assert!(truncated.len() <= 64);
        assert!(truncated.contains("..."));
    }

    #[test]
    fn test_format_timestamp() {
        // Test with known timestamp
        let ts = 1609459200; // 2021-01-01 00:00:00 UTC
        let formatted = format_timestamp(ts);
        assert!(formatted.contains("2021"));
        assert!(formatted.contains("UTC"));
    }

    #[test]
    fn test_tier_name() {
        assert_eq!(tier_name(0), "Null");
        assert_eq!(tier_name(2), "Privately Verifiable");
        assert_eq!(tier_name(4), "Publicly Reproducible");
        assert_eq!(tier_name(99), "Unknown");
    }
}
