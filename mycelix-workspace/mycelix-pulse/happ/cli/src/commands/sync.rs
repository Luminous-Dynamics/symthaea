// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::client::MycellixClient;
use anyhow::{Context, Result};

/// Update local cache from DHT and MATL
pub async fn handle_sync(client: &MycellixClient, force: bool) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if force {
        println!("                  FORCE SYNCHRONIZATION");
    } else {
        println!("                     SYNCHRONIZATION");
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if force {
        println!("⚠️  Force mode: Bypassing cache, fetching all data from sources");
        println!();
    }

    // Track sync results
    let mut sync_summary = SyncSummary::default();

    // 1. Sync messages from DHT
    println!("📬 Syncing messages from DHT...");
    match sync_messages(client).await {
        Ok(count) => {
            println!("   ✅ Synced {} new message(s)", count);
            sync_summary.messages_synced = count;
        }
        Err(e) => {
            println!("   ⚠️  Failed to sync messages: {}", e);
            sync_summary.messages_failed = true;
        }
    }
    println!();

    // 2. Sync trust scores from MATL
    println!("🔐 Syncing trust scores from MATL...");
    match sync_trust_scores(client).await {
        Ok(count) => {
            println!("   ✅ Synced {} trust score(s)", count);
            sync_summary.trust_scores_synced = count;
        }
        Err(e) => {
            println!("   ⚠️  Failed to sync trust scores: {}", e);
            sync_summary.trust_scores_failed = true;
        }
    }
    println!();

    // 3. Update mailbox statistics
    println!("📊 Updating mailbox statistics...");
    match update_stats(client).await {
        Ok(_) => {
            println!("   ✅ Statistics updated");
            sync_summary.stats_updated = true;
        }
        Err(e) => {
            println!("   ⚠️  Failed to update statistics: {}", e);
        }
    }
    println!();

    // Display summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                       SYNC SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("📬 Messages:      {} new", sync_summary.messages_synced);
    println!(
        "🔐 Trust Scores:  {} updated",
        sync_summary.trust_scores_synced
    );
    println!(
        "📊 Statistics:    {}",
        if sync_summary.stats_updated {
            "Updated"
        } else {
            "Not updated"
        }
    );

    if sync_summary.has_failures() {
        println!();
        println!("⚠️  Some operations failed. Check the output above for details.");
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("💡 Use 'mycelix-mail status' to view updated information");

    Ok(())
}

/// Sync messages from DHT
///
/// Fetches inbox and outbox messages from Holochain DHT
/// and returns the total count of messages synced
async fn sync_messages(client: &MycellixClient) -> Result<usize> {
    let mut total_synced = 0;

    // Check Holochain connection status
    if !client.is_holochain_connected().await {
        println!("   Holochain not connected, attempting to sync via HTTP fallback...");
    }

    // Sync inbox messages
    match client.get_inbox().await {
        Ok(inbox) => {
            let inbox_count = inbox.len();
            println!("   Fetched {} inbox message(s)", inbox_count);
            total_synced += inbox_count;
        }
        Err(e) => {
            println!("   Warning: Could not fetch inbox: {}", e);
        }
    }

    // Sync outbox (sent) messages
    match client.get_sent().await {
        Ok(sent) => {
            let sent_count = sent.len();
            println!("   Fetched {} sent message(s)", sent_count);
            total_synced += sent_count;
        }
        Err(e) => {
            println!("   Warning: Could not fetch sent messages: {}", e);
        }
    }

    Ok(total_synced)
}

/// Sync trust scores from MATL
async fn sync_trust_scores(client: &MycellixClient) -> Result<usize> {
    // Call client's sync_all_trust_scores method
    let trust_scores = client
        .sync_all_trust_scores()
        .await
        .context("Failed to sync trust scores from MATL")?;

    Ok(trust_scores.len())
}

/// Update local mailbox statistics
///
/// Calculates and displays current mailbox statistics
async fn update_stats(client: &MycellixClient) -> Result<()> {
    // Get statistics from client
    let stats = client
        .get_stats()
        .await
        .context("Failed to get statistics")?;

    println!("   Total messages: {}", stats.total_messages);
    println!("   Unread messages: {}", stats.unread_messages);
    println!("   Contacts: {}", stats.total_contacts);
    println!("   Trust scores: {}", stats.total_trust_scores);

    if let Some(last_sync) = stats.last_sync {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(last_sync, 0)
            .unwrap_or_else(|| chrono::Utc::now());
        println!("   Last sync: {}", dt.format("%Y-%m-%d %H:%M:%S UTC"));
    }

    Ok(())
}

/// Summary of sync operation results
#[derive(Default)]
struct SyncSummary {
    messages_synced: usize,
    messages_failed: bool,
    trust_scores_synced: usize,
    trust_scores_failed: bool,
    stats_updated: bool,
}

impl SyncSummary {
    /// Check if any operations failed
    fn has_failures(&self) -> bool {
        self.messages_failed || self.trust_scores_failed || !self.stats_updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_summary_default() {
        let summary = SyncSummary::default();
        assert_eq!(summary.messages_synced, 0);
        assert_eq!(summary.trust_scores_synced, 0);
        assert!(!summary.stats_updated);
        assert!(summary.has_failures()); // All failed in default state
    }

    #[test]
    fn test_sync_summary_success() {
        let summary = SyncSummary {
            messages_synced: 5,
            messages_failed: false,
            trust_scores_synced: 3,
            trust_scores_failed: false,
            stats_updated: true,
        };
        assert!(!summary.has_failures());
    }

    #[test]
    fn test_sync_summary_partial_failure() {
        let summary = SyncSummary {
            messages_synced: 5,
            messages_failed: false,
            trust_scores_synced: 0,
            trust_scores_failed: true,
            stats_updated: true,
        };
        assert!(summary.has_failures());
    }
}
