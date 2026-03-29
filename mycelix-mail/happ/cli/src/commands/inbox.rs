// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use crate::client::MycellixClient;
use crate::types::MailMessage;

/// List inbox messages with filtering and formatting
pub async fn handle_inbox(
    client: &MycellixClient,
    from: Option<String>,
    trust_min: Option<f64>,
    unread: bool,
    limit: usize,
    format: &str,
) -> Result<()> {
    println!("📬 Fetching inbox...");
    println!();

    // 1. Get inbox messages from client
    let mut messages = client
        .get_inbox()
        .await
        .context("Failed to fetch inbox messages")?;

    // Show filters being applied
    let mut filter_count = 0;
    if from.is_some() {
        filter_count += 1;
    }
    if trust_min.is_some() {
        filter_count += 1;
    }
    if unread {
        filter_count += 1;
    }

    if filter_count > 0 {
        println!("🔍 Applying {} filter(s):", filter_count);
        if let Some(ref sender) = from {
            println!("   • From: {}", sender);
        }
        if let Some(min_trust) = trust_min {
            println!("   • Minimum trust: {:.2}", min_trust);
        }
        if unread {
            println!("   • Unread only");
        }
        println!();
    }

    // 2. Build trust score cache for filtering
    let mut trust_scores = std::collections::HashMap::new();
    if trust_min.is_some() {
        // Collect unique sender DIDs
        let sender_dids: std::collections::HashSet<_> = messages.iter().map(|m| m.from_did.clone()).collect();

        // Fetch trust scores for each sender
        for did in sender_dids {
            if let Ok(Some(score)) = client.get_trust_score(did.clone()).await {
                trust_scores.insert(did, score.score);
            }
        }
    }

    // 3. Load read status from local storage
    let read_messages = load_read_messages();

    // 4. Apply filters
    messages = apply_filters(messages, from, trust_min, unread, &trust_scores, &read_messages);

    // 3. Sort by timestamp (newest first)
    messages.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // 4. Apply limit
    let total_count = messages.len();
    messages.truncate(limit);

    // Handle empty inbox
    if messages.is_empty() {
        if filter_count > 0 {
            println!("No messages match your filters.");
            println!(
                "Try removing some filters or check 'mycelix-mail inbox' to see all messages."
            );
        } else {
            println!("Your inbox is empty.");
            println!();
            println!("To receive messages:");
            println!("  1. Share your DID with contacts: mycelix-mail did");
            println!("  2. Wait for others to send you messages");
        }
        return Ok(());
    }

    // 5. Format and display
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Showing {} of {} message(s)", messages.len(), total_count);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match format {
        "json" => display_json(&messages)?,
        "raw" => display_raw(&messages),
        _ => display_table(&messages),
    }

    println!();

    // Show helpful tips
    if messages.len() < total_count {
        println!(
            "💡 Showing first {} messages. Use --limit {} to see more.",
            limit, total_count
        );
    }

    Ok(())
}

/// Get the path for storing read message status
fn get_read_messages_path() -> std::path::PathBuf {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("mycelix-mail");
    std::fs::create_dir_all(&config_dir).ok();
    config_dir.join("read_messages.json")
}

/// Load read message IDs from local storage
fn load_read_messages() -> std::collections::HashSet<String> {
    let path = get_read_messages_path();
    if let Ok(content) = std::fs::read_to_string(&path) {
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        std::collections::HashSet::new()
    }
}

/// Save read message IDs to local storage
fn save_read_messages(read_messages: &std::collections::HashSet<String>) -> Result<()> {
    let path = get_read_messages_path();
    let content = serde_json::to_string_pretty(read_messages)
        .context("Failed to serialize read messages")?;
    std::fs::write(&path, content).context("Failed to write read messages file")?;
    Ok(())
}

/// Mark a message as read
///
/// Adds the message ID to the read messages set and persists to disk.
pub fn mark_message_as_read(msg: &MailMessage) -> Result<()> {
    let msg_id = format!("{}:{}:{}", msg.from_did, msg.timestamp, msg.body_cid);
    let mut read_messages = load_read_messages();
    read_messages.insert(msg_id);
    save_read_messages(&read_messages)
}

/// Apply filters to message list
///
/// Filters messages by sender, trust score, and read status.
/// Trust scores are looked up from the provided cache map.
/// Read status is tracked via the read_messages set.
fn apply_filters(
    messages: Vec<MailMessage>,
    from: Option<String>,
    trust_min: Option<f64>,
    unread: bool,
    trust_scores: &std::collections::HashMap<String, f64>,
    read_messages: &std::collections::HashSet<String>,
) -> Vec<MailMessage> {
    messages
        .into_iter()
        .filter(|msg| {
            // Filter by sender
            if let Some(ref sender_did) = from {
                if !msg.from_did.contains(sender_did) {
                    return false;
                }
            }

            // Filter by trust score
            if let Some(min_trust) = trust_min {
                let sender_trust = trust_scores.get(&msg.from_did).copied().unwrap_or(0.0);
                if sender_trust < min_trust {
                    return false;
                }
            }

            // Filter by unread status
            if unread {
                // Generate a unique message identifier for read status tracking
                let msg_id = format!("{}:{}:{}", msg.from_did, msg.timestamp, msg.body_cid);
                if read_messages.contains(&msg_id) {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Display messages in table format
fn display_table(messages: &[MailMessage]) {
    println!(
        "{:<6} {:<40} {:<20} {:<20} {:<6}",
        "ID", "From", "Subject", "Time", "Tier"
    );
    println!("{}", "─".repeat(95));

    for (i, msg) in messages.iter().enumerate() {
        let msg_id = format!("#{}", i + 1);
        let from_short = truncate_did(&msg.from_did, 38);
        let subject = decrypt_subject(&msg.subject_encrypted);
        let subject_short = truncate_string(&subject, 18);
        let time_str = format_timestamp(msg.timestamp);
        let tier_short = format_tier_short(&msg.epistemic_tier);

        println!(
            "{:<6} {:<40} {:<20} {:<20} {:<6}",
            msg_id, from_short, subject_short, time_str, tier_short
        );
    }

    println!();
    println!("💡 Use 'mycelix-mail read <id>' to view full message");
}

/// Display messages in JSON format
fn display_json(messages: &[MailMessage]) -> Result<()> {
    let json =
        serde_json::to_string_pretty(messages).context("Failed to serialize messages to JSON")?;
    println!("{}", json);
    Ok(())
}

/// Display messages in raw format
fn display_raw(messages: &[MailMessage]) {
    for (i, msg) in messages.iter().enumerate() {
        println!("Message #{}", i + 1);
        println!("  From: {}", msg.from_did);
        println!("  To: {}", msg.to_did);
        println!(
            "  Subject (encrypted): {} bytes",
            msg.subject_encrypted.len()
        );
        println!("  Body CID: {}", msg.body_cid);
        println!(
            "  Timestamp: {} ({})",
            msg.timestamp,
            format_timestamp(msg.timestamp)
        );
        println!("  Tier: {:?}", msg.epistemic_tier);
        if let Some(ref thread) = msg.thread_id {
            println!("  Thread: {}", thread);
        }
        println!();
    }
}

/// Truncate a DID for display
fn truncate_did(did: &str, max_len: usize) -> String {
    if did.len() <= max_len {
        did.to_string()
    } else {
        // Show prefix and suffix
        let prefix_len = max_len.saturating_sub(10);
        let suffix_len = 7;
        format!(
            "{}...{}",
            &did[..prefix_len],
            &did[did.len() - suffix_len..]
        )
    }
}

/// Truncate a string for display
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Format timestamp as relative time
fn format_timestamp(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());

    let now = Utc::now();
    let diff = now.signed_duration_since(dt);

    if diff.num_seconds() < 60 {
        "just now".to_string()
    } else if diff.num_minutes() < 60 {
        format!("{}m ago", diff.num_minutes())
    } else if diff.num_hours() < 24 {
        format!("{}h ago", diff.num_hours())
    } else if diff.num_days() < 7 {
        format!("{}d ago", diff.num_days())
    } else {
        dt.format("%Y-%m-%d").to_string()
    }
}

/// Format epistemic tier as short string
fn format_tier_short(tier: &crate::types::EpistemicTier) -> String {
    use crate::types::EpistemicTier;
    match tier {
        EpistemicTier::Tier0Null => "T0".to_string(),
        EpistemicTier::Tier1Testimonial => "T1".to_string(),
        EpistemicTier::Tier2PrivatelyVerifiable => "T2".to_string(),
        EpistemicTier::Tier3CryptographicallyProven => "T3".to_string(),
        EpistemicTier::Tier4PubliclyReproducible => "T4".to_string(),
    }
}

/// Decrypt subject line from encrypted bytes
///
/// This is a placeholder implementation that handles the current "ENC:" prefix format.
/// Real NaCl box decryption requires:
/// 1. Fetching the sender's public key from DID registry
/// 2. Using our private key (from keyring) to decrypt via crypto_box_open
/// 3. Handling nonce extraction from the encrypted payload
///
/// The encryption format should be: nonce (24 bytes) || ciphertext
/// Decryption: crypto_box_open(ciphertext, nonce, sender_pubkey, our_privkey)
fn decrypt_subject(encrypted: &[u8]) -> String {
    // Current placeholder handles:
    // 1. "ENC:" prefixed strings (our test/stub format)
    // 2. Plain UTF-8 strings (for backwards compatibility)
    // 3. Binary data returns "<encrypted>" indicator

    if let Ok(s) = String::from_utf8(encrypted.to_vec()) {
        // Strip our test "ENC:" prefix if present
        if s.starts_with("ENC:") {
            s[4..].to_string()
        } else {
            s
        }
    } else {
        "<encrypted>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EpistemicTier;

    #[test]
    fn test_truncate_did() {
        let did = "did:mycelix:ATHMuhr4Mk9fx2VMUx5kzVPVkL5zyvQGZ1gofWQmJtG6";
        let truncated = truncate_did(did, 30);
        assert!(truncated.len() <= 30);
        assert!(truncated.contains("..."));
    }

    #[test]
    fn test_truncate_string() {
        let long = "This is a very long string that should be truncated";
        let short = truncate_string(long, 20);
        assert_eq!(short.len(), 20);
        assert!(short.ends_with("..."));
    }

    #[test]
    fn test_format_tier_short() {
        assert_eq!(format_tier_short(&EpistemicTier::Tier0Null), "T0");
        assert_eq!(
            format_tier_short(&EpistemicTier::Tier2PrivatelyVerifiable),
            "T2"
        );
        assert_eq!(
            format_tier_short(&EpistemicTier::Tier4PubliclyReproducible),
            "T4"
        );
    }

    #[test]
    fn test_decrypt_subject_placeholder() {
        let encrypted = b"ENC:Test Subject";
        let decrypted = decrypt_subject(encrypted);
        assert_eq!(decrypted, "Test Subject");
    }

    #[test]
    fn test_apply_filters_empty() {
        let messages = vec![];
        let trust_scores = std::collections::HashMap::new();
        let read_messages = std::collections::HashSet::new();
        let filtered = apply_filters(messages, None, None, false, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_apply_filters_by_sender() {
        let msg = MailMessage {
            from_did: "did:mycelix:ABC123".to_string(),
            to_did: "did:mycelix:XYZ789".to_string(),
            subject_encrypted: b"Test".to_vec(),
            body_cid: "bafyrei123".to_string(),
            timestamp: 1234567890,
            thread_id: None,
            epistemic_tier: EpistemicTier::Tier2PrivatelyVerifiable,
        };

        let messages = vec![msg.clone()];
        let trust_scores = std::collections::HashMap::new();
        let read_messages = std::collections::HashSet::new();

        // Should match
        let filtered = apply_filters(messages.clone(), Some("ABC".to_string()), None, false, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 1);

        // Should not match
        let filtered = apply_filters(messages.clone(), Some("ZZZ".to_string()), None, false, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_apply_filters_by_trust_score() {
        let msg = MailMessage {
            from_did: "did:mycelix:ABC123".to_string(),
            to_did: "did:mycelix:XYZ789".to_string(),
            subject_encrypted: b"Test".to_vec(),
            body_cid: "bafyrei123".to_string(),
            timestamp: 1234567890,
            thread_id: None,
            epistemic_tier: EpistemicTier::Tier2PrivatelyVerifiable,
        };

        let messages = vec![msg.clone()];
        let mut trust_scores = std::collections::HashMap::new();
        trust_scores.insert("did:mycelix:ABC123".to_string(), 0.8);
        let read_messages = std::collections::HashSet::new();

        // Should match (trust 0.8 >= 0.5)
        let filtered = apply_filters(messages.clone(), None, Some(0.5), false, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 1);

        // Should not match (trust 0.8 < 0.9)
        let filtered = apply_filters(messages.clone(), None, Some(0.9), false, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_apply_filters_by_read_status() {
        let msg = MailMessage {
            from_did: "did:mycelix:ABC123".to_string(),
            to_did: "did:mycelix:XYZ789".to_string(),
            subject_encrypted: b"Test".to_vec(),
            body_cid: "bafyrei123".to_string(),
            timestamp: 1234567890,
            thread_id: None,
            epistemic_tier: EpistemicTier::Tier2PrivatelyVerifiable,
        };

        let messages = vec![msg.clone()];
        let trust_scores = std::collections::HashMap::new();
        let mut read_messages = std::collections::HashSet::new();

        // Should match (unread filter on, message not read)
        let filtered = apply_filters(messages.clone(), None, None, true, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 1);

        // Mark as read
        let msg_id = format!("{}:{}:{}", msg.from_did, msg.timestamp, msg.body_cid);
        read_messages.insert(msg_id);

        // Should not match (unread filter on, message is read)
        let filtered = apply_filters(messages.clone(), None, None, true, &trust_scores, &read_messages);
        assert_eq!(filtered.len(), 0);
    }
}
