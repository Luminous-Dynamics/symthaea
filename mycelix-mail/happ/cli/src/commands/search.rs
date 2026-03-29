// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::client::MycellixClient;
use crate::types::{EpistemicTier, MailMessage};
use anyhow::{Context, Result};
use std::collections::HashMap;

/// Search messages across inbox and sent folders
pub async fn handle_search(
    client: &MycellixClient,
    query: String,
    in_field: &str,
    limit: usize,
    format: &str,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                      MESSAGE SEARCH");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🔍 Query:  \"{}\"", query);
    println!("📂 Field:  {}", format_field(in_field));
    println!("🔢 Limit:  {}", limit);
    println!();

    // 1. Fetch messages from both inbox and sent
    println!("📬 Searching inbox...");
    let inbox_messages = client
        .get_inbox()
        .await
        .context("Failed to fetch inbox messages")?;

    println!("📭 Searching sent messages...");
    let sent_messages = client
        .get_sent()
        .await
        .context("Failed to fetch sent messages")?;

    // 2. Combine and search
    let mut all_messages = inbox_messages;
    all_messages.extend(sent_messages);

    println!();
    println!("🔎 Searching {} total message(s)...", all_messages.len());
    println!();

    // 3. Pre-fetch body content if searching in body field
    let mut body_cache = HashMap::new();
    if in_field == "body" || in_field == "all" {
        println!("Fetching message bodies for search...");
        for msg in &all_messages {
            if let Some(body) = fetch_body_content(&msg.body_cid).await {
                body_cache.insert(msg.body_cid.clone(), body);
            }
        }
        if !body_cache.is_empty() {
            println!("Fetched {} message bod(ies)", body_cache.len());
        }
    }

    // 4. Apply search filter with body cache
    let mut results = search_messages_with_cache(&all_messages, &query, in_field, &body_cache);

    // 5. Sort by timestamp (newest first)
    results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // 6. Apply limit
    let total_results = results.len();
    results.truncate(limit);

    // Handle no results
    if results.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("No messages found matching your search.");
        println!();
        println!("💡 Tips:");
        println!("   • Try a shorter or more general search term");
        println!("   • Try searching in different fields (--field all/from/subject/body)");
        println!("   • Check for typos in your query");
        return Ok(());
    }

    // 6. Display results
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "Found {} result(s), showing {}",
        total_results,
        results.len()
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match format {
        "json" => display_json(&results)?,
        "raw" => display_raw(&results),
        _ => display_table(&results, &query),
    }

    println!();
    if results.len() < total_results {
        println!(
            "💡 Showing first {} results. Use --limit {} to see more.",
            limit, total_results
        );
    }

    Ok(())
}

/// Fetch body content from IPFS/DHT using CID
async fn fetch_body_content(cid: &str) -> Option<String> {
    // Try local IPFS gateway first
    let ipfs_gateways = [
        "http://localhost:8080/ipfs",
        "http://127.0.0.1:8080/ipfs",
        "https://ipfs.io/ipfs",
        "https://dweb.link/ipfs",
    ];

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    for gateway in ipfs_gateways {
        let url = format!("{}/{}", gateway, cid);
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                if let Ok(text) = response.text().await {
                    return Some(text);
                }
            }
            _ => continue,
        }
    }

    None
}

/// Search messages based on query and field with body content support
fn search_messages_with_cache(
    messages: &[MailMessage],
    query: &str,
    field: &str,
    body_cache: &HashMap<String, String>,
) -> Vec<MailMessage> {
    let query_lower = query.to_lowercase();

    messages
        .iter()
        .filter(|msg| {
            match field {
                "from" => msg.from_did.to_lowercase().contains(&query_lower),
                "to" => msg.to_did.to_lowercase().contains(&query_lower),
                "subject" => {
                    let subject = decrypt_subject(&msg.subject_encrypted);
                    subject.to_lowercase().contains(&query_lower)
                }
                "body" => {
                    // Search in fetched body content if available
                    if let Some(body) = body_cache.get(&msg.body_cid) {
                        body.to_lowercase().contains(&query_lower)
                    } else {
                        // Fall back to CID search
                        msg.body_cid.to_lowercase().contains(&query_lower)
                    }
                }
                "all" | _ => {
                    // Search in all fields
                    let subject = decrypt_subject(&msg.subject_encrypted);
                    let body_match = if let Some(body) = body_cache.get(&msg.body_cid) {
                        body.to_lowercase().contains(&query_lower)
                    } else {
                        msg.body_cid.to_lowercase().contains(&query_lower)
                    };

                    msg.from_did.to_lowercase().contains(&query_lower)
                        || msg.to_did.to_lowercase().contains(&query_lower)
                        || subject.to_lowercase().contains(&query_lower)
                        || body_match
                }
            }
        })
        .cloned()
        .collect()
}

/// Legacy search function for backward compatibility
fn search_messages(messages: &[MailMessage], query: &str, field: &str) -> Vec<MailMessage> {
    search_messages_with_cache(messages, query, field, &HashMap::new())
}

/// Display results in table format
fn display_table(messages: &[MailMessage], query: &str) {
    println!(
        "{:<6} {:<30} {:<30} {:<25} {:<6}",
        "ID", "From", "To", "Subject", "Tier"
    );
    println!("{}", "─".repeat(100));

    for (i, msg) in messages.iter().enumerate() {
        let msg_id = format!("#{}", i + 1);
        let from_short = truncate_string(&msg.from_did, 28);
        let to_short = truncate_string(&msg.to_did, 28);
        let subject = decrypt_subject(&msg.subject_encrypted);
        let subject_short = truncate_string(&subject, 23);
        let tier_short = format_tier_short(&msg.epistemic_tier);

        // Highlight query in subject if it matches
        let subject_display = if subject.to_lowercase().contains(&query.to_lowercase()) {
            format!("{}*", subject_short)
        } else {
            subject_short
        };

        println!(
            "{:<6} {:<30} {:<30} {:<25} {:<6}",
            msg_id, from_short, to_short, subject_display, tier_short
        );
    }

    println!();
    println!("💡 Use 'mycelix-mail read <id>' to view full message");
    println!("   (* = matches search query)");
}

/// Display results in JSON format
fn display_json(messages: &[MailMessage]) -> Result<()> {
    let json =
        serde_json::to_string_pretty(messages).context("Failed to serialize messages to JSON")?;
    println!("{}", json);
    Ok(())
}

/// Display results in raw format
fn display_raw(messages: &[MailMessage]) {
    for (i, msg) in messages.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Result #{}", i + 1);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("From:      {}", msg.from_did);
        println!("To:        {}", msg.to_did);
        println!("Subject:   {}", decrypt_subject(&msg.subject_encrypted));
        println!("Body CID:  {}", msg.body_cid);
        println!("Timestamp: {}", msg.timestamp);
        println!("Tier:      {:?}", msg.epistemic_tier);
        if let Some(ref thread) = msg.thread_id {
            println!("Thread:    {}", thread);
        }
        println!();
    }
}

/// Format field name for display
fn format_field(field: &str) -> String {
    match field {
        "from" => "Sender (From)",
        "to" => "Recipient (To)",
        "subject" => "Subject",
        "body" => "Message Body",
        "all" | _ => "All Fields",
    }
    .to_string()
}

/// Truncate string for display
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Format epistemic tier as short string
fn format_tier_short(tier: &EpistemicTier) -> String {
    match tier {
        EpistemicTier::Tier0Null => "T0",
        EpistemicTier::Tier1Testimonial => "T1",
        EpistemicTier::Tier2PrivatelyVerifiable => "T2",
        EpistemicTier::Tier3CryptographicallyProven => "T3",
        EpistemicTier::Tier4PubliclyReproducible => "T4",
    }
    .to_string()
}

/// Decrypt subject (placeholder implementation)
fn decrypt_subject(encrypted: &[u8]) -> String {
    if let Ok(s) = String::from_utf8(encrypted.to_vec()) {
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

    #[test]
    fn test_truncate_string() {
        let short = "Hello";
        assert_eq!(truncate_string(short, 10), "Hello");

        let long = "This is a very long string";
        let truncated = truncate_string(long, 15);
        assert_eq!(truncated.len(), 15);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_format_field() {
        assert_eq!(format_field("from"), "Sender (From)");
        assert_eq!(format_field("to"), "Recipient (To)");
        assert_eq!(format_field("subject"), "Subject");
        assert_eq!(format_field("body"), "Message Body");
        assert_eq!(format_field("all"), "All Fields");
        assert_eq!(format_field("unknown"), "All Fields");
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
    fn test_decrypt_subject() {
        let encrypted = b"ENC:Test Subject";
        let decrypted = decrypt_subject(encrypted);
        assert_eq!(decrypted, "Test Subject");

        let plain = b"Plain Subject";
        let result = decrypt_subject(plain);
        assert_eq!(result, "Plain Subject");
    }

    #[test]
    fn test_search_messages_by_from() {
        let messages = vec![
            MailMessage {
                from_did: "did:mycelix:ABC123".to_string(),
                to_did: "did:mycelix:XYZ789".to_string(),
                subject_encrypted: b"Test".to_vec(),
                body_cid: "bafyrei123".to_string(),
                timestamp: 1234567890,
                thread_id: None,
                epistemic_tier: EpistemicTier::Tier2PrivatelyVerifiable,
            },
            MailMessage {
                from_did: "did:mycelix:DEF456".to_string(),
                to_did: "did:mycelix:XYZ789".to_string(),
                subject_encrypted: b"Another".to_vec(),
                body_cid: "bafyrei456".to_string(),
                timestamp: 1234567891,
                thread_id: None,
                epistemic_tier: EpistemicTier::Tier1Testimonial,
            },
        ];

        // Should find message from ABC
        let results = search_messages(&messages, "ABC", "from");
        assert_eq!(results.len(), 1);
        assert!(results[0].from_did.contains("ABC"));

        // Should not find anything
        let results = search_messages(&messages, "ZZZ", "from");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_messages_all_fields() {
        let messages = vec![MailMessage {
            from_did: "did:mycelix:sender1".to_string(),
            to_did: "did:mycelix:recipient1".to_string(),
            subject_encrypted: b"ENC:Important Message".to_vec(),
            body_cid: "bafyrei123".to_string(),
            timestamp: 1234567890,
            thread_id: None,
            epistemic_tier: EpistemicTier::Tier2PrivatelyVerifiable,
        }];

        // Should find by sender
        let results = search_messages(&messages, "sender1", "all");
        assert_eq!(results.len(), 1);

        // Should find by subject
        let results = search_messages(&messages, "Important", "all");
        assert_eq!(results.len(), 1);

        // Should not find
        let results = search_messages(&messages, "nonexistent", "all");
        assert_eq!(results.len(), 0);
    }
}
