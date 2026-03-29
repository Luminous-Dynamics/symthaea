// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::client::MycellixClient;
use anyhow::{Context, Result};

/// Get trust score for a DID
pub async fn handle_get(client: &MycellixClient, did: String) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                   TRUST SCORE QUERY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🔍 DID:  {}", did);
    println!();

    // Get trust score from local store
    let trust_score = client
        .get_trust_score(did.clone())
        .await
        .context("Failed to get trust score")?;

    match trust_score {
        Some(score) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("                    TRUST SCORE");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("📊 Score:        {:.2}", score.score);
            println!("📅 Last Updated: {}", format_timestamp(score.last_updated));
            println!("🏷️  Source:       {}", score.source);
            println!();

            // Visual trust indicator
            let trust_bar = format_trust_bar(score.score);
            println!("Trust Level:  {}", trust_bar);
            println!();

            // Interpretation
            let interpretation = interpret_trust_score(score.score);
            println!("💡 {}", interpretation);
        }
        None => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("No trust score found for this DID.");
            println!();
            println!("💡 Tips:");
            println!(
                "   • Use 'mycelix-mail trust sync {}' to fetch from MATL",
                did
            );
            println!(
                "   • Use 'mycelix-mail trust set {} <score>' to set manually",
                did
            );
        }
    }

    Ok(())
}

/// Set trust score for a DID
pub async fn handle_set(client: &MycellixClient, did: String, score: f64) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                 SET TRUST SCORE");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🔍 DID:    {}", did);
    println!("📊 Score:  {:.2}", score);
    println!();

    // Validate score range
    if !(0.0..=1.0).contains(&score) {
        println!("❌ Error: Trust score must be between 0.0 and 1.0");
        return Ok(());
    }

    // Set trust score locally
    client
        .set_trust_score(did.clone(), score)
        .await
        .context("Failed to set trust score")?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                   SCORE UPDATED");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("✅ Trust score set to {:.2} for {}", score, did);
    println!();

    // Visual trust indicator
    let trust_bar = format_trust_bar(score);
    println!("Trust Level:  {}", trust_bar);
    println!();

    // Note about local vs MATL
    println!("💡 Note: This is a local trust score.");
    println!("   Use 'mycelix-mail trust sync' to update from MATL network.");

    Ok(())
}

/// List all trust scores
pub async fn handle_list(client: &MycellixClient, min: Option<f64>, sort: bool) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                  TRUST SCORES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if let Some(min_score) = min {
        println!("🔍 Filter:  Minimum score {:.2}", min_score);
    }
    if sort {
        println!("📊 Sort:    By score (descending)");
    }
    println!();

    // Get all trust scores
    let mut trust_scores = client
        .list_trust_scores()
        .await
        .context("Failed to list trust scores")?;

    // Apply minimum filter
    if let Some(min_score) = min {
        trust_scores.retain(|score| score.score >= min_score);
    }

    // Sort by score if requested
    if sort {
        trust_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }

    // Display results
    if trust_scores.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("No trust scores found.");
        println!();
        println!("💡 Tips:");
        println!("   • Use 'mycelix-mail trust sync' to fetch scores from MATL");
        println!("   • Use 'mycelix-mail trust set <did> <score>' to set manually");
        if min.is_some() {
            println!("   • Try removing the --min filter");
        }
        return Ok(());
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Found {} trust score(s)", trust_scores.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Table header
    println!(
        "{:<40} {:<8} {:<12} {:<10}",
        "DID", "Score", "Source", "Updated"
    );
    println!("{}", "─".repeat(70));

    // Table rows
    for score in trust_scores {
        let did_short = truncate_string(&score.did, 38);
        let score_str = format!("{:.2}", score.score);
        let source_short = truncate_string(&score.source, 10);
        let updated = format_relative_time(score.last_updated);

        println!(
            "{:<40} {:<8} {:<12} {:<10}",
            did_short, score_str, source_short, updated
        );
    }

    println!();
    println!("💡 Use 'mycelix-mail trust get <did>' to view full details");

    Ok(())
}

/// Sync trust scores from MATL
pub async fn handle_sync(client: &MycellixClient, did: Option<String>) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                TRUST SCORE SYNC");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    match did {
        Some(specific_did) => {
            // Sync single DID
            println!("🔄 Syncing trust score for: {}", specific_did);
            println!();

            let trust_score = client
                .sync_trust_score(specific_did.clone())
                .await
                .context("Failed to sync trust score from MATL")?;

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("                    SYNC COMPLETE");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("✅ Trust score synced for: {}", specific_did);
            println!();
            println!("📊 Score:  {:.2}", trust_score.score);
            println!("🏷️  Source: {}", trust_score.source);
            println!();

            // Visual trust indicator
            let trust_bar = format_trust_bar(trust_score.score);
            println!("Trust Level:  {}", trust_bar);
        }
        None => {
            // Sync all trust scores
            println!("🔄 Syncing all trust scores from MATL...");
            println!();

            let trust_scores = client
                .sync_all_trust_scores()
                .await
                .context("Failed to sync all trust scores from MATL")?;

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("                    SYNC COMPLETE");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("✅ Synced {} trust score(s)", trust_scores.len());
            println!();

            if !trust_scores.is_empty() {
                println!("Summary:");
                let avg_score: f64 =
                    trust_scores.iter().map(|s| s.score).sum::<f64>() / trust_scores.len() as f64;
                println!("   • Average Score: {:.2}", avg_score);
                println!("   • Total DIDs:    {}", trust_scores.len());
            }
        }
    }

    println!();
    println!("💡 Use 'mycelix-mail trust list' to view all scores");

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

/// Truncate string for display
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Format trust score as visual bar
fn format_trust_bar(score: f64) -> String {
    let filled = (score * 20.0).round() as usize;
    let empty = 20 - filled;

    let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));

    format!("{} {:.0}%", bar, score * 100.0)
}

/// Interpret trust score with human-readable description
fn interpret_trust_score(score: f64) -> String {
    if score >= 0.9 {
        "Highly trusted - Excellent reputation".to_string()
    } else if score >= 0.7 {
        "Trusted - Good reputation".to_string()
    } else if score >= 0.5 {
        "Moderately trusted - Average reputation".to_string()
    } else if score >= 0.3 {
        "Low trust - Be cautious".to_string()
    } else {
        "Very low trust - High caution recommended".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_trust_bar() {
        let bar_full = format_trust_bar(1.0);
        assert!(bar_full.contains("████████████████████"));
        assert!(bar_full.contains("100%"));

        let bar_half = format_trust_bar(0.5);
        assert!(bar_half.contains("50%"));

        let bar_zero = format_trust_bar(0.0);
        assert!(bar_zero.contains("░░░░░░░░░░░░░░░░░░░░"));
        assert!(bar_zero.contains("0%"));
    }

    #[test]
    fn test_interpret_trust_score() {
        assert!(interpret_trust_score(1.0).contains("Highly trusted"));
        assert!(interpret_trust_score(0.75).contains("Trusted"));
        assert!(interpret_trust_score(0.5).contains("Moderately"));
        assert!(interpret_trust_score(0.3).contains("Low trust"));
        assert!(interpret_trust_score(0.1).contains("Very low"));
    }

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
        let two_days_ago = Utc::now().timestamp() - (2 * 24 * 60 * 60);
        let relative = format_relative_time(two_days_ago);
        assert!(relative.contains("2d ago"));
    }
}
