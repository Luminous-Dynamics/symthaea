// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sentient Overlay — the Voice of the People.
//!
//! When viewing a fetched page, scans the page text for sentences that
//! match known claims in the search index. Matching sentences get
//! highlighted with their epistemic level (E0-E4), source, and trust score.
//!
//! This transforms every webpage into an epistemically annotated document.

use leptos::prelude::*;
use prism_common::{EmpiricalLevel, SearchResult};
use prism_search::SearchEngine;

const MAX_SEARCH_CALLS: usize = 50;
const MAX_ANNOTATIONS: usize = 10;

/// Scan page text for sentences that match known claims.
/// Returns the HTML with epistemic annotations injected.
pub fn annotate_html(html: &str, engine: &SearchEngine) -> String {
    // Extract visible text via tag stripping (avoids full DOM re-parse since
    // the HTML was already parsed and sanitized in engine.rs)
    let text = strip_tags(html);

    if text.len() < 50 {
        return html.to_string();
    }

    // Split text into sentences
    let sentences = split_sentences(&text);

    // Find matching claims for each sentence (capped for performance)
    let mut annotations: Vec<(String, Option<SearchResult>)> = Vec::new();
    let mut search_calls = 0;
    let mut match_count = 0;
    for sentence in &sentences {
        if sentence.len() < 30 {
            continue;
        }
        if search_calls >= MAX_SEARCH_CALLS || match_count >= MAX_ANNOTATIONS {
            break;
        }
        search_calls += 1;
        let results = engine.search(sentence, 1);
        if let Some(top) = results.first() {
            if top.query_similarity > 0.15 {
                annotations.push((sentence.clone(), Some(top.clone())));
                match_count += 1;
            }
        }
    }

    // If no annotations found, return original
    if annotations.is_empty() {
        return html.to_string();
    }

    // Build annotation summary to append after the content
    let mut overlay_html = String::new();
    overlay_html.push_str(r#"<div class="sentient-overlay-section">"#);
    overlay_html.push_str(r#"<h3 class="overlay-title">Epistemic Analysis</h3>"#);
    overlay_html.push_str(r#"<p class="overlay-subtitle">Prism found claims in the knowledge base that relate to this page:</p>"#);

    let mut found = 0;
    for (sentence, annotation) in &annotations {
        if let Some(result) = annotation {
            found += 1;
            let e_class = match result.empirical_level {
                EmpiricalLevel::E4 => "e4",
                EmpiricalLevel::E3 => "e3",
                EmpiricalLevel::E2 => "e2",
                EmpiricalLevel::E1 => "e1",
                EmpiricalLevel::E0 => "e0",
            };
            let e_label = match result.empirical_level {
                EmpiricalLevel::E4 => "E4 Established",
                EmpiricalLevel::E3 => "E3 Replicated",
                EmpiricalLevel::E2 => "E2 Tested",
                EmpiricalLevel::E1 => "E1 Preliminary",
                EmpiricalLevel::E0 => "E0 Unverified",
            };
            let source = result.sources.first().map(|s| s.as_str()).unwrap_or("—");
            let sim_pct = (result.query_similarity * 100.0) as u32;

            overlay_html.push_str(&format!(
                r##"<div class="overlay-claim">
                    <div class="overlay-match">
                        <span class="e-badge {e_class}">{e_label}</span>
                        <span class="overlay-sim">{sim_pct}% match</span>
                    </div>
                    <div class="overlay-page-text">"{sentence_short}"</div>
                    <div class="overlay-claim-text">{claim}</div>
                    <div class="overlay-source">{source}</div>
                </div>"##,
                e_class = e_class,
                e_label = e_label,
                sim_pct = sim_pct,
                sentence_short = &sentence[..sentence.len().min(100)],
                claim = result.content,
                source = source,
            ));
        }
    }

    overlay_html.push_str("</div>");

    if found > 0 {
        // Append the overlay section after the page content
        format!("{}\n{}", html, overlay_html)
    } else {
        html.to_string()
    }
}

/// Strip HTML tags to extract visible text (lightweight alternative to full DOM parse).
fn strip_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => { in_tag = false; result.push(' '); }
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result
}

/// Split text into sentences (simple heuristic).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') && current.len() > 15 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }
    // Remaining text
    let trimmed = current.trim().to_string();
    if trimmed.len() > 15 {
        sentences.push(trimmed);
    }

    sentences
}

/// Leptos component that wraps annotated content.
#[component]
pub fn AnnotatedContent(html: String, engine: StoredValue<SearchEngine>) -> impl IntoView {
    let annotated = engine.with_value(|e| annotate_html(&html, e));

    view! {
        <div class="reader-content" inner_html=annotated></div>
    }
}
