// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sentient Overlay — consciousness-aware epistemic annotation.
//!
//! Two-phase analysis:
//! 1. Fast HDC search: match page sentences against knowledge base (synchronous, sub-ms)
//! 2. Spore reasoning: feed matched sentences through the consciousness kernel
//!    for prediction error, confidence weighting, and epistemic honesty (async, batched)
//!
//! Phase 2 runs asynchronously in batches of BATCH_SIZE sentences,
//! yielding to the browser event loop between batches to avoid freezing.

use leptos::prelude::*;
use prism_common::{EmpiricalLevel, SearchResult};
use prism_search::SearchEngine;

const MAX_SEARCH_CALLS: usize = 50;
const MAX_ANNOTATIONS: usize = 10;
/// Sentences processed per async batch before yielding to the event loop.
const SPORE_BATCH_SIZE: usize = 3;

/// An annotation enriched with Spore consciousness data.
struct SporeAnnotation {
    sentence: String,
    result: SearchResult,
    /// Spore's prediction error for this sentence (surprise signal).
    prediction_error: f32,
    /// Spore's consciousness level when processing this sentence.
    consciousness: f32,
}

/// Scan page text for sentences that match known claims.
/// Returns the HTML with epistemic annotations injected.
///
/// This is the synchronous fast path (HDC-only). The Spore-enriched
/// overlay is rendered by the `ThinkingOverlay` component below.
pub fn annotate_html(html: &str, engine: &SearchEngine) -> String {
    let text = strip_tags(html);
    if text.len() < 50 {
        return crate::engine::sanitize_html(html, None);
    }

    let sentences = split_sentences(&text);
    let annotations = collect_hdc_annotations(&sentences, engine);

    if annotations.is_empty() {
        return crate::engine::sanitize_html(html, None);
    }

    let overlay_html = render_annotations(&annotations, None);
    crate::engine::sanitize_html(&format!("{}\n{}", html, overlay_html), None)
}

/// Escape HTML-significant characters in untrusted text before interpolating
/// it into a raw HTML string. Defense in depth alongside the sanitize_html
/// pass applied to the overall output.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Collect HDC-matched annotations from the search engine (fast, synchronous).
fn collect_hdc_annotations(
    sentences: &[String],
    engine: &SearchEngine,
) -> Vec<(String, SearchResult)> {
    let mut annotations = Vec::new();
    let mut search_calls = 0;

    for sentence in sentences {
        if sentence.len() < 30 {
            continue;
        }
        if search_calls >= MAX_SEARCH_CALLS || annotations.len() >= MAX_ANNOTATIONS {
            break;
        }

        search_calls += 1;
        if let Some(top) = engine.search(sentence, 1).into_iter().next() {
            if top.query_similarity > 0.15 {
                annotations.push((sentence.clone(), top));
            }
        }
    }
    annotations
}

/// Render annotations as HTML. If Spore data is available, includes
/// consciousness-weighted confidence and surprise indicators.
fn render_annotations(
    annotations: &[(String, SearchResult)],
    spore_data: Option<&[SporeAnnotation]>,
) -> String {
    let mut html = String::new();
    html.push_str(r#"<div class="sentient-overlay-section">"#);
    html.push_str(r#"<h3 class="overlay-title">Epistemic Analysis</h3>"#);

    if spore_data.is_some() {
        html.push_str(
            r#"<p class="overlay-subtitle">Consciousness-grounded analysis of this page:</p>"#,
        );
    } else {
        html.push_str(r#"<p class="overlay-subtitle">Prism found claims in the knowledge base that relate to this page:</p>"#);
    }

    for (i, (sentence, result)) in annotations.iter().enumerate() {
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
        let source = result
            .sources
            .first()
            .map(|s| s.as_str())
            .unwrap_or("\u{2014}");
        let sim_pct = (result.query_similarity * 100.0) as u32;

        // Spore enrichment: show consciousness confidence and surprise
        let spore_note = spore_data
            .and_then(|data| data.get(i))
            .map(|sa| {
                let conf_pct = (sa.consciousness * 100.0) as u32;
                if sa.prediction_error > 0.5 {
                    format!(
                        r#"<span class="overlay-surprise">{}% {} surprising</span>"#,
                        conf_pct, "\u{03A8} \u{00B7}"
                    )
                } else if sa.consciousness > 0.4 {
                    format!(
                        r#"<span class="overlay-confident">{}% {} confident</span>"#,
                        conf_pct, "\u{03A8} \u{00B7}"
                    )
                } else {
                    format!(
                        r#"<span class="overlay-uncertain">{}% {} uncertain</span>"#,
                        conf_pct, "\u{03A8} \u{00B7}"
                    )
                }
            })
            .unwrap_or_default();

        html.push_str(&format!(
            r##"<div class="overlay-claim">
                <div class="overlay-match">
                    <span class="e-badge {e_class}">{e_label}</span>
                    <span class="overlay-sim">{sim_pct}% match</span>
                    {spore_note}
                </div>
                <div class="overlay-page-text">"{sentence_short}"</div>
                <div class="overlay-claim-text">{claim}</div>
                <div class="overlay-source">{source}</div>
            </div>"##,
            e_class = e_class,
            e_label = e_label,
            sim_pct = sim_pct,
            spore_note = spore_note,
            sentence_short = escape_html(&sentence[..sentence.len().min(100)]),
            claim = escape_html(&result.content),
            source = escape_html(source),
        ));
    }

    html.push_str("</div>");
    html
}

/// Strip HTML tags to extract visible text (lightweight alternative to full DOM parse).
fn strip_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                result.push(' ');
            }
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
