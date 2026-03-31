// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;

// ─── Text Processing Tests ──────────────────────────────────────

#[test]
fn process_text_basic() {
    let terms = process_text("Hello World Test");
    assert_eq!(terms, vec!["hello", "world", "test"]);
}

#[test]
fn process_text_filters_short_words() {
    let terms = process_text("I am a big red bus");
    // "I", "am", "a" are <= 2 chars, should be filtered
    assert!(!terms.contains(&"i".to_string()));
    assert!(!terms.contains(&"am".to_string()));
    assert!(terms.contains(&"big".to_string()));
    assert!(terms.contains(&"red".to_string()));
    assert!(terms.contains(&"bus".to_string()));
}

#[test]
fn process_text_strips_punctuation() {
    let terms = process_text("hello, world! test.");
    assert!(terms.contains(&"hello".to_string()));
    assert!(terms.contains(&"world".to_string()));
    assert!(terms.contains(&"test".to_string()));
}

#[test]
fn process_text_lowercases() {
    let terms = process_text("MacBook PRO Laptop");
    assert!(terms.contains(&"macbook".to_string()));
    assert!(terms.contains(&"pro".to_string()));
    assert!(terms.contains(&"laptop".to_string()));
}

#[test]
fn process_text_empty_input() {
    let terms = process_text("");
    assert!(terms.is_empty());
}

#[test]
fn process_text_only_short_words() {
    let terms = process_text("I am a");
    assert!(terms.is_empty());
}

// ─── Term Frequency Tests ───────────────────────────────────────

#[test]
fn calculate_term_frequencies_basic() {
    let terms = vec!["hello".into(), "world".into(), "hello".into()];
    let freqs = calculate_term_frequencies(&terms);
    assert_eq!(freqs.get("hello"), Some(&2));
    assert_eq!(freqs.get("world"), Some(&1));
}

#[test]
fn calculate_term_frequencies_empty() {
    let terms: Vec<String> = vec![];
    let freqs = calculate_term_frequencies(&terms);
    assert!(freqs.is_empty());
}

#[test]
fn calculate_term_frequencies_single_term() {
    let terms = vec!["laptop".into()];
    let freqs = calculate_term_frequencies(&terms);
    assert_eq!(freqs.len(), 1);
    assert_eq!(freqs.get("laptop"), Some(&1));
}

#[test]
fn calculate_term_frequencies_all_same() {
    let terms = vec!["test".into(), "test".into(), "test".into()];
    let freqs = calculate_term_frequencies(&terms);
    assert_eq!(freqs.len(), 1);
    assert_eq!(freqs.get("test"), Some(&3));
}

// ─── Input Types Construction ───────────────────────────────────

#[test]
fn search_query_construction() {
    let q = SearchQuery {
        query: "macbook pro".into(),
        entity_types: Some(vec![EntityType::Listing]),
        limit: 20,
    };
    assert_eq!(q.query, "macbook pro");
    assert_eq!(q.limit, 20);
    assert!(q.entity_types.is_some());
}

#[test]
fn search_query_no_filter() {
    let q = SearchQuery {
        query: "test".into(),
        entity_types: None,
        limit: 10,
    };
    assert!(q.entity_types.is_none());
}

#[test]
fn search_response_empty() {
    let resp = SearchResponse {
        results: vec![],
        total_count: 0,
        query_time_ms: 5,
    };
    assert_eq!(resp.total_count, 0);
    assert_eq!(resp.query_time_ms, 5);
    assert!(resp.results.is_empty());
}
