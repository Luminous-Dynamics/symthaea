// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;

#[test]
fn create_security_log_input_construction() {
    let input = CreateSecurityLogInput {
        event_type: SecurityEventType::XssAttempt,
        severity: SecuritySeverity::High,
        zome_name: "listings_coordinator".into(),
        function_name: "create_listing".into(),
        description: "XSS payload detected in title".into(),
        trigger_input: Some("<script>alert(1)</script>".into()),
        metadata: "{}".into(),
    };
    assert_eq!(input.zome_name, "listings_coordinator");
    assert_eq!(input.function_name, "create_listing");
    assert!(input.trigger_input.is_some());
}

#[test]
fn create_security_log_input_without_trigger() {
    let input = CreateSecurityLogInput {
        event_type: SecurityEventType::RateLimitExceeded,
        severity: SecuritySeverity::Medium,
        zome_name: "search".into(),
        function_name: "search".into(),
        description: "Rate limit exceeded".into(),
        trigger_input: None,
        metadata: "{}".into(),
    };
    assert!(input.trigger_input.is_none());
}

#[test]
fn security_stats_output_construction() {
    let stats = SecurityStatsOutput {
        total_events: 42,
        critical_events: 2,
        high_events: 5,
        events_by_type: vec![("xss_attempt".into(), 10), ("rate_limit_exceeded".into(), 15)],
        top_offenders: vec![],
    };
    assert_eq!(stats.total_events, 42);
    assert_eq!(stats.critical_events, 2);
    assert_eq!(stats.events_by_type.len(), 2);
}

#[test]
fn security_logs_response_empty() {
    let response = SecurityLogsResponse {
        logs: vec![],
        total_count: 0,
    };
    assert_eq!(response.total_count, 0);
    assert!(response.logs.is_empty());
}

#[test]
fn security_severity_ordering() {
    assert!(SecuritySeverity::Low < SecuritySeverity::Medium);
    assert!(SecuritySeverity::Medium < SecuritySeverity::High);
    assert!(SecuritySeverity::High < SecuritySeverity::Critical);
}

#[test]
fn all_event_types_debug_format() {
    let types = vec![
        SecurityEventType::XssAttempt,
        SecurityEventType::InjectionAttempt,
        SecurityEventType::InvalidCid,
        SecurityEventType::RateLimitExceeded,
        SecurityEventType::ProfanityDetected,
        SecurityEventType::SuspiciousActivity,
        SecurityEventType::AuthorizationFailure,
        SecurityEventType::ValidationFailure,
    ];
    assert_eq!(types.len(), 8);
    for t in &types {
        let debug_str = format!("{:?}", t);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn all_severity_levels_equality() {
    assert_eq!(SecuritySeverity::Low, SecuritySeverity::Low);
    assert_eq!(SecuritySeverity::Medium, SecuritySeverity::Medium);
    assert_eq!(SecuritySeverity::High, SecuritySeverity::High);
    assert_eq!(SecuritySeverity::Critical, SecuritySeverity::Critical);
    assert_ne!(SecuritySeverity::Low, SecuritySeverity::Critical);
}
