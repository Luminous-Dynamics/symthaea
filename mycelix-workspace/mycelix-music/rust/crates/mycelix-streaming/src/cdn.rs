// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CDN (Content Delivery Network) optimization

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// CDN provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnConfig {
    pub provider: CdnProvider,
    pub base_url: String,
    pub signing_key: Option<String>,
    pub token_expiry: Duration,
}

/// Supported CDN providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdnProvider {
    CloudFront,
    Cloudflare,
    Akamai,
    Fastly,
    BunnyCdn,
    Custom,
}

/// Generate signed CDN URL
pub fn generate_signed_url(config: &CdnConfig, path: &str) -> String {
    let expires = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        + config.token_expiry.as_secs();

    match config.provider {
        CdnProvider::CloudFront => generate_cloudfront_url(config, path, expires),
        CdnProvider::Cloudflare => generate_cloudflare_url(config, path, expires),
        CdnProvider::Akamai => generate_akamai_url(config, path, expires),
        _ => format!("{}{}", config.base_url, path),
    }
}

fn generate_cloudfront_url(config: &CdnConfig, path: &str, expires: u64) -> String {
    // Simplified CloudFront signed URL
    let policy = format!(
        r#"{{"Statement":[{{"Resource":"{}{}","Condition":{{"DateLessThan":{{"AWS:EpochTime":{}}}}}}}]}}"#,
        config.base_url, path, expires
    );

    let signature = if let Some(key) = &config.signing_key {
        // Would use proper RSA signing in production
        simple_hash(&format!("{}:{}", policy, key))
    } else {
        String::new()
    };

    format!(
        "{}{}?Expires={}&Signature={}",
        config.base_url, path, expires, signature
    )
}

fn generate_cloudflare_url(config: &CdnConfig, path: &str, expires: u64) -> String {
    let token = if let Some(key) = &config.signing_key {
        simple_hash(&format!("{}:{}:{}", path, expires, key))
    } else {
        String::new()
    };

    format!(
        "{}{}?token={}&expires={}",
        config.base_url, path, token, expires
    )
}

fn generate_akamai_url(config: &CdnConfig, path: &str, expires: u64) -> String {
    let token = if let Some(key) = &config.signing_key {
        let data = format!("exp={}&acl={}&", expires, path);
        simple_hash(&format!("{}{}", data, key))
    } else {
        String::new()
    };

    format!(
        "{}{}?__token__=exp={}&acl={}&hmac={}",
        config.base_url, path, expires, path, token
    )
}

fn simple_hash(input: &str) -> String {
    // Simple hash for demonstration - would use proper HMAC in production
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// CDN cache headers
#[derive(Debug, Clone)]
pub struct CacheHeaders {
    pub cache_control: String,
    pub cdn_cache_control: Option<String>,
    pub surrogate_control: Option<String>,
    pub vary: Vec<String>,
}

impl CacheHeaders {
    pub fn for_segment(max_age: u32) -> Self {
        Self {
            cache_control: format!("public, max-age={}, immutable", max_age),
            cdn_cache_control: Some(format!("max-age={}", max_age * 2)),
            surrogate_control: Some(format!("max-age={}", max_age * 10)),
            vary: vec!["Accept-Encoding".to_string()],
        }
    }

    pub fn for_manifest(max_age: u32) -> Self {
        Self {
            cache_control: format!("public, max-age={}, stale-while-revalidate={}", max_age, max_age / 2),
            cdn_cache_control: Some(format!("max-age={}", max_age)),
            surrogate_control: None,
            vary: vec!["Accept".to_string()],
        }
    }

    pub fn no_cache() -> Self {
        Self {
            cache_control: "no-cache, no-store, must-revalidate".to_string(),
            cdn_cache_control: None,
            surrogate_control: None,
            vary: vec![],
        }
    }
}

/// Edge location hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLocation {
    pub code: String,
    pub region: String,
    pub city: Option<String>,
    pub latency_ms: Option<u32>,
}

/// Select best edge location based on client location
pub fn select_edge_location<'a>(client_country: &str, available_edges: &'a [EdgeLocation]) -> Option<&'a EdgeLocation> {
    // Prefer edges in same country/region
    available_edges
        .iter()
        .find(|e| e.code.starts_with(client_country))
        .or_else(|| {
            // Fall back to lowest latency
            available_edges.iter().min_by_key(|e| e.latency_ms.unwrap_or(u32::MAX))
        })
}
