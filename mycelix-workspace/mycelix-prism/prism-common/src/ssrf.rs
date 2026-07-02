// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared SSRF (Server-Side Request Forgery) protection for Prism's proxy layer.
//!
//! Both `prism-proxy` (standalone dev proxy) and `prism-serve` (unified
//! production server) accept a `url` query parameter and fetch it on the
//! server's behalf. Without validation, an attacker can point the proxy at
//! loopback addresses, RFC1918 private ranges, link-local addresses (incl.
//! cloud metadata endpoints like `169.254.169.254`), or other internal-only
//! schemes/hosts.
//!
//! This module is the single source of truth for that validation so a future
//! patch to one binary can't silently miss the other. It intentionally has
//! no dependency beyond `url` (already a `prism-common` dependency) so it
//! stays compilable for both native (proxy/serve) and `wasm32-unknown-unknown`
//! (prism-ui) targets that depend on `prism-common`.

/// Check if an IPv4 address is private/reserved and therefore unsafe to proxy to.
pub fn is_private_ipv4(ip: &std::net::Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || (ip.octets()[0] == 169 && ip.octets()[1] == 254)
}

/// Validate that a URL is safe to proxy — reject private/reserved addresses
/// and non-HTTP schemes.
///
/// Returns the parsed `url::Url` on success, or a static error string
/// describing why the URL was rejected.
pub fn validate_proxy_url(raw: &str) -> Result<url::Url, &'static str> {
    let parsed = url::Url::parse(raw).map_err(|_| "Invalid URL")?;

    match parsed.scheme() {
        "http" | "https" => {}
        _ => return Err("Only http and https URLs are allowed"),
    }

    let host_str = parsed.host_str().unwrap_or("");

    if host_str == "localhost" || host_str == "metadata.google.internal" {
        return Err("Access to internal addresses is forbidden");
    }

    // Use url::Host for proper IPv4/IPv6 discrimination
    match parsed.host() {
        Some(url::Host::Ipv4(ip)) => {
            if is_private_ipv4(&ip) {
                return Err("Access to private/reserved IP addresses is forbidden");
            }
        }
        Some(url::Host::Ipv6(ip)) => {
            if ip.is_loopback() || ip.is_unspecified() {
                return Err("Access to private/reserved IP addresses is forbidden");
            }

            let seg = ip.segments();

            // IPv4-mapped IPv6 (::ffff:x.x.x.x)
            if seg[0] == 0
                && seg[1] == 0
                && seg[2] == 0
                && seg[3] == 0
                && seg[4] == 0
                && seg[5] == 0xffff
            {
                let mapped = std::net::Ipv4Addr::new(
                    (seg[6] >> 8) as u8,
                    seg[6] as u8,
                    (seg[7] >> 8) as u8,
                    seg[7] as u8,
                );
                if is_private_ipv4(&mapped) {
                    return Err("Access to private/reserved IP addresses is forbidden");
                }
            }

            // IPv6 link-local (fe80::/10)
            if (seg[0] & 0xffc0) == 0xfe80 {
                return Err("Access to private/reserved IP addresses is forbidden");
            }

            // IPv6 unique-local (fc00::/7)
            if (seg[0] & 0xfe00) == 0xfc00 {
                return Err("Access to private/reserved IP addresses is forbidden");
            }
        }
        _ => {} // Domain names pass through (validated by DNS at fetch time)
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_localhost() {
        assert!(validate_proxy_url("http://localhost:8080").is_err());
        assert!(validate_proxy_url("http://127.0.0.1:5432").is_err());
        assert!(validate_proxy_url("http://[::1]:80").is_err());
        assert!(validate_proxy_url("http://0.0.0.0").is_err());
    }

    #[test]
    fn rejects_private_ips() {
        assert!(validate_proxy_url("http://10.0.0.1").is_err());
        assert!(validate_proxy_url("http://172.16.0.1").is_err());
        assert!(validate_proxy_url("http://192.168.1.1").is_err());
    }

    #[test]
    fn rejects_cloud_metadata() {
        assert!(validate_proxy_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_proxy_url("http://metadata.google.internal").is_err());
    }

    #[test]
    fn rejects_non_http_schemes() {
        assert!(validate_proxy_url("file:///etc/passwd").is_err());
        assert!(validate_proxy_url("ftp://example.com").is_err());
        assert!(validate_proxy_url("gopher://evil.com").is_err());
    }

    #[test]
    fn rejects_ipv4_mapped_ipv6() {
        // ::ffff:127.0.0.1 — loopback disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:127.0.0.1]").is_err());
        // ::ffff:192.168.1.1 — private IP disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:192.168.1.1]").is_err());
        // ::ffff:169.254.169.254 — cloud metadata disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:169.254.169.254]").is_err());
        // ::ffff:10.0.0.1 — private range disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:10.0.0.1]").is_err());
    }

    #[test]
    fn rejects_ipv6_link_local_and_ula() {
        // fe80:: — link-local
        assert!(validate_proxy_url("http://[fe80::1]").is_err());
        // fc00:: — unique-local address
        assert!(validate_proxy_url("http://[fc00::1]").is_err());
        // fd00:: — also unique-local
        assert!(validate_proxy_url("http://[fd12::1]").is_err());
    }

    #[test]
    fn allows_valid_external_urls() {
        assert!(validate_proxy_url("https://example.com").is_ok());
        assert!(validate_proxy_url("https://api.duckduckgo.com/?q=test").is_ok());
        assert!(validate_proxy_url("http://en.wikipedia.org/wiki/Rust").is_ok());
    }
}
