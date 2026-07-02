// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SSRF guard shared by prism-proxy and prism-serve.
//!
//! Was previously duplicated verbatim in both binaries — factored out so a
//! future patch to the allowlist can't silently miss one of them.

/// Check if an IPv4 address is private/reserved.
pub fn is_private_ipv4(ip: &std::net::Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || (ip.octets()[0] == 169 && ip.octets()[1] == 254)
}

/// Validate that a URL is safe to proxy — reject private/reserved addresses and non-HTTP schemes.
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
        assert!(validate_proxy_url("http://localhost/").is_err());
        assert!(validate_proxy_url("http://127.0.0.1/").is_err());
    }

    #[test]
    fn rejects_private_ranges() {
        assert!(validate_proxy_url("http://10.0.0.1/").is_err());
        assert!(validate_proxy_url("http://192.168.1.1/").is_err());
        assert!(validate_proxy_url("http://169.254.169.254/").is_err());
    }

    #[test]
    fn rejects_cloud_metadata() {
        assert!(validate_proxy_url("http://metadata.google.internal/").is_err());
    }

    #[test]
    fn rejects_ipv6_loopback_and_mapped() {
        assert!(validate_proxy_url("http://[::1]/").is_err());
        assert!(validate_proxy_url("http://[::ffff:127.0.0.1]/").is_err());
    }

    #[test]
    fn rejects_non_http_scheme() {
        assert!(validate_proxy_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn allows_public_https() {
        assert!(validate_proxy_url("https://example.com/").is_ok());
    }
}
