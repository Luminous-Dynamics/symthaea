// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-source-IP rate limiting pre-rspamd.
//!
//! Cheap cutoff before we spend CPU on SPF/DKIM/DMARC or rspamd. Two
//! governors: one per-minute (30 default), one per-hour (300 default).
//! Both must admit the packet or we 4xx-defer.

use governor::clock::DefaultClock;
use governor::state::keyed::DefaultKeyedStateStore;
use governor::{Quota, RateLimiter};
use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::Arc;

use crate::config::RateLimitConfig;

/// A composite limiter that runs per-minute AND per-hour checks, both keyed
/// by source IP. The packet is admitted only if both admit.
pub struct PerIpLimiter {
    per_minute: Arc<RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock>>,
    per_hour: Arc<RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock>>,
}

impl PerIpLimiter {
    pub fn new(cfg: &RateLimitConfig) -> crate::GatewayResult<Self> {
        let per_minute_quota =
            Quota::per_minute(NonZeroU32::new(cfg.per_ip_per_minute).ok_or_else(|| {
                crate::GatewayError::Config("per_ip_per_minute must be > 0".into())
            })?);
        let per_hour_quota =
            Quota::per_hour(NonZeroU32::new(cfg.per_ip_per_hour).ok_or_else(|| {
                crate::GatewayError::Config("per_ip_per_hour must be > 0".into())
            })?);
        Ok(Self {
            per_minute: Arc::new(RateLimiter::keyed(per_minute_quota)),
            per_hour: Arc::new(RateLimiter::keyed(per_hour_quota)),
        })
    }

    /// Returns Ok if both limiters admit, Err(RateLimit(_)) if either
    /// denies. Checked in HELO/MAIL FROM so we reject before DATA.
    pub fn check(&self, peer: IpAddr) -> crate::GatewayResult<()> {
        self.per_minute
            .check_key(&peer)
            .map_err(|_| crate::GatewayError::RateLimit(format!("{} exceeded per-minute", peer)))?;
        self.per_hour
            .check_key(&peer)
            .map_err(|_| crate::GatewayError::RateLimit(format!("{} exceeded per-hour", peer)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn limiter(per_min: u32, per_hr: u32) -> PerIpLimiter {
        PerIpLimiter::new(&RateLimitConfig {
            per_ip_per_minute: per_min,
            per_ip_per_hour: per_hr,
        })
        .unwrap()
    }

    #[test]
    fn admits_within_burst() {
        let l = limiter(5, 100);
        let peer: IpAddr = Ipv4Addr::new(198, 51, 100, 1).into();
        for _ in 0..5 {
            assert!(l.check(peer).is_ok());
        }
    }

    #[test]
    fn rejects_beyond_minute_quota() {
        let l = limiter(3, 100);
        let peer: IpAddr = Ipv4Addr::new(198, 51, 100, 2).into();
        assert!(l.check(peer).is_ok());
        assert!(l.check(peer).is_ok());
        assert!(l.check(peer).is_ok());
        let err = l.check(peer).unwrap_err();
        match err {
            crate::GatewayError::RateLimit(_) => (),
            other => panic!("expected RateLimit, got {:?}", other),
        }
    }

    #[test]
    fn isolates_peers() {
        let l = limiter(2, 100);
        let a: IpAddr = Ipv4Addr::new(198, 51, 100, 3).into();
        let b: IpAddr = Ipv4Addr::new(198, 51, 100, 4).into();
        l.check(a).unwrap();
        l.check(a).unwrap();
        // Exhausted for a, but b must still admit.
        assert!(l.check(a).is_err());
        assert!(l.check(b).is_ok());
    }
}
