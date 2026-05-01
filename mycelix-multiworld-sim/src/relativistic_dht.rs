// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Relativistic DHT: Lorentz transformation layer for Holochain at interstellar velocities.
//!
//! At significant fractions of light speed, time dilation causes the ship's
//! clocks to drift from Earth's. When the ship beams its gossip payload
//! back to Sol, the timestamps are out of phase. This module computes the
//! correction needed to reconcile causality.
//!
//! # Physics
//!
//! Lorentz factor: γ = 1 / √(1 - v²/c²)
//!
//! At 0.05c: γ = 1.00125 (time dilation: 0.125% — 1.1 hours per year)
//! At 0.10c: γ = 1.00504 (time dilation: 0.504% — 4.4 hours per year)
//! At 0.50c: γ = 1.15470 (time dilation: 15.5% — 56 days per year)
//!
//! For generation ships at 0.05c, the effect is tiny but cumulative:
//! over 85 years, the ship's clocks are 4.0 days behind Earth's.
//!
//! # Holochain Implications
//!
//! Holochain source chains use chronological timestamps. If ship time
//! drifts from Earth time, entries created "simultaneously" on ship and
//! Earth will have different timestamps. The reconciliation protocol
//! must account for this when merging divergent histories.

use serde::{Deserialize, Serialize};

/// Compute the Lorentz factor γ for a given velocity (as fraction of c).
///
/// γ = 1 / √(1 - v²/c²)
///
/// # Arguments
/// - `velocity_c`: Velocity as fraction of speed of light (0.0 to < 1.0)
pub fn lorentz_gamma(velocity_c: f64) -> f64 {
    if velocity_c <= 0.0 {
        return 1.0;
    }
    if velocity_c >= 1.0 {
        return f64::INFINITY;
    }
    1.0 / (1.0 - velocity_c * velocity_c).sqrt()
}

/// Compute proper time elapsed on the ship given coordinate time on Earth.
///
/// τ_ship = t_earth / γ
///
/// # Arguments
/// - `earth_time_years`: Time elapsed on Earth (years)
/// - `velocity_c`: Ship velocity as fraction of c
pub fn proper_time(earth_time_years: f64, velocity_c: f64) -> f64 {
    earth_time_years / lorentz_gamma(velocity_c)
}

/// Compute the time drift between ship and Earth clocks.
///
/// Returns the number of days the ship's clock is BEHIND Earth's.
///
/// # Arguments
/// - `earth_elapsed_years`: Earth time since launch
/// - `velocity_c`: Ship velocity as fraction of c
pub fn clock_drift_days(earth_elapsed_years: f64, velocity_c: f64) -> f64 {
    let gamma = lorentz_gamma(velocity_c);
    let ship_years = earth_elapsed_years / gamma;
    (earth_elapsed_years - ship_years) * 365.25
}

/// Relativistic timestamp reconciliation for Holochain DHT gossip.
///
/// When the ship sends its source chain entries to Earth, the timestamps
/// need adjustment to account for time dilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativisticReconciliation {
    /// Velocity of the ship (fraction of c).
    pub velocity_c: f64,
    /// Lorentz gamma factor.
    pub gamma: f64,
    /// Cumulative clock drift (days behind Earth).
    pub clock_drift_days: f64,
    /// Number of gossip payloads sent.
    pub gossip_payloads_sent: u64,
    /// Timestamp offset to apply when merging (seconds).
    pub timestamp_offset_seconds: f64,
}

impl RelativisticReconciliation {
    pub fn new(velocity_c: f64) -> Self {
        let gamma = lorentz_gamma(velocity_c);
        Self {
            velocity_c,
            gamma,
            clock_drift_days: 0.0,
            gossip_payloads_sent: 0,
            timestamp_offset_seconds: 0.0,
        }
    }

    /// Update the reconciliation state for a new tick.
    ///
    /// # Arguments
    /// - `earth_elapsed_years`: Total Earth time since ship launch
    pub fn tick(&mut self, earth_elapsed_years: f64) {
        self.clock_drift_days = clock_drift_days(earth_elapsed_years, self.velocity_c);
        self.timestamp_offset_seconds = self.clock_drift_days * 86400.0;
    }

    /// Transform a ship-local timestamp to Earth-equivalent timestamp.
    ///
    /// When the ship sends a source chain entry with timestamp T_ship,
    /// Earth should interpret it as T_earth = T_ship + offset.
    pub fn ship_to_earth_timestamp(&self, ship_timestamp_seconds: f64) -> f64 {
        ship_timestamp_seconds + self.timestamp_offset_seconds
    }

    /// Transform an Earth timestamp to ship-local timestamp.
    pub fn earth_to_ship_timestamp(&self, earth_timestamp_seconds: f64) -> f64 {
        earth_timestamp_seconds - self.timestamp_offset_seconds
    }

    /// Send a gossip payload (increment counter).
    pub fn send_gossip(&mut self) {
        self.gossip_payloads_sent += 1;
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "γ={:.6} | Drift: {:.2} days | Offset: {:.0}s | Gossip sent: {}",
            self.gamma,
            self.clock_drift_days,
            self.timestamp_offset_seconds,
            self.gossip_payloads_sent,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_at_rest() {
        assert!((lorentz_gamma(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gamma_at_005c() {
        let g = lorentz_gamma(0.05);
        // At 0.05c: γ ≈ 1.00125
        assert!((g - 1.00125).abs() < 0.001, "γ at 0.05c: {}", g);
    }

    #[test]
    fn gamma_at_010c() {
        let g = lorentz_gamma(0.10);
        // At 0.10c: γ ≈ 1.00504
        assert!((g - 1.00504).abs() < 0.001, "γ at 0.10c: {}", g);
    }

    #[test]
    fn gamma_at_050c() {
        let g = lorentz_gamma(0.50);
        // At 0.50c: γ ≈ 1.15470
        assert!((g - 1.15470).abs() < 0.001, "γ at 0.50c: {}", g);
    }

    #[test]
    fn drift_at_005c_85_years() {
        let drift = clock_drift_days(85.0, 0.05);
        // At 0.05c, γ=1.00125. Over 85yr: drift = 85 × (1 - 1/γ) × 365.25 ≈ 38.8 days
        assert!(
            drift > 35.0 && drift < 42.0,
            "85yr drift at 0.05c: {:.1} days",
            drift
        );
    }

    #[test]
    fn proper_time_less_than_earth() {
        let ship = proper_time(85.0, 0.05);
        assert!(ship < 85.0, "Ship time should be less than Earth: {}", ship);
        assert!(ship > 84.0, "Not that much less at 0.05c: {}", ship);
    }

    #[test]
    fn timestamp_roundtrip() {
        let recon = RelativisticReconciliation::new(0.05);
        let mut recon = recon;
        recon.tick(10.0); // 10 years into transit

        let ship_ts = 1000000.0; // Arbitrary ship timestamp
        let earth_ts = recon.ship_to_earth_timestamp(ship_ts);
        let back = recon.earth_to_ship_timestamp(earth_ts);
        assert!(
            (back - ship_ts).abs() < 0.01,
            "Roundtrip should be identity: {} → {} → {}",
            ship_ts,
            earth_ts,
            back
        );
    }

    #[test]
    fn reconciliation_tracks_gossip() {
        let mut recon = RelativisticReconciliation::new(0.05);
        assert_eq!(recon.gossip_payloads_sent, 0);
        recon.send_gossip();
        recon.send_gossip();
        assert_eq!(recon.gossip_payloads_sent, 2);
    }

    #[test]
    fn summary_produces_string() {
        let recon = RelativisticReconciliation::new(0.05);
        let s = recon.summary();
        assert!(s.contains("γ="));
        assert!(s.contains("Drift:"));
    }
}
