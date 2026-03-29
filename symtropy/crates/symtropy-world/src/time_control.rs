// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time controls for the civilization simulation.

/// Controls the speed and state of the simulation clock.
#[derive(Debug, Clone)]
pub struct TimeControl {
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Speed multiplier: 1.0 = real-time (1 tick/5s), 10.0 = 10x, etc.
    pub speed: f64,
    /// Current simulation tick (each tick = 1 month in-game).
    pub tick: u64,
    /// Accumulated time since last tick (seconds).
    pub accumulator: f64,
    /// Base interval between ticks at 1x speed (seconds).
    pub base_interval: f64,
}

impl TimeControl {
    /// Default: unpaused, 1x speed, 5-second tick interval.
    pub fn new() -> Self {
        Self {
            paused: false,
            speed: 1.0,
            tick: 0,
            accumulator: 0.0,
            base_interval: 5.0,
        }
    }

    /// Advance the clock by `dt` seconds. Returns the number of ticks to process.
    pub fn advance(&mut self, dt: f64) -> u64 {
        if self.paused {
            return 0;
        }

        self.accumulator += dt * self.speed;
        let ticks = (self.accumulator / self.base_interval) as u64;
        self.accumulator -= ticks as f64 * self.base_interval;
        self.tick += ticks;
        ticks
    }

    /// Fraction of progress toward the next tick [0.0, 1.0].
    /// Used for interpolating between snapshots.
    pub fn interpolation_factor(&self) -> f64 {
        if self.paused || self.base_interval <= 0.0 {
            return 0.0;
        }
        (self.accumulator / self.base_interval).clamp(0.0, 1.0)
    }

    /// Set speed (clamped to [0.0, 1000.0]).
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.clamp(0.0, 1000.0);
    }

    /// Toggle pause.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// In-game date: (year, month) from tick 0 = Year 1, Month 1.
    pub fn game_date(&self) -> (u64, u64) {
        let year = self.tick / 12 + 1;
        let month = self.tick % 12 + 1;
        (year, month)
    }
}

impl Default for TimeControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_produces_ticks() {
        let mut tc = TimeControl::new();
        // 5 seconds at 1x speed = 1 tick
        let ticks = tc.advance(5.0);
        assert_eq!(ticks, 1);
        assert_eq!(tc.tick, 1);
    }

    #[test]
    fn advance_10x_speed() {
        let mut tc = TimeControl::new();
        tc.set_speed(10.0);
        // 5 seconds at 10x = 50 effective seconds = 10 ticks
        let ticks = tc.advance(5.0);
        assert_eq!(ticks, 10);
    }

    #[test]
    fn paused_no_ticks() {
        let mut tc = TimeControl::new();
        tc.paused = true;
        let ticks = tc.advance(100.0);
        assert_eq!(ticks, 0);
        assert_eq!(tc.tick, 0);
    }

    #[test]
    fn accumulator_carries_remainder() {
        let mut tc = TimeControl::new();
        let ticks = tc.advance(7.0); // 5s interval → 1 tick, 2s remainder
        assert_eq!(ticks, 1);
        assert!((tc.accumulator - 2.0).abs() < 1e-10);

        let ticks = tc.advance(4.0); // 2 + 4 = 6s → 1 tick, 1s remainder
        assert_eq!(ticks, 1);
        assert!((tc.accumulator - 1.0).abs() < 1e-10);
    }

    #[test]
    fn interpolation_factor() {
        let mut tc = TimeControl::new();
        tc.advance(2.5); // Half a tick interval
        assert!((tc.interpolation_factor() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn game_date() {
        let mut tc = TimeControl::new();
        assert_eq!(tc.game_date(), (1, 1)); // Tick 0 = Year 1, Month 1

        tc.tick = 11;
        assert_eq!(tc.game_date(), (1, 12)); // Last month of year 1

        tc.tick = 12;
        assert_eq!(tc.game_date(), (2, 1)); // First month of year 2

        tc.tick = 24;
        assert_eq!(tc.game_date(), (3, 1));
    }

    #[test]
    fn toggle_pause() {
        let mut tc = TimeControl::new();
        assert!(!tc.paused);
        tc.toggle_pause();
        assert!(tc.paused);
        tc.toggle_pause();
        assert!(!tc.paused);
    }

    #[test]
    fn speed_clamped() {
        let mut tc = TimeControl::new();
        tc.set_speed(-5.0);
        assert_eq!(tc.speed, 0.0);
        tc.set_speed(5000.0);
        assert_eq!(tc.speed, 1000.0);
    }
}
