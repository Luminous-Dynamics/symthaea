// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The SIR compartment model (Kermack–McKendrick, 1927).
//!
//! Populations are tracked as fractions `S + I + R = 1`:
//! - `dS/dt = −β·S·I`
//! - `dI/dt =  β·S·I − γ·I`
//! - `dR/dt =  γ·I`
//!
//! where β is the transmission rate and γ the recovery rate.

/// SIR model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sir {
    /// Transmission rate β.
    pub beta: f64,
    /// Recovery rate γ.
    pub gamma: f64,
}

/// A population state as fractions (should sum to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub s: f64,
    pub i: f64,
    pub r: f64,
}

impl Sir {
    /// Basic reproduction number `R₀ = β/γ` — expected secondary infections
    /// from one case in a fully susceptible population.
    pub fn basic_reproduction_number(&self) -> f64 {
        self.beta / self.gamma
    }

    /// Herd-immunity threshold `1 − 1/R₀` — the immune fraction above which an
    /// epidemic cannot grow. Zero when `R₀ ≤ 1`.
    pub fn herd_immunity_threshold(&self) -> f64 {
        let r0 = self.basic_reproduction_number();
        if r0 <= 1.0 { 0.0 } else { 1.0 - 1.0 / r0 }
    }

    /// One explicit-Euler step of size `dt`. Conserves `S+I+R` exactly
    /// (the increments sum to zero by construction).
    pub fn step(&self, state: State, dt: f64) -> State {
        let ds = -self.beta * state.s * state.i;
        let di = self.beta * state.s * state.i - self.gamma * state.i;
        let dr = self.gamma * state.i;
        State {
            s: state.s + ds * dt,
            i: state.i + di * dt,
            r: state.r + dr * dt,
        }
    }

    /// Simulate from an initial state for `steps` steps of `dt`, returning the
    /// final state and the peak infected fraction reached.
    pub fn simulate(&self, initial: State, dt: f64, steps: usize) -> (State, f64) {
        let mut state = initial;
        let mut peak = initial.i;
        for _ in 0..steps {
            state = self.step(state, dt);
            if state.i > peak {
                peak = state.i;
            }
        }
        (state, peak)
    }

    /// Final epidemic size `R∞`: the total fraction infected over the whole
    /// outbreak (starting from ≈all-susceptible), solving the implicit
    /// `R∞ = 1 − e^(−R₀·R∞)` by fixed-point iteration. Zero when `R₀ ≤ 1`.
    pub fn final_size(&self) -> f64 {
        let r0 = self.basic_reproduction_number();
        if r0 <= 1.0 {
            return 0.0;
        }
        let mut z = 0.5;
        for _ in 0..1000 {
            let next = 1.0 - (-r0 * z).exp();
            if (next - z).abs() < 1e-12 {
                return next;
            }
            z = next;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r0_and_herd_immunity() {
        let m = Sir {
            beta: 0.3,
            gamma: 0.1,
        };
        assert!((m.basic_reproduction_number() - 3.0).abs() < 1e-12);
        assert!((m.herd_immunity_threshold() - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn subcritical_epidemic_does_not_grow() {
        let m = Sir {
            beta: 0.05,
            gamma: 0.1,
        }; // R0 = 0.5
        assert_eq!(m.herd_immunity_threshold(), 0.0);
        assert_eq!(m.final_size(), 0.0);
        let (_, peak) = m.simulate(
            State {
                s: 0.99,
                i: 0.01,
                r: 0.0,
            },
            0.1,
            2000,
        );
        assert!(
            peak <= 0.01 + 1e-9,
            "infection should not grow, peak={peak}"
        );
    }

    #[test]
    fn population_is_conserved() {
        let m = Sir {
            beta: 0.4,
            gamma: 0.1,
        };
        let (end, _) = m.simulate(
            State {
                s: 0.99,
                i: 0.01,
                r: 0.0,
            },
            0.05,
            5000,
        );
        assert!((end.s + end.i + end.r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn final_size_matches_transcendental_solution() {
        // R0 = 3 → R∞ ≈ 0.9403.
        let m = Sir {
            beta: 0.3,
            gamma: 0.1,
        };
        assert!(
            (m.final_size() - 0.940448).abs() < 1e-4,
            "R∞={}",
            m.final_size()
        );
    }

    #[test]
    fn supercritical_epidemic_grows_then_ends() {
        let m = Sir {
            beta: 0.4,
            gamma: 0.1,
        }; // R0 = 4
        let (end, peak) = m.simulate(
            State {
                s: 0.999,
                i: 0.001,
                r: 0.0,
            },
            0.05,
            8000,
        );
        assert!(
            peak > 0.1,
            "epidemic should peak substantially, peak={peak}"
        );
        assert!(end.i < 0.01, "epidemic should burn out, end.i={}", end.i);
        // Most of the population was infected (final size large).
        assert!(end.r > 0.5);
    }
}
