// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! M/M/1 queue: a single server, Poisson arrivals, exponential service.

/// An M/M/1 queue parameterised by arrival rate λ and service rate μ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MM1 {
    pub arrival_rate: f64,
    pub service_rate: f64,
}

impl MM1 {
    /// Server utilization `ρ = λ/μ`.
    pub fn utilization(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// The queue is stable (finite) iff `λ < μ`.
    pub fn is_stable(&self) -> bool {
        self.arrival_rate < self.service_rate
    }

    /// Mean number in the system `L = ρ/(1−ρ)`.
    pub fn avg_in_system(&self) -> f64 {
        let r = self.utilization();
        r / (1.0 - r)
    }

    /// Mean number waiting in the queue `Lq = ρ²/(1−ρ)`.
    pub fn avg_in_queue(&self) -> f64 {
        let r = self.utilization();
        r * r / (1.0 - r)
    }

    /// Mean time in system `W = 1/(μ−λ)`.
    pub fn avg_time_in_system(&self) -> f64 {
        1.0 / (self.service_rate - self.arrival_rate)
    }

    /// Mean waiting time in queue `Wq = Lq/λ` (Little's law).
    pub fn avg_time_in_queue(&self) -> f64 {
        self.avg_in_queue() / self.arrival_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mm1_known_metrics() {
        // λ=2, μ=3 → ρ=2/3, L=2, W=1, Lq=4/3, Wq=2/3.
        let q = MM1 {
            arrival_rate: 2.0,
            service_rate: 3.0,
        };
        assert!(q.is_stable());
        assert!((q.utilization() - 2.0 / 3.0).abs() < 1e-12);
        assert!((q.avg_in_system() - 2.0).abs() < 1e-12);
        assert!((q.avg_time_in_system() - 1.0).abs() < 1e-12);
        assert!((q.avg_in_queue() - 4.0 / 3.0).abs() < 1e-12);
        assert!((q.avg_time_in_queue() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn overloaded_queue_is_unstable() {
        assert!(
            !MM1 {
                arrival_rate: 5.0,
                service_rate: 3.0
            }
            .is_stable()
        );
    }

    #[test]
    fn littles_law_holds() {
        // L = λ·W.
        let q = MM1 {
            arrival_rate: 1.5,
            service_rate: 4.0,
        };
        assert!((q.avg_in_system() - q.arrival_rate * q.avg_time_in_system()).abs() < 1e-12);
    }
}
