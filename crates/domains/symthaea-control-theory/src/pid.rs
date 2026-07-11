// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Discrete PID controller.

/// A PID controller carrying integral and previous-error state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pid {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    initialized: bool,
}

impl Pid {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Pid {
        Pid {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            initialized: false,
        }
    }

    /// One control step: `u = Kp·e + Ki·∫e + Kd·de/dt`.
    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        self.integral += error * dt;
        let derivative = if self.initialized && dt > 0.0 {
            (error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = error;
        self.initialized = true;
        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_scales_error() {
        let mut pid = Pid::new(2.0, 0.0, 0.0);
        assert!((pid.update(3.0, 0.1) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn integral_accumulates_constant_error() {
        let mut pid = Pid::new(0.0, 1.0, 0.0);
        pid.update(1.0, 0.5); // integral = 0.5 → out 0.5
        let out = pid.update(1.0, 0.5); // integral = 1.0 → out 1.0
        assert!((out - 1.0).abs() < 1e-12);
    }

    #[test]
    fn closed_loop_drives_plant_to_setpoint() {
        // Integrator plant x' = u; PID should drive x → setpoint (1.0).
        let mut pid = Pid::new(2.0, 1.0, 0.1);
        let (setpoint, dt) = (1.0, 0.01);
        let mut x = 0.0;
        for _ in 0..5000 {
            let u = pid.update(setpoint - x, dt);
            x += u * dt;
        }
        assert!((x - setpoint).abs() < 1e-2, "x={x}");
    }
}
