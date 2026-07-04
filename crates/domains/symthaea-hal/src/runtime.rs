// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime orchestrator: sensor → callback → safety → servo loop.
//!
//! [`HalRuntime`] ties together all HAL components into a single update
//! loop. Each `tick()` reads all sensors, calls the user-provided callback
//! to get a `HumanoidCommand`, filters it through the safety interlock,
//! and applies it to the servos.

use embedded_hal::i2c::I2c;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::error::{HalError, HalResult};
use crate::gpio_estop::EstopPoller;
use crate::interlock::SafetyInterlock;
use crate::sensor::HalSensorAdapter;
use crate::servo::ServoOutput;

use symthaea_humanoid::types::HumanoidCommand;

// ============================================================================
// RUNTIME TELEMETRY
// ============================================================================

/// Snapshot of runtime performance metrics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RuntimeTelemetry {
    /// Total ticks executed.
    pub tick_count: u64,
    /// Number of ticks that exceeded the target period.
    pub deadline_misses: u64,
    /// Mean tick duration in microseconds.
    pub mean_tick_us: f64,
    /// Maximum tick duration in microseconds.
    pub max_tick_us: f64,
    /// Mean sensor read duration in microseconds.
    pub mean_sensor_us: f64,
    /// Mean callback duration in microseconds.
    pub mean_callback_us: f64,
    /// Actual average tick rate in Hz (0.0 if <2 ticks).
    pub actual_hz: f64,
    /// Median (p50) tick duration in microseconds.
    pub p50_tick_us: f64,
    /// 95th percentile tick duration in microseconds.
    pub p95_tick_us: f64,
    /// 99th percentile tick duration in microseconds.
    pub p99_tick_us: f64,
    /// Jitter: standard deviation of tick durations in microseconds.
    pub jitter_us: f64,
}

// ============================================================================
// HEALTH STATUS
// ============================================================================

/// Snapshot of overall system readiness.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    /// All registered sensors report available.
    pub sensors_ok: bool,
    /// Servo output is enabled.
    pub servos_enabled: bool,
    /// Interlock has not tripped.
    pub interlock_ok: bool,
    /// E-stop is currently active.
    pub estop_active: bool,
    /// Tick rate within ±20% of target (true if <2 ticks).
    pub tick_rate_ok: bool,
    /// Number of degraded sensor monitors (from consecutive-None tracking).
    pub degraded_count: usize,
    /// Detailed issue descriptions.
    pub issues: Vec<String>,
}

impl HealthStatus {
    /// Whether the system is ready for operation.
    ///
    /// True when: all sensors OK, servos enabled, interlock OK,
    /// no e-stop, tick rate OK, and no degraded sensors.
    pub fn is_ready(&self) -> bool {
        self.sensors_ok
            && self.servos_enabled
            && self.interlock_ok
            && !self.estop_active
            && self.tick_rate_ok
            && self.degraded_count == 0
    }
}

impl RuntimeTelemetry {
    /// Serialize to compact JSON string.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Internal accumulator for telemetry data (not public).
struct TelemetryAccumulator {
    start_time: Instant,
    total_tick_us: f64,
    max_tick_us: f64,
    total_sensor_us: f64,
    total_callback_us: f64,
    deadline_misses: u64,
    last_log_time: Instant,
    tick_history: VecDeque<f64>,
    history_capacity: usize,
}

impl TelemetryAccumulator {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            total_tick_us: 0.0,
            max_tick_us: 0.0,
            total_sensor_us: 0.0,
            total_callback_us: 0.0,
            deadline_misses: 0,
            last_log_time: now,
            tick_history: VecDeque::with_capacity(1000),
            history_capacity: 1000,
        }
    }

    fn record_tick(&mut self, tick_us: f64, sensor_us: f64, callback_us: f64) {
        self.total_tick_us += tick_us;
        if tick_us > self.max_tick_us {
            self.max_tick_us = tick_us;
        }
        self.total_sensor_us += sensor_us;
        self.total_callback_us += callback_us;
        // Ring buffer
        if self.tick_history.len() >= self.history_capacity {
            self.tick_history.pop_front();
        }
        self.tick_history.push_back(tick_us);
    }

    fn snapshot(&self, tick_count: u64) -> RuntimeTelemetry {
        let n = tick_count as f64;
        let actual_hz = if tick_count >= 2 {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                tick_count as f64 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        };
        // Percentiles and jitter from ring buffer
        let (p50, p95, p99, jitter) = if self.tick_history.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let mut sorted: Vec<f64> = self.tick_history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = sorted.len();
            let p50 = sorted[((len as f64 * 0.50) as usize).min(len - 1)];
            let p95 = sorted[((len as f64 * 0.95) as usize).min(len - 1)];
            let p99 = sorted[((len as f64 * 0.99) as usize).min(len - 1)];
            let mean = sorted.iter().sum::<f64>() / len as f64;
            let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len as f64;
            (p50, p95, p99, variance.sqrt())
        };
        RuntimeTelemetry {
            tick_count,
            deadline_misses: self.deadline_misses,
            mean_tick_us: if n > 0.0 { self.total_tick_us / n } else { 0.0 },
            max_tick_us: self.max_tick_us,
            mean_sensor_us: if n > 0.0 {
                self.total_sensor_us / n
            } else {
                0.0
            },
            mean_callback_us: if n > 0.0 {
                self.total_callback_us / n
            } else {
                0.0
            },
            actual_hz,
            p50_tick_us: p50,
            p95_tick_us: p95,
            p99_tick_us: p99,
            jitter_us: jitter,
        }
    }
}

// ============================================================================
// CURRENT MONITOR
// ============================================================================

/// Binds a sensor to a joint for automatic overcurrent checking.
///
/// When added to [`HalRuntime`], each tick reads the sensor and feeds
/// the current value (at `current_field_index`) into the interlock's
/// `check_current()` for the specified joint.
pub struct CurrentMonitor {
    sensor: Box<dyn HalSensorAdapter>,
    joint_index: usize,
    current_field_index: usize,
    consecutive_nones: u64,
    max_consecutive_nones: u64,
    warned: bool,
}

impl CurrentMonitor {
    /// Create a new current monitor.
    ///
    /// - `sensor`: A sensor whose `read_raw()` returns current data.
    /// - `joint_index`: Which joint to check (0-indexed).
    /// - `current_field_index`: Index within the sensor's `Vec<f32>` that
    ///   holds the current in amps (e.g., 0 for INA219).
    pub fn new(
        sensor: Box<dyn HalSensorAdapter>,
        joint_index: usize,
        current_field_index: usize,
    ) -> Self {
        Self {
            sensor,
            joint_index,
            current_field_index,
            consecutive_nones: 0,
            max_consecutive_nones: 50,
            warned: false,
        }
    }

    /// Set the threshold for consecutive `None` reads before marking degraded.
    pub fn with_max_consecutive_nones(mut self, max: u64) -> Self {
        self.max_consecutive_nones = max;
        self
    }

    /// Number of consecutive `None` reads from this sensor.
    pub fn consecutive_nones(&self) -> u64 {
        self.consecutive_nones
    }

    /// Whether this sensor is considered degraded (consecutive Nones >= threshold).
    pub fn is_degraded(&self) -> bool {
        self.consecutive_nones >= self.max_consecutive_nones
    }
}

// ============================================================================
// ANGLE MONITOR
// ============================================================================

/// Binds a sensor to a joint for automatic angle bounds checking.
///
/// When added to [`HalRuntime`], each tick reads the sensor and feeds
/// the angle value (at `angle_field_index`) into the interlock's
/// `check_angle()` for the specified joint.
///
/// **Note**: The sensor adapter must provide angles in degrees (e.g.,
/// from a complementary filter or AHRS). Raw accelerometer/gyroscope
/// readings (like MPU6050) need conversion before use.
pub struct AngleMonitor {
    sensor: Box<dyn HalSensorAdapter>,
    joint_index: usize,
    angle_field_index: usize,
    consecutive_nones: u64,
    max_consecutive_nones: u64,
    warned: bool,
}

impl AngleMonitor {
    /// Create a new angle monitor.
    ///
    /// - `sensor`: A sensor whose `read_raw()` returns angle data (degrees).
    /// - `joint_index`: Which joint to check (0-indexed).
    /// - `angle_field_index`: Index within the sensor's `Vec<f32>` that
    ///   holds the angle in degrees.
    pub fn new(
        sensor: Box<dyn HalSensorAdapter>,
        joint_index: usize,
        angle_field_index: usize,
    ) -> Self {
        Self {
            sensor,
            joint_index,
            angle_field_index,
            consecutive_nones: 0,
            max_consecutive_nones: 50,
            warned: false,
        }
    }

    /// Set the threshold for consecutive `None` reads before marking degraded.
    pub fn with_max_consecutive_nones(mut self, max: u64) -> Self {
        self.max_consecutive_nones = max;
        self
    }

    /// Number of consecutive `None` reads from this sensor.
    pub fn consecutive_nones(&self) -> u64 {
        self.consecutive_nones
    }

    /// Whether this sensor is considered degraded (consecutive Nones >= threshold).
    pub fn is_degraded(&self) -> bool {
        self.consecutive_nones >= self.max_consecutive_nones
    }
}

// ============================================================================
// HAL RUNTIME
// ============================================================================

/// Runtime orchestrator for the HAL pipeline.
///
/// Owns servo output, safety interlock, and a set of sensors. Each `tick()`
/// reads all sensors, invokes a callback, filters the resulting command
/// through the interlock, and applies it to the servos.
pub struct HalRuntime<I: I2c> {
    servo: ServoOutput<I>,
    interlock: SafetyInterlock,
    sensors: Vec<Box<dyn HalSensorAdapter>>,
    estop_poller: Option<Box<dyn EstopPoller>>,
    current_monitors: Vec<CurrentMonitor>,
    angle_monitors: Vec<AngleMonitor>,
    tick_count: u64,
    tick_hz: f64,
    telemetry: TelemetryAccumulator,
    telemetry_log_interval_secs: Option<f64>,
}

impl<I: I2c> HalRuntime<I> {
    /// Create a builder for configuring a new runtime.
    pub fn builder(servo: ServoOutput<I>, interlock: SafetyInterlock) -> HalRuntimeBuilder<I> {
        HalRuntimeBuilder::new(servo, interlock)
    }

    /// Create a new runtime with the given servo output and safety interlock.
    pub fn new(servo: ServoOutput<I>, interlock: SafetyInterlock) -> Self {
        Self {
            servo,
            interlock,
            sensors: Vec::new(),
            estop_poller: None,
            current_monitors: Vec::new(),
            angle_monitors: Vec::new(),
            tick_count: 0,
            tick_hz: 50.0,
            telemetry: TelemetryAccumulator::new(),
            telemetry_log_interval_secs: None,
        }
    }

    /// Set the target tick rate in Hz (used by `run()`).
    pub fn set_tick_hz(&mut self, hz: f64) {
        self.tick_hz = hz;
    }

    /// Add a sensor to the runtime.
    pub fn add_sensor(&mut self, sensor: Box<dyn HalSensorAdapter>) {
        debug!(name = sensor.name(), "added sensor to runtime");
        self.sensors.push(sensor);
    }

    /// Set an e-stop poller that is checked at the start of each tick.
    pub fn set_estop_poller(&mut self, poller: Box<dyn EstopPoller>) {
        self.estop_poller = Some(poller);
    }

    /// Add a current monitor for automatic overcurrent checking.
    pub fn add_current_monitor(&mut self, monitor: CurrentMonitor) {
        debug!(joint = monitor.joint_index, "added current monitor");
        self.current_monitors.push(monitor);
    }

    /// Add an angle monitor for automatic angle bounds checking.
    pub fn add_angle_monitor(&mut self, monitor: AngleMonitor) {
        debug!(joint = monitor.joint_index, "added angle monitor");
        self.angle_monitors.push(monitor);
    }

    /// Set the telemetry log interval (seconds). When set, runtime periodically
    /// emits telemetry via `tracing::info!` during `run()` / `run_with_shutdown()`.
    pub fn set_telemetry_log_interval(&mut self, secs: f64) {
        self.telemetry_log_interval_secs = Some(secs);
    }

    /// Set the tick history ring buffer capacity (default 1000).
    pub fn set_history_capacity(&mut self, capacity: usize) {
        self.telemetry.history_capacity = capacity;
        while self.telemetry.tick_history.len() > capacity {
            self.telemetry.tick_history.pop_front();
        }
    }

    /// Emit telemetry log if the configured interval has elapsed.
    fn maybe_log_telemetry(&mut self) {
        if let Some(interval) = self.telemetry_log_interval_secs {
            let elapsed = self.telemetry.last_log_time.elapsed().as_secs_f64();
            if elapsed >= interval {
                let t = self.telemetry.snapshot(self.tick_count);
                info!(
                    actual_hz = format!("{:.1}", t.actual_hz),
                    deadline_misses = t.deadline_misses,
                    max_tick_us = format!("{:.0}", t.max_tick_us),
                    mean_tick_us = format!("{:.0}", t.mean_tick_us),
                    tick_count = t.tick_count,
                    "HAL telemetry"
                );
                self.telemetry.last_log_time = Instant::now();
            }
        }
    }

    /// Execute one tick of the pipeline.
    ///
    /// 1. Read all sensors → `Vec<Option<Vec<f32>>>`
    /// 2. Call `callback` with sensor readings → `HumanoidCommand`
    /// 3. Filter through safety interlock
    /// 4. Apply to servos
    ///
    /// The callback receives one `Option<Vec<f32>>` per sensor. `None` means
    /// the sensor had no data this tick. Sensor indexing is stable (matches
    /// `add_sensor()` order).
    pub fn tick<F>(&mut self, callback: F) -> HalResult<()>
    where
        F: FnOnce(&[Option<Vec<f32>>]) -> HumanoidCommand,
    {
        let tick_start = Instant::now();

        // 0a. Poll e-stop (if wired)
        if let Some(ref mut poller) = self.estop_poller
            && poller.poll()
        {
            warn!("e-stop poller triggered");
            return Err(HalError::EStop);
        }

        // 0b. Read current monitors → check overcurrent
        for mon in &mut self.current_monitors {
            if let Some(values) = mon.sensor.read_raw() {
                mon.consecutive_nones = 0;
                mon.warned = false;
                if let Some(&amps) = values.get(mon.current_field_index) {
                    self.interlock.check_current(mon.joint_index, amps)?;
                }
            } else {
                mon.consecutive_nones += 1;
                if mon.consecutive_nones == mon.max_consecutive_nones && !mon.warned {
                    warn!(
                        sensor = mon.sensor.name(),
                        joint = mon.joint_index,
                        consecutive_nones = mon.consecutive_nones,
                        "current sensor degraded — no data for {} consecutive reads",
                        mon.consecutive_nones
                    );
                    mon.warned = true;
                }
            }
        }

        // 0c. Read angle monitors → check angle bounds
        for mon in &mut self.angle_monitors {
            if let Some(values) = mon.sensor.read_raw() {
                mon.consecutive_nones = 0;
                mon.warned = false;
                if let Some(&deg) = values.get(mon.angle_field_index) {
                    self.interlock.check_angle(mon.joint_index, deg)?;
                }
            } else {
                mon.consecutive_nones += 1;
                if mon.consecutive_nones == mon.max_consecutive_nones && !mon.warned {
                    warn!(
                        sensor = mon.sensor.name(),
                        joint = mon.joint_index,
                        consecutive_nones = mon.consecutive_nones,
                        "angle sensor degraded — no data for {} consecutive reads",
                        mon.consecutive_nones
                    );
                    mon.warned = true;
                }
            }
        }

        // 1. Read all sensors
        let sensor_start = Instant::now();
        let readings: Vec<Option<Vec<f32>>> =
            self.sensors.iter_mut().map(|s| s.read_raw()).collect();
        let sensor_us = sensor_start.elapsed().as_secs_f64() * 1e6;

        // 2. Get command from callback
        let cb_start = Instant::now();
        let command = callback(&readings);
        let callback_us = cb_start.elapsed().as_secs_f64() * 1e6;

        // 3. Filter through safety interlock
        let safe_command = self.interlock.filter_command(&command)?;

        // 4. Apply to servos
        self.servo.apply(&safe_command)?;

        self.tick_count += 1;

        let tick_us = tick_start.elapsed().as_secs_f64() * 1e6;
        self.telemetry.record_tick(tick_us, sensor_us, callback_us);

        Ok(())
    }

    /// Run a blocking loop at the configured tick rate.
    ///
    /// Exits when:
    /// - The e-stop is triggered
    /// - The interlock trips
    /// - The callback returns an error via `tick()`
    /// - `max_ticks` is reached (if `Some`)
    ///
    /// Returns the total number of ticks executed.
    pub fn run<F>(&mut self, max_ticks: Option<u64>, mut callback: F) -> HalResult<u64>
    where
        F: FnMut(&[Option<Vec<f32>>]) -> HumanoidCommand,
    {
        let period = Duration::from_secs_f64(1.0 / self.tick_hz);
        info!(hz = self.tick_hz, "HAL runtime starting");

        loop {
            let tick_start = Instant::now();

            // Check max ticks
            if let Some(max) = max_ticks
                && self.tick_count >= max
            {
                debug!(ticks = self.tick_count, "max ticks reached");
                return Ok(self.tick_count);
            }

            // Check e-stop before tick
            if self.interlock.is_estopped() {
                warn!("runtime stopping: e-stop active");
                return Err(HalError::EStop);
            }

            // Execute tick
            match self.tick(&mut callback) {
                Ok(()) => {}
                Err(e) => {
                    warn!(error = %e, ticks = self.tick_count, "runtime stopping on error");
                    return Err(e);
                }
            }

            // Sleep for remaining time in period
            let elapsed = tick_start.elapsed();
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            } else {
                self.telemetry.deadline_misses += 1;
            }

            self.maybe_log_telemetry();
        }
    }

    /// Get the total tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Get a reference to the safety interlock.
    pub fn interlock(&self) -> &SafetyInterlock {
        &self.interlock
    }

    /// Get a mutable reference to the safety interlock.
    pub fn interlock_mut(&mut self) -> &mut SafetyInterlock {
        &mut self.interlock
    }

    /// Get a reference to the servo output.
    pub fn servo(&self) -> &ServoOutput<I> {
        &self.servo
    }

    /// Get a mutable reference to the servo output.
    pub fn servo_mut(&mut self) -> &mut ServoOutput<I> {
        &mut self.servo
    }

    /// Number of registered sensors.
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }

    /// Return sensors that have exceeded their consecutive-None threshold.
    ///
    /// Each entry is `(sensor_name, monitor_type, consecutive_nones)`.
    pub fn degraded_sensors(&self) -> Vec<(&str, &str, u64)> {
        let mut result = Vec::new();
        for mon in &self.current_monitors {
            if mon.is_degraded() {
                result.push((mon.sensor.name(), "current", mon.consecutive_nones));
            }
        }
        for mon in &self.angle_monitors {
            if mon.is_degraded() {
                result.push((mon.sensor.name(), "angle", mon.consecutive_nones));
            }
        }
        result
    }

    /// Get a snapshot of runtime telemetry.
    pub fn telemetry(&self) -> RuntimeTelemetry {
        self.telemetry.snapshot(self.tick_count)
    }

    /// Get a snapshot of overall system health and readiness.
    pub fn health(&self) -> HealthStatus {
        let mut issues = Vec::new();

        // Sensors
        let mut sensors_ok = true;
        for s in &self.sensors {
            if !s.is_available() {
                sensors_ok = false;
                issues.push(format!("sensor '{}' unavailable", s.name()));
            }
        }

        // Servos
        let servos_enabled = self.servo.is_enabled();
        if !servos_enabled {
            issues.push("servos not enabled".to_string());
        }

        // Interlock
        let interlock_ok = !self.interlock.is_tripped();
        if !interlock_ok {
            issues.push(format!(
                "interlock tripped: {}",
                self.interlock.trip_reason().unwrap_or("unknown")
            ));
        }

        // E-stop
        let estop_active = self.interlock.is_estopped();
        if estop_active {
            issues.push("e-stop active".to_string());
        }

        // Tick rate (±20% tolerance, skip if <2 ticks)
        let t = self.telemetry.snapshot(self.tick_count);
        let tick_rate_ok = if self.tick_count < 2 {
            true
        } else {
            let ratio = t.actual_hz / self.tick_hz;
            let ok = (0.8..=1.2).contains(&ratio);
            if !ok {
                issues.push(format!(
                    "tick rate {:.1} Hz outside ±20% of target {:.1} Hz",
                    t.actual_hz, self.tick_hz
                ));
            }
            ok
        };

        // Degraded sensors
        let degraded = self.degraded_sensors();
        let degraded_count = degraded.len();
        for (name, kind, count) in &degraded {
            issues.push(format!(
                "{} sensor '{}' degraded ({} consecutive Nones)",
                kind, name, count
            ));
        }

        HealthStatus {
            sensors_ok,
            servos_enabled,
            interlock_ok,
            estop_active,
            tick_rate_ok,
            degraded_count,
            issues,
        }
    }

    /// Disable all servo outputs immediately.
    ///
    /// Turns off both PCA9685 boards. Safe to call multiple times.
    pub fn emergency_disable(&mut self) -> HalResult<()> {
        warn!("emergency disable — turning off all servos");
        self.servo.disable()
    }

    /// Run a blocking loop that handles SIGINT/SIGTERM gracefully.
    ///
    /// On any exit path (signal, error, e-stop, max_ticks), all servos are
    /// disabled before returning.
    #[cfg(feature = "signal")]
    pub fn run_with_shutdown<F>(
        &mut self,
        max_ticks: Option<u64>,
        mut callback: F,
    ) -> HalResult<u64>
    where
        F: FnMut(&[Option<Vec<f32>>]) -> HumanoidCommand,
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let _ = ctrlc::set_handler(move || {
            shutdown_clone.store(true, Ordering::SeqCst);
        });

        let period = Duration::from_secs_f64(1.0 / self.tick_hz);
        info!(
            hz = self.tick_hz,
            "HAL runtime starting (with signal handler)"
        );

        let result = loop {
            // Check shutdown signal
            if shutdown.load(Ordering::SeqCst) {
                info!("shutdown signal received");
                break Ok(self.tick_count);
            }

            let tick_start = Instant::now();

            // Check max ticks
            if let Some(max) = max_ticks {
                if self.tick_count >= max {
                    debug!(ticks = self.tick_count, "max ticks reached");
                    break Ok(self.tick_count);
                }
            }

            // Check e-stop before tick
            if self.interlock.is_estopped() {
                warn!("runtime stopping: e-stop active");
                break Err(HalError::EStop);
            }

            // Execute tick
            match self.tick(&mut callback) {
                Ok(()) => {}
                Err(e) => {
                    warn!(error = %e, ticks = self.tick_count, "runtime stopping on error");
                    break Err(e);
                }
            }

            // Sleep for remaining time in period
            let elapsed = tick_start.elapsed();
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            } else {
                self.telemetry.deadline_misses += 1;
            }

            self.maybe_log_telemetry();
        };

        // Always disable servos on exit
        let _ = self.servo.disable();
        result
    }
}

// ============================================================================
// RUNTIME BUILDER
// ============================================================================

/// Builder for [`HalRuntime`] with a fluent, consuming API.
///
/// ```rust,ignore
/// let runtime = HalRuntime::builder(servo, interlock)
///     .with_tick_hz(100.0)
///     .with_sensor(Box::new(imu))
///     .with_current_monitor(current_mon)
///     .with_angle_monitor(angle_mon)
///     .with_telemetry_log_interval(5.0)
///     .with_history_capacity(2000)
///     .build();
/// ```
pub struct HalRuntimeBuilder<I: I2c> {
    servo: ServoOutput<I>,
    interlock: SafetyInterlock,
    sensors: Vec<Box<dyn HalSensorAdapter>>,
    estop_poller: Option<Box<dyn EstopPoller>>,
    current_monitors: Vec<CurrentMonitor>,
    angle_monitors: Vec<AngleMonitor>,
    tick_hz: f64,
    telemetry_log_interval_secs: Option<f64>,
    history_capacity: usize,
}

impl<I: I2c> HalRuntimeBuilder<I> {
    fn new(servo: ServoOutput<I>, interlock: SafetyInterlock) -> Self {
        Self {
            servo,
            interlock,
            sensors: Vec::new(),
            estop_poller: None,
            current_monitors: Vec::new(),
            angle_monitors: Vec::new(),
            tick_hz: 50.0,
            telemetry_log_interval_secs: None,
            history_capacity: 1000,
        }
    }

    /// Set the target tick rate in Hz (default 50.0).
    pub fn with_tick_hz(mut self, hz: f64) -> Self {
        self.tick_hz = hz;
        self
    }

    /// Add a sensor.
    pub fn with_sensor(mut self, sensor: Box<dyn HalSensorAdapter>) -> Self {
        self.sensors.push(sensor);
        self
    }

    /// Set an e-stop poller.
    pub fn with_estop_poller(mut self, poller: Box<dyn EstopPoller>) -> Self {
        self.estop_poller = Some(poller);
        self
    }

    /// Add a current monitor.
    pub fn with_current_monitor(mut self, monitor: CurrentMonitor) -> Self {
        self.current_monitors.push(monitor);
        self
    }

    /// Add an angle monitor.
    pub fn with_angle_monitor(mut self, monitor: AngleMonitor) -> Self {
        self.angle_monitors.push(monitor);
        self
    }

    /// Set telemetry log interval in seconds.
    pub fn with_telemetry_log_interval(mut self, secs: f64) -> Self {
        self.telemetry_log_interval_secs = Some(secs);
        self
    }

    /// Set the tick history ring buffer capacity (default 1000).
    pub fn with_history_capacity(mut self, capacity: usize) -> Self {
        self.history_capacity = capacity;
        self
    }

    /// Build the [`HalRuntime`].
    pub fn build(self) -> HalRuntime<I> {
        let mut telemetry = TelemetryAccumulator::new();
        telemetry.history_capacity = self.history_capacity;
        telemetry.tick_history = VecDeque::with_capacity(self.history_capacity);
        HalRuntime {
            servo: self.servo,
            interlock: self.interlock,
            sensors: self.sensors,
            estop_poller: self.estop_poller,
            current_monitors: self.current_monitors,
            angle_monitors: self.angle_monitors,
            tick_count: 0,
            tick_hz: self.tick_hz,
            telemetry,
            telemetry_log_interval_secs: self.telemetry_log_interval_secs,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::CalibrationProfile;
    use crate::mock::{MockEstopPoller, MockHalSensor, MockI2cBus};

    fn make_runtime() -> HalRuntime<MockI2cBus> {
        let bus0 = MockI2cBus::new();
        let bus1 = MockI2cBus::new();
        let cal = CalibrationProfile::default_21();
        let mut servo = ServoOutput::new(bus0, bus1, cal);
        servo.init(50.0).unwrap();
        servo.enable();
        let interlock = SafetyInterlock::new();
        HalRuntime::new(servo, interlock)
    }

    #[test]
    fn test_runtime_tick_no_sensors() {
        let mut rt = make_runtime();
        rt.tick(|readings| {
            assert!(readings.is_empty());
            HumanoidCommand::zero()
        })
        .unwrap();
        assert_eq!(rt.tick_count(), 1);
    }

    #[test]
    fn test_runtime_tick_with_sensor() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("imu", vec![vec![1.0, 2.0, 3.0]]);
        rt.add_sensor(Box::new(sensor));
        assert_eq!(rt.sensor_count(), 1);

        rt.tick(|readings| {
            assert_eq!(readings.len(), 1);
            assert_eq!(readings[0], Some(vec![1.0, 2.0, 3.0]));
            HumanoidCommand::zero()
        })
        .unwrap();
    }

    #[test]
    fn test_runtime_tick_sensor_exhausted() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("imu", vec![vec![1.0]]);
        rt.add_sensor(Box::new(sensor));

        // First tick: data available
        rt.tick(|r| {
            assert!(r[0].is_some());
            HumanoidCommand::zero()
        })
        .unwrap();

        // Second tick: sensor exhausted → None
        rt.tick(|r| {
            assert!(r[0].is_none());
            HumanoidCommand::zero()
        })
        .unwrap();
    }

    #[test]
    fn test_runtime_run_max_ticks() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0); // fast

        let count = rt
            .run(Some(5), |_readings| HumanoidCommand::zero())
            .unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_runtime_estop_stops_run() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.interlock_mut().trigger_estop();

        let result = rt.run(Some(100), |_| HumanoidCommand::zero());
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_accessors() {
        let rt = make_runtime();
        assert!(rt.servo().is_enabled());
        assert!(!rt.interlock().is_estopped());
        assert_eq!(rt.sensor_count(), 0);
        assert_eq!(rt.tick_count(), 0);
    }

    #[test]
    fn test_runtime_multiple_sensors() {
        let mut rt = make_runtime();
        rt.add_sensor(Box::new(MockHalSensor::new("imu", vec![vec![1.0]])));
        rt.add_sensor(Box::new(MockHalSensor::new("current", vec![vec![2.0]])));
        assert_eq!(rt.sensor_count(), 2);

        rt.tick(|readings| {
            assert_eq!(readings.len(), 2);
            assert_eq!(readings[0], Some(vec![1.0]));
            assert_eq!(readings[1], Some(vec![2.0]));
            HumanoidCommand::zero()
        })
        .unwrap();
    }

    // ── E-stop poller tests ──────────────────────────────────────────────

    #[test]
    fn test_estop_poller_not_triggered() {
        let mut rt = make_runtime();
        rt.set_estop_poller(Box::new(MockEstopPoller::new(false)));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
        assert_eq!(rt.tick_count(), 1);
    }

    #[test]
    fn test_estop_poller_triggered() {
        let mut rt = make_runtime();
        rt.set_estop_poller(Box::new(MockEstopPoller::new(true)));
        let result = rt.tick(|_| HumanoidCommand::zero());
        assert!(matches!(result, Err(HalError::EStop)));
    }

    #[test]
    fn test_estop_poller_stops_run() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.set_estop_poller(Box::new(MockEstopPoller::new(true)));
        let result = rt.run(Some(100), |_| HumanoidCommand::zero());
        assert!(result.is_err());
        // Should stop on first tick (estop checked before tick body in tick(),
        // or before tick in run's estop check)
        assert!(rt.tick_count() < 100);
    }

    // ── Current monitor tests ──────────────────────────────────────────

    #[test]
    fn test_current_monitor_normal() {
        let mut rt = make_runtime();
        // 1.0A is below the 2.0A default limit
        let sensor = MockHalSensor::new("ina219", vec![vec![1.0, 5.0]; 3]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(sensor), 0, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_current_monitor_overcurrent() {
        let mut rt = make_runtime();
        // 3.0A exceeds the 2.0A default limit
        let sensor = MockHalSensor::new("ina219", vec![vec![3.0, 5.0]]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(sensor), 0, 0));
        let result = rt.tick(|_| HumanoidCommand::zero());
        assert!(matches!(result, Err(HalError::Overcurrent { .. })));
    }

    #[test]
    fn test_current_monitor_no_data() {
        let mut rt = make_runtime();
        // Empty sensor → read_raw returns None → no error
        let sensor = MockHalSensor::new("ina219", vec![]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(sensor), 0, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_multiple_current_monitors() {
        let mut rt = make_runtime();
        // Joint 0: 1.0A (ok), Joint 1: 1.5A (ok)
        let s0 = MockHalSensor::new("ina0", vec![vec![1.0]; 3]);
        let s1 = MockHalSensor::new("ina1", vec![vec![1.5]; 3]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(s0), 0, 0));
        rt.add_current_monitor(CurrentMonitor::new(Box::new(s1), 1, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_backward_compat_no_poller_no_monitors() {
        // Original test: no poller, no monitors → works as before
        let mut rt = make_runtime();
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
        assert_eq!(rt.tick_count(), 1);
    }

    // ── Telemetry tests ────────────────────────────────────────────────

    #[test]
    fn test_telemetry_initial_zeros() {
        let rt = make_runtime();
        let t = rt.telemetry();
        assert_eq!(t.tick_count, 0);
        assert_eq!(t.deadline_misses, 0);
        assert!((t.mean_tick_us - 0.0).abs() < f64::EPSILON);
        assert!((t.max_tick_us - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_telemetry_after_ticks() {
        let mut rt = make_runtime();
        for _ in 0..5 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        assert_eq!(t.tick_count, 5);
        assert!(t.mean_tick_us > 0.0, "mean_tick_us should be positive");
        assert!(t.max_tick_us > 0.0, "max_tick_us should be positive");
    }

    #[test]
    fn test_telemetry_deadline_misses() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1_000_000.0); // 1 MHz — every tick will miss
        let _ = rt.run(Some(3), |_| HumanoidCommand::zero());
        let t = rt.telemetry();
        // With 1 MHz target, real ticks will miss deadlines
        assert!(t.deadline_misses > 0, "expected deadline misses at 1MHz");
    }

    #[test]
    fn test_telemetry_actual_hz() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        let _ = rt.run(Some(10), |_| HumanoidCommand::zero());
        let t = rt.telemetry();
        assert!(t.actual_hz > 0.0, "actual_hz should be positive after run");
    }

    #[test]
    fn test_telemetry_sensor_time() {
        let mut rt = make_runtime();
        rt.add_sensor(Box::new(MockHalSensor::new("imu", vec![vec![1.0]; 5])));
        for _ in 0..3 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        // Sensor time should be non-negative (mock is fast, but > 0 is not guaranteed)
        assert!(t.mean_sensor_us >= 0.0);
    }

    #[test]
    fn test_telemetry_clone_debug() {
        let rt = make_runtime();
        let t = rt.telemetry();
        let t2 = t.clone();
        assert_eq!(t2.tick_count, t.tick_count);
        let _debug = format!("{:?}", t);
    }

    // ── Shutdown tests ─────────────────────────────────────────────────

    #[test]
    fn test_emergency_disable() {
        let mut rt = make_runtime();
        assert!(rt.servo().is_enabled());
        rt.emergency_disable().unwrap();
        assert!(!rt.servo().is_enabled());
    }

    #[cfg(feature = "signal")]
    #[test]
    fn test_run_with_shutdown_max_ticks() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        let count = rt
            .run_with_shutdown(Some(5), |_| HumanoidCommand::zero())
            .unwrap();
        assert_eq!(count, 5);
        // Servos should be disabled after return
        assert!(!rt.servo().is_enabled());
    }

    #[cfg(feature = "signal")]
    #[test]
    fn test_run_with_shutdown_estop() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.set_estop_poller(Box::new(MockEstopPoller::new(true)));
        let result = rt.run_with_shutdown(Some(100), |_| HumanoidCommand::zero());
        assert!(result.is_err());
        // Servos should be disabled even on error
        assert!(!rt.servo().is_enabled());
    }

    // ── Telemetry logging tests ───────────────────────────────────────

    #[test]
    fn test_telemetry_log_default_none() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        let count = rt.run(Some(5), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_telemetry_log_interval_no_panic() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.set_telemetry_log_interval(0.001);
        let count = rt.run(Some(10), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_telemetry_log_does_not_affect_tick_count() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.set_telemetry_log_interval(0.001);
        let count = rt.run(Some(10), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 10);
        assert_eq!(rt.tick_count(), 10);
    }

    #[test]
    fn test_telemetry_log_with_sensors() {
        let mut rt = make_runtime();
        rt.set_tick_hz(1000.0);
        rt.set_telemetry_log_interval(0.001);
        rt.add_sensor(Box::new(MockHalSensor::new("imu", vec![vec![1.0]; 10])));
        let count = rt.run(Some(10), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 10);
    }

    // ── Angle monitor tests ──────────────────────────────────────────

    #[test]
    fn test_angle_monitor_normal() {
        let mut rt = make_runtime();
        // 45° is within the ±90° default limit
        let sensor = MockHalSensor::new("angle", vec![vec![45.0]; 3]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_angle_monitor_out_of_bounds() {
        let mut rt = make_runtime();
        // 95° exceeds the ±90° default limit
        let sensor = MockHalSensor::new("angle", vec![vec![95.0]]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 0));
        let result = rt.tick(|_| HumanoidCommand::zero());
        assert!(matches!(result, Err(HalError::AngleBounds { .. })));
    }

    #[test]
    fn test_angle_monitor_negative_out_of_bounds() {
        let mut rt = make_runtime();
        // -95° exceeds the ±90° default limit
        let sensor = MockHalSensor::new("angle", vec![vec![-95.0]]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 0));
        let result = rt.tick(|_| HumanoidCommand::zero());
        assert!(matches!(result, Err(HalError::AngleBounds { .. })));
    }

    #[test]
    fn test_angle_monitor_no_data() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("angle", vec![]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_angle_monitor_field_index() {
        let mut rt = make_runtime();
        // angle_field_index=2 picks the third element (45°)
        let sensor = MockHalSensor::new("angle", vec![vec![0.0, 0.0, 45.0]; 3]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 2));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_angle_and_current_monitors_together() {
        let mut rt = make_runtime();
        // Current: 1.0A (ok), Angle: 45° (ok)
        let current_sensor = MockHalSensor::new("ina219", vec![vec![1.0]; 3]);
        let angle_sensor = MockHalSensor::new("angle", vec![vec![45.0]; 3]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(current_sensor), 0, 0));
        rt.add_angle_monitor(AngleMonitor::new(Box::new(angle_sensor), 0, 0));
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    // ── Ring buffer / percentile tests ────────────────────────────────

    #[test]
    fn test_percentiles_initial_zeros() {
        let rt = make_runtime();
        let t = rt.telemetry();
        assert!((t.p50_tick_us - 0.0).abs() < f64::EPSILON);
        assert!((t.p95_tick_us - 0.0).abs() < f64::EPSILON);
        assert!((t.p99_tick_us - 0.0).abs() < f64::EPSILON);
        assert!((t.jitter_us - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentiles_after_ticks() {
        let mut rt = make_runtime();
        for _ in 0..20 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        assert!(t.p50_tick_us > 0.0, "p50 should be positive after ticks");
        assert!(t.p95_tick_us > 0.0, "p95 should be positive after ticks");
        assert!(t.p99_tick_us > 0.0, "p99 should be positive after ticks");
    }

    #[test]
    fn test_percentile_ordering() {
        let mut rt = make_runtime();
        for _ in 0..20 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        assert!(t.p50_tick_us <= t.p95_tick_us, "p50 <= p95");
        assert!(t.p95_tick_us <= t.p99_tick_us, "p95 <= p99");
    }

    #[test]
    fn test_jitter_non_negative() {
        let mut rt = make_runtime();
        for _ in 0..20 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        assert!(t.jitter_us >= 0.0, "jitter should be non-negative");
    }

    #[test]
    fn test_custom_history_capacity() {
        let mut rt = make_runtime();
        rt.set_history_capacity(10);
        // Run 20 ticks, ring buffer should only hold last 10
        for _ in 0..20 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        assert_eq!(t.tick_count, 20);
        // Percentiles should still work with smaller buffer
        assert!(t.p50_tick_us > 0.0);
    }

    #[test]
    fn test_telemetry_clone_with_percentiles() {
        let mut rt = make_runtime();
        for _ in 0..5 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        let t2 = t.clone();
        assert_eq!(t.tick_count, t2.tick_count);
        assert!((t.p50_tick_us - t2.p50_tick_us).abs() < f64::EPSILON);
        assert!((t.jitter_us - t2.jitter_us).abs() < f64::EPSILON);
    }

    // ── Builder tests ─────────────────────────────────────────────────

    fn make_servo() -> ServoOutput<MockI2cBus> {
        let bus0 = MockI2cBus::new();
        let bus1 = MockI2cBus::new();
        let cal = CalibrationProfile::default_21();
        let mut servo = ServoOutput::new(bus0, bus1, cal);
        servo.init(50.0).unwrap();
        servo.enable();
        servo
    }

    #[test]
    fn test_builder_minimal() {
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new()).build();
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
        assert_eq!(rt.tick_count(), 1);
    }

    #[test]
    fn test_builder_with_tick_hz() {
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_tick_hz(100.0)
            .build();
        let count = rt.run(Some(5), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_builder_with_sensor() {
        let sensor = MockHalSensor::new("imu", vec![vec![1.0]; 3]);
        let rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_sensor(Box::new(sensor))
            .build();
        assert_eq!(rt.sensor_count(), 1);
    }

    #[test]
    fn test_builder_with_estop_poller() {
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_tick_hz(1000.0)
            .with_estop_poller(Box::new(MockEstopPoller::new(true)))
            .build();
        let result = rt.run(Some(100), |_| HumanoidCommand::zero());
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_current_monitor() {
        let sensor = MockHalSensor::new("ina219", vec![vec![1.0]; 3]);
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_current_monitor(CurrentMonitor::new(Box::new(sensor), 0, 0))
            .build();
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_builder_with_angle_monitor() {
        let sensor = MockHalSensor::new("angle", vec![vec![45.0]; 3]);
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_angle_monitor(AngleMonitor::new(Box::new(sensor), 0, 0))
            .build();
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
    }

    #[test]
    fn test_builder_full_chain() {
        let imu = MockHalSensor::new("imu", vec![vec![1.0]; 10]);
        let current_sensor = MockHalSensor::new("ina219", vec![vec![0.5]; 10]);
        let angle_sensor = MockHalSensor::new("angle", vec![vec![30.0]; 10]);
        let mut rt = HalRuntime::builder(make_servo(), SafetyInterlock::new())
            .with_tick_hz(1000.0)
            .with_sensor(Box::new(imu))
            .with_estop_poller(Box::new(MockEstopPoller::new(false)))
            .with_current_monitor(CurrentMonitor::new(Box::new(current_sensor), 0, 0))
            .with_angle_monitor(AngleMonitor::new(Box::new(angle_sensor), 0, 0))
            .with_telemetry_log_interval(0.001)
            .with_history_capacity(500)
            .build();
        let count = rt.run(Some(5), |_| HumanoidCommand::zero()).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_builder_defaults() {
        let rt = HalRuntime::builder(make_servo(), SafetyInterlock::new()).build();
        assert_eq!(rt.sensor_count(), 0);
        assert_eq!(rt.tick_count(), 0);
        // Default telemetry should show zeros
        let t = rt.telemetry();
        assert_eq!(t.tick_count, 0);
        assert!((t.p50_tick_us - 0.0).abs() < f64::EPSILON);
    }

    // ── Telemetry serde tests ────────────────────────────────────────

    #[test]
    fn test_telemetry_json_roundtrip() {
        let mut rt = make_runtime();
        for _ in 0..5 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let t = rt.telemetry();
        let json = t.to_json().unwrap();
        let t2: RuntimeTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(t.tick_count, t2.tick_count);
        assert_eq!(t.deadline_misses, t2.deadline_misses);
        assert!((t.mean_tick_us - t2.mean_tick_us).abs() < 1e-6);
    }

    #[test]
    fn test_telemetry_to_json_contains_fields() {
        let t = RuntimeTelemetry::default();
        let json = t.to_json().unwrap();
        assert!(json.contains("\"tick_count\""));
        assert!(json.contains("\"deadline_misses\""));
        assert!(json.contains("\"actual_hz\""));
        assert!(json.contains("\"p50_tick_us\""));
    }

    #[test]
    fn test_telemetry_to_json_pretty_has_newlines() {
        let t = RuntimeTelemetry::default();
        let pretty = t.to_json_pretty().unwrap();
        assert!(pretty.contains('\n'));
        assert!(pretty.lines().count() > 1);
    }

    // ── Sensor degradation tests ─────────────────────────────────────

    #[test]
    fn test_current_monitor_degradation() {
        let mut rt = make_runtime();
        // Empty sensor → always returns None
        let sensor = MockHalSensor::new("ina219", vec![]);
        let mon = CurrentMonitor::new(Box::new(sensor), 0, 0).with_max_consecutive_nones(3);
        rt.add_current_monitor(mon);

        // Tick 4 times — threshold is 3, so degraded after 3 Nones
        for _ in 0..4 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let degraded = rt.degraded_sensors();
        assert_eq!(degraded.len(), 1);
        assert_eq!(degraded[0].0, "ina219");
        assert_eq!(degraded[0].1, "current");
        assert!(degraded[0].2 >= 3);
    }

    #[test]
    fn test_angle_monitor_degradation() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("angle_imu", vec![]);
        let mon = AngleMonitor::new(Box::new(sensor), 0, 0).with_max_consecutive_nones(3);
        rt.add_angle_monitor(mon);

        for _ in 0..4 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        let degraded = rt.degraded_sensors();
        assert_eq!(degraded.len(), 1);
        assert_eq!(degraded[0].0, "angle_imu");
        assert_eq!(degraded[0].1, "angle");
    }

    #[test]
    fn test_degradation_counter_resets_on_data() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("ina219", vec![vec![1.0]]);
        let mon = CurrentMonitor::new(Box::new(sensor), 0, 0).with_max_consecutive_nones(5);
        rt.add_current_monitor(mon);

        // First tick: data arrives → counter should stay 0
        rt.tick(|_| HumanoidCommand::zero()).unwrap();
        assert!(rt.degraded_sensors().is_empty());

        // Next ticks: sensor exhausted → Nones, but threshold is 5
        for _ in 0..4 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        // Only 4 nones, threshold is 5 → not degraded yet
        assert!(rt.degraded_sensors().is_empty());
    }

    #[test]
    fn test_no_degraded_sensors_when_healthy() {
        let mut rt = make_runtime();
        let sensor = MockHalSensor::new("ina219", vec![vec![1.0]; 10]);
        rt.add_current_monitor(CurrentMonitor::new(Box::new(sensor), 0, 0));
        let angle_sensor = MockHalSensor::new("angle", vec![vec![45.0]; 10]);
        rt.add_angle_monitor(AngleMonitor::new(Box::new(angle_sensor), 0, 0));

        for _ in 0..5 {
            rt.tick(|_| HumanoidCommand::zero()).unwrap();
        }
        assert!(rt.degraded_sensors().is_empty());
    }

    #[test]
    fn test_degradation_default_threshold() {
        let sensor = MockHalSensor::new("ina219", vec![]);
        let mon = CurrentMonitor::new(Box::new(sensor), 0, 0);
        assert_eq!(mon.max_consecutive_nones, 50);
        assert!(!mon.is_degraded());
        assert_eq!(mon.consecutive_nones(), 0);
    }

    // ── Health check tests ──────────────────────────────────────────

    #[test]
    fn test_health_fully_ready() {
        let rt = make_runtime();
        let h = rt.health();
        assert!(h.is_ready(), "issues: {:?}", h.issues);
        assert!(h.sensors_ok);
        assert!(h.servos_enabled);
        assert!(h.interlock_ok);
        assert!(!h.estop_active);
        assert!(h.tick_rate_ok);
        assert_eq!(h.degraded_count, 0);
        assert!(h.issues.is_empty());
    }

    #[test]
    fn test_health_servos_disabled() {
        let bus0 = MockI2cBus::new();
        let bus1 = MockI2cBus::new();
        let cal = CalibrationProfile::default_21();
        let servo = ServoOutput::new(bus0, bus1, cal);
        // NOT enabled
        let rt = HalRuntime::new(servo, SafetyInterlock::new());
        let h = rt.health();
        assert!(!h.is_ready());
        assert!(!h.servos_enabled);
        assert!(h.issues.iter().any(|i| i.contains("servos")));
    }

    #[test]
    fn test_health_interlock_tripped() {
        let mut rt = make_runtime();
        rt.interlock_mut().trigger_estop();
        let _ = rt.interlock_mut().filter_command(&HumanoidCommand::zero());
        rt.interlock().release_estop();
        // Interlock is still tripped even after releasing estop
        let h = rt.health();
        assert!(!h.is_ready());
        assert!(!h.interlock_ok);
    }

    #[test]
    fn test_health_estop_active() {
        let mut rt = make_runtime();
        rt.interlock_mut().trigger_estop();
        let h = rt.health();
        assert!(!h.is_ready());
        assert!(h.estop_active);
        assert!(h.issues.iter().any(|i| i.contains("e-stop")));
    }

    #[test]
    fn test_health_unavailable_sensor() {
        let mut rt = make_runtime();
        // MockHalSensor with empty readings → is_available() returns false
        let mut s = MockHalSensor::new("dead_imu", vec![]);
        let _ = s.read_raw(); // now is_available() returns false
        rt.add_sensor(Box::new(s));
        let h = rt.health();
        assert!(!h.sensors_ok);
        assert!(h.issues.iter().any(|i| i.contains("dead_imu")));
    }

    #[test]
    fn test_health_status_serializable() {
        let rt = make_runtime();
        let h = rt.health();
        let json = serde_json::to_string(&h).unwrap();
        let h2: HealthStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(h.sensors_ok, h2.sensors_ok);
        assert_eq!(h.servos_enabled, h2.servos_enabled);
        assert_eq!(h.degraded_count, h2.degraded_count);
    }
}
