// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Backend-neutral contracts for measured hardware energy counters.
//!
//! This module has no NVML, sysfs, RAPL, CUDA, or platform dependency.
//! Concrete backends implement EnergyMeter and report actual capabilities.
//! An unsupported counter remains unsupported: callers must not synthesize a
//! measured energy value from TDP, clock rate, FLOP count, or a theoretical
//! Landauer reference.
//!
//! Counter readings preserve native integer units, device and sensor scope,
//! counter epoch (changed on driver/device reset), and monotonic timestamp.
//! EnergyDelta rejects incompatible/stale readings and only handles rollover
//! when the backend provides an explicit modulus.
//!
//! This layer measures only the declared sensor scope. A GPU counter is not
//! whole-system energy; CPU package, GPU, wall-meter and other domains remain
//! distinct unless their coverage is known to be disjoint and complete.

/// Backend family that supplied a cumulative energy counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyMeterBackend {
    /// NVIDIA Management Library cumulative device-energy counter.
    Nvml,
    /// Linux powercap energy counter (including RAPL zones).
    LinuxPowercap,
    /// Calibrated external energy meter.
    External,
    /// Another backend with an explicit stable identifier.
    Custom(String),
}

/// Physical or measurement domain covered by a sensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyScope {
    /// One GPU device; the UUID is stored in the descriptor.
    GpuDevice,
    /// One CPU package/socket.
    CpuPackage { package_id: u32 },
    /// One CPU core.
    CpuCore { cpu_id: u32 },
    /// A complete system wall-power meter.
    SystemWall,
    /// Another named measurement domain.
    Custom(String),
}

/// Unit used by the backend raw cumulative counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyCounterUnit {
    /// Joules.
    Joules,
    /// Millijoules.
    Millijoules,
    /// Microjoules.
    Microjoules,
}

impl EnergyCounterUnit {
    fn microjoules_per_unit(self) -> u128 {
        match self {
            Self::Joules => 1_000_000,
            Self::Millijoules => 1_000,
            Self::Microjoules => 1,
        }
    }
}

/// Stable identity and unit semantics for one hardware energy sensor.
///
/// wrap_modulus is the exclusive counter modulus in native units. For a
/// counter wrapping at 2^32, the modulus is 2^32, not 2^32 - 1. If a backend
/// cannot establish the wrap behavior, use None; a decreasing counter is
/// rejected rather than guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyMeterDescriptor {
    /// Backend family.
    pub backend: EnergyMeterBackend,
    /// Stable identifier for this particular sensor/counter.
    pub sensor_id: String,
    /// Device identity (GPU UUID, package ID string, meter serial, etc.).
    pub device_id: String,
    /// Physical scope covered by this counter.
    pub scope: EnergyScope,
    /// Unit of the raw integer counter.
    pub unit: EnergyCounterUnit,
    /// Exclusive rollover modulus, when documented by the backend.
    pub wrap_modulus: Option<u64>,
}

/// Runtime capability state for a sensor.
///
/// These states prevent unsupported sensors from being confused with zero
/// energy consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyMeterAvailability {
    /// The counter can currently be read.
    Available,
    /// The device/backend does not implement this counter.
    Unsupported { reason: String },
    /// The counter exists but the current process lacks permission.
    PermissionDenied { reason: String },
    /// The device or driver is unavailable.
    Unavailable { reason: String },
}

/// One raw cumulative counter reading.
///
/// monotonic_timestamp_ns must use a monotonic clock, not wall time.
/// counter_epoch must change whenever the counter may have reset, for example
/// after a driver reload or device reinitialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnergyCounterReading {
    /// Descriptor of the exact sensor that supplied this reading.
    pub descriptor: EnergyMeterDescriptor,
    /// Backend-defined counter generation; changes across resets/reloads.
    pub counter_epoch: u64,
    /// Raw cumulative value in the descriptor's native unit.
    pub raw_counter: u64,
    /// Monotonic observation time in nanoseconds.
    pub monotonic_timestamp_ns: u128,
}

/// Typed failures from a concrete meter backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyMeterError {
    /// Sensor is unsupported on this device/driver.
    Unsupported { reason: String },
    /// Sensor exists but the process cannot access it.
    PermissionDenied { reason: String },
    /// Device was lost or the driver is unavailable.
    Unavailable { reason: String },
    /// Backend returned an unexpected failure.
    BackendFailure { reason: String },
    /// Backend could not produce a monotonic timestamp.
    ClockUnavailable,
}

/// Backend-neutral interface for a cumulative hardware energy counter.
///
/// Implementations must not invent readings when the sensor is unsupported.
/// The descriptor and availability are discoverable before measurement;
/// readings preserve the exact descriptor and counter epoch used.
pub trait EnergyMeter: Send {
    /// Describe the sensor, device identity, measurement scope and raw unit.
    fn descriptor(&self) -> &EnergyMeterDescriptor;

    /// Report whether the counter is currently usable.
    fn availability(&self) -> EnergyMeterAvailability;

    /// Read one cumulative energy-counter sample.
    fn read_counter(&mut self) -> Result<EnergyCounterReading, EnergyMeterError>;
}

/// Errors produced while comparing two cumulative counter readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyDeltaError {
    /// Readings do not come from the same sensor/device/scope/unit.
    DescriptorMismatch,
    /// Counter reset or driver generation changed between readings.
    CounterEpochChanged,
    /// Later reading was not observed after the earlier reading.
    TimestampDidNotAdvance,
    /// Counter decreased and no rollover modulus was declared.
    CounterDecreasedWithoutModulus,
    /// Raw counter value is outside the declared rollover range.
    CounterOutsideModulus,
    /// Difference or unit conversion overflowed the supported integer range.
    ArithmeticOverflow,
}

/// Measured energy over one interval, converted exactly to microjoules.
///
/// The conversion uses integer arithmetic. This type does not assert that the
/// meter covers a full device or system; retain the descriptor scope in any
/// measurement receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyDelta {
    descriptor: EnergyMeterDescriptor,
    energy_microjoules: u128,
    elapsed_ns: u128,
}

impl EnergyDelta {
    /// Compute measured energy between two compatible cumulative readings.
    ///
    /// A counter decrease is accepted only when both readings share a
    /// descriptor with a declared exclusive rollover modulus. A driver reset
    /// must change counter_epoch and is never treated as rollover.
    pub fn between(
        earlier: &EnergyCounterReading,
        later: &EnergyCounterReading,
    ) -> Result<Self, EnergyDeltaError> {
        if earlier.descriptor != later.descriptor {
            return Err(EnergyDeltaError::DescriptorMismatch);
        }
        if earlier.counter_epoch != later.counter_epoch {
            return Err(EnergyDeltaError::CounterEpochChanged);
        }
        if later.monotonic_timestamp_ns <= earlier.monotonic_timestamp_ns {
            return Err(EnergyDeltaError::TimestampDidNotAdvance);
        }

        if let Some(modulus) = earlier.descriptor.wrap_modulus {
            if modulus == 0
                || earlier.raw_counter >= modulus
                || later.raw_counter >= modulus
            {
                return Err(EnergyDeltaError::CounterOutsideModulus);
            }
        }

        let raw_delta = match earlier.raw_counter.cmp(&later.raw_counter) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                u128::from(later.raw_counter - earlier.raw_counter)
            }
            std::cmp::Ordering::Greater => {
                let modulus = u128::from(
                    earlier
                        .descriptor
                        .wrap_modulus
                        .ok_or(EnergyDeltaError::CounterDecreasedWithoutModulus)?,
                );
                let earlier_value = u128::from(earlier.raw_counter);
                let later_value = u128::from(later.raw_counter);
                modulus
                    .checked_sub(earlier_value)
                    .and_then(|remaining| remaining.checked_add(later_value))
                    .ok_or(EnergyDeltaError::ArithmeticOverflow)?
            }
        };

        let energy_microjoules = raw_delta
            .checked_mul(earlier.descriptor.unit.microjoules_per_unit())
            .ok_or(EnergyDeltaError::ArithmeticOverflow)?;
        let elapsed_ns = later.monotonic_timestamp_ns - earlier.monotonic_timestamp_ns;

        Ok(Self {
            descriptor: earlier.descriptor.clone(),
            energy_microjoules,
            elapsed_ns,
        })
    }

    /// Stable sensor descriptor shared by the two readings.
    pub fn descriptor(&self) -> &EnergyMeterDescriptor {
        &self.descriptor
    }

    /// Measured energy difference in microjoules.
    pub fn energy_microjoules(&self) -> u128 {
        self.energy_microjoules
    }

    /// Elapsed monotonic time in nanoseconds.
    pub fn elapsed_ns(&self) -> u128 {
        self.elapsed_ns
    }

    /// Average power over the interval, in watts.
    ///
    /// One microjoule per nanosecond equals one kilowatt. This is a derived
    /// interval average, not an instantaneous power measurement.
    pub fn average_power_watts(&self) -> f64 {
        self.energy_microjoules as f64 * 1_000.0 / self.elapsed_ns as f64
    }

    /// Measured energy in joules.
    pub fn energy_joules(&self) -> f64 {
        self.energy_microjoules as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(unit: EnergyCounterUnit, wrap_modulus: Option<u64>) -> EnergyMeterDescriptor {
        EnergyMeterDescriptor {
            backend: EnergyMeterBackend::LinuxPowercap,
            sensor_id: "intel-rapl:0/energy".into(),
            device_id: "package-0".into(),
            scope: EnergyScope::CpuPackage { package_id: 0 },
            unit,
            wrap_modulus,
        }
    }

    fn reading(
        descriptor: EnergyMeterDescriptor,
        counter_epoch: u64,
        raw_counter: u64,
        timestamp_ns: u128,
    ) -> EnergyCounterReading {
        EnergyCounterReading {
            descriptor,
            counter_epoch,
            raw_counter,
            monotonic_timestamp_ns: timestamp_ns,
        }
    }

    #[test]
    fn microjoule_counter_delta_is_exact() {
        let d = descriptor(EnergyCounterUnit::Microjoules, None);
        let start = reading(d.clone(), 1, 50_000, 1_000);
        let end = reading(d, 1, 51_750, 2_000_000_000);

        let delta = EnergyDelta::between(&start, &end).unwrap();
        assert_eq!(delta.energy_microjoules(), 1_750);
        assert_eq!(delta.elapsed_ns(), 1_999_999_000);
        assert!((delta.energy_joules() - 0.00175).abs() < 1e-15);
    }

    #[test]
    fn joule_counter_is_converted_to_microjoules() {
        let d = descriptor(EnergyCounterUnit::Joules, None);
        let start = reading(d.clone(), 1, 2, 1);
        let end = reading(d, 1, 5, 2);

        let delta = EnergyDelta::between(&start, &end).unwrap();
        assert_eq!(delta.energy_microjoules(), 3_000_000);
        assert!((delta.energy_joules() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn average_power_uses_energy_and_monotonic_interval() {
        let d = descriptor(EnergyCounterUnit::Microjoules, None);
        let start = reading(d.clone(), 1, 0, 1_000_000_000);
        let end = reading(d, 1, 1_000_000, 2_000_000_000);

        let delta = EnergyDelta::between(&start, &end).unwrap();
        assert!((delta.average_power_watts() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn millijoule_counter_is_converted_to_microjoules() {
        let d = descriptor(EnergyCounterUnit::Millijoules, None);
        let start = reading(d.clone(), 1, 100, 1);
        let end = reading(d, 1, 103, 1_000_000_001);

        let delta = EnergyDelta::between(&start, &end).unwrap();
        assert_eq!(delta.energy_microjoules(), 3_000);
        assert!((delta.energy_joules() - 0.003).abs() < 1e-15);
    }

    #[test]
    fn documented_counter_wraparound_is_accounted_for() {
        let d = descriptor(EnergyCounterUnit::Microjoules, Some(1_000));
        let start = reading(d.clone(), 4, 950, 10);
        let end = reading(d, 4, 25, 20);

        let delta = EnergyDelta::between(&start, &end).unwrap();
        assert_eq!(delta.energy_microjoules(), 75);
    }

    #[test]
    fn counter_decrease_without_modulus_is_rejected() {
        let d = descriptor(EnergyCounterUnit::Microjoules, None);
        let start = reading(d.clone(), 4, 950, 10);
        let end = reading(d, 4, 25, 20);

        assert_eq!(
            EnergyDelta::between(&start, &end),
            Err(EnergyDeltaError::CounterDecreasedWithoutModulus)
        );
    }

    #[test]
    fn driver_reset_and_scope_mismatch_are_rejected() {
        let d = descriptor(EnergyCounterUnit::Microjoules, Some(1_000));
        let start = reading(d.clone(), 4, 950, 10);
        let reset = reading(d.clone(), 5, 25, 20);
        assert_eq!(
            EnergyDelta::between(&start, &reset),
            Err(EnergyDeltaError::CounterEpochChanged)
        );

        let other = EnergyMeterDescriptor {
            device_id: "package-1".into(),
            scope: EnergyScope::CpuPackage { package_id: 1 },
            ..d.clone()
        };
        let other_device = reading(other, 4, 975, 20);
        assert_eq!(
            EnergyDelta::between(&start, &other_device),
            Err(EnergyDeltaError::DescriptorMismatch)
        );
    }

    #[test]
    fn non_advancing_timestamp_and_invalid_wrap_are_rejected() {
        let d = descriptor(EnergyCounterUnit::Microjoules, Some(0));
        let start = reading(d.clone(), 1, 0, 100);
        let same_time = reading(d, 1, 1, 100);
        assert_eq!(
            EnergyDelta::between(&start, &same_time),
            Err(EnergyDeltaError::TimestampDidNotAdvance)
        );

        let invalid = descriptor(EnergyCounterUnit::Microjoules, Some(500));
        let start = reading(invalid.clone(), 1, 500, 10);
        let end = reading(invalid, 1, 2, 20);
        assert_eq!(
            EnergyDelta::between(&start, &end),
            Err(EnergyDeltaError::CounterOutsideModulus)
        );
    }
}
