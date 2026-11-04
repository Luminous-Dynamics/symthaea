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
//! counter epoch, and monotonic timestamp. Backends must assign a new epoch
//! whenever they detect a reset or reinitialization; some hardware interfaces
//! cannot distinguish an otherwise invisible reset from a documented wrap.
//! EnergyDelta rejects incompatible/stale readings and only handles rollover
//! when the backend provides an explicit modulus.
//!
//! This layer measures only the declared sensor scope. A GPU counter is not
//! whole-system energy; CPU package, GPU, wall-meter and other domains remain
//! distinct unless their coverage is known to be disjoint and complete.

use serde::{Deserialize, Serialize};

/// Backend family that supplied a cumulative energy counter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
/// counter_epoch must change whenever the backend detects a reset or
/// reinitialization. Some hardware counters do not expose enough information
/// to distinguish every silent reset from a documented rollover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        if earlier.counter_epoch != later.counter_epoch {
            return Err(EnergyDeltaError::CounterEpochChanged);
        }
        if earlier.descriptor != later.descriptor {
            return Err(EnergyDeltaError::DescriptorMismatch);
        }
        if later.monotonic_timestamp_ns <= earlier.monotonic_timestamp_ns {
            return Err(EnergyDeltaError::TimestampDidNotAdvance);
        }

        if let Some(modulus) = earlier.descriptor.wrap_modulus {
            if modulus == 0 || earlier.raw_counter >= modulus || later.raw_counter >= modulus {
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

/// Linux powercap/RAPL adapter backed by the kernel's energy counter files.
///
/// The caller supplies paths and physical sensor scope explicitly. This
/// avoids assuming a particular Intel/AMD hierarchy or treating a package
/// counter as whole-system energy. The adapter has no external dependency.
///
/// Each opened instance has a distinct epoch, and a changed wrap range starts
/// another epoch. A driver reset that leaves both sysfs files and the wrap
/// range unchanged may be indistinguishable from a genuine wrap; these two
/// counter files alone cannot prove otherwise.
#[cfg(target_os = "linux")]
pub mod linux_powercap {
    use super::*;
    use std::fs;
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    static NEXT_METER_EPOCH: AtomicU64 = AtomicU64::new(1);

    /// Linux powercap energy counter reader.
    pub struct LinuxPowercapMeter {
        descriptor: EnergyMeterDescriptor,
        energy_path: PathBuf,
        max_range_path: Option<PathBuf>,
        counter_epoch: u64,
        started_at: Instant,
    }

    impl LinuxPowercapMeter {
        /// Open a powercap zone using its energy_uj path and optional
        /// max_energy_range_uj path.
        ///
        /// sensor_id, device_id and scope are supplied by the caller so the
        /// adapter never guesses whether a zone is a package, core or system.
        pub fn open(
            sensor_id: impl Into<String>,
            device_id: impl Into<String>,
            scope: EnergyScope,
            energy_path: impl Into<PathBuf>,
            max_range_path: Option<PathBuf>,
        ) -> Result<Self, EnergyMeterError> {
            let energy_path = energy_path.into();
            let max_range = match max_range_path.as_ref() {
                Some(path) => {
                    let range = read_counter_file(path)?;
                    if range == 0 {
                        return Err(EnergyMeterError::Unsupported {
                            reason: "powercap counter reports a zero rollover range".into(),
                        });
                    }
                    Some(range)
                }
                None => None,
            };
            // Probe the counter at construction; permission and path failures
            // are surfaced now rather than converted into a zero reading.
            let initial_energy = read_counter_file(&energy_path)?;
            if max_range.is_some_and(|range| initial_energy >= range) {
                return Err(EnergyMeterError::BackendFailure {
                    reason: "initial energy counter is outside its declared range".into(),
                });
            }

            Ok(Self {
                descriptor: EnergyMeterDescriptor {
                    backend: EnergyMeterBackend::LinuxPowercap,
                    sensor_id: sensor_id.into(),
                    device_id: device_id.into(),
                    scope,
                    unit: EnergyCounterUnit::Microjoules,
                    wrap_modulus: max_range,
                },
                energy_path,
                max_range_path,
                counter_epoch: NEXT_METER_EPOCH.fetch_add(1, Ordering::Relaxed),
                started_at: Instant::now(),
            })
        }

        fn read_energy(&self) -> Result<u64, EnergyMeterError> {
            read_counter_file(&self.energy_path)
        }

        fn refresh_wrap_modulus(&mut self) -> Result<(), EnergyMeterError> {
            let Some(path) = self.max_range_path.as_ref() else {
                return Ok(());
            };
            let range = read_counter_file(path)?;
            if range == 0 {
                return Err(EnergyMeterError::Unsupported {
                    reason: "powercap counter reports a zero rollover range".into(),
                });
            }
            let current = Some(range);
            if current != self.descriptor.wrap_modulus {
                self.counter_epoch = self.counter_epoch.checked_add(1).ok_or_else(|| {
                    EnergyMeterError::BackendFailure {
                        reason: "counter epoch overflow".into(),
                    }
                })?;
                self.descriptor.wrap_modulus = current;
            }
            Ok(())
        }
    }

    impl EnergyMeter for LinuxPowercapMeter {
        fn descriptor(&self) -> &EnergyMeterDescriptor {
            &self.descriptor
        }

        fn availability(&self) -> EnergyMeterAvailability {
            let energy = match self.read_energy() {
                Ok(value) => value,
                Err(error) => return availability_from_error(error),
            };

            let Some(path) = self.max_range_path.as_ref() else {
                return EnergyMeterAvailability::Available;
            };
            let range = match read_counter_file(path) {
                Ok(value) => value,
                Err(error) => return availability_from_error(error),
            };

            if range == 0 {
                return EnergyMeterAvailability::Unsupported {
                    reason: "powercap counter reports a zero rollover range".into(),
                };
            }
            if energy >= range {
                return EnergyMeterAvailability::Unavailable {
                    reason: "energy counter is outside its declared rollover range".into(),
                };
            }

            EnergyMeterAvailability::Available
        }

        fn read_counter(&mut self) -> Result<EnergyCounterReading, EnergyMeterError> {
            self.refresh_wrap_modulus()?;
            let raw_counter = self.read_energy()?;
            if self
                .descriptor
                .wrap_modulus
                .is_some_and(|range| raw_counter >= range)
            {
                return Err(EnergyMeterError::BackendFailure {
                    reason: "energy counter is outside its declared range".into(),
                });
            }
            Ok(EnergyCounterReading {
                descriptor: self.descriptor.clone(),
                counter_epoch: self.counter_epoch,
                raw_counter,
                monotonic_timestamp_ns: self.started_at.elapsed().as_nanos(),
            })
        }
    }

    fn availability_from_error(error: EnergyMeterError) -> EnergyMeterAvailability {
        match error {
            EnergyMeterError::Unsupported { reason } => {
                EnergyMeterAvailability::Unsupported { reason }
            }
            EnergyMeterError::PermissionDenied { reason } => {
                EnergyMeterAvailability::PermissionDenied { reason }
            }
            EnergyMeterError::Unavailable { reason }
            | EnergyMeterError::BackendFailure { reason } => {
                EnergyMeterAvailability::Unavailable { reason }
            }
            EnergyMeterError::ClockUnavailable => EnergyMeterAvailability::Unavailable {
                reason: "monotonic clock unavailable".into(),
            },
        }
    }

    fn read_counter_file(path: &Path) -> Result<u64, EnergyMeterError> {
        let contents = fs::read_to_string(path).map_err(|error| match error.kind() {
            ErrorKind::PermissionDenied => EnergyMeterError::PermissionDenied {
                reason: format!("cannot read {}: {error}", path.display()),
            },
            ErrorKind::NotFound => EnergyMeterError::Unavailable {
                reason: format!("counter file not found: {}", path.display()),
            },
            _ => EnergyMeterError::BackendFailure {
                reason: format!("cannot read {}: {error}", path.display()),
            },
        })?;
        contents
            .trim()
            .parse::<u64>()
            .map_err(|error| EnergyMeterError::BackendFailure {
                reason: format!("invalid counter value in {}: {error}", path.display()),
            })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::{SystemTime, UNIX_EPOCH};

        struct TestZone {
            directory: PathBuf,
            energy_path: PathBuf,
            range_path: PathBuf,
        }

        impl TestZone {
            fn new(energy_uj: u64, max_range_uj: u64) -> Self {
                let nonce = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let directory = std::env::temp_dir()
                    .join(format!("symthaea-powercap-{}-{nonce}", std::process::id()));
                fs::create_dir(&directory).unwrap();
                let energy_path = directory.join("energy_uj");
                let range_path = directory.join("max_energy_range_uj");
                fs::write(&energy_path, energy_uj.to_string()).unwrap();
                fs::write(&range_path, max_range_uj.to_string()).unwrap();
                Self {
                    directory,
                    energy_path,
                    range_path,
                }
            }
        }

        impl Drop for TestZone {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.directory);
            }
        }

        #[test]
        fn reads_microjoule_counter_with_explicit_scope() {
            let zone = TestZone::new(10_000, 1_000_000);
            let mut meter = LinuxPowercapMeter::open(
                "intel-rapl:0",
                "package-0",
                EnergyScope::CpuPackage { package_id: 0 },
                &zone.energy_path,
                Some(zone.range_path.clone()),
            )
            .unwrap();

            assert_eq!(meter.availability(), EnergyMeterAvailability::Available);
            assert_eq!(meter.descriptor().unit, EnergyCounterUnit::Microjoules);
            assert_eq!(
                meter.descriptor().scope,
                EnergyScope::CpuPackage { package_id: 0 }
            );
            assert_eq!(meter.read_counter().unwrap().raw_counter, 10_000);
        }

        #[test]
        fn changed_wrap_range_starts_a_new_counter_epoch() {
            let zone = TestZone::new(10_000, 1_000_000);
            let mut meter = LinuxPowercapMeter::open(
                "intel-rapl:0",
                "package-0",
                EnergyScope::CpuPackage { package_id: 0 },
                &zone.energy_path,
                Some(zone.range_path.clone()),
            )
            .unwrap();
            let first = meter.read_counter().unwrap();
            fs::write(&zone.range_path, "2000000").unwrap();
            let second = meter.read_counter().unwrap();

            assert_ne!(first.counter_epoch, second.counter_epoch);
            assert_ne!(
                first.descriptor.wrap_modulus,
                second.descriptor.wrap_modulus
            );
            assert_eq!(
                EnergyDelta::between(&first, &second),
                Err(EnergyDeltaError::CounterEpochChanged)
            );
        }

        #[test]
        fn availability_rejects_zero_range_and_counter_outside_range() {
            let zone = TestZone::new(10_000, 1_000_000);
            let meter = LinuxPowercapMeter::open(
                "intel-rapl:0",
                "package-0",
                EnergyScope::CpuPackage { package_id: 0 },
                &zone.energy_path,
                Some(zone.range_path.clone()),
            )
            .unwrap();

            fs::write(&zone.range_path, "0").unwrap();
            assert!(matches!(
                meter.availability(),
                EnergyMeterAvailability::Unsupported { .. }
            ));

            fs::write(&zone.range_path, "5000").unwrap();
            assert!(matches!(
                meter.availability(),
                EnergyMeterAvailability::Unavailable { .. }
            ));
        }

        #[test]
        fn malformed_counter_file_is_not_reported_as_zero_energy() {
            let zone = TestZone::new(10_000, 1_000_000);
            let mut meter = LinuxPowercapMeter::open(
                "intel-rapl:0",
                "package-0",
                EnergyScope::CpuPackage { package_id: 0 },
                &zone.energy_path,
                Some(zone.range_path.clone()),
            )
            .unwrap();
            fs::write(&zone.energy_path, "not-an-integer").unwrap();

            assert!(matches!(
                meter.read_counter(),
                Err(EnergyMeterError::BackendFailure { .. })
            ));
        }
    }
}

/// NVIDIA NVML cumulative GPU energy backend.
///
/// This adapter is opt-in through the `nvml` Cargo feature and dynamically
/// loads the NVIDIA management library at runtime. A successful library load
/// does not imply that the selected GPU supports cumulative energy readings;
/// each device is probed and `NOT_SUPPORTED` remains an explicit capability
/// state. The counter is reported in millijoules since driver reload.
///
/// NVIDIA documents this counter for Volta-or-newer fully supported devices.
/// Even on those architectures, the backend must handle a device/driver that
/// returns NotSupported or loses the GPU. The adapter stores the immutable GPU
/// UUID and does not rely on an enumeration index after opening the device.
#[cfg(feature = "nvml")]
pub mod nvml_backend {
    use super::*;
    use nvml_wrapper::{Nvml, error::NvmlError};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    static NEXT_NVML_EPOCH: AtomicU64 = AtomicU64::new(1);

    /// NVML-backed cumulative energy meter for one GPU UUID.
    pub struct NvmlEnergyMeter {
        nvml: Arc<Nvml>,
        descriptor: EnergyMeterDescriptor,
        driver_version: String,
        nvml_version: Option<String>,
        model: Option<String>,
        counter_epoch: u64,
        last_raw_counter: u64,
        started_at: Instant,
    }

    impl NvmlEnergyMeter {
        /// Initialize NVML and open a meter for the GPU currently at index.
        ///
        /// The index is used only to discover the immutable UUID. For a
        /// multi-GPU process, prefer `from_shared` and pass one shared
        /// `Arc<Nvml>` to all meters rather than initializing NVML repeatedly.
        pub fn open_by_index(index: u32) -> Result<Self, EnergyMeterError> {
            let nvml = Arc::new(Nvml::init().map_err(map_nvml_error)?);
            Self::from_shared(nvml, index)
        }

        /// Open a meter using a shared NVML instance and the current device
        /// index. The resolved UUID is retained for every subsequent read.
        pub fn from_shared(
            nvml: Arc<Nvml>,
            index: u32,
        ) -> Result<Self, EnergyMeterError> {
            let device = nvml.device_by_index(index).map_err(map_nvml_error)?;
            let uuid = device.uuid().map_err(map_nvml_error)?;
            let initial_energy = device.total_energy_consumption().map_err(map_nvml_error)?;
            let driver_version = nvml.sys_driver_version().map_err(map_nvml_error)?;
            let nvml_version = nvml.sys_nvml_version().ok();
            let model = device.name().ok();

            Ok(Self {
                nvml,
                descriptor: EnergyMeterDescriptor {
                    backend: EnergyMeterBackend::Nvml,
                    sensor_id: format!("nvml:{uuid}:total-energy"),
                    device_id: uuid,
                    scope: EnergyScope::GpuDevice,
                    unit: EnergyCounterUnit::Millijoules,
                    // NVML exposes a cumulative counter but does not document
                    // a rollover modulus for this API. Never infer one.
                    wrap_modulus: None,
                },
                driver_version,
                nvml_version,
                model,
                counter_epoch: NEXT_NVML_EPOCH.fetch_add(1, Ordering::Relaxed),
                last_raw_counter: initial_energy,
                started_at: Instant::now(),
            })
        }

        /// NVIDIA driver version captured when this meter was opened.
        pub fn driver_version(&self) -> &str {
            &self.driver_version
        }

        /// NVML library version when the driver exposes it.
        pub fn nvml_version(&self) -> Option<&str> {
            self.nvml_version.as_deref()
        }

        /// GPU model name when the device exposes it.
        pub fn model(&self) -> Option<&str> {
            self.model.as_deref()
        }

        fn read_energy(&self) -> Result<u64, EnergyMeterError> {
            self.nvml
                .device_by_uuid(&self.descriptor.device_id)
                .map_err(map_nvml_error)?
                .total_energy_consumption()
                .map_err(map_nvml_error)
        }
    }

    impl EnergyMeter for NvmlEnergyMeter {
        fn descriptor(&self) -> &EnergyMeterDescriptor {
            &self.descriptor
        }

        fn availability(&self) -> EnergyMeterAvailability {
            match self.read_energy() {
                Ok(_) => EnergyMeterAvailability::Available,
                Err(EnergyMeterError::Unsupported { reason }) => {
                    EnergyMeterAvailability::Unsupported { reason }
                }
                Err(EnergyMeterError::PermissionDenied { reason }) => {
                    EnergyMeterAvailability::PermissionDenied { reason }
                }
                Err(EnergyMeterError::Unavailable { reason }) => {
                    EnergyMeterAvailability::Unavailable { reason }
                }
                Err(EnergyMeterError::BackendFailure { reason }) => {
                    EnergyMeterAvailability::Unavailable { reason }
                }
                Err(EnergyMeterError::ClockUnavailable) => EnergyMeterAvailability::Unavailable {
                    reason: "monotonic clock unavailable".into(),
                },
            }
        }

        fn read_counter(&mut self) -> Result<EnergyCounterReading, EnergyMeterError> {
            let raw_counter = self.read_energy()?;
            // NVML's total-energy API has no documented rollover modulus and
            // resets on driver reload. A decrease starts a new epoch; callers
            // cannot compute an interval across that boundary.
            advance_epoch_on_counter_decrease(
                self.last_raw_counter,
                raw_counter,
                &mut self.counter_epoch,
            )?;
            self.last_raw_counter = raw_counter;

            Ok(EnergyCounterReading {
                descriptor: self.descriptor.clone(),
                counter_epoch: self.counter_epoch,
                raw_counter,
                monotonic_timestamp_ns: self.started_at.elapsed().as_nanos(),
            })
        }
    }

    fn advance_epoch_on_counter_decrease(
        previous: u64,
        current: u64,
        epoch: &mut u64,
    ) -> Result<(), EnergyMeterError> {
        if current < previous {
            *epoch = epoch.checked_add(1).ok_or_else(|| {
                EnergyMeterError::BackendFailure {
                    reason: "NVML counter epoch overflow".into(),
                }
            })?;
        }
        Ok(())
    }

    fn map_nvml_error(error: NvmlError) -> EnergyMeterError {
        match error {
            NvmlError::NotSupported => EnergyMeterError::Unsupported {
                reason: "NVML cumulative energy counter is not supported by this GPU/driver".into(),
            },
            NvmlError::NoPermission => EnergyMeterError::PermissionDenied {
                reason: "NVML denied access to this GPU energy counter".into(),
            },
            NvmlError::DriverNotLoaded
            | NvmlError::GpuLost
            | NvmlError::LibraryNotFound
            | NvmlError::LibRmVersionMismatch => EnergyMeterError::Unavailable {
                reason: error.to_string(),
            },
            other => EnergyMeterError::BackendFailure {
                reason: other.to_string(),
            },
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_decreased_counter_advances_epoch_but_a_monotonic_counter_does_not() {
            let mut epoch = 7;
            advance_epoch_on_counter_decrease(100, 120, &mut epoch).unwrap();
            assert_eq!(epoch, 7);
            advance_epoch_on_counter_decrease(120, 3, &mut epoch).unwrap();
            assert_eq!(epoch, 8);
        }

        #[test]
        fn counter_epoch_overflow_is_rejected() {
            let mut epoch = u64::MAX;
            assert!(matches!(
                advance_epoch_on_counter_decrease(10, 1, &mut epoch),
                Err(EnergyMeterError::BackendFailure { .. })
            ));
            assert_eq!(epoch, u64::MAX);
        }

        #[test]
        fn not_supported_is_not_reported_as_zero_energy() {
            assert!(matches!(
                map_nvml_error(NvmlError::NotSupported),
                EnergyMeterError::Unsupported { .. }
            ));
        }

        #[test]
        fn permission_and_gpu_loss_remain_distinct_capabilities() {
            assert!(matches!(
                map_nvml_error(NvmlError::NoPermission),
                EnergyMeterError::PermissionDenied { .. }
            ));
            assert!(matches!(
                map_nvml_error(NvmlError::GpuLost),
                EnergyMeterError::Unavailable { .. }
            ));
        }
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
