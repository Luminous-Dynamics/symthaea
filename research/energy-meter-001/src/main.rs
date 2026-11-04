#![forbid(unsafe_code)]
#![deny(warnings)]

//! Probe real machine energy-counter availability and measure a short interval.
//!
//! Run with:
//! cargo run --locked --all-features --manifest-path research/energy-meter-001/Cargo.toml --bin energy-meter-probe -- 500
//!
//! The optional argument is the sample interval in milliseconds (default 500).
//! Values are device/sensor-scope readings, not process-energy attribution.

use serde::Serialize;
use std::path::Path;
use std::thread;
use std::time::Duration;
use symthaea_energy_meter_verification::physics::energy_meter::{
    EnergyCounterReading, EnergyDelta, EnergyMeter, EnergyMeterAvailability,
    EnergyMeterDescriptor, EnergyMeterError, EnergyScope,
};

#[derive(Serialize)]
struct ProbeResult {
    sensor_id: String,
    descriptor: Option<EnergyMeterDescriptor>,
    availability: Option<EnergyMeterAvailability>,
    status: String,
    start: Option<EnergyCounterReading>,
    end: Option<EnergyCounterReading>,
    energy_microjoules: Option<u128>,
    energy_joules: Option<f64>,
    average_power_watts: Option<f64>,
    error: Option<String>,
}

#[derive(Serialize)]
struct ProbeReport {
    schema_version: u32,
    interval_ms: u64,
    powercap_scan_enabled: bool,
    nvml_probe_enabled: bool,
    discovery_complete: bool,
    discovered_sensor_count: usize,
    sensor_results: Vec<ProbeResult>,
    notes: Vec<String>,
}

struct ActiveMeter {
    meter: Box<dyn EnergyMeter>,
    descriptor: EnergyMeterDescriptor,
    start: EnergyCounterReading,
}

fn failed_result(
    sensor_id: impl Into<String>,
    error: impl Into<String>,
    status: &str,
) -> ProbeResult {
    ProbeResult {
        sensor_id: sensor_id.into(),
        descriptor: None,
        availability: None,
        status: status.to_owned(),
        start: None,
        end: None,
        energy_microjoules: None,
        energy_joules: None,
        average_power_watts: None,
        error: Some(error.into()),
    }
}

fn status_for_error(error: &EnergyMeterError) -> &'static str {
    match error {
        EnergyMeterError::Unsupported { .. } => "unsupported",
        EnergyMeterError::PermissionDenied { .. } => "permission_denied",
        EnergyMeterError::Unavailable { .. } => "unavailable",
        EnergyMeterError::BackendFailure { .. } => "backend_failure",
        EnergyMeterError::ClockUnavailable => "clock_unavailable",
    }
}

#[cfg(target_os = "linux")]
fn discover_powercap(
    directory: &Path,
    depth: usize,
    meters: &mut Vec<Box<dyn EnergyMeter>>,
    errors: &mut Vec<ProbeResult>,
) {
    if depth > 4 {
        return;
    }

    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            errors.push(failed_result(
                directory.display().to_string(),
                error.to_string(),
                "discovery_directory_error",
            ));
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(failed_result(
                    directory.display().to_string(),
                    error.to_string(),
                    "discovery_entry_error",
                ));
                continue;
            }
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let energy_path = path.join("energy_uj");
        if energy_path.is_file() {
            let fallback_name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown-powercap-zone".into());
            let name = std::fs::read_to_string(path.join("name"))
                .map(|value| value.trim().to_owned())
                .unwrap_or(fallback_name);
            let range_path = path.join("max_energy_range_uj");
            let max_range_path = range_path.is_file().then_some(range_path);

            match symthaea_energy_meter_verification::physics::energy_meter::linux_powercap::LinuxPowercapMeter::open(
                format!("linux-powercap:{}", path.display()),
                name.clone(),
                EnergyScope::Custom(format!("Linux powercap zone: {name}")),
                energy_path,
                max_range_path,
            ) {
                Ok(meter) => meters.push(Box::new(meter)),
                Err(error) => errors.push(failed_result(
                    format!("linux-powercap:{}", path.display()),
                    format!("{error:?}"),
                    status_for_error(&error),
                )),
            }
        }

        discover_powercap(&path, depth + 1, meters, errors);
    }
}

#[cfg(not(target_os = "linux"))]
fn discover_powercap(
    _directory: &Path,
    _depth: usize,
    _meters: &mut Vec<Box<dyn EnergyMeter>>,
    _errors: &mut Vec<ProbeResult>,
) {
}

#[cfg(feature = "nvml")]
fn discover_nvml(
    meters: &mut Vec<Box<dyn EnergyMeter>>,
    errors: &mut Vec<ProbeResult>,
) {
    use std::sync::Arc;
    use nvml_wrapper::Nvml;
    use symthaea_energy_meter_verification::physics::energy_meter::nvml_backend::NvmlEnergyMeter;

    let nvml = match Nvml::init() {
        Ok(nvml) => Arc::new(nvml),
        Err(error) => {
            errors.push(failed_result("nvml", error.to_string(), "unavailable"));
            return;
        }
    };

    let device_count = match nvml.device_count() {
        Ok(count) => count,
        Err(error) => {
            errors.push(failed_result("nvml", error.to_string(), "unavailable"));
            return;
        }
    };

    for index in 0..device_count {
        match NvmlEnergyMeter::from_shared(Arc::clone(&nvml), index) {
            Ok(meter) => meters.push(Box::new(meter)),
            Err(error) => errors.push(failed_result(
                format!("nvml-device-index:{index}"),
                format!("{error:?}"),
                status_for_error(&error),
            )),
        }
    }
}

#[cfg(not(feature = "nvml"))]
fn discover_nvml(_meters: &mut Vec<Box<dyn EnergyMeter>>, _errors: &mut Vec<ProbeResult>) {
}

fn probe_meters(
    meters: Vec<Box<dyn EnergyMeter>>,
    interval_ms: u64,
    mut results: Vec<ProbeResult>,
) -> Vec<ProbeResult> {
    let mut active = Vec::new();

    for mut meter in meters {
        let descriptor = meter.descriptor().clone();
        let availability = meter.availability();
        if availability != EnergyMeterAvailability::Available {
            let status = match &availability {
                EnergyMeterAvailability::Available => "available",
                EnergyMeterAvailability::Unsupported { .. } => "unsupported",
                EnergyMeterAvailability::PermissionDenied { .. } => "permission_denied",
                EnergyMeterAvailability::Unavailable { .. } => "unavailable",
            };
            results.push(ProbeResult {
                sensor_id: descriptor.sensor_id.clone(),
                descriptor: Some(descriptor),
                availability: Some(availability),
                status: status.to_owned(),
                start: None,
                end: None,
                energy_microjoules: None,
                energy_joules: None,
                average_power_watts: None,
                error: None,
            });
            continue;
        }

        match meter.read_counter() {
            Ok(start) => active.push(ActiveMeter {
                meter,
                descriptor,
                start,
            }),
            Err(error) => results.push(failed_result(
                descriptor.sensor_id,
                format!("{error:?}"),
                status_for_error(&error),
            )),
        }
    }

    if !active.is_empty() {
        thread::sleep(Duration::from_millis(interval_ms));
    }

    for mut active_meter in active {
        let sensor_id = active_meter.descriptor.sensor_id.clone();
        match active_meter.meter.read_counter() {
            Ok(end) => match EnergyDelta::between(&active_meter.start, &end) {
                Ok(delta) => results.push(ProbeResult {
                    sensor_id,
                    descriptor: Some(active_meter.descriptor),
                    availability: Some(EnergyMeterAvailability::Available),
                    status: "measured".into(),
                    start: Some(active_meter.start),
                    end: Some(end),
                    energy_microjoules: Some(delta.energy_microjoules()),
                    energy_joules: Some(delta.energy_joules()),
                    average_power_watts: Some(delta.average_power_watts()),
                    error: None,
                }),
                Err(error) => results.push(ProbeResult {
                    sensor_id,
                    descriptor: Some(active_meter.descriptor),
                    availability: Some(EnergyMeterAvailability::Available),
                    status: "inconclusive".into(),
                    start: Some(active_meter.start),
                    end: Some(end),
                    energy_microjoules: None,
                    energy_joules: None,
                    average_power_watts: None,
                    error: Some(format!("{error:?}")),
                }),
            },
            Err(error) => results.push(ProbeResult {
                sensor_id,
                descriptor: Some(active_meter.descriptor),
                availability: None,
                status: status_for_error(&error).into(),
                start: Some(active_meter.start),
                end: None,
                energy_microjoules: None,
                energy_joules: None,
                average_power_watts: None,
                error: Some(format!("{error:?}")),
            }),
        }
    }

    results
}

fn main() {
    let interval_ms = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(500)
        .max(1);

    let mut meters: Vec<Box<dyn EnergyMeter>> = Vec::new();
    let mut errors = Vec::new();

    #[cfg(target_os = "linux")]
    discover_powercap(Path::new("/sys/class/powercap"), 0, &mut meters, &mut errors);

    #[cfg(not(target_os = "linux"))]
    errors.push(failed_result(
        "linux-powercap",
        "Linux powercap is unavailable on this operating system",
        "unsupported",
    ));

    discover_nvml(&mut meters, &mut errors);
    let sensor_results = probe_meters(meters, interval_ms, errors);
    let discovery_complete = !sensor_results
        .iter()
        .any(|result| result.status.starts_with("discovery_"));
    let discovered_sensor_count = sensor_results
        .iter()
        .filter(|result| result.descriptor.is_some())
        .count();

    let report = ProbeReport {
        schema_version: 1,
        interval_ms,
        powercap_scan_enabled: cfg!(target_os = "linux"),
        nvml_probe_enabled: cfg!(feature = "nvml"),
        discovery_complete,
        discovered_sensor_count,
        sensor_results,
        notes: vec![
            "NVML energy is whole-GPU device energy, not per-process energy.".into(),
            "Linux powercap scope is the named kernel zone supplied by sysfs.".into(),
            "Missing or unsupported sensors never become numeric zero.".into(),
            "A sampled cumulative-counter interval is not a whole-system energy claim unless sensor coverage is complete.".into(),
            "discovery_complete means the configured discovery paths were traversed without filesystem errors; it does not prove every physical power domain is exposed.".into(),
        ],
    };

    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize sensor probe report: {error}");
            std::process::exit(1);
        }
    }
}
