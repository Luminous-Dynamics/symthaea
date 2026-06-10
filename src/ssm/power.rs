// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Power telemetry SSM sensor (INA219 or simulated).

use anyhow::Result;
#[cfg(feature = "ssm-power-hal")]
use anyhow::anyhow;
use std::collections::HashMap;
#[cfg(feature = "ssm-power-hal")]
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(feature = "ssm-power-hal")]
use tracing::info;
use tracing::warn;

use symthaea_ssm::{SelectiveParams, SsmState};

use crate::consciousness::interoception::InteroceptionTag;
use crate::swarm::mesh::{MeshUrgency, SensorInput, SensorReading};

#[cfg(feature = "ssm-power-hal")]
use symthaea_hal::HalSensorAdapter;

const DEFAULT_HZ: f32 = 10.0;
const DEFAULT_D_STATE: usize = 16;
const DEFAULT_BASE_WATTS: f32 = 6.0;
const DEFAULT_AMPLITUDE_WATTS: f32 = 0.4;
const DEFAULT_PERIOD_S: f32 = 6.0;

pub struct PowerSsmConfig {
    pub sensor_id: String,
    pub hz: f32,
    pub d_state: usize,
    pub urgency: MeshUrgency,
    pub sim_base_watts: f32,
    pub sim_amplitude_watts: f32,
    pub sim_period_s: f32,
    pub use_real: bool,
}

impl PowerSsmConfig {
    pub fn from_env() -> Self {
        Self {
            sensor_id: std::env::var("SYMTHAEA_POWER_SENSOR_ID")
                .unwrap_or_else(|_| "ina219::power_ssm".to_string()),
            hz: env_f32("SYMTHAEA_SSM_HZ", DEFAULT_HZ),
            d_state: env_usize("SYMTHAEA_SSM_D_STATE", DEFAULT_D_STATE),
            urgency: MeshUrgency::Cruise,
            sim_base_watts: env_f32("SYMTHAEA_SIM_BASE_WATTS", DEFAULT_BASE_WATTS),
            sim_amplitude_watts: env_f32("SYMTHAEA_SIM_AMPLITUDE_WATTS", DEFAULT_AMPLITUDE_WATTS),
            sim_period_s: env_f32("SYMTHAEA_SIM_PERIOD_S", DEFAULT_PERIOD_S),
            use_real: env_bool("SYMTHAEA_INA219_REAL"),
        }
    }
}

trait PowerSource: Send + Sync {
    fn read_watts(&mut self) -> Result<f32>;
    fn is_available(&self) -> bool;
}

struct SimulatedIna219 {
    start: Instant,
    base_watts: f32,
    amplitude: f32,
    period_s: f32,
}

impl SimulatedIna219 {
    fn new(base_watts: f32, amplitude: f32, period_s: f32) -> Self {
        Self {
            start: Instant::now(),
            base_watts,
            amplitude,
            period_s,
        }
    }
}

impl PowerSource for SimulatedIna219 {
    fn read_watts(&mut self) -> Result<f32> {
        let t = self.start.elapsed().as_secs_f32();
        let omega = std::f32::consts::TAU / self.period_s.max(0.01);
        Ok(self.base_watts + self.amplitude * (omega * t).sin())
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(feature = "ssm-power-hal")]
struct Ina219PowerSource {
    sensor: Mutex<
        symthaea_hal::EmbeddedSensor<linux_embedded_hal::I2cdev, symthaea_hal::Ina219Decoder>,
    >,
}

#[cfg(feature = "ssm-power-hal")]
impl Ina219PowerSource {
    fn new(bus_path: &str, shunt_ohms: f32, address: u8) -> Result<Self> {
        let bus = linux_embedded_hal::I2cdev::new(bus_path)
            .map_err(|e| anyhow!("Failed to open I2C bus {}: {}", bus_path, e))?;
        let decoder = symthaea_hal::Ina219Decoder::new(shunt_ohms).with_address(address);
        let mut sensor = symthaea_hal::EmbeddedSensor::new(bus, decoder);
        let _ = sensor.probe()?;
        Ok(Self {
            sensor: Mutex::new(sensor),
        })
    }
}

#[cfg(feature = "ssm-power-hal")]
impl PowerSource for Ina219PowerSource {
    fn read_watts(&mut self) -> Result<f32> {
        let mut sensor = self
            .sensor
            .lock()
            .map_err(|_| anyhow!("INA219 sensor lock poisoned"))?;
        if let Some(values) = sensor.read_raw() {
            let current_a = values.get(0).copied().unwrap_or(0.0);
            let bus_v = values.get(1).copied().unwrap_or(0.0);
            Ok(current_a * bus_v)
        } else {
            Err(anyhow!("INA219 returned no data"))
        }
    }

    fn is_available(&self) -> bool {
        self.sensor
            .lock()
            .map(|sensor| sensor.is_available())
            .unwrap_or(false)
    }
}

pub struct PowerSsmSensor {
    sensor_id: String,
    urgency: MeshUrgency,
    source: Box<dyn PowerSource>,
    state: SsmState,
    params: SelectiveParams,
    input: Vec<f32>,
    output: Vec<f32>,
    last_watts: f32,
}

impl PowerSsmSensor {
    pub fn from_env() -> Result<Self> {
        let config = PowerSsmConfig::from_env();
        if config.use_real {
            #[cfg(feature = "ssm-power-hal")]
            {
                let bus_path = std::env::var("SYMTHAEA_INA219_BUS")
                    .unwrap_or_else(|_| "/dev/i2c-1".to_string());
                let shunt = env_f32("SYMTHAEA_INA219_SHUNT_OHMS", 0.1);
                let address = env_hex_u8("SYMTHAEA_INA219_ADDRESS", 0x45);
                match Ina219PowerSource::new(&bus_path, shunt, address) {
                    Ok(source) => {
                        info!(
                            target: "symthaea::ssm::power",
                            bus = %bus_path,
                            address = format!("0x{:02X}", address),
                            "INA219 power source enabled"
                        );
                        return Ok(Self::new(config, Box::new(source)));
                    }
                    Err(err) => {
                        warn!(
                            target: "symthaea::ssm::power",
                            error = %err,
                            "INA219 init failed; falling back to simulated source"
                        );
                    }
                }
            }

            #[cfg(not(feature = "ssm-power-hal"))]
            {
                warn!(
                    target: "symthaea::ssm::power",
                    "SYMTHAEA_INA219_REAL set but ssm-power-hal feature is disabled; using simulation"
                );
            }
        }

        let source = SimulatedIna219::new(
            config.sim_base_watts,
            config.sim_amplitude_watts,
            config.sim_period_s,
        );
        Ok(Self::new(config, Box::new(source)))
    }

    pub fn simulated() -> Self {
        let config = PowerSsmConfig::from_env();
        let source = SimulatedIna219::new(
            config.sim_base_watts,
            config.sim_amplitude_watts,
            config.sim_period_s,
        );
        Self::new(config, Box::new(source))
    }

    fn new(config: PowerSsmConfig, source: Box<dyn PowerSource>) -> Self {
        let d_model = 1;
        let mut params = SelectiveParams::new(d_model, config.d_state);
        params.set_delta(1.0 / config.hz.max(0.1));
        Self {
            sensor_id: config.sensor_id,
            urgency: config.urgency,
            source,
            state: SsmState::new(d_model, config.d_state),
            params,
            input: vec![0.0; d_model],
            output: vec![0.0; d_model],
            last_watts: 0.0,
        }
    }
}

impl SensorInput for PowerSsmSensor {
    fn name(&self) -> &str {
        &self.sensor_id
    }

    fn poll(&mut self) -> Option<SensorReading> {
        if !self.source.is_available() {
            return None;
        }

        if let Ok(watts) = self.source.read_watts() {
            self.last_watts = watts;
        }

        self.input[0] = self.last_watts;
        self.state.step(&self.input, &self.params, &mut self.output);

        let mut tags = vec![
            InteroceptionTag::ThermodynamicLoad.as_topic().to_string(),
            "interoception".to_string(),
            "power".to_string(),
            "watts".to_string(),
            "ssm".to_string(),
        ];
        tags.sort();
        tags.dedup();

        let mut metadata = HashMap::new();
        metadata.insert(
            "interoception_tag".to_string(),
            InteroceptionTag::ThermodynamicLoad.as_topic().to_string(),
        );
        metadata.insert(
            "interoception_label".to_string(),
            InteroceptionTag::ThermodynamicLoad.as_label().to_string(),
        );
        metadata.insert("watts".to_string(), format!("{:.6}", self.last_watts));
        metadata.insert("ssm_output".to_string(), format!("{:.6}", self.output[0]));
        metadata.insert("units".to_string(), "watts".to_string());

        Some(SensorReading {
            sensor_id: self.sensor_id.clone(),
            values: vec![self.last_watts, self.output[0]],
            tags,
            metadata,
            urgency_override: None,
            timestamp_s: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
        })
    }

    fn urgency_class(&self) -> MeshUrgency {
        self.urgency
    }

    fn is_available(&self) -> bool {
        self.source.is_available()
    }
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[cfg(feature = "ssm-power-hal")]
fn env_hex_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            u8::from_str_radix(trimmed, 16).ok()
        })
        .unwrap_or(default)
}
