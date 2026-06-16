// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::Result;
#[cfg(feature = "ina219-linux")]
use anyhow::anyhow;
use std::time::{Duration, Instant};

use symthaea_ssm::{SelectiveParams, SsmState};

#[cfg(feature = "ina219-linux")]
use symthaea_hal::HalSensorAdapter;

trait PowerSource {
    fn read_watts(&mut self) -> Result<f32>;
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
}

#[cfg(feature = "ina219-linux")]
struct Ina219PowerSource {
    sensor: symthaea_hal::sensor::EmbeddedSensor<
        linux_embedded_hal::I2cdev,
        symthaea_hal::Ina219Decoder,
    >,
}

#[cfg(feature = "ina219-linux")]
impl Ina219PowerSource {
    fn new(bus_path: &str, shunt_ohms: f32, address: u8) -> Result<Self> {
        let bus = linux_embedded_hal::I2cdev::new(bus_path)
            .map_err(|e| anyhow!("Failed to open I2C bus {}: {}", bus_path, e))?;
        let decoder = symthaea_hal::Ina219Decoder::new(shunt_ohms).with_address(address);
        let mut sensor = symthaea_hal::EmbeddedSensor::new(bus, decoder);
        let _ = sensor.probe()?;
        Ok(Self { sensor })
    }
}

#[cfg(feature = "ina219-linux")]
impl PowerSource for Ina219PowerSource {
    fn read_watts(&mut self) -> Result<f32> {
        if let Some(values) = self.sensor.read_raw() {
            let current_a = values.get(0).copied().unwrap_or(0.0);
            let bus_v = values.get(1).copied().unwrap_or(0.0);
            Ok(current_a * bus_v)
        } else {
            Err(anyhow!("INA219 returned no data"))
        }
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

#[cfg(feature = "ina219-linux")]
fn env_hex_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| {
            let trimmed = v.trim_start_matches("0x");
            u8::from_str_radix(trimmed, 16).ok()
        })
        .unwrap_or(default)
}

fn build_source() -> Result<Box<dyn PowerSource>> {
    let use_real = env_bool("SYMTHAEA_INA219_REAL");
    if use_real {
        #[cfg(feature = "ina219-linux")]
        {
            let bus_path =
                std::env::var("SYMTHAEA_INA219_BUS").unwrap_or_else(|_| "/dev/i2c-1".to_string());
            let shunt = env_f32("SYMTHAEA_INA219_SHUNT_OHMS", 0.1);
            let address = env_hex_u8("SYMTHAEA_INA219_ADDRESS", 0x45);
            let sensor = Ina219PowerSource::new(&bus_path, shunt, address)?;
            return Ok(Box::new(sensor));
        }
        #[cfg(not(feature = "ina219-linux"))]
        {
            eprintln!(
                "Real INA219 requested but feature 'ina219-linux' not enabled; using simulated source."
            );
        }
    }

    let base = env_f32("SYMTHAEA_SIM_BASE_WATTS", 6.0);
    let amplitude = env_f32("SYMTHAEA_SIM_AMPLITUDE_WATTS", 0.4);
    let period = env_f32("SYMTHAEA_SIM_PERIOD_S", 6.0);
    Ok(Box::new(SimulatedIna219::new(base, amplitude, period)))
}

fn main() -> Result<()> {
    let hz = env_f32("SYMTHAEA_SSM_HZ", 10.0);
    let seconds = env_f32("SYMTHAEA_SSM_SECONDS", 20.0);
    let d_model = env_usize("SYMTHAEA_SSM_D_MODEL", 1);
    let d_state = env_usize("SYMTHAEA_SSM_D_STATE", 16);

    let mut source = build_source()?;
    let mut state = SsmState::new(d_model, d_state);
    let mut params = SelectiveParams::new(d_model, d_state);
    params.set_delta(1.0 / hz.max(0.1));

    let mut input = vec![0.0_f32; d_model];
    let mut output = vec![0.0_f32; d_model];

    let steps = (seconds * hz).ceil() as usize;
    let tick = Duration::from_secs_f32(1.0 / hz.max(0.1));
    let mut last_watts = 0.0_f32;

    println!("SSM INA219 demo: {} steps at {:.2} Hz", steps, hz);
    for idx in 0..steps {
        let started = Instant::now();
        match source.read_watts() {
            Ok(watts) => {
                last_watts = watts;
            }
            Err(err) => {
                eprintln!("INA219 read failed (step {}): {}", idx, err);
            }
        }

        for v in &mut input {
            *v = last_watts;
        }
        state.step(&input, &params, &mut output);
        println!(
            "step {:04} | watts {:6.3} | ssm_out {:8.4}",
            idx, last_watts, output[0]
        );

        let elapsed = started.elapsed();
        if tick > elapsed {
            std::thread::sleep(tick - elapsed);
        }
    }

    Ok(())
}
