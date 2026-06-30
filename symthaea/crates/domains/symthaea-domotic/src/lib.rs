// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use tokio::net::UdpSocket;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct EnvironmentalTelemetry {
    pub lux: f32,
    pub temp_celsius: f32,
    pub open_ratio: f32,
}

pub struct DomoticBridge {
    pub dimension: usize,
    pub active_state: ContinuousHV,
    // Pre-allocated structural anchor boundaries
    min_temp_anchor: ContinuousHV,
    max_temp_anchor: ContinuousHV,
    dark_lux_anchor: ContinuousHV,
    bright_lux_anchor: ContinuousHV,
}

impl DomoticBridge {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            active_state: ContinuousHV::zero(dimension),
            min_temp_anchor: ContinuousHV::random(dimension, 1001),
            max_temp_anchor: ContinuousHV::random(dimension, 1002),
            dark_lux_anchor: ContinuousHV::random(dimension, 2001),
            bright_lux_anchor: ContinuousHV::random(dimension, 2002),
        }
    }

    /// Spawns an isolated non-blocking UDP receiver loop to process environmental state changes
    pub async fn listen_and_encode(&mut self, bind_address: &str) -> Result<()> {
        let socket = UdpSocket::bind(bind_address).await?;
        tracing::info!("📡 Domotic Bridge listening on UDP: {}", bind_address);
        let mut buffer = [0u8; 1024];

        loop {
            let (size, _) = socket.recv_from(&mut buffer).await?;
            if let Ok(frame) = serde_json::from_slice::<EnvironmentalTelemetry>(&buffer[..size]) {
                // 1. Map Temperature along a continuous geometric axis [0°C to 50°C]
                let temp_ratio = frame.temp_celsius.clamp(0.0, 50.0) / 50.0;
                let mut temp_hv = self.min_temp_anchor.clone();
                temp_hv.lerp_in_place(&self.max_temp_anchor, 1.0 - temp_ratio, temp_ratio);

                // 2. Map Light levels along a continuous luminance axis [0.0 to 1000.0 lux]
                let lux_ratio = frame.lux.clamp(0.0, 1000.0) / 1000.0;
                let mut lux_hv = self.dark_lux_anchor.clone();
                lux_hv.lerp_in_place(&self.bright_lux_anchor, 1.0 - lux_ratio, lux_ratio);

                // 3. Map Spatial boundaries [0.0 = Closed, 1.0 = Fully Open]
                let spatial_hv =
                    ContinuousHV::random(self.dimension, (frame.open_ratio * 100.0) as u64);

                // 4. Collapse the independent environmental channels into a clean batch superposition
                self.active_state = ContinuousHV::bundle(&[&lux_hv, &temp_hv, &spatial_hv]);

                tracing::info!(
                    "✔ Metric Frame -> Lux: {:.1}, Temp: {:.1}°C, Open: {:.2}. State Manifold Integrated.",
                    frame.lux,
                    frame.temp_celsius,
                    frame.open_ratio
                );
            }
        }
    }
}
