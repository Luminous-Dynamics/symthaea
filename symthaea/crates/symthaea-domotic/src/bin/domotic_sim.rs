// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::net::UdpSocket;
use std::thread::sleep;
use std::time::Duration;
use symthaea_domotic::EnvironmentalTelemetry;

fn main() -> anyhow::Result<()> {
    let socket = UdpSocket::bind("127.0.0.1:0")?;
    let target_addr = "127.0.0.1:4190";
    
    println!("🚀 Native Domotic Telemetry Simulator Engine Running.");
    println!("📡 Streaming environmental drift sequences to {}...\n", target_addr);

    let mut tick = 0.0f32;
    loop {
        // Model a smooth, cyclical day/night thermodynamic curve using wave variations
        let lux = (tick.sin() * 500.0 + 500.0).clamp(0.0, 1000.0);
        let temp_celsius = 20.0 + (tick.cos() * 5.0); // Drift between 15°C and 25°C
        let open_ratio = if tick.sin() > 0.5 { 1.0 } else { 0.0 }; // Simulate door switches

        let frame = EnvironmentalTelemetry {
            lux,
            temp_celsius,
            open_ratio,
        };

        let payload = serde_json::to_vec(&frame)?;
        socket.send_to(&payload, target_addr)?;

        println!("⚡ Broadcast -> Lux: {:5.1}, Temp: {:4.1}°C, Portal Open: {}", lux, temp_celsius, open_ratio > 0.5);
        
        tick += 0.1;
        sleep(Duration::from_millis(500)); // Update every 500ms
    }
}
