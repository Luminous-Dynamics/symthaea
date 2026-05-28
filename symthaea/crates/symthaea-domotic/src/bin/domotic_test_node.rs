// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_domotic::DomoticBridge;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize standard tracking outputs
    tracing_subscriber::fmt::init();
    
    println!("🧠 Initializing Domotic Mind Interface...");
    let mut bridge = DomoticBridge::new(16384); // 16K HDC resolution footprint
    
    // Bind to local pipeline conduit loop
    bridge.listen_and_encode("127.0.0.1:4190").await?;
    Ok(())
}
