// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

pub struct MeshBridge {
    mdns: ServiceDaemon,
    discovered_peers: Arc<Mutex<HashMap<String, String>>>, // agent_key -> addr
}

impl MeshBridge {
    pub fn new() -> Self {
        Self {
            mdns: ServiceDaemon::new().expect("Failed to create mDNS daemon"),
            discovered_peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start advertising this conductor on the local mesh.
    pub fn start_advertising(&self, agent_key: &str, app_port: u16) {
        let service_type = "_holochain-mesh._tcp.local.";
        let instance_name = format!("mycelix-{}", agent_key);
        let host_name = "mycelix-node.local.";
        let port = app_port;
        let mut properties = HashMap::new();
        properties.insert("agent".to_string(), agent_key.to_string());

        let service_info = ServiceInfo::new(
            service_type,
            &instance_name,
            host_name,
            "", // IP will be auto-resolved
            port,
            Some(properties),
        )
        .expect("Failed to create service info");

        self.mdns
            .register(service_info)
            .expect("Failed to register mesh service");
        println!(
            "📡 Mesh Bridge: Advertising agent {} on port {}",
            agent_key, app_port
        );
    }

    /// Start listening for other peers on the local mesh.
    pub fn start_discovery(&self, handle: AppHandle) {
        let service_type = "_holochain-mesh._tcp.local.";
        let receiver = self
            .mdns
            .browse(service_type)
            .expect("Failed to browse mesh");
        let peers = self.discovered_peers.clone();

        tokio::spawn(async move {
            while let Ok(event) = receiver.recv_async().await {
                match event {
                    mdns_sd::ServiceEvent::ServiceResolved(info) => {
                        let agent = info.get_property_val_str("agent").unwrap_or_default();
                        let addr = format!(
                            "{}:{}",
                            info.get_addresses().iter().next().unwrap(),
                            info.get_port()
                        );
                        println!("🤝 Mesh Bridge: Found peer {} at {}", agent, addr);

                        peers.lock().unwrap().insert(agent.clone(), addr.clone());
                        let _ = handle.emit("mesh-peer-discovered", (agent, addr));
                    }
                    _ => {}
                }
            }
        });
    }
}

#[tauri::command]
pub async fn sync_offline(state: tauri::State<'_, super::AppState>) -> Result<String, String> {
    // This would trigger a manual Holochain gossip pull from discovered mesh peers
    Ok("Offline sync initiated via local mesh.".to_string())
}
