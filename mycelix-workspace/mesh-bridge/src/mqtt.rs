// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MQTT gateway transport — bridges LoRa mesh ↔ internet via Meshtastic MQTT.
//!
//! Meshtastic devices can natively bridge their LoRa mesh to MQTT.
//! This transport connects to that MQTT broker, enabling:
//!   - Off-grid LoRa nodes ↔ internet-connected Holochain conductors
//!   - Multiple mesh networks bridged through a shared MQTT broker
//!   - Store-and-forward when LoRa gateway is intermittently online
//!
//! # Architecture
//!
//! ```text
//! Off-grid mesh nodes (LoRa only)
//!     ↓ Meshtastic LoRa mesh
//! Gateway node (LoRa + WiFi)
//!     ↓ Meshtastic MQTT bridge (native feature)
//! MQTT Broker (Mosquitto / HiveMQ / etc.)
//!     ↓ This transport subscribes
//! mesh-bridge → Holochain conductor
//! ```
//!
//! # Environment Variables
//!
//! - `MQTT_HOST`: Broker hostname (default: `localhost`)
//! - `MQTT_PORT`: Broker port (default: 1883)
//! - `MQTT_TOPIC_ROOT`: Topic prefix (default: `msh/mycelix`)
//! - `MQTT_CLIENT_ID`: Client ID (default: `mycelix-mesh-bridge`)
//! - `MQTT_USERNAME`: Auth username (optional)
//! - `MQTT_PASSWORD`: Auth password (optional)

#[cfg(feature = "mqtt")]
use anyhow::{Context, Result};
#[cfg(feature = "mqtt")]
use rumqttc::{AsyncClient, Event, EventLoop, MqttOptions, Packet, QoS};
#[cfg(feature = "mqtt")]
use std::sync::Arc;
#[cfg(feature = "mqtt")]
use tokio::sync::Mutex;

#[cfg(feature = "mqtt")]
use crate::transport::MeshTransport;

/// Default MQTT topic root for Meshtastic-Mycelix bridge traffic.
#[cfg(feature = "mqtt")]
const DEFAULT_TOPIC_ROOT: &str = "msh/mycelix";

/// Default MQTT QoS level.
#[cfg(feature = "mqtt")]
const DEFAULT_QOS: QoS = QoS::AtLeastOnce;

/// MQTT gateway configuration.
#[cfg(feature = "mqtt")]
#[derive(Debug, Clone)]
pub struct MqttConfig {
    pub host: String,
    pub port: u16,
    pub topic_root: String,
    pub client_id: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[cfg(feature = "mqtt")]
impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            host: "localhost".into(),
            port: 1883,
            topic_root: DEFAULT_TOPIC_ROOT.into(),
            client_id: "mycelix-mesh-bridge".into(),
            username: None,
            password: None,
        }
    }
}

#[cfg(feature = "mqtt")]
impl MqttConfig {
    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("MQTT_HOST").unwrap_or_else(|_| "localhost".into()),
            port: std::env::var("MQTT_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1883),
            topic_root: std::env::var("MQTT_TOPIC_ROOT")
                .unwrap_or_else(|_| DEFAULT_TOPIC_ROOT.into()),
            client_id: std::env::var("MQTT_CLIENT_ID")
                .unwrap_or_else(|_| "mycelix-mesh-bridge".into()),
            username: std::env::var("MQTT_USERNAME").ok(),
            password: std::env::var("MQTT_PASSWORD").ok(),
        }
    }

    /// Outbound topic: mesh-bridge publishes here for LoRa nodes.
    pub fn tx_topic(&self) -> String {
        format!("{}/tx", self.topic_root)
    }

    /// Inbound topic: mesh-bridge subscribes here for data from LoRa nodes.
    pub fn rx_topic(&self) -> String {
        format!("{}/rx/#", self.topic_root)
    }
}

/// MQTT mesh transport.
///
/// Publishes outbound payloads to `{topic_root}/tx` and subscribes
/// to `{topic_root}/rx/#` for inbound traffic from LoRa mesh nodes.
#[cfg(feature = "mqtt")]
pub struct MqttTransport {
    client: AsyncClient,
    event_loop: Arc<Mutex<EventLoop>>,
    config: MqttConfig,
    /// Buffered inbound payloads from the event loop.
    inbox: Arc<Mutex<Vec<Vec<u8>>>>,
}

#[cfg(feature = "mqtt")]
impl MqttTransport {
    /// Connect to the MQTT broker and subscribe to mesh topics.
    pub async fn new(config: MqttConfig) -> Result<Self> {
        let mut mqtt_opts = MqttOptions::new(
            &config.client_id,
            &config.host,
            config.port,
        );
        mqtt_opts.set_keep_alive(std::time::Duration::from_secs(30));

        if let (Some(user), Some(pass)) = (&config.username, &config.password) {
            mqtt_opts.set_credentials(user, pass);
        }

        let (client, event_loop) = AsyncClient::new(mqtt_opts, 256);

        // Subscribe to inbound topic
        let rx_topic = config.rx_topic();
        client.subscribe(&rx_topic, DEFAULT_QOS).await
            .context(format!("Failed to subscribe to {rx_topic}"))?;

        tracing::info!(
            "MQTT connected to {}:{}, pub={}, sub={}",
            config.host,
            config.port,
            config.tx_topic(),
            rx_topic
        );

        let inbox = Arc::new(Mutex::new(Vec::new()));

        // Spawn event loop pump that drains MQTT events into inbox
        let inbox_clone = inbox.clone();
        let event_loop = Arc::new(Mutex::new(event_loop));
        let el_clone = event_loop.clone();

        tokio::spawn(async move {
            loop {
                let notification = {
                    let mut el = el_clone.lock().await;
                    el.poll().await
                };

                match notification {
                    Ok(Event::Incoming(Packet::Publish(publish))) => {
                        let payload = publish.payload.to_vec();
                        if !payload.is_empty() {
                            tracing::debug!(
                                "MQTT RX on {}: {} bytes",
                                publish.topic,
                                payload.len()
                            );
                            inbox_clone.lock().await.push(payload);
                        }
                    }
                    Ok(_) => {} // ConnAck, SubAck, PingResp — ignore
                    Err(e) => {
                        tracing::warn!("MQTT event loop error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
            }
        });

        Ok(Self {
            client,
            event_loop,
            config,
            inbox,
        })
    }
}

#[cfg(feature = "mqtt")]
#[async_trait::async_trait]
impl MeshTransport for MqttTransport {
    fn name(&self) -> &str {
        "mqtt"
    }

    async fn send(&self, frame: &[u8]) -> Result<()> {
        let topic = self.config.tx_topic();
        self.client
            .publish(&topic, DEFAULT_QOS, false, frame)
            .await
            .context("MQTT publish failed")?;

        tracing::debug!("MQTT TX: {} bytes → {}", frame.len(), topic);
        Ok(())
    }

    async fn recv(&self, timeout_ms: u64) -> Result<Option<Vec<u8>>> {
        let deadline = tokio::time::Instant::now()
            + tokio::time::Duration::from_millis(timeout_ms);

        loop {
            {
                let mut inbox = self.inbox.lock().await;
                if !inbox.is_empty() {
                    return Ok(Some(inbox.remove(0)));
                }
            }

            if tokio::time::Instant::now() >= deadline {
                return Ok(None);
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }

    fn clone_box(&self) -> Box<dyn MeshTransport> {
        // MQTT transport shares the same client + inbox via Arc
        Box::new(Self {
            client: self.client.clone(),
            event_loop: self.event_loop.clone(),
            config: self.config.clone(),
            inbox: self.inbox.clone(),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "mqtt")]
mod tests {
    use super::*;

    #[test]
    fn test_mqtt_config_default() {
        let config = MqttConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 1883);
        assert_eq!(config.topic_root, "msh/mycelix");
        assert_eq!(config.client_id, "mycelix-mesh-bridge");
        assert!(config.username.is_none());
        assert!(config.password.is_none());
    }

    #[test]
    fn test_mqtt_topics() {
        let config = MqttConfig::default();
        assert_eq!(config.tx_topic(), "msh/mycelix/tx");
        assert_eq!(config.rx_topic(), "msh/mycelix/rx/#");
    }

    #[test]
    fn test_mqtt_custom_topic() {
        let config = MqttConfig {
            topic_root: "community/example".into(),
            ..Default::default()
        };
        assert_eq!(config.tx_topic(), "community/example/tx");
        assert_eq!(config.rx_topic(), "community/example/rx/#");
    }
}
