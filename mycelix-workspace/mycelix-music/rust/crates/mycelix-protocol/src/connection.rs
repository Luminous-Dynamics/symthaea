// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUIC connection management

use crate::{ProtocolConfig, ProtocolError, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Reconnecting,
    Disconnected,
    Failed,
}

/// Connection event
#[derive(Debug, Clone)]
pub enum ConnectionEvent {
    Connected,
    Disconnected,
    Reconnecting,
    Error(String),
    LatencyUpdate(f32),
}

/// QUIC connection wrapper
pub struct Connection {
    state: ConnectionState,
    remote_addr: SocketAddr,
    local_addr: Option<SocketAddr>,
    config: ProtocolConfig,
    event_tx: Option<mpsc::Sender<ConnectionEvent>>,
    reconnect_attempts: u32,
    max_reconnect_attempts: u32,
}

impl Connection {
    pub fn new(remote_addr: SocketAddr, config: ProtocolConfig) -> Self {
        Self {
            state: ConnectionState::Disconnected,
            remote_addr,
            local_addr: None,
            config,
            event_tx: None,
            reconnect_attempts: 0,
            max_reconnect_attempts: 5,
        }
    }

    /// Set event channel for connection events
    pub fn set_event_channel(&mut self, tx: mpsc::Sender<ConnectionEvent>) {
        self.event_tx = Some(tx);
    }

    /// Connect to remote peer
    pub async fn connect(&mut self) -> Result<()> {
        self.state = ConnectionState::Connecting;

        // In production, this would use quinn to establish QUIC connection
        // For now, simulate connection
        tokio::time::sleep(Duration::from_millis(10)).await;

        self.state = ConnectionState::Connected;
        self.send_event(ConnectionEvent::Connected).await;

        Ok(())
    }

    /// Disconnect from remote peer
    pub async fn disconnect(&mut self) -> Result<()> {
        self.state = ConnectionState::Disconnected;
        self.send_event(ConnectionEvent::Disconnected).await;
        Ok(())
    }

    /// Handle connection loss and attempt reconnection
    pub async fn handle_disconnect(&mut self) -> Result<()> {
        if self.reconnect_attempts >= self.max_reconnect_attempts {
            self.state = ConnectionState::Failed;
            self.send_event(ConnectionEvent::Error("Max reconnection attempts exceeded".to_string())).await;
            return Err(ProtocolError::ConnectionError("Failed to reconnect".to_string()));
        }

        self.state = ConnectionState::Reconnecting;
        self.reconnect_attempts += 1;
        self.send_event(ConnectionEvent::Reconnecting).await;

        // Exponential backoff
        let delay = Duration::from_millis(100 * (1 << self.reconnect_attempts.min(6)));
        tokio::time::sleep(delay).await;

        self.connect().await
    }

    /// Get current state
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }

    /// Get remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    async fn send_event(&self, event: ConnectionEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event).await;
        }
    }
}

/// Connection pool for managing multiple connections
pub struct ConnectionPool {
    connections: Vec<Connection>,
    config: ProtocolConfig,
    max_connections: usize,
}

impl ConnectionPool {
    pub fn new(config: ProtocolConfig, max_connections: usize) -> Self {
        Self {
            connections: Vec::new(),
            config,
            max_connections,
        }
    }

    /// Add a new connection to the pool
    pub fn add(&mut self, remote_addr: SocketAddr) -> Result<&mut Connection> {
        if self.connections.len() >= self.max_connections {
            return Err(ProtocolError::ConnectionError("Pool full".to_string()));
        }

        let conn = Connection::new(remote_addr, self.config.clone());
        self.connections.push(conn);
        Ok(self.connections.last_mut().unwrap())
    }

    /// Get connection by address
    pub fn get(&mut self, addr: &SocketAddr) -> Option<&mut Connection> {
        self.connections.iter_mut().find(|c| &c.remote_addr == addr)
    }

    /// Remove disconnected connections
    pub fn cleanup(&mut self) {
        self.connections.retain(|c| c.state != ConnectionState::Failed);
    }

    /// Get all active connections
    pub fn active_connections(&self) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.is_connected())
            .collect()
    }
}

/// Connection metrics
#[derive(Debug, Clone, Default)]
pub struct ConnectionMetrics {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub rtt_ms: f32,
    pub congestion_window: u64,
    pub bytes_in_flight: u64,
}

impl ConnectionMetrics {
    pub fn bandwidth_estimate_kbps(&self) -> f32 {
        if self.rtt_ms > 0.0 {
            (self.congestion_window as f32 * 8.0) / self.rtt_ms
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let config = ProtocolConfig::default();
        let mut conn = Connection::new(addr, config);

        assert_eq!(conn.state(), ConnectionState::Disconnected);

        conn.connect().await.unwrap();
        assert_eq!(conn.state(), ConnectionState::Connected);
        assert!(conn.is_connected());
    }

    #[test]
    fn test_connection_pool() {
        let config = ProtocolConfig::default();
        let mut pool = ConnectionPool::new(config, 10);

        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        pool.add(addr).unwrap();

        assert!(pool.get(&addr).is_some());
    }
}
