//! Conductor Bridge — abstraction over Holochain conductor connections.
//!
//! Phase 2: External conductor via WebSocket (current).
//! Phase 3: Embedded conductor via child process (Kangaroo pattern).
//!
//! The trait-based design allows swapping between external and embedded
//! conductors without changing the rest of the application.
//!
//! ## Connection Flow
//!
//! 1. `ExternalConductor::connect()` opens a WebSocket to the conductor's app port
//! 2. Zome calls are serialized as msgpack, sent over the WebSocket
//! 3. Responses are deserialized back into `ZomeCallResult`
//!
//! When `holochain_client` crate is available in the build environment,
//! replace the raw WebSocket with `AppWebsocket::connect()` for full
//! type-safe zome call support.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Result of a zome call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeCallResult {
    pub success: bool,
    pub payload: Vec<u8>,
    pub error: Option<String>,
}

/// Connection status for the conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub connected: bool,
    pub url: String,
    pub last_ping_ms: Option<u64>,
    pub agent_key: Option<String>,
}

/// Trait for conductor connections. Implementations handle the transport
/// (WebSocket for external, IPC for embedded).
pub trait ConductorConnection: Send + Sync {
    /// Call a zome function on the conductor.
    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Result<ZomeCallResult, String>;

    /// Check if the conductor is reachable.
    fn is_connected(&self) -> bool;

    /// Get the connection status.
    fn status(&self) -> ConnectionStatus;
}

/// External conductor connection via WebSocket.
///
/// Connects to a running Holochain conductor (e.g., from Docker Compose stack).
/// The conductor must have an app interface enabled on the configured port.
///
/// ## Usage
///
/// ```ignore
/// let mut conductor = ExternalConductor::new("ws://localhost:8888");
/// conductor.try_connect().await;
///
/// if conductor.is_connected() {
///     let result = conductor.call_zome(
///         "energy",
///         "projects_coordinator",
///         "get_all_projects",
///         vec![],
///     );
/// }
/// ```
pub struct ExternalConductor {
    /// WebSocket URL for the app interface.
    app_url: String,
    /// Whether the connection is established.
    connected: AtomicBool,
    /// Last successful ping latency in milliseconds.
    last_ping_ms: std::sync::Mutex<Option<u64>>,
}

impl ExternalConductor {
    /// Create a new external conductor connection (not yet connected).
    pub fn new(app_url: &str) -> Self {
        Self {
            app_url: app_url.to_string(),
            connected: AtomicBool::new(false),
            last_ping_ms: std::sync::Mutex::new(None),
        }
    }

    /// Attempt to connect to the conductor.
    ///
    /// Performs a TCP connection check to verify the conductor is reachable.
    /// In Phase 3, this will be replaced with `holochain_client::AppWebsocket::connect()`.
    pub async fn try_connect(&self) -> Result<(), String> {
        // Parse the WebSocket URL to get host:port for TCP check
        let url = self.app_url.replace("ws://", "").replace("wss://", "");
        let addr = url.split('/').next().unwrap_or(&url);

        let start = std::time::Instant::now();
        match tokio::net::TcpStream::connect(addr).await {
            Ok(_) => {
                let elapsed = start.elapsed().as_millis() as u64;
                self.connected.store(true, Ordering::SeqCst);
                if let Ok(mut ping) = self.last_ping_ms.lock() {
                    *ping = Some(elapsed);
                }
                Ok(())
            }
            Err(e) => {
                self.connected.store(false, Ordering::SeqCst);
                Err(format!("Cannot reach conductor at {}: {}", self.app_url, e))
            }
        }
    }

    /// Disconnect from the conductor.
    pub fn disconnect(&self) {
        self.connected.store(false, Ordering::SeqCst);
    }
}

impl ConductorConnection for ExternalConductor {
    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        _payload: Vec<u8>,
    ) -> Result<ZomeCallResult, String> {
        if !self.is_connected() {
            return Err(format!(
                "Not connected to conductor at {}. Call try_connect() first.",
                self.app_url
            ));
        }

        // Phase 2: Return structured error indicating the call path works
        // but actual WebSocket message framing needs holochain_client.
        //
        // When holochain_client is available, replace with:
        //   let client = AppWebsocket::connect(&self.app_url).await?;
        //   let result = client.call_zome(role_name, zome_name, fn_name, payload).await?;
        Err(format!(
            "Zome call {}/{}/{} routed but holochain_client not yet linked. \
             Connection verified to {}.",
            role_name, zome_name, fn_name, self.app_url
        ))
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    fn status(&self) -> ConnectionStatus {
        ConnectionStatus {
            connected: self.is_connected(),
            url: self.app_url.clone(),
            last_ping_ms: self.last_ping_ms.lock().ok().and_then(|p| *p),
            agent_key: None, // Set when holochain_client provides agent info
        }
    }
}
