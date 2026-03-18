//! Core connection manager for the Holochain conductor.
//!
//! Manages a persistent WebSocket connection to a Holochain conductor,
//! with automatic reconnection and exponential backoff on failure.

use crate::error::{BridgeError, Result};
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ExternIO, ZomeCallTarget};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Base backoff duration in seconds after a conductor failure.
const BACKOFF_BASE_SECS: u64 = 5;

/// Maximum backoff duration in seconds (5 minutes).
const BACKOFF_MAX_SECS: u64 = 300;

/// After this many consecutive failures, log at error level with diagnostic hints.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Manages a WebSocket connection to a Holochain conductor.
///
/// Provides connection caching, automatic reconnection, and exponential
/// backoff after consecutive failures. Thread-safe via interior mutability.
pub struct HolochainConductor {
    /// WebSocket URL of the conductor (e.g. "ws://localhost:8888").
    conductor_url: String,
    /// hApp identifier (e.g. "mycelix-unified").
    app_id: String,
    /// Cached WebSocket connection, guarded by async mutex.
    ws: Mutex<Option<AppWebsocket>>,
    /// Tracks consecutive connection/call failures for backoff.
    consecutive_failures: AtomicU32,
}

impl HolochainConductor {
    /// Create a new conductor handle. Does NOT connect immediately.
    ///
    /// # Arguments
    /// * `conductor_url` - WebSocket URL (e.g. "ws://localhost:8888")
    /// * `app_id` - hApp identifier for logging/diagnostics
    pub fn new(conductor_url: impl Into<String>, app_id: impl Into<String>) -> Self {
        Self {
            conductor_url: conductor_url.into(),
            app_id: app_id.into(),
            ws: Mutex::new(None),
            consecutive_failures: AtomicU32::new(0),
        }
    }

    /// Connect to the conductor. Reads `MYCELIX_APP_TOKEN` from env for auth.
    ///
    /// If already connected, this is a no-op. On failure, the error is logged
    /// and returned — the caller should retry via `ensure_connected()`.
    pub async fn connect(&self) -> Result<()> {
        let mut guard = self.ws.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        // Check backoff — if we've failed too many times, enforce a delay
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures > 0 {
            let backoff_secs = Self::compute_backoff(failures);
            tracing::debug!(
                app_id = %self.app_id,
                failures,
                backoff_secs,
                "Backoff active before reconnect attempt"
            );
            tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
        }

        let token: Vec<u8> = std::env::var("MYCELIX_APP_TOKEN")
            .unwrap_or_default()
            .into_bytes();
        let signer: Arc<dyn AgentSigner + Send + Sync> =
            Arc::new(ClientAgentSigner::default());

        tracing::info!(
            conductor_url = %self.conductor_url,
            app_id = %self.app_id,
            "Connecting to Holochain conductor"
        );

        match AppWebsocket::connect(&self.conductor_url, token, signer).await {
            Ok(websocket) => {
                self.consecutive_failures.store(0, Ordering::Relaxed);
                *guard = Some(websocket);
                tracing::info!(
                    app_id = %self.app_id,
                    "Connected to conductor"
                );
                Ok(())
            }
            Err(e) => {
                let count = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                let msg = format!("{e}");
                if count >= MAX_CONSECUTIVE_FAILURES {
                    tracing::error!(
                        app_id = %self.app_id,
                        failures = count,
                        "Conductor unreachable after {count} attempts: {msg}. \
                         Check conductor is running and MYCELIX_APP_TOKEN is set."
                    );
                } else {
                    tracing::warn!(
                        app_id = %self.app_id,
                        failures = count,
                        "Conductor connection failed: {msg}"
                    );
                }
                Err(BridgeError::ConnectionFailed(msg))
            }
        }
    }

    /// Disconnect from the conductor, dropping the cached WebSocket.
    pub async fn disconnect(&self) {
        let mut guard = self.ws.lock().await;
        if guard.take().is_some() {
            tracing::info!(app_id = %self.app_id, "Disconnected from conductor");
        }
    }

    /// Ensure we have an active connection, reconnecting if needed.
    pub async fn ensure_connected(&self) -> Result<()> {
        {
            let guard = self.ws.lock().await;
            if guard.is_some() {
                return Ok(());
            }
        }
        self.connect().await
    }

    /// Call a zome function on the conductor.
    ///
    /// Encodes the payload via `ExternIO`, dispatches the call, and decodes
    /// the response as `serde_json::Value`. On failure, the cached connection
    /// is dropped so the next call triggers a reconnect.
    ///
    /// # Arguments
    /// * `role` - Role name in the hApp manifest (e.g. "governance", "finance")
    /// * `zome` - Zome name (e.g. "agora", "finance_bridge")
    /// * `fn_name` - Function name in the zome (e.g. "create_proposal")
    /// * `payload` - JSON payload to encode as `ExternIO`
    pub async fn call_zome(
        &self,
        role: &str,
        zome: &str,
        fn_name: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.ensure_connected().await?;

        let input = ExternIO::encode(payload)
            .map_err(|e| BridgeError::SerializationError(format!("{e}")))?;

        let mut guard = self.ws.lock().await;
        let ws = guard.as_ref().ok_or(BridgeError::NotConnected)?;

        tracing::debug!(
            app_id = %self.app_id,
            role,
            zome,
            fn_name,
            "Calling zome function"
        );

        let response = ws
            .call_zome(
                ZomeCallTarget::RoleName(role.to_string().into()),
                zome.into(),
                fn_name.into(),
                input,
            )
            .await;

        match response {
            Ok(result) => {
                self.consecutive_failures.store(0, Ordering::Relaxed);
                match result.decode::<serde_json::Value>() {
                    Ok(data) => Ok(data),
                    Err(e) => Err(BridgeError::DeserializationError(format!("{e}"))),
                }
            }
            Err(e) => {
                let count = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                // Drop the connection so next call reconnects
                *guard = None;
                let msg = format!("{e}");
                tracing::warn!(
                    app_id = %self.app_id,
                    role,
                    zome,
                    fn_name,
                    failures = count,
                    "Zome call failed: {msg}"
                );
                Err(BridgeError::ZomeCallFailed(msg))
            }
        }
    }

    /// Returns `true` if we have a cached WebSocket connection.
    ///
    /// Note: this only checks the cache, not whether the connection is
    /// still alive. A stale connection will be detected on the next
    /// `call_zome()` and automatically dropped.
    pub fn is_connected(&self) -> bool {
        // Use try_lock to avoid blocking — if locked, someone is using it,
        // which means we are connected (or in the process of connecting).
        match self.ws.try_lock() {
            Ok(guard) => guard.is_some(),
            Err(_) => true,
        }
    }

    /// Returns the current consecutive failure count.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Returns the conductor URL.
    pub fn conductor_url(&self) -> &str {
        &self.conductor_url
    }

    /// Returns the app ID.
    pub fn app_id(&self) -> &str {
        &self.app_id
    }

    /// Compute exponential backoff duration from failure count.
    /// Formula: min(BACKOFF_BASE_SECS * 2^failures, BACKOFF_MAX_SECS)
    fn compute_backoff(failures: u32) -> u64 {
        let exponent = failures.min(6); // cap shift to avoid overflow
        let backoff = BACKOFF_BASE_SECS * (1u64 << exponent);
        backoff.min(BACKOFF_MAX_SECS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_conductor() {
        let c = HolochainConductor::new("ws://localhost:8888", "mycelix-unified");
        assert_eq!(c.conductor_url(), "ws://localhost:8888");
        assert_eq!(c.app_id(), "mycelix-unified");
        assert!(!c.is_connected());
        assert_eq!(c.consecutive_failures(), 0);
    }

    #[test]
    fn test_backoff_calculation() {
        // 0 failures → 5s
        assert_eq!(HolochainConductor::compute_backoff(0), 5);
        // 1 failure → 10s
        assert_eq!(HolochainConductor::compute_backoff(1), 10);
        // 2 failures → 20s
        assert_eq!(HolochainConductor::compute_backoff(2), 20);
        // 3 failures → 40s
        assert_eq!(HolochainConductor::compute_backoff(3), 40);
        // 6 failures → 320 → capped at 300s
        assert_eq!(HolochainConductor::compute_backoff(6), 300);
        // 10 failures → still capped at 6 bits → 300s
        assert_eq!(HolochainConductor::compute_backoff(10), 300);
    }

    #[test]
    fn test_backoff_constants() {
        assert_eq!(BACKOFF_BASE_SECS, 5);
        assert_eq!(BACKOFF_MAX_SECS, 300);
        assert_eq!(MAX_CONSECUTIVE_FAILURES, 5);
        assert!(BACKOFF_BASE_SECS < BACKOFF_MAX_SECS);
    }
}
