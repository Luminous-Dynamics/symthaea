// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Owner-backed core for the daemon's cognition-related protocol handlers.
//!
//! Transport/auth/voice remain outside this type. The purpose of this layer is to
//! make the daemon's cognitive request handling independent of direct mutable
//! `Symthaea` access while preserving operational counters and save-path policy.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use symthaea_service_runtime::{ProcessOrigin, ServiceCounters};

use crate::host::{ServiceHostCommandError, ServiceHostReadError, ServiceRuntimeHost};
use crate::read_wire::MeasuredIntrospectionV2;
use crate::wire::{
    LegacyIntrospectionPolicy, ServiceWireOutcome, ServiceWireResponse,
    legacy_introspection_response, partnership_response, status_response,
};

/// Daemon operational state that is not cognitive state.
///
/// These counters stay outside the single-owner cognition runtime because request
/// accounting and process uptime describe the service shell, not Symthaea's mind.
pub struct ServiceProtocolCore {
    started_at: Instant,
    requests_processed: AtomicU64,
    sleep_cycles: AtomicU32,
    state_file: Option<PathBuf>,
}

impl ServiceProtocolCore {
    pub fn new(state_file: Option<PathBuf>) -> Self {
        Self {
            started_at: Instant::now(),
            requests_processed: AtomicU64::new(0),
            sleep_cycles: AtomicU32::new(0),
            state_file,
        }
    }

    /// Record one accepted protocol request. The transport should call this once
    /// before dispatch, including for non-cognitive requests such as ping/protocol.
    /// The counter saturates instead of wrapping in a very long-lived daemon.
    pub fn record_request(&self) -> u64 {
        let previous = self
            .requests_processed
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(1))
            })
            .expect("saturating update closure always returns Some");
        previous.saturating_add(1)
    }

    pub fn counters(&self) -> ServiceCounters {
        ServiceCounters {
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            sleep_cycles: self.sleep_cycles.load(Ordering::Relaxed),
        }
    }

    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn state_file(&self) -> Option<&PathBuf> {
        self.state_file.as_ref()
    }

    /// Preserve the daemon's save-path precedence exactly:
    /// explicit request path -> configured state file -> `symthaea-state.bin`.
    pub fn resolve_save_path(&self, requested: Option<String>) -> PathBuf {
        requested
            .map(PathBuf::from)
            .or_else(|| self.state_file.clone())
            .unwrap_or_else(|| PathBuf::from("symthaea-state.bin"))
    }

    /// Lock-free/read-plane status projection. This never enters the owner mailbox.
    pub fn status(
        &self,
        host: &ServiceRuntimeHost,
    ) -> Result<ServiceWireResponse, ServiceHostReadError> {
        Ok(status_response(host.status()?, self.counters(), self.uptime()))
    }

    /// Measurement-only introspection for protocol-v2/new clients. Runtime activity
    /// and snapshot provenance remain explicit so a responsive client can distinguish
    /// current processing from the freshness of the last completed cognitive state.
    pub fn introspection_v2(
        &self,
        host: &ServiceRuntimeHost,
    ) -> Result<MeasuredIntrospectionV2, ServiceHostReadError> {
        Ok(MeasuredIntrospectionV2::from(host.introspection()?))
    }

    /// Explicit daemon-v1 compatibility projection. The heuristics remain a
    /// transport policy, not runtime state.
    pub fn introspection_legacy(
        &self,
        host: &ServiceRuntimeHost,
        policy: LegacyIntrospectionPolicy,
    ) -> Result<ServiceWireResponse, ServiceHostReadError> {
        Ok(legacy_introspection_response(host.introspection()?, policy))
    }

    /// Lock-free partnership projection from the last completed snapshot.
    pub fn partnership(
        &self,
        host: &ServiceRuntimeHost,
    ) -> Result<ServiceWireResponse, ServiceHostReadError> {
        Ok(partnership_response(host.partnership()?))
    }

    /// Owner-backed text query. `origin` lets voice/semantic-ear paths reuse the
    /// same bounded cognition owner without pretending every turn came from text UI.
    pub async fn query_with_origin(
        &self,
        host: &ServiceRuntimeHost,
        content: impl Into<String>,
        origin: ProcessOrigin,
    ) -> Result<ServiceWireOutcome, ServiceHostCommandError> {
        let started = Instant::now();
        let reply = host.query(content, origin).await?;
        Ok(ServiceWireOutcome::from_query(reply, started.elapsed()))
    }

    pub async fn query(
        &self,
        host: &ServiceRuntimeHost,
        content: impl Into<String>,
    ) -> Result<ServiceWireOutcome, ServiceHostCommandError> {
        self.query_with_origin(host, content, ProcessOrigin::ServiceQuery)
            .await
    }

    pub async fn sleep(
        &self,
        host: &ServiceRuntimeHost,
    ) -> Result<ServiceWireOutcome, ServiceHostCommandError> {
        let reply = host.sleep().await?;
        if reply.result.is_ok() {
            let _ = self
                .sleep_cycles
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_add(1))
                });
        }
        Ok(ServiceWireOutcome::from_sleep(reply))
    }

    pub async fn save(
        &self,
        host: &ServiceRuntimeHost,
        requested_path: Option<String>,
    ) -> Result<ServiceWireOutcome, ServiceHostCommandError> {
        let path = self.resolve_save_path(requested_path);
        let reply = host.save(path).await?;
        Ok(ServiceWireOutcome::from_save(reply))
    }

    /// Preserve shutdown's configured-state-file behavior. Persistence errors remain
    /// available in `ServiceWireDiagnostics` while the legacy wire response stays an
    /// acknowledgment.
    pub async fn shutdown(
        &self,
        host: &ServiceRuntimeHost,
    ) -> Result<ServiceWireOutcome, ServiceHostCommandError> {
        let reply = host.shutdown_persist(self.state_file.clone()).await?;
        Ok(ServiceWireOutcome::from_shutdown(reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_counters_are_separate_monotonic_and_saturating() {
        let core = ServiceProtocolCore::new(None);
        assert_eq!(core.counters(), ServiceCounters::default());
        assert_eq!(core.record_request(), 1);
        assert_eq!(core.record_request(), 2);
        assert_eq!(core.counters().requests_processed, 2);
        assert_eq!(core.counters().sleep_cycles, 0);

        core.requests_processed.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(core.record_request(), u64::MAX);
        assert_eq!(core.counters().requests_processed, u64::MAX);
    }

    #[test]
    fn save_path_precedence_matches_legacy_daemon() {
        let core = ServiceProtocolCore::new(Some(PathBuf::from("configured.bin")));
        assert_eq!(
            core.resolve_save_path(Some("explicit.bin".into())),
            PathBuf::from("explicit.bin")
        );
        assert_eq!(core.resolve_save_path(None), PathBuf::from("configured.bin"));

        let fallback = ServiceProtocolCore::new(None);
        assert_eq!(
            fallback.resolve_save_path(None),
            PathBuf::from("symthaea-state.bin")
        );
    }
}
