// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Safety Alert Channel — Human Notification Bridge
//!
//! Provides a synchronous, bounded channel for the cognitive loop to emit
//! safety-critical events that require human attention. The host application
//! drains these alerts and forwards to desktop notifications, Slack, email,
//! or any external monitoring system.
//!
//! ## Design
//!
//! - **Synchronous**: Uses `std::sync::mpsc::SyncSender` with bounded capacity
//!   to avoid unbounded memory growth during sustained crisis.
//! - **Non-blocking**: `try_send()` never blocks the cognitive loop — if the
//!   channel is full, the alert is logged but dropped (better than blocking
//!   consciousness during emergency).
//! - **Feature-independent**: Works regardless of `safety-agents` or `sentinel`
//!   feature flags. The minimal safety baseline can also emit alerts.

/// Safety-critical event requiring human notification.
#[derive(Debug, Clone)]
pub struct SafetyAlert {
    /// What happened.
    pub kind: SafetyAlertKind,
    /// Human-readable description.
    pub message: String,
    /// Cycle number when the alert was generated.
    pub cycle: u64,
    /// Current consciousness level (Phi) at alert time.
    pub phi: f64,
    /// Urgency level for notification routing.
    pub urgency: AlertUrgency,
}

/// Classification of safety-critical events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyAlertKind {
    /// Motor output halted due to Red safety level (Phi < 0.1).
    MotorHalt,
    /// Motor output restricted to read-only due to Orange safety level.
    MotorReadOnly,
    /// Integrity system detected critical anomaly (canary failure, hash mismatch).
    IntegrityCritical,
    /// Sentinel detected threat exceeding Orange severity threshold.
    ThreatEscalation,
    /// Conductor (Mycelix bridge) non-responsive for >60 seconds.
    ConductorTimeout,
    /// Conductor disconnected — local-only governance mode.
    ConductorDisconnected,
    /// Sustained consciousness collapse (Phi < 0.1 for >5 cycles).
    ConsciousnessCollapse,
    /// Collective immune response activated (swarm-wide threat).
    ImmuneActivation,
}

/// Urgency level for notification routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertUrgency {
    /// Informational — log, low-priority notification.
    Info,
    /// Warning — desktop notification, non-blocking.
    Warning,
    /// Critical — immediate human attention required.
    Critical,
}

impl SafetyAlertKind {
    /// Default urgency for each alert kind.
    pub fn default_urgency(&self) -> AlertUrgency {
        match self {
            Self::MotorHalt | Self::IntegrityCritical | Self::ConsciousnessCollapse => {
                AlertUrgency::Critical
            }
            Self::MotorReadOnly
            | Self::ThreatEscalation
            | Self::ConductorTimeout
            | Self::ImmuneActivation => AlertUrgency::Warning,
            Self::ConductorDisconnected => AlertUrgency::Critical,
        }
    }
}

/// Capacity of the safety alert channel.
/// Small — these are rare, high-value events. If we're generating more than
/// 32 per drain cycle, something is catastrophically wrong.
pub const SAFETY_ALERT_CHANNEL_CAPACITY: usize = 32;

/// Emit a safety alert on the given sender. Never blocks.
/// If the channel is full, logs a warning and drops the alert.
pub fn emit_alert(
    tx: &std::sync::mpsc::SyncSender<SafetyAlert>,
    kind: SafetyAlertKind,
    message: impl Into<String>,
    cycle: u64,
    phi: f64,
) {
    let alert = SafetyAlert {
        urgency: kind.default_urgency(),
        kind,
        message: message.into(),
        cycle,
        phi,
    };
    if let Err(std::sync::mpsc::TrySendError::Full(_)) = tx.try_send(alert) {
        tracing::error!(
            "Safety alert channel FULL — alert dropped: {:?} at cycle {}",
            kind,
            cycle
        );
    }
    // Disconnected is fine — host may not have connected a receiver yet.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_urgency_ordering() {
        assert!(AlertUrgency::Critical > AlertUrgency::Warning);
        assert!(AlertUrgency::Warning > AlertUrgency::Info);
    }

    #[test]
    fn test_motor_halt_is_critical() {
        assert_eq!(
            SafetyAlertKind::MotorHalt.default_urgency(),
            AlertUrgency::Critical
        );
    }

    #[test]
    fn test_emit_alert_full_channel() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        // Fill the channel
        emit_alert(&tx, SafetyAlertKind::MotorHalt, "test1", 1, 0.05);
        // This should not panic — it drops the alert silently
        emit_alert(&tx, SafetyAlertKind::MotorHalt, "test2", 2, 0.05);
        // Only the first alert should be in the channel (capacity=1)
        assert!(rx.try_recv().is_ok(), "first alert should be received");
    }

    #[test]
    fn test_emit_alert_disconnected() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        drop(rx); // Disconnect
        // Should not panic — gracefully handles disconnected receiver
        emit_alert(&tx, SafetyAlertKind::MotorHalt, "test", 1, 0.05);
        // Success: no panic on disconnected channel
        assert!(true);
    }
}
