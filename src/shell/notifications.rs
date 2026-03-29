// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Desktop Notifications via D-Bus
//!
//! E1: Desktop Integration - Send notifications for consciousness events
//!
//! Uses the freedesktop.org Notifications specification (org.freedesktop.Notifications)
//! to display desktop notifications for:
//! - Command completion (success/failure)
//! - Phi threshold warnings
//! - Consciousness state changes
//! - Safety vetoes
//!
//! ## Usage
//!
//! ```rust,ignore
//! let notifier = DesktopNotifier::new().await?;
//! notifier.notify_command_complete("install nginx", true, 0.85).await?;
//! ```

/// Urgency level for notifications (freedesktop spec)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NotificationUrgency {
    Low = 0,
    Normal = 1,
    Critical = 2,
}

/// Notification category for semantic hints
#[derive(Debug, Clone, Copy)]
pub enum NotificationCategory {
    /// Command execution completed
    CommandComplete,
    /// Consciousness level warning
    PhiWarning,
    /// Safety veto occurred
    SafetyVeto,
    /// Connection status change
    ConnectionStatus,
    /// General information
    Info,
}

impl NotificationCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CommandComplete => "device",
            Self::PhiWarning => "device.warning",
            Self::SafetyVeto => "device.error",
            Self::ConnectionStatus => "network",
            Self::Info => "presence",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::CommandComplete => "dialog-information",
            Self::PhiWarning => "dialog-warning",
            Self::SafetyVeto => "dialog-error",
            Self::ConnectionStatus => "network-transmit-receive",
            Self::Info => "dialog-information",
        }
    }
}

/// Desktop notification configuration
#[derive(Debug, Clone)]
pub struct NotificationConfig {
    /// Application name shown in notifications
    pub app_name: String,
    /// Default timeout in milliseconds (0 = server default)
    pub timeout_ms: i32,
    /// Whether to include Phi in notifications
    pub show_phi: bool,
    /// Minimum Phi level to trigger warning notifications
    pub phi_warning_threshold: f64,
    /// Whether notifications are enabled
    pub enabled: bool,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            app_name: "Symthaea".to_string(),
            timeout_ms: 5000,
            show_phi: true,
            phi_warning_threshold: 0.4,
            enabled: true,
        }
    }
}

/// A pending notification to be sent
#[derive(Debug, Clone)]
pub struct Notification {
    pub summary: String,
    pub body: String,
    pub icon: String,
    pub urgency: NotificationUrgency,
    pub category: NotificationCategory,
    pub timeout_ms: i32,
    pub actions: Vec<(String, String)>,
}

impl Notification {
    /// Create a new notification
    pub fn new(summary: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            body: body.into(),
            icon: "dialog-information".to_string(),
            urgency: NotificationUrgency::Normal,
            category: NotificationCategory::Info,
            timeout_ms: 5000,
            actions: Vec::new(),
        }
    }

    /// Set the icon
    pub fn icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = icon.into();
        self
    }

    /// Set urgency level
    pub fn urgency(mut self, urgency: NotificationUrgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// Set category
    pub fn category(mut self, category: NotificationCategory) -> Self {
        self.icon = category.icon().to_string();
        self.category = category;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout_ms: i32) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Add an action button
    pub fn action(mut self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.actions.push((id.into(), label.into()));
        self
    }
}

/// Desktop notifier - sends notifications via D-Bus
///
/// This is a platform-agnostic interface. The actual D-Bus integration
/// is only compiled when the `notifications` feature is enabled.
pub struct DesktopNotifier {
    config: NotificationConfig,
    #[cfg(feature = "notifications")]
    connection: Option<zbus::Connection>,
}

impl DesktopNotifier {
    /// Create a new notifier
    #[cfg(feature = "notifications")]
    pub async fn new() -> Result<Self, NotificationError> {
        let connection = zbus::Connection::session()
            .await
            .map_err(|e| NotificationError::ConnectionFailed(e.to_string()))?;

        Ok(Self {
            config: NotificationConfig::default(),
            connection: Some(connection),
        })
    }

    #[cfg(not(feature = "notifications"))]
    pub async fn new() -> Result<Self, NotificationError> {
        Ok(Self {
            config: NotificationConfig::default(),
        })
    }

    /// Create a notifier with custom config
    pub async fn with_config(config: NotificationConfig) -> Result<Self, NotificationError> {
        let mut notifier = Self::new().await?;
        notifier.config = config;
        Ok(notifier)
    }

    /// Create a disabled notifier (no-op)
    pub fn disabled() -> Self {
        Self {
            config: NotificationConfig {
                enabled: false,
                ..Default::default()
            },
            #[cfg(feature = "notifications")]
            connection: None,
        }
    }

    /// Check if notifications are available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "notifications")]
        {
            self.config.enabled && self.connection.is_some()
        }
        #[cfg(not(feature = "notifications"))]
        {
            false
        }
    }

    /// Send a notification
    #[cfg(feature = "notifications")]
    pub async fn send(&self, notification: &Notification) -> Result<u32, NotificationError> {
        if !self.config.enabled {
            return Ok(0);
        }

        let conn = self
            .connection
            .as_ref()
            .ok_or(NotificationError::NotConnected)?;

        // Build hints
        let mut hints: std::collections::HashMap<&str, zbus::zvariant::Value> =
            std::collections::HashMap::new();
        hints.insert(
            "urgency",
            zbus::zvariant::Value::U8(notification.urgency as u8),
        );
        hints.insert(
            "category",
            zbus::zvariant::Value::Str(notification.category.as_str().into()),
        );

        // Flatten actions
        let actions: Vec<&str> = notification
            .actions
            .iter()
            .flat_map(|(id, label)| vec![id.as_str(), label.as_str()])
            .collect();

        // Call org.freedesktop.Notifications.Notify
        let proxy = zbus::Proxy::new(
            conn,
            "org.freedesktop.Notifications",
            "/org/freedesktop/Notifications",
            "org.freedesktop.Notifications",
        )
        .await
        .map_err(|e| NotificationError::ProxyFailed(e.to_string()))?;

        let notification_id: u32 = proxy
            .call(
                "Notify",
                &(
                    &self.config.app_name,
                    0u32, // replaces_id
                    &notification.icon,
                    &notification.summary,
                    &notification.body,
                    &actions,
                    &hints,
                    notification.timeout_ms,
                ),
            )
            .await
            .map_err(|e| NotificationError::SendFailed(e.to_string()))?;

        Ok(notification_id)
    }

    #[cfg(not(feature = "notifications"))]
    pub async fn send(&self, _notification: &Notification) -> Result<u32, NotificationError> {
        if !self.config.enabled {
            return Ok(0);
        }
        // No-op when D-Bus is not available
        tracing::debug!("Notifications disabled (compiled without 'notifications' feature)");
        Ok(0)
    }

    /// Notify command completion
    pub async fn notify_command_complete(
        &self,
        command: &str,
        success: bool,
        phi: f64,
    ) -> Result<u32, NotificationError> {
        let (summary, urgency) = if success {
            ("Command Completed", NotificationUrgency::Low)
        } else {
            ("Command Failed", NotificationUrgency::Normal)
        };

        let body = if self.config.show_phi {
            format!("{command}\nPhi: {phi:.2}")
        } else {
            command.to_string()
        };

        let notification = Notification::new(summary, body)
            .category(NotificationCategory::CommandComplete)
            .urgency(urgency);

        self.send(&notification).await
    }

    /// Notify Phi threshold warning
    pub async fn notify_phi_warning(
        &self,
        current_phi: f64,
        required_phi: f64,
    ) -> Result<u32, NotificationError> {
        if current_phi >= self.config.phi_warning_threshold {
            return Ok(0); // No warning needed
        }

        let notification = Notification::new(
            "Low Consciousness Level",
            format!(
                "Current Phi ({current_phi:.2}) is below required level ({required_phi:.2})\nConsider centering before proceeding"
            ),
        )
        .category(NotificationCategory::PhiWarning)
        .urgency(NotificationUrgency::Normal)
        .action("center", "Center Now")
        .action("dismiss", "Dismiss");

        self.send(&notification).await
    }

    /// Notify safety veto
    pub async fn notify_safety_veto(
        &self,
        command: &str,
        reason: &str,
    ) -> Result<u32, NotificationError> {
        let notification = Notification::new(
            "Command Vetoed",
            format!("Command: {command}\nReason: {reason}"),
        )
        .category(NotificationCategory::SafetyVeto)
        .urgency(NotificationUrgency::Critical)
        .timeout(0); // Don't auto-dismiss

        self.send(&notification).await
    }

    /// Notify connection status change
    pub async fn notify_connection(
        &self,
        connected: bool,
        service_name: &str,
    ) -> Result<u32, NotificationError> {
        let (summary, body) = if connected {
            ("Connected", format!("Connected to {service_name}"))
        } else {
            ("Disconnected", format!("Lost connection to {service_name}"))
        };

        let notification = Notification::new(summary, body)
            .category(NotificationCategory::ConnectionStatus)
            .urgency(if connected {
                NotificationUrgency::Low
            } else {
                NotificationUrgency::Normal
            });

        self.send(&notification).await
    }

    /// Notify consciousness state change
    pub async fn notify_consciousness_change(
        &self,
        is_conscious: bool,
        phi: f64,
        coherence: f64,
    ) -> Result<u32, NotificationError> {
        let (summary, urgency) = if is_conscious {
            ("Consciousness Awakened", NotificationUrgency::Low)
        } else {
            ("Consciousness Dormant", NotificationUrgency::Normal)
        };

        let body = format!("Phi: {:.2}\nCoherence: {:.0}%", phi, coherence * 100.0);

        let notification = Notification::new(summary, body)
            .category(NotificationCategory::Info)
            .urgency(urgency);

        self.send(&notification).await
    }
}

/// Notification errors
#[derive(Debug, Clone)]
pub enum NotificationError {
    /// D-Bus connection failed
    ConnectionFailed(String),
    /// Not connected to D-Bus
    NotConnected,
    /// Failed to create proxy
    ProxyFailed(String),
    /// Failed to send notification
    SendFailed(String),
    /// Notifications disabled
    Disabled,
}

impl std::fmt::Display for NotificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(e) => write!(f, "D-Bus connection failed: {e}"),
            Self::NotConnected => write!(f, "Not connected to D-Bus"),
            Self::ProxyFailed(e) => write!(f, "Failed to create notification proxy: {e}"),
            Self::SendFailed(e) => write!(f, "Failed to send notification: {e}"),
            Self::Disabled => write!(f, "Notifications are disabled"),
        }
    }
}

impl std::error::Error for NotificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_builder() {
        let notif = Notification::new("Test", "Body")
            .urgency(NotificationUrgency::Critical)
            .category(NotificationCategory::SafetyVeto)
            .timeout(10000)
            .action("ok", "OK")
            .action("cancel", "Cancel");

        assert_eq!(notif.summary, "Test");
        assert_eq!(notif.body, "Body");
        assert_eq!(notif.urgency, NotificationUrgency::Critical);
        assert_eq!(notif.timeout_ms, 10000);
        assert_eq!(notif.actions.len(), 2);
    }

    #[test]
    fn test_notification_config_default() {
        let config = NotificationConfig::default();
        assert_eq!(config.app_name, "Symthaea");
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.show_phi);
        assert!(config.enabled);
    }

    #[test]
    fn test_disabled_notifier() {
        let notifier = DesktopNotifier::disabled();
        assert!(!notifier.is_available());
    }
}
