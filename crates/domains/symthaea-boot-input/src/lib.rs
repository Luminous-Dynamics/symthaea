// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ephemeral input adapter for the early Spore boot renderer.
//!
//! This library recognizes only F1, F2, and Escape key-down events. It does not
//! grab devices, retain device names, retain ordinary key events, or perform any
//! presentation/VT action itself. Its intended lifetime is exactly the renderer's
//! early-boot lifetime so global keyboard observation does not survive into login.

#![forbid(unsafe_code)]

use std::io;
use std::time::{Duration, Instant};

use evdev::{Device, EventSummary, KeyCode};
use symthaea_boot_control::{PresentationMode, PresentationRequest};

const RESCAN_INTERVAL: Duration = Duration::from_secs(5);
const MAX_REQUESTS_PER_POLL: usize = 16;

pub struct BootInputAdapter {
    devices: Vec<Device>,
    last_scan: Instant,
    next_sequence: u64,
}

impl Default for BootInputAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl BootInputAdapter {
    pub fn new() -> Self {
        let mut adapter = Self {
            devices: Vec::new(),
            last_scan: Instant::now()
                .checked_sub(RESCAN_INTERVAL)
                .unwrap_or_else(Instant::now),
            next_sequence: 1,
        };
        adapter.rescan();
        adapter
    }

    /// Poll currently readable input devices and return only recognized
    /// presentation requests. All other events are dropped immediately.
    pub fn poll(&mut self) -> Vec<PresentationRequest> {
        if self.devices.is_empty() || self.last_scan.elapsed() >= RESCAN_INTERVAL {
            self.rescan();
        }

        let mut requested = Vec::new();
        for device in &mut self.devices {
            loop {
                match device.fetch_events() {
                    Ok(events) => {
                        for event in events {
                            if let Some(mode) = mode_from_event(event.destructure()) {
                                if requested.len() < MAX_REQUESTS_PER_POLL {
                                    requested.push(mode);
                                }
                            }
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }

        requested
            .into_iter()
            .map(|mode| {
                let sequence = self.next_sequence;
                self.next_sequence = self.next_sequence.saturating_add(1).max(1);
                PresentationRequest::new(sequence, mode)
            })
            .collect()
    }

    fn rescan(&mut self) {
        let mut devices = Vec::new();
        for (_path, device) in evdev::enumerate() {
            let supports_control_keys = device.supported_keys().is_some_and(|keys| {
                keys.contains(KeyCode::KEY_F1)
                    && keys.contains(KeyCode::KEY_F2)
                    && keys.contains(KeyCode::KEY_ESC)
            });
            if !supports_control_keys {
                continue;
            }
            if device.set_nonblocking(true).is_ok() {
                devices.push(device);
            }
        }
        self.devices = devices;
        self.last_scan = Instant::now();
    }
}

fn mode_from_event(event: EventSummary) -> Option<PresentationMode> {
    match event {
        EventSummary::Key(_, KeyCode::KEY_F1, 1) => Some(PresentationMode::Ambient),
        EventSummary::Key(_, KeyCode::KEY_F2, 1) => Some(PresentationMode::Diagnostics),
        EventSummary::Key(_, KeyCode::KEY_ESC, 1) => Some(PresentationMode::RawLogs),
        // Releases (0), repeats (2), and every unrelated key/event disappear here.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evdev::{EventType, InputEvent};

    fn key(code: KeyCode, value: i32) -> EventSummary {
        // EventSummary owns the small event wrapper/value; this helper constructs
        // only synthetic unit-test input and never touches /dev/input.
        InputEvent::new(EventType::KEY.0, code.code(), value).destructure()
    }

    #[test]
    fn only_control_key_down_events_map_to_modes() {
        assert_eq!(mode_from_event(key(KeyCode::KEY_F1, 1)), Some(PresentationMode::Ambient));
        assert_eq!(
            mode_from_event(key(KeyCode::KEY_F2, 1)),
            Some(PresentationMode::Diagnostics)
        );
        assert_eq!(
            mode_from_event(key(KeyCode::KEY_ESC, 1)),
            Some(PresentationMode::RawLogs)
        );
        assert_eq!(mode_from_event(key(KeyCode::KEY_A, 1)), None);
        assert_eq!(mode_from_event(key(KeyCode::KEY_F2, 0)), None);
        assert_eq!(mode_from_event(key(KeyCode::KEY_F2, 2)), None);
    }
}
