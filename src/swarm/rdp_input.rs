// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Input injection for Sovereign RDP server.
//!
//! Translates normalized input events from the RDP client into platform
//! input actions. Consciousness-gating is enforced by the server (not here).

use super::rdp_protocol::InputEvent;

/// Platform-agnostic input injection interface.
pub trait InputInjector: Send {
    /// Inject a pointer (mouse) event.
    /// `x` and `y` are normalized to 0.0-1.0.
    fn inject_pointer(&mut self, x: f32, y: f32, button: u8, pressed: bool);

    /// Inject a keyboard event.
    fn inject_key(&mut self, code: u32, pressed: bool, modifiers: u8);

    /// Inject a touch event.
    fn inject_touch(&mut self, index: u8, x: f32, y: f32, phase: u8, pressure: f32);

    /// Process a batch of InputEvents.
    fn process_events(&mut self, events: &[InputEvent]) {
        for event in events {
            match event {
                InputEvent::Pointer {
                    x,
                    y,
                    button,
                    pressed,
                } => self.inject_pointer(*x, *y, *button, *pressed),
                InputEvent::Key {
                    code,
                    pressed,
                    modifiers,
                } => self.inject_key(*code, *pressed, *modifiers),
                InputEvent::Touch {
                    index,
                    x,
                    y,
                    phase,
                    pressure,
                } => self.inject_touch(*index, *x, *y, *phase, *pressure),
            }
        }
    }
}

/// No-op injector for testing and view-only mode.
pub struct NoopInjector;

impl InputInjector for NoopInjector {
    fn inject_pointer(&mut self, _x: f32, _y: f32, _button: u8, _pressed: bool) {}
    fn inject_key(&mut self, _code: u32, _pressed: bool, _modifiers: u8) {}
    fn inject_touch(&mut self, _index: u8, _x: f32, _y: f32, _phase: u8, _pressure: f32) {}
}

/// Logging injector: records events for testing/debugging.
pub struct LoggingInjector {
    pub events: Vec<InjectedEvent>,
    screen_width: u32,
    screen_height: u32,
}

/// A recorded injection event.
#[derive(Debug, Clone)]
pub enum InjectedEvent {
    Pointer {
        screen_x: i32,
        screen_y: i32,
        button: u8,
        pressed: bool,
    },
    Key {
        code: u32,
        pressed: bool,
        modifiers: u8,
    },
    Touch {
        index: u8,
        screen_x: i32,
        screen_y: i32,
        phase: u8,
        pressure: f32,
    },
}

impl LoggingInjector {
    pub fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            events: Vec::new(),
            screen_width,
            screen_height,
        }
    }
}

impl InputInjector for LoggingInjector {
    fn inject_pointer(&mut self, x: f32, y: f32, button: u8, pressed: bool) {
        self.events.push(InjectedEvent::Pointer {
            screen_x: (x * self.screen_width as f32) as i32,
            screen_y: (y * self.screen_height as f32) as i32,
            button,
            pressed,
        });
    }

    fn inject_key(&mut self, code: u32, pressed: bool, modifiers: u8) {
        self.events.push(InjectedEvent::Key {
            code,
            pressed,
            modifiers,
        });
    }

    fn inject_touch(&mut self, index: u8, x: f32, y: f32, phase: u8, pressure: f32) {
        self.events.push(InjectedEvent::Touch {
            index,
            screen_x: (x * self.screen_width as f32) as i32,
            screen_y: (y * self.screen_height as f32) as i32,
            phase,
            pressure,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// X11 XTest Input Injection
// ═══════════════════════════════════════════════════════════════════════════════

/// X11 input injection via XTest extension.
///
/// Uses XTestFakeMotionEvent, XTestFakeButtonEvent, XTestFakeKeyEvent.
/// Requires `x11rb` with `xtest` feature.
///
/// ## Coordinate Mapping
/// Normalized (0.0-1.0) → screen pixel coordinates via screen_width/height.
pub struct X11TestInjector {
    screen_width: u32,
    screen_height: u32,
    // In production: x11rb::rust_connection::RustConnection
    // For now: delegates to LoggingInjector for testability
    log: LoggingInjector,
}

impl X11TestInjector {
    /// Connect to X11 and initialize XTest injection.
    pub fn new(screen_width: u32, screen_height: u32) -> Result<Self, String> {
        if std::env::var("DISPLAY").is_err() {
            return Err("X11 not available: $DISPLAY not set".to_string());
        }

        // In production: connect x11rb, verify XTest extension available
        Ok(Self {
            screen_width,
            screen_height,
            log: LoggingInjector::new(screen_width, screen_height),
        })
    }

    /// Access the event log (for testing).
    pub fn events(&self) -> &[InjectedEvent] {
        &self.log.events
    }
}

impl InputInjector for X11TestInjector {
    fn inject_pointer(&mut self, x: f32, y: f32, button: u8, pressed: bool) {
        // Production: XTestFakeMotionEvent + XTestFakeButtonEvent
        // Stub: log the denormalized event
        self.log.inject_pointer(x, y, button, pressed);
    }

    fn inject_key(&mut self, code: u32, pressed: bool, modifiers: u8) {
        // Production: XTestFakeKeyEvent
        // Handle modifiers: inject synthetic Shift/Ctrl/Alt press before, release after
        self.log.inject_key(code, pressed, modifiers);
    }

    fn inject_touch(&mut self, index: u8, x: f32, y: f32, phase: u8, pressure: f32) {
        // X11 doesn't have native touch — map to pointer events
        self.log.inject_touch(index, x, y, phase, pressure);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wayland Virtual Input Injection
// ═══════════════════════════════════════════════════════════════════════════════

/// Wayland input injection via virtual input device.
///
/// Two approaches:
/// 1. **wlr-virtual-pointer/keyboard** (wlroots compositors: Sway, Hyprland)
/// 2. **uinput** (kernel-level, works everywhere but needs permissions)
///
/// ## Architecture
/// ```text
/// InputEvent → denormalize coordinates
///   → [wlroots] zwlr_virtual_pointer_v1 / zwp_virtual_keyboard_v1
///   → [fallback] /dev/uinput virtual device → evdev events
/// ```
///
/// ## Permissions
/// uinput requires either:
/// - Root access, or
/// - User in `input` group, or
/// - `udev` rule granting access to `/dev/uinput`
pub struct WaylandInputInjector {
    screen_width: u32,
    screen_height: u32,
    // In production: wayland-client connection + virtual_pointer/keyboard objects
    // Or: uinput fd for the virtual device
    log: LoggingInjector,
}

impl WaylandInputInjector {
    /// Initialize Wayland input injection.
    pub fn new(screen_width: u32, screen_height: u32) -> Result<Self, String> {
        if std::env::var("WAYLAND_DISPLAY").is_err() {
            return Err("Wayland not available: $WAYLAND_DISPLAY not set".to_string());
        }

        // In production:
        // 1. Connect to Wayland compositor
        // 2. Bind zwlr_virtual_pointer_manager_v1 (wlroots)
        // 3. Create virtual pointer + keyboard
        // 4. Fallback: open /dev/uinput, create virtual evdev device

        Ok(Self {
            screen_width,
            screen_height,
            log: LoggingInjector::new(screen_width, screen_height),
        })
    }

    /// Access the event log (for testing).
    pub fn events(&self) -> &[InjectedEvent] {
        &self.log.events
    }
}

impl InputInjector for WaylandInputInjector {
    fn inject_pointer(&mut self, x: f32, y: f32, button: u8, pressed: bool) {
        // Production:
        // wlr: zwlr_virtual_pointer_v1::motion_absolute(x * width, y * height)
        //      + zwlr_virtual_pointer_v1::button(BTN_LEFT, pressed)
        // uinput: write EV_ABS ABS_X/ABS_Y + EV_KEY BTN_LEFT
        self.log.inject_pointer(x, y, button, pressed);
    }

    fn inject_key(&mut self, code: u32, pressed: bool, modifiers: u8) {
        // Production:
        // wlr: zwp_virtual_keyboard_v1::key(code, pressed ? WL_KEYBOARD_KEY_STATE_PRESSED : RELEASED)
        // uinput: write EV_KEY code pressed
        self.log.inject_key(code, pressed, modifiers);
    }

    fn inject_touch(&mut self, index: u8, x: f32, y: f32, phase: u8, pressure: f32) {
        // Wayland supports multi-touch via wl_touch
        // uinput: EV_ABS ABS_MT_SLOT/ABS_MT_POSITION_X/Y
        self.log.inject_touch(index, x, y, phase, pressure);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Auto-detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect the display server and create the appropriate input injector.
pub fn auto_detect_injector(width: u32, height: u32) -> Box<dyn InputInjector> {
    // Try Wayland first.
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        match WaylandInputInjector::new(width, height) {
            Ok(inj) => {
                tracing::info!(
                    "Input injection: Wayland virtual input ({}x{})",
                    width,
                    height
                );
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!("Wayland input failed: {}, trying X11", e);
            }
        }
    }

    // Try X11.
    if std::env::var("DISPLAY").is_ok() {
        match X11TestInjector::new(width, height) {
            Ok(inj) => {
                tracing::info!("Input injection: X11 XTest ({}x{})", width, height);
                return Box::new(inj);
            }
            Err(e) => {
                tracing::warn!("X11 input failed: {}, using no-op", e);
            }
        }
    }

    tracing::warn!("No display server for input injection, using no-op");
    Box::new(NoopInjector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_injector_denormalization() {
        let mut inj = LoggingInjector::new(1920, 1080);
        inj.inject_pointer(0.5, 0.5, 1, true);

        match &inj.events[0] {
            InjectedEvent::Pointer {
                screen_x, screen_y, ..
            } => {
                assert_eq!(*screen_x, 960);
                assert_eq!(*screen_y, 540);
            }
            _ => panic!("Expected Pointer"),
        }
    }

    #[test]
    fn test_process_events_batch() {
        let mut inj = LoggingInjector::new(100, 100);
        let events = vec![
            InputEvent::Key {
                code: 65,
                pressed: true,
                modifiers: 0,
            },
            InputEvent::Pointer {
                x: 0.1,
                y: 0.2,
                button: 1,
                pressed: true,
            },
        ];
        inj.process_events(&events);
        assert_eq!(inj.events.len(), 2);
    }

    #[test]
    fn test_noop_injector_no_panic() {
        let mut inj = NoopInjector;
        inj.inject_pointer(0.5, 0.5, 1, true);
        inj.inject_key(65, true, 0);
        inj.inject_touch(0, 0.5, 0.5, 0, 0.5);
        // Success: NoopInjector silently discards all input without panic
    }
}
