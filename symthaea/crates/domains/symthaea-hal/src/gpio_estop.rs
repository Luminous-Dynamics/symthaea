// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GPIO-based emergency stop.
//!
//! [`GpioEstop`] polls a physical button on a GPIO pin and updates the
//! [`SafetyInterlock`](crate::SafetyInterlock)'s shared e-stop flag.
//! Pin read errors are treated as e-stop triggers (fail-safe).
//!
//! # Usage
//!
//! ```rust,no_run
//! use symthaea_hal::{SafetyInterlock, gpio_estop::GpioEstop};
//! use symthaea_hal::mock::MockInputPin;
//!
//! let interlock = SafetyInterlock::new();
//! let pin = MockInputPin::new(false); // not pressed
//! let mut estop = GpioEstop::new(pin, interlock.estop_handle());
//! estop.poll(); // reads pin, updates e-stop flag
//! ```

use embedded_hal::digital::InputPin;
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::warn;

// ============================================================================
// ESTOP POLLER TRAIT
// ============================================================================

/// Trait-object-safe e-stop poller.
///
/// This allows the runtime to poll an e-stop without being generic over
/// the pin type. [`GpioEstop`] implements this automatically for any
/// `InputPin + Send`.
pub trait EstopPoller: Send {
    /// Poll the e-stop. Returns `true` if triggered.
    fn poll(&mut self) -> bool;
}

impl<P: InputPin + Send> EstopPoller for GpioEstop<P> {
    fn poll(&mut self) -> bool {
        GpioEstop::poll(self)
    }
}

// ============================================================================
// GPIO E-STOP
// ============================================================================

/// GPIO-based emergency stop controller.
///
/// Generic over any `embedded_hal::digital::InputPin`. Reads the pin state
/// on each `poll()` call and sets the shared e-stop flag accordingly.
///
/// The pin is active-low by default: a low reading (pressed/grounded) triggers
/// the e-stop. Pin read errors also trigger e-stop (fail-safe).
pub struct GpioEstop<P: InputPin> {
    pin: P,
    estop: Arc<Mutex<bool>>,
    /// If true, low = e-stop (default). If false, high = e-stop.
    active_low: bool,
}

impl<P: InputPin> GpioEstop<P> {
    /// Create a new GPIO e-stop (active-low by default).
    pub fn new(pin: P, estop: Arc<Mutex<bool>>) -> Self {
        Self {
            pin,
            estop,
            active_low: true,
        }
    }

    /// Set active-high mode (high = e-stop triggered).
    pub fn active_high(mut self) -> Self {
        self.active_low = false;
        self
    }

    /// Poll the GPIO pin and update the e-stop flag.
    ///
    /// Returns `true` if *this poll* observed the physical button pressed
    /// (or a pin read error).
    ///
    /// Only ever *sets* the shared e-stop flag on this poller's own
    /// trigger condition -- it never clears it. The flag is shared with
    /// [`SafetyInterlock`](crate::SafetyInterlock), which may also set it
    /// via `trigger_estop()` from an unrelated source (e.g. a software or
    /// remote e-stop); if `poll()` cleared the flag whenever the physical
    /// button merely isn't being held down, it would silently undo any
    /// e-stop triggered by that other source on the very next poll.
    /// Clearing an e-stop (physical or otherwise) is the deliberate
    /// responsibility of [`SafetyInterlock::release_estop`](crate::SafetyInterlock::release_estop).
    pub fn poll(&mut self) -> bool {
        let triggered = match self.pin.is_low() {
            Ok(low) => {
                if self.active_low {
                    low
                } else {
                    !low
                }
            }
            Err(_) => {
                // Pin read error → fail-safe: trigger e-stop
                warn!("GPIO e-stop pin read error — triggering e-stop (fail-safe)");
                true
            }
        };

        if triggered {
            *self.estop.lock() = true;
        }
        triggered
    }

    /// Whether the e-stop is currently active (last poll result).
    pub fn is_triggered(&self) -> bool {
        *self.estop.lock()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockInputPin;

    #[test]
    fn test_gpio_estop_not_pressed() {
        let estop = Arc::new(Mutex::new(false));
        let pin = MockInputPin::new(false); // not pressed (high)
        let mut gpio = GpioEstop::new(pin, Arc::clone(&estop));

        let triggered = gpio.poll();
        assert!(!triggered);
        assert!(!gpio.is_triggered());
    }

    #[test]
    fn test_gpio_estop_pressed() {
        let estop = Arc::new(Mutex::new(false));
        let pin = MockInputPin::new(true); // pressed (low)
        let mut gpio = GpioEstop::new(pin, Arc::clone(&estop));

        let triggered = gpio.poll();
        assert!(triggered);
        assert!(gpio.is_triggered());
    }

    #[test]
    fn test_gpio_estop_active_high() {
        let estop = Arc::new(Mutex::new(false));
        let pin = MockInputPin::new(false); // pin is high
        let mut gpio = GpioEstop::new(pin, Arc::clone(&estop)).active_high();

        // In active-high mode, high = triggered
        let triggered = gpio.poll();
        assert!(triggered);
    }

    #[test]
    fn test_gpio_estop_error_failsafe() {
        let estop = Arc::new(Mutex::new(false));
        let pin = MockInputPin::with_error();
        let mut gpio = GpioEstop::new(pin, Arc::clone(&estop));

        let triggered = gpio.poll();
        assert!(triggered, "pin error should trigger e-stop (fail-safe)");
    }

    #[test]
    fn test_gpio_estop_shared_with_interlock() {
        let interlock = crate::SafetyInterlock::new();
        let handle = interlock.estop_handle();
        let pin = MockInputPin::new(true); // pressed
        let mut gpio = GpioEstop::new(pin, handle);

        gpio.poll();
        assert!(interlock.is_estopped());
    }

    #[test]
    fn test_gpio_estop_poll_does_not_clear_software_triggered_estop() {
        // Regression test: a software/remote e-stop (trigger_estop()) must
        // survive a poll() where the physical button simply isn't pressed --
        // poll() must never clear an e-stop it didn't itself trigger.
        let interlock = crate::SafetyInterlock::new();
        let handle = interlock.estop_handle();

        interlock.trigger_estop();
        assert!(interlock.is_estopped());

        let pin = MockInputPin::new(false); // NOT pressed
        let mut gpio = GpioEstop::new(pin, handle);

        let triggered = gpio.poll();
        assert!(
            !triggered,
            "this poll's own reading should be 'not pressed'"
        );
        assert!(
            interlock.is_estopped(),
            "poll() must not silently clear an e-stop triggered by another source"
        );
    }

    #[test]
    fn test_gpio_estop_release_still_works_after_button_released() {
        // The physical button being released does NOT auto-clear the estop;
        // release_estop() is what actually clears it.
        let estop = Arc::new(Mutex::new(false));
        let pin = MockInputPin::new(true); // pressed
        let mut gpio = GpioEstop::new(pin, Arc::clone(&estop));
        gpio.poll();
        assert!(gpio.is_triggered());

        // Physical button released -- flag should remain set.
        gpio.pin = MockInputPin::new(false);
        let triggered = gpio.poll();
        assert!(!triggered);
        assert!(
            gpio.is_triggered(),
            "flag must stay latched until explicitly released"
        );

        // Explicit release.
        *estop.lock() = false;
        assert!(!gpio.is_triggered());
    }
}
