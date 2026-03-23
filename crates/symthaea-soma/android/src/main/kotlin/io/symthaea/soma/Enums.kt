package io.symthaea.soma

/**
 * Sleep/wake state of the metabolism state machine.
 *
 * Maps to Rust `WakeState` in `metabolism.rs`.
 */
enum class WakeState(val code: Int) {
    Sleep(0),
    Drowsy(1),
    Alert(2),
    Focused(3);

    companion object {
        fun fromCode(code: Int): WakeState =
            entries.firstOrNull { it.code == code } ?: Alert
    }
}

/**
 * Signals that can be sent to the metabolism to trigger state transitions.
 *
 * Maps to Rust `WakeSignal` in `metabolism.rs`.
 */
enum class WakeSignal(val code: Int) {
    /** Phone picked up / screen on. */
    PhonePickup(0),
    /** User typed or tapped. */
    UserInput(1),
    /** Explicit request to enter sleep. */
    ExplicitSleep(2);
}

/**
 * Motion state derived from accelerometer magnitude.
 *
 * Maps to Rust `MotionState` in `sensor_bridge.rs`.
 */
enum class MotionState(val code: Int) {
    Stationary(0),
    Walking(1),
    Running(2),
    InVehicle(3);

    companion object {
        fun fromCode(code: Int): MotionState =
            entries.firstOrNull { it.code == code } ?: Stationary
    }
}
