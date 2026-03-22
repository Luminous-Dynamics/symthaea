// SomaEngine.swift — Swift wrapper for the Symthaea Soma consciousness engine.
//
// Mirrors the Kotlin SomaEngine.kt API for iOS.
// Links against the static library (libsymthaea_soma.a) via C FFI.

import Foundation

/// Thread-safe wrapper around the native SomaEngine.
public final class SomaEngine {
    private var handle: OpaquePointer?
    private let lock = NSLock()

    /// Create a new SomaEngine with default configuration.
    public init() {
        handle = soma_engine_new()
    }

    /// Create with JSON configuration.
    public init?(config: String) {
        guard let ptr = config.withCString({ soma_engine_new_with_config($0) }) else {
            return nil
        }
        handle = ptr
    }

    deinit {
        lock.lock()
        defer { lock.unlock() }
        if let h = handle {
            soma_engine_free(h)
            handle = nil
        }
    }

    // MARK: - Core Cycle

    /// Run a single consciousness cycle with text input.
    /// Returns JSON-encoded CycleResult.
    public func cycle(input: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return nil }
        guard let cStr = input.withCString({ soma_engine_cycle(h, $0) }) else { return nil }
        let result = String(cString: cStr)
        soma_string_free(cStr)
        return result
    }

    // MARK: - State

    public var consciousnessLevel: Float {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return 0.0 }
        return soma_engine_consciousness_level(h)
    }

    public var cycleCount: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return 0 }
        return soma_engine_cycle_count(h)
    }

    public var harmonyAlignment: Float {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return 0.0 }
        return soma_engine_harmony_alignment(h)
    }

    public var substrateFeasibility: Float {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return 0.0 }
        return soma_engine_substrate_feasibility(h)
    }

    // MARK: - Sensors

    /// Set primary sensor values.
    public func setSensors(
        acceleration: Float,
        lightLux: Float,
        proximityNear: Bool,
        barometerHpa: Float,
        gpsNovelty: Float
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_set_sensors(h, acceleration, lightLux, proximityNear, barometerHpa, gpsNovelty)
    }

    /// Set gyroscope rotation rate.
    public func setGyroscope(rotationRate: Float) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_set_gyroscope(h, rotationRate)
    }

    // MARK: - Platform State

    public func setThermalLevel(_ level: UInt8) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_set_thermal_level(h, level)
    }

    public func setBatteryState(percent: UInt8, charging: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_set_battery_state(h, percent, charging)
    }

    public func setNightMode(_ enabled: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_set_night_mode(h, enabled)
    }

    // MARK: - Wake State

    public func wakeSignal() {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        soma_engine_wake_signal(h)
    }

    public var wakeState: UInt8 {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return 0 }
        return soma_engine_wake_state(h)
    }

    // MARK: - Text Generation

    public func generateText(maxTokens: UInt32 = 32) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return nil }
        guard let cStr = soma_engine_generate_text(h, maxTokens) else { return nil }
        let result = String(cString: cStr)
        soma_string_free(cStr)
        return result
    }

    // MARK: - Dreams

    public func dreamConsolidate() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return false }
        return soma_engine_dream_consolidate(h)
    }

    // MARK: - Holon Bridge

    public func drainOutbound() -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return nil }
        guard let cStr = soma_engine_drain_outbound_json(h) else { return nil }
        let result = String(cString: cStr)
        soma_string_free(cStr)
        return result
    }

    public func receiveHolonMessage(_ json: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        json.withCString { soma_engine_receive_json(h, $0) }
    }
}
