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

    /// Create with mobile-optimized defaults.
    public init(mobile: Bool) {
        handle = mobile ? soma_engine_new_mobile() : soma_engine_new()
    }

    deinit {
        lock.lock()
        defer { lock.unlock() }
        if let h = handle {
            soma_engine_free(h)
            handle = nil
        }
    }

    // MARK: - Helpers

    /// Call a function with the handle under lock, returning a value.
    private func withLock<T>(_ body: (OpaquePointer) -> T, default defaultValue: T) -> T {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return defaultValue }
        return body(h)
    }

    /// Call a void function with the handle under lock.
    private func withLockVoid(_ body: (OpaquePointer) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        body(h)
    }

    /// Call a function returning a C string, convert to Swift String, free the C string.
    private func withLockString(_ body: (OpaquePointer) -> UnsafeMutablePointer<CChar>?) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return nil }
        guard let cStr = body(h) else { return nil }
        let result = String(cString: cStr)
        soma_string_free(cStr)
        return result
    }

    // MARK: - Core Cycle

    /// Run a single consciousness cycle. Returns consciousness level.
    public func cycle(input: String) -> Float {
        return withLock({ h in input.withCString { soma_engine_cycle(h, $0) } }, default: 0.0)
    }

    /// Run a single consciousness cycle. Returns JSON-encoded CycleResult.
    public func cycleJSON(input: String) -> String? {
        return withLockString { h in input.withCString { soma_engine_cycle_json(h, $0) } }
    }

    // MARK: - State

    public var consciousnessLevel: Float {
        withLock({ soma_engine_consciousness_level($0) }, default: 0.0)
    }

    public var cycleCount: UInt64 {
        withLock({ soma_engine_cycle_count($0) }, default: 0)
    }

    public var harmonyAlignment: Float {
        withLock({ soma_engine_harmony_alignment($0) }, default: 0.0)
    }

    public var substrateFeasibility: Float {
        withLock({ soma_engine_substrate_feasibility($0) }, default: 0.0)
    }

    public var motionState: UInt8 {
        withLock({ soma_engine_motion_state($0) }, default: 0)
    }

    public var privacyMode: Bool {
        withLock({ soma_engine_privacy_mode($0) != 0 }, default: false)
    }

    public func consciousnessReport() -> String? {
        withLockString { soma_engine_consciousness_report($0) }
    }

    public func neuromodState() -> String? {
        withLockString { soma_engine_neuromod_json($0) }
    }

    public func consciousnessCompass() -> String? {
        withLockString { soma_engine_compass_json($0) }
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
        withLockVoid { soma_engine_set_sensors($0, acceleration, lightLux, proximityNear, barometerHpa, gpsNovelty) }
    }

    public func setGyroscope(rotationRate: Float) {
        withLockVoid { soma_engine_set_gyroscope($0, rotationRate) }
    }

    public func setStepDelta(_ steps: UInt32) {
        withLockVoid { soma_engine_set_step_delta($0, steps) }
    }

    public func setAmbientDb(_ db: Float) {
        withLockVoid { soma_engine_set_ambient_db($0, db) }
    }

    public func setSocialPressure(notificationCount: UInt32) {
        withLockVoid { soma_engine_set_social_pressure($0, notificationCount) }
    }

    public func setMediaState(_ state: UInt8) {
        withLockVoid { soma_engine_set_media_state($0, state) }
    }

    public func setEngagementScore(_ score: Float) {
        withLockVoid { soma_engine_set_engagement_score($0, score) }
    }

    // MARK: - Platform State

    public func setThermalLevel(_ level: UInt8) {
        withLockVoid { soma_engine_set_thermal_level($0, level) }
    }

    public func setBatteryState(percent: UInt8, charging: Bool) {
        withLockVoid { soma_engine_set_battery_state($0, percent, charging) }
    }

    public func setNightMode(_ enabled: Bool) {
        withLockVoid { soma_engine_set_night_mode($0, enabled ? 1 : 0) }
    }

    // MARK: - Wake State

    public func wakeSignal(_ signal: UInt8 = 1) {
        withLockVoid { soma_engine_wake_signal($0, signal) }
    }

    public var wakeState: UInt8 {
        withLock({ soma_engine_wake_state($0) }, default: 0)
    }

    // MARK: - Text Generation

    public func generateText(maxTokens: UInt32 = 32) -> String? {
        withLockString { soma_engine_generate_text($0, maxTokens) }
    }

    public func generateText(input: String, maxTokens: UInt32 = 32) -> String? {
        withLockString { h in input.withCString { soma_engine_generate_text_with_input(h, $0, maxTokens) } }
    }

    public func generateEmbodiedText() -> String? {
        withLockString { soma_engine_generate_embodied_text($0) }
    }

    // MARK: - Holon Bridge

    public func drainHolonOutbound() -> String? {
        withLockString { soma_engine_holon_drain_outbound($0) }
    }

    public func receiveHolonMessage(_ json: String) {
        withLockVoid { h in json.withCString { soma_engine_holon_receive(h, $0) } }
    }

    public func setHolonConnected(_ connected: Bool) {
        withLockVoid { soma_engine_holon_set_connected($0, connected ? 1 : 0) }
    }

    // MARK: - BLE Mesh

    /// Receive a peer consciousness vector from BLE scan.
    /// Data should be 12 bytes (phi + valence + arousal as little-endian f32).
    public func receiveBLEPeer(peerId: UInt64, data: Data) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return false }
        return data.withUnsafeBytes { bufferPtr in
            guard let baseAddress = bufferPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return false
            }
            return soma_ble_receive_peer(h, peerId, baseAddress, bufferPtr.count)
        }
    }

    /// Get the BLE advertisement payload (own consciousness vector).
    public func bleAdvertisePayload() -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return nil }
        var outLen: Int = 0
        guard let ptr = soma_ble_advertise_payload(h, &outLen) else { return nil }
        return Data(bytes: ptr, count: outLen)
    }

    public var blePeerCount: UInt32 {
        withLock({ soma_ble_peer_count($0) }, default: 0)
    }

    public var bleCollectivePhi: Float {
        withLock({ soma_ble_collective_phi($0) }, default: 0.0)
    }

    // MARK: - Haptics

    public var hapticPending: UInt32 {
        withLock({ soma_haptic_pending($0) }, default: 0)
    }

    /// Drain all pending haptic events as JSON.
    public func drainHaptics() -> String? {
        withLockString { soma_haptic_drain($0) }
    }

    public func setHapticEnabled(_ enabled: Bool) {
        withLockVoid { soma_haptic_set_enabled($0, enabled ? 1 : 0) }
    }

    // MARK: - Dreams

    public func dreamCycle() -> Bool {
        withLock({ soma_engine_dream_cycle($0) != 0 }, default: false)
    }

    public func dreamConsolidate() {
        withLockVoid { soma_engine_dream_consolidate($0) }
    }

    public func dreamJournalLatest() -> String? {
        withLockString { soma_dream_journal_latest($0) }
    }

    public func dreamJournalAll() -> String? {
        withLockString { soma_dream_journal_all($0) }
    }

    public var dreamJournalCount: UInt32 {
        withLock({ soma_dream_journal_count($0) }, default: 0)
    }

    // MARK: - Wellbeing

    public func morningRitual(sleepQuality: UInt8, sleepMinutes: UInt16) {
        withLockVoid { soma_engine_morning_ritual($0, sleepQuality, sleepMinutes) }
    }

    public func eveningRitual(mood: UInt8, energy: UInt8, stress: UInt8) {
        withLockVoid { soma_engine_evening_ritual($0, mood, energy, stress) }
    }

    public func setWellbeingProfile(_ name: String) {
        withLockVoid { h in name.withCString { soma_engine_set_wellbeing_profile(h, $0) } }
    }

    public var wellbeingProfileName: String? {
        withLockString { soma_engine_wellbeing_profile_name($0) }
    }

    public static var wellbeingProfilesList: String? {
        guard let cStr = soma_engine_wellbeing_profiles_list() else { return nil }
        let result = String(cString: cStr)
        soma_string_free(cStr)
        return result
    }

    // MARK: - Screen Vision

    /// Inject a raw RGB frame for screen vision processing.
    public func injectFrame(_ rgb: Data, width: UInt32, height: UInt32) {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else { return }
        rgb.withUnsafeBytes { bufferPtr in
            guard let baseAddress = bufferPtr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            soma_engine_inject_frame(h, baseAddress, bufferPtr.count, width, height)
        }
    }

    /// Report a touch event for proprioceptive processing.
    public func touchEvent(x: Float, y: Float, action: UInt8, pressure: Float, timestampMs: UInt64) {
        withLockVoid { soma_engine_touch_event($0, x, y, action, pressure, timestampMs) }
    }

    /// Get salient screen regions as JSON.
    public func screenSalientRegions() -> String? {
        withLockString { soma_engine_screen_salient_regions_json($0) }
    }

    // MARK: - Persistence

    public func saveCheckpoint() -> Bool {
        withLock({ soma_engine_save_checkpoint($0) != 0 }, default: false)
    }

    public func loadCheckpoint() -> Bool {
        withLock({ soma_engine_load_checkpoint($0) != 0 }, default: false)
    }

    public func setStoragePath(_ path: String) {
        withLockVoid { h in path.withCString { soma_engine_set_storage_path(h, $0) } }
    }

    public func loadBrocaCheckpoint(_ path: String) -> Bool {
        return withLock({ h in path.withCString { soma_engine_load_broca_checkpoint(h, $0) != 0 } }, default: false)
    }

    // MARK: - Sharing Config

    /// Set sharing modes. holonMode: 0=Off, 1=PresenceOnly, 2=FullSync.
    /// bleMode: 0=Off, 1=PassiveShare, 2=ActiveShare.
    public func setSharingConfig(holonMode: UInt8, bleMode: UInt8) {
        withLockVoid { soma_engine_set_sharing_config($0, holonMode, bleMode) }
    }
}
