package io.symthaea.soma

/**
 * JNI bindings to the native symthaea-soma Rust library.
 *
 * All functions map 1:1 to the `extern "C"` surface in `native_ffi.rs`.
 * The engine handle is an opaque pointer (Long) — callers must not
 * interpret or fabricate handle values.
 *
 * Thread safety: The native engine is NOT thread-safe. All callers
 * must serialize access externally (see [SomaEngine]).
 */
internal object NativeBindings {
    init {
        // Load LiteRT shim first — libsymthaea_soma.so depends on litert_* symbols
        try { System.loadLibrary("litert_shim") } catch (_: UnsatisfiedLinkError) {}
        System.loadLibrary("soma_jni")
    }

    // Lifecycle
    @JvmStatic external fun engineNew(): Long
    @JvmStatic external fun engineNewMobile(): Long
    @JvmStatic external fun engineNewWithConfig(configJson: String): Long
    @JvmStatic external fun engineFree(handle: Long)

    // Core cycle
    @JvmStatic external fun cycle(handle: Long, input: String?): Float
    @JvmStatic external fun cycleJson(handle: Long, input: String?): String?

    // State inspection
    @JvmStatic external fun consciousnessLevel(handle: Long): Float
    @JvmStatic external fun cycleCount(handle: Long): Long
    @JvmStatic external fun substrateFeasibility(handle: Long): Float
    @JvmStatic external fun harmonyAlignment(handle: Long): Float
    @JvmStatic external fun consciousnessReport(handle: Long): String?
    @JvmStatic external fun neuromodJson(handle: Long): String?

    // Platform integration
    @JvmStatic external fun setThermalLevel(handle: Long, level: Int)
    @JvmStatic external fun setBatteryState(handle: Long, percent: Int, isCharging: Boolean)
    @JvmStatic external fun setNightMode(handle: Long, isNight: Boolean)

    // Dream engine
    @JvmStatic external fun dreamCycle(handle: Long): Boolean

    // Metabolism
    @JvmStatic external fun wakeSignal(handle: Long, signal: Int)
    @JvmStatic external fun wakeState(handle: Long): Int

    // Sensor bridge
    @JvmStatic external fun setSensors(handle: Long, accel: Float, light: Float, proximityNear: Boolean, barometer: Float, gpsNovelty: Float)
    @JvmStatic external fun motionState(handle: Long): Int
    @JvmStatic external fun privacyMode(handle: Long): Boolean

    // Expanded senses
    @JvmStatic external fun setGyroscope(handle: Long, rotationRate: Float)
    @JvmStatic external fun setStepDelta(handle: Long, steps: Int)
    @JvmStatic external fun setAmbientDb(handle: Long, db: Float)
    @JvmStatic external fun setSocialPressure(handle: Long, notificationCount: Int)
    @JvmStatic external fun setMediaState(handle: Long, state: Int)

    // Compass
    @JvmStatic external fun compassJson(handle: Long): String?

    // Sharing config
    @JvmStatic external fun setSharingConfig(handle: Long, json: String)

    // Haptic
    @JvmStatic external fun hapticDrain(handle: Long): String?
    @JvmStatic external fun hapticPending(handle: Long): Int
    @JvmStatic external fun hapticSetEnabled(handle: Long, enabled: Boolean)

    // Dream journal
    @JvmStatic external fun dreamJournalLatest(handle: Long): String?
    @JvmStatic external fun dreamJournalAll(handle: Long): String?
    @JvmStatic external fun dreamJournalCount(handle: Long): Int
    @JvmStatic external fun dreamConsolidate(handle: Long)

    // Holon bridge
    @JvmStatic external fun holonDrainOutbound(handle: Long): String?
    @JvmStatic external fun holonReceive(handle: Long, json: String)
    @JvmStatic external fun holonSetConnected(handle: Long, connected: Boolean)

    // Broca language generation
    @JvmStatic external fun generateText(handle: Long, maxTokens: Int): String?
    /** Generate text responding to user input. */
    @JvmStatic external fun generateTextWithInput(handle: Long, input: String, maxTokens: Int): String?
    /** Full 20-channel embodied Broca pipeline (returns richer JSON with coherence/veto). */
    @JvmStatic external fun generateEmbodiedText(handle: Long): String?

    // Broca checkpoint loading
    @JvmStatic external fun loadBrocaCheckpoint(handle: Long, data: ByteArray): Boolean

    // Engagement signal
    @JvmStatic external fun setEngagementScore(handle: Long, score: Float)

    // Persistence
    @JvmStatic external fun saveCheckpoint(handle: Long): Boolean
    @JvmStatic external fun loadCheckpoint(handle: Long): Boolean
    @JvmStatic external fun setStoragePath(handle: Long, path: String)

    // BLE mesh
    @JvmStatic external fun bleReceivePeer(handle: Long, peerId: Long, cvData: ByteArray): Boolean
    @JvmStatic external fun bleAdvertisePayload(handle: Long): ByteArray?
    @JvmStatic external fun blePeerCount(handle: Long): Int
    @JvmStatic external fun bleCollectivePhi(handle: Long): Float

    // Prism epistemic search (requires prism-search feature in Rust)
    @JvmStatic external fun prismInit(handle: Long)
    @JvmStatic external fun prismSearch(handle: Long, query: String, topK: Int): String?
    @JvmStatic external fun prismAvailable(handle: Long): Boolean

    // Screen vision (requires screen-vision feature in Rust)
    @JvmStatic external fun injectFrame(handle: Long, data: ByteArray, width: Int, height: Int, channels: Int): Float
    @JvmStatic external fun touchEvent(handle: Long, x: Float, y: Float, action: Int, pressure: Float)
    @JvmStatic external fun screenSalientRegionsJson(handle: Long): String?
}
