package io.symthaea.soma

import android.content.Context
import android.util.Log
import kotlinx.serialization.json.Json
import java.io.Closeable
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Soma consciousness engine — Android binding.
 *
 * Wraps the native Rust engine with thread-safe access (ReentrantLock)
 * and automatic resource cleanup (Closeable). Every public method
 * acquires the lock and checks that the engine has not been closed.
 *
 * State persistence: Use [initStorage] with an Android [Context] to enable
 * automatic checkpoint save/restore. Call [persistState] before lifecycle
 * transitions (onPause, onStop) and [restoreState] after creation or
 * process death recovery.
 *
 * Usage:
 * ```kotlin
 * SomaEngine.createMobile().use { engine ->
 *     engine.initStorage(context)
 *     engine.restoreState()
 *     val result = engine.cycle("hello")
 *     engine.persistState()
 * }
 * ```
 */
class SomaEngine private constructor(private var handle: Long) : Closeable {
    private val lock = ReentrantLock()
    private var closed = false
    private val json = Json { ignoreUnknownKeys = true }

    /** Whether storage path has been configured for persistence. */
    @Volatile private var storageInitialized = false

    companion object {
        private const val TAG = "SomaEngine"

        fun create(): SomaEngine {
            val h = NativeBindings.engineNew()
            require(h != 0L) { "Failed to create SomaEngine" }
            return SomaEngine(h)
        }

        fun createMobile(): SomaEngine {
            val h = NativeBindings.engineNewMobile()
            require(h != 0L) { "Failed to create mobile SomaEngine" }
            return SomaEngine(h)
        }

        fun create(config: SomaConfig): SomaEngine {
            val configJson = Json.encodeToString(SomaConfig.serializer(), config)
            val h = NativeBindings.engineNewWithConfig(configJson)
            require(h != 0L) { "Failed to create SomaEngine from config" }
            return SomaEngine(h)
        }
    }

    /**
     * Initialize storage for state persistence using the app's private files directory.
     * Must be called before [persistState] or [restoreState].
     */
    fun initStorage(context: Context) {
        val storageDir = context.filesDir.resolve("soma_state").absolutePath
        setStoragePath(storageDir)
        storageInitialized = true
    }

    /**
     * Save engine state to persistent storage. Safe to call on any lifecycle transition.
     * Returns true if checkpoint was saved, false if storage not initialized or save failed.
     */
    fun persistState(): Boolean {
        if (!storageInitialized) return false
        return try {
            saveCheckpoint()
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to persist engine state", ex)
            false
        }
    }

    /**
     * Restore engine state from persistent storage. Call after engine creation.
     * Returns true if a checkpoint was found and restored.
     */
    fun restoreState(): Boolean {
        if (!storageInitialized) return false
        return try {
            val restored = loadCheckpoint()
            if (restored) {
                Log.i(TAG, "Engine state restored from checkpoint")
            }
            restored
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to restore engine state", ex)
            false
        }
    }

    // Core cycle

    fun cycle(input: String? = null): CycleResult = lock.withLock {
        check(!closed) { "SomaEngine is closed" }
        val resultJson = NativeBindings.cycleJson(handle, input)
            ?: error("cycle returned null")
        json.decodeFromString(CycleResult.serializer(), resultJson)
    }

    fun cycleRaw(input: String? = null): Float = lock.withLock {
        check(!closed) { "SomaEngine is closed" }
        NativeBindings.cycle(handle, input)
    }

    // State inspection

    val consciousnessLevel: Float get() = lock.withLock {
        check(!closed); NativeBindings.consciousnessLevel(handle)
    }

    val cycleCount: Long get() = lock.withLock {
        check(!closed); NativeBindings.cycleCount(handle)
    }

    val substrateFeasibility: Float get() = lock.withLock {
        check(!closed); NativeBindings.substrateFeasibility(handle)
    }

    val harmonyAlignment: Float get() = lock.withLock {
        check(!closed); NativeBindings.harmonyAlignment(handle)
    }

    fun consciousnessReport(): String = lock.withLock {
        check(!closed); NativeBindings.consciousnessReport(handle) ?: ""
    }

    fun neuromodStateJson(): String = lock.withLock {
        check(!closed); NativeBindings.neuromodJson(handle) ?: "{}"
    }

    fun compassSnapshot(): CompassSnapshot = lock.withLock {
        check(!closed)
        val s = NativeBindings.compassJson(handle) ?: error("compass returned null")
        json.decodeFromString(CompassSnapshot.serializer(), s)
    }

    // Platform integration

    fun setThermalLevel(level: Int) = lock.withLock {
        check(!closed); NativeBindings.setThermalLevel(handle, level.coerceIn(0, 4))
    }

    fun setBatteryState(chargePercent: Int, isCharging: Boolean) = lock.withLock {
        check(!closed); NativeBindings.setBatteryState(handle, chargePercent.coerceIn(0, 100), isCharging)
    }

    fun setNightMode(isNight: Boolean) = lock.withLock {
        check(!closed); NativeBindings.setNightMode(handle, isNight)
    }

    // Metabolism

    fun sendWakeSignal(signal: WakeSignal) = lock.withLock {
        check(!closed); NativeBindings.wakeSignal(handle, signal.code)
    }

    val wakeState: WakeState get() = lock.withLock {
        check(!closed); WakeState.fromCode(NativeBindings.wakeState(handle))
    }

    // Sensor bridge

    fun setSensors(
        accelMagnitude: Float,
        lightLux: Float,
        proximityNear: Boolean,
        barometerHpa: Float,
        gpsNovelty: Float,
    ) = lock.withLock {
        check(!closed)
        NativeBindings.setSensors(handle, accelMagnitude, lightLux, proximityNear, barometerHpa, gpsNovelty)
    }

    val motionState: MotionState get() = lock.withLock {
        check(!closed); MotionState.fromCode(NativeBindings.motionState(handle))
    }

    val privacyMode: Boolean get() = lock.withLock {
        check(!closed); NativeBindings.privacyMode(handle)
    }

    // Expanded senses

    fun setGyroscope(rotationRate: Float) = lock.withLock {
        check(!closed); NativeBindings.setGyroscope(handle, rotationRate)
    }

    fun setStepDelta(steps: Int) = lock.withLock {
        check(!closed); NativeBindings.setStepDelta(handle, steps)
    }

    fun setAmbientDb(db: Float) = lock.withLock {
        check(!closed); NativeBindings.setAmbientDb(handle, db)
    }

    fun setSocialPressure(notificationCount: Int) = lock.withLock {
        check(!closed); NativeBindings.setSocialPressure(handle, notificationCount)
    }

    fun setMediaState(state: Int) = lock.withLock {
        check(!closed); NativeBindings.setMediaState(handle, state)
    }

    /** Drain haptic events as JSON and return raw string. */
    fun hapticDrain(): String = lock.withLock {
        check(!closed); NativeBindings.hapticDrain(handle) ?: "[]"
    }

    // Broca language generation

    /** Generate text from current consciousness state. Returns JSON with text/num_tokens/eos. */
    fun generateText(maxTokens: Int = 12): String = lock.withLock {
        check(!closed); NativeBindings.generateText(handle, maxTokens) ?: "{\"text\":\"\",\"num_tokens\":0}"
    }

    /** Generate text responding to user input context. */
    fun generateTextWithInput(input: String, maxTokens: Int = 20): String = lock.withLock {
        check(!closed); NativeBindings.generateTextWithInput(handle, input, maxTokens) ?: "{\"text\":\"\",\"num_tokens\":0}"
    }

    /** Load trained BrocaLite checkpoint for higher-quality generation. */
    fun loadBrocaCheckpoint(data: ByteArray): Boolean = lock.withLock {
        check(!closed); NativeBindings.loadBrocaCheckpoint(handle, data)
    }

    /** Inject user engagement signal (0.0-1.0) into neuromodulator bath. */
    fun setEngagementScore(score: Float) = lock.withLock {
        check(!closed); NativeBindings.setEngagementScore(handle, score)
    }

    /** Full 20-channel embodied Broca pipeline with coherence/veto metadata. */
    fun generateEmbodiedText(): String = lock.withLock {
        check(!closed); NativeBindings.generateEmbodiedText(handle) ?: "{\"text\":\"\",\"num_tokens\":0}"
    }

    // Persistence

    fun saveCheckpoint(): Boolean = lock.withLock {
        check(!closed); NativeBindings.saveCheckpoint(handle)
    }

    fun loadCheckpoint(): Boolean = lock.withLock {
        check(!closed); NativeBindings.loadCheckpoint(handle)
    }

    fun setStoragePath(path: String) = lock.withLock {
        check(!closed); NativeBindings.setStoragePath(handle, path)
    }

    // Sharing config

    fun setSharingConfig(config: SharingConfig) = lock.withLock {
        check(!closed)
        NativeBindings.setSharingConfig(handle, Json.encodeToString(SharingConfig.serializer(), config))
    }

    // Haptic

    fun drainHapticEvents(): String = lock.withLock {
        check(!closed); NativeBindings.hapticDrain(handle) ?: "[]"
    }

    val hapticPendingCount: Int get() = lock.withLock {
        check(!closed); NativeBindings.hapticPending(handle)
    }

    fun setHapticEnabled(enabled: Boolean) = lock.withLock {
        check(!closed); NativeBindings.hapticSetEnabled(handle, enabled)
    }

    // Dream engine + journal

    fun dreamCycle(): Boolean = lock.withLock {
        check(!closed); NativeBindings.dreamCycle(handle)
    }

    fun dreamConsolidate() = lock.withLock {
        check(!closed); NativeBindings.dreamConsolidate(handle)
    }

    fun dreamJournalLatest(): DreamFragment? = lock.withLock {
        check(!closed)
        val s = NativeBindings.dreamJournalLatest(handle)
        if (s == null || s == "null") null
        else json.decodeFromString(DreamFragment.serializer(), s)
    }

    fun dreamJournalAll(): List<DreamFragment> = lock.withLock {
        check(!closed)
        val s = NativeBindings.dreamJournalAll(handle) ?: "[]"
        json.decodeFromString(s)
    }

    val dreamJournalCount: Int get() = lock.withLock {
        check(!closed); NativeBindings.dreamJournalCount(handle)
    }

    // Holon bridge

    fun holonDrainOutbound(): String = lock.withLock {
        check(!closed); NativeBindings.holonDrainOutbound(handle) ?: "[]"
    }

    fun holonReceive(msgJson: String) = lock.withLock {
        check(!closed); NativeBindings.holonReceive(handle, msgJson)
    }

    fun setHolonConnected(connected: Boolean) = lock.withLock {
        check(!closed); NativeBindings.holonSetConnected(handle, connected)
    }

    // Screen vision

    /**
     * Inject a screen frame (RGB bytes) for visual processing.
     * Returns the overall surprise level (0.0-1.0).
     */
    fun injectFrame(data: ByteArray, width: Int, height: Int, channels: Int = 3): Float = lock.withLock {
        check(!closed); NativeBindings.injectFrame(handle, data, width, height, channels)
    }

    /**
     * Inject a touch event for proprioceptive processing.
     * @param action 0=Down, 1=Move, 2=Up, 3=Cancel
     */
    fun onTouchEvent(x: Float, y: Float, action: Int, pressure: Float) = lock.withLock {
        check(!closed); NativeBindings.touchEvent(handle, x, y, action, pressure)
    }

    /** Get screen vision telemetry as JSON. */
    fun screenTelemetryJson(): String = lock.withLock {
        check(!closed); NativeBindings.screenSalientRegionsJson(handle) ?: "{}"
    }

    /** The raw engine handle for direct bridge access (e.g., SomaScreenCaptureBridge). */
    val nativeHandle: Long get() = lock.withLock {
        check(!closed); handle
    }

    // BLE mesh

    fun bleReceivePeer(peerId: Long, cvData: ByteArray): Boolean = lock.withLock {
        check(!closed); NativeBindings.bleReceivePeer(handle, peerId, cvData)
    }

    fun bleAdvertisePayload(): ByteArray? = lock.withLock {
        check(!closed); NativeBindings.bleAdvertisePayload(handle)
    }

    val blePeerCount: Int get() = lock.withLock {
        check(!closed); NativeBindings.blePeerCount(handle)
    }

    val bleCollectivePhi: Float get() = lock.withLock {
        check(!closed); NativeBindings.bleCollectivePhi(handle)
    }

    // Prism epistemic search (offline, sub-ms)

    /** Initialize Prism epistemic search with pre-seeded claims. */
    fun prismInit() = lock.withLock {
        check(!closed)
        try { NativeBindings.prismInit(handle) } catch (_: UnsatisfiedLinkError) {}
    }

    /** Search Prism epistemic claims. Returns JSON array or null. */
    fun prismSearch(query: String, topK: Int = 3): String? = lock.withLock {
        check(!closed)
        try { NativeBindings.prismSearch(handle, query, topK) } catch (_: UnsatisfiedLinkError) { null }
    }

    /** Whether Prism search is initialized. */
    val prismAvailable: Boolean get() = lock.withLock {
        check(!closed)
        try { NativeBindings.prismAvailable(handle) } catch (_: UnsatisfiedLinkError) { false }
    }

    // Lifecycle

    override fun close() = lock.withLock {
        if (!closed) {
            closed = true
            NativeBindings.engineFree(handle)
            handle = 0
        }
    }
}
