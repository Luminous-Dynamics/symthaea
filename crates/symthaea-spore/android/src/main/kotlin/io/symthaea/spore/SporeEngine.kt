package io.symthaea.spore

import android.content.Context
import android.util.Log
import kotlinx.serialization.json.Json
import java.io.Closeable
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Spore consciousness engine — Android binding.
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
 * SporeEngine.createMobile().use { engine ->
 *     engine.initStorage(context)
 *     engine.restoreState()
 *     val result = engine.cycle("hello")
 *     engine.persistState()
 * }
 * ```
 */
class SporeEngine private constructor(private var handle: Long) : Closeable {
    private val lock = ReentrantLock()
    private var closed = false
    private val json = Json { ignoreUnknownKeys = true }

    /** Whether storage path has been configured for persistence. */
    @Volatile private var storageInitialized = false

    companion object {
        private const val TAG = "SporeEngine"

        fun create(): SporeEngine {
            val h = NativeBindings.engineNew()
            require(h != 0L) { "Failed to create SporeEngine" }
            return SporeEngine(h)
        }

        fun createMobile(): SporeEngine {
            val h = NativeBindings.engineNewMobile()
            require(h != 0L) { "Failed to create mobile SporeEngine" }
            return SporeEngine(h)
        }

        fun create(config: SporeConfig): SporeEngine {
            val configJson = Json.encodeToString(SporeConfig.serializer(), config)
            val h = NativeBindings.engineNewWithConfig(configJson)
            require(h != 0L) { "Failed to create SporeEngine from config" }
            return SporeEngine(h)
        }
    }

    /**
     * Initialize storage for state persistence using the app's private files directory.
     * Must be called before [persistState] or [restoreState].
     */
    fun initStorage(context: Context) {
        val storageDir = context.filesDir.resolve("spore_state").absolutePath
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
        check(!closed) { "SporeEngine is closed" }
        val resultJson = NativeBindings.cycleJson(handle, input)
            ?: error("cycle returned null")
        json.decodeFromString(CycleResult.serializer(), resultJson)
    }

    fun cycleRaw(input: String? = null): Float = lock.withLock {
        check(!closed) { "SporeEngine is closed" }
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

    // Lifecycle

    override fun close() = lock.withLock {
        if (!closed) {
            closed = true
            NativeBindings.engineFree(handle)
            handle = 0
        }
    }
}
