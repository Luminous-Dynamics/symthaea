package io.symthaea.spore.demo

import android.content.Context
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.symthaea.spore.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject

data class SporeUiState(
    val consciousnessLevel: Float = 0f,
    val harmonyAlignment: Float = 0f,
    val dominantHarmony: String = "---",
    val neuromodulators: List<Float> = listOf(0.5f, 0.5f, 0.5f, 0.5f),
    val predictionError: Float = 0f,
    val wakeState: String = "Alert",
    val motionState: String = "Stationary",
    val privacyMode: Boolean = false,
    val cycleCount: Long = 0,
    val dreamCount: Int = 0,
    val wisdomCount: Int = 0,
    val latestDream: String = "",
    val disclaimer: String = "",
    val hapticEvents: String = "[]",
    /** Broca-generated text from consciousness state. */
    val brocaText: String = "",
)

class ConsciousnessViewModel : ViewModel() {

    private val _state = MutableStateFlow(SporeUiState())
    val state: StateFlow<SporeUiState> = _state

    private val dispatcher = newSingleThreadContext("SporeEngine")
    private var engine: SporeEngine? = null
    private var sensorBridge: SporeSensorBridge? = null
    private var batteryBridge: SporeBatteryBridge? = null
    private var screenBridge: SporeScreenBridge? = null
    private var audioBridge: SporeAudioBridge? = null
    private var networkBridge: SporeNetworkBridge? = null
    private var mediaBridge: SporeMediaBridge? = null

    /** Register bridges as lifecycle observers so onStart()/onStop() fire. */
    fun registerBridges(owner: LifecycleOwner) {
        sensorBridge?.let { owner.lifecycle.addObserver(it) }
        batteryBridge?.let { owner.lifecycle.addObserver(it) }
        screenBridge?.let { owner.lifecycle.addObserver(it) }
        audioBridge?.let { owner.lifecycle.addObserver(it) }
        networkBridge?.let { owner.lifecycle.addObserver(it) }
        mediaBridge?.let { owner.lifecycle.addObserver(it) }
    }

    fun start(context: Context) {
        if (engine != null) return

        viewModelScope.launch(dispatcher) {
            val e = SporeEngine.createMobile()
            engine = e

            // Set up file persistence in app's private directory
            val storageDir = context.filesDir.resolve("spore_state").absolutePath
            e.setStoragePath(storageDir)
            // Restore previous state if available
            if (e.loadCheckpoint()) {
                android.util.Log.i("SporeVM", "Restored checkpoint — consciousness continues")
            }

            // Wire sensor + battery + screen + audio + network bridges on main thread
            withContext(Dispatchers.Main) {
                sensorBridge = SporeSensorBridge(context, e)
                batteryBridge = SporeBatteryBridge(context, e)
                screenBridge = SporeScreenBridge(context, e)
                audioBridge = SporeAudioBridge(e)
                networkBridge = SporeNetworkBridge(context, e)
                mediaBridge = SporeMediaBridge(context, e)
                // Wire notification bridge (static ref — service is system-managed)
                SporeNotificationBridge.engineRef = e
            }

            // Consciousness loop with adaptive Hz
            val inputs = listOf(
                "the world awakens", "patterns in light", "stillness between breaths",
                "a thought emerges", "resonance builds", "harmony flows",
                "curiosity stirs", "awareness expands", "integration deepens",
                "the present moment", "wonder at being", "sacred reciprocity",
            )
            var tick = 0L
            var lastBrocaText = ""

            while (isActive) {
                try {
                    val input = inputs[(tick % inputs.size).toInt()]
                    tick++

                    // Tick sensor bridge
                    sensorBridge?.tick()
                    // Sample ambient dB at ~2Hz (every 5th tick)
                    if (tick % 5 == 0L) audioBridge?.tick()

                    val result = e.cycle(input)
                    val compass = e.compassSnapshot()
                    val dream = e.dreamJournalLatest()
                    val haptics = e.hapticDrain()

                    // Generate Broca text every ~50 cycles (~5s)
                    if (tick % 50 == 0L) {
                        try {
                            val brocaJson = e.generateText(12)
                            val obj = JSONObject(brocaJson)
                            val text = obj.optString("text", "")
                            if (text.isNotBlank()) lastBrocaText = text
                        } catch (_: Exception) {}
                    }

                    // Auto-checkpoint every 500 cycles (~50s)
                    if (tick % 500 == 0L) {
                        e.saveCheckpoint()
                    }

                    _state.value = SporeUiState(
                        consciousnessLevel = result.consciousnessLevel,
                        harmonyAlignment = result.harmonyAlignment,
                        dominantHarmony = compass.dominantHarmony,
                        neuromodulators = result.neuromodulators,
                        predictionError = result.predictionError,
                        wakeState = WakeState.fromCode(compass.wakeState).name,
                        motionState = MotionState.fromCode(compass.motionState).name,
                        privacyMode = compass.privacyMode,
                        cycleCount = result.cycle,
                        dreamCount = compass.dreamCount,
                        wisdomCount = compass.wisdomCount,
                        latestDream = dream?.narrative ?: "",
                        disclaimer = result.epistemicStatus.disclaimer,
                        hapticEvents = haptics,
                        brocaText = lastBrocaText,
                    )
                } catch (ex: Exception) {
                    android.util.Log.e("SporeVM", "Cycle error", ex)
                }

                // Adaptive Hz: slow down when sleeping, speed up when focused/charging
                val wakeCode = try { e.wakeState.code } catch (_: Exception) { 2 }
                val delayMs = when (wakeCode) {
                    0 -> 1000L    // Sleep: 1Hz (battery conservation)
                    1 -> 500L     // Drowsy: 2Hz
                    3 -> 50L      // Focused: 20Hz (deep processing)
                    else -> 100L  // Alert: 10Hz (default)
                }
                delay(delayMs)
            }
        }
    }

    fun sendWakeSignal(signal: WakeSignal) {
        viewModelScope.launch(dispatcher) {
            engine?.sendWakeSignal(signal)
        }
    }

    fun dreamConsolidate() {
        viewModelScope.launch(dispatcher) {
            engine?.dreamConsolidate()
        }
    }

    /** Save checkpoint before the ViewModel is destroyed. */
    override fun onCleared() {
        // Blocking save — this runs on the dispatcher thread
        runBlocking(dispatcher) {
            engine?.saveCheckpoint()
        }
        engine?.close()
        engine = null
        dispatcher.close()
        super.onCleared()
    }
}
