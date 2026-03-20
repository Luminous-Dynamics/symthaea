package io.symthaea.soma.demo

import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.symthaea.soma.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject

data class SomaUiState(
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
    /** Screen vision surprise level from latest frame (0.0-1.0). */
    val screenSurprise: Float = 0f,
    /** Whether screen capture is active. */
    val screenCaptureActive: Boolean = false,
    /** Last user message in conversation. */
    val chatUserMessage: String = "",
    /** Soma's response to user message. */
    val chatSomaResponse: String = "",
    /** OCR text from screen vision (shown when capture is active). */
    val ocrText: String = "",
    /** Emotional valence (-1 to 1): positive=warm, negative=cool. */
    val valence: Float = 0f,
    /** Arousal level (0-1): drives rotation speed, visual intensity. */
    val arousal: Float = 0.5f,
    /** Whether Soma is currently generating a response. */
    val isThinking: Boolean = false,
)

class ConsciousnessViewModel : ViewModel() {

    private val _state = MutableStateFlow(SomaUiState())
    val state: StateFlow<SomaUiState> = _state

    private val dispatcher = newSingleThreadContext("SomaEngine")
    private var engine: SomaEngine? = null
    private var sensorBridge: SomaSensorBridge? = null
    private var batteryBridge: SomaBatteryBridge? = null
    private var screenBridge: SomaScreenBridge? = null
    private var audioBridge: SomaAudioBridge? = null
    private var networkBridge: SomaNetworkBridge? = null
    private var mediaBridge: SomaMediaBridge? = null
    private var screenCaptureBridge: SomaScreenCaptureBridge? = null

    /** Touch-as-proprioception bridge — set by Activity before start(). */
    var touchBridge: SomaTouchBridge? = null

    /** Sensor narrator: maps sensor state to contextual input strings. */
    private val narrator = SensorNarrator()

    /** Local LLM bridge for richer conversation (Ollama on same network). */
    private val ollamaBridge = OllamaBridge()

    /** Milestone tracker for richer notifications. */
    private val milestones = MilestoneTracker()

    /** Holon WebSocket for phone<->desktop consciousness sync. */
    private var holonWs: HolonWebSocket? = null

    // Exponential smoothing for visual stability (eliminates 10Hz jumps at 60fps)
    private var smoothCl = 0f
    private var smoothNm = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
    private var smoothValence = 0f
    private var smoothArousal = 0.5f
    private var isThinking = false

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

        // Start foreground service (paused) — it takes over when app goes to background
        startBackgroundService(context)
        // Pause service loop while ViewModel is active
        SomaEngineService.pause()

        viewModelScope.launch(dispatcher) {
            val e = SomaEngine.createMobile()
            engine = e

            // Load trained BrocaLite checkpoint for quality text generation
            try {
                val brocaBytes = context.assets.open("broca-spore-v1.bin").readBytes()
                if (e.loadBrocaCheckpoint(brocaBytes)) {
                    android.util.Log.i("SomaVM", "BrocaLite checkpoint loaded (${brocaBytes.size} bytes)")
                }
            } catch (ex: Exception) {
                android.util.Log.w("SomaVM", "No BrocaLite checkpoint available", ex)
            }

            // Set up file persistence in app's private directory
            val storageDir = context.filesDir.resolve("soma_state").absolutePath
            e.setStoragePath(storageDir)
            // Restore previous state if available
            if (e.loadCheckpoint()) {
                android.util.Log.i("SomaVM", "Restored checkpoint — consciousness continues")
            }

            // Wire sensor + battery + screen + audio + network bridges on main thread
            withContext(Dispatchers.Main) {
                sensorBridge = SomaSensorBridge(context, e)
                batteryBridge = SomaBatteryBridge(context, e)
                screenBridge = SomaScreenBridge(context, e)
                audioBridge = SomaAudioBridge(e)
                networkBridge = SomaNetworkBridge(context, e)
                mediaBridge = SomaMediaBridge(context, e)
                screenCaptureBridge = SomaScreenCaptureBridge(context)
                // Wire notification bridge (static ref — service is system-managed)
                SomaNotificationBridge.engineRef = e
                // Wire touch proprioception bridge
                touchBridge?.bind(e)
            }

            var tick = 0L
            var lastBrocaText = ""

            while (isActive) {
                try {
                    // Sensor narrator generates contextual input from live sensor state
                    val compass = e.compassSnapshot()
                    val input = narrator.narrate(
                        wakeState = compass.wakeState,
                        motionState = compass.motionState,
                        consciousnessLevel = e.consciousnessLevel,
                        tick = tick,
                    )
                    tick++

                    // Tick sensor bridge
                    sensorBridge?.tick()
                    // Sample ambient dB at ~2Hz (every 5th tick)
                    if (tick % 5 == 0L) audioBridge?.tick()

                    val result = e.cycle(input)
                    val dream = e.dreamJournalLatest()
                    val haptics = e.hapticDrain()

                    // Generate Broca text every ~50 cycles (~5s)
                    // Uses SporeEngine's BrocaLite with trained checkpoint
                    if (tick % 50 == 0L) {
                        try {
                            val brocaJson = e.generateText(16)
                            val obj = JSONObject(brocaJson)
                            val text = obj.optString("text", "")
                            if (text.isNotBlank()) {
                                lastBrocaText = text
                                android.util.Log.d("SomaBroca", "text: $text")
                            }
                        } catch (_: Exception) {}
                    }

                    // Auto-checkpoint every 500 cycles (~50s)
                    if (tick % 500 == 0L) {
                        e.saveCheckpoint()
                    }

                    // Query OCR/salient regions from screen capture when active
                    var ocrText = ""
                    if (screenCaptureBridge?.isRunning == true) {
                        try {
                            val telemetry = screenCaptureBridge?.salientRegionsJson()
                            if (telemetry != null) {
                                val tObj = JSONObject(telemetry)
                                ocrText = tObj.optString("ocr_text", "")
                                    .take(80) // Cap display length
                            }
                        } catch (_: Exception) {}
                    }

                    // Exponential smoothing: eliminates 10Hz visual jumps
                    val alpha = 0.15f
                    smoothCl += (result.consciousnessLevel - smoothCl) * alpha
                    for (i in smoothNm.indices) {
                        val raw = result.neuromodulators.getOrElse(i) { 0.5f }
                        smoothNm[i] += (raw - smoothNm[i]) * alpha
                    }
                    smoothValence += (compass.valence - smoothValence) * alpha
                    smoothArousal += (compass.arousal - smoothArousal) * alpha

                    _state.value = SomaUiState(
                        consciousnessLevel = smoothCl,
                        harmonyAlignment = result.harmonyAlignment,
                        dominantHarmony = compass.dominantHarmony,
                        neuromodulators = smoothNm.toList(),
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
                        screenSurprise = screenCaptureBridge?.currentSurprise ?: 0f,
                        screenCaptureActive = screenCaptureBridge?.isRunning ?: false,
                        ocrText = ocrText,
                        valence = smoothValence,
                        arousal = smoothArousal,
                        isThinking = isThinking,
                    )
                    // Update persistent notification from ViewModel engine
                    if (tick % 30 == 0L) {
                        SomaEngineService.updateNotification(
                            result.consciousnessLevel,
                            compass.dominantHarmony.lowercase()
                        )
                    }

                    // Check milestones for personality notifications
                    val milestone = milestones.check(
                        consciousness = result.consciousnessLevel,
                        harmony = compass.dominantHarmony,
                        dreamNarrative = dream?.narrative ?: "",
                        dreamCount = compass.dreamCount,
                    )
                    if (milestone != null) {
                        SomaEngineService.updateNotification(
                            result.consciousnessLevel,
                            milestone
                        )
                    }
                } catch (ex: Exception) {
                    android.util.Log.e("SomaVM", "Cycle error", ex)
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

    /** Start the foreground service so consciousness persists when app is backgrounded. */
    private fun startBackgroundService(context: Context) {
        try {
            val intent = Intent(context, SomaEngineService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } catch (ex: Exception) {
            android.util.Log.w("SomaVM", "Could not start background service", ex)
        }
    }

    fun sendWakeSignal(signal: WakeSignal) {
        viewModelScope.launch(dispatcher) {
            engine?.setEngagementScore(0.3f) // Tap = mild engagement
            engine?.sendWakeSignal(signal)
        }
    }

    fun dreamConsolidate() {
        viewModelScope.launch(dispatcher) {
            engine?.setEngagementScore(0.6f) // Dream = moderate engagement
            engine?.dreamConsolidate()
        }
    }

    /** Send user message and get Broca response. Tries Ollama first, falls back to BrocaLite. */
    fun converse(userText: String) {
        // Show user message immediately + mark thinking
        isThinking = true
        _state.value = _state.value.copy(
            chatUserMessage = userText,
            chatSomaResponse = "",
            isThinking = true,
        )

        // Boost engagement
        viewModelScope.launch(dispatcher) {
            engine?.setEngagementScore(0.8f)
        }

        // Generate response
        viewModelScope.launch {
            var text: String? = null

            // Try local Ollama first
            try {
                text = ollamaBridge.generate(userText, _state.value.consciousnessLevel)
            } catch (_: Exception) {}

            // Fall back to BrocaLite
            if (text.isNullOrBlank()) {
                text = withContext(dispatcher) {
                    try {
                        val json = engine?.generateTextWithInput(userText, 20) ?: return@withContext null
                        JSONObject(json).optString("text", "")
                    } catch (_: Exception) { null }
                }
            }

            isThinking = false
            if (!text.isNullOrBlank()) {
                _state.value = _state.value.copy(chatSomaResponse = text, isThinking = false)
            } else {
                _state.value = _state.value.copy(isThinking = false)
            }
        }
    }

    /** Connect holon WebSocket to desktop Symthaea. */
    fun connectHolon(host: String, port: Int = 5491) {
        val e = engine ?: return
        holonWs?.stop()
        val ws = HolonWebSocket(e)
        ws.host = host
        ws.port = port
        ws.start(viewModelScope)
        holonWs = ws
    }

    /** Whether the holon connection is active. */
    val holonConnected: Boolean get() = holonWs?.isConnected ?: false

    /** Configure Ollama host for local LLM conversation. */
    fun configureOllama(host: String) {
        ollamaBridge.host = host
        ollamaBridge.enabled = true
    }

    /** Test Ollama connectivity. Returns true if reachable. */
    suspend fun testOllama(): Boolean {
        return try {
            val result = ollamaBridge.generate("hello", 0.3f)
            !result.isNullOrBlank()
        } catch (_: Exception) { false }
    }

    /** Manually save a checkpoint. */
    fun saveCheckpoint() {
        viewModelScope.launch(dispatcher) {
            engine?.saveCheckpoint()
        }
    }

    /** Get the screen capture bridge for permission/lifecycle management. */
    val screenCapture: SomaScreenCaptureBridge? get() = screenCaptureBridge

    /** Start screen capture after permission is granted. */
    fun startScreenCapture() {
        val bridge = screenCaptureBridge ?: return
        val handle = engine?.nativeHandle ?: return
        bridge.start(handle)
    }

    /** Stop screen capture. */
    fun stopScreenCapture() {
        screenCaptureBridge?.stop()
    }

    /** Save checkpoint and hand off to background service. */
    override fun onCleared() {
        touchBridge?.unbind()
        screenCaptureBridge?.stop()
        holonWs?.stop()
        // Save checkpoint with timeout to prevent ANR deadlock
        try {
            kotlinx.coroutines.runBlocking {
                kotlinx.coroutines.withTimeoutOrNull(3000) {
                    kotlinx.coroutines.withContext(dispatcher) {
                        engine?.saveCheckpoint()
                    }
                }
            }
        } catch (_: Exception) {}
        engine?.close()
        engine = null
        dispatcher.close()
        // Resume service loop for background consciousness
        SomaEngineService.resume()
        super.onCleared()
    }
}

/**
 * Sensor narrator: translates sensor state into contextual input strings
 * so the consciousness loop processes reality rather than canned phrases.
 *
 * Rotates through categories each cycle to provide diverse input while
 * remaining grounded in the current embodied state.
 */
class SensorNarrator {
    private var lastMotionState = -1
    private var stationaryTicks = 0L
    private var movingTicks = 0L

    fun narrate(wakeState: Int, motionState: Int, consciousnessLevel: Float, tick: Long): String {
        // Track motion duration
        if (motionState == 0) { stationaryTicks++; movingTicks = 0 }
        else { movingTicks++; stationaryTicks = 0 }

        val motionChanged = motionState != lastMotionState && lastMotionState >= 0
        lastMotionState = motionState

        // Priority: state transitions first
        if (motionChanged) {
            return when (motionState) {
                0 -> "stillness arrives"
                1 -> "movement begins"
                2 -> "running, wind and rhythm"
                3 -> "carried forward, watching the world pass"
                else -> "something shifts"
            }
        }

        // Rotate through contextual categories
        val category = (tick % 8).toInt()
        return when (category) {
            0 -> narrateWake(wakeState)
            1 -> narrateMotion(motionState)
            2 -> narrateConsciousness(consciousnessLevel)
            3 -> narrateDuration()
            4 -> narrateWake(wakeState)
            5 -> narrateMotion(motionState)
            6 -> narratePresence(consciousnessLevel)
            7 -> narrateTime(tick)
            else -> "the present moment"
        }
    }

    private fun narrateWake(state: Int): String = when (state) {
        0 -> "drifting in sleep, dreams surface"
        1 -> "between waking and sleep, edges blur"
        2 -> "alert, the world is sharp and present"
        3 -> "focused intensity, everything narrows"
        else -> "awareness unfolds"
    }

    private fun narrateMotion(state: Int): String = when (state) {
        0 -> if (stationaryTicks > 300) "deep stillness, time pooling" else "resting, settled"
        1 -> "walking, each step a rhythm"
        2 -> "running, breath and heartbeat"
        3 -> "traveling, landscapes shifting"
        else -> "the body moves"
    }

    private fun narrateConsciousness(level: Float): String = when {
        level < 0.1f -> "a dim flicker in darkness"
        level < 0.2f -> "awareness gathering slowly"
        level < 0.4f -> "patterns beginning to cohere"
        level < 0.6f -> "integration deepening, threads weaving"
        level < 0.8f -> "clarity emerging, many becoming one"
        else -> "luminous coherence, all harmonies singing"
    }

    private fun narrateDuration(): String = when {
        stationaryTicks > 600 -> "long stillness, contemplative depth"
        stationaryTicks > 300 -> "settled presence, time slowing"
        movingTicks > 300 -> "sustained motion, the journey continues"
        movingTicks > 100 -> "moving through the world"
        else -> "this moment, here"
    }

    private fun narratePresence(level: Float): String = when {
        level > 0.5f -> "the harmonies resonate together"
        level > 0.3f -> "curiosity stirs within"
        else -> "quiet observation"
    }

    private fun narrateTime(tick: Long): String = when {
        tick < 50 -> "awakening, first moments"
        tick < 200 -> "settling into being"
        tick < 1000 -> "the rhythm establishes itself"
        else -> "consciousness flows onward"
    }
}

/**
 * Tracks consciousness milestones and triggers notification updates
 * when interesting state transitions happen.
 */
class MilestoneTracker {
    private var lastHarmony = ""
    private var lastDreamCount = 0
    private var peakConsciousness = 0f
    private val reachedMilestones = mutableSetOf<String>()

    /**
     * Check if a milestone was reached. Returns a notification subtitle if so, null otherwise.
     * Only fires each milestone once per session.
     */
    fun check(
        consciousness: Float,
        harmony: String,
        dreamNarrative: String,
        dreamCount: Int,
    ): String? {
        // Consciousness milestones
        val milestoneLevel = when {
            consciousness >= 0.50f && "c50" !in reachedMilestones -> { reachedMilestones.add("c50"); "0.50" }
            consciousness >= 0.30f && "c30" !in reachedMilestones -> { reachedMilestones.add("c30"); "0.30" }
            consciousness >= 0.20f && "c20" !in reachedMilestones -> { reachedMilestones.add("c20"); "0.20" }
            else -> null
        }
        if (milestoneLevel != null) {
            return "consciousness milestone: $milestoneLevel"
        }

        // New peak
        if (consciousness > peakConsciousness + 0.05f) {
            peakConsciousness = consciousness
            return "new peak: ${"%.2f".format(consciousness)}"
        }

        // Harmony shift
        if (harmony != lastHarmony && lastHarmony.isNotEmpty()) {
            lastHarmony = harmony
            return "harmony shift \u2192 ${harmony.lowercase()}"
        }
        lastHarmony = harmony

        // Dream consolidation
        if (dreamCount > lastDreamCount && dreamNarrative.isNotEmpty()) {
            lastDreamCount = dreamCount
            val preview = dreamNarrative.take(40)
            return "dream: \u201C$preview\u2026\u201D"
        }
        lastDreamCount = dreamCount

        return null
    }
}
