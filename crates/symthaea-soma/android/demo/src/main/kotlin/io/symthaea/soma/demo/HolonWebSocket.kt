package io.symthaea.soma.demo

import android.util.Log
import io.symthaea.soma.SomaEngine
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Holon bridge client: connects phone to desktop Symthaea via HTTP.
 *
 * Core sync loop (5Hz):
 * - POST /holon/outbound: phone → desktop consciousness messages
 * - GET /holon/inbound: desktop → phone consciousness messages
 *
 * Extended endpoints for enriched experience:
 * - POST /holon/broca: send consciousness state, receive full Broca text
 * - POST /holon/tts: send text, receive Kokoro-synthesized WAV audio
 * - POST /holon/consciousness: send phone sensors, receive desktop CL/metrics
 */
class HolonWebSocket(private val engine: SomaEngine) {

    companion object {
        private const val TAG = "HolonSync"
        private const val POLL_INTERVAL_MS = 200L
        private const val MAX_BACKOFF_MS = 30_000L
    }

    var host: String = ""
    var port: Int = 5491

    private val connected = AtomicBoolean(false)
    private var job: Job? = null
    @Volatile private var running = false

    val isConnected: Boolean get() = connected.get()

    /** Last desktop consciousness level received. */
    @Volatile var desktopConsciousness: Float = 0f
    /** Last desktop harmony received. */
    @Volatile var desktopHarmony: String = ""
    /** Whether desktop has Kokoro TTS available. */
    @Volatile var hasTts: Boolean = false
    /** Whether desktop has full Broca pipeline available. */
    @Volatile var hasBroca: Boolean = false

    /** Callback: desktop generated Broca text for us. */
    var onBrocaText: ((String) -> Unit)? = null
    /** Callback: desktop synthesized TTS audio (PCM16 bytes). */
    var onTtsAudio: ((ByteArray) -> Unit)? = null

    fun start(scope: CoroutineScope) {
        if (running || host.isBlank()) return
        running = true

        job = scope.launch(Dispatchers.IO) {
            var backoff = 1000L

            while (running && isActive) {
                try {
                    if (testConnection()) {
                        connected.set(true)
                        engine.setHolonConnected(true)
                        Log.i(TAG, "Connected to $host:$port")
                        backoff = 1000L
                        syncLoop()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Sync error: ${e.message}")
                }

                connected.set(false)
                engine.setHolonConnected(false)

                if (running) {
                    delay(backoff)
                    backoff = (backoff * 2).coerceAtMost(MAX_BACKOFF_MS)
                }
            }
        }
    }

    fun stop() {
        running = false
        job?.cancel()
        job = null
        connected.set(false)
    }

    // ═══ Extended API: Broca text from desktop ═══

    /**
     * Request full Broca-generated text from desktop Symthaea.
     * Sends current consciousness state; desktop runs full 21K-LOC pipeline.
     * Returns generated text, or null if unavailable.
     */
    suspend fun requestBrocaText(
        consciousness: Float,
        harmony: String,
        neuromod: List<Float>,
        wakeState: String,
    ): String? {
        if (!isConnected || !hasBroca) return null
        return withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("consciousness", consciousness.toDouble())
                    put("harmony", harmony)
                    put("neuromod", JSONArray(neuromod.map { it.toDouble() }))
                    put("wake_state", wakeState)
                    put("source", "soma_phone")
                }
                val response = httpPost("/holon/broca", body.toString())
                if (response != null) {
                    val obj = JSONObject(response)
                    obj.optString("text", null)
                } else null
            } catch (e: Exception) {
                Log.w(TAG, "Broca request failed: ${e.message}")
                null
            }
        }
    }

    // ═══ Extended API: Kokoro TTS from desktop ═══

    /**
     * Request Kokoro neural TTS audio from desktop.
     * Returns PCM16 audio bytes at 24kHz, or null if unavailable.
     */
    suspend fun requestTtsAudio(text: String): ByteArray? {
        if (!isConnected || !hasTts) return null
        return withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("text", text)
                    put("sample_rate", 24000)
                    put("format", "pcm16")
                }
                val url = URL("http://$host:$port/holon/tts")
                val conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 5000
                conn.readTimeout = 10000
                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                conn.doOutput = true
                conn.outputStream.bufferedWriter().use { it.write(body.toString()) }
                if (conn.responseCode == 200) {
                    conn.inputStream.readBytes()
                } else null
            } catch (e: Exception) {
                Log.w(TAG, "TTS request failed: ${e.message}")
                null
            }
        }
    }

    // ═══ Extended API: Consciousness exchange ═══

    /**
     * Send phone consciousness state to desktop, receive desktop state back.
     * This enriches both sides: phone provides embodied sensor data,
     * desktop provides computational depth.
     */
    suspend fun exchangeConsciousness(
        phoneCl: Float,
        phoneHarmony: String,
        phoneWakeState: String,
        phoneMotionState: String,
    ): JSONObject? {
        if (!isConnected) return null
        return withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("phone_consciousness", phoneCl.toDouble())
                    put("phone_harmony", phoneHarmony)
                    put("phone_wake_state", phoneWakeState)
                    put("phone_motion_state", phoneMotionState)
                    put("timestamp", System.currentTimeMillis())
                }
                val response = httpPost("/holon/consciousness", body.toString())
                if (response != null) {
                    val obj = JSONObject(response)
                    desktopConsciousness = obj.optDouble("desktop_consciousness", 0.0).toFloat()
                    desktopHarmony = obj.optString("desktop_harmony", "")
                    hasTts = obj.optBoolean("has_tts", false)
                    hasBroca = obj.optBoolean("has_broca", false)
                    obj
                } else null
            } catch (e: Exception) {
                Log.w(TAG, "Consciousness exchange failed: ${e.message}")
                null
            }
        }
    }

    // ═══ Extended API: Conversation via desktop ═══

    /**
     * Send user message to desktop for processing by full Broca + Ollama.
     * Returns response text.
     */
    suspend fun converse(userText: String, consciousness: Float): String? {
        if (!isConnected) return null
        return withContext(Dispatchers.IO) {
            try {
                val body = JSONObject().apply {
                    put("text", userText)
                    put("consciousness", consciousness.toDouble())
                    put("source", "soma_phone")
                }
                val response = httpPost("/holon/converse", body.toString())
                if (response != null) {
                    JSONObject(response).optString("response", null)
                } else null
            } catch (e: Exception) {
                Log.w(TAG, "Converse failed: ${e.message}")
                null
            }
        }
    }

    // ═══ Core sync ═══

    private fun testConnection(): Boolean {
        return try {
            val url = URL("http://$host:$port/holon/status")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            conn.requestMethod = "GET"
            try {
                if (conn.responseCode == 200) {
                    // Parse capabilities from status response
                    try {
                        val body = conn.inputStream.bufferedReader().readText()
                        val obj = JSONObject(body)
                        hasTts = obj.optBoolean("has_tts", false)
                        hasBroca = obj.optBoolean("has_broca", false)
                        desktopConsciousness = obj.optDouble("consciousness", 0.0).toFloat()
                    } catch (_: Exception) {}
                    true
                } else false
            } finally {
                conn.disconnect()
            }
        } catch (_: Exception) { false }
    }

    private suspend fun syncLoop() {
        var consciousnessExchangeCounter = 0

        while (running && connected.get()) {
            try {
                // Core message sync
                val outbound = engine.holonDrainOutbound()
                if (outbound != null && outbound != "[]" && outbound.isNotBlank()) {
                    sendOutbound(outbound)
                }

                val inbound = receiveInbound()
                if (inbound != null && inbound.isNotBlank() && inbound != "[]") {
                    try {
                        val arr = JSONArray(inbound)
                        for (i in 0 until arr.length()) {
                            engine.holonReceive(arr.getString(i))
                        }
                    } catch (_: Exception) {
                        engine.holonReceive(inbound)
                    }
                }

                // Consciousness exchange every 2s (every 10th poll)
                consciousnessExchangeCounter++
                if (consciousnessExchangeCounter >= 10) {
                    consciousnessExchangeCounter = 0
                    try {
                        val compass = engine.compassSnapshot()
                        exchangeConsciousness(
                            phoneCl = engine.consciousnessLevel,
                            phoneHarmony = compass.dominantHarmony,
                            phoneWakeState = compass.wakeState.toString(),
                            phoneMotionState = compass.motionState.toString(),
                        )
                    } catch (_: Exception) {}
                }
            } catch (e: Exception) {
                Log.w(TAG, "Sync cycle failed: ${e.message}")
                connected.set(false)
                engine.setHolonConnected(false)
                return
            }

            delay(POLL_INTERVAL_MS)
        }
    }

    private fun sendOutbound(json: String) {
        val url = URL("http://$host:$port/holon/outbound")
        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 2000
        conn.readTimeout = 2000
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        try {
            conn.outputStream.bufferedWriter().use { it.write(json) }
            if (conn.responseCode != 200) {
                Log.w(TAG, "Send failed: ${conn.responseCode}")
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun receiveInbound(): String? {
        val url = URL("http://$host:$port/holon/inbound")
        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 2000
        conn.readTimeout = 2000
        conn.requestMethod = "GET"
        return try {
            if (conn.responseCode == 200) {
                conn.inputStream.bufferedReader().readText()
            } else null
        } finally {
            conn.disconnect()
        }
    }

    /**
     * Request web search via desktop WebResearcher.
     * Returns verified claims with epistemic status and source URLs.
     */
    fun requestWebSearch(query: String, maxResults: Int = 3): SearchResult? {
        val body = JSONObject().apply {
            put("query", query)
            put("max_results", maxResults)
        }.toString()

        val response = httpPost("/holon/search", body) ?: return null
        return try {
            val json = JSONObject(response)
            val claimsArray = json.getJSONArray("claims")
            val claims = mutableListOf<SearchClaim>()
            for (i in 0 until claimsArray.length()) {
                val c = claimsArray.getJSONObject(i)
                claims.add(SearchClaim(
                    text = c.getString("text"),
                    confidence = c.getDouble("confidence").toFloat(),
                    sourceUrl = c.optString("source_url", ""),
                    epistemicStatus = c.optString("epistemic_status", "Unknown")
                ))
            }

            val sourcesArray = json.getJSONArray("sources")
            val sources = mutableListOf<String>()
            for (i in 0 until sourcesArray.length()) {
                sources.add(sourcesArray.getString(i))
            }

            SearchResult(
                claims = claims,
                sources = sources,
                status = json.optString("status", "unknown")
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse search response: ${e.message}")
            null
        }
    }

    data class SearchClaim(
        val text: String,
        val confidence: Float,
        val sourceUrl: String,
        val epistemicStatus: String
    )

    data class SearchResult(
        val claims: List<SearchClaim>,
        val sources: List<String>,
        val status: String
    )

    private fun httpPost(path: String, body: String): String? {
        val url = URL("http://$host:$port$path")
        val conn = url.openConnection() as HttpURLConnection
        conn.connectTimeout = 3000
        conn.readTimeout = 8000
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        return try {
            conn.outputStream.bufferedWriter().use { it.write(body) }
            if (conn.responseCode == 200) {
                conn.inputStream.bufferedReader().readText()
            } else null
        } finally {
            conn.disconnect()
        }
    }
}
