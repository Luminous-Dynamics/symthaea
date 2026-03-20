package io.symthaea.soma.demo

import android.util.Log
import io.symthaea.soma.SomaEngine
import kotlinx.coroutines.*
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Holon bridge client: connects phone to desktop Symthaea via HTTP polling.
 *
 * Drains outbound holon messages from the engine at 5Hz and POSTs them
 * to the desktop. GETs inbound messages and injects into the engine.
 *
 * Uses simple HTTP instead of WebSocket for Android compatibility
 * (java.net.http.WebSocket not available on Android).
 *
 * Desktop endpoint: http://<host>:<port>/holon
 * - POST /holon/outbound: send phone → desktop messages
 * - GET /holon/inbound: receive desktop → phone messages
 */
class HolonWebSocket(private val engine: SomaEngine) {

    companion object {
        private const val TAG = "HolonSync"
        private const val POLL_INTERVAL_MS = 200L // 5Hz
        private const val MAX_BACKOFF_MS = 30_000L
    }

    var host: String = ""
    var port: Int = 5491

    private val connected = AtomicBoolean(false)
    private var job: Job? = null
    @Volatile private var running = false

    val isConnected: Boolean get() = connected.get()

    fun start(scope: CoroutineScope) {
        if (running || host.isBlank()) return
        running = true

        job = scope.launch(Dispatchers.IO) {
            var backoff = 1000L

            while (running && isActive) {
                try {
                    // Test connectivity
                    if (testConnection()) {
                        connected.set(true)
                        engine.setHolonConnected(true)
                        Log.i(TAG, "Connected to $host:$port")
                        backoff = 1000L

                        // Main sync loop
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

    private fun testConnection(): Boolean {
        return try {
            val url = URL("http://$host:$port/holon/status")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            conn.requestMethod = "GET"
            try {
                conn.responseCode == 200
            } finally {
                conn.disconnect()
            }
        } catch (_: Exception) {
            false
        }
    }

    private suspend fun syncLoop() {
        while (running && connected.get()) {
            try {
                // Send outbound messages
                val outbound = engine.holonDrainOutbound()
                if (outbound != null && outbound != "[]" && outbound.isNotBlank()) {
                    sendOutbound(outbound)
                }

                // Receive inbound messages
                val inbound = receiveInbound()
                if (inbound != null && inbound.isNotBlank() && inbound != "[]") {
                    // Inject each message into engine
                    try {
                        val arr = JSONArray(inbound)
                        for (i in 0 until arr.length()) {
                            engine.holonReceive(arr.getString(i))
                        }
                    } catch (_: Exception) {
                        // Single message, not array
                        engine.holonReceive(inbound)
                    }
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
}
