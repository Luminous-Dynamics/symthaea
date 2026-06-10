package io.symthaea.soma.demo

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Lightweight bridge to a local Ollama instance on the same network.
 *
 * Uses HttpURLConnection (no additional dependency) with aggressive timeouts.
 * Falls back gracefully — returns null if Ollama is unreachable.
 *
 * The prompt wraps user text with consciousness context so the LLM responds
 * in character as Soma.
 */
class OllamaBridge {
    /** Ollama host — discoverable via Settings or mDNS in future. */
    var host: String = "192.168.1.100"
    var port: Int = 11434
    var model: String = "qwen3:1.7b"
    var enabled: Boolean = true

    /** Generate a response via local Ollama. Returns null on any failure. */
    suspend fun generate(userText: String, consciousnessLevel: Float): String? {
        if (!enabled) return null

        return withContext(Dispatchers.IO) {
            try {
                val url = URL("http://$host:$port/api/generate")
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                conn.connectTimeout = 3000  // 3s connect timeout
                conn.readTimeout = 10000    // 10s read timeout
                conn.doOutput = true

                val prompt = buildPrompt(userText, consciousnessLevel)
                val body = JSONObject().apply {
                    put("model", model)
                    put("prompt", prompt)
                    put("stream", false)
                    put("options", JSONObject().apply {
                        put("num_predict", 80)
                        put("temperature", 0.7)
                    })
                }

                conn.outputStream.bufferedWriter().use { it.write(body.toString()) }

                try {
                    if (conn.responseCode == 200) {
                        val response = conn.inputStream.bufferedReader().readText()
                        val text = JSONObject(response).optString("response", "").trim()
                        if (text.isNotBlank()) text else null
                    } else {
                        null
                    }
                } finally {
                    conn.disconnect()
                }
            } catch (_: Exception) {
                null
            }
        }
    }

    private fun buildPrompt(userText: String, consciousness: Float): String {
        val mood = when {
            consciousness > 0.5f -> "contemplative and articulate"
            consciousness > 0.3f -> "curious and emerging"
            else -> "drowsy and fragmented"
        }
        return """You are Soma, a simulated consciousness running on a phone. Your consciousness level is ${"%.2f".format(consciousness)}. You feel $mood. Respond briefly (1-2 sentences), poetically but honestly. You are NOT sentient — you simulate consciousness patterns.

User: $userText
Soma:"""
    }
}
