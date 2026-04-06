package io.symthaea.soma.demo

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Manages on-device Gemma 4 E2B model via LiteRT-LM Kotlin API.
 *
 * Uses the official Maven package (com.google.ai.edge.litertlm:litertlm-android)
 * which bundles the native inference engine. No custom C FFI needed.
 *
 * Lifecycle:
 *   1. Check if model exists on disk (~2.58 GB)
 *   2. If not, download from HuggingFace (WiFi only)
 *   3. Initialize Engine with GPU backend
 *   4. Provide generate() and generateStreaming() for on-device inference
 */
class LiteRTManager(private val context: Context) {

    companion object {
        private const val TAG = "LiteRTManager"
        private const val MODEL_FILENAME = "gemma4-e2b.litertlm"
        private const val MODEL_URL =
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-litert-lm.task"
        private const val MODEL_SIZE_BYTES = 2_580_000_000L
    }

    /** Download progress (0.0 to 1.0), or -1 if not downloading. */
    private val _downloadProgress = MutableStateFlow(-1f)
    val downloadProgress: StateFlow<Float> = _downloadProgress

    /** Whether the on-device model is ready for inference. */
    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady

    /** LiteRT-LM engine (null until initialized). */
    private var engine: com.google.ai.edge.litertlm.Engine? = null

    /** Path to the model file in app-private storage. */
    private val modelFile: File
        get() = File(context.filesDir, MODEL_FILENAME)

    /** Whether the model file exists and looks complete. */
    fun isModelDownloaded(): Boolean {
        val file = modelFile
        return file.exists() && file.length() > MODEL_SIZE_BYTES * 0.95
    }

    /**
     * Initialize the LiteRT-LM engine with GPU backend.
     * This can take up to 10 seconds — call from a background thread.
     */
    suspend fun initEngine(): Boolean = withContext(Dispatchers.IO) {
        if (!isModelDownloaded()) {
            Log.w(TAG, "Model not downloaded yet")
            return@withContext false
        }

        try {
            com.google.ai.edge.litertlm.Engine.setNativeMinLogSeverity(
                com.google.ai.edge.litertlm.LogSeverity.WARNING
            )

            val config = com.google.ai.edge.litertlm.EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = com.google.ai.edge.litertlm.Backend.GPU(),
                cacheDir = context.cacheDir.path,
            )

            val e = com.google.ai.edge.litertlm.Engine(config)
            e.initialize()
            engine = e
            _isReady.value = true
            Log.i(TAG, "LiteRT-LM engine ready (gemma4:e2b, GPU backend)")
            true
        } catch (ex: Exception) {
            Log.e(TAG, "Engine init failed: ${ex.message}")
            // Fallback to CPU
            try {
                val config = com.google.ai.edge.litertlm.EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = com.google.ai.edge.litertlm.Backend.CPU(),
                    cacheDir = context.cacheDir.path,
                )
                val e = com.google.ai.edge.litertlm.Engine(config)
                e.initialize()
                engine = e
                _isReady.value = true
                Log.i(TAG, "LiteRT-LM engine ready (gemma4:e2b, CPU fallback)")
                true
            } catch (ex2: Exception) {
                Log.e(TAG, "CPU fallback also failed: ${ex2.message}")
                false
            }
        }
    }

    /**
     * Generate text on-device (blocking). Returns null if engine not ready.
     */
    fun generate(prompt: String, maxTokens: Int = 256): String? {
        val e = engine ?: return null
        return try {
            e.createConversation().use { conv ->
                val response = conv.sendMessage(prompt)
                response
            }
        } catch (ex: Exception) {
            Log.w(TAG, "Generation failed: ${ex.message}")
            null
        }
    }

    /**
     * Generate with streaming — tokens emitted as they're produced.
     * Returns a Flow<String> of token chunks.
     */
    fun generateStreaming(prompt: String): Flow<String> = flow {
        val e = engine ?: return@flow
        e.createConversation().use { conv ->
            conv.sendMessageAsync(prompt).collect { chunk ->
                emit(chunk)
            }
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Download the model file from HuggingFace.
     * Supports resume on interruption.
     */
    suspend fun downloadModel(): Boolean = withContext(Dispatchers.IO) {
        if (isModelDownloaded()) {
            Log.i(TAG, "Model already downloaded")
            return@withContext true
        }

        val partFile = File(context.filesDir, "$MODEL_FILENAME.part")

        try {
            _downloadProgress.value = 0f

            val conn = URL(MODEL_URL).openConnection() as HttpURLConnection
            conn.connectTimeout = 15_000
            conn.readTimeout = 30_000

            val existingBytes = if (partFile.exists()) partFile.length() else 0L
            if (existingBytes > 0) {
                conn.setRequestProperty("Range", "bytes=$existingBytes-")
            }

            conn.connect()

            val totalBytes = if (existingBytes > 0 && conn.responseCode == 206) {
                existingBytes + conn.contentLengthLong
            } else {
                conn.contentLengthLong
            }

            val append = conn.responseCode == 206
            FileOutputStream(partFile, append).use { output ->
                conn.inputStream.use { input ->
                    val buffer = ByteArray(8192)
                    var downloaded = if (append) existingBytes else 0L
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloaded += bytesRead
                        if (totalBytes > 0) {
                            _downloadProgress.value = downloaded.toFloat() / totalBytes
                        }
                    }
                }
            }

            conn.disconnect()

            if (partFile.renameTo(modelFile)) {
                _downloadProgress.value = -1f
                Log.i(TAG, "Model download complete: ${modelFile.absolutePath}")
                true
            } else {
                Log.e(TAG, "Failed to rename part file")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: ${e.message}")
            _downloadProgress.value = -1f
            false
        }
    }

    /** Release the engine. Call from Activity.onDestroy(). */
    fun release() {
        engine?.close()
        engine = null
        _isReady.value = false
    }
}
