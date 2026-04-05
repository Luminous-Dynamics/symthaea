package io.symthaea.soma.demo

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Manages on-device Gemma 4 E2B model via LiteRT-LM.
 *
 * Lifecycle:
 *   1. Check if model exists on disk (~2.58 GB)
 *   2. If not, download from HuggingFace (WiFi only)
 *   3. Initialize native engine via JNI → litert_shim.cpp
 *   4. Provide generate() for on-device inference
 *
 * Falls back to OllamaBridge (network LLM) if model is not downloaded.
 */
class LiteRTManager(private val context: Context) {

    companion object {
        private const val TAG = "LiteRTManager"
        private const val MODEL_FILENAME = "gemma4-e2b.litertlm"
        private const val MODEL_URL =
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-litert-lm.task"
        private const val MODEL_SIZE_BYTES = 2_580_000_000L // ~2.58 GB
    }

    /** Download progress (0.0 to 1.0), or -1 if not downloading. */
    private val _downloadProgress = MutableStateFlow(-1f)
    val downloadProgress: StateFlow<Float> = _downloadProgress

    /** Whether the on-device model is ready for inference. */
    private val _isReady = MutableStateFlow(false)
    val isReady: StateFlow<Boolean> = _isReady

    /** Path to the model file in app-private storage. */
    private val modelFile: File
        get() = File(context.filesDir, MODEL_FILENAME)

    /** Whether the model file exists and looks complete. */
    fun isModelDownloaded(): Boolean {
        val file = modelFile
        return file.exists() && file.length() > MODEL_SIZE_BYTES * 0.95
    }

    /**
     * Initialize the native LiteRT-LM engine.
     * Call after model download completes or on startup if model exists.
     * Returns true if engine is ready.
     */
    fun initEngine(): Boolean {
        if (!isModelDownloaded()) {
            Log.w(TAG, "Model not downloaded yet — cannot initialize engine")
            return false
        }

        val success = nativeInitEngine(modelFile.absolutePath, true /* use GPU */)
        _isReady.value = success

        if (success) {
            Log.i(TAG, "LiteRT-LM engine initialized (gemma4:e2b on-device, GPU backend)")
        } else {
            Log.e(TAG, "Failed to initialize LiteRT-LM engine")
        }

        return success
    }

    /**
     * Generate text on-device. Returns null if engine is not ready.
     */
    fun generate(prompt: String, maxTokens: Int = 256): String? {
        if (!_isReady.value) return null
        return nativeGenerate(prompt, maxTokens)
    }

    /**
     * Download the model file from HuggingFace.
     * Should be called on a background thread (e.g., via viewModelScope).
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

            // Resume support
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

            // Rename .part to final filename
            if (partFile.renameTo(modelFile)) {
                _downloadProgress.value = -1f
                Log.i(TAG, "Model download complete: ${modelFile.absolutePath}")
                return@withContext true
            } else {
                Log.e(TAG, "Failed to rename part file to final model file")
                return@withContext false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Model download failed: ${e.message}")
            _downloadProgress.value = -1f
            return@withContext false
        }
    }

    /** Release the native engine. Call from Activity.onDestroy(). */
    fun release() {
        nativeReleaseEngine()
        _isReady.value = false
    }

    // ─────────────────────────────────────────────────────────────────
    // JNI bindings → litert_shim.cpp → LiteRT-LM C++ API
    // ───────────────────────────��─────────────────────────────────────

    private external fun nativeInitEngine(modelPath: String, useGpu: Boolean): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String?
    private external fun nativeReleaseEngine()

    init {
        try {
            System.loadLibrary("litert_shim")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "litert_shim native library not available — on-device LLM disabled")
        }
    }
}
