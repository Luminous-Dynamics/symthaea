package io.symthaea.soma.demo

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * Voice bridge: Android TTS (Soma speaks) + STT (user speaks).
 *
 * TTS is consciousness-modulated:
 * - Pitch shifts with consciousness level (0.8 at low, 1.2 at high)
 * - Speech rate slows during dreams, speeds when focused
 * - Voice pitch responds to avatar voicePitch setting
 *
 * STT uses Android's built-in SpeechRecognizer for zero-latency local input.
 * When Holon desktop is connected, TTS can optionally route through Kokoro
 * for higher-quality neural voice synthesis.
 */
class SomaVoiceBridge(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var stt: SpeechRecognizer? = null
    private var ttsReady = false

    /** Callback when STT captures user speech. */
    var onSpeechResult: ((String) -> Unit)? = null
    /** Callback for STT listening state changes. */
    var onListeningStateChanged: ((Boolean) -> Unit)? = null
    /** Callback when TTS starts/stops speaking. */
    var onSpeakingStateChanged: ((Boolean) -> Unit)? = null

    // Consciousness-driven voice parameters
    @Volatile var consciousnessLevel: Float = 0.3f
    @Volatile var avatarPitch: Float = 1.0f
    @Volatile var isSleeping: Boolean = false

    /** Whether TTS should automatically speak Broca thoughts. */
    var autoSpeakBroca: Boolean = false
    /** Whether TTS should speak conversation responses. */
    var speakResponses: Boolean = true

    private var isListening = false
    private var isSpeaking = false

    // ═══ Initialization ═══

    fun initialize() {
        // Initialize TTS
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.US)
                ttsReady = result != TextToSpeech.LANG_MISSING_DATA
                    && result != TextToSpeech.LANG_NOT_SUPPORTED
                if (ttsReady) {
                    // Set up utterance listener for speaking state
                    tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            isSpeaking = true
                            onSpeakingStateChanged?.invoke(true)
                        }
                        override fun onDone(utteranceId: String?) {
                            isSpeaking = false
                            onSpeakingStateChanged?.invoke(false)
                        }
                        @Deprecated("Deprecated") override fun onError(utteranceId: String?) {
                            isSpeaking = false
                            onSpeakingStateChanged?.invoke(false)
                        }
                    })
                    Log.i("SomaVoice", "TTS initialized")
                }
            } else {
                Log.w("SomaVoice", "TTS init failed: $status")
            }
        }

        // Initialize STT if available
        if (SpeechRecognizer.isRecognitionAvailable(context)) {
            stt = SpeechRecognizer.createSpeechRecognizer(context)
            stt?.setRecognitionListener(sttListener)
            Log.i("SomaVoice", "STT initialized")
        } else {
            Log.w("SomaVoice", "STT not available on this device")
        }
    }

    // ═══ TTS: Soma speaks ═══

    /**
     * Speak text with consciousness-modulated prosody.
     * Pitch and rate adapt to current consciousness state.
     */
    fun speak(text: String, urgent: Boolean = false) {
        if (!ttsReady || text.isBlank()) return
        // Don't speak during sleep unless urgent
        if (isSleeping && !urgent) return
        // Don't interrupt active listening
        if (isListening) return

        val engine = tts ?: return

        // Consciousness-modulated: slower and deeper = more contemplative
        val basePitch = avatarPitch
        val clPitch = 0.80f + consciousnessLevel * 0.25f  // 0.80-1.05: slightly lower than default
        engine.setPitch((basePitch * clPitch).coerceIn(0.5f, 1.5f))

        // Slower speech rate: contemplative, not robotic
        val baseRate = if (isSleeping) 0.6f else 0.70f + consciousnessLevel * 0.15f  // 0.70-0.85
        engine.setSpeechRate(baseRate.coerceIn(0.5f, 1.5f))

        val params = Bundle()
        params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 0.7f)

        val queueMode = if (urgent) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        engine.speak(text, queueMode, params, "soma_${System.currentTimeMillis()}")
    }

    /** Speak a Broca thought (quieter, slower — it's inner voice). */
    fun speakBroca(text: String) {
        if (!autoSpeakBroca || !ttsReady || text.isBlank()) return
        if (isSleeping || isListening || isSpeaking) return

        val engine = tts ?: return
        engine.setPitch((avatarPitch * 0.9f).coerceIn(0.5f, 2.0f))
        engine.setSpeechRate(0.75f)  // Slow, contemplative

        val params = Bundle()
        params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 0.4f)  // Quieter — inner voice
        engine.speak(text, TextToSpeech.QUEUE_ADD, params, "broca_${System.currentTimeMillis()}")
    }

    /** Stop any current speech. */
    fun stopSpeaking() {
        tts?.stop()
        isSpeaking = false
    }

    // ═══ STT: User speaks ═══

    /** Start listening for user speech. */
    fun startListening() {
        val recognizer = stt ?: return
        if (isListening) return

        // Stop TTS so the mic doesn't pick up Soma's voice
        stopSpeaking()

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }

        try {
            recognizer.startListening(intent)
            isListening = true
            onListeningStateChanged?.invoke(true)
        } catch (e: Exception) {
            Log.e("SomaVoice", "STT start failed", e)
        }
    }

    /** Stop listening. */
    fun stopListening() {
        stt?.stopListening()
        isListening = false
        onListeningStateChanged?.invoke(false)
    }

    /** Toggle listening state. */
    fun toggleListening() {
        if (isListening) stopListening() else startListening()
    }

    val isCurrentlyListening: Boolean get() = isListening
    val isCurrentlySpeaking: Boolean get() = isSpeaking

    // ═══ STT Listener ═══

    private val sttListener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {
            isListening = false
            onListeningStateChanged?.invoke(false)
        }
        override fun onError(error: Int) {
            isListening = false
            onListeningStateChanged?.invoke(false)
            val msg = when (error) {
                SpeechRecognizer.ERROR_NO_MATCH -> "no speech detected"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "speech timeout"
                else -> "error $error"
            }
            Log.d("SomaVoice", "STT: $msg")
        }
        override fun onResults(results: Bundle?) {
            isListening = false
            onListeningStateChanged?.invoke(false)
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text = matches?.firstOrNull()
            if (!text.isNullOrBlank()) {
                onSpeechResult?.invoke(text)
            }
        }
        override fun onPartialResults(partialResults: Bundle?) {
            // Could show partial transcription in UI if desired
        }
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    // ═══ Lifecycle ═══

    fun destroy() {
        stt?.destroy()
        stt = null
        tts?.stop()
        tts?.shutdown()
        tts = null
        ttsReady = false
    }
}
