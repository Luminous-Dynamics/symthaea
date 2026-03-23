package io.symthaea.spore

import android.media.MediaRecorder
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges ambient sound level to the Spore engine.
 *
 * Samples ambient dB via [MediaRecorder.getMaxAmplitude]. Only amplitude
 * is captured — no audio content is recorded or stored (sovereign privacy).
 *
 * Requires `RECORD_AUDIO` permission (requested at runtime by the activity).
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SporeAudioBridge(engine))
 * ```
 */
class SporeAudioBridge(
    private val engine: SporeEngine,
) : DefaultLifecycleObserver {

    private var recorder: MediaRecorder? = null

    override fun onStart(owner: LifecycleOwner) {
        try {
            val mr = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                setOutputFile("/dev/null") // Discard audio — only amplitude matters
                prepare()
                start()
            }
            recorder = mr
        } catch (e: Exception) {
            Log.w(TAG, "Audio bridge unavailable (missing permission?): ${e.message}")
            recorder = null
        }
    }

    override fun onStop(owner: LifecycleOwner) {
        try {
            recorder?.stop()
            recorder?.release()
        } catch (_: Exception) {}
        recorder = null
    }

    /** Sample ambient dB and push to engine. Call at ~2Hz. */
    fun tick() {
        val mr = recorder ?: return
        try {
            val amplitude = mr.maxAmplitude
            if (amplitude > 0) {
                // Convert amplitude to approximate dB (20 * log10(amp / ref))
                val db = 20.0 * kotlin.math.log10(amplitude.toDouble()).coerceIn(0.0, 120.0)
                engine.setAmbientDb(db.toFloat())
            }
        } catch (e: Exception) {
            Log.w(TAG, "Audio sample failed: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "SporeAudio"
    }
}
