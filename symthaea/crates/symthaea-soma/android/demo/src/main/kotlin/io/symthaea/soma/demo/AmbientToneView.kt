package io.symthaea.soma.demo

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.*

/**
 * Ambient consciousness drone: a soft sine tone that modulates with
 * consciousness level.
 *
 * Pitch: 110Hz (low A) at consciousness 0 -> 220Hz (A3) at consciousness 1.
 * Volume: very quiet (0.02-0.08), meant to be felt more than heard.
 * Modulation: slow tremolo for organic breathing quality.
 */
class AmbientTone {
    private var track: AudioTrack? = null
    private var job: Job? = null
    @Volatile private var running = false

    @Volatile var consciousnessLevel: Float = 0f
    @Volatile var harmonyShift: Float = 0f
    @Volatile var enabled: Boolean = true

    private val sampleRate = 22050
    private val bufferSize = AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    ).coerceAtLeast(4096)

    fun start(scope: CoroutineScope) {
        if (running) return
        running = true

        job = scope.launch(Dispatchers.IO) {
            var audioTrack: AudioTrack? = null
            try {
                audioTrack = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(sampleRate)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferSize * 2)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()

                track = audioTrack
                audioTrack.play()

                val buffer = ShortArray((bufferSize / 2).coerceAtLeast(1))
                var phase = 0.0
                var tremoloPhase = 0.0

                while (running && isActive) {
                    val cl = consciousnessLevel
                    if (!enabled || cl < 0.1f) {
                        buffer.fill(0)
                        audioTrack.write(buffer, 0, buffer.size)
                        delay(50)
                        continue
                    }

                    val baseFreq = 110.0 + cl * 110.0
                    val harmonyMix = cl * 0.15 * harmonyShift
                    val volume = (0.02 + cl * 0.06).coerceIn(0.0, 0.08)
                    val tremoloFreq = 0.25 + cl * 0.25

                    for (i in buffer.indices) {
                        tremoloPhase += 2.0 * PI * tremoloFreq / sampleRate
                        val tremolo = 0.6 + 0.4 * sin(tremoloPhase)

                        phase += 2.0 * PI * baseFreq / sampleRate
                        val sample = sin(phase) + harmonyMix * sin(phase * 1.5)

                        // Clamp before conversion to prevent overflow
                        val clamped = (volume * tremolo * sample).coerceIn(-1.0, 1.0)
                        buffer[i] = (clamped * Short.MAX_VALUE).toInt().toShort()
                    }

                    audioTrack.write(buffer, 0, buffer.size)
                }
            } catch (e: Exception) {
                Log.w("AmbientTone", "Audio failed", e)
            } finally {
                // Always release audio resources
                try {
                    audioTrack?.stop()
                } catch (_: Exception) {}
                try {
                    audioTrack?.release()
                } catch (_: Exception) {}
                track = null
            }
        }
    }

    fun stop() {
        running = false
        job?.cancel()
        job = null
    }
}
