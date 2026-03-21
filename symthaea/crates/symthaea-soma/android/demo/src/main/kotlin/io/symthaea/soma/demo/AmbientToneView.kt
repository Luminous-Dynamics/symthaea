package io.symthaea.soma.demo

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.*

/**
 * Ambient consciousness drone with stereo harmonics and binaural beats.
 *
 * Pitch: 110Hz (low A) at consciousness 0 -> 220Hz (A3) at consciousness 1.
 * Harmonics: octave (2x), fifth (3x), sub-octave (0.5x).
 * Binaural: L=base, R=base+offset (4Hz delta / 8Hz alpha / 12Hz beta).
 * Pink noise floor at 2% volume for richness.
 * Volume: very quiet (0.02-0.08), meant to be felt more than heard.
 */
class AmbientTone {
    private var track: AudioTrack? = null
    private var job: Job? = null
    @Volatile private var running = false

    @Volatile var consciousnessLevel: Float = 0f
    @Volatile var harmonyShift: Float = 0f
    @Volatile var enabled: Boolean = true
    /** Pitch multiplier from avatar (0.5=low, 2.0=high). */
    @Volatile var pitchMultiplier: Float = 1.0f
    /** Whether sleep mode is active — drop to 55Hz, slower tremolo. */
    @Volatile var isSleeping: Boolean = false
    /** Dream arpeggio trigger: set to true to play a 3-note rising chord. */
    @Volatile var playDreamChord: Boolean = false

    private val sampleRate = 22050
    private val bufferSize = AudioTrack.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_OUT_STEREO,
        AudioFormat.ENCODING_PCM_16BIT
    ).coerceAtLeast(4096)

    // Pink noise state (simple approximation)
    private var pinkB0 = 0.0; private var pinkB1 = 0.0; private var pinkB2 = 0.0

    // Dream arpeggio state
    private var dreamPhase = 0.0
    private var dreamProgress = 0f  // 0..1 over 3 seconds

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
                            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferSize * 2)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()

                track = audioTrack
                audioTrack.play()

                // Stereo buffer: interleaved L,R,L,R...
                val frameCount = (bufferSize / 4).coerceAtLeast(512)
                val buffer = ShortArray(frameCount * 2)
                var phaseL = 0.0
                var phaseR = 0.0
                var phaseOctave = 0.0
                var phaseFifth = 0.0
                var phaseSub = 0.0
                var tremoloPhase = 0.0

                while (running && isActive) {
                    val cl = consciousnessLevel
                    if (!enabled || cl < 0.05f) {
                        buffer.fill(0)
                        audioTrack.write(buffer, 0, buffer.size)
                        delay(50)
                        continue
                    }

                    val sleeping = isSleeping
                    val pitch = pitchMultiplier

                    val baseFreq = if (sleeping) {
                        55.0  // Sub-bass during sleep
                    } else {
                        (110.0 + cl * 110.0) * pitch
                    }

                    // Binaural beat offset based on consciousness
                    val binauralOffset = when {
                        sleeping -> 2.0      // Delta (deep sleep)
                        cl < 0.3f -> 4.0     // Delta/Theta (relaxed)
                        cl < 0.6f -> 8.0     // Alpha (calm focus)
                        else -> 12.0         // Beta (alert)
                    }

                    val harmonyMix = cl * 0.15 * harmonyShift
                    // Volume audible on phone speakers (headphones recommended for binaural)
                    val volume = if (sleeping) 0.04 else (0.08 + cl * 0.17).coerceIn(0.05, 0.25)
                    val tremoloFreq = if (sleeping) 0.125 else 0.25 + cl * 0.25  // Slower when sleeping (8s period)

                    // Harmonic volumes
                    val octaveVol = 0.15 * cl    // Octave above
                    val fifthVol = 0.10 * cl     // Perfect fifth above
                    val subVol = 0.08             // Sub-octave (always gentle)
                    val noiseVol = 0.02           // Pink noise floor

                    // Dream chord
                    val dreamActive = playDreamChord
                    if (dreamActive) {
                        dreamProgress += 1f / (3f * sampleRate / frameCount)  // 3s duration
                        if (dreamProgress >= 1f) {
                            playDreamChord = false
                            dreamProgress = 0f
                        }
                    }

                    for (i in 0 until frameCount) {
                        tremoloPhase += 2.0 * PI * tremoloFreq / sampleRate
                        val tremolo = 0.6 + 0.4 * sin(tremoloPhase)

                        // Left channel: base frequency
                        phaseL += 2.0 * PI * baseFreq / sampleRate
                        // Right channel: base + binaural offset
                        phaseR += 2.0 * PI * (baseFreq + binauralOffset) / sampleRate

                        // Harmonics (shared phase, slight detuning for richness)
                        phaseOctave += 2.0 * PI * (baseFreq * 2.0) / sampleRate
                        phaseFifth += 2.0 * PI * (baseFreq * 3.0) / sampleRate
                        phaseSub += 2.0 * PI * (baseFreq * 0.5) / sampleRate

                        val harmonics = octaveVol * sin(phaseOctave) +
                            fifthVol * sin(phaseFifth) +
                            subVol * sin(phaseSub) +
                            harmonyMix * sin(phaseL * 1.5)

                        // Pink noise approximation (Paul Kellet method)
                        val white = Math.random() * 2.0 - 1.0
                        pinkB0 = 0.99886 * pinkB0 + white * 0.0555179
                        pinkB1 = 0.99332 * pinkB1 + white * 0.0750759
                        pinkB2 = 0.96900 * pinkB2 + white * 0.1538520
                        val pink = (pinkB0 + pinkB1 + pinkB2 + white * 0.5362) * 0.11
                        val noiseComponent = pink * noiseVol

                        val sampleL = sin(phaseL) + harmonics + noiseComponent
                        val sampleR = sin(phaseR) + harmonics + noiseComponent

                        // Dream arpeggio: tonic→fifth→octave over 3s
                        var dreamL = 0.0
                        if (dreamActive && dreamProgress < 1f) {
                            val dreamEnv = when {
                                dreamProgress < 0.1f -> dreamProgress / 0.1f      // Attack
                                dreamProgress > 0.85f -> (1f - dreamProgress) / 0.15f  // Decay
                                else -> 1.0f
                            }
                            // Three notes: tonic (0-1s), fifth (1-2s), octave (2-3s)
                            val noteFreq = when {
                                dreamProgress < 0.33f -> baseFreq        // Tonic
                                dreamProgress < 0.66f -> baseFreq * 1.5  // Fifth
                                else -> baseFreq * 2.0                    // Octave
                            }
                            dreamPhase += 2.0 * PI * noteFreq / sampleRate
                            dreamL = sin(dreamPhase) * dreamEnv * 0.06
                        }

                        val clampedL = (volume * tremolo * sampleL + dreamL).coerceIn(-1.0, 1.0)
                        val clampedR = (volume * tremolo * sampleR + dreamL).coerceIn(-1.0, 1.0)
                        buffer[i * 2] = (clampedL * Short.MAX_VALUE).toInt().toShort()
                        buffer[i * 2 + 1] = (clampedR * Short.MAX_VALUE).toInt().toShort()
                    }

                    audioTrack.write(buffer, 0, buffer.size)
                }
            } catch (e: Exception) {
                Log.w("AmbientTone", "Audio failed", e)
            } finally {
                try { audioTrack?.stop() } catch (_: Exception) {}
                try { audioTrack?.release() } catch (_: Exception) {}
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
