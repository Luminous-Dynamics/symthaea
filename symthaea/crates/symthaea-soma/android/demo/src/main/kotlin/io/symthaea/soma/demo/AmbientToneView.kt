package io.symthaea.soma.demo

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Living consciousness drone: formant voice synthesis with breath sync.
 *
 * Instead of a bare sine wave, this produces a vocal "aaah" → "eee" drone
 * that morphs with consciousness level. Three formant filters shape the
 * timbre. Amplitude syncs to the BreathModel so the audio literally
 * breathes with the mandala.
 *
 * Features:
 * - Formant voice: 3 resonant peaks (F1/F2/F3) shift with consciousness
 * - Breath-synced amplitude: swells on inhale, softens on exhale
 * - Binaural beats: L/R offset for brainwave entrainment
 * - Stochastic micro-events: faint pitch bends every 15-30s
 * - Neuromod timbral coloring: DA=rhythmic pulse, NE=high shimmer,
 *   5-HT=warmth, OT=low resonance
 * - Dream soundscape: spectral morph during dream consolidation
 * - Pink noise floor for richness
 */
class AmbientTone {
    private var track: AudioTrack? = null
    private var job: Job? = null
    @Volatile private var running = false

    @Volatile var consciousnessLevel: Float = 0f
    @Volatile var harmonyShift: Float = 0f
    @Volatile var enabled: Boolean = true
    @Volatile var pitchMultiplier: Float = 1.0f
    @Volatile var isSleeping: Boolean = false
    @Volatile var playDreamChord: Boolean = false
    /** Neuromod levels [DA, NE, 5-HT, OT] for timbral coloring. */
    @Volatile var neuromodLevels: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
    /** Breath value (0-1) from BreathModel for amplitude sync. */
    @Volatile var breathValue: Float = 0.5f

    private val sampleRate = 22050
    private val bufferSize = AudioTrack.getMinBufferSize(
        sampleRate, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT
    ).coerceAtLeast(4096)

    // Formant filter state (3 biquad bandpass filters)
    private val formants = arrayOf(FormantFilter(), FormantFilter(), FormantFilter())

    // Stochastic micro-event state
    private var microEventTimer = 0
    private var microEventTarget = randomMicroInterval()
    private var pitchBend = 0.0  // Current pitch bend (-0.02 to +0.02)
    private var pitchBendDecay = 0.0

    // Dream state
    private var dreamPhase = 0.0
    private var dreamProgress = 0f
    private var dreamMorphAmount = 0f  // 0=normal, 1=full dream soundscape

    // Pink noise
    private var pinkB0 = 0.0; private var pinkB1 = 0.0; private var pinkB2 = 0.0

    fun start(scope: CoroutineScope) {
        if (running) return
        running = true
        Log.i("AmbientTone", "start() called, bufferSize=$bufferSize")

        job = scope.launch(Dispatchers.IO) {
            Log.i("AmbientTone", "Coroutine started, building AudioTrack...")
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
                Log.i("AmbientTone", "Audio PLAYING: rate=$sampleRate buf=$bufferSize state=${audioTrack.state}")

                val frameCount = (bufferSize / 4).coerceAtLeast(512)
                val buffer = ShortArray(frameCount * 2)
                var phaseL = 0.0
                var phaseR = 0.0
                var phaseOctave = 0.0
                var phaseFifth = 0.0

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
                    val breath = breathValue
                    val nm = neuromodLevels

                    // Base frequency: 220-440Hz, audible on phone speakers
                    val baseFreq = if (sleeping) 220.0 * pitch
                        else (220.0 + cl * 220.0) * pitch + pitchBend * 220.0

                    // Formant targets: consciousness morphs vowel "uh" → "ah" → "ee"
                    // These are the resonant frequencies that make it sound like a voice
                    val f1 = 300.0 + cl * 400.0   // F1: 300Hz (uh) → 700Hz (ah) → back down
                    val f2 = 870.0 + cl * 1400.0  // F2: 870Hz (uh) → 2270Hz (ee)
                    val f3 = 2240.0 + cl * 800.0   // F3: 2240 → 3040
                    formants[0].setTarget(f1, 0.92, sampleRate)
                    formants[1].setTarget(f2, 0.90, sampleRate)
                    formants[2].setTarget(f3, 0.88, sampleRate)

                    // Binaural offset
                    val binauralOffset = when {
                        sleeping -> 2.0
                        cl < 0.3f -> 4.0
                        cl < 0.6f -> 8.0
                        else -> 12.0
                    }

                    // Volume: breath-synced (swells on inhale, softens on exhale)
                    val baseVol = if (sleeping) 0.12 else (0.20 + cl * 0.35).coerceIn(0.15, 0.55)
                    val breathMod = 0.6 + breath * 0.4  // 0.6-1.0 modulation from breath
                    val volume = baseVol * breathMod

                    // Harmonic volumes
                    val octaveVol = 0.20 * cl
                    val fifthVol = 0.12 * cl
                    val noiseVol = 0.015 + cl * 0.015

                    // Neuromod timbral coloring
                    val daPulse = nm.getOrElse(0) { 0.5f }   // Dopamine: rhythmic brightness
                    val neShimmer = nm.getOrElse(1) { 0.5f }  // NE: high frequency shimmer
                    val htWarmth = nm.getOrElse(2) { 0.5f }   // 5-HT: warm low resonance
                    val otDepth = nm.getOrElse(3) { 0.5f }    // OT: depth/fullness

                    // Stochastic micro-events: pitch bend every 15-30s
                    microEventTimer++
                    if (microEventTimer >= microEventTarget) {
                        microEventTimer = 0
                        microEventTarget = randomMicroInterval()
                        pitchBend = (Random.nextDouble() - 0.5) * 0.04  // ±2% pitch bend
                        pitchBendDecay = pitchBend
                    }
                    // Decay pitch bend smoothly
                    pitchBend *= 0.9998

                    // Dream chord
                    val dreamActive = playDreamChord
                    if (dreamActive) {
                        dreamProgress += 1f / (4f * sampleRate / frameCount)  // 4s duration
                        dreamMorphAmount = (dreamMorphAmount + 0.01f).coerceAtMost(1f)
                        if (dreamProgress >= 1f) {
                            playDreamChord = false
                            dreamProgress = 0f
                        }
                    } else {
                        dreamMorphAmount = (dreamMorphAmount - 0.005f).coerceAtLeast(0f)
                    }

                    for (i in 0 until frameCount) {
                        // Source: sawtooth wave (rich harmonics for formant filtering)
                        phaseL += 2.0 * PI * baseFreq / sampleRate
                        phaseR += 2.0 * PI * (baseFreq + binauralOffset) / sampleRate
                        if (phaseL > 2 * PI) phaseL -= 2 * PI
                        if (phaseR > 2 * PI) phaseR -= 2 * PI

                        // Sawtooth + triangle blend for richer source
                        val sawL = 2.0 * (phaseL / (2 * PI)) - 1.0
                        val sawR = 2.0 * (phaseR / (2 * PI)) - 1.0
                        val triL = 2.0 * abs(sawL) - 1.0
                        val sourceL = sawL * 0.6 + triL * 0.4  // Blend
                        val sourceR = sawR * 0.6 + triL * 0.4

                        // Formant filtering: shape the raw waveform into a vocal timbre
                        val f1out = formants[0].process(sourceL)
                        val f2out = formants[1].process(sourceL)
                        val f3out = formants[2].process(sourceL)
                        val formantMix = f1out * 0.5 + f2out * 0.35 + f3out * 0.15

                        // Harmonics
                        phaseOctave += 2.0 * PI * (baseFreq * 2.0) / sampleRate
                        phaseFifth += 2.0 * PI * (baseFreq * 1.5) / sampleRate
                        val harmonics = octaveVol * sin(phaseOctave) + fifthVol * sin(phaseFifth)

                        // Neuromod coloring
                        val neHighFreq = neShimmer * 0.03 * sin(phaseL * 7.0)  // High shimmer
                        val daRhythm = daPulse * 0.02 * sin(phaseL * 0.1)  // Slow pulse
                        val neuroColor = neHighFreq + daRhythm

                        // Pink noise
                        val white = Random.nextDouble() * 2.0 - 1.0
                        pinkB0 = 0.99886 * pinkB0 + white * 0.0555179
                        pinkB1 = 0.99332 * pinkB1 + white * 0.0750759
                        pinkB2 = 0.96900 * pinkB2 + white * 0.1538520
                        val pink = (pinkB0 + pinkB1 + pinkB2 + white * 0.5362) * 0.11
                        val noise = pink * noiseVol

                        // Mix: formant voice + harmonics + neuromod color + noise
                        val voiceMix = formantMix * (0.7 + htWarmth * 0.3) * (0.8 + otDepth * 0.2)
                        val mixL = voiceMix + harmonics + neuroColor + noise
                        val mixR = voiceMix * 0.9 + sin(phaseR) * 0.1 + harmonics + neuroColor + noise

                        // Dream morph: blend toward shimmering pad
                        var finalL = mixL
                        var finalR = mixR
                        if (dreamMorphAmount > 0.01) {
                            val dm = dreamMorphAmount.toDouble()
                            // Shimmer pad: multiple detuned sines
                            val shimmer = 0.3 * (
                                sin(phaseL * 1.002) + sin(phaseL * 0.998) +
                                sin(phaseL * 2.003) * 0.5 + sin(phaseL * 1.501) * 0.4
                            )
                            finalL = mixL * (1 - dm) + shimmer * dm
                            finalR = mixR * (1 - dm) + shimmer * dm

                            // Dream arpeggio on top
                            if (dreamActive && dreamProgress < 1f) {
                                val env = when {
                                    dreamProgress < 0.1f -> (dreamProgress / 0.1f).toDouble()
                                    dreamProgress > 0.85f -> ((1f - dreamProgress) / 0.15f).toDouble()
                                    else -> 1.0
                                }
                                val noteFreq = when {
                                    dreamProgress < 0.33f -> baseFreq
                                    dreamProgress < 0.66f -> baseFreq * 1.5
                                    else -> baseFreq * 2.0
                                }
                                dreamPhase += 2.0 * PI * noteFreq / sampleRate
                                val dreamNote = sin(dreamPhase) * env * 0.12
                                finalL += dreamNote
                                finalR += dreamNote
                            }
                        }

                        val clampedL = (volume * finalL).coerceIn(-1.0, 1.0)
                        val clampedR = (volume * finalR).coerceIn(-1.0, 1.0)
                        buffer[i * 2] = (clampedL * Short.MAX_VALUE).toInt().toShort()
                        buffer[i * 2 + 1] = (clampedR * Short.MAX_VALUE).toInt().toShort()
                    }

                    audioTrack.write(buffer, 0, buffer.size)
                }
            } catch (e: Exception) {
                Log.e("AmbientTone", "Audio FAILED: ${e.message}", e)
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

    private fun randomMicroInterval(): Int {
        // 15-30 seconds at ~22050/frameCount Hz buffer writes
        val frameCount = (bufferSize / 4).coerceAtLeast(512)
        val writesPerSecond = sampleRate.toFloat() / frameCount
        return ((15 + Random.nextFloat() * 15) * writesPerSecond).toInt()
    }

    /**
     * Simple biquad bandpass filter for formant synthesis.
     * Each formant is a resonant peak at a target frequency.
     */
    private class FormantFilter {
        private var a0 = 1.0; private var a1 = 0.0; private var a2 = 0.0
        private var b1 = 0.0; private var b2 = 0.0
        private var x1 = 0.0; private var x2 = 0.0
        private var y1 = 0.0; private var y2 = 0.0

        fun setTarget(freq: Double, q: Double, sampleRate: Int) {
            val w0 = 2.0 * PI * freq / sampleRate
            val alpha = sin(w0) / (2.0 * q / (1.0 - q))  // bandwidth from Q
            val cosW0 = cos(w0)
            val norm = 1.0 + alpha
            a0 = alpha / norm
            a1 = 0.0
            a2 = -alpha / norm
            b1 = -2.0 * cosW0 / norm
            b2 = (1.0 - alpha) / norm
        }

        fun process(input: Double): Double {
            val output = a0 * input + a1 * x1 + a2 * x2 - b1 * y1 - b2 * y2
            x2 = x1; x1 = input
            y2 = y1; y1 = output
            return output
        }
    }
}
