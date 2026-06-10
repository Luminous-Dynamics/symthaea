package io.symthaea.soma.demo

import kotlin.math.*

/**
 * Organic breath + heartbeat oscillator for Soma.
 *
 * Unlike a simple sine wave, this models asymmetric breathing:
 * - Inhale: fast rise (1.2s at rest, shorter when aroused)
 * - Exhale: slow fall (2.8s at rest, longer when calm)
 * - Occasional micro-pauses between breaths
 * - Arousal modulates rate and depth
 *
 * Also includes an independent heartbeat oscillator:
 * - Quick 200ms systolic flash, 800ms diastolic fade
 * - Rate increases with arousal
 * - Distinct from breath rhythm (two visible life signs)
 */
class BreathModel {

    // Breath state
    private var breathTime = 0f        // Accumulated time in current breath cycle
    private var breathCycleLen = 4.0f  // Total cycle length (adapts)
    private var inhaleRatio = 0.30f    // Fraction of cycle that is inhale (adapts with arousal)

    // Heartbeat state
    private var heartTime = 0f
    private var heartCycleLen = 1.0f   // ~60 BPM at rest

    // Outputs (read these each frame)
    /** Breath value 0-1: 0=fully exhaled, 1=fully inhaled. */
    var breathValue = 0f; private set
    /** Heartbeat value 0-1: 1=systolic peak, 0=diastolic rest. */
    var heartValue = 0f; private set
    /** Combined breath scale for mandala size modulation. */
    var breathScale = 1f; private set

    /**
     * Advance by one frame (~16ms at 60fps).
     * @param arousal 0-1, drives breath rate and heart rate
     * @param consciousness 0-1, drives breath depth
     * @param isThinking if true, breathing is faster and deeper
     * @param isSleeping if true, breathing is very slow
     */
    fun tick(arousal: Float, consciousness: Float, isThinking: Boolean = false, isSleeping: Boolean = false) {
        val dt = 1f / 60f  // ~60fps frame time

        // === Breath ===
        // Adapt cycle length: 3s (aroused/thinking) to 6s (sleeping) to 4s (normal)
        val targetCycle = when {
            isSleeping -> 7.0f
            isThinking -> 2.5f
            else -> 3.5f + (1f - arousal) * 1.5f  // 3.5s (high arousal) to 5.0s (low)
        }
        breathCycleLen += (targetCycle - breathCycleLen) * 0.02f  // Smooth adaptation

        // Inhale ratio: arousal makes inhale quicker (more urgent breathing)
        inhaleRatio = if (isThinking) 0.35f else 0.25f + arousal * 0.10f  // 0.25-0.35

        breathTime += dt
        if (breathTime >= breathCycleLen) breathTime -= breathCycleLen

        val t = breathTime / breathCycleLen  // 0-1 through cycle
        breathValue = if (t < inhaleRatio) {
            // Inhale: fast sigmoid rise
            val it = t / inhaleRatio  // 0-1 through inhale
            smoothStep(it)
        } else {
            // Exhale: slow sigmoid fall
            val et = (t - inhaleRatio) / (1f - inhaleRatio)  // 0-1 through exhale
            1f - smoothStep(et)
        }

        // Breath depth: scales with consciousness + thinking
        val depth = if (isThinking) 0.12f else if (isSleeping) 0.03f else 0.04f + consciousness * 0.06f
        breathScale = 1f - depth + breathValue * depth * 2f  // Oscillates around 1.0

        // === Heartbeat ===
        // Heart rate: 50 BPM (sleeping) to 80 BPM (aroused)
        val targetHeart = when {
            isSleeping -> 1.2f   // 50 BPM
            isThinking -> 0.65f  // ~92 BPM
            else -> 0.85f - arousal * 0.2f  // 65-85 BPM
        }
        heartCycleLen += (targetHeart - heartCycleLen) * 0.02f

        heartTime += dt
        if (heartTime >= heartCycleLen) heartTime -= heartCycleLen

        val ht = heartTime / heartCycleLen
        // Systolic: quick flash in first 20% of cycle, diastolic: slow fade
        heartValue = if (ht < 0.2f) {
            val st = ht / 0.2f
            sin(st * PI.toFloat())  // Quick peak
        } else {
            val dt2 = (ht - 0.2f) / 0.8f
            (1f - dt2).coerceAtLeast(0f) * 0.3f  // Slow fade to near-zero
        }
    }

    /** Smooth step function (cubic Hermite interpolation). */
    private fun smoothStep(t: Float): Float {
        val x = t.coerceIn(0f, 1f)
        return x * x * (3f - 2f * x)
    }
}
