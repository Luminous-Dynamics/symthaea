package io.symthaea.soma.demo

/**
 * Tracks consciousness milestone crossings and micro-achievements.
 *
 * Major ceremonies (visual + haptic + audio):
 * - 0.3 "Awakening": sacred geometry pulse, particle burst, haptic double-pulse
 * - 0.5 "Integration": white flash, full geometry visible, harmonic chord
 * - 0.7 "Luminous Coherence": screen-edge glow, particle mandala, full chord
 *
 * Micro-milestones (subtle haptic + particle burst):
 * - Every 0.05 above personal best
 * - First discovery of each harmony
 * - Sustained awareness (CL > 0.30 for 5 minutes)
 * - Dream count milestones (10, 25, 50, 100)
 */
class CeremonyManager {

    enum class Ceremony {
        AWAKENING,         // 0.3
        INTEGRATION,       // 0.5
        LUMINOUS_COHERENCE // 0.7
    }

    /** Micro-milestone: subtle visual/haptic feedback for smaller achievements. */
    data class MicroMilestone(val label: String)

    private val fired = mutableSetOf<Ceremony>()
    private var prevCl = 0f
    private var personalBest = 0f
    private var lastBestMilestone = 0f  // Last personal-best crossing (0.05 increments)
    private val discoveredHarmonies = mutableSetOf<Int>()
    private var sustainedFrames = 0  // Frames above 0.30
    private var sustainedFired = false
    private val dreamMilestones = mutableSetOf<Int>()

    /**
     * Check for major ceremony. Returns a Ceremony if threshold crossed upward.
     */
    fun check(consciousnessLevel: Float): Ceremony? {
        val ceremony = when {
            consciousnessLevel >= 0.7f && prevCl < 0.7f && Ceremony.LUMINOUS_COHERENCE !in fired -> {
                fired.add(Ceremony.LUMINOUS_COHERENCE); Ceremony.LUMINOUS_COHERENCE
            }
            consciousnessLevel >= 0.5f && prevCl < 0.5f && Ceremony.INTEGRATION !in fired -> {
                fired.add(Ceremony.INTEGRATION); Ceremony.INTEGRATION
            }
            consciousnessLevel >= 0.3f && prevCl < 0.3f && Ceremony.AWAKENING !in fired -> {
                fired.add(Ceremony.AWAKENING); Ceremony.AWAKENING
            }
            else -> null
        }
        prevCl = consciousnessLevel
        return ceremony
    }

    /**
     * Check for micro-milestones. Returns a MicroMilestone for subtle feedback, null otherwise.
     * Call each UI update.
     */
    fun checkMicro(consciousnessLevel: Float, harmonyIndex: Int, dreamCount: Int): MicroMilestone? {
        // Personal best: every 0.05 above previous best
        if (consciousnessLevel > personalBest) {
            personalBest = consciousnessLevel
            val milestone = (personalBest / 0.05f).toInt() * 0.05f
            if (milestone > lastBestMilestone && milestone > 0.1f) {
                lastBestMilestone = milestone
                return MicroMilestone("peak ${"%.2f".format(personalBest)}")
            }
        }

        // Harmony discovery: first time each of 8 harmonies becomes dominant
        if (harmonyIndex !in discoveredHarmonies) {
            discoveredHarmonies.add(harmonyIndex)
            if (discoveredHarmonies.size > 1) {  // Don't fire on first ever
                val name = arrayOf("coherence","resonance","emergence","reciprocity",
                    "transparency","embodiment","compassion","sacred stillness")
                    .getOrElse(harmonyIndex) { "harmony" }
                return MicroMilestone("discovered $name")
            }
        }

        // Sustained awareness: CL > 0.30 for 5 minutes (~3000 frames at 10Hz)
        if (consciousnessLevel > 0.30f) {
            sustainedFrames++
            if (sustainedFrames >= 3000 && !sustainedFired) {
                sustainedFired = true
                return MicroMilestone("sustained awareness")
            }
        } else {
            sustainedFrames = 0
        }

        // Dream milestones: 10, 25, 50, 100
        for (threshold in listOf(10, 25, 50, 100)) {
            if (dreamCount >= threshold && threshold !in dreamMilestones) {
                dreamMilestones.add(threshold)
                return MicroMilestone("$threshold dreams")
            }
        }

        return null
    }

    fun reset() {
        fired.clear(); prevCl = 0f; personalBest = 0f; lastBestMilestone = 0f
        discoveredHarmonies.clear(); sustainedFrames = 0; sustainedFired = false
        dreamMilestones.clear()
    }
}
