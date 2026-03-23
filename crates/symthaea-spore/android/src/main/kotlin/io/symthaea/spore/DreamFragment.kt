package io.symthaea.spore

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A single dream narrative fragment from overnight consolidation.
 *
 * Maps to the Rust `DreamFragment` struct in `dream_journal.rs`.
 */
@Serializable
data class DreamFragment(
    /** BrocaLite-generated dream narrative text. */
    val narrative: String,
    /** Phi improvement from the consolidated wisdom. */
    @SerialName("phi_improvement") val phiImprovement: Float,
    /** Dominant emotion derived from neuromodulator state. */
    @SerialName("dominant_emotion") val dominantEmotion: String,
    /** Cycle number when this fragment was recorded. */
    @SerialName("cycle_recorded") val cycleRecorded: Long,
)
