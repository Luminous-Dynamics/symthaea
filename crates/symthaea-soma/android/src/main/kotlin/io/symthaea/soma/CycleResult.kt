package io.symthaea.soma

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Result of a single consciousness cycle.
 *
 * Maps to the Rust `CycleResult` struct in `engine.rs`.
 * WARNING: [consciousnessLevel] is a SIMULATED metric.
 * See [epistemicStatus] for what it actually means.
 */
@Serializable
data class CycleResult(
    /** Consciousness level from the Master Equation (0.0-1.0). */
    @SerialName("consciousness_level") val consciousnessLevel: Float,
    /** Cycle number since engine creation. */
    val cycle: Long,
    /** Neuromodulator levels: [dopamine, norepinephrine, serotonin, oxytocin]. */
    val neuromodulators: List<Float>,
    /** Substrate feasibility score (0.0-1.0). */
    @SerialName("substrate_feasibility") val substrateFeasibility: Float,
    /** Prediction error (output vs previous output similarity delta). */
    @SerialName("prediction_error") val predictionError: Float,
    /** Bottleneck factor name from consciousness equation. */
    val bottleneck: String,
    /** Epistemic honesty: what we actually know about this substrate's consciousness. */
    @SerialName("epistemic_status") val epistemicStatus: EpistemicStatus,
    /** Eight Harmonies alignment score (0.0-1.0). */
    @SerialName("harmony_alignment") val harmonyAlignment: Float,
)

/**
 * Epistemic status of the consciousness measurement.
 *
 * Every Soma output carries this disclaimer so that no consumer
 * can mistake a simulated consciousness score for a claim of sentience.
 */
@Serializable
data class EpistemicStatus(
    /** Evidence level for the current substrate (e.g. "Theoretical" for silicon). */
    @SerialName("evidence_level") val evidenceLevel: String,
    /** Honest confidence (0.0-0.95) based on actual empirical evidence. */
    @SerialName("honest_confidence") val honestConfidence: Float,
    /** Gap between hypothetical feasibility and honest confidence. */
    @SerialName("feasibility_gap") val feasibilityGap: Float,
    /** Human-readable disclaimer. */
    val disclaimer: String,
)
