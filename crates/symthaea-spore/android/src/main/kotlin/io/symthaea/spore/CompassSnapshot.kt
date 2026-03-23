package io.symthaea.spore

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Compact consciousness snapshot for widget / notification rendering.
 *
 * Maps to the Rust `CompassSnapshot` struct in `compass.rs`.
 */
@Serializable
data class CompassSnapshot(
    /** Consciousness level (0.0-1.0). */
    @SerialName("consciousness_level") val consciousnessLevel: Float,
    /** Name of the dominant Eight Harmony. */
    @SerialName("dominant_harmony") val dominantHarmony: String,
    /** [dopamine, norepinephrine, serotonin, oxytocin] normalized 0-1. */
    @SerialName("neuromod_balance") val neuromodBalance: List<Float>,
    /** Wake state code (0=Sleep, 1=Drowsy, 2=Alert, 3=Focused). */
    @SerialName("wake_state") val wakeState: Int,
    /** Motion state code (0=Stationary, 1=Walking, 2=Running, 3=InVehicle). */
    @SerialName("motion_state") val motionState: Int,
    /** Whether privacy mode is active. */
    @SerialName("privacy_mode") val privacyMode: Boolean,
    /** Emotional valence (-1.0 to 1.0). Derived from DA + OT. */
    val valence: Float,
    /** Arousal level (0.0-1.0). Derived from NE. */
    val arousal: Float,
    /** Number of stored dream fragments. */
    @SerialName("dream_count") val dreamCount: Int,
    /** Number of wisdom entries generated. */
    @SerialName("wisdom_count") val wisdomCount: Int,
    /** Current cycle number. */
    val cycle: Long,
    /** Snapshot timestamp (epoch milliseconds). */
    @SerialName("timestamp_epoch_ms") val timestampEpochMs: Long,
)
