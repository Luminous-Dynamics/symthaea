package io.symthaea.spore

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Configuration for the Spore consciousness kernel.
 *
 * Maps to the Rust `SporeConfig` struct in `config.rs`.
 * Default values match the Rust defaults (full-fidelity desktop config).
 * For mobile, use [SporeEngine.createMobile] or set reduced values.
 */
@Serializable
data class SporeConfig(
    /** HDC dimension (default: 16,384 = full fidelity). Minimum 4,096 recommended. */
    @SerialName("hdc_dim") val hdcDim: Int = 16_384,
    /** Number of CfC neurons per layer. */
    @SerialName("neurons_per_layer") val neuronsPerLayer: Int = 64,
    /** Number of network layers. */
    @SerialName("network_layers") val networkLayers: Int = 3,
    /** Compute Phi every N cycles (1=every, 3=amortized). */
    @SerialName("phi_every_n_cycles") val phiEveryNCycles: Int = 1,
    /** Substrate type name (e.g. "SiliconDigital", "BiologicalNeurons"). */
    val substrate: String = "SiliconDigital",
    /** Target cycle rate in Hz. */
    @SerialName("target_hz") val targetHz: Float = 50f,
)
