package io.symthaea.soma

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Controls all outbound sharing. Sovereign privacy by default.
 *
 * Maps to the Rust `SharingConfig` struct in `config.rs`.
 * Mode strings must match Rust enum variant names exactly:
 * - holon_mode: "Off", "PresenceOnly", "FullSync"
 * - ble_mode: "Off", "PassiveDiscover", "ActiveShare"
 */
@Serializable
data class SharingConfig(
    /** Holon sync mode: "Off", "PresenceOnly", or "FullSync". */
    @SerialName("holon_mode") val holonMode: String = "Off",
    /** BLE mesh mode: "Off", "PassiveDiscover", or "ActiveShare". */
    @SerialName("ble_mode") val bleMode: String = "Off",
    /** Whether haptic events are generated. */
    @SerialName("haptic_enabled") val hapticEnabled: Boolean = true,
    /** Whether telemetry export is enabled. */
    @SerialName("telemetry_export") val telemetryExport: Boolean = false,
)
