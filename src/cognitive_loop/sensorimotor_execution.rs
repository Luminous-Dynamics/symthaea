// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sensorimotor Execution — extracted from CognitiveLoopService
//!
//! Groups the 6 (+ 3 feature-gated) sensorimotor subsystems that were
//! previously individual fields on `CognitiveLoopService`:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `vision_sensory` | `VisionAndSensoryManager` | Vision pipeline, coherence field, virtual body, foveation |
//! | `motor_rendering` | `MotorRenderingManager` | Motor output bridge, canvas, pending requests |
//! | `somatic_bridge` | `SomaticErrorBridge` | Infrastructure errors → felt interoceptive signals |
//! | `pain_tx` | `Option<PainSender>` | Pain channel sender for subsystem error reporting |
//! | `thermal_bridge` | `ThermalBridge` | Platform thermal state → CfC tau modulation |
//! | `thermal_tx` | `Option<ThermalSender>` | Thermal channel sender for platform integration |
//! | `embodiment_bridge` | `Option<Box<dyn EmbodimentBridge>>` | Physical motor control (humanoid) |
//! | `last_proprioceptive_hv` | `Option<ContinuousHV>` | Last proprioceptive HV (humanoid) |
//! | `embodiment_telemetry` | `EmbodimentTelemetry` | Embodiment telemetry (humanoid) |
//!
//! ## Design
//!
//! - **Zero behavior change**: Pure structural refactor, same field paths via delegation.
//! - **Public API preserved**: All accessor methods on `CognitiveLoopService` still work.
//! - **Field access**: Internal code accesses via `self.sensorimotor.field_name`.

use super::motor_rendering_manager;
use super::vision_sensory_manager;

/// Groups 6 sensorimotor subsystems (+ 3 feature-gated embodiment fields).
///
/// Extracted from `CognitiveLoopService` to reduce its field count
/// and provide a cohesive sensorimotor execution boundary.
pub(crate) struct SensoriMotorExecution {
    /// Vision & sensory: coherence field, virtual body, vision, foveation.
    pub vision_sensory: vision_sensory_manager::VisionAndSensoryManager,

    /// Motor rendering: output bridge, pending request, last result, phi, canvas.
    pub motor_rendering: motor_rendering_manager::MotorRenderingManager,

    /// Somatic error bridge: converts infrastructure failures into felt stress.
    /// Lock poisoning, task panics, DB errors → arousal, thermodynamic load, tau slowdown.
    pub somatic_bridge: crate::infrastructure::somatic_error_bridge::SomaticErrorBridge,

    /// Pain channel sender for distributing to subsystems.
    /// Subsystems clone this to report infrastructure errors.
    pub pain_tx: Option<crate::infrastructure::somatic_error_bridge::PainSender>,

    /// Thermal bridge: converts platform thermal state into CfC tau modulation.
    /// Hardware heat → tau slowdown → slower integration → less heat generated.
    /// Science: Angilletta (2009) thermal performance curves.
    pub thermal_bridge: crate::infrastructure::thermal_bridge::ThermalBridge,

    /// Thermal channel sender for platform integration code.
    /// Android PowerManager / iOS ProcessInfo / Linux sysfs thermal zones.
    pub thermal_tx: Option<crate::infrastructure::thermal_bridge::ThermalSender>,

    /// Embodiment bridge: physical motor control with proprioceptive feedback.
    /// When `Some`, each cycle steps the bridge, blending proprioceptive HV
    /// into the next cycle's perception at `embodiment_blend_weight`.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "phone"
    ))]
    pub embodiment_bridge: Option<Box<dyn symthaea_core::embodiment::EmbodimentBridge>>,

    /// Last proprioceptive HV from the embodiment bridge.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "phone"
    ))]
    pub last_proprioceptive_hv: Option<symthaea_core::hdc::ContinuousHV>,

    /// Embodiment telemetry from the most recent step.
    #[cfg(any(
        feature = "humanoid",
        feature = "helicopter",
        feature = "flight",
        feature = "vehicle",
        feature = "auv",
        feature = "manipulator",
        feature = "exoskeleton",
        feature = "surgical",
        feature = "orbital",
        feature = "quadruped",
        feature = "phone"
    ))]
    pub embodiment_telemetry: symthaea_core::embodiment::EmbodimentTelemetry,
}

impl SensoriMotorExecution {
    /// Construct from individually built components.
    ///
    /// Called from `CognitiveLoopService::new()` after each subsystem is created.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vision_sensory: vision_sensory_manager::VisionAndSensoryManager,
        motor_rendering: motor_rendering_manager::MotorRenderingManager,
        somatic_bridge: crate::infrastructure::somatic_error_bridge::SomaticErrorBridge,
        pain_tx: Option<crate::infrastructure::somatic_error_bridge::PainSender>,
        thermal_bridge: crate::infrastructure::thermal_bridge::ThermalBridge,
        thermal_tx: Option<crate::infrastructure::thermal_bridge::ThermalSender>,
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "phone"
        ))]
        embodiment_bridge: Option<Box<dyn symthaea_core::embodiment::EmbodimentBridge>>,
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "phone"
        ))]
        last_proprioceptive_hv: Option<symthaea_core::hdc::ContinuousHV>,
        #[cfg(any(
            feature = "humanoid",
            feature = "helicopter",
            feature = "flight",
            feature = "vehicle",
            feature = "auv",
            feature = "manipulator",
            feature = "exoskeleton",
            feature = "surgical",
            feature = "orbital",
            feature = "quadruped",
            feature = "phone"
        ))]
        embodiment_telemetry: symthaea_core::embodiment::EmbodimentTelemetry,
    ) -> Self {
        Self {
            vision_sensory,
            motor_rendering,
            somatic_bridge,
            pain_tx,
            thermal_bridge,
            thermal_tx,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "phone"
            ))]
            embodiment_bridge,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "phone"
            ))]
            last_proprioceptive_hv,
            #[cfg(any(
                feature = "humanoid",
                feature = "helicopter",
                feature = "flight",
                feature = "vehicle",
                feature = "auv",
                feature = "manipulator",
                feature = "exoskeleton",
                feature = "surgical",
                feature = "orbital",
                feature = "quadruped",
                feature = "phone"
            ))]
            embodiment_telemetry,
        }
    }
}
