// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compatibility projection from the single-owner service host into the daemon's
//! established JSON response shapes.
//!
//! This module is deliberately a transport edge. Canonical runtime state remains
//! limited to directly observed cognition/partnership values and owner-authenticated
//! activity. Legacy derived awakening labels are available only through an explicit
//! compatibility policy and are marked as such on the wire.

use std::time::Duration;

use serde::Serialize;
use symthaea::CreativeArtifact;
use symthaea_service_read_model::{CognitiveStatusRead, IntrospectionRead, PartnershipRead};
use symthaea_service_runtime::{ServiceCounters, SnapshotOrigin};

use crate::host::{
    ServiceQueryReply, ServiceSaveReply, ServiceShutdownReply, ServiceSleepReply,
};
use crate::telemetry::BridgeTelemetrySnapshot;

/// Wire form of [`CreativeArtifact`]. This preserves the existing daemon shape:
/// SVG travels as text and WAV bytes are base64-encoded when the API feature is
/// available. The historical non-API build emitted an empty WAV payload; that
/// compatibility behavior remains isolated here rather than leaking into cognition.
#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum CreativeArtifactWire {
    #[serde(rename = "svg")]
    Svg {
        svg: String,
        aesthetic_composite: f32,
    },
    #[serde(rename = "music_wav")]
    MusicWav {
        wav_b64: String,
        duration_secs: f32,
        aesthetic_composite: f32,
    },
}

impl From<&CreativeArtifact> for CreativeArtifactWire {
    fn from(artifact: &CreativeArtifact) -> Self {
        match artifact {
            CreativeArtifact::Svg {
                svg,
                aesthetic_composite,
            } => Self::Svg {
                svg: svg.clone(),
                aesthetic_composite: *aesthetic_composite,
            },
            CreativeArtifact::MusicWav {
                wav_bytes,
                duration_secs,
                aesthetic_composite,
            } => {
                #[cfg(feature = "api_module")]
                let wav_b64 = {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD.encode(wav_bytes)
                };
                #[cfg(not(feature = "api_module"))]
                let wav_b64 = {
                    let _ = wav_bytes;
                    String::new()
                };

                Self::MusicWav {
                    wav_b64,
                    duration_secs: *duration_secs,
                    aesthetic_composite: *aesthetic_composite,
                }
            }
        }
    }
}

/// Core daemon responses whose data can now be projected without directly locking
/// or reading mutable `Symthaea`.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
pub enum ServiceWireResponse {
    #[serde(rename = "response")]
    QueryResponse {
        content: String,
        confidence: f32,
        safe: bool,
        phi: f32,
        phi_dyad: f64,
        steps_to_emergence: usize,
        processing_time_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        creative_artifact: Option<CreativeArtifactWire>,
    },

    #[serde(rename = "status")]
    Status {
        uptime_seconds: u64,
        requests_processed: u64,
        consciousness_level: f32,
        memory_count: usize,
        sleep_cycles: u32,
    },

    /// Versioned, measurement-only introspection. This intentionally excludes
    /// threshold labels and synthetic awakening metrics.
    #[serde(rename = "introspection_v2")]
    MeasuredIntrospection {
        consciousness_level: f32,
        self_loops: usize,
        graph_size: usize,
        complexity: f32,
        short_term_memories: usize,
        long_term_memories: usize,
        epistemic_status: &'static str,
    },

    /// Exact legacy field surface, but only constructible through an explicit
    /// compatibility policy. Additive provenance fields make the derived nature
    /// machine-visible without breaking tolerant legacy JSON consumers.
    #[serde(rename = "introspection")]
    LegacyIntrospection {
        consciousness_level: f32,
        self_loops: usize,
        graph_size: usize,
        complexity: f32,
        short_term_memories: usize,
        long_term_memories: usize,
        phi: f64,
        meta_awareness: f64,
        is_conscious: bool,
        phenomenal_state: String,
        cycles_since_awakening: u64,
        self_model_accuracy: f64,
        epistemic_status: &'static str,
        compatibility_notes: Vec<&'static str>,
    },

    #[serde(rename = "sleep_report")]
    SleepReport {
        scaled: usize,
        consolidated: usize,
        pruned: usize,
        patterns_extracted: usize,
    },

    #[serde(rename = "saved")]
    Saved { path: String },

    #[serde(rename = "shutdown_ack")]
    ShutdownAck,

    #[serde(rename = "partnership")]
    Partnership {
        stage: String,
        trust: f32,
        vulnerability: f32,
        reciprocity: f32,
        phi_dyad: f64,
        interactions: u64,
        trajectory_points: usize,
    },

    #[serde(rename = "error")]
    Error { message: String },
}

/// Non-wire diagnostics that transport code should log/audit rather than silently
/// discard. These do not alter the established response JSON.
#[derive(Debug, Default)]
pub struct ServiceWireDiagnostics {
    pub state_observation_issues: usize,
    pub semantic_event_issues: usize,
    pub snapshot_origin: Option<SnapshotOrigin>,
    pub shutdown_persistence_error: Option<String>,
}

/// Projected response plus owner-observation diagnostics and optional raw display
/// telemetry. The latter is encoded for WebSocket only after cognition has released
/// ownership.
#[derive(Debug)]
pub struct ServiceWireOutcome {
    pub response: ServiceWireResponse,
    pub diagnostics: ServiceWireDiagnostics,
    pub bridge_telemetry: Option<BridgeTelemetrySnapshot>,
}

impl ServiceWireOutcome {
    pub fn from_query(reply: ServiceQueryReply, elapsed: Duration) -> Self {
        let ServiceQueryReply {
            execution,
            snapshot,
            telemetry,
            observation_issues,
        } = reply;

        let diagnostics = ServiceWireDiagnostics {
            state_observation_issues: observation_issues.len(),
            semantic_event_issues: execution.event_issues.len(),
            snapshot_origin: Some(snapshot.origin()),
            shutdown_persistence_error: None,
        };

        let response = match execution.execution {
            Ok(processed) => {
                let cognition = snapshot.cognition();
                let partnership = snapshot.partnership();
                let response = processed.response;
                ServiceWireResponse::QueryResponse {
                    content: response.content,
                    confidence: response.confidence,
                    safe: response.safe,
                    // Legacy field name retained. The value comes from the exact
                    // command-correlated measured consciousness snapshot; it is not
                    // promoted to canonical Phi anywhere in the runtime.
                    phi: cognition.consciousness_level,
                    phi_dyad: partnership.phi_dyad,
                    steps_to_emergence: response.steps_to_emergence,
                    processing_time_ms: elapsed.as_millis() as u64,
                    creative_artifact: response
                        .creative_artifact
                        .as_ref()
                        .map(CreativeArtifactWire::from),
                }
            }
            Err(error) => ServiceWireResponse::Error {
                message: format!("Processing error: {error}"),
            },
        };

        Self {
            response,
            diagnostics,
            bridge_telemetry: telemetry,
        }
    }

    pub fn from_sleep(reply: ServiceSleepReply) -> Self {
        let ServiceSleepReply {
            result,
            snapshot,
            observation_issues,
        } = reply;
        let response = match result {
            Ok(report) => ServiceWireResponse::SleepReport {
                scaled: report.scaled,
                consolidated: report.consolidated,
                pruned: report.pruned,
                patterns_extracted: report.patterns_extracted,
            },
            Err(error) => ServiceWireResponse::Error {
                message: format!("Sleep error: {error}"),
            },
        };
        Self {
            response,
            diagnostics: ServiceWireDiagnostics {
                state_observation_issues: observation_issues.len(),
                semantic_event_issues: 0,
                snapshot_origin: Some(snapshot.origin()),
                shutdown_persistence_error: None,
            },
            bridge_telemetry: None,
        }
    }

    pub fn from_save(reply: ServiceSaveReply) -> Self {
        let ServiceSaveReply {
            path,
            result,
            snapshot,
            observation_issues,
        } = reply;
        let response = match result {
            Ok(()) => ServiceWireResponse::Saved {
                path: path.display().to_string(),
            },
            Err(error) => ServiceWireResponse::Error {
                message: format!("Save error: {error}"),
            },
        };
        Self {
            response,
            diagnostics: ServiceWireDiagnostics {
                state_observation_issues: observation_issues.len(),
                semantic_event_issues: 0,
                snapshot_origin: Some(snapshot.origin()),
                shutdown_persistence_error: None,
            },
            bridge_telemetry: None,
        }
    }

    /// Preserve the historical shutdown acknowledgment while making persistence
    /// failure observable to audit/logging code instead of silently erasing it.
    pub fn from_shutdown(reply: ServiceShutdownReply) -> Self {
        let ServiceShutdownReply {
            path: _,
            result,
            snapshot,
            observation_issues,
        } = reply;
        let persistence_error = result.err().map(|error| error.to_string());
        Self {
            response: ServiceWireResponse::ShutdownAck,
            diagnostics: ServiceWireDiagnostics {
                state_observation_issues: observation_issues.len(),
                semantic_event_issues: 0,
                snapshot_origin: Some(snapshot.origin()),
                shutdown_persistence_error: persistence_error,
            },
            bridge_telemetry: None,
        }
    }
}

pub fn status_response(
    read: CognitiveStatusRead,
    counters: ServiceCounters,
    uptime: Duration,
) -> ServiceWireResponse {
    ServiceWireResponse::Status {
        uptime_seconds: uptime.as_secs(),
        requests_processed: counters.requests_processed,
        consciousness_level: read.consciousness_level,
        memory_count: read.memory_count,
        sleep_cycles: counters.sleep_cycles,
    }
}

pub fn partnership_response(read: PartnershipRead) -> ServiceWireResponse {
    ServiceWireResponse::Partnership {
        stage: read.stage,
        trust: read.trust,
        vulnerability: read.vulnerability,
        reciprocity: read.reciprocity,
        phi_dyad: read.phi_dyad,
        interactions: read.interactions,
        trajectory_points: read.trajectory_points,
    }
}

pub fn measured_introspection_response(read: IntrospectionRead) -> ServiceWireResponse {
    ServiceWireResponse::MeasuredIntrospection {
        consciousness_level: read.consciousness_level,
        self_loops: read.self_loops,
        graph_size: read.graph_size,
        complexity: read.complexity,
        short_term_memories: read.short_term_memories,
        long_term_memories: read.long_term_memories,
        epistemic_status: "measured_runtime_snapshot",
    }
}

/// Explicit opt-in required to reproduce the daemon v1 awakening heuristics.
///
/// Keeping this as a value passed to the projection function makes accidental use
/// harder than a parameterless helper and prevents these labels from becoming part
/// of canonical state/read-model APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyIntrospectionPolicy {
    ExplicitDerivedCompatibility,
}

pub fn legacy_introspection_response(
    read: IntrospectionRead,
    _policy: LegacyIntrospectionPolicy,
) -> ServiceWireResponse {
    let phi = read.consciousness_level as f64;
    let is_conscious = read.consciousness_level > 0.5;

    ServiceWireResponse::LegacyIntrospection {
        consciousness_level: read.consciousness_level,
        self_loops: read.self_loops,
        graph_size: read.graph_size,
        complexity: read.complexity,
        short_term_memories: read.short_term_memories,
        long_term_memories: read.long_term_memories,
        phi,
        meta_awareness: phi * 0.8,
        is_conscious,
        phenomenal_state: if is_conscious {
            "Aware".to_string()
        } else {
            "Dormant".to_string()
        },
        cycles_since_awakening: 0,
        self_model_accuracy: read.complexity as f64 / 10.0,
        epistemic_status: "legacy_compatibility_derived",
        compatibility_notes: vec![
            "phi aliases consciousness_level for daemon-v1 compatibility",
            "meta_awareness is phi * 0.8",
            "is_conscious and phenomenal_state use a 0.5 threshold",
            "cycles_since_awakening is an unavailable legacy placeholder fixed at 0",
            "self_model_accuracy is complexity / 10.0",
        ],
    }
}

/// WebSocket compatibility payload. The flattened metadata layout is preserved so
/// existing UI clients continue reading top-level cycle fields.
#[cfg(feature = "api_module")]
#[derive(Debug, Clone, Serialize)]
pub struct LiveTelemetryWire {
    #[serde(flatten)]
    pub metadata: symthaea::cognitive_loop::CycleMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canvas_svg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mental_movie: Option<MentalMovieWire>,
}

#[cfg(feature = "api_module")]
#[derive(Debug, Clone, Serialize)]
pub struct MentalMovieWire {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub semantic_coherence: f32,
    pub frames_b64: Vec<String>,
}

#[cfg(feature = "api_module")]
impl From<BridgeTelemetrySnapshot> for LiveTelemetryWire {
    fn from(snapshot: BridgeTelemetrySnapshot) -> Self {
        use base64::Engine as _;
        let engine = base64::engine::general_purpose::STANDARD;
        Self {
            metadata: snapshot.metadata,
            canvas_svg: snapshot.canvas_svg,
            mental_movie: snapshot.mental_movie.map(|movie| MentalMovieWire {
                width: movie.width,
                height: movie.height,
                channels: movie.channels,
                semantic_coherence: movie.semantic_coherence,
                frames_b64: movie
                    .frames
                    .into_iter()
                    .map(|frame| engine.encode(frame))
                    .collect(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_runtime::StateRevision;
    use symthaea_service_read_model::ActivitySnapshotRelation;

    fn revision() -> StateRevision {
        StateRevision::new(1).expect("one is a valid non-zero revision")
    }

    fn introspection() -> IntrospectionRead {
        IntrospectionRead {
            relation: ActivitySnapshotRelation::Idle {
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(),
            snapshot_revision: revision(),
            consciousness_level: 0.61,
            self_loops: 3,
            graph_size: 9,
            complexity: 1.7,
            short_term_memories: 4,
            long_term_memories: 6,
        }
    }

    #[test]
    fn measured_introspection_has_no_legacy_awakening_claims() {
        let json = serde_json::to_value(measured_introspection_response(introspection())).unwrap();
        assert_eq!(json["type"], "introspection_v2");
        assert_eq!(json["epistemic_status"], "measured_runtime_snapshot");
        for forbidden in [
            "phi",
            "meta_awareness",
            "is_conscious",
            "phenomenal_state",
            "cycles_since_awakening",
            "self_model_accuracy",
        ] {
            assert!(json.get(forbidden).is_none(), "unexpected {forbidden}");
        }
    }

    #[test]
    fn legacy_introspection_is_machine_labeled_as_derived_compatibility() {
        let json = serde_json::to_value(legacy_introspection_response(
            introspection(),
            LegacyIntrospectionPolicy::ExplicitDerivedCompatibility,
        ))
        .unwrap();
        assert_eq!(json["type"], "introspection");
        assert!((json["phi"].as_f64().unwrap() - 0.61).abs() < 1e-6);
        assert_eq!(json["is_conscious"], true);
        assert_eq!(json["epistemic_status"], "legacy_compatibility_derived");
        assert!(json["compatibility_notes"].as_array().unwrap().len() >= 5);
    }

    #[test]
    fn status_preserves_legacy_wire_field_names_without_fake_cognition() {
        let read = CognitiveStatusRead {
            relation: ActivitySnapshotRelation::Idle {
                snapshot_origin: SnapshotOrigin::Initialized,
            },
            activity_revision: revision(),
            snapshot_revision: revision(),
            consciousness_level: 0.4,
            memory_count: 12,
        };
        let response = status_response(
            read,
            ServiceCounters {
                requests_processed: 8,
                sleep_cycles: 2,
            },
            Duration::from_secs(42),
        );
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["type"], "status");
        assert_eq!(json["uptime_seconds"], 42);
        assert_eq!(json["requests_processed"], 8);
        assert_eq!(json["memory_count"], 12);
        assert_eq!(json["sleep_cycles"], 2);
    }
}
