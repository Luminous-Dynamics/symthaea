// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure translation between the shell-facing request/response vocabulary and
//! the legacy protocol-v1 IPC wire vocabulary.
//!
//! Keeping this mapping independent from socket ownership lets the shell migrate
//! from the historical request/read loop to [`super::ipc_connection::DuplexShellConnection`]
//! without mixing transport concurrency changes with application semantics.

use super::ipc_client::{
    IpcRequest, IpcResponse, Request, Response, SafetyLevelData,
};

/// Deterministic context required when translating a wire response.
///
/// `latest_phi` preserves the legacy IntelliSense response contract while making
/// the dependency explicit instead of letting transport code reach into client
/// state. `now_ms` makes Pong mapping deterministic and directly testable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponseMapContext {
    pub latest_phi: f64,
    pub now_ms: u64,
}

impl ResponseMapContext {
    pub const fn new(latest_phi: f64, now_ms: u64) -> Self {
        Self { latest_phi, now_ms }
    }
}

/// Translate one shell-facing request into the protocol-v1 wire request.
///
/// This function is intentionally pure: it does not write to a socket, inspect
/// connection state, or mutate cognitive telemetry.
pub fn map_request(request: &Request) -> IpcRequest {
    match request {
        Request::Ping => IpcRequest::Ping,
        Request::GetStatus => IpcRequest::GetMetrics,
        Request::Execute {
            command,
            phi_required,
            dry_run,
        } => {
            if *dry_run {
                IpcRequest::Validate {
                    command: command.clone(),
                }
            } else {
                IpcRequest::Execute {
                    command: command.clone(),
                    require_phi: Some(*phi_required),
                }
            }
        }
        Request::GetIntelliSense {
            input, cursor_pos, ..
        } => IpcRequest::GetCompletions {
            input: input.clone(),
            cursor_pos: *cursor_pos,
        },
        Request::Validate { command, .. } => IpcRequest::Validate {
            command: command.clone(),
        },
        Request::SubscribeMetrics => IpcRequest::SubscribeMetrics,
        Request::UnsubscribeMetrics => IpcRequest::UnsubscribeMetrics,
    }
}

/// Translate one protocol-v1 response into the shell-facing response vocabulary.
///
/// The match order intentionally preserves the historical `ShellIpcClient`
/// semantics for request-specific cases before applying generic fallbacks.
pub fn map_response(
    request: &Request,
    response: IpcResponse,
    context: ResponseMapContext,
) -> Response {
    match (request, response) {
        (Request::Ping, IpcResponse::Pong) => Response::Pong {
            timestamp_ms: context.now_ms,
        },
        (Request::GetStatus, IpcResponse::Metrics(metrics)) => Response::Status {
            phi: metrics.phi,
            coherence: metrics.coherence,
            consciousness_level: metrics.consciousness_level,
            is_conscious: metrics.is_conscious,
            uptime_secs: metrics.uptime_secs,
        },
        (Request::GetIntelliSense { .. }, IpcResponse::Completions(items)) => {
            let confidence = items
                .iter()
                .map(|c| c.similarity)
                .fold(0.0_f32, f32::max);
            Response::IntelliSenseResult {
                completions: items,
                command_preview: None,
                phi: context.latest_phi,
                confidence,
            }
        }
        (
            Request::Validate { .. },
            IpcResponse::ValidationResult {
                valid,
                safety_level,
                warnings,
                ..
            },
        ) => Response::ValidationResult {
            valid,
            safety_level: safety_level_from_string(&safety_level),
            phi_required: 0.0,
            warnings,
        },
        (
            Request::Execute { dry_run: true, .. },
            IpcResponse::ValidationResult {
                valid,
                preview,
                warnings,
                ..
            },
        ) => {
            let output = preview.unwrap_or_else(|| warnings.join("\n"));
            Response::ExecutionResult {
                executed: false,
                output,
                phi_at_execution: 0.0,
                gate_reason: Some(if valid {
                    "dry-run".to_string()
                } else {
                    "dry-run failed".to_string()
                }),
            }
        }
        (
            Request::Execute { .. },
            IpcResponse::ExecutionResult {
                success,
                output,
                phi_at_execution,
                vetoed,
                veto_reason,
            },
        ) => Response::ExecutionResult {
            executed: success && !vetoed,
            output,
            phi_at_execution,
            gate_reason: if vetoed {
                veto_reason.or(Some("vetoed".to_string()))
            } else {
                veto_reason
            },
        },
        (Request::SubscribeMetrics, IpcResponse::Subscribed)
        | (Request::UnsubscribeMetrics, IpcResponse::Subscribed) => Response::Subscribed,
        (_, IpcResponse::HelloAck { server_version }) => Response::Error {
            code: 2,
            message: format!("Unexpected hello ack (server version {server_version})"),
        },
        (_, IpcResponse::Metrics(metrics)) => Response::Status {
            phi: metrics.phi,
            coherence: metrics.coherence,
            consciousness_level: metrics.consciousness_level,
            is_conscious: metrics.is_conscious,
            uptime_secs: metrics.uptime_secs,
        },
        (
            _,
            IpcResponse::ValidationResult {
                valid,
                safety_level,
                warnings,
                ..
            },
        ) => Response::ValidationResult {
            valid,
            safety_level: safety_level_from_string(&safety_level),
            phi_required: 0.0,
            warnings,
        },
        (_, IpcResponse::Completions(items)) => Response::IntelliSenseResult {
            completions: items,
            command_preview: None,
            phi: context.latest_phi,
            confidence: 0.0,
        },
        (_, IpcResponse::Error(message)) => Response::Error { code: 1, message },
        (_, IpcResponse::Pong) => Response::Pong {
            timestamp_ms: context.now_ms,
        },
        (_, IpcResponse::Subscribed) => Response::Subscribed,
        (
            _,
            IpcResponse::ExecutionResult {
                success,
                output,
                phi_at_execution,
                vetoed,
                veto_reason,
            },
        ) => Response::ExecutionResult {
            executed: success && !vetoed,
            output,
            phi_at_execution,
            gate_reason: if vetoed {
                veto_reason.or(Some("vetoed".to_string()))
            } else {
                veto_reason
            },
        },
    }
}

/// Convert the server's historical string safety label to the shell UI structure.
pub fn safety_level_from_string(level: &str) -> SafetyLevelData {
    let lower = level.to_lowercase();
    let color = if lower.contains("destructive")
        || lower.contains("critical")
        || lower.contains("red")
    {
        "red"
    } else if lower.contains("confirm") || lower.contains("yellow") || lower.contains("warn") {
        "yellow"
    } else {
        "green"
    };

    SafetyLevelData {
        level: level.to_string(),
        color: color.to_string(),
        description: level.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::ipc_client::{CompletionItem, MetricsSnapshot, ShellContextData};

    fn context() -> ResponseMapContext {
        ResponseMapContext::new(0.73, 42)
    }

    fn metrics(phi: f64) -> MetricsSnapshot {
        MetricsSnapshot {
            phi,
            coherence: 0.81,
            is_conscious: true,
            consciousness_level: 0.79,
            uptime_secs: 123,
            ..Default::default()
        }
    }

    #[test]
    fn request_mapping_covers_every_shell_variant() {
        assert!(matches!(map_request(&Request::Ping), IpcRequest::Ping));
        assert!(matches!(
            map_request(&Request::GetStatus),
            IpcRequest::GetMetrics
        ));

        let execute = map_request(&Request::Execute {
            command: "echo safe".into(),
            phi_required: 0.61,
            dry_run: false,
        });
        assert!(matches!(
            execute,
            IpcRequest::Execute {
                command,
                require_phi: Some(phi)
            } if command == "echo safe" && (phi - 0.61).abs() < f64::EPSILON
        ));

        assert!(matches!(
            map_request(&Request::Execute {
                command: "rm -rf /tmp/example".into(),
                phi_required: 0.9,
                dry_run: true,
            }),
            IpcRequest::Validate { command } if command == "rm -rf /tmp/example"
        ));

        assert!(matches!(
            map_request(&Request::GetIntelliSense {
                input: "nix bu".into(),
                cursor_pos: 6,
                context: ShellContextData::default(),
            }),
            IpcRequest::GetCompletions { input, cursor_pos }
                if input == "nix bu" && cursor_pos == 6
        ));
        assert!(matches!(
            map_request(&Request::Validate {
                command: "nix build".into(),
                dry_run: true,
            }),
            IpcRequest::Validate { command } if command == "nix build"
        ));
        assert!(matches!(
            map_request(&Request::SubscribeMetrics),
            IpcRequest::SubscribeMetrics
        ));
        assert!(matches!(
            map_request(&Request::UnsubscribeMetrics),
            IpcRequest::UnsubscribeMetrics
        ));
    }

    #[test]
    fn status_mapping_preserves_observed_metrics() {
        let response = map_response(
            &Request::GetStatus,
            IpcResponse::Metrics(metrics(0.77)),
            context(),
        );
        assert!(matches!(
            response,
            Response::Status {
                phi,
                coherence,
                consciousness_level,
                is_conscious: true,
                uptime_secs: 123,
            } if (phi - 0.77).abs() < f64::EPSILON
                && (coherence - 0.81).abs() < f64::EPSILON
                && (consciousness_level - 0.79).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn intellisense_uses_latest_phi_and_max_similarity() {
        let items = vec![
            CompletionItem {
                text: "build".into(),
                label: "build".into(),
                kind: "Command".into(),
                similarity: 0.42,
                docs: None,
            },
            CompletionItem {
                text: "bundle".into(),
                label: "bundle".into(),
                kind: "Command".into(),
                similarity: 0.91,
                docs: None,
            },
        ];
        let response = map_response(
            &Request::GetIntelliSense {
                input: "bu".into(),
                cursor_pos: 2,
                context: ShellContextData::default(),
            },
            IpcResponse::Completions(items),
            context(),
        );
        assert!(matches!(
            response,
            Response::IntelliSenseResult { phi, confidence, .. }
                if (phi - 0.73).abs() < f64::EPSILON
                    && (confidence - 0.91).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn dry_run_validation_never_claims_execution() {
        let response = map_response(
            &Request::Execute {
                command: "danger".into(),
                phi_required: 0.9,
                dry_run: true,
            },
            IpcResponse::ValidationResult {
                valid: true,
                safety_level: "YELLOW".into(),
                preview: Some("would execute danger".into()),
                warnings: vec![],
            },
            context(),
        );
        assert!(matches!(
            response,
            Response::ExecutionResult {
                executed: false,
                output,
                phi_at_execution,
                gate_reason: Some(reason),
            } if output == "would execute danger"
                && phi_at_execution == 0.0
                && reason == "dry-run"
        ));
    }

    #[test]
    fn vetoed_execution_stays_non_executed() {
        let response = map_response(
            &Request::Execute {
                command: "danger".into(),
                phi_required: 0.9,
                dry_run: false,
            },
            IpcResponse::ExecutionResult {
                success: true,
                output: "blocked".into(),
                phi_at_execution: 0.5,
                vetoed: true,
                veto_reason: None,
            },
            context(),
        );
        assert!(matches!(
            response,
            Response::ExecutionResult {
                executed: false,
                gate_reason: Some(reason),
                ..
            } if reason == "vetoed"
        ));
    }

    #[test]
    fn ping_timestamp_is_injected_not_wall_clock_owned() {
        let response = map_response(&Request::Ping, IpcResponse::Pong, context());
        assert!(matches!(response, Response::Pong { timestamp_ms: 42 }));
    }

    #[test]
    fn safety_label_mapping_is_stable() {
        assert_eq!(safety_level_from_string("critical").color, "red");
        assert_eq!(safety_level_from_string("confirm").color, "yellow");
        assert_eq!(safety_level_from_string("safe").color, "green");
    }
}
