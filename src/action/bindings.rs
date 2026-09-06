// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Primitive Action Bindings
//!
//! This module implements the "Hands" of Symthaea, bridging the abstract
//! HDC primitives ("Thinking") with concrete `ActionIR` ("Doing").

use super::{ActionError, ActionIR};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

/// Context passed to action generators
#[derive(Debug, Clone, Default)]
pub struct ActionContext {
    /// Target file path (if applicable)
    pub target_path: Option<PathBuf>,
    /// Target content/code (if applicable)
    pub content: Option<String>,
    /// Additional arguments
    pub args: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
}

/// A binding between a primitive and an action generator
pub struct PrimitiveActionBinding {
    /// Name of the primitive (e.g., "PARSE")
    pub primitive_name: String,
    /// Function that generates the action from context
    pub generator: Box<dyn Fn(&ActionContext) -> Result<ActionIR, ActionError> + Send + Sync>,
}

/// Registry of primitive bindings
#[derive(Default, Clone)]
pub struct ActionRegistry {
    bindings: HashMap<String, Arc<PrimitiveActionBinding>>,
}

impl ActionRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Register a new binding
    pub fn register<F>(mut self, primitive: &str, generator: F) -> Self
    where
        F: Fn(&ActionContext) -> Result<ActionIR, ActionError> + Send + Sync + 'static,
    {
        let binding = PrimitiveActionBinding {
            primitive_name: primitive.to_string(),
            generator: Box::new(generator),
        };
        self.bindings
            .insert(primitive.to_string(), Arc::new(binding));
        self
    }

    /// Resolve a primitive to an action
    pub fn resolve(
        &self,
        primitive: &str,
        context: &ActionContext,
    ) -> Result<ActionIR, ActionError> {
        let binding = self.bindings.get(primitive).ok_or_else(|| {
            ActionError::ValidationFailed(format!("Unknown primitive: {}", primitive))
        })?;

        (binding.generator)(context)
    }

    /// Create the standard registry with default bindings
    pub fn standard() -> Self {
        Self::new()
            .register("PARSE", |ctx| {
                let path = ctx.target_path.as_ref().ok_or_else(|| {
                    ActionError::ValidationFailed("PARSE requires target_path".into())
                })?;

                Ok(ActionIR::RunCommand {
                    program: "tree-sitter".into(),
                    args: vec!["parse".into(), path.to_string_lossy().to_string()],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("READ", |ctx| {
                let path = ctx.target_path.as_ref().ok_or_else(|| {
                    ActionError::ValidationFailed("READ requires target_path".into())
                })?;

                Ok(ActionIR::ReadFile {
                    path: path.clone(),
                    encoding: None,
                })
            })
            .register("WRITE", |ctx| {
                let path = ctx.target_path.as_ref().ok_or_else(|| {
                    ActionError::ValidationFailed("WRITE requires target_path".into())
                })?;
                let content = ctx.content.as_ref().ok_or_else(|| {
                    ActionError::ValidationFailed("WRITE requires content".into())
                })?;

                Ok(ActionIR::WriteFile {
                    path: path.clone(),
                    content: content.as_bytes().to_vec(),
                    create_dirs: true,
                })
            })
            .register("LIST", |ctx| {
                let path = ctx
                    .target_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::ListDirectory {
                    path,
                    recursive: false,
                })
            })
            .register("NIX_BUILD", |ctx| {
                let _path = ctx
                    .target_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::RunCommand {
                    program: "nix".into(),
                    args: vec!["--version".into()],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("CARGO_TEST", |ctx| {
                let path = ctx
                    .target_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::RunCommand {
                    program: "cargo".into(),
                    args: vec!["test".into()],
                    env: BTreeMap::new(),
                    working_dir: Some(path),
                })
            })
            .register("CARGO_CHECK", |ctx| {
                let path = ctx
                    .target_path
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::RunCommand {
                    program: "cargo".into(),
                    args: vec!["check".into()],
                    env: BTreeMap::new(),
                    working_dir: Some(path),
                })
            })
            .register("GIT_PUSH", |_ctx| {
                Ok(ActionIR::RunCommand {
                    program: "git".into(),
                    args: vec!["push".into()],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("GIT_COMMIT", |ctx| {
                let msg = ctx
                    .content
                    .clone()
                    .unwrap_or_else(|| "Conscious update".into());
                Ok(ActionIR::RunCommand {
                    program: "git".into(),
                    args: vec!["commit".into(), "-am".into(), msg],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("WEB_SEARCH", |ctx| {
                let query = ctx.content.clone().ok_or_else(|| {
                    ActionError::ValidationFailed("WEB_SEARCH requires content".into())
                })?;
                Ok(ActionIR::RunCommand {
                    program: "nix".into(),
                    args: vec!["search".into(), "nixpkgs".into(), query],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("READ_SENSOR", |ctx| {
                let sensor_id = ctx
                    .args
                    .first()
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .ok_or_else(|| {
                        ActionError::ValidationFailed(
                            "READ_SENSOR requires an explicit sensor_id".into(),
                        )
                    })?;
                let channels = ctx.args.get(1..).unwrap_or_default().to_vec();
                if channels.is_empty() || channels.iter().any(|channel| channel.trim().is_empty()) {
                    return Err(ActionError::ValidationFailed(
                        "READ_SENSOR requires at least one explicit non-empty channel".into(),
                    ));
                }
                Ok(ActionIR::ReadSensor {
                    sensor_id,
                    channels,
                })
            })
            .register("WRITE_SERVO", |ctx| {
                let id_raw = ctx.args.first().ok_or_else(|| {
                    ActionError::ValidationFailed(
                        "WRITE_SERVO requires an explicit servo_id".into(),
                    )
                })?;
                let id = id_raw.parse::<u32>().map_err(|_| {
                    ActionError::ValidationFailed(format!(
                        "WRITE_SERVO servo_id must be a u32, got '{id_raw}'"
                    ))
                })?;

                let value_raw = ctx.args.get(1).ok_or_else(|| {
                    ActionError::ValidationFailed(
                        "WRITE_SERVO requires an explicit value".into(),
                    )
                })?;
                let value = value_raw.parse::<f32>().map_err(|_| {
                    ActionError::ValidationFailed(format!(
                        "WRITE_SERVO value must be an f32, got '{value_raw}'"
                    ))
                })?;
                if !value.is_finite() {
                    return Err(ActionError::ValidationFailed(
                        "WRITE_SERVO value must be finite".into(),
                    ));
                }

                Ok(ActionIR::WriteServo {
                    servo_id: id,
                    value,
                })
            })
            .register("SWARM_GOSSIP", |ctx| {
                let topic = ctx
                    .args
                    .first()
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .ok_or_else(|| {
                        ActionError::ValidationFailed(
                            "SWARM_GOSSIP requires an explicit non-empty topic".into(),
                        )
                    })?;
                let content = ctx.content.as_ref().filter(|value| !value.is_empty()).ok_or_else(
                    || {
                        ActionError::ValidationFailed(
                            "SWARM_GOSSIP requires explicit non-empty content".into(),
                        )
                    },
                )?;
                Ok(ActionIR::SwarmGossip {
                    topic,
                    payload: content.as_bytes().to_vec(),
                })
            })
            .register("WASM_VERIFY", |ctx| {
                let path = ctx.target_path.clone().ok_or_else(|| {
                    ActionError::ValidationFailed("WASM_VERIFY requires target_path".into())
                })?;
                let func = ctx.args.first().cloned().unwrap_or_else(|| "verify".into());
                Ok(ActionIR::WasmSandbox {
                    module_path: path,
                    function_name: func,
                    input_data: vec![],
                })
            })
    }
}

/// Executor for sequences of primitives
pub struct PrimitiveExecutor {
    registry: ActionRegistry,
}

impl PrimitiveExecutor {
    pub fn new(registry: ActionRegistry) -> Self {
        Self { registry }
    }

    /// Translate a sequence of primitives into a sequence of actions.
    ///
    /// Translation is all-or-nothing: an unknown primitive or invalid binding
    /// returns `Err` rather than silently dropping that requested action and
    /// returning a shorter sequence.
    pub fn translate(
        &self,
        primitives: &[String],
        context: &ActionContext,
    ) -> Result<Vec<ActionIR>, ActionError> {
        let mut actions = Vec::with_capacity(primitives.len());
        for prim in primitives {
            actions.push(self.registry.resolve(prim, context)?);
        }
        Ok(actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_sensor_requires_explicit_identity_and_channel() {
        let registry = ActionRegistry::standard();

        let missing = registry.resolve("READ_SENSOR", &ActionContext::default());
        assert!(missing.is_err());

        let no_channel = ActionContext {
            args: vec!["imu-0".into()],
            ..Default::default()
        };
        assert!(registry.resolve("READ_SENSOR", &no_channel).is_err());

        let valid = ActionContext {
            args: vec!["imu-0".into(), "accel_x".into()],
            ..Default::default()
        };
        let action = registry.resolve("READ_SENSOR", &valid).unwrap();
        match action {
            ActionIR::ReadSensor {
                sensor_id,
                channels,
            } => {
                assert_eq!(sensor_id, "imu-0");
                assert_eq!(channels, vec!["accel_x"]);
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn write_servo_rejects_missing_malformed_and_non_finite_values() {
        let registry = ActionRegistry::standard();

        assert!(
            registry
                .resolve("WRITE_SERVO", &ActionContext::default())
                .is_err()
        );

        let malformed_id = ActionContext {
            args: vec!["servo-zero".into(), "0.5".into()],
            ..Default::default()
        };
        assert!(registry.resolve("WRITE_SERVO", &malformed_id).is_err());

        let malformed_value = ActionContext {
            args: vec!["3".into(), "half".into()],
            ..Default::default()
        };
        assert!(registry.resolve("WRITE_SERVO", &malformed_value).is_err());

        let nan = ActionContext {
            args: vec!["3".into(), "NaN".into()],
            ..Default::default()
        };
        assert!(registry.resolve("WRITE_SERVO", &nan).is_err());

        let valid = ActionContext {
            args: vec!["3".into(), "0.5".into()],
            ..Default::default()
        };
        let action = registry.resolve("WRITE_SERVO", &valid).unwrap();
        match action {
            ActionIR::WriteServo { servo_id, value } => {
                assert_eq!(servo_id, 3);
                assert_eq!(value, 0.5);
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn swarm_gossip_requires_explicit_topic_and_payload() {
        let registry = ActionRegistry::standard();

        assert!(
            registry
                .resolve("SWARM_GOSSIP", &ActionContext::default())
                .is_err()
        );

        let missing_payload = ActionContext {
            args: vec!["research".into()],
            ..Default::default()
        };
        assert!(registry.resolve("SWARM_GOSSIP", &missing_payload).is_err());

        let valid = ActionContext {
            args: vec!["research".into()],
            content: Some("result".into()),
            ..Default::default()
        };
        let action = registry.resolve("SWARM_GOSSIP", &valid).unwrap();
        match action {
            ActionIR::SwarmGossip { topic, payload } => {
                assert_eq!(topic, "research");
                assert_eq!(payload, b"result");
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn primitive_translation_is_all_or_nothing() {
        let executor = PrimitiveExecutor::new(ActionRegistry::standard());
        let context = ActionContext {
            target_path: Some(PathBuf::from("/tmp/input.txt")),
            ..Default::default()
        };

        let result = executor.translate(
            &["READ".to_string(), "UNKNOWN_PRIMITIVE".to_string()],
            &context,
        );
        assert!(result.is_err());
    }

    #[test]
    fn primitive_translation_preserves_order_when_complete() {
        let executor = PrimitiveExecutor::new(ActionRegistry::standard());
        let context = ActionContext {
            target_path: Some(PathBuf::from("/tmp/input.txt")),
            ..Default::default()
        };

        let actions = executor
            .translate(&["READ".to_string(), "LIST".to_string()], &context)
            .unwrap();
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], ActionIR::ReadFile { .. }));
        assert!(matches!(actions[1], ActionIR::ListDirectory { .. }));
    }
}
