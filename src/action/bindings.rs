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
            .register("NIX_BUILD", |_ctx| {
                Err(ActionError::ValidationFailed(
                    "NIX_BUILD requires an explicit build target/profile; legacy nix --version probe disabled"
                        .into(),
                ))
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
                let sensor_id = ctx.args.first().cloned().unwrap_or_else(|| "ina219".into());
                let channels = if ctx.args.len() > 1 {
                    ctx.args[1..].to_vec()
                } else {
                    vec!["voltage".into(), "current".into()]
                };
                Ok(ActionIR::ReadSensor {
                    sensor_id,
                    channels,
                })
            })
            .register("WRITE_SERVO", |ctx| {
                let id = ctx
                    .args
                    .first()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                let val = ctx
                    .args
                    .get(1)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(ActionIR::WriteServo {
                    servo_id: id,
                    value: val,
                })
            })
            .register("SWARM_GOSSIP", |ctx| {
                let topic = ctx
                    .args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "optimization".into());
                let payload = ctx.content.clone().unwrap_or_default().into_bytes();
                Ok(ActionIR::SwarmGossip { topic, payload })
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

    /// Translate a sequence of primitives into a sequence of actions
    pub fn translate(
        &self,
        primitives: &[String],
        context: &ActionContext,
    ) -> Result<Vec<ActionIR>, ActionError> {
        let mut actions = Vec::new();
        for prim in primitives {
            if let Ok(action) = self.registry.resolve(prim, context) {
                actions.push(action);
            }
        }
        Ok(actions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nix_build_cannot_resolve_to_version_probe() {
        let registry = ActionRegistry::standard();
        let err = registry
            .resolve("NIX_BUILD", &ActionContext::default())
            .expect_err("NIX_BUILD must fail closed until real build semantics are explicit");
        let message = format!("{err}");
        assert!(message.contains("explicit build target/profile"));
        assert!(!message.contains("--version"));
    }

    #[test]
    fn nix_version_probe_is_not_exposed_as_nix_build() {
        let registry = ActionRegistry::standard();
        let context = ActionContext {
            target_path: Some(PathBuf::from(".")),
            ..Default::default()
        };
        let result = registry.resolve("NIX_BUILD", &context);
        assert!(result.is_err());
    }
}
