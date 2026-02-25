//! Primitive Action Bindings
//!
//! This module implements the "Hands" of Symthaea, bridging the abstract
//! HDC primitives ("Thinking") with concrete `ActionIR` ("Doing").

use super::{ActionError, ActionIR};
use std::collections::{HashMap, BTreeMap};
use std::path::PathBuf;
use std::sync::Arc;

/// Context passed to action generators
#[derive(Debug, Clone)]
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

impl Default for ActionContext {
    fn default() -> Self {
        Self {
            target_path: None,
            content: None,
            args: Vec::new(),
            env: HashMap::new(),
        }
    }
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
        let binding = self
            .bindings
            .get(primitive)
            .ok_or_else(|| ActionError::ValidationFailed(format!("Unknown primitive: {}", primitive)))?;

        (binding.generator)(context)
    }

    /// Create the standard registry with default bindings
    pub fn standard() -> Self {
        Self::new()
            .register("PARSE", |ctx| {
                let path = ctx
                    .target_path
                    .as_ref()
                    .ok_or_else(|| ActionError::ValidationFailed("PARSE requires target_path".into()))?;

                Ok(ActionIR::RunCommand {
                    program: "tree-sitter".into(),
                    args: vec!["parse".into(), path.to_string_lossy().to_string()],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("READ", |ctx| {
                let path = ctx
                    .target_path
                    .as_ref()
                    .ok_or_else(|| ActionError::ValidationFailed("READ requires target_path".into()))?;

                Ok(ActionIR::ReadFile {
                    path: path.clone(),
                    encoding: None,
                })
            })
            .register("WRITE", |ctx| {
                let path = ctx
                    .target_path
                    .as_ref()
                    .ok_or_else(|| ActionError::ValidationFailed("WRITE requires target_path".into()))?;
                let content = ctx
                    .content
                    .as_ref()
                    .ok_or_else(|| ActionError::ValidationFailed("WRITE requires content".into()))?;

                Ok(ActionIR::WriteFile {
                    path: path.clone(),
                    content: content.as_bytes().to_vec(),
                    create_dirs: true,
                })
            })
            .register("LIST", |ctx| {
                let path = ctx.target_path.clone().unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::ListDirectory {
                    path,
                    recursive: false,
                })
            })
            .register("NIX_BUILD", |ctx| {
                let _path = ctx.target_path.clone().unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::RunCommand {
                    program: "nix".into(),
                    args: vec!["--version".into()],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("CARGO_TEST", |ctx| {
                let path = ctx.target_path.clone().unwrap_or_else(|| PathBuf::from("."));
                Ok(ActionIR::RunCommand {
                    program: "cargo".into(),
                    args: vec!["test".into()],
                    env: BTreeMap::new(),
                    working_dir: Some(path),
                })
            })
            .register("CARGO_CHECK", |ctx| {
                let path = ctx.target_path.clone().unwrap_or_else(|| PathBuf::from("."));
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
                let msg = ctx.content.clone().unwrap_or_else(|| "Conscious update".into());
                Ok(ActionIR::RunCommand {
                    program: "git".into(),
                    args: vec!["commit".into(), "-am".into(), msg],
                    env: BTreeMap::new(),
                    working_dir: None,
                })
            })
            .register("WEB_SEARCH", |ctx| {
                let query = ctx.content.clone().ok_or_else(|| ActionError::ValidationFailed("WEB_SEARCH requires content".into()))?;
                Ok(ActionIR::RunCommand {
                    program: "nix".into(),
                    args: vec!["search".into(), "nixpkgs".into(), query],
                    env: BTreeMap::new(),
                    working_dir: None,
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
