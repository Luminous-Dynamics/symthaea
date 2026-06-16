// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell Module - Command Interface
//!
//! Provides a rich, consciousness-aware command shell interface including:
//! - Alias management and expansion
//! - Epistemic overlays for uncertainty
//! - Error explanation with context
//! - Flake environment management
//! - IPC client/server for service communication
//! - IntelliSense completions with HDC similarity
//! - Keybindings and theming
//! - Notifications and service state
//! - Phi-gated execution
//! - Syntax highlighting
//! - Undo/redo operations
//! - What-if speculation

use serde::{Deserialize, Serialize};

pub mod aliases;
pub mod context;
pub mod epistemic_overlay;
pub mod error_explainer;
pub mod flake_context;
pub mod ipc_client;
pub mod ipc_server;
pub mod keybindings;
pub mod notifications;
pub mod service_state;
pub mod syntax_highlight;
pub mod theming;
pub mod undo;
pub mod whatif;

// Re-export commonly used types
pub use aliases::{Alias, AliasManager};
pub use context::{
    CommandClassification,
    CommandContext as ShellCommandContext, // Renamed to avoid conflict
    Completion,
    CompletionKind,
    CompletionPreview,
    ExecutionRequest,
    GateDecision,
    IntelliSenseEngine,
    PhiGate,
    PreviewStep,
    ShellContext,
    classify_command_destructiveness,
};
pub use epistemic_overlay::{
    CommandContext, // Epistemic command context (with k_index, etc.)
    EpistemicOverlay,
    EpistemicOverlayEngine,
    EpistemicStyle,
    KnowledgeSource,
    OverlayPosition,
    OverlayType,
};
pub use error_explainer::{ErrorExplainer, ErrorExplanation, quick_error_check};
pub use flake_context::{ContextualSuggestion, FlakeContext, SuggestionSource};
pub use ipc_client::{
    ConnectionState,
    IpcClientConfig,
    IpcRequest,
    IpcResponse,
    MetricsSnapshot,
    // Wire protocol types
    Request,
    RequestEnvelope,
    Response,
    ResponseEnvelope,
    SafetyLevelData,
    ShellContextData,
    ShellIpcClient,
    WireProtocol,
    discover_socket,
};
pub use ipc_server::{
    CommandExecutor, ExecutionResult, IpcServer, IpcServerConfig, MetricsProvider,
    StubCommandExecutor, StubMetricsProvider, ValidationResult,
};
pub use service_state::StateManager;
pub use undo::{ActionData, ActionType, UndoAction};
pub use whatif::{WhatIfResult, WhatIfSimulator};

/// Shell configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// History size
    pub history_size: usize,
    /// Enable suggestions
    pub suggestions_enabled: bool,
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
    /// Enable epistemic overlays
    pub epistemic_overlays: bool,
    /// Theme name
    pub theme: String,
    /// Undo stack size
    pub undo_stack_size: usize,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            suggestions_enabled: true,
            syntax_highlighting: true,
            epistemic_overlays: true,
            theme: "consciousness-dark".to_string(),
            undo_stack_size: 100,
        }
    }
}
