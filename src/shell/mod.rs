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
pub use aliases::{AliasManager, Alias};
pub use context::{
    ShellContext, IntelliSenseEngine, PhiGate, GateDecision,
    Completion, CompletionKind, ExecutionRequest, CommandClassification,
    CompletionPreview, PreviewStep, classify_command_destructiveness,
    CommandContext as ShellCommandContext, // Renamed to avoid conflict
};
pub use epistemic_overlay::{
    EpistemicOverlayEngine, EpistemicOverlay, EpistemicStyle,
    OverlayPosition, OverlayType, KnowledgeSource,
    CommandContext, // Epistemic command context (with k_index, etc.)
};
pub use ipc_client::{
    ShellIpcClient, IpcClientConfig, ConnectionState,
    discover_socket, MetricsSnapshot, IpcRequest, IpcResponse,
    // Wire protocol types
    Request, Response, RequestEnvelope, ResponseEnvelope,
    WireProtocol, ShellContextData, SafetyLevelData,
};
pub use ipc_server::{
    IpcServer, IpcServerConfig, MetricsProvider, CommandExecutor,
    ExecutionResult, ValidationResult,
    StubMetricsProvider, StubCommandExecutor,
};
pub use service_state::StateManager;
pub use flake_context::{FlakeContext, ContextualSuggestion, SuggestionSource};
pub use error_explainer::{ErrorExplainer, ErrorExplanation, quick_error_check};
pub use whatif::{WhatIfSimulator, WhatIfResult};
pub use undo::{UndoAction, ActionType, ActionData};

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
