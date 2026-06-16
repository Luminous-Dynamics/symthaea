// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GUI Bridge Module - Bidirectional Widget↔Nix Mapping
//!
//! Provides bidirectional translation between GUI widgets and NixOS configuration:
//! - Widget changes → Nix expression updates
//! - Nix config parsing → Widget state synchronization
//! - HDC semantic translation for intelligent mapping
//! - Constraint validation via knowledge graph
//!
//! ## Architecture
//!
//! ```text
//! GUI Widget Change                    Nix Config Change
//!       │                                    │
//!       ▼                                    ▼
//! ┌─────────────┐                    ┌─────────────┐
//! │ widget_to_  │                    │ nix_to_     │
//! │ nix()       │                    │ widgets()   │
//! └──────┬──────┘                    └──────┬──────┘
//!        │                                  │
//!        ▼                                  ▼
//! ┌─────────────────────────────────────────────────┐
//! │              HDC Translator                      │
//! │  bind(widget_hv, role_hv) ↔ unbind(nix_hv)     │
//! └─────────────────────────────────────────────────┘
//!        │                                  │
//!        ▼                                  ▼
//! ┌─────────────────────────────────────────────────┐
//! │           Constraint Validator                   │
//! │  Knowledge Graph: type/dependency/conflict      │
//! └─────────────────────────────────────────────────┘
//!        │                                  │
//!        ▼                                  ▼
//!    Nix Expression                   Widget States
//! ```

pub mod constraint_validator;
pub mod hdc_translator;
pub mod widget_mapper;
pub mod widgets;

pub use constraint_validator::{
    Constraint, ConstraintKind, ConstraintValidator, Suggestion, ValidationError, ValidationResult,
};
pub use hdc_translator::{
    HdcTranslator, SemanticBinding, SemanticConfidence, SemanticRole, TranslationResult,
};
pub use widget_mapper::{
    MappingResult, NixPath, ValueTransformer, WidgetBinding, WidgetId, WidgetMapper, WidgetValue,
};
pub use widgets::{
    DashboardStats,
    DiffHunk,
    DiffLine,
    DiffStats,
    DiffType,
    Generation,
    GenerationChanges,
    // C2: Generation Timeline
    GenerationTimeline,
    // C3: Live Config Diff
    LiveConfigDiff,
    // C1: NixOS Module Browser
    ModuleBrowser,
    ModuleCategory,
    NixOption,
    ServiceAction,
    // C4: Service Status Dashboard
    ServiceDashboard,
    SystemdUnit,
    UnitState,
};

use std::collections::HashMap;

/// Category of NixOS configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigCategory {
    /// System packages (environment.systemPackages)
    Packages,
    /// Services (services.*)
    Services,
    /// Networking (networking.*)
    Networking,
    /// Users and groups (users.*)
    Users,
    /// Boot configuration (boot.*)
    Boot,
    /// Hardware (hardware.*)
    Hardware,
    /// Programs (programs.*)
    Programs,
    /// Security (security.*)
    Security,
    /// Nix settings (nix.*)
    NixSettings,
    /// Custom/other
    Custom,
}

impl ConfigCategory {
    /// Get the NixOS option path prefix for this category
    pub fn path_prefix(&self) -> &'static str {
        match self {
            Self::Packages => "environment.systemPackages",
            Self::Services => "services",
            Self::Networking => "networking",
            Self::Users => "users",
            Self::Boot => "boot",
            Self::Hardware => "hardware",
            Self::Programs => "programs",
            Self::Security => "security",
            Self::NixSettings => "nix",
            Self::Custom => "",
        }
    }

    /// Infer category from a Nix option path
    pub fn from_path(path: &str) -> Self {
        if path.starts_with("environment.systemPackages") {
            Self::Packages
        } else if path.starts_with("services.") {
            Self::Services
        } else if path.starts_with("networking.") {
            Self::Networking
        } else if path.starts_with("users.") {
            Self::Users
        } else if path.starts_with("boot.") {
            Self::Boot
        } else if path.starts_with("hardware.") {
            Self::Hardware
        } else if path.starts_with("programs.") {
            Self::Programs
        } else if path.starts_with("security.") {
            Self::Security
        } else if path.starts_with("nix.") {
            Self::NixSettings
        } else {
            Self::Custom
        }
    }

    /// Get icon for category
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Packages => "📦",
            Self::Services => "⚙️",
            Self::Networking => "🌐",
            Self::Users => "👤",
            Self::Boot => "🚀",
            Self::Hardware => "🖥️",
            Self::Programs => "📱",
            Self::Security => "🔒",
            Self::NixSettings => "❄️",
            Self::Custom => "📝",
        }
    }
}

/// Synchronization state between GUI and Nix
#[derive(Debug, Clone, Default)]
pub struct SyncState {
    /// Current widget states (widget_id -> value)
    pub widget_states: HashMap<WidgetId, WidgetValue>,

    /// Pending changes not yet applied to Nix
    pub pending_changes: Vec<PendingChange>,

    /// Validation errors for current state
    pub validation_errors: Vec<ValidationError>,

    /// Last sync timestamp
    pub last_sync_ms: u64,

    /// Whether Nix config is dirty (needs rebuild)
    pub nix_dirty: bool,
}

/// A pending change to be applied
#[derive(Debug, Clone)]
pub struct PendingChange {
    /// Widget that triggered the change
    pub widget_id: WidgetId,

    /// New value
    pub new_value: WidgetValue,

    /// Nix path to update
    pub nix_path: NixPath,

    /// Generated Nix expression
    pub nix_expr: String,

    /// Semantic confidence of the change
    pub confidence: f64,

    /// Timestamp of change
    pub timestamp_ms: u64,
}

/// Complete GUI bridge for bidirectional synchronization
pub struct GuiBridge {
    /// Widget mapper for bidirectional translation
    pub mapper: WidgetMapper,

    /// HDC translator for semantic operations
    pub translator: HdcTranslator,

    /// Constraint validator
    pub validator: ConstraintValidator,

    /// Current sync state
    pub state: SyncState,

    /// Registered widget bindings
    bindings: Vec<WidgetBinding>,
}

impl GuiBridge {
    /// Create new GUI bridge
    pub fn new() -> Self {
        Self {
            mapper: WidgetMapper::new(),
            translator: HdcTranslator::new(),
            validator: ConstraintValidator::new(),
            state: SyncState::default(),
            bindings: Vec::new(),
        }
    }

    /// Register a widget binding
    pub fn register_binding(&mut self, mut binding: WidgetBinding) {
        // Encode semantic HV for the binding and store it
        binding.semantic_hv = self.translator.encode_path(&binding.nix_path);

        self.bindings.push(binding);
    }

    /// Handle widget value change
    pub fn on_widget_change(
        &mut self,
        widget_id: WidgetId,
        new_value: WidgetValue,
    ) -> MappingResult {
        // Find binding for this widget
        let binding = self
            .bindings
            .iter()
            .find(|b| b.widget_id == widget_id)
            .cloned();

        if let Some(binding) = binding {
            // Transform value to Nix expression
            let nix_expr = self.mapper.widget_to_nix(&binding, &new_value);

            // Validate the change
            let validation = self.validator.validate_change(&binding.nix_path, &nix_expr);

            // Update state
            self.state
                .widget_states
                .insert(widget_id.clone(), new_value.clone());

            if validation.is_valid {
                // Add to pending changes
                self.state.pending_changes.push(PendingChange {
                    widget_id,
                    new_value,
                    nix_path: binding.nix_path.clone(),
                    nix_expr: nix_expr.clone(),
                    confidence: validation.confidence,
                    timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
                });
                self.state.nix_dirty = true;

                MappingResult::Success {
                    nix_expr,
                    confidence: validation.confidence,
                }
            } else {
                self.state
                    .validation_errors
                    .extend(validation.errors.clone());

                MappingResult::ValidationFailed {
                    errors: validation.errors,
                    suggestions: validation.suggestions,
                }
            }
        } else {
            MappingResult::UnknownWidget { widget_id }
        }
    }

    /// Parse Nix configuration and update widget states
    pub fn sync_from_nix(&mut self, nix_content: &str) -> Vec<WidgetUpdate> {
        let mut updates = Vec::new();

        // Parse Nix content and extract values
        let parsed_values = self.mapper.parse_nix_content(nix_content);

        for (nix_path, nix_value) in parsed_values {
            // Find binding for this path
            if let Some(binding) = self.bindings.iter().find(|b| b.nix_path == nix_path) {
                // Transform Nix value to widget value
                let widget_value = self.mapper.nix_to_widget(binding, &nix_value);

                // Update state
                self.state
                    .widget_states
                    .insert(binding.widget_id.clone(), widget_value.clone());

                updates.push(WidgetUpdate {
                    widget_id: binding.widget_id.clone(),
                    new_value: widget_value,
                    source_path: nix_path,
                });
            }
        }

        self.state.last_sync_ms = chrono::Utc::now().timestamp_millis() as u64;
        updates
    }

    /// Generate complete Nix configuration from pending changes
    pub fn generate_nix_diff(&self) -> String {
        let mut diff = String::new();

        for change in &self.state.pending_changes {
            diff.push_str(&format!(
                "# Widget: {:?}\n{} = {};\n\n",
                change.widget_id, change.nix_path.0, change.nix_expr
            ));
        }

        diff
    }

    /// Apply pending changes (mark as applied)
    pub fn apply_pending(&mut self) {
        self.state.pending_changes.clear();
        self.state.nix_dirty = false;
    }

    /// Get semantic search results for a query
    pub fn semantic_search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        self.translator.search(query, &self.bindings, limit)
    }

    /// Get all bindings for a category
    pub fn bindings_for_category(&self, category: ConfigCategory) -> Vec<&WidgetBinding> {
        self.bindings
            .iter()
            .filter(|b| ConfigCategory::from_path(&b.nix_path.0) == category)
            .collect()
    }
}

impl Default for GuiBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Widget update notification
#[derive(Debug, Clone)]
pub struct WidgetUpdate {
    /// Widget to update
    pub widget_id: WidgetId,

    /// New value
    pub new_value: WidgetValue,

    /// Source Nix path
    pub source_path: NixPath,
}

/// Semantic search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Matching binding
    pub binding: WidgetBinding,

    /// Semantic similarity score
    pub similarity: f64,

    /// Category of the result
    pub category: ConfigCategory,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_category_from_path() {
        assert_eq!(
            ConfigCategory::from_path("services.nginx.enable"),
            ConfigCategory::Services
        );
        assert_eq!(
            ConfigCategory::from_path("networking.firewall.enable"),
            ConfigCategory::Networking
        );
        assert_eq!(
            ConfigCategory::from_path("environment.systemPackages"),
            ConfigCategory::Packages
        );
    }

    #[test]
    fn test_gui_bridge_creation() {
        let bridge = GuiBridge::new();
        assert!(bridge.bindings.is_empty());
        assert!(bridge.state.pending_changes.is_empty());
    }
}
