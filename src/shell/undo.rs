// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! G4: Undo/Redo Stack
//!
//! Change history for configuration edits with rollback support.

use std::collections::VecDeque;
use std::time::SystemTime;

/// An action that can be undone/redone
#[derive(Debug, Clone)]
pub struct UndoAction {
    /// Unique action ID
    pub id: u64,
    /// Description of the action
    pub description: String,
    /// Type of action
    pub action_type: ActionType,
    /// Forward action data
    pub forward: ActionData,
    /// Reverse action data (to undo)
    pub reverse: ActionData,
    /// When the action was performed
    pub timestamp: SystemTime,
    /// Whether this action has been applied
    pub applied: bool,
}

/// Type of undoable action
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionType {
    /// Text edit in shell input
    TextEdit,
    /// Configuration file change
    ConfigChange,
    /// Package operation
    Package,
    /// Service operation
    Service,
    /// Option change
    Option,
    /// Composite (multiple sub-actions)
    Composite,
}

/// Data for performing/reversing an action
#[derive(Debug, Clone)]
pub enum ActionData {
    /// Text replacement
    TextReplace {
        /// Position in text
        position: usize,
        /// Text to insert
        insert: String,
        /// Text to delete
        delete: String,
    },
    /// File content change
    FileChange {
        /// File path
        path: String,
        /// Old content
        old_content: String,
        /// New content
        new_content: String,
    },
    /// Nix expression change
    NixChange {
        /// Option path (e.g., "services.nginx.enable")
        path: String,
        /// Old value (Nix expression)
        old_value: String,
        /// New value (Nix expression)
        new_value: String,
    },
    /// Package add/remove
    PackageChange {
        /// Package name
        package: String,
        /// Whether it was added (vs removed)
        added: bool,
    },
    /// Service enable/disable
    ServiceChange {
        /// Service name
        service: String,
        /// Whether it was enabled (vs disabled)
        enabled: bool,
    },
    /// Composite of multiple actions
    Composite(Vec<ActionData>),
    /// No-op (for markers)
    Noop,
}

impl ActionData {
    /// Get the reverse action data
    pub fn reverse(&self) -> Self {
        match self {
            ActionData::TextReplace {
                position,
                insert,
                delete,
            } => ActionData::TextReplace {
                position: *position,
                insert: delete.clone(),
                delete: insert.clone(),
            },
            ActionData::FileChange {
                path,
                old_content,
                new_content,
            } => ActionData::FileChange {
                path: path.clone(),
                old_content: new_content.clone(),
                new_content: old_content.clone(),
            },
            ActionData::NixChange {
                path,
                old_value,
                new_value,
            } => ActionData::NixChange {
                path: path.clone(),
                old_value: new_value.clone(),
                new_value: old_value.clone(),
            },
            ActionData::PackageChange { package, added } => ActionData::PackageChange {
                package: package.clone(),
                added: !added,
            },
            ActionData::ServiceChange { service, enabled } => ActionData::ServiceChange {
                service: service.clone(),
                enabled: !enabled,
            },
            ActionData::Composite(actions) => {
                ActionData::Composite(actions.iter().rev().map(|a| a.reverse()).collect())
            }
            ActionData::Noop => ActionData::Noop,
        }
    }
}

impl UndoAction {
    /// Create a new undo action
    pub fn new(
        description: impl Into<String>,
        action_type: ActionType,
        forward: ActionData,
    ) -> Self {
        let reverse = forward.reverse();
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            description: description.into(),
            action_type,
            forward,
            reverse,
            timestamp: SystemTime::now(),
            applied: false,
        }
    }

    /// Create a text edit action
    pub fn text_edit(
        description: impl Into<String>,
        position: usize,
        insert: impl Into<String>,
        delete: impl Into<String>,
    ) -> Self {
        Self::new(
            description,
            ActionType::TextEdit,
            ActionData::TextReplace {
                position,
                insert: insert.into(),
                delete: delete.into(),
            },
        )
    }

    /// Create a config change action
    pub fn config_change(
        path: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> Self {
        let path = path.into();
        Self::new(
            format!("Change {path}"),
            ActionType::Option,
            ActionData::NixChange {
                path,
                old_value: old_value.into(),
                new_value: new_value.into(),
            },
        )
    }

    /// Create a package add action
    pub fn package_add(package: impl Into<String>) -> Self {
        let pkg = package.into();
        Self::new(
            format!("Add package {pkg}"),
            ActionType::Package,
            ActionData::PackageChange {
                package: pkg,
                added: true,
            },
        )
    }

    /// Create a package remove action
    pub fn package_remove(package: impl Into<String>) -> Self {
        let pkg = package.into();
        Self::new(
            format!("Remove package {pkg}"),
            ActionType::Package,
            ActionData::PackageChange {
                package: pkg,
                added: false,
            },
        )
    }

    /// Create a service enable action
    pub fn service_enable(service: impl Into<String>) -> Self {
        let svc = service.into();
        Self::new(
            format!("Enable service {svc}"),
            ActionType::Service,
            ActionData::ServiceChange {
                service: svc,
                enabled: true,
            },
        )
    }

    /// Create a composite action
    pub fn composite(description: impl Into<String>, actions: Vec<UndoAction>) -> Self {
        let forward_data: Vec<ActionData> = actions.into_iter().map(|a| a.forward).collect();
        Self::new(
            description,
            ActionType::Composite,
            ActionData::Composite(forward_data),
        )
    }
}

/// Undo/redo stack
pub struct UndoStack {
    /// Actions that can be undone
    undo_stack: VecDeque<UndoAction>,
    /// Actions that can be redone
    redo_stack: VecDeque<UndoAction>,
    /// Maximum stack size
    max_size: usize,
    /// Group ID for action grouping
    current_group: Option<u64>,
    /// Pending grouped actions
    pending_group: Vec<UndoAction>,
}

impl UndoStack {
    /// Create a new undo stack
    pub fn new() -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_size: 100,
            current_group: None,
            pending_group: Vec::new(),
        }
    }

    /// Create with custom max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size,
            ..Self::new()
        }
    }

    /// Push an action onto the undo stack
    pub fn push(&mut self, mut action: UndoAction) {
        action.applied = true;

        // If grouping, add to pending group
        if self.current_group.is_some() {
            self.pending_group.push(action);
            return;
        }

        // Clear redo stack (new action invalidates redo history)
        self.redo_stack.clear();

        // Add to undo stack
        self.undo_stack.push_back(action);

        // Trim if over max size
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.pop_front();
        }
    }

    /// Start grouping actions
    pub fn begin_group(&mut self) {
        static GROUP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        self.current_group = Some(GROUP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
    }

    /// End grouping and create composite action
    pub fn end_group(&mut self, description: impl Into<String>) {
        if self.current_group.is_none() {
            return;
        }

        self.current_group = None;

        if !self.pending_group.is_empty() {
            let actions = std::mem::take(&mut self.pending_group);
            let composite = UndoAction::composite(description, actions);
            self.push(composite);
        }
    }

    /// Cancel current group
    pub fn cancel_group(&mut self) {
        self.current_group = None;
        self.pending_group.clear();
    }

    /// Undo the last action
    pub fn undo(&mut self) -> Option<UndoAction> {
        if let Some(mut action) = self.undo_stack.pop_back() {
            action.applied = false;
            self.redo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }

    /// Redo the last undone action
    pub fn redo(&mut self) -> Option<UndoAction> {
        if let Some(mut action) = self.redo_stack.pop_back() {
            action.applied = true;
            self.undo_stack.push_back(action.clone());
            Some(action)
        } else {
            None
        }
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get next undo action description
    pub fn peek_undo(&self) -> Option<&str> {
        self.undo_stack.back().map(|a| a.description.as_str())
    }

    /// Get next redo action description
    pub fn peek_redo(&self) -> Option<&str> {
        self.redo_stack.back().map(|a| a.description.as_str())
    }

    /// Get undo history (most recent first)
    pub fn undo_history(&self) -> Vec<&UndoAction> {
        self.undo_stack.iter().rev().collect()
    }

    /// Get redo history (most recent first)
    pub fn redo_history(&self) -> Vec<&UndoAction> {
        self.redo_stack.iter().rev().collect()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Get total history size
    pub fn len(&self) -> usize {
        self.undo_stack.len() + self.redo_stack.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.undo_stack.is_empty() && self.redo_stack.is_empty()
    }
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Text buffer with undo support
pub struct UndoableBuffer {
    /// Current text content
    content: String,
    /// Cursor position
    cursor: usize,
    /// Undo stack
    undo: UndoStack,
    /// Saved position (for detecting changes)
    saved_at: Option<u64>,
}

impl UndoableBuffer {
    /// Create new empty buffer
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor: 0,
            undo: UndoStack::new(),
            saved_at: None,
        }
    }

    /// Create with initial content
    pub fn with_content(content: impl Into<String>) -> Self {
        let content = content.into();
        let cursor = content.len();
        Self {
            content,
            cursor,
            undo: UndoStack::new(),
            saved_at: None,
        }
    }

    /// Get current content
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get cursor position
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Set cursor position
    pub fn set_cursor(&mut self, pos: usize) {
        self.cursor = pos.min(self.content.len());
    }

    /// Insert text at cursor (with undo support)
    pub fn insert(&mut self, text: &str) {
        let action = UndoAction::text_edit(
            format!(
                "Insert '{}'",
                if text.len() > 20 {
                    format!("{}...", &text[..20])
                } else {
                    text.to_string()
                }
            ),
            self.cursor,
            text,
            "",
        );

        self.content.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.undo.push(action);
    }

    /// Delete text before cursor (with undo support)
    pub fn delete_back(&mut self, count: usize) {
        if self.cursor == 0 || count == 0 {
            return;
        }

        let delete_count = count.min(self.cursor);
        let start = self.cursor - delete_count;
        let deleted: String = self.content[start..self.cursor].to_string();

        let action = UndoAction::text_edit("Delete", start, "", &deleted);

        self.content.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.undo.push(action);
    }

    /// Delete text after cursor (with undo support)
    pub fn delete_forward(&mut self, count: usize) {
        if self.cursor >= self.content.len() || count == 0 {
            return;
        }

        let delete_count = count.min(self.content.len() - self.cursor);
        let end = self.cursor + delete_count;
        let deleted: String = self.content[self.cursor..end].to_string();

        let action = UndoAction::text_edit("Delete", self.cursor, "", &deleted);

        self.content.replace_range(self.cursor..end, "");
        self.undo.push(action);
    }

    /// Replace a range of text
    pub fn replace(&mut self, start: usize, end: usize, text: &str) {
        let start = start.min(self.content.len());
        let end = end.min(self.content.len()).max(start);

        let deleted: String = self.content[start..end].to_string();

        let action = UndoAction::text_edit("Replace", start, text, &deleted);

        self.content.replace_range(start..end, text);
        self.cursor = start + text.len();
        self.undo.push(action);
    }

    /// Clear content
    pub fn clear(&mut self) {
        if self.content.is_empty() {
            return;
        }

        let action = UndoAction::text_edit("Clear", 0, "", &self.content);

        self.content.clear();
        self.cursor = 0;
        self.undo.push(action);
    }

    /// Undo last change
    pub fn undo(&mut self) -> bool {
        if let Some(action) = self.undo.undo() {
            self.apply_action_data(&action.reverse);
            true
        } else {
            false
        }
    }

    /// Redo last undone change
    pub fn redo(&mut self) -> bool {
        if let Some(action) = self.undo.redo() {
            self.apply_action_data(&action.forward);
            true
        } else {
            false
        }
    }

    fn apply_action_data(&mut self, data: &ActionData) {
        match data {
            ActionData::TextReplace {
                position,
                insert,
                delete,
            } => {
                let pos = *position;
                let end = pos + delete.len();

                if end <= self.content.len() {
                    self.content.replace_range(pos..end, insert);
                    self.cursor = pos + insert.len();
                }
            }
            ActionData::Composite(actions) => {
                for action in actions {
                    self.apply_action_data(action);
                }
            }
            _ => {}
        }
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        self.undo.can_undo()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        self.undo.can_redo()
    }

    /// Mark current state as saved
    pub fn mark_saved(&mut self) {
        self.saved_at = self.undo.undo_stack.back().map(|a| a.id);
    }

    /// Check if buffer has unsaved changes
    pub fn is_modified(&self) -> bool {
        let current = self.undo.undo_stack.back().map(|a| a.id);
        current != self.saved_at
    }
}

impl Default for UndoableBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_stack() {
        let mut stack = UndoStack::new();

        stack.push(UndoAction::text_edit("Type 'a'", 0, "a", ""));
        stack.push(UndoAction::text_edit("Type 'b'", 1, "b", ""));

        assert!(stack.can_undo());
        assert_eq!(stack.peek_undo(), Some("Type 'b'"));

        let undone = stack.undo().unwrap();
        assert_eq!(undone.description, "Type 'b'");
        assert!(stack.can_redo());

        let redone = stack.redo().unwrap();
        assert_eq!(redone.description, "Type 'b'");
    }

    #[test]
    fn test_undoable_buffer() {
        let mut buf = UndoableBuffer::new();

        buf.insert("Hello");
        assert_eq!(buf.content(), "Hello");

        buf.insert(" World");
        assert_eq!(buf.content(), "Hello World");

        buf.undo();
        assert_eq!(buf.content(), "Hello");

        buf.redo();
        assert_eq!(buf.content(), "Hello World");
    }

    #[test]
    fn test_buffer_delete() {
        let mut buf = UndoableBuffer::with_content("Hello World");

        buf.set_cursor(5);
        buf.delete_back(5);
        assert_eq!(buf.content(), " World");

        buf.undo();
        assert_eq!(buf.content(), "Hello World");
    }

    #[test]
    fn test_action_grouping() {
        let mut stack = UndoStack::new();

        stack.begin_group();
        stack.push(UndoAction::text_edit("Type 'a'", 0, "a", ""));
        stack.push(UndoAction::text_edit("Type 'b'", 1, "b", ""));
        stack.push(UndoAction::text_edit("Type 'c'", 2, "c", ""));
        stack.end_group("Type 'abc'");

        // Should be a single composite action
        assert_eq!(stack.undo_stack.len(), 1);
        assert_eq!(stack.peek_undo(), Some("Type 'abc'"));
    }

    #[test]
    fn test_config_change_action() {
        let action = UndoAction::config_change("services.nginx.enable", "false", "true");

        assert_eq!(action.action_type, ActionType::Option);

        if let ActionData::NixChange {
            path, new_value, ..
        } = &action.forward
        {
            assert_eq!(path, "services.nginx.enable");
            assert_eq!(new_value, "true");
        } else {
            panic!("Expected NixChange");
        }
    }
}
