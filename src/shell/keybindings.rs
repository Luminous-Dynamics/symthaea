// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! G1: Vim/Emacs Keybindings
//!
//! Modal editing support for the Symthaea shell with configurable
//! keybinding schemes.

use std::collections::HashMap;

/// Editing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EditingMode {
    /// Normal (command) mode
    Normal,
    /// Insert mode
    #[default]
    Insert,
    /// Visual (selection) mode
    Visual,
    /// Command-line mode (for ex commands)
    Command,
}

/// Keybinding scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeybindingScheme {
    /// Emacs-style (default readline)
    #[default]
    Emacs,
    /// Vim-style modal editing
    Vim,
    /// Minimal (basic editing)
    Minimal,
}

/// A key event
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyEvent {
    /// Single character
    Char(char),
    /// Control + character
    Ctrl(char),
    /// Alt/Meta + character
    Alt(char),
    /// Function keys (F1-F12)
    F(u8),
    /// Arrow keys
    Up,
    Down,
    Left,
    Right,
    /// Special keys
    Enter,
    Tab,
    Backspace,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,
    Escape,
    /// Sequence (for vim commands like "dd")
    Sequence(String),
}

impl KeyEvent {
    /// Parse from crossterm key event
    pub fn from_char(c: char) -> Self {
        KeyEvent::Char(c)
    }

    /// Check if this is an escape key
    pub fn is_escape(&self) -> bool {
        matches!(self, KeyEvent::Escape)
    }

    /// Check if this is enter
    pub fn is_enter(&self) -> bool {
        matches!(self, KeyEvent::Enter)
    }
}

/// Action to perform on keybinding
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditAction {
    /// Insert character at cursor
    InsertChar(char),
    /// Delete character before cursor
    DeleteBack,
    /// Delete character at cursor
    DeleteForward,
    /// Delete word before cursor
    DeleteWord,
    /// Delete to end of line
    DeleteToEnd,
    /// Delete to start of line
    DeleteToStart,
    /// Delete entire line
    DeleteLine,
    /// Move cursor left
    CursorLeft,
    /// Move cursor right
    CursorRight,
    /// Move to word start
    WordLeft,
    /// Move to word end
    WordRight,
    /// Move to line start
    LineStart,
    /// Move to line end
    LineEnd,
    /// Previous history entry
    HistoryPrev,
    /// Next history entry
    HistoryNext,
    /// Search history backwards
    HistorySearchBack,
    /// Search history forwards
    HistorySearchForward,
    /// Accept current line
    Accept,
    /// Cancel/abort current input
    Cancel,
    /// Trigger completion
    Complete,
    /// Clear screen
    ClearScreen,
    /// Transpose characters
    Transpose,
    /// Uppercase word
    UppercaseWord,
    /// Lowercase word
    LowercaseWord,
    /// Capitalize word
    CapitalizeWord,
    /// Yank (paste) from kill ring
    Yank,
    /// Kill (cut) region
    Kill,
    /// Undo last edit
    Undo,
    /// Redo last undone edit
    Redo,
    /// Switch to insert mode (vim)
    EnterInsert,
    /// Switch to normal mode (vim)
    EnterNormal,
    /// Switch to visual mode (vim)
    EnterVisual,
    /// Enter command mode (vim :)
    EnterCommand,
    /// Execute vim ex command
    ExCommand(String),
    /// No action
    Noop,
}

/// Keybinding manager
pub struct KeybindingManager {
    /// Current scheme
    scheme: KeybindingScheme,
    /// Current mode (for vim)
    mode: EditingMode,
    /// Emacs bindings
    emacs_bindings: HashMap<KeyEvent, EditAction>,
    /// Vim normal mode bindings
    vim_normal_bindings: HashMap<KeyEvent, EditAction>,
    /// Vim insert mode bindings
    vim_insert_bindings: HashMap<KeyEvent, EditAction>,
    /// Custom bindings (override defaults)
    custom_bindings: HashMap<(EditingMode, KeyEvent), EditAction>,
    /// Pending sequence (for vim multi-key commands)
    pending_sequence: String,
    /// Count prefix (for vim repeat)
    count_prefix: Option<usize>,
}

impl KeybindingManager {
    /// Create with default Emacs bindings
    pub fn new() -> Self {
        let mut mgr = Self {
            scheme: KeybindingScheme::Emacs,
            mode: EditingMode::Insert,
            emacs_bindings: HashMap::new(),
            vim_normal_bindings: HashMap::new(),
            vim_insert_bindings: HashMap::new(),
            custom_bindings: HashMap::new(),
            pending_sequence: String::new(),
            count_prefix: None,
        };
        mgr.init_emacs_bindings();
        mgr.init_vim_bindings();
        mgr
    }

    /// Create with specific scheme
    pub fn with_scheme(scheme: KeybindingScheme) -> Self {
        let mut mgr = Self::new();
        mgr.set_scheme(scheme);
        mgr
    }

    /// Set keybinding scheme
    pub fn set_scheme(&mut self, scheme: KeybindingScheme) {
        self.scheme = scheme;
        self.mode = match scheme {
            KeybindingScheme::Vim => EditingMode::Normal,
            _ => EditingMode::Insert,
        };
    }

    /// Get current scheme
    pub fn scheme(&self) -> KeybindingScheme {
        self.scheme
    }

    /// Get current mode
    pub fn mode(&self) -> EditingMode {
        self.mode
    }

    /// Set editing mode
    pub fn set_mode(&mut self, mode: EditingMode) {
        self.mode = mode;
        self.pending_sequence.clear();
        self.count_prefix = None;
    }

    /// Add custom keybinding
    pub fn bind(&mut self, mode: EditingMode, key: KeyEvent, action: EditAction) {
        self.custom_bindings.insert((mode, key), action);
    }

    /// Process a key event and return action
    pub fn process(&mut self, key: KeyEvent) -> EditAction {
        // Check custom bindings first
        if let Some(action) = self.custom_bindings.get(&(self.mode, key.clone())) {
            return action.clone();
        }

        match self.scheme {
            KeybindingScheme::Emacs => self.process_emacs(key),
            KeybindingScheme::Vim => self.process_vim(key),
            KeybindingScheme::Minimal => self.process_minimal(key),
        }
    }

    fn process_emacs(&self, key: KeyEvent) -> EditAction {
        self.emacs_bindings.get(&key).cloned().unwrap_or({
            match key {
                KeyEvent::Char(c) => EditAction::InsertChar(c),
                _ => EditAction::Noop,
            }
        })
    }

    fn process_vim(&mut self, key: KeyEvent) -> EditAction {
        match self.mode {
            EditingMode::Normal => self.process_vim_normal(key),
            EditingMode::Insert => self.process_vim_insert(key),
            EditingMode::Visual => self.process_vim_visual(key),
            EditingMode::Command => self.process_vim_command(key),
        }
    }

    fn process_vim_normal(&mut self, key: KeyEvent) -> EditAction {
        // Handle count prefix
        if let KeyEvent::Char(c) = &key {
            if c.is_ascii_digit() && (*c != '0' || self.count_prefix.is_some()) {
                let digit = c.to_digit(10).expect("c verified as ascii digit above") as usize;
                self.count_prefix = Some(self.count_prefix.unwrap_or(0) * 10 + digit);
                return EditAction::Noop;
            }
        }

        // Handle sequences
        self.pending_sequence.push_str(&match &key {
            KeyEvent::Char(c) => c.to_string(),
            _ => String::new(),
        });

        // Check for complete sequences
        let action = match self.pending_sequence.as_str() {
            "dd" => {
                self.pending_sequence.clear();
                EditAction::DeleteLine
            }
            "dw" => {
                self.pending_sequence.clear();
                EditAction::DeleteWord
            }
            "d$" => {
                self.pending_sequence.clear();
                EditAction::DeleteToEnd
            }
            "d0" => {
                self.pending_sequence.clear();
                EditAction::DeleteToStart
            }
            "yy" => {
                self.pending_sequence.clear();
                EditAction::Kill // Yank line
            }
            _ if self.pending_sequence.len() > 2 => {
                self.pending_sequence.clear();
                EditAction::Noop
            }
            _ if self.pending_sequence.starts_with('d')
                || self.pending_sequence.starts_with('y')
                || self.pending_sequence.starts_with('c') =>
            {
                EditAction::Noop // Wait for more input
            }
            _ => {
                self.pending_sequence.clear();
                self.vim_normal_bindings
                    .get(&key)
                    .cloned()
                    .unwrap_or(EditAction::Noop)
            }
        };

        self.count_prefix = None;
        action
    }

    fn process_vim_insert(&self, key: KeyEvent) -> EditAction {
        self.vim_insert_bindings.get(&key).cloned().unwrap_or({
            match key {
                KeyEvent::Char(c) => EditAction::InsertChar(c),
                KeyEvent::Escape => EditAction::EnterNormal,
                _ => EditAction::Noop,
            }
        })
    }

    fn process_vim_visual(&mut self, key: KeyEvent) -> EditAction {
        match key {
            KeyEvent::Escape => {
                self.mode = EditingMode::Normal;
                EditAction::EnterNormal
            }
            KeyEvent::Char('d') | KeyEvent::Char('x') => {
                self.mode = EditingMode::Normal;
                EditAction::Kill
            }
            KeyEvent::Char('y') => {
                self.mode = EditingMode::Normal;
                EditAction::Kill // Copy selection
            }
            _ => EditAction::Noop,
        }
    }

    fn process_vim_command(&mut self, key: KeyEvent) -> EditAction {
        match key {
            KeyEvent::Escape => {
                self.pending_sequence.clear();
                self.mode = EditingMode::Normal;
                EditAction::EnterNormal
            }
            KeyEvent::Enter => {
                let cmd = self.pending_sequence.clone();
                self.pending_sequence.clear();
                self.mode = EditingMode::Normal;
                EditAction::ExCommand(cmd)
            }
            KeyEvent::Char(c) => {
                self.pending_sequence.push(c);
                EditAction::Noop
            }
            KeyEvent::Backspace => {
                self.pending_sequence.pop();
                if self.pending_sequence.is_empty() {
                    self.mode = EditingMode::Normal;
                    EditAction::EnterNormal
                } else {
                    EditAction::Noop
                }
            }
            _ => EditAction::Noop,
        }
    }

    fn process_minimal(&self, key: KeyEvent) -> EditAction {
        match key {
            KeyEvent::Char(c) => EditAction::InsertChar(c),
            KeyEvent::Backspace => EditAction::DeleteBack,
            KeyEvent::Delete => EditAction::DeleteForward,
            KeyEvent::Left => EditAction::CursorLeft,
            KeyEvent::Right => EditAction::CursorRight,
            KeyEvent::Home => EditAction::LineStart,
            KeyEvent::End => EditAction::LineEnd,
            KeyEvent::Up => EditAction::HistoryPrev,
            KeyEvent::Down => EditAction::HistoryNext,
            KeyEvent::Enter => EditAction::Accept,
            KeyEvent::Tab => EditAction::Complete,
            KeyEvent::Ctrl('c') => EditAction::Cancel,
            _ => EditAction::Noop,
        }
    }

    fn init_emacs_bindings(&mut self) {
        use EditAction::*;
        use KeyEvent::*;

        let bindings = [
            // Navigation
            (Ctrl('a'), LineStart),
            (Ctrl('e'), LineEnd),
            (Ctrl('b'), CursorLeft),
            (Ctrl('f'), CursorRight),
            (Alt('b'), WordLeft),
            (Alt('f'), WordRight),
            (Left, CursorLeft),
            (Right, CursorRight),
            (Home, LineStart),
            (End, LineEnd),
            // Deletion
            (Ctrl('d'), DeleteForward),
            (Backspace, DeleteBack),
            (Delete, DeleteForward),
            (Ctrl('h'), DeleteBack),
            (Ctrl('w'), DeleteWord),
            (Ctrl('k'), DeleteToEnd),
            (Ctrl('u'), DeleteToStart),
            (Alt('d'), DeleteWord),
            // History
            (Up, HistoryPrev),
            (Down, HistoryNext),
            (Ctrl('p'), HistoryPrev),
            (Ctrl('n'), HistoryNext),
            (Ctrl('r'), HistorySearchBack),
            (Ctrl('s'), HistorySearchForward),
            // Actions
            (Enter, Accept),
            (Tab, Complete),
            (Ctrl('c'), Cancel),
            (Ctrl('l'), ClearScreen),
            (Ctrl('t'), Transpose),
            (Ctrl('y'), Yank),
            (Ctrl('_'), Undo),
            (Ctrl('x'), Redo), // Ctrl-X Ctrl-U in full Emacs
            // Case
            (Alt('u'), UppercaseWord),
            (Alt('l'), LowercaseWord),
            (Alt('c'), CapitalizeWord),
        ];

        for (key, action) in bindings {
            self.emacs_bindings.insert(key, action);
        }
    }

    fn init_vim_bindings(&mut self) {
        use EditAction::*;
        use KeyEvent::*;

        // Normal mode
        let normal_bindings = [
            // Mode switching
            (KeyEvent::Char('i'), EnterInsert),
            (KeyEvent::Char('a'), EnterInsert), // Should also move cursor right
            (KeyEvent::Char('I'), EnterInsert), // Should also go to line start
            (KeyEvent::Char('A'), EnterInsert), // Should also go to line end
            (KeyEvent::Char('v'), EnterVisual),
            (KeyEvent::Char(':'), EnterCommand),
            // Navigation
            (KeyEvent::Char('h'), CursorLeft),
            (KeyEvent::Char('l'), CursorRight),
            (KeyEvent::Char('0'), LineStart),
            (KeyEvent::Char('$'), LineEnd),
            (KeyEvent::Char('^'), LineStart),
            (KeyEvent::Char('w'), WordRight),
            (KeyEvent::Char('b'), WordLeft),
            (Left, CursorLeft),
            (Right, CursorRight),
            (Home, LineStart),
            (End, LineEnd),
            // Deletion (single char)
            (KeyEvent::Char('x'), DeleteForward),
            (KeyEvent::Char('X'), DeleteBack),
            // History
            (KeyEvent::Char('k'), HistoryPrev),
            (KeyEvent::Char('j'), HistoryNext),
            (Up, HistoryPrev),
            (Down, HistoryNext),
            // Actions
            (Enter, Accept),
            (KeyEvent::Char('p'), Yank),
            (KeyEvent::Char('u'), Undo),
            (Ctrl('r'), Redo),
        ];

        for (key, action) in normal_bindings {
            self.vim_normal_bindings.insert(key, action);
        }

        // Insert mode
        let insert_bindings = [
            (Escape, EnterNormal),
            (Ctrl('c'), EnterNormal),
            (Backspace, DeleteBack),
            (Delete, DeleteForward),
            (Left, CursorLeft),
            (Right, CursorRight),
            (Up, HistoryPrev),
            (Down, HistoryNext),
            (Home, LineStart),
            (End, LineEnd),
            (Tab, Complete),
            (Enter, Accept),
            (Ctrl('w'), DeleteWord),
            (Ctrl('u'), DeleteToStart),
        ];

        for (key, action) in insert_bindings {
            self.vim_insert_bindings.insert(key, action);
        }
    }

    /// Get mode indicator for display
    pub fn mode_indicator(&self) -> &'static str {
        match (self.scheme, self.mode) {
            (KeybindingScheme::Vim, EditingMode::Normal) => "[N]",
            (KeybindingScheme::Vim, EditingMode::Insert) => "[I]",
            (KeybindingScheme::Vim, EditingMode::Visual) => "[V]",
            (KeybindingScheme::Vim, EditingMode::Command) => "[:]",
            _ => "",
        }
    }

    /// Check if in insert mode
    pub fn is_inserting(&self) -> bool {
        self.mode == EditingMode::Insert || self.scheme != KeybindingScheme::Vim
    }
}

impl Default for KeybindingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emacs_bindings() {
        let mgr = KeybindingManager::new();

        assert_eq!(
            mgr.emacs_bindings.get(&KeyEvent::Ctrl('a')),
            Some(&EditAction::LineStart)
        );
        assert_eq!(
            mgr.emacs_bindings.get(&KeyEvent::Ctrl('e')),
            Some(&EditAction::LineEnd)
        );
    }

    #[test]
    fn test_vim_mode_switching() {
        let mut mgr = KeybindingManager::with_scheme(KeybindingScheme::Vim);

        assert_eq!(mgr.mode(), EditingMode::Normal);

        let action = mgr.process(KeyEvent::Char('i'));
        assert_eq!(action, EditAction::EnterInsert);
    }

    #[test]
    fn test_custom_binding() {
        let mut mgr = KeybindingManager::new();
        mgr.bind(EditingMode::Insert, KeyEvent::Ctrl('z'), EditAction::Undo);

        let action = mgr.process(KeyEvent::Ctrl('z'));
        assert_eq!(action, EditAction::Undo);
    }

    #[test]
    fn test_mode_indicator() {
        let mut mgr = KeybindingManager::with_scheme(KeybindingScheme::Vim);

        assert_eq!(mgr.mode_indicator(), "[N]");

        mgr.set_mode(EditingMode::Insert);
        assert_eq!(mgr.mode_indicator(), "[I]");
    }

    #[test]
    fn test_minimal_scheme() {
        let mgr = KeybindingManager::with_scheme(KeybindingScheme::Minimal);

        let action = mgr.process_minimal(KeyEvent::Char('a'));
        assert_eq!(action, EditAction::InsertChar('a'));

        let action = mgr.process_minimal(KeyEvent::Backspace);
        assert_eq!(action, EditAction::DeleteBack);
    }
}
