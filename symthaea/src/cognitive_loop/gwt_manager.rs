// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GWT Manager — groups Global Workspace Theory integration fields.
//!
//! Consolidates 3 previously scattered GWT fields from CognitiveLoopService
//! into a single coherent module.

use crate::consciousness::gwt_integration::UnifiedGlobalWorkspace;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

/// Consolidated GWT manager.
///
/// Groups the GWT workspace, memory consolidation flag, and perception
/// broadcast counter into a single struct.
pub(crate) struct GwtManager {
    /// Global Workspace Theory integration.
    /// When enabled, submits encodings to workspace for conscious broadcast.
    pub gwt: Option<UnifiedGlobalWorkspace>,

    /// GWT handler flag: memory consolidation requested via broadcast.
    pub memory_flag: Arc<AtomicBool>,

    /// GWT handler counter: perception broadcast events consumed.
    pub perception_count: Arc<AtomicUsize>,
}

impl Default for GwtManager {
    fn default() -> Self {
        Self {
            gwt: None,
            memory_flag: Arc::new(AtomicBool::new(false)),
            perception_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl GwtManager {
    /// Create a new GwtManager with pre-built Arc flags (for sharing with handlers).
    pub fn new(
        gwt: Option<UnifiedGlobalWorkspace>,
        memory_flag: Arc<AtomicBool>,
        perception_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            gwt,
            memory_flag,
            perception_count,
        }
    }
}
