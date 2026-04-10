// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Shared Leptos component library for the Mycelix ecosystem.
//!
//! Provides reusable UI components for Holochain-backed Leptos applications
//! including EduNet, LUCID, Observatory, and other Mycelix frontends.
//!
//! # Components
//!
//! - [`HolochainProvider`] — Generic context provider wrapping a [`HolochainTransport`]
//! - [`HolochainProviderAuto`] — Auto-connecting provider with mock fallback
//! - [`ConnectionStatusIndicator`] — Navbar-sized connection state indicator
//! - [`ConnectionBadge`] — Compact connection badge for navbars
//! - [`TrustBadge`] — Consciousness-gated trust tier badge
//! - [`TierGate`] — Gating component that shows/hides children by tier
//! - [`LoadingSkeleton`] — Pulsing skeleton placeholder for loading states
//! - [`AppErrorBoundary`] — Styled error boundary with retry
//! - [`ZomeCallButton`] — Button with loading/success/error states
//! - [`ProgressBar`] — Configurable progress bar
//! - [`StatCard`] — Labeled statistic card
//!
//! # Providers
//!
//! - [`provide_theme_context`] — localStorage-persisted theme with `data-theme` attribute
//! - [`provide_consciousness_context`] — 4D consciousness profile (identity/reputation/community/engagement)
//! - [`provide_thermodynamic_context`] — Battery API polling with torpor detection
//! - [`provide_homeostasis_context`] — Multi-counter pending work tracking
//! - [`init_consciousness_ui`] — Wires consciousness + thermodynamic state to CSS variables
//!
//! # Hooks
//!
//! - [`use_holochain`] — Retrieve the [`HolochainTransport`] from context
//! - [`use_zome_call`] — Reactive helper for executing zome calls
//! - [`use_theme_state`] — Retrieve the current theme state
//! - [`use_consciousness`] — Retrieve consciousness profile and tier
//! - [`use_thermodynamic`] — Retrieve device energy and torpor state
//! - [`use_homeostasis`] — Retrieve homeostasis (all-pending-zero) state

// --- Core transport ---
pub mod provider;
pub mod holochain_provider;
pub mod connection_status;

// --- UI components ---
pub mod trust_badge;
pub mod loading;
pub mod error_boundary;
pub mod zome_call_button;
pub mod progress_bar;
pub mod stat_card;
pub mod tier_gate;
pub mod app_shell;
pub mod forms;
pub mod modal;
pub mod data_table;
pub mod tabs;
pub mod empty_state;
pub mod search_bar;
pub mod badge;
pub mod cluster_launcher;
pub mod indlela;
pub mod spore_bridge;
pub mod local_identity;

// --- Reactive systems ---
pub mod theme;
pub mod consciousness;
pub mod consciousness_ui;
pub mod thermodynamic;
pub mod homeostasis;
pub mod toasts;

// --- Utilities ---
pub mod util;

// Re-exports for convenience — transport
pub use provider::{HolochainProvider, use_holochain, use_zome_call};
pub use holochain_provider::{
    HolochainProviderAuto, HolochainProviderConfig, HolochainCtx,
    ConnectionStatus, ConnectStrategy, ConnectionBadge,
};
// Note: holochain_provider::use_holochain() returns concrete HolochainCtx,
// while provider::use_holochain::<T>() is generic. Access the concrete one
// via mycelix_leptos_core::holochain_provider::use_holochain().

// Re-exports — types
pub use personal_leptos_types::TrustTier;

// Re-exports — UI components
pub use connection_status::ConnectionStatusIndicator;
pub use trust_badge::TrustBadge;
pub use loading::LoadingSkeleton;
pub use error_boundary::AppErrorBoundary;
pub use zome_call_button::ZomeCallButton;
pub use progress_bar::ProgressBar;
pub use stat_card::StatCard;
pub use tier_gate::TierGate;

// Re-exports — reactive systems
pub use theme::{AppTheme, ThemeState, provide_theme_context, use_theme_state};
pub use consciousness::{
    ConsciousnessProfile, ConsciousnessState,
    provide_consciousness_context, use_consciousness,
};
pub use consciousness_ui::init_consciousness_ui;
pub use thermodynamic::{ThermodynamicState, provide_thermodynamic_context, use_thermodynamic};
pub use homeostasis::{HomeostasisState, provide_homeostasis_context, use_homeostasis};
pub use toasts::{Toast, ToastKind, ToastState, ToastContainer, provide_toast_context, use_toasts};

// Re-exports — new components
pub use app_shell::{AppShell, AppNav, MobileBottomNav, NavLink, NavTab};
pub use forms::{FormField, TextInput, TextArea, Select, Checkbox, SelectOption};
pub use modal::{Modal, ConfirmDialog, ModalSize};
pub use data_table::{DataTable, Column, Pagination};
pub use tabs::{Tabs, TabPanel};
pub use empty_state::EmptyState;
pub use search_bar::SearchBar;
pub use badge::{Badge, BadgeVariant, StatusDot};
pub use cluster_launcher::{ClusterLauncher, ClusterLink, default_clusters};
pub use indlela::{GrowthStage, community_warmth, knowledge_freshness};
pub use spore_bridge::{SporeState, provide_spore_bridge, use_spore};
pub use local_identity::{LocalIdentity, local_did, provide_local_identity, use_local_identity, load_json, save_json};

// Re-exports — utilities
pub use util::{set_css_var, set_root_attribute};
