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
//! - [`StatCard`] — Labeled statistic card (static number)
//! - [`TelemetryLine`] — Horizontal sparkline strip for streaming metrics (complements StatCard)
//! - [`GraphNode`] + [`GraphEdge`] — Relational graph primitives (kinship, reciprocity, credentials)
//! - [`FlowIndicator`] — Animated pulse showing cross-cluster dispatch / value flow
//! - [`Showcase`] — Reference-impl page exercising every primitive on one cohesive scene
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
// bridge_finance.rs is disabled: it's an unfinished manual proof-of-concept
// ("validate the Bridge Codegen pattern", per its own header) with zero
// consumers anywhere in the workspace, and it doesn't compile — it imports
// `use_holochain` from `mycelix_leptos_client` (that function actually lives
// in *this* crate, re-exported below) and re-exports `ActionHash`/`Record`/
// `SapBalanceResponse` from `personal_leptos_types`, none of which that crate
// actually defines (its own comment: "placeholders for types not in
// personal-leptos-types"). This has silently broken `cargo check --lib` for
// the entire crate — which every consuming Leptos app depends on — since it
// was added. Re-enable once it's actually finished against real types,
// rather than leaving broken scaffolding in the default build of a crate
// ~20 apps depend on.
// pub mod bridge_finance;
pub mod connection_status;
pub mod holochain_provider;
pub mod provider;

// --- UI components ---
pub mod activity_feed;
pub mod app_shell;
pub mod availability_state;
pub use mycelix_leptos_ui::badge;
pub mod cluster_launcher;
pub use mycelix_leptos_ui::data_table;
pub use mycelix_leptos_ui::empty_state;
pub mod error_boundary;
pub mod flow_indicator;
pub mod forms;
pub mod freshness;
pub mod graph_node;
pub mod indlela;
pub use mycelix_leptos_ui::loading;
pub mod local_identity;
pub mod modal;
pub use mycelix_leptos_ui::progress_bar;
pub mod search_bar;
pub mod showcase;
pub use mycelix_leptos_ui::sovereign_radar;
pub mod spore_bridge;
pub use mycelix_leptos_ui::stat_card;
pub mod summary_card;
pub use mycelix_leptos_ui::tabs;
pub mod telemetry_line;
pub mod tier_gate;
pub mod trust_badge;
pub mod zome_call_button;

// --- Reactive systems ---
pub mod consciousness;
pub mod consciousness_ui;
pub mod homeostasis;
pub mod theme;
pub mod thermodynamic;
pub use mycelix_leptos_ui::toasts;

// --- Utilities ---
pub mod util;

// Re-exports for convenience — transport
pub use holochain_provider::{
    ConnectStrategy, ConnectionBadge, ConnectionStatus, HolochainCtx, HolochainProviderAuto,
    HolochainProviderConfig,
};
pub use provider::{HolochainProvider, use_holochain, use_zome_call};
// Note: holochain_provider::use_holochain() returns concrete HolochainCtx,
// while provider::use_holochain::<T>() is generic. Access the concrete one
// via mycelix_leptos_core::holochain_provider::use_holochain().

// Re-exports — types
pub use personal_leptos_types::TrustTier;

// Re-exports — UI components
pub use connection_status::ConnectionStatusIndicator;
pub use error_boundary::AppErrorBoundary;
pub use loading::LoadingSkeleton;
pub use progress_bar::ProgressBar;
pub use stat_card::StatCard;
pub use telemetry_line::TelemetryLine;
pub use tier_gate::TierGate;
pub use trust_badge::TrustBadge;
pub use zome_call_button::ZomeCallButton;

// Re-exports — reactive systems
pub use consciousness::{
    ConsciousnessProfile, ConsciousnessState, provide_consciousness_context, use_consciousness,
};
pub use consciousness_ui::init_consciousness_ui;
pub use homeostasis::{HomeostasisState, provide_homeostasis_context, use_homeostasis};
pub use theme::{AppTheme, ThemeState, provide_theme_context, use_theme_state};
pub use thermodynamic::{ThermodynamicState, provide_thermodynamic_context, use_thermodynamic};
pub use toasts::{Toast, ToastContainer, ToastKind, ToastState, provide_toast_context, use_toasts};

// Re-exports — new components
pub use activity_feed::{ActivityFeed, ActivityFeedItem};
pub use app_shell::{AppNav, AppShell, MobileBottomNav, NavLink, NavTab};
pub use availability_state::{AvailabilityState, AvailabilityStateKind};
pub use badge::{Badge, BadgeVariant, StatusDot};
pub use cluster_launcher::{ClusterLauncher, ClusterLink, default_clusters};
pub use data_table::{Column, DataTable, Pagination};
pub use empty_state::EmptyState;
pub use flow_indicator::FlowIndicator;
pub use forms::{Checkbox, FormField, Select, SelectOption, TextArea, TextInput};
pub use freshness::{FreshnessBadge, FreshnessLevel};
pub use graph_node::{GraphEdge, GraphNode, NodeEmphasis};
pub use indlela::{GrowthStage, community_warmth, knowledge_freshness};
pub use local_identity::{
    LocalIdentity, load_json, local_did, provide_local_identity, save_json, use_local_identity,
};
pub use modal::{ConfirmDialog, Modal, ModalSize};
pub use search_bar::SearchBar;
pub use showcase::Showcase;
pub use sovereign_radar::{SovereignRadar, SovereignRadarSize};
pub use spore_bridge::{SporeState, provide_spore_bridge, use_spore};
pub use summary_card::{
    SummaryActionItem, SummaryAttentionItem, SummaryCard, SummaryMetricItem, SummaryStatusBadge,
};
pub use tabs::{TabPanel, Tabs};

// Re-exports — utilities
pub use util::{set_css_var, set_root_attribute};
