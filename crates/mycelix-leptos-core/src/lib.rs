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
//! - [`HolochainProvider`] — Context provider wrapping a [`HolochainTransport`]
//! - [`ConnectionStatusIndicator`] — Navbar-sized connection state indicator
//! - [`TrustBadge`] — Consciousness-gated trust tier badge
//! - [`LoadingSkeleton`] — Pulsing skeleton placeholder for loading states
//! - [`AppErrorBoundary`] — Styled error boundary with retry
//! - [`ZomeCallButton`] — Button with loading/success/error states
//! - [`ProgressBar`] — Configurable progress bar
//! - [`StatCard`] — Labeled statistic card
//!
//! # Hooks
//!
//! - [`use_holochain`] — Retrieve the [`HolochainTransport`] from context
//! - [`use_zome_call`] — Reactive helper for executing zome calls

pub mod provider;
pub mod connection_status;
pub mod trust_badge;
pub mod loading;
pub mod error_boundary;
pub mod zome_call_button;
pub mod progress_bar;
pub mod stat_card;

// Re-exports for convenience
pub use provider::{HolochainProvider, use_holochain, use_zome_call};
pub use connection_status::ConnectionStatusIndicator;
pub use trust_badge::TrustBadge;
pub use loading::LoadingSkeleton;
pub use error_boundary::AppErrorBoundary;
pub use zome_call_button::ZomeCallButton;
pub use progress_bar::ProgressBar;
pub use stat_card::StatCard;
