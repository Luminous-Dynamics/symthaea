// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finance (FIN) Module for Mycelix ERP
//!
//! Provides general ledger, accounts payable/receivable, invoicing,
//! bank reconciliation, and financial reporting with cryptographic auditability.

pub mod models;
pub mod api;
pub mod api_reconciliation;
pub mod api_currency;
pub mod api_ai;
pub mod ledger;
pub mod invoicing;
pub mod payments;
pub mod reporting;
pub mod reconciliation;
pub mod currency;
pub mod ai_processing;

pub use models::*;
pub use api::*;
pub use api_reconciliation::*;
pub use api_currency::*;
pub use api_ai::*;
pub use ledger::*;
pub use invoicing::*;
pub use payments::*;
pub use reporting::*;
pub use reconciliation::*;
pub use currency::*;
pub use ai_processing::*;
