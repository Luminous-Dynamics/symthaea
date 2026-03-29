// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix ERP Service Library
//!
//! Blockchain-auditable Enterprise Resource Planning with:
//! - Supply Chain Management (SCM) - Verifiable provenance tracking
//! - Finance (FIN) - Double-entry bookkeeping with cryptographic signatures
//! - CRM - Customer Relationship Management
//! - HR - Human Resources Management
//! - Auth - JWT-based authentication with multi-tenant support
//! - Integrations - Plaid (banking) & Stripe (payments)
//! - More modules coming: MRP, PM, ASSET

pub mod api;
pub mod auth;  // Authentication & authorization
pub mod batch;
pub mod config;
pub mod crm;  // CRM module (contacts, leads, opportunities, activities)
pub mod db;
pub mod dkg_client;
pub mod error;  // Shared error types
pub mod fin;  // Finance module (GL, invoices, bills, payments)
pub mod health;
pub mod hr;  // HR module (employees, departments, leave, payroll)
pub mod inventory;  // Inventory module (products, warehouses, stock)
pub mod integrations;  // External integrations (Plaid, Stripe)
pub mod lineage;
pub mod lineage_api;
pub mod logging;
pub mod metrics;
pub mod middleware;
pub mod observability;
pub mod pipeline;
pub mod realtime;  // Real-time WebSocket collaboration
pub mod security;
pub mod validation;
pub mod vc;

/// Application state shared across handlers
pub struct AppState {
    pub keypair: crypto::KeyPair,
    /// Database for persistent storage
    pub db: Option<db::Database>,
    /// Fallback in-memory storage for development/testing
    pub claims: tokio::sync::RwLock<std::collections::HashMap<String, claim_model::DkgClaim>>,
    /// PostgreSQL pool for SQL-based modules
    pub pool: Option<sqlx::PgPool>,
}

/// Simple pool state for inventory and other SQL modules
#[derive(Clone)]
pub struct PoolState {
    pub pool: sqlx::PgPool,
}
