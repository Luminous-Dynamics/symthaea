// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Durable shared authorization-consumption domain.
//!
//! SQLite is the authoritative reservation/consumption state machine. The
//! external effect remains a separate boundary: an uncertain effect becomes
//! Indeterminate and requires explicit reconciliation.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::{Duration, DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use url::Url;

use super::{
    ActionAuthorizationWitness, AuthorizationConsumptionError, AuthorizationLease,