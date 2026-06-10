// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! System scanner library — detects installed apps, hardware, and system info.
//!
//! Used by:
//! - `sovereign-scan` CLI binary
//! - `symthaea-installer` Tauri app
//! - SSH relay for remote scanning

pub mod scanner;