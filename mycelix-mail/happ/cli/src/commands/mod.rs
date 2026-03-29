// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
pub mod did;
pub mod export;
pub mod inbox;
/// Command implementations for Mycelix Mail CLI
///
/// This module contains all command handlers that implement the CLI functionality.
/// Each command is in its own submodule for maintainability.
pub mod init;
pub mod read;
pub mod search;
pub mod send;
pub mod status;
pub mod sync;
pub mod trust;
