// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing transport trade-study schemas.
//!
//! This crate sits above `symthaea-infrastructure`. It defines assumptions,
//! demand scenarios, lifecycle/reliability records, and Pareto-study inputs,
//! while deliberately containing no routing optimizer, transport physics, or
//! actuator authority.

#![deny(unsafe_code)]

pub mod claims;
pub mod trade;
