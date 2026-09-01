// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Symthaea Quicken Framebuffer — state-aware procedural Spore boot animation.
///
/// A bare-metal DRM/KMS renderer with an exact offline preview path.
///
/// # Architecture
///
/// - `ecology_renderer` — deterministic state-aware procedural curve renderer
/// - `framebuffer` — DRM device management, dumb buffer allocation, mmap
/// - `mycelium` — legacy L-system renderer retained for compatibility
/// - `progress` — optional named-pipe / disk-I/O progress events
/// - `color` — Solarpunk RGBA palette with interpolation and compositing
pub mod color;
pub mod ecology_renderer;
pub mod framebuffer;
pub mod mycelium;
pub mod progress;
