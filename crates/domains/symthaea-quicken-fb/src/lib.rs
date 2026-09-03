// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Symthaea Quicken Framebuffer — mycelial colonization boot animation.
///
/// A bare-metal DRM/KMS framebuffer renderer for the NixOS installer boot sequence.
/// Renders procedural presentation from a stable, explicitly non-secret visual seed.
///
/// # Architecture
///
/// - `boot_protocol` — fail-open typed boot telemetry consumer
/// - `renderer_bridge` — validated semantic ecology projection with explicit legacy fallback
/// - `framebuffer` — DRM device management, dumb buffer allocation, mmap
/// - `handoff` — post-DRM-release diagnostic acknowledgement
/// - `mycelium` — L-system branch growth, Bresenham rendering, node pulsing
/// - `perf` — deterministic timing summaries and opt-in live performance receipts
/// - `progress` — Named pipe reader for installer progress events
/// - `visual_seed` — bounded presentation-only seed loading/validation
/// - `color` — Solarpunk RGBA palette with interpolation and compositing
pub mod boot_protocol;
pub mod color;
#[path = "ecology_renderer.rs"]
pub mod ecology_renderer_base;
pub mod ecology_renderer_fidelity_v2;
pub mod ecology_renderer_holo;
pub mod ecology_renderer_identity;
pub mod framebuffer;
pub mod handoff;
pub mod microtype;
pub mod mycelium;
pub mod perf;
pub mod progress;
pub mod renderer_bridge;
pub mod visual_seed;
