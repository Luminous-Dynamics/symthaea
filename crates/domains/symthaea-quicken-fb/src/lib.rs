// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Symthaea Quicken Framebuffer — state-aware procedural Spore boot animation.
///
/// A bare-metal DRM/KMS renderer with an exact offline preview path.
///
/// # Architecture
///
/// - `ecology_renderer_base` — deterministic state-aware procedural curve renderer
/// - `ecology_renderer_holo` — bounded holographic spatial field wrapper
/// - `ecology_renderer` — final visual-fidelity wrapper used by live + preview paths
/// - `inoculation_renderer` — shared installation/incubation phase grammar
/// - `inoculation_ceremony` — install-path-specific visual signatures
/// - `preview` — exact offline frame capture using the same renderer
/// - `framebuffer` — DRM device management, dumb buffer allocation, mmap
/// - `mycelium` — legacy L-system renderer retained for compatibility
/// - `progress` — optional named-pipe / disk-I/O progress events
/// - `color` — Solarpunk RGBA palette with interpolation and compositing
pub mod color;
#[path = "ecology_renderer.rs"]
pub mod ecology_renderer_base;
#[path = "ecology_renderer_holo.rs"]
pub mod ecology_renderer_holo;
#[path = "ecology_renderer_fidelity.rs"]
pub mod ecology_renderer;
pub mod framebuffer;
pub mod inoculation_ceremony;
pub mod inoculation_renderer;
pub mod mycelium;
pub mod preview;
pub mod progress;
