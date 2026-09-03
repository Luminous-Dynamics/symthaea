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
/// - `ecology_renderer_fidelity_v2` — membrane, caustics, and CPU bloom
/// - `ecology_renderer` — final factual identity wrapper used by live + preview paths
/// - `visual_composition` — pure bounded perceptual-attention policy
/// - `temporal_choreography` — semantic-progress-driven motion policy
/// - `visual_semantics` — descriptive anti-collapse regression signatures
/// - `visual_sampling` — deterministic node-density / negative-space selector
/// - `hardware_bud` — deterministic localized structure for persistent hardware change
/// - `inoculation_renderer` — shared installation/incubation phase grammar
/// - `inoculation_ceremony` — install-path-specific visual signatures
/// - `microtype` — tiny dependency-free uppercase factual label renderer
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
pub mod ecology_renderer_fidelity_v2;
#[path = "ecology_renderer_identity.rs"]
pub mod ecology_renderer;
pub mod framebuffer;
pub mod hardware_bud;
pub mod inoculation_ceremony;
pub mod inoculation_renderer;
pub mod microtype;
pub mod mycelium;
pub mod preview;
pub mod progress;
pub mod temporal_choreography;
pub mod visual_composition;
pub mod visual_sampling;
pub mod visual_semantics;
