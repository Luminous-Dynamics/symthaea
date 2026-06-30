// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Symthaea Quicken Framebuffer — mycelial colonization boot animation.
///
/// A bare-metal DRM/KMS framebuffer renderer for the NixOS installer boot sequence.
/// Renders procedural L-system mycelial growth seeded by the installation's genesis phrase.
///
/// # Architecture
///
/// - `framebuffer` — DRM device management, dumb buffer allocation, mmap
/// - `mycelium` — L-system branch growth, Bresenham rendering, node pulsing
/// - `progress` — Named pipe reader for installer progress events
/// - `color` — Solarpunk RGBA palette with interpolation and compositing
pub mod color;
pub mod framebuffer;
pub mod mycelium;
pub mod progress;
