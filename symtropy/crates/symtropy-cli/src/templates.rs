// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Embedded project templates. Each template is a small Rust starter that
//! pulls in the Symtropy distribution stack with sensible defaults.
//!
//! Templates evolve with the CLI; users get fresh starters per `symtropy-cli`
//! version.

pub struct Template {
    pub name: &'static str,
    pub description: &'static str,
    pub cargo_toml: &'static str,
    pub main_rs: &'static str,
    pub readme_md: &'static str,
}

pub const TEMPLATES: &[Template] = &[
    Template {
        name: "3d-research",
        description: "3D scene with one swinging pendulum + dev console",
        cargo_toml: include_str!("templates/3d_research/Cargo.toml.tpl"),
        main_rs: include_str!("templates/3d_research/main.rs.tpl"),
        readme_md: include_str!("templates/3d_research/README.md.tpl"),
    },
    Template {
        name: "4d-research",
        description: "4D scene with hyperplane slicing + dev console",
        cargo_toml: include_str!("templates/4d_research/Cargo.toml.tpl"),
        main_rs: include_str!("templates/4d_research/main.rs.tpl"),
        readme_md: include_str!("templates/4d_research/README.md.tpl"),
    },
    Template {
        name: "2d-game",
        description: "2D scene with a single physics-bodied sprite",
        cargo_toml: include_str!("templates/2d_game/Cargo.toml.tpl"),
        main_rs: include_str!("templates/2d_game/main.rs.tpl"),
        readme_md: include_str!("templates/2d_game/README.md.tpl"),
    },
];
