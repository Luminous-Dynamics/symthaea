[package]
name = "{{project_name}}"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.19"

# The Permissive Symtropy distribution: re-exports symtropy-bevy-core
# + symtropy-bevy-scene + symtropy-devconsole.
# NO AGPL dependencies.
#
# Currently unpublished — git URL until next release lands on crates.io.
symtropy-core = { git = "https://github.com/luminous-dynamics/symtropy", branch = "main" }
