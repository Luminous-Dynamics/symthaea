[package]
name = "{{project_name}}"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.19"

# The Symtropy distribution: re-exports symtropy-bevy + symtropy-bevy-scene
# + symtropy-devconsole. One dep instead of three.
#
# Currently unpublished — git URL until next release lands on crates.io.
# Once published: `symtropy = { version = "0.1", features = ["devconsole-phi"] }`.
symtropy = { git = "https://github.com/luminous-dynamics/symtropy", branch = "main", features = ["devconsole-phi"] }
