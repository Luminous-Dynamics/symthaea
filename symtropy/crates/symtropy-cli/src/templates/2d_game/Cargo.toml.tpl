[package]
name = "{{project_name}}"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.18"
symtropy-bevy = "0.2"

# Unpublished as of writing — git URL until next release lands on crates.io.
symtropy-devconsole = { git = "https://github.com/luminous-dynamics/symtropy", branch = "main", features = ["phi-panel"] }
