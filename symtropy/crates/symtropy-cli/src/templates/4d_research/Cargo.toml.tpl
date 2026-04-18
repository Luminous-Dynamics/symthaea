[package]
name = "{{project_name}}"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy = "0.18"

# Published on crates.io.
symtropy-bevy = "0.2"

# Unpublished as of writing — git URLs until the next release lands on
# crates.io. Switch to versioned dependencies once available.
symtropy-bevy-scene = { git = "https://github.com/luminous-dynamics/symtropy", branch = "main" }
symtropy-devconsole = { git = "https://github.com/luminous-dynamics/symtropy", branch = "main", features = ["phi-panel"] }
