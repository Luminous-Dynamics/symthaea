# Luminous Dynamics: System Constitution

## Environment: NixOS 26.05 (Yarara)
- **Immutable Root**: NEVER attempt to install global packages via `pip`, `cargo install`, or `npm`.
- **Nix Flakes**: All builds MUST be wrapped: `nix develop --command cargo build`.
- **Acceleration**: Use `mold` and `sccache` (pre-configured in system) for all Rust operations.

## Secrets & Credentials
- **Vault**: Use `~/.cargo/bin/bws secret get <id>` for tokens. 
- **Crates.io**: Token is `736da236-a95f-4dd2-8efc-b42800c9106a`.

## Coding Standard: Symthaea Core
- **Math**: Strictly follow Lie Theory and su(2) representations in `symthaea-core`.
- **Traits**: Every new platform MUST implement `symthaea-core/src/embodiment.rs:EmbodimentBridge`.
- **Memory**: After every major module change, update the local `MEMORY.md` to maintain holographic state.

## Autonomous Protocol
- **Commits**: Commit after every phase (Rule 8). Stage only authored files.
- **Telemetry**: Ensure all new modules export to the `event_bus` for visualization.
