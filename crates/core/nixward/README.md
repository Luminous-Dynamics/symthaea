# nixward: A NixOS systems intelligence and execution runtime

Nixward is the NixOS-specific observation, semantic evaluation, planning, execution, rollback, and verification runtime for the Luminous Dynamics infrastructure stack.

Its architectural boundary is deliberately separate from cognition and from the platform-neutral Sovereign State Compiler (SSC):

```
Symthaea cognition
      │
      ▼
DeploymentIntent
      │
      ▼
Sovereign State Compiler (platform-neutral)
      │
      ▼
NixOS deployment plan
      │
      ▼
Nixward
 ┌────┼───────────────┐
 │    │               │
observe execute   verify/rollback
 │    │               │
 └────┴───────┬───────┘
               ▼
        post-state receipt
```

Nixward owns NixOS-specific realization. SSC does not execute commands, and Nixward does not become the authority for unrelated operating systems.

## Architecture

Nixward is organized around six responsibilities:

1. **Parser** — Nix source and flake structure.
2. **Observation** — live NixOS state, systemd, journal, store, hardware, and generation identity.
3. **Semantic evaluation** — NixOS options, packages, dependencies, drift, and causal relationships.
4. **Planning and execution** — Nix-specific lowering, native process execution, rollback, and compensation.
5. **Verification** — fresh post-state observation and typed postconditions; process exit status is not equivalent to verified success.
6. **Cognitive integration** — optional Symthaea-facing adapters for prediction, explanation, and recommendation. Cognitive state is advisory and is not an authorization primitive.

The privileged boundary is therefore:

```
observation
  → plan
  → explicit authorization
  → mechanical execution
  → fresh observation
  → postcondition verification
  → evidence-bearing receipt
```

## Current capabilities

The current in-tree implementation includes:

- Nix parsing and semantic analysis
- NixOS state observation
- HDC-based representations and predictive models
- causal and drift analysis
- CLI and TUI interfaces
- background daemon support
- NixOS module integration
- generation inspection and rollback
- native Nix/NixOS command execution
- post-execution verification work
- optional Symthaea cognitive integration

## Standalone-repository direction

Nixward is currently hosted inside Symthaea while the SSC/Nixward boundary is being proven.

The intended eventual repository boundary is:

- **Symthaea** — cognition, reasoning, simulation, and recommendations
- **Sovereign State Compiler** — platform-neutral deployment protocol, planning, authorization, and evidence contracts
- **Nixward** — NixOS observation, realization, execution, rollback, and verification
- **Spore** — portable boot/recovery embodiment
- **Sovereign Ops** — cross-system operator/control-plane composition

The extraction gate is not "copy the directory into another repository." It is independent buildability and qualification, with no duplicate privileged NixOS executor left behind.

## CLI

```bash
nixward search "web server"
nixward observe services
nixward doctor
nixward rebuild switch --flake ".#myhost"
nixward generations list
nixward rollback
nixward service status nginx
nixward service restart postgresql
nixward flake check
nixward flake show
```

## NixOS module

Nixward can be integrated as a NixOS service with a dedicated system user and hardened service configuration. The exact module interface is versioned with the runtime and should not be treated as a substitute for SSC authorization.

## Development

```bash
nix develop ./crates/nixward
cargo build -p nixward --features cli
cargo test -p nixward --features tui --lib
cargo test -p nixward --features cli --test cli_integration
cargo test -p nixward --test e2e_consciousness_loop
cargo test -p nixward --test proptest_hdc
cargo clippy -p nixward --features tui --all-targets
```

## Authority and safety model

Nixward must preserve the distinction between:

- **recommendation** — cognition or semantic analysis suggests a change;
- **authorization** — an explicit policy grants the exact capability and state transition;
- **execution** — Nixward performs only an authorized plan;
- **verification** — fresh observations establish whether required postconditions hold;
- **receipt** — evidence binds intent, authorization, pre-state, execution, post-state, and verification.

A command returning exit code zero is not by itself proof that the requested system state was achieved.

## License

AGPL-3.0-or-later

Commercial licensing: see `COMMERCIAL_LICENSE.md` at the repository root.
