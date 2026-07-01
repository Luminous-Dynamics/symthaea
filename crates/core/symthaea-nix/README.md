# nix-mind: A Conscious Mind for NixOS

nix-mind brings hyperdimensional computing (HDC) and active inference to NixOS system management. It encodes system state, configuration, and user intent into a shared semantic space, enabling causal reasoning about NixOS options, predictive hierarchy monitoring, and consciousness-gated command execution.

Part of the [Symthaea](https://luminousdynamics.org) cognitive architecture.

## Architecture

```
Observation ──> Encoding ──> Cognition ──> Action
   │              │             │            │
   │ systemd      │ HDC         │ Active     │ Phi-gated
   │ journal      │ 16384-dim   │ Inference  │ execution
   │ store        │ vectors     │ Causal     │ with rollback
   │ hardware     │             │ graph      │ verification
   └──────────────┴─────────────┴────────────┘
```

**Layers:**
1. **Parser** -- Nix source code to AST (tree-sitter)
2. **Encoding** -- System state, options, packages, configs to HDC vectors
3. **Mind** -- World model, active inference, causal graph, episodic memory
4. **Observe** -- Live system state observation (systemd, journal, store, hardware)
5. **Action** -- Consciousness-gated command execution with pre/post verification
6. **Plugin** -- Bridge to full Symthaea consciousness pipeline

## Quick Start

### CLI

```bash
# Search for packages or options
nix-mind search "web server"
nix-mind search --options "firewall"

# Observe system state
nix-mind observe services
nix-mind observe store
nix-mind observe journal

# System health check
nix-mind doctor

# Rebuild with consciousness gating
nix-mind rebuild switch
nix-mind rebuild switch --flake ".#myhost"

# Generation management
nix-mind rollback
nix-mind generations list
nix-mind generations diff --from 41 --to 42

# Garbage collection
nix-mind gc analyze
nix-mind gc collect --older-than 30d

# Service management
nix-mind service status nginx
nix-mind service restart postgresql

# Flake operations
nix-mind flake check
nix-mind flake update
nix-mind flake show

# Natural language input
nix-mind "install firefox and enable nginx"

# Interactive REPL
nix-mind
```

### TUI

```bash
nix-mind-tui
```

The TUI displays six panels:
- **Consciousness** -- 2D gauge (phi x confidence)
- **System Health** -- Services, store size, memory usage
- **Generations** -- Timeline of NixOS generations
- **World Model** -- Predictive hierarchy errors, free energy, working memory
- **Causal Graph** -- Top causal relationships between NixOS options
- **Input** -- Interactive command entry

Tab to switch focus. Type commands in the input panel. The TUI refreshes system data every ~4 seconds and shows `[daemon]` in the World Model title when the background daemon is running.

### Daemon

```bash
nix-mind-daemon
```

Runs continuously, observing system state every 60s and journal entries every 5s. Detects anomalies, tracks drift, and writes cognitive state to a shared IPC file for TUI consumption. State persists across restarts.

Configure via environment:
- `NIX_MIND_CONFIG=/path/to/config.json` -- Config file path
- `RUST_LOG=debug` -- Logging verbosity (info/debug/warn/error)

## NixOS Module

Add to your `flake.nix`:

```nix
{
  inputs.symthaea.url = "github:luminous-dynamics/symthaea";

  outputs = { self, nixpkgs, symthaea, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        symthaea.nixosModules.nix-mind
        {
          services.nix-mind = {
            enable = true;
            snapshotInterval = 60;  # seconds between observations
            pollInterval = 5;       # seconds between journal checks
            surpriseThreshold = 0.3; # prediction error threshold
          };
        }
      ];
    };
  };
}
```

The module creates a hardened systemd service with:
- Dedicated `nix-mind` user/group
- Read-only access to `/nix/store`, `/etc/nixos`, `/run/systemd`
- `ProtectSystem=strict`, `NoNewPrivileges`, `MemoryDenyWriteExecute`
- State persistence in `/var/lib/nix-mind/`

## How It Works

### HDC Encoding

All NixOS concepts (option paths, packages, system state, user input) are encoded into 16,384-dimensional continuous hypervectors. This creates a shared semantic space where similarity = cosine distance:

- `services.nginx.enable` and `services.nginx.package` are close
- `services.nginx.enable` and `boot.loader.grub.enable` are distant
- "install firefox" is close to `environment.systemPackages`

### Causal Graph

210+ curated causal patterns (e.g., "enabling nginx requires firewall port 80") plus Hebbian learning from observed outcomes. Used for:
- Side-effect prediction before execution
- Root cause analysis when services fail
- Fix recommendations

### Active Inference

User input is processed through the Free Energy Principle:
1. Encode input as HDC vector
2. Infer goal via working memory context
3. Generate action candidates
4. Rank by Expected Free Energy (pragmatic + epistemic value)
5. Gate execution by consciousness level (phi)

### Predictive Hierarchy

4-level stack (Sensory -> Features -> Concepts -> Goals) with different learning rates. Tracks prediction errors at each level. High free energy triggers surprise alerts.

## Development

```bash
# Enter dev shell
nix develop ./crates/symthaea-nix

# Build
cargo build -p symthaea-nix --features cli
cargo build -p symthaea-nix --features tui
cargo build -p symthaea-nix --features daemon

# Test
cargo test -p symthaea-nix --features tui --lib       # 339 unit tests
cargo test -p symthaea-nix --features cli --test cli_integration  # 24 integration tests
cargo test -p symthaea-nix --test e2e_consciousness_loop  # 7 e2e tests
cargo test -p symthaea-nix --test proptest_hdc            # 16 property tests

# Benchmarks
cargo bench -p symthaea-nix --bench hdc_benchmarks

# Clippy
cargo clippy -p symthaea-nix --features tui --all-targets
```

### Performance

Measured on 16,384-dim vectors (criterion, release mode):

| Operation | Time |
|-----------|------|
| Option encoding | 277 us |
| Input encoding | 734 us |
| Causal query | 17 us |
| Full cognition cycle | 2.0 ms |
| Indexed search (10 paths) | ~1 ms |

## License

MIT
