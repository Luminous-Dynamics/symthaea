# symtropy-mycelix-bridge

**Status:** Milestone 1 spike (2026-04-17). See `symtropy/plans/mycelix-bridge-plan.md` for the full plan.

Wraps Mycelix Holochain zome calls as a Bevy `Resource` so Bevy systems (and NPCs) can call real Mycelix governance, finance, and other zomes via the shared Holochain conductor.

## Architecture: subprocess IPC, not in-process linking

Holochain's Rust client pins `serde = 1.0.203` transitively via `holochain_client 0.6.0`. Bevy 0.18 requires `serde_core >= 1.0.221` (via `hashbrown 0.16`). They cannot coexist in a single Rust compilation unit.

This crate solves that by running the Holochain client as a **separate process** (`mycelix-conductor-bridge` from the monorepo) and exchanging JSON over the subprocess's stdin/stdout. The architectural boundary eliminates the dep conflict permanently:

- `symtropy-mycelix-bridge` has zero Holochain dependencies.
- Bevy's compilation unit is clean.
- If Holochain bumps serde or Bevy bumps again, nothing here breaks.

## Prerequisites

1. Build `mycelix-conductor-bridge` (from the monorepo root):
   ```bash
   cd /srv/luminous-dynamics/mycelix-conductor-bridge
   cargo build --release
   ```
2. Export an app-auth token before starting your Bevy app:
   ```bash
   export MYCELIX_APP_TOKEN="<base64-token>"
   ```
3. Point `MycelixConfig::bridge_binary` at the binary, or put it on `PATH`.

## Licensing

**AGPL-3.0-or-later.** Pulls the Mycelix zome logic via `symthaea-mycelix-holochain`. If you want AGPL-free Bevy physics, use `symtropy-bevy-core` (Apache/MIT) instead — this crate is only for code that wants Mycelix integration.

## Quick API

```rust
use bevy::prelude::*;
use symtropy_mycelix_bridge::{BevyMycelixPlugin, MycelixConfig, MycelixClient,
                              MycelixRequest, MycelixResponse};

App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(BevyMycelixPlugin::new(MycelixConfig::default()))
    .add_systems(Update, (send_request, handle_response))
    .run();

fn send_request(client: Res<MycelixClient>) {
    client.send(MycelixRequest::GetActiveProposals {
        requester: Entity::PLACEHOLDER,
    });
}

fn handle_response(mut ev: EventReader<MycelixResponse>) {
    for response in ev.read() {
        tracing::info!(?response, "got response");
    }
}
```

## Milestone 1 scope

One round-trip zome call from a Bevy system through a tokio background task into `HolochainConductor` and back. No UI. No scenario harness. No retries. Just: does the plumbing work?

Next milestones (M2–M4) expand the zome surface, add a scenario harness for CI, and ship a 50-NPC visual demo. See the plan.
