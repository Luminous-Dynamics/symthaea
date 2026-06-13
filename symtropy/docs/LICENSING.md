# Symtropy Licensing

Symtropy uses a **dual-track license model**. The split is designed to:

1. **Make the core engine freely adoptable** by game studios, indies, research groups, and industrial users — including in proprietary and commercial contexts.
2. **Protect the research IP** (consciousness-physics, Mycelix governance/economy/crypto integration) with strong copyleft so improvements flow back to the commons.

## At a glance

| Crate | License | Rationale |
|---|---|---|
| `symtropy-math` | **Apache-2.0 OR MIT** | Core: ND geometric algebra. Zero AGPL deps. |
| `symtropy-physics` | **Apache-2.0 OR MIT** | Core: GJK+EPA, CCD, joints, raycasting, deterministic replay. Zero AGPL deps. |
| `symtropy-render-bridge` | **Apache-2.0 OR MIT** | Core: ND→Bevy projection, 4D cross-section slicing. Zero AGPL deps. |
| `symtropy-bevy-core` | **Apache-2.0 OR MIT** | Core: Bevy plugin with generic `PhysicsCallback` coupling. Zero AGPL deps. Use directly for proprietary games; `symtropy-bevy` (AGPL) adds ConsciousnessField. |
| `symtropy-robotics-bridge` | **AGPL-3.0-or-later** (*see note 1*) | Currently depends on AGPL `consciousness-physics` + Symthaea FEP. Phase 1 will split into a permissive `-core` + AGPL integration crate. |
| `symtropy-net` | **AGPL-3.0-or-later** (*see note 1*) | Currently depends on AGPL `symtropy-holochain-relay`. Phase 1 will extract permissive `symtropy-net-core`. |
| `symtropy-bevy` | **AGPL-3.0-or-later** (*see note 1*) | Currently depends on AGPL `consciousness-physics` throughout `plugin.rs`, `biometrics.rs`, `macro_bridge.rs`. Phase 0.5 extracts `symtropy-bevy-core` (permissive) via feature-gating. |
| `symtropy-consciousness-physics` | **AGPL-3.0-or-later** | Research hero: Φ coupling, 63 experiments, Master Equation, ThermodynamicLedger. |
| `symtropy-sim-bridge` | **AGPL-3.0-or-later** | Mycelix governance / economy / federated learning integration. |
| `symtropy-world` | **AGPL-3.0-or-later** | Depends on consciousness-physics (transitively AGPL). |
| `symtropy-holochain-relay` | **AGPL-3.0-or-later** | Holochain DHT persistence for consciousness profiles. |
| `symtropy-lightyear` | **AGPL-3.0-or-later** | Game-tier netcode wrapper (can become permissive later). |
| `symthaea-bevy-brain` | **AGPL-3.0-or-later** | Full Symthaea cognitive loop integration. |
| `symtropy-gravcraft-demo` | **AGPL-3.0-or-later** | Demo game. |
| `symtropy-manipulator-demo` | **AGPL-3.0-or-later** | Demo game. |
| Root `symtropy` (game crate) | **AGPL-3.0-or-later** | *The Room That Remembers You* + Sol Atlas. |

> **Note 1** — These three crates were initially targeted for permissive licensing to support generalist adoption, but each has required dependencies on AGPL crates. AGPL is viral: any crate that *requires* AGPL code cannot itself be permissively licensed without misrepresenting the combined work's terms. The Phase 0.5 / Phase 1 roadmap extracts permissive `-core` variants by feature-gating the AGPL integrations:
>
> - **`symtropy-bevy-core`** — physics plugin, gizmos, input. No consciousness-physics. Apache/MIT.
> - **`symtropy-bevy`** — re-exports `-core` + adds consciousness/biometrics systems. AGPL.
> - **`symtropy-net-core`** — spatial authority + lockstep protocol. No Holochain. Apache/MIT.
> - **`symtropy-net`** — adds Holochain DHT persistence. AGPL.
> - **`symtropy-robotics-bridge-core`** — platform traits + `PhysicsCallback` hook. No Symthaea. Apache/MIT.
> - **`symtropy-robotics-bridge`** — current functionality. AGPL.
>
> The `symtropy-math`, `symtropy-physics`, `symtropy-render-bridge` crates already have zero AGPL deps, so they're truly permissive today.

## What this means in practice

### You are free to

Under the permissive (Apache-2.0 OR MIT) core crates:

- **Build and ship commercial games** on Symtropy's physics, rendering, and robotics core without open-sourcing your game.
- **Run proprietary SaaS / network services** using the core crates without AGPL obligations.
- **Integrate with proprietary middleware** (other physics backends, proprietary renderers, closed-source tools).
- **Fork, modify, and redistribute** the core crates under either Apache-2.0 or MIT at your option.

### You must open-source (AGPL-3.0) if you use

- `symtropy-consciousness-physics` — if you ship or run as a service, your modifications must be released under AGPL.
- `symtropy-sim-bridge` — Mycelix governance integration is copyleft.
- Any of the game-layer crates.

### You need a commercial license if you

- Want to integrate any of the **AGPL crates** into a **proprietary** product or service without releasing your modifications.
- See the parent `COMMERCIAL_LICENSE.md` in `/srv/luminous-dynamics/COMMERCIAL_LICENSE.md` for terms.
- Cooperatives, B-corps, and mission-aligned organisations may qualify for favourable terms.

### Contact for commercial licensing

- Email: tristan.stoltz@evolvingresonantcocreationism.com
- Web: https://luminousdynamics.org

## Contributor License Agreement

By submitting a contribution to any Symtropy crate, you agree that your contribution is licensed under the same terms as the crate you are contributing to:

- Contributions to **permissively-licensed core crates** are dual-licensed Apache-2.0 OR MIT.
- Contributions to **AGPL-licensed crates** are licensed AGPL-3.0-or-later.

This matches the Rust ecosystem convention (same as the Rust compiler and Bevy itself). See `CONTRIBUTING.md` for details.

## Why dual-track

AGPL alone is an adoption blocker for studios and many indie developers — they won't touch a codebase whose license requires releasing proprietary server code. Apache-2.0 OR MIT alone would give away the research IP that makes Symtropy unique.

The split resolves this:

- The **physics + rendering + networking core** — generic infrastructure that happens to be very good — is given away freely. This is the widest possible funnel for adoption and contribution.
- The **consciousness-physics + Mycelix** layer — the research contribution that is genuinely novel and took years to develop — stays copyleft. Studios and proprietary users who want it negotiate a commercial license; the commons stays protected.

This is the same structural model used by many successful OSS projects (e.g. MongoDB's SSPL / Server Side Public License before, Redis Labs' modules, Elastic before 7.11), adapted to a consciousness-research context.

## crates.io history note

As of the initial license split (2026-04-17), the following crates had already been published to crates.io at version `0.1.0` under **AGPL-3.0-or-later**:

- `symtropy-math` 0.1.0 (AGPL)
- `symtropy-physics` 0.1.0 (AGPL)
- `symtropy-bevy` 0.1.0 (AGPL)
- `symtropy-consciousness-physics` 0.1.0 (AGPL — stays AGPL, no change)

Published versions on crates.io are immutable — `0.1.0` remains under AGPL forever for anyone who pulls that exact version.

**Published 2026-04-17 under Apache-2.0 OR MIT** (zero AGPL deps, safe for proprietary use):

- `symtropy-math` **0.2.0** ✅
- `symtropy-physics` **0.2.0** ✅
- `symtropy-render-bridge` **0.1.0** ✅ (first publish)
- `symtropy-bevy-core` **0.1.0** ✅ (first publish — Phase 0.5 permissive Bevy plugin)

`symtropy-bevy` 0.1.0 stays AGPL on crates.io. Its refactor to re-export `symtropy-bevy-core` and layer `ConsciousnessField` on top is queued in Phase 0.5.
- `symtropy-consciousness-physics` unchanged under **AGPL-3.0-or-later**

Users who want permissive licensing should depend on `symtropy-math >= 0.2.0`, `symtropy-physics >= 0.2.0`, `symtropy-render-bridge >= 0.1.0`, `symtropy-bevy-core >= 0.1.0`. The version bump on math/physics signals the license-model change so automated tooling (cargo-audit, license scanners) can flag it.

## References

- Apache License 2.0: `LICENSE-APACHE`
- MIT License: `LICENSE-MIT`
- GNU AGPL v3: `/srv/luminous-dynamics/LICENSE`
- Commercial licensing terms: `/srv/luminous-dynamics/COMMERCIAL_LICENSE.md`
- CLA: `/srv/luminous-dynamics/CLA.md`
