# Crate overview

Symtropy is a Cargo workspace with 14 crates. Each has a single clear responsibility.

## Core (permissive, Apache-2.0 OR MIT)

| Crate | Description | crates.io |
|---|---|---|
| `symtropy-math` | ND geometric algebra, shapes, transforms | [link](https://crates.io/crates/symtropy-math) |
| `symtropy-physics` | `PhysicsWorld<D>`, GJK+EPA, CCD, joints, replay | [link](https://crates.io/crates/symtropy-physics) |
| `symtropy-render-bridge` | ND→Bevy projection, 4D cross-section slicing | (Phase 0) |
| `symtropy-robotics-bridge` | FEP agents, 6 embodied platforms | (Phase 0) |
| `symtropy-net` | P2P spatial authority, lockstep protocol | (Phase 0) |
| `symtropy-bevy` | Drop-in Bevy plugin | [link](https://crates.io/crates/symtropy-bevy) |

## Research (copyleft, AGPL-3.0-or-later)

| Crate | Description | crates.io |
|---|---|---|
| `symtropy-consciousness-physics` | Φ coupling, 63 experiments, thermodynamic ledger | [link](https://crates.io/crates/symtropy-consciousness-physics) |
| `symtropy-sim-bridge` | Mycelix governance/economy/FL integration | private |
| `symtropy-world` | Macro/micro sim bridge | private |
| `symtropy-holochain-relay` | Holochain DHT persistence | private |
| `symtropy-lightyear` | Game-tier netcode wrapper | private |
| `symthaea-bevy-brain` | Full Symthaea cognitive loop as Bevy plugin | private |

## Game / demo (AGPL)

| Crate | Description |
|---|---|
| `symtropy` (root) | *The Room That Remembers You* + Sol Atlas |
| `symtropy-gravcraft-demo` | Gravity craft game demo |
| `symtropy-manipulator-demo` | Manipulator arm demo |

## Dependency graph

```
symtropy-math                                   (no deps)
  └→ symtropy-physics                           (permissive)
        ├→ symtropy-render-bridge               (permissive)
        ├→ symtropy-robotics-bridge             (permissive)
        ├→ symtropy-net                         (permissive)
        ├→ symtropy-bevy                        (permissive)
        └→ symtropy-consciousness-physics       (AGPL)
              ├→ symtropy-sim-bridge            (AGPL)
              └→ symtropy-world                 (AGPL)
```

## Which do I need?

| Task | Crates |
|---|---|
| ND collision detection only | `symtropy-math` + `symtropy-physics` |
| Generic state-coupled physics, proprietary OK | Core crates only |
| Φ-coupled research | + `symtropy-consciousness-physics` (AGPL) |
| Bevy game | + `symtropy-bevy` |
| Mycelix governance testing | + `symtropy-sim-bridge` (AGPL) |
| Symthaea robotics | + `symtropy-robotics-bridge` + relevant platform crate |
