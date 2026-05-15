# Crate overview

Symtropy is a Cargo workspace with 29 crate manifests. The crates fall into three tracks: permissive core infrastructure, AGPL research/integration layers, and AGPL demos that prove the stack in embodied scenarios.

## Core (permissive, Apache-2.0 OR MIT)

| Crate | Description | crates.io |
|---|---|---|
| `symtropy-math` | ND geometric algebra, shapes, transforms | [link](https://crates.io/crates/symtropy-math) |
| `symtropy-physics` | `PhysicsWorld<D>`, GJK+EPA, CCD, joints, replay | [link](https://crates.io/crates/symtropy-physics) |
| `symtropy-render-bridge` | ND->Bevy projection, 4D cross-section slicing | [link](https://crates.io/crates/symtropy-render-bridge) |
| `symtropy-bevy-core` | Generic Bevy physics plugin over `PhysicsCallback` | [link](https://crates.io/crates/symtropy-bevy-core) |
| `symtropy-net-core` | Spatial authority and lockstep protocol (permissive part) | private / planned |
| `symtropy-robotics-bridge-core` | `PlatformType` and `RoboticAgent` trait (permissive part) | private / planned |
| `symtropy-bevy-scene` | Opinionated scene/camera scaffolding for examples | private / planned |
| `symtropy-devconsole` | Debug console and inspector scaffolding | private / planned |
| `symtropy-cli` | Project/demo generation and calibration helpers | private / planned |

## Research (copyleft, AGPL-3.0-or-later)

| Crate | Description | crates.io |
|---|---|---|
| `symtropy-consciousness-physics` | Φ coupling, 63 experiments, thermodynamic ledger | [link](https://crates.io/crates/symtropy-consciousness-physics) |
| `symtropy-sim-bridge` | Mycelix governance/economy/FL integration | private |
| `symtropy-world` | Macro/micro sim bridge | private |
| `symtropy-holochain-relay` | Holochain DHT persistence | private |
| `symtropy-lightyear` | Game-tier netcode wrapper | private |
| `symtropy-net` | P2P spatial authority + Holochain relay integration | private |
| `symtropy-robotics-bridge` | FEP agents and Symthaea platform coupling | private |
| `symtropy-bevy` | `symtropy-bevy-core` plus `ConsciousnessField` integration | [link](https://crates.io/crates/symtropy-bevy) |
| `symthaea-bevy-brain` | Full Symthaea cognitive loop as Bevy plugin | private |

## Game / demo (AGPL)

| Crate | Description |
|---|---|
| `symtropy-launcher` (root) | *The Room That Remembers You* + Sol Atlas launcher |
| `symtropy` (meta-crate) | Distribution/re-export crate; license status depends on default dependencies |
| `symtropy-gravcraft-demo` | Gravity craft game demo |
| `symtropy-manipulator-demo` | Manipulator arm demo |
| `symtropy-flight-demo` | Quadrotor demo |
| `symtropy-vehicle-demo` | Vehicle demo |
| `symtropy-auv-demo` | Underwater vehicle demo |
| `symtropy-helicopter-demo` | Helicopter demo |
| `symtropy-exoskeleton-demo` | Exoskeleton demo |
| `symtropy-orbital-demo` | Orbital mechanics demo |
| `symtropy-surgical-demo` | Surgical robotics demo |
| `symtropy-humanoid-demo` | Humanoid demo |
| `symtropy-quadruped-demo` | Quadruped demo |
| `symtropy-demo-capture` | Screenshot capture support for demo verification |

## Dependency graph

```
symtropy-math                                   (no deps)
  └→ symtropy-physics                           (permissive)
        ├→ symtropy-render-bridge               (permissive)
        ├→ symtropy-bevy-core                   (permissive)
        ├→ symtropy-net-core                    (permissive)
        ├→ symtropy-robotics-bridge-core        (permissive)
        └→ symtropy-consciousness-physics       (AGPL)
              ├→ symtropy-bevy                  (AGPL)
              ├→ symtropy-robotics-bridge       (AGPL)
              ├→ symtropy-net                   (AGPL)
              ├→ symtropy-sim-bridge            (AGPL)
              └→ symtropy-world                 (AGPL)
```

## Which do I need?

| Task | Crates |
|---|---|
| ND collision detection only | `symtropy-math` + `symtropy-physics` |
| Generic state-coupled physics, proprietary OK | Core crates only; implement `PhysicsCallback` yourself or use a permissive callback |
| Φ-coupled research | + `symtropy-consciousness-physics` (AGPL) |
| Bevy game, permissive core only | + `symtropy-bevy-core` |
| Bevy game with Φ coupling | + `symtropy-bevy` (AGPL) |
| Mycelix governance testing | + `symtropy-sim-bridge` (AGPL) |
| Symthaea robotics | + `symtropy-robotics-bridge` + relevant platform crate |
