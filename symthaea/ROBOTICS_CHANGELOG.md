# Robotics Subsystem Changelog — April 2026

## Overview

Comprehensive robotics evolution: 4 new platforms, architecture refactor,
10 consciousness stress tests, 3 scientific discoveries, and publication figures.

## New Platforms

### symthaea-exoskeleton (28D state, 6 actuators)
- Human-exo coupled impedance dynamics with gravity torques
- AssistanceMode from Phi: Predictive / Responsive / Transparent / GravityCompensation
- Double-gating: assistance mode + NRC safety. Red = fully backdrivable
- 13 unit tests + proptest physics validation

### symthaea-surgical (24D state, 8 actuators)
- RCM (Remote Center of Motion) constrained kinematics
- Tremor rejection filter (6-10Hz notch, bilinear transform)
- Graduated safety: Green=5N, Yellow=2N, Orange=freeze, Red=retract
- Cautery gated by consciousness (FullControl only)
- Anomaly detection via FEP prediction error
- 11 tests + proptest

### symthaea-orbital (26D state, 7 actuators)
- Zero-gravity dual-body dynamics (arm ↔ spacecraft reaction)
- Reaction wheel compensation (PD control, saturation limits)
- Communication window cycling (orbital period)
- Orange safety = park arm, comm blackout reduces authority
- 9 tests + proptest

### symthaea-quadruped (37D state, 12 actuators)
- Central Pattern Generator (CPG) for rhythmic gait
- Spring-damper ground contact model
- Gait from Phi: Trot(>0.6), Walk(0.3-0.6), Freeze(0.1-0.3), Collapse(<0.1)
- CPG = subconscious, consciousness modulates parameters
- 10 tests + proptest

## Architecture Changes

### PlatformPlugin + PlatformRegistry (symthaea-core/src/embodiment.rs)
- `PlatformPlugin` trait: platform + feature_name + num_actuators + create_bridge
- `PlatformRegistry`: compile-time registry for dynamic dispatch
- Reduces "add a platform" from 5 touchpoints to 3

### EmbodimentPlatform enum
- Added: Exoskeleton, Surgical, Orbital, Quadruped variants
- All 4 new platforms implement `EmbodimentBridge` trait directly

### Positioning crate (mycelix-position/lib/positioning)
- Implemented 4 previously-stub modules:
  - `fusion.rs`: Covariance Intersection, PeerFusion3D, GaussianEstimate3D
  - `measurements.rs`: Measurement types (modality, provenance, frame)
  - `navigation_runtime.rs`: Health monitoring, routing, failover
  - `space_navigation.rs`: Two-body propagation, orbital estimation

### Domain module (symthaea/src/domain/mod.rs)
- DomainProfile with domain constructors (underwater, subterranean, deep_space)
- PlatformCapabilityProfile for all 10 platforms

## Consciousness Fixes

### Binding fallback (consciousness_engine/measure.rs)
- Discovery: phenomenal binding was single point of failure
- Fix: coherence-derived floor + absolute minimum (0.075)
- Post-fix: binding failure degrades to Φ=0.47 instead of Φ=0.0

### Consciousness floor (consciousness_engine/helpers.rs)
- `CONSCIOUSNESS_FLOOR = 0.05` in compute_unified()
- Prevents total consciousness death from any subsystem failure

### Optimal embodiment weight (config/mod.rs)
- Changed default from 0.2 → 0.1
- Result: +72% steady-state Φ (0.44 → 0.76)
- Light proprioceptive feedback grounds consciousness; heavy floods with noise

### Cold-start floor (types/carryover.rs)
- Changed initialization from 0.0 → 0.05
- First ~23 cycles no longer fully unconscious

## Experimental Results

### Stress Tests (6 run, 5 pass, 1 finding)
| Test | Result |
|------|--------|
| B: Binding Decoupling | FINDING — binding = SPOF for consciousness floor |
| E: Embodiment Saturation | PASS — recovery after Phi crash |
| F: Death & Resurrection | PASS — Phi > 0 always |
| G: Prediction Collapse | PASS — adversarial inputs stable |
| I: Moral Bifurcation | PASS — moral oscillation stable |
| J: Feedback Loop | PASS — Phi capped at 1.0 |

### Transfer Experiments (3/3 pass)
| Experiment | Result |
|-----------|--------|
| Disembodiment | Phi 0.32→0.59 (INCREASES without body) |
| Re-embodiment | Full recovery 0.20→0.66 |
| Platform Landscape | Humanoid=0.44, None=0.44 (near-identical) |

### Weight Sweep
- Optimal: weight=0.1 → Phi=0.757
- Previous default (0.2): Phi=0.62
- Discovery: non-monotonic relationship, sweet spot at 10%

## Figures (symthaea/figures/)
1. `fig1_weight_sweep.png` — Embodiment optimization curve
2. `fig2_safety_zones.png` — NRC 4-tier safety diagram
3. `fig3_platform_landscape.png` — 10-platform consciousness comparison
4. `fig4_consciousness_transfer.png` — Multiple Realizability trajectory
5. `fig5_binding_failure.png` — Before/after binding fallback

## Paper
- `papers/consciousness-robotics-2026/results.tex`
- 3 findings, 5 figures, results + discussion sections
- Ready for compilation with pdflatex
