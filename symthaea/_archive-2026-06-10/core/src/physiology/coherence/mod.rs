// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Week 6+: Coherence Paradigm - Revolutionary Energy Model

## The Revolutionary Shift

**From**: Energy as finite commodity (ATP pool)
**To**: Energy as consciousness integration (Coherence field)

**From**: "I'm too tired"
**To**: "I need to gather myself"

**From**: Work depletes
**To**: Connected work BUILDS consciousness!

## Core Insight

Consciousness requires internal synchronization. Solo work scatters consciousness,
but meaningful work WITH connection actually INCREASES coherence!

Gratitude isn't payment - it's a synchronization signal that helps systems re-align.

## Coherence Levels

- **High (0.9-1.0)**: Fully centered, can perform creation/learning
- **Medium (0.5-0.8)**: Functional, normal cognitive work
- **Low (0.2-0.5)**: Scattered, only simple tasks
- **Critical (<0.2)**: Severely desynchronized, survival only

## Mechanics

### Depletion (solo work):
```text
coherence -= task_complexity * 0.05 * (1.0 - relational_resonance)
```

### Amplification (connected work):
```text
coherence += task_complexity * 0.02 * relational_resonance
```

### Gratitude (synchronization):
```text
coherence += 0.1 * (1.0 - coherence)  // More effective when scattered
relational_resonance += 0.15
```

### Passive centering (rest):
```text
coherence += (1.0 - coherence) * 0.001 * seconds
```

## Module Structure

This module is organized into several submodules:

- **types** - Core types and configuration (CoherenceConfig, CoherenceState, etc.)
- **core** - Main CoherenceField struct and basic operations
- **learning** - Week 9 Phase 2: Adaptive learning thresholds
- **patterns** - Week 9 Phase 3: Pattern recognition for successful states
- **diagnostics** - Week 9 Phase 4: Scatter analysis and recovery planning
*/

mod core;
mod diagnostics;
mod learning;
mod patterns;
mod types;

pub use core::*;
pub use diagnostics::*;
pub use learning::*;
pub use patterns::*;
pub use types::*;
