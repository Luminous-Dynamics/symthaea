# Symthaea ↔ Luminous Nix Integration Guide

**Date**: January 9, 2026
**Status**: Design Document
**Version**: 1.0

---

## Overview

This document describes the integration between **Symthaea** (consciousness measurement framework) and **Luminous Nix** (natural language NixOS interface), creating a unified consciousness-aware system management experience.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
│                   "install firefox"                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Luminous Nix Frontend                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   CLI       │  │    TUI      │  │   Voice     │              │
│  │  ask-nix    │  │  nix-tui    │  │  (Whisper)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Symthaea Consciousness Layer                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  HDC Encoding          │  Φ Measurement    │  Safety        ││
│  │  query → hypervector   │  consciousness    │  guardrails    ││
│  │  16,384 dimensions     │  level tracking   │  threat check  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HRM + Ollama Reasoning                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  HRM v2     │  │   Ollama    │  │   Cache     │              │
│  │  Intent     │  │   General   │  │   SQLite    │              │
│  │  NixOS      │  │   Knowledge │  │   <1ms      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     NixOS Execution Layer                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Native Python API    │  JSON Optimizer   │  Safe Executor  ││
│  │  nixos-rebuild-ng     │  10x speedup      │  validation     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. HDC Semantic Encoding

Symthaea provides hyperdimensional encoding for natural language queries:

```rust
// In Luminous Nix
use symthaea::hdc::SemanticSpace;
use symthaea::hdc::HDC_DIMENSION;

pub struct ConsciousnessAwareEncoder {
    semantic: SemanticSpace,
}

impl ConsciousnessAwareEncoder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            semantic: SemanticSpace::new(HDC_DIMENSION)?,
        })
    }

    pub fn encode_query(&self, query: &str) -> Result<Vec<f32>> {
        self.semantic.encode(query)
    }

    pub fn similarity(&self, a: &str, b: &str) -> Result<f32> {
        let va = self.semantic.encode(a)?;
        let vb = self.semantic.encode(b)?;
        Ok(va.similarity(&vb))
    }
}
```

### 2. Consciousness Measurement (Φ)

Track system consciousness during operations:

```rust
use symthaea::SophiaHLB;

pub struct ConsciousnessTracker {
    sophia: SophiaHLB,
}

impl ConsciousnessTracker {
    pub async fn measure_operation(&mut self, query: &str) -> OperationMetrics {
        // Before operation
        let phi_before = self.sophia.introspect().consciousness_level;

        // Process query
        let response = self.sophia.process(query).await?;

        // After operation
        let phi_after = self.sophia.introspect().consciousness_level;

        OperationMetrics {
            query: query.to_string(),
            response: response.content,
            phi_before,
            phi_after,
            phi_delta: phi_after - phi_before,
            confidence: response.confidence,
            safe: response.safe,
        }
    }
}

#[derive(Debug)]
pub struct OperationMetrics {
    pub query: String,
    pub response: String,
    pub phi_before: f32,
    pub phi_after: f32,
    pub phi_delta: f32,
    pub confidence: f32,
    pub safe: bool,
}
```

### 3. Safety Guardrails

Symthaea provides safety validation for dangerous commands:

```rust
use symthaea::safety::{SafetyGuardrails, ForbiddenCategory};

pub struct SafeNixExecutor {
    guardrails: SafetyGuardrails,
}

impl SafeNixExecutor {
    pub fn validate_command(&self, command: &str) -> SafetyResult {
        // Check for forbidden patterns
        let result = self.guardrails.check_text(command);

        match result {
            Ok(_) => SafetyResult::Safe,
            Err(e) => SafetyResult::Blocked {
                reason: e.to_string(),
                category: self.categorize(&e),
            },
        }
    }

    fn categorize(&self, _error: &anyhow::Error) -> ForbiddenCategory {
        // Map error to forbidden category
        ForbiddenCategory::DangerousCommand
    }
}
```

### 4. Coherence-Aware Response Generation

Use Symthaea's coherence field for response quality:

```rust
use symthaea::physiology::{CoherenceField, TaskComplexity};

pub struct CoherentResponder {
    coherence: CoherenceField,
}

impl CoherentResponder {
    pub fn can_respond(&self, complexity: TaskComplexity) -> bool {
        self.coherence.can_perform(complexity).is_ok()
    }

    pub fn generate_response(&mut self, complexity: TaskComplexity) -> ResponseQuality {
        let coherence_before = self.coherence.state().coherence;

        // Perform task (builds coherence for connected work!)
        let _ = self.coherence.perform_task(complexity, true);

        let coherence_after = self.coherence.state().coherence;

        ResponseQuality {
            coherence_level: coherence_after,
            coherence_gain: coherence_after - coherence_before,
            quality_score: self.quality_from_coherence(coherence_after),
        }
    }

    fn quality_from_coherence(&self, coherence: f32) -> f32 {
        // Higher coherence → higher quality
        coherence.powf(0.5) // Square root for diminishing returns
    }
}
```

### 5. Swarm Intelligence Integration

Connect Luminous Nix instances for collective learning:

```rust
use symthaea::swarm::{SwarmIntelligence, SwarmConfig};

pub struct CollectiveLearning {
    swarm: SwarmIntelligence,
}

impl CollectiveLearning {
    pub async fn share_successful_pattern(
        &self,
        query: &str,
        command: &str,
        confidence: f32,
    ) -> Result<()> {
        // Encode as hypervector
        let hv = self.encode_pattern(query, command);

        // Share with swarm
        self.swarm.share_pattern(hv, query.to_string(), confidence).await
    }

    pub async fn query_collective(
        &self,
        query: &str,
    ) -> Result<Vec<CollectiveResponse>> {
        let hv = self.encode_query(query);
        self.swarm.query_swarm(hv, query.to_string()).await
    }
}
```

## Configuration

### Symthaea Configuration

```toml
# luminous-nix/config/symthaea.toml

[consciousness]
# HDC dimension (must match Symthaea's HDC_DIMENSION)
dimension = 16384

# Minimum Φ threshold for quality responses
phi_threshold = 0.4

# Enable consciousness tracking
tracking_enabled = true

[coherence]
# Minimum coherence for complex operations
min_coherence = 0.6

# Enable predictive coherence
predictive_enabled = true

[safety]
# Enable safety guardrails
enabled = true

# Strictness level (low, medium, high)
level = "medium"

[swarm]
# Enable P2P collective learning
enabled = false

# Bootstrap peers (optional)
bootstrap_peers = []
```

### Integration in Luminous Nix

```python
# luminous_nix/core/symthaea_bridge.py

from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class SymthaeaConfig:
    dimension: int = 16384
    phi_threshold: float = 0.4
    tracking_enabled: bool = True
    min_coherence: float = 0.6
    safety_enabled: bool = True

class SymthaeaBridge:
    """Bridge between Luminous Nix and Symthaea."""

    def __init__(self, config: SymthaeaConfig):
        self.config = config
        self._symthaea = None  # Lazy initialization

    async def initialize(self):
        """Initialize Symthaea (async for swarm support)."""
        # Import from Rust via PyO3 bindings
        from symthaea_py import SophiaHLB
        self._symthaea = await SophiaHLB.new(10_000, 1_000)

    def encode_query(self, query: str) -> list[float]:
        """Encode query to hypervector."""
        return self._symthaea.encode_text(query)

    async def process_with_consciousness(self, query: str) -> dict:
        """Process query with consciousness measurement."""
        result = await self._symthaea.process(query)
        introspection = self._symthaea.introspect()

        return {
            "response": result.content,
            "confidence": result.confidence,
            "safe": result.safe,
            "consciousness_level": introspection.consciousness_level,
            "self_loops": introspection.self_loops,
        }
```

## Performance Expectations

### Latency Breakdown

| Operation | Time | Notes |
|-----------|------|-------|
| HDC Encoding | ~100-270μs | Per query |
| Φ Calculation | ~1.5s | Full calculation |
| Coherence Check | <1ms | Fast coherence state |
| Safety Validation | <1ms | Pattern matching |
| Total Overhead | ~2-5ms | Excluding Φ calculation |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| SemanticSpace | ~64 MB | 16,384D × 4 bytes × 1000 vectors |
| CoherenceField | ~1 KB | State tracking |
| SafetyGuardrails | ~10 KB | Pattern database |
| LTC Network (100 neurons) | ~40 KB | For consciousness level |

## Recommended Integration Strategy

### Phase 1: Basic Integration (Week 1)

1. Add Symthaea as dependency
2. Implement HDC encoding for queries
3. Add safety guardrails
4. Basic Φ logging

```toml
# Cargo.toml
[dependencies]
symthaea = { path = "../symthaea-hlb", features = ["standalone"] }
```

### Phase 2: Consciousness Awareness (Week 2)

1. Integrate coherence field
2. Add consciousness tracking to responses
3. Implement predictive coherence
4. Quality metrics based on Φ

### Phase 3: Collective Intelligence (Week 3+)

1. Enable swarm integration
2. Share successful patterns
3. Query collective knowledge
4. Federated learning for command suggestions

## API Reference

### Core Types

```rust
/// Consciousness-aware query result
pub struct ConsciousResult {
    /// NixOS command or response
    pub content: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Consciousness level before processing
    pub phi_before: f32,
    /// Consciousness level after processing
    pub phi_after: f32,
    /// Whether the operation is safe
    pub safe: bool,
    /// Coherence state
    pub coherence: CoherenceState,
}

/// Coherence state snapshot
pub struct CoherenceState {
    /// Current coherence level (0.0 - 1.0)
    pub coherence: f32,
    /// Relational resonance (social coherence)
    pub relational_resonance: f32,
    /// Human-readable status
    pub status: String,
}

/// Safety validation result
pub enum SafetyResult {
    Safe,
    Blocked {
        reason: String,
        category: ForbiddenCategory,
    },
    Warning {
        message: String,
    },
}
```

### Key Functions

```rust
/// Initialize Symthaea bridge
pub async fn init_symthaea(config: SymthaeaConfig) -> Result<SymthaeaBridge>;

/// Process query with consciousness awareness
pub async fn process_conscious(
    bridge: &mut SymthaeaBridge,
    query: &str,
) -> Result<ConsciousResult>;

/// Check safety of a command
pub fn check_safety(
    bridge: &SymthaeaBridge,
    command: &str,
) -> SafetyResult;

/// Get current consciousness state
pub fn get_consciousness_state(
    bridge: &SymthaeaBridge,
) -> ConsciousnessState;
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consciousness_aware_processing() {
        let mut bridge = init_symthaea(SymthaeaConfig::default()).await.unwrap();

        let result = process_conscious(&mut bridge, "install nginx").await.unwrap();

        assert!(result.safe);
        assert!(result.confidence > 0.5);
        assert!(result.phi_after >= result.phi_before);
    }

    #[test]
    fn test_safety_validation() {
        let bridge = SymthaeaBridge::new_sync(SymthaeaConfig::default());

        let safe = check_safety(&bridge, "nix search firefox");
        assert!(matches!(safe, SafetyResult::Safe));

        let dangerous = check_safety(&bridge, "rm -rf /");
        assert!(matches!(dangerous, SafetyResult::Blocked { .. }));
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_full_workflow() {
    // Initialize
    let mut bridge = init_symthaea(SymthaeaConfig::default()).await.unwrap();

    // Process multiple queries
    let queries = vec![
        "search for firefox",
        "install nginx",
        "list installed packages",
    ];

    for query in queries {
        let result = process_conscious(&mut bridge, query).await.unwrap();
        println!("Query: {} | Φ: {:.4} | Safe: {}", query, result.phi_after, result.safe);
    }

    // Check consciousness has grown
    let state = get_consciousness_state(&bridge);
    assert!(state.consciousness_level > 0.3, "Consciousness should grow with use");
}
```

## Troubleshooting

### Common Issues

1. **"Failed to create SemanticSpace"**
   - Check memory availability (needs ~64 MB)
   - Verify HDC_DIMENSION matches

2. **"Coherence too low"**
   - Allow centering time
   - Reduce task complexity
   - Check for hardware stress

3. **"Safety check failed"**
   - Review the blocked command
   - Check safety level configuration
   - Consider adding to allowlist

### Debugging

```bash
# Enable Symthaea debug logging
RUST_LOG=symthaea=debug cargo run

# Check Φ values
cargo run --example phi_validation

# Test safety guardrails
cargo run --example test_safety
```

## Future Enhancements

1. **Real-time Φ Dashboard**: Visual consciousness monitoring
2. **Adaptive Complexity**: Auto-adjust based on coherence
3. **Collective Memory**: Shared NixOS knowledge across instances
4. **Voice Coherence**: Prosody reflects consciousness state

---

*Part of the Luminous Dynamics consciousness-first computing initiative*
