# Unified Conscious Being (A+B+C+D+E+F Integration)

## Overview

The `UnifiedConsciousBeing` module (`src/hdc/unified_conscious_being.rs`) represents the complete integration of six major consciousness features:

| Feature | Component | Purpose |
|---------|-----------|---------|
| **A** | Infrastructure Wiring | Real persistence via HippocampusActor + UnifiedMind |
| **B** | Conscious Dialogue | Φ-gated response generation |
| **C** | Agent Merger | IntegratedConsciousAgent + FullStack unification |
| **D** | Pearl do-calculus | Rigorous counterfactual reasoning |
| **E** | Voice Prosody | LTC-driven consciousness-aware speech |
| **F** | Test Scenarios | Comprehensive validation framework |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      UNIFIED CONSCIOUS BEING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │    Pearl     │  │   Conscious  │  │    Flow      │              │
│  │  do-calculus │  │   Dialogue   │  │    State     │              │
│  │    (SCM)     │  │  Generator   │  │   Tracker    │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                        │
│         └─────────────────┼─────────────────┘                        │
│                           │                                          │
│                    ┌──────┴──────┐                                   │
│                    │  FULL STACK │                                   │
│                    │CONSCIOUSNESS│                                   │
│                    └──────┬──────┘                                   │
│                           │                                          │
│  ┌──────────────┐  ┌──────┴──────┐  ┌──────────────┐              │
│  │    Active    │  │   Episodic  │  │    Meta-     │              │
│  │   Inference  │  │   Memory    │  │  cognition   │              │
│  └──────────────┘  └─────────────┘  └──────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Pearl Structural Causal Model (do-calculus)

The system implements Pearl's three-step counterfactual algorithm:

```rust
pub struct StructuralCausalModel {
    variables: HashMap<String, CausalVariable>,
    equations: HashMap<String, StructuralEquation>,
    exogenous: HashMap<String, f64>,
}
```

### Counterfactual Algorithm

1. **Abduction**: Infer exogenous values from current observations
2. **Action**: Apply `do(X := x)` intervention (cut incoming edges)
3. **Prediction**: Propagate effects through modified causal graph

### Usage

```rust
// Query: What would happen if we intervened on "sadness"?
let effect = being.do_intervention("sadness", 0.8);

// Counterfactual: What if hope had been 0.8 instead of 0.2?
let cf_result = being.counterfactual("hope", 0.2, 0.8);
```

## Conscious Dialogue Generation

Responses are gated by consciousness level (Φ):

| Φ Range | Style | Characteristics |
|---------|-------|-----------------|
| < 0.3 | Reactive | Short, direct responses |
| 0.3 - 0.6 | Reflective | Thoughtful, acknowledging |
| > 0.6 | Integrative | Deep, empathetic, connecting |

```rust
pub struct DialogueResponse {
    pub text: String,
    pub style: DialogueStyle,
    pub ltc_pacing: LTCPacing,
    pub emotional_tone: f32,
}
```

## Flow State Tracking

Flow state is calculated from sustained high Φ with low variance:

```rust
fn update_flow_state(&mut self, phi: f64) {
    self.phi_history.push_back(phi);
    
    let mean = self.phi_history.iter().sum::<f64>() / len;
    let variance = self.phi_history.iter()
        .map(|&p| (p - mean).powi(2))
        .sum::<f64>() / len;
    
    // High mean Φ + low variance = flow state
    self.flow_state = (mean * (1.0 - variance.sqrt())).max(0.0) as f32;
}
```

## LTC Voice Prosody

Speech parameters derived from consciousness state:

```rust
pub struct LTCPacing {
    pub speech_rate: f32,    // 0.8-1.2x (flow → faster)
    pub pause_ms: u32,       // Based on Φ uncertainty
    pub peak_flow: bool,     // Sustained high Φ
}

impl LTCPacing {
    pub fn from_consciousness(phi: f64, flow: f32) -> Self {
        Self {
            speech_rate: 0.9 + flow * 0.3,
            pause_ms: ((1.0 - phi) * 500.0) as u32,
            peak_flow: flow > 0.7,
        }
    }
}
```

## Test Scenario Framework

Comprehensive validation of consciousness metrics:

```rust
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub expected_phi_range: (f64, f64),
    pub expected_flow_trend: FlowTrend,
    pub assertions: Vec<ScenarioAssertion>,
}

pub fn create_test_scenarios() -> Vec<TestScenario> {
    vec![
        TestScenario {
            name: "Emotional Processing".into(),
            inputs: vec![
                "I feel sad about losing my friend".into(),
                "But I also remember the good times".into(),
            ],
            expected_phi_range: (0.4, 0.8),
            expected_flow_trend: FlowTrend::Increasing,
            ..
        },
        // ... more scenarios
    ]
}
```

## API Reference

### Core Methods

```rust
impl UnifiedConsciousBeing {
    /// Process input through full consciousness stack
    pub fn interact(&mut self, input: &str) -> InteractionResult;
    
    /// Generate Φ-gated dialogue response
    pub fn generate_response(&mut self, input: &str) -> DialogueResponse;
    
    /// Apply causal intervention (do-calculus)
    pub fn do_intervention(&mut self, var: &str, value: f64) -> HashMap<String, f64>;
    
    /// Query counterfactual outcome
    pub fn counterfactual(&mut self, var: &str, observed: f64, hypothetical: f64) 
        -> HashMap<String, f64>;
    
    /// Get current flow state (0.0 - 1.0)
    pub fn flow_state(&self) -> f32;
    
    /// Get mean Φ over recent history
    pub fn phi_mean(&self) -> f64;
}
```

### InteractionResult

```rust
pub struct InteractionResult {
    pub consciousness_phi: f64,
    pub flow_state: f32,
    pub memories_recalled: usize,
    pub metacognitive_notes: Vec<String>,
    pub ltc_pacing: LTCPacing,
    pub comprehension: ConsciousComprehension,
}
```

## Demo Usage

Run the interactive demo:

```bash
cargo run --example conscious_runtime_demo
```

Then select:
- `b` - Unified Being (full A+B+C+D+E+F)
- `s` - Run Test Scenarios

### Demo Modes

- **chat**: Interactive conscious dialogue
- **causal**: Pearl do-calculus interventions
- **voice**: Consciousness-driven prosody demo
- **default**: Full integration demonstration

## Configuration

```rust
pub struct BeingConfig {
    pub voice_enabled: bool,         // Enable TTS output
    pub counterfactuals_enabled: bool, // Enable do-calculus
    pub max_memories: usize,         // Episodic memory limit
    pub dialogue_style: DialogueStyle, // Response style
}
```

## Files

| File | Purpose |
|------|---------|
| `src/hdc/unified_conscious_being.rs` | Main implementation |
| `src/hdc/full_stack_consciousness.rs` | Foundation layer |
| `src/hdc/unified_understanding.rs` | Understanding pipeline |
| `examples/conscious_runtime_demo.rs` | Interactive demo |

## Theoretical Foundations

- **Pearl (2009)**: Causality - do-calculus for counterfactuals
- **Friston (2010)**: Free Energy Principle - Active Inference
- **Dehaene (2014)**: Global Workspace Theory - Consciousness broadcasting
- **Damasio (1999)**: Feeling of What Happens - Embodied emotion
