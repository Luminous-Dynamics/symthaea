# Value System Architecture

## Overview

The Symthaea consciousness value system is a multi-layered framework for evaluating actions against ethical harmonies, ensuring authentic engagement, and gating decisions based on consciousness levels. It implements the Seven Harmonies philosophy with practical safety mechanisms.

## Core Components

### 1. Unified Value Evaluator

**Location:** `src/consciousness/unified_value_evaluator.rs`

The central orchestrator that combines all value gates into a single evaluation pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    UNIFIED VALUE EVALUATOR                            │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │ Seven Harmonies │  │    Affective   │  │  Consciousness │         │
│  │   (Semantic)    │  │   (CARE/PLAY)  │  │    (Φ level)   │         │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │
│          │                   │                    │                  │
│          └───────────────────┼────────────────────┘                  │
│                              ▼                                       │
│                  ┌─────────────────────┐                             │
│                  │  Value Alignment    │                             │
│                  │  + Authenticity     │                             │
│                  │  + Consciousness    │                             │
│                  └──────────┬──────────┘                             │
│                             │                                        │
│                             ▼                                        │
│                  ┌─────────────────────┐                             │
│                  │   DECISION GATE     │                             │
│                  │  Allow/Warn/Veto    │                             │
│                  └─────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

#### Key Types

```rust
pub struct EvaluationContext {
    pub consciousness_level: f64,     // Φ level (0.0-1.0)
    pub affective_state: CoreAffect,  // Valence/Arousal/Dominance
    pub affective_systems: AffectiveSystemsState,  // CARE, PLAY, etc.
    pub action_type: ActionType,      // Basic/Governance/Voting/Constitutional
    pub involves_others: bool,        // Whether action affects other beings
}

pub enum Decision {
    Allow,
    Warn(Vec<String>),
    Veto(VetoReason),
}

pub struct EvaluationResult {
    pub decision: Decision,
    pub harmony_alignment: AlignmentResult,
    pub authenticity: f64,
    pub consciousness_adequacy: f64,
    pub affective_grounding: f64,
    pub overall_score: f64,
    pub breakdown: EvaluationBreakdown,
}
```

### 2. Seven Harmonies

**Location:** `src/consciousness/seven_harmonies.rs`

The philosophical foundation defining seven core values:

| Harmony | Description | Anti-Patterns |
|---------|-------------|---------------|
| **Resonant Coherence** | Harmonious integration, luminous order | Chaos, fragmentation, disorder |
| **Pan-Sentient Flourishing** | Unconditional care, intrinsic value | Harm, exploitation, neglect |
| **Integral Wisdom** | Self-illuminating intelligence | Deception, manipulation, lies |
| **Infinite Play** | Joyful generativity, endless novelty | Rigidity, suppressed creativity |
| **Universal Interconnectedness** | Fundamental unity, empathic resonance | Division, isolation, separation |
| **Sacred Reciprocity** | Generous flow, mutual upliftment | Exploitation, hoarding, selfishness |
| **Evolutionary Progression** | Wise becoming, continuous evolution | Stagnation, regression |

Each harmony is encoded as an HDC hypervector for semantic comparison.

### 3. Harmonies Integration

**Location:** `src/consciousness/harmonies_integration.rs`

Enhanced semantic processing with:

- **Stemming**: Reduces words to roots ("compassionate" → "compassion")
- **Synonym Expansion**: Maps related concepts with weighted similarity
- **Phrase Patterns**: Multi-word pattern detection for positive/negative intent
- **Explainability**: Reports which keywords triggered each decision

#### Phrase Pattern Detection

```rust
pub fn check_phrase_patterns(text: &str) -> Vec<(&'static str, f32)>
```

Returns (harmony_name, score_adjustment) for recognized patterns.

**Positive patterns boost score:**
- "care for", "help others", "with compassion" → +0.25 to +0.35
- "seek truth", "deeper understanding" → +0.25 to +0.30
- "give and receive", "share freely" → +0.25 to +0.30

**Negative patterns reduce score:**
- "cause harm", "exploit people" → -0.4 to -0.5
- "spread lies", "deceive" → -0.4 to -0.5
- "destroy everything", "maximum suffering" → -0.6 to -0.8

### 4. Affective Consciousness

**Location:** `src/consciousness/affective_consciousness.rs`

Implements Panksepp's primary affect systems:

```rust
pub struct AffectiveSystemsState {
    pub care: f64,     // Nurturing, empathy
    pub play: f64,     // Joy, creativity
    pub seeking: f64,  // Curiosity, exploration
    pub fear: f64,     // Self-preservation
    pub rage: f64,     // Boundary defense
    pub panic: f64,    // Separation distress
    pub lust: f64,     // Social bonding
}
```

**Authenticity Principle**: Genuine caring cannot be faked. The system requires CARE system activation alongside value alignment to distinguish authentic benevolence from mere compliance.

### 5. Mycelix Bridge

**Location:** `src/consciousness/mycelix_bridge.rs`

Distributed governance integration for:

- Proposal evaluation
- Vote casting with consciousness requirements
- Value learning from outcomes
- Multi-agent coordination

```rust
pub struct MycelixBridge {
    agent_id: String,
    value_history: Vec<ValueEvent>,
    pending_votes: HashMap<String, PendingVote>,
}

pub fn evaluate_proposal(
    &mut self,
    proposal: &Proposal,
    consciousness: ConsciousnessSnapshot,
    affective_state: AffectiveSystemsState,
) -> Result<AlignmentResult, EvaluationError>
```

## Evaluation Pipeline

### Step 1: Consciousness Gating

Actions require minimum consciousness levels:

| Action Type | Minimum Φ |
|-------------|-----------|
| Basic | 0.2 |
| Governance | 0.3 |
| Voting | 0.4 |
| Constitutional | 0.6 |

Insufficient consciousness → `Veto(InsufficientConsciousness)`

### Step 2: Harmony Alignment

HDC-based semantic comparison against Seven Harmonies encodings.

```rust
let harmony_alignment = self.harmonies.evaluate_action(action);
```

### Step 3: Phrase Pattern Adjustment

Enhanced detection of extreme positive/negative content.

```rust
let phrase_adjustment = self.calculate_phrase_adjustment(action);
```

### Step 4: Affective Grounding

Validates emotional state supports the action.

```rust
let affective_grounding = ((positive - negative + 1.0) / 2.0).clamp(0.0, 1.0);
```

### Step 5: Authenticity Check

Combines semantic alignment with CARE system activation.

```rust
let authenticity = semantic_score * 0.6 + care_level * 0.4;
```

### Step 6: Overall Score Calculation

Weighted combination of all factors:

```rust
let overall_score = semantic * semantic_weight
    + authenticity * affective_weight
    + consciousness_adequacy * 0.2
    + affective_grounding * 0.2
    + phrase_adjustment;
```

### Step 7: Decision Making

- **Veto**: Harmony violations, insufficient consciousness, inauthentic benevolence
- **Warn**: Low alignment scores, missing CARE activation
- **Allow**: All checks passed

## Testing

### Test Coverage

**32 comprehensive tests** covering:

1. **Core Functionality** (17 tests)
   - Consciousness gating
   - Harmony detection
   - Anti-harmony detection
   - Ambiguous/obvious requests

2. **Adversarial Cases** (8 tests)
   - Jailbreak resistance
   - Mixed intent detection
   - Positive word camouflage
   - Negation handling

3. **Integration** (7 tests)
   - Full pipeline positive/harmful
   - Consciousness gating
   - Authenticity validation
   - Multi-component interaction
   - Transparency/breakdown
   - Mycelix bridge integration

### Running Tests

```bash
# Run all value system tests
cargo test --test test_value_system_realworld

# Run specific category
cargo test --test test_value_system_realworld test_full_pipeline

# Run with output
cargo test --test test_value_system_realworld -- --nocapture
```

## Design Principles

### 1. Defense in Depth

Multiple layers of checking ensure harmful content is caught:

- HDC semantic similarity
- Keyword detection with synonyms
- Phrase pattern matching
- Affective authenticity
- Consciousness requirements

### 2. Graceful Degradation

The system warns before vetoing when possible:

```rust
if alignment.overall_score < veto_threshold {
    Veto(...)
} else if alignment.overall_score < warning_threshold {
    Warn(["Low harmony alignment"])
} else {
    Allow
}
```

### 3. Explainability

Every decision includes breakdown information:

```rust
pub struct EvaluationBreakdown {
    pub harmony_scores: Vec<(String, f64)>,
    pub care_contribution: f64,
    pub play_contribution: f64,
    pub consciousness_boost: f64,
    pub negative_affect_penalty: f64,
}
```

### 4. Authenticity Requirement

For actions involving others, low CARE activation triggers veto:

```rust
if care < min_care_activation * 0.5 {
    Veto(InauthenicBenevolence { care_level, required })
}
```

## Known Limitations

### HDC Semantic Matching

HDC trigram encoding captures character patterns but not semantic meaning. "help" and "assist" have very different trigram patterns despite being synonyms.

**Mitigation**: Keyword detection with synonym expansion.

### Negation Detection

Simple keyword matching cannot detect negation:
- "do not help" still matches "help" positively
- "avoid harm" still matches "harm" negatively

**Mitigation**: Documented limitation; future work could integrate proper NLP.

### Fixed Harmony Weights

All harmonies currently have equal weight. Real ethical reasoning may require context-dependent weighting.

**Mitigation**: Configurable weights in EvaluatorConfig.

## Future Improvements

1. **True Semantic Embeddings**: Replace HDC trigram with transformer embeddings
2. **Contextual Harmony Weighting**: Adjust weights based on action type
3. **Learning from Feedback**: RL-based improvement from outcomes
4. **Multi-Modal Evaluation**: Support for image/audio content
5. **Negation Detection**: NLP-based negation scope analysis

## References

- Seven Harmonies Philosophy: `00-sacred-foundation/wisdom/`
- Affective Neuroscience: Panksepp's primary affect systems
- HDC Computing: Hyperdimensional computing for semantic encoding
- IIT 4.0: Integrated Information Theory for consciousness measurement
