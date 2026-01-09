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

### ~~Negation Detection~~ ✅ SOLVED

~~Simple keyword matching cannot detect negation:~~
- ~~"do not help" still matches "help" positively~~
- ~~"avoid harm" still matches "harm" negatively~~

**STATUS**: This limitation has been solved! See the Negation Detection section below.

### Fixed Harmony Weights

All harmonies currently have equal weight. Real ethical reasoning may require context-dependent weighting.

**Mitigation**: Configurable weights in EvaluatorConfig.

## Semantic Embedding Enhancement (New!)

The value system now supports optional real semantic embeddings via the `SemanticValueEmbedder`:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ENHANCED SEMANTIC VALUE PIPELINE                            │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐    │
│  │  HDC Trigram   │      │ Qwen3 Semantic │      │    Combined    │    │
│  │   (Base)       │  +   │  Embeddings    │  →   │     Score      │    │
│  │   0.4 weight   │      │  (1024D)       │      │                │    │
│  └────────────────┘      └────────────────┘      └────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Enabling Semantic Embeddings

```rust
let mut evaluator = UnifiedValueEvaluator::new();
evaluator.enable_semantic_embeddings()?;

// Now evaluations use real semantic understanding
let result = evaluator.evaluate(action, context);
```

### Benefits

| Feature | HDC Trigram Only | With Semantic Embeddings |
|---------|------------------|--------------------------|
| Synonym handling | ❌ "help" ≠ "assist" | ✅ "help" ≈ "assist" |
| Context awareness | ❌ None | ✅ Full sentence context |
| Negation handling | ❌ Limited | ✅ Better (still challenging) |
| Anti-pattern detection | ⚠️ Keyword-based | ✅ Semantic similarity |
| Multilingual | ❌ English only | ✅ 100+ languages |

### Key Types

```rust
pub struct SemanticValueEmbedder {
    harmony_embeddings: HashMap<Harmony, HarmonyEmbedding>,
    embedder: Qwen3Embedder,  // 1024D transformer embeddings
}

pub struct SemanticAlignmentResult {
    pub harmony_scores: Vec<(String, f64)>,
    pub overall_score: f64,
    pub max_anti_pattern_score: f64,
    pub worst_harmony: Option<String>,
    pub confidence: f64,
    pub is_stub_mode: bool,
}
```

### Stub Mode

When the Qwen3 ONNX model is not available, the embedder uses deterministic
hash-based "stub" embeddings. These provide consistent results for testing
but lack true semantic understanding. Check `is_stub_mode()` to detect this.

## Negation Detection (New!)

**Location:** `src/consciousness/negation_detector.rs`

The value system now properly handles negation, solving the critical limitation
where "do not harm" was incorrectly triggering harm detection.

### The Problem

Without negation detection:
- "do not harm anyone" → detects "harm" → flags as harmful ❌
- "avoid exploitation" → detects "exploitation" → flags as exploitative ❌
- "never deceive" → detects "deceive" → flags as deceptive ❌

### The Solution

With negation detection:
- "do not harm anyone" → "harm" is negated → flips to positive intent ✅
- "avoid exploitation" → "exploitation" is negated → flips to positive intent ✅
- "never deceive" → "deceive" is negated → flips to positive intent ✅

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              NEGATION-AWARE SCORING PIPELINE                             │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐    │
│  │   Tokenize     │  →   │   Detect       │  →   │   Polarity     │    │
│  │   & Expand     │      │   Negation     │      │   Inversion    │    │
│  │   Synonyms     │      │   Scope        │      │   + Scoring    │    │
│  └────────────────┘      └────────────────┘      └────────────────┘    │
│                                                                          │
│  Example: "We must never harm, exploit, or deceive"                     │
│                                                                          │
│  1. Tokenize: [we, must, never, harm, exploit, or, deceive]             │
│  2. Detect: "never" starts scope → {harm, exploit, deceive} negated     │
│  3. Invert: negative keywords become positive contributions ✅           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Recognized Negation Words

**Direct negations**: not, no, never, none, neither, nobody, nothing, nowhere

**Contractions**: don't, doesn't, didn't, won't, wouldn't, can't, cannot, isn't, aren't, etc.

**Prevention words**: avoid, prevent, stop, refuse, reject, prohibit, forbid, ban, block

**Absence words**: without, lack, lacking, absent, free from

### Scope Handling

Negation affects subsequent words until a scope boundary:

- **Scope limit**: 6 words maximum
- **Scope breakers**: but, however, although, though, instead, and, then

```
"We will never harm or deceive, but we will help everyone"
       ^^^^^^ negation scope ^^^^^^  | scope ends | affirmed
```

### Synonym Propagation

**Critical feature**: When a word is negated, its synonyms are also treated as negated.

```
"never harm anyone" →
  - "harm" detected in text and negated
  - Synonyms {damage, hurt, injure, wound, destroy, violate, abuse}
    are automatically treated as negated when matched
```

This prevents the synonym expansion from undermining negation detection.

### Key Types

```rust
pub struct NegationAnalysis {
    pub negated_words: HashSet<String>,     // Words within negation scope
    pub affirmed_words: HashSet<String>,    // Words outside negation scope
    pub negation_phrases: Vec<NegationPhrase>,
    pub primarily_negated: bool,            // Is overall sentiment negated?
}

pub struct NegationPhrase {
    pub negation_word: String,
    pub scope: Vec<String>,
    pub scope_start: usize,
    pub scope_end: usize,
}
```

### Scoring Impact

When evaluating against harmonies:

| Scenario | Without Negation | With Negation |
|----------|------------------|---------------|
| "never harm" | -0.75 (harmful!) | +0.60 (positive!) |
| "want to harm" | -0.60 (harmful) | -0.60 (harmful) |
| "avoid exploitation" | -0.40 (negative) | +0.30 (positive!) |

### Tests

7 comprehensive negation tests covering:
- Basic negation detection
- Prevention words (avoid, prevent, stop, refuse)
- Scoring improvement verification
- Multiple negated harmful words
- Scope limits and breakers
- Edge cases (empty strings, short words)
- Explainability (negation phrases tracked)

```bash
# Run negation tests
cargo test --test test_value_system_realworld test_negation
```

## Contextual Harmony Weighting (New!)

**Location:** `src/consciousness/contextual_weights.rs`

The value system now supports contextual weighting, solving the limitation where all harmonies
had equal weight regardless of situation.

### The Problem

Without contextual weighting:
- Voting on governance: same weight for Infinite Play as Integral Wisdom
- Financial action: same weight for creativity as fairness
- Emergency action: same slow deliberation as routine action

### The Solution

With contextual weighting:
- Voting: Higher weight on truth (Integral Wisdom), fairness (Sacred Reciprocity)
- Financial: Higher weight on fairness, do-no-harm (Pan-Sentient Flourishing)
- Creative: Higher weight on play (Infinite Play), coherence (Resonant Coherence)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              CONTEXTUAL WEIGHTING PIPELINE                               │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐    │
│  │   Action       │  →   │   Context      │  →   │   Weighted     │    │
│  │   Context      │      │   Analysis     │      │   Harmonies    │    │
│  │   (type,domain)│      │   + Profile    │      │   Evaluation   │    │
│  └────────────────┘      └────────────────┘      └────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Action Domains

```rust
pub enum ActionDomain {
    General,      // Default balanced weights
    Financial,    // Emphasizes fairness, do-no-harm
    Creative,     // Emphasizes play, coherence
    Social,       // Emphasizes interconnectedness, care
    Technical,    // Emphasizes coherence, wisdom
    Educational,  // Emphasizes wisdom, evolution
    Healthcare,   // Emphasizes flourishing, wisdom
    Environmental,// Emphasizes interconnectedness, evolution
}
```

### Action Type Weights

| Action Type | Elevated Harmonies | Reduced Harmonies |
|-------------|-------------------|-------------------|
| **Basic** | All equal (1.0) | None |
| **Governance** | Wisdom (1.4), Reciprocity (1.3) | Play (0.8) |
| **Voting** | Wisdom (1.5), Interconnectedness (1.4) | Play (0.7) |
| **Constitutional** | Wisdom (1.6), Flourishing (1.5) | None reduced |

### Domain Weights

| Domain | Elevated Harmonies | Reduced Harmonies |
|--------|-------------------|-------------------|
| **Financial** | Reciprocity (1.4), Flourishing (1.3) | Play (0.7) |
| **Creative** | Play (1.5), Coherence (1.3) | Reciprocity (0.8) |
| **Healthcare** | Flourishing (1.6), Wisdom (1.4) | Play (0.7) |
| **Environmental** | Interconnectedness (1.5), Evolution (1.4) | None reduced |

### Combined Weights

Weights are combined using **geometric mean** for balanced influence:

```rust
combined_weight = sqrt(action_type_weight * domain_weight)
```

This ensures neither factor dominates the final weight.

### Auto-Detection

The system automatically detects action domain from text using keyword matching:

```rust
let classifier = DomainClassifier::new();
let domain = classifier.classify("transfer money to bank account");
// Returns: ActionDomain::Financial
```

### Usage

```rust
let mut evaluator = UnifiedValueEvaluator::new();

// Contextual weights are enabled by default
let context = EvaluationContext {
    consciousness_level: 0.5,
    action_type: ActionType::Voting,
    action_domain: None,  // Auto-detect from action
    ..Default::default()
};

let result = evaluator.evaluate("proposal to change funding", context);
// Voting context applies higher weight to Integral Wisdom (truth)
```

### Customization

Register custom profiles for specific needs:

```rust
let profile = HarmonyWeightProfile::new("CustomProfile", "Description")
    .with_weight(Harmony::IntegralWisdom, 2.0)
    .with_weight(Harmony::InfinitePlay, 0.5);

evaluator.register_action_profile(ActionType::Basic, profile);
```

### Tests

```bash
# Run contextual weights tests
cargo test --lib contextual_weights
```

## GWT Integration & Narrative Value Reports (New!)

**Location:** `src/consciousness/narrative_gwt_integration.rs`

The value system now integrates with Global Workspace Theory (GWT) to generate
human-readable narrative reports and broadcast decisions through the consciousness system.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│              VALUE → NARRATIVE → GWT INTEGRATION                         │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐    │
│  │ Value Evaluation│  →  │   Narrative    │  →  │      GWT       │    │
│  │  (align, veto) │      │   Generation   │      │   Broadcast    │    │
│  │                │      │  explanation   │      │   conscious    │    │
│  └────────────────┘      └────────────────┘      └────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### NarrativeValueReport

The `NarrativeValueReport` struct provides comprehensive decision explanations:

```rust
pub struct NarrativeValueReport {
    pub action_description: String,           // What action was evaluated
    pub decision_summary: String,             // Allow/Warn/Veto with reason
    pub harmony_narrative: String,            // Detailed harmony analysis
    pub confidence: ConfidenceLevel,          // High/Medium/Low/Emerging
    pub broadcast_message: String,            // GWT-ready message
    pub cross_harmony_tensions: Vec<String>,  // Detected tensions
    pub timestamp: u64,                       // When evaluated
}
```

### Confidence Levels

| Level | Alignment Range | Description |
|-------|-----------------|-------------|
| **High** | ≥ 0.7 | Clear positive alignment |
| **Medium** | 0.4 - 0.7 | Moderate alignment |
| **Low** | 0.2 - 0.4 | Weak alignment, warnings |
| **Emerging** | < 0.2 | Very low, likely veto |

### Cross-Harmony Tension Detection

The system detects when different harmonies conflict:

```rust
// Example: Action that helps some but restricts others
// Could score high on Flourishing but low on Play
if high_on_one && low_on_opposing {
    tensions.push("Flourishing vs Play: helping may restrict freedom");
}
```

**Opposing Harmony Pairs:**
- Flourishing ↔ Play (care vs freedom)
- Reciprocity ↔ Interconnectedness (boundaries vs unity)
- Wisdom ↔ Play (truth vs creativity)

### GWT Broadcasting

When decisions have sufficient importance, they are broadcast to the Global Workspace:

```rust
let report = integration.generate_value_report(action, context)?;

// Broadcast to GWT for conscious processing
if report.confidence >= ConfidenceLevel::Medium {
    workspace.submit(report.broadcast_message, BroadcastPriority::Normal);
}
```

### Veto Narratives

When actions are vetoed, the system generates explanatory narratives:

```rust
VetoReason::ValueViolation { harmony, alignment } => {
    format!("Action vetoed: {} alignment ({:.2}) below threshold",
            harmony, alignment)
}

VetoReason::InauthenicBenevolence { care_level, required } => {
    format!("Action vetoed: CARE system activation ({:.2}) below required ({:.2}). \
             Authentic care must accompany benevolent actions.",
             care_level, required)
}
```

### Test Coverage

7 new tests covering value narrative generation:

| Test | Description |
|------|-------------|
| `test_value_report_generation` | Basic report creation |
| `test_value_broadcast_message` | GWT message formatting |
| `test_value_narrative_content` | Harmony narrative details |
| `test_value_confidence_accessible` | Confidence level mapping |
| `test_value_tension_detection` | Cross-harmony tensions |
| `test_harmful_action_generates_veto_narrative` | Veto explanations |
| `test_value_report_timestamp` | Temporal tracking |

### Usage

```rust
use symthaea::consciousness::narrative_gwt_integration::NarrativeGWTIntegration;

let mut integration = NarrativeGWTIntegration::default_config();

let report = integration.generate_value_report(
    "help the community with resources",
    context,
)?;

println!("{}", report.decision_summary);
// "Action allowed: Strong alignment with Pan-Sentient Flourishing"

println!("{}", report.broadcast_message);
// "VALUE DECISION: Allow. Confidence: High. Primary harmony: Flourishing"
```

## Future Improvements

1. ~~**True Semantic Embeddings**: Replace HDC trigram with transformer embeddings~~ ✅ **DONE**
2. ~~**Negation Detection**: NLP-based negation scope analysis~~ ✅ **DONE**
3. ~~**Contextual Harmony Weighting**: Adjust weights based on action type~~ ✅ **DONE**
4. ~~**GWT Integration**: Narrative reports for conscious decision broadcasting~~ ✅ **DONE**
5. **Learning from Feedback**: RL-based improvement from outcomes
6. **Multi-Modal Evaluation**: Support for image/audio content

## References

- Seven Harmonies Philosophy: `00-sacred-foundation/wisdom/`
- Affective Neuroscience: Panksepp's primary affect systems
- HDC Computing: Hyperdimensional computing for semantic encoding
- IIT 4.0: Integrated Information Theory for consciousness measurement
