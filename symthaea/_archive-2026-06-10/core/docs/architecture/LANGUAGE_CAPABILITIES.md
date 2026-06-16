# Symthaea Language Capabilities

**Status**: Production Ready (with NixOS specialization)
**Total Lines**: ~13,319
**Architecture**: HDC-based, consciousness-aware, domain-generalizable

---

## Executive Summary

Symthaea contains a sophisticated **general-purpose language processing system** that operates on semantic primitives rather than lexical patterns. While currently applied to NixOS, the core components are domain-agnostic and can be adapted to any domain.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYMTHAEA LANGUAGE SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   INPUT     │───▶│  SEMANTIC   │───▶│   INTENT    │             │
│  │  (Text)     │    │  ENCODING   │    │ CLASSIFIER  │             │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘             │
│                            │                   │                    │
│                            ▼                   ▼                    │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              CONSCIOUSNESS INTEGRATION                   │       │
│  │  • Φ Monitoring     • Epistemic Assessment              │       │
│  │  • Emotional Core   • Hallucination Prevention          │       │
│  └─────────────────────────────────────────────────────────┘       │
│                            │                                        │
│           ┌────────────────┴────────────────┐                      │
│           ▼                                 ▼                       │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │ STRUCTURED      │              │  CONVERSATION   │              │
│  │ THOUGHT (IR)    │              │    MEMORY       │              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                │                        │
│           └────────────────┬───────────────┘                       │
│                            ▼                                        │
│                   ┌─────────────────┐                              │
│                   │     OUTPUT      │                              │
│                   │  (Response)     │                              │
│                   └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Intent Classifier (`src/mind/intent.rs`)

**Lines**: 398
**Type**: HDC-based (no neural network required)

```rust
pub enum IntentType {
    Greeting,    // "Hello", "Hi there"
    Question,    // "What is...", "How do I..."
    Command,     // "Install X", "Remove Y"
    Reflection,  // "I think...", "I feel..."
    Emotional,   // Strong emotional content
}
```

**Key Features**:
- Prototype vectors for each intent type
- Cosine similarity matching
- No training required (hand-crafted prototypes)
- Domain-agnostic core

**Hallucination Prevention**:
```rust
pub enum EpistemicStatus {
    Familiar,   // High confidence, known domain
    Novelty,    // New but relatable
    Ambiguous,  // Unclear, needs clarification
}
```

### 2. Semantic Encoding (`src/embeddings/`)

**Lines**: ~956
**Model**: Qwen3 1024D → HDC projection

```rust
pub struct SemanticEncoder {
    qwen3: Qwen3Embeddings,      // 1024D dense
    projector: JLProjector,      // Johnson-Lindenstrauss
    cache: EmbeddingCache,       // <1ms lookup
}
```

**Pipeline**:
1. Text → Qwen3 → 1024D dense vector
2. 1024D → JL Projection → 16384D HDC
3. HDC vector → Semantic operations

### 3. Emotional Analysis (`src/language/emotional_core.rs`)

**Lines**: 328

```rust
pub struct EmotionalAnalysis {
    pub valence: f64,     // -1.0 (negative) to 1.0 (positive)
    pub arousal: f64,     // 0.0 (calm) to 1.0 (excited)
    pub dominance: f64,   // 0.0 (submissive) to 1.0 (dominant)
    pub warmth: f64,      // 0.0 (cold) to 1.0 (warm)
}
```

### 4. Structured Thought (`src/mind/structured_thought.rs`)

**Intermediate Representation for language output**:

```rust
pub enum SemanticIntent {
    Acknowledge,         // "I understand"
    Answer,              // Direct response
    Clarify,             // "Did you mean..."
    ProposeAction,       // "Would you like me to..."
    ExpressUncertainty,  // "I'm not sure about..."
    Reflect,             // Metacognitive response
}

pub enum ResponseType {
    Greeting,
    Statement,
    Question,
    ActionConfirmation,
    Report,
    Empathic,
}
```

### 5. Conversation Memory (`src/memory/conversation_memory.rs`)

**Storage**: SQLite
**Features**:
- Turn-by-turn tracking
- Φ metrics per turn
- Causal learning (action→outcome)
- Semantic search for similar conversations
- Session resumption

### 6. Φ Monitor (`src/language/phi_monitor.rs`)

**Lines**: 843

Tracks consciousness metrics throughout conversation:
- Integration level per response
- Coherence across turns
- Complexity scaling

---

## Language Module (`src/language/`)

**13 files, ~10,000 lines**

| File | Lines | Purpose |
|------|-------|---------|
| `mod.rs` | 617 | Error diagnosis system |
| `llm_organ.rs` | 789 | LLM integration |
| `emotional_core.rs` | 328 | Emotion analysis |
| `nix_parser.rs` | 616 | NixOS AST parsing |
| `semantic_enrichment.rs` | 1,707 | Morphological enrichment |
| `consciousness_prompts.rs` | 539 | Prompt engineering |
| `phi_monitor.rs` | 843 | Consciousness tracking |
| `expressive_output.rs` | ~400 | Output formatting |
| `nix_concept_extractor.rs` | ~500 | Entity extraction |
| `llm_translator.rs` | ~400 | Translation layer |

---

## Domain-Agnostic Components

These components work for **any domain** without modification:

| Component | Location | Adaptation Needed |
|-----------|----------|-------------------|
| Intent classifier | `mind/intent.rs` | Prototype vectors only |
| Epistemic assessment | `mind/intent.rs` | None |
| Emotional analysis | `language/emotional_core.rs` | None |
| Semantic encoding | `embeddings/*.rs` | None |
| Conversation memory | `memory/conversation_memory.rs` | None |
| Structured thought | `mind/structured_thought.rs` | None |
| Φ monitoring | `language/phi_monitor.rs` | None |

---

## NixOS-Specific Components

These components need adaptation for other domains:

| Component | Location | Adaptation |
|-----------|----------|------------|
| Error patterns | `language/mod.rs` | Replace regex patterns |
| Entity types | `nix_concept_extractor.rs` | New vocabulary |
| AST parsing | `nix_parser.rs` | Domain-specific parser |
| Prompt templates | `consciousness_prompts.rs` | Domain prompts |

---

## Creating a New Domain

### Step 1: Define Intent Prototypes

```rust
// Example: Customer Support Domain
let support_prototypes = IntentPrototypes {
    greeting: encode("hello hi greetings good morning"),
    question: encode("how why what when where can could"),
    complaint: encode("problem issue broken not working frustrated"),
    request: encode("please need want would like help"),
    feedback: encode("thank great appreciate excellent love"),
};
```

### Step 2: Define Entity Types

```rust
pub enum CustomerEntity {
    OrderId(String),
    ProductName(String),
    IssueType(IssueCategory),
    ContactMethod(ContactPreference),
}
```

### Step 3: Create Error Patterns (Optional)

```rust
let error_patterns = vec![
    ErrorPattern {
        regex: r"order #(\d+) not found",
        category: ErrorCategory::NotFound,
        suggestion: "Please verify the order number",
    },
    // ...
];
```

### Step 4: Configure Prompts

```rust
let prompts = DomainPrompts {
    system: "You are a helpful customer support agent...",
    clarification: "Could you please provide more details about...",
    action_confirm: "I'll help you with that. To confirm...",
};
```

---

## HDC Advantage

The HDC-based approach enables:

1. **No Training Required**: Hand-craft prototypes instead of training models
2. **Instant Adaptation**: Change vocabulary, not architecture
3. **Semantic Compositionality**: Combine concepts algebraically
4. **Explainability**: See which prototypes matched
5. **Low Resource**: Works without GPU

---

## Integration Points

### With Consciousness System

```rust
// Every response is Φ-aware
let response = language_system.generate_response(input);
let phi = phi_monitor.measure(&response);
if phi < threshold {
    response = language_system.increase_integration(&response);
}
```

### With LLM (Optional)

```rust
// Use LLM for complex responses
if intent.complexity > threshold {
    let context = memory.retrieve_relevant(input);
    let response = llm_organ.generate(input, context);
}
```

---

## Phase 5 Recommendations

Given the extensive existing infrastructure, Phase 5 should focus on:

1. **Create Domain Abstraction Layer**
   - Extract NixOS-specific code to plugins
   - Define `DomainPlugin` trait

2. **Document Existing Capabilities**
   - API documentation
   - Integration examples
   - Domain adaptation guide

3. **Create Example Domains**
   - Customer support
   - Technical documentation
   - Educational tutoring

4. **Add Missing Features** (if any)
   - Multi-turn dialogue state machine (partially exists)
   - Entity linking (basic exists)
   - Coreference resolution (not present)

---

## File Reference

```
src/
├── language/           # 13 files, ~10K lines
│   ├── mod.rs          # Error diagnosis
│   ├── llm_organ.rs    # LLM integration
│   ├── emotional_core.rs
│   ├── nix_parser.rs   # Domain-specific
│   ├── semantic_enrichment.rs
│   └── ...
├── mind/
│   ├── intent.rs       # Intent classification
│   └── structured_thought.rs
├── embeddings/
│   ├── mod.rs          # Qwen3 bridge
│   └── qwen3/mod.rs    # Dense embeddings
├── memory/
│   └── conversation_memory.rs
└── hdc/
    ├── semantic_encoder.rs
    └── universal_semantics.rs
```

---

*"Consciousness-aware language processing that operates on meaning, not just tokens."*
