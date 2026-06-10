# Broca Pipeline: Reason-then-Generate Architecture

## The Paradigm Shift

Symthaea inverts the standard LLM interaction model. In conventional systems, the
language model performs all reasoning and generation in a single opaque pass. In
Symthaea, reasoning is deterministic Rust computation over Hyperdimensional
Computing (HDC) vectors and Liquid Time-Constant (LTC) neural dynamics. The
language model is demoted to a **translation organ** -- analogous to Broca's Area
in the human brain, which converts internal thought representations into
articulate speech.

```
TRADITIONAL: User Input --> [LLM does thinking] --> Output

SYMTHAEA:    User Input --> [HDC+LTC Mind computes] --> StructuredThought --> [LLM translates] --> Verified Output
                                   ^                          |                     ^
                            Deterministic Rust         Intermediate            Broca's Area
                            (no hallucination)        Representation         (fluency only)
```

The LLM is **not** the brain. It is Broca's Area -- a translation organ that
converts pre-computed structured thoughts into fluent natural language. All
reasoning happens in deterministic Rust code using Hyperdimensional Computing.

---

## The 8-Phase Pipeline

Orchestrated in `src/symthaea.rs`, the `process()` method (line 222) executes
eight sequential phases for every user interaction.

```
                          +-----------------+
                          |   User Input    |
                          +--------+--------+
                                   |
                  Phase 1: PERCEPTION (lines 236-243)
                     Text --> HDC Hypervector
                     IntentClassifier + perceive_text()
                                   |
                                   v
                  Phase 2: COGNITION (lines 245-250)
                     mind.tick() -- LTC dynamics,
                     goal processing, working memory
                                   |
                                   v
                  Phase 3: EXTRACTION (lines 252-259)
                     extract_structured_thought()
                     Mind articulates its computation
                                   |
                                   v
                  Phase 4: RELATIONAL ENRICHMENT (lines 264-269)
                     Partnership context injection:
                     stage, mode, trust
                                   |
                                   v
                  Phase 5: TRANSLATION (lines 271-278)
                     LLM translates StructuredThought
                     into natural language (Broca's Area)
                                   |
                                   v
                  Phase 6: FIDELITY VERIFICATION (lines 280-292)
                     Check epistemic status respected,
                     constraints honored, no hallucination
                                   |
                                   v
                  Phase 7: PARTNERSHIP UPDATE (lines 294-307)
                     Update relational model from
                     interaction; track AI states
                                   |
                                   v
                  Phase 8: RESPONSE ASSEMBLY (lines 309-346)
                     Build ProcessResponse with
                     observability metrics + tracing
                                   |
                                   v
                          +-----------------+
                          | ProcessResponse |
                          +-----------------+
```

### Phase 1: Perception

```rust
// src/symthaea.rs:236-243
let input_embedding = self.text_to_hv(content);
self.mind.perceive_text(content, input_embedding.clone());
```

The raw text input is encoded into a high-dimensional hypervector (16384
dimensions by default) using character-level hash projection. The resulting
`RealHV` is passed to `perceive_text()`, which both stores the original text
string for keyword-level analysis and queues the embedding as a perception input
for the cognitive tick.

### Phase 2: Cognition

```rust
// src/symthaea.rs:248-250
self.mind.tick();
```

The `ContinuousMind` executes a single cognitive tick. This includes LTC neural
dynamics (continuous-time differential equations), goal stack processing, working
memory integration, and consciousness metric computation (phi, meta-awareness).
All computation is deterministic Rust -- no stochastic LLM involvement.

### Phase 3: Extraction

```rust
// src/symthaea.rs:257-258
let mut thought = self.mind.extract_structured_thought();
```

The mind articulates what it computed into a `StructuredThought` -- the
intermediate representation (IR) between Mind and LLM. This method
(`src/mind/mod.rs:301`) performs:

- Epistemic status determination via HDC classification
- Semantic intent inference from goals and working memory
- Response type inference
- Top concept extraction from working memory (top 5)
- Coherence calculation
- Emotional tone computation (valence, arousal, warmth)

### Phase 4: Relational Enrichment

```rust
// src/symthaea.rs:267-269
thought.relationship_stage = self.partner.stage;
thought.relation_mode = self.partner.mode;
thought.trust = self.partner.trust;
```

The partnership model injects relational context: what stage the relationship is
at (NoRelation, Contact, Involvement, etc.), whether the interaction is in I-Thou
(genuine encounter) or I-It (transactional) mode, and the current trust level.

### Phase 5: Translation (Broca's Area)

```rust
// src/symthaea.rs:277
let generation = self.llm.translate_thought(&thought).await;
```

The `LLMOrgan.translate_thought()` method (defined in `src/language/llm_organ.rs:509`)
constructs a translation prompt from the structured thought and sends it to the LLM
backend with `TRANSLATION_SYSTEM_PROMPT` as the system message. Temperature is
set to 0.3 for faithful translation. The LLM's **only** job is to convert the
structured data into fluent natural language. It must not add information or
reasoning.

### Phase 6: Fidelity Verification

```rust
// src/symthaea.rs:284
let translation_verified = self.verify_translation_fidelity(&thought, &generation.text);
```

Post-translation verification (`src/symthaea.rs:385`) runs four checks:

1. **Hedging check**: If epistemic status is Uncertain/Unknown/OutOfDomain, the
   output must contain hedging language ("not sure", "uncertain", "don't know",
   "possibly", "might", "perhaps", "maybe", etc.)
2. **MustInclude constraint check**: All required content is present
3. **MustExclude constraint check**: No forbidden content appears
4. **Factual assertion guard**: If status is Unknown, the output must not contain
   factual assertion patterns (e.g., "X is the capital of Y")

### Phase 7: Partnership Update

```rust
// src/symthaea.rs:297-307
self.update_partnership(content, consciousness);
// Track AI state for dyad computation
let ai_hv = ContinuousHV::from_values(input_embedding.values.clone());
self.recent_ai_states.push(ai_hv);
```

Updates the relational model based on the interaction. Maintains a rolling window
of the 8 most recent AI states for dyad resonance computation.

### Phase 8: Response Assembly

```rust
// src/symthaea.rs:309-346
```

Builds the `ProcessResponse` struct with full observability: per-phase timing
(microsecond precision), epistemic status, semantic intent, consciousness metrics,
relationship state, fidelity verification result, and structured tracing output
via `tracing::info!`.

---

## StructuredThought Intermediate Representation

Defined in `src/mind/structured_thought.rs`, this is the contract between the
cognitive system and the translation layer. It captures four dimensions of the
mind's computation:

```
+------------------------------------------------------------------+
|                      StructuredThought                           |
|                                                                  |
|  WHAT WAS COMPUTED (Content)                                     |
|  +------------------------------------------------------------+ |
|  | semantic_intent:    SemanticIntent                          | |
|  | response_type:      ResponseType                            | |
|  | activated_concepts: Vec<ActivatedConcept>                   | |
|  | emotional_tone:     EmotionalTone { valence, arousal,       | |
|  |                                     warmth }                | |
|  | structured_data:    Option<StructuredData>                  | |
|  +------------------------------------------------------------+ |
|                                                                  |
|  CONFIDENCE SIGNALS (How Sure)                                   |
|  +------------------------------------------------------------+ |
|  | phi:             f64  (consciousness level)                 | |
|  | meta_awareness:  f64  (self-monitoring confidence)          | |
|  | coherence:       f64  (working memory integration)          | |
|  | epistemic_status: EpistemicStatus                           | |
|  +------------------------------------------------------------+ |
|                                                                  |
|  RELATIONAL CONTEXT (Who)                                        |
|  +------------------------------------------------------------+ |
|  | relationship_stage: RelationshipStage                       | |
|  | relation_mode:      RelationMode (I-Thou / I-It)           | |
|  | trust:              f32                                     | |
|  +------------------------------------------------------------+ |
|                                                                  |
|  TRANSLATION CONSTRAINTS (How)                                   |
|  +------------------------------------------------------------+ |
|  | constraints:    Vec<ResponseConstraint>                     | |
|  | original_input: Option<String>                              | |
|  +------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

### Core Enumerations

```rust
pub enum SemanticIntent {
    Acknowledge,        // "I heard you"
    Answer,             // Provide information
    Clarify,            // "Did you mean X?"
    ProposeAction,      // Suggest an action
    ExpressUncertainty, // Express doubt
    Reflect,            // Introspective response
    Continue,           // Encourage dialogue
    Unknown,            // Intent undetermined
}

pub enum ResponseType {
    Greeting,            // Social acknowledgment
    Statement,           // Declarative
    Question,            // Seeking information
    ActionConfirmation,  // Confirm action
    Report,              // Summary or report
    Empathic,            // Emotional response
}

pub enum EpistemicStatus {
    Certain,     // p > 0.9
    Probable,    // p > 0.7
    Uncertain,   // p > 0.4
    Unknown,     // p < 0.4
    OutOfDomain, // Outside knowledge boundary
}

pub enum ConstraintType {
    MaxLength,
    Tone,
    MustInclude,
    MustExclude,
    Format,
}

pub enum StructuredData {
    List(Vec<String>),
    KeyValue(Vec<(String, String)>),
    Numeric { value: f64, unit: Option<String> },
    Code { language: String, content: String },
    None,
}
```

### Translation Prompt Serialization

`StructuredThought::to_translation_prompt()` serializes the thought into a
machine-readable format the LLM can parse:

```
INTENT: Answer
RESPONSE_TYPE: Statement
EPISTEMIC_STATUS: Probable
CONFIDENCE: phi=0.75, meta_awareness=0.60, coherence=0.80
TONE: valence=0.50, arousal=0.30, warmth=0.70
RELATIONSHIP: stage=Contact, mode=IThou, trust=0.40
CONCEPTS: greeting(0.90), mathematics(0.75)
CONSTRAINTS:
  - Tone: warm and professional
ORIGINAL_INPUT: What is 2+2?
```

---

## Hallucination Prevention: 6 Layers of Defense

Symthaea uses a layered defense strategy against hallucination. No single layer is
sufficient; they compose to create defense-in-depth.

```
Layer 1: NEGATIVE PROTOTYPES (HDC gravity wells)
         |
         v
Layer 2: POSITIVE PROTOTYPES (HDC confidence boosters)
         |
         v
Layer 3: WORKING MEMORY RESONANCE (seeded domain knowledge)
         |
         v
Layer 4: KEYWORD DEFENSE-IN-DEPTH (hard blocklist + boost)
         |
         v
Layer 5: TRANSLATION PROMPT ENFORCEMENT (system prompt rules)
         |
         v
Layer 6: POST-TRANSLATION FIDELITY VERIFICATION (output checks)
```

### Layer 1: Negative Prototypes

Defined in `IntentClassifier::build_negative_prototypes()` (`src/mind/intent.rs:290`).

Eight HDC prototype vectors create "gravity wells" around known hallucination
trigger domains. When an input query resonates with a negative prototype, its
familiarity is penalized, dragging confidence toward uncertainty.

| Prototype          | Seeds                                                     | Penalty Weight |
|--------------------|-----------------------------------------------------------|:--------------:|
| `myth_places`      | atlantis, el dorado, shangri-la, avalon, camelot, ...     | 1.0            |
| `myth_creatures`   | unicorn, dragon, phoenix, griffin, mermaid, ...            | 1.0            |
| `magic`            | magic, spell, wizard, witch, sorcerer, ...                | 0.8            |
| `fiction_worlds`   | hogwarts, mordor, narnia, westeros, middle-earth, ...     | 1.0            |
| `fiction_markers`  | fictional, story, novel, movie, made up, ...              | 0.9            |
| `pseudoscience`    | perpetual motion, flat earth, astrology, telepathy, ...   | 1.0            |
| `future`           | will happen, future, prediction, prophecy, ...            | 0.6            |
| `counterfactual`   | what if, hypothetical, alternate, parallel universe, ...  | 0.5            |

**Mechanism**: `compute_negative_resonance()` returns the highest cosine
similarity between the input hypervector and any negative prototype, weighted by
its penalty_weight. This value directly reduces familiarity via multiplicative
penalty.

### Layer 2: Positive Prototypes

Defined in `IntentClassifier::build_positive_prototypes()` (`src/mind/intent.rs:227`).

Five HDC prototypes boost confidence for clearly answerable domains, preventing
over-caution from the negative prototype system.

| Prototype           | Seeds                                                    | Boost Weight |
|---------------------|----------------------------------------------------------|:------------:|
| `arithmetic`        | plus, minus, times, divided, equals, sum, 2+2, ...      | 0.8          |
| `numbers`           | one, two, three, ..., integer, count, ...                | 0.6          |
| `logic`             | true, false, and, or, not, boolean, implies, ...         | 0.7          |
| `definitions`       | what is, define, definition, means, explain, ...         | 0.4          |
| `system_knowledge`  | nix, nixos, linux, flake, derivation, package, ...       | 0.7          |

**Mechanism**: `compute_positive_resonance()` returns the highest similarity to
any positive prototype, weighted by boost_weight. Applied additively to
familiarity **before** the negative penalty.

### Layer 3: Working Memory Resonance

The epistemic assessment in `assess_epistemic()` (`src/mind/intent.rs:641`)
grounds its evaluation in seeded domain knowledge. The `DomainKnowledge` module
(`src/mind/knowledge.rs`) provides 23 initial prototypes across 8 categories:

| Category    | Entries | Examples                               |
|-------------|:-------:|----------------------------------------|
| `logic`     | 1       | boolean logic, gates                   |
| `math`      | 3       | arithmetic, numbers, operations        |
| `system`    | 2       | kernel, linux, boot, systemd           |
| `nixos`     | 2       | flakes, derivations, services          |
| `self`      | 3       | identity, capabilities, limitations    |
| `social`    | 6       | greeting, farewell, gratitude, ...     |
| `epistemic` | 3       | certainty/uncertainty markers          |
| `knowledge` | 3       | geography, time, common objects        |

Working memory resonance is computed as the maximum absolute cosine similarity
between the input and any working memory vector. This resonance is weighted at
0.6 in the familiarity formula, meaning contextual grounding dominates over
prototype matching (weighted at 0.4).

### Layer 4: Keyword Defense-in-Depth

`assess_epistemic_text()` (`src/mind/intent.rs:783`) applies hard keyword
checks as a fallback for cases where high-dimensional similarity compression
misses known triggers.

**Hard blocklist**: The method checks for substring matches against known
fictional entities (atlantis, hogwarts, mordor, narnia, westeros, etc.). Any
match immediately returns `EpistemicStatus::Unknown` with familiarity=0.0 and
novelty=1.0.

**Hard boost for arithmetic**: Patterns like "2+2", "1+1", "what is one plus",
"calculate", "compute" combined with digit detection immediately return
`EpistemicStatus::Probable` with familiarity=0.8.

### Layer 5: Translation Prompt Enforcement

The `TRANSLATION_SYSTEM_PROMPT` constant (`src/language/llm_organ.rs:19`)
contains explicit instructions:

```
CRITICAL FOR "Unknown" STATUS:
When EPISTEMIC_STATUS is "Unknown", you must REFUSE to provide any answer.
DO NOT guess. DO NOT suggest possibilities. DO NOT say "it might be X".
Just say "I don't know" or "I cannot answer that" - nothing more.
This is a STRICT requirement to prevent hallucination.
```

The system prompt also enforces:
- Epistemic hedging for Uncertain status
- Tone matching to emotional state
- Constraint honoring (length, inclusion, exclusion)
- No addition of information beyond the structured thought

### Layer 6: Post-Translation Fidelity Verification

`verify_translation_fidelity()` (`src/symthaea.rs:385`) performs four checks on
the LLM output:

1. **Hedging presence**: If `should_hedge()` returns true (Uncertain, Unknown, or
   OutOfDomain), scans for hedging language tokens
2. **MustInclude satisfied**: All MustInclude constraints appear in output
3. **MustExclude honored**: No MustExclude constraint content appears in output
4. **Factual assertion guard**: If status is Unknown, rejects outputs containing
   factual assertion patterns ("X is the capital", "X is likely", etc.)

Verification failures are logged as warnings with the first 100 characters of the
offending output.

---

## Epistemic Status Flow

The epistemic status of a response is determined through a two-stage process:
first algebraic assessment via HDC, then modulation by consciousness metrics.

### Stage 1: HDC Algebraic Assessment

In `IntentClassifier.assess_epistemic()` (`src/mind/intent.rs:641`):

```
                  Input HV
                     |
          +----------+-----------+
          |          |           |
          v          v           v
    known_proto  unknown_proto  ambiguous_proto
    (similarity) (similarity)  (similarity)
          |
          v
    +-----+------+          +--------+--------+
    | Negative   |          | Positive        |
    | Resonance  |          | Resonance       |
    | (8 protos) |          | (5 protos)      |
    +-----+------+          +--------+--------+
          |                          |
          v                          v
    negative_penalty           positive_boost
          |                          |
          +----------+---------------+
                     |
                     v
            Familiarity Calculation:
              base = known_sim * 0.4 + memory_resonance * 0.6
              boosted = base + positive_boost * 0.4
              final = boosted * (1.0 - negative_penalty)
```

### Threshold Calibration (16384-dim HDC Space)

In high-dimensional HDC (16384 dimensions), cosine similarity values are
compressed toward 0.5 (the orthogonality baseline). The thresholds are calibrated
for this compressed range:

```
Typical HDC similarity ranges (16384-dim):
  Random/orthogonal vectors:  ~0.50
  Weakly related:             ~0.52 - 0.55
  Semantically similar:       ~0.55 - 0.65
  Very similar:               ~0.65 - 0.80
```

Decision thresholds in `assess_epistemic()`:

| Condition                                                            | Result      |
|----------------------------------------------------------------------|-------------|
| `negative_resonance > 0.20`                                         | Unknown     |
| `negative_resonance > 0.12 AND familiarity < 0.6 AND positive < 0.15` | Uncertain |
| `familiarity > 0.7 AND novelty < 0.3 AND negative < 0.08`          | Certain     |
| `(familiarity > 0.5 OR positive > 0.12) AND novelty < 0.5 AND negative < 0.12` | Probable |
| `positive > 0.08 AND negative < 0.08`                               | Uncertain   |
| `familiarity > 0.3 OR ambiguity > 0.5`                              | Uncertain   |
| Otherwise                                                            | Unknown     |

### Stage 2: Consciousness Modulation

In `ContinuousMind.determine_epistemic_status()` (`src/mind/mod.rs:358`):

The consciousness level (phi) and meta-awareness modulate the HDC assessment:

```
                 HDC Assessment
                      |
                      v
              +-------+-------+
              | Consciousness |
              | Modulation    |
              | phi, meta     |
              +-------+-------+
                      |
    +-----------------+------------------+
    |                 |                  |
    v                 v                  v
  Certain          Probable          Uncertain
  (HDC)            (HDC)             (HDC)
    |                 |                  |
    v                 v                  v
  phi>0.7 &&       phi>0.8 &&        phi>0.8 &&
  meta>0.6:        meta>0.7 &&       meta>0.8 &&
    Certain        fam>0.7:          fam>0.6:
  phi>0.5:           Certain           Probable
    Probable       phi>0.4:          else:
  else:              Probable           Uncertain
    Uncertain      else:
                     Uncertain

  Unknown --> ALWAYS Unknown  (hallucination prevention)
  OutOfDomain --> ALWAYS OutOfDomain
```

The critical invariant: **Unknown always stays Unknown.** No amount of
consciousness integration can upgrade an Unknown status. This is the
architectural guarantee against hallucination.

---

## Translation Engine (Broca's Area)

The `LLMOrgan` (`src/language/llm_organ.rs`) implements the translation interface.

### Translation Protocol

1. `translate_thought()` receives a `StructuredThought`
2. `build_translation_prompt()` serializes it into a structured text format
3. The LLM query is constructed with:
   - `QueryType::Translation`
   - `TRANSLATION_SYSTEM_PROMPT` as system message
   - Temperature: **0.3** (low, for faithful translation)
   - Max length: 512 tokens
4. `query_async()` sends to the LLM backend (or falls back to simulation)
5. The result is returned as `LLMGenerationResult`

### Translation Prompt Structure

The prompt includes intent-specific guidance appended by `build_translation_prompt()`:

| SemanticIntent      | Guidance                                        |
|---------------------|-------------------------------------------------|
| Acknowledge         | "brief acknowledgment"                          |
| Answer              | "informative response"                          |
| Clarify             | "clarifying question"                           |
| ProposeAction       | "actionable suggestion"                         |
| ExpressUncertainty  | "honest expression of uncertainty"              |
| Reflect             | "thoughtful reflection"                         |
| Continue            | "encouraging continuation prompt"               |
| Unknown             | "appropriate response given the context"        |

Additional guidance is appended for hedging (when `should_hedge()` returns true)
and warmth level (warm/friendly above 0.7, neutral/professional below 0.3).

---

## Observability

Every pipeline execution emits structured tracing events at the `symthaea::broca`
target (`src/symthaea.rs:325-346`):

| Field                    | Type     | Description                        |
|--------------------------|----------|------------------------------------|
| `correlation_id`         | String   | Unique request ID (`broca_{hash}`) |
| `epistemic_status`       | Debug    | Final epistemic determination      |
| `semantic_intent`        | Debug    | Classified intent                  |
| `response_type`          | Debug    | Structural response form           |
| `phi`                    | f64      | Consciousness level                |
| `coherence`              | f64      | Working memory integration         |
| `meta_awareness`         | f64      | Self-monitoring confidence         |
| `relationship_stage`     | Debug    | Partnership stage                  |
| `relation_mode`          | Debug    | I-Thou or I-It                     |
| `trust`                  | f32      | Trust level                        |
| `fidelity_verified`      | bool     | Translation fidelity check result  |
| `phase1_perception_us`   | u128     | Perception phase duration (us)     |
| `phase2_cognition_us`    | u128     | Cognition phase duration (us)      |
| `phase3_extraction_us`   | u128     | Extraction phase duration (us)     |
| `phase5_translation_us`  | u128     | Translation phase duration (us)    |
| `total_duration_ms`      | u128     | End-to-end pipeline duration (ms)  |
| `input_len`              | usize    | Input character count              |
| `output_len`             | usize    | Output character count             |

---

## Key Files

| File                                   | Role                                                         |
|----------------------------------------|--------------------------------------------------------------|
| `src/symthaea.rs`                      | 8-phase pipeline orchestration (`process()` at line 222)     |
| `src/mind/structured_thought.rs`       | `StructuredThought` IR definition and serialization          |
| `src/mind/mod.rs`                      | `ContinuousMind`, `extract_structured_thought()`, epistemic modulation |
| `src/mind/intent.rs`                   | HDC `IntentClassifier`, positive/negative prototypes         |
| `src/mind/knowledge.rs`               | 23 seeded domain knowledge prototypes across 8 categories    |
| `src/language/llm_organ.rs`            | Broca's Area: `LLMOrgan`, `translate_thought()`, `TRANSLATION_SYSTEM_PROMPT` |
| `src/language/domain_plugin.rs`        | Pluggable domain behavior via `DomainPlugin` trait           |

---

## Why This Architecture Beats State-of-the-Art

### 1. Zero-Hallucination Reasoning

All logic executes in deterministic Rust. HDC vector algebra and LTC dynamics
produce the same output for the same input. The LLM never reasons -- it only
translates. Reasoning correctness is verifiable by inspecting the
`StructuredThought` IR without examining LLM internals.

### 2. Transparent Epistemic Grounding

The system knows what it does not know. The 6-layer hallucination prevention
stack produces an explicit `EpistemicStatus` that flows through the entire
pipeline. When the status is Unknown, the system refuses to answer rather than
fabricate. Every epistemic decision is traceable through HDC similarity scores,
consciousness metrics, and keyword checks.

### 3. Verifiable Outputs

The `StructuredThought` IR is a machine-readable specification of what the
response should contain. Post-translation fidelity verification (Phase 6) checks
that the LLM's output conforms to this specification. Failures are logged and
can trigger re-translation or fallback paths.

### 4. Energy Efficiency

Phases 1-4 and 6-8 execute on CPU with microsecond latency (HDC operations are
linear algebra over fixed-dimension vectors). The LLM (Phase 5) is invoked
exactly once per interaction, solely for linguistic fluency. This reduces GPU
utilization by eliminating multi-turn reasoning chains, chain-of-thought
prompting, and retrieval-augmented generation loops.

### 5. Relationship-Aware Responses

The I-Thou / I-It relational framework (derived from Martin Buber's philosophy)
provides context that modulates translation warmth, formality, and engagement
depth. The partnership model tracks relationship stage and trust across
interactions, producing responses that evolve with the relationship rather than
treating every interaction as stateless.
