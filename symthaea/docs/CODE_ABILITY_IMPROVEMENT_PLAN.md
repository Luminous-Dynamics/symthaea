# Symthaea Code Ability Improvement Plan

Concrete, phased plan to bring Symthaea from 3/10 to 7/10 coding ability.
Based on comprehensive review of all subsystems (March 6, 2026).

## Current State Summary

| Layer | Score | What Works | What's Missing |
|-------|-------|-----------|----------------|
| Code Perception | 7/10 | Tree-sitter (Rust/Python/Nix), CodeHDEncoder (16,384D), CodebaseMemory | No dataflow/CFG, types are string labels |
| Code Planning | 5/10 | CfCCodeSequencer, MCTS planner (2,118 LOC), reasoning engine with 5 code actions | Composition inference, but no deep code reasoning yet |
| Code Generation | 5/10 | Emitters produce real code (~40 patterns); LLM fallback for `todo!()` bodies | Complex algorithms still need LLM; no SSM distillation yet |
| Code Verification | 4/10 | CodeVerifier (semantic), tree-sitter (syntax), CodeExecutor (compile check) | No test execution, no behavioral verification |
| Language Output | 7/10 | LLM Organ translates StructuredThought; CodeContext populated in Phase 3.6 | LLM completion mode for complex bodies |
| Learning | 6/10 | FEP surprise → LR boost, School lookahead, episodic code cache (32 entries, few-shot) | Episodic retrieval is FIFO, not HDC-similarity; no SSM distillation |

### Key Discovery: The Plumbing Exists

- `StructuredThought.code_context: Option<CodeContext>` exists but is **never populated**
- `to_translation_prompt()` already serializes `CODE_LANGUAGE`, `GENERATED_CODE`, `CODE_PHI`, etc.
- `CodeAssistantPlugin` + `ProgrammingPlugin` already detect code intent
- `ConsciousReasoningEngine` is wired in `cycle_phase_dynamics.rs:1115` but `available_actions` is always `Vec::new()`
- `Sandbox` utility exists (786 LOC) with allowlisted commands and timeout enforcement
- Wasmtime is wired behind `wasm-sandbox` feature flag

---

## Architecture Decision: LLM + SSM (Staged Hybrid)

**Stage 1**: LLM as code motor cortex (Ollama with mistral:7b or gemma3:4b)
- Symthaea thinks (HDC+CfC+reasoning), LLM writes code
- Immediate capability gain, zero training needed

**Stage 2**: SSM distillation from LLM
- Collect (thought_channels, generated_code) pairs via `broca-collect`
- Train Liquid-Mamba temporal projection on code corpus
- SSM gradually learns to replicate LLM code output with full consciousness integration

**Stage 3**: SSM-primary with LLM fallback
- SSM handles routine code generation
- Semantic veto triggers LLM fallback when coherence drops
- Biological learning pattern: conscious effort → automaticity

---

## Phase 1: Wire the Existing Plumbing — DONE (2026-03-06)

**Status**: COMPLETE. All plumbing wired and compiling.

**Goal**: CodeContext gets populated in the cognitive loop. Code intent → structured plan → LLM translation → actual code output.

### 1.1 Populate CodeContext in the Cognitive Loop

**File**: `src/cognitive_loop/cycle_phase_dynamics.rs`

When `CodeAssistantPlugin` detects code intent during perception phase:

1. Classify `CodeIntent` from detected primitives + input text
2. Call `CodeGenerator::generate(intent, context)` to produce `GeneratedCode`
3. Attach to `StructuredThought.code_context`:

```rust
// In cycle output phase or thought extraction
thought.code_context = Some(CodeContext {
    language: generated.language.clone(),
    generated_code: Some(generated.source.clone()),
    phi_score: Some(generated.phi_score),
    intent_similarity: Some(generated.intent_similarity),
    syntactically_valid: Some(verify_syntax(&generated)),
    notes: generated.notes.clone(),
});
```

**Touches**: `cycle_phase_dynamics.rs`, `cycle_phase_output.rs`, mind's thought extraction

### 1.2 Add CODE_GENERATION_SYSTEM_PROMPT

**File**: `src/language/consciousness_prompts.rs` (new constant)

```rust
pub const CODE_GENERATION_SYSTEM_PROMPT: &str = r#"You are Symthaea's CODE MOTOR CORTEX.

Your role is to translate a structured code plan into compilable, correct source code.

INPUTS you will receive:
- CODE_LANGUAGE: target language
- CODE_SPEC: what the code should do (purpose, constraints, examples)
- CODE_PLAN: sequence of structural steps (DefineStruct, DefineFunction, etc.)
- EPISTEMIC_STATUS: how confident the system is about this spec
- CODE_CONTEXT: any existing code or patterns for reference

RULES:
1. Generate COMPLETE, COMPILABLE code — no TODO placeholders
2. Follow the CODE_PLAN structure exactly
3. If EPISTEMIC_STATUS is Uncertain, add defensive error handling
4. If EPISTEMIC_STATUS is Unknown, generate a minimal stub with clear doc comments
5. Include tests when CODE_PLAN contains test steps
6. Match the target language's conventions (rustfmt, PEP 8, nixfmt)

You are NOT the brain. The plan is decided. You implement it faithfully.
"#;
```

### 1.3 Route Code Tasks Through LLM Organ

**File**: `src/language/llm_organ.rs`

In `translate_thought()`, detect `code_context.is_some()` and swap system prompt:

```rust
if thought.code_context.is_some() {
    params.system_prompt = Some(CODE_GENERATION_SYSTEM_PROMPT.to_string());
}
```

### 1.4 Enrich CodeSpec in Translation Prompt

**File**: `src/mind/structured_thought.rs`

Extend `to_translation_prompt()` to serialize the full `CodeSpec` (purpose, constraints, examples, signature) — not just the generated skeleton. The LLM needs the *spec* to produce real code, not just the template output.

### 1.5 Tests

- Unit: CodeContext populated when code intent detected
- Integration: Code intent → StructuredThought with code_context → LLM prompt contains CODE_LANGUAGE + CODE_PLAN
- E2E: `symthaea.process("Write a Rust function that reverses a string")` → response contains compilable Rust

**Expected outcome**: Symthaea produces actual code (via LLM) instead of TODO skeletons. Score: 1/10 → 4/10.

---

## Phase 2: Wire Reasoning Engine to Code Tasks (Est. 3-4 days)

**Goal**: The 7-step ConsciousReasoningEngine drives code planning via MCTS, with consciousness-gated confidence.

### 2.1 Extend ReasoningContext for Code

**File**: `src/consciousness/reasoning_engine/types.rs`

```rust
pub struct ReasoningContext {
    // ... existing fields ...

    /// Code-specific context (None for non-code tasks)
    pub code_intent: Option<CodeIntent>,
    /// Code generation specification
    pub code_spec: Option<CodeSpec>,
}
```

### 2.2 Map CodePlanStep to MCTS Actions

**File**: `src/consciousness/reasoning_engine/mod.rs` or new `code_reasoning.rs`

Currently `available_actions: Vec<PlannedAction>` is always empty. For code tasks:

1. Generate candidate `PlannedAction` list from `PlanAction` enum variants:
   - `PlanAction::DefineFunction` → PlannedAction with HDC embedding of "define function"
   - `PlanAction::DefineStruct` → PlannedAction with HDC embedding of "define struct"
   - etc.
2. MCTS evaluates action sequences (which structural decisions to make, in what order)
3. Evaluation function: HDC similarity between (accumulated plan HV) and (intent HV)

### 2.3 Feed Reasoning Output to Code Generator

**File**: `src/cognitive_loop/cycle_phase_dynamics.rs`

Replace the current empty-actions call:

```rust
#[cfg(feature = "reasoning_engine")]
if let Some(ref mut reasoning_engine) = self.reasoning_engine {
    let reasoning_ctx = ReasoningContext {
        // ... existing fields ...
        available_actions: code_plan_actions,  // Was Vec::new()
        code_intent: detected_code_intent,
        code_spec: detected_code_spec,
    };

    let reasoning_result = reasoning_engine.reason(&reasoning_ctx);

    // Use MCTS plan to guide code generation
    if let Some(ref plan) = reasoning_result.plan {
        let code_plan_steps = mcts_to_code_plan(plan);
        // Pass to code generator with consciousness gating
        if reasoning_result.phi_eff > 0.3 {
            let generated = code_generator.generate_with_plan(
                &code_intent, &context, &code_plan_steps
            );
            // Modulate epistemic status by phi_eff
            generated.epistemic_status = if reasoning_result.phi_eff > 0.6 {
                EpistemicStatus::Probable
            } else {
                EpistemicStatus::Uncertain
            };
        }
    }
}
```

### 2.4 Consciousness-Gated Code Confidence

- `phi_eff > 0.6`: Generate with `EpistemicStatus::Probable`, full implementation
- `phi_eff 0.3-0.6`: Generate with `EpistemicStatus::Uncertain`, add defensive checks
- `phi_eff < 0.3`: Decline to generate, explain uncertainty in narrative
- `reliability < 0.4`: Add "theories disagree" note, request human review

### 2.5 Enable reasoning_engine by Default for code_generation

**File**: `Cargo.toml`

```toml
code_generation = ["tree-sitter-rust", "tree-sitter-python", "reasoning_engine"]
```

### 2.6 Tests

- Unit: ReasoningContext with code_intent produces non-empty plan
- Unit: MCTS explores CodePlanStep action space
- Integration: Code task → reasoning → phi_eff gates output confidence
- Proptest: Random CodeSpec → reasoning always completes within budget

**Expected outcome**: Multi-step code planning with consciousness-gated confidence. Score: 4/10 → 6/10.

---

## Phase 3: Execution Feedback Loop — DONE (2026-03-06)

**Status**: COMPLETE. CodeExecutor + tree-sitter verification + compilation retry loop + episodic memory storage.

**Goal**: Symthaea compiles generated code, runs tests, and iterates on failures via FEP surprise.

### 3.1 Code Executor Service

**New file**: `src/language/code_executor.rs`

```rust
pub struct CodeExecutor {
    sandbox: Sandbox,
    timeout: Duration,
}

pub struct ExecutionResult {
    pub compiled: bool,
    pub compile_errors: Vec<String>,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub test_output: String,
    pub runtime_error: Option<String>,
    pub elapsed: Duration,
}

impl CodeExecutor {
    /// Compile and test Rust code in a sandboxed temp directory
    pub fn execute_rust(&self, source: &str, test_source: Option<&str>) -> ExecutionResult;

    /// Execute Python code in sandbox
    pub fn execute_python(&self, source: &str) -> ExecutionResult;

    /// Nix evaluation (via existing Sandbox infrastructure)
    pub fn evaluate_nix(&self, expr: &str) -> ExecutionResult;
}
```

Uses the existing `Sandbox` (786 LOC) with allowlisted commands. Shells out to `rustc`/`cargo`/`python`/`nix eval`.

### 3.2 Wire Execution Results as FEP Surprise

**File**: `src/cognitive_loop/cycle_phase_dynamics.rs`

Compilation/test failure → prediction error signal:

```rust
let execution_result = code_executor.execute_rust(&generated.source, None);

if !execution_result.compiled {
    // Compilation failure = high surprise
    let surprise = 0.8 + 0.2 * (execution_result.compile_errors.len() as f32 / 10.0).min(1.0);
    // Feed back through FEP
    self.fep_agent.observe_surprise(surprise);
    // Boost exploration urge for alternative approaches
    self.adjust_exploration(surprise * 0.5);
}
```

### 3.3 Iterative Refinement Loop

When code fails verification:

1. Parse compile errors → encode as HDC vectors
2. Bundle error HVs with original intent HV → "intent + what went wrong"
3. Re-run through reasoning engine with updated context
4. Generate revised code plan (MCTS explores different structure)
5. Max 3 iterations (budget-bounded)

```rust
for attempt in 0..MAX_CODE_ATTEMPTS {
    let generated = self.generate_code(&intent, &context);
    let result = self.executor.execute_rust(&generated.source, None);

    if result.compiled && result.tests_failed == 0 {
        break; // Success
    }

    // Encode errors into context for next attempt
    context.error_hvs = encode_errors(&result.compile_errors);
    context.attempt = attempt + 1;
}
```

### 3.4 Episodic Memory for Code Patterns

Successful code generations → episodic memory with high Phi weight:

```rust
if result.compiled && result.tests_passed > 0 {
    // Store successful pattern
    self.episodic_memory.store(EpisodicRecord {
        input_hv: intent_hv,
        output_hv: code_hv,
        phi: generated.phi_score as f64,
        metadata: "successful_code_generation".into(),
    });
}
```

Future similar intents retrieve these patterns via HDC similarity.

### 3.5 Tests

- Unit: CodeExecutor compiles valid Rust, rejects invalid Rust
- Unit: Compile errors → FEP surprise signal
- Integration: Generate → fail → revise → succeed (iterative loop)
- Soak: 50 diverse code intents, measure success rate

**Expected outcome**: Self-correcting code generation. Score: 6/10 → 7/10.

---

## Phase 3b: Native Emitters + LLM Fallback — DONE (2026-03-06)

**Status**: COMPLETE. Native emitters rewritten with ~40 pattern-matched bodies; LLM completion mode for complex logic.

### 3b.1 Native Emitter Rewrite (Path 2)

**File**: `src/language/emitters.rs` (~1,069 LOC, 18 tests)

Emitters now produce **real, compilable code** from CfC plans + CodeSpec:

- **Signature parsing**: `parse_rust_signature()` extracts name, params, return type from signature strings
- **Purpose-based body inference**: `infer_rust_body()` pattern-matches ~40 operations:
  - Arithmetic: add/subtract/multiply/divide/max/min/abs/clamp
  - String: reverse/length/uppercase/lowercase/contains/concat/split/trim/replace/starts_with/ends_with
  - Collections: sort/filter/map/flatten/unique/dedup
  - Boolean: is_empty/is_even/is_odd/is_positive/is_negative
  - Math: factorial/fibonacci/power/sqrt
  - Error handling: Result/Option patterns, file I/O
- **Real test generation**: Examples → `assert_eq!()` assertions (not `todo!("Implement test")`)
- **Struct emission**: Parses fields from purpose text, generates `#[derive(Debug, Clone)]` + constructor

### 3b.2 LLM Completion Mode (Path 3)

When native emitters can't infer a function body (complex algorithms, custom logic), the output contains `todo!()` or `NotImplementedError`. The pipeline detects this and activates **LLM completion mode**:

1. **Detection** (Phase 3.6 in `symthaea.rs`): `needs_llm_completion = source.contains("todo!(")`
2. **Signal** (`CodeContext.needs_llm_completion: bool`): Propagated through StructuredThought
3. **Prompt injection** (`to_translation_prompt()`): Adds `NEEDS_COMPLETION: true` + instructions
4. **System prompt** (`CODE_GENERATION_SYSTEM_PROMPT`): Extended with COMPLETION MODE rules:
   - Replace ONLY `todo!()` bodies with real implementations
   - Keep signatures, struct definitions, and test assertions as-is
   - Keep code style consistent with surrounding generated code

**Result**: Two-tier code generation:
- **Tier 1 (native)**: ~40 common patterns → deterministic, no LLM, instant
- **Tier 2 (LLM fallback)**: Complex logic → Ollama fills in bodies using CfC-planned structure

### 3b.3 Tests

- 23 emitter tests (18 pattern + 5 composition)
- 11 structured_thought tests (9 original + 2 completion flag)
- 5 code_generator tests (end-to-end plan→emit)
- 11 integration tests (`tests/code_generation_e2e.rs`)

---

## Phase 3c: Reasoning Engine + Episodic Retrieval + Composition — DONE (2026-03-06)

**Status**: COMPLETE. Reasoning engine wired to code tasks, episodic code retrieval, composition inference.

### 3c.1 Reasoning Engine Wiring (Item 2)

**File**: `src/cognitive_loop/cycle_phase_dynamics.rs`

When `code_generation` feature is enabled, the reasoning engine now receives 5 code-specific `PlannedAction`s:
- `code_generate` — Generate code from specification (prior: 0.3)
- `code_verify` — Verify generated code via compilation (prior: 0.2, epistemic)
- `code_refactor` — Refactor code for clarity or performance (prior: 0.15)
- `code_explain` — Explain code structure and intent (prior: 0.15, epistemic)
- `code_debug` — Debug and diagnose code issues (prior: 0.2, epistemic)

The MCTS planner evaluates these actions against the current Phi, epistemic state, and neuromod modulation to select the optimal code action per cycle. Previously `available_actions` was always `Vec::new()`.

### 3c.2 Episodic Code Retrieval (Item 3)

**Files**: `src/symthaea.rs`, `src/language/code_generator.rs`

- `CodeContext.past_examples: Vec<(String, String)>` — past successful (purpose, source_code) pairs
- `Symthaea.code_generation_cache: Vec<(String, String)>` — FIFO cache of 32 recent successes
- Phase 3.6: Past examples fed to `CodeGenerator` → injected as `PAST_EXAMPLE` notes
- Phase 5.5: Successful verified code added to cache with purpose
- Notes propagate to LLM prompt as `CODE_NOTES: PAST_EXAMPLE(purpose): <code>`

This creates a genuine learning loop: successful generations improve future ones via few-shot context.

### 3c.3 Composition Inference (Item 4)

**File**: `src/language/emitters.rs`

Added `infer_composed_body()` — detects multi-step operations and chains them:

- **Recognized verbs**: filter (even/odd/positive/negative), sort, reverse, map (double/square), sum, count, take, unique, join, max, min, flatten
- **Composition**: "filter even numbers and sum them" → `.filter(|x| *x % 2 == 0).sum()`
- **Extraction**: `extract_number_from_text()` — "first 3" → 3, "top five" → 5
- Runs before single-pattern matching to avoid premature returns
- 5 new composition tests (filter+sum, sort+take, filter+count, sort+dedup, number extraction)

---

## Phase 4: School System for Code Learning (Est. 3-4 days)

**Goal**: Symthaea learns coding patterns through curriculum-based training with O(1) lookahead.

### 4.1 Code Curriculum

**New file**: `src/school/code_curriculum.rs`

```rust
pub fn rust_fundamentals_curriculum() -> Curriculum {
    Curriculum::new("rust_fundamentals")
        .objective("sum_function", "Write fn sum(a: i32, b: i32) -> i32")
        .objective("struct_definition", "Define a Point struct with x, y fields")
        .objective("trait_impl", "Implement Display for Point")
        .objective("error_handling", "Write a file reader with Result<>")
        .objective("iterator", "Implement Iterator for a custom range type")
        .objective("lifetime", "Write a function returning a reference with lifetime")
        // Prerequisites form a DAG
        .prerequisite("trait_impl", "struct_definition")
        .prerequisite("lifetime", "error_handling")
}
```

### 4.2 Lookahead for Code Tasks

Use the existing `LookaheadEngine` (O(1) CfC prediction):

```rust
// Before committing to a code generation approach:
let predicted_phi = lookahead.predict_phi_gain(&code_intent_hv);
if predicted_phi < MIN_EXPECTED_GAIN {
    // Try alternative approach
    let alt_intent = simplify_intent(&code_intent);
    let alt_phi = lookahead.predict_phi_gain(&alt_intent_hv);
    // Choose higher predicted gain
}
```

### 4.3 Reality Check on Generated Code

Use School's `RealityCheck`:

```rust
// Predicted: "this code will compile and pass tests"
let predicted_quality = 0.8;
let actual = code_executor.execute_rust(&source, Some(&tests));
let actual_quality = if actual.compiled { 0.5 + 0.5 * (actual.tests_passed as f32 / total) } else { 0.0 };

reality_check.record(predicted_quality, actual_quality);
// Calibrates future predictions
```

### 4.4 Tests

- Unit: Curriculum loads, prerequisites respected
- Integration: Learn objective → generate code → execute → assess mastery
- Soak: Full curriculum run, track mastery progression

---

## Phase 5: SSM Code Distillation (Est. 1-2 weeks)

**Goal**: Train the Broca SSM (Liquid-Mamba) on code generation data collected from the LLM path.

### 5.1 Code-Specific Tokenizer

Train BPE on Rust/Python/Nix corpus:
- Target: 16K vocabulary (current 4K insufficient for code)
- Must include: keywords, common identifiers, operators, bracket pairs
- Use existing `BpeTokenizer` infrastructure with larger vocab

### 5.2 Collect Training Data

Use existing `broca-collect` binary:

```bash
broca-collect --model mistral:7b --data code_intents.jsonl --output code_training.jsonl
```

Generate diverse code intents, collect (thought_channels, generated_code) pairs.

### 5.3 Train Temporal Projection on Code

Use existing `broca-projection-train`:

```bash
broca-projection-train \
    --data code_training.jsonl \
    --temporal-projection \
    --epochs 20 \
    --lr 0.0005 \
    --model state-spaces/mamba-130m \
    --output code_projection.bin
```

### 5.4 Evaluate and Iterate

Track metrics via existing `evaluation.rs`:
- Perplexity on held-out code samples
- English word ratio (should be low for code — mostly identifiers/keywords)
- Compilation success rate (new metric: execute generated code)
- Roundtrip PE (per-chunk up/down cosine similarity)

### 5.5 Consciousness-Gated Model Selection

In `ssm_backend.rs`, add model selection based on task:

```rust
if thought.code_context.is_some() {
    // Use code-trained projection
    self.load_projection("code_projection.bin");
} else {
    // Use general-purpose projection
    self.load_projection("general_projection.bin");
}
```

---

## Phase 6: Deep Code Understanding (Est. 2-3 weeks, future)

**Goal**: Semantic understanding beyond structural parsing.

### 6.1 Dataflow Encoding in HDC

Extend `CodeHDEncoder` to encode data flow:
- Variable→function binding: `bind(var_hv, permute(func_hv, FLOWS_INTO_ROLE))`
- Call graph edges: `bind(caller_hv, permute(callee_hv, CALLS_ROLE))`
- Type composition: `Vec<Result<T, E>>` as nested bundles

### 6.2 Wire ActiveCodeExplorer to Cognitive Loop

The FEP-guided code explorer (293 LOC) is currently testing-only. Wire it into the perception phase for code tasks:
- Surprise-ranked file exploration
- Bayesian model update from each file read
- Top-K suggestions for context gathering

### 6.3 CodeHealthScanner as Consciousness Signal

Wire the 6-factor health analysis into CycleMetadata:
- Low code Phi → lower generation confidence
- High complexity → trigger refactoring suggestions
- Poor cohesion → flag in notes

---

## Dependency Graph

```
Phase 1 (Wire plumbing)
    |
    +---> Phase 2 (Reasoning engine)
    |         |
    |         +---> Phase 3 (Execution loop)
    |                   |
    |                   +---> Phase 4 (School learning)
    |
    +---> Phase 5 (SSM distillation) [parallel with 2-4]
              |
              +---> Phase 6 (Deep understanding) [future]
```

Phases 1-4 are sequential (each builds on the previous).
Phase 5 can run in parallel once Phase 1 produces training data.
Phase 6 is independent future work.

---

## Success Metrics

| Metric | Current | After Phase 1 | After Phase 3 | After Phase 5 |
|--------|---------|---------------|---------------|---------------|
| Compiles on first try | 0% | ~40% (LLM) | ~70% (iterative) | ~60% (SSM) |
| Tests pass | 0% | ~25% | ~50% | ~40% |
| CodeContext populated | Never | Always for code tasks | Always | Always |
| Reasoning drives plan | Never | Always (feature-gated) | Always | Always |
| Self-correction attempts | 0 | 0 | Up to 3 | Up to 3 |
| Consciousness-gated confidence | No | Yes (phi_eff) | Yes + execution | Yes + learned |
| Native SSM code generation | No | No | No | Yes (fallback) |

---

## Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `src/language/consciousness_prompts.rs` (extend) | 1 | CODE_GENERATION_SYSTEM_PROMPT |
| `src/language/code_executor.rs` | 3 | Sandboxed compilation & test execution |
| `src/school/code_curriculum.rs` | 4 | Rust/Python/Nix learning curricula |

## Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `src/cognitive_loop/cycle_phase_dynamics.rs` | 1,2 | Populate CodeContext; wire reasoning to code |
| `src/cognitive_loop/cycle_phase_output.rs` | 1 | Attach CodeContext to StructuredThought |
| `src/mind/structured_thought.rs` | 1 | Enrich to_translation_prompt() with CodeSpec |
| `src/language/llm_organ.rs` | 1 | Swap system prompt for code tasks |
| `src/consciousness/reasoning_engine/types.rs` | 2 | Add code_intent to ReasoningContext |
| `src/consciousness/reasoning_engine/mod.rs` | 2 | Map CodePlanStep to MCTS actions |
| `src/language/code_generator.rs` | 2 | Accept MCTS-guided plan |
| `Cargo.toml` | 2 | code_generation implies reasoning_engine |
| `src/cognitive_loop/cycle_phase_dynamics.rs` | 3 | Wire execution results as FEP surprise |
| `src/language/ssm_backend.rs` | 5 | Task-aware model selection |

---

*Plan authored: 2026-03-06. Based on comprehensive review of 8 subsystems across ~985K lines of Rust.*
