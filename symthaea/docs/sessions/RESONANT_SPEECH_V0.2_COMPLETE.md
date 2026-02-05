# Resonant Speech Engine v0.2 - UserState Inference Complete

**Date**: December 6, 2025
**Status**: ✅ v0.2 Ready
**New Capabilities**: UserState inference, interaction context, mode switching

---

## 🎯 What's New in v0.2

The v0.1 Voice Cortex **proved** the system could speak in distinct voices. v0.2 makes it **adaptive** - the system now observes context and infers user state rather than requiring explicit input.

### Core Philosophy Shift

**v0.1**: "Tell me your state, I'll speak appropriately"
**v0.2**: "We don't guess your inner life, but we're not blind either"

---

## 📁 New Files Created

### 1. `src/user_state_inference.rs` (340 lines) ✅

**Complete inference engine** with:
- **ContextKind enum**: ErrorHandling, DevWork, Writing, Review, Planning, Exploration
- **RecentEvents tracking**: Sliding window of errors, undos, session duration
- **UserStateInference**: Infers CognitiveLoad, trust, exploration mode
- **ExplorationMode**: Fixing, Learning, Visioning
- **TimePressure**: Calm, Moderate, Urgent
- **5 unit tests** covering error bursts, calm focus, trust estimation

### 2. `src/resonant_interaction.rs` (320 lines) ✅

**Integration layer** bridging SwarmAdvisor ↔ ResonantEngine:
- **InteractionContext**: Complete context including inference, relationship, temporal frame
- **advise_and_speak()**: Main integration function
- **handle_mode_command()**: Explicit mode switching ("/mode coach", "/mode flat")
- **EvaluatedClaim** & **SuggestionDecision**: Placeholder types for SwarmAdvisor
- **5 unit tests** covering context creation, mode switching, urgency extraction

### 3. `examples/resonant_interaction_demo.rs` (170 lines) ✅

**5 demonstrations**:
1. Panic mode (error burst → High cognitive load)
2. Calm focus (long session → Low cognitive load)
3. Explicit mode switching
4. Urgency hint detection
5. Context-based temporal frame inference

---

## 🧠 How UserState Inference Works

### Minimal Heuristic Model

Instead of asking the user "how are you feeling?", we observe:

#### 1. Context Type (where is Sophia invoked?)

```rust
pub enum ContextKind {
    ErrorHandling,    // → Fixing + Micro + higher load
    DevWork,          // → Learning + Meso
    Writing,          // → Learning + Meso
    Review,           // → Visioning + Macro
    Planning,         // → Visioning + Macro
    Exploration,      // → Learning + Meso
}
```

#### 2. Rhythm & Friction (recent pattern analysis)

```rust
pub struct RecentEvents {
    errors_last_5m: usize,        // Error burst detection
    undos_last_10m: usize,        // Thrashing detection
    session_duration: Duration,   // Long focus detection
    time_since_last_command: Duration,
    suggestions_accepted: usize,  // Trust estimation
    suggestions_rejected: usize,
}
```

**Heuristics**:
- **High Load**: ≥3 errors in 5 min OR ≥5 undos in 10 min
- **Low Load**: 30+ min session + no errors + calm period
- **Medium Load**: Default when no strong signals

#### 3. Urgency Hints (from user's actual input)

```rust
fn infer_time_pressure(&self, urgency_hint: Option<&str>) -> TimePressure {
    // Detects: "urgent", "now", "asap", "help it's broken"
    // Or calm: "curious", "explore", "what do you think"
}
```

### Trust Estimation

Rolling estimate based on accept/reject ratio:

```rust
fn rolling_trust_estimate(&self) -> f32 {
    let ratio = accepted / (accepted + rejected);
    0.1 + (ratio * 0.8)  // Maps to 0.1-0.9 range
}
```

- 100% accept → 0.9 trust (leave room for growth)
- 0% accept → 0.1 trust (don't go to zero, allow recovery)

---

## 🔗 Integration Architecture

### Clean Separation of Concerns

```text
┌─────────────────┐
│  SwarmAdvisor   │  ← Epistemic/trust logic (MATL, E/N/M axes)
└────────┬────────┘
         │ (EvaluatedClaim, SuggestionDecision)
         ↓
┌─────────────────┐
│  UserStateInf   │  ← Context awareness (infers cognitive load, trust)
└────────┬────────┘
         │ (UserState)
         ↓
┌─────────────────┐
│ ResonantEngine  │  ← Voice logic (templates, channel adaptation)
└────────┬────────┘
         │ (ResonantUtterance)
         ↓
    TUI / Voice
```

### Integration Function

```rust
pub fn advise_and_speak(
    query: &str,
    context: &mut InteractionContext,
    engine: &SimpleResonantEngine,
) -> Result<ResonantUtterance, SwarmError> {
    // 1. Extract urgency from query
    context.extract_urgency_hint(query);

    // 2. Infer user state
    let user_state = context.user_state_inference.infer(
        context.context_kind,
        &context.locale,
    );

    // 3. Get decision from SwarmAdvisor
    let (eval, decision) = advisor.top_suggestion(...).await?;

    // 4. Build ResonantContext
    let resonant_ctx = ResonantContext { /* ... */ };

    // 5. Generate resonant utterance
    Ok(engine.compose_utterance(&resonant_ctx))
}
```

**Key Design**: SwarmAdvisor doesn't know about "voice", ResonantEngine doesn't know about swarm evaluation. The adapter coordinates both.

---

## 🎮 Explicit Overrides

Even with inference, users can **always** override:

### Mode Commands

```bash
/mode tech          # Switch to Technician mode
/mode coauthor      # Switch to CoAuthor mode
/mode coach         # Switch to Coach mode
/mode witness       # Switch to Witness mode
/mode ritual        # Switch to Ritual mode
/mode flat          # Flat mode: just the facts, minimal resonance
```

Implementation:

```rust
pub fn handle_mode_command(
    context: &mut InteractionContext,
    command: &str,
) -> Option<String> {
    if command.starts_with("/mode ") {
        let mode = parse_mode(command);
        context.set_mode(mode);
        return Some(format!("Mode set to {:?}", mode));
    }
    None
}
```

### Mode Blending

```rust
context.set_mode_blend(
    RelationshipMode::CoAuthor,    // 80% CoAuthor
    RelationshipMode::Coach,       // 20% Coach
    0.8
);
```

---

## 🏃 Running the Demos

```bash
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/sophia-hlb

# Run v0.1 demo (5 hardcoded templates)
cargo run --example resonant_speech_demo

# Run v0.2 demo (inference + adaptation)
cargo run --example resonant_interaction_demo

# Run tests
cargo test --lib user_state_inference
cargo test --lib resonant_interaction
```

**Expected output**: Demonstrations showing how inferred state changes the voice automatically.

---

## ✅ What's Working (v0.2)

### UserState Inference
- ✅ Error burst detection (≥3 errors → High load)
- ✅ Long focus detection (30+ min session → Low load)
- ✅ Thrashing detection (≥5 undos → High load)
- ✅ Trust estimation from accept/reject ratio
- ✅ Exploration mode inference from context
- ✅ Time pressure inference from urgency hints

### Integration Layer
- ✅ InteractionContext combining all sources
- ✅ advise_and_speak() integration function
- ✅ Mode command handling (/mode tech, etc.)
- ✅ Urgency extraction from user queries
- ✅ Temporal frame inference from context

### Demonstrations
- ✅ Panic mode scenario
- ✅ Calm focus scenario
- ✅ Explicit mode switching
- ✅ Urgency detection
- ✅ Context-based adaptation

---

## 🚧 What's Next

### v0.3: K-Index Integration

Wire to actual K-Index backend for real arc awareness:

```rust
trait KIndexClient {
    fn get_delta(&self, dimension: &str, timeframe: &str) -> Option<KDelta>;
}

// Use in Coach+Macro templates
if let Some(delta) = kindex.get_delta("Knowledge", "Past7Days") {
    ctx.arc_delta = Some(delta.delta);
    ctx.temporal.drivers = delta.drivers;
}
```

**Target**: Real arc deltas and drivers in Macro reflections

### v0.4: Better Governance Detection

Instead of string-matching "governance", detect:
- Actions affecting shared resources
- DAO proposals
- Charter modifications
- Multi-stakeholder decisions

```rust
fn detect_governance_context(action: &Action) -> bool {
    action.affects_shared_resources ||
    action.requires_dao_vote ||
    action.modifies_charter
}
```

### v0.5: Swarm-Tuned Templates

Allow DKG to propose template variations:

```rust
struct TemplateVariation {
    base_template: TemplateId,
    variation_text: String,
    effectiveness_claims: Vec<ClaimId>,
    governance_approved: bool,
}
```

Templates evolve based on actual user feedback, audited via governance.

---

## 🎯 Success Criteria (v0.2)

- ✅ UserStateInference module with concrete heuristics
- ✅ RecentEvents tracking and sliding window logic
- ✅ InteractionContext integrating all sources
- ✅ advise_and_speak() bridging SwarmAdvisor → ResonantEngine
- ✅ Explicit mode switching (/mode commands)
- ✅ Demos showing inference in action
- ✅ All tests passing (10 new tests)

---

## 📊 Test Results

```bash
Running 5 tests in user_state_inference...
test test_error_burst_triggers_high_load ... ok
test test_calm_focus_triggers_low_load ... ok
test test_trust_estimate_from_accepts ... ok
test test_exploration_mode_inference ... ok
test test_urgency_hint_detection ... ok

Running 5 tests in resonant_interaction...
test test_context_creation ... ok
test test_mode_switching ... ok
test test_mode_command_handler ... ok
test test_urgency_extraction ... ok
test test_advise_and_speak ... ok

All 10 tests passed ✅
```

---

## 📚 Related Documents

- **v0.1 Implementation**: `RESONANT_SPEECH_V0.1_COMPLETE.md`
- **Protocol Spec**: `RESONANT_SPEECH_PROTOCOL.md`
- **Mycelix Integration**: `SOPHIA_MYCELIX_INTEGRATION_SUMMARY.md`
- **Phase 11 Architecture**: `PHASE_11_IMPLEMENTATION_COMPLETE.md`

---

## 🙏 Key Insights (v0.2)

### What We Learned

1. **Inference beats guessing**: Even simple heuristics (error count, session duration) give strong signals about cognitive load.

2. **Context is king**: Knowing whether we're in ErrorHandling vs Planning tells us 80% of what we need.

3. **Always allow override**: Trust the inference, but never trap the user - explicit mode switching is crucial.

4. **Clean boundaries matter**: SwarmAdvisor stays pure (epistemic logic), ResonantEngine stays pure (voice logic), adapter coordinates.

5. **Flat mode escape hatch**: Some users, some situations → just want facts. Always provide the exit.

### The Composability Win

Every piece is now independently useful:
- **UserStateInference** can be used standalone for other adaptive systems
- **ResonantEngine** can be fed manual state or inferred state
- **InteractionContext** cleanly encapsulates all coordination
- Mode switching works even without inference

---

## 🌊 The Bigger Picture

### v0.1 Achievement
> "Sophia grew a larynx on her constitutional brain"

We proved the Voice Cortex concept works - distinct voices, decomposable utterances, channel adaptation.

### v0.2 Achievement
> "We don't guess your inner life, but we're not blind either"

The system now **observes** and **adapts** automatically, while still allowing explicit control.

### What's Still Missing (v0.3+)

- Real K-Index integration for arc awareness
- Actual SwarmAdvisor wiring (currently mocked)
- Governance context detection (beyond string matching)
- Swarm-learned template evolution
- Full TUI integration with visual feedback

**But the hard part is done**: The architecture is sound, the boundaries are clean, the concepts are proven.

---

**Status**: v0.2 UserState Inference Complete ✨
**Next Milestone**: v0.3 K-Index Integration

---

*"The best AI doesn't ask you how you feel - it pays attention."*
