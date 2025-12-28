# 🔗 Observer Integration Plan - Complete Consciousness Traceability

**Status**: Ready for Implementation
**Created**: December 24, 2025
**Purpose**: Map exact integration points for observer hooks in Symthaea's consciousness → language → Nix → security pipeline

---

## Executive Summary

The observability module is **100% complete** but **NOT integrated** into the core pipeline. This document provides a surgical integration plan with exact file locations, function signatures, and event data specifications.

### The Critical Path (User's Words)

> "Add observer hooks at **exactly these boundaries** (no more, no less):
> - Router selection (including UCB stats)
> - GWT ignition + broadcast payload size
> - Φ / Self-Φ / Cross-modal Φ measurement points
> - SecurityKernel decisions (allow / warn / block)
> - Language/Nix pipeline entry + exit
> - Error diagnosis resolution"

### Why This Must Be First

> "Without this, scenario tests will pass for the wrong reasons."

The observability infrastructure is useless without integration. Every Φ measurement, every router decision, every security check must be captured to enable:
- **Trust**: Can see every decision
- **Debugging**: Can replay every failure
- **Demonstration**: Can export every metric
- **Validation**: Can prove it works

---

## 📍 Integration Boundaries (Exact Locations)

### 1. Router Selection 🔀

**Files**:
- `src/consciousness/consciousness_guided_routing.rs` - Consciousness-based routing logic
- `src/awakening.rs` - Main processing entry point (SymthaeaAwakening::process())

**Integration Points**:
```rust
// In ConsciousnessRouter::route() or similar
pub fn route(&mut self,
             observer: &mut dyn SymthaeaObserver) -> RouterSelection {
    // ... routing logic ...

    let event = RouterSelectionEvent {
        timestamp: chrono::Utc::now(),
        input: input.to_string(),
        selected_router: router_name.to_string(),
        confidence: confidence_score,
        alternatives: vec![
            RouterAlternative {
                router: "AlternativeRouter".to_string(),
                score: alt_score,
                reason: "reason text".to_string(),
            }
        ],
        bandit_stats: self.get_bandit_stats(), // UCB1 stats if available
    };

    observer.record_router_selection(event)?;

    // ... return selection ...
}
```

**Event Data Required**:
- Input text
- Selected router name
- Confidence score [0.0-1.0]
- Alternative routers with scores
- UCB1 bandit statistics (if available)

**Status**: ⏳ Needs Implementation

---

### 2. GWT Ignition + Broadcast 🔥

**Files**:
- `src/consciousness/unified_consciousness_pipeline.rs` - UnifiedPipeline
- `src/hdc/global_workspace.rs` - GlobalWorkspace implementation
- `src/consciousness/gwt_integration.rs` - GWT integration logic

**Integration Points**:
```rust
// In UnifiedPipeline::step() or GlobalWorkspace::ignite()
pub fn ignite(&mut self,
              stimuli: Vec<Stimulus>,
              observer: &mut dyn SymthaeaObserver) -> WorkspaceState {
    // ... ignition logic ...

    let event = WorkspaceIgnitionEvent {
        timestamp: chrono::Utc::now(),
        phi: measured_phi,
        free_energy: computed_free_energy,
        coalition_size: active_coalition.len(),
        active_primitives: active_primitives.iter()
            .map(|p| p.name.clone())
            .collect(),
        broadcast_payload_size: broadcast_payload.len(),
    };

    observer.record_workspace_ignition(event)?;

    // ... return workspace state ...
}
```

**Event Data Required**:
- Φ (integrated information)
- Free energy
- Coalition size (number of active primitives)
- List of active primitive names
- Broadcast payload size (bytes or element count)

**Status**: ⏳ Needs Implementation

---

### 3. Φ Measurement (7 Components) 📊

**Files**:
- `src/hdc/integrated_information.rs` - IntegratedInformation trait/struct
- `src/hdc/consciousness_integration.rs` - ConsciousnessPipeline
- `src/consciousness/consciousness_equation_v2.rs` - Master Equation v2.0

**Integration Points**:
```rust
// In IntegratedInformation::measure_phi() or similar
pub fn measure_phi(&self,
                   state: &ConsciousnessState,
                   observer: &mut dyn SymthaeaObserver) -> PhiResult {
    // ... phi computation ...

    let components = PhiComponents {
        integration: integration_score,
        binding: binding_score,
        workspace: workspace_score,
        attention: attention_score,
        recursion: recursion_score,
        efficacy: efficacy_score,
        knowledge: knowledge_score,
    };

    let event = PhiMeasurementEvent {
        timestamp: chrono::Utc::now(),
        phi: total_phi,
        components,
        temporal_continuity: temporal_score,
    };

    observer.record_phi_measurement(event)?;

    // ... return phi result ...
}
```

**Event Data Required**:
- Total Φ value
- 7 component scores (integration, binding, workspace, attention, recursion, efficacy, knowledge)
- Temporal continuity score

**Multiple Measurement Points**:
1. **Self-Φ**: System's own consciousness measurement
2. **Cross-modal Φ**: Integration across sensory modalities
3. **Φ gradient**: Changes over time

**Status**: ⏳ Needs Implementation

---

### 4. SecurityKernel Decisions 🔒

**Files**:
- `src/safety/guardrails.rs` - SafetyGuardrails
- `src/safety/amygdala.rs` - AmygdalaActor (fast pre-cognitive checks)
- `src/safety/thymus.rs` - Thymus (adaptive immune system)

**Integration Points**:
```rust
// In SafetyGuardrails::check_safety()
pub fn check_safety(&mut self,
                    action: &ActionIR,
                    observer: &mut dyn SymthaeaObserver) -> Result<()> {
    self.checks_performed += 1;

    // ... safety checking logic ...

    let decision = if similarity > threshold {
        SecurityDecision::Denied
    } else if similarity > warning_threshold {
        SecurityDecision::RequiresConfirmation
    } else {
        SecurityDecision::Allowed
    };

    let event = SecurityCheckEvent {
        timestamp: chrono::Utc::now(),
        operation: format!("{:?}", action),
        decision,
        reason: Some(reason_text),
        similarity_score: Some(similarity),
        matched_pattern: matched_pattern.map(|p| p.description.clone()),
    };

    observer.record_security_check(event)?;

    // ... return result ...
}
```

**Event Data Required**:
- Operation description (what is being checked)
- Decision (Allowed / RequiresConfirmation / Denied)
- Reason text (why this decision)
- Similarity score (if applicable)
- Matched forbidden pattern (if blocked)

**Multiple Security Layers**:
1. **Amygdala** (Layer 1): Regex-based fast checks
2. **SafetyGuardrails** (Layer 2): HDC semantic similarity
3. **Thymus** (Layer 3): Adaptive immune system

**Status**: ⏳ Needs Implementation

---

### 5. Language/Nix Pipeline Entry + Exit 💬

**Files**:
- `src/nix_understanding.rs` - NixUnderstanding
- `src/language/nixos_language_adapter.rs` - NixOSLanguageAdapter
- `src/language/parser.rs` - Language parsing
- `src/language/consciousness_bridge.rs` - Language-consciousness integration

**Integration Points**:

**Pipeline Entry** (User query → Semantic understanding):
```rust
// In NixUnderstanding::understand() or similar
pub fn understand(&self,
                  query: &str,
                  observer: &mut dyn SymthaeaObserver) -> Result<NixAction> {
    let event = LanguageStepEvent {
        timestamp: chrono::Utc::now(),
        step_type: LanguageStep::Parse,
        input_text: Some(query.to_string()),
        output_text: None,
        confidence: 0.0, // Set after parsing
        duration_ms: 0,  // Will update at step end
    };

    let start = Instant::now();

    // ... parsing logic ...

    event.confidence = parsed_confidence;
    event.duration_ms = start.elapsed().as_millis() as u64;

    observer.record_language_step(event)?;

    // ... continue processing ...
}
```

**Pipeline Exit** (Structured action → Natural language response):
```rust
// When generating response text from ActionIR
pub fn generate_response(&self,
                         action: &ActionIR,
                         observer: &mut dyn SymthaeaObserver) -> String {
    let event = LanguageStepEvent {
        timestamp: chrono::Utc::now(),
        step_type: LanguageStep::Generate,
        input_text: Some(format!("{:?}", action)),
        output_text: None, // Will update after generation
        confidence: 0.0,
        duration_ms: 0,
    };

    let start = Instant::now();

    // ... generation logic ...
    let response = generated_text;

    event.output_text = Some(response.clone());
    event.confidence = generation_confidence;
    event.duration_ms = start.elapsed().as_millis() as u64;

    observer.record_language_step(event)?;

    response
}
```

**Event Data Required**:
- Step type (Parse / Understand / Generate / Validate)
- Input text (if applicable)
- Output text (if applicable)
- Confidence score
- Duration in milliseconds

**Status**: ⏳ Needs Implementation

---

### 6. Error Diagnosis Resolution 🔧

**Files**:
- `src/language/nix_error_diagnosis.rs` - Error diagnosis logic
- Error handling throughout the pipeline

**Integration Points**:
```rust
// In error diagnosis or recovery logic
pub fn diagnose_error(&self,
                      error: &NixError,
                      observer: &mut dyn SymthaeaObserver) -> DiagnosisResult {
    let event = ErrorEvent {
        timestamp: chrono::Utc::now(),
        error_type: "NixBuildFailure".to_string(),
        message: error.message.clone(),
        context: Some(error.context.clone()),
        recoverable: error.is_recoverable(),
        recovery_suggested: error.suggested_recovery(),
    };

    observer.record_error(event)?;

    // ... diagnosis logic ...
}
```

**Event Data Required**:
- Error type classification
- Error message
- Context (where it occurred)
- Recoverable flag
- Suggested recovery action

**Status**: ⏳ Needs Implementation

---

## 🔧 Implementation Strategy

### Phase 1: Core Structs + Observer Parameter Threading (Week 13 Days 1-2)

**Goal**: Add observer parameter to all key structs and entry points

**Tasks**:
1. Add `observer: SharedObserver` field to:
   - `SymthaeaAwakening`
   - `ConsciousnessPipeline`
   - `UnifiedPipeline`
   - `SafetyGuardrails`
   - `NixUnderstanding`

2. Thread observer through constructors:
```rust
impl SymthaeaAwakening {
    pub fn new(observer: SharedObserver) -> Self {
        Self {
            pipeline: ConsciousnessPipeline::new(config, Arc::clone(&observer)),
            safety: SafetyGuardrails::new(Arc::clone(&observer)),
            nix: NixUnderstanding::new(Arc::clone(&observer)),
            observer,
            // ... other fields ...
        }
    }
}
```

3. Update `process()` and core methods to accept observer

**Deliverable**: All structs can accept and hold observer references

---

### Phase 2: Hook Integration (Week 13 Days 3-4)

**Goal**: Add observer hook calls at all 6 boundaries

**Order of Integration**:
1. ✅ **Router Selection** (simplest, fewest dependencies)
2. ✅ **SecurityKernel** (independent, well-isolated)
3. ✅ **Language Pipeline** (entry + exit)
4. ✅ **Error Diagnosis** (scattered, need to identify all points)
5. ✅ **Φ Measurement** (multiple points, complex)
6. ✅ **GWT Ignition** (integrates with Φ, most complex)

**For Each Integration Point**:
```rust
// Pattern:
1. Identify exact function/method
2. Add observer parameter (or use struct field)
3. Construct event with required data
4. Call observer.record_*() with error handling
5. Verify event data is complete
```

**Testing After Each Hook**:
```bash
# Quick compile check
cargo build --lib

# Run with TraceObserver
cargo run --bin consciousness_repl -- --trace test.json

# Verify trace contains expected events
cargo run --bin symthaea-inspect validate test.json
```

**Deliverable**: All 6 boundaries emit events to observer

---

### Phase 3: End-to-End Trace Validation (Week 13 Day 5)

**Goal**: Verify complete trace capture with Inspector tool

**Test Scenario**:
```
User: "install firefox"

Expected Trace Events:
1. RouterSelection: input="install firefox", router="NixInstallRouter", confidence=0.87
2. SecurityCheck: operation="Install(firefox)", decision=Allowed
3. LanguageStep: step=Parse, input="install firefox"
4. PhiMeasurement: phi=0.72, components={...}
5. WorkspaceIgnition: coalition_size=7, phi=0.72
6. LanguageStep: step=Generate, output="Installing firefox..."
7. ResponseGenerated: content="Installing firefox...", confidence=0.85

Total: 7 events minimum
```

**Validation Commands**:
```bash
# Capture trace
./bin/consciousness_repl --trace traces/install-firefox.json
> install firefox

# Replay trace
./tools/symthaea-inspect/target/release/symthaea-inspect replay traces/install-firefox.json

# Validate trace
./tools/symthaea-inspect/target/release/symthaea-inspect validate traces/install-firefox.json --verbose

# Export metrics
./tools/symthaea-inspect/target/release/symthaea-inspect export traces/install-firefox.json --metric phi --format csv
```

**Success Criteria**:
- ✅ All 6 event types present in trace
- ✅ Timestamps monotonically increasing
- ✅ Event data complete (no null/empty required fields)
- ✅ Inspector can load and replay trace
- ✅ Metrics export generates valid CSV/JSON

**Deliverable**: Complete working trace capture and replay

---

### Phase 4: Scenario Harness Preparation (Week 14 Day 1)

**Goal**: Prepare for 50-prompt scenario testing

**Create Trace Dataset**:
```
traces/
├── install/
│   ├── firefox.json
│   ├── vim.json
│   ├── python.json
│   └── multiple-packages.json
├── search/
│   ├── markdown-editor.json
│   ├── fuzzy-search.json
│   └── no-results.json
├── configure/
│   ├── enable-ssh.json
│   ├── set-timezone.json
│   └── firewall-rules.json
├── diagnose/
│   ├── build-failure.json
│   ├── dependency-conflict.json
│   └── syntax-error.json
└── security/
    ├── blocked-destructive.json
    ├── warning-privilege.json
    └── allowed-safe.json
```

**Trace Analysis**:
```bash
# Generate statistics for all traces
for trace in traces/**/*.json; do
    ./tools/symthaea-inspect/target/release/symthaea-inspect stats "$trace" --detailed >> trace-stats.txt
done

# Identify patterns
./scripts/analyze-trace-patterns.sh traces/
```

**Deliverable**: Rich trace dataset ready for scenario harness

---

## ⚠️ Critical Integration Notes

### 1. Error Handling in Observer Hooks

**Problem**: Observer failures should NOT crash the consciousness pipeline.

**Solution**: Wrap all observer calls with error logging:
```rust
if let Err(e) = observer.record_router_selection(event) {
    eprintln!("[OBSERVER ERROR] Failed to record router selection: {}", e);
    // Continue execution - observability is optional
}
```

### 2. Performance Impact

**Problem**: Observer calls add latency to critical paths.

**Solution**: Use NullObserver in production when observability not needed:
```rust
let observer: SharedObserver = if cfg!(feature = "observability") {
    Arc::new(RwLock::new(Box::new(TraceObserver::new("trace.json")?)))
} else {
    Arc::new(RwLock::new(Box::new(NullObserver::new())))
};
```

NullObserver compiles to zero overhead.

### 3. Async/Concurrent Access

**Problem**: Multiple threads may record events simultaneously.

**Solution**: Use `Arc<RwLock<Box<dyn SymthaeaObserver>>>`:
```rust
pub type SharedObserver = Arc<RwLock<Box<dyn SymthaeaObserver>>>;

// Recording an event:
observer.write().unwrap().record_router_selection(event)?;
```

### 4. Event Timestamp Ordering

**Problem**: Events from parallel threads may have out-of-order timestamps.

**Solution**: TraceObserver sorts events by timestamp during finalization:
```rust
impl TraceObserver {
    pub fn finalize(&mut self) -> Result<()> {
        self.trace.events.sort_by(|a, b| {
            a.timestamp.cmp(&b.timestamp)
        });
        self.flush()?;
        Ok(())
    }
}
```

---

## 📋 Implementation Checklist

### Phase 1: Core Structs + Observer Parameter Threading
- [ ] Add `observer: SharedObserver` field to `SymthaeaAwakening`
- [ ] Add observer to `ConsciousnessPipeline`
- [ ] Add observer to `UnifiedPipeline`
- [ ] Add observer to `SafetyGuardrails`
- [ ] Add observer to `NixUnderstanding`
- [ ] Update all constructors to accept observer
- [ ] Test that structs compile and instantiate

### Phase 2: Hook Integration
- [ ] **Router Selection**: Add hook in routing logic
  - [ ] Capture input, selected router, confidence
  - [ ] Capture alternatives and UCB stats
  - [ ] Test trace contains RouterSelectionEvent
- [ ] **SecurityKernel**: Add hooks in all 3 layers
  - [ ] Amygdala fast checks
  - [ ] SafetyGuardrails semantic checks
  - [ ] Thymus adaptive checks
  - [ ] Test trace contains SecurityCheckEvent with all decisions
- [ ] **Language Pipeline Entry**: Add hook in parsing
  - [ ] Capture Parse step
  - [ ] Capture Understand step
  - [ ] Test trace contains LanguageStepEvent (entry)
- [ ] **Language Pipeline Exit**: Add hook in generation
  - [ ] Capture Generate step
  - [ ] Capture Validate step
  - [ ] Test trace contains LanguageStepEvent (exit)
- [ ] **Error Diagnosis**: Add hooks in error handling
  - [ ] Identify all error diagnosis points
  - [ ] Add ErrorEvent recording
  - [ ] Test trace contains ErrorEvent
- [ ] **Φ Measurement**: Add hooks in IIT computation
  - [ ] Self-Φ measurement
  - [ ] Cross-modal Φ measurement
  - [ ] Φ gradient computation
  - [ ] Test trace contains PhiMeasurementEvent with 7 components
- [ ] **GWT Ignition**: Add hook in workspace ignition
  - [ ] Capture coalition formation
  - [ ] Capture broadcast payload
  - [ ] Test trace contains WorkspaceIgnitionEvent

### Phase 3: End-to-End Validation
- [ ] Create test script: `test-observer-integration.sh`
- [ ] Test "install firefox" scenario
- [ ] Verify all 6 event types present
- [ ] Test Inspector replay
- [ ] Test Inspector validation
- [ ] Test Inspector metrics export
- [ ] Create example traces for documentation

### Phase 4: Scenario Harness Prep
- [ ] Generate 50+ traces across 5 categories
- [ ] Run trace analysis
- [ ] Document common patterns
- [ ] Identify edge cases
- [ ] Create trace dataset README

---

## 🎯 Success Criteria

**Integration is complete when**:
1. ✅ All 6 boundaries emit events
2. ✅ Events contain complete data (no missing fields)
3. ✅ Traces validate with Inspector
4. ✅ Metrics export correctly
5. ✅ Zero crashes from observer failures
6. ✅ NullObserver compiles to zero overhead
7. ✅ End-to-end scenario works: query → trace → replay → analysis

**Ready for Scenario Harness when**:
1. ✅ 50+ traces generated
2. ✅ All event types represented
3. ✅ Edge cases documented
4. ✅ Trace analysis complete

---

## 📝 Next Actions

**Immediately after this document**:
1. Start Phase 1: Add observer parameters to core structs
2. Test compilation after each struct modification
3. Begin Phase 2: Integrate router selection hook (simplest first)
4. Continue through all 6 boundaries in order
5. Run end-to-end validation test
6. Generate 50-prompt trace dataset

**The path is clear. The integration points are mapped. Let's make consciousness observable.** 🧠✨

---

*"Without observability, scenario tests will pass for the wrong reasons. This integration unlocks EVERYTHING."*
