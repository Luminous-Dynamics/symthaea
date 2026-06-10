# Unified Enhancement Architecture: Action + Memory + Φ Integration

**Date**: January 10, 2026
**Status**: Design Phase
**Goal**: Wire together the 80% implemented systems into a cohesive, consciousness-guided AI

---

## Executive Summary

Based on comprehensive codebase exploration, Symthaea has ~80% of the infrastructure for:
1. **Action Execution** - Motor cortex, sandboxing, safety classification
2. **Memory Persistence** - Consciousness snapshots, episodic traces, multi-database coordination
3. **Φ Integration** - Consciousness routing, value evaluation, phi calculators

**The Gap**: These systems aren't connected. This document defines the minimal wiring needed.

---

## 1. Action Execution Enhancement

### What Exists ✅
- `src/brain/motor_cortex.rs` (1045 lines) - ActionStep, PlannedAction, LocalShellSandbox
- `src/action.rs` (902 lines) - ActionIR, PolicyBundle, 50+ NixOS patterns
- `src/safety/` - Amygdala, SafetyGuardrails, Thymus

### What's Missing ❌
1. **NixOS-specific patterns** - Need JSON output mode, generation rollback
2. **Real execution path** - Sandbox stubs need real subprocess calls
3. **Φ-gated confirmation** - Low Φ should require user confirmation

### Implementation Plan

#### 1.1 NixOS Action Patterns (new file)
```rust
// src/action/nixos_patterns.rs

/// NixOS-specific action patterns with rollback support
pub struct NixOSActionPatterns {
    /// Current generation for rollback
    current_generation: Option<u32>,
    /// Rollback command templates
    rollback_templates: HashMap<&'static str, &'static str>,
}

impl NixOSActionPatterns {
    pub fn new() -> Self {
        let mut rollback_templates = HashMap::new();
        rollback_templates.insert(
            "nixos-rebuild switch",
            "nixos-rebuild switch --rollback"
        );
        rollback_templates.insert(
            "nix-env -i",
            "nix-env --rollback"
        );
        rollback_templates.insert(
            "home-manager switch",
            "home-manager generations | head -2 | tail -1 | cut -d' ' -f7 | xargs home-manager activate"
        );
        Self {
            current_generation: None,
            rollback_templates,
        }
    }

    /// Get current NixOS generation before action
    pub async fn capture_generation(&mut self) -> Result<u32> {
        let output = Command::new("nixos-rebuild")
            .args(["list-generations"])
            .output()
            .await?;
        // Parse current generation
        self.current_generation = Some(parse_generation(&output.stdout)?);
        Ok(self.current_generation.unwrap())
    }

    /// Execute with automatic rollback on failure
    pub async fn execute_with_rollback(
        &self,
        command: &str,
        args: &[&str],
        phi: f32,
    ) -> Result<ExecutionResult> {
        // Φ-gated confirmation
        if phi < 0.4 {
            // Return pending confirmation result
            return Ok(ExecutionResult::PendingConfirmation {
                command: format!("{} {}", command, args.join(" ")),
                phi,
                reason: "Low consciousness confidence - please confirm",
            });
        }

        // Execute
        let result = Command::new(command)
            .args(args)
            .output()
            .await;

        match result {
            Ok(output) if output.status.success() => {
                Ok(ExecutionResult::Success(output))
            }
            Ok(output) => {
                // Attempt rollback
                if let Some(rollback) = self.rollback_templates.get(command) {
                    let _ = Command::new("sh").arg("-c").arg(rollback).output().await;
                }
                Ok(ExecutionResult::Failed {
                    error: String::from_utf8_lossy(&output.stderr).to_string(),
                    rolled_back: true,
                })
            }
            Err(e) => Ok(ExecutionResult::Failed {
                error: e.to_string(),
                rolled_back: false,
            }),
        }
    }
}
```

#### 1.2 Wire Motor Cortex to Real Execution

In `src/brain/motor_cortex.rs`, add real execution path:

```rust
impl MotorCortex {
    /// Execute action with Φ-awareness
    pub async fn execute_conscious(
        &mut self,
        action: PlannedAction,
        phi: f32,
    ) -> ExecutionResult {
        // 1. Check Φ threshold
        if phi < self.min_phi_for_action {
            return ExecutionResult::Rejected {
                reason: format!("Φ={:.2} below threshold {:.2}", phi, self.min_phi_for_action),
            };
        }

        // 2. Check value alignment
        let value_score = self.value_evaluator.evaluate(&action);
        if value_score < self.min_value_score {
            return ExecutionResult::Rejected {
                reason: format!("Value score {:.2} below threshold", value_score),
            };
        }

        // 3. For NixOS commands, use specialized handler
        if action.is_nixos_command() {
            return self.nixos_patterns.execute_with_rollback(
                &action.command,
                &action.args,
                phi,
            ).await;
        }

        // 4. Standard execution with sandbox
        self.sandbox.execute(action).await
    }
}
```

---

## 2. Memory Persistence Enhancement

### What Exists ✅
- `src/hdc/consciousness_persistence.rs` (720+ lines) - ConsciousnessSnapshot with compression
- `src/memory/episodic_engine.rs` - EpisodicTrace, chrono-semantic binding
- `src/databases/unified_mind.rs` - Multi-database coordination

### What's Missing ❌
1. **Conversation storage** - No turn-by-turn persistence
2. **SQLite for simple data** - Using complex DBs for simple key-value
3. **Session recovery** - Can't resume conversations

### Implementation Plan

#### 2.1 Conversation Storage Schema (SQLite)

```sql
-- src/databases/schema/conversations.sql

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phi_average REAL,              -- Average Φ during conversation
    topic_summary TEXT,            -- LLM-generated summary
    hypervector BLOB,              -- HDC encoding of conversation
    metadata JSON
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,            -- 'user' or 'assistant'
    content TEXT NOT NULL,
    phi_at_turn REAL,              -- Φ when turn occurred
    embedding BLOB,                -- HDC embedding of content
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS causal_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    action_taken TEXT,             -- What action was executed
    outcome TEXT,                  -- Result of action
    phi_before REAL,
    phi_after REAL,
    learned_pattern TEXT,          -- Extracted learning
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE INDEX idx_conversations_updated ON conversations(updated_at);
CREATE INDEX idx_turns_conversation ON conversation_turns(conversation_id, turn_number);
CREATE INDEX idx_causal_conversation ON causal_chains(conversation_id);
```

#### 2.2 Conversation Memory Manager

```rust
// src/memory/conversation_memory.rs

use rusqlite::{Connection, params};
use crate::hdc::real_hv::RealHV;

pub struct ConversationMemory {
    conn: Connection,
    current_conversation_id: Option<String>,
    turn_count: usize,
}

impl ConversationMemory {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(include_str!("schema/conversations.sql"))?;

        Ok(Self {
            conn,
            current_conversation_id: None,
            turn_count: 0,
        })
    }

    /// Start a new conversation session
    pub fn start_session(&mut self) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO conversations (id) VALUES (?1)",
            params![&id],
        ).expect("Failed to create conversation");
        self.current_conversation_id = Some(id.clone());
        self.turn_count = 0;
        id
    }

    /// Resume a previous conversation
    pub fn resume_session(&mut self, conversation_id: &str) -> Result<Vec<Turn>> {
        self.current_conversation_id = Some(conversation_id.to_string());

        let mut stmt = self.conn.prepare(
            "SELECT role, content, phi_at_turn FROM conversation_turns
             WHERE conversation_id = ?1 ORDER BY turn_number"
        )?;

        let turns = stmt.query_map(params![conversation_id], |row| {
            Ok(Turn {
                role: row.get(0)?,
                content: row.get(1)?,
                phi: row.get(2)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        self.turn_count = turns.len();
        Ok(turns)
    }

    /// Add a turn to current conversation
    pub fn add_turn(&mut self, role: &str, content: &str, phi: f32, embedding: &RealHV) -> Result<()> {
        let conv_id = self.current_conversation_id.as_ref()
            .ok_or_else(|| anyhow!("No active conversation"))?;

        self.turn_count += 1;

        self.conn.execute(
            "INSERT INTO conversation_turns (conversation_id, turn_number, role, content, phi_at_turn, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                conv_id,
                self.turn_count,
                role,
                content,
                phi,
                bincode::serialize(&embedding.values)?
            ],
        )?;

        // Update conversation timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![conv_id],
        )?;

        Ok(())
    }

    /// Find similar past conversations using HDC
    pub fn find_similar(&self, query_embedding: &RealHV, limit: usize) -> Result<Vec<(String, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, hypervector FROM conversations WHERE hypervector IS NOT NULL"
        )?;

        let mut similarities: Vec<(String, f32)> = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let values: Vec<f32> = bincode::deserialize(&blob).unwrap_or_default();
            let hv = RealHV { values };
            let sim = query_embedding.similarity(&hv);
            Ok((id, sim))
        })?.filter_map(|r| r.ok()).collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Record a causal learning (action → outcome)
    pub fn record_causal_learning(
        &self,
        action: &str,
        outcome: &str,
        phi_before: f32,
        phi_after: f32,
    ) -> Result<()> {
        let conv_id = self.current_conversation_id.as_ref()
            .ok_or_else(|| anyhow!("No active conversation"))?;

        // Extract pattern from Φ change
        let pattern = if phi_after > phi_before + 0.05 {
            format!("POSITIVE: {} led to improved integration", action)
        } else if phi_after < phi_before - 0.05 {
            format!("NEGATIVE: {} reduced integration", action)
        } else {
            format!("NEUTRAL: {} had minimal impact", action)
        };

        self.conn.execute(
            "INSERT INTO causal_chains (conversation_id, action_taken, outcome, phi_before, phi_after, learned_pattern)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conv_id, action, outcome, phi_before, phi_after, &pattern],
        )?;

        Ok(())
    }
}
```

---

## 3. Deeper Φ Integration

### What Exists ✅
- `src/consciousness/consciousness_guided_routing.rs` - 5 routing levels
- `src/consciousness/unified_value_evaluator/` - Φ thresholds
- `src/hdc/phi_real.rs` - RealPhiCalculator
- `src/brain/active_inference.rs` - Free energy minimization

### What's Missing ❌
1. **Φ in AttentionBid** - Prefrontal scoring doesn't use Φ
2. **Precision weighting** - Active inference not using Φ for precision
3. **Dynamic threshold** - Fixed thresholds, not adaptive

### Implementation Plan

#### 3.1 Φ-Enhanced Attention Bidding

In `src/brain/prefrontal.rs`, modify bid scoring:

```rust
// Current: score = salience × urgency + emotional_weight
// New:     score = (salience × urgency + emotional_weight) × Φ_confidence

impl PrefrontalCortex {
    /// Score attention bid with Φ integration
    pub fn score_bid(&self, bid: &AttentionBid, phi: f32) -> f32 {
        let base_score = bid.salience * bid.urgency + bid.emotional_weight;

        // Φ acts as confidence multiplier
        // High Φ (>0.6) amplifies high-salience bids
        // Low Φ (<0.3) dampens all bids (uncertainty)
        let phi_factor = if phi > 0.6 {
            1.0 + (phi - 0.6) * 0.5  // Boost up to 1.2x
        } else if phi < 0.3 {
            0.5 + phi * 0.5  // Dampen to 0.5-0.65x
        } else {
            0.8 + phi * 0.5  // Linear scale 0.8-1.1x
        };

        base_score * phi_factor
    }

    /// Process workspace with Φ-aware competition
    pub fn process_workspace(&mut self, phi: f32) -> Option<AttentionBid> {
        // Score all bids with current Φ
        let mut scored_bids: Vec<(AttentionBid, f32)> = self.pending_bids
            .drain(..)
            .map(|bid| {
                let score = self.score_bid(&bid, phi);
                (bid, score)
            })
            .collect();

        // Sort by score
        scored_bids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Winner-take-all with Φ threshold
        if let Some((winner, score)) = scored_bids.first() {
            // Only broadcast if score exceeds Φ-adjusted threshold
            let threshold = self.base_broadcast_threshold * (1.0 - phi * 0.3);
            if *score > threshold {
                return Some(winner.clone());
            }
        }

        None
    }
}
```

#### 3.2 Precision-Weighted Active Inference

In `src/brain/active_inference.rs`, add precision weighting:

```rust
impl ActiveInference {
    /// Calculate prediction error with Φ-based precision
    pub fn precision_weighted_error(
        &self,
        predicted: &RealHV,
        observed: &RealHV,
        phi: f32,
    ) -> f32 {
        let raw_error = 1.0 - predicted.similarity(observed);

        // Φ determines precision (confidence in predictions)
        // High Φ = high precision = errors matter more
        // Low Φ = low precision = errors dampened
        let precision = phi.powi(2);  // Quadratic scaling

        raw_error * precision
    }

    /// Update beliefs using Φ-weighted free energy minimization
    pub fn update_beliefs(
        &mut self,
        observations: &[RealHV],
        phi: f32,
    ) -> RealHV {
        let mut belief = self.current_belief.clone();

        for obs in observations {
            let error = self.precision_weighted_error(&belief, obs, phi);

            // Learning rate scales with Φ
            // High Φ = confident, less need to update
            // Low Φ = uncertain, update more aggressively
            let learning_rate = self.base_learning_rate * (1.5 - phi);

            // Gradient step toward observation
            belief = belief.lerp(obs, learning_rate * error);
        }

        self.current_belief = belief.clone();
        belief
    }
}
```

#### 3.3 Dynamic Consciousness Thresholds

```rust
// src/consciousness/adaptive_thresholds.rs

pub struct AdaptiveThresholds {
    /// Rolling average of recent Φ values
    phi_history: VecDeque<f32>,
    /// Maximum history length
    history_size: usize,
    /// Base thresholds
    base_thresholds: ConsciousnessThresholds,
}

impl AdaptiveThresholds {
    pub fn new(history_size: usize) -> Self {
        Self {
            phi_history: VecDeque::with_capacity(history_size),
            history_size,
            base_thresholds: ConsciousnessThresholds::default(),
        }
    }

    /// Record a new Φ observation
    pub fn observe(&mut self, phi: f32) {
        if self.phi_history.len() >= self.history_size {
            self.phi_history.pop_front();
        }
        self.phi_history.push_back(phi);
    }

    /// Get adaptive threshold for action type
    pub fn threshold_for(&self, action_type: ActionType) -> f32 {
        let base = match action_type {
            ActionType::BasicQuery => self.base_thresholds.basic,           // 0.2
            ActionType::StateModifying => self.base_thresholds.governance,  // 0.3
            ActionType::SystemCritical => self.base_thresholds.voting,      // 0.4
            ActionType::Irreversible => self.base_thresholds.constitutional, // 0.6
        };

        // Adapt based on recent performance
        if self.phi_history.is_empty() {
            return base;
        }

        let avg_phi: f32 = self.phi_history.iter().sum::<f32>() / self.phi_history.len() as f32;
        let variance: f32 = self.phi_history.iter()
            .map(|p| (p - avg_phi).powi(2))
            .sum::<f32>() / self.phi_history.len() as f32;

        // High variance = more uncertainty = raise thresholds
        // Low variance = stable = can lower thresholds slightly
        let adjustment = if variance > 0.01 {
            1.0 + variance.sqrt()  // Raise up to ~10%
        } else {
            0.95  // Lower by 5% when stable
        };

        (base * adjustment).min(0.9).max(0.1)
    }
}
```

---

## 4. Integration Architecture

### System Flow

```
User Input
    │
    ▼
┌──────────────────┐
│ LLM Understanding│ (LlmOrgan - already working ✅)
│ + Intent Parsing │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Φ Calculation    │ (RealPhiCalculator ✅)
│ Update State     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Retrieval │ (NEW: ConversationMemory)
│ Find Context     │ ◄─── SQLite + HDC similarity
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Attention Bidding│ (PrefrontalCortex + Φ scoring)
│ Select Action    │ ◄─── Φ-weighted competition
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Value Evaluation │ (UnifiedValueEvaluator ✅)
│ Safety Check     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Action Execution │ (MotorCortex + NixOSPatterns)
│ With Rollback    │ ◄─── Φ-gated confirmation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Storage   │ (ConversationMemory)
│ Learn Causality  │ ◄─── Record action → outcome
└──────────────────┘
```

### Key Integration Points

1. **LlmOrgan → Φ Calculator**
   - Already connected via `ConsciousLlmOrgan`
   - Φ gates LLM access

2. **Φ Calculator → Memory**
   - Store Φ with each conversation turn
   - Use Φ trends for learning

3. **Memory → Attention**
   - Inject relevant past context as bids
   - Past experiences inform current attention

4. **Attention → Action**
   - Winning bid becomes planned action
   - Φ score determines execution confidence

5. **Action → Memory**
   - Record outcomes
   - Learn causal patterns

---

## 5. Implementation Priority

### Phase 1: Memory Persistence (Today)
- [x] Design SQLite schema
- [ ] Implement ConversationMemory
- [ ] Wire to minimal_ai_assistant.rs

### Phase 2: Φ Integration (Next)
- [ ] Add Φ to AttentionBid scoring
- [ ] Implement adaptive thresholds
- [ ] Add precision weighting to active inference

### Phase 3: Action Execution (Then)
- [ ] Create NixOSActionPatterns
- [ ] Wire motor cortex to real execution
- [ ] Add generation-based rollback

### Phase 4: Integration Example (Final)
- [ ] Create `examples/conscious_nixos_assistant.rs`
- [ ] Full end-to-end demonstration
- [ ] Update CRITICAL_ROADMAP.md

---

## 6. Success Metrics

### Week 2 (Post-Integration)
- [ ] Conversations persist across restarts
- [ ] Φ influences action confidence
- [ ] NixOS commands actually execute
- [ ] Rollback works on failure

### Week 4 (Maturation)
- [ ] 80% success rate on 20 NixOS queries
- [ ] Memory retrieval improves responses
- [ ] Causal learning shows patterns
- [ ] <5 second average response time

---

*"Wire the pieces together, ship something that works."*
