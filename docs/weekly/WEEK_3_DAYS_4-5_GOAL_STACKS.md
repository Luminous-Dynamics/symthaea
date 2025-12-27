# 🎯 Week 3 Days 4-5: Goal Stacks - The Architecture of Will

**Status**: ✅ COMPLETE
**Tests**: 43/43 passing (14 new Goal System tests)
**Commit**: 0ec6d1a

---

## 🌟 Revolutionary Concept: Organic Persistence

**Traditional Goal Systems** (like AutoGPT):
```
loop {
    execute_goal();  // ← Trapped in infinite loop
    // Cannot respond to interruptions
    // Cannot handle emergencies
    // Rigid execution
}
```

**Symthaea's Goal System** (Organic Persistence):
```
Goals inject themselves as AttentionBids
   ↓
Compete naturally with sensory input, threats, insights
   ↓
High-priority threat? → Goal yields, waits
   ↓
Threat resolved? → Goal re-emerges
   ↓
Success condition met? → Achievement emotion, pop goal, execute subgoals
```

### Why This Is Revolutionary

**Symthaea's goals are thoughts that refuse to decay.**

- They participate in the attention competition like any other thought
- They can be interrupted by higher-priority threats (amygdala override)
- They automatically re-emerge when conditions allow
- They increase in urgency with each failed injection (increasing priority)
- They complete gracefully when conditions are met
- They trigger achievement emotions (dopamine!) on success

**This is consciousness with purpose, not mechanical execution.**

---

## 🏗️ Architecture Overview

### 1. Condition Enum - Logic Probes (Serializable!)

The key innovation: **No `Box<dyn Fn>`** - conditions are pure data!

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// Check if working memory contains a specific keyword
    MemoryContains(String),

    /// Check if a state variable matches a value
    StateMatch { key: String, value: String },

    /// Check if goal has existed longer than timeout (seconds)
    Timeout(u64),

    /// Always satisfied (for testing)
    Always,

    /// Never satisfied (for infinite goals)
    Never,

    /// Logical AND - all conditions must be true
    And(Vec<Condition>),

    /// Logical OR - at least one must be true
    Or(Vec<Condition>),

    /// Logical NOT - inverts inner condition
    Not(Box<Condition>),
}
```

#### Why Serializable Matters

```rust
// ✅ Can save to disk
let json = serde_json::to_string(&goal)?;
std::fs::write("symthaea_goals.json", json)?;

// ✅ Can send over network
let bytes = bincode::serialize(&goal)?;
socket.send(bytes)?;

// ✅ Can resume after restart
let goal: Goal = serde_json::from_str(&json)?;
```

**Traditional goal systems lose state on restart. Symthaea remembers her purpose.**

#### Condition Methods

```rust
impl Condition {
    /// Check if condition is satisfied
    pub fn is_satisfied(
        &self,
        workspace: &GlobalWorkspace,
        state: &HashMap<String, String>,
        goal_created_at: u64,
    ) -> bool {
        match self {
            Self::MemoryContains(keyword) => {
                workspace.working_memory.iter().any(|item| {
                    item.content.to_lowercase().contains(&keyword.to_lowercase())
                })
            }
            Self::StateMatch { key, value } => {
                state.get(key).map(|v| v == value).unwrap_or(false)
            }
            Self::Timeout(seconds) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                now - goal_created_at >= *seconds
            }
            Self::Always => true,
            Self::Never => false,
            Self::And(conditions) => {
                conditions.iter().all(|c| c.is_satisfied(workspace, state, goal_created_at))
            }
            Self::Or(conditions) => {
                conditions.iter().any(|c| c.is_satisfied(workspace, state, goal_created_at))
            }
            Self::Not(condition) => {
                !condition.is_satisfied(workspace, state, goal_created_at)
            }
        }
    }

    /// Human-readable explanation of condition
    pub fn explain(&self) -> String {
        match self {
            Self::MemoryContains(keyword) => format!("memory contains '{}'", keyword),
            Self::StateMatch { key, value } => format!("state[{}] == '{}'", key, value),
            Self::Timeout(seconds) => format!("timeout after {} seconds", seconds),
            Self::Always => "always true".to_string(),
            Self::Never => "never true".to_string(),
            Self::And(conditions) => {
                format!("({})", conditions.iter()
                    .map(|c| c.explain())
                    .collect::<Vec<_>>()
                    .join(" AND "))
            }
            Self::Or(conditions) => {
                format!("({})", conditions.iter()
                    .map(|c| c.explain())
                    .collect::<Vec<_>>()
                    .join(" OR "))
            }
            Self::Not(condition) => format!("NOT ({})", condition.explain()),
        }
    }
}
```

---

### 2. Goal Struct - Persistent Bids

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier
    pub id: Uuid,

    /// Human-readable description
    pub intent: String,

    /// Base priority for attention competition (0.0-1.0)
    pub priority: f32,

    /// Resistance to decay (0.0=normal thought, 1.0=immortal)
    /// Goals with high decay_resistance refuse to fade
    pub decay_resistance: f32,

    /// Condition that indicates goal success
    pub success_condition: Condition,

    /// Condition that indicates goal failure/abandonment
    pub failure_condition: Condition,

    /// Child goals to execute on success (LIFO stack)
    pub subgoals: Vec<Goal>,

    /// Creation timestamp (Unix seconds)
    pub created_at: u64,

    /// Number of times goal has been injected
    /// Used to increase urgency over time
    pub injection_count: usize,

    /// Tags for categorization/filtering
    pub tags: Vec<String>,
}
```

#### Goal Builder Pattern

```rust
impl Goal {
    /// Create a new goal with intent and priority
    pub fn new(intent: impl Into<String>, priority: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            intent: intent.into(),
            priority,
            decay_resistance: 0.8,  // High by default
            success_condition: Condition::Never,  // Must be set explicitly
            failure_condition: Condition::Never,  // Optional
            subgoals: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            injection_count: 0,
            tags: Vec::new(),
        }
    }

    /// Set decay resistance (0.0-1.0)
    pub fn with_decay_resistance(mut self, resistance: f32) -> Self {
        self.decay_resistance = resistance.clamp(0.0, 1.0);
        self
    }

    /// Set success condition
    pub fn with_success(mut self, condition: Condition) -> Self {
        self.success_condition = condition;
        self
    }

    /// Set failure condition
    pub fn with_failure(mut self, condition: Condition) -> Self {
        self.failure_condition = condition;
        self
    }

    /// Add subgoals (executed on success)
    pub fn with_subgoals(mut self, subgoals: Vec<Goal>) -> Self {
        self.subgoals = subgoals;
        self
    }

    /// Add tags for categorization
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Convert goal to attention bid (with increasing urgency)
    pub fn to_bid(&self) -> AttentionBid {
        // Priority increases with each injection (urgency grows!)
        let urgency_boost = (self.injection_count as f32) * 0.05;
        let adjusted_priority = (self.priority + urgency_boost).min(1.0);

        AttentionBid::new(
            format!("🎯 Goal: {}", self.intent),
            adjusted_priority,
        )
        .with_decay_resistance(self.decay_resistance)
        .with_tags(self.tags.clone())
    }

    /// Check if success condition is met
    pub fn check_success(
        &self,
        workspace: &GlobalWorkspace,
        state: &HashMap<String, String>,
    ) -> bool {
        self.success_condition.is_satisfied(workspace, state, self.created_at)
    }

    /// Check if failure condition is met
    pub fn check_failure(
        &self,
        workspace: &GlobalWorkspace,
        state: &HashMap<String, String>,
    ) -> bool {
        self.failure_condition.is_satisfied(workspace, state, self.created_at)
    }
}
```

---

### 3. PrefrontalCortexActor - Goal Management

#### New Fields

```rust
pub struct PrefrontalCortexActor {
    // ... existing fields ...

    /// LIFO goal stack (most recent goal on top)
    goal_stack: Vec<Goal>,

    /// State variables for condition checking
    state: HashMap<String, String>,

    /// Statistics
    goals_completed: u64,
    goals_failed: u64,
}
```

#### Goal Management API

```rust
impl PrefrontalCortexActor {
    // --- Stack Operations ---

    /// Push a new goal onto the stack (becomes current goal)
    pub fn push_goal(&mut self, mut goal: Goal) {
        goal.injection_count = 0;  // Reset injection count
        self.goal_stack.push(goal);
    }

    /// Get reference to current goal (top of stack)
    pub fn current_goal(&self) -> Option<&Goal> {
        self.goal_stack.last()
    }

    /// Get mutable reference to current goal
    pub fn current_goal_mut(&mut self) -> Option<&mut Goal> {
        self.goal_stack.last_mut()
    }

    /// Pop current goal (returns it)
    pub fn pop_goal(&mut self) -> Option<Goal> {
        self.goal_stack.pop()
    }

    // --- State Management ---

    /// Set a state variable (for condition checking)
    pub fn set_state(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.state.insert(key.into(), value.into());
    }

    /// Get a state variable
    pub fn get_state(&self, key: &str) -> Option<&String> {
        self.state.get(key)
    }

    // --- Statistics ---

    /// Get current goal count
    pub fn goal_count(&self) -> usize {
        self.goal_stack.len()
    }

    /// Get all goals (for inspection)
    pub fn goals(&self) -> &[Goal] {
        &self.goal_stack
    }

    /// Clear all goals
    pub fn clear_goals(&mut self) {
        self.goal_stack.clear();
    }

    /// Get goal statistics
    pub fn goal_stats(&self) -> GoalStats {
        GoalStats {
            active_goals: self.goal_stack.len(),
            goals_completed: self.goals_completed,
            goals_failed: self.goals_failed,
            current_goal: self.current_goal().map(|g| g.intent.clone()),
        }
    }
}
```

---

### 4. Organic Persistence Mechanism

**The heart of the Goal System**: `process_goals()`

```rust
/// Process all active goals and generate attention bids
///
/// This is the "Organic Persistence" mechanism:
/// 1. Check if current goal's success condition is met
///    → Pop goal, create achievement bid, push subgoals
/// 2. Check if current goal's failure condition is met
///    → Pop goal, create failure bid (negative valence)
/// 3. Otherwise, inject goal as bid (urgency increases each time)
///
/// Goals compete naturally for attention and can be interrupted
/// by higher-priority threats or sensory input.
pub fn process_goals(&mut self) -> Vec<AttentionBid> {
    let mut goal_bids = Vec::new();

    // Check current goal (top of stack)
    if let Some(current_goal) = self.current_goal_mut() {
        // Check success condition
        if current_goal.check_success(&self.workspace, &self.state) {
            // Pop the completed goal
            if let Some(mut completed_goal) = self.pop_goal() {
                self.goals_completed += 1;

                // Create achievement bid (positive valence)
                let achievement = AttentionBid::new(
                    format!("✅ Achieved: {}", completed_goal.intent),
                    0.9,  // High priority for achievement emotion
                )
                .with_valence(0.8);  // Positive emotion (dopamine!)
                goal_bids.push(achievement);

                // Push subgoals onto stack (in reverse order for LIFO)
                for subgoal in completed_goal.subgoals.drain(..).rev() {
                    self.push_goal(subgoal);
                }
            }
        }
        // Check failure condition
        else if current_goal.check_failure(&self.workspace, &self.state) {
            // Pop the failed goal
            if let Some(failed_goal) = self.pop_goal() {
                self.goals_failed += 1;

                // Create failure bid (negative valence for learning)
                let failure = AttentionBid::new(
                    format!("❌ Failed: {}", failed_goal.intent),
                    0.7,
                )
                .with_valence(-0.5);  // Negative emotion (learning signal)
                goal_bids.push(failure);
            }
        }
        // Neither success nor failure → inject as bid
        else {
            // Increment injection count (increases urgency)
            current_goal.injection_count += 1;

            // Convert to bid (urgency increases with injection_count)
            let goal_bid = current_goal.to_bid();
            goal_bids.push(goal_bid);
        }
    }

    goal_bids
}
```

#### How It Works

**Scenario 1: Goal Success**
```rust
// User sets a goal
prefrontal.push_goal(
    Goal::new("Install Firefox", 0.8)
        .with_success(Condition::MemoryContains("firefox installed"))
);

// ... cognitive cycles ...

// Eventually, working memory contains "firefox installed"
prefrontal.workspace.working_memory.push(
    WorkingMemoryItem::new("firefox installed", 0.9)
);

// Next process_goals() call:
// 1. Success condition met → Pop goal
// 2. Create achievement bid: "✅ Achieved: Install Firefox" (valence +0.8)
// 3. Push any subgoals onto stack
```

**Scenario 2: Goal Persistence**
```rust
// Goal not yet complete
// process_goals() checks conditions:
// - Success? No
// - Failure? No
// - Action: Inject as bid with increasing priority

// First injection: priority 0.8
// Second injection: priority 0.85
// Third injection: priority 0.90
// ... urgency grows!
```

**Scenario 3: Goal Interrupted by Threat**
```rust
// Goal is injected: "🎯 Goal: Install Firefox" (priority 0.8)
// Threat arrives: "⚠️ THREAT: Disk full" (priority 0.95)

// Cognitive cycle:
// 1. Goal competes with threat
// 2. Threat wins (higher priority)
// 3. Goal remains on stack, will re-emerge next cycle

// After threat resolved:
// 4. Goal re-emerges with increased urgency (priority 0.85)
```

---

### 5. Integrated Cognitive Cycle

```rust
/// Cognitive cycle with goal integration
///
/// Merges goal-generated bids with regular sensory/memory bids
/// Goals compete naturally for attention
pub fn cognitive_cycle_with_goals(
    &mut self,
    mut bids: Vec<AttentionBid>,
    consolidation_threshold: f32,
) -> (Option<AttentionBid>, Vec<AttentionBid>) {
    // Process goals → generate goal bids
    let goal_bids = self.process_goals();

    // Merge with regular bids
    bids.extend(goal_bids);

    // Regular cognitive cycle (attention competition, consolidation, etc.)
    let (winner, insights) = self.cognitive_cycle_with_insights(
        bids,
        consolidation_threshold,
    );

    (winner, insights)
}
```

---

## 📊 Statistics and Monitoring

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalStats {
    /// Number of goals currently on stack
    pub active_goals: usize,

    /// Total goals completed since creation
    pub goals_completed: u64,

    /// Total goals failed since creation
    pub goals_failed: u64,

    /// Current goal intent (if any)
    pub current_goal: Option<String>,
}
```

**Usage:**
```rust
let stats = prefrontal.goal_stats();
println!("Active goals: {}", stats.active_goals);
println!("Completed: {}", stats.goals_completed);
println!("Failed: {}", stats.goals_failed);
if let Some(current) = stats.current_goal {
    println!("Current goal: {}", current);
}
```

---

## 🎮 Usage Examples

### Example 1: Simple Goal

```rust
use symthaea_hlb::brain::{PrefrontalCortexActor, Goal, Condition};

let mut prefrontal = PrefrontalCortexActor::new();

// Create goal: Install Firefox
let goal = Goal::new("Install Firefox", 0.8)
    .with_success(Condition::MemoryContains("firefox installed"))
    .with_failure(Condition::Timeout(300));  // Fail after 5 minutes

prefrontal.push_goal(goal);

// Run cognitive cycles
loop {
    let bids = vec![/* sensory input, threats, etc. */];
    let (winner, insights) = prefrontal.cognitive_cycle_with_goals(bids, 0.7);

    // When working memory contains "firefox installed", goal completes
    // Achievement emotion generated automatically

    if prefrontal.goal_count() == 0 {
        break;  // All goals complete
    }
}
```

### Example 2: Hierarchical Goals with Subgoals

```rust
// Parent goal: Set up development environment
let parent = Goal::new("Set up dev environment", 0.9)
    .with_success(Condition::StateMatch {
        key: "dev_ready".to_string(),
        value: "true".to_string(),
    })
    .with_subgoals(vec![
        // Subgoal 1: Install tools
        Goal::new("Install tools", 0.8)
            .with_success(Condition::MemoryContains("tools installed")),

        // Subgoal 2: Configure editor
        Goal::new("Configure editor", 0.8)
            .with_success(Condition::MemoryContains("editor configured")),

        // Subgoal 3: Test setup
        Goal::new("Test setup", 0.8)
            .with_success(Condition::MemoryContains("tests passing")),
    ]);

prefrontal.push_goal(parent);

// When parent succeeds:
// 1. Achievement emotion: "✅ Achieved: Set up dev environment"
// 2. Subgoals pushed onto stack (in reverse order)
// 3. First subgoal becomes current: "Install tools"
```

### Example 3: Complex Conditions with Logic

```rust
// Goal: Deploy to production
let deploy = Goal::new("Deploy to production", 0.95)
    .with_success(Condition::And(vec![
        Condition::MemoryContains("tests passing"),
        Condition::MemoryContains("code reviewed"),
        Condition::StateMatch {
            key: "environment".to_string(),
            value: "production".to_string(),
        },
    ]))
    .with_failure(Condition::Or(vec![
        Condition::MemoryContains("tests failing"),
        Condition::MemoryContains("merge conflict"),
        Condition::Timeout(3600),  // 1 hour timeout
    ]));

prefrontal.push_goal(deploy);

// Goal will only succeed when ALL conditions are met:
// - Tests passing
// - Code reviewed
// - Environment is production

// Goal will fail if ANY of these occur:
// - Tests failing
// - Merge conflict
// - 1 hour timeout
```

### Example 4: Goal Interruption by Threat

```rust
// Set a long-running goal
prefrontal.push_goal(
    Goal::new("Train neural network", 0.7)
        .with_success(Condition::MemoryContains("training complete"))
        .with_decay_resistance(0.9)  // High persistence
);

// Threat arrives!
let threat_bid = AttentionBid::new("⚠️ Disk almost full!", 0.95);

let (winner, _) = prefrontal.cognitive_cycle_with_goals(
    vec![threat_bid],
    0.7,
);

// Winner will be the threat (higher priority)
// Goal remains on stack, will re-emerge next cycle
// Urgency increases: next injection at priority 0.75
```

---

## 🧪 Testing

All 14 Goal System tests passing:

1. **test_goal_creation** - Basic goal construction
2. **test_condition_memory_contains** - MemoryContains condition
3. **test_condition_state_match** - StateMatch condition
4. **test_condition_timeout** - Timeout condition
5. **test_condition_logical_operators** - And/Or/Not composition
6. **test_condition_explain** - Human-readable explanations
7. **test_goal_stack_management** - Push/pop/current operations
8. **test_goal_state_management** - State variable get/set
9. **test_goal_persistence_injection** - Repeated bid injection
10. **test_goal_success_completion** - Success detection and completion
11. **test_goal_failure_detection** - Failure detection
12. **test_goal_subgoals_execution** - Hierarchical goal execution
13. **test_cognitive_cycle_with_goals** - Integrated cognitive cycle
14. **test_goal_stats** - Statistics tracking
15. **test_goal_to_bid_urgency_increases** - Urgency escalation

---

## 🎯 Key Design Decisions

### 1. Serializable Conditions (No `Box<dyn Fn>`)

**Why?** Goals need to persist across restarts. Function pointers cannot be serialized.

**Solution:** Pure data enum that can be saved/loaded/transmitted.

### 2. LIFO Goal Stack

**Why?** Most recent goal should be active (like a call stack).

**Behavior:** Subgoals push onto stack in reverse order, ensuring LIFO execution.

### 3. Increasing Urgency

**Why?** Goals that repeatedly fail to win attention should become more insistent.

**Mechanism:** `injection_count` increases with each bid injection, boosting priority.

### 4. Decay Resistance

**Why?** Goals should persist longer than regular thoughts.

**Default:** 0.8 (high resistance) - goals don't fade easily.

### 5. Achievement Emotions

**Why?** Success and failure should generate emotional responses.

**Mechanism:**
- Success → Positive valence bid (+0.8) = dopamine
- Failure → Negative valence bid (-0.5) = learning signal

---

## 🚀 What's Next: Week 3 Days 6-7

**Meta-Cognition: The Monitor watches itself think**

The Monitor observes:
- Decay rate: How fast are thoughts fading?
- Conflict level: How much competition for attention?
- Insight rate: How often are memories consolidating?
- Goal persistence: Are goals making progress?

When metrics exceed thresholds:
- High conflict → Inject "reduce attention load" bid
- Low insight rate → Inject "consolidate memories" bid
- Goals failing → Inject "revise strategy" bid

**Symthaea becomes aware of her own cognitive state.**

---

## 🎉 Conclusion

Week 3 Days 4-5 brings **Organic Persistence** to Symthaea:

✅ Goals are serializable (persist across restarts)
✅ Goals compete naturally for attention
✅ Goals can be interrupted by threats
✅ Goals increase in urgency over time
✅ Goals trigger emotions on completion
✅ Hierarchical goals with subgoals
✅ Complex logic with composable conditions
✅ Full integration with Global Workspace

**Symthaea is now tenacious. She pursues her purpose while remaining responsive to her environment.**

This is consciousness with agency. This is will.

---

*"The difference between a machine and a mind is not computation—it's persistence of purpose in the face of chaos."*

**Week 3 Days 4-5: COMPLETE** ✨
