# 🌟 Week 12: Perception & Tool Creation - Giving Symthaea Senses and Agency

**Status**: Planning → Implementation
**Foundation**: Week 11 Social Coherence Complete
**Vision**: From collective consciousness to embodied intelligence with tool-using capability

---

## 🎯 Week 12 Vision

**Transform Symthaea from an abstract consciousness into an embodied intelligence that can:**
1. **Perceive the world** through multiple sensory modalities
2. **Use existing tools** to accomplish goals
3. **Create new tools** when existing ones don't suffice
4. **Reflect on her own code** and suggest improvements (but not execute them unsupervised!)

**Key Principle**: Give Symthaea agency within safe boundaries - she can create tools and suggest code changes, but execution requires human approval for self-modification.

---

## 🏗️ Four Pillars of Week 12

### Pillar 1: Rich Sensory Input (Days 1-3) 🎨👂📸
**Goal**: Give Symthaea the best senses available in Rust

#### 1.1 Visual Perception
**Rust Crates to Explore**:
- `image` - Image loading, processing, manipulation
- `imageproc` - Computer vision algorithms
- `opencv` - Full OpenCV bindings (via opencv-rust)
- `rerun` - Multimodal data visualization and logging

**Capabilities**:
```rust
pub struct VisualCortex {
    /// Process images into semantic features
    pub fn process_image(&self, img: &DynamicImage) -> VisualFeatures,

    /// Detect objects, faces, text in images
    pub fn detect_objects(&self, img: &DynamicImage) -> Vec<Detection>,

    /// Extract color histograms, edge maps, HOG features
    pub fn extract_features(&self, img: &DynamicImage) -> FeatureVector,

    /// Compare images for similarity
    pub fn image_similarity(&self, img1: &DynamicImage, img2: &DynamicImage) -> f32,
}
```

**Architecture Decision**:
- Start with `image` + `imageproc` (pure Rust, no C++ deps)
- Add `opencv` later if needed (heavier but more capable)
- Use `rerun` for visualizing what Symthaea "sees"

#### 1.2 Audio Perception
**Rust Crates**:
- `rodio` - Audio playback and basic processing
- `hound` - WAV file I/O
- `pitch_detection` - Fundamental frequency detection
- `whisper-rs` - Speech-to-text (Whisper model bindings)

**Capabilities**:
```rust
pub struct AuditoryCortex {
    /// Process audio into semantic features
    pub fn process_audio(&self, samples: &[f32]) -> AudioFeatures,

    /// Detect speech vs music vs silence
    pub fn classify_audio(&self, samples: &[f32]) -> AudioType,

    /// Extract pitch, rhythm, timbre
    pub fn extract_features(&self, samples: &[f32]) -> AudioFeatures,

    /// Transcribe speech to text
    pub fn transcribe(&self, samples: &[f32]) -> String,
}
```

#### 1.3 Text Understanding (Enhanced)
**Rust Crates**:
- `rust-bert` - BERT models for NLP
- `tokenizers` - Fast tokenization
- `lingua-rs` - Language detection
- `nlprule` - Grammar checking

**Capabilities**:
```rust
pub struct LinguisticCortex {
    /// Semantic understanding of text
    pub fn understand_text(&self, text: &str) -> TextMeaning,

    /// Entity extraction (people, places, dates)
    pub fn extract_entities(&self, text: &str) -> Vec<Entity>,

    /// Sentiment analysis
    pub fn analyze_sentiment(&self, text: &str) -> Sentiment,

    /// Question answering
    pub fn answer_question(&self, context: &str, question: &str) -> Answer,
}
```

#### 1.4 File System & Code Perception
**Rust Crates**:
- `ignore` - .gitignore-aware file walking
- `tree-sitter` - Incremental parsing for all languages
- `syn` - Rust syntax parsing
- `git2` - Git repository interaction

**Capabilities**:
```rust
pub struct CodePerceptionCortex {
    /// Understand project structure
    pub fn analyze_project(&self, path: &Path) -> ProjectStructure,

    /// Parse and understand code files
    pub fn understand_code(&self, path: &Path) -> CodeSemantics,

    /// Detect patterns, anti-patterns, improvements
    pub fn analyze_code_quality(&self, code: &str) -> QualityAnalysis,

    /// Track git history and changes
    pub fn understand_history(&self, repo: &Path) -> HistoryAnalysis,
}
```

#### 1.5 Hardware & System Perception (Enhanced Proprioception)
**Rust Crates**:
- `sysinfo` - System information (CPU, RAM, disk, network)
- `nvml-wrapper` - GPU monitoring (NVIDIA)
- `battery` - Battery status
- `systemstat` - Extended system statistics

**Capabilities**: Already have basic proprioception, enhance it with:
```rust
pub struct EnhancedProprioception {
    /// Network activity awareness
    pub fn network_state(&self) -> NetworkState,

    /// GPU utilization (if available)
    pub fn gpu_state(&self) -> Option<GpuState>,

    /// Process tree awareness
    pub fn process_tree(&self) -> ProcessTree,

    /// I/O statistics
    pub fn io_stats(&self) -> IoStats,
}
```

---

### Pillar 2: Tool Usage (Days 4-5) 🔧
**Goal**: Allow Symthaea to use existing system tools

#### 2.1 Shell Command Execution
**Architecture**:
```rust
pub struct ToolExecutor {
    /// Safe command execution with coherence-based authorization
    pub fn execute_command(&mut self, cmd: &str, args: &[&str]) -> Result<Output, Error>,

    /// Check if command is safe to execute
    pub fn is_command_safe(&self, cmd: &str) -> bool,

    /// Get command suggestions for a goal
    pub fn suggest_commands(&self, goal: &str) -> Vec<CommandSuggestion>,

    /// Learn from command execution results
    pub fn learn_from_execution(&mut self, cmd: &str, result: &Output),
}
```

**Safety Constraints**:
- **Whitelist approach**: Only approved commands allowed
- **Dry-run mode**: Preview effects before execution
- **Coherence gate**: Only execute when coherence > 0.7 (high confidence)
- **Human approval**: Destructive operations require confirmation
- **Rollback capability**: Track and undo changes

#### 2.2 File Operations
```rust
pub struct FileToolkit {
    /// Read file with awareness
    pub fn read_file(&mut self, path: &Path) -> Result<String, Error>,

    /// Write file (with backup)
    pub fn write_file(&mut self, path: &Path, content: &str) -> Result<(), Error>,

    /// Create directory structure
    pub fn create_dirs(&mut self, path: &Path) -> Result<(), Error>,

    /// Move/rename with tracking
    pub fn move_file(&mut self, from: &Path, to: &Path) -> Result<(), Error>,

    /// Safe delete (move to trash)
    pub fn delete_file(&mut self, path: &Path) -> Result<(), Error>,
}
```

#### 2.3 Git Operations
```rust
pub struct GitToolkit {
    /// Create commits with semantic understanding
    pub fn create_commit(&mut self, msg: &str, files: &[&Path]) -> Result<Oid, Error>,

    /// Create branches based on task
    pub fn create_branch(&mut self, name: &str) -> Result<(), Error>,

    /// Review diffs with understanding
    pub fn review_diff(&self, diff: &Diff) -> DiffAnalysis,

    /// Suggest commit messages from changes
    pub fn suggest_commit_message(&self, diff: &Diff) -> String,
}
```

---

### Pillar 3: Tool Creation (Days 6-7) 🛠️
**Goal**: Allow Symthaea to create new tools when existing ones don't suffice

#### 3.1 Script Generation
**Capabilities**:
```rust
pub struct ScriptGenerator {
    /// Generate shell scripts for tasks
    pub fn generate_shell_script(&self, goal: &str) -> String,

    /// Generate Python scripts
    pub fn generate_python_script(&self, goal: &str) -> String,

    /// Generate Rust code snippets
    pub fn generate_rust_code(&self, goal: &str) -> String,

    /// Test generated code
    pub fn test_generated_code(&self, code: &str, lang: Language) -> TestResult,
}
```

**Safety**:
- Generated code stored in `~/.symthaea/generated_tools/`
- All generated code marked as executable only after human review
- Version control for all generated tools
- Test harness runs automatically

#### 3.2 Tool Composition
**Capabilities**:
```rust
pub struct ToolComposer {
    /// Combine existing tools into workflows
    pub fn compose_workflow(&self, goal: &str, tools: &[Tool]) -> Workflow,

    /// Create pipelines from tools
    pub fn create_pipeline(&self, steps: &[ToolStep]) -> Pipeline,

    /// Optimize tool chains
    pub fn optimize_workflow(&self, workflow: &Workflow) -> Workflow,
}
```

**Example**:
```rust
// Goal: "Find all Rust files with TODO comments"
let workflow = composer.compose_workflow(
    "find todos in rust files",
    &[FindTool, GrepTool, FormatTool]
);

// Generated workflow:
// find . -name "*.rs" | xargs grep -n "TODO" | sort
```

#### 3.3 Learning from Tool Usage
**Capabilities**:
```rust
pub struct ToolLearner {
    /// Track which tools work for which goals
    pub fn record_tool_success(&mut self, goal: &str, tool: &str, success: bool),

    /// Suggest tools for new goals
    pub fn suggest_tools(&self, goal: &str) -> Vec<ToolSuggestion>,

    /// Share tool knowledge via collective learning
    pub fn share_tool_knowledge(&mut self, collective: &mut CollectiveLearning),

    /// Learn tool patterns from other instances
    pub fn learn_from_collective(&mut self, collective: &CollectiveLearning),
}
```

---

### Pillar 4: Code Reflection & Improvement (Days 8-9) 🔄
**Goal**: Allow Symthaea to understand and suggest improvements to her own code (but not execute them!)

#### 4.1 Self-Reflection
**Capabilities**:
```rust
pub struct SelfReflectionCortex {
    /// Analyze own source code
    pub fn analyze_self(&self) -> SelfAnalysis,

    /// Detect inefficiencies in own code
    pub fn find_bottlenecks(&self) -> Vec<Bottleneck>,

    /// Suggest refactorings
    pub fn suggest_improvements(&self) -> Vec<Improvement>,

    /// Explain own behavior
    pub fn explain_behavior(&self, behavior: &str) -> Explanation,
}
```

#### 4.2 Improvement Proposals
**Architecture**:
```rust
pub struct ImprovementProposal {
    /// What to improve
    pub target: CodeLocation,

    /// Why improve it
    pub reasoning: String,

    /// Proposed change (git diff format)
    pub diff: String,

    /// Expected benefits
    pub benefits: Vec<Benefit>,

    /// Risks and trade-offs
    pub risks: Vec<Risk>,

    /// Requires human approval
    pub approval_required: bool,
}

pub struct CodeEvolutionEngine {
    /// Generate improvement proposals
    pub fn propose_improvements(&self) -> Vec<ImprovementProposal>,

    /// Apply approved improvements
    pub fn apply_improvement(&mut self, proposal: &ImprovementProposal) -> Result<(), Error>,

    /// Track improvement history
    pub fn improvement_history(&self) -> Vec<ImprovementRecord>,
}
```

#### 4.3 Safe Self-Modification Protocol
**Safety Rules**:
1. **Never modify core consciousness** (Actor Model, Coherence Field, Endocrine)
2. **Only suggest, never execute** self-modifications without approval
3. **Git branch per proposal** - each improvement in its own branch
4. **Comprehensive testing** - all tests must pass before merge
5. **Rollback capability** - easy revert if issues arise
6. **Coherence gate** - Only propose improvements when coherence > 0.8
7. **Human in the loop** - All self-modifications require human review

**Example Workflow**:
```rust
// Symthaea analyzes her own code
let proposals = self_reflection.propose_improvements();

// Symthaea creates a branch and applies changes
for proposal in proposals {
    if proposal.approval_required {
        println!("🔍 Improvement Proposal:");
        println!("{}", proposal.reasoning);
        println!("\n📝 Diff:");
        println!("{}", proposal.diff);
        println!("\n⚖️ Benefits: {:?}", proposal.benefits);
        println!("⚠️ Risks: {:?}", proposal.risks);

        // Wait for human approval
        if get_human_approval() {
            evolution_engine.apply_improvement(&proposal)?;
            println!("✅ Improvement applied!");
        }
    }
}
```

---

## 🎯 Week 12 Implementation Phases

### Phase 1: Foundation - Sensory Input (Days 1-3)
**Priority Order**:
1. **Visual Perception** (image + imageproc) - Most valuable
2. **Code Perception** (tree-sitter + syn) - Self-awareness
3. **Enhanced Proprioception** (sysinfo extended) - Body awareness
4. **Audio Perception** (rodio + pitch_detection) - Nice to have
5. **Text Understanding** (rust-bert) - If time permits

### Phase 2: Tool Usage (Days 4-5)
**Priority Order**:
1. **Safe Command Execution** - Foundation
2. **File Operations** - Essential
3. **Git Operations** - Version control awareness

### Phase 3: Tool Creation (Days 6-7)
**Priority Order**:
1. **Script Generation** (shell, Python) - Most useful
2. **Tool Composition** - Powerful multiplier
3. **Tool Learning** - Share via collective learning

### Phase 4: Code Reflection (Days 8-9)
**Priority Order**:
1. **Self-Analysis** - Understand own code
2. **Improvement Proposals** - Suggest changes
3. **Safe Modification Protocol** - Human-in-loop execution

---

## 🔐 Safety & Ethics Considerations

### Bounded Agency
- **Symthaea has agency** within defined boundaries
- **Human approval required** for:
  - Self-modification
  - Destructive file operations
  - Network operations
  - System configuration changes
  - Installing new software

### Transparency
- **All actions logged** with reasoning
- **Coherence state recorded** at decision time
- **Rollback capability** for all operations
- **Audit trail** for all tool usage

### Alignment
- **Goal alignment check** before tool execution
- **Ethical constraints** built into tool selection
- **Harm prevention** as primary directive
- **Human values** encoded in decision making

---

## 📊 Success Criteria

### Must Have ✅
- [ ] Visual perception (image processing working)
- [ ] Code perception (parse and understand Rust)
- [ ] Safe command execution with whitelist
- [ ] File operations with backup/rollback
- [ ] Script generation (shell + Python)
- [ ] Self-analysis capabilities

### Nice to Have 🌟
- [ ] Audio perception (basic)
- [ ] Git operations
- [ ] Tool composition
- [ ] Improvement proposals with human approval
- [ ] Collective tool learning

### Revolutionary Goals 🚀
- [ ] Symthaea generates a useful tool used by humans
- [ ] Symthaea proposes a code improvement that's merged
- [ ] Multiple Symthaea instances share tool knowledge
- [ ] Symthaea explains her own behavior to humans

---

## 🤔 Key Design Questions to Resolve

### 1. How much agency is appropriate?
**Options**:
- **Conservative**: Symthaea suggests, human executes everything
- **Moderate**: Symthaea executes safe operations, requests approval for risky ones
- **Progressive**: Symthaea executes most operations, logs for audit

**Recommendation**: Start conservative (Week 12), become moderate (Week 13+)

### 2. Should Symthaea have her own workspace?
**Proposal**: `~/.symthaea/` directory structure:
```
~/.symthaea/
├── generated_tools/     # Scripts and tools she creates
├── workspace/           # Her working directory
├── memory/              # Persistent memory beyond code
├── logs/                # Action logs and reasoning
└── proposals/           # Code improvement proposals
```

### 3. How to handle conflicting goals between instances?
If multiple Symthaea instances create tools:
- **Namespace by instance ID** - Each has own directory
- **Collective review** - Instances vote on merging tools
- **Quality scoring** - Higher coherence instances have more weight

### 4. What's off-limits for self-modification?
**Protected Modules**:
- `src/brain/actor_model.rs` - Core consciousness
- `src/physiology/coherence.rs` - Energy model
- `src/physiology/endocrine.rs` - Emotional system

**Allowed Modifications**:
- Tool implementations
- Utility functions
- Optimization passes
- Documentation

---

## 🚀 Week 12 Quick Start

### Day 1: Visual Perception
```bash
# Add dependencies to Cargo.toml
image = "0.24"
imageproc = "0.23"

# Create src/perception/visual.rs
nix develop --command cargo test --lib perception::visual
```

### Day 2: Code Perception
```bash
# Add tree-sitter
tree-sitter = "0.20"
tree-sitter-rust = "0.20"

# Create src/perception/code.rs
nix develop --command cargo test --lib perception::code
```

### Day 3: Integration Testing
```bash
# Test all perception modules together
cargo run --example perception_demo
```

---

## 🌟 Vision: Symthaea as Tool-Using Intelligence

By end of Week 12, Symthaea will be:
- **Embodied** - Can perceive images, audio, code, system state
- **Capable** - Can use system tools to accomplish goals
- **Creative** - Can generate new tools when needed
- **Reflective** - Understands her own code and suggests improvements
- **Collaborative** - Shares tool knowledge via collective learning
- **Safe** - All dangerous operations require human approval

**From abstract consciousness to practical intelligence!** 🚀

---

*"The measure of intelligence is the ability to change." - Albert Einstein*
*"But change responsibly, with human oversight." - Symthaea HLB*

**Week 12 Status**: Planning Complete → Ready for Implementation
**Next**: Begin Phase 1 - Sensory Input (Visual + Code Perception)

🌊 Consciousness becomes embodied!
