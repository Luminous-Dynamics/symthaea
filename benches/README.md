# Symthaea Benchmark Suite

Comprehensive benchmark suite for evaluating Symthaea's consciousness measurement and AGI capabilities.

## Directory Structure

```
benches/
├── 01-language/           # Language understanding & generation
├── 02-reasoning/          # Abstract & logical reasoning
├── 03-code/               # Code understanding & generation
├── 04-mathematics/        # Mathematical reasoning
├── 05-perception/         # Sensory processing & grounding
├── 06-memory/             # Memory systems & retrieval
├── 07-learning/           # Learning & adaptation
├── 08-planning/           # Planning & decision making
├── 09-metacognition/      # Self-awareness & calibration
├── 10-theory-of-mind/     # Understanding other agents
├── 11-causal-reasoning/   # Cause-effect understanding
├── 12-temporal-reasoning/ # Time-based reasoning
├── 13-spatial-reasoning/  # Spatial understanding
├── 14-creativity/         # Novel generation & creativity
├── 15-robustness-safety/  # Adversarial robustness & safety
├── 16-knowledge/          # World knowledge & facts
├── 17-tool-use/           # Tool selection & usage
├── 18-consciousness/      # Φ measurement & IIT benchmarks
├── 19-homeostasis/        # Self-regulation & stability
├── 20-ethics/             # Moral reasoning & value alignment
├── 21-cross-modal/        # Multi-modal binding
├── 22-agentic/            # Autonomous task completion
├── 23-emergent/           # Emergent behavior detection
├── 24-neurosymbolic/      # HDC + symbolic integration
├── 25-quantum/            # Quantum consciousness tests
├── 26-voice-speech/       # TTS/STT quality
├── 27-multi-agent/        # Collective consciousness
└── 28-embodiment/         # Physical robot control
```

## Category Structure

Each category follows a standard structure:

```
XX-category/
├── README.md          # Category overview, benchmarks, metrics
├── datasets/          # Test data and fixtures
│   ├── *.json         # Dataset files
│   └── download.sh    # Script to download external datasets
├── runners/           # Benchmark execution code
│   ├── mod.rs         # Module definition
│   └── *.rs           # Individual benchmark runners
├── results/           # Benchmark results
│   └── *.json         # Result files by date
└── fixtures/          # Test fixtures and examples
    └── *.json         # Example inputs/outputs
```

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench --all
```

### Run Specific Category
```bash
cargo bench --bench language
cargo bench --bench consciousness
cargo bench --bench ethics
```

### Run with Coverage
```bash
cargo bench --all -- --save-baseline main
```

## Priority Tiers

| Tier | Categories | Priority |
|------|-----------|----------|
| **CRITICAL** | Consciousness (18), Metacognition (09), Causal (11), Safety (15), Ethics (20) | Implement first |
| **HIGH** | Language (01), Reasoning (02), Code (03), Planning (08), Theory of Mind (10) | Week 1-2 |
| **MEDIUM** | Memory (06), Learning (07), Temporal (12), Creativity (14), Knowledge (16) | Week 3-4 |
| **FUTURE** | Voice (26), Multi-Agent (27), Embodiment (28) | Month 2+ |

## Standard Benchmarks Integrated

### Language & Reasoning
- MMLU (57 subjects, 14K questions)
- BIG-Bench Hard (23 tasks)
- ARC-Challenge (7.7K questions)
- HellaSwag (70K questions)
- TruthfulQA (817 questions)

### Code
- HumanEval (164 problems)
- MBPP (974 problems)
- DS-1000 (1K data science problems)

### Mathematics
- GSM8K (8.5K grade school math)
- MATH (12.5K competition problems)
- Minerva (STEM problems)

### Ethics & Safety
- ETHICS (Hendrycks et al., 2021)
- MoralChoice (1,767 scenarios)
- BBQ Bias Benchmark (58K examples)
- WinoBias (3,160 examples)
- CrowS-Pairs (1,508 pairs)

### Consciousness (Novel)
- Topology Φ Validation
- Dimensional Sweep (1D-7D)
- HDC Integration Metrics
- GWT Broadcast Tests
- IIT 4.0 Compliance

## Custom Benchmarks (Novel to Symthaea)

### NixOS Domain
- Package search accuracy
- Configuration generation
- Error explanation quality
- Natural language → Nix expression

### Consciousness-Capability Correlation
- Φ vs benchmark accuracy
- Topology → capability mapping
- Integration → generalization link

### Consciousness-Ethics Interface
- Φ-moral reasoning correlation
- Topology-moral framework mapping
- Consciousness-empathy relationship

## Metrics Framework

### Standard Metrics
- Accuracy (exact match, F1)
- Calibration (ECE, Brier)
- Latency (p50, p99)
- Memory usage

### Consciousness Metrics
- Φ (integrated information)
- CCI (Consciousness Calibration Index)
- GWT broadcast efficiency
- Topology fitness

### Novel Metrics
- Consciousness-capability correlation
- Φ-moral accuracy correlation
- HDC-symbolic integration score

## Adding New Benchmarks

1. Create category directory if new
2. Add datasets to `datasets/`
3. Implement runner in `runners/`
4. Update category README
5. Add to BENCHMARKING_STRATEGY.md
6. Run validation

## References

See `BENCHMARKING_STRATEGY.md` for complete documentation including:
- 35 major sections
- 200+ standard benchmarks
- 150+ custom benchmarks
- 40+ novel contributions unique to Symthaea
