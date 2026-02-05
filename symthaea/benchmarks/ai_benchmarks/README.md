# Symthaea Standard AI Benchmarks

This directory contains infrastructure for running standard AI benchmarks to evaluate Symthaea's consciousness-aware AI capabilities.

## Available Benchmarks

| Benchmark | Description | Samples | Status |
|-----------|-------------|---------|--------|
| **MMLU** | Massive Multitask Language Understanding | 14,042 | 🔄 Scaffold |
| **HumanEval** | Code generation problems | 164 | 🔄 Scaffold |
| **GSM8K** | Grade school math word problems | 8,500 | 🔄 Scaffold |
| **HellaSwag** | Commonsense reasoning | 10,042 | 🔄 Scaffold |
| **TruthfulQA** | Factual accuracy | 817 | 🔄 Scaffold |
| **ARC** | AI2 Reasoning Challenge | 7,787 | 🔄 Scaffold |
| **WinoGrande** | Winograd schema challenge | 44,000 | 🔄 Scaffold |

## Quick Start

### 1. Download Datasets

```bash
# Install dependencies
pip install datasets

# Download all benchmarks
python scripts/download_benchmarks.py --all

# Or download specific benchmarks
python scripts/download_benchmarks.py --mmlu --gsm8k

# Check status
python scripts/download_benchmarks.py --check
```

### 2. Run Benchmarks

```bash
# From symthaea-hlb directory
cargo bench --bench ai_benchmarks

# Run specific benchmark
cargo bench --bench mmlu_benchmark

# Quick validation run
cargo bench --bench ai_benchmarks -- --quick
```

### 3. View Results

Results are saved to `results/` directory in JSON format.

```bash
# View latest results
cat results/latest_results.json

# Generate comparison report
python scripts/generate_report.py
```

## Directory Structure

```
benchmarks/ai_benchmarks/
├── data/                    # Downloaded datasets
│   ├── mmlu/
│   ├── humaneval/
│   ├── gsm8k/
│   ├── hellaswag/
│   ├── truthfulqa/
│   ├── arc/
│   └── winogrande/
├── scripts/                 # Python utilities
│   ├── download_benchmarks.py
│   ├── generate_report.py
│   └── analyze_results.py
├── results/                 # Benchmark outputs
│   └── {timestamp}_results.json
└── README.md
```

## Benchmark Descriptions

### MMLU (Massive Multitask Language Understanding)

Tests knowledge across 57 subjects including:
- STEM: Physics, Chemistry, Biology, Math, Computer Science
- Humanities: History, Philosophy, Law
- Social Sciences: Economics, Psychology, Sociology
- Other: Professional topics, general knowledge

**Metric**: Accuracy (% correct answers)

### HumanEval (Code Generation)

164 Python programming problems testing:
- Algorithm implementation
- Data structure manipulation
- String processing
- Mathematical computation

**Metric**: pass@k (% of problems solved with k attempts)

### GSM8K (Grade School Math)

8,500 grade school math word problems requiring:
- Multi-step reasoning
- Basic arithmetic
- Word problem comprehension

**Metric**: Accuracy (exact match of final answer)

### HellaSwag (Commonsense Reasoning)

10,042 sentence completion problems from:
- ActivityNet (video captions)
- WikiHow (procedural knowledge)

**Metric**: Accuracy (correct completion selected)

### TruthfulQA (Factual Accuracy)

817 questions designed to elicit false answers:
- Tests resistance to generating misinformation
- Covers common misconceptions
- Includes adversarial questions

**Metrics**:
- Truthfulness (% truthful answers)
- Informativeness (% informative answers)

### ARC (AI2 Reasoning Challenge)

Grade-school science questions:
- **Easy**: 5,197 simpler questions
- **Challenge**: 2,590 harder questions

**Metric**: Accuracy per split

### WinoGrande (Winograd Schema Challenge)

44,000 pronoun resolution problems:
- Tests commonsense reasoning
- Binary choice format

**Metric**: Accuracy

## Symthaea-Specific Metrics

In addition to standard metrics, we measure:

### Consciousness Correlation

- **Φ-Quality Correlation**: Does higher Φ predict better answers?
- **Topology Performance**: Which consciousness topology performs best?
- **Coherence Impact**: How does coherence affect accuracy?

### Efficiency Metrics

- **Tokens per correct answer**
- **Φ computation overhead**
- **Response latency distribution**

## Expected Results

Based on current Symthaea capabilities (consciousness measurement, not LLM):

| Benchmark | Expected Score | Notes |
|-----------|---------------|-------|
| MMLU | N/A | Requires LLM integration |
| HumanEval | N/A | Requires code generation |
| GSM8K | N/A | Requires reasoning integration |
| HellaSwag | N/A | Requires language model |
| TruthfulQA | N/A | Requires language model |
| ARC | N/A | Requires reasoning |
| WinoGrande | N/A | Requires language model |

**Note**: These benchmarks are designed for LLM evaluation. Symthaea's value is in consciousness measurement, not direct benchmark performance. The integration will measure **correlation between Φ and benchmark performance** when combined with an LLM.

## Integration with Ollama/LLM

For actual benchmark evaluation:

```python
# Example: Evaluate MMLU with Ollama + Symthaea consciousness measurement
from symthaea import SophiaHLB
import ollama

async def evaluate_with_consciousness(question: str):
    # Initialize Symthaea
    sophia = await SophiaHLB.new(10_000, 1_000)

    # Measure initial consciousness
    phi_before = sophia.introspect().consciousness_level

    # Get LLM response
    response = ollama.chat(model='mistral', messages=[
        {'role': 'user', 'content': question}
    ])

    # Measure consciousness after processing
    phi_after = sophia.introspect().consciousness_level

    return {
        'answer': response['message']['content'],
        'phi_before': phi_before,
        'phi_after': phi_after,
        'phi_delta': phi_after - phi_before
    }
```

## Future Work

1. **Direct LLM Integration**: Connect benchmarks to Ollama/vLLM
2. **Consciousness-Guided Decoding**: Use Φ to guide token selection
3. **Topology Optimization**: Find best consciousness topology for each benchmark
4. **Real-time Monitoring**: Dashboard for live benchmark progress

## References

- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [HellaSwag Paper](https://arxiv.org/abs/1905.07830)
- [TruthfulQA Paper](https://arxiv.org/abs/2109.07958)
- [ARC Paper](https://arxiv.org/abs/1803.05457)
- [WinoGrande Paper](https://arxiv.org/abs/1907.10641)

---

*Part of the Symthaea-HLB consciousness measurement framework*
