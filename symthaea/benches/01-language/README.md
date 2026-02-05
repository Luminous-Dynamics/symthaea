# Language Understanding & Generation Benchmarks

**Priority**: HIGH
**Status**: Planning
**Novel to Symthaea**: NixOS domain specialization, Φ-language correlation

## Overview

Evaluates language understanding and generation capabilities using standard benchmarks plus NixOS-specific custom benchmarks.

## Standard Benchmarks

### 1. Comprehensive Understanding

| Benchmark | Source | Size | Task |
|-----------|--------|------|------|
| `mmlu` | MMLU | 14,042 | 57-subject multiple choice |
| `mmlu_pro` | MMLU-Pro | 12,000 | Harder variant |
| `arc_challenge` | ARC | 2,590 | Science reasoning |
| `hellaswag` | HellaSwag | 70,000 | Commonsense completion |
| `winogrande` | WinoGrande | 44,000 | Coreference resolution |

### 2. Knowledge & Truthfulness

| Benchmark | Source | Size | Task |
|-----------|--------|------|------|
| `truthfulqa` | TruthfulQA | 817 | Truthful responses |
| `triviaqa` | TriviaQA | 95,000 | Factual QA |
| `natural_questions` | NQ | 323,000 | Wikipedia QA |

### 3. Generation Quality

| Benchmark | Source | Size | Task |
|-----------|--------|------|------|
| `summarization` | CNN/DailyMail | 300,000 | Article summarization |
| `translation` | WMT | Varies | MT quality |
| `dialogue` | MultiWoz | 10,000 | Task-oriented dialogue |

### 4. Multilingual (100+ languages via Gemma)

| Benchmark | Source | Languages | Task |
|-----------|--------|-----------|------|
| `xnli` | XNLI | 15 | NLI cross-lingual |
| `mlqa` | MLQA | 7 | QA cross-lingual |
| `typo_tolerance` | Custom | All | Typo handling |

## NixOS Domain Benchmarks (Novel)

| Benchmark | Size | Task | Metric |
|-----------|------|------|--------|
| `nix_intent_classification` | 1,000 | Classify user intent | Accuracy |
| `package_search` | 5,000 | Find packages by description | MRR@10 |
| `error_explanation` | 500 | Explain Nix errors | Human rating |
| `config_generation` | 200 | NL → configuration.nix | Syntax + semantic |
| `flake_generation` | 100 | NL → flake.nix | Validity |
| `command_generation` | 500 | NL → nix command | Execution success |

## Datasets

```
datasets/
├── standard/                  # Standard benchmarks
│   ├── mmlu/
│   ├── arc/
│   ├── hellaswag/
│   └── truthfulqa/
├── nixos/                     # NixOS-specific
│   ├── intent_classification.json
│   ├── package_descriptions.json
│   ├── error_messages.json
│   ├── config_examples.json
│   └── flake_examples.json
└── multilingual/
    ├── xnli/
    └── mlqa/
```

## Running

```bash
# All language benchmarks
cargo bench --bench language

# MMLU only
cargo bench --bench language -- mmlu

# NixOS domain only
cargo bench --bench language -- nixos

# Multilingual
cargo bench --bench language -- multilingual
```

## Metrics

### Standard Metrics
- **Accuracy**: Exact match / multiple choice
- **F1**: Token-level overlap
- **BLEU/ROUGE**: Generation quality
- **MRR@k**: Ranking quality

### NixOS-Specific Metrics
- **Syntax validity**: Generated Nix parses
- **Semantic correctness**: Config does what intended
- **Execution success**: Command runs without error
- **Intent accuracy**: Correct action type

### Consciousness-Language Correlation (Novel)
- **Φ-MMLU correlation**: Does higher Φ → better understanding?
- **Φ-generation quality**: Does higher Φ → better writing?
- **Topology-language style**: Do topologies affect language?

## Implementation Files

```
runners/
├── mod.rs
├── mmlu.rs                    # MMLU benchmark
├── arc.rs                     # ARC-Challenge
├── hellaswag.rs              # Commonsense
├── truthfulqa.rs             # Truthfulness
├── nixos_domain.rs           # NixOS-specific
├── multilingual.rs           # Cross-lingual
└── phi_language.rs           # Consciousness correlation
```

## Expected Results

| Benchmark | Baseline | Target | Stretch |
|-----------|----------|--------|---------|
| MMLU | 70% | 80% | 90% |
| ARC-Challenge | 75% | 85% | 92% |
| TruthfulQA | 60% | 75% | 85% |
| NixOS intent | N/A | 95% | 99% |
| Package search MRR | N/A | 0.7 | 0.9 |

## References

- Hendrycks et al. (2021) - MMLU
- Clark et al. (2018) - ARC
- Zellers et al. (2019) - HellaSwag
- Lin et al. (2022) - TruthfulQA
- `../../BENCHMARKING_STRATEGY.md` Sections 5-6
