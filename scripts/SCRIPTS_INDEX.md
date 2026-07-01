# Phenomenal Corridor Research Scripts Index

**Location**: `/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/scripts/`
**Last Updated**: January 30, 2026

---

## Quick Reference

### Running Scripts (NixOS)

All scripts require specific Python packages. Use nix-shell:

```bash
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb

# For most scripts (numpy, torch, transformers):
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/SCRIPT_NAME.py"

# For scripts with sklearn:
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
    --run "python3 scripts/SCRIPT_NAME.py"
```

---

## Core Research Scripts

### Multi-Architecture Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `multi_arch_analysis.py` | Test phenomenal corridor across 6 architectures | `data/architecture_depth_map.json` |
| `cross_lingual_phenomenal.py` | Test 5 languages on XLM-RoBERTa | `data/cross_lingual_phenomenal.json` |
| `layerwise_phi_analysis.py` | Layer-by-layer phenomenal analysis | `data/layerwise_phi_trajectory.json` |

### Attention Mechanism Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `attention_head_analysis.py` | Identify phenomenal-discriminating heads | `data/attention_head_analysis.json` |
| `attention_entropy_analysis.py` | Layer-wise entropy patterns | `data/attention_entropy_analysis.json` |
| `attention_topology_correlation.py` | Entropy-topology correlation | `data/attention_topology_correlation.json` |
| `token_attention_analysis.py` | Token-level attention weights | `data/token_attention_analysis.json` |

### Causal Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `causal_head_ablation.py` | Ablate heads via head_mask | `data/causal_head_ablation.json` |

### Decoder-Only Architecture

| Script | Description | Output |
|--------|-------------|--------|
| `gpt2_phenomenal_corridor.py` | Test GPT-2 (12 layers) | `data/gpt2_phenomenal_corridor.json` |
| `gpt2_scaling_analysis.py` | Compare GPT-2, medium, large | `data/gpt2_scaling_analysis.json` |

### Classifier & Generalization

| Script | Description | Output |
|--------|-------------|--------|
| `train_phenomenal_classifier.py` | Train logistic regression on L11 features | Console output |
| `real_world_classifier_test.py` | Test on philosophy, poetry, meditation, etc. | `data/real_world_classifier_test.json` |

---

## Script Details

### attention_head_analysis.py

Analyzes all 144 attention heads (12 layers x 12 heads) for phenomenal discrimination.

**Key Findings**:
- L11.H4 shows strongest phenomenal discrimination (d=-2.63)
- 4/5 top phenomenal heads are in Layer 11

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/attention_head_analysis.py"
```

---

### causal_head_ablation.py

Tests whether phenomenal heads are CAUSALLY necessary using BERT's head_mask.

**Key Findings**:
- L11.H4 is NOT uniquely causal
- Phenomenal processing is DISTRIBUTED across L11
- Ablating ALL L11 heads reduces discrimination by 0.04

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/causal_head_ablation.py"
```

---

### cross_lingual_phenomenal.py

Tests phenomenal concepts in 5 languages on XLM-RoBERTa-base.

**Key Findings**:
- English: 50% depth (anomaly)
- Chinese: 92% depth (expected)
- European languages: 75-83% depth

**Languages**: en, fr, de, es, zh

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/cross_lingual_phenomenal.py"
```

---

### gpt2_phenomenal_corridor.py

Tests the phenomenal corridor in decoder-only architecture (GPT-2).

**Key Findings**:
- Peak at Layer 12 (100% depth)
- Discrimination: 1.39 (comparable to BERT)
- Late-layer pattern preserved in decoders

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/gpt2_phenomenal_corridor.py"
```

---

### gpt2_scaling_analysis.py

Compares phenomenal corridor across GPT-2 model sizes.

**Models**: gpt2 (12L), gpt2-medium (24L), gpt2-large (36L)

**Usage** (long running):
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/gpt2_scaling_analysis.py"
```

---

### multi_arch_analysis.py

Tests phenomenal corridor across 6 transformer architectures.

**Models**: BERT-base, BERT-large, RoBERTa-base, DistilBERT, BGE-M3, XLM-RoBERTa

**Key Findings**:
- 5/6 models show >70% depth
- Mean depth: 82.6% (std: 17.1%)
- XLM-RoBERTa-base (50%) is the anomaly

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers])" \
    --run "python3 scripts/multi_arch_analysis.py"
```

---

### real_world_classifier_test.py

Tests the phenomenal classifier on diverse real-world text.

**Categories**:
- Philosophy of Mind (Nagel, Chalmers, Dennett)
- Poetry & Literature (Dickinson, Wordsworth)
- Meditation & Mindfulness
- AI & Technology
- Science
- Edge Cases

**Key Findings**:
- Overall accuracy: 81.5%
- Strongest: Poetry, Meditation (100%)
- Weakest: Science (40%)

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
    --run "python3 scripts/real_world_classifier_test.py"
```

---

### train_phenomenal_classifier.py

Trains a logistic regression classifier on Layer 11 features.

**Features**: Attention entropy + reduced hidden state (54 dimensions)

**Key Findings**:
- Training accuracy: 100%
- Entropy coefficient: -0.148 (negative = lower entropy predicts phenomenal)

**Usage**:
```bash
nix-shell -p "python313.withPackages(ps: with ps; [numpy torch transformers scikit-learn])" \
    --run "python3 scripts/train_phenomenal_classifier.py"
```

---

## Data Files

All output files are in `/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/`:

| File | Description |
|------|-------------|
| `architecture_depth_map.json` | 6 architectures' layer-wise results |
| `attention_entropy_analysis.json` | 12 layers' entropy patterns |
| `attention_head_analysis.json` | 144 heads' discrimination scores |
| `attention_topology_correlation.json` | Entropy-unity correlation |
| `causal_head_ablation.json` | Head ablation results |
| `cross_lingual_phenomenal.json` | 5-language comparison |
| `gpt2_phenomenal_corridor.json` | GPT-2 layer-wise analysis |
| `gpt2_scaling_analysis.json` | GPT-2 model size comparison |
| `real_world_classifier_test.json` | Classifier generalization |
| `token_attention_analysis.json` | Token-level attention |

---

## Research Documentation

See `/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/papers/`:

| Document | Contents |
|----------|----------|
| `phenomenal_signature_paper.md` | Main research paper |
| `corridor_depth_hypothesis.md` | Depth variation analysis |
| `mechanistic_interpretability_findings.md` | Consolidated mechanistic findings |
| `anomaly_analysis.md` | XLM-RoBERTa and BERT-large anomalies |

---

*Index last updated: January 30, 2026*
