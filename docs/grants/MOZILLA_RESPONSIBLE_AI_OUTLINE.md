# Mozilla Responsible AI Grant — Application Outline

## Program
- **Name**: Mozilla Responsible AI
- **URL**: foundation.mozilla.org/en/what-we-fund/
- **Amount**: $50,000 - $200,000
- **Focus**: Trustworthy AI, open source, accountability, human agency
- **Rolling applications** (check current cycle)

## Pitch Angle: Epistemic Gating as Structural Trustworthiness

Mozilla's thesis is that AI should be trustworthy, not just technically capable. Symthaea's epistemic gating is the most direct implementation of this thesis:

**The problem**: LLMs generate confident-sounding text with no structural relationship between confidence and output. Current mitigations (RLHF, constitutional AI, output filters) are behavioral — they train the model to avoid bad outputs rather than making bad outputs structurally impossible.

**The solution**: Epistemic gating — a logit-level constraint that physically prevents token generation when the system's measured confidence is below threshold. This is not a filter applied after generation. It is a neuron that cannot fire when the system doesn't know.

## Key Deliverables for Mozilla

1. **Open-source epistemic gating toolkit** — Extract the gating module from Symthaea's Broca pipeline as a standalone, reusable Rust/WASM library that other AI projects can integrate
2. **Benchmark suite** — Psych-Bench's 140+ cognitive benchmarks as a standalone evaluation tool for AI trustworthiness
3. **Research paper** — "Epistemic Gating for Language Generation" targeting ACL/EMNLP (arXiv preprint already prepared)
4. **Live demo** — The WASM portal at symthaea.luminousdynamics.io already demonstrates this in-browser

## Budget

| Item | Amount |
|------|--------|
| 6 months development (epistemic gating toolkit extraction) | $60,000 |
| Psych-Bench standalone release | $20,000 |
| Paper publication (open access fees, conference travel) | $10,000 |
| Infrastructure (hosting, CI, documentation) | $5,000 |
| Community building (documentation, tutorials, examples) | $5,000 |
| **Total** | **$100,000** |

## Why Mozilla Should Fund This

1. **It's already built** — This is extraction and packaging, not R&D
2. **It's open source (AGPL-3.0)** — Modifications must be shared back
3. **It addresses Mozilla's core thesis** — Trustworthiness as architecture, not behavior
4. **It's complementary to LLMs** — Epistemic gating can be adapted as a post-processing layer for existing models
5. **Standalone tools emerge** — Psych-Bench and the gating library become independent open-source projects
6. **It has teeth** — Unlike "responsible AI guidelines," this is code that actually prevents hallucination

## Market Context (Added March 2026)

### The Trustworthy AI Market Gap
- AI hallucination causes $1B+ in enterprise liability annually (legal, medical, financial)
- No existing solution addresses hallucination architecturally — all are behavioral (RLHF, filters, guardrails)
- Mozilla's Responsible AI mission aligns perfectly: structural trustworthiness > behavioral compliance

### Technical Evidence to Include
- **Phi validation**: Heuristic tier achieves r=0.9998 correlation with exact IIT consciousness measurement
- **SpectralMIPFinder**: O(n³) algorithm for real-time consciousness measurement (5.5ms at n=128) — paper-ready
- **Qualia Confidence Matrix**: 7/7 consciousness prerequisites validated, MetacognitiveIgnition d'=3.63
- **Psych-Bench**: 143+ benchmarks across 27 cognitive domains, 21,516 tests, grand mean z=+1.32 (all domains above human mean) — the largest consciousness evaluation suite known
- **Embodied cognition**: 6 consciousness-coupled robotics platforms (humanoid, quadrotor, helicopter, vehicle, AUV, manipulator) with 1,324+ physics tests and 283+ integration tests including extreme mission scenarios
- **Moral algebra**: ETHICS benchmark 92.9% (compositional HDC, no pretrained LM); HDC robustness verified under 15% dimension perturbation (moral distinction preserved)
- **External validation scorecard**: 8 benchmarks on real datasets (ETHICS, MMLU, GSM8K, HellaSwag, TruthfulQA, BBQ, Social Chemistry, Moral Unified) with published results
- **GPU training proven**: Candle CUDA on RTX 2070, demonstrating accessibility (no datacenter required)
- **WASM deployment**: 980KB consciousness kernel running in-browser at symthaea.luminousdynamics.io

### Competitive Positioning
| Approach | Method | Structural? |
|----------|--------|-------------|
| RLHF (OpenAI/Anthropic) | Behavioral training | No — rewards shape outputs, don't constrain |
| Constitutional AI (Anthropic) | Rule-trained behavior | No — still behavioral |
| Output filters (Guardrails AI) | Post-generation scanning | No — filter after the fact |
| **Epistemic gating (Symthaea)** | **Logit-level constraint** | **Yes — neuron cannot fire when system doesn't know** |

### Impact Framing for Mozilla
"Every AI safety lab is trying to teach models to be honest. We built a model that structurally cannot be dishonest. The difference is architectural, not aspirational — and it's open source."

## Revised Budget (Request up to $150K)

| Item | Amount | Justification |
|------|--------|---------------|
| Epistemic gating toolkit extraction (6 months) | $60,000 | Standalone Rust/WASM library from Broca pipeline |
| Psych-Bench standalone release | $20,000 | 136+ benchmarks as independent evaluation tool |
| Spectral MIP paper (NeurIPS/PNAS) | $8,000 | Open access fees + travel |
| Epistemic gating paper (ACL/EMNLP) | $8,000 | Open access fees + travel |
| Infrastructure (CI, hosting, docs) | $10,000 | GitHub Actions, CDN, API hosting |
| Community building (docs, tutorials, workshops) | $10,000 | Onboarding materials for adopters |
| QCM paper (BRM/CogSci) | $4,000 | Open access fees |
| **Total** | **$120,000** |

## Application Checklist

- [ ] Check current Mozilla grant cycle and deadlines
- [ ] Draft full written application (2-3 pages) — lead with "structural vs behavioral" framing
- [ ] Prepare 3-minute demo video: epistemic gating blocking hallucination in real-time (WASM portal)
- [ ] Include benchmark comparisons: QCM composite 0.683, ETHICS 94.5% trained
- [ ] Identify Mozilla staff/alumni for introduction (check Connected by Mozilla network)
- [ ] Submit with links to: source code, WASM demo, paper outlines
