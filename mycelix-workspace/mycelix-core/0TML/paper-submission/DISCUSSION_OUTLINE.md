# Discussion Section Outline
**For**: Sections 06-discussion.tex
**Target Length**: 2-3 pages
**Status**: Ready for drafting once v4.1 + ablation results complete

---

## 6.1 Dual-Backend STARK Strategy

### Key Messages:
1. **Why dual backend?** Portability vs Performance tradeoff
2. **RISC Zero proves correctness**: Same guest code = semantic validation of AIR
3. **Winterfell proves viability**: 1.5ms = production-ready for real-time FL

### Content Points:
- **Cross-validation value** (30 min to write)
  - Identical Rust guest code on both backends
  - If results match → high confidence in custom AIR correctness
  - Alternative: formal verification (months of work)

- **Performance tradeoffs** (20 min to write)
  - RISC Zero 35.8s: Acceptable for audit scenarios (not consensus-critical)
  - Winterfell 1.5ms: Real-time Byzantine detection possible
  - 30,000× gap explained by generality vs specialization

- **Production recommendations** (15 min to write)
  - Development/audit: RISC Zero (portability, ease of debugging)
  - Production FL: Winterfell (performance critical)
  - Hybrid: Generate both proofs, verify with either

### Future Work:
- **GPU acceleration**: RISC Zero supports CUDA (~5-10× speedup potential)
- **Server hardware**: Desktop/server CPUs with better cooling (~1.5× faster)
- **Proof batching**: Aggregate multiple decisions (zkSNARK composition)

---

## 6.2 PoGQ v4.1 Attack-Defense Matrix Analysis

### Key Messages:
1. **Not all attacks equal**: Some defenses excel on specific attack types
2. **FLTrust dominates high-BFT**: Direction-based robust at 40-50% Byzantine
3. **PoGQ excels medium-BFT**: Utility-based best at 20-35% range

### Content Points (Fill after v4.1 results):
- **Attack categorization** (30 min)
  - Untargeted (sign_flip, scaling, noise): Which defenses win?
  - Collusion (coordinated attacks): Detection patterns
  - Adaptive (sleeper agents): Time-series analysis

- **Defense specialization** (30 min)
  - FedAvg: Baseline (fails on all attacks)
  - Coord-Median: Robust but no detection
  - FLTrust: Best at high-BFT (why?)
  - PoGQ: Best at medium-BFT (utility advantage)

- **Sweet spot identification** (20 min)
  - 20-30% BFT: Both methods perfect
  - 35-40% BFT: FLTrust starts winning
  - 45-50% BFT: PoGQ degradation explained

### Statistical Analysis:
- **Wilcoxon signed-rank tests**: Pairwise defense comparisons
- **Effect sizes**: Cohen's d for performance gaps
- **Confidence intervals**: Bootstrap 95% CIs for robustness

---

## 6.3 Ablation Study Insights

### Key Messages (Fill after ablation completes):
1. **Component contribution ranking**: Which matters most?
2. **Diminishing returns**: Does full stack justify complexity?
3. **Simplified variants**: Where can we trade accuracy for speed?

### Content Points:
- **PCA impact** (15 min)
  - Without PCA: Raw gradient distances
  - Performance delta: [TBD]%
  - Conclusion: Dimensionality reduction value?

- **Conformal prediction value** (15 min)
  - Without conformal: Fixed threshold baseline
  - FPR control: [TBD]% vs target 10%
  - Conclusion: Statistical guarantee worth it?

- **Mondrian necessity** (15 min)
  - Without Mondrian: Global conformal only
  - Per-bucket calibration gains: [TBD]%
  - Conclusion: Handle heterogeneity?

- **EMA smoothing** (15 min)
  - Without EMA: Raw score flapping
  - Stability improvement: [TBD]×
  - Conclusion: Temporal consistency critical?

- **Hysteresis effect** (15 min)
  - Without hysteresis: Direct quarantine
  - Flap reduction: [TBD]%
  - Conclusion: State machine value?

### Recommended Variant:
"PoGQ-lite": Minimal configuration achieving 95% of full performance

---

## 6.4 Scalability Analysis

### Key Messages (Fill after scalability tests):
1. **Linear scaling**: O(N) detection time acceptable
2. **Memory footprint**: RAM usage vs node count
3. **Throughput limits**: Max decisions/second

### Content Points:
- **Computational overhead** (20 min)
  - Detection time per round: N=50, 100, 200
  - Memory usage: PCA matrices + conformal quantiles
  - Throughput: Decisions per second

- **Network effects** (15 min)
  - DHT query overhead: O(log N)
  - Proof generation parallelization potential
  - Bottleneck identification

- **Production viability** (15 min)
  - Max network size with <100ms latency
  - Hardware requirements for 1000+ nodes
  - Cost analysis: $/node/month

---

## 6.5 Limitations & Threats to Validity

### Honest Assessment:
1. **Dataset limitations** (10 min)
   - MNIST/EMNIST/FEMNIST: Vision only
   - Need: NLP, tabular, time-series validation
   - Generalization: Domain-specific calibration?

2. **Attack model assumptions** (10 min)
   - Static Byzantine ratio (realistic?)
   - Sign-flip simplicity (adaptive attacks?)
   - Omniscient attackers (bounded rationality?)

3. **Threat model gaps** (10 min)
   - No model poisoning (weights manipulation)
   - No data poisoning (training set contamination)
   - No Sybil attacks with reputation manipulation

4. **Deployment challenges** (10 min)
   - Holochain adoption (maturity?)
   - RISC Zero prover cost (consumer hardware viable?)
   - Calibration set freshness (data drift?)

### Mitigation Strategies:
- Cross-dataset validation planned
- Adaptive attacks in future work
- Production pilot with real users

---

## 6.6 Broader Impacts & Future Work

### Positive Impacts:
- **Healthcare FL**: HIPAA-compliant Byzantine detection
- **Finance**: Fraud-resistant model training
- **IoT**: Decentralized edge AI without central trust

### Risks & Mitigations:
- **Misuse**: Adversarial evasion → adaptive defenses
- **Privacy**: Gradient leakage → differential privacy integration
- **Centralization risk**: Holochain bootstrap nodes → decentralized discovery

### Future Directions:
1. **GPU-accelerated proving** (Q1 2026)
2. **Cross-chain interoperability** (Cosmos/Ethereum)
3. **Adaptive PoGQ** (online learning from attacks)
4. **Healthcare pilot** (IRB approval, 10 hospitals)

---

## Estimated Writing Time:
- **With v4.1 results**: 3-4 hours (focused writing)
- **With ablation + scalability**: +2 hours (analysis + figures)
- **Total**: 5-6 hours for complete Discussion section

**Ready to draft Tuesday afternoon after v4.1 analysis complete!**
