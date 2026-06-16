# Neural Causal Discovery with HDC+LTC: Improved Architecture

## The Problem with Current CausalCantorNetwork

The current architecture treats causal discovery as a temporal pattern recognition problem.
But causal asymmetry in bivariate data is fundamentally about **distributional asymmetry**:

- P(Y|X) ≠ P(X|Y) in complexity
- The cause has independent mechanism from its marginal
- Effect = f(Cause) + Noise, where Noise ⊥ Cause

## Proposed Architecture: Asymmetric Contrastive Causal Network (ACCN)

### Core Insight

Instead of using LTC for temporal dynamics, use it for **iterative refinement of causal belief**.
The "time" dimension becomes "inference steps" - how confidence evolves as we process more evidence.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASYMMETRIC CONTRASTIVE CAUSAL NET            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │   X data    │     │   Y data    │                           │
│  └──────┬──────┘     └──────┬──────┘                           │
│         │                   │                                   │
│         ▼                   ▼                                   │
│  ┌─────────────────────────────────────┐                       │
│  │     HDC ENCODER (Shared Weights)    │                       │
│  │  • Statistical moments → HV         │                       │
│  │  • Distribution shape → HV          │                       │
│  │  • Tail behavior → HV               │                       │
│  └─────────────────────────────────────┘                       │
│         │                   │                                   │
│         ▼                   ▼                                   │
│     HV_X (10000-d)     HV_Y (10000-d)                          │
│         │                   │                                   │
│         └────────┬──────────┘                                   │
│                  ▼                                              │
│  ┌─────────────────────────────────────┐                       │
│  │     ASYMMETRY DETECTOR MODULE       │                       │
│  │                                     │                       │
│  │  Forward Evidence:                  │                       │
│  │    E_fwd = encode(X→Y regression)   │                       │
│  │                                     │                       │
│  │  Backward Evidence:                 │                       │
│  │    E_bwd = encode(Y→X regression)   │                       │
│  │                                     │                       │
│  │  Asymmetry:                         │                       │
│  │    A = HV_X ⊗ E_fwd - HV_Y ⊗ E_bwd  │                       │
│  └─────────────────────────────────────┘                       │
│                  │                                              │
│                  ▼                                              │
│  ┌─────────────────────────────────────┐                       │
│  │     LTC REFINEMENT LAYER            │                       │
│  │                                     │                       │
│  │  • Input: Asymmetry HV              │                       │
│  │  • τ (time constant) = confidence   │                       │
│  │  • Iterates until convergence       │                       │
│  │  • Output: Refined causal belief    │                       │
│  └─────────────────────────────────────┘                       │
│                  │                                              │
│                  ▼                                              │
│  ┌─────────────────────────────────────┐                       │
│  │     MULTI-SCALE CANTOR AGGREGATOR   │                       │
│  │                                     │                       │
│  │  Level 1: Local evidence (small τ)  │                       │
│  │  Level 2: Regional evidence (med τ) │                       │
│  │  Level 3: Global evidence (large τ) │                       │
│  │                                     │                       │
│  │  Φ = integrated information across  │                       │
│  │      levels (indicates confidence)  │                       │
│  └─────────────────────────────────────┘                       │
│                  │                                              │
│                  ▼                                              │
│           [forward/backward]                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Architectural Changes

### 1. Statistical Feature Encoding (Not Raw Data)

Current approach encodes raw (x, y) pairs. Better approach:

```rust
struct CausalFeatures {
    // Marginal statistics
    x_mean: f64, x_std: f64, x_skew: f64, x_kurt: f64,
    y_mean: f64, y_std: f64, y_skew: f64, y_kurt: f64,

    // Joint statistics
    correlation: f64,
    mutual_info: f64,

    // Regression asymmetry
    mse_xy: f64,        // X→Y regression error
    mse_yx: f64,        // Y→X regression error
    res_indep_xy: f64,  // Independence of X from residuals(X→Y)
    res_indep_yx: f64,  // Independence of Y from residuals(Y→X)

    // Distribution shape
    heteroscedasticity: f64,
    tail_weight_x: f64,
    tail_weight_y: f64,

    // Method outputs (as soft features)
    reci_score: f64,
    igci_score: f64,
    anm_score: f64,
    info_score: f64,
}
```

### 2. Contrastive Training Objective

Train on pairs where we know the answer, using contrastive loss:

```rust
fn contrastive_loss(
    pair: &CauseEffectPair,
    network: &mut ACCN,
) -> f64 {
    // Get network's prediction for X→Y
    let score_forward = network.predict(&pair.x, &pair.y);

    // Get network's prediction for Y→X (should be opposite)
    let score_backward = network.predict(&pair.y, &pair.x);

    // The forward direction should have higher score if ground truth is forward
    let margin = 1.0;
    let target = if pair.ground_truth == "forward" { 1.0 } else { -1.0 };

    // Margin-based contrastive loss
    max(0, margin - target * (score_forward - score_backward))
}
```

### 3. LTC as Confidence Refinement

Instead of processing time series, use LTC to refine causal belief:

```rust
impl LTCRefinement {
    fn refine(&self, initial_belief: HV, evidence: &[HV]) -> (HV, f64) {
        let mut belief = initial_belief;
        let mut confidence = 0.0;

        // Iterate until convergence (LTC dynamics)
        for step in 0..self.max_steps {
            let prev_belief = belief.clone();

            // Update belief based on evidence
            for e in evidence {
                let compatibility = belief.similarity(e);
                let update_rate = self.tau.get_rate(compatibility);
                belief = belief.lerp(e, update_rate);
            }

            // Check convergence
            let delta = belief.similarity(&prev_belief);
            if delta > 0.99 {
                confidence = delta;
                break;
            }
        }

        (belief, confidence)
    }
}
```

### 4. Φ as Calibrated Confidence

Make Φ actually correlate with correctness:

```rust
impl PhiCalculation {
    fn compute(&self, levels: &[HV]) -> f64 {
        // Φ should measure how much information is lost when we partition
        // If all levels agree → high Φ → high confidence
        // If levels disagree → low Φ → low confidence

        let whole = self.aggregate(levels);
        let partition_loss = self.partition_and_compare(levels, &whole);

        // Calibrate so Φ predicts accuracy
        self.calibrate(partition_loss)
    }
}
```

## Training Strategy

### Phase 1: Pre-training on Synthetic Data

Generate synthetic causal pairs with known ground truth:

```rust
fn generate_synthetic_pair() -> CauseEffectPair {
    // Sample cause from various distributions
    let x = sample_distribution(random_distribution_type());

    // Generate effect with random mechanism
    let f = random_mechanism();  // linear, polynomial, GP, etc.
    let noise = sample_noise(random_noise_type());
    let y = f(x) + noise;

    CauseEffectPair { x, y, ground_truth: "forward" }
}

// Pre-train on 10,000+ synthetic pairs
for _ in 0..10000 {
    let pair = generate_synthetic_pair();
    network.train_step(&pair);
}
```

### Phase 2: Fine-tuning on Real Data

Use leave-one-out CV on Tübingen:

```rust
for i in 0..tuebingen.len() {
    let test_pair = &tuebingen[i];
    let train_pairs = tuebingen.iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, p)| p);

    network.fine_tune(train_pairs);
    let pred = network.predict(test_pair);
}
```

## Why This Should Work Better

1. **Feature Space**: Operates on causal-relevant statistics, not raw data
2. **Contrastive Learning**: Explicitly learns to distinguish X→Y from Y→X
3. **Multi-Scale**: Cantor aggregation captures evidence at different granularities
4. **Calibrated Confidence**: Φ trained to predict correctness
5. **Robust to Heavy Tails**: Features include kurtosis, tail weight explicitly

## Expected Performance

| Component | Contribution |
|-----------|--------------|
| Better features | +3-5% |
| Contrastive training | +2-4% |
| Calibrated Φ for abstention | +2-3% |
| **Total expected** | **75-80%** |

This would still leave a gap to oracle (90.7%), but represents a significant improvement
over current 65.7% and even Majority Voting (63.9%).

## Implementation Priority

1. **First**: Implement statistical feature extraction
2. **Second**: Add contrastive training objective
3. **Third**: LTC refinement with convergence-based confidence
4. **Fourth**: Calibrate Φ on validation set
