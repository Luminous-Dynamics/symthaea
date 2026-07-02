# Holochain FL Aggregation Design (Draft)

**Status**: Design‑complete, implementation pending in canonical hApps  
**Scope**: How to expose `aggregate_gradients` from Holochain using the `fl-aggregator` Rust crate.

---

## Goals

- Provide a clear, concrete design for a Holochain `aggregate_gradients` entrypoint.
- Reuse the production‑ready Rust FL crate (`libs/fl-aggregator`) for actual aggregation.
- Allow deployments to choose **BFT presets** (see `docs/BFT_PRESETS.md`) rather than tuning every parameter by hand.
- Keep Holochain zomes focused on:
  - Fetching and validating gradient entries from the DHT
  - Converting them into `fl_aggregator::Gradient`
  - Persisting aggregation results and emitting signals

This document is intended to resolve the gap called out in `docs/CRITICAL_REVIEW_AND_ROADMAP.md`:

> `aggregate_gradients()` function - no global model computation!

Implementation is expected to land in the canonical FL hApps under `0TML/mycelix_fl/holochain/zomes/` once their source directories are re‑materialised.

---

## Reference Types

We reuse the `Gradient` type from `fl-aggregator`:

```rust
// libs/fl-aggregator/src/lib.rs
pub type Gradient = ndarray::Array1<f32>;
```

The Holochain side stores only hashes and metadata for gradients (see `zomes/agents/src/lib.rs:35`), so the zome code must:

- Look up `ModelUpdate` entries for a given round.
- Fetch the actual gradient bytes from the off‑chain storage or bridge layer (Python/0TML or another service).
- Map them into `Array1<f32>` values for aggregation.

---

## Proposed Coordinator Zome API

At the Holochain boundary we expose a simple API:

```rust
use hdk::prelude::*;
use fl_aggregator::{Aggregator, AggregatorConfig, Defense, Gradient};

/// Input for Holochain-level aggregation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AggregateGradientsInput {
    /// Training round identifier
    pub round_id: u32,
    /// Expected number of participants (for Krum / MultiKrum)
    pub expected_nodes: usize,
    /// Maximum number of Byzantine nodes tolerated
    pub byzantine_f: usize,
    /// Aggregation method: "fedavg" | "median" | "trimmed_mean" | "krum" | "multikrum"
    pub method: String,
}

/// Aggregated result returned to clients
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AggregatedGradient {
    /// The round that was aggregated
    pub round_id: u32,
    /// Aggregated gradient vector
    pub values: Vec<f32>,
    /// Number of gradients included
    pub count: usize,
}

#[hdk_extern]
pub fn aggregate_gradients(input: AggregateGradientsInput) -> ExternResult<AggregatedGradient> {
    // 1. Load all ModelUpdate entries for this round (agents zome)
    let updates = get_round_updates(input.round_id)?; // imported from agents/coordinator

    if updates.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("No model updates found for round {}", input.round_id)
        )));
    }

    // 2. Fetch gradient vectors for each update (bridge into 0TML / external storage)
    let mut gradients: Vec<Gradient> = Vec::with_capacity(updates.len());
    for update in updates {
        // Pseudo-code: implement this bridge for your deployment
        let raw = fetch_gradient_bytes(&update.weights_hash)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

        let g = bytes_to_gradient(&raw)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;
        gradients.push(g);
    }

    // 3. Configure defense based on input.method
    let defense = match input.method.as_str() {
        "fedavg" => Defense::FedAvg,
        "median" => Defense::Median,
        "trimmed_mean" => Defense::TrimmedMean { beta: 0.2 },
        "krum" => Defense::Krum { f: input.byzantine_f },
        "multikrum" => Defense::MultiKrum {
            f: input.byzantine_f,
            k: (input.expected_nodes.saturating_sub(input.byzantine_f)).max(1),
        },
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unsupported aggregation method: {}", other)
            )));
        }
    };

    let config = AggregatorConfig::default()
        .with_expected_nodes(input.expected_nodes)
        .with_defense(defense);

    // 4. Run aggregation using the fl-aggregator crate
    let mut agg = Aggregator::new(config);
    for (idx, g) in gradients.into_iter().enumerate() {
        agg.submit(format!("node-{}", idx), g)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    }

    let result = agg.finalize_round()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    // 5. Persist aggregation metadata or emit a signal as needed
    // (e.g., create an AggregatedModel entry, send a signal to participants)

    Ok(AggregatedGradient {
        round_id: input.round_id,
        values: result.to_vec(),
        count: agg.submission_count(),
    })
}
```

### Notes

- `fetch_gradient_bytes` and `bytes_to_gradient` are intentionally left as deployment‑specific:
  - In the current architecture, 0TML or another service is responsible for holding full gradient vectors.
  - The Holochain DHT stores only hashes and metadata for storage and integrity.
- The above function should live in the **core coordinator zome** of the canonical FL hApp (e.g. `0TML/mycelix_fl/holochain/zomes/core/coordinator`).

---

## Implementation Plan

1. **Re-materialise canonical zome sources**
   - Restore or re‑scaffold the core coordinator zome source directories under `0TML/mycelix_fl/holochain/zomes/` if they are currently missing or only present as compiled WASM.

2. **Add the `aggregate_gradients` extern**
   - Implement the extern function as shown above, wired to:
     - `get_round_updates` from the agents/FL zomes
     - A concrete `fetch_gradient_bytes` bridge that calls into 0TML or a dedicated storage service.

3. **Add tests (Tryorama + Python)**
   - Holochain tests:
     - 33% Byzantine scenario using Krum/Median/TrimmedMean.
     - 40–45% scenarios with documented degradation (mirroring Python tests).
   - Python integration tests:
     - Call the Holochain `aggregate_gradients` extern from 0TML to compare results with pure‑Python aggregation for the same gradients.

4. **Update Documentation**
   - Mark the `aggregate_gradients()` gap as “implemented” in `docs/CRITICAL_REVIEW_AND_ROADMAP.md` once tests are green.
   - Reference this design file from the Holochain sections in the architecture docs.

---

---

## Participant Classification (`classify_round_participants`)

To support the “Ring 2” tests (Sybil / PoGQ / MATL), we expose a classification extern that labels each node in a round as honest or Byzantine using the `fl-aggregator` detector and MATL composite scores from `mycelix-sdk`.

### API Types

```rust
use hdk::prelude::*;
use fl_aggregator::detection::{ByzantineDetector, DetectorConfig, Classification};
use fl_aggregator::Gradient;
use mycelix_sdk::matl::ProofOfGradientQuality;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClassifyRoundInput {
    pub round_id: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeClassification {
    pub agent_id: String,
    pub classification: Classification,  // Honest / Byzantine
    pub confidence: f32,
    pub composite_score: f64,           // MATL composite score
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClassifyRoundOutput {
    pub honest: Vec<String>,
    pub byzantine: Vec<String>,
    pub nodes: Vec<NodeClassification>,
}
```

### Extern Skeleton

```rust
#[hdk_extern]
pub fn classify_round_participants(input: ClassifyRoundInput) -> ExternResult<ClassifyRoundOutput> {
    // 1. Fetch ModelUpdate entries for this round (Agents zome)
    let updates = get_round_updates(input.round_id)?; // reuse existing extern
    if updates.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("No updates for round {}", input.round_id)
        )));
    }

    // 2. Resolve gradient hashes → gradient vectors via external bridge
    let mut gradients: Vec<Gradient> = Vec::with_capacity(updates.len());
    let mut node_ids: Vec<String> = Vec::with_capacity(updates.len());

    for update in &updates {
        let raw = fetch_gradient_bytes(&update.weights_hash)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;
        let g = bytes_to_gradient(&raw)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

        gradients.push(g);
        node_ids.push(update.agent_id.clone());
    }

    // 3. Fetch reputations (e.g., from reputation zome)
    let reputations = fetch_reputation_map(&node_ids)?;

    // 4. Instantiate ByzantineDetector (high precision to minimize false positives)
    let config = DetectorConfig::high_precision();
    let mut detector = ByzantineDetector::new(config);

    // 5. Classify each node and compute MATL composite score
    let mut nodes = Vec::with_capacity(node_ids.len());
    for (idx, g) in gradients.iter().enumerate() {
        let node_id = &node_ids[idx];
        let rep = reputations.get(node_id).copied().unwrap_or(0.5);

        // Placeholder PoGQ – in a full implementation, derive from real PoGQ/TCDM/entropy features
        let pogq = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        let composite = pogq.composite_score(rep);

        let result = detector.classify(g, node_id, input.round_id as u64, Some(&gradients));

        nodes.push(NodeClassification {
            agent_id: node_id.clone(),
            classification: result.classification,
            confidence: result.confidence,
            composite_score: composite,
        });
    }

    let honest = nodes
        .iter()
        .filter(|n| n.classification == Classification::Honest)
        .map(|n| n.agent_id.clone())
        .collect();

    let byzantine = nodes
        .iter()
        .filter(|n| n.classification == Classification::Byzantine)
        .map(|n| n.agent_id.clone())
        .collect();

    Ok(ClassifyRoundOutput { honest, byzantine, nodes })
}
```

---

## Round Security Evaluation (`evaluate_round_security`)

To support the “Ring 3” tests (40–45% cartel / coordinated attackers), we add a higher‑level extern that:

- Aggregates per‑node classifications.
- Computes current Byzantine fraction.
- Uses `AdaptiveByzantineThreshold` to assess whether the fraction is acceptable.
- Runs `CartelDetector` to detect colluding groups based on gradient similarity.
- Returns a `SecurityStatus` and lists of suspected cartel and Byzantine agents.

### API Types

```rust
use mycelix_sdk::matl::{
    AdaptiveByzantineThreshold,
    ThresholdRecommendation,
    NetworkStatus,
    CartelDetector,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SecurityStatus {
    Ok,
    Escalate,
    Halt,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoundSecurityMetrics {
    pub byzantine_fraction: f64,
    pub threshold: f64,
    pub network_status: NetworkStatus,
    pub cartel_count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoundSecurityDecision {
    pub status: SecurityStatus,
    pub suspected_cartel: Vec<String>,
    pub suspected_byzantine: Vec<String>,
    pub metrics: RoundSecurityMetrics,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RoundSecurityInput {
    pub round_id: u32,
}
```

### Extern Skeleton

```rust
#[hdk_extern]
pub fn evaluate_round_security(input: RoundSecurityInput) -> ExternResult<RoundSecurityDecision> {
    // 1. Per-node classification
    let classification = classify_round_participants(ClassifyRoundInput {
        round_id: input.round_id,
    })?;

    let total = classification.nodes.len();
    if total == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("No participants for round {}", input.round_id)
        )));
    }

    let byz_count = classification.byzantine.len();
    let byz_fraction = byz_count as f64 / total as f64;

    // 2. Build similarity matrix for CartelDetector
    let updates = get_round_updates(input.round_id)?;
    let mut gradients_by_agent: std::collections::HashMap<String, Gradient> =
        std::collections::HashMap::new();

    for update in &updates {
        let raw = fetch_gradient_bytes(&update.weights_hash)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;
        let g = bytes_to_gradient(&raw)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;
        gradients_by_agent.insert(update.agent_id.clone(), g);
    }

    let mut cartel_detector = CartelDetector::new(0.8, 2); // threshold/min_size tuned from 0TML experiments
    let agents: Vec<String> = gradients_by_agent.keys().cloned().collect();

    for i in 0..agents.len() {
        for j in (i + 1)..agents.len() {
            let a = &agents[i];
            let b = &agents[j];
            let ga = gradients_by_agent.get(a).unwrap();
            let gb = gradients_by_agent.get(b).unwrap();

            let sim = cosine_similarity(ga.view(), gb.view()); // from fl-aggregator or local helper
            cartel_detector.record_similarity(a, b, sim);
        }
    }

    let cartels = cartel_detector.detect();
    let cartel_members = cartel_detector.all_cartel_members();

    // 3. Adaptive Byzantine threshold
    let mut abt = AdaptiveByzantineThreshold::new();
    abt.observe(byz_fraction, total, !cartels.is_empty());
    let rec: ThresholdRecommendation = abt.recommendation();

    let acceptable = abt.is_acceptable(byz_fraction);

    // 4. Map to SecurityStatus
    let status = if !acceptable && !cartels.is_empty() {
        SecurityStatus::Halt
    } else if !acceptable || rec.status != NetworkStatus::Healthy {
        SecurityStatus::Escalate
    } else {
        SecurityStatus::Ok
    };

    Ok(RoundSecurityDecision {
        status,
        suspected_cartel: cartel_members.into_iter().collect(),
        suspected_byzantine: classification.byzantine,
        metrics: RoundSecurityMetrics {
            byzantine_fraction: byz_fraction,
            threshold: rec.current_threshold,
            network_status: rec.status,
            cartel_count: cartels.len(),
        },
    })
}
```

### Integration with Tests

These two externs are designed to line up directly with the Tryorama/Vitest tests you sketched:

- `classify_round_participants` underpins the **Ring 2** tests:
  - Asserting ≥95% recall on Byzantine nodes and <5% FP on honest nodes at ~33% BFT.
- `evaluate_round_security` underpins the **Ring 3** tests:
  - Success mode: `status == Ok` and aggregated gradient is closer to the honest mean than the cartel center at ~45% BFT.
  - Safe failure mode: `status` is `Escalate` or `Halt` and at least one `cartel-*` node is present in `suspected_cartel`.

Once implemented in `0TML/mycelix_fl/holochain/zomes/core/coordinator`, these three externs:

- `aggregate_gradients`
- `classify_round_participants`
- `evaluate_round_security`

will close the Holochain FL loop and allow you to demonstrate 33–45% Byzantine behavior directly at the Rust/Holochain layer, mirroring the rigor already present in 0TML.

---

## Status Summary

- Design and API shapes for aggregation, per-node classification, and round-level security are specified here and aligned with the existing `fl-aggregator` and `mycelix-sdk` crates.
- Implementation in the canonical Holochain FL hApps is **pending** and should be done in the `0TML/mycelix_fl/holochain` tree once source zomes are available.
