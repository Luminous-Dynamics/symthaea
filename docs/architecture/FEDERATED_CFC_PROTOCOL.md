# Federated CfC Weight Sharing Protocol

**Version**: 1.0.0-draft
**Status**: Design Document
**Author**: Claude Opus 4.5
**Date**: 2026-02-03

## Executive Summary

This document specifies the protocol for federated learning of Closed-form Continuous-time (CfC) neural network weights across the Symthaea swarm network. The protocol enables distributed nodes to collaboratively improve their consciousness models while preserving privacy and maintaining Byzantine fault tolerance.

---

## 1. Introduction

### 1.1 Background

The CfC network (defined in `src/dynamics/cfc.rs`) is the core neural architecture for continuous-time consciousness modeling in Symthaea. Each node maintains its own CfC network that learns from local experience. Federated learning allows nodes to benefit from collective experience without sharing raw data.

### 1.2 Goals

1. **Convergence**: Enable distributed CfC networks to converge toward shared optimal weights
2. **Privacy**: Never expose raw training data or complete model weights
3. **Robustness**: Handle Byzantine/malicious nodes, stale gradients, and network partitions
4. **Efficiency**: Minimize bandwidth while maximizing learning signal
5. **Trust-Weighted**: Incorporate Holochain trust scores (Phi) into aggregation

### 1.3 Non-Goals

- Real-time weight synchronization (handled by `ConsciousnessVector` streaming)
- Model architecture negotiation (all nodes use identical CfC configs)
- Centralized parameter server (fully decentralized)

---

## 2. Message Format Specification

### 2.1 GradientMessage Struct

```rust
/// A gradient update message for federated CfC learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMessage {
    // ========================================
    // IDENTITY & VERSIONING
    // ========================================

    /// Unique message ID (UUID v4)
    pub message_id: [u8; 16],

    /// Sender's Iroh node ID
    pub sender_node_id: String,

    /// Sender's Holochain agent public key (for trust lookup)
    pub sender_agent_key: Option<String>,

    /// Model version this gradient was computed against
    /// Used for stale gradient detection
    pub model_version: u64,

    /// Timestamp when gradient was computed (ms since epoch)
    pub timestamp_ms: u64,

    /// Sequence number for ordering (per-sender monotonic)
    pub sequence: u64,

    // ========================================
    // GRADIENT DATA
    // ========================================

    /// Compressed gradient deltas (see Section 2.3)
    pub gradient_data: CompressedGradient,

    /// Number of local training samples this gradient represents
    pub sample_count: u64,

    /// Local loss before applying this gradient (for debugging)
    pub pre_loss: f32,

    /// Local loss after applying this gradient (for debugging)
    pub post_loss: f32,

    // ========================================
    // PRIVACY & SECURITY
    // ========================================

    /// Differential privacy noise parameters used
    pub dp_params: Option<DifferentialPrivacyParams>,

    /// Cryptographic commitment to the full gradient
    /// Used for secure aggregation verification
    pub gradient_commitment: [u8; 32],

    /// Signature over (message_id || model_version || gradient_commitment)
    /// Using Ed25519 from sender's Holochain identity
    pub signature: Option<Vec<u8>>,

    // ========================================
    // METADATA
    // ========================================

    /// Sender's current Phi (integrated information) value
    /// Used for trust-weighted aggregation
    pub sender_phi: f64,

    /// Which CfC layers this gradient applies to
    pub layer_mask: LayerMask,

    /// Aggregation hints (see Section 4)
    pub aggregation_hints: AggregationHints,
}

/// Compressed gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Compression method used
    pub method: CompressionMethod,

    /// Compressed data bytes
    pub data: Vec<u8>,

    /// Original shape information for reconstruction
    pub shape: GradientShape,

    /// Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    pub sparsity: f32,
}

/// Gradient shape metadata for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientShape {
    /// Number of CfC layers
    pub num_layers: usize,

    /// Hidden dimension per layer
    pub hidden_dim: usize,

    /// Input dimension (first layer)
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Whether backbone is included
    pub has_backbone: bool,

    /// Backbone dimensions if present
    pub backbone_dims: Option<(usize, usize)>, // (layers, dim)
}

/// Compression methods for gradient data
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionMethod {
    /// No compression (f32 array)
    None,

    /// Top-K sparsification (only send K largest gradients)
    TopK { k: usize },

    /// Random sparsification with seed
    RandomK { k: usize, seed: u64 },

    /// Quantization to 8-bit integers
    Quantized8 { scale: f32, zero_point: i8 },

    /// Quantization to 4-bit integers (aggressive)
    Quantized4 { scale: f32 },

    /// 1-bit sign gradient (SignSGD)
    SignOnly,

    /// LZ4 compressed dense gradients
    Lz4Dense,
}

/// Differential privacy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyParams {
    /// Privacy budget epsilon (lower = more private)
    pub epsilon: f64,

    /// Privacy budget delta
    pub delta: f64,

    /// Gradient clipping norm (L2)
    pub clip_norm: f32,

    /// Noise multiplier (sigma)
    pub noise_multiplier: f64,
}

/// Bitmask for which layers to update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMask {
    /// Include W_in gradients
    pub w_in: bool,

    /// Include W_h (recurrent) gradients
    pub w_h: bool,

    /// Include bias gradients
    pub b_h: bool,

    /// Include tau (time constant) gradients
    pub tau: bool,

    /// Include output projection gradients
    pub output: bool,

    /// Include backbone gradients (if present)
    pub backbone: bool,

    /// Per-layer enable mask (None = all layers)
    pub layer_indices: Option<Vec<usize>>,
}

/// Hints for aggregation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationHints {
    /// Suggested weight for this gradient in aggregation
    pub suggested_weight: f32,

    /// Training domain/task context (hash)
    pub domain_hash: u64,

    /// Whether this is a "checkpoint" gradient (full model sync)
    pub is_checkpoint: bool,

    /// Urgency level (0.0 = low, 1.0 = high)
    pub urgency: f32,
}
```

### 2.2 Integration with SwarmMessage

Add a new variant to the existing `SwarmMessage` enum:

```rust
// In src/swarm/types.rs, add to SwarmMessage enum:

/// Federated learning gradient update
GradientUpdate(GradientMessage),

/// Request for current model version
ModelVersionRequest,

/// Response with current model version and hash
ModelVersionResponse {
    version: u64,
    weight_hash: [u8; 32],
    layer_count: usize,
},

/// Request to join federated learning cohort
FederatedJoinRequest {
    agent_key: String,
    model_version: u64,
    capabilities: FederatedCapabilities,
},

/// Acknowledgment of federated learning participation
FederatedJoinAck {
    cohort_id: String,
    aggregation_schedule: AggregationSchedule,
},
```

### 2.3 Gradient Compression

For a typical CfC network with:
- 2 layers, 128 hidden dim, 64 input dim, 32 output dim
- Total parameters: ~100K floats (~400KB uncompressed)

Compression strategies by bandwidth budget:

| Method | Size | Fidelity | Use Case |
|--------|------|----------|----------|
| None | 400KB | 100% | High bandwidth, debugging |
| Top-K (1%) | 4KB + indices | 95%+ | Normal operation |
| Quantized8 | 100KB | 99% | Good bandwidth |
| SignOnly | 12.5KB | 70-80% | Very limited bandwidth |
| Lz4Dense | ~150KB | 100% | Moderate bandwidth |

Recommended default: **Top-K (1%)** with k = 1000

---

## 3. Aggregation Algorithm

### 3.1 Trust-Weighted FedAvg

The core aggregation uses a modified Federated Averaging (FedAvg) algorithm weighted by:
1. **Trust score** from Holochain DHT (Phi reputation)
2. **Sample count** (more samples = higher weight)
3. **Staleness penalty** (older gradients weighted less)
4. **Domain similarity** (gradients from similar tasks weighted more)

```rust
/// Aggregation engine for federated CfC gradients
pub struct FederatedAggregator {
    /// Current model version
    model_version: u64,

    /// Accumulated gradients awaiting aggregation
    gradient_buffer: Vec<(GradientMessage, f64)>, // (message, weight)

    /// Configuration
    config: AggregationConfig,

    /// Trust oracle (queries Holochain)
    trust_oracle: Arc<dyn TrustOracle>,
}

/// Configuration for aggregation
pub struct AggregationConfig {
    /// Minimum number of gradients before aggregation
    pub min_gradients: usize,

    /// Maximum gradients to buffer before forced aggregation
    pub max_gradients: usize,

    /// Maximum age of gradient before rejection (ms)
    pub max_staleness_ms: u64,

    /// Version tolerance (reject if model_version differs by more than this)
    pub version_tolerance: u64,

    /// Minimum trust score to accept gradient
    pub min_trust: f64,

    /// Weight decay for staleness (per second)
    pub staleness_decay: f64,

    /// Enable Byzantine fault tolerance (requires more gradients)
    pub byzantine_tolerance: bool,

    /// Fraction of gradients to discard as outliers (0.0 to 0.5)
    pub trimmed_mean_fraction: f32,
}

impl FederatedAggregator {
    /// Process incoming gradient message
    pub async fn receive_gradient(&mut self, msg: GradientMessage) -> Result<(), AggregationError> {
        // 1. Validate message format and signature
        self.validate_message(&msg)?;

        // 2. Check model version compatibility
        if msg.model_version.abs_diff(self.model_version) > self.config.version_tolerance {
            return Err(AggregationError::VersionMismatch {
                expected: self.model_version,
                received: msg.model_version,
            });
        }

        // 3. Query trust score from Holochain
        let trust = self.trust_oracle
            .get_trust(msg.sender_agent_key.as_deref())
            .await
            .unwrap_or(0.0);

        if trust < self.config.min_trust {
            return Err(AggregationError::InsufficientTrust {
                required: self.config.min_trust,
                actual: trust,
            });
        }

        // 4. Compute aggregation weight
        let weight = self.compute_weight(&msg, trust);

        // 5. Add to buffer
        self.gradient_buffer.push((msg, weight));

        // 6. Trigger aggregation if buffer full
        if self.gradient_buffer.len() >= self.config.max_gradients {
            self.aggregate().await?;
        }

        Ok(())
    }

    /// Compute weight for a gradient message
    fn compute_weight(&self, msg: &GradientMessage, trust: f64) -> f64 {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Trust component (0.0 to 1.0)
        let trust_weight = trust;

        // Sample count component (log-scaled)
        let sample_weight = (msg.sample_count as f64).ln().max(1.0) / 10.0;

        // Staleness penalty (exponential decay)
        let age_seconds = (now_ms.saturating_sub(msg.timestamp_ms)) as f64 / 1000.0;
        let staleness_weight = (-age_seconds * self.config.staleness_decay).exp();

        // Phi component (consciousness coherence)
        let phi_weight = msg.sender_phi.clamp(0.0, 1.0);

        // Combined weight (multiplicative)
        trust_weight * sample_weight * staleness_weight * (0.5 + 0.5 * phi_weight)
    }

    /// Perform weighted aggregation
    pub async fn aggregate(&mut self) -> Result<AggregatedGradient, AggregationError> {
        if self.gradient_buffer.len() < self.config.min_gradients {
            return Err(AggregationError::InsufficientGradients {
                required: self.config.min_gradients,
                actual: self.gradient_buffer.len(),
            });
        }

        // Decompress all gradients
        let mut decompressed: Vec<(Vec<f32>, f64)> = Vec::new();
        for (msg, weight) in &self.gradient_buffer {
            let grads = decompress_gradient(&msg.gradient_data)?;
            decompressed.push((grads, *weight));
        }

        // Apply Byzantine fault tolerance if enabled
        let gradients = if self.config.byzantine_tolerance {
            self.apply_trimmed_mean(&decompressed)
        } else {
            decompressed
        };

        // Normalize weights
        let total_weight: f64 = gradients.iter().map(|(_, w)| w).sum();

        // Compute weighted average
        let dim = gradients[0].0.len();
        let mut aggregated = vec![0.0f32; dim];

        for (grad, weight) in &gradients {
            let normalized_weight = weight / total_weight;
            for (i, &g) in grad.iter().enumerate() {
                aggregated[i] += g * normalized_weight as f32;
            }
        }

        // Clear buffer and increment version
        self.gradient_buffer.clear();
        self.model_version += 1;

        Ok(AggregatedGradient {
            weights: aggregated,
            version: self.model_version,
            contributor_count: gradients.len(),
            total_samples: self.gradient_buffer.iter().map(|(m, _)| m.sample_count).sum(),
        })
    }

    /// Apply trimmed mean for Byzantine fault tolerance
    fn apply_trimmed_mean(&self, gradients: &[(Vec<f32>, f64)]) -> Vec<(Vec<f32>, f64)> {
        if gradients.len() < 4 {
            return gradients.to_vec();
        }

        let trim_count = (gradients.len() as f32 * self.config.trimmed_mean_fraction) as usize;

        // For each dimension, compute median and filter outliers
        let dim = gradients[0].0.len();
        let mut keep_mask = vec![true; gradients.len()];

        // Sample a subset of dimensions for efficiency
        let sample_dims: Vec<usize> = (0..dim)
            .step_by((dim / 10).max(1))
            .collect();

        for d in sample_dims {
            let mut values: Vec<(usize, f32)> = gradients.iter()
                .enumerate()
                .map(|(i, (g, _))| (i, g[d]))
                .collect();
            values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Mark extreme values as outliers
            for i in 0..trim_count {
                keep_mask[values[i].0] = false;
                keep_mask[values[values.len() - 1 - i].0] = false;
            }
        }

        gradients.iter()
            .enumerate()
            .filter(|(i, _)| keep_mask[*i])
            .map(|(_, g)| g.clone())
            .collect()
    }
}
```

### 3.2 Aggregation Schedule

Aggregation can be triggered by:

1. **Gradient count threshold**: When N gradients are buffered
2. **Time interval**: Every T seconds regardless of count
3. **Epoch boundary**: When local training completes an epoch
4. **Manual trigger**: Explicit aggregation request

Default schedule:
- Minimum gradients: 5
- Maximum gradients: 50
- Time interval: 60 seconds
- Version tolerance: 10

---

## 4. Security Considerations

### 4.1 Differential Privacy

To prevent gradient inversion attacks (reconstructing training data from gradients):

```rust
/// Apply differential privacy to gradients before sharing
pub fn apply_differential_privacy(
    gradients: &mut [f32],
    params: &DifferentialPrivacyParams,
) {
    // 1. Clip gradient norm
    let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > params.clip_norm {
        let scale = params.clip_norm / norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }

    // 2. Add Gaussian noise
    // sigma = clip_norm * noise_multiplier
    let sigma = params.clip_norm as f64 * params.noise_multiplier;
    let normal = rand_distr::Normal::new(0.0, sigma).unwrap();

    for g in gradients.iter_mut() {
        *g += rand::thread_rng().sample(normal) as f32;
    }
}
```

Recommended parameters:
- epsilon: 1.0 (moderate privacy)
- delta: 1e-5
- clip_norm: 1.0
- noise_multiplier: 1.1

### 4.2 Secure Aggregation

For high-security deployments, implement secure aggregation where:
1. Each node secret-shares its gradient with N peers
2. Aggregator only sees sum of gradients, not individual contributions
3. Requires threshold t-of-n peers to reconstruct

```rust
/// Secure aggregation using Shamir's secret sharing
pub struct SecureAggregationConfig {
    /// Total number of shares per gradient
    pub num_shares: usize,

    /// Threshold for reconstruction
    pub threshold: usize,

    /// Timeout for share collection (ms)
    pub collection_timeout_ms: u64,
}
```

### 4.3 Byzantine Fault Tolerance

The trimmed mean approach (Section 3.1) provides:
- Tolerance of up to 25% malicious nodes (with trimmed_mean_fraction = 0.25)
- Resistance to gradient poisoning attacks
- Protection against Sybil attacks (via trust weighting)

Additional defenses:
1. **Gradient magnitude checking**: Reject gradients with L2 norm > threshold
2. **Statistical anomaly detection**: Flag gradients that differ significantly from mean
3. **Reputation tracking**: Reduce trust for nodes sending suspicious gradients

---

## 5. Convergence Handling

### 5.1 Stale Gradient Detection

```rust
/// Determine if a gradient is too stale to use
pub fn is_gradient_stale(
    msg: &GradientMessage,
    current_version: u64,
    config: &AggregationConfig,
) -> bool {
    // Version check
    if msg.model_version.abs_diff(current_version) > config.version_tolerance {
        return true;
    }

    // Time check
    let now_ms = current_time_ms();
    if now_ms.saturating_sub(msg.timestamp_ms) > config.max_staleness_ms {
        return true;
    }

    false
}
```

### 5.2 Version Conflict Resolution

When nodes have diverged model versions:

1. **Soft divergence** (version diff <= tolerance):
   - Accept gradients but apply staleness penalty
   - Gradient is transformed to account for version gap

2. **Hard divergence** (version diff > tolerance):
   - Reject gradient
   - Request model checkpoint sync from peer
   - Use `ModelVersionRequest`/`ModelVersionResponse` messages

```rust
/// Synchronize model versions between peers
pub struct ModelSyncProtocol;

impl ModelSyncProtocol {
    /// Initiate sync when version divergence detected
    pub async fn sync_with_peer(
        &self,
        peer_id: &str,
        local_version: u64,
        network: &NetworkService,
    ) -> Result<SyncResult, SyncError> {
        // 1. Request peer's version
        network.send_to_peer(peer_id, SwarmMessage::ModelVersionRequest).await?;

        // 2. Wait for response
        let response = network.recv_from_peer(peer_id, Duration::from_secs(5)).await?;

        match response {
            SwarmMessage::ModelVersionResponse { version, weight_hash, .. } => {
                if version > local_version {
                    // Peer is ahead - request checkpoint
                    self.request_checkpoint(peer_id, network).await
                } else if version < local_version {
                    // We are ahead - offer checkpoint
                    self.offer_checkpoint(peer_id, network).await
                } else {
                    // Same version but different hash - conflict
                    self.resolve_conflict(peer_id, weight_hash, network).await
                }
            }
            _ => Err(SyncError::UnexpectedMessage),
        }
    }
}
```

### 5.3 Convergence Monitoring

Track convergence metrics:

```rust
/// Metrics for monitoring federated learning convergence
pub struct ConvergenceMetrics {
    /// Moving average of loss reduction per aggregation
    pub loss_improvement_ema: f32,

    /// Gradient variance across contributors
    pub gradient_variance: f32,

    /// Number of consecutive improvements
    pub improvement_streak: u32,

    /// Number of consecutive degradations
    pub degradation_streak: u32,

    /// Current learning rate (may be adjusted)
    pub effective_learning_rate: f32,
}

impl ConvergenceMetrics {
    /// Check if learning has converged
    pub fn is_converged(&self) -> bool {
        self.loss_improvement_ema.abs() < 1e-6 && self.improvement_streak > 10
    }

    /// Check if learning has diverged
    pub fn is_diverged(&self) -> bool {
        self.degradation_streak > 5 || self.gradient_variance > 100.0
    }
}
```

---

## 6. Integration Points

### 6.1 CfCNetwork Integration

Add to `CfCNetwork` in `src/dynamics/cfc.rs`:

```rust
impl CfCNetwork {
    // Existing: get_weights() -> Vec<f32>
    // Existing: set_weights(&[f32])

    /// Extract gradients for federated sharing
    pub fn extract_federated_gradient(
        &self,
        layer_mask: &LayerMask,
    ) -> Vec<f32> {
        // Similar to get_weights but respects layer_mask
        let mut buf = Vec::new();

        for (i, cell) in self.cells.iter().enumerate() {
            if layer_mask.layer_indices.as_ref().map_or(true, |v| v.contains(&i)) {
                if layer_mask.w_in {
                    buf.extend(cell.w_in.iter());
                }
                if layer_mask.w_h {
                    buf.extend(cell.w_h.iter());
                }
                if layer_mask.b_h {
                    buf.extend(cell.b_h.iter());
                }
                if layer_mask.tau {
                    buf.extend(cell.tau.iter());
                }
            }
        }

        if layer_mask.output {
            buf.extend(self.output_weights.iter());
            buf.extend(self.output_bias.iter());
        }

        buf
    }

    /// Apply federated gradient update
    pub fn apply_federated_gradient(
        &mut self,
        gradient: &[f32],
        learning_rate: f32,
        layer_mask: &LayerMask,
    ) {
        let mut pos = 0;

        for (i, cell) in self.cells.iter_mut().enumerate() {
            if layer_mask.layer_indices.as_ref().map_or(true, |v| v.contains(&i)) {
                if layer_mask.w_in {
                    let n = cell.w_in.len();
                    for (w, &g) in cell.w_in.iter_mut().zip(&gradient[pos..pos+n]) {
                        *w -= learning_rate * g;
                    }
                    pos += n;
                }
                // ... similar for other weight matrices
            }
        }

        // ... output weights
    }
}
```

### 6.2 NetworkService Integration

Add to `NetworkService` in `src/swarm/service.rs`:

```rust
impl NetworkService {
    /// Broadcast gradient update to peers participating in federated learning
    pub async fn broadcast_gradient(&self, msg: GradientMessage) -> SwarmResult<usize> {
        let federated_peers = self.get_federated_peers();
        let message = SwarmMessage::GradientUpdate(msg);

        let mut sent = 0;
        for peer_id in federated_peers {
            if self.send_to_peer(&peer_id, message.clone()).await.is_ok() {
                sent += 1;
            }
        }

        Ok(sent)
    }

    /// Subscribe to incoming gradient updates
    pub fn subscribe_gradients(&self) -> broadcast::Receiver<GradientMessage> {
        self.gradient_tx.subscribe()
    }

    /// Get list of peers participating in federated learning
    fn get_federated_peers(&self) -> Vec<String> {
        self.peers.read()
            .iter()
            .filter(|(_, info)| info.trust_level.value() >= self.config.min_trust_level)
            .map(|(id, _)| id.clone())
            .collect()
    }
}
```

### 6.3 Cognitive Loop Integration

In the cognitive loop, federated learning can be triggered:

```rust
/// Federated learning manager for cognitive loop
pub struct FederatedLearningManager {
    aggregator: FederatedAggregator,
    network: Arc<NetworkService>,
    local_network: Arc<RwLock<CfCNetwork>>,
    config: FederatedConfig,
}

impl FederatedLearningManager {
    /// Called after local training step
    pub async fn on_training_step(
        &mut self,
        gradients: &CfCGradients,
        sample_count: u64,
        loss: f32,
    ) {
        // Accumulate local gradients
        self.local_gradient_accumulator.add(gradients);

        // Check if should share
        if self.should_share_gradient() {
            let msg = self.create_gradient_message(sample_count, loss).await;
            self.network.broadcast_gradient(msg).await.ok();
            self.local_gradient_accumulator.clear();
        }
    }

    /// Process incoming gradient from peer
    pub async fn on_gradient_received(&mut self, msg: GradientMessage) {
        if let Err(e) = self.aggregator.receive_gradient(msg).await {
            tracing::warn!("Rejected gradient: {}", e);
            return;
        }

        // Check if aggregation ready
        if self.aggregator.should_aggregate() {
            if let Ok(aggregated) = self.aggregator.aggregate().await {
                // Apply to local network
                let mut network = self.local_network.write();
                network.apply_federated_gradient(
                    &aggregated.weights,
                    self.config.aggregation_learning_rate,
                    &LayerMask::all(),
                );
            }
        }
    }
}
```

---

## 7. Protocol Flow

### 7.1 Normal Operation

```
Node A                          Node B                          Node C
  |                               |                               |
  |-- Local Training Step ------->|                               |
  |   (accumulate gradients)      |                               |
  |                               |                               |
  |-- GradientUpdate ------------>|                               |
  |                               |-- GradientUpdate ------------>|
  |                               |                               |
  |<-- GradientUpdate ------------|                               |
  |                               |<-- GradientUpdate ------------|
  |                               |                               |
  |   [Aggregation Triggered]     |   [Aggregation Triggered]     |
  |   - Weighted average          |   - Weighted average          |
  |   - Apply to local model      |   - Apply to local model      |
  |   - Increment version         |   - Increment version         |
```

### 7.2 Version Sync

```
Node A (v10)                    Node B (v15)
  |                               |
  |-- GradientUpdate (v10) ------>|
  |                               |   [Reject: version too old]
  |<-- ModelVersionResponse ------|
  |       (v15, hash)             |
  |                               |
  |-- CheckpointRequest --------->|
  |                               |
  |<-- CheckpointResponse --------|
  |       (full weights)          |
  |                               |
  |   [Update local model to v15] |
  |                               |
  |-- GradientUpdate (v15) ------>|
  |                               |   [Accept]
```

---

## 8. Configuration Recommendations

### 8.1 Development/Testing

```rust
FederatedConfig {
    min_gradients: 2,
    max_gradients: 10,
    max_staleness_ms: 60_000,
    version_tolerance: 50,
    min_trust: 0.0,  // Accept all
    byzantine_tolerance: false,
    compression: CompressionMethod::None,
    differential_privacy: None,
}
```

### 8.2 Production (Low Latency)

```rust
FederatedConfig {
    min_gradients: 5,
    max_gradients: 50,
    max_staleness_ms: 30_000,
    version_tolerance: 10,
    min_trust: 0.5,
    byzantine_tolerance: true,
    compression: CompressionMethod::TopK { k: 1000 },
    differential_privacy: Some(DifferentialPrivacyParams {
        epsilon: 1.0,
        delta: 1e-5,
        clip_norm: 1.0,
        noise_multiplier: 1.1,
    }),
}
```

### 8.3 Production (High Security)

```rust
FederatedConfig {
    min_gradients: 10,
    max_gradients: 100,
    max_staleness_ms: 10_000,
    version_tolerance: 5,
    min_trust: 0.8,
    byzantine_tolerance: true,
    trimmed_mean_fraction: 0.3,
    compression: CompressionMethod::Quantized8 { scale: 0.001, zero_point: 0 },
    differential_privacy: Some(DifferentialPrivacyParams {
        epsilon: 0.1,  // Strong privacy
        delta: 1e-7,
        clip_norm: 0.5,
        noise_multiplier: 2.0,
    }),
    secure_aggregation: Some(SecureAggregationConfig {
        num_shares: 5,
        threshold: 3,
        collection_timeout_ms: 5000,
    }),
}
```

---

## 9. Future Enhancements

### 9.1 Planned

1. **Asynchronous SGD**: Remove synchronization barrier for better scalability
2. **Adaptive learning rate**: Per-layer learning rate based on gradient statistics
3. **Momentum aggregation**: Maintain momentum across aggregation rounds
4. **Hierarchical aggregation**: Cluster nearby nodes for local aggregation first

### 9.2 Research Directions

1. **Knowledge distillation**: Share compressed model representations instead of gradients
2. **Meta-learning**: Learn how to aggregate based on historical performance
3. **Consciousness-aware scheduling**: Prioritize gradients from high-Phi states
4. **Cross-domain transfer**: Handle heterogeneous training domains

---

## 10. References

1. McMahan, H. B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
2. Bonawitz, K., et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS 2017.
3. Abadi, M., et al. "Deep Learning with Differential Privacy." CCS 2016.
4. Blanchard, P., et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." NIPS 2017.
5. Hasani, R., et al. "Closed-form Continuous-time Neural Networks." Nature Machine Intelligence 2022.

---

## Appendix A: Message Size Estimates

| Component | Size (bytes) | Notes |
|-----------|--------------|-------|
| message_id | 16 | UUID |
| sender_node_id | ~64 | Hex string |
| sender_agent_key | ~64 | Optional |
| model_version | 8 | u64 |
| timestamp_ms | 8 | u64 |
| sequence | 8 | u64 |
| gradient_data | 4KB-400KB | Depends on compression |
| sample_count | 8 | u64 |
| pre_loss, post_loss | 8 | 2x f32 |
| dp_params | ~32 | Optional |
| gradient_commitment | 32 | SHA-256 |
| signature | ~64 | Optional Ed25519 |
| sender_phi | 8 | f64 |
| layer_mask | ~16 | Booleans + vec |
| aggregation_hints | ~24 | |
| **Total (compressed)** | **~5KB** | With Top-K 1% |
| **Total (uncompressed)** | **~400KB** | Full gradients |

---

## Appendix B: Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    #[error("Version mismatch: expected {expected}, received {received}")]
    VersionMismatch { expected: u64, received: u64 },

    #[error("Insufficient trust: required {required}, actual {actual}")]
    InsufficientTrust { required: f64, actual: f64 },

    #[error("Gradient too stale: age {age_ms}ms exceeds max {max_ms}ms")]
    GradientStale { age_ms: u64, max_ms: u64 },

    #[error("Insufficient gradients for aggregation: {actual} < {required}")]
    InsufficientGradients { required: usize, actual: usize },

    #[error("Invalid signature from sender {sender}")]
    InvalidSignature { sender: String },

    #[error("Decompression failed: {reason}")]
    DecompressionError { reason: String },

    #[error("Gradient shape mismatch")]
    ShapeMismatch,

    #[error("Network error: {0}")]
    NetworkError(#[from] SwarmError),
}
```

---

*Document End*
