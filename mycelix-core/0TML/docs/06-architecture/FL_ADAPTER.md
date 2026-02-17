# Federated Learning Adapter Architecture

## Overview

The Federated Learning (FL) adapter is the flagship industry adapter for the Zero-TrustML Meta-Framework. It demonstrates how the universal primitives (Quality, Security, Validation, Contribution) can be instantiated for decentralized machine learning.

## Core Innovation

Traditional federated learning relies on a central server to coordinate training and aggregate model updates. This creates:
- **Single point of failure**
- **Censorship vector**
- **Trust assumption** in the coordinator

The Zero-TrustML FL adapter eliminates these issues through:

1. **Byzantine-Resistant Aggregation** - Correct model even with malicious participants
2. **Cryptoeconomic Incentives** - Economic alignment for honest behavior
3. **Fully P2P Architecture** - No central coordinator required

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Nodes                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node 4  │  ...  │
│  │ (Alice) │  │ (Bob)   │  │(Charlie)│  │ (Dave)  │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                         │                                   │
│                         ▼                                   │
│              ┌────────────────────┐                         │
│              │  Holochain DNA     │                         │
│              │  "FL Coordinator"  │                         │
│              └────────────────────┘                         │
│                         │                                   │
│       ┌─────────────────┴─────────────────┐                │
│       ▼                                   ▼                │
│  ┌─────────────────┐            ┌─────────────────┐        │
│  │ Validator Nodes │            │  Hierarchical   │        │
│  │  (VRF-selected) │            │   Aggregation   │        │
│  └─────────────────┘            └─────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Proof of Quality Gradient (PoGQ)

### Concept

PoGQ is a cryptoeconomic mechanism that:
1. Quantifies the quality of a submitted gradient
2. Enables decentralized validation
3. Provides basis for reputation and rewards

### Process

```python
class PoGQValidator:
    """
    Validates gradient quality using private test data.
    """

    def __init__(self, test_dataset, model_architecture):
        self.test_data = test_dataset
        self.model = model_architecture

    def compute_pogq(self, gradient):
        """
        Compute Proof of Quality Gradient score.

        Args:
            gradient: Model parameter updates from training

        Returns:
            PoGQScore with accuracy, loss, and proof
        """
        # 1. Apply gradient to current model
        updated_model = self.model.apply_gradient(gradient)

        # 2. Evaluate on private test set
        accuracy = updated_model.evaluate(self.test_data)
        loss = updated_model.compute_loss(self.test_data)

        # 3. Generate cryptographic proof
        proof = self._generate_proof(gradient, accuracy)

        return PoGQScore(
            accuracy=accuracy,
            loss=loss,
            proof=proof,
            validator_signature=self.sign(proof)
        )

    def _generate_proof(self, gradient, accuracy):
        """
        Create cryptographic commitment to validation.

        Future: Replace with ZK-SNARK for privacy.
        """
        commitment = hash(
            gradient.hash() +
            str(accuracy) +
            self.test_data.hash()
        )
        return Proof(commitment=commitment)
```

### Validation Dataset Strategy

**Challenge:** Where does the private test data come from?

**Solution: Multi-Phase Approach**

**Phase 1: Bootstrap (Current)**
```python
# Use well-established public datasets
test_data = load_mnist_validation_set()
```

**Phase 2: DAO Curation**
```python
# Community proposes and vets new datasets
proposal = DatasetProposal(
    name="Medical Imaging Validation v2",
    ipfs_hash="Qm...",  # Stored on IPFS
    description="Curated by DAO vote #42"
)

# DAO votes to approve
if dao.vote(proposal) == Vote.APPROVED:
    approved_datasets.add(proposal)
```

**Phase 3: Privacy-Preserving (Future)**
```python
# Use ZK-proofs to validate without revealing test data
zk_proof = validator.prove_quality(
    gradient=gradient,
    secret_test_data=private_data,
    public_parameters=params
)

# Anyone can verify without seeing private_data
assert verify_zk_proof(zk_proof)
```

## Hierarchical Federated Learning

### Problem: Communication Complexity

**Naive P2P Approach:**
- n nodes, each sends gradient to n-1 others
- Total communication: O(n²)
- Infeasible for large n

**Example:**
```
100 nodes × 100MB gradient × 99 recipients
= 990 GB total network traffic per round
```

### Solution: Hierarchical Aggregation

**Two-Tier Architecture:**

```
Round t:
  ├─ Phase 1: Intra-Cluster Aggregation
  │   ├─ Cluster 1 (10 nodes) → Aggregator 1
  │   ├─ Cluster 2 (10 nodes) → Aggregator 2
  │   ├─ Cluster 3 (10 nodes) → Aggregator 3
  │   └─ ... (10 clusters total)
  │
  └─ Phase 2: Inter-Cluster Aggregation
      ├─ 10 Aggregators exchange results
      └─ Global model update produced
```

**Communication Reduction:**

```
Hierarchical:
  10 nodes/cluster × 100MB × 1 aggregator
    = 1 GB per cluster

  10 aggregators × 100MB × 9 other aggregators
    = 9 GB inter-cluster

  Total: ~10 GB (99x improvement!)
```

### Implementation

```python
class HierarchicalFL:
    """
    Implements communication-efficient hierarchical FL.
    """

    def __init__(self, num_nodes, cluster_size=10):
        self.num_nodes = num_nodes
        self.cluster_size = cluster_size
        self.num_clusters = num_nodes // cluster_size

    def run_round(self, round_num):
        """
        Execute one round of hierarchical FL.
        """
        # Step 1: Form clusters (dynamic per round)
        clusters = self._form_clusters(round_num)

        # Step 2: Select cluster aggregators (VRF)
        aggregators = self._select_aggregators(clusters)

        # Step 3: Intra-cluster aggregation
        cluster_results = []
        for cluster, aggregator in zip(clusters, aggregators):
            # Nodes send gradients to their aggregator
            gradients = [node.get_gradient() for node in cluster]

            # Aggregator performs Byzantine-resistant aggregation
            cluster_gradient = aggregator.aggregate(
                gradients,
                method="krum"  # or bulyan, median, etc.
            )
            cluster_results.append(cluster_gradient)

        # Step 4: Inter-cluster aggregation
        global_gradient = self._aggregate_cluster_results(
            cluster_results,
            method="weighted_average"
        )

        # Step 5: Distribute global model update
        self._broadcast_update(global_gradient)

        return global_gradient

    def _form_clusters(self, round_num):
        """
        Dynamically form clusters based on:
        - Geographic proximity
        - Reputation similarity
        - Random shuffling (prevents collusion)
        """
        # Mix of deterministic and random
        seed = hash(str(round_num) + "cluster_formation")
        random.seed(seed)

        nodes = list(self.all_nodes)
        random.shuffle(nodes)

        clusters = [
            nodes[i:i + self.cluster_size]
            for i in range(0, len(nodes), self.cluster_size)
        ]
        return clusters

    def _select_aggregators(self, clusters):
        """
        Use VRF to select one aggregator per cluster.
        """
        aggregators = []
        for cluster in clusters:
            # Each node computes VRF
            vrf_outputs = [
                node.compute_vrf(round_num)
                for node in cluster
            ]

            # Lowest VRF output wins
            aggregator = min(
                cluster,
                key=lambda node: vrf_outputs[cluster.index(node)]
            )
            aggregators.append(aggregator)

        return aggregators
```

## VRF-Based Validator Selection

### Problem: Predictable Selection is Vulnerable

If selection is deterministic:
- Attackers know who to target/bribe
- Colluding nodes can coordinate
- Single point of attack

### Solution: Verifiable Random Function

**Properties of VRF:**
1. **Unpredictable**: Output appears random
2. **Verifiable**: Anyone can check it was computed correctly
3. **Deterministic**: Same input always produces same output
4. **Unforgeable**: Only holder of secret key can compute

**Selection Process:**

```python
class VRFValidator:
    """
    VRF-based validator selection.
    """

    def __init__(self, secret_key, public_key):
        self.sk = secret_key
        self.pk = public_key

    def compute_vrf(self, round_num, seed):
        """
        Compute VRF output for this round.

        Args:
            round_num: Current FL round number
            seed: Public randomness (e.g., hash of previous model)

        Returns:
            (output, proof)
        """
        # Input is public seed + round number
        vrf_input = hash(str(seed) + str(round_num))

        # Only this node can compute output with their SK
        vrf_output = vrf_hash(self.sk, vrf_input)

        # Generate proof that output is correct
        proof = vrf_prove(self.sk, vrf_input)

        return (vrf_output, proof)

    def verify_vrf(self, output, proof, public_key, vrf_input):
        """
        Anyone can verify the VRF output.
        """
        return vrf_verify(public_key, vrf_input, output, proof)


class ValidatorSelection:
    """
    Select validators using reputation-weighted VRF lottery.
    """

    def select_validators(self, all_nodes, num_validators, round_num, seed):
        """
        Select num_validators from all_nodes using VRF.
        """
        # Step 1: Filter by minimum reputation
        eligible = [
            node for node in all_nodes
            if node.reputation >= MIN_REPUTATION
        ]

        # Step 2: Each eligible node computes VRF
        vrf_results = []
        for node in eligible:
            output, proof = node.compute_vrf(round_num, seed)

            # Verify proof (reject if invalid)
            if not node.verify_vrf(output, proof, node.pk, seed):
                continue

            # Weight by reputation
            weighted_output = output / node.reputation

            vrf_results.append((node, weighted_output, proof))

        # Step 3: Select top num_validators by weighted VRF output
        selected = sorted(vrf_results, key=lambda x: x[1])[:num_validators]

        return [node for node, _, _ in selected]
```

## Byzantine Attack Mitigation

### Attack Types

| Attack | Description | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Gaussian Noise** | Add random noise to gradients | Degrades accuracy | PoGQ filters low-quality |
| **Sign Flip** | Reverse gradient direction | Prevents convergence | Krum/Bulyan detect outliers |
| **Label Flip** | Train on incorrect labels | Poisons model | Cross-validation reveals |
| **Model Replacement** | Submit unrelated gradient | Catastrophic | Merkle proofs + validation |
| **Adaptive Attack** | Sophisticated mimicry | Subtle degradation | Multi-round reputation |
| **Sybil Attack** | Create many fake identities | Outvote honest nodes | Reputation-gated entry |

### Defense Mechanisms

**Implemented (Phase 10):**

```python
# 1. Krum - Select most consistent gradient
def krum(gradients, f):
    """
    f = number of Byzantine nodes
    Selects gradient with smallest sum of distances to others.
    """
    scores = []
    for i, g_i in enumerate(gradients):
        distances = [
            distance(g_i, g_j)
            for j, g_j in enumerate(gradients)
            if i != j
        ]
        # Sum of n-f-2 smallest distances
        score = sum(sorted(distances)[:len(gradients) - f - 2])
        scores.append((score, g_i))

    # Return gradient with best score
    return min(scores, key=lambda x: x[0])[1]


# 2. Multi-Krum - Average top k gradients
def multikrum(gradients, f, k):
    """
    More robust than single Krum.
    """
    scores = compute_krum_scores(gradients, f)
    top_k = sorted(scores, key=lambda x: x[0])[:k]
    return average([g for _, g in top_k])


# 3. Bulyan - Multi-Krum + coordinate-wise median
def bulyan(gradients, f):
    """
    Most robust, but slowest.
    """
    # Step 1: Multi-Krum selection
    k = len(gradients) - 2*f
    selected = multikrum(gradients, f, k)

    # Step 2: Coordinate-wise median
    result = []
    for i in range(len(selected[0])):
        coordinates = [g[i] for g in selected]
        result.append(median(coordinates))

    return result


# 4. Coordinate-wise Median
def median_aggregation(gradients):
    """
    Simple and reasonably robust.
    """
    result = []
    for i in range(len(gradients[0])):
        coordinates = [g[i] for g in gradients]
        result.append(median(coordinates))
    return result
```

**Testing Results:**

```
Byzantine Attack Suite (7 attacks × 5 defenses = 35 experiments)

Attack: Gaussian Noise (30% Byzantine)
├─ FedAvg:    Accuracy: 0.42 ❌ (baseline fails)
├─ Krum:      Accuracy: 0.91 ✅ (robust)
├─ Multi-Krum: Accuracy: 0.93 ✅ (more robust)
├─ Bulyan:    Accuracy: 0.94 ✅ (most robust)
└─ Median:    Accuracy: 0.88 ✅ (good balance)

Attack: Sign Flip (30% Byzantine)
├─ FedAvg:    Diverges ❌
├─ Krum:      Accuracy: 0.89 ✅
├─ Multi-Krum: Accuracy: 0.92 ✅
├─ Bulyan:    Accuracy: 0.93 ✅
└─ Median:    Accuracy: 0.86 ✅
```

## Integration with Meta-Core

### Reputation Updates

```python
class FLReputationIntegration:
    """
    Maps FL contribution quality to Meta-Core reputation.
    """

    def update_reputation_from_pogq(self, node_id, pogq_score):
        """
        Convert PoGQ score to reputation delta.
        """
        # Map accuracy [0, 1] to reputation delta
        if pogq_score.accuracy > 0.95:
            delta = +10  # Excellent contribution
        elif pogq_score.accuracy > 0.85:
            delta = +5   # Good contribution
        elif pogq_score.accuracy > 0.70:
            delta = +1   # Acceptable contribution
        elif pogq_score.accuracy > 0.50:
            delta = 0    # Neutral
        else:
            delta = -5   # Poor contribution (possible attack)

        # Update in Meta-Core
        meta_core.update_reputation(
            agent_id=node_id,
            delta=delta,
            evidence=pogq_score.proof
        )
```

### Currency Rewards

```python
class FLRewardDistribution:
    """
    Distribute Zero-TrustML Credits based on contribution quality.
    """

    def distribute_rewards(self, round_gradients, total_reward=1000):
        """
        Allocate rewards proportional to quality.
        """
        # Compute quality scores
        scores = [
            pogq.compute(gradient)
            for gradient in round_gradients
        ]

        # Normalize to sum = total_reward
        total_quality = sum(s.accuracy for s in scores)
        rewards = [
            (total_reward * s.accuracy / total_quality)
            for s in scores
        ]

        # Issue Zero-TrustML Credits on Holochain
        for node, reward in zip(round_gradients.keys(), rewards):
            zerotrustml_credits.mint(
                recipient=node.agent_id,
                amount=reward,
                reason=f"FL Round Contribution"
            )
```

## Performance Characteristics

### Computational Complexity

**Per Node:**
- Local training: O(n_local × d) where n_local = local dataset size, d = model dimensions
- Gradient computation: O(d)
- Aggregation (if cluster aggregator): O(k × d) where k = cluster size

**Network:**
- Hierarchical: O(n × d) total communication
- Flat: O(n² × d) total communication

### Latency

**Typical Round Timeline:**
1. Local training: 30-60 seconds
2. Gradient upload: 1-5 seconds
3. Validation: 5-10 seconds
4. Aggregation: 1-2 seconds
5. Model update distribution: 1-5 seconds

**Total: ~60-80 seconds per round**

### Scalability

**Current Implementation:**
- 10-100 nodes: Production-ready
- 100-1000 nodes: Requires hierarchical aggregation
- 1000+ nodes: Multiple aggregation layers

**Theoretical Limit:**
- Holochain: Millions of nodes
- Byzantine algorithms: Hundreds per cluster
- Bridge: Thousands of transactions per batch

---

## Related Documentation

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [Hierarchical FL Details](./HIERARCHICAL_FL.md)
- [Byzantine Attack Mitigation](../07-security/ATTACK_MITIGATION.md)
- [PoGQ Specification](../02-core-concepts/FOUR_PRIMITIVES.md)
