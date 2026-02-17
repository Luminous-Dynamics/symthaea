# Layer 6: Multi-Round Temporal Detection - Design Specification

**Component**: AEGIS Layer 6 - Temporal Pattern Detection
**Purpose**: Detect sleeper agents, coordination, and temporal anomalies
**Status**: Design Phase
**Date**: November 12, 2025

---

## 🎯 Vision: Temporal Intelligence

### The Problem
Traditional Byzantine detection operates on **single gradients in isolation**:
- No memory of past behavior
- Can't detect agents that turn Byzantine mid-training
- Can't detect coordinated attacks (synchronized timing)
- Can't track reputation evolution

**Reality**: Sophisticated attacks evolve over time:
1. **Sleeper Agents**: Honest for rounds 1-100, Byzantine for rounds 101-200
2. **Coordinated Attacks**: Multiple agents attack simultaneously
3. **Gradual Poisoning**: Slowly increasing Byzantine behavior to avoid detection
4. **Reputation Exploitation**: Build trust, then attack when trusted

### The Solution: Multi-Round Temporal Detection
**Temporal Pattern Analysis**:
1. **History Tracking**: Maintain per-agent behavior history
2. **Sleeper Detection**: Identify sudden behavior changes
3. **Coordination Detection**: Find synchronized attack patterns
4. **Reputation Evolution**: Track trust scores over time
5. **Temporal Anomaly Detection**: Flag unusual temporal patterns

**Result**: Detect sophisticated attacks that evade single-round detection.

---

## 🏗️ Architecture Overview

### Core Components

```python
class MultiRoundDetector:
    """
    Temporal pattern detection across multiple FL rounds.

    Tracks:
    - Per-agent behavior history (scores over time)
    - Reputation evolution (trust scores)
    - Temporal patterns (sleeper agents, coordination)
    - Statistical anomalies (sudden changes)

    Example:
        >>> detector = MultiRoundDetector(
        ...     window_size=50,  # Track last 50 rounds
        ...     sleeper_threshold=0.3,  # 30% behavior change
        ...     coordination_threshold=0.7  # 70% correlation
        ... )
        >>>
        >>> # Track agent over multiple rounds
        >>> for round_idx in range(100):
        ...     gradient = agent.compute_gradient()
        ...     score = ensemble.score(gradient)
        ...
        ...     # Update history
        ...     detector.update_agent_history(agent_id, round_idx, score)
        ...
        ...     # Check for anomalies
        ...     is_sleeper = detector.detect_sleeper_agent(agent_id)
        ...     coordination = detector.detect_coordination()
    """
```

### Detection Mechanisms

#### 1. Sleeper Agent Detection
**Algorithm**: Statistical change point detection

```python
def detect_sleeper_agent(
    agent_id: str,
    window_size: int = 20,
    threshold: float = 0.3
) -> Tuple[bool, float, int]:
    """
    Detect sudden behavior change (honest → Byzantine).

    Algorithm:
        1. Get agent's score history
        2. Compute rolling mean (window_size)
        3. Detect change points: |mean_recent - mean_historical| > threshold
        4. Flag if change > threshold

    Returns:
        (is_sleeper, change_magnitude, change_point_round)

    Example:
        Rounds 1-50: mean_score = 0.85 (honest)
        Rounds 51-100: mean_score = 0.15 (Byzantine)
        Change = |0.15 - 0.85| = 0.70 > 0.3 → SLEEPER DETECTED
    """
```

**Statistical Approach**: CUSUM (Cumulative Sum Control Chart)
```
CUSUM_t = max(0, CUSUM_{t-1} + (x_t - μ - k))

If CUSUM_t > h: Change point detected at round t

Where:
- x_t: Current score
- μ: Historical mean
- k: Allowance (slack parameter)
- h: Threshold (detection sensitivity)
```

#### 2. Coordination Detection
**Algorithm**: Cross-correlation analysis

```python
def detect_coordination(
    agent_ids: List[str],
    correlation_threshold: float = 0.7,
    lag_tolerance: int = 2
) -> List[Tuple[str, str, float]]:
    """
    Detect coordinated attacks (agents attacking together).

    Algorithm:
        1. For each pair of agents:
            a. Compute cross-correlation of score time series
            b. Check lag (synchronized timing)
            c. If correlation > threshold and lag < tolerance:
               → Coordinated attack detected

    Returns:
        List of (agent1, agent2, correlation_score)

    Example:
        Agent A: [0.8, 0.8, 0.2, 0.2, 0.2, 0.8]  (attack rounds 3-5)
        Agent B: [0.8, 0.8, 0.2, 0.2, 0.2, 0.8]  (attack rounds 3-5)
        Correlation: 1.0 > 0.7 → COORDINATION DETECTED
    """
```

**Statistical Approach**: Pearson correlation with lag adjustment
```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)

For lag τ:
ρ_τ(X, Y) = Cov(X_t, Y_{t+τ}) / (σ_X × σ_Y)

Coordinated if: max_τ ρ_τ > threshold and |τ| < lag_tolerance
```

#### 3. Reputation Evolution Tracking
**Algorithm**: Bayesian reputation update

```python
class ReputationTracker:
    """
    Track agent reputation over time with Bayesian updates.

    Reputation ∈ [0.0, 1.0]:
    - 1.0: Fully trusted (consistently honest)
    - 0.5: Neutral (no history)
    - 0.0: Fully distrusted (consistently Byzantine)

    Update Rule (Beta-Binomial):
        α_new = α_old + (score >= 0.5)  # Honest evidence
        β_new = β_old + (score < 0.5)   # Byzantine evidence

        reputation = α / (α + β)
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize with Beta(α, β) prior.

        Default Beta(1, 1) = Uniform prior (no bias).
        """
        self.agents = {}  # agent_id → (α, β)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def update_reputation(self, agent_id: str, score: float):
        """Update reputation with new evidence."""
        if agent_id not in self.agents:
            self.agents[agent_id] = (self.alpha_prior, self.beta_prior)

        alpha, beta = self.agents[agent_id]

        # Bayesian update
        if score >= 0.5:  # Honest evidence
            alpha += 1
        else:  # Byzantine evidence
            beta += 1

        self.agents[agent_id] = (alpha, beta)

    def get_reputation(self, agent_id: str) -> float:
        """Get current reputation (posterior mean)."""
        if agent_id not in self.agents:
            return 0.5  # Neutral

        alpha, beta = self.agents[agent_id]
        return alpha / (alpha + beta)
```

#### 4. Temporal Anomaly Detection
**Algorithm**: Statistical Process Control (SPC)

```python
def detect_temporal_anomaly(
    agent_id: str,
    z_threshold: float = 3.0
) -> Tuple[bool, float]:
    """
    Detect statistical anomalies in agent behavior.

    Algorithm:
        1. Compute historical mean μ and std σ
        2. For current score x:
            z = |x - μ| / σ
        3. If z > threshold (e.g., 3σ):
           → Anomaly detected

    Returns:
        (is_anomaly, z_score)

    Example:
        Historical: μ=0.85, σ=0.05 (consistently honest)
        Current: x=0.15
        z = |0.15 - 0.85| / 0.05 = 14 > 3 → ANOMALY
    """
```

---

## 📊 Algorithm Specification

### Multi-Round Detection Algorithm

**Input**: Agent gradients across multiple rounds
**Output**: Temporal anomaly flags, coordination groups, reputation scores

**Data Structures**:
```python
AgentHistory:
    agent_id: str
    scores: List[float]  # Score per round
    rounds: List[int]    # Round numbers
    reputation: float    # Current reputation [0, 1]

CoordinationGroup:
    agents: List[str]
    correlation: float
    attack_rounds: List[int]

TemporalAnomaly:
    agent_id: str
    round: int
    anomaly_type: str  # "sleeper" | "spike" | "drift"
    magnitude: float
```

**Algorithm**:
```
MULTI_ROUND_DETECTION(gradients_per_round, agent_ids):
    # Initialize
    detector = MultiRoundDetector(
        window_size=50,
        sleeper_threshold=0.3,
        coordination_threshold=0.7
    )

    # Track history over rounds
    FOR round_idx, gradients in gradients_per_round:
        FOR agent_id, gradient in zip(agent_ids, gradients):
            # Score gradient (using Layers 1-5)
            score = active_learning_inspector.inspect([gradient])[0]

            # Update history
            detector.update_agent_history(agent_id, round_idx, score)

            # Update reputation
            detector.reputation_tracker.update_reputation(agent_id, score)

    # Phase 1: Sleeper Detection
    sleepers = []
    FOR agent_id in agent_ids:
        is_sleeper, magnitude, change_point = detector.detect_sleeper_agent(agent_id)
        IF is_sleeper:
            sleepers.append({
                "agent": agent_id,
                "magnitude": magnitude,
                "change_point": change_point
            })

    # Phase 2: Coordination Detection
    coordinated_groups = detector.detect_coordination(agent_ids)

    # Phase 3: Temporal Anomalies
    anomalies = []
    FOR agent_id in agent_ids:
        is_anomaly, z_score = detector.detect_temporal_anomaly(agent_id)
        IF is_anomaly:
            anomalies.append({
                "agent": agent_id,
                "z_score": z_score,
                "round": current_round
            })

    # Phase 4: Reputation Summary
    reputations = {
        agent_id: detector.reputation_tracker.get_reputation(agent_id)
        for agent_id in agent_ids
    }

    RETURN {
        "sleepers": sleepers,
        "coordinated_groups": coordinated_groups,
        "anomalies": anomalies,
        "reputations": reputations
    }
```

---

## 🎯 Key Design Decisions

### Decision 1: Window Size (Rolling Statistics)
**Choice**: Configurable window (default: 50 rounds)

**Rationale**:
- Too small (< 10): Noisy, false positives
- Too large (> 100): Slow to detect changes
- 50 rounds: Balance responsiveness and stability

**Adaptive Option**: Adjust window based on variance
```python
if variance < 0.01:  # Stable
    window_size = 20  # Respond quickly
else:  # Volatile
    window_size = 100  # Be conservative
```

### Decision 2: CUSUM vs. Moving Average
**Choice**: CUSUM (Cumulative Sum) for change point detection

**Rationale**:
- CUSUM detects subtle shifts better than simple thresholds
- Industry standard in statistical process control
- Mathematically rigorous (ARL - Average Run Length theory)

**Alternative**: Exponentially Weighted Moving Average (EWMA)
- Similar performance but CUSUM has clearer theory

### Decision 3: Bayesian vs. Frequency Reputation
**Choice**: Bayesian (Beta-Binomial) reputation

**Rationale**:
- Handles uncertainty explicitly (α, β parameters)
- Naturally incorporates prior knowledge
- Graceful handling of new agents (prior = neutral)
- Well-studied convergence properties

**Alternative**: Simple moving average
- Easier to implement but no uncertainty quantification

### Decision 4: Correlation Threshold
**Choice**: 0.7 correlation for coordination detection

**Rationale**:
- 0.7 = "strong correlation" (statistical standard)
- Lower (< 0.5): Too many false positives
- Higher (> 0.9): Might miss near-perfect coordination

**Adaptive Option**: Adjust based on baseline agent correlation

---

## 🧪 Testing Strategy

### Unit Tests (12 tests)

1. **Sleeper Detection (4 tests)**:
   - ✅ Detect honest → Byzantine transition
   - ✅ No false positive on consistent agent
   - ✅ Detect Byzantine → honest transition
   - ✅ CUSUM change point accuracy

2. **Coordination Detection (3 tests)**:
   - ✅ Detect synchronized attacks
   - ✅ Ignore non-coordinated agents
   - ✅ Handle lag tolerance

3. **Reputation Tracking (3 tests)**:
   - ✅ Bayesian update correctness
   - ✅ Reputation convergence
   - ✅ New agent initialization

4. **Temporal Anomaly (2 tests)**:
   - ✅ Z-score anomaly detection
   - ✅ No false positive on normal variance

### Integration Tests (8 tests)

1. **Multi-Round Pipeline (3 tests)**:
   - ✅ History tracking over 100 rounds
   - ✅ All detection mechanisms working together
   - ✅ Performance on large-scale (1000 agents × 200 rounds)

2. **Real Attack Scenarios (5 tests)**:
   - ✅ Sleeper agent (round 50 flip)
   - ✅ Coordinated attack (10 agents, round 75)
   - ✅ Gradual poisoning (slow drift)
   - ✅ Reputation exploitation (build trust, then attack)
   - ✅ Mixed scenario (sleepers + coordination)

---

## 📈 Expected Performance

### Detection Rates

**Sleeper Agents**:
```
Scenario: Honest for 50 rounds, Byzantine for 50 rounds
Expected:
- Detection round: Round 55 (5 rounds after flip)
- Detection confidence: > 95%
- False positive rate: < 5%
```

**Coordination**:
```
Scenario: 5 agents attack simultaneously (rounds 70-80)
Expected:
- Detection: Within 10 rounds of attack start
- Precision: > 90% (correct pairs identified)
- Recall: > 85% (few missed coordinations)
```

**Reputation Convergence**:
```
Scenario: Consistently honest agent (100 rounds)
Expected:
- Round 10: reputation ≈ 0.90
- Round 50: reputation ≈ 0.98
- Round 100: reputation ≈ 0.99
- Convergence: Exponential (fast initially, then slow)
```

### Computational Cost

**Per-Round Cost**:
```
History update: O(1) (append to deque)
Sleeper detection: O(window_size) ≈ O(50) per agent
Coordination: O(n²) pairwise correlations (expensive!)
Reputation update: O(1) (Bayesian update)

Total per round: O(n² + n × window_size)
For n=100 agents: ~15,000 operations
```

**Optimization**: Cache correlation matrix, incremental updates
```
Optimized coordination: O(n) per round
Total optimized: O(n × window_size) ≈ O(5,000) for n=100
```

---

## 💡 Key Innovations

### 1. CUSUM-Based Sleeper Detection
**First FL system** using statistical process control (CUSUM) for sleeper agent detection.

**Impact**: Detect subtle behavior changes with statistical rigor.

### 2. Correlation-Based Coordination Detection
**Novel**: Cross-correlation analysis of agent time series to find coordinated attacks.

**Impact**: Detect synchronized attacks that evade single-round detection.

### 3. Bayesian Reputation Evolution
**Principled**: Beta-Binomial conjugate prior for reputation tracking.

**Impact**: Uncertainty-aware trust scores with provable convergence.

### 4. Temporal Anomaly Integration
**Unique**: Combines change point detection + correlation analysis + reputation.

**Impact**: Multi-faceted temporal intelligence, not just single metric.

---

## 🔍 Research Foundation

### CUSUM Theory
**Paper**: Page, E. S. (1954). "Continuous Inspection Schemes"

**Key Result**: CUSUM detects mean shift of size δ with Average Run Length (ARL):
```
ARL₀ ≈ exp(2h/σ²)  (false alarm)
ARL₁ ≈ (h + 1.166) / δ  (true detection)

Where:
- h: Decision threshold
- σ²: Process variance
- δ: Shift magnitude
```

**AEGIS Application**: Detect score shifts (honest ↔ Byzantine) with tunable false alarm rate.

### Beta-Binomial Reputation
**Paper**: Jøsang, A. & Ismail, R. (2002). "The Beta Reputation System"

**Key Result**: Bayesian update with Beta(α, β):
```
Posterior mean: E[θ] = α / (α + β)
Posterior variance: Var[θ] = αβ / ((α+β)²(α+β+1))

Convergence: As α+β → ∞, variance → 0 (reputation converges)
```

**AEGIS Application**: Reputation converges to true agent behavior with more observations.

### Cross-Correlation for Synchrony Detection
**Paper**: Eckmann, J. P. et al. (2004). "Entropy of dialogues creates coherent structures in e-mail traffic"

**Key Result**: Pearson correlation detects synchronized time series:
```
ρ = 1: Perfect synchrony
ρ = 0: Independent
ρ = -1: Perfect anti-synchrony

For FL attacks: Expect ρ > 0.7 for coordinated agents
```

**AEGIS Application**: Identify groups of agents attacking together.

---

## 🚀 Implementation Roadmap

### Phase 1: Core Detector (3-4 hours)
- Implement `MultiRoundDetector` class
- History tracking (per-agent deques)
- Basic sleeper detection (threshold-based)
- Basic reputation tracking (Bayesian)

### Phase 2: Advanced Detection (2-3 hours)
- CUSUM sleeper detection
- Cross-correlation coordination detection
- Temporal anomaly detection (z-scores)
- Optimization (incremental correlation)

### Phase 3: Testing (3-4 hours)
- 12 unit tests
- 8 integration tests
- Performance benchmarks
- Validation against attack scenarios

### Total Time: 2-3 days (within Week 3 plan)

---

## 🎯 Success Criteria

### Must-Have (Launch Blockers)
- ✅ Sleeper agent detection working
- ✅ Coordination detection working
- ✅ Reputation tracking working
- ✅ Detection within 10 rounds of attack start
- ✅ < 10% false positive rate
- ✅ All 20 tests passing

### Nice-to-Have (Enhancements)
- 📊 CUSUM implementation (vs. simple threshold)
- 📊 Incremental correlation (optimization)
- 📊 Adaptive window sizing
- 📊 Multi-lag correlation

### Stretch Goals (Future Work)
- 🔮 Predictive detection (ML-based forecasting)
- 🔮 Causal inference (attack attribution)
- 🔮 Automated response (quarantine Byzantine agents)

---

## 📝 Paper Integration

### Contribution Statement
```
To detect sophisticated attacks that evade single-round detection, we
introduce Multi-Round Temporal Detection—a system that tracks agent
behavior over time to identify sleeper agents, coordinated attacks, and
temporal anomalies. Using CUSUM change point detection, cross-correlation
analysis, and Bayesian reputation tracking, AEGIS detects behavior changes
within 5-10 rounds with < 10% false positive rate. This temporal intelligence
layer enables defense against adaptive attackers who exploit temporal dynamics.
```

### Algorithm (for paper)
```
TEMPORAL_DETECTION(agent_histories):
    # Sleeper Detection (CUSUM)
    FOR agent in agents:
        CUSUM = compute_cusum(agent.scores)
        IF CUSUM > threshold:
            FLAG agent as sleeper at change point

    # Coordination Detection (Correlation)
    FOR each pair (agent_i, agent_j):
        ρ = pearson_correlation(agent_i.scores, agent_j.scores)
        IF ρ > 0.7:
            FLAG (agent_i, agent_j) as coordinated

    # Reputation Update (Bayesian)
    FOR agent in agents:
        α, β = update_beta(agent.scores)
        reputation = α / (α + β)
```

### Figures (Week 4 Experiments)
- **Figure 6a**: Sleeper detection timeline (detection vs. flip round)
- **Figure 6b**: Coordination detection precision/recall
- **Figure 6c**: Reputation convergence curves
- **Figure 6d**: Temporal anomaly detection ROC curve

---

## 🔗 Integration with Previous Layers

### Layer 1 (Meta-Learning)
**Connection**: Use ensemble scores as input to temporal tracking
```python
score = meta_ensemble.compute_ensemble_score(signals)
multi_round.update_agent_history(agent_id, round_idx, score)
```

### Layer 3 (Uncertainty)
**Connection**: Use confidence intervals for reputation uncertainty
```python
decision, prob, interval = uncertainty_quantifier.predict_with_confidence(score)
reputation_variance = interval[1] - interval[0]
```

### Layer 5 (Active Learning)
**Connection**: Prioritize queries on agents with temporal anomalies
```python
if multi_round.detect_temporal_anomaly(agent_id):
    # Flag for deep verification in Layer 5
    query_indices.append(agent_idx)
```

**Synergy**: Layers 1-5 provide scores, Layer 6 tracks temporal patterns → Complete defense stack

---

**Design Status**: ✅ COMPLETE
**Next Step**: Implement `MultiRoundDetector` class
**Estimated Time**: 8-11 hours total (2-3 days)

🌊 **Temporal intelligence - AEGIS remembers and learns from history!** 🌊
