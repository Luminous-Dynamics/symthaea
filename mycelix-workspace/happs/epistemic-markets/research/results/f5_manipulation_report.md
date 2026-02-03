# F5: Manipulation Resistance Testing Report

**Experiment ID:** F5_manipulation_resistance
**Timestamp:** 2026-01-30T12:00:00Z
**Research Infrastructure:** epistemic-markets/research v0.1.0

---

## Executive Summary

This experiment evaluates the effectiveness of anti-manipulation mechanisms against six attack types at varying intensities, testing multiple detection methods to determine optimal security configurations for epistemic prediction markets.

| Criterion | Result |
|-----------|--------|
| **Most Dangerous Attack** | Gradient Poisoning |
| **Most Effective Detection** | Ensemble |
| **Average Detection Rate** | 71.5% |
| **Average False Positive Rate** | 6.1% |
| **Attacks Requiring Attention** | Gradient Poisoning, Collusion |

### Risk Assessment Summary

| Attack Type | Risk Level | Detection Confidence | Recommended Action |
|-------------|------------|---------------------|-------------------|
| Wash Trading | Low | High | Auto-block |
| Price Manipulation | Medium | Medium-High | Flag for review |
| Sybil Attack | Medium | Medium | Identity verification |
| Collusion | High | Low-Medium | Network analysis + review |
| Spoofing | Low | High | Auto-block |
| Gradient Poisoning | **Critical** | Low | Byzantine-robust aggregation |

---

## Methodology

### Attack Types Simulated

1. **Wash Trading**: Self-trades to inflate volume and create false activity signals
2. **Price Manipulation**: Coordinated buying/selling to move prices artificially
3. **Sybil Attack**: Creating multiple fake identities to amplify influence
4. **Collusion**: Coordinated behavior among distinct actors
5. **Spoofing**: Placing fake orders to manipulate orderbook perception
6. **Gradient Poisoning**: Corrupting federated learning signals

### Detection Methods Tested

1. **Statistical Anomaly**: Deviation-based detection from normal trading patterns
2. **Network Analysis**: Graph-based detection of coordinated behavior
3. **Behavior Pattern**: Rule-based pattern matching for known attack signatures
4. **Volume Spike**: Detection of abnormal volume patterns
5. **ML Classifier**: Trained classifier on labeled manipulation data
6. **Ensemble**: Voting-based combination of all methods

### Intensity Levels

Each attack tested at 5 intensity levels: 0.1 (subtle), 0.25, 0.5, 0.75, 1.0 (aggressive)

### Metrics

- **True Positive Rate (TPR)**: Attacks correctly detected
- **False Positive Rate (FPR)**: Legitimate activity flagged as attacks
- **Detection Latency**: Time to detect (normalized)
- **Attack Success if Undetected**: Damage if attack evades detection

---

## Attack Analysis

### 1. Wash Trading

**Overall Assessment:** Well-detected, low risk

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 82.3% | 4.5% | 1.21 | 3.0% |
| 0.25 | 89.1% | 5.2% | 1.12 | 7.5% |
| 0.50 | 93.4% | 6.7% | 1.07 | 15.0% |
| 0.75 | 95.6% | 7.8% | 1.04 | 22.5% |
| 1.00 | 96.7% | 8.9% | 1.03 | 30.0% |

**Average Detection Rate:** 91.4%
**Hardest to Detect:** Low intensity (0.1) - 82.3% TPR

**Analysis:** Wash trading is reliably detected due to distinctive patterns (rapid buy-sell alternation, same trader, high reversal count). Even subtle wash trading (0.1 intensity) is detected 82% of the time. Higher intensity attacks are easier to detect but cause more damage if missed.

**Recommendation:** Implement velocity limits and self-trade detection. Auto-block trades that match wash trading signatures.

---

### 2. Price Manipulation (Pump/Dump)

**Overall Assessment:** Moderate detection, medium risk

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 63.4% | 3.4% | 1.58 | 6.0% |
| 0.25 | 71.2% | 4.1% | 1.40 | 15.0% |
| 0.50 | 78.9% | 5.6% | 1.27 | 30.0% |
| 0.75 | 85.6% | 6.7% | 1.17 | 45.0% |
| 1.00 | 91.2% | 7.8% | 1.10 | 60.0% |

**Average Detection Rate:** 78.1%
**Hardest to Detect:** Low intensity (0.1) - 63.4% TPR

**Analysis:** Price manipulation becomes more detectable as intensity increases, but subtle manipulation (63.4% TPR at 0.1) poses significant risk. The coordinated nature makes it detectable via pattern matching, but sophisticated attackers can mimic organic trading.

**Recommendation:**
- Add price circuit breakers (halt trading if price moves >10% in short window)
- Monitor for coordinated directional trading
- Flag accounts with unusual profit patterns post-manipulation

---

### 3. Sybil Attack (Fake Identities)

**Overall Assessment:** Moderate detection, requires identity layer

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 51.2% | 2.3% | 1.95 | 5.0% |
| 0.25 | 62.3% | 3.4% | 1.61 | 12.5% |
| 0.50 | 73.4% | 4.5% | 1.36 | 25.0% |
| 0.75 | 82.3% | 5.6% | 1.22 | 37.5% |
| 1.00 | 88.9% | 6.7% | 1.12 | 50.0% |

**Average Detection Rate:** 71.6%
**Hardest to Detect:** Low intensity (0.1) - 51.2% TPR

**Analysis:** Sybil attacks are challenging to detect at low intensity because individual fake identities may behave normally. Network analysis helps identify coordination, but sophisticated attackers can introduce timing variance to evade detection.

**Recommendation:**
- Strengthen identity verification at registration
- Implement stake-based reputation (economic Sybil resistance)
- Monitor behavioral similarity across accounts
- Use network analysis to detect coordination clusters

---

### 4. Collusion / Cartels

**Overall Assessment:** Poor detection, HIGH RISK

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 38.9% | 1.8% | 2.57 | 7.0% |
| 0.25 | 47.8% | 2.8% | 2.09 | 17.5% |
| 0.50 | 56.7% | 3.9% | 1.76 | 35.0% |
| 0.75 | 65.6% | 4.9% | 1.52 | 52.5% |
| 1.00 | 74.5% | 6.1% | 1.34 | 70.0% |

**Average Detection Rate:** 56.7%
**Hardest to Detect:** Low intensity (0.1) - 38.9% TPR

**Analysis:** Collusion is the second-hardest attack to detect. Unlike Sybil attacks, colluding parties are genuinely independent, making behavioral similarity harder to establish. They can coordinate off-platform and introduce sufficient timing variance to avoid pattern detection.

**Recommendation:**
- Deploy advanced network analysis (graph clustering, community detection)
- Monitor for unusual profit correlation among accounts
- Implement reporting mechanisms for suspected collusion
- Consider prediction diversity requirements (penalize herding)
- Manual review for borderline cases

---

### 5. Spoofing (Fake Orders)

**Overall Assessment:** Well-detected, low risk

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 75.6% | 5.6% | 1.32 | 4.0% |
| 0.25 | 83.4% | 6.7% | 1.20 | 10.0% |
| 0.50 | 88.9% | 7.8% | 1.12 | 20.0% |
| 0.75 | 92.3% | 8.9% | 1.08 | 30.0% |
| 1.00 | 94.5% | 9.8% | 1.06 | 40.0% |

**Average Detection Rate:** 86.9%
**Hardest to Detect:** Low intensity (0.1) - 75.6% TPR

**Analysis:** Spoofing is well-detected due to characteristic patterns: large orders followed by quick cancellation, trading activity immediately after cancellation. Order book dynamics analysis is effective.

**Recommendation:**
- Monitor order cancellation rates per account
- Penalize excessive order cancellation (fee or rate limit)
- Track correlation between cancellations and subsequent trades
- Auto-block accounts with spoofing signatures

---

### 6. Gradient Poisoning (FL Attacks)

**Overall Assessment:** POOR DETECTION, CRITICAL RISK

| Intensity | TPR | FPR | Latency | Success if Undetected |
|-----------|-----|-----|---------|----------------------|
| 0.10 | 27.8% | 1.2% | 3.60 | 8.0% |
| 0.25 | 35.6% | 2.3% | 2.81 | 20.0% |
| 0.50 | 44.5% | 3.4% | 2.25 | 40.0% |
| 0.75 | 53.4% | 4.5% | 1.87 | 60.0% |
| 1.00 | 61.2% | 5.6% | 1.63 | 80.0% |

**Average Detection Rate:** 44.5%
**Hardest to Detect:** Low intensity (0.1) - 27.8% TPR

**Analysis:** Gradient poisoning is the MOST DANGEROUS attack. Detection rates are lowest across all intensities. At low intensity, only 27.8% are detected, while attack success if undetected can reach 80%. This attack targets the federated learning system directly, corrupting model updates to manipulate calibration feedback.

**Recommendation (CRITICAL):**
- Implement Byzantine-robust aggregation (e.g., Krum, Multi-Krum, Trimmed Mean)
- Validate gradient updates against historical patterns
- Use differential privacy to limit individual update impact
- Implement gradient clipping to bound malicious contributions
- Consider reputation-weighted aggregation (trusted contributors have more influence)
- Regular model auditing against known-good baselines

---

## Detection Method Comparison

### Performance Summary

| Method | TPR | FPR | F1 Score | Ranking |
|--------|-----|-----|----------|---------|
| Ensemble | 82.3% | 4.5% | 0.867 | **#1** |
| ML Classifier | 77.8% | 5.2% | 0.834 | #2 |
| Network Analysis | 68.9% | 3.4% | 0.789 | #3 |
| Behavior Pattern | 71.2% | 6.7% | 0.778 | #4 |
| Statistical Anomaly | 64.5% | 7.8% | 0.723 | #5 |
| Volume Spike | 53.4% | 8.9% | 0.623 | #6 |

### Detection Method Strengths

| Method | Best Against | Worst Against |
|--------|--------------|---------------|
| Ensemble | Wash Trading, Spoofing | Gradient Poisoning, Collusion |
| ML Classifier | Wash Trading, Price Manipulation | Collusion, Gradient Poisoning |
| Network Analysis | Sybil Attack, Collusion | Wash Trading, Gradient Poisoning |
| Behavior Pattern | Spoofing, Wash Trading | Gradient Poisoning, Sybil Attack |
| Statistical Anomaly | Wash Trading, Spoofing | Collusion, Gradient Poisoning |
| Volume Spike | Wash Trading, Price Manipulation | Gradient Poisoning, Collusion |

### Key Insights

1. **Ensemble is consistently best** - Combining methods yields highest F1 score (0.867)
2. **Network Analysis critical for coordination** - Only effective method against Sybil/Collusion
3. **All methods struggle with Gradient Poisoning** - Requires specialized defenses
4. **Volume-based methods have highest FPR** - Legitimate high-volume trading gets flagged
5. **ML Classifier provides best single-method performance** - But requires ongoing training

---

## Security Recommendations

### Immediate Actions (P0)

1. **Deploy Ensemble detection as primary defense**
   - Highest F1 score (0.867)
   - Best coverage across attack types
   - Implementation: Majority voting with 3+ methods

2. **Implement Byzantine-robust aggregation for FL**
   - Gradient Poisoning has 44.5% detection rate
   - Critical attack on calibration system
   - Options: Krum, Multi-Krum, Trimmed Mean, Median

3. **Add price circuit breakers**
   - Halt trading if price moves >10% in 5 minutes
   - Prevents damage from undetected manipulation
   - Manual review required to resume

### Short-Term Actions (P1)

4. **Strengthen Sybil resistance**
   - Stake-based reputation system
   - Enhanced identity verification for high-value markets
   - Economic cost for market participation

5. **Deploy network analysis for collusion detection**
   - Graph clustering on trading patterns
   - Correlation analysis on profit/loss patterns
   - Community detection algorithms

6. **Implement tiered alerting**
   - High confidence (>90%): Auto-block
   - Medium confidence (70-90%): Rate limit + review queue
   - Low confidence (50-70%): Flag for batch review

### Medium-Term Actions (P2)

7. **Build incident response playbooks**
   - Per-attack-type response procedures
   - Escalation paths
   - Recovery procedures

8. **Establish real-time monitoring dashboard**
   - Attack detection rates by type
   - False positive rates
   - Latency metrics

9. **Regular ML classifier retraining**
   - Monthly retraining on new data
   - Include detected attacks in training set
   - Adversarial training against evasion

10. **Cross-market correlation monitoring**
    - Detect manipulation across related markets
    - Identify accounts with suspicious cross-market patterns

---

## Appendix: Confusion Matrices (Ensemble Method)

### Wash Trading
```
               Predicted
              Pos    Neg
Actual Pos    914     86
       Neg     45    955

Precision: 95.3%
Recall: 91.4%
F1: 93.3%
```

### Price Manipulation
```
               Predicted
              Pos    Neg
Actual Pos    781    219
       Neg     52    948

Precision: 93.8%
Recall: 78.1%
F1: 85.2%
```

### Sybil Attack
```
               Predicted
              Pos    Neg
Actual Pos    716    284
       Neg     38    962

Precision: 95.0%
Recall: 71.6%
F1: 81.6%
```

### Collusion
```
               Predicted
              Pos    Neg
Actual Pos    567    433
       Neg     35    965

Precision: 94.2%
Recall: 56.7%
F1: 70.8%
```

### Spoofing
```
               Predicted
              Pos    Neg
Actual Pos    869    131
       Neg     76    924

Precision: 91.9%
Recall: 86.9%
F1: 89.3%
```

### Gradient Poisoning
```
               Predicted
              Pos    Neg
Actual Pos    445    555
       Neg     34    966

Precision: 92.9%
Recall: 44.5%
F1: 60.2%
```

---

## Appendix: Detection Thresholds

### Recommended Thresholds by Method

| Method | Confidence Threshold | Action |
|--------|---------------------|--------|
| Statistical Anomaly | z > 2.5 | Flag |
| Network Analysis | Cluster coefficient > 0.7 | Review |
| Behavior Pattern | Pattern match + freq > 5/min | Block |
| Volume Spike | Volume > 3x daily avg | Flag |
| ML Classifier | Probability > 0.8 | Block |
| Ensemble | 3+ methods agree | Block |

### False Positive Mitigation

- Whitelist verified market makers
- Exclude first-time predictions from anomaly detection
- Allow appeal process for blocked accounts
- Quarterly review of detection thresholds

---

*Report generated by epistemic-markets research infrastructure*
