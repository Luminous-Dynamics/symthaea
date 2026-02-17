---
title: Interactive Playground - Byzantine Resistance Simulator
description: Hands-on Byzantine resistance simulations. Explore trust scoring, attack detection, and 45% Byzantine tolerance with live interactive widgets.
keywords: Byzantine simulator, trust score calculator, interactive federated learning, MATL playground, attack detection demo
search:
  boost: 1.8
---

# 🎮 Interactive Learning Playground

**Hands-on exploration of Byzantine resistance and trust scoring**

---

## Welcome to the Playground!

Experiment with MATL's Byzantine resistance mechanisms through interactive simulations. Adjust parameters, run scenarios, and see how the system responds in real-time.

---

## 🛡️ Byzantine Tolerance Calculator

<div id="byzantine-calculator"></div>

### What You're Seeing

This calculator demonstrates how **reputation-weighted validation** enables MATL to exceed the classical 33% Byzantine fault tolerance limit.

**Key Insight:** Even with >33% malicious **nodes**, the system remains secure if their **Byzantine power** (Σ reputation²) stays below the threshold.

**Try this:**
1. Set Byzantine Ratio to 45%
2. Keep Byzantine Reputation at 0.1
3. Keep Honest Reputation at 0.9
4. ✅ System stays SAFE! 45% tolerance achieved.

---

## 📊 Trust Score Evolution Simulator

<div id="trust-score-simulator"></div>

### How It Works

Watch how trust scores evolve over 20 training rounds. See what happens when a node starts attacking:

**Scenarios to try:**
- **No Attack** (Round 0): All nodes build trust steadily
- **Early Attack** (Round 5): Attacker detected and isolated quickly
- **Sleeper Agent** (Round 15): Node builds trust then attacks - watch MATL respond!

**Key Observation:** Once a node starts attacking, its trust score drops rapidly and it gets excluded from aggregation.

---

## ⚔️ Attack Type Comparison

<div id="attack-comparison"></div>

### Understanding Different Attacks

Click on each attack type to learn how MATL detects and mitigates it. Detection rates show how effectively MATL identifies each attack type.

**Detection Rates:**
- **Sign Flip**: 98% (easiest to detect)
- **Sleeper Agent**: 87% (hardest - builds trust first)
- **Average**: 92% across all attack types

---

## 💡 Key Takeaways

### 1. Reputation is Power (Squared!)

Byzantine power = Σ(malicious_reputation²)

This quadratic weighting means:
- Low-reputation attackers have minimal impact
- New nodes can't immediately poison the system
- Trust must be earned over multiple rounds

### 2. System Self-Heals

As training progresses:
- Honest nodes: Trust ↑ (0.5 → 0.95)
- Malicious nodes: Trust ↓ (0.5 → 0.05)
- System becomes **more secure** with each round

### 3. Multiple Defense Layers

MATL doesn't rely on a single detection method:
- PoGQ: Statistical validation
- TCDM: Reputation-weighted mean
- Entropy: Information theory
- Cartel: Graph-based clustering

### 4. Real-World Resilience

These aren't theoretical guarantees - they're measured results from production testing with 100+ continuous training rounds.

---

## 🔬 Experiment Ideas

Try these scenarios in the calculators above:

### Scenario 1: Gradual Attack
- Start with no attack
- Run simulation to round 10
- Watch how honest nodes build strong reputation
- **Insight**: Early trust makes later attacks harder

### Scenario 2: High Byzantine Ratio
- Set Byzantine Ratio to 60%
- Lower their reputation to 0.05
- System STAYS SAFE despite >50% malicious nodes!
- **Insight**: Reputation matters more than count

### Scenario 3: Reputation Equilibrium
- Find the maximum Byzantine ratio for different reputation values
- Try: Byzantine Rep = 0.3, what's the max ratio?
- **Insight**: There's always a tolerance threshold

---

## 📚 Related Resources

Want to go deeper? Check out:

- **[MATL Integration Tutorial](../tutorials/matl_integration.md)** - Build your own FL system
- **[Visual Diagrams](../03-architecture/visual_diagrams.md)** - More architecture visualizations
- **[MATL Architecture](../0TML/docs/06-architecture/matl_architecture.md)** - Technical deep dive
- **[Healthcare Tutorial](../tutorials/healthcare_federated_learning.md)** - Real-world application

---

## 🎯 Challenge Questions

Test your understanding:

1. **Why does quadratic weighting (reputation²) help security?**
   <details>
   <summary>Click to reveal answer</summary>
   It amplifies the difference between high and low reputation nodes. An honest node with 0.9 reputation has 0.81 power, while a malicious node with 0.1 reputation has only 0.01 power - an 81× difference!
   </details>

2. **Can a sleeper agent defeat MATL by building trust first?**
   <details>
   <summary>Click to reveal answer</summary>
   Partially - they can cause temporary damage, but MATL quickly detects the change in behavior and reduces their trust. The damage is limited to a few rounds before isolation.
   </details>

3. **What happens if ALL nodes start malicious?**
   <details>
   <summary>Click to reveal answer</summary>
   The system would converge slowly as nodes with better gradients gradually gain more trust. However, MATL assumes at least some honest bootstrap nodes exist initially.
   </details>

---

## 🤝 Share Your Discoveries

Found an interesting scenario? Tweet it with **#MATLPlayground** or share in our [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix/discussions)!

---

**Ready to build with MATL?** Start with the [MATL Integration Tutorial →](../tutorials/matl_integration.md)

<script src="../javascripts/interactive-widgets.js"></script>
