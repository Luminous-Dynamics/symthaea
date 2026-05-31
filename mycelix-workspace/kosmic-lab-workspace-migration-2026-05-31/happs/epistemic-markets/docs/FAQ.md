# Frequently Asked Questions

*Everything You Wanted to Know But Were Afraid to Ask*

---

> "The only stupid question is the one you don't ask."
> — Every good teacher

> "Actually, some questions reveal deep insight. We hope these do."
> — Us

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the System](#understanding-the-system)
3. [Making Predictions](#making-predictions)
4. [Stakes and Rewards](#stakes-and-rewards)
5. [Resolution and Oracles](#resolution-and-oracles)
6. [Reputation and Trust](#reputation-and-trust)
7. [Question Markets](#question-markets)
8. [Community and Governance](#community-and-governance)
9. [Technical Questions](#technical-questions)
10. [Privacy and Security](#privacy-and-security)
11. [Philosophy and Values](#philosophy-and-values)
12. [Troubleshooting](#troubleshooting)

---

## Getting Started

### What is Epistemic Markets?

Epistemic Markets is a decentralized platform for collective truth-seeking through prediction markets. Unlike traditional prediction markets focused purely on price discovery, we prioritize calibration, reasoning quality, and collaborative truth-seeking.

Think of it as a place where:
- You make predictions about the future
- You stake something valuable (money, reputation, or commitments)
- You share your reasoning
- The community learns from right and wrong predictions
- Wisdom accumulates for future generations

### How is this different from betting?

| Aspect | Betting | Epistemic Markets |
|--------|---------|-------------------|
| Goal | Profit | Truth-seeking |
| Stakes | Money only | Multi-dimensional |
| Reasoning | Not required | Central |
| Learning | Individual | Collective |
| Calibration | Not tracked | Core metric |
| Community | Adversarial | Collaborative |

You can profit here, but that's a side effect of being right—not the primary goal.

### Do I need to be an expert?

No! We're designed for everyone:

- **Complete beginners**: Start with low-stakes predictions on topics you know
- **Domain experts**: Your expertise is recognized and weighted
- **Data analysts**: Deep analytics and API access
- **Philosophers**: Engage with the deeper questions
- **Community members**: Contribute wisdom without technical expertise

### How do I make my first prediction?

1. Browse open markets or create one
2. Choose your position and confidence level
3. Decide what to stake (money, reputation, or commitment)
4. Write your reasoning (required for higher stakes)
5. Submit and wait for resolution

See [GETTING_STARTED.md](../GETTING_STARTED.md) for a detailed walkthrough.

### Is there a tutorial mode?

Yes! The Training Ground offers:
- Practice predictions with no real stakes
- Calibration training exercises
- Historical markets to test yourself against
- Community mentors for guidance

---

## Understanding the System

### What makes a market "epistemic"?

Every market is classified on three dimensions:

**E (Empirical)**: How verifiable is the outcome?
- E0: Subjective (no external verification)
- E1: Testimonial (witnesses)
- E2: Private verification (ZK proofs)
- E3: Cryptographic (on-chain)
- E4: Scientific (reproducible)

**N (Normative)**: Whose agreement matters?
- N0: Personal opinion
- N1: Community consensus
- N2: Network-wide
- N3: Universal truth

**M (Materiality)**: How long does this matter?
- M0: Hours
- M1: Days/weeks
- M2: Months/years
- M3: Permanent

The classification determines how the market gets resolved.

### What is MATL?

MATL (Multi-dimensional Adaptive Trust Layer) is our reputation system. It tracks:

- **Merit**: Calibration accuracy over time
- **Accountability**: Following through on commitments
- **Trust**: Community trust signals
- **Legacy**: Long-term contribution

Your MATL score affects your oracle voting weight, stake limits, and governance participation.

### What are the different roles in the system?

| Role | Description | How to Become |
|------|-------------|---------------|
| **Predictor** | Makes predictions | Anyone can start |
| **Oracle** | Votes on resolution | High MATL score |
| **Wisdom Keeper** | Plants wisdom seeds | Consistent contribution |
| **Mentor** | Helps newcomers | Earn through teaching |
| **Elder** | Governance participation | Long tenure + contribution |
| **Guardian** | Security vigilance | Designated by protocol |

### How does decentralization work?

Epistemic Markets runs on Holochain, which means:
- No central server controls the data
- Each participant runs a node
- Data is validated by the network
- No single point of failure
- Resistant to censorship

---

## Making Predictions

### How do I choose what to predict?

**Good predictions:**
- Clear, unambiguous outcomes
- Reasonable resolution timeline
- Something you have genuine insight on
- Appropriate for your risk tolerance

**Avoid:**
- Vague or subjective questions
- Unknowable outcomes
- Questions where you have no edge
- Stakes higher than you can afford to lose

### What's the difference between confidence and stake?

**Confidence** (0-100%): Your belief about probability
- 80% means "I think there's an 80% chance of this outcome"
- Calibrated predictors' 80% predictions come true about 80% of the time

**Stake**: What you're risking
- Monetary: Actual value
- Reputation: MATL score percentage
- Commitment: Actions you'll take based on outcome

Higher stakes require more reasoning and have bigger consequences.

### Do I have to explain my reasoning?

| Stake Level | Reasoning Requirement |
|-------------|----------------------|
| Minimal | Optional |
| Low | Encouraged (150+ chars) |
| Medium | Required (300+ chars) |
| High | Required (500+ chars) + sources |
| Very High | Required (1000+ chars) + structured |

Better reasoning improves your reputation even if your prediction is wrong.

### Can I update my prediction?

Yes, with constraints:
- Updates before market close are allowed
- Each update may have a small cost
- Update history is visible
- Frequent updates may signal low confidence (affects calibration)

### What if I change my mind completely?

You can:
1. **Update**: Change your position in place
2. **Hedge**: Make a counter-prediction (costs more)
3. **Exit**: Withdraw (partial stake loss may apply)

We encourage genuine belief updates over stubbornness.

---

## Stakes and Rewards

### What types of stakes are there?

**Monetary Stakes**
- HAM tokens (protocol currency)
- Stablecoins (if integrated)
- Locked until resolution

**Reputation Stakes**
- Percentage of your MATL score
- Domain-specific reputation
- Affects your standing if wrong

**Social Stakes**
- Public visibility of prediction
- Identity linking (optional)
- Community accountability

**Commitment Stakes**
- Actions you'll take if right/wrong
- Verified through smart contracts
- Builds trust regardless of outcome

### How are rewards calculated?

Rewards come from multiple sources:

```
Total Reward = Accuracy Reward + Calibration Bonus + Reasoning Bonus + Early Bonus

Where:
- Accuracy Reward: Share of losing stakes (proportional to your stake)
- Calibration Bonus: Extra for well-calibrated predictions
- Reasoning Bonus: Community-rated reasoning quality
- Early Bonus: Small bonus for earlier predictions
```

### What happens if I'm wrong?

You lose your stake, but:
- Monetary stake goes to correct predictors
- Reputation stake decreases your MATL score
- Commitment stakes must still be honored
- Good reasoning softens reputation loss

Being wrong is part of learning. We don't punish honest mistakes.

### Can I lose more than I stake?

No. Your maximum loss is your stake. There are no margin calls or leveraged positions in the base protocol.

### What fees does the platform charge?

| Fee Type | Amount | Purpose |
|----------|--------|---------|
| Market creation | 0.1% | Anti-spam, quality |
| Prediction | 0.05% | Protocol sustainability |
| Resolution | 0.1% | Oracle compensation |
| Withdrawal | Variable | Network costs |

Fees are minimal and support protocol sustainability.

---

## Resolution and Oracles

### How do markets get resolved?

Resolution depends on the E-N-M classification:

**Cryptographic (E3)**
- Automatic resolution from on-chain events
- No human intervention needed

**Measurable (E4)**
- Data feeds from trusted sources
- Cross-verified by multiple oracles

**Testimonial (E1-E2)**
- Oracle voting weighted by MATL
- Requires supermajority consensus

**Subjective (E0)**
- Community consensus process
- May remain "opinion" (partial refunds)

### Who are the oracles?

Oracles are high-MATL participants qualified to vote on resolution:
- Selected based on domain expertise
- Weighted by their MATL composite score
- Multiple oracles per resolution (Byzantine resistant)
- Can be penalized for dishonesty

### What if I disagree with a resolution?

1. **Dispute Period**: Challenge within 72 hours
2. **Stake Required**: Dispute stake prevents spam
3. **Extended Oracle Pool**: More oracles invited
4. **Governance Escalation**: If still disputed
5. **Final Decision**: Protocol Council ruling

### What about markets that can't be resolved?

If a market becomes unresolvable:
- Stakes are refunded proportionally
- Market creator may lose deposit
- Marked as "Unknowable" in history
- Wisdom seeds capture what was learned

---

## Reputation and Trust

### How does MATL scoring work?

Your MATL score is calculated from:

```
MATL = α·Merit + β·Accountability + γ·Trust + δ·Legacy

Where:
- Merit (40%): Prediction accuracy and calibration
- Accountability (25%): Commitment follow-through
- Trust (20%): Community vouching
- Legacy (15%): Long-term contribution
```

Scores are domain-specific and time-weighted (recent matters more).

### How do I improve my reputation?

**Do:**
- Make calibrated predictions (accuracy at stated confidence)
- Write high-quality reasoning
- Honor your commitments
- Help newcomers
- Participate constructively

**Don't:**
- Make wild claims
- Abandon commitments
- Attack other participants
- Game the system

### Can I lose reputation?

Yes, through:
- Consistent miscalibration
- Broken commitments
- Community reports (verified)
- Sybil/manipulation detection

Reputation decay is gradual, allowing recovery.

### How do I get vouched for?

Vouching is earned through:
- Helping community members
- Quality contributions over time
- Being vouched by already-trusted members
- Verification of real-world credentials

### Is reputation transferable?

No. Reputation is:
- Earned, not bought
- Non-transferable
- Domain-specific
- Tied to your identity

---

## Question Markets

### What is a Question Market?

Before predicting answers, trade on which questions are worth answering!

Question Markets let you:
- Propose questions
- Signal curiosity (lightweight interest)
- Buy "value shares" (bet the answer is worth knowing)
- Earn rewards if the question becomes a prediction market

### How do questions become markets?

1. **Proposal**: Someone proposes a question
2. **Curiosity Signals**: Others express interest
3. **Value Accumulation**: Shares are bought
4. **Threshold**: When value > threshold, market spawns
5. **Value Distribution**: Early supporters earn rewards

### What makes a good question?

- Clear and unambiguous
- Resolvable (not permanently unknowable)
- Relevant to the community
- Not already answered
- Appropriate epistemic classification

### Can I propose any question?

Most questions are allowed, except:
- Illegal content
- Personal harm questions
- Spam or duplicates
- Questions designed to manipulate

See [GOVERNANCE.md](./GOVERNANCE.md) for full policies.

---

## Community and Governance

### How is the protocol governed?

```
┌─────────────────────────────────────┐
│       Constitutional Layer          │  ← Core values (hardest to change)
├─────────────────────────────────────┤
│         Protocol Council            │  ← Major decisions
├─────────────────────────────────────┤
│       Parameter Committee           │  ← Tuning
├─────────────────────────────────────┤
│        Operations Team              │  ← Day-to-day
├─────────────────────────────────────┤
│       Community Assembly            │  ← All participants
└─────────────────────────────────────┘
```

### How do I participate in governance?

| Level | Requirement | Abilities |
|-------|-------------|-----------|
| Observe | Any account | Read proposals, discussions |
| Signal | Verified account | React to proposals |
| Vote | MATL > 0.3 | Vote on proposals |
| Propose | MATL > 0.5 | Submit proposals |
| Council | Elder status | Major decisions |

### What can I change through governance?

**Changeable:**
- Fee structures
- Threshold parameters
- Reward formulas
- Integration decisions

**Constitutional (hard to change):**
- Core epistemic principles
- Fundamental rights
- Red lines (see [GOVERNANCE.md](./GOVERNANCE.md))

### How do I report problems?

- **Bug**: GitHub issues or in-app reporting
- **Dispute**: Formal dispute process
- **Abuse**: Community moderation
- **Security**: security@epistemic-markets.org

---

## Technical Questions

### What blockchain does this run on?

We run on Holochain, which is:
- Agent-centric (not blockchain)
- No mining or staking for consensus
- Each user has their own chain
- Validated by network peers
- Energy efficient

### Do I need a crypto wallet?

Yes, for:
- Identity (cryptographic keys)
- Monetary stakes (if any)
- Signing predictions

The app guides you through setup.

### What are the system requirements?

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, macOS 10.15, Linux | Latest versions |
| RAM | 4GB | 8GB+ |
| Storage | 10GB | 50GB+ |
| Network | Stable internet | Low latency |

### Is there an API?

Yes! Full API access for:
- Reading market data
- Submitting predictions (with authentication)
- Analytics and historical data
- Webhook notifications

See developer documentation for details.

### Can I run my own node?

Yes, and we encourage it:
- Improves network resilience
- Faster local access
- Full data sovereignty
- Contribute to decentralization

---

## Privacy and Security

### Who can see my predictions?

| Setting | Before Resolution | After Resolution |
|---------|-------------------|------------------|
| Private | Only you | Only you + oracles |
| Limited | You + chosen people | Community |
| Public | Everyone | Everyone |

Default is "Limited" for most stake levels.

### Is my data encrypted?

Yes:
- All data encrypted in transit
- Private predictions encrypted at rest
- Only you control your private keys
- Zero-knowledge proofs for private aggregation

### What if I forget my password?

- Social recovery (if configured)
- Backup seed phrase (if saved)
- No central authority can reset

**Save your seed phrase securely!**

### Can the platform be hacked?

We take security seriously:
- Regular security audits
- Bug bounty program
- Multi-sig treasury
- Circuit breakers for emergencies
- See [SECURITY.md](./SECURITY.md)

No system is perfectly secure, but we minimize risks.

### Can anyone see my real identity?

Only if you choose to link it:
- Pseudonymous by default
- Optional credential verification
- Identity reveals are opt-in
- Even oracles don't always see identities

---

## Philosophy and Values

### Why "epistemic" markets?

"Epistemic" means "relating to knowledge." We care about:
- How we know things
- The quality of our reasoning
- Collective sense-making
- Wisdom accumulation

Not just "are you right?" but "how did you think about it?"

### What's the difference between truth and consensus?

We distinguish:

| Concept | Definition | Example |
|---------|------------|---------|
| Truth (E4) | Objective reality | "The sun rose today" |
| Consensus (N2) | Shared agreement | "This policy is good" |
| Opinion (E0) | Individual view | "This movie is better" |

Different questions require different resolution approaches.

### Why require reasoning?

Reasoning:
- Improves your own thinking
- Helps others learn
- Creates lasting knowledge
- Distinguishes luck from skill
- Builds community intelligence

A correct prediction with good reasoning is worth more than luck.

### What about manipulation and bad actors?

We design against:
- Information cascades (independent predictions)
- Sybil attacks (identity verification)
- Oracle collusion (MATL weighting)
- Market manipulation (economic penalties)

See [FAILURE_MODES.md](./FAILURE_MODES.md) for comprehensive analysis.

### Is this trying to replace experts?

No! We aim to:
- Surface expertise (MATL recognizes it)
- Aggregate distributed knowledge
- Combine expert and crowd wisdom
- Make expertise more visible

Experts are valuable. We make their expertise count more.

---

## Troubleshooting

### My prediction wasn't accepted

Check:
- Is the market still open?
- Is your stake within allowed range?
- Did you meet reasoning requirements?
- Is your account in good standing?

### I can't withdraw my stake

Stakes are locked until resolution. Check:
- Market resolution status
- Dispute period status
- Your withdrawal limits

### My reputation suddenly dropped

Possible causes:
- Recent miscalibrated predictions resolved
- Community reports verified
- Commitment not honored
- Decay from inactivity

### The market resolved wrong

1. Check if dispute period is open
2. Review resolution evidence
3. If valid concern, file dispute
4. Stake required to dispute

### I found a bug

- Non-security: GitHub issues
- Security: security@epistemic-markets.org
- Bug bounty available for serious issues

### I need more help

- **Discord**: Community support
- **Documentation**: This FAQ + other docs
- **Mentors**: Request mentor matching
- **Email**: support@epistemic-markets.org

---

## Still Have Questions?

If your question isn't answered here:

1. **Search the docs**: Use search in the documentation
2. **Ask the community**: Discord or forums
3. **Request mentor**: Get 1-1 help
4. **Submit feedback**: Help us improve this FAQ

---

*"Questions are the beginning of wisdom. Predictions are the beginning of knowledge. Reflection is the beginning of understanding."*

*Ask freely. Predict honestly. Reflect deeply.*

