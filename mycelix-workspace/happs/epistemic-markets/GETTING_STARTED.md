# Getting Started with Epistemic Markets

*Your journey from first prediction to wisdom keeper*

---

## Quick Start (5 Minutes)

### 1. Understand the Core Concept

Epistemic markets let you:
- **Predict** outcomes with honest probability estimates
- **Stake** things you value (not just money)
- **Learn** from the collective and from your own calibration
- **Contribute** wisdom for future generations

### 2. Make Your First Prediction

Find an open market that interests you. Ask yourself:
- What do I actually believe will happen?
- How confident am I? (Express as a probability)
- What am I willing to stake on this belief?
- Why do I believe this? (Write it down)

### 3. Share Your Reasoning

The most valuable part of a prediction isn't the number - it's the reasoning behind it. Share:
- Your key assumptions
- What would change your mind
- What you might be wrong about

### 4. Wait, Learn, Update

As new information arrives:
- Update your prediction if evidence warrants
- Don't update just because others do (avoid cascades)
- Record why you updated

### 5. Reflect on Resolution

When the market resolves:
- Were you right? Wrong? Partially correct?
- What does this teach you?
- Plant a wisdom seed for future predictors

---

## The Prediction Journey

### Stage 1: Newcomer (First 10 Predictions)

**Focus on:**
- Learning to express uncertainty honestly
- Making small stakes to feel the system
- Reading others' reasoning traces
- Getting comfortable being wrong

**Avoid:**
- High stakes on early predictions
- Overconfidence in unfamiliar domains
- Copying others without independent thought

**Milestone:** Complete 10 predictions with honest probability estimates and written reasoning.

---

### Stage 2: Participant (10-50 Predictions)

**Focus on:**
- Building calibration in specific domains
- Using multiple stake types (reputation, social)
- Engaging with disagreements productively
- Learning from your calibration curve

**Practices:**
- Review your calibration monthly
- Find a prediction partner who challenges you
- Start identifying your epistemic strengths/weaknesses

**Milestone:** Achieve calibration score below 0.25 Brier in at least one domain.

---

### Stage 3: Contributor (50-200 Predictions)

**Focus on:**
- Writing reasoning traces that help others learn
- Engaging in structured disagreement
- Identifying cruxes in contested questions
- Mentoring newcomers

**Practices:**
- Write one detailed reasoning trace per week
- Participate in at least one productive disagreement per month
- Plant wisdom seeds on resolved predictions

**Milestone:** Have your reasoning cited by at least 5 other predictors.

---

### Stage 4: Mentor (200+ Predictions)

**Focus on:**
- Helping newcomers develop calibration
- Contributing to market design discussions
- Identifying systemic issues (cascades, manipulation)
- Building long-horizon predictions

**Practices:**
- Mentor at least one newcomer
- Create a question market for an important unexplored question
- Make at least one prediction with 5+ year horizon

**Milestone:** Successfully help a newcomer reach Stage 2.

---

### Stage 5: Elder (Long-term Commitment)

**Focus on:**
- Planting wisdom seeds for future generations
- Contributing to protocol evolution
- Legacy stakes and intergenerational commitments
- Caring for the health of the whole system

**Practices:**
- Review and tend to your wisdom seeds annually
- Participate in meta-epistemic reflection
- Write letters to future predictors
- Consider generational stakes

**Milestone:** Have a wisdom seed germinate and influence future predictors.

---

## Understanding Stakes

### Monetary Stakes
The simplest form - you put tokens on your prediction.

**When to use:** When you have genuine financial skin in the game
**Amount:** Start small (1-10 HAM), increase as confidence warrants

### Reputation Stakes
You stake credibility in a specific domain.

**When to use:** When you have expertise you want to demonstrate
**Amount:** 10-50 points for moderate confidence, more for high confidence

### Social Stakes
You involve witnesses who will know if you're right or wrong.

**When to use:** When social accountability matters to you
**How:** Select 1-3 trusted witnesses who understand the question

### Commitment Stakes
You promise to take action based on the outcome.

**When to use:** When you want to align beliefs with behavior
**Examples:** "If X happens, I will do Y"

### Time Stakes
You lock your prediction early, earning temporal bonuses.

**When to use:** When you have early insight
**How:** Lock until resolution, earning bonus for early accuracy

### Legacy Stakes
Long-term commitments that mature over years.

**When to use:** For predictions about the long-term future
**Consideration:** Only for deep, considered predictions

---

## Understanding Epistemic Position

Every market has an **E-N-M classification** that determines how it resolves:

### Empirical Axis (How Verifiable)

| Level | Meaning | Resolution |
|-------|---------|------------|
| Measurable | Can be objectively measured | Automated oracle |
| Cryptographic | Can be verified on-chain | Smart contract |
| Private Verify | Some can verify privately | Trusted oracles |
| Testimonial | Based on witness accounts | Community consensus |
| Subjective | Personal experience | Self-report |

### Normative Axis (Whose Values)

| Level | Meaning | Stakeholders |
|-------|---------|--------------|
| Personal | Individual perspective | Just you |
| Communal | Community values | Your community |
| Network | Network-wide norms | All participants |
| Universal | Applies everywhere | Everyone |

### Materiality Axis (How Long It Matters)

| Level | Meaning | Timescale |
|-------|---------|-----------|
| Ephemeral | Momentary | Hours to days |
| Temporal | Short-term | Days to months |
| Persistent | Long-term | Months to years |
| Foundational | Permanent | Generations |

---

## Best Practices

### Before Predicting
- [ ] Read the full market description
- [ ] Understand the resolution criteria
- [ ] Check the epistemic position
- [ ] Review existing predictions and reasoning
- [ ] Form your own view before seeing others

### While Predicting
- [ ] Express honest probability (not binary yes/no)
- [ ] Write clear reasoning summary
- [ ] List key assumptions
- [ ] Acknowledge weaknesses in your reasoning
- [ ] Choose appropriate stake types

### After Predicting
- [ ] Monitor for new information
- [ ] Update if evidence warrants
- [ ] Engage with disagreements
- [ ] Don't herd-follow others

### After Resolution
- [ ] Reflect on what you learned
- [ ] Update your mental models
- [ ] Check your calibration
- [ ] Plant wisdom seeds if you have insights

---

## Common Mistakes

### Overconfidence
**Symptom:** Predictions cluster at 90%+ or 10%-
**Fix:** Practice expressing genuine uncertainty

### Underconfidence
**Symptom:** All predictions near 50%
**Fix:** You probably know more than you think - be willing to commit

### Herding
**Symptom:** Your predictions follow the crowd
**Fix:** Form views before seeing others; value independence

### Sunk Cost Bias
**Symptom:** Refusing to update despite evidence
**Fix:** Changing your mind is a virtue, not weakness

### Outcome Bias
**Symptom:** Judging predictions by outcome, not process
**Fix:** Focus on calibration, not individual results

### Domain Overreach
**Symptom:** Predicting confidently in unfamiliar areas
**Fix:** Build expertise domains; express uncertainty elsewhere

---

## Developer Quick Start

### Prerequisites
- Holochain development environment
- Node.js 18+
- Rust (for zome development)

### Installation

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-workspace
cd mycelix-workspace/happs/epistemic-markets

# Install dependencies
npm install

# Build zomes
cd zomes && cargo build --release --target wasm32-unknown-unknown

# Run tests
npm test
```

### SDK Usage

```typescript
import { EpistemicMarketsClient } from '@mycelix/epistemic-markets';

// Connect to your Holochain cell
const client = new EpistemicMarketsClient(cell);

// Create a market
const market = await client.markets.createMarket({
  title: "Will our community project succeed?",
  description: "Success = 50+ active participants by year end",
  outcomes: ["Yes", "No"],
  epistemic_position: {
    empirical: "testimonial",
    normative: "communal",
    materiality: "persistent"
  },
  closes_at: Date.now() + 90 * 24 * 60 * 60 * 1000 // 90 days
});

// Make a prediction
const prediction = await client.predictions.createPrediction({
  market_id: market.id,
  outcome: "Yes",
  probability: 0.72,
  stake: {
    monetary: { amount: 100, currency: "HAM" },
    reputation: { amount: 20, domain: "community_organizing" }
  },
  reasoning: {
    summary: "Strong early engagement suggests success",
    assumptions: [{ statement: "Funding remains stable", sensitivity: 0.3 }],
    acknowledged_weaknesses: ["Limited historical data for similar projects"]
  },
  wisdom_seed: {
    if_correct: { lesson: "Early enthusiasm can predict sustained success" },
    if_incorrect: { lesson: "Enthusiasm alone is not enough" }
  }
});
```

---

## Resources

### Documentation
- [README](./README.md) - Project overview
- [MANIFESTO](./MANIFESTO.md) - Our values and vision
- [ECOSYSTEM_INTEGRATION](./docs/ECOSYSTEM_INTEGRATION.md) - Cross-hApp patterns
- [LONG_TERM_VISION](./docs/LONG_TERM_VISION.md) - Where we're going
- [THE_DEEPER_VISION](./docs/THE_DEEPER_VISION.md) - Spiritual dimensions
- [THE_LIVING_PROTOCOL](./docs/THE_LIVING_PROTOCOL.md) - The system as life

### Technical
- [SDK Documentation](./sdk-ts/README.md) - TypeScript SDK
- [DNA Configuration](./dna.yaml) - Holochain DNA setup
- [Integration Tests](./tests/) - Example usage patterns

### Community
- [Mycelix Network](https://mycelix.net) - Parent ecosystem
- [Observatory](../observatory) - Visual dashboard

---

## Your First Market

Once you're comfortable predicting, consider creating your first market:

1. **Choose a question** that matters to your community
2. **Define clear outcomes** that are mutually exclusive and exhaustive
3. **Set appropriate epistemic position** based on how it can be resolved
4. **Write clear resolution criteria** so everyone knows what counts
5. **Set a reasonable timeframe** - not too short, not too long
6. **Invite diverse predictors** to ensure independent perspectives

---

## Welcome

You're joining a community of truth-seekers. We don't all agree - that's the point. But we share a commitment to honest inquiry, to updating our beliefs, to learning from each other.

The world needs better tools for collective intelligence. By participating, you're helping build them.

Make your first prediction. Stake something real. Share your reasoning.

The journey begins now.

---

*Questions? Reach out to the Mycelix community or open an issue in the repository.*

*Remember: The goal isn't to be right. It's to be honest about what you believe and why - and to get better at believing true things over time.*
