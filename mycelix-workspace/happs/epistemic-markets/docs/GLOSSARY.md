# Glossary

*The Language of Collective Truth-Seeking*

---

> "To understand a culture, learn its vocabulary."
> — Linguistic wisdom

> "We chose these words carefully. They encode our values."
> — Us

---

## How to Use This Glossary

Terms are organized into categories. Each entry includes:
- **Definition**: What the term means
- **Context**: Where and how it's used
- **Related**: Connected concepts
- **Example**: Concrete illustration (where helpful)

Cross-references are marked with →arrows.

---

## Core Concepts

### Epistemic Markets

**Definition**: A decentralized platform and protocol for collective truth-seeking through prediction markets, emphasizing calibration, reasoning quality, and collaborative knowledge-building over pure profit maximization.

**Context**: The overall system described in this documentation.

**Related**: →Prediction Market, →Collective Intelligence, →Truth-Seeking

**Note**: "Epistemic" means "relating to knowledge or the conditions for acquiring it." We chose this term to distinguish from prediction markets focused purely on speculation.

---

### Prediction Market

**Definition**: A market where participants trade on the outcomes of future events, with prices reflecting aggregated probability estimates.

**Context**: The fundamental mechanism underlying Epistemic Markets.

**Related**: →Market, →Outcome, →Resolution

**Example**: A market on "Will it rain tomorrow in Tokyo?" where 70¢ for "Yes" implies 70% probability.

---

### Collective Intelligence

**Definition**: The emergent knowledge that arises from aggregating the beliefs, reasoning, and insights of many independent minds.

**Context**: The primary goal of Epistemic Markets—not just price discovery, but wisdom generation.

**Related**: →Aggregation, →Wisdom, →Independence

---

### Truth-Seeking

**Definition**: The orientation toward discovering accurate beliefs about reality, valuing correctness over winning or social approval.

**Context**: The fundamental value that distinguishes Epistemic Markets from gambling or speculation.

**Related**: →Calibration, →Epistemic Virtue, →Honesty

---

## The E-N-M Framework

### E-N-M Classification

**Definition**: A three-dimensional framework for classifying questions based on their Empirical verifiability, Normative scope, and Materiality duration.

**Context**: Every market receives an E-N-M classification that determines its resolution mechanism.

**Related**: →Empirical Axis, →Normative Axis, →Materiality Axis

**Example**: E4-N3-M2 describes a scientifically measurable, universally agreed, persistent question.

---

### Empirical Axis (E)

**Definition**: The dimension measuring how the outcome can be verified.

**Levels**:
| Level | Name | Description | Example |
|-------|------|-------------|---------|
| E0 | Subjective | No external verification possible | "Was this movie good?" |
| E1 | Testimonial | Verified by witness testimony | "Did the meeting happen?" |
| E2 | Private Verify | Verifiable via credentials/ZK proofs | "Did they graduate?" |
| E3 | Cryptographic | On-chain verifiable events | "Did the transaction occur?" |
| E4 | Measurable | Scientific reproducibility | "What was the temperature?" |

**Related**: →Resolution, →Oracle, →Verification

---

### Normative Axis (N)

**Definition**: The dimension measuring whose agreement determines truth.

**Levels**:
| Level | Name | Description | Example |
|-------|------|-------------|---------|
| N0 | Personal | Individual perspective only | "Do I like this?" |
| N1 | Communal | Community consensus | "Is this appropriate here?" |
| N2 | Network | Network-wide agreement | "Is this protocol-compliant?" |
| N3 | Universal | Objective fact | "Did the sun rise?" |

**Related**: →Consensus, →Subjectivity, →Objectivity

---

### Materiality Axis (M)

**Definition**: The dimension measuring how long the question's answer matters.

**Levels**:
| Level | Name | Duration | Example |
|-------|------|----------|---------|
| M0 | Ephemeral | Hours | "Will traffic be bad this evening?" |
| M1 | Temporal | Days/weeks | "Will this bill pass this session?" |
| M2 | Persistent | Months/years | "Will this company succeed?" |
| M3 | Foundational | Permanent/indefinite | "Is this theorem true?" |

**Related**: →Long-term Markets, →Legacy Stakes

---

## MATL System

### MATL

**Definition**: Multi-dimensional Adaptive Trust Layer. The reputation and trust system that weights participant influence based on demonstrated reliability across multiple dimensions.

**Pronunciation**: "mat-ul" (rhymes with "battle")

**Context**: Used for oracle weighting, stake limits, governance participation, and access control.

**Related**: →Merit, →Accountability, →Trust Dimension, →Legacy

---

### MATL Score

**Definition**: A composite numerical score (0-1) representing a participant's overall trustworthiness, calculated from weighted components.

**Formula**: `MATL = α·Merit + β·Accountability + γ·Trust + δ·Legacy`

**Default weights**: Merit 40%, Accountability 25%, Trust 20%, Legacy 15%

**Related**: →Domain MATL, →Composite Score

---

### Merit (M in MATL)

**Definition**: The component of MATL measuring prediction accuracy and calibration over time.

**Calculation**: Based on Brier scores, calibration curves, and accuracy rates.

**Context**: The "skill" dimension—are you good at predicting?

**Related**: →Calibration, →Brier Score, →Accuracy

---

### Accountability (A in MATL)

**Definition**: The component of MATL measuring follow-through on commitments.

**Calculation**: Based on commitment stake fulfillment rate.

**Context**: The "reliability" dimension—do you do what you say?

**Related**: →Commitment Stake, →Follow-through

---

### Trust Dimension (T in MATL)

**Definition**: The component of MATL measuring community trust signals and vouching.

**Calculation**: Based on vouches received, social graph position, and trust ratings.

**Context**: The "social" dimension—do others trust you?

**Related**: →Vouching, →Social Capital

---

### Legacy (L in MATL)

**Definition**: The component of MATL measuring long-term contribution to collective wisdom.

**Calculation**: Based on wisdom seed quality, mentorship, and intergenerational contribution.

**Context**: The "wisdom" dimension—have you contributed lasting value?

**Related**: →Wisdom Seed, →Elder, →Intergenerational

---

### Domain MATL

**Definition**: A MATL score specific to a particular topic domain (e.g., climate, finance, technology).

**Context**: Participants may have different MATL scores in different domains, reflecting domain-specific expertise.

**Related**: →Domain, →Expertise, →Oracle Selection

---

### Byzantine Tolerance

**Definition**: The system's ability to reach correct consensus even when a fraction of participants are adversarial or faulty.

**Context**: MATL-weighted oracles achieve 45% Byzantine tolerance (can tolerate up to 45% malicious oracles).

**Related**: →Oracle, →Consensus, →Adversarial

---

## Stakes and Rewards

### Stake

**Definition**: Something of value risked on a prediction, lost if wrong and potentially multiplied if right.

**Types**: →Monetary Stake, →Reputation Stake, →Social Stake, →Commitment Stake, →Time Stake, →Legacy Stake

**Related**: →Risk, →Accountability, →Skin in the Game

---

### Monetary Stake

**Definition**: Currency (HAM tokens or equivalent) risked on a prediction.

**Context**: The traditional form of prediction market stake.

**Example**: "I stake 100 HAM that Bitcoin will exceed $100k."

**Related**: →HAM, →Reward Pool

---

### Reputation Stake

**Definition**: A percentage of one's MATL score risked on a prediction, decreased if wrong.

**Context**: Allows expertise to be staked without monetary cost.

**Example**: "I stake 10% of my climate MATL on this prediction."

**Related**: →MATL, →Domain MATL

---

### Social Stake

**Definition**: Visibility and identity exposure risked on a prediction.

**Levels**: Private, Limited, Community, Public

**Context**: Being publicly wrong has social consequences.

**Related**: →Visibility, →Identity

---

### Commitment Stake

**Definition**: Pledged actions conditional on prediction outcome.

**Types**: Actions if correct, actions if wrong

**Context**: Aligns beliefs with behavior.

**Example**: "If I'm wrong, I'll donate $100 to climate research."

**Related**: →Follow-through, →Accountability

---

### Time Stake

**Definition**: Research effort and evidence contribution invested in a prediction.

**Context**: Recognizes that thorough research has value regardless of outcome.

**Related**: →Evidence, →Research, →Reasoning

---

### Legacy Stake

**Definition**: Long-term reputation and intergenerational standing risked on significant predictions.

**Context**: For foundational predictions that will be remembered.

**Related**: →Legacy, →Wisdom Seed, →Elder

---

### HAM

**Definition**: Holochain Asset Module tokens, the protocol's native currency.

**Context**: Used for monetary stakes, rewards, fees, and governance.

**Related**: →Monetary Stake, →Reward, →Fee

---

### Reward Pool

**Definition**: The collected stakes from incorrect predictions, distributed to correct predictors.

**Context**: The primary source of monetary rewards.

**Related**: →Stake, →Distribution, →Winner

---

### Calibration Bonus

**Definition**: Additional reward for predictions at confidence levels matching historical accuracy.

**Context**: Rewards epistemic honesty over mere correctness.

**Example**: If you predict at 70% confidence and are right about 70% of the time at that level, you earn calibration bonuses.

**Related**: →Calibration, →Brier Score

---

### Reasoning Bonus

**Definition**: Additional reward for high-quality reasoning, regardless of prediction outcome.

**Context**: Values the thinking process, not just the answer.

**Related**: →Reasoning, →Quality Assessment

---

## Markets and Questions

### Market

**Definition**: A trading venue for predictions on a specific question with defined outcomes.

**States**: Draft, Open, Closed, Resolving, Resolved, Disputed, Voided

**Related**: →Question, →Outcome, →Resolution

---

### Market Creator

**Definition**: The participant who created a market, responsible for clear question formulation.

**Responsibilities**: Question clarity, appropriate classification, initial parameters

**Related**: →Creation Fee, →Market Quality

---

### Outcome

**Definition**: One of the possible results of a market's question.

**Types**: Binary (Yes/No), Categorical (multiple options), Scalar (numerical range)

**Related**: →Market, →Resolution, →Prediction

---

### Question Market

**Definition**: A meta-market where participants trade on whether a question is worth answering, before any prediction market exists.

**Context**: Discovers which questions deserve attention and resources.

**Related**: →Curiosity Signal, →Value Share, →Spawning Threshold

---

### Curiosity Signal

**Definition**: A lightweight, low-cost expression of interest in a question.

**Context**: The first step in elevating a question toward market creation.

**Related**: →Question Market, →Interest

---

### Value Share

**Definition**: A tradeable stake in a question's worthiness, paying out if the question spawns a successful market.

**Context**: Bet that the answer to this question is worth knowing.

**Related**: →Question Market, →Spawning

---

### Spawning Threshold

**Definition**: The accumulated value in a question market required to automatically create a prediction market.

**Context**: Questions must demonstrate sufficient interest before becoming markets.

**Related**: →Question Market, →Market Creation

---

## Predictions and Reasoning

### Prediction

**Definition**: A stated belief about a market's outcome, with associated confidence and stake.

**Components**: Outcome choice, confidence level, stake, reasoning

**Related**: →Confidence, →Stake, →Reasoning

---

### Confidence

**Definition**: The probability (0-100%) a predictor assigns to their chosen outcome.

**Context**: Distinguishes strong beliefs (95%) from uncertain ones (55%).

**Calibration**: Well-calibrated predictors' 70% predictions come true about 70% of the time.

**Related**: →Calibration, →Probability, →Uncertainty

---

### Reasoning

**Definition**: The explanation for why a predictor believes their prediction is correct.

**Context**: Required for higher-stake predictions, rewarded for quality.

**Quality factors**: Evidence cited, logic clarity, consideration of alternatives, appropriate uncertainty

**Related**: →Reasoning Bonus, →Evidence, →Quality

---

### Evidence

**Definition**: Data, sources, or observations supporting a prediction's reasoning.

**Context**: Strong evidence improves reasoning quality scores.

**Related**: →Reasoning, →Sources, →Knowledge Graph

---

### Calibration

**Definition**: The alignment between stated confidence levels and actual accuracy rates.

**Context**: The primary measure of epistemic skill.

**Example**: Perfect calibration means 70% confident predictions are correct 70% of the time.

**Related**: →Brier Score, →Overconfidence, →Underconfidence

---

### Brier Score

**Definition**: A proper scoring rule measuring prediction accuracy, calculated as the mean squared error between predicted probabilities and outcomes.

**Formula**: `Brier = (1/N) × Σ(prediction - outcome)²`

**Range**: 0 (perfect) to 1 (worst)

**Context**: Used internally for Merit calculation.

**Related**: →Calibration, →Accuracy, →Scoring Rule

---

### Overconfidence

**Definition**: Systematic tendency to assign higher probabilities than warranted by accuracy.

**Context**: A common calibration failure—thinking you're more certain than you should be.

**Related**: →Calibration, →Confidence, →Bias

---

### Underconfidence

**Definition**: Systematic tendency to assign lower probabilities than warranted by accuracy.

**Context**: Less common but also a calibration failure—not trusting your knowledge.

**Related**: →Calibration, →Confidence, →Bias

---

## Resolution and Oracles

### Resolution

**Definition**: The process of determining a market's outcome and distributing rewards.

**Stages**: Initiation, Oracle voting, Consensus check, Finalization, Distribution

**Related**: →Oracle, →Consensus, →Outcome

---

### Oracle

**Definition**: A participant qualified to vote on market resolution, selected based on domain MATL.

**Requirements**: Minimum domain MATL, no conflict of interest, available during resolution

**Related**: →Oracle Vote, →MATL, →Resolution

---

### Oracle Vote

**Definition**: An oracle's declaration of what they believe the correct outcome is.

**Weight**: Proportional to oracle's domain MATL score squared

**Related**: →Oracle, →Weighted Voting, →Consensus

---

### Consensus

**Definition**: Sufficient agreement among weighted oracle votes to finalize resolution.

**Threshold**: Typically 67% of weighted votes for supermajority

**Related**: →Oracle Vote, →Resolution, →Byzantine Tolerance

---

### Dispute

**Definition**: A formal challenge to a market's resolution, requiring stake and triggering review.

**Context**: Safety mechanism against incorrect or manipulated resolution.

**Process**: Dispute stake → Extended oracle pool → Re-resolution → Governance escalation if needed

**Related**: →Resolution, →Appeal, →Governance

---

### Appeal

**Definition**: Escalation of a disputed resolution to higher governance bodies.

**Context**: Final recourse when normal resolution fails.

**Related**: →Dispute, →Governance, →Protocol Council

---

## Wisdom and Learning

### Wisdom Seed

**Definition**: A structured insight, lesson, or piece of knowledge contributed for future generations.

**Context**: The mechanism for accumulating collective wisdom beyond individual predictions.

**Components**: Content, context, author, verification status, citations

**Related**: →Wisdom, →Legacy, →Intergenerational

---

### Wisdom Keeper

**Definition**: A participant recognized for consistently contributing valuable wisdom seeds.

**Context**: An earned role with special privileges for wisdom contribution.

**Related**: →Wisdom Seed, →Legacy, →Elder

---

### Elder

**Definition**: A participant with long tenure and demonstrated wisdom, eligible for governance participation.

**Requirements**: Sustained high MATL, significant wisdom contribution, community trust

**Related**: →Elder Council, →Governance, →Legacy

---

### Elder Council

**Definition**: A governance body of Elders with special responsibilities for long-term protocol health.

**Responsibilities**: Constitutional interpretation, red line enforcement, wisdom curation

**Related**: →Elder, →Governance, →Protocol Council

---

### Calibration Training

**Definition**: Exercises and games designed to improve participants' calibration skills.

**Context**: Educational feature to help predictors become more epistemically skilled.

**Related**: →Calibration, →Training Ground, →Learning

---

### Training Ground

**Definition**: A practice environment with simulated stakes for learning without real risk.

**Context**: Safe space for newcomers to develop skills.

**Related**: →Calibration Training, →Newcomer, →Practice

---

## Community and Roles

### Predictor

**Definition**: Any participant who makes predictions in markets.

**Context**: The basic role—anyone can be a predictor.

**Related**: →Prediction, →Participant

---

### Mentor

**Definition**: An experienced participant who helps guide newcomers.

**Context**: Earned through demonstrated helpfulness and community contribution.

**Related**: →Newcomer, →Teaching, →Community

---

### Newcomer

**Definition**: A participant in their first period of engagement (typically first 3 months).

**Context**: Special protections, lower requirements, mentor matching available.

**Related**: →Mentor, →Training Ground, →Onboarding

---

### Guardian

**Definition**: A participant designated to monitor for security threats and manipulation.

**Context**: Special access to monitoring tools and alert systems.

**Related**: →Security, →Manipulation, →Monitoring

---

### Community Assembly

**Definition**: The full body of all verified participants, the ultimate source of legitimacy.

**Context**: Can override other governance bodies through supermajority vote.

**Related**: →Governance, →Democracy, →Participation

---

## Governance

### Protocol Council

**Definition**: The primary governance body for major protocol decisions.

**Composition**: Elected members plus Elder representatives

**Related**: →Governance, →Proposal, →Voting

---

### Parameter Committee

**Definition**: The governance body responsible for tuning protocol parameters.

**Scope**: Fee levels, thresholds, weights, limits

**Related**: →Governance, →Parameters, →Tuning

---

### Proposal

**Definition**: A formal suggestion for protocol change, subject to governance process.

**Types**: Parameter change, Feature addition, Policy change, Constitutional amendment

**Related**: →Governance, →Voting, →Approval

---

### Quorum

**Definition**: The minimum participation required for a governance vote to be valid.

**Context**: Prevents small groups from making decisions for everyone.

**Related**: →Voting, →Participation, →Legitimacy

---

### Futarchy

**Definition**: A governance method where decisions are made based on prediction market outcomes about their effects.

**Context**: Integration with governance hApp for policy decisions.

**Example**: "Approve policy X if the market predicts it will increase metric Y"

**Related**: →Governance, →Prediction Market, →Policy

---

## Technical Terms

### Holochain

**Definition**: The distributed computing framework on which Epistemic Markets runs, featuring agent-centric architecture.

**Key features**: No global consensus, each agent owns their data, peer validation

**Related**: →DNA, →Zome, →Agent

---

### DNA

**Definition**: A Holochain application package containing code and configuration.

**Context**: The epistemic-markets DNA contains all zomes.

**Related**: →Holochain, →Zome, →hApp

---

### Zome

**Definition**: A module within a Holochain DNA providing specific functionality.

**Zomes in Epistemic Markets**: markets, predictions, resolution, scoring, question_markets, markets_bridge

**Related**: →DNA, →Holochain, →Module

---

### Entry

**Definition**: A piece of data stored on a Holochain source chain.

**Types**: Market entries, prediction entries, resolution entries, etc.

**Related**: →Holochain, →Validation, →Storage

---

### Validation

**Definition**: The process by which peers verify that entries follow protocol rules.

**Context**: Ensures no invalid data enters the system.

**Related**: →Entry, →Rules, →Peer

---

### Entry Hash

**Definition**: A unique identifier for an entry, derived from its content.

**Context**: Used to reference markets, predictions, and other entries.

**Related**: →Entry, →Hash, →Reference

---

### Agent

**Definition**: A participant's node in the Holochain network, identified by public key.

**Context**: Each user runs an agent that holds their source chain.

**Related**: →Holochain, →Public Key, →Node

---

### Source Chain

**Definition**: An agent's personal record of all their actions, cryptographically linked.

**Context**: The authoritative record of what a participant has done.

**Related**: →Agent, →Holochain, →History

---

## Mechanisms

### LMSR (Logarithmic Market Scoring Rule)

**Definition**: An automated market maker algorithm that provides liquidity and price discovery.

**Context**: Used for market pricing when direct peer trading is thin.

**Parameters**: Liquidity parameter (b) controls price sensitivity

**Related**: →Market Maker, →Liquidity, →Price

---

### Commit-Reveal

**Definition**: A two-phase protocol where predictions are first committed (hashed) then revealed.

**Context**: Ensures prediction independence by hiding predictions until reveal phase.

**Related**: →Independence, →Hash, →Privacy

---

### Zero-Knowledge Proof (ZK)

**Definition**: A cryptographic proof that something is true without revealing the underlying data.

**Context**: Used for private prediction aggregation and credential verification.

**Related**: →Privacy, →Cryptography, →Verification

---

### Quadratic Voting

**Definition**: A voting mechanism where vote cost increases quadratically with vote strength.

**Context**: Used in some governance decisions to balance intensity with breadth.

**Formula**: Cost to cast n votes = n²

**Related**: →Governance, →Voting, →Weighting

---

### Conviction Voting

**Definition**: A voting mechanism where votes gain weight the longer they're held.

**Context**: Rewards patient, considered positions over reactive voting.

**Related**: →Governance, →Voting, →Time-weighting

---

## Culture and Practice

### Ritual

**Definition**: A recurring practice that sustains community culture and epistemic health.

**Types**: →Daily Rituals, →Weekly Rituals, →Monthly Rituals, →Annual Rituals

**Related**: →Culture, →Practice, →Community

---

### Daily Rituals

**Definition**: Practices performed daily by engaged participants.

**Examples**: Morning Calibration Check, Prediction Pause, Evening Review

**Related**: →Ritual, →Habit, →Practice

---

### Calibration Check

**Definition**: A daily review of one's prediction accuracy and calibration metrics.

**Context**: Supports continuous improvement and self-awareness.

**Related**: →Calibration, →Ritual, →Self-assessment

---

### Genesis Market

**Definition**: The first market created in the system, typically about the system itself.

**Context**: A ceremonial and practical founding moment.

**Related**: →Genesis, →Founding, →Bootstrap

---

### Red Line

**Definition**: An absolute limit that cannot be crossed regardless of governance votes.

**Context**: Constitutional protections against catastrophic changes.

**Examples**: No prediction markets on individual harm, no removal of core epistemic values

**Related**: →Constitution, →Governance, →Limits

---

## Anti-Patterns and Failures

### Information Cascade

**Definition**: A failure mode where predictors follow others rather than their own information.

**Cause**: Visible early predictions influencing later ones

**Mitigation**: Commit-reveal, independent prediction mode

**Related**: →Independence, →Herding, →Bias

---

### Sybil Attack

**Definition**: An attack using multiple fake identities to gain disproportionate influence.

**Mitigation**: Identity verification, stake requirements, MATL weighting

**Related**: →Security, →Identity, →Attack

---

### Oracle Collusion

**Definition**: Coordinated false voting by multiple oracles to manipulate resolution.

**Mitigation**: MATL weighting, collusion detection, diversity requirements

**Related**: →Oracle, →Security, →Manipulation

---

### Epistemic Capture

**Definition**: Systematic bias where certain viewpoints dominate regardless of accuracy.

**Cause**: Social pressure, filter bubbles, homogeneous community

**Mitigation**: Diversity metrics, independence requirements, devil's advocate roles

**Related**: →Bias, →Diversity, →Independence

---

## See Also

### Cross-References

- For detailed mechanisms: →ADVANCED_MECHANISMS.md
- For system design: →DESIGN_PRINCIPLES.md
- For governance details: →GOVERNANCE.md
- For economic model: →ECONOMICS.md
- For security: →SECURITY.md
- For common questions: →FAQ.md

---

## Contributing to the Glossary

This glossary is a living document. To suggest additions or corrections:

1. Terms must be used in official documentation
2. Definitions should be clear to newcomers
3. Examples are encouraged
4. Cross-references help navigation

Submit suggestions via the standard contribution process.

---

*"The limits of my language mean the limits of my world."*
*— Ludwig Wittgenstein*

*We expand our language to expand our collective understanding.*

