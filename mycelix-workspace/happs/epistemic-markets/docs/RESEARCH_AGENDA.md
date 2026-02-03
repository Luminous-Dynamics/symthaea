# Research Agenda

*The Questions We Haven't Answered Yet*

---

> "The important thing is not to stop questioning."
> — Albert Einstein

> "The most important questions are the ones we haven't thought to ask."
> — Us

---

## Introduction

Epistemic Markets is built on hypotheses—predictions about what will work. Many of these hypotheses are unproven. This document catalogs the open research questions that could improve our understanding and design.

This is an invitation to researchers, practitioners, and curious minds: help us answer these questions.

---

## Part I: Foundational Questions

### F1: Does aggregation actually work?

**The claim:** Aggregating predictions from diverse, independent minds produces better forecasts than any individual.

**Open questions:**
- Under what conditions does aggregation fail?
- How much independence is needed?
- What's the minimum viable diversity?
- Does aggregation work equally well for all types of questions?

**Research approaches:**
- Comparative studies vs. expert prediction
- Systematic analysis of aggregation failures
- Simulation of different aggregation conditions
- Historical analysis of prediction market accuracy

**Priority:** Critical—this is a foundational assumption

---

### F2: Can calibration be taught?

**The claim:** Humans can improve their calibration through practice and feedback.

**Open questions:**
- What training methods work best?
- How transferable is calibration across domains?
- Are there limits to calibration improvement?
- Does improved calibration persist over time?
- Are some people unable to improve?

**Research approaches:**
- Longitudinal training studies
- Cross-domain transfer experiments
- Comparison of training methodologies
- Neuroimaging of calibration processes

**Priority:** High—underlies our training systems

---

### F3: What makes reasoning high-quality?

**The claim:** Some reasoning is better than other reasoning, and we can identify and reward it.

**Open questions:**
- Can reasoning quality be reliably measured?
- Is quality domain-specific or general?
- Do people agree on what constitutes good reasoning?
- Can AI evaluate reasoning quality?
- Does rewarding reasoning quality improve predictions?

**Research approaches:**
- Expert evaluation studies
- Inter-rater reliability analysis
- AI reasoning evaluation benchmarks
- Incentive experiments

**Priority:** High—central to our value proposition

---

### F4: Does skin in the game improve predictions?

**The claim:** Stakes make people more honest and accurate.

**Open questions:**
- What types of stakes work best?
- Is there a minimum effective stake?
- Can stakes be too high (and counterproductive)?
- Do non-monetary stakes work as well as monetary?
- Does the type of stake interact with question type?

**Research approaches:**
- Comparative experiments with different stake types
- Stake magnitude studies
- Non-monetary stake validation
- Cross-cultural stake research

**Priority:** High—central mechanism

---

### F5: Can wisdom be transmitted?

**The claim:** Wisdom seeds plant knowledge that future generations can use.

**Open questions:**
- Do wisdom seeds actually influence future predictors?
- What makes a wisdom seed "germinate"?
- How should wisdom be structured for transmission?
- Does intergenerational knowledge accumulate?
- What's the half-life of wisdom?

**Research approaches:**
- Longitudinal tracking of wisdom seed influence
- A/B testing of wisdom seed formats
- Citation network analysis
- Qualitative interviews with seed users

**Priority:** Medium—important for long-term vision

---

## Part II: Mechanism Design Questions

### M1: Optimal market structure

**Questions:**
- LMSR vs. order book vs. other mechanisms?
- Optimal liquidity parameters?
- When should markets use automated market makers?
- How to handle thin markets?
- Optimal market duration by question type?

**Specific hypotheses to test:**
- LMSR works better for low-liquidity questions
- Longer markets have better calibration
- Subsidy improves price discovery more than volume

---

### M2: Resolution mechanism design

**Questions:**
- Optimal oracle selection criteria?
- Best voting mechanisms for resolution?
- How to handle genuinely ambiguous outcomes?
- Optimal challenge and appeal processes?
- When should resolution be automated vs. human?

**Specific hypotheses to test:**
- MATL weighting outperforms equal weighting
- Multi-round resolution improves accuracy
- Delayed resolution reduces manipulation

---

### M3: Reputation system dynamics

**Questions:**
- Optimal decay rates for reputation?
- How to bootstrap reputation for newcomers?
- Domain-specific vs. general reputation?
- Can reputation systems resist gaming?
- How should reputation transfer across contexts?

**Specific hypotheses to test:**
- Faster decay encourages continued participation
- Domain-specific reputation outperforms general
- Reputation diversity improves system resilience

---

### M4: Anti-manipulation mechanisms

**Questions:**
- Which manipulation attacks are most likely?
- Effectiveness of detection vs. prevention?
- Trade-offs between security and usability?
- Can manipulation be made economically unviable?
- How to handle sophisticated attackers?

**Specific hypotheses to test:**
- Economic penalties deter more than detection
- Coordination detection reduces collusion
- Transparency deters manipulation

---

### M5: Question market design

**Questions:**
- How to value questions before answers are known?
- Optimal spawning thresholds?
- How to prevent question spam?
- How to surface underexplored questions?
- Meta-questions about questions?

**Specific hypotheses to test:**
- Higher spawning thresholds improve market quality
- Expert endorsement improves question value estimation
- Question markets improve attention allocation

---

## Part III: Behavioral Questions

### B1: Information cascades

**Questions:**
- How common are cascades in practice?
- What triggers cascade formation?
- Can interface design reduce cascades?
- Do cascades always harm accuracy?
- Recovery from cascade failures?

**Specific hypotheses to test:**
- Hiding early predictions reduces cascades
- Diversity incentives break cascade formation
- Independent prediction mode improves outcomes

---

### B2: Overconfidence and underconfidence

**Questions:**
- Prevalence in prediction market participants?
- Domain-specific patterns?
- Effectiveness of debiasing interventions?
- Does experience reduce miscalibration?
- Cultural variations in confidence bias?

**Specific hypotheses to test:**
- Immediate feedback improves calibration
- Experts are miscalibrated outside their domain
- Calibration training reduces overconfidence

---

### B3: Motivation and participation

**Questions:**
- What motivates sustained participation?
- Role of intrinsic vs. extrinsic motivation?
- Optimal balance of gamification?
- How to prevent burnout?
- What causes dropout?

**Specific hypotheses to test:**
- Intrinsic motivation predicts sustained participation
- Leaderboards increase but also decrease participation
- Community belonging reduces dropout

---

### B4: Collaborative truth-seeking

**Questions:**
- Do adversarial collaborations actually produce synthesis?
- When does disagreement become productive?
- Optimal size for deliberation groups?
- Role of facilitation in productive disagreement?
- Can structured disagreement be scaled?

**Specific hypotheses to test:**
- Structured protocols improve synthesis rates
- Smaller groups produce better synthesis
- Crux identification improves resolution speed

---

### B5: Trust and credibility

**Questions:**
- How do users develop trust in the system?
- Role of transparency in trust building?
- Trust recovery after failures?
- Trust calibration (appropriate vs. inappropriate trust)?
- Cultural variations in trust formation?

**Specific hypotheses to test:**
- Visible track records build trust faster
- Acknowledged failures increase trust more than hidden successes
- Gradual stake increases reflect trust development

---

## Part IV: Technical Questions

### T1: Scalability limits

**Questions:**
- Maximum participants per market?
- Maximum concurrent markets?
- Resolution mechanism scaling?
- Storage requirements at scale?
- Latency requirements for fair participation?

**Specific investigations:**
- Load testing under realistic conditions
- Holochain-specific scaling studies
- Comparative architecture analysis

---

### T2: Privacy-preserving aggregation

**Questions:**
- Can we aggregate predictions without revealing individuals?
- Performance trade-offs of privacy techniques?
- What level of privacy is actually needed?
- Privacy vs. transparency trade-offs?
- Zero-knowledge proofs for prediction markets?

**Specific investigations:**
- MPC performance benchmarks
- ZK proof implementation studies
- User privacy preference research

---

### T3: Cross-chain integration

**Questions:**
- Interoperability with other prediction markets?
- Resolution oracle sharing?
- Reputation portability?
- Cross-chain market creation?
- Standards for prediction market interoperability?

**Specific investigations:**
- Protocol design for interoperability
- Cross-chain resolution mechanisms
- Reputation translation standards

---

### T4: AI integration

**Questions:**
- Role of AI in prediction aggregation?
- AI as oracle or participant?
- Human-AI collaborative prediction?
- AI detection and handling?
- AI for reasoning quality assessment?

**Specific investigations:**
- AI predictor performance studies
- Human-AI teaming experiments
- AI oracle reliability testing

---

### T5: Mobile and offline

**Questions:**
- Prediction market UX for mobile?
- Offline prediction with later sync?
- Low-bandwidth participation?
- Accessibility on limited devices?
- Voice-based participation?

**Specific investigations:**
- Mobile UX research
- Offline-first architecture studies
- Accessibility testing on diverse devices

---

## Part V: Social Questions

### S1: Community formation

**Questions:**
- How do epistemic communities form?
- Role of shared predictions in community building?
- Optimal community size for epistemic health?
- How to prevent community fragmentation?
- Cross-community collaboration patterns?

**Specific investigations:**
- Network analysis of prediction communities
- Community health metrics development
- Comparative community studies

---

### S2: Expertise and authority

**Questions:**
- How should expertise be weighted?
- Credential verification vs. track record?
- Expert capture risks?
- Democratic vs. meritocratic balance?
- Expertise recognition across domains?

**Specific investigations:**
- Expert prediction comparison studies
- Credential value analysis
- Authority emergence patterns

---

### S3: Cultural adaptation

**Questions:**
- Prediction market patterns across cultures?
- Culturally-specific biases?
- Localization beyond translation?
- Cultural appropriateness of mechanisms?
- Universal vs. culturally-specific features?

**Specific investigations:**
- Cross-cultural prediction studies
- Cultural bias identification
- Localization impact analysis

---

### S4: Power and inequality

**Questions:**
- Does the system create or reduce inequality?
- Access barriers and their effects?
- Wealth concentration patterns?
- Power accumulation risks?
- Equity interventions and their effects?

**Specific investigations:**
- Inequality metrics tracking
- Access barrier analysis
- Power concentration monitoring
- Intervention effectiveness studies

---

### S5: Long-term social effects

**Questions:**
- Does participation improve reasoning in daily life?
- Effects on public discourse quality?
- Spillover to other institutions?
- Generational effects of epistemic training?
- Unintended social consequences?

**Specific investigations:**
- Longitudinal participant studies
- Discourse quality analysis
- Institutional spillover research
- Long-term impact assessment

---

## Part VI: Philosophical Questions

### P1: Epistemology of prediction markets

**Questions:**
- What epistemological status do market prices have?
- Are aggregated predictions "knowledge"?
- Relationship to other epistemic institutions?
- Limits of prediction market epistemology?
- Novel epistemic phenomena in markets?

**Specific investigations:**
- Philosophical analysis of market epistemology
- Comparison with scientific consensus
- Epistemological case studies

---

### P2: Ethics of prediction markets

**Questions:**
- Moral status of betting on outcomes?
- Ethical boundaries for market creation?
- Responsibility for self-fulfilling prophecies?
- Justice in prediction market design?
- Rights in epistemic systems?

**Specific investigations:**
- Ethical framework development
- Case-by-case ethical analysis
- Cross-cultural ethics research

---

### P3: Ontology of collective belief

**Questions:**
- What is collective belief?
- Emergence of group epistemics?
- Reductionism vs. emergence in group knowledge?
- Identity of epistemic communities?
- Persistence of collective beliefs?

**Specific investigations:**
- Philosophical analysis of group epistemics
- Empirical study of belief aggregation
- Emergence pattern identification

---

### P4: Wisdom and knowledge

**Questions:**
- Distinction between knowledge and wisdom?
- Can wisdom be formalized?
- Transmission mechanisms for tacit knowledge?
- Relationship between calibration and wisdom?
- Metrics for wisdom?

**Specific investigations:**
- Philosophical analysis of wisdom
- Tacit knowledge transmission studies
- Wisdom measurement development

---

### P5: Truth and politics

**Questions:**
- Can truth be separated from power?
- Political implications of epistemic infrastructure?
- Prediction markets in democratic societies?
- Technocratic risks?
- Epistemic justice concerns?

**Specific investigations:**
- Political philosophy analysis
- Democratic theory implications
- Epistemic justice framework application

---

## Part VII: Research Infrastructure

### For researchers

We provide:
- **Data access** - Anonymized prediction data for analysis
- **API access** - Programmatic access to system data
- **Research partnerships** - Collaboration on priority questions
- **Publication support** - Help getting research published
- **Funding** - Grants for priority research

### How to get involved

1. **Identify** a question from this agenda (or propose your own)
2. **Propose** a research design
3. **Contact** us with your proposal
4. **Collaborate** on research design and execution
5. **Publish** and share findings

### Research ethics

All research must:
- Respect participant privacy
- Obtain appropriate consent
- Minimize harm
- Share findings openly
- Acknowledge funding sources

---

## Priority Matrix

| Question | Impact | Feasibility | Priority |
|----------|--------|-------------|----------|
| F1: Aggregation effectiveness | Critical | Moderate | Highest |
| F2: Calibration training | High | High | Highest |
| M4: Anti-manipulation | High | Moderate | High |
| B1: Information cascades | High | High | High |
| F3: Reasoning quality | High | Moderate | High |
| M1: Market structure | Medium | High | Medium |
| S1: Community formation | Medium | Moderate | Medium |
| P1: Market epistemology | Medium | Moderate | Medium |
| T2: Privacy-preserving | Medium | Low | Lower |
| P5: Truth and politics | High | Low | Long-term |

---

## Conclusion

These questions are not obstacles—they are opportunities. Every unanswered question is a chance to learn something important about collective truth-seeking.

We don't have all the answers. That's the point.

If you can help answer any of these questions, please reach out. The future of collective intelligence depends on getting this right.

---

*"The only true wisdom is in knowing you know nothing."*
*— Socrates*

*We know something. We don't know everything. Help us learn.*

---

## How to Contribute

- **Email**: research@mycelix.net
- **Proposals**: Submit via GitHub issues with "research" label
- **Data requests**: data-access@mycelix.net
- **Collaboration**: partnerships@mycelix.net

We are especially interested in hearing from:
- Epistemologists
- Behavioral economists
- Mechanism designers
- Social scientists
- Data scientists
- Ethicists
- Anyone with rigorous methodology and genuine curiosity
