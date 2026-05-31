# Design Principles

*The Philosophy Behind Every Decision*

---

> "Design is not just what it looks like and feels like.
> Design is how it works."
> — Steve Jobs

> "Design is how it feels to use it.
> And how it feels to be used by it."
> — Us

---

## The Meta-Principle

**Every design decision should be answerable with: "This serves truth-seeking by..."**

If we cannot complete that sentence, we should not make that decision.

---

## Part I: Human-Centered Design

### Principle 1: Dignity Before Efficiency

The system must preserve human dignity even when efficiency suffers.

**Implications:**
- Never shame users for being wrong
- Never gamify to the point of addiction
- Never optimize engagement over well-being
- Never treat humans as data sources

**Anti-patterns to avoid:**
- Leaderboards that humiliate
- Notifications designed to create anxiety
- Dark patterns that exploit cognitive biases
- Metrics that reduce humans to numbers

**Design test:** Would I want my grandmother to use this? Would I want my child to use this? Would I be proud to explain exactly how this works?

### Principle 2: Progressive Disclosure

Reveal complexity as users are ready for it.

**The Onion Model:**
```
Layer 0: Make a prediction (anyone can do this)
    ↓
Layer 1: Add reasoning (optional, encouraged)
    ↓
Layer 2: Multi-dimensional stakes (discovered naturally)
    ↓
Layer 3: Belief graphs (for those who want depth)
    ↓
Layer 4: Meta-markets (for system thinkers)
    ↓
Layer 5: Protocol governance (for stewards)
```

**Each layer should:**
- Be complete in itself (not feel like "lite version")
- Hint at deeper layers without overwhelming
- Allow retreat without shame
- Reward exploration without requiring it

### Principle 3: Friction by Design

Some friction is intentional and valuable.

**Good friction:**
- Pause before high-stakes predictions (reflection)
- Confirmation for irreversible actions (care)
- Cooling-off periods for disputes (wisdom)
- Requirement to articulate reasoning (depth)

**Bad friction:**
- Unnecessary clicks (annoyance)
- Confusing navigation (frustration)
- Unclear error messages (helplessness)
- Slow performance (disrespect)

**Design test:** Is this friction serving the user's long-term interests, or just getting in their way?

### Principle 4: Failure as Teacher

Every failure should be a learning opportunity.

**When predictions are wrong:**
```
Not: "You lost 50 points"
But: "Your prediction didn't match reality. Here's what we can learn..."

Include:
- What the base rate was
- How others predicted
- What information was available
- What might have helped
- Invitation to reflect
```

**When the system fails:**
```
Not: "Error 500"
But: "Something went wrong on our end. Here's what happened,
     what we're doing about it, and how you can help us improve."
```

### Principle 5: Accessibility as Foundation

Accessibility is not a feature—it is a foundation.

**Requirements:**
- Screen reader compatible from day one
- Keyboard navigation for everything
- High contrast modes
- Adjustable text sizes
- No time-limited interactions without extension options
- Clear, simple language options
- Multiple input modalities

**Design test:** Can someone with visual impairment, motor difficulties, cognitive differences, or limited technical literacy use every feature?

---

## Part II: Epistemic Design

### Principle 6: Calibration Over Correctness

The goal is not to be right, but to know how right you are.

**Implications:**
- Reward well-calibrated wrong predictions over lucky right ones
- Track calibration curves prominently
- Celebrate uncertainty honestly expressed
- Penalize overconfidence more than underconfidence

**UI implications:**
- Show calibration graphs, not just win/loss records
- Highlight "good losses" (well-reasoned predictions that happened to be wrong)
- De-emphasize lucky wins
- Make probability expression intuitive (sliders, not just numbers)

### Principle 7: Independence Before Aggregation

Protect independent thinking from social pressure.

**Mechanisms:**
- Option to predict before seeing others
- Delayed revelation of aggregate positions
- Diversity bonuses in scoring
- Detection and dampening of cascades
- Private prediction modes

**UI implications:**
- Default: make prediction, then see others
- Clear visual distinction between "your view" and "the crowd"
- Warnings when predicting in direction of recent movement
- Celebration of dissenting views (when well-reasoned)

### Principle 8: Reasoning is Primary

The reasoning behind a prediction is more valuable than the prediction itself.

**Implications:**
- Reasoning traces should be prominent, not hidden
- Good reasoning should be rewarded even when wrong
- Reasoning quality should affect weighting
- Reasoning should be searchable, citable, buildable

**UI implications:**
- Large, prominent reasoning input fields
- Templates and prompts for good reasoning
- Easy citation of others' reasoning
- Threading for reasoning discussions

### Principle 9: Temporal Humility

The future is uncertain; design for uncertainty.

**Implications:**
- Long-term predictions should be treated differently from short-term
- Update mechanisms should be easy and encouraged
- Historical predictions should be preserved with context
- Resolution mechanisms should handle ambiguity

**UI implications:**
- Clear visual distinction by time horizon
- Easy update flows with change tracking
- "What I knew when I predicted" context preservation
- Graceful handling of unresolvable predictions

### Principle 10: Epistemic Diversity

Different ways of knowing have different strengths.

**Recognize multiple epistemic modes:**
- Analytical (formal reasoning)
- Intuitive (pattern recognition)
- Experiential (lived knowledge)
- Social (collective sensing)
- Embodied (felt sense)

**Design implications:**
- Don't require everything to be articulated formally
- Allow "gut feeling" predictions with appropriate weighting
- Value experiential expertise
- Create space for qualitative insights

---

## Part III: Social Design

### Principle 11: Generosity Over Competition

Truth-seeking is collaborative, not zero-sum.

**Mechanisms:**
- Synthesis rewards (finding common ground)
- Crux identification rewards (clarifying disagreement)
- Teaching rewards (helping others improve)
- Attribution systems (crediting intellectual debts)

**Anti-patterns:**
- Pure zero-sum betting
- Winner-take-all tournaments
- Leaderboards that emphasize ranking over growth
- Systems where helping others hurts you

### Principle 12: Repairable Relationships

Conflicts should be healable.

**Dispute resolution design:**
```
Level 1: Direct dialogue (structured, facilitated)
    ↓
Level 2: Mediation (neutral third party)
    ↓
Level 3: Community deliberation (jury of peers)
    ↓
Level 4: Formal arbitration (designated authorities)
    ↓
Level 5: Fork (irreconcilable differences)
```

**Each level should:**
- Emphasize understanding over winning
- Preserve dignity of all parties
- Create learning opportunities
- Allow face-saving exits
- Document lessons for the community

### Principle 13: Power Transparency

Make power visible and accountable.

**What must be transparent:**
- How decisions are made
- Who has influence and why
- How influence is earned and lost
- What cannot be changed (and why)

**UI implications:**
- Visible reputation/influence scores
- Audit trails for all governance decisions
- Clear documentation of rules and their origins
- Easy access to "why does X have power over Y"

### Principle 14: Newcomer Welcome

The system is only as healthy as its ability to integrate new minds.

**Newcomer experience design:**
- Clear, non-intimidating entry points
- Low-stakes practice environments
- Mentorship connections
- Gradual responsibility increases
- Protection from predatory behavior
- Celebration of beginner's mind

**Anti-patterns:**
- Insider jargon without explanation
- High barriers to participation
- Hazing or trial-by-fire
- Mockery of newcomer mistakes
- Exclusive cliques

### Principle 15: Exit Rights

Users must always be able to leave.

**Exit design:**
- Clear data export
- Reputation portability (where possible)
- No lock-in mechanisms
- Graceful account deletion
- Community memory of contributions (with consent)

**Anti-patterns:**
- Making exit difficult or shameful
- Data hostage-taking
- Reputation that only exists inside the system
- Social pressure against leaving

---

## Part IV: Technical Design

### Principle 16: Resilience Over Optimization

The system should degrade gracefully, not fail catastrophically.

**Implications:**
- No single points of failure
- Fallback mechanisms for every critical path
- Graceful degradation under load
- Clear communication when things break

**Architecture implications:**
- Distributed by default (Holochain)
- Local-first when possible
- Async-tolerant design
- Self-healing mechanisms

### Principle 17: Privacy by Default

Minimize data collection; maximize user control.

**Data minimization:**
- Collect only what's necessary
- Delete when no longer needed
- Aggregate when individual data isn't required
- Anonymize when identity isn't relevant

**User control:**
- Clear privacy settings
- Granular sharing controls
- Easy data export
- Right to be forgotten (where technically possible)

### Principle 18: Auditability

Every action should be traceable and verifiable.

**Implications:**
- Immutable action logs
- Cryptographic verification
- Open algorithms
- Reproducible computations

**But also:**
- Privacy-preserving where needed
- Aggregated audit trails when individual privacy matters
- Clear distinction between public and private audit

### Principle 19: Evolvability

The system must be able to change.

**Upgrade design:**
- Clear versioning
- Migration paths
- Backward compatibility where possible
- Graceful deprecation
- Community input on changes

**Governance of change:**
- Transparent proposal process
- Stakeholder input mechanisms
- Testing requirements
- Rollback capabilities

### Principle 20: Interoperability

The system should play well with others.

**Standards:**
- Open protocols
- Documented APIs
- Standard data formats
- Bridge mechanisms

**Philosophy:**
- Other systems are partners, not competitors
- Data should flow where it's useful
- Don't build walled gardens
- Enable unexpected integrations

---

## Part V: Aesthetic Design

### Principle 21: Beauty Serves Function

Aesthetics should support, not distract from, truth-seeking.

**Visual design principles:**
- Clarity over decoration
- Calm over excitement
- Consistency over novelty
- Information density that matches cognitive load

**What beauty means here:**
- Clean typography that aids reading
- Color that conveys meaning
- Whitespace that creates breathing room
- Animation that guides, not distracts

### Principle 22: Emotional Resonance

The interface should feel like it cares.

**Tone:**
- Warm but not saccharine
- Serious but not stern
- Encouraging but not pushy
- Honest but not harsh

**Micro-interactions:**
- Acknowledge effort
- Celebrate growth
- Comfort in failure
- Delight in discovery

### Principle 23: Temporal Rhythm

Different modes for different times.

**Activity modes:**
- Focus mode (minimal distraction)
- Exploration mode (discovery-oriented)
- Reflection mode (review and learning)
- Social mode (community interaction)

**Time-aware design:**
- Respect circadian rhythms (optional dark mode at night)
- Support natural work rhythms (pomodoro-compatible)
- Honor rest (no guilt for inactivity)
- Recognize seasons (annual reviews, fresh starts)

---

## Part VI: Ethical Design

### Principle 24: Do No Harm

The system must not be weaponizable.

**Safeguards:**
- Cannot be used for harassment
- Cannot be used for manipulation at scale
- Cannot be used for surveillance
- Cannot be used to harm vulnerable populations

**Design tests:**
- Could this be used to hurt someone?
- Could this be used to deceive at scale?
- Could this be used to concentrate power unhealthily?
- Would we be comfortable if our adversaries used this?

### Principle 25: Align Incentives

Individual benefit should align with collective benefit.

**Mechanism design:**
- Personal growth → community strength
- Helping others → personal reputation
- System health → individual success
- Long-term thinking → short-term rewards

**Warning signs:**
- Zero-sum dynamics
- Tragedy of the commons
- Race to the bottom
- Exploitation opportunities

### Principle 26: Question Everything (Including This)

These principles are hypotheses, not commandments.

**Meta-design:**
- Regular review of principles
- Mechanisms to challenge any principle
- Evidence-based evaluation
- Evolution over dogma

**The ultimate test:**
- Are these principles serving truth-seeking?
- Are they serving human flourishing?
- Should they change?

---

## Applying the Principles

### Decision Framework

When facing a design decision:

1. **Identify the tension** - Which principles are in conflict?
2. **Understand the context** - What matters most here?
3. **Consider stakeholders** - Who is affected and how?
4. **Explore alternatives** - What other options exist?
5. **Test against principles** - How does each option score?
6. **Decide and document** - Record the reasoning
7. **Monitor and learn** - Was this the right call?

### Example: Notification Design

**Question:** Should we send notifications when someone disagrees with your prediction?

**Principles in tension:**
- Independence (notifications might create pressure)
- Learning (disagreement is information)
- Dignity (might feel like attack)
- Engagement (keeps users active)

**Context analysis:**
- Users have different preferences
- Disagreement quality varies
- Timing matters

**Decision:**
- Default: No notifications for disagreement
- Option: "Notify me of high-quality disagreements"
- When enabled: Delay notifications, batch them
- Frame as: "Someone has a different perspective"
- Include: Easy mute options

**Documentation:**
- Decision recorded with reasoning
- Metrics defined for evaluation
- Review scheduled for 3 months

---

## Living Principles

These principles are themselves predictions about what will serve truth-seeking. They should be:

- **Tested** against experience
- **Updated** when evidence warrants
- **Challenged** by those who disagree
- **Evolved** as we learn

The best design principles are those that make themselves obsolete—replaced by something better.

---

*"Good design is obvious. Great design is transparent."*
*— Joe Sparano*

*Great epistemic design is invisible—it just feels like thinking clearly together.*
