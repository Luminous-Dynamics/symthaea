# Moderation

*Cultivating Truth Through Community Care*

---

> "The goal of moderation is not to punish, but to heal. Not to exclude, but to guide back to truth."

---

## Introduction

Moderation in Epistemic Markets is fundamentally different from traditional content moderation. We're not primarily concerned with offensive content or spam (though those exist). Our central concern is **epistemic hygiene**—maintaining an environment where truth can emerge.

This document describes our moderation philosophy, policies, tools, and processes.

---

## Part I: Moderation Philosophy

### Our Beliefs About Moderation

1. **Truth requires diversity**: We don't moderate disagreement; we protect the space for productive disagreement

2. **Intent matters**: A genuine mistake is different from deliberate manipulation

3. **Education over punishment**: First responses should teach, not punish

4. **Community ownership**: The community moderates itself; we provide tools

5. **Transparency**: Moderation actions should be visible and explainable

6. **Proportionality**: Responses should match the severity of the issue

7. **Rehabilitation**: There should always be a path back

### What We Moderate

| We Moderate | We Don't Moderate |
|-------------|-------------------|
| Manipulation attempts | Unpopular predictions |
| Coordinated misinformation | Minority viewpoints |
| Harassment | Passionate disagreement |
| Gaming the system | Creative strategies |
| Bad faith participation | Being wrong |
| Impersonation | Anonymity |
| Spam and noise | High activity |
| Plagiarism | Citing others |

### The Spectrum of Intervention

```
Observation → Nudge → Warning → Restriction → Suspension → Ban
    ↓           ↓         ↓           ↓             ↓         ↓
  Watch     Suggest    Notify     Limit        Pause     Remove
  pattern   better     clearly    temporarily  access    from
            behavior   problem                           community
```

Most issues should resolve at the "Nudge" stage. Bans should be rare.

---

## Part II: Types of Violations

### Category 1: Epistemic Manipulation

**Definition**: Deliberate attempts to distort collective truth-seeking.

**Examples**:
- Creating fake evidence
- Systematic misinformation campaigns
- Sybil attacks to skew predictions
- Paying others to predict falsely
- Insider information abuse

**Severity**: High
**Default Response**: Investigation + Warning → Suspension

**Detection Methods**:
- Pattern analysis on prediction clusters
- Network analysis for coordination
- Evidence verification
- Whistleblower reports
- AI anomaly detection

### Category 2: Market Manipulation

**Definition**: Gaming the market mechanisms for unfair advantage.

**Examples**:
- Wash trading (trading with yourself)
- Pump and dump on question markets
- Timing exploits
- Oracle manipulation
- Front-running based on insider knowledge

**Severity**: High
**Default Response**: Economic penalty + Investigation

**Detection Methods**:
- Trade pattern analysis
- Account linkage detection
- Timing anomaly detection
- Economic flow analysis

### Category 3: Social Harm

**Definition**: Behavior that damages community members or culture.

**Examples**:
- Harassment of predictors
- Doxxing
- Threatening behavior
- Discrimination
- Bullying after wrong predictions

**Severity**: High for individuals, Very High for patterns
**Default Response**: Immediate restriction + Investigation

**Detection Methods**:
- Report system
- Sentiment analysis
- Pattern detection
- Community flags

### Category 4: Quality Degradation

**Definition**: Actions that reduce the quality of collective intelligence.

**Examples**:
- Spamming low-quality predictions
- Copy-paste reasoning
- Irrelevant market creation
- Off-topic discussions
- Noise-to-signal degradation

**Severity**: Low to Medium
**Default Response**: Nudge → Warning

**Detection Methods**:
- Quality scoring algorithms
- Duplicate detection
- Relevance analysis
- Community downvotes

### Category 5: Bad Faith Participation

**Definition**: Participating in ways that undermine the system's purpose.

**Examples**:
- Never updating predictions regardless of evidence
- Refusing to accept clear resolutions
- Persistent trolling
- Deliberate confusion-sowing
- Arguing in bad faith

**Severity**: Medium
**Default Response**: Education → Warning → Restriction

**Detection Methods**:
- Calibration analysis (never updating)
- Dispute pattern analysis
- Community reports
- Behavioral consistency analysis

### Category 6: Technical Violations

**Definition**: Abusing technical systems or violating technical rules.

**Examples**:
- Creating multiple accounts
- Automated trading bots without disclosure
- API abuse
- Circumventing rate limits
- Exploiting bugs

**Severity**: Varies
**Default Response**: Depends on intent

**Detection Methods**:
- Technical monitoring
- API analysis
- Bug bounty reports
- Security audits

---

## Part III: Moderation Roles

### 1. Community Members (Everyone)

**Powers**:
- Flag content for review
- Downvote low-quality contributions
- Report violations
- Participate in community votes
- Provide context on flags

**Responsibilities**:
- Flag in good faith
- Don't abuse the flag system
- Engage constructively
- Help newcomers understand norms

### 2. Wisdom Keepers

**Requirements**:
- MATL composite > 70
- 1+ years of participation
- Track record of constructive engagement
- Community nomination + vote

**Powers**:
- Fast-track flags to review
- Issue community nudges
- Provide authoritative context
- Mentor flagged participants
- Vote on moderation decisions

**Responsibilities**:
- Prioritize education over punishment
- Remain impartial in disputes
- Recuse when personally involved
- Maintain confidentiality appropriately

### 3. Domain Stewards

**Requirements**:
- Domain expertise (MATL domain score > 80)
- Technical understanding of that domain
- Track record in domain markets

**Powers**:
- Review domain-specific flags
- Assess evidence quality
- Advise on resolution disputes
- Flag domain-specific manipulation

**Responsibilities**:
- Apply domain expertise fairly
- Distinguish honest mistakes from manipulation
- Stay current with domain developments

### 4. Trust Council

**Requirements**:
- Elected by community
- Demonstrated commitment to truth-seeking
- Diverse representation
- Term-limited (1 year)

**Powers**:
- Final decisions on appeals
- Set moderation policy
- Handle complex cases
- Issue suspensions
- Approve bans

**Responsibilities**:
- Act as final arbiter
- Ensure consistency
- Maintain appeals process
- Regular transparency reports

### 5. Emergency Responders

**Requirements**:
- Rapid availability
- Security training
- Trust Council approval

**Powers**:
- Immediate account suspension
- Market halt
- Evidence preservation
- Coordinate with authorities if needed

**Responsibilities**:
- Use powers only in genuine emergencies
- Document everything
- Report to Trust Council within 24 hours

---

## Part IV: Moderation Process

### Stage 1: Detection

Violations are detected through multiple channels:

```typescript
interface DetectionSource {
  // Automatic detection
  automatic: {
    patternAnalysis: boolean;
    anomalyDetection: boolean;
    qualityScoring: boolean;
    networkAnalysis: boolean;
  };

  // Community detection
  community: {
    flags: Flag[];
    downvotes: Downvote[];
    reports: Report[];
    whispers: ConfidentialReport[];
  };

  // Expert review
  expert: {
    wisdomKeeperFlags: Flag[];
    domainStewardFlags: Flag[];
    securityAudit: Finding[];
  };
}
```

### Stage 2: Triage

Every flag is triaged within 24 hours:

```
Priority Assignment:
├── Critical (< 1 hour)
│   ├── Active harm to individuals
│   ├── Ongoing manipulation with large impact
│   └── Security breach
├── High (< 24 hours)
│   ├── Market manipulation
│   ├── Epistemic manipulation
│   └── Harassment
├── Medium (< 72 hours)
│   ├── Quality degradation
│   ├── Bad faith participation
│   └── Technical violations
└── Low (< 1 week)
    ├── Minor quality issues
    ├── Community disputes
    └── Feature abuse
```

### Stage 3: Investigation

For non-trivial cases:

```typescript
interface Investigation {
  caseId: string;
  assignedTo: AgentPubKey[];
  priority: Priority;

  evidence: {
    automaticEvidence: Evidence[];  // System logs, patterns
    reportedEvidence: Evidence[];   // From flags
    gatheredEvidence: Evidence[];   // Investigation findings
  };

  timeline: Event[];  // Full chronology

  participantStatements: Statement[];  // If relevant

  analysis: {
    factualFindings: string;
    patternAssessment: string;
    intentAssessment: IntentLevel;
    impactAssessment: ImpactLevel;
  };

  recommendation: ActionRecommendation;

  status: "Open" | "InProgress" | "Review" | "Closed";
}
```

### Stage 4: Decision

Decision-making framework:

```
Input: Investigation findings + Policy guidance + Precedent

Decision Process:
1. Is this a clear policy violation?
   - Yes → Apply standard response
   - No → Escalate for judgment

2. What was the intent?
   - Malicious → Stronger response
   - Negligent → Educational response
   - Innocent → Minimal or no response

3. What was the impact?
   - High → More serious response
   - Low → More lenient response

4. Is there a pattern?
   - First offense → Lighter response
   - Repeat offense → Escalated response

5. Are there mitigating factors?
   - Yes → Consider in response
   - No → Standard response

Output: Proportional response action
```

### Stage 5: Action

Available actions, from lightest to heaviest:

**1. No Action**
- When investigation clears the participant
- Communicate findings to reporter

**2. Nudge**
- Private, friendly guidance
- No permanent record
- For minor first-time issues

**3. Warning**
- Official notice of violation
- Recorded but not public
- Clear explanation of issue
- Guidance for improvement

**4. Content Action**
- Remove specific content
- Correct misinformation
- Add context or warning labels

**5. Restriction**
- Temporary limits on specific activities
- Can still participate in most ways
- Duration: typically 1-30 days

**6. Suspension**
- All participation paused
- Account in read-only mode
- Duration: 7-90 days
- Requires Trust Council approval

**7. Ban**
- Permanent removal from community
- Requires Trust Council supermajority
- Appeals process available

**8. Criminal Referral**
- For serious violations of law
- Coordinated with legal authorities
- Rare and serious

### Stage 6: Communication

All moderation actions must include:

```typescript
interface ModerationCommunication {
  // To the participant
  notification: {
    what: string;          // What action was taken
    why: string;           // Clear explanation of violation
    evidence: string;      // What evidence supported this
    impact: string;        // What this means for them
    duration?: string;     // How long (if temporary)
    nextSteps: string;     // What they can/should do
    appeal: string;        // How to appeal
  };

  // To the community (if public)
  announcement?: {
    anonymized: boolean;   // Most are anonymized
    summary: string;       // What happened
    lesson: string;        // What we all learned
    policyLink?: string;   // Relevant policy
  };

  // To reporter
  reporterUpdate: {
    acknowledged: boolean;
    outcome: "ActionTaken" | "NoViolation" | "Ongoing";
    thankYou: boolean;
  };
}
```

### Stage 7: Follow-Up

After action:

- **Monitor**: Watch for recurrence or escalation
- **Support**: Offer resources for rehabilitation
- **Learn**: Update patterns and policies
- **Review**: Periodic review of ongoing restrictions

---

## Part V: Specific Policies

### Policy 1: Sybil Accounts

**Definition**: Multiple accounts controlled by one person to gain unfair advantage.

**Rules**:
1. One primary account per person
2. Secondary accounts permitted only with disclosure
3. Voting/staking must be from primary account only

**Detection**:
- Behavioral fingerprinting
- Transaction pattern analysis
- Device/IP correlation (privacy-preserving)
- Community reports

**Response**:
- First offense: Merge accounts + warning
- With manipulation: Suspension + economic penalty
- Organized attack: Ban + potential legal action

### Policy 2: Coordinated Manipulation

**Definition**: Multiple actors working together to distort truth-seeking.

**Rules**:
1. Coordination to share information is permitted
2. Coordination to manipulate outcomes is prohibited
3. Prediction groups must disclose coordination

**Detection**:
- Network analysis
- Timing correlation
- Prediction similarity analysis
- Whistleblower reports

**Response**:
- Investigation of all participants
- Graduated response based on role
- Leaders face stronger action
- Followers may receive reduced penalty for cooperation

### Policy 3: Harassment

**Definition**: Behavior targeting individuals to intimidate, demean, or harm.

**Zero Tolerance For**:
- Threats of violence
- Doxxing
- Sexual harassment
- Discrimination
- Persistent unwanted contact

**Response**:
- Immediate restriction pending investigation
- Content removal
- Escalated response for severity
- Support for affected party

### Policy 4: Market Creation

**Definition**: Standards for what markets can be created.

**Prohibited Markets**:
- Markets on private individuals' personal lives
- Markets incentivizing harm
- Markets violating law
- Duplicate markets without clear purpose

**Response**:
- Market closure
- Creator warning
- Repeat violations lead to market creation restriction

### Policy 5: Oracle Behavior

**Definition**: Standards for resolution oracles.

**Prohibited**:
- Voting against clear evidence
- Accepting bribes
- Coordinating votes
- Resolving markets with conflicts of interest

**Response**:
- Oracle status revocation
- Economic penalties
- Potential ban for serious violations

### Policy 6: Wisdom Theft

**Definition**: Claiming credit for others' reasoning or evidence.

**Rules**:
- Must cite sources
- Must credit inspiration
- Must not misrepresent authorship

**Response**:
- Attribution correction
- Warning
- Repeat violations affect reputation

---

## Part VI: Appeals Process

### Who Can Appeal

Any participant who has received a moderation action can appeal.

### Grounds for Appeal

Valid grounds:
1. **Factual error**: The facts were wrong
2. **Policy misapplication**: The policy was applied incorrectly
3. **New evidence**: Evidence not considered in original decision
4. **Procedural error**: Process wasn't followed correctly
5. **Disproportionate**: Action doesn't match violation

Invalid grounds:
1. "I disagree with the policy"
2. "Others do it too"
3. "It's not fair" (without specifics)

### Appeal Process

```
Step 1: Submit Appeal (within 14 days)
   ├── Clear statement of grounds
   ├── Supporting evidence
   └── Requested outcome

Step 2: Review (within 7 days)
   ├── Different reviewers than original
   ├── Full case review
   └── Additional investigation if needed

Step 3: Decision (within 7 days)
   ├── Uphold: Original decision stands
   ├── Modify: Adjust the action
   ├── Overturn: Remove the action
   └── Escalate: Send to Trust Council

Step 4: Trust Council (if escalated, within 14 days)
   ├── Full panel review
   ├── May request hearing
   └── Final decision (binding)
```

### Appeal Rights

- One appeal per moderation action
- Continued appeal to Trust Council if initial appeal denied
- Right to present evidence
- Right to know grounds for decision
- No retaliation for appealing

---

## Part VII: Moderation Tools

### Automated Tools

```typescript
// Pattern detection
interface PatternDetector {
  detectSybilClusters(): SybilCluster[];
  detectCoordinatedBehavior(): CoordinationPattern[];
  detectAnomalies(): Anomaly[];
  assessQuality(content: Content): QualityScore;
}

// Risk scoring
interface RiskScorer {
  accountRiskScore(account: AgentPubKey): RiskScore;
  contentRiskScore(content: Content): RiskScore;
  marketRiskScore(market: Market): RiskScore;
  predictionRiskScore(prediction: Prediction): RiskScore;
}

// Auto-moderation
interface AutoModerator {
  // Actions that can be automated
  quarantineForReview(content: Content): void;
  rateLimitAccount(account: AgentPubKey, limits: RateLimits): void;
  flagForHumanReview(item: FlaggableItem, reason: string): void;

  // NOT automated: suspensions, bans, permanent actions
}
```

### Human Tools

```typescript
// Moderator dashboard
interface ModeratorDashboard {
  // Queue management
  queue: {
    view(): ModerationItem[];
    claim(itemId: string): void;
    prioritize(itemId: string, priority: Priority): void;
  };

  // Investigation
  investigate: {
    viewEvidence(caseId: string): Evidence[];
    addNote(caseId: string, note: string): void;
    linkCases(caseIds: string[]): void;
    timeline(account: AgentPubKey): TimelineEvent[];
  };

  // Actions
  actions: {
    nudge(account: AgentPubKey, message: string): void;
    warn(account: AgentPubKey, warning: Warning): void;
    restrict(account: AgentPubKey, restriction: Restriction): void;
    proposeAction(action: ProposedAction): void;
  };

  // Communication
  communicate: {
    contactParticipant(account: AgentPubKey, message: string): void;
    updateReporter(flagId: string, update: string): void;
    postAnnouncement(announcement: Announcement): void;
  };
}
```

### Transparency Tools

```typescript
// Public moderation data
interface TransparencyDashboard {
  // Aggregate statistics
  stats: {
    actionsLastMonth(): ModerationStats;
    categoriesBreakdown(): CategoryStats;
    appealOutcomes(): AppealStats;
  };

  // Policy enforcement
  enforcement: {
    recentActions(): AnonymizedAction[];
    policyChanges(): PolicyChange[];
    precedents(): Precedent[];
  };

  // Reports
  reports: {
    monthlyReport(): TransparencyReport;
    quarterlyReport(): TransparencyReport;
    annualReport(): TransparencyReport;
  };
}
```

---

## Part VIII: Prevention and Education

### Proactive Prevention

**1. Clear Norms Communication**
- Onboarding includes norm education
- Regular reminders of community standards
- Positive examples highlighted

**2. Friction for Risk**
- Higher friction for high-risk actions
- Confirmation for potentially problematic content
- Cooling-off periods for conflicts

**3. Early Intervention**
- Reach out before violations
- Offer guidance when patterns emerging
- Connect with mentors

**4. Community Building**
- Strong community reduces violations
- Sense of belonging increases compliance
- Social bonds create accountability

### Educational Resources

**For All Participants**:
- "What We Expect" guide
- "How to Disagree Well" tutorial
- "Epistemic Hygiene 101" course
- "Being Wrong Gracefully" guide

**For Flaggers**:
- "How to Flag Effectively" guide
- "What Is vs. Isn't a Violation"
- "Flagging Best Practices"

**For Those Warned**:
- Custom resources based on violation type
- Mentor pairing option
- Rehabilitation pathway

---

## Part IX: Transparency and Accountability

### Regular Reporting

**Monthly Report**:
- Number of flags received
- Actions taken by category
- Appeal outcomes
- Notable cases (anonymized)

**Quarterly Report**:
- Trend analysis
- Policy effectiveness
- Tool performance
- Areas for improvement

**Annual Report**:
- Full statistics
- Policy changes made
- Lessons learned
- Plans for next year

### Moderator Accountability

**Performance Metrics**:
- Decision quality (appeal overturn rate)
- Timeliness
- Communication quality
- Community feedback

**Oversight**:
- Trust Council reviews moderator actions
- Community can raise concerns
- Regular moderator evaluations

**Term Limits**:
- No perpetual moderator positions
- Rotation to prevent entrenchment
- Fresh perspectives regularly

### Community Oversight

**Transparency Mechanisms**:
- All policies public
- Moderation statistics public
- Major decisions explained
- Regular AMAs with Trust Council

**Feedback Channels**:
- Anonymous feedback form
- Monthly community meetings
- Direct contact with Trust Council
- Ombudsperson available

---

## Part X: Special Situations

### During Resolutions

**Heightened Monitoring**:
- Resolution periods see increased moderation
- Manipulation attempts more common
- Faster response times

**Special Rules**:
- No new evidence claims without verification
- Cooling-off before dispute escalation
- Oracle protection from harassment

### During Governance

**Protected Periods**:
- Governance votes have additional protections
- Manipulation detection enhanced
- Lower threshold for intervention

**Special Considerations**:
- All viewpoints must be heard
- No suppression of legitimate dissent
- Results must be verifiable

### Emergency Situations

**Circuit Breaker Triggers**:
- Mass manipulation detected
- External attack
- Critical bug exploitation

**Emergency Powers**:
- Temporary market halts
- Mass account restrictions
- Enhanced verification requirements

**Post-Emergency**:
- Full incident review
- Community communication
- Policy updates if needed

---

## Part XI: Evolution

### Policy Updates

**Process**:
1. Issue identified (through reports, data, community)
2. Policy proposal drafted
3. Community comment period (minimum 7 days)
4. Trust Council vote
5. Implementation
6. Review effectiveness

**Criteria**:
- Does it serve truth-seeking?
- Is it proportional?
- Is it enforceable?
- Does it protect the vulnerable?
- Is it clear?

### Tool Improvements

**Continuous Development**:
- Detection algorithms refined with feedback
- False positive reduction
- New manipulation pattern detection
- Efficiency improvements

### Community Evolution

As the community grows:
- More distributed moderation
- Stronger community norms
- Less need for explicit enforcement
- Self-policing increases

---

## Conclusion

Moderation in Epistemic Markets exists to protect the conditions for truth-seeking. Our goal is never punishment for its own sake, but healing—of the community, the culture, and when possible, the participant who strayed.

The best moderation is invisible: norms so clear, tools so effective, community so strong, that violations are rare and handled swiftly. We aspire to a community where moderation withers away because it's no longer needed—where truth-seeking is so deeply embedded that gaming the system becomes unthinkable.

Until then, we moderate with wisdom, compassion, and an unwavering commitment to truth.

---

> "The measure of our moderation is not how many we exclude,
> but how many we guide back to the path of honest inquiry."

---

*Moderation serves truth. Truth serves all.*

---

## Appendix: Quick Reference

### When to Flag

| Situation | Flag? | Category |
|-----------|-------|----------|
| Someone makes a prediction I think is wrong | No | |
| Someone is using fake evidence | Yes | Epistemic Manipulation |
| Someone is mean in comments | Maybe | Context matters |
| Someone is threatening another user | Yes | Harassment (urgent) |
| Someone is making many low-quality predictions | Maybe | Quality Degradation |
| Someone has multiple accounts | Maybe | If causing manipulation |
| Someone won't accept a resolution | Maybe | Bad Faith Participation |

### Response Expectations

| Action | Response Time | Appeal Window |
|--------|---------------|---------------|
| Acknowledgment | < 24 hours | N/A |
| Critical resolution | < 1 hour | Immediate |
| Standard resolution | < 72 hours | 14 days |
| Appeal decision | < 7 days | N/A |
| Trust Council appeal | < 14 days | None (final) |

### Who to Contact

| Situation | Contact |
|-----------|---------|
| General flag | Use flag button |
| Urgent safety issue | Emergency response team |
| Appeal | Appeals process |
| Policy question | Wisdom Keeper |
| Systemic concern | Trust Council |
| Moderation feedback | Feedback form |
