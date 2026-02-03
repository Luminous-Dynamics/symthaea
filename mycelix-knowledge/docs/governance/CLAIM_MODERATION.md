# Claim Moderation

Guidelines for community-driven claim quality and moderation.

## Overview

Knowledge is a decentralized knowledge graph without central authority. Quality is maintained through:

1. **Epistemic Classification** - Claims self-describe verifiability
2. **Credibility Scores** - MATL-weighted reputation
3. **Verification Markets** - Economic incentives for truth
4. **Community Flagging** - Peer review and reporting
5. **Automated Detection** - Pattern recognition for abuse

## Moderation Principles

### 1. No Central Censor

There is no central authority that can delete claims. Instead:
- Claims can be flagged and disputed
- Credibility scores reflect community assessment
- Low-credibility claims rank lower in searches
- Authors can retract their own claims

### 2. Transparency Over Removal

Rather than hiding disputed content:
- Contradicting evidence is linked
- Dispute status is visible
- Users see the full picture
- History is preserved

### 3. Reputation Consequences

Bad actors face natural consequences:
- MATL trust decreases
- Author credibility drops
- Future claims start with lower scores
- Domain expertise can be revoked

## Claim Quality Tiers

### Tier 1: Verified (Green)
- E ≥ 0.8 after verification market
- Multiple corroborating claims
- Author credibility > 0.7
- No unresolved disputes

### Tier 2: Standard (Blue)
- E ≥ 0.5
- Some supporting evidence
- Author in good standing
- Default tier for new claims

### Tier 3: Disputed (Yellow)
- Active contradictions
- Ongoing verification market
- Mixed community assessment
- Requires caution

### Tier 4: Low Credibility (Red)
- E < 0.3 after review
- Failed verification market
- Multiple flags
- Author credibility < 0.3

## Flagging System

### Flag Types

| Flag | Description | Action |
|------|-------------|--------|
| `factual_error` | Claim contains factual inaccuracies | Triggers review |
| `misleading` | True but deceptively framed | Adds context |
| `outdated` | Information no longer current | Requests update |
| `duplicate` | Substantially duplicates another claim | Links to original |
| `spam` | Promotional or off-topic | Reduces visibility |
| `abuse` | Harassment or harmful content | Escalated review |

### Flagging Process

1. **User submits flag** with reason and evidence
2. **Flag counter increments** on claim
3. **At threshold (5 flags)**, claim enters review
4. **Reviewers assess** (random MATL-weighted selection)
5. **Outcome determined** by reviewer consensus
6. **Action applied** (status change, warning, etc.)

### Flag Requirements

To submit a flag:
- Account age > 7 days
- MATL trust > 0.3
- Not previously flagged this claim
- Must provide reason

## Review Process

### Reviewer Selection

Reviewers are selected based on:
- **MATL² weighting** - Higher trust = higher chance
- **Domain expertise** - Relevant domain experience
- **No conflicts** - Not involved with claim or author
- **Diversity** - Geographic and temporal spread

### Review Actions

| Action | Effect | When Used |
|--------|--------|-----------|
| `dismiss` | Flag rejected, no change | Flag was incorrect |
| `add_context` | Note added to claim | Clarification needed |
| `request_update` | Author asked to update | Outdated information |
| `mark_disputed` | Status changed | Legitimate dispute |
| `reduce_visibility` | Lower search ranking | Quality concerns |
| `author_warning` | Warning issued | Pattern of issues |

### Appeal Process

Authors can appeal moderation decisions:
1. Submit appeal within 30 days
2. Provide new evidence or argument
3. Different reviewer panel assesses
4. Final decision rendered
5. No further appeal (but can create new claim)

## Automated Detection

### Pattern Detection

The system automatically detects:
- **Spam patterns** - Repetitive, promotional content
- **Sybil attacks** - Coordinated fake accounts
- **Citation rings** - Mutual citation networks
- **Copy-paste** - Duplicated content
- **Rapid-fire posting** - Unusual posting velocity

### Automated Actions

| Detection | Automatic Response |
|-----------|-------------------|
| Spam pattern | Queue for review |
| Sybil attack | Freeze affected accounts |
| Citation ring | Flag network, reduce scores |
| Duplicate | Link to original |
| Rapid posting | Rate limit author |

## Author Responsibilities

### Good Standing Requirements

To maintain good standing:
- Respond to update requests within 14 days
- Retract claims proven false
- Disclose conflicts of interest
- Provide evidence when challenged
- Maintain MATL trust > 0.4

### Consequences for Violations

| Violation | First | Second | Third |
|-----------|-------|--------|-------|
| Unresponsive | Warning | Temp restriction | Account freeze |
| False claims | Score penalty | Domain ban | Global restriction |
| Harassment | Warning | 30-day ban | Permanent ban |
| Sybil activity | Account deletion | IP range ban | - |

## Moderation Governance

### Moderation Council

A rotating council oversees moderation policies:
- 7 members, 6-month terms
- Elected by MATL-weighted vote
- Responsible for policy updates
- Handles escalated cases

### Policy Changes

Moderation policy changes require:
1. Community proposal
2. 14-day discussion period
3. MATL-weighted vote
4. >60% approval
5. Implementation by council

### Transparency Reports

Monthly reports include:
- Flags submitted and outcomes
- Appeals and resolutions
- Automated detections
- Policy enforcement actions
- System health metrics

## Best Practices for Authors

### Creating Quality Claims

1. **Be specific** - Precise claims are easier to verify
2. **Cite sources** - Link supporting evidence
3. **Classify accurately** - Honest E-N-M positioning
4. **Acknowledge uncertainty** - Don't overclaim
5. **Update promptly** - Keep information current

### Responding to Flags

1. **Don't ignore** - Unresponded flags escalate
2. **Be constructive** - Address concerns directly
3. **Provide evidence** - Support your claim
4. **Update if needed** - Correct genuine errors
5. **Appeal respectfully** - If flag was wrong

## Reporting Abuse

For serious violations (harassment, threats, illegal content):

1. Use the `abuse` flag type
2. Include specific details and evidence
3. Report will be escalated immediately
4. Response within 24 hours
5. Serious cases referred to authorities

## Technical Implementation

### Flag Entry

```rust
pub struct Flag {
    pub claim_id: String,
    pub flag_type: FlagType,
    pub reason: String,
    pub evidence: Option<String>,
    pub flagger: AgentPubKey,
    pub created_at: u64,
    pub status: FlagStatus,
}
```

### Review Entry

```rust
pub struct Review {
    pub claim_id: String,
    pub reviewer: AgentPubKey,
    pub action: ReviewAction,
    pub reasoning: String,
    pub reviewed_at: u64,
}
```

### SDK Usage

```typescript
// Flag a claim
await knowledge.moderation.flagClaim({
  claimId: "uhCkk...",
  flagType: "factual_error",
  reason: "The cited study was retracted in 2023",
  evidence: "https://retractionwatch.com/..."
});

// Check claim moderation status
const status = await knowledge.moderation.getClaimStatus(claimId);
console.log(`Flags: ${status.flagCount}, Status: ${status.moderationStatus}`);
```

---

## Related Documentation

- [Epistemic Charter](./EPISTEMIC_CHARTER.md) - Foundational principles
- [Credibility Engine](../concepts/CREDIBILITY_ENGINE.md) - Scoring system
- [Security](../operations/SECURITY.md) - Security measures

---

*Truth emerges through collective vigilance.*
