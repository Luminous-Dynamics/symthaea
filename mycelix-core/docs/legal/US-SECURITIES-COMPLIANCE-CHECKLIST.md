# US Securities Law Compliance Checklist

**Mycelix Protocol - Utility Token Analysis**
**Document Version**: 1.0
**Last Updated**: 2026-01-18

---

> **LEGAL NOTICE**: This checklist is for planning and internal assessment purposes only. It does not constitute legal advice. Consult qualified US securities counsel before any token distribution or marketing activities affecting US persons.

---

## Executive Summary

This document provides a compliance checklist for ensuring that [TOKEN NAME] is structured and marketed as a utility token that is not a security under US law. The analysis is based on the SEC's framework for analyzing digital assets, particularly the Howey test and SEC guidance.

---

## 1. Howey Test Analysis

### 1.1 Overview

A token is a security under US law if it meets ALL four prongs of the Howey test:

1. **Investment of Money** - Purchaser provides capital
2. **Common Enterprise** - Purchaser's fortunes tied to promoter/others
3. **Expectation of Profits** - Purchaser expects financial returns
4. **Derived from Efforts of Others** - Returns depend on promoter's efforts

**Strategy**: Structure token to fail at least one prong (preferably #3 and #4).

### 1.2 Prong-by-Prong Analysis

| Prong | Analysis | Risk Level | Mitigation |
|-------|----------|------------|------------|
| Investment of Money | Tokens acquired in exchange for value | HIGH | Cannot eliminate; focus on other prongs |
| Common Enterprise | Network participants share outcomes | MEDIUM | Decentralize, reduce issuer dependency |
| Expectation of Profits | **KEY FOCUS** | MEDIUM | Emphasize utility, prohibit investment language |
| Efforts of Others | **KEY FOCUS** | MEDIUM | Decentralize, network functionality from day 1 |

---

## 2. Marketing and Communications Checklist

### 2.1 Prohibited Language

**NEVER use the following language in any materials:**

| Prohibited | Reason | Alternative |
|------------|--------|-------------|
| "Investment" | Implies securities | "Participation" |
| "Profit" | Implies financial return | "Rewards for contribution" |
| "Return" (financial) | Implies investment | "Utility benefits" |
| "Appreciation" | Implies speculative gain | "Network growth" |
| "Hold and wait" | Implies passive investment | "Use and participate" |
| "Buy low, sell high" | Speculative trading | Never mention |
| "Token price" | Price speculation | "Token utility" |
| "Get in early" | FOMO investing | "Participate in launch" |
| "Limited supply" (as investment) | Scarcity investing | "Network design" |

### 2.2 Required Disclosures

**ALWAYS include these disclosures:**

| Disclosure | Location | Text |
|------------|----------|------|
| Not an Investment | All materials | "[TOKEN NAME] is a utility token and is not an investment product. No expectation of profit should be derived from holding [TOKEN NAME]." |
| No Guarantee | Token sale | "There is no guarantee of any value, utility, or functionality." |
| Use Only | Website | "[TOKEN NAME] is designed for use within the Mycelix network. Acquire only the amount you intend to use." |
| Risk | All materials | "Digital assets carry risk of total loss." |

### 2.3 Marketing Review Checklist

Before publishing ANY material:

- [ ] No investment language (see 2.1)
- [ ] No price predictions or projections
- [ ] No comparisons to securities or traditional investments
- [ ] No emphasis on secondary market trading
- [ ] No promises of returns or profits
- [ ] Utility use cases prominently featured
- [ ] Required disclosures included (see 2.2)
- [ ] Reviewed by compliance team
- [ ] Legal sign-off obtained

---

## 3. Token Distribution Checklist

### 3.1 Pre-Network Distribution

**HIGH RISK**: Distributing tokens before a functional network exists increases securities risk.

| Factor | Risk | Mitigation |
|--------|------|------------|
| Network not live | HIGH | Launch network before/concurrent with token |
| No immediate utility | HIGH | Ensure day-1 utility |
| Purchaser reliance on issuer | HIGH | Decentralize development |

**Recommendation**:
- [ ] Network is functional before token distribution
- [ ] Token has immediate utility at distribution
- [ ] Clear documentation of utility features

### 3.2 Distribution Tied to Participation

| Distribution Method | Risk Level | Notes |
|--------------------|------------|-------|
| Rewards for network participation | LOW | Preferred - earned, not purchased |
| Airdrops to users | MEDIUM | Must have utility basis |
| Public sale | HIGH | Avoid in US or restrict |
| Private placement | HIGH | Accredited only if proceeds |

**Recommended Approach**:
- [ ] Primary distribution through network rewards (validators, contributors)
- [ ] No public token sale to US persons
- [ ] Airdrops tied to network activity, not investment
- [ ] Foundation allocation with long vesting

### 3.3 Geographic Restrictions

| Jurisdiction | Action |
|--------------|--------|
| United States | [ ] Exclude from public sale |
| | [ ] Geo-block US IPs from sale interface |
| | [ ] Require attestation of non-US person status |
| | [ ] No marketing targeted at US persons |

---

## 4. Token Economics Checklist

### 4.1 Utility-First Design

- [ ] Token is REQUIRED for network functionality (not optional)
- [ ] Clear and documented use cases:
  - [ ] Compute credits (pay for aggregation)
  - [ ] Staking (validator collateral)
  - [ ] Governance (protocol voting)
  - [ ] Fee payment
- [ ] Token consumption mechanism (spent, not just held)
- [ ] No dividend or profit-sharing features
- [ ] No buyback commitments

### 4.2 Anti-Securities Features

| Feature | Status | Notes |
|---------|--------|-------|
| No promise of dividends | [ ] | Never implemented |
| No revenue sharing | [ ] | Network fees go to validators, not token holders passively |
| No buyback commitments | [ ] | Foundation may burn tokens, but no price support |
| No guaranteed liquidity | [ ] | No market making commitments |
| Decentralized governance | [ ] | Token holders control, not Foundation |

### 4.3 Vesting and Lockups

| Allocation | Vesting | Rationale |
|------------|---------|-----------|
| Team | 4-year cliff/linear | Prevents immediate sale |
| Foundation | Governance-controlled | Long-term alignment |
| Investors (if any) | 2-year lockup | Prevent speculation |
| Network rewards | Continuous | Earned through participation |

---

## 5. Decentralization Checklist

### 5.1 Technical Decentralization

- [ ] Open-source code (GPL/MIT/Apache)
- [ ] Permissionless node operation
- [ ] No single point of failure
- [ ] Multiple independent validators
- [ ] Decentralized governance mechanism
- [ ] Community-controlled upgrades

### 5.2 Operational Decentralization

| Factor | Current State | Target |
|--------|---------------|--------|
| Validator count | [X] | 21+ independent |
| Foundation control | [X]% | <20% |
| Upgrade authority | Foundation | Token holders |
| Treasury control | Foundation | Multi-sig + governance |

### 5.3 Information Decentralization

- [ ] Public documentation
- [ ] Open governance forums
- [ ] Transparent decision-making
- [ ] Regular community updates
- [ ] No asymmetric information advantage

---

## 6. Terms of Service Requirements

### 6.1 Required Clauses

The Terms of Service MUST include:

```
1. NO INVESTMENT PRODUCT

[TOKEN NAME] is a utility token for use within the Mycelix network.
[TOKEN NAME] is NOT:
- A security or investment product
- A share, equity, or ownership interest
- A debt instrument or loan
- A derivative or futures contract

You should NOT acquire [TOKEN NAME] with any expectation of profit.
Acquire only the amount you intend to use for network services.

2. NO REPRESENTATIONS REGARDING VALUE

We make NO representations regarding:
- The future value or price of [TOKEN NAME]
- The liquidity of [TOKEN NAME]
- The availability of secondary markets
- Any return on your acquisition of [TOKEN NAME]

3. ACKNOWLEDGMENT

By using this service, you acknowledge that you:
- Understand [TOKEN NAME] is a utility token, not an investment
- Have no expectation of profit from [TOKEN NAME]
- Are acquiring [TOKEN NAME] solely for use within the network
- Accept full responsibility for your decision to acquire [TOKEN NAME]

4. US PERSONS

If you are a US person (citizen, resident, or located in the US), you
acknowledge additional restrictions may apply and agree to comply with
all applicable US laws.
```

### 6.2 Checklist

- [ ] "Not an investment" language included
- [ ] "No expectation of profit" language included
- [ ] "Utility purpose" clearly stated
- [ ] User acknowledgment required
- [ ] US person restrictions stated
- [ ] Risk disclosures included
- [ ] Legal reviewed and approved

---

## 7. Secondary Market Considerations

### 7.1 Exchange Listings

| Action | Risk | Guidance |
|--------|------|----------|
| Actively pursuing listings | HIGH | Implies focus on trading |
| Announcing listings | MEDIUM | Keep factual, not promotional |
| Celebrating listings | HIGH | Implies investment value |
| Providing liquidity | HIGH | Implies price support |

**Policy**:
- [ ] Do not actively pursue exchange listings as a project goal
- [ ] Do not celebrate or promote listings
- [ ] Do not provide liquidity or market making
- [ ] Keep any announcements factual and brief

### 7.2 Price Discussion Policy

| Permitted | Prohibited |
|-----------|------------|
| Factual statements about token economics | Price predictions |
| Network utility statistics | Price targets |
| Technical documentation | "Undervalued" claims |
| | "Moon" or "pump" language |

---

## 8. Ongoing Compliance

### 8.1 Regular Reviews

| Review | Frequency | Responsible |
|--------|-----------|-------------|
| Marketing materials | Before publication | Compliance + Legal |
| Website content | Monthly | Marketing + Legal |
| Social media | Weekly | Community + Compliance |
| Terms of Service | Quarterly | Legal |
| Token economics | Annual | Foundation Board |

### 8.2 Training Requirements

- [ ] All team members complete securities law basics training
- [ ] Marketing team completes communications compliance training
- [ ] Community managers trained on prohibited language
- [ ] Annual refresher for all staff

### 8.3 Documentation

Maintain records of:
- [ ] All marketing materials (with approval dates)
- [ ] Legal opinions obtained
- [ ] Compliance reviews conducted
- [ ] Training completion records
- [ ] Distribution records (who received tokens, why)

---

## 9. Red Flags Checklist

If ANY of these are true, STOP and consult legal counsel:

- [ ] Token is distributed primarily through sale (vs. earning)
- [ ] Marketing emphasizes potential price appreciation
- [ ] Purchasers have no immediate utility for tokens
- [ ] Network is not yet functional
- [ ] Foundation retains significant control post-launch
- [ ] Team allocation is not subject to vesting
- [ ] Revenue or profits are distributed to token holders
- [ ] Buyback or price support mechanisms exist
- [ ] US persons are targeted in marketing
- [ ] Secondary market trading is emphasized

---

## 10. Documentation Checklist

### 10.1 Required Documents

| Document | Status | Last Updated |
|----------|--------|--------------|
| Legal opinion (US counsel) | [ ] | |
| Terms of Service | [ ] | |
| Privacy Policy | [ ] | |
| Token documentation | [ ] | |
| Marketing guidelines | [ ] | |
| Compliance training materials | [ ] | |
| Distribution records | [ ] | |

### 10.2 Legal Opinion Scope

Request US securities counsel opinion on:
- [ ] Howey test analysis for [TOKEN NAME]
- [ ] Marketing guidelines review
- [ ] Distribution structure review
- [ ] Geographic restriction adequacy
- [ ] Terms of Service review
- [ ] Ongoing compliance recommendations

---

## Appendix A: SEC Guidance References

| Document | Date | Relevance |
|----------|------|-----------|
| SEC Framework for Digital Assets | April 2019 | Primary Howey analysis |
| FinHub Statement on Token Distribution | 2019 | Distribution guidance |
| Hinman Speech (ETH) | June 2018 | Decentralization factors |
| SEC v. Telegram | 2020 | Pre-network distribution risk |
| SEC v. Ripple | Ongoing | Ongoing sale issues |

---

## Appendix B: Template Attestation

For any token distribution, consider requiring:

```
ATTESTATION

I hereby attest that:

1. I am NOT a US person (as defined under Regulation S)
2. I am NOT acquiring tokens on behalf of a US person
3. I understand that [TOKEN NAME] is a utility token and not an investment
4. I have no expectation of profit from acquiring [TOKEN NAME]
5. I am acquiring [TOKEN NAME] solely for use within the Mycelix network
6. I will not resell or transfer tokens to US persons

Signature: _____________
Date: _____________
```

---

*This checklist is for planning purposes only. Consult qualified US securities counsel.*
