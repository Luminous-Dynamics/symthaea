# MiCA Crypto-Asset White Paper Template

**Markets in Crypto-Assets Regulation (EU) 2023/1114**
**Article 6 Compliance Template**
**Version**: 1.0
**Last Updated**: 2026-01-18

---

> **LEGAL NOTICE**: This template is provided for informational purposes only and does not constitute legal advice. Consult qualified legal counsel in your jurisdiction before publishing. This white paper must be notified to the competent national authority at least 20 working days before publication.

---

# [TOKEN NAME] WHITE PAPER

## Cover Page (Required)

**Date of White Paper**: [DATE]
**Version**: [VERSION NUMBER]

> **Warning**: This crypto-asset white paper has [not been approved / been approved] by any competent authority in any Member State of the European Union. The offeror of the crypto-asset is solely responsible for the content of this crypto-asset white paper.

**Issuer**: [LEGAL ENTITY NAME]
**Registered Office**: [ADDRESS]
**Registration Number**: [COMPANY NUMBER]

**Contact Details**:
- Website: [URL]
- Email: [EMAIL]
- Phone: [PHONE]

---

## 1. Summary (Article 6(1)(a))

*[Maximum 500 words. Must include warnings about risks.]*

### 1.1 Overview

[TOKEN NAME] is a [UTILITY/PAYMENT/E-MONEY] token issued by [ISSUER NAME] for use within the Mycelix federated learning ecosystem.

### 1.2 Key Information

| Attribute | Value |
|-----------|-------|
| Token Name | [NAME] |
| Token Symbol | [SYMBOL] |
| Token Type | Utility Token |
| Blockchain | [BLOCKCHAIN] |
| Token Standard | [ERC-20/etc.] |
| Total Supply | [AMOUNT] |
| Issuer | [LEGAL NAME] |

### 1.3 Risk Warning

> **IMPORTANT WARNINGS**:
> - This crypto-asset may lose some or all of its value
> - This crypto-asset may not always be transferable
> - This crypto-asset may not be liquid
> - If the crypto-asset is covered by compensation or guarantee schemes, such coverage may not cover all losses
> - This crypto-asset is not covered by the investor protection schemes

---

## 2. Information About the Issuer (Article 6(1)(b))

### 2.1 Identity and Contact Details

| Field | Information |
|-------|-------------|
| Legal Name | [FULL LEGAL NAME] |
| Legal Form | [e.g., Swiss Foundation, GmbH, etc.] |
| Registered Address | [FULL ADDRESS] |
| Registration Number | [NUMBER] |
| LEI (if applicable) | [LEI CODE] |
| Date of Registration | [DATE] |
| Jurisdiction | [COUNTRY] |

### 2.2 Website and Contact Information

- **Website**: [URL]
- **Email**: [EMAIL]
- **Contact Form**: [URL]
- **Social Media**: [LINKS]

### 2.3 Principal Activities

[ISSUER NAME] is engaged in the following principal activities:
1. Development and maintenance of the Mycelix federated learning protocol
2. Operation of the token distribution and governance systems
3. Research and development of privacy-preserving machine learning technologies

---

## 3. Information About the Offeror (Article 6(1)(c))

*[If different from issuer]*

| Field | Information |
|-------|-------------|
| Legal Name | [NAME] |
| Legal Form | [FORM] |
| Registered Address | [ADDRESS] |
| Registration Number | [NUMBER] |
| Relationship to Issuer | [DESCRIPTION] |

---

## 4. Information About the Operator (Article 6(1)(d))

*[If seeking admission to trading on a trading platform]*

| Field | Information |
|-------|-------------|
| Trading Platform | [NAME] |
| Platform License | [LICENSE NUMBER] |
| Jurisdiction | [COUNTRY] |
| Contact | [EMAIL] |

---

## 5. Reasons for the Offer/Admission (Article 6(1)(e))

### 5.1 Purpose of the Token

The [TOKEN NAME] token serves the following purposes within the Mycelix ecosystem:

1. **Compute Credits**: Pay for federated learning aggregation services
2. **Governance Participation**: Vote on protocol upgrades and parameter changes
3. **Staking**: Secure the network as a validator node
4. **Reputation Rewards**: Receive rewards for quality ML contributions

### 5.2 Use of Proceeds

Funds raised through token distribution will be allocated as follows:

| Category | Percentage | Description |
|----------|------------|-------------|
| Development | [X]% | Protocol development and maintenance |
| Operations | [X]% | Infrastructure and operational costs |
| Research | [X]% | Privacy and ML research |
| Marketing | [X]% | Community building and adoption |
| Legal/Compliance | [X]% | Regulatory compliance |
| Reserve | [X]% | Emergency fund and future development |

---

## 6. Detailed Description of the Crypto-Asset (Article 6(1)(f))

### 6.1 Technical Characteristics

| Characteristic | Specification |
|----------------|---------------|
| Blockchain | [e.g., Ethereum, Holochain, etc.] |
| Token Standard | [e.g., ERC-20] |
| Consensus Mechanism | [e.g., Proof of Stake] |
| Contract Address | [ADDRESS - after deployment] |
| Decimals | [NUMBER] |
| Transferability | Yes/No/Conditional |

### 6.2 Technology Description

The Mycelix protocol is a decentralized federated learning system that:

1. **Aggregates ML Models**: Combines local model updates without accessing raw data
2. **Preserves Privacy**: Uses differential privacy and cryptographic techniques
3. **Ensures Quality**: MATL trust layer validates contribution quality
4. **Provides Proofs**: Zero-knowledge proofs verify computations

### 6.3 Smart Contract Functionality

```
Main Functions:
- stake(amount): Lock tokens for network participation
- delegate(validator): Delegate stake to a validator
- claimRewards(): Claim accumulated participation rewards
- vote(proposalId, support): Cast governance vote
- withdraw(amount): Unstake tokens (subject to unbonding period)
```

---

## 7. Rights and Obligations (Article 6(1)(g))

### 7.1 Rights of Token Holders

Token holders have the following rights:

| Right | Description |
|-------|-------------|
| **Network Access** | Use tokens to access federated learning services |
| **Governance** | Vote on protocol proposals proportional to stake |
| **Staking Rewards** | Earn rewards for network participation |
| **Transferability** | Transfer tokens to other addresses |
| **Information** | Receive updates about material protocol changes |

### 7.2 Obligations of Token Holders

Token holders must:
- Comply with applicable laws in their jurisdiction
- Accept responsibility for secure key management
- Accept that tokens are not investment products

### 7.3 Issuer Obligations

The issuer commits to:
- Maintain and develop the protocol
- Publish regular transparency reports
- Notify holders of material changes
- Maintain adequate reserves

### 7.4 Procedures for Exercise of Rights

| Right | Procedure |
|-------|-----------|
| Voting | Connect wallet to governance portal, submit vote |
| Staking | Use staking interface or contract interaction |
| Rewards | Call claimRewards() function |
| Transfer | Standard token transfer |

---

## 8. Technology and Risks (Article 6(1)(h))

### 8.1 Technology Description

#### 8.1.1 Distributed Ledger Technology

[TOKEN NAME] operates on [BLOCKCHAIN NAME], which provides:
- Decentralized consensus
- Immutable transaction history
- Smart contract execution

#### 8.1.2 Consensus Mechanism

[Describe the consensus mechanism - e.g., Proof of Stake with specific parameters]

#### 8.1.3 Security Measures

| Measure | Implementation |
|---------|----------------|
| Smart Contract Audits | [AUDITOR NAME], [DATE] |
| Bug Bounty Program | [DETAILS] |
| Multi-sig Treasury | [THRESHOLD]-of-[TOTAL] signers |
| Upgrade Mechanism | [TIMELOCK/GOVERNANCE PROCESS] |

### 8.2 Risk Factors

#### 8.2.1 Technology Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Smart Contract Vulnerabilities | Bugs in code could result in loss | Third-party audits, bug bounty |
| Network Attacks | 51% attacks, DDoS | Decentralization, Byzantine tolerance |
| Key Management | Loss of private keys | User education, recovery options |
| Protocol Upgrades | Changes could introduce issues | Governance process, timelocks |

#### 8.2.2 Market Risks

| Risk | Description |
|------|-------------|
| Price Volatility | Token value may fluctuate significantly |
| Liquidity Risk | May not be possible to sell tokens quickly |
| Competition | Other protocols may reduce utility |

#### 8.2.3 Regulatory Risks

| Risk | Description |
|------|-------------|
| Regulatory Changes | Laws may change adversely |
| Classification | Token classification may change |
| Geographic Restrictions | Access may be limited in jurisdictions |

#### 8.2.4 Operational Risks

| Risk | Description |
|------|-------------|
| Team Risk | Key personnel may leave |
| Funding Risk | Development funding may be insufficient |
| Adoption Risk | Network may not achieve adoption |

---

## 9. Environmental Impact (Article 6(1)(i))

### 9.1 Consensus Mechanism Environmental Impact

[TOKEN NAME] uses [CONSENSUS MECHANISM], which has the following environmental characteristics:

| Metric | Value | Comparison |
|--------|-------|------------|
| Energy per Transaction | [X] kWh | [vs. Bitcoin/Visa/etc.] |
| Annual Network Energy | [X] GWh | [comparison] |
| Carbon Footprint | [X] tons CO2/year | [comparison] |

### 9.2 Sustainability Measures

1. **Proof of Stake**: [X]% more energy efficient than Proof of Work
2. **Carbon Offset**: [DETAILS of any offset program]
3. **Renewable Energy**: Validators encouraged to use renewable energy
4. **Efficient Design**: Federated learning reduces data transfer vs. centralized ML

### 9.3 Environmental Reporting

We commit to publishing annual environmental impact reports including:
- Network energy consumption
- Carbon emissions
- Sustainability improvements

---

## 10. Complaint Handling (Article 6(1)(j))

### 10.1 Complaint Procedure

Token holders may submit complaints through the following channels:

| Channel | Contact | Response Time |
|---------|---------|---------------|
| Email | complaints@[DOMAIN] | 5 business days |
| Online Form | [URL] | 5 business days |
| Mail | [ADDRESS] | 10 business days |

### 10.2 Complaint Process

1. **Submission**: Submit complaint with relevant details
2. **Acknowledgment**: Receive acknowledgment within 2 business days
3. **Investigation**: Internal review within 10 business days
4. **Resolution**: Written response with resolution or explanation
5. **Escalation**: Option to escalate to competent authority if unsatisfied

### 10.3 Competent Authority

If complaints are not resolved satisfactorily, token holders may contact:

**Competent Authority**: [NATIONAL AUTHORITY NAME]
**Address**: [ADDRESS]
**Website**: [URL]

---

## 11. Governance (Article 6(6) - Additional for Utility Tokens)

### 11.1 Governance Structure

| Body | Role | Composition |
|------|------|-------------|
| Foundation Board | Strategic oversight | [NUMBER] members |
| Technical Committee | Protocol decisions | [NUMBER] members |
| Token Holders | Governance votes | All holders |

### 11.2 Voting Mechanism

- **Voting Power**: 1 token = 1 vote (or describe quadratic voting if applicable)
- **Quorum**: [X]% of circulating supply
- **Threshold**: Simple majority / Supermajority for [specific actions]
- **Timelock**: [X] days between approval and execution

### 11.3 Protocol Upgrades

Upgrades follow this process:
1. Proposal submitted on-chain
2. [X]-day discussion period
3. [X]-day voting period
4. [X]-day timelock if approved
5. Automatic execution

---

## 12. Token Allocation and Vesting

### 12.1 Total Supply Allocation

| Category | Percentage | Amount | Vesting |
|----------|------------|--------|---------|
| Network Rewards | [X]% | [AMOUNT] | Distributed over [X] years |
| Foundation | [X]% | [AMOUNT] | [VESTING SCHEDULE] |
| Team | [X]% | [AMOUNT] | [X]-month cliff, [X]-month linear |
| Early Participants | [X]% | [AMOUNT] | [VESTING SCHEDULE] |
| Community | [X]% | [AMOUNT] | Unlocked at launch |
| Reserve | [X]% | [AMOUNT] | Governance-controlled |

### 12.2 Emission Schedule

[Include chart or table showing token emission over time]

---

## 13. Disclaimers and Legal Notices

### 13.1 No Investment Advice

This white paper does not constitute investment advice, financial advice, trading advice, or any other sort of advice. You should conduct your own due diligence and consult a financial advisor before making any decision.

### 13.2 Forward-Looking Statements

This white paper contains forward-looking statements based on current expectations. Actual results may differ materially due to various factors including those described in the Risk Factors section.

### 13.3 Regulatory Status

This crypto-asset [has/has not] been approved by any competent authority. The white paper [has/has not] been notified to [NATIONAL COMPETENT AUTHORITY].

### 13.4 Geographic Restrictions

This offering is not available to residents of [LIST RESTRICTED JURISDICTIONS].

### 13.5 No Guarantee

There is no guarantee of any return, utility, or value. Participation is at your own risk.

---

## Annex A: Glossary

| Term | Definition |
|------|------------|
| DLT | Distributed Ledger Technology |
| Federated Learning | Machine learning where data remains on local devices |
| MATL | Mycelix Adaptive Trust Layer |
| ZK Proof | Zero-Knowledge Proof |
| Validator | Network participant that verifies transactions |

---

## Annex B: Technical Specifications

[Include detailed technical specifications, smart contract addresses, ABIs, etc.]

---

## Annex C: Legal Opinions

[Reference to legal opinions obtained, if applicable]

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | [DATE] | Initial publication |

---

*End of White Paper*
