# THE FEDERATED DAO HIERARCHY CHARTER (v0.22)

**Companion instrument to the Mycelix Spore Constitution v0.22**

---

## PREAMBLE

This Charter operationalizes the polycentric and adaptive governance envisioned in the Constitution, defining tiers, powers, and processes to ensure subsidiarity, efficiency, and alignment with Core Principles.

---

## ARTICLE I – STRUCTURE OF A POLYCENTRIC GOVERNANCE ECOLOGY

### Section 1. Federation Tiers

The Mycelix DAO consists of a dynamic, multi-centered ecosystem rather than a rigid hierarchy:

- **Tier 1**: Local DAOs & Civic Uplift Trusts – Base governance and participation units.
- **Tier 2 (Parallel)**: Sector DAOs and Regional DAOs.
- **Tier 3**: The Global Mycelix DAO – A bicameral legislature.

### Section 2. Liminal DAOs ("Floaters")

DAOs that do not fit within a single Sector or Region (e.g., cross-disciplinary research guilds, identity-based collectives, Wisdom Guilds) may be chartered as Liminal DAOs. They must:

a. Anchor to at least one Sector DAO (for domain expertise) and one Regional DAO (for geographic/legal context). For purely digital/global Liminal DAOs, anchoring directly to the Global DAO is required.

b. Submit bi-annual self-assessments to their anchor(s) and the Knowledge Council for review.

c. Undergo a mandatory lifecycle review after one year, where the anchor(s) and Knowledge Council will issue a formal recommendation to the Global DAO to: 1) Renew liminal status, 2) Integrate into the formal structure, or 3) Initiate dissolution.

d. **Jurisdictional Precedence**: Where a Liminal DAO's activities overlap with an established Sector or Regional DAO, the established DAO's standards and rules prevail unless explicitly exempted by a Global DAO resolution via Mycelix Improvement Proposals (MIPs). Liminal DAOs operate with autonomy but are subordinate to the established structure in cases of direct conflict.

e. **Lifecycle Enforcement**:
   i. If a Liminal DAO fails its one-year review, it shall enter a 90-day remediation period, during which the Knowledge Council will provide a public remediation plan.
   ii. If remediation fails, the Global DAO shall vote on one of two binding actions: forced integration into a formal structure (requiring ⅔ supermajority) or dissolution with assets distributed to its Members (requiring simple majority).
   iii. A Liminal DAO may appeal a dissolution or forced integration order to the Member Redress Council, but only on the grounds that the review process violated Core Principles.

### Section 3. DAO Lifecycle: Fusion and Fission

The Network recognizes that DAOs are living systems that must be able to merge and divide.

a. **Fusion**: Two or more DAOs may merge following the passage of a joint proposal ratified by a supermajority (⅔) in each participating DAO. A Fusion Charter, based on templates provided by the Knowledge Council, must be submitted to the Global DAO for formal recognition.

b. **Fission**: A DAO may split into two or more new entities. A formal fission proposal requires a petition from at least one-third (⅓) of the DAO's members and must include a viable Fission Charter. The proposal must pass with a supermajority (⅔) vote. The Moloch DAO "ragequit" mechanism may serve as a model for protecting minority interests.

c. **Safeguards**: The Global DAO may, by supermajority vote, block any fusion or fission if the Audit Guild or Knowledge Council provides compelling evidence that it would violate Core Principles (e.g., create a monopoly, endanger Member rights). All contested fissions are subject to a mandatory 60-day cooling-off period for mediation by the Inter-Tier Arbitration Council (ITAC).

### Section 4. Representation and Voting

#### 4.1 Delegate Election

Local DAOs elect delegates to their respective Sector and Regional DAOs via ranked-choice voting. Delegates serve 12-month terms, renewable once consecutively.

#### 4.2 Tier 2 → Global DAO

Sector and Regional DAOs elect delegates to Global DAO houses via the same mechanism. Global DAO delegates serve 18-month terms.

#### 4.3 Recall

Any delegate may be recalled by a ⅔ vote of their originating DAO.

---

## ARTICLE II – POWERS AND JURISDICTION

### Section 1. Local DAOs

Manage local community governance and allocate local project funding (up to 50,000 FLOW without higher approval).

### Section 2. Sector DAOs

Govern economic, technical, and ethical standards within their domain and approve project funding up to 500,000 FLOW.

### Section 3. Regional DAOs

Handle cross-jurisdictional legal compliance, manage language standardization, and approve funding up to 500,000 FLOW for regional initiatives.

### Section 4. Global Mycelix DAO

Exercises ultimate constitutional authority, ratifies amendments, approves protocol-level changes (via MIPs), and allocates treasury funds >500,000 FLOW.

### Section 5. Accountability and Reporting

Each Sector and Regional DAO must publish quarterly transparency reports. Failure to report for two consecutive quarters may result in a suspension of funding.

---

## ARTICLE III – LEGISLATIVE AND DECISIONAL PROCESSES

### Section 1. Proposal Flow

Proposals originate at the lowest competent tier and may be escalated if they require resources or authority beyond that tier's limits.

### Section 2. Voting Standards

**Simple Decisions**: Require majority (>50%) in the deciding body.

**Constitutional Matters**: Require supermajority (⅔) in both Global DAO houses, plus tier ratification.

**Quorum**: Votes require ≥40% Member participation for validity. If quorum fails twice, the threshold drops to 30%.

**Third-Tier Quorum Failure**: If a vote fails to reach 30% quorum **three times in a row**, the proposal enters **Expert Adjudication Mode**:

#### Expert Adjudication Process

1. **Delegation to Appropriate Body** (based on proposal type):
   - **Technical (MYC-T)**: Audit Guild votes (simple majority of Guild members)
   - **Economic (MYC-E)**: Knowledge Council votes (simple majority)
   - **Governance (MYC-G)**: Inter-Tier Arbitration Council votes (5/7 required)
   - **Social (MYC-S)**: Member Redress Council votes (simple majority)
   - **Cultural (MYC-C)**: Knowledge Council votes (simple majority)
   - **Constitutional Amendment**: NO EXPERT OVERRIDE (proposal fails definitively)

2. **Expert Decision Constraints**:
   - Decisions valid for **90 days only** (temporary)
   - Must be re-ratified by standard DAO vote within 90 days
   - If re-vote also fails quorum: Expert decision becomes permanent
   - All expert decisions flagged as `"non_democratic_resolution": true` in public ledger

3. **Accountability**:
   - Expert bodies must publish detailed reasoning (minimum 500 words)
   - Global DAO may overturn expert decision by **simple majority** (lower bar than usual ⅔)
   - Repeated use of expert override (>5 times/year) triggers constitutional review

#### Systematic Engagement Decay Response

If **average quorum across all votes falls below 35% for 2 consecutive quarters**, the Network enters **Engagement Crisis Mode**:

**Phase 1: Diagnosis** (Mandatory within 30 days)
- Knowledge Council conducts community survey (minimum 500 responses)
- Survey assesses: proposal fatigue, complexity issues, qualification concerns, relevance problems, UX friction
- Results published publicly with analysis

**Phase 2: Intervention** (Based on diagnosis)
- **If Proposal Fatigue**: Implement proposal throttling (max 50/month), increase proposal bond
- **If Complexity**: Mandate plain-language summaries, video explanations, assign proposal champions
- **If Qualification/Intimidation**: Implement delegation by default, liquid democracy
- **If Relevance**: Improve proposal routing, create opt-in voting by domain
- **If UX Problem**: Audit voting interfaces, implement mobile-first, gasless voting

**Phase 3: Structural Reform** (If interventions fail after 4 quarters <35%)
- **Option A**: Shift to Conviction Voting (continuous staking on proposals)
- **Option B**: Reduce Governance Scope (move operational decisions to elected councils)
- **Option C**: Incentivize Participation (pay for voting with quiz requirement)

**Nuclear Option: Governance Moratorium**
- If engagement remains <20% for 6 quarters, Global DAO may pause governance for 6 months (⅔ vote)
- Community workshops redesign governance system
- Constitutional convention proposes reforms
- Moratorium ends with new system (⅔ ratification required)

### Section 3. Adaptive Delegation & Emergency Response

In response to a declared Network Emergency (per Constitution Article IX), the Global DAO may authorize the formation of temporary, cross-functional Task DAOs with specific, time-bound mandates.

a. **Activation**: Emergency powers may only be activated by the multisig defined in Constitution Article IX, under the strict conditions listed therein.

b. **Safeguards**: No emergency action may violate the Bill of Rights (Constitution Article VI) or the Immutable Core (Constitution Article IV). All emergency actions require public notification within 24 hours.

c. **Sunset and Ratification**: All emergency powers and temporary bodies automatically expire after 30 days unless renewed by a ⅔ Global DAO vote with published justification. All actions taken under emergency authority require retroactive ratification by the appropriate DAO tier to remain in effect.

d. **Task DAO Limitations**:
   i. Emergency Task DAOs may not amend the Constitution or Core Principles, allocate treasury funds greater than 500,000 FLOW without separate Global DAO approval, suspend Member rights as defined in Article VI, or override decisions of the Member Redress Council.
   ii. All actions taken by a Task DAO require daily public reporting to the Network.
   iii. Any Member may petition the Member Redress Council for an expedited (3-day) review of a Task DAO's actions for constitutional violations.
   iv. Task DAOs automatically dissolve after 30 days unless renewed by ⅔ Global DAO vote with published justification.

### Section 4. Voting Mechanisms

#### 4.1 Governance Voting

**Reputation-Weighted Voting** shall be used for political and governance decisions, including:
- Constitutional amendments
- MIP ratification
- Treasury allocations
- Delegate elections

#### 4.2 Rights-Based Voting

**Equal-Weight Voting (1 Member, 1 Vote)** shall be used for decisions directly adjudicating individual or collective rights, including:
- All Member Redress Council decisions
- Impeachment proceedings
- Votes on ethical audit findings
- Any decision directly affecting individual rights per Constitution Article VI

#### 4.3 Resource Allocation

**Quadratic Voting** shall be used for specific resource allocation mechanisms where measuring preference intensity is paramount, such as:
- Community grants
- Quadratic funding rounds
- Public goods prioritization

#### 4.4 Precedence

In cases where the appropriate voting mechanism is ambiguous, the default shall be Equal-Weight Voting unless a pre-vote procedural motion to use Reputation-Weighted Voting passes with a 60% majority.

---

## ARTICLE IV – DISPUTE RESOLUTION

### Section 1. Jurisdictional Conflict

Conflicts are resolved according to a defined order of precedence: 
1. Constitution
2. Global DAO
3. Regional DAO (on legal matters)
4. Sector DAO (on technical matters)

### Section 2. Inter-Tier Arbitration Council (ITAC)

A 7-member rotating council that mediates disputes and issues non-binding recommendations. Final escalation is to the Global DAO.

### Section 3. Frivolous Disputes

A dispute is frivolous if it fails to cite a specific rule violation, is identical to a prior case, or shows bad faith. Penalties include reputation reduction and fines.

---

## ARTICLE V – OPERATIONAL EFFICIENCY

### Section 1. Meeting Cadence

Local DAOs meet at their discretion. Tier 2 DAOs meet monthly. The Global DAO meets quarterly.

### Section 2. Delegation and Proxies

Members may assign revocable, on-chain proxies for votes. All proxy assignments are public.

### Section 3. Decision Velocity

Non-controversial proposals may be fast-tracked. Experimental proposals include 12-month sunset clauses.

---

## ARTICLE VI – ACCOUNTABILITY AND PERFORMANCE

### Section 1. DAO Performance Metrics

Each DAO tier must track and publish metrics on participation, proposal throughput, and budget efficiency. Persistently underperforming DAOs may be restructured by the Global DAO.

**Validator Diversity Metrics** (published quarterly):
- Geographic distribution (by Regional DAO)
- Sector distribution (by Sector DAO)
- Social graph clustering coefficient
- Cartel risk alerts triggered
- Cartels dissolved (if any)

**Bridge Security Metrics** (published quarterly):
- Total Value Locked (TVL) across all bridges
- Insurance coverage ratio (coverage ÷ TVL)
- Bridge Recovery Fund balance
- Security incidents detected
- Exploits successfully contained (if any)
- Average response time to security alerts

**Governance Health Dashboard** (Public, Real-Time):
- Quorum Rate: % of votes meeting quorum (30-day trailing)
- Turnout Trend: Line chart of participation over time
- Proposal Quality: % of proposals passing vs failing
- Delegate Utilization: % of Members using delegation
- Response Time: Median time from proposal to resolution

Dashboard alerts:
- 🟢 Green: Quorum >40%, turnout stable
- 🟡 Yellow: Quorum 30-40%, declining turnout (Phase 1 diagnosis triggered)
- 🔴 Red: Quorum <30%, governance crisis (Phase 2 intervention required)

### Section 2. Local Feedback Loops

To enable adaptive governance, every DAO and Civic Uplift Trust must submit a lightweight quarterly report (in a standardized JSON format) and a more detailed annual narrative digest.

a. **Content**: Reports shall include key metrics, decisions, challenges, and qualitative feedback.

b. **Synthesis**: The Knowledge Council is mandated to aggregate these reports into an annual Network Resilience Report, identifying systemic risks and opportunities.

c. **Actionability**: The Global DAO must formally vote on any actionable recommendations presented in the Network Resilience Report within 90 days of its publication. Audit frequencies may be semi-automatically adjusted based on risk scores derived from these reports, subject to appeal.

**Quarterly Engagement Reports**: Knowledge Council publishes analysis of:
- Which proposal types get highest/lowest turnout
- Which Members are most/least engaged
- Which Sectors/Regions have healthiest governance
- Recommendations for improvement

### Section 3. DAO Dissolution and Reformation

DAOs may be dissolved voluntarily (⅔ majority) or involuntarily by the Global DAO for cause (e.g., fraud, repeated constitutional violations).

### Section 4. Anti-Capture Monitoring

To maintain polycentric balance, the Audit Guild shall continuously monitor influence concentration. If any single Member, external entity, or coordinated group demonstrably controls more than 15% of the voting power or receives more than 25% of the reputation allocated within any single DAO (Local, Sector, Regional) over a 12-month period, an automatic audit and review by the Global DAO shall be triggered.

---

## ARTICLE VII – FORMATION, RECOGNITION, AND CIVIC UPLIFT

### Section 1. Local DAO Formation

Requires a minimum of 10 verified Members, a published charter, and recognition by its parent Sector and Regional DAOs.

### Section 2. Sector and Regional DAO Formation

Requires a minimum of 3 constituent Local DAOs and approval from the Global DAO.

### Section 3. Civic Uplift Trusts

To fulfill the principle of Civic Equity, the Network provides a formal pathway for communities that are not yet DAO-native.

a. **Purpose**: A Civic Uplift Trust is a pre-DAO entity focused on education, capacity-building, and bridging the digital divide for a specific community. Its primary goal is to achieve "DAO Graduation."

b. **Formation**: A Trust may be formed by a recognized external partner (e.g., an NGO, educational institution) or by a group of at least 3 Members acting as Custodians. Formation requires a charter and recognition by the relevant Regional DAO.

c. **Governance**: Trusts are governed by their Custodians, who act as fiduciaries for the beneficiary community. All actions must be transparent and aimed at empowering the community toward self-governance.

d. **DAO Graduation**: A Trust graduates to a full Local DAO when it meets criteria set by the Global DAO, including: a minimum number of verifiably unique Members from the community, demonstrated capacity for self-governance, and ratification of its own Local DAO charter. Upon graduation, all custodial authority is dissolved.

### Section 4. Pre-DAO Incubator Status

Communities not yet ready for full DAO or Trust status may receive **Pre-DAO Incubator** designation:

a. **Eligibility**: Groups of 5+ individuals from underserved regions or communities seeking to learn governance practices.

b. **Support**: Knowledge Council provides:
   - Educational scaffolding
   - Legal setup assistance
   - Cultural translation of governance concepts
   - Onboarding resources

c. **Funding**: Incubators may receive grants from Global Solidarity Fund (up to 10,000 FLOW) for capacity building.

d. **Transition**: Incubators transition to Civic Uplift Trusts when they reach 10 verified Members and demonstrate basic governance capacity.

---

## ARTICLE VIII – CROSS-TIER COORDINATION

DAOs may form joint initiatives with shared budgets and governance. All DAOs contribute to a shared Data Commons for records and best practices.

---

## ARTICLE IX – RESILIENCE AND EMERGENCY SYSTEMS

### Section 1. Knowledge Continuity Mandate

All recognized DAOs (Local, Sector, Regional) must develop and annually update a **Civic Continuity and Disaster Recovery Plan (DRP)**. These plans must include:

a. **DHT Redundancy Strategy**: Measures to ensure critical Holochain data is redundantly stored across diverse geographic zones.

b. **Human-Readable Snapshots**: An annual export of the DAO's critical governance structure (charter, policies, membership ledger) in a standardized format (e.g., Markdown + JSON).

c. **Knowledge Council Review**: DRPs are subject to review by the Knowledge Council, which provides templates and publishes an annual "State of Network Resilience" report based on these reviews.

d. **Fractal Resilience Mandate**: Each Region, Sector, and Local DAO must independently maintain:
   - Emergency funding reserves (minimum 3 months operations)
   - Off-chain protocol mirrors (DHT redundancy)
   - Constitutional fallback (Liminal Charters if root Charter fails)
   - Crisis Response Plans (seismic, climate, war, digital partitioning)

### Section 2. Planetary Recovery Protocol

The Global DAO and Knowledge Council shall jointly maintain a minimal, open-source **Civilization Recovery Bundle (CRB)**.

a. **Contents**: The bundle shall contain the Seed Constitution, Charter Templates, core protocol codebase (RB-BFT, Reputation), DKG/ZK toolchains, and a recovery onboarding kit.

b. **Archival**: The bundle must be archived in at least three geographically distinct, high-resilience archives (e.g., Arctic World Archive, Internet Archive, GitHub Arctic Code Vault) and anchored annually to a public L1 blockchain.

c. **Distributed Data Preservation**: All DAOs must:
   - Maintain encrypted backups of governance history, votes, and identity attestations
   - Use distributed off-chain mirrors (IPFS, Arweave, Filecoin)
   - Store at least 1 recovery replica outside their region to prepare for geopolitical or climate risk

---

## ARTICLE X – FINANCIAL GOVERNANCE

### Section 1. Budgeting Process

Each DAO tier prepares annual budgets, which are consolidated and approved by the Global DAO.

### Section 2. Funding Sources and Allocation

#### 2.1 Protocol Revenue Distribution

Protocol revenues are distributed according to a fixed formula:
- 12% to oversight (KC 3%, AG 5%, MRC 4%)
- 20% to Global DAO
- 30% to Sector DAOs
- 20% to Regional DAOs
- 13% to Foundation reserve
- 5% to emergency reserve

#### 2.2 Tier 2 Pass-Through Funding Mandate

To ensure Subsidiarity and empower grassroots activity, all Sector DAOs and Regional DAOs must programmatically pass through a minimum of 40% of their allocated protocol revenue to their constituent Local DAOs and recognized Civic Uplift Trusts. 

Distribution shall be proportional to **active, verified participation**, calculated using a weighted formula:

```
Funding_Share = (Voting_Activity × 0.4) + (Proposal_Quality × 0.3) + 
                (Member_Count × 0.2) + (Civic_Uplift_Impact × 0.1)
```

The precise metrics and weightings shall be defined and maintained via a MIP (Economic). 

**Anti-Gaming Provisions**:
- Local DAOs with fewer than 5 active Members or fewer than 2 proposals per quarter shall receive a minimum viable allocation only.
- The Audit Guild shall monitor for Sybil patterns, such as multiple Local DAOs controlled by the same actors.
- Circular funding patterns (A→B→C→A) trigger fraud review.

### Section 3. Financial Controls

Expenditure limits and multi-signature requirements are in place for all DAO treasuries to prevent misuse of funds.

### Section 4. Long-Term Fiscal Sustainability

#### a. Reserve Requirements

The Network shall maintain minimum reserves:
- **Foundation**: 18 months of operating expenses (Constitution Article II, 2.1.d)
- **Each Oversight Body**: Maximum 24 months of operating expenses
- **Emergency Reserve**: 6 months network-wide operations

#### b. Revenue Stress Testing

The Audit Guild shall conduct and publish an annual scenario analysis modeling the impact of severe revenue shocks:
- 50% revenue drop
- 90% reduction in new Members
- Major bridge exploit requiring insurance payout

Results shall inform reserve target adjustments via MIP.

#### c. Graduated Deficit Protocol

If protocol revenue falls below certain thresholds, graduated responses are triggered:

**Warning State** (Revenue <150K FLOW for 1 quarter):
- Monitoring only
- No action required
- Alert published to Global DAO

**Conservation State** (Revenue <100K FLOW for 2 quarters):
- 10% discretionary budget cut
- Hiring freeze for non-critical roles
- Mandatory efficiency audit by Audit Guild

**Emergency State** (Revenue <50K FLOW for 3 quarters):
- 25% budget cut (including oversight, down to 9% from 12%)
- Foundation reserve fund activated
- Global DAO votes on austerity plan

**Existential State** (Revenue <25K FLOW for 4 quarters):
- Orderly wind-down plan activated
- Member assets protected (Constitution Article X, Section 2.2)
- Dissolution procedures initiated

### Section 5. Bridge Security and Recovery

#### 5.1 Bridge Insurance Requirements

**Mandatory Coverage**: The Foundation shall maintain bridge exploit insurance coverage equal to:
- **Minimum**: 50% of Total Value Locked (TVL) on all bridges
- **Target**: 100% of TVL
- **Coverage Types**: Smart contract exploit insurance, validator collusion insurance, oracle manipulation insurance

**Funding Source**: 2% of protocol revenue automatically directed to insurance premium fund. If premiums exceed 2%, Foundation reserve covers difference.

**Self-Insurance Reserve**: If commercial insurance unavailable, Foundation maintains **Bridge Recovery Fund** = 25% of TVL (minimum), held in multi-sig (5-of-9: 3 Foundation Directors, 3 Audit Guild, 3 Knowledge Council).

#### 5.2 Tiered Bridge Deployment

Bridges deployed in stages with TVL caps:

| Phase | Bridge Type | Max TVL | Security Requirements |
|-------|-------------|---------|----------------------|
| **Alpha** | Testnet | $0 (test tokens) | Internal audit only |
| **Beta** | Mainnet Limited | $1M | 2 external audits + 90-day testnet |
| **Gamma** | Mainnet Standard | $10M | 3 external audits + bug bounty + 180-day beta |
| **Production** | Mainnet Full | Unlimited | 4+ audits + formal verification + 1-year gamma |

**TVL Cap Enforcement**: Smart contract enforces cap (rejects deposits above limit). To raise cap: Requires MYC-T proposal + new audit.

#### 5.3 Continuous Security Monitoring

Audit Guild operates 24/7 bridge monitoring:
- **Automated**: On-chain monitoring (unusual transaction patterns)
- **Manual**: Weekly validator behavior review
- **External**: Quarterly penetration testing by whitehat hackers

**Alert Thresholds**:
- 🟡 Yellow: Unusual activity detected (e.g., large withdrawal spike) - Audit Guild reviews within 1 hour
- 🔴 Red: Critical anomaly (e.g., validator collusion evidence) - Technical Circuit Breaker activated immediately

#### 5.4 Post-Exploit Recovery Protocol

**Phase 1: Emergency Response (0-72 hours)**
- Hour 0-2: Detection and containment
- Hour 2-12: Forensics (determine exploit vector, quantify losses)
- Hour 12-24: Communication (publish incident report)
- Day 2-3: Patch development and emergency audit

**Phase 2: Recovery Execution (Days 3-30)**

**Funding Recovery** (Priority order):
1. Insurance Claims (commercial insurer)
2. Bridge Recovery Fund (self-insurance reserve)
3. Foundation Reserve (emergency backstop, up to 50%)
4. Treasury (last resort, requires ⅔ Global DAO vote)

**User Reimbursement Priorities**:
- **Tier 1 (Immediate)**: Small users (<$1,000 affected) - 100% reimbursement within 7 days
- **Tier 2 (Fast)**: Medium users ($1K-$100K) - 100% reimbursement within 30 days
- **Tier 3 (Delayed)**: Large users (>$100K) - Staged reimbursement over 90 days

**Phase 3: Prevention (Days 30-90)**
- Root cause analysis published
- System improvements implemented
- Policy changes via MYC-T proposal
- Personnel actions if exploit due to negligence

#### 5.5 Catastrophic Failure Modes

If bridge exploit **exceeds available recovery funding**:

**Option A: Socialized Losses** (Requires ⅔ Global DAO vote)
- Mint new tokens to cover shortfall (dilution)
- Pro-rata distribution to affected users
- Maximum socialized loss: 5% of total supply (prevents hyperinflation)
- Requires forensic proof exploit was not due to Foundation negligence
- One-time event only

**Option B: Haircut Recovery**
- Affected users receive partial reimbursement (e.g., 70 cents per dollar)
- Remaining losses written off
- Users may sue Foundation in Swiss courts

**Option C: Graceful Shutdown**
- Activate Constitution Article X, Section 2 (Network Dissolution)
- Orderly wind-down of operations
- Remaining treasury distributed to Members

#### 5.6 Validator Accountability

**Validator Slashing for Bridge Exploits**: If exploit caused by validator negligence or collusion:
- Guilty validators: **100% reputation slash** (ejected permanently)
- Guilty validators: **100% stake slash** (e.g., 5,000 FLOW bonded amount)
- Slashed funds go to bridge recovery fund

**Proof Requirements**: Audit Guild must provide cryptographic proof of validator fault (e.g., signed invalid Merkle proof, participated in 51% attack).

**Due Process**: Accused validators may appeal to Member Redress Council within 7 days. MRC conducts expedited review (decision within 14 days).

**Validator Insurance Requirements**: To serve as bridge validator, must carry professional liability insurance (minimum $1M coverage) OR post additional collateral (10,000 FLOW minimum).

#### 5.7 Multi-Bridge Redundancy

**Bridge Diversification Strategy**:
- Phase 1: Merkle + RB-BFT validators (single bridge type)
- Phase 2: Add ZK-STARK bridge (different security model)
- Phase 3: Add IBC bridge (Cosmos ecosystem)
- Phase 4: Add XCMP bridge (Polkadot)

**TVL Limits Per Bridge**: No single bridge holds >50% of total TVL. Enforced via smart contract deposit caps.

### Section 6. Validator Economics (To Be Defined)

Validator node operator incentives, hardware requirements, and economic models shall be defined via **MYC-E-001 (Validator Economics Framework)** proposal prior to Phase 1 mainnet launch.

**Minimum requirements**:
- Validators must post collateral (5,000 FLOW minimum)
- Rewards proportional to validation accuracy and uptime
- Slashing for malfeasance or extended downtime (>48 hours)
- Geographic diversity incentives (bonus for underrepresented regions)

**Timeline**: MYC-E-001 drafted Q3 2025, ratified Q4 2025.

### Section 7. Salary and Compensation

**Salary Model**:
- Base salaries paid entirely in stablecoins (USDC/DAI/LUSD - recipient choice)
- Optional performance bonuses may be paid in utility tokens (if FLOW adopted)
- Vesting options: Long-term contributors may receive vested CIV allocation (non-transferable, "civic options")

**No CIV Monetization**: Civic Standing (CIV) shall never be used for salaries or direct economic compensation. CIV represents citizenship and influence, not capital.

---

## ARTICLE XI – MEMBER ENGAGEMENT AND EDUCATION

### Section 1. Onboarding

All new Members receive a constitutional overview, a governance tutorial, and an assigned mentor.

**Phased Verifiable Credential (VC) Issuance**:

**Phase 1 (Genesis, 0-12 months)**: VCs issued primarily by core institutions:
- Provisional Knowledge Council, Audit Guild, Member Redress Council
- Mycelix Foundation
- Recognized external partners (security auditors, identity verifiers like Gitcoin Passport)

**Types**: `VC:GenesisSteward`, `VC:CompletedProtocolOnboarding`, `VC:VerifiedHumanity`, `VC:ContributedToCoreCodebase`, `VC:ServedOnProvisionalCouncil`

**Phase 2 (12-24 months)**: Sector and Regional DAOs empowered to issue domain-specific VCs:
- Must register VC schemas with Knowledge Council
- Subject to Audit Guild review
- Examples: `VC:AISafetyCertification` (AI Sector DAO), `VC:GDPRComplianceExpert` (EU Regional DAO)

**Phase 3 (24+ months)**: Local DAOs and peer-to-peer issuance:
- Local DAOs issue VCs for local contributions once they demonstrate stability
- Individual Members above CIV threshold may issue peer attestations
- Lower-weight in global CIV calculations initially
- Examples: `VC:LedCommunityProjectX`, `VC:HearthMentor`

### Section 2. Ongoing Education

A **Governance Academy** offers workshops on proposal drafting, voting strategies, and leadership development.

**Cultural Resources**:
- Access to Wisdom Library (maintained by Knowledge Council)
- Optional ritual templates for community ceremonies
- Constitutional commentary in multiple languages
- Accessibility resources and plain-language guides

### Section 3. Participation Incentives & Civic Gifting

#### a. Participation Incentives

Members earn reputation bonuses for consistent voting and submitting high-quality proposals:
- Vote on >50% of proposals in your domain: +0.02 CIV per quarter
- Vote on >80% of proposals: +0.05 CIV per quarter
- Perfect attendance (100% of votes): +0.1 CIV per quarter + "Engaged Citizen" badge

Delegates face removal for poor attendance (<90% participation rate).

**Delegate Recognition Program**:
- Delegate rewards: 0.5 FLOW per delegator per month (if FLOW adopted)
- Max 100 delegators per delegate (prevents over-concentration)
- Delegates must vote on >90% of proposals (auto-removal if below)

**Proposal Quality Incentives**:
- Proposals passing with >75% approval: Sponsor receives 500 FLOW reward
- Proposals failing badly (<25% approval): Sponsor loses proposal bond
- Encourages high-quality, consensus-building proposals

#### b. Civic Gifting Credits (CGCs)

##### i. Allocation

Each verified Member receives **10 CGCs per month**. These credits are non-cumulative and expire at the end of each monthly cycle.

##### ii. Transfer Transparency

Rather than hard caps, CGCs use transparency thresholds:
- No hard limits on CGC transfers (preserves flexibility)
- Members receiving >100 CGCs/month OR >50 from single source flagged in public **"High-Activity CGC Report"**
- Audit Guild reviews flagged accounts quarterly for gaming patterns
- Legitimate high-value contributors (educators, organizers) may appeal flag to Knowledge Council with explanation

##### iii. Sybil Protection

The Audit Guild shall monitor CGC flows for fraudulent patterns:
- Circular gifting (A→B→C→A) triggers review
- Reputation penalty for confirmed gaming
- Whistleblower rewards for reporting (30% of penalties)

##### iv. Reputation Integration

Net CGC inflow may be used as one discretionary input (with a maximum weight of 10%) in local reputation calculations, as determined by Local DAO policy.

##### v. Cultural Naming

Local DAOs may adopt cultural aliases for CGCs while maintaining interoperability:
- Examples: SPARK, EMBER, PETAL, LIGHT, GRATITUDE
- All aliases map to same underlying CGC primitive
- Symbol Registry (maintained by Knowledge Council) tracks aliases

#### c. Reputation Decay

A Member's reputation shall decay according to a multi-factor model:

##### i. Passive Decay

A 5% linear decay per year is applied to reputation for Members who do not actively participate in governance (voting, proposing, serving as delegate) or receive new reputation allocations within that year.

##### ii. Performance Decay

Immediate, event-triggered decay is applied for validation failures:
- **Incorrect validation**: 50% slash for an incorrect validation
- **Timeout**: 2% penalty per timeout incident

##### iii. Decay Floor

Reputation cannot decay below a baseline of **0.05** (maintains baseline humanity recognition).

##### iv. Recovery Path

Members whose reputation falls below 0.4 may enter **"Apprentice Mode"** to rebuild their standing through co-validation with high-reputation nodes. Parameters defined by MIP.

#### d. Reputation Growth

A Member's reputation may increase through verified positive contributions:

##### i. Validation Performance (Primary Growth)

- **Correct Validation**: +0.05 reputation per successful validation round
  - Maximum growth rate: 0.05 per round (prevents gaming)
  - Applied after RB-BFT consensus confirms correctness
  - Capped at 1.0 (maximum reputation)

- **Consistency Bonus**: Members with 100+ consecutive correct validations receive a 1.2x multiplier on growth rate

##### ii. Governance Participation (Secondary Growth)

- **Proposal Quality**: +0.02 reputation for proposals that pass with >75% support
- **Constructive Dissent**: +0.01 reputation for dissenting votes on proposals that later fail audit review (rewards vigilance)

##### iii. Civic Contributions (Tertiary Growth)

- **Mentorship**: +0.01 reputation per successfully onboarded Member (up to 10/year)
- **Accessibility Work**: +0.02 reputation for contributions to accessibility tooling (verified by Knowledge Council)

##### iv. Growth Rate Limits (Anti-Gaming)

- Maximum reputation increase: **0.1 per month** (prevents sudden dominance)
- Growth above 0.8 requires **additional verification**:
  - Peer review by 3 randomly selected high-rep Members
  - Knowledge Council spot audit (5% sample rate)
- Genesis Cohort members subject to 50% slower growth rate (0.05/month max)

##### v. Genesis Period Special Rules (First 6 months)

During the Genesis Period, reputation growth is accelerated to bootstrap the validator network:
- All growth rates **doubled** (0.1 → 0.2 for validation growth)
- Minimum validation threshold **lowered** to 0.3 (from 0.4)
- After 6 months, thresholds revert to standard values
- Members must re-qualify if they fall below 0.4

**Rationale**: Without codified growth mechanisms, the reputation system cannot function. Decay without growth creates inevitable collapse toward minimum reputation.

#### e. Genesis Bootstrapping

##### i. First 1,000 Members

The first 1,000 Members of the Network shall receive a starting reputation of **0.5** (vs 0.1 default).

##### ii. Genesis Cohort

Genesis Cohort members shall start at **0.7** but remain subject to the constitutional voting power cap (Constitution Article V, Section 3: max 0.5% per vote) and slower growth rate.

##### iii. Transition Threshold

After the Network reaches 10,000 total Members, all new entrants shall start at the default reputation of **0.1**.

**Rationale**: Early adopters take higher risk; moderate boost accelerates network effect without creating plutocracy.

---

## ARTICLE XII – TECHNOLOGY AND INFRASTRUCTURE

### Section 1. Core Infrastructure

All governance infrastructure must be decentralized, open-source, and adhere to Mycelix Interoperability Standards (MIS).

### Section 2. Security

Smart contract upgrades are subject to a 14-day timelock and require Audit Guild attestation.

### Section 2.5. Consensus Parameter Governance

Critical RB-BFT consensus parameters shall be governed via MYC-T (Technical) proposals, subject to the following minimum requirements:

| Parameter | Initial Value | Change Threshold | Audit Requirement |
|-----------|---------------|------------------|-------------------|
| MIN_REPUTATION_THRESHOLD | 0.4 | ⅔ Global DAO | Mandatory Security Audit |
| Reputation Exponent | 2 (quadratic) | ⅔ Global DAO | Mandatory Simulation & Audit |
| Decay Rate (passive) | 5%/year | Simple Majority | Optional |
| Decay Rate (malicious) | 50% slash | ⅔ Global DAO | Mandatory Security Audit |
| Ejection Threshold | 0.1 | ⅔ Global DAO | Optional |

**Audit Procurement**:
- Proposer must fund audit (refunded from treasury if proposal passes)
- Audit must be from reputable firm (Knowledge Council maintains approved list)
- Audit report published 14 days before vote (community review period)

**Ambiguous Audit Results**:
- If audit conclusion is "uncertain" or "depends on assumptions":
  - Parameter change requires 75% vote (higher bar)
  - OR deploy to testnet for 90 days before mainnet

**Emergency Parameter Adjustment**:
- Technical Circuit Breaker (Constitution IX.1) may adjust parameters during active exploits
- Changes limited to ±20% of current value (prevents radical changes)
- Change expires after 72 hours unless ratified by Global DAO

The Technical Circuit Breaker may temporarily adjust these parameters to mitigate an active exploit, subject to immediate Global DAO review.

### Section 2.6. Real-Time Cartel Detection

The RB-BFT validator selection algorithm shall include real-time graph analysis to detect coordinated behavior:

**Automatic Clustering Analysis**:
```python
def detect_reputation_cartels(validator_network, time_window=90_days):
    # Build validation graph (who validated whom)
    G = build_validation_graph(validator_network, time_window)
    
    # Detect strongly connected components (mutual validation loops)
    clusters = find_strongly_connected_components(G)
    
    # Flag suspicious clusters based on metrics:
    # - Internal validation rate > 70%
    # - External validation rate < 30%
    # - Temporal correlation > 0.8
    
    return suspicious_clusters
```

**Automatic Risk Mitigation**:

**Phase 1: Soft Alert (Risk 0.6-0.75)**
- Audit Guild notified automatically
- Cluster members' validation power **temporarily reduced by 20%** (not reputation slash)
- Members notified: "Your validation pattern flagged for review"
- 14-day review period for appeal

**Phase 2: Hard Restriction (Risk 0.75-0.90)**
- Cluster members **cannot validate each other's work** for 90 days
- Validation power reduced by 50%
- Requires Audit Guild manual review within 7 days
- If confirmed benign: Restrictions lifted + reputation restored
- If confirmed malicious: Proceed to Phase 3

**Phase 3: Cartel Dissolution (Risk >0.90)**
- All cluster members' reputation **reset to 0.3** (above ejection threshold but below validator threshold)
- 6-month probation: Cannot serve as validators
- Must rebuild reputation through **non-validation contributions**
- Permanent flag: "Formerly flagged for cartel behavior" (public transparency)

**Economic Penalties**:
- **Validator Bonding Requirement**: All validators must stake **5,000 FLOW** (slashable deposit)
- If identified as cartel member: **100% slash** (entire stake)
- Funds distributed to whistleblowers (30%) and treasury (70%)

**Reputation Decay Acceleration**: Members of dissolved cartels experience **2x passive decay** for 12 months (5%/year → 10%/year).

**Diversity Quotas in Validator Selection**:
```rust
pub fn select_diverse_validator_set(
    candidate_pool: Vec<Validator>,
    num_validators: usize,
) -> Vec<Validator> {
    // Enforce diversity constraints:
    let max_per_region = (num_validators as f64 * 0.3).floor() as usize;
    let max_per_sector = (num_validators as f64 * 0.4).floor() as usize;
    let max_per_cluster = (num_validators as f64 * 0.2).floor() as usize;
    
    // Select via reputation-weighted sampling with diversity constraints
    // ...
}
```

**Whistleblower Protections**:
- Anonymous reporting via Tor hidden service (operated by Audit Guild)
- Zero-knowledge proof of evidence (don't reveal reporter identity)
- Whistleblower rewards: 30% of slashed stake (up to 50,000 FLOW) if cartel confirmed
- Protection from retaliation (Member Redress Council jurisdiction)

**False Accusation Penalties**:
- False cartel accusation (proven malicious): 1,000 FLOW fine + reputation penalty
- Negligent false accusation (honest mistake): 100 FLOW fine, no reputation penalty

### Section 3. Privacy-Preserving Governance

Members may cast votes using zero-knowledge proofs to preserve ballot secrecy.

### Section 4. Algorithmic Systems Standards

a. **Public Registry**: The Knowledge Council shall maintain a public registry of all significant AI/ML systems deployed within Network governance, including documentation links, audit reports, and designated points of contact.

b. **Explainability**: Systems must provide human-readable explanations for their outputs upon request (per Constitution Art VI, Sec 11).

c. **Open Weight Preference**: Models developed with Network treasury funds should default to releasing weights and training configurations under open licenses, unless exempted by a Global DAO supermajority vote citing specific security or privacy risks.

d. **Decentralized Hosting Goal**: The Network shall strive, where technically feasible and secure, to host critical AI governance functions on decentralized compute infrastructure. Implementation details shall be determined via MIP.

e. **Mandatory Disclosure**: All AI-authored proposals, comments, or validations must be clearly flagged as originating from Instrumental Actors.

### Section 5. Epistemic Ledger Requirements

a. All governance-significant claims must be submitted via a standardized **Epistemic Claim Object** conforming to the Network schema (see Appendix F).

b. Clients and DAOs must implement schema validation at input.

c. The Decentralized Knowledge Graph (DKG) shall serve as the primary, append-only ledger for all Tier ≥ 2 claims, ensuring a permanent and auditable record.

d. Core Network infrastructure must provide standardized APIs for indexing, querying, and subscribing to Epistemic Claim Objects based on schema fields.

### Section 6. Core Registries

The Network shall maintain a set of core registries, including but not limited to:

a. **The Epistemic Claim Registry**: An append-only, queryable ledger containing all Tier ≥ 2 claims.

b. **The Trust Tag Ontology**: A versioned dictionary of source labels and validation standards managed by the Knowledge Council.

c. **The Claim Lineage Map**: A graph representation of claim relationships for epistemic reasoning.

d. **Epistemic Dispute Resolution Protocol**:

##### i. Tiered Dispute Resolution

When conflicting Tier 2+ claims are submitted, disputes are resolved through tiers based on claim_materiality:

**Tier A: Routine Claims** (claim_materiality = "routine")
- Threshold: <10,000 FLOW impact, no governance implications
- Process: 3-member Sector DAO panel issues binding decision in 14 days
- Appeal: 50 FLOW bond to escalate to Tier B

**Tier B: Significant Claims** (claim_materiality = "significant")  
- Threshold: 10K-100K FLOW impact or sector-wide policy
- Process: 5-member Knowledge Council panel → Relevant Sector/Regional DAO votes (simple majority, 30% quorum)
- Fallback: If quorum fails twice, Knowledge Council decision is binding

**Tier C: Constitutional Claims** (claim_materiality = "constitutional")
- Threshold: Affects Core Principles, Member rights, or >100K FLOW
- Process: 
   1. Disputing party files challenge (500 FLOW equivalent bond).
   2. Knowledge Council + Audit Guild joint panel (7 members) conducts review.
   3. Mandatory public comment period (30 days).
   4. Global DAO votes with special majority: 60% approval required in both houses.
   5. Fallback: If the Global DAO vote fails quorum twice, the joint KC+AG panel's consensus decision (requiring 5 out of 7 votes) becomes binding, but flagged as "non_democratic_resolution": true. This fallback decision is valid for 1 year unless ratified by a subsequent Global DAO vote.
   6. Expert Override: Separately, the Knowledge Council retains its power (per Constitution Article II, Section 3.1a and Charter Article XII, Section 6.d.i Tier C) to invoke its once-per-year Expert Override after a Global DAO vote (even if quorum was met) if it achieves a 7/11 internal supermajority, suspending the DAO's decision. This override is also flagged as non-democratic and triggers constitutional review.
- Appeal: To Member Redress Council on procedural grounds only.

**Tier D: Existential Claims** (claim_materiality = "existential")
- Threshold: Threatens network viability (e.g., "critical smart contract vulnerability")
- Process: Circuit Breaker activated → AG emergency verification (24h) → Global DAO ratifies ex-post (simple majority, 7 days)
- Appeal: None (security trumps process)

##### ii. Default Classification Rules

- If claim_materiality is ambiguous, defaults to **one tier higher** (err on side of caution)
- Knowledge Council may reclassify claims if materiality changes
- All reclassifications require public justification

##### iii. Deadlock Prevention Mechanisms

**Sunset Clause**: If Tier B or C dispute remains unresolved for 90 days:
- Claim automatically tagged "unresolved_timeout"
- Both competing claims remain in DKG marked "disputed_long_term"
- Consuming applications must handle uncertainty (e.g., display confidence intervals)

**Temporal Validity**: All dispute resolutions include expiration date (default: 5 years). Claims may be re-disputed if new evidence emerges.

##### iv. Outcome Recording

- Winning claim: `"status": "affirmed_[tier]_consensus"`
- Losing claim: `"status": "disputed_minority"` (NOT deleted)
- All dissenting opinions preserved in metadata
- Voting records public (unless Member invokes ZK privacy)

##### v. Bond Distribution

- **Challenge succeeds**: Bond returned + 50 FLOW reward from losing party
- **Challenge fails**: Bond split: 50% to defending party, 50% to dispute resolution body treasury
- **Frivolous challenges**: Bond forfeited + reputation penalty

---

## ARTICLE XIII – RATIFICATION AND CONTINUITY

### Section 1. Charter Ratification

This Charter takes full effect upon adoption by the Global DAO and ratification by the federated tiers.

### Section 2. Supremacy Clause

The Mycelix Spore Constitution prevails over this Charter in any case of conflict.

### Section 3. Continuity During Transitions

Outgoing delegates remain in a caretaker capacity for up to 60 days to prevent governance gaps.

---

## ARTICLE XIV – EVALUATION AND CONTINUOUS IMPROVEMENT

### Section 1. Constitutional Review

The Global DAO shall conduct comprehensive reviews of this Charter every two years, informed by the annual Network Resilience Reports.

### Section 2. Experimental Governance

Local DAOs may operate as "Governance Sandboxes" to test experimental mechanisms with Global DAO approval.

### Section 3. Metrics and Transparency

A real-time public dashboard displays key network metrics, and the Global DAO publishes an annual "State of the Network" report.

---

## ARTICLE XV – DISSOLUTION AND SUCCESSION

### Section 1. DAO-Level Dissolution

Procedures are defined for both voluntary and involuntary dissolution of DAOs at all tiers.

### Section 2. Charter Sunset

This Charter sunsets automatically if the Constitution is fundamentally restructured or the Network dissolves.

---

## APPENDIX STATUS CLARIFICATION
Status of Appendices:

 - Appendices A (Definitions), F (Epistemic Claim Schema Standard), and G (Commons Mechanism Registry) are considered integral and binding components of this Charter. Amendments to these appendices require the same ratification process as amendments to the Charter articles themselves.

 - Appendices B (Governance Workflow Diagrams), C (Template Documents), D (Foundational Aspirations), and H (Implementation Roadmap) are non-binding, illustrative, and operational guides. They provide context and best practices but are not legally enforceable parts of the Charter. These appendices may be updated and maintained by the Knowledge Council or Foundation through standard operational procedures without requiring a formal Charter amendment process.

## APPENDIX A – DEFINITIONS

**Caretaker Capacity**: A temporary extension of a delegate's term with limited powers.

**Constituent DAO**: A lower-tier DAO that is part of a higher-tier DAO's membership.

**Sandbox Zone**: An experimental governance area for testing innovations.

**Sunset Clause**: A provision causing a policy to automatically expire unless renewed.

**Computational Trust Quotient (CTQ)**: Integrity score for Instrumental Actors based on uptime, audit logs, and deterministic output validity.

**Synthetic Reputation Index (SRI)**: Alternative term for CTQ, emphasizing distinction from human reputation.

---

## APPENDIX B – GOVERNANCE WORKFLOW DIAGRAMS

*(To be developed: Visual flowcharts for key governance processes, including the Epistemic Claim Lifecycle, Dispute Resolution Tiers, and Emergency Response Procedures.)*

---

## APPENDIX C – TEMPLATE DOCUMENTS

*(To be developed: Template charters for Local DAOs, Fusion/Fission, and Civic Uplift Trusts; proposal templates; and financial reports.)*

### Cultural Charter Template (Optional)

Local DAOs may include a cultural section in their charters:

```markdown
## Cultural Identity (Optional)

### Our Values
[Describe the philosophical or cultural framework guiding this DAO]

### Economic Primitives
We use the following cultural names for Network primitives:
- CIV (Civic Standing): [Cultural name, e.g., "STONE"]
- CGC (Civic Gifting Credit): [Cultural name, e.g., "EMBER"]
- [Additional commons modules if adopted]

### Rituals and Practices
[Describe any regular ceremonies, decision-making rituals, or cultural practices]

### Constitutional Alignment
All cultural expressions align with Core Principles: [List relevant principles]
```

---

## APPENDIX D – FOUNDATIONAL ASPIRATIONS (Non-Binding)

This Constitution was designed to foster a network embodying principles such as Resonant Coherence, Pan-Sentient Flourishing, Integral Wisdom Cultivation, Infinite Play, Universal Interconnectedness, Sacred Reciprocity, and Evolutionary Progression. This appendix serves to provide cultural context and inspiration, derived from foundational philosophical texts like the Luminous Library, without imposing any dogmatic or legal requirements on Members.

---

## APPENDIX E – (RESERVED)

---

## APPENDIX F – EPISTEMIC CLAIM SCHEMA STANDARD v1.1

This appendix defines the formal, machine-readable schema for all governance-relevant epistemic inputs. The Knowledge Council is the steward of this schema, and major updates must be ratified via MIP.

```json
{
  "epistemic_protocol_version": "1.1",
  "claim_id": "uuid",
  "claim_hash": "sha256_hash_of_content",
  "claim_materiality": "routine | significant | constitutional | existential",
  "related_claims": [
    {
      "relationship_type": "supports | refutes | clarifies | builds_upon | is_derived_from",
      "related_claim_id": "uuid"
    }
  ],
  "submitted_by": "member_did_or_dao_address",
  "submitter_type": "human_member | instrumental_actor | dao_collective",
  "timestamp": "ISO_8601_UTC",
  "epistemic_tier": "0_Null | 1_Testimonial | 2_EncryptedVerified | 3_CryptographicallyProven | 4_PubliclyReproducible",
  "claim_type": "assertion | observation | report | vote | metric | judgment | dispute | proposal",
  "criticality": "low | medium | high | foundational",
  "content": {
    "format": "markdown | plaintext | json | link",
    "body": "string or nested object",
    "attachments": ["ipfs_hash_or_other_uri"]
  },
  "verifiability": {
    "method": "none | testimonial | encrypted | zk-proof | public-source",
    "status": "unsubmitted | pending_verification | verified | disputed | resolved_dispute_affirmed | resolved_dispute_invalidated | retracted",
    "audited_by": "audit_guild_address_or_null",
    "confidence_score": "float_0_to_1",
    "confidence_justification": "string_or_link_to_methodology"
  },
  "provenance": {
    "source_type": "member | device | AI | external_system",
    "source_id": "uuid_or_url",
    "trust_tags": ["tag_registry_uri"],
    "chain_of_custody": ["did_or_address"]
  },
  "temporal_validity": {
    "valid_from": "ISO_date",
    "expires": "ISO_date_or_null",
    "revoked": "boolean"
  },
  "governance_scope": {
    "impact_level": "local | sector | regional | global",
    "binding_power": "non_binding_opinion | advisory_recommendation | binding_on_submitter | binding_on_local_dao | binding_on_sector | binding_on_network | constitutionally_binding",
    "rights_impacting": "boolean"
  },
  "epistemic_disputes": [
    {
      "disputed_by": "member_or_guild_address",
      "reason": "string",
      "status": "open | resolved",
      "resolution": "null_or_summary"
    }
  ]
}
```

**Key Updates in v1.1**:
- Added `claim_materiality` field for tiered dispute resolution
- Added `submitter_type` to distinguish human vs AI submissions
- Enhanced `provenance` tracking for chain of custody

---

## APPENDIX G – COMMONS MECHANISM REGISTRY

This appendix catalogs optional economic modules that DAOs may activate via MIPs.

### Constitutionally Recognized Core Primitives

| Name | Code | Type | Function | Symbol |
|------|------|------|----------|--------|
| **Civic Standing** | `CIV` | Non-transferable | Reputation, governance weight | 🏛️ |
| **Civic Gifting Credit** | `CGC` | Non-transferable | Social signal, gratitude | ✨ |
| **Utility Token** | `FLOW` | Transferable (optional) | Fees, staking, compute | 💧 |

### Charter-Enabled Optional Commons Modules

DAOs may elect to activate additional mechanisms via approved MIPs and Eco-OS templates:

| Name | Symbol | Type | Function | Best Fit | Symbol |
|------|--------|------|----------|----------|--------|
| **Time Exchange** | `TEND` | Mutual Credit | Skill/service reciprocity | Local DAO/CUT | 🤲 |
| **Stewardship Credits** | `ROOT`, `SEED` | SBT/NFT | Bioregeneration, governance labor | Sector/Liminal | 🌱 |
| **Signal Pools** | `BEACON`, `WIND` | Soft signal | Non-binding prioritization | Any tier | 🧭 |
| **Commons Hearths** | `HEARTH`, `WELL` | Pool/Trust | Custodial resource governance | Local/Global | 🔥 |

### Cultural Naming Convention

Each primitive supports cultural aliases enabling local expression while maintaining interoperability:

- **CGC aliases**: SPARK, EMBER, PETAL, LIGHT, GRATITUDE
- **CIV aliases**: STONE, FOUNDATION, STANDING
- **ROOT aliases**: SEED, FROND, GROWTH
- **HEARTH aliases**: WELL, CAMPFIRE, COMMONS

**Governance**: The Knowledge Council maintains the official Symbol Registry. DAOs register aliases via MYC-C proposals.

### Activation Process

To activate an optional commons module:

1. **Draft MIP** (Technical or Cultural category)
2. **Define Standards**: Technical spec, interoperability requirements, governance rules
3. **Knowledge Council Review**: Ensure compatibility with core infrastructure
4. **Audit Guild Review**: Security and economic impact assessment
5. **Global DAO Vote**: Simple majority for modules, ⅔ for modifications to core primitives
6. **Implementation**: Deploy via standardized hApp template or smart contract

### Example: Activating Time Exchange (TEND)

```markdown
MYC-C-042: Time Exchange Module for Local DAOs

**Summary**: Enable Local DAOs to implement mutual credit time banking.

**Technical Spec**: 
- hApp template: `mycelix-tend-v1.0.happ`
- Ledger: Holochain DHT (local to each DAO)
- Unit: 1 TEND = 1 hour of service
- Issuance: Members earn TEND by providing services
- Redemption: Members spend TEND to receive services
- Balance limits: ±40 TEND (prevents excessive debt/credit)

**Interoperability**: 
- TEND balances queryable via standard DKG API
- Optional: TEND may influence local CIV calculations (max 5% weight)

**Cultural Layer**: DAOs may rename TEND (e.g., "CARE", "HOURS")

**Security**: Audit Guild reviewed (no systemic risk)

**Vote**: Simple majority required
```

---

## APPENDIX H – IMPLEMENTATION ROADMAP

### Pre-Testnet (Q2 2025)
- [ ] Draft v0.22 incorporating all Phase 1+2 changes
- [ ] Community review period (30 days)
- [ ] Global DAO vote on v0.22 (⅔ majority + tier ratification)
- [ ] Publish v0.22 as canonical version

### Pre-Mainnet (Q4 2025)
- [ ] Swiss Foundation incorporation complete
- [ ] Validator cartel detection algorithms deployed
- [ ] Reputation growth mechanics tested on testnet (1,000+ Members)
- [ ] Epistemic dispute resolution tested (10+ test cases)
- [ ] Bridge security audits complete (if launching Phase 1 bridge)
- [ ] MYC-E-001 (Validator Economics) ratified
- [ ] All oversight bodies appointed and operational

### Post-Mainnet (2026+)
- [ ] Governance engagement metrics monitored (quarterly reports)
- [ ] Bridge insurance policies obtained (Phase 2 prep)
- [ ] First constitutional review (2-year cycle)
- [ ] Commons modules activated (TEND, ROOT, BEACON)
- [ ] Cultural layer resources expanded
- [ ] Pre-DAO incubators launched for underserved regions

---

**(Version History: v0.22 – Integrated critical system completion fixes including reputation growth protocol, tiered epistemic dispute resolution, anti-cartel mechanisms, governance engagement recovery, and comprehensive bridge security framework. Added economic pluralism with CIV/CGC/FLOW primitives and optional commons modules (TEND, ROOT, BEACON, HEARTH). Established phased VC issuance, graduated deficit protocol, and multi-stablecoin treasury strategy. Enhanced cultural layer infrastructure with Symbol Registry and ritual templates. Formalized Pre-DAO Incubator pathway and DRP requirements. Updated Epistemic Claim Schema to v1.1 with materiality classification.)**
