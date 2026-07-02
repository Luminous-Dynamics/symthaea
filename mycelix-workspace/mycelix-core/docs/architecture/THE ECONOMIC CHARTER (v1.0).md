# **THE ECONOMIC CHARTER (v1.1)**

*Updated March 2026: CIV renamed MYCEL (soulbound reputation), FLOW renamed SAP (circulation medium with demurrage) — aligned with production implementation*

**Companion instrument to the Mycelix Spore Constitution (v0.24) and part of the modular Mycelix Charter Set (v1.0).**

***Editor's Note:** This Economic Charter (v1.0) is a refactored module of the original THE FEDERATED DAO HIERARCHY CHARTER (v0.24). It isolates all articles pertaining to financial governance, treasury management, bridge security, validator economics, and reputation/identity mechanics (formerly Articles X and XI). All information is sourced directly from the v0.24 Charter.*

## **ARTICLE I – FINANCIAL GOVERNANCE**

### **Section 1\. Budgeting Process**

Each DAO tier prepares annual budgets, which are consolidated and approved by the Global DAO.

### **Section 2\. Funding Sources and Allocation**

#### **2.1 Protocol Revenue Distribution**

Protocol revenues are distributed according to a fixed formula:

* 12% to oversight (KC 3%, AG 5%, MRC 4%)  
* 20% to Global DAO  
* 30% to Sector DAOs  
* 20% to Regional DAOs  
* 13% to Foundation reserve  
* 5% to emergency reserve

#### **2.2 Tier 2 Pass-Through Funding Mandate**

To ensure Subsidiarity and empower grassroots activity, all Sector DAOs and Regional DAOs must programmatically pass through a minimum of 40% of their allocated protocol revenue to their constituent Local DAOs and recognized Civic Uplift Trusts.

Distribution shall be proportional to **active, verified participation**, calculated using a weighted formula:

Funding\_Share \= (Voting\_Activity × 0.4) \+ (Proposal\_Quality × 0.3) \+ (Member\_Count × 0.2) \+ (Civic\_Uplift\_Impact × 0.1)

The precise metrics and weightings shall be defined and maintained via a MIP (Economic).

**Anti-Gaming Provisions**:

* Local DAOs with fewer than 5 active Members or fewer than 2 proposals per quarter shall receive a minimum viable allocation only.  
* The Audit Guild shall monitor for Sybil patterns, such as multiple Local DAOs controlled by the same actors.  
* Circular funding patterns (A→B→C→A) trigger fraud review.

### **Section 3\. Financial Controls**

Expenditure limits and multi-signature requirements are in place for all DAO treasuries to prevent misuse of funds.

### **Section 4\. Long-Term Fiscal Sustainability**

#### **a. Reserve Requirements**

The Network shall maintain minimum reserves:

* **Foundation**: 18 months of operating expenses (Constitution Article II, 2.1.d)  
* **Each Oversight Body**: Maximum 24 months of operating expenses  
* **Emergency Reserve**: 6 months network-wide operations

#### **b. Revenue Stress Testing**

The Audit Guild shall conduct and publish an annual scenario analysis modeling the impact of severe revenue shocks:

* 50% revenue drop  
* 90% reduction in new Members  
* Major bridge exploit requiring insurance payout

Results shall inform reserve target adjustments via MIP.

#### **c. Graduated Deficit Protocol**

If protocol revenue falls below certain thresholds, graduated responses are triggered:

**Warning State** (Revenue \<150K SAP for 1 quarter):

* Monitoring only  
* No action required  
* Alert published to Global DAO

**Conservation State** (Revenue \<100K SAP for 2 quarters):

* 10% discretionary budget cut  
* Hiring freeze for non-critical roles  
* Mandatory efficiency audit by Audit Guild

**Emergency State** (Revenue \<50K SAP for 3 quarters):

* 25% budget cut (including oversight, down to 9% from 12%)  
* Foundation reserve fund activated  
* Global DAO votes on austerity plan

**Existential State** (Revenue \<25K SAP for 4 quarters):

* Orderly wind-down plan activated  
* Member assets protected (Constitution Article X, Section 2.2)  
* Dissolution procedures initiated

### **Section 5\. Bridge Security and Recovery**

#### **5.1 Bridge Insurance Requirements**

**Mandatory Coverage**: The Foundation shall maintain bridge exploit insurance coverage equal to:

* **Minimum**: 50% of Total Value Locked (TVL) on all bridges  
* **Target**: 100% of TVL  
* **Coverage Types**: Smart contract exploit insurance, validator collusion insurance, oracle manipulation insurance

**Funding Source**: 2% of protocol revenue automatically directed to insurance premium fund. If premiums exceed 2%, Foundation reserve covers difference.

**Self-Insurance Reserve**: If commercial insurance unavailable, Foundation maintains **Bridge Recovery Fund** \= 25% of TVL (minimum), held in multi-sig (5-of-9: 3 Foundation Directors, 3 Audit Guild, 3 Knowledge Council).

#### **5.2 Tiered Bridge Deployment**

Bridges deployed in stages with TVL caps:

| Phase | Bridge Type | Max TVL | Security Requirements |
| :---- | :---- | :---- | :---- |
| **Alpha** | Testnet | $0 (test tokens) | Internal audit only |
| **Beta** | Mainnet Limited | $100K | 2 external audits \+ 90-day testnet |
| **Gamma** | Mainnet Standard | $1M | 3 external audits \+ bug bounty \+ 180-day beta |
| **Production** | Mainnet Full | $10M | 4+ audits \+ formal verification \+ 1-year gamma |
| **Unlimited** | Mainnet Uncapped | Unlimited | 2+ years incident-free operation |

**TVL Cap Enforcement**: Smart contract enforces cap (rejects deposits above limit). To raise cap: Requires MIP-T proposal \+ new audit.

#### **5.3 Continuous Security Monitoring**

Audit Guild operates 24/7 bridge monitoring:

* **Automated**: On-chain monitoring (unusual transaction patterns)  
* **Manual**: Weekly validator behavior review  
* **External**: Quarterly penetration testing by whitehat hackers

**Alert Thresholds**:

* 🟡 Yellow: Unusual activity detected (e.g., large withdrawal spike) \- Audit Guild reviews within 1 hour  
* 🔴 Red: Critical anomaly (e.g., validator collusion evidence) \- Technical Circuit Breaker activated immediately

#### **5.4 Post-Exploit Recovery Protocol**

**Phase 1: Emergency Response (0-72 hours)**

* Hour 0-2: Detection and containment  
* Hour 2-12: Forensics (determine exploit vector, quantify losses)  
* Hour 12-24: Communication (publish incident report)  
* Day 2-3: Patch development and emergency audit

**Phase 2: Recovery Execution (Days 3-30)**

**Funding Recovery** (Priority order):

1. Insurance Claims (commercial insurer)  
2. Bridge Recovery Fund (self-insurance reserve)  
3. Foundation Reserve (emergency backstop, up to 50%)  
4. Treasury (last resort, requires ⅔ Global DAO vote)

**User Reimbursement Priorities**:

* **Tier 1 (Immediate)**: Small users (\<$1,000 affected) \- 100% reimbursement within 7 days  
* **Tier 2 (Fast)**: Medium users ($1K-$100K) \- 100% reimbursement within 30 days  
* **Tier 3 (Delayed)**: Large users (\>$100K) \- Staged reimbursement over 90 days

**Phase 3: Prevention (Days 30-90)**

* Root cause analysis published  
* System improvements implemented  
* Policy changes via MIP-T proposal  
* Personnel actions if exploit due to negligence

#### **5.5 Catastrophic Failure Modes**

If bridge exploit **exceeds available recovery funding**:

**Option A: Socialized Losses** (Requires ⅔ Global DAO vote)

* Mint new tokens to cover shortfall (dilution)  
* Pro-rata distribution to affected users  
* Maximum socialized loss: 5% of total supply (prevents hyperinflation)  
* Requires forensic proof exploit was not due to Foundation negligence  
* One-time event only

**Option B: Haircut Recovery**

* Affected users receive partial reimbursement (e.g., 70 cents per dollar)  
* Remaining losses written off  
* Users may sue Foundation in Swiss courts

**Option C: Graceful Shutdown**

* Activate Constitution Article X, Section 2 (Network Dissolution)  
* Orderly wind-down of operations  
* Remaining treasury distributed to Members

#### **5.6 Validator Accountability**

**Validator Slashing for Bridge Exploits**: If exploit caused by validator negligence or collusion:

* Guilty validators: **100% reputation slash** (ejected permanently)  
* Guilty validators: **100% stake slash** (e.g., 5,000 SAP bonded amount)  
* Slashed funds go to bridge recovery fund

**Proof Requirements**: Audit Guild must provide cryptographic proof of validator fault (e.g., signed invalid Merkle proof, participated in 51% attack).

**Due Process**: Accused validators may appeal to Member Redress Council within 7 days. MRC conducts expedited review (decision within 14 days).

**Validator Insurance Requirements**: To serve as bridge validator, must carry professional liability insurance (minimum $1M coverage) OR post additional collateral (10,000 SAP minimum).

#### **5.7 Multi-Bridge Redundancy**

**Bridge Diversification Strategy**:

* Phase 1: Merkle \+ RB-BFT validators (single bridge type)  
* Phase 2: Add ZK-STARK bridge (different security model)  
* Phase 3: Add IBC bridge (Cosmos ecosystem)  
* Phase 4: Add XCMP bridge (Polkadot)

**TVL Limits Per Bridge**: No single bridge holds \>50% of total TVL. Enforced via smart contract deposit caps.

### **Section 6\. Validator Economics (MIP-E-001)**

Validator node operator incentives, hardware requirements, and economic models shall be defined via **MIP-E-001 (Validator Economics Framework)** proposal prior to Phase 1 mainnet launch.

**Timeline**:

* **Q2 2025**: MIP-E-001 drafted by working group (Foundation \+ Audit Guild \+ 5 community members)  
* **Q3 2025**: Community review period (30 days minimum)  
* **Q4 2025**: Ratification by Global DAO (⅔ supermajority required for economics) **BEFORE mainnet launch**

**MIP-E-001 Required Specifications**:

The proposal must address the following components or mainnet launch will be delayed:

**1\. Hardware Requirements**:

* Minimum CPU (e.g., 4 cores @ 2.5 GHz)  
* Minimum RAM (e.g., 16 GB)  
* Minimum storage (e.g., 500 GB SSD)  
* Minimum bandwidth (e.g., 100 Mbps up/down)  
* Recommended redundancy (backup power, internet)

**2\. Compensation Structure**:

* Base rewards (e.g., X SAP per epoch for participation)  
* Performance bonuses (e.g., \+Y% for 100% uptime, \+Z% for perfect validation accuracy)  
* Geographic diversity bonuses (e.g., 2x multiplier for underrepresented regions \<10% of validator set)

**3\. Collateral and Staking**:

* Minimum stake: 5,000 SAP (confirmed)  
* Staking period: Minimum lock-up duration (e.g., 90 days)  
* Unstaking cooldown: Period before stake can be withdrawn after exit (e.g., 14 days)  
* Stake utility: Used for slashing in case of misbehavior

**4\. Slashing Conditions**:

* **Partial Slashing (10% of stake)**:  
  * Extended downtime: \>48 consecutive hours offline  
  * Repeated timeouts: \>5 validation timeouts in a 30-day period  
* **Substantial Slashing (50% of stake)**:  
  * Incorrect validation: Signing invalid validation result  
  * Performance below threshold: \<90% accuracy over 90-day period  
* **Full Slashing (100% of stake \+ reputation reset)**:  
  * Malicious behavior: Proven cartel participation, signing fraudulent bridge proofs  
  * Double-signing: Validating conflicting states simultaneously  
  * Critical security violation: Deliberately compromising network security  
* **MATL-Informed Slashing**: Slashing multipliers may be adjusted based on MATL Composite Trust Score (per Article II, Section 2.b of this document).

**5\. Uptime and Performance Requirements**:

* Minimum uptime: 95% (measured over 30-day rolling window)  
* Maximum acceptable downtime: 36 hours per month  
* Validation accuracy: 95% minimum (incorrect validations \<5%)  
* Response time: Must respond to validation requests within 30 seconds

**6\. Geographic Diversity Incentives**:

* **Underrepresented regions** (\<10% of validator set): 2x reward multiplier  
* **Moderately represented regions** (10-20%): 1.5x multiplier  
* **Well-represented regions** (20-30%): 1x multiplier (standard)  
* **Over-represented regions** (\>30%): 0.5x multiplier (penalty to encourage diversity)

Regions defined by: North America, South America, Europe, Africa, Middle East, Asia-Pacific, Oceania.

**7\. Validator Onboarding Process**:

* Registration: Submit DID, hardware specs, stake collateral  
* Probationary period: 30 days at 50% reward rate (prove reliability)  
* Full activation: After 30 days with \>95% uptime and \>95% accuracy  
* Mentorship: New validators paired with experienced validators for first 90 days

**8\. Validator Exit Process**:

* Voluntary exit: 7-day notice period, stake returned after 14-day cooldown  
* Involuntary exit (ejection): Immediate for full slashing events, stake forfeited  
* Graceful degradation: Validators may enter "maintenance mode" (up to 72 hours/month) with advance notice, no slashing

**9\. Economic Sustainability Model**:

* Revenue sources: Bridge fees (estimated X% of TVL annually), transaction fees (if SAP adopted)  
* Validator pool allocation: Y% of protocol revenue distributed to validator pool  
* Reserve buffer: Z% of validator rewards held in reserve for low-revenue periods

**10\. Anti-Centralization Measures**:

* Maximum validators per entity: 3 (enforced via reputation linkage analysis)  
* Maximum stake per entity: 5% of total staked (prevents whale dominance)  
* Delegation prohibited: Validators must self-custody infrastructure (no validator-as-a-service initially)

**Ratification Requirement**: MIP-E-001 must be ratified by **September 30, 2025**.

## **ARTICLE II – REPUTATION & IDENTITY ECONOMICS**

### **Section 1\. Phased Verifiable Credential (VC) Issuance**

**Phase 1 (Genesis, 0-12 months)**: VCs issued primarily by core institutions:

* Provisional Knowledge Council, Audit Guild, Member Redress Council  
* Mycelix Foundation  
* Recognized external partners (security auditors, identity verifiers like Gitcoin Passport)

**Types**: VC:GenesisSteward, VC:CompletedProtocolOnboarding, VC:VerifiedHumanity, VC:ContributedToCoreCodebase, VC:ServedOnProvisionalCouncil

**Phase 2 (12-24 months)**: Sector and Regional DAOs empowered to issue domain-specific VCs:

* Must register VC schemas with Knowledge Council  
* Subject to Audit Guild review  
* Examples: VC:AISafetyCertification (AI Sector DAO), VC:GDPRComplianceExpert (EU Regional DAO)

**Phase 3 (24+ months)**: Local DAOs and peer-to-peer issuance:

* Local DAOs issue VCs for local contributions once they demonstrate stability  
* Individual Members above MYCEL threshold may issue peer attestations  
* Lower-weight in global MYCEL calculations initially  
* Examples: VC:LedCommunityProjectX, VC:HearthMentor

### **Section 2\. Participation Incentives & Reputation Dynamics**

#### **a. Participation Incentives**

Members earn reputation bonuses for consistent voting and submitting high-quality proposals:

* Vote on \>50% of proposals in your domain: \+0.02 MYCEL per quarter  
* Vote on \>80% of proposals: \+0.05 MYCEL per quarter  
* Perfect attendance (100% of votes): \+0.1 MYCEL per quarter \+ "Engaged Citizen" badge

Delegates face removal for poor attendance (\<90% participation rate).

**Delegate Recognition Program**:

* Delegate rewards: 0.5 SAP per delegator per month (if SAP adopted)  
* Max 100 delegators per delegate (prevents over-concentration)  
* Delegates must vote on \>90% of proposals (auto-removal if below)

**Proposal Quality Incentives**:

* Proposals passing with \>75% approval: Sponsor receives 500 SAP reward  
* Proposals failing badly (\<25% approval): Sponsor loses proposal bond  
* Encourages high-quality, consensus-building proposals

#### **b. Reputation Decay**

A Member's reputation shall decay according to a multi-factor model:

##### **i. Passive Decay**

A 5% linear decay per year is applied to reputation for Members who do not actively participate in governance (voting, proposing, serving as delegate) or receive new reputation allocations within that year.

##### **ii. Performance Decay (MATL-Informed)**

Immediate, event-triggered decay is applied for validation failures, informed by the MATL Composite Trust Score (per the Epistemic Charter):

* **Incorrect validation**: A base reputation slash of **50%** (\* 0.5) is applied.  
* **MATL Multiplier**: This base slash is adjusted by a multiplier based on the Member's MATL score:  
  * **Suspected Sybil (Score \< 0.3)**: Multiplier is **1.0x**. The Member receives the *full 50% slash*.  
  * **Honest Mistake (Score \>= 0.3)**: Multiplier is 0.5 \+ (1.0 \- composite\_score) \* 0.5. This *reduces the penalty* for high-reputation Members who make an honest mistake. (e.g., a score of 0.9 results in a 0.5 \+ (0.1\*0.5) \= 0.55 multiplier, for a total slash of 50% \* 0.55 \= 27.5%).  
* **Timeout**: 2% penalty per timeout incident.

##### **iii. Decay Floor**

Reputation cannot decay below a baseline of **0.05** (maintains baseline humanity recognition).

##### **iv. Recovery Path**

Members whose reputation falls below 0.4 may enter **"Apprentice Mode"** to rebuild their standing through co-validation with high-reputation nodes. Parameters defined by MIP.

#### **c. Reputation Growth (MATL-Gated)**

A Member's reputation may increase through verified positive contributions, gated by the **Mycelix Adaptive Trust Layer (MATL)** Composite Trust Score (per the Epistemic Charter).

##### **i. Validation Performance (Primary Growth)**

* **Correct Validation**: A base growth of **\+0.05 reputation** is calculated per successful validation round.  
* **MATL Multiplier**: This base growth is multiplied by a MATL quality multiplier:  
  * **High Trust (Score ≥ 0.8)**: Multiplier is **1.0x**. Member receives *full \+0.05 growth*.  
  * **Medium Trust (Score 0.4 \- 0.8)**: Multiplier is **proportional** (e.g., a score of 0.6 results in a 0.6x multiplier for \+0.03 growth).  
  * **Low Trust / Suspected Sybil (Score \< 0.4)**: Multiplier is **0.0x**. Reputation growth is *frozen* until the Member improves their behavioral score.  
* Reputation is capped at 1.0.  
* **Consistency Bonus**: Members with 100+ consecutive correct validations receive a 1.2x multiplier on their *adjusted* growth rate.

##### **ii. Governance Participation (Secondary Growth)**

* **Proposal Quality**: \+0.02 reputation for proposals that pass with \>75% support  
* **Constructive Dissent**: \+0.01 reputation for dissenting votes on proposals that later fail audit review (rewards vigilance)

##### **iii. Civic Contributions (Tertiary Growth)**

* **Mentorship**: \+0.01 reputation per successfully onboarded Member (up to 10/year)  
* **Accessibility Work**: \+0.02 reputation for contributions to accessibility tooling (verified by Knowledge Council)

##### **iv. Growth Rate Limits (Anti-Gaming)**

* Maximum reputation increase: **0.1 per month** (prevents sudden dominance).  
* Growth above 0.8 requires **additional verification**:  
  * Peer review by 3 randomly selected high-rep Members  
  * Knowledge Council spot audit (5% sample rate)  
* Genesis Cohort members subject to 50% slower growth rate (0.05/month max).

#### **d. Genesis Bootstrapping**

##### **i. First 1,000 Members**

The first 1,000 Members of the Network shall receive a starting reputation of **0.5** (vs 0.1 default).

##### **ii. Genesis Cohort**

Genesis Cohort members shall start at **0.7** but remain subject to the constitutional voting power cap (Constitution Article V, Section 3: max 0.5% per vote) and slower growth rate.

##### **iii. Transition Threshold**

After the Network reaches 10,000 total Members, all new entrants shall start at the default reputation of **0.1**.

## **ARTICLE III – RATIFICATION AND CONTINUITY**

### **Section 1\. Charter Ratification**

This Charter takes full effect upon adoption by the Global DAO and ratification by the federated tiers.

### **Section 2\. Supremacy Clause**

The Mycelix Spore Constitution prevails over this Charter in any case of conflict.

## **ARTICLE IV – EVALUATION AND CONTINUOUS IMPROVEMENT**

### **Section 1\. Constitutional Review**

The Global DAO shall conduct comprehensive reviews of this Charter every two years, informed by the annual Network Resilience Reports.

## **APPENDIX A – DEFINITIONS**

*(Editor's Note: This appendix includes only definitions from the original Charter relevant to the Economic Charter.)*

**Sunset Clause**: A provision causing a policy to automatically expire unless renewed.

