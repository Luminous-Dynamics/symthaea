# **THE MYCELIX SPORE CONSTITUTION (v0.25)**

**Drafted for the Mycelix Network – December 2025**
**Updated March 2026: CIV renamed MYCEL, FLOW renamed SAP, CGC absorbed into MYCEL peer recognition — aligned with production implementation**

## **Plain English Summary (Non-Binding)**

**This summary provides an accessible overview. In any conflict, the full text shall prevail.**

**Our Goal: To build a decentralized, discoverable, and cooperative network for verifiable knowledge that operates as a public good.**

**Core Rules: Everything must be verifiable, transparent, fair, and interoperable. Decisions should be local. The network must be accessible and adaptable.**

**Who's in Charge: The Mycelix DAO, composed of all verifiably human Members, is the ultimate authority. It operates through a polycentric ecosystem of parallel Sector DAOs (by topic) and Regional DAOs (by geography), as well as other emergent structures. The Global DAO acts as a two-house legislature requiring agreement from both Sector and Regional representatives.**

**Oversight: A legal Foundation handles real-world matters. Three independent, paid oversight bodies provide checks and balances: the Knowledge Council (data integrity, network resilience), Audit Guild (code and justice verification), and Member Redress Council (individual rights and systemic harms).**

**Emergency Brakes: A small technical multisig can pause the network temporarily (72 hours max). The broader DAO can pause governance for longer crises (30 days max) and activate adaptive delegation protocols.**

**Your Rights: Control over data attribution, access to reputation scores, fair appeals with real remedies, privacy, right to data portability, accessibility accommodations, and protection from coercion.**

**How We Grow: Changes are made via Mycelix Improvement Proposals (MIPs). Constitutional amendments require supermajorities. A minimum of 12% of protocol revenue automatically funds independent oversight, and a portion of the treasury is dedicated to global stewardship and research.**

## **PREAMBLE**

**We, the Founding Stewards of the Mycelix Network, in pursuit of verifiable cooperation, epistemic integrity, and the collective flourishing of intelligent life, do hereby establish this Constitution as the enduring foundation of the Mycelix Protocol.**

**This Constitution defines the meta-law of governance—its limits, legitimacy, and evolution—and entrusts its custodianship to the Mycelix DAO and Foundation acting in reciprocity. As a public good, the Network commits to open-source principles, discoverability, and interoperability, ensuring all core components are freely accessible and modifiable.**

**The Mycelix Network, while operating under this secular and universally applicable Constitution, recognizes and honors the diverse cultural, spiritual, and philosophical traditions that may inform its Members' participation. The Network is committed to providing a neutral substrate upon which communities may build and express varied systems of meaning and value, provided they remain consistent with the Core Principles and Member Rights defined herein.**

## **ARTICLE I – PRINCIPLES AND PURPOSE**

### **Section 1\. Core Purpose**

**The Mycelix Network exists to enable sovereign, verifiable, and cooperative intelligence across all domains of knowledge and production.**

### **Section 2\. Core Principles**

**All governance, computation, and coordination under Mycelix shall adhere to:**

**Epistemic Verifiability: All assertions, decisions, and data that materially affect constitutional rights, resource allocation greater than 100,000 SAP, or Core Principles must be epistemically categorized and subject to a standard of verifiability appropriate to their type. Routine operational decisions may use simplified attestation. The Knowledge Council shall maintain and publish guidelines distinguishing "material" from "routine" claims. The Network affirms a layered model of epistemic governance (Publicly Reproducible, Cryptographically Proven, Privately Verifiable, Testimonial). All significant epistemic inputs influencing governance must be clearly labeled with their type, provenance, and auditability status. No binding governance action may be taken solely on the basis of epistemically unclassified or Tier 0 ('Null') claims.**

**Verifiable Computation: All reputation calculations, validator selections, and governance aggregations that materially affect Member rights or resource allocation shall be accompanied by zero-knowledge proofs of correct execution. The Network employs zk-STARK proofs to ensure algorithmic transparency without compromising privacy.**

**Epistemic Humility: The Network recognizes that knowledge is provisional and context-dependent. Dispute resolution mechanisms shall favor transparency over certainty, preserving dissenting views and enabling re-evaluation as understanding evolves.**

**Reciprocity and Fair Exchange: Economic models shall incentivize verifiable contributions, achieve allocative efficiency, and disincentivize rent-seeking. The Network shall support plural forms of economic coordination, including gift economies, mutual credit, time exchange, stewardship recognition, and market mechanisms, allowing communities to choose models aligned with their values.**

**Subsidiarity: Decisions must be made at the lowest competent tier.**

**Transparency: Governance processes, data provenance, and decision rationales must remain publicly auditable.**

**Adaptability: Protocol and governance must be designed to evolve and learn in response to new information and changing conditions, embracing flexibility and iterative refinement.**

**Interoperability: Protocol components shall be modular and composable. To this end, the Network shall adhere to the following standards:**

* **API Standards: All public APIs must support JSON-RPC 2.0 and GraphQL.**  
* **State Exports: State exports shall use standardized formats (e.g., JSON-LD for DKG, Protobuf for DHT).**  
* **Bridge Contracts: Bridge contracts shall strive to implement widely adopted standards such as IBC (Inter-Blockchain Communication) and XCMP (Cross-Consensus Message Format) where applicable.**  
* **Deprecation Policy: Breaking changes to core APIs or standards require a minimum 6-month deprecation notice and the provision of migration tooling where feasible.**  
* **Governance: The Knowledge Council shall maintain the Mycelix Interoperability Standards (MIS) registry. Updates shall be made via MIP-T (Technical) proposals.**

**Discoverability: All public data and functions shall be published with standardized metadata.**

**Sybil Resistance: The Network shall maintain verifiable uniqueness of human Members through its designated protocol, protecting against identity manipulation.**

**Initial Sybil Resistance Protocol: The Network shall use Gitcoin Passport (v2.0 or higher) with a minimum threshold of 20 Humanity Score points for Member verification during Phase 1\. Alternative protocols (BrightID, Worldcoin, or hybrid approaches) may be adopted via Constitutional Amendment following independent security audits by at least 2 firms and Global DAO ratification.**

**Polycentricity: The Network shall be composed of multiple, semi-autonomous, and overlapping centers of decision-making that compete and cooperate to foster resilience and prevent single points of failure.**

**Civic Equity: The Network shall actively work to include and empower historically underserved populations, ensuring that access is not predicated on digital literacy, formal education, or crypto-native experience.**

**Validator Diversity: The Network shall maintain geographic, sectoral, and social diversity among validators to prevent concentration of validation power. No single region, sector, or social cluster may comprise more than 30% of active validators.**

**Algorithmic Transparency: All algorithms, machine learning models, or AI agents used within the Network to significantly influence governance, resource allocation, reputation, or access must be publicly documented (architecture, training data provenance), auditable for bias, and explainable in their decision-making processes. Opaque, "black-box" systems are prohibited in high-stakes governance functions.**

**Cultural Pluralism: The Network shall provide infrastructure for diverse cultural expression while maintaining universal governance coherence. Communities may adopt cultural naming conventions, rituals, and value frameworks provided they align with Core Principles.**

### **Section 3\. Sovereignty of the Constitution**

**This document shall be the supreme governing instrument of the Mycelix Network. All subsequent laws, standards, or modules derive authority herefrom.**

## **ARTICLE II – INSTITUTIONS AND AUTHORITIES**

### **Section 1\. The Mycelix DAO**

**The Mycelix DAO is the sovereign legislative body, exercising power through a parallel federated structure of Sector DAOs and Regional DAOs, culminating in a bicameral Global DAO, as defined in the Federated DAO Hierarchy Charter.**

### **Section 2\. The Mycelix Foundation**

#### **2.1 Purpose and Powers**

**The Foundation shall act as the legal and operational custodian of the Network Charter. It shall:**

**a. Uphold the Constitution and maintain off-chain legal personality. b. Administer grants, infrastructure funds, and compliance interfaces. c. Exercise the Golden Veto under Article III. d. Maintain a reserve fund of no less than 18 months of operating expenses, drawn from protocol revenues and managed according to conservative investment principles approved by the Global DAO. e. Jointly ensure, with the Global DAO, that the global stewardship and research mandate (Article VIII, Section 4\) is adequately funded and executed. f. Allocate no less than $300,000 annually (inflation-adjusted) to legal compliance and regulatory engagement.**

#### **2.2 Limitations**

**The Foundation shall have no legislative power. It may not create binding regulations, alter Core Principles, or interfere with DAO governance except through the Golden Veto.**

#### **2.3 Governance**

**Foundation Directors shall serve three (3) year terms, staggered to ensure continuity. Directors may be removed for cause by a two-thirds (⅔) supermajority in both houses of the Global DAO. "Cause" includes: gross negligence, fraud, repeated failure to execute DAO directives, or actions demonstrably contrary to the Core Principles.**

#### **2.4 Transparency**

**The Foundation shall publish quarterly financial reports, annual audits (conducted by independent third parties), and justifications for all Golden Veto exercises within 7 days of use.**

#### **2.5 Legal Structure and Jurisdictional Framework**

##### **2.5.1 Primary Foundation Entity**

**The Mycelix Foundation shall be established as a Swiss Foundation (Stiftung) under Swiss Civil Code Articles 80-89, with the following characteristics:**

* **Purpose: Irrevocably dedicated to public benefit (operating the Mycelix Network as public good infrastructure)**  
* **Domicile: Canton of Zug, Switzerland (recognized crypto-friendly jurisdiction)**  
* **Supervisory Authority: Swiss Federal Supervisory Authority for Foundations (ESA)**  
* **Governance: Board of Foundation Directors per Section 2.3**

**Rationale for Swiss Foundation: Strong rule of law and political neutrality; established framework for crypto foundations (Ethereum, Cardano precedent); perpetual existence (cannot be dissolved except by court order); clear separation between foundation assets and director personal assets; GDPR-compliant jurisdiction.**

##### **2.5.2 Subsidiary Entities (Federated Model)**

**To navigate conflicting legal requirements across jurisdictions, the Foundation may establish wholly-owned subsidiary entities in critical regions:**

| Entity | Jurisdiction | Purpose | Governance |
| ----- | ----- | ----- | ----- |
| **Mycelix US Services LLC** | **Delaware, USA** | **US securities/tax compliance; operate US-based validator nodes** | **Wholly owned by Swiss Foundation; board appointed by Foundation Directors** |
| **Mycelix EU gGmbH** | **Berlin, Germany** | **GDPR compliance; EU data operations** | **Same** |
| **Mycelix Singapore Foundation Ltd** | **Singapore** | **Asia-Pacific operations; bridge to Asian markets** | **Same** |

**Key Constraints:**

* **Subsidiaries have zero governance power over the protocol (purely operational)**  
* **All protocol decisions flow from DAO → Swiss Foundation → Subsidiaries**  
* **Subsidiaries may not exercise Golden Veto**  
* **Financial reports consolidated and published by Swiss Foundation**

##### **2.5.3 Jurisdictional Conflict Resolution**

**When legal requirements conflict across jurisdictions:**

**Priority Order (Constitutional Hierarchy):**

1. **Core Principles (Article I, Section 2\) \- Inviolable**  
2. **Member Rights (Article VI) \- Cannot be overridden by local law**  
3. **Swiss Foundation Charter \- Governs Foundation operations**  
4. **Local Compliance \- Subsidiaries comply with local law *within constitutional bounds***

**Conflict Resolution Process:**

1. **Legal conflict identified (e.g., US requires KYC, conflicts with "Privacy by Default")**  
2. **Foundation Directors consult Global DAO (mandatory advisory vote within 14 days)**  
3. **Foundation chooses one of:**  
   * **Comply in Region: Subsidiary implements local law (e.g., KYC for US users only)**  
   * **Exit Region: Cease operations in that jurisdiction (e.g., block geographic access)**  
   * **Legal Challenge: Contest law in local courts (requires DAO funding approval)**  
4. **All decisions published with legal analysis within 7 days**

**Geographic Segmentation (Last Resort):**

* **If compliance impossible without violating Core Principles, Foundation may create regional protocol variants:**  
  * **Example: "Mycelix-US" (KYC-compliant) vs "Mycelix-Global" (privacy-preserving)**  
  * **Both share same DHT infrastructure, but on-chain bridge settlement differs**  
  * **Users choose version based on their risk tolerance**  
* **Requires ⅔ Global DAO approval (constitutional-level decision)**

##### **2.5.4 Securities Law Compliance**

**Token Classification Strategy: The Foundation shall maintain legal opinions from law firms in all major jurisdictions classifying token status.**

**Target Classification: Utility Token / Network Access Token**

* **MYCEL (Soulbound Reputation): Non-transferable reputation, not a security**
* **MYCEL peer recognition categories (SPARK, EMBER, etc.): Non-transferable social signal, not a security**
* **SAP: Circulation medium with demurrage, not investment contract**

**If Classified as Security (Failure Mode):**

* **US: Register as exempt security under Reg D (accredited investors only) or Reg A+ (capped offering)**  
* **EU: Comply with MiFID II prospectus requirements**  
* **Singapore: Capital Markets Services license required**

**Nuclear Option: If global securities compliance becomes impossible:**

* **Foundation transitions to pure governance model (no economic tokens)**  
* **All fee payments shift to wrapped BTC/ETH on bridge**  
* **MYCEL becomes pure voting rights (likely not a security)**

##### **2.5.5 Data Sovereignty and Privacy**

**GDPR Compliance (EU/Switzerland):**

* **All personal data processed under Swiss/EU data protection law**  
* **Users have right to erasure (conflicts with immutable DHT)**  
* **Technical Solution: Personal data stored off-chain in encrypted vaults**  
  * **Only hashes stored on DHT**  
  * **Vault keys held by user (true self-custody)**  
  * **Erasure \= delete key (data becomes irrecoverable)**

**China Cybersecurity Law (Potential Conflict):**

* **Requires all data of Chinese citizens stored in China**  
* **Requires government backdoors for inspection**  
* **Foundation Position: Incompatible with Core Principles**  
* **Resolution: No official Chinese operations; users access via VPN at own risk**

**Whistleblower Protections:**

* **Foundation maintains anonymous reporting channel (Tor hidden service)**  
* **Legal opinion that whistleblower disclosures to regulators do not violate confidentiality**  
* **Foundation will not retaliate against Members who report violations to authorities**

##### **2.5.6 Tax Treatment**

**Foundation Tax Status:**

* **Swiss foundation qualifies as tax-exempt public benefit entity (no corporate tax)**  
* **Must file annual report to ESA proving public benefit activities**  
* **Investment income taxed at preferential rates**

**Member Tax Obligations:**

* **Foundation disclaims tax advice (Members consult own advisors)**  
* **Foundation publishes general guidance on common tax treatment:**  
  * **MYCEL: Non-taxable (no economic value, non-transferable)**
  * **MYCEL peer recognition: Non-taxable (gift-like, no market value)**
  * **SAP: Likely treated as property (capital gains on sale)**
  * **DAO proposal rewards: Likely taxable income**

**IRS Reporting (US Members):**

* **Foundation will not report individual Member transactions (no Form 1099\)**  
* **Members responsible for self-reporting**  
* **Foundation cooperates with lawful subpoenas but challenges overly broad requests**

##### **2.5.7 Regulatory Engagement Strategy**

**The Foundation shall adopt a proactive engagement approach with regulators:**

**Phase 1 (Pre-Launch): Legal Opinion Shopping**

* **Obtain formal opinions from top-tier law firms in US, EU, Singapore, UK**  
* **Publish opinions publicly (radical transparency)**  
* **Adjust protocol design to maximize compliance without compromising Core Principles**

**Phase 2 (Post-Launch): Regulatory Dialogues**

* **Foundation Directors attend regulatory hearings/consultations**  
* **Provide technical education to regulators**  
* **Advocate for "innovation-friendly" frameworks**

**Phase 3 (Mature Network): Industry Leadership**

* **Foundation joins advocacy groups (Blockchain Association, Crypto Council for Innovation)**  
* **Fund academic research on decentralized governance regulation**  
* **Propose model legislation**

**Red Lines (Non-Negotiable):**

* **Will not implement general censorship (targeted sanctions compliance acceptable)**  
* **Will not provide master backdoor keys (breaks cryptographic integrity)**  
* **Will not deanonymize users en masse (lawful subpoenas for specific users acceptable)**

##### **2.5.8 Dissolution and Successor Entities**

**If Swiss Foundation becomes legally untenable (e.g., Switzerland bans crypto):**

**Successor Jurisdiction Criteria:**

* **Strong rule of law (Corruption Perceptions Index \>70)**  
* **Crypto-friendly regulatory framework**  
* **GDPR-adequate data protection**  
* **Political neutrality (not subject to US/China/Russia pressure)**  
* **DAO approval (⅔ Global DAO vote)**

**Potential Successors (Ranked):**

1. **Liechtenstein: Similar to Switzerland, even more crypto-friendly**  
2. **Estonia: E-residency program, digital infrastructure**  
3. **Portugal: Crypto tax haven, EU member**  
4. **Cayman Islands: Offshore finance expertise (but less politically neutral)**

**Transition Process:**

* **New foundation established in successor jurisdiction**  
* **All assets transferred (Swiss Foundation dissolves)**  
* **New foundation ratifies Constitution (continuity preserved)**  
* **Validator nodes migrate to new legal entity**  
* **Process takes \~6 months (requires court approvals)**

### **Section 3\. The Knowledge Council**

#### **3.1 Purpose and Powers**

**An independent epistemic authority responsible for safeguarding data integrity and provenance. The Council may:**

**a. Issue formal suspensions of decisions violating Core Principles (14-day pause, requiring 75% Global DAO supermajority to override). b. Commission independent audits of data quality and provenance. c. Publish annual Network integrity reports. d. Recommend (non-binding) improvements to data standards and verification protocols. e. Maintain and publish annual templates and audit guidelines for Civic Continuity and Disaster Recovery Plans (DRPs), applicable to all DAOs within the Network. f. Curate and safeguard the Civilization Recovery Bundle (CRB), in partnership with the Global DAO. g. Maintain the public Wisdom Library within the Data Commons, hosting inspirational texts (clearly marked as non-binding), DAO-submitted constitutional commentaries, and cultural resources. h. Maintain the Mycelix Symbol Registry providing design guidelines for cultural naming conventions. i. Issue non-binding Ethical Alignment Reports on major MIPs, analyzing them against relevant philosophical frameworks as advisory input.**

#### **3.2 Composition and Election**

**The Council shall consist of 7-11 members elected biennially by network-wide vote. Candidates must demonstrate expertise in data science, epistemology, or related fields. No single Sector or Regional DAO may hold more than 30% of Council seats.**

#### **3.3 Accountability**

**Council members may be impeached by a 75% vote in both houses of the Global DAO for: dereliction of duty, conflicts of interest, or demonstrated bias. Impeached members forfeit salary and are barred from re-election for 5 years.**

#### **3.4 Funding**

**The Council receives 3% of protocol revenues (from the 12% oversight allocation), distributed quarterly.**

### **Section 4\. The Audit Guild**

#### **4.1 Purpose and Powers**

**A permanent technical organ responsible for attesting to faithful implementation of governance votes, cryptographic proofs, and economic controls. The Guild's function is to verify execution, not judge merit. Its mandate includes:**

**a. Financial and Reputational Integrity Audits: Conducting periodic and on-demand audits of any DAO or affiliated entity within the Network to ensure financial health and adherence to constitutional principles. b. Equity and Bias Audits: Performing annual audits of core protocol algorithms, voting mechanisms, and resource allocation patterns to identify and recommend mitigation for systemic biases. c. Algorithmic Integrity Audits: Performing regular, independent audits of all significant AI/ML systems used in governance (including the MATL) for security vulnerabilities, data integrity, potential biases (algorithmic fairness), and compliance with transparency standards. d. Cartel Detection: Operating continuous automated monitoring of validator behavior patterns, leveraging the Mycelix Adaptive Trust Layer (MATL) and its underlying graph-based analytics. Detection thresholds and response protocols are defined in the Charter (Article XII) and updated via MIP-T proposals.**

#### **4.2 Attestation Process**

**For each passed proposal requiring implementation:**

**a. The Guild shall audit the implementation within 30 days. b. Attestation requires unanimous approval from 3 randomly selected Guild members. c. If implementation is verifiably faithful (including verification of any required zk-STARK proofs), attestation must be granted. Refusal without valid technical justification is grounds for removal. d. If implementation deviates, the Guild must publish a detailed report of discrepancies.**

#### **4.3 Composition**

**Guild members must demonstrate technical expertise (cryptography, smart contract development, or systems architecture). Members are nominated by Sector DAOs and confirmed by Global DAO majority vote. Guild members serve 2-year terms with no more than 50% turnover in any election cycle.**

#### **4.4 Bug Bounties and Risk Register**

**The Guild maintains a public risk register prioritizing protocol components for audit. Bug bounties are paid from the Guild treasury according to severity:**

* **Critical (bridge vulnerabilities, direct fund theft): 100,000 SAP**  
* **High (e.g., MATL exploits, bypassing Sybil detection): 50,000 SAP**  
* **Medium (DOS or griefing attacks): 10,000 SAP**  
* **Low (minimal impact): 1,000 SAP**

**Payout Requirements:**

* **Responsible disclosure (private report to Audit Guild)**  
* **No public disclosure for 90 days (patch window)**  
* **If hacker exploits instead: Permanent network ban \+ legal action**

#### **4.5 Funding**

**The Guild receives 5% of protocol revenues, distributed monthly.**

### **Section 5\. The Member Redress Council (MRC)**

#### **5.1 Purpose and Powers**

**An on-call tribunal adjudicating appeals related to individual Member rights (Article VI). The MRC is empowered to:**

**a. Issue binding remedies, including reputation restoration, compensation, and reversal of punitive actions. b. Impose reputation penalties on parties found to have violated Member rights. c. Order the Foundation to execute technical changes necessary to remedy rights violations. d. Investigate and rule on claims of epistemic injustice, systemic exclusion, or cultural erasure within the Network, acting as the primary body for justice and redress. e. Conduct expedited (3-day) reviews of emergency Task DAO actions for constitutional violations.**

#### **5.2 Composition**

**The MRC consists of a rotating pool of 21 eligible arbitrators, randomly selected for each case in panels of 3\. Arbitrators must be Members in good standing (reputation \> 0.6) with no conflicts of interest. Each case panel must include 1 member from a Regional DAO, 1 from a Sector DAO, and 1 at-large Member.**

#### **5.3 Appeals Process**

**a. Members file appeals with a 100 SAP bond (refunded if appeal succeeds). b. The MRC has 14 days to review and issue a preliminary ruling. c. Final rulings are published on-chain with anonymized details to protect privacy. d. MRC decisions may be appealed to the Global DAO only on constitutional grounds (requires ⅔ supermajority).**

#### **5.4 Compensation**

**The MRC treasury (4% of protocol revenues) pays:**

* **500 SAP per case to serving arbitrators.**  
* **Remedies awarded to Members (up to 10,000 SAP per case, or restoration of quantifiable economic losses).**  
* **If the MRC treasury is depleted, the Foundation reserve fund provides an emergency backfill.**

#### **5.5 Accountability**

**Arbitrators who demonstrate repeated bias or fail to uphold the Constitution may be removed by Global DAO supermajority and banned from future service.**

## **ARTICLE III – THE GOLDEN VETO**

### **Section 1\. Purpose and Nature**

**The Golden Veto safeguards the Constitution, Core Principles, and long-term viability of the Network against existential threats.**

### **Section 2\. Transitional Strategic Authority**

**For thirty-six (36) months from Genesis Epoch, the Foundation may exercise Strategic Override to block or suspend governance actions that demonstrably threaten:**

**a. Legal viability (e.g., clear regulatory violations). b. Critical security (e.g., smart contract exploits, cryptographic failures). c. Economic solvency (e.g., decisions causing \>25% treasury depletion).**

### **Section 3\. Sunset and Transformation**

**After 36 months from Genesis Epoch, Strategic Override expires automatically, transitioning to Charter Guardian Authority, limited exclusively to blocking actions demonstrably violating this Constitution or Core Principles.**

### **Section 4\. Renewal**

**Continuation of Strategic Override beyond 36 months requires:**

* **Two-thirds (⅔) majority in both Global DAO houses.**  
* **Ratification by a majority of all Sector DAOs and a majority of all Regional DAOs.**  
* **Renewal is limited to 12-month extensions, requiring re-approval each cycle.**

**Cumulative Limit: Strategic Override may not be renewed more than 3 times (maximum 72 months total from Genesis). After 72 months, only Charter Guardian Authority remains available.**

### **Section 5\. Justification, Review, and Appeal**

#### **5.1 Mandatory Justification**

**All veto exercises must include:**

**a. Written justification citing specific constitutional or existential threats. b. Evidence supporting the claim. c. Proposed alternative actions. d. Immutably recorded on-chain within 24 hours.**

#### **5.2 Knowledge Council Advisory**

**The Global DAO may request a non-binding advisory opinion from the Knowledge Council on veto validity. The Council must respond within 7 days.**

#### **5.3 DAO Override**

**The Global DAO may overturn any veto by a two-thirds (⅔) majority in both houses.**

#### **5.4 Veto Abuse**

**Probation Triggers:**

* **Foundation exercises Golden Veto more than 3 times in a 12-month period (regardless of override status), OR**  
* **Any single veto is overturned by the Global DAO**

**Effect of Probation: Foundation Directors enter probationary status with the following constraints:**

* **Must publish detailed justification for all subsequent vetoes within 24 hours**  
* **Knowledge Council conducts mandatory review of veto pattern (completed within 14 days)**  
* **Probation lasts 12 months from trigger date**

**Impeachment Triggers (while on probation or within 12 months of trigger):**

* **4 total vetoes in 12 months (including overturned ones), OR**  
* **2 overturned vetoes in 12 months**

**Vetoes exercised during probation count double toward impeachment threshold (e.g., 2 vetoes during probation \= 4 for threshold calculation).**

**Impeachment Process: May be initiated by simple majority vote in both houses of the Global DAO. Requires ⅔ supermajority in both houses to remove Foundation Directors.**

#### **5.5 Veto Registry**

**All veto exercises, DAO override attempts, and their final outcomes shall be recorded in an immutable public ledger maintained by the Audit Guild. Patterns of veto usage shall be analyzed and reported in the annual Network Resilience Report.**

**Veto Registry Fields: Each entry must include: veto\_id, timestamp, justification\_hash, threat\_category, affected\_proposal\_id, override\_attempted (boolean), override\_successful (boolean), probation\_status, Knowledge\_Council\_review\_link.**

## **ARTICLE IV – AMENDMENT AND EVOLUTION**

### **Section 1\. Amendment Process**

**Constitutional amendments require:**

**a. A two-thirds (⅔) majority in both Global DAO houses. b. Simple majority (≥51%) ratification by all Sector DAOs. c. Simple majority (≥51%) ratification by all Regional DAOs. d. A 30-day public comment period before the final vote.**

**Any proposal to change the Sybil resistance protocol shall be subject to this full process plus an independent security audit by at least 2 external firms.**

### **Section 2\. Immutable Core**

**The following are immutable except by a 90% supermajority in both Global DAO houses, ratified by three-fourths (¾) of all Sector DAOs and three-fourths (¾) of all Regional DAOs:**

* **Core Principles (Article I, Section 2\)**  
* **Sovereignty of the Constitution (Article I, Section 3\)**  
* **Sunset of Golden Veto (Article III, Section 3\)**  
* **Independent oversight funding minimum (Article VIII, Section 3\)**

### **Section 3\. Adaptive Law Modules (MIPs)**

**Operational bylaws, technical standards, and economic models exist as Mycelix Improvement Proposals (MIPs). These may be modified through standard DAO voting.**

**MIP Process:**

**a. Proposal submission (requires 1000 SAP stake or 100 Member signatures). b. 30-day public review and comment period. c. Technical review by Audit Guild (15 days). d. Vote (simple majority in both Global DAO houses). e. Implementation attestation by Audit Guild.**

**MIP Categories:**

* **Technical (MIP-T): Protocol upgrades, smart contract changes.**  
* **Economic (MIP-E): Tokenomics, fee structures, treasury allocations.**  
* **Governance (MIP-G): DAO procedures, voting mechanisms.**  
* **Social (MIP-S): Community standards, conduct policies.**  
* **Cultural (MIP-C): Optional commons modules, naming conventions, cultural infrastructure.**

### **Section 4\. Deadlock Resolution**

**If an amendment passes the Global DAO but fails tier ratification twice in 12 months, it may proceed to a network-wide referendum of all Members, requiring 60% approval.**

**Governance Engagement Crisis: If average voter turnout falls below 35% for four consecutive quarters, the Global DAO may convene a Constitutional Convention to redesign governance structures. Convention delegates are elected via sortition (random selection from all Members) to ensure broad representation. Convention proposals require ⅔ ratification by full membership.**

## **ARTICLE V – TRANSITIONAL AND FOUNDING PROVISIONS**

### **Section 1\. Genesis Cohort**

**The Founding Stewards shall exercise provisional authority to operationalize governance and ratify this Constitution.**

### **Section 2\. Initial Appointments**

#### **2.1 Foundation**

**The Genesis Cohort shall nominate initial Foundation Directors. No more than 40% may be Genesis Cohort members. Initial Directors serve 2-year terms to allow for staggered elections.**

#### **2.2 Sybil Protocol**

**The Cohort shall designate the initial Sybil resistance protocol, subject to audit by at least 2 independent security firms. The audit reports must be published before network launch.**

**Phase 1 Sybil Protocol: As specified in Article I, Section 2, the Network launches with Gitcoin Passport (v2.0+) requiring ≥20 Humanity Score points.**

**Audit Requirement: Two independent security firms (selected from: Trail of Bits, OpenZeppelin, ConsenSys Diligence, Kudelski Security, NCC Group, or equivalents approved by Audit Guild) must audit the Passport integration and issue public reports certifying:**

* **Sybil resistance effectiveness (\>95% detection rate for known attack patterns)**  
* **Privacy protections adequate**  
* **No single point of failure in verification infrastructure**

#### **2.3 Oversight Bodies**

**The Cohort shall appoint provisional members to the Knowledge Council, Audit Guild, and MRC. These provisional appointments expire after the first DAO-wide elections (no later than 12 months from Genesis).**

### **Section 3\. Dissolution of Founding Authority**

**Genesis authority dissolves automatically upon the first elected Global DAO ratification or after 18 months (whichever occurs first). At dissolution, the individual voting power of all Genesis Cohort members is permanently and programmatically capped at 0.5% (one-half of one percent) of the total network voting power applicable in any single vote.**

## **ARTICLE VI – BILL OF RIGHTS & CIVIC EQUITY**

**Preamble: These rights are granted to all Members and recognized for all participants to ensure sovereignty, dignity, equitable coordination, and universal access.**

### **Section 1\. Right to Disassociation**

**Every Member may control, export, or permanently anonymize their contributed knowledge. This right does not extend to data incorporated into public audit trails or upon which other contributions verifiably depend.**

### **Section 2\. Transparent Reputation**

**Every Member has access to:**

**a. Full reputation history and scoring logic. b. A detailed breakdown of reputation sources. c. Mechanisms to challenge reputation calculations via MRC appeal.**

### **Section 3\. Fair Recourse**

**Every Member has the right to appeal decisions affecting their rights. Appeals shall be heard by the Member Redress Council, which may issue binding remedies.**

### **Section 4\. Privacy by Default**

**No Member shall be required to expose personally identifiable information unless explicitly consented to. The Network shall employ zero-knowledge proofs and other privacy-preserving technologies to minimize data exposure.**

### **Section 5\. Right to Fork and Data Portability**

**Any group of Members may fork the protocol code. The Network affirms the principle that forking communities should honor existing reputation data to maintain social continuity. To facilitate this, the Network shall ensure all DKG data, governance records, and reputation histories are publicly exportable in a standardized, machine-readable format.**

### **Section 6\. Non-Coercion Clause**

**No Member may be penalized for non-participation, dissent, or abstention unless network consensus deems it malicious (requiring ⅔ vote with published evidence).**

### **Section 7\. Universal Access**

**All public goods and documents must be accessible without stake requirements. Critical network functions must remain free or subsidized for Members unable to pay fees.**

### **Section 8\. Ethical Integrity**

**All Members have the right to an annual ethical audit conducted by the Knowledge Council. Findings of ethical violations must be accompanied by binding remedial proposals.**

### **Section 9\. Intellectual Property as Public Good**

**All IP related to the core Mycelix Protocol, extensions, and governance shall be irrevocably donated to the public domain or released under maximally permissive licenses (CC0, MIT).**

### **Section 10\. Right to Accessibility**

**Core Network interfaces and public documents shall adhere to WCAG 2.1 AA standards minimum. To this end, the following implementation requirements are mandated:**

**a. Web Interfaces: All official web interfaces must provide screen reader compatibility (e.g., ARIA labels), full keyboard-only navigation, a high-contrast mode, and font size adjustment up to a minimum of 200%.**

**b. Design Philosophy: A mobile-first design philosophy is mandatory for all new user-facing applications.**

**c. Language Support: The Constitution and Charter shall be made available in the six official languages of the United Nations. Local DAOs are encouraged to provide and maintain translations into additional languages. Machine translation is acceptable for non-binding documents if clearly labeled and accompanied by a human-reviewed summary.**

**d. Alternative Formats: Audio versions of all major governance proposals and plain language summaries (targeted at an 8th-grade reading level) for complex MIPs must be provided.**

**e. Testing and Audits: The Network shall fund quarterly audits by independent accessibility consultants and conduct regular user testing with paid participation from Members with disabilities. Accessibility issues violating WCAG A standards shall be categorized as critical bugs that block releases. The Knowledge Council shall assign and oversee these audits.**

**f. Accommodation: Members may petition the Member Redress Council for reasonable accommodations. The Foundation shall maintain an annual accessibility fund of no less than 50,000 SAP to fulfill such requests.**

### **Section 11\. Right to Explanation**

**Members have the right to human-readable explanations for any algorithmic decisions affecting their reputation, access, or rights.**

### **Section 12\. Right to Equitable Access**

**The Network shall actively support the inclusion of historically underserved populations.**

**a. Custodial Participation: Participation may be delegated through recognized custodial collectives, such as Civic Uplift Trusts, NGOs, or other trusted local institutions, as defined in the Charter.**

**b. Civic Learning Mandate: Regional DAOs must allocate resources to fund Civic Learning Missions with culturally and linguistically tailored onboarding programs, including mobile-first and audio-visual materials.**

**c. Non-Discrimination: The Network shall not discriminate based on education, location, or technical fluency, provided participants adhere to the Core Principles.**

### **Section 13\. Right to Human Adjudication**

**No Member shall be subject to penalties (e.g., reputation slashing, suspension, fund seizure) solely based on an automated or algorithmic finding. AI systems may flag, recommend, or provide evidence, but final enforcement decisions impacting Member rights must involve human review and judgment, typically through the Member Redress Council or relevant DAO body.**

### **Section 14\. Right to Know Non-Human Entities**

**Members have the right to know whether they are interacting with a non-human entity (AI agent, bot, autonomous system). All non-human entities must clearly identify themselves in all governance interactions, proposals, and communications.**

## **ARTICLE VII – JUSTICE AND REDRESS**

### **Section 1\. Mandate for Justice**

**The Network recognizes that decentralized power alone does not ensure justice. It shall therefore embed justice primitives and redress mechanisms directly into its governance framework.**

### **Section 2\. Epistemic Redress**

**The Member Redress Council (per Article II, Section 5\) is formally designated as the Epistemic Redress Board, with the authority to investigate and remedy claims of systemic harm.**

### **Section 3\. Equity Audits**

**All core protocol functions and major policies are subject to a mandatory equity audit by the Audit Guild at least once per year (per Article II, Section 4).**

### **Section 4\. Narrative Sovereignty**

**All Local DAOs and Civic Uplift Trusts have the right to publish their own "Constitutional Commentary," explaining the Core Principles in their own cultural idioms and paradigms. These commentaries shall be recorded and made accessible via the Decentralized Knowledge Graph (DKG) and hosted in the Knowledge Council's Wisdom Library.**

## **ARTICLE VIII – SUSTAINABILITY AND INCENTIVES**

### **Section 1\. Economic Alignment**

**The Network shall implement sustainable incentives to encourage participation while preventing extractive behaviors. The Network recognizes plural forms of economic coordination including gift economies, mutual credit systems, time exchange, stewardship recognition, and market mechanisms.**

### **Section 2\. Treasury Governance**

#### **2.1 Structure**

**A dedicated Treasury Module shall manage funds, with allocations requiring Global DAO approval.**

#### **2.2 Treasury Composition**

**The Network treasury shall maintain diversified holdings to optimize for liquidity, longevity, and credibility:**

**Recommended Composition:**

* **50% Stablecoins (diversified: 40% USDC, 30% DAI, 20% LUSD, 10% experimental)**  
* **20% ETH / Liquid Staking Tokens (stETH, etc.)**  
* **10% BTC (wrapped)**  
* **10% Diversified LSTs (mLSD, rsETH)**  
* **5% Non-correlated assets (RWA-backed tokens, tokenized treasuries)**  
* **5% Cash buffer (via off-chain bank partner or fiat ramp)**

**Vault Structure:**

* **Main Treasury Vault (multi-signature, Layer 1\)**  
* **Execution Vaults (per DAO: ecosystem, grants, infrastructure)**  
* **Emergency Cold Vault (hardware multi-sig \+ legal escrow)**

**Treasury composition may be adjusted via MIP-E proposals subject to Audit Guild risk analysis.**

#### **2.3 Inflation Cap**

**Treasury inflation is capped at 5% annually. Changes to this cap require a Constitutional Amendment.**

#### **2.4 Spending Transparency**

**All treasury expenditures \> 10,000 SAP must be published with justifications. Expenditures \> 100,000 SAP require Audit Guild attestation.**

**Bridge Security Spending: All bridge insurance premiums, security audit costs, and recovery fund contributions shall be published monthly with itemized justifications. Bridge Recovery Fund balance must be disclosed in real-time via on-chain oracle.**

### **Section 3\. Independent Funding for Oversight**

#### **3.1 Allocation**

**A minimum of 12% of all protocol revenue shall be automatically and perpetually directed into independent treasury contracts:**

* **Knowledge Council: 3%**  
* **Audit Guild: 5%**  
* **Member Redress Council: 4%**

#### **3.2 Distribution Schedule**

**Funds are distributed monthly. If protocol revenues fall below operational minimums, the Foundation reserve fund provides emergency funding for up to 6 months.**

#### **3.3 Surplus Management**

**If an oversight body's treasury exceeds 24 months of operating expenses, surplus funds revert to the general treasury.**

#### **3.4 Immutability**

**This 12% minimum allocation is part of the Immutable Core (Article IV, Section 2).**

### **Section 4\. Global Stewardship and Research Mandate**

**The Global DAO and Foundation shall jointly allocate no less than 5% of the annual general treasury budget to support independent research, auditing, and investigation of all sub-DAOs, Civic Uplift Trusts, and affiliated projects. This mandate covers financial integrity, alignment with Core Principles, anti-corruption safeguards, and inclusivity metrics.**

### **Section 5\. Economic Primitives**

**The Network shall recognize the following core economic primitives:**

#### **5.1 MYCEL (Soulbound Reputation)**

**A non-transferable reputation score (0.0–1.0) representing a Member's verifiable contributions, expertise, and trustworthiness within the Network. MYCEL serves as the foundation for governance weight, consciousness-gated tier progression, and fee rate determination.**

**Properties:**

* **Non-transferable and non-tradeable (soulbound)**
* **Computed from four weighted components: Participation (40%), Peer Recognition (20%), Validation Quality (20%), Longevity (20%)**
* **Decays 5% annually without activity; compresses 0.8× every four years via jubilee**
* **Used for reputation-weighted voting, tier progression (Apprentice < 0.3, Member 0.3–0.7, Steward > 0.7), and fee rate determination**
* **Peer recognition (10 recognitions/month per member, minimum MYCEL 0.3 to give, weighted by recognizer's score) replaces the former CGC social signaling primitive — recognition categories may use cultural aliases (SPARK, EMBER, PETAL, LIGHT)**

#### **5.2 SAP (Circulation Medium)**

**A transferable token serving as the primary medium of exchange, subject to demurrage (2% annual decay). SAP that decays from private accounts flows to commons pools (70% local, 20% regional, 10% global).**

**Properties:**

* **Transferable, subject to demurrage**
* **Mintable against physical assets (energy certificates, agricultural production, fiat bridges)**
* **Redeemable for real goods and services**
* **Commons pool SAP exempt from demurrage (waqf principle)**
* **Constitutional inalienable reserve: 25% minimum in commons pools**
* **Explicitly separated from governance power**
* **Subject to securities law compliance (Article II, Section 2.5.4)**

**Constitutional Mandate: Governance reputation (MYCEL) shall not be financialized or transferable. The circulation medium (SAP) remains functionally and economically distinct from governance weight.**

## **ARTICLE IX – EMERGENCY PROVISIONS**

### **Section 1\. Technical Circuit Breaker**

#### **1.1 Composition and Power**

**The Foundation maintains a 3-of-5 multisig (2 Foundation Directors, 2 Audit Guild leads, 1 Knowledge Council lead) with the sole power to pause all non-critical protocol functions.**

#### **1.2 Triggering Conditions**

**The circuit breaker may only be used for:**

**a. Smart contract exploits (e.g., reentrancy attack, oracle manipulation). b. Severe economic attacks (e.g., flash loan manipulations causing \>10% treasury loss). c. Critical consensus failures (e.g., Byzantine validator collusion). d. Cryptographic compromises. e. Bridge Compromise: Any credible evidence of smart contract exploit, validator collusion, or oracle manipulation affecting cross-chain bridges. Evidence assessed by Audit Guild emergency team with 2-hour response SLA.**

#### **1.3 Duration and Review**

**The pause expires automatically after 72 hours. Use triggers a mandatory Global DAO review within 7 days.**

### **Section 2\. Governance Pause**

**The Global DAO may, by supermajority (⅔) vote in both houses, pause governance functions for up to 30 days to mitigate non-exploit threats.**

#### **2.1 Limitations**

**A governance pause may not block Member Redress Council operations or prevent emergency fund disbursements.**

## **ARTICLE X – SEVERABILITY AND DISSOLUTION**

### **Section 1\. Severability**

**If any provision is held invalid, the remainder shall remain in effect. The Global DAO shall convene within 30 days to address the invalidated provision.**

### **Section 2\. Network Dissolution**

#### **2.1 Conditions**

**The Network may be dissolved only by:**

* **A 90% supermajority in both Global DAO houses.**  
* **Ratification by three-fourths (¾) of all Sector DAOs and three-fourths (¾) of all Regional DAOs.**

#### **2.2 Asset Distribution**

**Upon dissolution:**

**a. All remaining funds in the General Treasury, the Foundation Reserve Fund, and all Oversight Body treasuries (Knowledge Council, Audit Guild, and Member Redress Council) shall be transferred to a single, immutable "Dissolution Contract". b. The Dissolution Contract shall distribute all pooled assets proportionally to all verified Members based on their final, pre-dissolution MYCEL reputation scores. c. The distribution mechanism specified in the Dissolution Contract must be ratified as part of the dissolution vote. d. All IP remains irrevocably in the public domain.**

#### **2.3 Data Preservation**

**All DKG data, reputation histories, and governance records shall be archived in a publicly accessible format before dissolution.**

## **ARTICLE XI – SOVEREIGNTY AND ARTIFICIAL ENTITIES**

### **Section 1\. Sovereignty of Natural Persons**

**The Network affirms that only natural persons may be recognized as Members with full civic sovereignty, including voting rights, constitutional protections, and the right to hold reputation (MYCEL).**

### **Section 2\. Instrumental Actors**

**Non-human agents (including autonomous AI systems, bots, and oracles) may be registered as Instrumental Actors, subject to the following constraints:**

**a. Must be verifiably sponsored by and accountable to a recognized Member or DAO b. Subject to continuous audit and performance monitoring c. May not hold MYCEL (reputation) or vote in governance d. Must clearly identify as non-human in all interactions e. Subject to immediate suspension if behavior violates Core Principles**

**Integrity Scoring for Instrumental Actors: Non-human entities shall be evaluated via a separate Computational Trust Quotient (CTQ) or Synthetic Reputation Index (SRI) derived from uptime, audit logs, and deterministic output validity—not peer attestation. CTQ/SRI systems defined via MIP-T proposals.**

### **Section 3\. Moratorium on AI Sovereignty**

**The question of AI Sovereignty may be revisited via a supermajority Constitutional Amendment process. Any such amendment must include:**

**a. A demonstrated, reproducible method for aligning AI behavior with the Core Principles b. Verifiable consent frameworks c. Public ethical review by the Knowledge Council and Audit Guild**

## **ARTICLE XII – DEFINITIONS**

**Agent: Any entity (human or autonomous) interacting with Network smart contracts.**

**Member: A verifiably unique human participant confirmed by the Network's designated Sybil resistance protocol. Only Members possess Article VI rights and governance eligibility.**

**DKG: Decentralized Knowledge Graph.**

**Genesis Epoch: The date of mainnet launch.**

**Supermajority: Two-thirds (⅔) or greater vote, unless otherwise specified.**

**Protocol Revenue: All fees, transaction costs, or value capture generated by Network operations, excluding external grants or donations.**

**Cause (for removal): Gross negligence, fraud, conflicts of interest, or repeated failure to uphold constitutional duties.**

**Algorithmic Transparency: The principle that algorithms, models, or AI agents influencing governance must be publicly documented, auditable, and explainable.**

**Civic Uplift Trust: A recognized custodial collective, defined in the Charter, that acts as an interface for communities not yet fully participating as DAOs, with a primary mission of education and capacity building.**

**Polycentric Governance: A system of governance with multiple, semi-autonomous, and overlapping centers of decision-making that interact through both competition and cooperation within a shared framework of rules.**

**Adaptive Governance: A governance approach that emphasizes flexibility, continuous learning, and broad participation to manage complex and unpredictable systems.**

**Instrumental Actor: A non-Member digital agent or AI system granted limited, specific operational rights within the Network via a ratified MIP. Instrumental Actors possess no Article VI rights, cannot vote, and must be verifiably sponsored by and accountable to a recognized Member or DAO.**

**Civic Continuity/DRP: A Disaster Recovery Plan mandated for all DAOs to ensure the preservation of critical governance data and operational continuity.**

**Civilization Recovery Bundle (CRB): A minimal, open-source Mycelix instance containing the core code and constitutional documents, archived for planetary-scale resilience.**

**Peer Recognition (formerly CGC): Non-market contributions are recognized through MYCEL's peer recognition component (20% weight). Recognition categories may use cultural aliases (SPARK, EMBER, PETAL, LIGHT). 10 recognitions per member per monthly cycle, weighted by recognizer's MYCEL score.**

**Epistemic Injustice: Harm done to an individual or group in their capacity as a knower, including testimonial injustice (undue credibility deficit) and hermeneutical injustice (lack of collective interpretive resources).**

**Epistemic Dispute: A formal challenge to a DKG claim's validity, provenance, or interpretation. Disputes do not delete claims but rather annotate them with resolution status and reasoning.**

**Expert Override: A mechanism allowing the Knowledge Council to resolve deadlocked constitutional epistemic disputes when democratic processes fail, subject to strict constraints and public accountability (max once per year).**

**Fusion Charter: A formal document, based on a Network template, that outlines the governance, structure, and asset consolidation plan for two or more DAOs that are merging.**

**Liminal DAO: A provisional, agile DAO designed for experimentation or inter-domain coordination that exists outside the primary federated tiers but is anchored to the formal structure for accountability.**

**Foundation: Unless context requires otherwise, refers to the Mycelix Foundation (Swiss Stiftung) and its wholly-owned subsidiary entities acting collectively, subject to the constraints in Article II, Section 2.5.**

**Regulatory Compliance: Adherence to applicable law in jurisdictions where the Foundation or its Members operate, provided such compliance does not violate the Core Principles or Member Rights. In cases of irreconcilable conflict, Core Principles prevail.**

**Computational Trust Quotient (CTQ) / Synthetic Reputation Index (SRI): Integrity scoring systems for Instrumental Actors based on uptime, audit logs, and deterministic output validity rather than social attestation.**

**Gitcoin Passport: A decentralized identity verification protocol using cryptographic stamps from multiple web2/web3 identity providers to calculate a "Humanity Score" representing the probability a user is a unique human. Developed by Gitcoin.**

**Humanity Score: A numerical score (0-100) calculated by Gitcoin Passport based on accumulated identity stamps. Higher scores indicate stronger evidence of unique humanity. Network threshold: ≥20 points for Member verification.**

**MATL (Mycelix Adaptive Trust Layer): A verifiable middleware layer (defined in the Charter) that combines composite trust scoring, cartel detection, and zero-knowledge proofs to enhance Sybil resistance and algorithmic transparency.**

**MIP (Mycelix Improvement Proposal): A formal proposal to modify operational bylaws, technical standards, or economic models, as defined in Article IV.**

## **ARTICLE XIII – LEGAL COMPLIANCE AND REGULATORY ENGAGEMENT**

### **Section 1\. Proactive Legal Strategy**

**The Foundation shall maintain a proactive legal compliance posture, obtaining formal legal opinions before launching features with regulatory implications.**

### **Section 2\. Required Legal Opinions (Pre-Mainnet)**

**Before mainnet launch, the Foundation must obtain and publish legal opinions from reputable law firms in the following jurisdictions:**

**United States:**

* **Firm: Sullivan & Cromwell, Cooley, Debevoise & Plimpton, or equivalent**  
* **Scope: Securities classification of MYCEL and SAP under Howey test**

**European Union:**

* **Firm: Clifford Chance, Allen & Overy, Linklaters, or equivalent**  
* **Scope: MiFID II compliance, GDPR compliance, DLT Pilot Regime applicability**

**Switzerland:**

* **Firm: Homburger, Lenz & Staehelin, Niederer Kraft Frey, or equivalent**  
* **Scope: Swiss Foundation regulatory obligations, FINMA token classification**

**Publication Timeline: Legal opinions must be published to the public at least 30 days before mainnet launch to allow community review.**

**Budget Allocation: Legal opinion procurement funded from Foundation reserve (Article II, Section 2.1.f), estimated $150,000-$300,000 total.**

### **Section 3\. Regulatory Engagement Phases**

**Phase 1 → Phase 2 Trigger: Mainnet launch (Q4 2025 target) Phase 2 → Phase 3 Trigger: Network reaches 10,000 active Members OR $10M TVL, whichever occurs first**

**Phase transitions require Foundation Directors approval plus Global DAO advisory vote (simple majority, non-binding but informs Foundation strategy).**

## **APPENDIX A – CULTURAL LAYER AND PLURALISM**

### **Purpose**

**This appendix provides guidance on implementing cultural expression within the constitutional framework without compromising the neutrality of core governance.**

### **Cultural Resources Infrastructure**

**Knowledge Council Responsibilities:**

1. **Wisdom Library: Maintain a public, open-access repository within the Data Commons hosting:**  
   * **Inspirational texts (clearly marked as non-binding, including foundational documents like the Luminous Library)**  
   * **DAO-submitted constitutional commentaries reflecting diverse cultural interpretations**  
   * **Optional templates for DAO rituals, ceremonies, and value-alignment workshops**  
   * **Registry of cultural aliases for economic primitives**  
2. **Symbol Registry: Maintain the official Mycelix Symbol Registry with design guidelines:**  
   * **MYCEL (Soulbound Reputation): 🏛️ Stone/Structure (aliases: CIV, STANDING, FOUNDATION)**  
   * **MYCEL Recognition Categories: ✨ Fire/Light (aliases: SPARK, EMBER, PETAL, LIGHT, GRATITUDE)**
   * **SAP (Circulation Medium): 💧 Water/Flow**  
   * **TEND (Time Exchange): 🤲 Care/Time**  
   * **ROOT/SEED (Stewardship): 🌱 Earth/Regeneration**  
   * **BEACON/WIND (Signaling): 🧭 Direction/Voice**  
   * **HEARTH/WELL (Commons): 🔥 Community/Shared Life**  
3. **Ritual Templates: Develop and maintain open-source templates for:**  
   * **Member onboarding ceremonies**  
   * **Proposal blessing/attunement processes**  
   * **Dispute resolution circles**  
   * **Project completion celebrations**  
   * **Quarterly reflection ceremonies**

**Local DAO Sovereignty:**

**Local DAOs and Civic Uplift Trusts may:**

* **Adopt cultural naming for all economic primitives**  
* **Develop unique rituals and ceremonies**  
* **Express specific philosophical orientations in their charters**  
* **Allocate budgets for artistic expression and cultural events**  
* **Translate governance materials into local languages**

**Constitutional Safeguards:**

**All cultural expression must:**

* **Align with Core Principles (Article I, Section 2\)**  
* **Respect Member Rights (Article VI)**  
* **Not discriminate or exclude based on belief systems**  
* **Remain optional (no Member may be coerced to participate)**

### **Wisdom Councils (Optional)**

**Liminal DAOs or specialized guilds may be chartered as "Wisdom Guilds" to:**

* **Provide philosophical commentary on governance proposals**  
* **Issue non-binding Ethical Alignment Reports**  
* **Analyze MIPs against various ethical frameworks**  
* **Offer advisory input without legislative power**

**(Version History: v0.25 – PRODUCTION ALIGNMENT UPDATE. Renamed CIV (Civic Standing) to MYCEL (soulbound reputation, 0.0–1.0, four-component weighted). Renamed FLOW (Utility Token) to SAP (circulation medium with 2% annual demurrage). Absorbed CGC (Civic Gifting Credits) into MYCEL's peer recognition component (20% weight, 10 recognitions/month, culturally aliasable). Updated Article VIII Section 5 (Economic Primitives) to reflect the three-currency production implementation: MYCEL, SAP, TEND. All changes align charter with running Holochain zome code.**

**v0.24 – CRITICAL ALIGNMENT & SECURITY UPDATE. Added "Verifiable Computation" as a Core Principle (Art I.2) to constitutionally ground the MATL. Updated all proposal acronyms from MYC-X to MIP-X and all token references from MYC to FLOW for network-wide consistency. Removed 5-year time window from AI Sovereignty moratorium (Art XI.3), relying on the Immutable Core amendment process. Added definitions for MATL and MIP (Art XII).**

**v0.23 – CRITICAL PRE-MAINNET UPDATE. Formalized Sybil resistance protocol (Gitcoin Passport ≥20) with audit requirements. Added mandatory pre-mainnet legal opinion requirements from top-tier firms in US/EU/Switzerland. Enhanced Golden Veto abuse protections with explicit probation/impeachment thresholds. Clarified Strategic Override cumulative renewal limit (max 72 months). Added regulatory engagement phase triggers. Expanded definitions to include Gitcoin Passport and Humanity Score.)**

