# **THE EPISTEMIC CHARTER (v1.0)**

**Companion instrument to the Mycelix Spore Constitution (v0.24) and part of the modular Mycelix Charter Set (v1.0).**

***Editor's Note:** This Epistemic Charter (v1.0) is a refactored module of the original THE FEDERATED DAO HIERARCHY CHARTER (v0.24). It isolates all articles and appendices pertaining to technology standards, the Mycelix Adaptive Trust Layer (MATL), consensus parameter governance, cartel detection, privacy, algorithmic standards, and the epistemic framework (LEM and Claim Schema). All information is sourced directly from the v0.24 Charter.*

## **ARTICLE I – TECHNOLOGY AND INFRASTRUCTURE**

### **Section 1\. Core Infrastructure**

All governance infrastructure must be decentralized, open-source, and adhere to Mycelix Interoperability Standards (MIS).

### **Section 2\. Security**

Smart contract upgrades are subject to a 14-day timelock and require Audit Guild attestation.

### **Section 3\. Consensus Parameter Governance**

Critical RB-BFT consensus parameters shall be governed via MIP-T (Technical) proposals, subject to the following minimum requirements:

| Parameter | Initial Value | Change Threshold | Audit Requirement |
| :---- | :---- | :---- | :---- |
| MIN\_REPUTATION\_THRESHOLD | 0.4 | ⅔ Global DAO | Mandatory Security Audit |
| Reputation Exponent | 2 (quadratic) | ⅔ Global DAO | Mandatory Simulation & Audit |
| Decay Rate (passive) | 5%/year | Simple Majority | Optional |
| Decay Rate (malicious) | 50% base slash | ⅔ Global DAO | Mandatory Security Audit |
| Ejection Threshold | 0.1 | ⅔ Global DAO | Optional |
| MATL Weights (PoGQ/TCDM/Entropy) | 0.4 / 0.3 / 0.3 | ⅔ Global DAO | Mandatory Simulation & Audit |

**Audit Procurement**:

* Proposer must fund audit (refunded from treasury if proposal passes).  
* Audit must be from reputable firm (Knowledge Council maintains approved list).  
* Audit report published 14 days before vote (community review period).

**Ambiguous Audit Results**:

* If audit conclusion is "uncertain" or "depends on assumptions":  
  * Parameter change requires 75% vote (higher bar)  
  * OR deploy to testnet for 90 days before mainnet.

**Emergency Parameter Adjustment**:

* Technical Circuit Breaker (Constitution IX.1) may adjust parameters during active exploits.  
* Changes limited to ±20% of current value (prevents radical changes).  
* Change expires after 72 hours unless ratified by Global DAO.

The Technical Circuit Breaker may temporarily adjust these parameters to mitigate an active exploit, subject to immediate Global DAO review.

### **Section 4\. Real-Time Cartel Detection (MATL-Driven)**

The Network's anti-cartel defense is managed by the **Mycelix Adaptive Trust Layer (MATL)**, which performs real-time graph analysis to detect coordinated behavior and calculate a cluster risk score.

Risk Score Calculation: MATL computes a CartelRiskScore for validator clusters based on a weighted formula.

(Code implementation details omitted for brevity, see original Charter)

Automatic Risk Mitigation: MATL automatically applies mitigations based on the overall\_risk score.

(Code implementation details omitted for brevity, see original Charter)

**Economic Penalties**:

* **Validator Bonding Requirement**: All validators must stake **5,000 SAP** (slashable deposit).  
* If identified as a malicious cartel member (Phase 3 Dissolution): **100% slash** (entire stake).  
* Slashed funds distributed to whistleblowers (30%) and treasury (70%).

**Reputation Decay Acceleration**: Members of dissolved cartels experience **2x passive decay** for 12 months (5%/year → 10%/year).

**Whistleblower Protections**:

* Anonymous reporting via Tor hidden service (operated by Audit Guild).  
* Zero-knowledge proof of evidence (don't reveal reporter identity).  
* Whistleblower rewards: 30% of slashed stake (up to 50,000 SAP) if cartel confirmed.  
* Protection from retaliation (Member Redress Council jurisdiction).

**False Accusation Penalties**:

* False cartel accusation (proven malicious): 1,000 SAP fine \+ reputation penalty.  
* Negligent false accusation (honest mistake): 100 SAP fine, no reputation penalty.

### **Section 5\. Mycelix Adaptive Trust Layer (MATL)**

#### **5.1 Purpose and Architecture**

The Network shall employ a verifiable middleware layer (MATL) that combines:

* Composite trust scoring for Sybil detection (PoGQ, TCDM, Entropy).  
* Graph-based cartel detection.  
* zk-STARK proof generation for algorithmic transparency.  
* Adaptive differential privacy for governance.

MATL operates as Layer 4.5 in the protocol stack, interfacing between Identity (Layer 4\) and Governance (Layer 5).

#### **5.2 Composite Trust Score Formula**

Score \= (PoGQ × 0.4) \+ (TCDM × 0.3) \+ (Entropy × 0.3)

Where:

* **PoGQ** (Proof of Quality): Validation accuracy over last 90 days.  
* **TCDM** (Temporal/Community Diversity): 1.0 \- (internal\_validation\_rate × 0.6 \+ temporal\_correlation × 0.4).  
* **Entropy**: Shannon entropy of validation patterns, normalized and penalizing extremes.

Members with final\_score \< 0.4 flagged as suspected Sybils (reputation growth frozen).

#### **5.3 Integration with RB-BFT**

Composite scores act as quality multipliers on reputation growth (per the Economic Charter) and performance decay (per the Economic Charter).

#### **5.4 zk-STARK Proof Requirements**

The aggregator (Foundation or delegated service) shall generate zk-STARK proofs using Risc0 zkVM (or equivalent) for:

* Each validator selection round (proving scores computed correctly).  
* Governance vote aggregations (proving DP noise added as specified).  
* Cartel detection decisions (proving risk scores and mitigation actions).

Proofs published to public bulletin (IPFS \+ Ethereum L1 anchor) within 24 hours.

#### **5.5 Adaptive DP for Governance**

Voting on Constitutional matters (per Governance Charter Article III, Section 2\) shall employ adaptive differential privacy:

* Base epsilon: 0.1 (strong privacy).  
* Noise scale increases if \<100 participants (shortage\_factor \= sqrt(100 / actual\_participants)).  
* Median aggregation before noise (Byzantine-resistant).  
* Proof attests: "Laplace noise with scale X added".

#### **5.6 Audit and Security**

MATL components subject to same audit requirements as core protocol (per Economic Charter Article I, Section 5):

* Security Audit \#2 must include MATL composite scoring, cartel detection, and zkVM integration.  
* Bug bounty: MATL exploits (bypassing Sybil detection) classified as High severity (50,000 SAP).  
* Knowledge Council reviews MATL parameter tuning annually.

#### **5.7 Evolution and Middleware**

The Global DAO may license MATL as standalone middleware via MIP-E proposal. Revenue from commercial licensing (e.g., enterprises using MATL for federated learning) allocated:

* 40% Treasury  
* 30% R\&D (improve MATL)  
* 20% Validator rewards  
* 10% Knowledge Council (oversight)

### **Section 6\. Privacy-Preserving Governance**

Members may cast votes using zero-knowledge proofs to preserve ballot secrecy. This is enforced for high-stakes votes via Adaptive Differential Privacy (per Section 5.5 of this Article).

### **Section 7\. Algorithmic Systems Standards**

a. **Public Registry**: The Knowledge Council shall maintain a public registry of all significant AI/ML systems deployed within Network governance (including MATL), including documentation links, audit reports, and designated points of contact.

b. **Explainability**: Systems must provide human-readable explanations for their outputs upon request (per Constitution Art VI, Sec 11).

c. **Open Weight Preference**: Models developed with Network treasury funds should default to releasing weights and training configurations under open licenses, unless exempted by a Global DAO supermajority vote citing specific security or privacy risks.

d. **Decentralized Hosting Goal**: The Network shall strive, where technically feasible and secure, to host critical AI governance functions on decentralized compute infrastructure. Implementation details shall be determined via MIP.

e. **Mandatory Disclosure**: All AI-authored proposals, comments, or validations must be clearly flagged as originating from Instrumental Actors.

### **Section 8\. Epistemic Ledger Requirements**

a. All governance-significant claims must be submitted via a standardized **Epistemic Claim Object** conforming to the Network schema (see Appendix B of this Charter).

b. Clients and DAOs must implement schema validation at input.

c. The Decentralized Knowledge Graph (DKG) shall serve as the primary, append-only ledger for all Tier ≥ 2 claims, ensuring a permanent and auditable record.

d. Core Network infrastructure must provide standardized APIs for indexing, querying, and subscribing to Epistemic Claim Objects based on schema fields.

### **Section 9\. Core Registries**

The Network shall maintain a set of core registries, including but not limited to:

a. **The Epistemic Claim Registry**: An append-only, queryable ledger containing all Tier ≥ 2 claims.

b. **The Trust Tag Ontology**: A versioned dictionary of source labels and validation standards managed by the Knowledge Council.

c. **The Claim Lineage Map**: A graph representation of claim relationships for epistemic reasoning.

d. **Epistemic Dispute Resolution Protocol**:

##### **i. Tiered Dispute Resolution**

When conflicting Tier 2+ claims are submitted, disputes are resolved through tiers based on claim\_materiality:

**Tier A: Routine Claims** (claim\_materiality \= "routine")

* Threshold: \<10,000 SAP impact, no governance implications.  
* Process: 3-member Sector DAO panel issues binding decision in 14 days.  
* Appeal: 50 SAP bond to escalate to Tier B.

**Tier B: Significant Claims** (claim\_materiality \= "significant")

* Threshold: 10K-100K SAP impact or sector-wide policy.  
* Process: 5-member Knowledge Council panel → Relevant Sector/Regional DAO votes (simple majority, 30% quorum).  
* Fallback: If quorum fails twice, Knowledge Council decision is binding.

**Tier C: Constitutional Claims** (claim\_materiality \= "constitutional")

* Threshold: Affects Core Principles, Member rights, or \>100K SAP.  
* Process:  
  1. Disputing party files challenge (500 SAP equivalent bond).  
  2. Knowledge Council \+ Audit Guild joint panel (7 members) conducts review.  
  3. Mandatory public comment period (30 days).  
  4. Global DAO votes with special majority: 60% approval required in both houses.  
  5. Fallback: If the Global DAO vote fails quorum twice, the joint KC+AG panel's consensus decision (requiring 5 out of 7 votes) becomes binding, but flagged as "non\_democratic\_resolution": true. This fallback decision is valid for 1 year unless ratified by a subsequent Global DAO vote.  
  6. Expert Override: Separately, the Knowledge Council retains its power (per Constitution Article II, Section 3.1a and this Charter's Article I, Section 9.d.i Tier C) to invoke its once-per-year Expert Override after a Global DAO vote (even if quorum was met) if it achieves a 7/11 internal supermajority, suspending the DAO's decision. This override is also flagged as non-democratic and triggers constitutional review.  
* Appeal: To Member Redress Council on procedural grounds only.

**Tier D: Existential Claims** (claim\_materiality \= "existential")

* Threshold: Threatens network viability (e.g., "critical smart contract vulnerability").  
* Process: Circuit Breaker activated → AG emergency verification (24h) → Global DAO ratifies ex-post (simple majority, 7 days).  
* Appeal: None (security trumps process).

##### **ii. Default Classification Rules**

* If claim\_materiality is ambiguous, defaults to **one tier higher** (err on side of caution).  
* Knowledge Council may reclassify claims if materiality changes.  
* All reclassifications require public justification.

##### **iii. Deadlock Prevention Mechanisms**

**Sunset Clause**: If Tier B or C dispute remains unresolved for 90 days:

* Claim automatically tagged "unresolved\_timeout".  
* Both competing claims remain in DKG marked "disputed\_long\_term".  
* Consuming applications must handle uncertainty (e.g., display confidence intervals).

**Temporal Validity**: All dispute resolutions include expiration date (default: 5 years). Claims may be re-disputed if new evidence emerges.

##### **iv. Outcome Recording**

* Winning claim: "status": "affirmed\_\[tier\]\_consensus".  
* Losing claim: "status": "disputed\_minority" (NOT deleted).  
* All dissenting opinions preserved in metadata.  
* Voting records public (unless Member invokes ZK privacy).

##### **v. Bond Distribution**

* **Challenge succeeds**: Bond returned \+ 50 SAP reward from losing party.  
* **Challenge fails**: Bond split: 50% to defending party, 50% to dispute resolution body treasury.  
* **Frivolous challenges**: Bond forfeited \+ reputation penalty.

## **ARTICLE II – RATIFICATION AND CONTINUITY**

### **Section 1\. Charter Ratification**

This Charter takes full effect upon adoption by the Global DAO and ratification by the federated tiers.

### **Section 2\. Supremacy Clause**

The Mycelix Spore Constitution prevails over this Charter in any case of conflict.

## **APPENDIX A – LAYERED EPISTEMIC MODEL (LEM) v1.0**

*(Formerly Appendix E)*

### **Purpose**

This appendix operationalizes the **Epistemic Verifiability** Core Principle (Constitution Article I, Section 2). It defines the standard tiers for classifying the verifiability of claims within the Mycelix Network and outlines the governance implications associated with each tier. This model serves as the foundation for the Epistemic Claim Schema Standard (Appendix B).

### **Stewardship**

The **Knowledge Council** is the designated steward of this Layered Epistemic Model. It is responsible for maintaining the definitions, providing implementation guidance, and proposing updates via MIPs (Governance).

### **Epistemic Tiers and Governance Implications**

All significant epistemic inputs influencing governance must be classified into one of the following tiers. Claims default to Tier 0 unless explicitly classified otherwise and verified according to tier requirements.

| Tier | Name | Description | Verification Method | Governance Implications | Examples |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **0** | **Null / Unverifiable** | Claims lacking verifiable source, evidence, or clear meaning. Anonymous or unsubstantiated assertions. | None Possible | **Cannot be used as the sole basis for any binding governance action** (resource allocation, policy change, rights adjudication). May inform discussion but holds no formal weight. | Rumors, unauthenticated messages, vague opinions, claims tagged "frivolous" during dispute resolution. |
| **1** | **Testimonial** | Claims based on subjective experience, direct observation, or personal attestation, linked to a verified Member DID. | Member Signature / Attestation | Can be used for **low-stakes decisions**, signaling preference (e.g., CGCs, local polls), or as **supporting evidence** in disputes. Insufficient for high-stakes actions or constitutional matters alone. | Member reports, qualitative user feedback, witness statements in disputes, expressions of cultural value. |
| **2** | **Privately Verifiable** | Claims whose veracity can be confirmed by authorized parties (e.g., Audit Guild, specific DAOs) under defined procedures, often involving encrypted data. | Audit Guild Attestation, Multi-Sig Verification | Can support **significant decisions** (e.g., budget approvals, validator performance reviews) if verification procedures are documented and audited. Suitable for sensitive data requiring controlled access. | Confidential financial audits, results of private DAO votes, sensitive security reports, KYC/AML checks. |
| **3** | **Cryptographically Proven** | Claims verified through mathematical proofs (e.g., ZKPs, digital signatures, cryptographic hashes) without revealing underlying private data. | ZKP Verification, Signature Validation | Sufficient for **high-stakes decisions**, automated actions, and establishing objective facts within the protocol. Forms the basis for verifiable computation and trustless interactions. | zk-SNARK/STARK proofs (e.g., MATL outputs), transaction signatures, Merkle proofs, verifiable credentials (VCs). |
| **4** | **Publicly Reproducible** | Claims verifiable by anyone through open data, open-source code, or repeatable experiments/calculations. Represents the highest standard. | Public Data Analysis, Code Execution | **Gold standard** for constitutional amendments, core protocol parameters, scientific claims, and any decision requiring maximum public legitimacy. Preferred tier where feasible. | Mathematical proofs, open-source code execution results, publicly queryable DKG data, results from published scientific papers. |

### **Implementation Notes**

* **Claim Materiality:** The required Epistemic Tier for a claim often depends on its **materiality** (Routine, Significant, Constitutional, Existential), as defined in Appendix B. Higher materiality generally requires higher-tier evidence.  
* **Tier Classification:** Claims submitted via the Epistemic Claim Schema (Appendix B) must include the epistemic\_tier field. The Knowledge Council provides guidelines and may review classifications.  
* **Disputes:** Challenges to a claim's validity or its assigned tier are handled via the Tiered Epistemic Dispute Resolution protocol (Article I, Section 9.d).

## **APPENDIX B – EPISTEMIC CLAIM SCHEMA STANDARD v1.1**

*(Formerly Appendix F)*

This appendix defines the formal, machine-readable schema for all governance-relevant epistemic inputs. The Knowledge Council is the steward of this schema, and major updates must be ratified via MIP.

JSON

```

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

* Added claim\_materiality field for tiered dispute resolution.  
* Added submitter\_type to distinguish human vs AI submissions.  
* Enhanced provenance tracking for chain of custody.

## **APPENDIX C – DEFINITIONS**

*(Editor's Note: This appendix includes only definitions from the original Charter relevant to the Epistemic Charter.)*

**Computational Trust Quotient (CTQ)**: Integrity score for Instrumental Actors based on uptime, audit logs, and deterministic output validity.

**Synthetic Reputation Index (SRI)**: Alternative term for CTQ, emphasizing distinction from human reputation.

**Epistemic Dispute**: A formal challenge to a DKG claim's validity, provenance, or interpretation. Disputes do not delete claims but rather annotate them with resolution status and reasoning.

**Expert Override**: A mechanism allowing the Knowledge Council to resolve deadlocked constitutional epistemic disputes when democratic processes fail, subject to strict constraints and public accountability (max once per year).

**MATL (Mycelix Adaptive Trust Layer)**: A verifiable middleware layer (defined in this Charter) that combines composite trust scoring, cartel detection, and zero-knowledge proofs to enhance Sybil resistance and algorithmic transparency.

**MIP (Mycelix Improvement Proposal)**: A formal proposal to modify operational bylaws, technical standards, or economic models, as defined in Constitution Article IV.

