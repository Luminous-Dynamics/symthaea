THE EPISTEMIC CHARTER (v2.1)

*Updated March 2026: FLOW renamed SAP — aligned with production implementation*

**Companion instrument to the Mycelix Spore Constitution (v0.25) and part of the modular Mycelix Charter Set (v2.0).**

---

## Editor's Note: v2.0 (The Epistemic Cube Update)

This Charter supersedes v1.0. It represents a fundamental upgrade to the protocol's understanding of "truth."

The 1-dimensional Layered Epistemic Model (LEM) is deprecated. It is replaced by the **3-dimensional Epistemic Cube (LEM v2.0)**, a 3-axis framework that classifies claims based on **Empirical (E-Axis)**, **Normative (N-Axis)**, and **Materiality (M-Axis)** criteria. This new model is capable of classifying all forms of knowledge, from mathematical proofs (E4, N3, M3) and constitutional laws (E0, N3, M3) to scientific theories (E4, N1, M3) and ephemeral messages (E1, N0, M0).

Accordingly, the **Epistemic Claim Schema is upgraded to v2.0** to include these three axes and a formal `related_claims` field to manage the evolution of knowledge (e.g., "superseded" claims). The **Dispute Resolution framework** (Article I, Section 10) has been completely rewritten to be "axis-based," providing separate paths for factual (E-Axis) and governance (N-Axis) challenges.

---

## ARTICLE I – TECHNOLOGY AND INFRASTRUCTURE

### Section 1. Core Infrastructure
All governance infrastructure must be decentralized, open-source, and adhere to Mycelix Interoperability Standards (MIS).

### Section 2. Security
Smart contract upgrades are subject to a 14-day timelock and require Audit Guild attestation.

### Section 3. Consensus Parameter Governance
Critical RB-BFT consensus parameters shall be governed via MIP-T (Technical) proposals, subject to the following minimum requirements:

| Parameter | Initial Value | Change Threshold | Audit Requirement |
|-----------|---------------|------------------|-------------------|
| MIN_REPUTATION_THRESHOLD | 0.4 | ⅔ Global DAO | Mandatory Security Audit |
| Reputation Exponent | 2 (quadratic) | ⅔ Global DAO | Mandatory Simulation & Audit |
| Decay Rate (passive) | 5%/year | Simple Majority | Optional |
| Decay Rate (malicious) | 50% base slash | ⅔ Global DAO | Mandatory Security Audit |
| Ejection Threshold | 0.1 | ⅔ Global DAO | Optional |
| MATL Weights (PoGQ/TCDM/Entropy) | 0.4 / 0.3 / 0.3 | ⅔ Global DAO | Mandatory Simulation & Audit |

### Section 4. Real-Time Cartel Detection (MATL-Driven)
(...This section remains as in v1.0, as MATL's function as a detection engine is unchanged...)

### Section 5. Mycelix Adaptive Trust Layer (MATL)
(...This section remains as in v1.0, as MATL's role as the scoring, ZKP, and DP engine is unchanged. Its inputs will now include N-Axis and M-Axis data, but its function is the same...)

### Section 6. Privacy-Preserving Governance
Members may cast votes using zero-knowledge proofs. This is enforced for high-stakes votes (N-Axis) via Adaptive Differential Privacy (per Section 5.5).

### Section 7. Algorithmic Systems Standards
(...This section remains as in v1.0. All AI agents are "Instrumental Actors" and must be registered and accountable...)

### Section 8. Epistemic Ledger Requirements
a. All governance-significant claims must be submitted via a standardized **Epistemic Claim Object v2.0** (see Appendix B).
b. All DKG nodes and clients must validate claims against this schema.
c. The **Decentralized Knowledge Graph (DKG)** shall serve as the primary, append-only graph ledger for all claims with `epistemic_tier_m` (Materiality) of **M2 (Persistent)** or **M3 (Foundational)**.
d. **M0 (Ephemeral)** and **M1 (Temporal)** claims may be pruned by the DHT/DKG network according to the rules defined in the Mycelix Protocol: Integrated System Architecture.

### Section 9. Core Registries
The Network shall maintain a set of core registries, including:

a. **The Epistemic Claim Registry (DKG)**: An append-only, 3-dimensional, queryable graph containing all persistent and foundational claims.
b. **The Trust Tag Ontology**: A versioned dictionary of validation standards managed by the Knowledge Council.
c. **The Claim Lineage Map**: The graph relationships (`related_claims` field) for tracing the evolution of knowledge (e.g., claims that SUPPORT, REFUTE, or SUPERCEDE others).

### Section 10. Epistemic Dispute Resolution Protocol (v2.0)

Disputes are no longer categorized by "materiality" (which is now an axis, M-Tier). Disputes are categorized by **which axis of truth is being challenged**.

#### 10.1 E-Axis Disputes (Challenges of Fact)
**Definition**: A challenge to a claim's Empirical (E-Axis) coordinate.

**Examples**: "This E1 testimony is false," "This E3 VC is a forgery," "This E4 data is fraudulent," "This product is counterfeit."

**Routing**: All E-Axis disputes are routed to the **Member Redress Council (MRC)** for binding adjudication. The MRC may commission the Audit Guild for technical analysis.

**Outcome**: If the dispute succeeds, the MRC publishes a new, binding E2 claim that marks the original claim's status as `"ResolvedDisputeInvalidated"`. This triggers automatic reputation and stake slashing for the original submitter (per the Economic Charter).

#### 10.2 N-Axis Disputes (Challenges of Law)
**Definition**: A challenge to a claim's Normative (N-Axis) authority or its compliance with a higher N-Tier law.

**Examples**: "This N1 (Communal) DAO vote violated its own charter," "This N2 (Network) MIP is unconstitutional because it violates an N3 (Axiomatic) Core Principle."

**Routing (by Tier)**:
- **vs. N1 (Communal)**: Handled by the Member Redress Council (MRC), which acts as the judiciary.
- **vs. N2 (Network)**: Handled by the Member Redress Council (MRC), which reviews for constitutional compliance.
- **vs. N3 (Axiomatic)**: This is a **Constitutional Challenge**. The MRC can review it, but a final ruling requires the full Constitutional Amendment process (per Constitution Art. IV, Sec 1).

**Outcome**: If the dispute succeeds, the MRC or Global DAO publishes a new claim that invalidates the offending law/proposal (e.g., `status: "ResolvedNormativeConflict"`).

#### 10.3 M-Axis Disputes (Challenges of State)
**Definition**: A challenge to a claim's Materiality (M-Axis) classification, typically for state management.

**Examples**: "This M1 (Temporal) claim should be M3 (Foundational) and must not be pruned," or "This M3 claim is obsolete and should be downgraded to M2 (Persistent)."

**Routing**: All M-Axis disputes are routed to the **Knowledge Council** for a binding ruling on archival policy.

#### 10.4 Scientific "Disputes" (The Evolution of Knowledge)
**Definition**: A challenge that does not allege fraud or illegality, but posits a better model. (e.g., Einstein vs. Newton).

**Process**: This is not a formal dispute. The new claim is submitted with a `related_claims` type of `SUPERCEDES`.

**Routing**: The claim is routed to the relevant Sector DAO (e.g., "Physics DAO") for N1 (Communal) consensus.

**Outcome**: If the Sector DAO votes to accept the new claim, the DKG links the two. The original claim's status is graphically updated to `status: "SUPERCEDED_BY: [new_claim_id]"`. No slashing occurs; this is seen as a healthy evolution of knowledge.

#### 10.5 Bond & Deadlock Protocols
(...Bonding, deadlock, and outcome recording rules remain as in v1.0, but are applied to the new E/N/M dispute flows...)

---

## ARTICLE II – RATIFICATION AND CONTINUITY

### Section 1. Charter Ratification
This Charter (v2.0) takes full effect upon adoption by the Global DAO and ratification by the federated tiers.

### Section 2. Supremacy Clause
The Mycelix Spore Constitution (v0.25+) prevails over this Charter in any case of conflict.

---

## APPENDIX A – THE EPISTEMIC CUBE (LEM v2.0)

*(This appendix replaces the v1.0 LEM)*

The **Layered Epistemic Model (LEM) v2.0** classifies all claims along three independent axes, forming the "Epistemic Cube." A claim's coordinate (e.g., E4, N1, M3) defines its nature, authority, and permanence.

### Axis 1: E-Axis (Empirical Verifiability)
**Question**: How do we know this claim is true?

| Tier | Name | Description | Verification Method | Example |
|------|------|-------------|---------------------|---------|
| E0 | Null | An unverifiable assertion of belief or opinion. | None Possible | "I believe this song is bad." "The sky should be green." |
| E1 | Testimonial | A claim of personal experience or observation, bound to a user's DID. | Member Signature | "I, did:alice, attest I received this product." "I saw this event." |
| E2 | Privately Verifiable | A claim verified by an authorized, high-trust 3rd party (like the Audit Guild), where the raw data remains private. | Audit Guild Attestation | "The Audit Guild attests this DAO's (private) books are balanced." |
| E3 | Cryptographically Proven | A claim whose truth is proven by a mathematical algorithm, without revealing all data. | ZKP / Signature Validation | "This VC proves I am over 18." "This MATL score was calculated correctly." |
| E4 | Publicly Reproducible | The highest standard. A claim whose truth can be independently reproduced by any agent with the provided open data and open code. | Public Data / Open Code | "Running this zerotrustml_paper.pdf code on this dataset proves the 100% detection rate." |

### Axis 2: N-Axis (Normative Authority)
**Question**: Who agrees this claim is binding?

| Tier | Name | Description | Verification Method | Example |
|------|------|-------------|---------------------|---------|
| N0 | Personal (A-Normative) | A claim that is not a "law" and is only binding on the individual who made it. | Self-Attestation | "I am selling this bike." (A testimonial) |
| N1 | Communal Consensus | A law or rule agreed upon by a specific, local community (a "Hearth" or Local/Sector/Regional DAO). | Verified DAO Vote | "The Aura Music DAO agrees that artist royalties are 70%." |
| N2 | Network Consensus | A binding law (MIP) that applies to the entire Mycelix Network. | Verified Global DAO Vote | "The MIP-E-001 (Validator Economics) is now binding network law." |
| N3 | Axiomatic | A foundational, "Platonic" truth of the system. This includes both the Constitution and pure logic/mathematics. | Constitutional Reference / Logical Proof | "The 'Right to Privacy' is a Core Principle." "2 + 2 = 4." |

### Axis 3: M-Axis (Materiality / Modality)
**Question**: Why, where, and for how long does this claim matter?

| Tier | Name | Description | State Management | Example |
|------|------|-------------|------------------|---------|
| M0 | Ephemeral | A transient signal with no lasting value. | Discard. Not stored on DKG. | "This user's cursor is here." "A 'like'." A P2P network ACK packet. |
| M1 | Temporal (Routine) | A claim that is only valid until its state changes. | Prune. Stored on DHT, pruned after state change. | "This Mycelix Mail message is 'unread'." "This Agora listing is 'active'." |
| M2 | Persistent (Significant) | A claim that must be archived for auditing and reputation, but is not foundational. | Archive (Cold Storage). Moved from DKG to IPFS/Filecoin after (e.g.) 1 year. | "This Agora sale was completed." "This DAO vote passed." "Your MATL score from last month." |
| M3 | Foundational (Permanent) | A claim that is "constitutional" or "existential" and must never be pruned. | Preserve. Held in permanent DKG/validator state. | "This Aura song is my original work." "This MIP is law." "The Constitution." |

---

## APPENDIX B – EPISTEMIC CLAIM SCHEMA STANDARD v2.0

*(This appendix replaces the v1.1 schema)*

This JSON Schema defines the mandatory structure for all claims published to the DKG.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Mycelix Epistemic Claim Schema v2.0",
  "type": "object",
  "properties": {
    "claim_id": {
      "description": "Unique UUIDv4 for the claim.",
      "type": "string",
      "format": "uuid"
    },
    "claim_hash": {
      "description": "SHA-256 hash of the 'content' object (canonicalized JSON).",
      "type": "string",
      "pattern": "^[a-f0-9]{64}$"
    },
    "submitted_by_did": {
      "description": "The Mycelix DID of the agent submitting the claim.",
      "type": "string",
      "format": "uri",
      "pattern": "^did:mycelix:"
    },
    "submitter_type": {
      "description": "Mandatory declaration of agent type, per Constitution Art. VI, Sec 14.",
      "type": "string",
      "enum": ["HumanMember", "InstrumentalActor", "DAOCollective"]
    },
    "timestamp": {
      "description": "ISO 8601 UTC timestamp of claim submission.",
      "type": "string",
      "format": "date-time"
    },
    "epistemic_tier_e": {
      "description": "E-Axis: The Empirical verifiability of the claim.",
      "type": "string",
      "enum": ["E0", "E1", "E2", "E3", "E4"]
    },
    "epistemic_tier_n": {
      "description": "N-Axis: The Normative authority/consensus backing the claim.",
      "type": "string",
      "enum": ["N0", "N1", "N2", "N3"]
    },
    "epistemic_tier_m": {
      "description": "M-Axis: The Materiality/permanence of the claim (for state pruning).",
      "type": "string",
      "enum": ["M0", "M1", "M2", "M3"]
    },
    "claim_type": {
      "description": "The functional purpose of the claim.",
      "type": "string",
      "enum": ["Assertion", "Testimony", "Report", "Proposal", "Vote", "Dispute", "Ruling", "VC", "ProductListing", "MediaPublication", "Message"]
    },
    "content": {
      "description": "The immutable body of the claim. Hashed in 'claim_hash'.",
      "type": "object",
      "properties": {
        "format": { "type": "string", "enum": ["application/json", "text/markdown", "text/plain", "application/ld+json"] },
        "body": { "description": "The claim's core data, as a JSON-escaped string or nested object.", "type": ["string", "object"] },
        "attachments": {
          "description": "Array of links to 'Cold Storage' (IPFS/Filecoin) for large files.",
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "cid": { "type": "string" },
              "mime_type": { "type": "string" },
              "filename": { "type": "string" }
            },
            "required": ["cid"]
          }
        }
      },
      "required": ["format", "body"]
    },
    "verifiability": {
      "description": "Metadata on how to verify the claim's E-Tier.",
      "type": "object",
      "properties": {
        "method": { "type": "string", "enum": ["None", "Signature", "ZKProof", "AuditGuildReview", "PublicCode"] },
        "status": { "type": "string", "enum": ["Unverified", "Verified", "Pending", "Disputed", "ResolvedDisputeInvalidated", "Superseded"] },
        "proof_cid": { "description": "Link to ZKP or other proof data in cold storage.", "type": "string" }
      },
      "required": ["method", "status"]
    },
    "related_claims": {
      "description": "The 'Graph' part of the DKG. Links this claim to others.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "relationship_type": {
            "type": "string",
            "enum": ["SUPPORTS", "REFUTES", "CLARIFIES", "CONTEXTUALIZES", "SUPERCEDES", "REFERENCES", "AUTHENTICATES"]
          },
          "related_claim_id": { "type": "string", "format": "uuid" },
          "context": { "type": "string", "description": "Optional note, e.g., 'Is a low-velocity approximation of this model.'" }
        },
        "required": ["relationship_type", "related_claim_id"]
      }
    }
  },
  "required": [
    "claim_id",
    "claim_hash",
    "submitted_by_did",
    "submitter_type",
    "timestamp",
    "epistemic_tier_e",
    "epistemic_tier_n",
    "epistemic_tier_m",
    "claim_type",
    "content",
    "verifiability"
  ]
}
```

---

## APPENDIX C – CLAIM CLASSIFICATION EXAMPLES (v2.0)

This table provides examples of how the Epistemic Cube (LEM v2.0) classifies common data types.

| Claim Type | Content | E-Tier | N-Tier | M-Tier | (E, N, M) | Rationale |
|------------|---------|--------|--------|--------|-----------|-----------|
| A "Like" / Emote | "User 'Alice' liked post 'X'" | E0 | N0 | M0 | (E0, N0, M0) | An unverifiable, personal, and transient signal. Not stored on DKG. |
| Mycelix Mail Message | "To: Bob, From: Alice..." | E1 | N0 | M1 | (E1, N0, M1) | A personal testimony from Alice. It is not a law. It is temporary and can be pruned by the DHT once delivered and confirmed by Bob. |
| Agora Listing | "Selling this bike for 50 SAP" | E1 | N1 | M1 | (E1, N1, M1) | A personal testimony from the seller. It is binding only on the seller (a "communal" promise). It is temporary (active until sold). |
| Agora Sale | "Sale X completed for 50 SAP" | E2 | N1 | M2 | (E2, N1, M2) | Verified by the escrow/payment system. It's a binding community record. It must be persisted for reputation/auditing, but not forever. |
| Aura Song Copyright | "I, did:artist, am the author of song X" | E3 | N1 | M3 | (E3, N1, M3) | Proven by the artist's digital signature. It's a binding law for royalty splits within the "Music DAO" community. It is permanent. |
| Mathematical Proof | "2 + 2 = 4" | E4 | N3 | M3 | (E4, N3, M3) | The highest form of truth. It can be reproduced by anyone, is axiomatically binding on all, and is permanent. |
| Mycelix Constitution | "The Core Principle of Subsidiarity" | E0 | N3 | M3 | (E0, N3, M3) | A foundational, permanent "belief." It is not empirically verifiable, but it is axiomatically binding on the entire network. |
| Newtonian Physics | "F=ma" | E4 | N1 | M3 | (E4, N1, M3) | It can be publicly reproduced (in its context). It was the N1 (Communal) consensus of the "Physics DAO." It was not N3 (Axiomatic), as it was a model, not a definition. |
| Einstein's Theory | "General Relativity" | E4 | N1 | M3 | (E4, N1, M3) | Same as Newton, but it also includes a `related_claims` field that SUPERCEDES Newton's, providing the DKG with the graph of scientific evolution. |

---

## APPENDIX D – DEFINITIONS

*(Editor's Note: This appendix includes only definitions relevant to the Epistemic Charter v2.0.)*

**Epistemic Cube (LEM v2.0)**: The 3-axis framework (Empirical, Normative, Materiality) used to classify all claims on the network.

**E-Axis (Empirical)**: The axis that defines a claim's verifiability and factual basis.

**N-Axis (Normative)**: The "Platonic" axis that defines a claim's legal, ethical, or consensual authority.

**M-Axis (Materiality)**: The axis that defines a claim's relevance over time, used for state pruning and scalability.

**Epistemic Dispute**: A formal challenge to a claim's coordinate on one or more axes (e.g., an E-Axis dispute for fraud, an N-Axis dispute for a constitutional violation).

**Epistemic Claim (v2.0)**: The standardized data object (per Appendix B) that represents a piece of information on the DKG, classified with a 3-axis (E, N, M) coordinate.

**MATL (Mycelix Adaptive Trust Layer)**: A verifiable middleware layer (defined in this Charter) that combines composite trust scoring, cartel detection, and zero-knowledge proofs to enhance Sybil resistance and algorithmic transparency. Its inputs include claim classifications from the Epistemic Cube.

**MIP (Mycelix Improvement Proposal)**: A formal proposal to modify operational bylaws, technical standards, or economic models. A passed MIP is an (E0, N2, M3) claim.

---

**Version**: 2.0
**Status**: Current
**Published**: November 2025
**Supersedes**: [Epistemic Charter v1.0](./THE%20EPISTEMIC%20CHARTER%20(v1.0).md)
**Related Documents**:
- [Mycelix Spore Constitution v0.24](./THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
- [Governance Charter v1.0](./THE%20GOVERNANCE%20CHARTER%20(v1.0).md)
- [Economic Charter v1.0](./THE%20ECONOMIC%20CHARTER%20(v1.0).md)
- [Commons Charter v1.0](./THE%20COMMONS%20CHARTER%20(v1.0).md)

---

📍 **Navigation**: [← Commons Charter](./THE%20COMMONS%20CHARTER%20(v1.0).md) | [↑ Charters Index](../README.md#constitutional-framework-the-four-charters) | [Constitution →](./THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
