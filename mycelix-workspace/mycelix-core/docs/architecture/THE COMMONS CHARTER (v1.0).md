# **THE COMMONS CHARTER (v1.1)**

**Companion instrument to the Mycelix Spore Constitution (v0.25) and part of the modular Mycelix Charter Set (v1.1).**

*Updated March 2026: CIV renamed MYCEL, FLOW renamed SAP, CGC absorbed into MYCEL peer recognition — aligned with production implementation*

***Editor's Note:** This Commons Charter (v1.1) is a refactored module of the original THE FEDERATED DAO HIERARCHY CHARTER. It covers peer recognition (formerly CGCs) and the registry of optional commons mechanisms.*

## **ARTICLE I – PEER RECOGNITION (MYCEL Recognition Component)**

*The former Civic Gifting Credit (CGC) primitive has been absorbed into MYCEL's peer recognition component (20% of MYCEL score weight). Recognition events serve the same social-signaling function but feed directly into soulbound reputation rather than existing as a standalone token.*

### **Section 1\. Allocation**

Each verified Member may issue **10 peer recognitions per monthly cycle**. The allocation resets at the end of each cycle.

### **Section 2\. Recognition Mechanics**

a. Minimum MYCEL score of 0.3 required to give recognition.

b. Recognition weight is proportional to the recognizer's own MYCEL score (Sybil-resistant).

c. Members receiving disproportionate recognition volume are flagged in public audit reports.

d. Legitimate high-value contributors (educators, organizers) may appeal flags to Knowledge Council.

### **Section 3\. Sybil Protection**

The Audit Guild shall monitor recognition flows for fraudulent patterns:

a. Circular recognition (A→B→C→A) triggers review.

b. MYCEL penalty for confirmed gaming.

c. Whistleblower rewards for reporting (30% of penalties).

### **Section 4\. Reputation Integration**

Peer recognition constitutes 20% of the MYCEL composite score, alongside Participation (40%), Validation Quality (20%), and Longevity (20%).

### **Section 5\. Cultural Naming**

Local DAOs may adopt cultural aliases for recognition categories while maintaining interoperability:

a. Examples: SPARK, EMBER, PETAL, LIGHT, GRATITUDE.

b. All aliases map to the same underlying MYCEL recognition event type.

c. Symbol Registry (maintained by Knowledge Council) tracks aliases.

## **ARTICLE II – COMMONS MECHANISM REGISTRY**

This article catalogs optional economic modules that DAOs may activate via MIPs.

### **Section 1\. Constitutionally Recognized Core Primitives**

| Name | Code | Type | Function | Symbol |
| :---- | :---- | :---- | :---- | :---- |
| **Soulbound Reputation** | MYCEL | Non-transferable | Reputation, governance weight, peer recognition | 🏛️ |
| **Circulation Medium** | SAP | Transferable | Exchange, demurrage, commons composting | 💧 |

### **Section 2\. Charter-Enabled Optional Commons Modules**

DAOs may elect to activate additional mechanisms via approved MIPs and Eco-OS templates:

| Name | Symbol | Type | Function | Best Fit | Symbol |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Time Exchange** | TEND | Mutual Credit | Skill/service reciprocity | Local DAO/CUT | 🤲 |
| **Stewardship Credits** | ROOT, SEED | SBT/NFT | Bioregeneration, governance labor | Sector/Liminal | 🌱 |
| **Signal Pools** | BEACON, WIND | Soft signal | Non-binding prioritization | Any tier | 🧭 |
| **Commons Hearths** | HEARTH, WELL | Pool/Trust | Custodial resource governance | Local/Global | 🔥 |

### **Section 3\. Cultural Naming Convention**

Each primitive supports cultural aliases enabling local expression while maintaining interoperability:

* **MYCEL recognition aliases**: SPARK, EMBER, PETAL, LIGHT, GRATITUDE
* **MYCEL reputation aliases**: STONE, FOUNDATION, STANDING
* **ROOT aliases**: SEED, FROND, GROWTH  
* **HEARTH aliases**: WELL, CAMPFIRE, COMMONS

**Governance**: The Knowledge Council maintains the official Symbol Registry. DAOs register aliases via MIP-C proposals.

### **Section 4\. Activation Process**

To activate an optional commons module:

1. **Draft MIP** (Technical or Cultural category).  
2. **Define Standards**: Technical spec, interoperability requirements, governance rules.  
3. **Knowledge Council Review**: Ensure compatibility with core infrastructure.  
4. **Audit Guild Review**: Security and economic impact assessment.  
5. **Global DAO Vote**: Simple majority for modules, ⅔ for modifications to core primitives.  
6. **Implementation**: Deploy via standardized hApp template or smart contract.

### **Section 5\. Example: Activating Time Exchange (TEND)**

```

MIP-C-042: Time Exchange Module for Local DAOs

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
- Optional: TEND may influence local MYCEL calculations (max 5% weight)

**Cultural Layer**: DAOs may rename TEND (e.g., "CARE", "HOURS")

**Security**: Audit Guild reviewed (no systemic risk)

**Vote**: Simple majority required

```

## **ARTICLE III – RATIFICATION AND CONTINUITY**

### **Section 1\. Charter Ratification**

This Charter takes full effect upon adoption by the Global DAO and ratification by the federated tiers.

### **Section 2\. Supremacy Clause**

The Mycelix Spore Constitution prevails over this Charter in any case of conflict.

## **APPENDIX A – DEFINITIONS**

*(Editor's Note: Definitions relevant to the Commons Charter. MYCEL and SAP definitions included for context, though primarily governed by the Economic Charter.)*

**MYCEL (Soulbound Reputation)**: A non-transferable reputation score (0.0–1.0) computed from Participation (40%), Peer Recognition (20%), Validation Quality (20%), and Longevity (20%). Serves as foundation for governance weight, tier progression, and fee rates. Includes peer recognition component (formerly CGC) with cultural aliases (SPARK, EMBER, PETAL, LIGHT). Also known by reputation aliases STONE or FOUNDATION.

**SAP (Circulation Medium)**: A transferable token serving as the primary medium of exchange. Subject to 2% annual demurrage; decayed SAP flows to commons pools. Mintable against physical assets.

**Optional Commons Modules**: Additional economic or social mechanisms (e.g., Time Exchange TEND, Stewardship Credits ROOT/SEED) that DAOs can activate via MIP.

## **APPENDIX B – CULTURAL CHARTER TEMPLATE (Optional)**

*(Non-binding)*

Local DAOs may include a cultural section in their charters to define local names and practices related to commons primitives:

Markdown

```

## Cultural Identity (Optional)

### Our Values
[Describe the philosophical or cultural framework guiding this DAO]

### Economic Primitives
We use the following cultural names for Network primitives:
- MYCEL (Reputation): [Cultural name, e.g., "STONE"]
- MYCEL Recognition: [Cultural name, e.g., "EMBER"]
- [Additional commons modules if adopted]

### Rituals and Practices
[Describe any regular ceremonies, decision-making rituals, or cultural practices]

### Constitutional Alignment
All cultural expressions align with Core Principles: [List relevant principles]

```

