# **THE COMMONS CHARTER (v1.0)**

**Companion instrument to the Mycelix Spore Constitution (v0.24) and part of the modular Mycelix Charter Set (v1.0).**

***Editor's Note:** This Commons Charter (v1.0) is a refactored module of the original THE FEDERATED DAO HIERARCHY CHARTER (v0.24). It isolates articles and appendices pertaining to Civic Gifting Credits (CGCs) and the registry of optional commons mechanisms (formerly Article XI.3.b and Appendix G). All information is sourced directly from the v0.24 Charter.*

## **ARTICLE I – CIVIC GIFTING CREDITS (CGCs)**

### **Section 1\. Allocation**

Each verified Member receives **10 CGCs per month**. These credits are non-cumulative and expire at the end of each monthly cycle.

### **Section 2\. Transfer Transparency**

Rather than hard caps, CGCs use transparency thresholds:

a. No hard limits on CGC transfers (preserves flexibility).

b. Members receiving \>100 CGCs/month OR \>50 from single source flagged in public "High-Activity CGC Report".

c. Audit Guild reviews flagged accounts quarterly for gaming patterns.

d. Legitimate high-value contributors (educators, organizers) may appeal flag to Knowledge Council with explanation.

### **Section 3\. Sybil Protection**

The Audit Guild shall monitor CGC flows for fraudulent patterns:

a. Circular gifting (A→B→C→A) triggers review.

b. Reputation penalty for confirmed gaming.

c. Whistleblower rewards for reporting (30% of penalties).

### **Section 4\. Reputation Integration**

Net CGC inflow may be used as one discretionary input (with a maximum weight of 10%) in local reputation calculations, as determined by Local DAO policy.

### **Section 5\. Cultural Naming**

Local DAOs may adopt cultural aliases for CGCs while maintaining interoperability:

a. Examples: SPARK, EMBER, PETAL, LIGHT, GRATITUDE.

b. All aliases map to same underlying CGC primitive.

c. Symbol Registry (maintained by Knowledge Council) tracks aliases.

## **ARTICLE II – COMMONS MECHANISM REGISTRY**

This article catalogs optional economic modules that DAOs may activate via MIPs.

### **Section 1\. Constitutionally Recognized Core Primitives**

| Name | Code | Type | Function | Symbol |
| :---- | :---- | :---- | :---- | :---- |
| **Civic Standing** | CIV | Non-transferable | Reputation, governance weight | 🏛️ |
| **Civic Gifting Credit** | CGC | Non-transferable | Social signal, gratitude | ✨ |
| **Utility Token** | FLOW | Transferable (optional) | Fees, staking, compute | 💧 |

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

* **CGC aliases**: SPARK, EMBER, PETAL, LIGHT, GRATITUDE  
* **CIV aliases**: STONE, FOUNDATION, STANDING  
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
- Optional: TEND may influence local CIV calculations (max 5% weight)

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

*(Editor's Note: Definitions relevant to the Commons Charter. CIV and FLOW definitions included for context, though primarily governed by the Economic Charter.)*

**Civic Standing (CIV)**: A non-transferable reputation score representing a Member's verifiable contributions, expertise, and trustworthiness within the Network. CIV serves as the foundation for governance weight and validator selection. Also known by cultural aliases such as STONE or FOUNDATION.

**Civic Gifting Credit (CGC)**: A non-monetary, non-transferable signaling primitive used to recognize non-market contributions within the Network. Also known by cultural aliases such as SPARK, EMBER, PETAL, LIGHT.

**Utility Token (FLOW)**: A transferable token (optional activation) used for fees, staking, compute, and other economic functions separate from governance power.

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
- CIV (Civic Standing): [Cultural name, e.g., "STONE"]
- CGC (Civic Gifting Credit): [Cultural name, e.g., "EMBER"]
- [Additional commons modules if adopted]

### Rituals and Practices
[Describe any regular ceremonies, decision-making rituals, or cultural practices]

### Constitutional Alignment
All cultural expressions align with Core Principles: [List relevant principles]

```

