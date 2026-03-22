# **COMMONS CHARTER: COMPOSTING EXTENSION (v1.1)**

*Updated March 2026: CGC renamed to MYCEL recognition, CIV renamed to MYCEL — aligned with production implementation*

**Amendment to the Commons Charter (v1.0) under the Living Protocol Initiative**

---

## Editor's Note

This document specifies the **Composting Protocol** as an extension to the Commons Charter (v1.0), adding a formal mechanism for gracefully retiring failed patterns, outdated knowledge, and deprecated structures.

Composting joins the existing Commons mechanisms (MYCEL recognition, TEND, ROOT, BEACON, HEARTH) as an optional module that DAOs may activate via MIP-C proposal.

**Constitutional Alignment**: Constitution Article I, Section 2 (Adaptability, Epistemic Humility), Epistemic Charter v2.0 (claim lifecycle).

---

## **ARTICLE III – COMPOSTING PROTOCOL**

### **Section 1. Purpose and Philosophy**

**1.1 Definition**: Composting is the process of gracefully retiring entities (patterns, claims, structures, relationships) that no longer serve, while extracting their value for reuse.

**1.2 Philosophical Basis**: Recognizing that:
- Living systems transform waste into nutrients
- Failed experiments contain valuable lessons
- Clinging to outdated structures creates burden
- Endings, properly honored, enable beginnings

**1.3 Purpose**: The Composting Protocol enables:
- Graceful deprecation of outdated claims (DKG)
- Retirement of failed governance patterns
- Dissolution of defunct relationships and agreements
- Harvesting of lessons from failures

**1.4 Key Principle**: **Composting is not deletion.** Composted entities remain in the historical record. Composting changes their status and extracts their value.

### **Section 2. Compostable Entity Types**

#### **2.1 Entity Classification**

| Entity Type | Examples | Composting Outcome |
|-------------|----------|-------------------|
| **Knowledge Claims** | DKG assertions, predictions | Status → "Superseded"; links preserved |
| **Governance Patterns** | Failed policies, deprecated processes | Lessons extracted; archived |
| **Relationships** | Dissolved partnerships, ended agreements | History preserved; active links removed |
| **Code/Structures** | Deprecated contracts, old schemas | Components catalogued for reuse |
| **Proposals** | Rejected or abandoned MIPs | Failure analysis published |
| **Organizations** | Dissolved DAOs, ended projects | Assets distributed; legacy recorded |

#### **2.2 Non-Compostable Entities**

The following cannot be composted:
- Constitutional provisions (require amendment process)
- Active wounds in healing (must complete first)
- Entities under dispute (must resolve first)
- Core identity records (subject to GDPR erasure instead)

### **Section 3. Composting Phases**

#### **3.1 Initiation Phase**

**Purpose**: Propose and validate composting.

**Requirements**:
- Composting proposal submitted by authorized party
- Justification documented
- Stakeholder notification (7-day notice for significant entities)

**Authorized Initiators**:
| Entity Type | Who May Initiate |
|-------------|------------------|
| Personal claims | Claim author |
| DAO proposals | Original proposer or DAO governance |
| Relationships | Any party to the relationship |
| Organizations | DAO supermajority vote |
| Code | Maintainers or Audit Guild |

**Outputs**:
- Composting Proposal Record
- Stakeholder notification log
- Objection window opened

#### **3.2 Objection Window**

**Duration**: 7 days (standard) / 1 day (expedited for minor entities)

**Objection Grounds**:
- Entity still in active use
- Premature composting (value not yet extracted)
- Procedural violation
- Stakeholder harm

**Objection Process**:
1. Objection filed with justification
2. Initiator may respond
3. If unresolved, escalate to relevant body:
   - Knowledge Claims → Knowledge Council
   - Governance → Inter-Tier Arbitration Council
   - Relationships → Member Redress Council

#### **3.3 Decomposition Phase**

**Purpose**: Break down entity; extract value.

**Duration**: 7-28 days depending on complexity

**Activities**:
- Value extraction (lessons, components, patterns)
- Dependency analysis (what depends on this entity?)
- Migration planning (how to update dependents)
- Documentation of extracted value

**Outputs**:
- Extracted Value Manifest
- Dependency Migration Plan
- Decomposition Report

#### **3.4 Mineralization Phase**

**Purpose**: Complete transformation; make nutrients available.

**Activities**:
- Entity status updated to "Composted"
- Extracted values published to commons
- Dependencies migrated or notified
- Composting ceremony (optional)

**Outputs**:
- Final Composting Record
- Lessons Learned entry (Wisdom Library)
- Component Catalog entries (if applicable)
- Successor references updated

### **Section 4. Value Extraction**

#### **4.1 Extractable Value Types**

| Value Type | Description | Destination |
|------------|-------------|-------------|
| **Lessons** | What we learned from failure | Wisdom Library |
| **Components** | Reusable code, templates, patterns | Component Catalog |
| **Relationships** | Contact networks, trust connections | Relationship Archive |
| **Data** | Historical records, metrics | DKG Archive |
| **Resources** | Remaining funds, assets | Designated successor or treasury |

#### **4.2 Extraction Requirements**

For entities with Materiality M2 (Persistent) or higher:
- At least one Lesson must be extracted
- Extraction requires at least 2 contributors
- Knowledge Council may audit extraction quality

#### **4.3 Extraction Incentives**

Contributors to value extraction receive:
- MYCEL peer recognition (10 recognition events per quality extraction)
- MYCEL bonus (+0.01 per verified lesson)
- Author credit in Wisdom Library

### **Section 5. Composting and the DKG**

#### **5.1 Epistemic Claim Composting**

When DKG claims are composted:

```rust
pub struct CompostedClaim {
    pub original_claim: EpistemicClaim,
    pub composted_at: Timestamp,
    pub composted_by: DID,
    pub reason: CompostingReason,

    // Preservation
    pub successor_claims: Vec<ClaimId>,  // What supersedes this
    pub extracted_lessons: Vec<LessonId>,

    // Status
    pub status: ClaimStatus::Composted,
    pub searchable: bool,  // Still queryable with filter
}

pub enum CompostingReason {
    Superseded { by: ClaimId },
    Invalidated { evidence: ContentHash },
    Obsolete { explanation: String },
    Retracted { by_author: bool },
    Merged { into: ClaimId },
}
```

#### **5.2 Claim Lifecycle Integration**

```
Created → Verified → [Active] → Composted
                         ↓
                    Disputed → Resolved → [Active or Composted]
```

Composted claims:
- Remain in DKG with "Composted" status
- Excluded from default queries
- Included in historical/lineage queries
- Preserve all provenance links

### **Section 6. Composting and Metabolism**

Composting aligns with the Metabolism Cycle:

| Phase | Composting Activity |
|-------|---------------------|
| **Release** | Composting initiation encouraged |
| **Stillness** | Objection window; reflection |
| **Creation** | Value extraction; lesson writing |
| **Integration** | Mineralization; completion |

**Phase Bonuses**:
- Initiation during Release: +10% MYCEL recognition bonus
- Extraction during Creation: +10% MYCEL recognition bonus
- Completion during Integration: +10% MYCEL recognition bonus

### **Section 7. Governance Integration**

#### **7.1 Failed Proposal Composting**

MIPs that fail are automatically queued for composting:

| Failure Type | Composting Trigger |
|--------------|-------------------|
| Rejected (voted down) | Immediate |
| Abandoned (no vote after 2 cycles) | Automatic |
| Withdrawn (by proposer) | Upon withdrawal |
| Superseded (by newer MIP) | Upon supersession |

**Required Extraction**: Proposer must submit failure analysis or forfeit proposal bond.

#### **7.2 DAO Dissolution Composting**

When a DAO dissolves (Governance Charter Article XII):

1. **Initiation**: Dissolution vote triggers composting
2. **Decomposition**:
   - Charter lessons extracted
   - Member histories preserved
   - Asset distribution executed
3. **Mineralization**:
   - DAO Record archived
   - Lessons published
   - Successor DAO (if any) linked

### **Section 8. Technical Implementation**

#### **8.1 Composting Record Schema**

```rust
pub struct CompostingRecord {
    pub composting_id: Uuid,
    pub entity_type: EntityType,
    pub entity_id: String,

    // Lifecycle
    pub initiated_at: Timestamp,
    pub initiated_by: DID,
    pub phase: CompostingPhase,
    pub completed_at: Option<Timestamp>,

    // Justification
    pub reason: CompostingReason,
    pub justification: String,

    // Objections
    pub objection_deadline: Timestamp,
    pub objections: Vec<Objection>,
    pub objection_resolution: Option<ObjectionResolution>,

    // Value extraction
    pub extracted_values: Vec<ExtractedValue>,
    pub extractors: Vec<DID>,

    // Outputs
    pub lessons_published: Vec<LessonId>,
    pub components_catalogued: Vec<ComponentId>,
    pub successor_entities: Vec<EntityReference>,
}

pub enum CompostingPhase {
    Initiated,
    ObjectionWindow,
    Decomposing,
    Mineralized,
    Cancelled,  // If objection sustained
}

pub struct ExtractedValue {
    pub value_type: ValueType,
    pub title: String,
    pub description: String,
    pub content_hash: ContentHash,
    pub extracted_by: DID,
    pub extracted_at: Timestamp,
    pub quality_score: Option<f64>,  // KC assessment
}
```

#### **8.2 Query Interface**

```rust
pub trait CompostingQueries {
    /// Get all composted entities of a type
    fn get_composted(entity_type: EntityType) -> Vec<CompostingRecord>;

    /// Get composting history for an entity
    fn get_composting_history(entity_id: &str) -> Option<CompostingRecord>;

    /// Get lessons extracted from composting
    fn get_lessons_from(composting_id: Uuid) -> Vec<Lesson>;

    /// Check if entity is composted
    fn is_composted(entity_id: &str) -> bool;

    /// Get successors of composted entity
    fn get_successors(entity_id: &str) -> Vec<EntityReference>;
}
```

### **Section 9. Revival Protocol**

Composted entities may be revived under exceptional circumstances:

#### **9.1 Revival Window**

- **Standard entities**: 28 days post-mineralization
- **Significant entities (M2+)**: 90 days post-mineralization
- **After window**: Revival requires new creation (referencing composted original)

#### **9.2 Revival Requirements**

- Petition by original author or 10+ members
- Justification for revival
- Knowledge Council review
- Simple majority vote (relevant DAO)

#### **9.3 Revival Outcome**

If revived:
- Status changed to "Revived"
- Composting record preserved (historical)
- Extracted values remain in commons
- Revival reason documented

### **Section 10. Metrics and Reporting**

The Knowledge Council tracks:

| Metric | Description |
|--------|-------------|
| Composting Rate | Entities composted per cycle |
| Value Extraction Quality | Average quality score of extracted lessons |
| Revival Rate | Percentage of composted entities revived |
| Objection Rate | Percentage of compostings objected |
| Cycle Alignment | Percentage of composting aligned with Metabolism phases |

Published quarterly in Network Resilience Report.

---

## **APPENDIX A – CEREMONY TEMPLATE**

### **Composting Ceremony (Optional)**

Held during Release phase for significant compostings:

**Opening**:
"We gather to honor what has served us and is now complete."

**Acknowledgment**:
- Name the entity being composted
- Acknowledge its contributions
- Thank those who created/maintained it

**Lesson Sharing**:
- Extractors share lessons learned
- Community reflects on what was valuable

**Release**:
"We release this [entity] with gratitude. May its nutrients feed new growth."

**Closing**:
- Moment of silence
- Commitment to apply lessons

---

## **APPENDIX B – COMMONS MECHANISM REGISTRY UPDATE**

Add to Commons Charter Article II, Section 2:

| Name | Symbol | Type | Function | Best Fit | Symbol |
|------|--------|------|----------|----------|--------|
| **Composting** | MULCH | Process | Graceful retirement; value extraction | Any tier | 🍂 |

**Cultural Aliases**: MULCH, HUMUS, RETURN, CYCLE, RELEASE

---

**Version**: 1.0
**Status**: Draft for Ratification
**Published**: February 2026
**Amends**: Commons Charter v1.0 (Article III)
**Related Documents**:
- [Commons Charter v1.0](./THE%20COMMONS%20CHARTER%20(v1.0).md)
- [Metabolism Charter v1.0](./THE%20METABOLISM%20CHARTER%20(v1.0).md)
- [Epistemic Charter v2.0](./THE%20EPISTEMIC%20CHARTER%20(v2.0).md)

---

*"Nothing is wasted. Everything transforms."*
