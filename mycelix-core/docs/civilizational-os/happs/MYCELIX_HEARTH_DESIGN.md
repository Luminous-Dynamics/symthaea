# Mycelix-Hearth: Media, Publishing & Cultural Commons

## Vision Statement

*"Culture is the fire around which we gather. Hearth provides the infrastructure for creating, sharing, and preserving our collective stories, art, and knowledge - a media commons owned by no one and enriched by everyone."*

---

## Executive Summary

Mycelix-Hearth provides infrastructure for cultural production and sharing:

1. **Publishing** - Decentralized publishing without gatekeepers
2. **Curation** - Community-driven content curation and recommendation
3. **Attribution** - Creator attribution and provenance tracking
4. **Monetization** - Fair compensation models for creators
5. **Preservation** - Cultural archive and commons management

---

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HEARTH ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CREATION LAYER                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Authorship │ Collaboration │ Versioning │ Attribution          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  PUBLISHING LAYER                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Distribution │ Licensing │ Access Control │ Monetization       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  CURATION LAYER                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Community Curation │ Recommendation │ Collections │ Channels   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  COMMONS LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Public Domain │ Creative Commons │ Remix │ Archive             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Data Types

```rust
/// Published work
pub struct Work {
    pub work_id: String,
    pub title: String,
    pub work_type: WorkType,
    pub creators: Vec<Creator>,
    pub content: ContentReference,
    pub metadata: WorkMetadata,
    pub license: License,
    pub access: AccessTerms,
    pub provenance: Provenance,
    pub status: PublicationStatus,
}

pub enum WorkType {
    // Text
    Article,
    Essay,
    Story,
    Poetry,
    Book,
    Newsletter,
    Documentation,

    // Audio
    Music,
    Podcast,
    AudioBook,
    SoundDesign,

    // Visual
    Image,
    Photography,
    Illustration,
    Video,
    Film,
    Animation,

    // Interactive
    Game,
    InteractiveStory,
    Software,

    // Mixed
    Multimedia,
    Performance,

    // Derivative
    Translation { original: ActionHash },
    Adaptation { original: ActionHash },
    Remix { sources: Vec<ActionHash> },
    Commentary { subject: ActionHash },
}

pub struct Creator {
    pub agent: AgentPubKey,
    pub role: CreatorRole,
    pub contribution_percentage: Option<u8>,
    pub pseudonym: Option<String>,
    pub attribution_text: String,
}

pub enum CreatorRole {
    Author,
    Artist,
    Composer,
    Performer,
    Editor,
    Translator,
    Director,
    Producer,
    Contributor,
    Collaborator,
}

pub struct ContentReference {
    pub storage_type: StorageType,
    pub hash: String,
    pub size: u64,
    pub mime_type: String,
    pub preview: Option<String>,
}

pub enum StorageType {
    Holochain,                              // DHT
    IPFS,
    Arweave,
    External { url: String },
}

/// License terms
pub enum License {
    // Open
    PublicDomain,
    CC0,
    CCBY,
    CCBYShare,
    CCBYNonCommercial,
    CCBYNonCommercialShare,

    // Software
    MIT,
    GPL,
    Apache,

    // Custom
    Custom {
        terms: String,
        commercial: bool,
        derivatives: DerivativeTerms,
        attribution: bool,
    },

    // Traditional
    AllRightsReserved,
}

pub enum DerivativeTerms {
    Allowed,
    ShareAlike,
    NonCommercialOnly,
    WithApproval,
    NotAllowed,
}

/// Access and monetization terms
pub struct AccessTerms {
    pub access_type: AccessType,
    pub price: Option<Price>,
    pub subscription: Option<SubscriptionTerms>,
    pub patronage: Option<PatronageInfo>,
    pub pay_what_you_want: bool,
    pub free_tier: Option<FreeTier>,
}

pub enum AccessType {
    Free,
    Paid,
    Subscription,
    Patronage,
    CommunityFunded,
    TimeLocked { free_after: Duration },
}

/// Community curation
pub struct CurationChannel {
    pub channel_id: String,
    pub name: String,
    pub description: String,
    pub curators: Vec<Curator>,
    pub curation_policy: CurationPolicy,
    pub works: Vec<CuratedWork>,
    pub followers: u64,
    pub governance: Option<ActionHash>,
}

pub struct Curator {
    pub agent: AgentPubKey,
    pub role: CuratorRole,
    pub expertise: Vec<String>,
    pub curation_count: u64,
    pub reputation: f64,
}

pub enum CuratorRole {
    Owner,
    Editor,
    Contributor,
    CommunityVoter,
}

pub struct CuratedWork {
    pub work_hash: ActionHash,
    pub added_by: AgentPubKey,
    pub added_at: Timestamp,
    pub curator_note: Option<String>,
    pub community_score: f64,
    pub featured: bool,
}

/// Content moderation
pub struct ModerationAction {
    pub action_id: String,
    pub content_hash: ActionHash,
    pub action_type: ModerationType,
    pub reason: ModerationReason,
    pub moderator: AgentPubKey,
    pub community: ActionHash,
    pub status: ModerationStatus,
    pub appeal: Option<ActionHash>,
}

pub enum ModerationType {
    Flag,
    Hide,
    Remove,
    Warn,
    RestrictDistribution,
    AgeGate,
    ContentWarning(String),
}

pub enum ModerationReason {
    CommunityGuidelines { guideline: String },
    LegalRequirement { jurisdiction: String },
    CopyrightClaim { claimant: AgentPubKey },
    HarmfulContent { harm_type: String },
    Spam,
    Misinformation,
    Other(String),
}
```

---

## Key Workflows

### Workflow 1: Creator Publishing

```
Creator uploads work →
Metadata and attribution defined →
License selected →
Access/monetization terms set →
Work published to DHT →
Optionally archived to Chronicle (M2+) →
Discoverable via search and curation
```

### Workflow 2: Community Curation

```
Curator discovers work →
Adds to channel with note →
Community members vote/engage →
High-quality works rise in visibility →
Creators gain reputation →
Curators gain reputation for good picks
```

### Workflow 3: Collaborative Creation

```
Lead creator initiates collaborative work →
Invites collaborators →
Contributions tracked with attribution →
Revenue share defined via Covenant →
Work published with all creators credited →
Royalties distributed automatically via Treasury
```

### Workflow 4: Remix/Derivative

```
Creator wants to remix existing work →
Original license checked →
If allowed, derivative created →
Provenance links to original →
Original creator credited/compensated per license →
Remix enters commons with linked provenance
```

---

## Moderation Philosophy

### Community-Based Moderation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODERATION MODEL                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRINCIPLES:                                                            │
│  • No global censorship - communities set own standards                │
│  • Transparency - all moderation decisions visible                     │
│  • Appeals process via Arbiter                                         │
│  • Creator rights protected                                            │
│  • Context preserved even when hidden                                  │
│                                                                         │
│  LAYERS:                                                                │
│  1. Creator self-classification (content warnings, etc.)               │
│  2. Community-specific guidelines                                       │
│  3. Legal compliance (per jurisdiction)                                │
│  4. Child safety (universal)                                           │
│                                                                         │
│  ACTIONS:                                                               │
│  • Content warnings (not hiding)                                       │
│  • Community-specific hiding                                           │
│  • Age gating                                                          │
│  • Removal (only for illegal content)                                  │
│  • Context preservation (even removed content has record)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

| Integration | Purpose |
|-------------|---------|
| **Attest** | Creator identity, verified accounts |
| **MATL** | Creator and curator reputation |
| **Chronicle** | Cultural archive, permanent preservation |
| **Loom** | Narrative integration, story weaving |
| **Treasury** | Creator payments, royalties |
| **Covenant** | Collaboration agreements, licensing |
| **Marketplace** | Work sales, commissions |
| **Praxis** | Educational content integration |
| **Agora** | Community governance of channels |
| **Arbiter** | Copyright disputes, moderation appeals |

---

## Creator Economics

### Compensation Models

```rust
pub enum CompensationModel {
    // Direct
    OnePurchase { price: Decimal },
    PayWhatYouWant { minimum: Option<Decimal>, suggested: Option<Decimal> },

    // Recurring
    Subscription { price: Decimal, period: Duration },
    Patronage { tiers: Vec<PatronTier> },

    // Collective
    CommunityFunded { goal: Decimal, deadline: Timestamp },
    Retroactive { pool: ActionHash },      // Quadratic funding style

    // Hybrid
    FreemiumModel {
        free_content: Vec<ActionHash>,
        paid_content: Vec<ActionHash>,
    },

    // Commons
    DonationBased,
    GiftEconomy,
    PublicGood { funded_by: ActionHash },  // Treasury/grant funded
}

pub struct PatronTier {
    pub name: String,
    pub price: Decimal,
    pub period: Duration,
    pub benefits: Vec<PatronBenefit>,
}

pub enum PatronBenefit {
    EarlyAccess,
    ExclusiveContent,
    BehindTheScenes,
    DirectCommunication,
    Credits,
    PhysicalRewards(String),
    Custom(String),
}
```

---

## Conclusion

Hearth provides the infrastructure for a media commons that is owned by no one and enriched by everyone. By separating creation, curation, and distribution, it enables creator sovereignty while building collective cultural wealth.

*"Every story told strengthens the fire. Hearth keeps that fire burning for all."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Attest, MATL, Chronicle, Loom, Treasury, Covenant, Marketplace, Agora, Arbiter*
