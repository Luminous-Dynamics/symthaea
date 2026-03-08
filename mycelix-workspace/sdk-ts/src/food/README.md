# Mycelix Food Sovereignty Module

**Status**: Alpha
**Version**: 0.1.0

Food sovereignty through decentralized infrastructure. This module is NOT a new hApp—it's a configuration layer unifying existing Mycelix modules for food systems.

## The Five Pillars

| Pillar | Underlying Module | Purpose |
|--------|-------------------|---------|
| **TEND** | Finance | CSA subscriptions, mutual credit, farm finance |
| **NOURISH** | Health | Allergies, nutrition, food-drug interactions |
| **CULTIVATE** | SupplyChain | Farm-to-table provenance, regenerative verification |
| **STEWARD** | Property | Land commons, food forests, community gardens |
| **WISDOM** | Knowledge | Seed library, growing guides, traditional knowledge |

## Quick Start

```typescript
import { getFoodService } from '@mycelix/sdk/food';

const food = getFoodService();

// Find local farms
const farms = food.findFarms({
  practices: ['no_till', 'cover_crop']
});

// Subscribe to CSA
const share = food.subscribeToCSA({
  farmId: farms[0].id,
  memberDid: 'did:mycelix:member-123',
  shareType: 'half',
  currency: 'TEND',
  startWeek: 12,
  weeks: 20,
  pickupLocationId: 'pickup-1'
});

// Check allergen safety
const check = food.checkAllergens('tomato-batch-123', 'did:mycelix:member-123');
if (!check.safe) {
  console.log('Allergen warning:', check.recommendations);
}

// Get full provenance
const provenance = food.getProvenance('tomato-batch-123');
console.log(`Organic: ${provenance.organicVerified}`);
console.log(`Food miles: ${provenance.foodMiles}km`);
console.log(`Freshness: ${(provenance.freshnessScore * 100).toFixed(0)}%`);
```

## TEND - CSA Economics

Community-Supported Agriculture with mutual credit.

```typescript
// Member subscribes
const share = food.subscribeToCSA({
  farmId: 'farm-1',
  memberDid: 'did:mycelix:member-123',
  shareType: 'full',      // full, half, quarter, custom
  currency: 'TEND',       // Interest-free mutual credit
  startWeek: 12,
  weeks: 20,
  pickupLocationId: 'community-center-1'
});

// Get member's shares
const myShares = food.getMemberShares('did:mycelix:member-123');
```

**Why TEND?** Interest-free mutual credit means farmers get value NOW, members pay back through the season. No bank extraction.

## NOURISH - Food as Medicine

Allergy management and nutrition integration with Health hApp.

```typescript
// Set up food profile
food.setFoodProfile({
  did: 'did:mycelix:member-123',
  allergies: [
    { id: 'a1', allergen: 'peanuts', severity: 'anaphylactic' },
    { id: 'a2', allergen: 'tree_nuts', severity: 'moderate' }
  ],
  medications: ['lisinopril', 'metformin'],
  dietaryRestrictions: [
    { type: 'vegetarian' }
  ]
});

// Check if food is safe
const result = food.checkAllergens('peanut-sauce-batch', 'did:mycelix:member-123');

// result.safe = false
// result.allergens = [{ allergen: 'peanuts', present: true, ... }]
// result.drugInteractions = [{ medication: 'lisinopril', food: 'high potassium', ... }]
```

## CULTIVATE - Farm-to-Table Provenance

Full supply chain tracking with regenerative verification.

```typescript
// Record farm checkpoint
food.recordFarmCheckpoint({
  farmId: 'farm-1',
  productId: 'tomato-batch-2026-001',
  cropType: 'tomato',
  variety: 'Cherokee Purple',
  practicePhase: 'harvesting',
  location: 'Field 3, Sunrise Farm',
  coordinates: { lat: 32.9483, lng: -96.7299 },
  practices: [
    { type: 'no_till', description: 'No tillage', ... }
  ],
  evidence: [
    { type: 'photo', data: { url: 'harvest.jpg' }, timestamp: Date.now(), verified: true },
    { type: 'iot_sensor', data: { soilMoisture: 45 }, timestamp: Date.now(), verified: true }
  ],
  yieldData: { cropType: 'tomato', variety: 'Cherokee Purple', quantity: 50, unit: 'lb', harvestDate: Date.now() }
});

// Verify provenance
const verification = food.getProvenance('tomato-batch-2026-001');

// verification.organicVerified = true
// verification.regenerativeVerified = true
// verification.foodMiles = 15
// verification.freshnessScore = 0.95
// verification.carbonFootprint = 2.3 (kg CO2)
```

## STEWARD - Land Commons

Food forests and community gardens as shared resources.

```typescript
// Create food forest commons
const forest = food.createFoodForest({
  name: 'Richardson Community Food Forest',
  location: '123 Park Lane, Richardson TX',
  coordinates: { lat: 32.9483, lng: -96.7299 },
  areaHectares: 0.5,
  initialStewards: [
    { did: 'did:mycelix:steward-1', name: 'Alice', role: 'lead_steward' },
    { did: 'did:mycelix:steward-2', name: 'Bob', role: 'steward' }
  ]
});

// Log stewardship hours
food.logStewardship({
  forestId: forest.id,
  stewardDid: 'did:mycelix:steward-1',
  hours: 3,
  activityType: 'planting',
  description: 'Planted fruit guild around apple tree',
  witnesses: ['did:mycelix:steward-2']
});
```

## WISDOM - Agricultural Knowledge

Seed library and growing guides with epistemic transparency.

```typescript
// Get growing guide
const guide = food.getGrowingGuide('tomato', 'Cherokee Purple');

// guide.whenToPlant = "Plant in zones 3-11 when soil is workable"
// guide.spacing = "24-36 inches"
// guide.goodCompanions = [{ plant: 'basil', reason: 'Beneficial relationship' }, ...]
// guide.sources = [{ claim: { ... }, confidence: { sourceCredibility: 0.9, ... } }]

// Search seed library
const seeds = food.searchSeeds({
  query: 'tomato',
  organic: true,
  availableForTrade: true
});

// Add seeds to library
food.addToSeedLibrary({
  seedKnowledgeId: 'tomato-cherokee',
  stewartDid: 'did:mycelix:farmer-1',
  variety: 'Cherokee Purple',
  yearHarvested: 2026,
  quantity: 100,
  unit: 'seeds',
  germinationRate: 0.94,
  organicStatus: 'organic',
  notes: 'Selected from best producers',
  availableForTrade: true,
  tradePreferences: ['other heirlooms', 'peppers']
});
```

## SMS Commands

Via Civitas adapter:

```
FOOD JOIN [farm-id]          → Subscribe to CSA
FOOD BOX                     → This week's box contents
FOOD ALLERGEN [item]         → Check allergens for item
FOOD TRACE [item-id]         → Full provenance chain
FOOD GARDEN [plot-id]        → Garden plot status

SEED FIND [variety]          → Search seed library
SEED SHARE [variety] [qty]   → Offer seeds for trade

KNOWLEDGE [crop]             → Get growing guide
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FoodSovereigntyService                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   TEND   │  │ NOURISH  │  │CULTIVATE │  │ STEWARD  │   │
│  │ (Finance)│  │ (Health) │  │ (Supply) │  │(Property)│   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴──────┬──────┴─────────────┘          │
│                            │                                │
│                    ┌───────▼───────┐                       │
│                    │    WISDOM     │                       │
│                    │   (Knowledge) │                       │
│                    └───────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │ Mycelix Bridge│
                    │  Cross-hApp   │
                    └───────────────┘
```

## Consciousness-First Food

This module embodies the Eight Harmonies:

1. **Truth** - Verifiable provenance, epistemic claims on growing knowledge
2. **Kindness** - Food security, allergy safety, nutrition access
3. **Justice** - Fair farmer compensation, equitable distribution
4. **Sustainability** - Regenerative practices, carbon tracking
5. **Freedom** - Seed sovereignty, land commons ownership
6. **Creativity** - Traditional knowledge, polyculture design
7. **Connection** - Farm-to-table relationships, community

## Contributing

See `/docs/MYCELIX_FOOD_REVOLUTIONARY_ARCHITECTURE.md` for full design.

---

*Food sovereignty is not a new system—it's Mycelix configured for the most fundamental human need.*
