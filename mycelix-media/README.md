# Mycelix Media

**Decentralized journalism and content verification for the Mycelix Civilizational OS**

## Overview

Mycelix Media provides censorship-resistant publication infrastructure, author attribution with royalties, fact-checking using the Epistemic Charter, and community-driven curation. This enables trustworthy information sharing in the Mycelix ecosystem.

## Zomes

### publication
Immutable content anchoring:
- Article and content publishing
- Version history tracking
- Content type classification
- Encryption for source protection
- Cross-reference linking

### attribution
Author DID linking and royalties:
- Author verification via DID
- Co-author and contributor credits
- Royalty distribution rules
- License management
- Usage tracking

### factcheck
Epistemic claim verification:
- Claim extraction from content
- Epistemic classification (E/N/M axes)
- Verification status tracking
- Evidence linking
- Source credibility assessment

### curation
Community-driven quality signals:
- Upvotes and endorsements
- Topic categorization
- Collection management
- Quality scoring
- Featured content selection

## Architecture

```
mycelix-media/
├── dna/
│   └── dna.yaml              # DNA manifest
├── client/                   # TypeScript client + Knowledge bridge
├── zomes/
│   ├── publication/
│   │   ├── integrity/        # Content validation
│   │   └── coordinator/      # Publishing management
│   ├── attribution/
│   │   ├── integrity/        # Attribution validation
│   │   └── coordinator/      # Royalty management
│   ├── factcheck/
│   │   ├── integrity/        # Fact-check validation
│   │   └── coordinator/      # Verification workflow
│   └── curation/
│       ├── integrity/        # Curation validation
│       └── coordinator/      # Quality signals
├── client/                   # TypeScript client
└── tests/                    # Integration tests
```

## Epistemic Classification

Content claims are classified on three axes:
- **E (Empirical)**: Verifiable through observation/experiment
- **N (Normative)**: Value judgments and opinions
- **M (Mythic)**: Narrative and meaning-making

## Integration Points

- **mycelix-identity**: Author verification
- **mycelix-knowledge**: Claim storage and linking
- **mycelix-justice**: Defamation disputes
- **mycelix-finance**: Royalty payments

### Client & Knowledge Integration

The `client/` package provides a small TypeScript helper for linking publications
into the Mycelix Knowledge Graph:

- `MediaKnowledgeBridge` wraps the `@mycelix/knowledge-client` `KnowledgeService`.
- `submitPublicationClaim` creates an epistemic claim for a given publication
  (identified by `publicationId`) and attaches a `happ://media/publication/<id>` source.

Example (Node/TypeScript):

```ts
import { AppClient } from "@holochain/client";
import { MediaKnowledgeBridge } from "@mycelix/media-client";

const appClient = {} as AppClient; // obtain from Holochain conductor
const bridge = new MediaKnowledgeBridge(appClient);

await bridge.submitPublicationClaim({
  publicationId: "article-123",
  title: "Solar capacity increased 15% in Q3",
  claimText: "Solar capacity increased 15% in Q3",
  authorDid: "did:mycelix:uhCAkX...",
  domain: "journalism",
  topics: ["energy", "climate"],
});
```

## Building

```bash
# Build all zomes
cargo build --release --target wasm32-unknown-unknown

# Package the hApp
hc app pack .
```

## License

Apache-2.0
