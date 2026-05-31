---
layout: home

hero:
  name: Mycelix Knowledge
  text: Decentralized Knowledge Graph
  tagline: Epistemic classification, credibility scoring, and fact-checking for the decentralized web
  image:
    src: /hero-image.svg
    alt: Mycelix Knowledge
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/Luminous-Dynamics/mycelix-knowledge
    - theme: alt
      text: Try Demo
      link: https://demo.mycelix.net

features:
  - icon: 🎯
    title: Epistemic Classification
    details: Classify claims along three dimensions - Empirical (verifiable), Normative (value-based), and Mythic (meaning-making) - for nuanced understanding.
  - icon: 📊
    title: Credibility Scoring
    details: Multi-factor credibility assessment including source diversity, author reputation, temporal consistency, and cross-validation.
  - icon: 🔗
    title: Knowledge Graph
    details: Build interconnected webs of claims with typed relationships - supports, contradicts, refines, depends_on - enabling belief propagation.
  - icon: ✅
    title: Fact-Checking API
    details: Query the knowledge graph to fact-check statements. Get verdicts, confidence scores, supporting/contradicting evidence.
  - icon: 💹
    title: Markets Integration
    details: Bidirectional integration with Epistemic Markets - spawn verification markets for claims, update credibility from resolutions.
  - icon: 🤖
    title: AI-Powered Analysis
    details: LLM integration for automatic classification, evidence extraction, contradiction detection, and bias analysis.
---

## Quick Example

```typescript
import { KnowledgeClient } from '@mycelix/knowledge-client';

// Connect to the knowledge network
const client = new KnowledgeClient({
  url: 'ws://localhost:8888',
  appId: 'mycelix-knowledge',
});

// Create a claim with epistemic classification
const claimId = await client.claims.createClaim({
  content: 'Climate change is primarily caused by human activities.',
  classification: {
    empirical: 0.85,  // Highly verifiable through data
    normative: 0.10,  // Some value implications
    mythic: 0.05,     // Minimal narrative element
  },
  tags: ['climate', 'science'],
});

// Fact-check a statement
const result = await client.factcheck.factCheck({
  statement: 'Global temperatures have risen 1.1°C since pre-industrial times.',
  minConfidence: 0.7,
});

console.log(result.verdict);     // "TRUE" | "MOSTLY_TRUE" | "MIXED" | ...
console.log(result.confidence);  // 0.92
console.log(result.explanation); // Detailed reasoning
```

## The E-N-M Framework

<div class="enm-grid">

### 🔴 Empirical
Claims verifiable through observation, experiment, or data.
- "Water boils at 100°C at sea level"
- "The population of Tokyo is 14 million"
- "COVID-19 is caused by SARS-CoV-2"

### 🟢 Normative
Claims expressing values, ethics, or prescriptions.
- "Healthcare should be universal"
- "Privacy is a fundamental right"
- "We ought to reduce emissions"

### 🔵 Mythic
Claims involving meaning-making, narrative, or cultural significance.
- "Democracy represents human progress"
- "Nature has intrinsic value"
- "Technology will save humanity"

</div>

## Ecosystem

<div class="ecosystem-grid">

### SDK & Libraries
- **@mycelix/knowledge-client** - TypeScript SDK
- **@mycelix/knowledge-react** - React hooks & components
- **@mycelix/knowledge-svelte** - Svelte stores & components
- **@mycelix/knowledge-llm** - AI/LLM integration

### Tools & Extensions
- **VS Code Extension** - Fact-check in your editor
- **Browser Extension** - Fact-check any webpage
- **Embeddable Widgets** - Add to any website
- **GraphQL API** - Flexible querying

### Infrastructure
- **Holochain DNA** - Decentralized backend
- **GraphQL Server** - API layer
- **Real-time Subscriptions** - WebSocket updates
- **Storybook** - Component documentation

</div>

## Use Cases

- **Journalism** - Verify sources and claims in articles
- **Research** - Build interconnected literature reviews
- **Education** - Teach critical thinking with epistemic classification
- **Social Media** - Community fact-checking at scale
- **Policy** - Evidence-based decision making
- **AI Safety** - Ground language models in verified knowledge

## Get Involved

<div class="cta-grid">

### 🚀 Start Building
[Quick Start Guide](/guide/quickstart) • [SDK Documentation](/sdk/) • [API Reference](/api/)

### 💬 Join Community
[Discord](https://discord.gg/mycelix) • [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-knowledge/discussions)

### 🤝 Contribute
[Contributing Guide](https://github.com/Luminous-Dynamics/mycelix-knowledge/blob/main/CONTRIBUTING.md) • [Good First Issues](https://github.com/Luminous-Dynamics/mycelix-knowledge/labels/good%20first%20issue)

</div>

<style>
.enm-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 30px 0;
}

.enm-grid h3 {
  margin-top: 0;
}

@media (max-width: 768px) {
  .enm-grid {
    grid-template-columns: 1fr;
  }
}

.ecosystem-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 30px 0;
}

@media (max-width: 768px) {
  .ecosystem-grid {
    grid-template-columns: 1fr;
  }
}

.cta-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 30px 0;
  text-align: center;
}

@media (max-width: 768px) {
  .cta-grid {
    grid-template-columns: 1fr;
  }
}
</style>
