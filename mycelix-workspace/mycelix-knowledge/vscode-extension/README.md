# Mycelix Knowledge - Fact Checker for VS Code

Fact-check text, explore claims, and contribute to the decentralized knowledge graph directly from VS Code.

![Mycelix Knowledge Extension](media/screenshot.png)

## Features

### Fact-Check Any Text
Select text in your editor and instantly fact-check it against the decentralized knowledge graph.

- **Keyboard shortcut**: `Ctrl+Shift+F` (Windows/Linux) or `Cmd+Shift+F` (Mac)
- **Context menu**: Right-click selected text → "Fact Check Selection"

### Submit Claims
Contribute to the knowledge graph by submitting verified claims.

- **Keyboard shortcut**: `Ctrl+Shift+C` (Windows/Linux) or `Cmd+Shift+C` (Mac)
- Classify claims as Empirical, Normative, or Mythic
- Add sources and tags

### Epistemic Classification
Analyze text to understand its epistemic nature:

- **Empirical**: Verifiable through observation (e.g., "Water boils at 100°C")
- **Normative**: Value-based claims (e.g., "Privacy is important")
- **Mythic**: Meaning-making narratives (e.g., "Progress is inevitable")

### Knowledge Graph Explorer
Browse the interconnected web of claims in the sidebar panel:

- View recent claims
- See fact-check results history
- Explore claim relationships

### Credibility Scoring
See multi-factor credibility scores including:

- Source diversity
- Author reputation
- Temporal consistency
- Cross-validation with related claims

## Commands

| Command | Description | Shortcut |
|---------|-------------|----------|
| `Mycelix: Fact Check Selection` | Fact-check selected text | `Ctrl+Shift+F` |
| `Mycelix: Submit as Claim` | Submit selection as a new claim | `Ctrl+Shift+C` |
| `Mycelix: Search Knowledge Graph` | Search for claims | - |
| `Mycelix: Classify Epistemic Type` | Analyze epistemic classification | - |
| `Mycelix: Show Credibility Score` | View credibility details | - |
| `Mycelix: Connect to Holochain` | Connect to the network | - |

## Requirements

- **Holochain Conductor**: Running at `ws://localhost:8888` (configurable)
- **Mycelix Knowledge hApp**: Installed on your Holochain conductor

### Setting Up Holochain

1. Install Holochain: https://developer.holochain.org/quick-start/
2. Install the Mycelix Knowledge hApp
3. Start the conductor
4. Connect from VS Code

## Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `mycelix.holochainUrl` | `ws://localhost:8888` | Holochain conductor WebSocket URL |
| `mycelix.appId` | `mycelix-knowledge` | Holochain app ID |
| `mycelix.autoFactCheck` | `false` | Auto fact-check on selection |
| `mycelix.minCredibilityThreshold` | `0.5` | Minimum credibility to show |
| `mycelix.showInlineDecorations` | `true` | Show inline credibility markers |

## Inline Decorations

When enabled, the extension shows inline decorations for fact-checked text:

- 🟢 **Green**: High credibility (>0.8)
- 🟡 **Yellow**: Medium credibility (0.5-0.8)
- 🔴 **Red**: Low credibility (<0.5)
- ⚪ **Gray**: Unverifiable

## The E-N-M Framework

This extension uses the Epistemic-Normative-Mythic classification framework:

```
┌─────────────────────────────────────────┐
│             CLAIM TYPES                  │
├─────────────┬─────────────┬─────────────┤
│  EMPIRICAL  │  NORMATIVE  │   MYTHIC    │
│  (E)        │  (N)        │   (M)       │
├─────────────┼─────────────┼─────────────┤
│ Observable  │ Value-based │ Meaning     │
│ Measurable  │ Ethical     │ Narrative   │
│ Testable    │ Preferential│ Cultural    │
└─────────────┴─────────────┴─────────────┘
```

Most claims are mixtures. For example, "We should reduce carbon emissions" is:
- 60% Normative (value judgment)
- 30% Empirical (measurable outcomes)
- 10% Mythic (progress narrative)

## Privacy

All fact-checking queries go through the decentralized Holochain network:

- No central server stores your queries
- Your local node participates in the DHT
- You control your data

## Known Issues

- Initial connection may take a few seconds while discovering peers
- Large fact-check requests may timeout on slow connections

## Release Notes

### 0.1.0

- Initial release
- Fact-check, submit claims, search
- Epistemic classification
- Sidebar views
- Inline decorations

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/Luminous-Dynamics/mycelix-knowledge/blob/main/CONTRIBUTING.md).

## Learn More

- [Mycelix Knowledge Documentation](https://knowledge.mycelix.net)
- [GitHub Repository](https://github.com/Luminous-Dynamics/mycelix-knowledge)
- [Discord Community](https://discord.gg/mycelix)

## License

MIT - See [LICENSE](LICENSE) for details.

---

**Enjoy fact-checking!** - [Luminous Dynamics](https://luminousdynamics.org)
