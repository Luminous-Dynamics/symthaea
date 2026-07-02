# Mycelix Protocol Architecture Diagrams

This directory contains visual diagrams explaining the Mycelix Protocol architecture, mechanisms, and Ethereum ecosystem integration.

## Available Diagrams

### 1. [System Architecture](01-system-architecture.md)
**Purpose**: Complete 4-layer architecture overview

**Shows**:
- Layer 4: Industry Adapters (Federated Learning, Healthcare, DAO, DeFi)
- Layer 3: Meta-Core (DIDs, VCs, Reputation, Governance)
- Layer 2: Hybrid DLT (Holochain DHT + Ethereum L2)
- Layer 1: Cryptographic Primitives (Ed25519, SHA-256, Merkle, ZKP)

**Use Cases**:
- Grant applications (Ethereum Foundation, NSF)
- Technical presentations
- Architecture documentation
- Onboarding new contributors

**Key Insights**:
- Pragmatic decentralization (right tool for right job)
- Clear evolution path (Phase 1 Merkle → Phase 2 ZK)
- Data flow from user training to global model update

### 2. [PoGQ Mechanism](02-pogq-mechanism.md)
**Purpose**: Detailed flowchart of Proof-of-Gradient-Quality validation

**Shows**:
- Complete training round workflow
- Validator selection (VRF-based)
- Quality score computation
- Byzantine detection and filtering
- Reputation updates
- Ethereum L2 anchoring

**Use Cases**:
- Academic papers and research presentations
- Explaining core innovation to technical audiences
- Security analysis and auditing
- Implementation reference for developers

**Key Insights**:
- Privacy-preserving validation (no training data shared)
- +23.2pp improvement over baseline FedAvg
- 100% Byzantine detection rate in experiments
- Reputation-weighted aggregation prevents Sybil attacks

### 3. [Ethereum Integration](03-ethereum-integration.md)
**Purpose**: How Mycelix integrates with Ethereum ecosystem

**Shows**:
- DID/VC integration (ERC-1056, ERC-725)
- Cross-chain bridge architecture
- DAO governance integration (Snapshot, Aragon)
- DeFi integration (reputation-based lending)
- External ecosystem connections (Gitcoin, Compound)

**Use Cases**:
- Ethereum Foundation grant application
- Partnership discussions with DAOs and DeFi protocols
- Developer integration guides
- Ecosystem alignment demonstration

**Key Insights**:
- Complete W3C DID + Ethereum standards compliance
- Phase 1 gas costs: ~$3-6 per 1,000 users
- Phase 2 projected: ~$0.01 per user (100x improvement)
- Multiple integration points (identity, reputation, governance)

## Viewing the Diagrams

### On GitHub
All diagrams use Mermaid syntax and render automatically on GitHub. Just click the links above.

### Export to Images

**Option 1: Mermaid Live Editor** (Easiest)
1. Visit [mermaid.live](https://mermaid.live/)
2. Copy the Mermaid code from the diagram file
3. Paste into the editor
4. Export as PNG or SVG

**Option 2: Command Line** (Automated)
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i 01-system-architecture.md -o system-architecture.png
mmdc -i 02-pogq-mechanism.md -o pogq-mechanism.png
mmdc -i 03-ethereum-integration.md -o ethereum-integration.png

# Generate SVG images (scalable, better for presentations)
mmdc -i 01-system-architecture.md -o system-architecture.svg
mmdc -i 02-pogq-mechanism.md -o pogq-mechanism.svg
mmdc -i 03-ethereum-integration.md -o ethereum-integration.svg
```

**Option 3: VS Code Extension**
1. Install "Markdown Preview Mermaid Support" extension
2. Open diagram file in VS Code
3. Click "Preview" (Ctrl+Shift+V)
4. Right-click diagram → Export as PNG/SVG

## Using Diagrams in Grant Applications

### Ethereum Foundation ESP Application

Include these diagrams to demonstrate:
1. **System Architecture** → Shows technical sophistication and clear design
2. **PoGQ Mechanism** → Demonstrates novel research contribution
3. **Ethereum Integration** → Proves ecosystem alignment and value

**Suggested Placement**:
- Executive Summary: System Architecture (high-level overview)
- Technical Approach: All three diagrams with detailed explanations
- Ethereum Alignment: Ethereum Integration diagram

### NSF CISE Application (Future)

Include diagrams to show:
1. **Intellectual Merit**: PoGQ mechanism innovation
2. **Broader Impacts**: Ethereum Integration (real-world applicability)
3. **Technical Feasibility**: System Architecture (proven implementation)

## Creating New Diagrams

When adding diagrams:
1. Use Mermaid syntax for consistency
2. Follow naming convention: `XX-descriptive-name.md`
3. Include explanatory text before/after diagrams
4. Add export instructions
5. Update this README with new diagram

**Mermaid Resources**:
- [Official Documentation](https://mermaid.js.org/)
- [Flowchart Syntax](https://mermaid.js.org/syntax/flowchart.html)
- [Sequence Diagrams](https://mermaid.js.org/syntax/sequenceDiagram.html)
- [Class Diagrams](https://mermaid.js.org/syntax/classDiagram.html)

## Design Guidelines

**For Grant Applications**:
- ✅ Clear, professional appearance
- ✅ Minimal but sufficient detail
- ✅ Consistent color scheme
- ✅ Labels and legends included

**For Technical Audiences**:
- ✅ Accurate technical details
- ✅ Standard notation (UML, flowchart symbols)
- ✅ Complete data flow paths
- ✅ Security considerations highlighted

**For Non-Technical Audiences**:
- ✅ Simplified version with key concepts only
- ✅ Visual metaphors where appropriate
- ✅ Avoid jargon in labels
- ✅ Focus on benefits over mechanisms

## Diagram Maintenance

These diagrams should be updated when:
- Architecture changes significantly
- New phases added to roadmap
- Experimental results change key metrics
- Integration points with Ethereum ecosystem expand

**Update Frequency**: Review quarterly or before major grant submissions

## Questions?

- **Technical questions**: See [System Architecture](../06-architecture/SYSTEM_ARCHITECTURE.md) for detailed documentation
- **Grant application help**: See [Ethereum Foundation Checklist](../../grants/ETHEREUM-FOUNDATION-CHECKLIST.md)
- **Implementation questions**: See [Contributing Guide](../../CONTRIBUTING.md)

---

**Last Updated**: October 7, 2025
**Created For**: Ethereum Foundation ESP Application (November 15, 2025 deadline)
**Maintained By**: Tristan Stoltz (tristan.stoltz@evolvingresonantcocreationism.com)

**Status**: ✅ Complete - All three required diagrams created and ready for grant application
