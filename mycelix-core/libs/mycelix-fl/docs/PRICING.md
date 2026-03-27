# Mycelix FL — Enterprise Pricing Guide
*Internal Document — Not for Distribution*

## Pricing Philosophy
- Scope-based, not per-seat or per-server
- Favorable terms for cooperatives, B-corps, mission-aligned organizations
- Academic/research: free under AGPL
- Evaluation: 90-day trial, no commitment

## Tier Structure

### Tier 1: Starter ($50K/year)
- Up to 10 participating nodes
- Single coordinator instance
- FedAvg + Krum + TrimmedMean defenses
- Email support (48h response)
- No PoGQ (use open-source AGPL for basic defenses)

### Tier 2: Professional ($150K/year)
- Up to 50 participating nodes
- HA coordinator (active-passive)
- **Full PoGQ v4.1 + Adaptive PoGQ**
- Differential privacy module
- HyperFeel V2 compression
- Slack support (24h response)
- Quarterly security updates

### Tier 3: Enterprise ($500K/year)
- Unlimited nodes
- Multi-region coordinator deployment
- **Full detection stack** (PoGQ + Shapley + Self-Healing)
- **ZK-STARK proof generation** for compliance
- Custom defense algorithm development
- Dedicated support engineer
- SLA: 99.9% uptime, 4h response
- Annual security audit

### Tier 4: Strategic Partnership ($1M+/year)
- Everything in Enterprise
- Source code access (not AGPL — proprietary license)
- Joint development roadmap
- Co-branding rights
- Board advisory seat
- Custom integrations (EHR, financial systems)

## Add-Ons
| Add-On | Price |
|--------|-------|
| Holochain deployment (decentralized) | +$50K/year |
| HIPAA compliance package | +$30K/year |
| Custom defense algorithm | $75K one-time |
| Training workshop (2-day) | $15K |
| Integration consulting | $250/hour |

## Volume Discounts
| Term | Discount |
|------|----------|
| 2-year commitment | 15% |
| 3-year commitment | 25% |
| Multi-product (FL + Civic + Finance) | 20% |

## Comparison to Alternatives

| | Mycelix FL | TF Federated | NVIDIA Clara | PySyft |
|---|-----------|-------------|-------------|--------|
| Byzantine Tolerance | **45%** | 0% | 0% | 0% |
| Detection | **PoGQ v4.1** | None | Basic | Basic |
| Proof System | **ZK-STARK** | None | None | None |
| Deployment | Binary + Docker | Python | Container | Python |
| Decentralized | **Holochain** | No | No | Partial |
| Price | $50-500K | Free (Google lock-in) | $$$ (NVIDIA lock-in) | Free (limited) |

## Revenue Projections (Conservative)

### Year 1 (5 customers)
- 2x Starter: $100K
- 2x Professional: $300K
- 1x Enterprise: $500K
- **Total: $900K**

### Year 2 (15 customers)
- 5x Starter: $250K
- 7x Professional: $1.05M
- 3x Enterprise: $1.5M
- **Total: $2.8M**

### Year 3 (30 customers)
- 8x Starter: $400K
- 15x Professional: $2.25M
- 5x Enterprise: $2.5M
- 2x Strategic: $2M
- **Total: $7.15M**

## Negotiation Guidelines
- Never discount below Starter price for commercial use
- Cooperatives: 50% discount on any tier
- B-corps: 30% discount
- Government: standard pricing, multi-year preferred
- Startups (<$5M revenue): Starter at $25K/year for first 2 years

## Competitive Response
- If customer mentions TF Federated: emphasize Byzantine detection (they have none)
- If customer mentions NVIDIA: emphasize no vendor lock-in, Holochain decentralization
- If customer mentions cost: emphasize single binary deployment, no Python dependency management
- If customer mentions compliance: emphasize ZK-STARK proofs, HIPAA package
