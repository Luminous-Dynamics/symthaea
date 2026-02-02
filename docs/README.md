# Luminous Dynamics Documentation

Central documentation repository for the Luminous Dynamics organization, providing comprehensive guides, API references, and architectural documentation for all projects.

**Live Documentation**: [https://luminousdynamics.org](https://luminousdynamics.org)

## Documentation Structure

```
docs/
├── philosophy/           # Core principles and design philosophy
│   ├── CONSCIOUSNESS_FIRST_COMPUTING.md
│   ├── TECHNICAL_PHILOSOPHY.md
│   └── CHARTER.md
├── guides/               # Implementation and setup guides
│   ├── DEVELOPER_ONBOARDING.md
│   ├── SECURITY_IMPLEMENTATION_GUIDE.md
│   └── PERFORMANCE_IMPLEMENTATION_GUIDE.md
├── technical/            # Architecture and API documentation
├── vision/               # Roadmaps and project vision
├── terra-lumina/         # Terra Atlas project documentation
├── active/               # Currently active project docs
└── reports/              # Project reports and analyses
```

## Projects Covered

### Terra Atlas

Global energy investment platform democratizing access to renewable energy projects worldwide.

- **Documentation**: `/docs/terra-lumina/`
- **Live Platform**: [atlas.luminousdynamics.io](https://atlas.luminousdynamics.io)
- **Key Features**:
  - 3D visualization of energy sites globally
  - Investment pledge system ($10+ minimum)
  - Community ownership through Luminous Chimera Model
  - Swiss foundation governance for mission permanence

### Luminous Nix

Natural language interface for NixOS that makes system management accessible to everyone through AI-powered assistance.

- **Documentation**: [11-meta-consciousness/luminous-nix/docs/](https://github.com/Luminous-Dynamics/luminous-dynamics/tree/main/11-meta-consciousness/luminous-nix/docs)
- **Project Site**: [nixforhumanity.org](https://nixforhumanity.org)
- **Key Features**:
  - Natural language NixOS commands
  - AI-powered intent recognition (98.5% accuracy)
  - Voice interface support
  - GUI and TUI interfaces

### Mycelix-Core

Decentralized machine learning protocol enabling collaborative model training while preserving data privacy.

- **Repository**: [Luminous-Dynamics/Mycelix-Core](https://github.com/Luminous-Dynamics/Mycelix-Core)
- **Key Features**:
  - Federated learning infrastructure
  - Privacy-preserving ML training
  - Peer-to-peer model synchronization
  - Holochain integration

## Viewing the Documentation

### Online

Access the published documentation at [https://luminousdynamics.org](https://luminousdynamics.org)

The site includes:
- Interactive 3D visualizations
- Economic calculators
- Project dashboards
- API references

### Local Development

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/docs.git
cd docs

# Serve locally (any static file server works)
python -m http.server 8000
# or
npx serve .

# Open in browser
open http://localhost:8000
```

## Contributing to Documentation

We welcome documentation contributions! Clear, accurate documentation helps everyone in our community.

### Getting Started

1. **Read the prerequisites**:
   - [Consciousness-First Computing Philosophy](./philosophy/CONSCIOUSNESS_FIRST_COMPUTING.md)
   - [Contributing Guide](./CONTRIBUTING.md)
   - [Technical Philosophy](./philosophy/TECHNICAL_PHILOSOPHY.md)

2. **Find what needs work**:
   - Check issues labeled `documentation`
   - Look for `TODO` or `WIP` markers in existing docs
   - Review outdated content

3. **Fork and branch**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/docs.git
   cd docs
   git checkout -b docs/your-improvement
   ```

### Writing Guidelines

#### Style

- **Be precise**: Every numerical claim needs verification or "estimated/projected" qualifier
- **Be clear**: Write for developers who may not be familiar with the project
- **Be honest**: "We don't know" is better than false certainty
- **Avoid jargon**: When technical terms are necessary, define them

#### Structure

- Use clear headings (H1 for title, H2 for sections, H3 for subsections)
- Include code examples where helpful
- Add links to related documentation
- Include a "Last Updated" date for time-sensitive content

#### Formatting

```markdown
# Document Title

Brief description of what this document covers.

## Section Name

Content with clear explanations.

### Subsection

- Bullet points for lists
- Code blocks for examples

```language
code example
```

## Related Documentation

- [Link to related doc](./path/to/doc.md)
```

### Pull Request Process

1. **Create focused PRs**: One topic per PR
2. **Write clear descriptions**: Explain what changed and why
3. **Update related docs**: If your change affects other documents, update them too
4. **Test locally**: Ensure links work and formatting renders correctly
5. **Request review**: Tag relevant maintainers

### Areas Needing Documentation

**High Priority**:
- API reference documentation
- Tutorial content for new users
- Deployment guides
- Troubleshooting guides

**Medium Priority**:
- Architecture decision records
- Performance optimization guides
- Security best practices
- Internationalization

## Tech Stack

- **Static Site**: GitHub Pages with custom HTML/CSS/JS
- **Visualizations**: Three.js for 3D content
- **Styling**: Custom CSS with glass morphism design
- **Build**: No build step required (pure static files)
- **Hosting**: GitHub Pages at luminousdynamics.org

### Key Files

- `_config.yml` - GitHub Pages configuration
- `CNAME` - Custom domain configuration
- `index.html` - Main landing page
- `styles.css` / `luminous-styles.css` - Site styling
- `main.js` / `luminous-main.js` - Interactive functionality

## Quick Links

| Resource | Description |
|----------|-------------|
| [Developer Onboarding](./DEVELOPER_ONBOARDING.md) | Getting started as a contributor |
| [API Documentation](./API_DOCUMENTATION.md) | API reference and endpoints |
| [Service Catalog](./SERVICE_CATALOG.md) | Available services and ports |
| [Tech Stack 2025](./TECH_STACK_2025.md) | Current technology choices |
| [Contributing](./CONTRIBUTING.md) | How to contribute |

## Organization Resources

- **GitHub**: [github.com/Luminous-Dynamics](https://github.com/Luminous-Dynamics)
- **Main Domain**: [luminousdynamics.org](https://luminousdynamics.org)
- **Platform**: [luminousdynamics.io](https://luminousdynamics.io)
- **Email**: contact@luminousdynamics.org

## License

This documentation is released under the [MIT License](../LICENSE).

You are free to use, modify, and distribute this documentation with attribution.

---

*Part of the Luminous Dynamics ecosystem - Building technology that serves consciousness and humanity's evolution.*
