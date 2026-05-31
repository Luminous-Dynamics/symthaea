# Changelog

All notable changes to the "Mycelix Knowledge - Fact Checker" extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Multi-file fact-checking
- Markdown preview integration
- CodeLens for inline fact-checks
- Workspace-wide claim scanning

## [0.1.0] - 2024-01-15

### Added
- **Fact-Check Command**: Select text and fact-check against the knowledge graph
  - Keyboard shortcut: `Ctrl+Shift+F` / `Cmd+Shift+F`
  - Context menu integration
  - Quick pick result display

- **Submit Claim Command**: Contribute new claims to the network
  - Keyboard shortcut: `Ctrl+Shift+C` / `Cmd+Shift+C`
  - E-N-M classification wizard
  - Tag and source management

- **Search Command**: Search the decentralized knowledge graph
  - Full-text search
  - Filter by epistemic type
  - Sort by credibility or date

- **Epistemic Classification**: Analyze text for E-N-M breakdown
  - Visual breakdown display
  - Explanation of classification

- **Credibility Display**: View detailed credibility factors
  - Source diversity score
  - Author reputation
  - Temporal consistency
  - Cross-validation results

- **Sidebar Views**:
  - Recent Claims panel
  - Fact Check Results panel
  - Knowledge Graph explorer

- **Inline Decorations**: Visual markers for fact-checked text
  - Color-coded by credibility
  - Hover for details
  - Configurable visibility

- **Configuration Options**:
  - Custom Holochain URL
  - App ID configuration
  - Auto fact-check toggle
  - Credibility threshold
  - Decoration toggle

### Technical
- WebSocket connection to Holochain conductor
- Webpack bundling for optimal size
- TypeScript with strict mode
- ESLint configuration

## [0.0.1] - 2024-01-01

### Added
- Initial development release
- Basic project structure
- Extension manifest

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-15 | First public release with full feature set |
| 0.0.1 | 2024-01-01 | Development preview |

## Upgrade Notes

### Upgrading to 0.1.0

This is the first public release. If upgrading from a development version:

1. Update your settings - configuration keys have changed
2. Re-connect to Holochain after upgrade
3. Clear any cached fact-check results

## Roadmap

### 0.2.0 (Planned)
- CodeLens integration
- Markdown preview support
- Batch fact-checking

### 0.3.0 (Planned)
- Workspace scanning
- Report generation
- Export functionality

### 1.0.0 (Future)
- Stable API
- Full test coverage
- Performance optimization
