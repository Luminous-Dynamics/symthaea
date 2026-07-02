# Contributing to Mycelix Praxis

Thank you for your interest in contributing to Mycelix Praxis! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Branch Strategy](#branch-strategy)
- [Commit Convention](#commit-convention)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Rust** (latest stable): Install via [rustup](https://rustup.rs/)
- **Node.js** (>= 18): Install via [nvm](https://github.com/nvm-sh/nvm) or [official installer](https://nodejs.org/)
- **Holochain dev tools**: Follow [Holochain installation guide](https://developer.holochain.org/install/)
- **pnpm** or **npm**: Package manager for Node.js

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git
cd mycelix-praxis

# Install dependencies and build
make build

# Run development environment
make dev

# Run tests
make test
```

## Development Workflow

```
┌──────────────┐
│ 1. Find/Create│
│     Issue     │
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 2. Fork Repo │
│ (if external)│
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 3. Create    │
│ Feature Branch│ (from dev)
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 4. Make      │
│  Changes     │ (atomic commits)
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 5. Add Tests │
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 6. Update    │
│     Docs     │
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 7. Run Tests │
│  & Linters   │
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 8. Push &    │
│ Create PR    │
└───────┬───────┘
        │
        ▼
┌──────────────┐
│ 9. Address   │
│  Feedback    │ ◄─┐
└───────┬───────┘   │
        │           │
        ├───────────┘
        ▼
┌──────────────┐
│10. Merged!   │
│   Celebrate! │
└──────────────┘
```

### Step-by-Step

1. **Find or create an issue** describing the work
2. **Fork the repository** (external contributors)
3. **Create a feature branch** from `dev`
4. **Make your changes** with clear, atomic commits
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Run tests and linters** locally
8. **Push your branch** and create a pull request
9. **Address review feedback**
10. **Celebrate** when merged!

## Branch Strategy

- **`main`** — Stable, production-ready code
- **`dev`** — Integration branch for next release
- **`feat/*`** — New features (e.g., `feat/fl-aggregation`)
- **`fix/*`** — Bug fixes (e.g., `fix/zome-validation`)
- **`docs/*`** — Documentation updates
- **`chore/*`** — Maintenance tasks (e.g., `chore/deps-update`)
- **`refactor/*`** — Code refactoring without functional changes

**Branch from `dev`**, not `main`, unless it's a hotfix.

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **style**: Code style/formatting (no logic change)
- **refactor**: Code restructuring without feature/fix
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, deps, tooling
- **security**: Security-related changes

### Scope

Optional. Examples: `zome/fl`, `web/ui`, `core`, `agg`, `protocol`, `ci`

### Examples

```
feat(zome/fl): add trimmed mean aggregation

Implements robust aggregation using trimmed mean to mitigate
poisoning attacks. Adds validation for gradient norms.

Closes #42
```

```
fix(web): correct round state display logic

Round status was showing incorrect phase due to missing
case for AGGREGATE state.

Fixes #87
```

## Pull Request Process

### Before Opening a PR

- [ ] Code builds without errors (`make build`)
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make fmt`)
- [ ] Linters pass (`make lint`)
- [ ] Documentation updated (if needed)
- [ ] Changelog updated (for significant changes)

### PR Checklist

When you open a PR, ensure:

- [ ] **Title** follows conventional commit format
- [ ] **Description** clearly explains what and why
- [ ] **Tests** added or updated for new functionality
- [ ] **Docs** updated (protocol, threat model, API docs)
- [ ] **Security impact** noted if applicable
- [ ] **Breaking changes** clearly marked and justified
- [ ] **Schema version** bumped if entry definitions changed
- [ ] **Issue linked** (e.g., "Closes #123")

### Review Process

- At least **1 approval** required from maintainers
- All **CI checks** must pass
- **Address feedback** promptly and professionally
- Use **"Request re-review"** after making changes

### Code Review Checklist

**For Reviewers**:

#### Functionality
- [ ] Code does what the PR claims
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs or logic errors

#### Code Quality
- [ ] Code is readable and maintainable
- [ ] No unnecessary complexity
- [ ] Follows project conventions
- [ ] Good naming (variables, functions, types)
- [ ] Adequate comments for complex logic

#### Tests
- [ ] Tests cover new functionality
- [ ] Tests cover edge cases
- [ ] Tests are clear and maintainable
- [ ] All tests passing

#### Security
- [ ] No hardcoded credentials
- [ ] Input validation present
- [ ] No SQL injection / command injection risks
- [ ] Crypto operations use safe libraries
- [ ] Privacy implications considered

#### Performance
- [ ] No obvious performance issues
- [ ] Algorithms are efficient
- [ ] No memory leaks
- [ ] Database queries optimized (if applicable)

#### Documentation
- [ ] Public APIs documented
- [ ] Complex logic explained
- [ ] README updated (if needed)
- [ ] Breaking changes noted

## Code Style

### Rust

- **Format**: `cargo fmt --all`
- **Lint**: `cargo clippy --all -- -D warnings`
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Document public APIs with `///` doc comments
- Prefer `Result<T, E>` over panics

### JavaScript/TypeScript

- **Format**: `prettier` (auto-configured)
- **Lint**: `eslint` (auto-configured)
- Use TypeScript strict mode
- Document complex functions with JSDoc

### General

- **Line length**: 100 characters (soft limit)
- **Indentation**: See `.editorconfig`
- **Comments**: Explain *why*, not *what*
- **TODOs**: Include issue number or rationale

## Testing

### Rust

```bash
# Run all Rust tests
cargo test --all

# Run specific crate tests
cargo test -p praxis-core

# Run with output
cargo test -- --nocapture
```

### Web

```bash
cd apps/web

# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch
```

### Integration Tests

```bash
# Run full integration suite (when available)
make test-integration
```

## Documentation

### What to Document

- **Public APIs**: All public functions, structs, enums
- **Protocol changes**: Update `docs/protocol.md`
- **Security implications**: Update `docs/threat-model.md`
- **Governance decisions**: Create ADR in `docs/adr/`
- **Breaking changes**: Note in PR and CHANGELOG

### Documentation Standards

- Use **Markdown** for all docs
- Include **code examples** where helpful
- Keep **diagrams** in source format (`.drawio`, `.mermaid`)
- Link to **related issues** and ADRs
- Update **README** for major features

## Security

### Reporting Vulnerabilities

**Do NOT** open public issues for security vulnerabilities. Instead:

1. Email **security@mycelix.org**
2. Or use **GitHub Security Advisories**
3. Include detailed description and reproduction steps
4. Allow 90 days for embargo before public disclosure

See [SECURITY.md](SECURITY.md) for full policy.

### Security Considerations

When contributing, consider:

- **Input validation**: Validate all external inputs
- **Cryptographic operations**: Use reviewed libraries
- **Privacy**: Minimize data exposure
- **Threat model impact**: Note in PR if your change affects threat surface
- **Dependencies**: Audit new dependencies

## Configuration and Constants

### Application Configuration

All application constants live in `apps/web/src/config/constants.ts`. **Never hardcode** magic numbers, strings, or URLs in components.

#### Adding a New Constant

```typescript
// apps/web/src/config/constants.ts

export const MY_FEATURE_CONFIG = {
  MAX_ITEMS: 100,
  DEFAULT_TIMEOUT: 5000, // 5 seconds
  API_ENDPOINT: import.meta.env.VITE_MY_API_URL || '/api/my-feature',
} as const;

export const MY_FEATURE_MESSAGES = {
  SUCCESS: 'Operation completed successfully!',
  ERROR: 'An error occurred. Please try again.',
} as const;
```

#### Using Constants in Components

```typescript
import { MY_FEATURE_CONFIG, MY_FEATURE_MESSAGES } from '../config/constants';

const handleAction = () => {
  if (items.length > MY_FEATURE_CONFIG.MAX_ITEMS) {
    toast.error(MY_FEATURE_MESSAGES.ERROR);
  }
};
```

#### Environment Variables

1. Copy `.env.example` to `.env`:

```bash
cd apps/web
cp .env.example .env
```

2. Edit `.env` with your configuration:

```bash
VITE_API_URL=http://localhost:3000
VITE_ENABLE_MOCK_DATA=true
```

3. Access in code via `import.meta.env`:

```typescript
const apiUrl = import.meta.env.VITE_API_URL;
```

**Important**: Never commit `.env` files! They're gitignored.

## Troubleshooting

### Web App Issues

#### App won't start

```bash
cd apps/web

# Clear cache and reinstall
rm -rf node_modules package-lock.json dist
npm install
npm run dev
```

#### Build fails with TypeScript errors

```bash
# Check TypeScript errors
npm run build

# Or run type check only
npx tsc --noEmit
```

#### Environment variables not loading

- Ensure `.env` file exists (copy from `.env.example`)
- Restart dev server after changing `.env`
- Environment variables must start with `VITE_`

### Rust Issues

#### Cargo build fails

```bash
# Update Rust toolchain
rustup update stable

# Clean and rebuild
cargo clean
make build
```

#### Tests failing after rebase

```bash
# Update dependencies
cargo update

# Run specific failing test
cargo test test_name -- --nocapture --test-threads=1
```

#### Clippy warnings

```bash
# Auto-fix some issues
cargo clippy --fix --allow-dirty

# Or manually fix based on warnings
cargo clippy --all -- -D warnings
```

### Git Issues

#### Merge conflicts during rebase

```bash
# Option 1: Resolve manually
git status  # See conflicted files
# Edit files to resolve conflicts
git add .
git rebase --continue

# Option 2: Abort and try again
git rebase --abort
```

#### Accidentally committed to wrong branch

```bash
# Stash changes
git stash

# Switch to correct branch
git checkout correct-branch

# Apply stashed changes
git stash pop
```

## npm Scripts Reference

### Web App (`apps/web/`)

```bash
# Development
npm run dev              # Start dev server (http://localhost:5173)
npm run build            # Build for production
npm run preview          # Preview production build

# Code Quality
npm run lint             # Run ESLint
npm run format           # Format with Prettier
npm run type-check       # TypeScript type checking

# Testing
npm test                 # Run tests
npm test -- --watch      # Run tests in watch mode
npm test -- --coverage   # Run tests with coverage
```

### Makefile Commands (Root)

```bash
# Development
make dev                 # Start development environment
make build               # Build all components
make clean               # Remove build artifacts

# Testing
make test                # Run all tests (Rust + Web)
make test-rust           # Rust tests only
make test-web            # Web tests only

# Code Quality
make fmt                 # Format all code
make lint                # Run all linters
make check               # Quick check (fmt + lint + test)

# Holochain
make reset               # Reset Holochain state (dev only)
```

## Questions?

- **General questions**: Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- **Bug reports**: Open an [Issue](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)
- **Feature requests**: Open an [Issue](https://github.com/Luminous-Dynamics/mycelix-praxis/issues) with `type: feature` label
- **Real-time chat**: Join our community (link TBD)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache-2.0). See [LICENSE](LICENSE) for details.

---

Thank you for contributing to a more equitable, privacy-preserving education platform!
