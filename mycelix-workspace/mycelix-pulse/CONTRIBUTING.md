# Contributing to Mycelix Mail

Thank you for your interest in contributing to Mycelix Mail! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- **Rust** 1.75+ (for backend)
- **Node.js** 20+ (for frontend)
- **PostgreSQL** 15+
- **Redis** 7+
- **Docker** and Docker Compose (optional but recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mycelix/mycelix-mail.git
cd mycelix-mail

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Install frontend dependencies
cd ui/frontend && npm install

# Start the development server
npm run dev
```

## Development Setup

### Backend (Rust)

```bash
cd backend

# Install dependencies
cargo build

# Run tests
cargo test

# Run the API server
cargo run --bin api

# Run with hot reload
cargo watch -x run
```

### Frontend (TypeScript/React)

```bash
cd ui/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Type check
npm run typecheck

# Lint
npm run lint
```

### Database Migrations

```bash
# Run migrations
cargo run --bin migrate up

# Create a new migration
cargo run --bin migrate create <migration_name>

# Rollback last migration
cargo run --bin migrate down
```

## Architecture Overview

```
mycelix-mail/
├── backend/
│   ├── api/           # REST API server
│   ├── core/          # Core business logic (DNA)
│   ├── trust/         # Web-of-trust implementation
│   ├── crypto/        # Encryption and key management
│   └── worker/        # Background job processing
├── ui/
│   └── frontend/      # React TypeScript frontend
├── mobile/            # React Native mobile app
├── extension/         # Browser extension
├── sdk/               # Python and TypeScript SDKs
├── deploy/            # Deployment configurations
├── docs/              # Documentation
└── tests/             # Integration and E2E tests
```

### Key Concepts

1. **DNA (Decentralized Network Architecture)**: The core protocol layer
2. **Web of Trust**: Decentralized trust attestation system
3. **End-to-End Encryption**: PGP-based encryption for all messages
4. **Privacy First**: No tracking, minimal data collection

## Contribution Guidelines

### Types of Contributions

We welcome:

- **Bug fixes**: Report and fix bugs
- **Features**: Propose and implement new features
- **Documentation**: Improve docs, examples, and guides
- **Tests**: Add or improve test coverage
- **Performance**: Optimization and benchmarking
- **Translations**: Help translate the UI
- **Security**: Report vulnerabilities responsibly

### Before You Start

1. **Check existing issues**: Look for similar issues or features
2. **Open a discussion**: For major changes, discuss first
3. **Create an issue**: Document what you plan to work on
4. **Fork the repository**: Work on your own fork

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(trust): add attestation expiry notifications
fix(email): handle unicode subjects correctly
docs(api): add webhook examples
```

### 4. Open a Pull Request

- Fill out the PR template completely
- Link related issues
- Request reviews from maintainers
- Respond to feedback promptly

### 5. Review Process

- All PRs require at least one approval
- CI must pass (tests, linting, type checking)
- Security-sensitive changes need security review
- Breaking changes require additional review

## Code Style

### Rust

We follow the Rust style guide with `rustfmt`:

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features
```

Key conventions:
- Use `Result<T, E>` for fallible operations
- Prefer `?` operator over `unwrap()`
- Document public APIs with `///` comments
- Use `#[derive]` for common traits

### TypeScript/React

We use ESLint and Prettier:

```bash
# Lint and format
npm run lint
npm run format

# Type check
npm run typecheck
```

Key conventions:
- Use functional components with hooks
- Prefer named exports
- Use TypeScript strict mode
- Avoid `any` type

### File Organization

```typescript
// Imports (grouped)
import React from 'react';
import { useQuery } from '@tanstack/react-query';

import { Button } from '@/components/ui/button';
import { useAuth } from '@/hooks/useAuth';

// Types
interface Props {
  // ...
}

// Component
export const MyComponent: React.FC<Props> = ({ ... }) => {
  // Hooks first
  const { user } = useAuth();

  // State
  const [state, setState] = useState();

  // Effects
  useEffect(() => {}, []);

  // Handlers
  const handleClick = () => {};

  // Render
  return (
    // ...
  );
};
```

## Testing

### Backend Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with coverage
cargo tarpaulin --out Html
```

### Frontend Tests

```bash
# Run tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Test Guidelines

1. **Unit tests**: Test individual functions/components
2. **Integration tests**: Test module interactions
3. **E2E tests**: Test complete user flows
4. **Property tests**: Use `proptest` for invariants

Example test structure:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_works() {
        // Arrange
        let input = ...;

        // Act
        let result = function(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

## Documentation

### Code Documentation

- Document all public APIs
- Include examples in doc comments
- Keep documentation up to date

```rust
/// Calculates the trust score for an email address.
///
/// # Arguments
///
/// * `email` - The email address to evaluate
/// * `attestations` - Known trust attestations
///
/// # Returns
///
/// A trust score between 0.0 and 1.0
///
/// # Example
///
/// ```
/// let score = calculate_trust_score("user@example.com", &attestations);
/// assert!(score >= 0.0 && score <= 1.0);
/// ```
pub fn calculate_trust_score(email: &str, attestations: &[Attestation]) -> f64 {
    // ...
}
```

### API Documentation

- Update OpenAPI specs for API changes
- Include request/response examples
- Document error codes

### Architecture Decision Records (ADRs)

For significant decisions, create an ADR in `docs/adr/`:

```markdown
# ADR-001: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
Why is this decision needed?

## Decision
What is the change?

## Consequences
What are the results?
```

## Community

### Getting Help

- **Discord**: [Join our Discord](https://discord.gg/mycelix)
- **Discussions**: GitHub Discussions for questions
- **Issues**: GitHub Issues for bugs

### Communication

- Be respectful and constructive
- Assume good faith
- Help others learn
- Celebrate contributions

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Monthly community highlights

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Mycelix Mail! Your efforts help build a more private and trustworthy email experience for everyone.
