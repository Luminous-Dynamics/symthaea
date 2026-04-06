# Contributing to @mycelix/sdk

Thank you for your interest in contributing to the Mycelix SDK! This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- Node.js 20.x or later
- npm 10.x or later
- Git

### Getting Started

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix.git
cd mycelix/sdk-ts

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test
```

### Recommended VS Code Extensions

- ESLint
- Prettier
- TypeScript and JavaScript Language Features

## Project Structure

```
sdk-ts/
├── src/                    # Source code
│   ├── index.ts           # Main entry point
│   ├── matl/              # MATL (Trust Layer) module
│   ├── epistemic/         # Epistemic (Truth Classification) module
│   ├── bridge/            # Cross-hApp communication
│   ├── fl/                # Federated Learning
│   ├── client/            # Holochain client bindings
│   ├── security/          # Cryptographic utilities
│   ├── validation/        # Input validation
│   ├── config/            # Configuration management
│   ├── utils/             # Shared utilities
│   ├── errors.ts          # Error types and handling
│   ├── integrations/      # Domain-specific integrations
│   │   ├── mail/
│   │   ├── marketplace/
│   │   ├── praxis/
│   │   └── ...
│   ├── react/             # React hooks
│   ├── svelte/            # Svelte stores
│   └── vue/               # Vue composables
├── tests/                  # Test files
│   ├── *.test.ts          # Unit tests
│   ├── property/          # Property-based tests (fast-check)
│   ├── integration/       # Integration tests
│   ├── e2e/               # End-to-end tests
│   └── conductor/         # Holochain conductor tests
├── benchmarks/             # Performance benchmarks
├── examples/               # Usage examples
└── docs/                   # Generated documentation
```

## Development Workflow

### Available Scripts

```bash
# Build
npm run build              # Compile TypeScript
npm run clean              # Remove build artifacts

# Testing
npm test                   # Run unit tests
npm run test:watch         # Run tests in watch mode
npm run test:coverage      # Run tests with coverage
npm run bench              # Run performance benchmarks

# Code Quality
npm run lint               # Check for linting errors
npm run lint:fix           # Auto-fix linting errors
npm run format             # Format code with Prettier
npm run format:check       # Check formatting
npm run typecheck          # Type checking without emit

# Size Analysis
npm run size:check         # Check bundle sizes against limits
npm run size:why           # Analyze what's in each bundle

# Documentation
npm run docs               # Generate TypeDoc documentation
npm run docs:serve         # Generate and serve docs locally

# Advanced
npm run mutate             # Run mutation testing
npm run mutate:incremental # Incremental mutation testing
npm run changelog          # Generate changelog from commits
```

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

**Examples:**
```
feat(matl): add adaptive threshold support
fix(bridge): handle empty message payloads
docs(readme): update installation instructions
test(fl): add property tests for gradient aggregation
perf(security): optimize constant-time comparison
```

## Code Standards

### TypeScript Guidelines

1. **Use strict typing**: Avoid `any` when possible
2. **Export types**: All public types should be exported
3. **Document public APIs**: Use JSDoc comments

```typescript
/**
 * Creates a new reputation score for an agent.
 *
 * @param agentId - The agent's unique identifier
 * @param initialScore - Starting score (0-1), defaults to 0.5
 * @returns A new ReputationScore instance
 *
 * @example
 * ```typescript
 * const reputation = createReputation('agent-123', 0.8);
 * console.log(reputation.value); // 0.8
 * ```
 */
export function createReputation(
  agentId: string,
  initialScore = 0.5
): ReputationScore {
  // ...
}
```

### Error Handling

Use the SDK's error system:

```typescript
import { ValidationError, MATLError, ErrorCode } from '@mycelix/sdk/errors';

// Throw typed errors
throw new ValidationError('Invalid score', ErrorCode.InvalidScore, {
  received: score,
  expected: '0-1 range'
});
```

### Security Considerations

- Never log sensitive data
- Use constant-time comparison for secrets
- Validate all inputs at module boundaries
- Follow the security patterns in `src/security/`

## Testing

### Test Structure

```typescript
import { describe, it, expect, beforeEach } from 'vitest';

describe('ModuleName', () => {
  describe('functionName', () => {
    it('should handle normal case', () => {
      // Arrange
      const input = createTestInput();

      // Act
      const result = functionName(input);

      // Assert
      expect(result).toEqual(expectedOutput);
    });

    it('should throw on invalid input', () => {
      expect(() => functionName(null)).toThrow(ValidationError);
    });
  });
});
```

### Property-Based Testing

For complex logic, add property tests using fast-check:

```typescript
import fc from 'fast-check';

describe('Property Tests', () => {
  it('aggregate should be bounded [0, 1]', () => {
    fc.assert(
      fc.property(
        fc.array(scoreArbitrary, { minLength: 1, maxLength: 20 }),
        (scores) => {
          const result = calculateAggregate(scores);
          expect(result).toBeGreaterThanOrEqual(0);
          expect(result).toBeLessThanOrEqual(1);
        }
      )
    );
  });
});
```

### Coverage Requirements

- Statements: 70%
- Branches: 60%
- Functions: 65%
- Lines: 70%

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** following the code standards

3. **Write tests** for new functionality

4. **Run the full check suite**:
   ```bash
   npm run typecheck && npm run lint && npm test && npm run size:check
   ```

5. **Commit** using conventional commits

6. **Push** and create a pull request

7. **PR Checks** (automatic):
   - Type checking
   - Linting
   - Unit tests with coverage
   - Bundle size check
   - Benchmarks
   - Mutation testing (for PRs)

8. **Review**: Address any feedback from maintainers

## Release Process

Releases are handled automatically when changes are merged to `main`:

1. Version is checked against npm registry
2. If changed, package is published
3. Changelog is generated from commits
4. Documentation is deployed to GitHub Pages

### Manual Release

```bash
# Patch release (0.6.0 → 0.6.1)
npm run release:patch

# Minor release (0.6.0 → 0.7.0)
npm run release:minor

# Major release (0.6.0 → 1.0.0)
npm run release:major
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix/discussions)
- **Email**: dev@luminousdynamics.org

## Code of Conduct

We are committed to providing a welcoming and inclusive experience. Please be respectful and constructive in all interactions.

---

Thank you for contributing to Mycelix! Your efforts help build decentralized trust infrastructure for everyone. 🍄
