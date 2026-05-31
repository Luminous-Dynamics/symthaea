# Contributing to Mycelix Knowledge

Thank you for your interest in contributing to Mycelix Knowledge! This guide will help you get started.

## Ways to Contribute

### 1. Submit Knowledge Claims
The simplest way to contribute is by submitting high-quality claims:
- Well-sourced factual claims
- Evidence from reputable sources
- Proper epistemic classification

### 2. Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation

### 3. Documentation
- Improve existing docs
- Add examples
- Write tutorials
- Translate documentation

### 4. Testing
- Write integration tests
- Run simulations
- Report bugs
- Test on different platforms

### 5. Review
- Review pull requests
- Test new features
- Provide feedback

## Development Setup

### Prerequisites
- Holochain v0.4+
- Rust with wasm32-unknown-unknown target
- Node.js 18+
- Nix (recommended)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-knowledge
cd mycelix-knowledge

# Enter Nix development environment
nix develop

# Build the zomes
cargo build --release --target wasm32-unknown-unknown

# Run tests
cargo test

# Build SDK
cd client && npm install && npm run build
```

## Code Style

### Rust
- Follow Rust naming conventions
- Use `rustfmt` for formatting
- Run `cargo clippy` before committing
- Document public functions

```rust
/// Creates a new claim in the knowledge graph.
///
/// # Arguments
/// * `input` - The claim creation input
///
/// # Returns
/// * `ExternResult<ActionHash>` - The hash of the created claim
#[hdk_extern]
pub fn create_claim(input: CreateClaimInput) -> ExternResult<ActionHash> {
    // Implementation
}
```

### TypeScript
- Use TypeScript strict mode
- Run `npm run lint` before committing
- Document exported functions

```typescript
/**
 * Creates a new knowledge claim.
 * @param input - Claim creation input
 * @returns Hash of the created claim
 */
async createClaim(input: CreateClaimInput): Promise<ActionHash> {
  // Implementation
}
```

## Pull Request Process

### 1. Fork and Branch
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR-USERNAME/mycelix-knowledge
cd mycelix-knowledge
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write tests for new features
- Update documentation
- Follow code style guidelines

### 3. Test
```bash
# Run Rust tests
cargo test

# Run TypeScript tests
cd client && npm test

# Run integration tests
cd tests && npm test
```

### 4. Commit
```bash
# Use conventional commits
git commit -m "feat: add new credibility factor"
git commit -m "fix: correct belief propagation convergence"
git commit -m "docs: update API reference for factcheck"
```

### 5. Push and PR
```bash
git push origin feature/your-feature-name
# Open PR on GitHub
```

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(claims): add support for multi-evidence claims
fix(graph): correct circular dependency detection
docs(api): add examples for batch operations
test(inference): add credibility calculation tests
```

## Testing Guidelines

### Unit Tests
Test individual functions:
```rust
#[test]
fn test_epistemic_validation() {
    let claim = Claim {
        classification: EpistemicPosition { e: 0.5, n: 0.3, m: 0.2 },
        ..Default::default()
    };
    assert!(validate_epistemic(&claim.classification).is_ok());
}
```

### Integration Tests
Test zome interactions:
```typescript
test("create and retrieve claim", async () => {
  const hash = await knowledge.claims.createClaim({
    content: "Test claim",
    classification: { empirical: 0.5, normative: 0.3, mythic: 0.2 },
    domain: "test"
  });

  const claim = await knowledge.claims.getClaim(hash);
  expect(claim.content).toBe("Test claim");
});
```

## Documentation Guidelines

- Use clear, concise language
- Include code examples
- Keep examples up to date
- Cross-reference related docs

## Issue Reporting

### Bug Reports
Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details

### Feature Requests
Include:
- Problem description
- Proposed solution
- Use cases
- Alternatives considered

## Code of Conduct

### Our Pledge
We pledge to make participation welcoming and harassment-free.

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy

### Enforcement
Violations may be reported to: conduct@luminousdynamics.org

## Getting Help

- [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-knowledge/issues)
- [Discord Community](https://discord.gg/mycelix)
- [Documentation](./README.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

*Thank you for helping build the decentralized knowledge commons!*
