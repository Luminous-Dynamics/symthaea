# Contributing to Symthaea

*How to help build consciousness-first AI.*

---

## Welcome

Thank you for your interest in contributing to Symthaea! This project is built on the belief that AI should enhance human consciousness, not diminish it. Every contribution - code, documentation, ideas, or feedback - helps advance that vision.

---

## Ways to Contribute

### Code Contributions
- Bug fixes
- Performance improvements
- New features
- Test coverage
- Documentation in code

### Documentation
- Improve clarity of existing docs
- Add examples and tutorials
- Translate documentation
- Fix typos and errors

### Research
- Validate findings
- Propose new approaches
- Analyze results
- Write papers

### Community
- Answer questions
- Help new contributors
- Share your use cases
- Provide feedback

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/symthaea-hlb.git
cd symthaea-hlb
```

### 2. Set Up Development Environment

```bash
# Standard Rust
cargo build

# Or with Nix
nix develop
```

### 3. Run Tests

```bash
cargo test
```

### 4. Make Changes

Create a branch for your work:
```bash
git checkout -b feature/your-feature-name
```

### 5. Submit Pull Request

Push your branch and open a PR on GitHub with:
- Clear description of changes
- Link to related issues
- Test coverage for new code

---

## Code Guidelines

### Style

We follow standard Rust conventions:

```rust
// Use snake_case for functions and variables
fn compute_phi_value(topology: &ConsciousnessTopology) -> f32

// Use CamelCase for types
struct ConsciousnessGraph

// Use SCREAMING_SNAKE_CASE for constants
const HDC_DIMENSION: usize = 16_384;

// Document public APIs
/// Computes the integrated information (Φ) for a topology.
///
/// # Arguments
/// * `representations` - Node representations in HDC space
///
/// # Returns
/// The Φ value in range [0, 1]
pub fn compute(&self, representations: &[RealHV]) -> f32
```

### Formatting

```bash
# Format code
cargo fmt

# Check lints
cargo clippy
```

### Testing

Every new feature should have tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_calculation() {
        let topology = ring(8, HDC_DIMENSION, 42);
        let calc = RealPhiCalculator::new();
        let phi = calc.compute(&topology.node_representations);
        assert!(phi > 0.49 && phi < 0.51, "Ring Φ should be ~0.495");
    }
}
```

### Documentation

Document significant code:

```rust
/// Represents a node in the consciousness graph.
///
/// Each node contains:
/// - Semantic representation (HDC vector)
/// - Dynamic state (LTC activation)
/// - Consciousness level (0.0 - 1.0)
/// - Optional self-reference (for autopoiesis)
pub struct ConsciousnessNode {
    /// HDC representation of this conscious state
    pub semantic: RealHV,

    /// Dynamic state from LTC network
    pub dynamic: Vec<f32>,

    /// Current consciousness level
    pub level: f32,

    /// Self-loop for autopoietic consciousness
    pub self_loop: Option<NodeId>,
}
```

---

## Commit Guidelines

### Format

```
type(scope): short description

Longer description if needed.

Fixes #123
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `chore`: Maintenance

### Examples

```
feat(hdc): add Möbius strip topology generator

Implements the Möbius strip as a consciousness topology,
demonstrating how 1D non-orientability affects Φ.

This is part of the topology research for the paper.
```

```
fix(consciousness): correct self-loop detection

Self-loops were not being detected when the consciousness
level threshold was exactly 0.9. Changed to >= comparison.

Fixes #42
```

---

## Pull Request Process

### Before Submitting

1. **Run all tests**: `cargo test`
2. **Run formatter**: `cargo fmt`
3. **Run linter**: `cargo clippy`
4. **Update documentation** if needed
5. **Add tests** for new functionality

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Motivation
Why this change is needed.

## Changes
- Change 1
- Change 2

## Testing
How this was tested.

## Related Issues
Fixes #123
```

### Review Process

1. Maintainer reviews within a few days
2. Address feedback in new commits
3. Squash if requested
4. Maintainer merges when approved

---

## Architecture Decisions

### When to Discuss First

Open an issue to discuss before implementing:

- Changes to core architecture
- New dependencies
- Breaking API changes
- Significant new features

### Decision Records

Major decisions are recorded in `docs/decisions/`. If your change warrants one:

```markdown
# Decision: [Title]

## Context
What is the situation?

## Decision
What was decided?

## Consequences
What are the implications?
```

---

## Areas Needing Help

### High Priority

1. **Test coverage** - We need more integration tests
2. **Documentation** - Examples and tutorials
3. **Performance** - Profiling and optimization
4. **Validation** - Testing claims empirically

### Good First Issues

Look for issues labeled `good-first-issue` on GitHub.

### Research Opportunities

- Validate Φ calculations against PyPhi
- Test new consciousness topologies
- Benchmark against other HDC implementations
- Explore new LTC architectures

---

## Communication

### GitHub Issues

- Bug reports
- Feature requests
- Questions about implementation

### GitHub Discussions

- General questions
- Ideas and proposals
- Show and tell

### Code of Conduct

Be respectful, constructive, and kind. We're building technology to enhance consciousness - let's model good consciousness in our interactions.

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Paper acknowledgments (for significant research contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

*"Consciousness grows through connection. Thank you for connecting with this project."*
