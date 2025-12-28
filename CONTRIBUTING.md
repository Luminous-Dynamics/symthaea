# Contributing to Symthaea-HLB

Thank you for your interest in contributing to Symthaea-HLB! This document provides guidelines for contributing to this consciousness research project.

---

## 🌟 Project Overview

Symthaea-HLB is a Rust-based framework for consciousness measurement using:
- **Hyperdimensional Computing (HDC)** - High-dimensional vector operations
- **Integrated Information Theory (IIT)** - Φ (phi) measurement for consciousness
- **Topology Analysis** - Network structure → consciousness relationships

**Current Status**: Research-focused with production-ready foundations. We've achieved genuine scientific breakthroughs (Hypercube 4D dimensional invariance) and are preparing for publication.

---

## 🎯 Ways to Contribute

### Research Contributions
- Validate results on different hardware/configurations
- Test with larger topologies (n > 8 nodes)
- Apply to real neural data (C. elegans, human cortex)
- Propose new exotic topologies
- Improve Φ calculation methods

### Code Contributions
- Fix compilation warnings (194 currently)
- Add unit tests (targeting 70% coverage)
- Improve documentation
- Optimize performance
- Add new features from roadmap

### Documentation
- Improve README clarity
- Add examples
- Write tutorials
- Fix typos and broken links
- Translate documentation

---

## 🚀 Getting Started

### Prerequisites

**Required**:
- Rust 1.70+ (`rustup install stable`)
- Cargo (comes with Rust)

**Recommended**:
- `rustfmt` for code formatting: `rustup component add rustfmt`
- `clippy` for linting: `rustup component add clippy`
- `cargo-tarpaulin` for coverage: `cargo install cargo-tarpaulin`

### Setup

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/symthaea-hlb.git
cd symthaea-hlb

# Build the project
cargo build --lib

# Run tests
cargo test --lib

# Run examples (research validation)
cargo run --example tier_3_exotic_topologies --release
```

### Development Environment

We use:
- **Rust edition 2021**
- **16,384 HDC dimensions** (configurable)
- **Deterministic seeds** for reproducibility
- **Release mode** for research (performance-critical)

---

## 📋 Contribution Process

### 1. Find or Create an Issue

- Check [existing issues](https://github.com/Luminous-Dynamics/symthaea-hlb/issues)
- For bugs: Provide minimal reproduction steps
- For features: Describe use case and expected behavior
- For research: Explain hypothesis and expected findings

### 2. Fork and Branch

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/symthaea-hlb.git
cd symthaea-hlb

# Create a feature branch
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/issue-number-description
```

### 3. Make Changes

**Code Style**:
- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Follow Rust naming conventions (snake_case for functions/variables)
- Add doc comments for public APIs

**Testing**:
- Add tests for new functionality
- Ensure all tests pass: `cargo test --lib`
- Aim for >70% coverage for new code

**Documentation**:
- Update relevant `.md` files
- Add doc comments to code
- Update examples if API changes

### 4. Commit

**Commit Message Format**:
```
type(scope): Brief description

Detailed explanation if needed.

Closes #issue-number
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat(topology): Add 5D/6D hypercube generators"
git commit -m "fix(phi): Correct resonator-based Φ calculation for n=1"
git commit -m "docs: Add tutorial for custom topology creation"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear title describing the change
- Description of what and why
- Link to related issue(s)
- Screenshots/results if applicable

---

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all library tests
cargo test --lib

# Run specific test
cargo test test_hypercube_4d

# Run with output
cargo test -- --nocapture

# Run in release mode (for benchmarks)
cargo test --release
```

### Writing Tests

**Unit Tests** (in source files):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_generation() {
        let topology = ConsciousnessTopology::ring(8, 16384, 42);
        assert_eq!(topology.n_nodes, 8);
        assert_eq!(topology.dim, 16384);
    }
}
```

**Integration Tests** (in `tests/`):
```rust
// tests/phi_calculation.rs
use symthaea::hdc::phi_real::RealPhiCalculator;

#[test]
fn test_phi_star_greater_than_random() {
    // Test the fundamental hypothesis
    let star_phi = /* ... */;
    let random_phi = /* ... */;
    assert!(star_phi > random_phi);
}
```

### Test Requirements

- All tests must be deterministic (use fixed seeds)
- Tests should complete in <10s each
- Use descriptive test names
- Test edge cases (n=1, n=large, empty, etc.)

---

## 📝 Documentation Standards

### Code Documentation

```rust
/// Calculates integrated information (Φ) for a consciousness topology.
///
/// # Arguments
///
/// * `topology` - The network structure to analyze
/// * `method` - The calculation method (Heuristic, Spectral, Exact)
///
/// # Returns
///
/// The Φ value representing integrated information (0.0 to 1.0)
///
/// # Examples
///
/// ```
/// use symthaea::hdc::*;
///
/// let topology = ConsciousnessTopology::ring(8, 16384, 42);
/// let calculator = RealPhiCalculator::new();
/// let phi = calculator.compute(&topology.node_representations);
/// ```
pub fn calculate_phi(topology: &ConsciousnessTopology) -> f64 {
    // Implementation
}
```

### Markdown Documentation

- Use clear headings
- Include code examples
- Add diagrams where helpful
- Link to related docs
- Keep line length <120 chars

---

## 🔬 Research Contributions

### Validating Results

If you're validating our findings:

1. **Use same parameters**:
   - HDC dimensions: 16,384
   - Seeds: Document what you use
   - Sample size: ≥10 per topology

2. **Report complete results**:
   - Mean Φ and standard deviation
   - Hardware/OS used
   - Execution time
   - Any deviations from our findings

3. **Share data**:
   - Raw results (CSV preferred)
   - Analysis scripts
   - Visualization code

### Proposing New Topologies

1. **Describe the topology**:
   - Mathematical definition
   - Node count and connectivity
   - Biological/theoretical motivation

2. **Implement generator**:
   - Add to `consciousness_topology_generators.rs`
   - Follow existing patterns
   - Include edge generation
   - Use deterministic randomness

3. **Validate**:
   - Run with multiple seeds
   - Compare to existing topologies
   - Report Φ statistics
   - Document findings

---

## 🐛 Bug Reports

### Good Bug Reports Include

1. **Description**: What happened vs. what you expected
2. **Reproduction**: Minimal code to reproduce
3. **Environment**: OS, Rust version, hardware
4. **Output**: Error messages, stack traces
5. **Investigation**: What you've tried

### Example

```markdown
**Bug**: Ring topology Φ calculation returns NaN for n=1

**To Reproduce**:
```rust
let topology = ConsciousnessTopology::ring(1, 16384, 42);
let calc = RealPhiCalculator::new();
let phi = calc.compute(&topology.node_representations); // Returns NaN
```

**Environment**:
- OS: Ubuntu 22.04
- Rust: 1.75.0
- Hardware: AMD Ryzen 9 5950X

**Expected**: Φ = 0.0 (single node has no integration)

**Actual**: NaN (division by zero in eigenvalue calculation)
```

---

## 🎨 Code Style Guide

### Rust Conventions

**Naming**:
- `snake_case` for functions, variables, modules
- `CamelCase` for types, traits, enums
- `SCREAMING_SNAKE_CASE` for constants
- Descriptive names (avoid abbreviations)

**Example**:
```rust
const HDC_DIMENSION: usize = 16_384;

pub struct ConsciousnessTopology {
    pub n_nodes: usize,
    pub node_representations: Vec<RealHV>,
}

pub fn calculate_integrated_information(topology: &ConsciousnessTopology) -> f64 {
    // ...
}
```

**Formatting**:
- Run `cargo fmt` before committing
- Max line length: 100 chars
- 4-space indentation
- Trailing commas in multi-line lists

**Clippy**:
- Fix all clippy warnings
- Run: `cargo clippy -- -D warnings`
- Some lints can be allowed with `#[allow(clippy::lint_name)]`

---

## 📊 Performance Guidelines

### Benchmarking

```rust
// benches/phi_calculation.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_phi_calculation(c: &mut Criterion) {
    let topology = ConsciousnessTopology::ring(8, 16384, 42);

    c.bench_function("phi_ring_8_nodes", |b| {
        b.iter(|| {
            let calc = RealPhiCalculator::new();
            calc.compute(black_box(&topology.node_representations))
        })
    });
}

criterion_group!(benches, bench_phi_calculation);
criterion_main!(benches);
```

### Optimization Guidelines

- Profile before optimizing (`cargo flamegraph`)
- Measure improvements with benchmarks
- Document performance characteristics
- Consider memory vs. speed tradeoffs
- Use release mode for research: `--release`

---

## 🤝 Code Review Process

### What Reviewers Look For

1. **Correctness**: Does it work as intended?
2. **Tests**: Are there tests? Do they pass?
3. **Style**: Does it follow conventions?
4. **Documentation**: Is it documented?
5. **Performance**: Any obvious inefficiencies?
6. **Breaking changes**: Is API compatibility maintained?

### Responding to Feedback

- Be open to suggestions
- Ask questions if unclear
- Make requested changes
- Explain your reasoning if you disagree
- Thank reviewers for their time

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## ❓ Questions?

- **Documentation**: Start with `README.md` and `CLAUDE.md`
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/symthaea-hlb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/symthaea-hlb/discussions)
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com

---

## 🙏 Thank You!

Your contributions help advance consciousness research and make this project better for everyone. Whether you're fixing a typo or proposing a breakthrough topology, every contribution matters!

---

*This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.*
