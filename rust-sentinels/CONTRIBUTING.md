# Contributing to Rust Sentinels

Thank you for your interest in contributing to Rust Sentinels! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We welcome contributors of all backgrounds and experience levels.

## Ways to Contribute

### 1. Bug Reports

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Rust version, etc.)
- Sample data if applicable (anonymized)

### 2. Feature Requests

We welcome feature requests! Please open an issue with:
- A clear description of the proposed feature
- The problem it solves or value it adds
- Any relevant research or references
- Example use cases

### 3. Code Contributions

#### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/luminous-dynamics/rust-sentinels
cd rust-sentinels

# Build
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Format code
cargo fmt

# Check lints
cargo clippy
```

#### Code Style

- Follow Rust idioms and conventions
- Use `rustfmt` for formatting
- Address all `clippy` warnings
- Write documentation for public APIs
- Include unit tests for new functionality

#### Commit Messages

Use conventional commits:
```
feat: add new feature
fix: fix a bug
docs: documentation changes
test: add or update tests
refactor: code refactoring
perf: performance improvements
chore: maintenance tasks
```

#### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Run clippy: `cargo clippy`
6. Format code: `cargo fmt`
7. Commit with a descriptive message
8. Push to your fork
9. Open a Pull Request

#### PR Requirements

- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Code is formatted
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG updated (for user-facing changes)

### 4. Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples
- Improve API documentation
- Write tutorials

### 5. Dataset Validation

Help us validate our algorithms:
- Run validation scripts on new datasets
- Report results and findings
- Suggest threshold improvements
- Identify edge cases

## Areas of Interest

### High Priority

1. **Extended Proofs Implementation**
   - AttentionSentinel
   - FlowSentinel
   - EngagementSentinel

2. **Python Bindings (PyO3)**
   - Wrap core functionality
   - NumPy array support
   - Pythonic API design

3. **WebAssembly Support**
   - Browser-compatible build
   - JavaScript API
   - Demo application

### Medium Priority

4. **Multi-Channel Support**
   - Spatial filtering
   - Source localization
   - Channel selection algorithms

5. **Artifact Rejection**
   - Eye blink detection
   - Muscle artifact removal
   - Automatic quality assessment

6. **Additional Datasets**
   - DEAP emotion validation
   - DREAMER emotion validation
   - Clinical sleep data

### Good First Issues

Look for issues labeled `good first issue`:
- Documentation improvements
- Test coverage expansion
- Small bug fixes
- Code cleanup

## Research Contributions

If you're a researcher:
- Share validation results
- Propose algorithm improvements
- Contribute literature references
- Suggest new applications

## Questions?

- Open a GitHub Discussion for general questions
- Open an Issue for specific problems
- Email: tristan.stoltz@evolvingresonantcocreationism.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for helping make consciousness detection accessible to everyone!*
