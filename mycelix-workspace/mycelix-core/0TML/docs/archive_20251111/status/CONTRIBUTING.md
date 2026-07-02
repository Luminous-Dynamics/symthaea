# Contributing to Mycelix Protocol

Thank you for your interest in contributing to the Mycelix Protocol! This is an active research project with grant applications in progress, and we welcome contributions from the community.

## 🎯 Current Status

**Mycelix Protocol** is a research implementation combining:
- Byzantine-resistant federated learning (PoGQ mechanism)
- Hybrid DLT architecture (Holochain + Ethereum Layer-2)
- Self-sovereign identity and reputation systems

**Current Phase**: Phase 11 - Experimental validation (Stage 1 comprehensive experiments 67% complete)

## 🤝 How to Contribute

### Priority Areas

We're currently focused on these areas and would especially welcome help:

1. **Experimental Validation** - Help run Stage 1 experiments (Sets B and D)
   - Byzantine attack scenario testing
   - Performance benchmarking with GPU acceleration
   - Data analysis and visualization

2. **Documentation** - Improve clarity and accessibility
   - User guides and tutorials
   - API documentation
   - Architecture diagrams

3. **Integration Testing** - Test Ethereum L2 integration
   - Polygon testnet deployment
   - Smart contract testing
   - Cross-chain bridge validation

4. **Security Auditing** - Review PoGQ mechanism and reputation system
   - Cryptographic analysis
   - Attack vector identification
   - Security recommendations

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Luminous-Dynamics/Mycelix-Core
   cd Mycelix-Core/0TML
   ```

2. **Set Up Development Environment**
   ```bash
   # Enter Nix development environment
   nix develop

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```bash
   # Run quick validation
   python run_mini_validation.py

   # Run comprehensive tests
   pytest tests/
   ```

4. **Make Your Changes**
   - Create a feature branch: `git checkout -b feature/your-feature-name`
   - Make your changes with clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed

5. **Submit a Pull Request**
   - Push to your fork: `git push origin feature/your-feature-name`
   - Open a Pull Request with:
     - Clear description of the changes
     - Reference to any related issues
     - Test results showing your changes work
     - Documentation updates if applicable

## 📋 Contribution Guidelines

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Rust**: Follow Rust standard formatting (use `rustfmt`)
- **TypeScript/JavaScript**: Follow Airbnb style guide
- **Documentation**: Use clear, accessible language with examples

### Commit Messages

Use descriptive commit messages following this format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat(pogq): Add adaptive reputation threshold

Implement dynamic threshold adjustment based on network health metrics.
This improves Byzantine detection by 5.2% in high-attack scenarios.

Closes #42
```

### Testing Requirements

All contributions must include appropriate tests:

- **Unit Tests**: Test individual functions and modules
- **Integration Tests**: Test component interactions
- **Experimental Validation**: For PoGQ mechanism changes, include experimental results

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_pogq.py

# Run with coverage
pytest --cov=src tests/
```

### Documentation Requirements

- All public functions must have docstrings
- Complex algorithms should include inline comments
- New features require updates to relevant documentation files
- Architecture changes need diagram updates

## 🔬 Research Contributions

This is a research project, so we especially welcome:

- **Experimental Results**: Share findings from running experiments
- **Theoretical Analysis**: Mathematical proofs, security analyses
- **Literature Reviews**: Related work and citations
- **Algorithm Improvements**: Enhancements to PoGQ or aggregation mechanisms

When contributing research:
1. Clearly state assumptions and limitations
2. Provide reproducible experimental setup
3. Include statistical significance tests
4. Share raw data when possible (privacy-permitting)

## 🐛 Reporting Bugs

Use GitHub Issues to report bugs. Please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal example that demonstrates the problem
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: NixOS version, Python version, hardware specs
6. **Logs**: Relevant error messages or stack traces

**Example**:
```
**Description**: PoGQ validation fails with NaN values on CIFAR-10

**Steps to Reproduce**:
1. Run `python run_byzantine_suite.py --dataset cifar10`
2. Observe validation scores in results/

**Expected**: Quality scores in range [0, 1]
**Actual**: NaN values for 30% of gradients

**Environment**: NixOS 23.11, Python 3.11, NVIDIA RTX 3090
**Logs**: [paste relevant logs]
```

## 💡 Suggesting Features

We track feature requests in GitHub Issues with the `enhancement` label.

Before suggesting a feature:
1. Check existing issues to avoid duplicates
2. Review the [roadmap](README.md#-roadmap) to see if it's already planned
3. Consider how it aligns with project goals (Byzantine resistance, privacy, sovereignty)

Feature requests should include:
- **Use Case**: Who needs this and why?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches you considered
- **Ecosystem Impact**: How does it benefit Ethereum/Holochain ecosystems?

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bugs and feature requests
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com (project lead)

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md acknowledgments section
- Academic papers (for significant research contributions)
- Grant applications and reports (when applicable)

## 📄 License

By contributing to Mycelix Protocol, you agree that your contributions will be licensed under the project's open source license (MIT or Apache-2.0, to be finalized pending grant decisions).

All code will be open source to ensure permissionless use by the Ethereum ecosystem and broader research community.

## 🙏 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

**Current Grant Status**: Ethereum Foundation ESP application in progress (deadline November 15, 2025). Contributions during this period are especially valuable as they demonstrate community engagement.

**Thank you for helping build decentralized identity and reputation infrastructure for the Ethereum ecosystem!**
