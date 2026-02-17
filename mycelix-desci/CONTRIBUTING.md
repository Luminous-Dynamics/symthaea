# Contributing to Mycelix-DeSci

Thank you for your interest in contributing to Mycelix-DeSci! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in discussions
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Rust**: 1.75+ ([rustup](https://rustup.rs/))
- **Python**: 3.11+ ([python.org](https://www.python.org/))
- **Node.js**: 20+ ([nodejs.org](https://nodejs.org/))
- **Git**: For version control

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/mycelix-desci.git
   cd mycelix-desci
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/luminousdynamics/mycelix-desci.git
   ```

### Setup Development Environment

#### Rust Components

```bash
# Build Rust workspace
cargo build

# Run tests
cargo test
```

#### Python ML Components

```bash
cd src/ml
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"
pytest
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Development Workflow

### Creating a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring
- `test/*` - Test additions/improvements

### Making Changes

1. Make your changes in logical commits
2. Write clear, descriptive commit messages
3. Add tests for new functionality
4. Update documentation as needed

### Commit Message Format

Follow conventional commits format:

```
type(scope): brief description

Longer explanation if needed

Fixes #issue-number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code restructuring
- `test`: Test additions/modifications
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(pogq): implement Byzantine detection algorithm

Add statistical outlier detection for gradient validation
using median-based consensus mechanism.

Fixes #42
```

```
fix(storage): handle IPFS timeout errors

Add retry logic with exponential backoff for IPFS uploads.
```

## Coding Standards

### Rust

- Follow the [Rust Style Guide](https://doc.rust-lang.org/style-guide/)
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes with no warnings
- Add documentation for public APIs
- Write unit tests for new functionality

Example:
```rust
/// Validates a batch of gradient updates for Byzantine behavior
///
/// # Arguments
/// * `gradients` - Slice of gradient updates to validate
///
/// # Returns
/// Vector of quality scores for each gradient
///
/// # Errors
/// Returns `Error::PoGQ` if no gradients provided
pub fn validate_gradients(&self, gradients: &[GradientUpdate]) -> Result<Vec<QualityScore>> {
    // Implementation
}
```

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Run `black` for formatting
- Use type hints for function signatures
- Add docstrings for classes and functions
- Ensure `mypy` type checking passes

Example:
```python
def validate_gradients(
    self, gradients: List[GradientUpdate]
) -> List[QualityScore]:
    """Validate a batch of gradient updates.

    Args:
        gradients: List of gradient updates from participants

    Returns:
        List of quality scores for each gradient

    Raises:
        ValueError: If no gradients provided
    """
    # Implementation
```

### TypeScript/Svelte

- Use TypeScript for type safety
- Follow [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)
- Run `npm run format` before committing
- Use meaningful variable and function names
- Add JSDoc comments for complex functions

## Testing

### Rust Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with coverage (requires cargo-llvm-cov)
cargo llvm-cov
```

### Python Tests

```bash
cd src/ml
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest -k test_name       # Run specific test
pytest --cov              # With coverage
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Test Coverage

Aim for:
- **Rust**: 80%+ coverage for core modules
- **Python**: 80%+ coverage for ML components
- **Frontend**: 70%+ coverage for components

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Rust
   cargo fmt --all -- --check
   cargo clippy --all-features -- -D warnings
   cargo test

   # Python
   cd src/ml
   black --check .
   mypy mycelix_desci_ml
   pytest

   # Frontend
   cd frontend
   npm run check
   npm run lint
   npm run build
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Submitting

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub

3. Fill out the PR template completely

4. Link related issues using "Fixes #123" or "Closes #456"

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] No new warnings
```

### Review Process

- At least one maintainer must approve
- All CI checks must pass
- Address review comments promptly
- Squash commits if requested
- Be patient - reviews may take a few days

## Community

### Communication Channels

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat (coming soon)

### Getting Help

- Check existing documentation in `/docs`
- Search closed issues
- Ask in GitHub Discussions
- Join community calls (schedule TBD)

### Bounties and Rewards

We offer bounties for significant contributions:

- **Bug fixes**: $50-$500
- **Features**: $500-$2000
- **Integrations**: $1000-$5000
- **Security findings**: Up to $10,000

See [BOUNTIES.md](docs/BOUNTIES.md) for details (coming soon).

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Eligible for governance tokens (future)

## Questions?

Don't hesitate to ask! Open a discussion or reach out to maintainers.

Thank you for contributing to decentralized science! 🔬
