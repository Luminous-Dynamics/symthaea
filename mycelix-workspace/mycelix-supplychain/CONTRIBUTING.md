# Contributing to mycelix-supplychain

Thank you for your interest in contributing to the Mycelix Supply Chain project!

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR-USERNAME/mycelix-supplychain.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Test** thoroughly
6. **Commit** with clear messages
7. **Push** and create a Pull Request

## Development Setup

### Rust

```bash
cd rust/service
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

### TypeScript

```bash
cd ts/sdk
npm install
npm run build
npm test
npm run lint
```

### Python (optional adapters)

```bash
cd adapters/python
pip install -r requirements.txt
python -m pytest
```

## Code Standards

### Rust
- Use `rustfmt` (run `cargo fmt`)
- Pass `clippy` with no warnings
- Write unit tests for new functionality
- Document public APIs with `///` doc comments
- Follow naming conventions: `snake_case` for functions/variables, `PascalCase` for types

### TypeScript
- Use ESLint + Prettier (configured in repo)
- Prefer `const` over `let`; avoid `any`
- Write JSDoc for exported functions
- Use strict TypeScript (`strict: true`)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add batch lineage query endpoint
fix: correct timestamp parsing in CSV adapter
docs: update OpenAPI schema for ShipEvent
test: add e2e test for certification flow
chore: upgrade tokio to 1.35
```

## Pull Request Process

1. **Update documentation** if you change APIs or behavior
2. **Add tests** that cover your changes
3. **Ensure CI passes** (all workflows green)
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested before merge

### PR Checklist

- [ ] Code builds without warnings
- [ ] Tests added/updated and passing
- [ ] Documentation updated (README, OpenAPI, schemas)
- [ ] No secrets or credentials committed
- [ ] Breaking changes clearly marked in PR description

## Reporting Issues

### Bugs

Please include:
- **Description** of the bug
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment** (OS, Rust version, Node version)
- **Logs or screenshots** if applicable

### Feature Requests

Please include:
- **What** should change
- **Why** this matters (use case)
- **How** you'd measure success (acceptance criteria)

## Code of Conduct

Be respectful, inclusive, and professional. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

**Do not** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for reporting process.

## 🎯 Areas We Need Help

### High Priority
- **Authentication & Multi-tenancy**: Implementing JWT auth and tenant isolation
- **React Dashboard**: Building the web UI
- **Integration Testing**: More comprehensive test scenarios
- **Performance Optimization**: Query optimization, caching strategies

### Good First Issues
Look for issues tagged with `good-first-issue` - these are well-defined, good for newcomers, and include clear guidance.

### Community Contributions
- **Translations**: Internationalization support
- **Demo Scenarios**: Industry-specific examples beyond our current 6
- **Integrations**: Connect with other tools (QuickBooks, Stripe, etc.)
- **Deployment Guides**: Docker, Kubernetes, cloud platforms

## 💬 Questions?

- Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-supplychain/discussions)
- Tag issues with `question` label
- Join our [Discord community](https://discord.gg/mycelix)
- Email: dev@mycelix.net

## 🌟 Recognition

We value all contributions! Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Offered early access to new features

## 📅 Release Cycle

- **Alpha** (current): Monthly releases, breaking changes allowed
- **Beta** (Q1 2026): Bi-weekly releases, limited breaking changes
- **Stable** (Q2 2026): Monthly releases, semantic versioning

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.

---

## 🙏 Thank You!

Every contribution makes Mycelix better for everyone. Whether you're fixing a typo, reporting a bug, adding a feature, or helping other users - **you're making a difference!**

**Happy coding! 🚀**

*Last updated: December 30, 2025*
