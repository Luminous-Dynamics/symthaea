# Contributing to Mycelix Music

Thank you for your interest in contributing to Mycelix Music! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Code**: Frontend, backend, smart contracts, tests
- **Documentation**: Improve guides, fix typos, add examples
- **Design**: UI/UX improvements, graphics, branding
- **Testing**: Report bugs, test new features
- **Ideas**: Suggest features, provide feedback

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mycelix-music.git
cd mycelix-music

# Add upstream remote
git remote add upstream https://github.com/YOUR_ORG/mycelix-music.git
```

### 2. Set Up Development Environment

```bash
# Install dependencies
npm install

# Copy environment template
cp .env.example .env

# Start development servers
npm run dev
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/amazing-feature

# Or a bugfix branch
git checkout -b fix/bug-description
```

## 📝 Development Guidelines

### Code Style

- **TypeScript**: Use strict mode, proper types
- **React**: Functional components with hooks
- **Naming**: descriptive camelCase for variables, PascalCase for components
- **Comments**: Explain *why*, not *what*

### Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run specific test file
npm test -- apps/web/src/components/MusicPlayer.test.tsx
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add artist profile page
fix: resolve music player skip bug
docs: update API documentation
style: format code with prettier
refactor: simplify payment routing logic
test: add tests for upload component
chore: update dependencies
```

## 🔄 Pull Request Process

### 1. Keep Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Make Your Changes

- Write clear, concise code
- Add tests for new features
- Update documentation as needed
- Follow existing code style

### 3. Test Thoroughly

```bash
# Run all tests
npm test

# Check TypeScript types
npm run type-check

# Run linter
npm run lint

# Format code
npm run format
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

### 5. Create Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Select your feature branch
- Fill out the PR template:
  - **Description**: What does this PR do?
  - **Motivation**: Why is this change needed?
  - **Testing**: How was this tested?
  - **Screenshots**: If UI changes

### 6. Code Review

- Address feedback from reviewers
- Push additional commits if needed
- Be responsive and collaborative

## 🐛 Reporting Bugs

### Before Submitting

1. Check existing issues
2. Try latest version
3. Reproduce the bug

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. macOS 13.0]
 - Browser: [e.g. Chrome 120]
 - Version: [e.g. v1.0.0]

**Additional context**
Any other context about the problem.
```

## ✨ Suggesting Features

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other context or screenshots.
```

## 📚 Documentation Contributions

- Fix typos and grammar
- Improve clarity
- Add examples
- Update outdated information
- Translate to other languages

## 🎨 Design Contributions

- Follow existing design system
- Maintain accessibility standards
- Provide Figma files or mockups
- Consider mobile responsiveness

## 💰 Smart Contract Contributions

**IMPORTANT**: Smart contracts require extra care.

### Testing Requirements

- 100% test coverage
- Gas optimization analysis
- Security audit considerations
- Integration tests with frontend

### Deployment Process

1. Test on local network
2. Deploy to testnet (Gnosis Chiado)
3. Frontend integration testing
4. Security review
5. Mainnet deployment (with multisig)

## 🔒 Security

**Do NOT open public issues for security vulnerabilities.**

Email security@mycelix.net with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Eligible for community rewards (when available)

## 💬 Community

- **Discord**: [Join our community](https://discord.gg/mycelix)
- **Twitter**: [@MycelixMusic](https://twitter.com/MycelixMusic)
- **Forum**: [discussions.mycelix.net](https://discussions.mycelix.net)

## ❓ Questions?

- Check [FAQ](docs/guides/FAQ.md)
- Ask in Discord #dev-help channel
- Email dev@mycelix.net

---

**Thank you for contributing to Mycelix Music!** 🎵

Your contributions help create a fairer music ecosystem for artists worldwide.
