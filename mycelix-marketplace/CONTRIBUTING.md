# Contributing to Mycelix Marketplace

Thank you for your interest in contributing to Mycelix Marketplace! 🍄

We're building a truly decentralized peer-to-peer marketplace on Holochain, and we welcome contributions from developers, designers, writers, and community members of all skill levels.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to tristan.stoltz@evolvingresonantcocreationism.com.

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Node.js** 18+ (LTS recommended)
- **npm** 9+ or **pnpm** 8+
- **Git** for version control
- **Holochain conductor + `hc`** (recommended: `nix develop` for a pinned toolchain)
- Basic knowledge of TypeScript and Svelte (or willingness to learn!)

### Fork and Clone

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Mycelix-Marketplace.git
   cd Mycelix-Marketplace
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/Luminous-Dynamics/Mycelix-Marketplace.git
   ```

## 💻 Development Setup

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment example
cp .env.example .env.local

# Edit .env.local with your configuration
# (Pinata JWT, conductor URL, etc.)

# Start development server
npm run dev
```

The app will be available at http://localhost:5173

### Verify Setup

```bash
# Type check
npm run check

# Lint
npm run lint

# Build (ensure no errors)
npm run build
```

**Expected results:**
- ✅ 0 TypeScript errors
- ✅ Build completes successfully
- ⚠️ 8 accessibility warnings (acceptable)

## 🤝 How to Contribute

### Types of Contributions

We welcome many types of contributions:

#### 🐛 Bug Fixes
- Found a bug? Check if it's already reported in [Issues](https://github.com/Luminous-Dynamics/Mycelix-Marketplace/issues)
- If not, create a new issue using the bug report template
- Fork, fix, and submit a PR

#### ✨ New Features
- Have an idea? Open a feature request issue first to discuss
- Get feedback from maintainers before starting implementation
- Follow the established architecture patterns

#### 📚 Documentation
- Improve README, guides, or code comments
- Add examples or tutorials
- Fix typos or clarify confusing sections

#### 🎨 Design & UX
- Improve accessibility (our goal: WCAG 2.1 AA)
- Enhance user interface or user experience
- Create mockups or prototypes

#### 🧪 Testing
- Add unit tests
- Add E2E tests (Playwright)
- Improve test coverage

## 💎 Code Standards

### TypeScript

- **100% TypeScript**: All code must be fully typed
- **Strict mode**: No `any` types without justification
- **0 errors**: Maintain the perfect type safety record

```typescript
// ✅ Good
interface Listing {
  title: string;
  price: number;
}

// ❌ Bad
const listing: any = { ... };
```

### Accessibility

- **WCAG 2.1 AA**: All new features must be accessible
- **Semantic HTML**: Use proper HTML elements
- **ARIA attributes**: Add when necessary
- **Keyboard navigation**: All interactive elements must be keyboard accessible

```svelte
<!-- ✅ Good -->
<button on:click={handleClick} aria-label="Add to cart">
  Add to Cart
</button>

<!-- ❌ Bad -->
<div on:click={handleClick}>
  Add to Cart
</div>
```

### Svelte Components

- **Single responsibility**: Each component does one thing well
- **Props validation**: Use TypeScript interfaces
- **Reactive statements**: Use `$:` for derived state
- **Accessibility**: Include ARIA attributes

### Code Style

We use ESLint and Prettier for consistent code formatting:

```bash
# Format code
npm run format

# Check formatting
npm run format:check

# Lint
npm run lint
```

**Key conventions:**
- 2 spaces for indentation
- Single quotes for strings
- Semicolons required
- Trailing commas in multi-line structures

## 🔄 Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our code standards

3. **Test thoroughly**:
   ```bash
   npm run check   # Type check
   npm run lint    # Linting
   npm run build   # Build
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add amazing feature

   - Implement X
   - Update Y
   - Fix Z"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `style:` - Code style changes (formatting)
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

### Submitting the PR

1. Go to the [Pull Requests](https://github.com/Luminous-Dynamics/Mycelix-Marketplace/pulls) page
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:
   - Clear title describing the change
   - Description of what changed and why
   - Reference related issues
   - Screenshots (if UI changes)
   - Testing steps

### PR Review Process

- A maintainer will review your PR
- Address any requested changes
- Once approved, your PR will be merged!

**Review criteria:**
- ✅ Follows code standards
- ✅ Includes tests (when applicable)
- ✅ Documentation updated
- ✅ No TypeScript errors
- ✅ Accessibility maintained
- ✅ Builds successfully

## 🧪 Testing

### Running Tests

```bash
# Unit tests (coming in Phase 5)
npm run test

# E2E tests (coming in Phase 5)
npm run test:e2e

# Watch mode
npm run test:watch
```

### Writing Tests

- Write tests for new features
- Maintain or improve test coverage
- Test edge cases and error conditions
- Use descriptive test names

```typescript
// Example test structure
describe('CartStore', () => {
  it('should add item to cart', () => {
    // Test implementation
  });

  it('should update quantity', () => {
    // Test implementation
  });
});
```

## 📚 Documentation

### Code Documentation

- **Inline comments**: Explain complex logic
- **JSDoc**: Document public functions and interfaces
- **README updates**: Keep documentation current

```typescript
/**
 * Calculates trust score using PoGQ algorithm
 * @param transactions - Array of user transactions
 * @param reviews - Array of user reviews
 * @returns Normalized trust score (0-100)
 */
function calculateTrustScore(
  transactions: Transaction[],
  reviews: Review[]
): number {
  // Implementation
}
```

### Documentation Files

- Update relevant `.md` files for significant changes
- Add examples for new features
- Keep CHANGELOG.md updated

## 🌟 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions, ideas, and community chat
- **Holochain Forum**: Broader Holochain community discussions

### Getting Help

- Check existing documentation
- Search [Issues](https://github.com/Luminous-Dynamics/Mycelix-Marketplace/issues)
- Ask in [Discussions](https://github.com/Luminous-Dynamics/Mycelix-Marketplace/discussions)
- Join the Holochain community

### Recognition

All contributors are recognized in our README and release notes. Thank you for making Mycelix Marketplace better!

## 🎯 Good First Issues

New to the project? Look for issues labeled:
- `good first issue` - Perfect for newcomers
- `help wanted` - We'd love your contribution
- `documentation` - Improve our docs

## 📝 License

By contributing to Mycelix Marketplace, you agree that your contributions will be licensed under the Apache License 2.0.

## ❓ Questions?

Don't hesitate to ask questions! We're here to help:
- Open a [Discussion](https://github.com/Luminous-Dynamics/Mycelix-Marketplace/discussions)
- Comment on an issue
- Reach out to maintainers

---

**Thank you for contributing to the future of decentralized commerce!** 🍄

*Together, we're building something amazing.*

---

*Last Updated: November 13, 2025*
