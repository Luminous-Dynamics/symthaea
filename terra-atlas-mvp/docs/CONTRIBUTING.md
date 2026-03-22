# Contributing to Terra Atlas 🌍

Thank you for your interest in contributing to Terra Atlas! We're building the future of democratic energy investment together.

## 🎯 Our Mission

Terra Atlas democratizes global energy investment by connecting everyday investors with renewable energy opportunities worldwide. Every contribution helps make clean energy accessible to all.

## 🤝 Ways to Contribute

### 1. Code Contributions
- **Bug fixes** - Help us squash bugs
- **Feature development** - Add new capabilities
- **Performance improvements** - Make it faster
- **UI/UX enhancements** - Make it beautiful
- **Testing** - Improve test coverage
- **Documentation** - Help others understand

### 2. Data Contributions
- **Energy project data** - Add new renewable projects
- **Validation** - Verify existing data accuracy
- **Geographic expansion** - Add new regions
- **Translation** - Internationalize the platform

### 3. Community Contributions
- **Bug reports** - Tell us what's broken
- **Feature requests** - Share your ideas
- **Documentation** - Improve guides and tutorials
- **Community support** - Help other users

## 🚀 Getting Started

### Prerequisites
- Node.js 20+
- Git
- GitHub account
- Supabase account (for database features)

### Setup Process

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/terra-atlas.git
   cd terra-atlas
   ```

3. **Set up upstream**
   ```bash
   git remote add upstream https://github.com/Luminous-Dynamics/terra-atlas.git
   ```

4. **Install dependencies**
   ```bash
   npm install
   ```

5. **Set up environment**
   ```bash
   cp .env.example .env.local
   # Add your Supabase credentials
   ```

6. **Run development server**
   ```bash
   npm run dev
   ```

## 📋 Development Workflow

### 1. Create a branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make your changes
- Write clean, readable code
- Follow existing patterns
- Add comments for complex logic
- Update tests if needed

### 3. Test your changes
```bash
npm run lint        # Check code style
npm run typecheck   # TypeScript validation
npm test           # Run tests
npm run build      # Build production
```

### 4. Commit your changes
```bash
git add .
git commit -m "feat: add new feature" # Use conventional commits
```

### Commit Message Format
We use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Testing
- `chore:` Maintenance

### 5. Push to your fork
```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Go to GitHub and create a PR
- Fill out the PR template
- Link related issues
- Wait for review

## 🎨 Code Style

### TypeScript/JavaScript
- Use TypeScript for all new code
- Follow ESLint configuration
- Use meaningful variable names
- Prefer functional components
- Use hooks over class components

### CSS/Styling
- Use Tailwind CSS classes
- Follow mobile-first approach
- Ensure accessibility (WCAG 2.1)
- Test on multiple screen sizes

### Best Practices
- **DRY** - Don't Repeat Yourself
- **KISS** - Keep It Simple, Stupid
- **YAGNI** - You Aren't Gonna Need It
- **Clean Code** - Readable > Clever

## 🧪 Testing

### Running Tests
```bash
npm test              # Run all tests
npm run test:watch    # Watch mode
npm run test:coverage # Coverage report
```

### Writing Tests
- Test user behavior, not implementation
- Cover edge cases
- Maintain >80% coverage for new code
- Use meaningful test descriptions

## 📊 Data Standards

### Energy Project Data
```json
{
  "id": "unique-identifier",
  "name": "Project Name",
  "type": "solar|wind|hydro|nuclear|storage",
  "capacity_mw": 100,
  "location": {
    "lat": 40.7128,
    "lng": -74.0060,
    "country": "US",
    "state": "NY"
  },
  "status": "planning|construction|operational",
  "investment_needed": 50000000,
  "expected_roi": 0.12
}
```

### Data Quality
- Verify sources
- Include references
- Update timestamps
- Flag uncertain data

## 🔒 Security

### Reporting Security Issues
**DO NOT** create public issues for security vulnerabilities.
Email: security@luminousdynamics.org

### Security Best Practices
- Never commit secrets or API keys
- Use environment variables
- Validate all user input
- Sanitize database queries
- Follow OWASP guidelines

## 📝 Documentation

### Code Documentation
- Add JSDoc comments for functions
- Document complex algorithms
- Update README for new features
- Keep API documentation current

### User Documentation
- Write clear, simple language
- Include examples
- Add screenshots when helpful
- Test instructions yourself

## 🌐 Internationalization

We aim to support multiple languages:
- Use translation keys, not hardcoded text
- Support RTL languages
- Consider cultural differences
- Test with different locales

## 🎯 Priority Areas

Current high-priority contributions:
1. **FERC data integration** - Real-time queue updates
2. **USACE dam database** - 87,000 sites
3. **Investment calculator** - Financial modeling
4. **Mobile optimization** - Responsive design
5. **Performance** - Load time optimization
6. **Accessibility** - WCAG compliance

## 🙏 Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Eligible for special badges
- Invited to contributor calls

## 📬 Communication

- **GitHub Issues** - Bug reports and features
- **Discussions** - General questions
- **Discord** - Real-time chat
- **Email** - contribute@luminousdynamics.org

## 🚫 What NOT to Do

- Don't break existing functionality
- Don't introduce dependencies without discussion
- Don't commit large binary files
- Don't change core architecture without RFC
- Don't harass or discriminate

## 📜 Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be kind, respectful, and inclusive.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🎉 Thank You!

Every contribution, no matter how small, helps democratize energy investment globally. Together, we're building a sustainable future!

---

*Questions? Join our [Discord](https://discord.gg/luminous) or open a [Discussion](https://github.com/Luminous-Dynamics/terra-atlas/discussions)*