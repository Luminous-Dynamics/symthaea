# Pull Request

## Description

<!-- Provide a clear and concise description of your changes -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] 🎨 Code style/refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test addition or update
- [ ] 🔧 Configuration change

## Related Issues

<!-- Link to related issues using # notation, e.g., "Closes #123" or "Relates to #456" -->

Closes #
Relates to #

## Changes Made

<!-- List the specific changes you made -->

-
-
-

## Testing

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing locally (`make test`)
- [ ] Test coverage maintained or improved

### Manual Testing

<!-- Describe the manual testing you performed -->

**Steps to test**:
1.
2.
3.

**Expected behavior**:


**Actual behavior**:


## Checklist

### Code Quality

- [ ] Code follows the project's style guidelines
- [ ] Self-review performed
- [ ] Code is commented where necessary (especially complex logic)
- [ ] No new warnings introduced
- [ ] `cargo fmt` and `cargo clippy` passing
- [ ] TypeScript/ESLint checks passing (for web changes)

### Documentation

- [ ] Updated relevant documentation (README, docs/, inline comments)
- [ ] Added/updated API documentation (if applicable)
- [ ] Updated CHANGELOG.md (if applicable)

### Security

- [ ] No hardcoded secrets or sensitive data
- [ ] Input validation added where necessary
- [ ] No new security vulnerabilities introduced
- [ ] `cargo audit` passing (for Rust changes)
- [ ] `npm audit` passing (for web changes)

### Performance

- [ ] No significant performance regressions
- [ ] Benchmarks run (if performance-critical change)
- [ ] Memory usage considered

## Screenshots/Videos

<!-- If applicable, add screenshots or videos demonstrating the changes -->

## Deployment Notes

<!-- Any special considerations for deploying this change? -->

- [ ] Requires database migration
- [ ] Requires environment variable changes
- [ ] Breaking changes that require coordination
- [ ] None

## Reviewer Guidance

<!-- Help reviewers understand what to focus on -->

**Areas requiring extra attention**:


**Questions for reviewers**:


## Additional Context

<!-- Add any other context, background, or rationale here -->

---

## For Maintainers

### Merge Checklist

- [ ] PR title follows conventional commits format
- [ ] All CI checks passing
- [ ] At least one approval from maintainer
- [ ] No unresolved conversations
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing change)
