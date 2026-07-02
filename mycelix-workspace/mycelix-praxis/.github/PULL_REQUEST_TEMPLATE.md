# Pull Request

## Summary

**What does this PR do?**
<!-- Brief description of the changes -->

**Related issue(s):**
<!-- Fixes #123, Closes #456, Related to #789 -->

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Security fix

## Changes Made

**High-level summary:**
<!-- What files/components were changed and why? -->

**Key changes:**
- Change 1
- Change 2
- Change 3

## Testing

**How was this tested?**
<!-- Describe the tests you ran and their results -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Adversarial/security testing (if applicable)

**Test coverage:**
<!-- Link to coverage report or describe coverage -->

**Test plan for reviewers:**
<!-- How can reviewers verify this works? -->

1. Step 1
2. Step 2
3. Expected result

## Documentation

- [ ] Code comments added/updated
- [ ] API documentation updated (if applicable)
- [ ] Protocol documentation updated (`docs/protocol.md`)
- [ ] Threat model updated (`docs/threat-model.md`)
- [ ] Privacy model updated (`docs/privacy.md`)
- [ ] ADR created/updated (if architectural change)
- [ ] README updated (if user-facing change)
- [ ] CHANGELOG updated (if notable change)

## Security & Privacy

**Security impact:**
<!-- Does this change the threat model? Any new attack vectors? -->
<!-- If yes, explain and link to updated threat-model.md -->

- [ ] No security impact
- [ ] Security impact analyzed and documented

**Privacy impact:**
<!-- Does this involve collecting, storing, or sharing new data? -->
<!-- If yes, explain and link to updated privacy.md -->

- [ ] No privacy impact
- [ ] Privacy impact analyzed and documented

## Breaking Changes

**Does this introduce breaking changes?**
- [ ] No
- [ ] Yes (explain below)

**If yes, describe:**
<!-- What breaks? What's the migration path? -->

**Schema version bump:**
- [ ] Not applicable
- [ ] Version bumped in relevant `Cargo.toml` or schemas

## Checklist

**Before submitting:**
- [ ] Code builds without errors (`make build`)
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make fmt`)
- [ ] Linters pass (`make lint`)
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Branch is up-to-date with `dev` (or `main` if hotfix)

**Required for merge:**
- [ ] All CI checks passing
- [ ] At least 1 maintainer approval
- [ ] All review comments addressed
- [ ] No merge conflicts

## Screenshots (if applicable)

<!-- Before/After screenshots for UI changes -->

## Additional Notes

<!-- Anything else reviewers should know? -->
<!-- Deployment notes, feature flags, configuration changes, etc. -->

---

**Reviewers:** Please check:
- [ ] Code quality and style
- [ ] Test coverage and correctness
- [ ] Security and privacy implications
- [ ] Documentation completeness
- [ ] Breaking change handling
