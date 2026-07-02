# Praxis Phase 2 Execution Plan

**Version**: 2.0
**Date**: 2025-11-15
**Phase**: Developer Experience & Community Readiness
**Duration**: 6-8 hours (this session)

---

## Overview

Phase 2 focuses on polishing the developer experience, adding tooling automation, and preparing for community contributions. This builds on Phase 1's foundation to create a truly world-class contributor experience.

---

## Objectives

1. **Eliminate friction** for new contributors
2. **Automate quality checks** to maintain high standards
3. **Demonstrate the vision** through working prototypes
4. **Attract contributors** with professional presentation

---

## Execution Plan (Prioritized)

### 🔥 CRITICAL (Do First)

#### 1. FAQ Documentation ⏱️ 30 min
**File**: `docs/faq.md`

**Questions to answer** (25+):
- What is Praxis?
- Why Holochain instead of blockchain?
- How does federated learning work?
- Is my data private?
- Can I use this for X?
- How do credentials work?
- What's the DAO for?
- How can I contribute?
- What's the roadmap?
- When will v1.0 launch?
- Is this production-ready?
- How do I run tests?
- Where are the zomes?
- Why doesn't the web client connect to Holochain yet?
- What's the difference between praxis-core and praxis-agg?
- How do I add a new aggregation method?
- How do I create a course?
- How do I issue credentials?
- What are the security guarantees?
- Can this scale to 1M users?
- What's the business model?
- Is there a paper/whitepaper?
- How does governance work?
- What if I find a security vulnerability?
- Who maintains this project?

**Impact**: Reduces support burden, answers common questions upfront

---

#### 2. VS Code Configuration ⏱️ 20 min
**Files**:
- `.vscode/settings.json` - Optimal settings for Rust + TS
- `.vscode/extensions.json` - Recommended extensions
- `.vscode/tasks.json` - Build/test tasks

**Extensions to recommend**:
- `rust-lang.rust-analyzer` - Rust language support
- `tamasfe.even-better-toml` - TOML syntax
- `vadimcn.vscode-lldb` - Rust debugging
- `dbaeumer.vscode-eslint` - ESLint
- `esbenp.prettier-vscode` - Prettier
- `EditorConfig.EditorConfig` - EditorConfig support

**Settings**:
- Format on save
- Clippy on save
- Auto-import organization
- Consistent indentation

**Impact**: Consistent development environment, faster onboarding

---

#### 3. Rust Toolchain Pinning ⏱️ 5 min
**File**: `rust-toolchain.toml`

Pin to stable Rust version for reproducible builds.

**Impact**: Eliminates "works on my machine" issues

---

### 🎯 HIGH PRIORITY (Do Next)

#### 4. Enhanced Web App ⏱️ 60 min
**Files**:
- `apps/web/src/services/mockHolochainClient.ts` - Mock client
- `apps/web/src/components/CourseCard.tsx` - Course component
- `apps/web/src/components/FLRoundCard.tsx` - FL round component
- `apps/web/src/components/CredentialCard.tsx` - Credential component
- `apps/web/src/data/mockData.ts` - Sample data from examples/
- `apps/web/src/App.tsx` - Enhanced with routing

**Features**:
- Course discovery page
- FL round participation flow
- Credential viewer
- Mock Holochain connection status

**Impact**: Demonstrates the vision, feels like a real app

---

#### 5. ROADMAP.md ⏱️ 20 min
**File**: `ROADMAP.md`

Public-facing roadmap with:
- Milestones (v0.1, v0.2, v0.3, v1.0)
- Feature timeline
- Community goals
- How to influence the roadmap

**Impact**: Transparency, shows momentum, attracts contributors

---

#### 6. README Enhancements ⏱️ 15 min
**Changes to `README.md`**:

Add shields.io badges:
```markdown
![Build Status](https://github.com/Luminous-Dynamics/mycelix-praxis/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Stars](https://img.shields.io/github/stars/Luminous-Dynamics/mycelix-praxis)
```

Add:
- Screenshots (when web app is enhanced)
- Video demo link (future)
- "Star us on GitHub" CTA
- Contributor count badge

**Impact**: Professional appearance, social proof

---

### 💪 MEDIUM PRIORITY (If Time Permits)

#### 7. Test Fixtures ⏱️ 30 min
**File**: `tests/fixtures/mod.rs`

Data generators:
```rust
pub fn mock_course() -> Course { ... }
pub fn mock_fl_round() -> FlRound { ... }
pub fn mock_credential() -> VerifiableCredential { ... }
pub fn random_gradient(dim: usize) -> Vec<f32> { ... }
```

**Impact**: Makes testing easier and more consistent

---

#### 8. Pre-commit Hooks ⏱️ 15 min
**File**: `.pre-commit-config.yaml`

Hooks:
- Rust format check
- Rust clippy
- Web eslint
- Web prettier
- Trailing whitespace removal
- Large file check

**Impact**: Prevents bad commits, maintains quality automatically

---

#### 9. Dependabot Configuration ⏱️ 10 min
**File**: `.github/dependabot.yml`

Auto-update:
- Cargo dependencies (weekly)
- npm dependencies (weekly)
- GitHub Actions (monthly)

**Impact**: Keeps dependencies current, reduces security risks

---

#### 10. Seeded GitHub Issues ⏱️ 30 min
**Create 10 initial issues**:

Labels: `good first issue`, `help wanted`, `enhancement`, etc.

Issues:
1. Add more aggregation methods to praxis-agg
2. Implement HDK entry definitions for learning_zome
3. Create FL round visualization
4. Add dark mode to web app
5. Write integration tests for FL round lifecycle
6. Implement DAO proposal creation
7. Add credential verification UI
8. Optimize trimmed mean performance
9. Add i18n support
10. Create Jupyter notebook for FL simulation

**Impact**: Gives contributors clear entry points

---

### 🌟 BONUS (Stretch Goals)

#### 11. Social Media Assets ⏱️ 20 min
**Files**:
- `.github/social-preview.png` - OG image for sharing
- `docs/assets/logo.svg` - Project logo (simple design)

**Impact**: Better social media presence

---

#### 12. Docker Compose ⏱️ 20 min
**File**: `docker-compose.yml`

Services:
- Holochain conductor (future)
- PostgreSQL (for analytics, future)
- Redis (for caching, future)

**Impact**: One-command local environment

---

#### 13. Stale Bot Configuration ⏱️ 5 min
**File**: `.github/stale.yml`

Auto-close inactive issues after 90 days.

**Impact**: Keeps issue tracker clean

---

#### 14. Code Coverage Integration ⏱️ 15 min
**File**: `.github/workflows/coverage.yml`

Upload coverage to codecov.io

**Impact**: Visibility into test coverage

---

## Success Metrics

### Developer Experience
- [ ] Time to first successful build: <5 min (from 10 min)
- [ ] VS Code users have consistent setup
- [ ] Pre-commit hooks prevent 90% of CI failures
- [ ] FAQ answers 80% of questions

### Repository Quality
- [ ] All badges green
- [ ] 10+ good first issues
- [ ] Dependencies auto-updated weekly
- [ ] Test coverage visible

### Community Readiness
- [ ] Roadmap publicly visible
- [ ] Professional README with badges
- [ ] Clear contribution paths
- [ ] Active issue tracker

---

## Execution Order (This Session)

**Wave 1** (Critical - 1 hour):
1. FAQ documentation ✅
2. VS Code configuration ✅
3. Rust toolchain pinning ✅
4. ROADMAP.md ✅

**Wave 2** (High Priority - 1.5 hours):
5. Enhanced web app ✅
6. README badges ✅
7. Test fixtures ✅

**Wave 3** (Medium Priority - 45 min):
8. Pre-commit hooks ✅
9. Dependabot ✅
10. Seeded issues ✅

**Wave 4** (Bonus - if time):
11. Docker Compose
12. Stale bot
13. Code coverage

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Too ambitious for one session | Prioritized list, focus on Wave 1-2 |
| Web app changes break build | Test after each change |
| Pre-commit hooks too strict | Make them warnings initially |
| Issues too complex for newcomers | Mix simple and complex issues |

---

## Post-Execution Checklist

- [ ] All files committed
- [ ] Pushed to remote
- [ ] CI passes (green badges)
- [ ] README looks professional
- [ ] Getting started guide still accurate
- [ ] Examples still work
- [ ] Tests still pass

---

**Let's execute! 🚀**

Next: Start with FAQ, then VS Code config, then web app enhancements.
