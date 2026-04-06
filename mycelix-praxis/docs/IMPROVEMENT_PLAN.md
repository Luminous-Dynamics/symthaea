# Mycelix Praxis - Improvement Plan

**Version**: 1.0
**Date**: 2025-11-15
**Status**: In Progress

---

## Executive Summary

This document outlines a comprehensive improvement plan to evolve the mycelix-praxis repository from initial scaffold to implementation-ready state. The plan is organized into 5 phases over 4-6 weeks.

---

## Phase 1: Build System & Developer Experience (Week 1) 🔧

**Goal**: Ensure clean builds, passing tests, and smooth developer onboarding

### Priority: CRITICAL

#### 1.1 Fix Rust Compilation
- [x] Remove HDK dependencies from zomes (move to separate branch when ready)
- [ ] Add library-only implementations for v0.1
- [ ] Ensure `cargo build --workspace` succeeds
- [ ] Ensure `cargo test --workspace` passes
- [ ] Add rustdoc comments to all public APIs

**Rationale**: CI must be green for contributors to work confidently

#### 1.2 Improve Web Build
- [ ] Add mock Holochain client (stub implementation)
- [ ] Add sample data for UI components
- [ ] Ensure `npm run build` succeeds
- [ ] Add basic component tests

#### 1.3 Documentation Enhancements
- [ ] Add `docs/getting-started.md` - Step-by-step tutorial
- [ ] Add `docs/architecture.md` - Visual diagrams (ASCII/Mermaid)
- [ ] Add `docs/faq.md` - Common questions
- [ ] Add `examples/` directory with sample data

#### 1.4 Developer Tooling
- [ ] Add `CHANGELOG.md` (semantic versioning)
- [ ] Add `.vscode/settings.json` - Recommended VS Code settings
- [ ] Add `.vscode/extensions.json` - Recommended extensions
- [ ] Add `rust-toolchain.toml` - Pin Rust version
- [ ] Improve `Makefile` with more commands (check, watch, coverage)

**Success Criteria**:
- ✅ Green CI on all checks
- ✅ New contributor can build in <10 minutes
- ✅ All tests pass
- ✅ Documentation covers 80% of common questions

---

## Phase 2: Example Data & Fixtures (Week 2) 📊

**Goal**: Provide realistic examples and test fixtures

### Priority: HIGH

#### 2.1 Example Credentials
- [ ] `examples/credentials/` - Sample VCs (valid + invalid)
- [ ] JSON files with full VC structure
- [ ] Verification examples

#### 2.2 Example FL Rounds
- [ ] `examples/fl-rounds/` - Complete round lifecycle
- [ ] Mock gradient data
- [ ] Aggregation examples (trimmed mean, median)

#### 2.3 Example Courses
- [ ] `examples/courses/` - Sample course structures
- [ ] Language learning course (MVP use case)
- [ ] Skills marketplace course

#### 2.4 Test Fixtures
- [ ] `tests/fixtures/` - Reusable test data
- [ ] Data generators (random courses, rounds, credentials)
- [ ] Assertion helpers

#### 2.5 Example Notebooks (optional)
- [ ] Jupyter notebook: FL simulation
- [ ] Jupyter notebook: Aggregation comparison
- [ ] Visualizations of attack scenarios

**Success Criteria**:
- ✅ 10+ example files covering all major features
- ✅ Fixtures used in ≥50% of tests
- ✅ README links to examples

---

## Phase 3: Enhanced Documentation (Week 3) 📚

**Goal**: Comprehensive, accessible documentation for all audiences

### Priority: MEDIUM-HIGH

#### 3.1 Architecture Documentation
- [ ] `docs/architecture.md` - System diagrams
  - Component diagram (zomes, crates, web)
  - Data flow diagram (FL round lifecycle)
  - Deployment architecture
- [ ] `docs/architecture/` - Detailed component docs
  - Per-zome architecture
  - Per-crate architecture
- [ ] Mermaid diagrams (render in GitHub)

#### 3.2 Developer Guides
- [ ] `docs/dev/setup.md` - Development environment setup
- [ ] `docs/dev/testing.md` - Testing guide
- [ ] `docs/dev/debugging.md` - Debugging tips
- [ ] `docs/dev/contributing-code.md` - Code contribution workflow
- [ ] `docs/dev/zome-development.md` - How to write zomes

#### 3.3 User Guides (future)
- [ ] `docs/user/learner-guide.md` - For learners
- [ ] `docs/user/creator-guide.md` - For course creators
- [ ] `docs/user/coordinator-guide.md` - For FL coordinators

#### 3.4 API Documentation
- [ ] `docs/api/` - Generated rustdoc
- [ ] `docs/api/rest.md` - Web API endpoints (future)
- [ ] `docs/api/websocket.md` - Real-time API (future)

#### 3.5 FAQ & Troubleshooting
- [ ] `docs/faq.md` - Common questions
- [ ] `docs/troubleshooting.md` - Common issues + fixes

**Success Criteria**:
- ✅ Architecture diagrams for all major components
- ✅ Developer guides cover setup → first PR
- ✅ FAQ answers ≥20 questions
- ✅ Rustdoc coverage ≥80%

---

## Phase 4: Tooling & Automation (Week 4) 🤖

**Goal**: Automate repetitive tasks, improve DX

### Priority: MEDIUM

#### 4.1 Development Scripts
- [ ] `scripts/test-all.sh` - Run all tests (Rust + Web + integration)
- [ ] `scripts/lint-all.sh` - Run all linters
- [ ] `scripts/format-all.sh` - Format all code
- [ ] `scripts/check-licenses.sh` - Verify license headers
- [ ] `scripts/generate-changelog.sh` - Auto-generate from commits

#### 4.2 GitHub Actions Enhancements
- [ ] Add codecov.io integration (test coverage)
- [ ] Add dependabot.yml (auto-update dependencies)
- [ ] Add stale.yml (close inactive issues)
- [ ] Add auto-labeler (label PRs by file paths)

#### 4.3 Local Development
- [ ] Add `docker-compose.yml` (optional: PostgreSQL, Redis, etc.)
- [ ] Add `.env.example` - Environment variable template
- [ ] Add `justfile` alternative to Makefile (optional)

#### 4.4 Pre-commit Hooks
- [ ] `pre-commit-config.yaml` - Auto-format, lint on commit
- [ ] Rust fmt + clippy
- [ ] Web eslint + prettier
- [ ] Markdown linting

**Success Criteria**:
- ✅ 90% of manual checks automated
- ✅ Pre-commit hooks prevent bad commits
- ✅ CI runs <5 minutes
- ✅ Dependabot keeps dependencies current

---

## Phase 5: Community & Project Management (Week 5-6) 🌍

**Goal**: Prepare for public launch and community contributions

### Priority: MEDIUM-LOW

#### 5.1 Issue Seeding
- [ ] Create 20+ initial issues from roadmap
- [ ] Label issues (good first issue, help wanted, etc.)
- [ ] Assign to milestones (v0.1, v0.2, v1.0)
- [ ] Add issue templates for each zome/crate

#### 5.2 Project Boards
- [ ] GitHub Project: "Praxis v0.1 MVP"
- [ ] Columns: Backlog, Ready, In Progress, Review, Done
- [ ] Automation: auto-move on PR/issue state changes

#### 5.3 Community Docs
- [ ] `CONTRIBUTORS.md` - List of contributors (auto-generated)
- [ ] `docs/community/` - Community resources
  - Code of Conduct (already exists)
  - Communication channels (Discord, Discussions, etc.)
  - Meeting notes template
- [ ] `ROADMAP.md` - Public roadmap (link from README)

#### 5.4 Marketing & Communication
- [ ] Blog post: "Introducing Praxis" (draft)
- [ ] Twitter announcement (draft)
- [ ] Holochain forum post (draft)
- [ ] DevForum post (future)

#### 5.5 Onboarding
- [ ] Contributor onboarding checklist
- [ ] Maintainer playbook
- [ ] Release process documentation

**Success Criteria**:
- ✅ 20+ issues ready for contributors
- ✅ Project board active and up-to-date
- ✅ Public roadmap visible
- ✅ Onboarding docs complete

---

## Phase 6: Implementation Readiness (Ongoing) 🚀

**Goal**: Begin actual feature implementation

### Priority: VARIES

#### 6.1 Holochain DNA
- [ ] Add `dna/` directory
- [ ] Create DNA manifest (`dna.yaml`)
- [ ] Configure zomes in DNA
- [ ] Add integrity + coordinator zome split

#### 6.2 Implement FL Zome (v0.1 MVP)
- [ ] Entry definitions (FlRound, FlUpdate)
- [ ] Validation functions
- [ ] Zome functions (create_round, join_round, submit_update)
- [ ] Links (round → updates, agent → updates)
- [ ] Integration tests

#### 6.3 Implement Learning Zome (v0.1 MVP)
- [ ] Entry definitions (Course, LearnerProgress)
- [ ] Validation functions
- [ ] Zome functions (create_course, enroll, update_progress)
- [ ] Links (creator → courses, learner → progress)
- [ ] Integration tests

#### 6.4 Implement Credential Zome (v0.1 MVP)
- [ ] Entry definitions (VerifiableCredential)
- [ ] Validation functions (W3C VC spec compliance)
- [ ] Zome functions (issue_credential, verify_credential)
- [ ] Signature verification (Ed25519)
- [ ] Integration tests

#### 6.5 Web Client Integration
- [ ] Holochain client setup
- [ ] Connect to conductor
- [ ] Call zome functions
- [ ] Display real data (not mocks)
- [ ] Offline support (PWA)

**Success Criteria**:
- ✅ DNA compiles and runs in conductor
- ✅ All zome functions callable from web client
- ✅ End-to-end FL round completes successfully (local testing)
- ✅ Credentials issuable and verifiable

---

## Quick Wins (Can be done anytime) ⚡

Low-effort, high-impact improvements:

- [ ] Add shields.io badges to README (build status, coverage, license)
- [ ] Add GitHub Sponsors / OpenCollective
- [ ] Add "Star us on GitHub" CTA
- [ ] Add social media preview image (og:image)
- [ ] Add sitemap for docs (if hosted as static site)
- [ ] Add search functionality to docs
- [ ] Add dark mode to web app
- [ ] Add keyboard shortcuts to web app
- [ ] Add accessibility audit (WCAG 2.1 AA)
- [ ] Add internationalization (i18n) framework

---

## Metrics & Success Tracking

### Developer Experience Metrics
- Time to first successful build: **Target <10 min**
- Time to first PR: **Target <2 hours**
- CI run time: **Target <5 min**
- Test coverage: **Target >80%**

### Documentation Metrics
- Pages in docs/: **Target >20**
- Rustdoc coverage: **Target >80%**
- External links (blog posts, tutorials): **Target >5**

### Community Metrics
- GitHub stars: **Target >50 (month 1)**
- Contributors: **Target >5 (month 1)**
- Issues closed: **Target >10 (month 1)**
- PRs merged: **Target >5 (month 1)**

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Holochain version breaking changes | HIGH | MEDIUM | Pin Holochain version, monitor releases |
| FL security vulnerabilities | HIGH | LOW | External audit before v1.0 |
| Poor performance (large rounds) | MEDIUM | MEDIUM | Performance testing, profiling |
| W3C VC spec changes | LOW | LOW | Monitor W3C working group |

### Community Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Low contributor interest | MEDIUM | MEDIUM | Active outreach, good first issues |
| Toxic community members | MEDIUM | LOW | Enforce Code of Conduct strictly |
| Maintainer burnout | HIGH | MEDIUM | Distribute responsibilities, onboard co-maintainers |

---

## Resource Allocation

### Time Budget (per phase)

- **Phase 1** (Build System): 20 hours
- **Phase 2** (Examples): 15 hours
- **Phase 3** (Docs): 25 hours
- **Phase 4** (Tooling): 15 hours
- **Phase 5** (Community): 10 hours
- **Phase 6** (Implementation): 100+ hours (ongoing)

**Total**: ~185 hours for Phases 1-5

### Prioritization

**This session** (immediate execution):
1. Fix Rust compilation ✅
2. Add example data ✅
3. Add architecture diagrams ✅
4. Add CHANGELOG.md ✅
5. Add getting-started guide ✅

**Next session**:
- Remaining Phase 1 tasks
- Phase 2 (examples)
- Phase 4 (tooling)

**Future sessions**:
- Phase 3 (deep docs)
- Phase 5 (community)
- Phase 6 (implementation)

---

## Appendix: Reference Materials

### Inspiration (other well-structured repos)
- [tokio-rs/tokio](https://github.com/tokio-rs/tokio) - Excellent Rust DX
- [holochain/holochain](https://github.com/holochain/holochain) - Reference for DNA structure
- [facebook/react](https://github.com/facebook/react) - Community management
- [tensorflow/federated](https://github.com/tensorflow/federated) - FL reference

### Tools
- [cargo-make](https://github.com/sagiegurari/cargo-make) - Advanced Makefile alternative
- [mdbook](https://rust-lang.github.io/mdBook/) - Generate book from Markdown
- [mermaid-js](https://mermaid.js.org/) - Diagram as code

---

**End of Plan**

Next: Execute high-priority items from Phase 1 & 2 immediately.
