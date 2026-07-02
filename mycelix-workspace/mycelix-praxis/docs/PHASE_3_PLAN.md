# Praxis Phase 3 Execution Plan

**Version**: 3.0
**Date**: 2025-11-15
**Phase**: Polish, Prototypes & Community Seeding
**Duration**: 6-8 hours (this session)

---

## Overview

Phase 3 focuses on creating working prototypes, adding test infrastructure, and preparing the first wave of contributor issues. This transforms the repo from "documentation-complete" to "demo-ready" and "contributor-ready."

---

## Strategic Objectives

1. **Demonstrate the vision** through working web UI
2. **Enable rigorous testing** with fixtures and generators
3. **Attract first contributors** with clear, approachable issues
4. **Polish the brand** with visual identity
5. **Measure performance** to establish baselines

---

## 🔥 WAVE 1: Test Infrastructure (High Priority)

### 1. Test Fixtures & Generators ⏱️ 45 min
**Files**:
- `tests/fixtures/mod.rs` - Root fixture module
- `tests/fixtures/courses.rs` - Course generators
- `tests/fixtures/fl_rounds.rs` - FL round generators
- `tests/fixtures/credentials.rs` - Credential generators
- `tests/fixtures/aggregation.rs` - Gradient/data generators

**Capabilities**:
```rust
// Generate realistic test data
let course = fixtures::course::random();
let fl_round = fixtures::fl_round::with_participants(37);
let credential = fixtures::credential::for_course(&course);
let gradients = fixtures::gradients::gaussian(100, 512);  // 100 participants, 512 dims
```

**Features**:
- Deterministic random (seeded for reproducibility)
- Configurable parameters
- Builder pattern for flexibility
- Realistic data distributions

**Impact**:
- Makes writing tests 10x faster
- Ensures consistent test data
- Enables property-based testing (future)

---

### 2. Benchmark Harness ⏱️ 30 min
**File**: `benches/aggregation_bench.rs`

**Benchmarks**:
- Trimmed mean vs median vs weighted mean
- Varying participant counts (10, 100, 1000)
- Varying gradient dimensions (128, 512, 2048)
- Clipping overhead

**Output**:
```
trimmed_mean/100_participants/512_dims   time: [15.2 ms 15.4 ms 15.6 ms]
median/100_participants/512_dims         time: [22.1 ms 22.3 ms 22.5 ms]
weighted_mean/100_participants/512_dims  time: [12.8 ms 13.0 ms 13.2 ms]
```

**Impact**:
- Establishes performance baselines
- Identifies optimization opportunities
- Tracks regressions over time

---

## 🚀 WAVE 2: Web App Prototype (Critical)

### 3. Mock Holochain Client ⏱️ 30 min
**File**: `apps/web/src/services/mockHolochainClient.ts`

**Simulates**:
- Conductor connection
- Zome function calls (returns mock data from examples/)
- Realistic latency (100-300ms delays)
- Error scenarios (10% failure rate)

**API**:
```typescript
const client = new MockHolochainClient();
const courses = await client.callZome('learning_zome', 'get_courses', {});
const round = await client.callZome('fl_zome', 'create_round', params);
```

**Impact**:
- Web app feels real
- Can demo without Holochain conductor
- Tests UI error handling

---

### 4. Course Discovery Page ⏱️ 45 min
**Files**:
- `apps/web/src/pages/CoursesPage.tsx` - Main page
- `apps/web/src/components/CourseCard.tsx` - Course card
- `apps/web/src/components/CourseFilters.tsx` - Search/filter
- `apps/web/src/data/mockCourses.ts` - Load from examples/

**Features**:
- Grid of course cards
- Search by title/description
- Filter by tags
- Sort by popularity/recent
- Click to view details

**Design** (Simple, Clean):
```
┌─────────────────────────────────────────┐
│  Praxis  [Search...]  [Filter ▼]       │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │Course│  │Course│  │Course│  ...    │
│  │Card 1│  │Card 2│  │Card 3│         │
│  └──────┘  └──────┘  └──────┘         │
│                                         │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │Course│  │Course│  │Course│  ...    │
│  │Card 4│  │Card 5│  │Card 6│         │
│  └──────┘  └──────┘  └──────┘         │
└─────────────────────────────────────────┘
```

**Impact**: Demonstrates core user flow

---

### 5. FL Round Participation Page ⏱️ 45 min
**Files**:
- `apps/web/src/pages/FLRoundsPage.tsx` - List rounds
- `apps/web/src/components/RoundCard.tsx` - Round card
- `apps/web/src/components/RoundTimeline.tsx` - Phase visualization
- `apps/web/src/data/mockRounds.ts` - Load from examples/

**Features**:
- List active/completed rounds
- Join button (simulated)
- Phase timeline (DISCOVER → RELEASE)
- Progress indicator
- Participant count

**Timeline Visualization**:
```
DISCOVER → JOIN → ASSIGN → UPDATE → AGGREGATE → RELEASE
  ✓        ✓       ✓        [●]        ○            ○
                          (Current)
```

**Impact**: Shows off FL feature, coolest part of the project

---

### 6. Credential Viewer ⏱️ 30 min
**Files**:
- `apps/web/src/pages/CredentialsPage.tsx` - List credentials
- `apps/web/src/components/CredentialCard.tsx` - Credential card
- `apps/web/src/components/CredentialVerifier.tsx` - Verify modal
- `apps/web/src/data/mockCredentials.ts` - Load from examples/

**Features**:
- Display user's credentials
- W3C VC badge design
- Click to verify (shows JSON structure)
- Share button (copy to clipboard)

**Credential Card Design**:
```
┌──────────────────────────────────────┐
│  🎓 EduAchievementCredential         │
│                                      │
│  Rust Fundamentals                   │
│  Completed: Nov 15, 2025             │
│  Score: 92.5 (A)                     │
│                                      │
│  Issued by: Mycelix Academy          │
│  [Verify] [Share]                    │
└──────────────────────────────────────┘
```

**Impact**: Shows W3C VC integration

---

### 7. Routing & Navigation ⏱️ 20 min
**Files**:
- `apps/web/src/App.tsx` (updated) - Add React Router
- `apps/web/src/components/NavBar.tsx` - Navigation

**Routes**:
- `/` - Home/Dashboard
- `/courses` - Course discovery
- `/rounds` - FL rounds
- `/credentials` - My credentials
- `/about` - About Praxis

**Impact**: Feels like a real app

---

## 🌟 WAVE 3: Community Seeding (High Priority)

### 8. Seed GitHub Issues ⏱️ 60 min
**Create 15 issues** (mix of difficulties):

**Good First Issues (5)**:
1. Add `geometric_mean` aggregation method
2. Improve README with screenshots
3. Add more example courses (Math, Art, etc.)
4. Fix typos in documentation
5. Add tests for `clip_l2_norm` edge cases

**Help Wanted (5)**:
6. Implement `learning_zome` HDK entry definitions
7. Add dark mode to web app
8. Create FL round visualization (chart.js)
9. Add i18n support (react-i18next)
10. Implement credential revocation UI

**Enhancement (5)**:
11. Optimize trimmed mean performance
12. Add asynchronous FL support
13. Implement DAO proposal creation UI
14. Add Jupyter notebook for FL simulation
15. Create mobile app (React Native) POC

**Labels for each**:
- `good first issue` or `help wanted` or `enhancement`
- Area: `rust`, `web`, `docs`, `zomes`, `protocol`
- Priority: `p1`, `p2`, `p3`

**Impact**: Clear entry points for contributors

---

### 9. CONTRIBUTORS.md ⏱️ 15 min
**File**: `CONTRIBUTORS.md`

**Sections**:
- How to get added
- Current contributors (start with maintainers)
- Contribution types (code, docs, design, testing)
- Hall of fame (future: top contributors)

**Auto-generation** (future): Use all-contributors bot

**Impact**: Recognizes contributions, encourages participation

---

## 🎨 WAVE 4: Visual Identity (Medium Priority)

### 10. Simple Logo Design ⏱️ 30 min
**File**: `docs/assets/logo.svg`

**Concept**: Simple, modern, represents:
- Education (book, graduation cap, brain)
- Decentralization (network, nodes)
- Privacy (lock, shield)

**Style**: Minimalist, two-color (primary + accent)

**Use SVG** for scalability

**Impact**: Brand recognition, professional appearance

---

### 11. Social Media Preview Image ⏱️ 20 min
**File**: `.github/social-preview.png` (1200x630)

**Content**:
- Praxis logo
- Tagline: "Privacy-Preserving Decentralized Education"
- Key features (3 bullet points)
- "Star us on GitHub"

**Impact**: Better social sharing (Twitter, LinkedIn, etc.)

---

## 🔧 WAVE 5: Developer Experience Polish

### 12. Enhanced Makefile ⏱️ 15 min
**File**: `Makefile` (updated)

**New commands**:
```makefile
make watch       # Auto-rebuild on file change
make bench       # Run benchmarks
make coverage    # Generate coverage report
make check       # Quick sanity check (fmt + clippy + test)
make docs        # Generate rustdoc + serve
make install     # Install dev tools (pre-commit, etc.)
```

**Impact**: One-command workflows

---

### 13. Installation Script ⏱️ 20 min
**File**: `scripts/install.sh`

**Does**:
1. Check prerequisites (Rust, Node, etc.)
2. Install if missing (rustup, nvm, etc.)
3. Set up pre-commit hooks
4. Install npm dependencies
5. Build everything
6. Run tests
7. Print success message

**One-liner install**:
```bash
curl -sSf https://raw.githubusercontent.com/.../install.sh | bash
```

**Impact**: Zero-friction onboarding

---

## 📊 Success Metrics

### Wave 1: Test Infrastructure
- [ ] 10+ test fixtures available
- [ ] Benchmarks run successfully
- [ ] Test writing time reduced 50%

### Wave 2: Web App
- [ ] 3 pages fully functional
- [ ] Mock client works seamlessly
- [ ] Can demo full user journey

### Wave 3: Community
- [ ] 15 issues created
- [ ] Mix of easy/medium/hard
- [ ] CONTRIBUTORS.md live

### Wave 4: Visual Identity
- [ ] Logo designed
- [ ] Social preview created
- [ ] README includes logo

### Wave 5: DX Polish
- [ ] Makefile enhanced
- [ ] Install script works
- [ ] One-command setup

---

## Execution Order (This Session)

**Priority 1** (Must Do - 3 hours):
1. Test fixtures ✅
2. Mock Holochain client ✅
3. Course discovery page ✅
4. FL round participation page ✅
5. Seed 10 GitHub issues ✅

**Priority 2** (Should Do - 2 hours):
6. Credential viewer ✅
7. CONTRIBUTORS.md ✅
8. Benchmark harness ✅
9. Enhanced Makefile ✅

**Priority 3** (Nice to Have - 1 hour):
10. Simple logo ✅
11. Social preview ✅

**Priority 4** (Stretch):
12. Install script
13. Additional polish

---

## Post-Execution Validation

### Functionality Checks
- [ ] All tests pass (`make test`)
- [ ] Web app runs (`make dev`)
- [ ] Benchmarks run (`make bench`)
- [ ] All pages accessible and functional

### Quality Checks
- [ ] No lint errors
- [ ] No TypeScript errors
- [ ] Examples load correctly
- [ ] Mock client returns realistic data

### Community Readiness
- [ ] Issues have clear descriptions
- [ ] Issues have appropriate labels
- [ ] CONTRIBUTORS.md is inviting
- [ ] README includes new features

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Web app changes break build | Test after each component |
| Mock data doesn't match schema | Validate against examples/ |
| Issues too vague | Use templates, clear acceptance criteria |
| Logo design takes too long | Keep it simple, text-based OK |

---

## Files to Create/Modify

**New Files** (~25):
- `tests/fixtures/*.rs` (5 files)
- `benches/aggregation_bench.rs`
- `apps/web/src/services/mockHolochainClient.ts`
- `apps/web/src/pages/*.tsx` (3 pages)
- `apps/web/src/components/*.tsx` (6 components)
- `apps/web/src/data/mock*.ts` (3 files)
- `CONTRIBUTORS.md`
- `docs/assets/logo.svg`
- `.github/social-preview.png`
- `scripts/install.sh`

**Modified Files**:
- `apps/web/src/App.tsx` (routing)
- `Makefile` (new commands)
- `README.md` (logo, screenshots)

---

## Expected Outcomes

After Phase 3, the repository will be:
1. **Demo-ready**: Working web app showcases vision
2. **Test-ready**: Fixtures make testing easy
3. **Contributor-ready**: 15 issues waiting for contributors
4. **Performance-tracked**: Benchmarks establish baselines
5. **Brand-recognized**: Logo and social preview

---

**Let's execute! 🚀**

Next: Start with test fixtures, then web app enhancements, then issue seeding.
