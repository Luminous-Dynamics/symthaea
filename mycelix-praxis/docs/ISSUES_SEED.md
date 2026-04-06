# Seed GitHub Issues for Praxis

This document contains template issues to seed the GitHub repository with contributor-friendly tasks. These should be created manually via the GitHub UI or `gh` CLI.

---

## Good First Issues (5)

### Issue #1: Add geometric_mean aggregation method
**Labels**: `good first issue`, `rust`, `enhancement`, `p2`

**Title**: Add geometric_mean aggregation method to praxis-agg

**Description**:
Add a geometric mean aggregation method to complement the existing trimmed_mean, median, and weighted_mean methods.

**Task**:
Implement `geometric_mean` function in `crates/praxis-agg/src/methods.rs` that:
- Takes a vector of gradient updates
- Computes the geometric mean per dimension
- Handles edge cases (zeros, negative values)
- Includes comprehensive tests
- Adds benchmark to `benches/aggregation_bench.rs`

**Acceptance Criteria**:
- [ ] Function implemented with correct geometric mean formula
- [ ] Edge cases handled (zeros → skip or add small epsilon)
- [ ] Unit tests added with at least 90% coverage
- [ ] Benchmark added
- [ ] Documentation updated
- [ ] All tests pass

**Resources**:
- See existing methods in `crates/praxis-agg/src/methods.rs`
- Geometric mean formula: nth root of product of n numbers
- Test fixtures available in `tests/fixtures/aggregation.rs`

**Estimated Effort**: 2-3 hours

---

### Issue #2: Improve README with screenshots
**Labels**: `good first issue`, `docs`, `enhancement`, `p2`

**Title**: Add screenshots of web UI to README

**Description**:
Our web app now has working UI pages (courses, FL rounds, credentials), but the README doesn't show them! Let's add screenshots to make the project more visually appealing.

**Task**:
1. Run `make dev` and start the web app
2. Take screenshots of:
   - Home page
   - Course discovery page
   - FL rounds page
   - Credentials page
3. Save screenshots to `docs/assets/screenshots/`
4. Update README.md with embedded screenshots
5. Add a "Screenshots" section below "Features"

**Acceptance Criteria**:
- [ ] High-quality screenshots (1920x1080 or similar)
- [ ] At least 4 screenshots
- [ ] Screenshots show key features clearly
- [ ] Markdown properly renders images
- [ ] File sizes reasonable (<500KB each)

**Resources**:
- Web app at `apps/web/`
- README at root: `README.md`
- Example image syntax: `![Alt text](docs/assets/screenshots/home.png)`

**Estimated Effort**: 1 hour

---

### Issue #3: Add more example courses
**Labels**: `good first issue`, `docs`, `data`, `p3`

**Title**: Create example courses for Math, Art, and History

**Description**:
We have example courses for Rust and Spanish, but need more variety! Add 3 new courses in different subject areas.

**Task**:
Create JSON files in `examples/courses/` for:
1. `mathematics-algebra.json` - Algebra fundamentals
2. `digital-art-basics.json` - Digital art and design
3. `world-history-101.json` - Introduction to world history

Each should include:
- Detailed syllabus with 4-6 modules
- Learning outcomes per module
- Realistic tags and difficulty level
- Prerequisites (if applicable)

**Acceptance Criteria**:
- [ ] 3 new JSON files created
- [ ] Each matches the schema of existing courses
- [ ] Realistic and complete module descriptions
- [ ] Valid JSON (run through validator)
- [ ] Added to `mockCourses.ts` data (optional)

**Resources**:
- Existing examples: `examples/courses/rust-fundamentals.json`
- Schema guide: `docs/protocol.md`

**Estimated Effort**: 1-2 hours

---

### Issue #4: Fix typos in documentation
**Labels**: `good first issue`, `docs`, `p3`

**Title**: Proofread and fix typos in documentation files

**Description**:
Help improve documentation quality by finding and fixing typos, grammar issues, and formatting inconsistencies.

**Task**:
1. Read through key documentation files:
   - `README.md`
   - `CONTRIBUTING.md`
   - `docs/protocol.md`
   - `docs/architecture.md`
   - `docs/faq.md`
2. Fix typos, grammar issues, formatting problems
3. Ensure consistent capitalization (e.g., "Federated Learning" vs "federated learning")
4. Check links are working
5. Improve readability where needed

**Acceptance Criteria**:
- [ ] At least 5 fixes made
- [ ] No new typos introduced
- [ ] Links verified
- [ ] Consistent terminology

**Resources**:
- Docs in `docs/` directory
- Grammarly or similar tool helpful
- US English preferred

**Estimated Effort**: 1-2 hours

---

### Issue #5: Add tests for clip_l2_norm edge cases
**Labels**: `good first issue`, `rust`, `testing`, `p2`

**Title**: Improve test coverage for clip_l2_norm function

**Description**:
The `clip_l2_norm` function in `praxis-agg` needs better edge case testing.

**Task**:
Add tests in `crates/praxis-agg/src/methods.rs` for:
- Zero vector (norm = 0)
- Very small vectors (norm < 0.001)
- Very large vectors (norm > 1000)
- Negative values
- NaN/Infinity handling
- Empty vector

**Acceptance Criteria**:
- [ ] At least 6 new test cases
- [ ] Tests cover edge cases listed above
- [ ] Tests are deterministic and reproducible
- [ ] All tests pass
- [ ] Code coverage increased

**Resources**:
- Function: `crates/praxis-agg/src/methods.rs:clip_l2_norm`
- Existing tests in same file
- Test fixtures: `tests/fixtures/aggregation.rs`

**Estimated Effort**: 1-2 hours

---

## Help Wanted Issues (5)

### Issue #6: Implement learning_zome HDK entry definitions
**Labels**: `help wanted`, `rust`, `holochain`, `zomes`, `p1`

**Title**: Add HDK entry definitions and zome functions to learning_zome

**Description**:
The `learning_zome` currently has library-only types. Let's implement actual Holochain zome functions using HDK!

**Task**:
In `zomes/learning_zome/`:
1. Add HDK dependencies to Cargo.toml (uncomment existing lines)
2. Define integrity zome with entries:
   - `Course` (with validation)
   - `LearnerProgress`
   - `LearningActivity`
3. Define coordinator zome with functions:
   - `create_course`
   - `get_course`
   - `enroll`
   - `update_progress`
   - `get_learner_progress`
4. Add links: creator → courses, learner → progress
5. Write integration tests

**Acceptance Criteria**:
- [ ] HDK v0.4+ used correctly
- [ ] Integrity/coordinator split implemented
- [ ] Validation functions for all entries
- [ ] All zome functions working
- [ ] Integration tests pass
- [ ] Documentation updated

**Resources**:
- HDK docs: https://docs.rs/hdk/latest/hdk/
- Existing type definitions in `src/lib.rs`
- Holochain gym exercises

**Estimated Effort**: 6-8 hours

---

### Issue #7: Add dark mode to web app
**Labels**: `help wanted`, `web`, `ui`, `enhancement`, `p2`

**Title**: Implement dark mode toggle for web UI

**Description**:
Add a dark mode theme to the web app with persistent storage.

**Task**:
1. Create theme context (`ThemeContext.tsx`)
2. Define dark/light color schemes
3. Add toggle button to NavBar
4. Store preference in localStorage
5. Apply theme to all components:
   - HomePage
   - CoursesPage
   - FLRoundsPage
   - CredentialsPage
   - All cards and modals

**Acceptance Criteria**:
- [ ] Toggle button in NavBar
- [ ] Dark mode looks good (readable, accessible)
- [ ] Preference persists across sessions
- [ ] No color contrast issues (WCAG AA)
- [ ] Smooth theme transitions

**Resources**:
- Components in `apps/web/src/`
- React Context API
- CSS-in-JS (inline styles used currently)
- Color schemes: Tailwind colors recommended

**Estimated Effort**: 4-6 hours

---

### Issue #8: Create FL round visualization chart
**Labels**: `help wanted`, `web`, `visualization`, `enhancement`, `p2`

**Title**: Add Chart.js visualization for FL round progress

**Description**:
Visualize FL round aggregation metrics (loss over rounds, participant contributions) using Chart.js.

**Task**:
1. Install Chart.js: `npm install chart.js react-chartjs-2`
2. Create `RoundMetricsChart.tsx` component
3. Display charts in FL round detail modal:
   - Validation loss over time
   - Participant count per round
   - Aggregation quality metrics
4. Use mock data initially
5. Make charts responsive

**Acceptance Criteria**:
- [ ] Chart.js integrated
- [ ] At least 2 chart types (line, bar)
- [ ] Charts render correctly
- [ ] Responsive design
- [ ] Legend and labels clear
- [ ] Mock data realistic

**Resources**:
- Chart.js docs: https://www.chartjs.org/
- FL round data: `apps/web/src/data/mockRounds.ts`
- Example provenance data in `examples/fl-rounds/`

**Estimated Effort**: 3-4 hours

---

### Issue #9: Add i18n support with react-i18next
**Labels**: `help wanted`, `web`, `i18n`, `enhancement`, `p3`

**Title**: Internationalization support for web UI

**Description**:
Add multi-language support starting with English and Spanish.

**Task**:
1. Install `react-i18next`: `npm install react-i18next i18next`
2. Set up i18n configuration
3. Create translation files:
   - `locales/en/translation.json`
   - `locales/es/translation.json`
4. Replace hard-coded strings with translation keys
5. Add language selector to NavBar
6. Translate at least:
   - HomePage
   - Navigation
   - Common buttons/labels

**Acceptance Criteria**:
- [ ] i18next configured correctly
- [ ] English and Spanish translations
- [ ] Language selector works
- [ ] Preference persists
- [ ] At least 50 strings translated
- [ ] Fallback to English if missing

**Resources**:
- react-i18next docs: https://react.i18next.com/
- Components in `apps/web/src/`
- Translation guide: https://www.i18next.com/

**Estimated Effort**: 4-5 hours

---

### Issue #10: Implement credential revocation UI
**Labels**: `help wanted`, `web`, `credentials`, `enhancement`, `p2`

**Title**: Add revocation status display for credentials

**Description**:
Show revocation status on credentials and allow checking against revocation lists.

**Task**:
1. Add revocation status to `VerifiableCredential` type
2. Create revocation checking function
3. Display status badge on CredentialCard:
   - Green checkmark: Valid
   - Yellow warning: Expiring soon
   - Red X: Revoked
4. Add revocation check to verify modal
5. Mock revocation list for testing

**Acceptance Criteria**:
- [ ] Status badge shows correctly
- [ ] Revoked credentials clearly marked
- [ ] Verification checks revocation
- [ ] Expiration warnings work
- [ ] Mock data realistic
- [ ] UI polished and accessible

**Resources**:
- W3C VC spec on revocation: https://www.w3.org/TR/vc-data-model/
- Credential types: `apps/web/src/data/mockCredentials.ts`
- Credential components: `apps/web/src/components/CredentialCard.tsx`

**Estimated Effort**: 3-4 hours

---

## Enhancement Issues (5)

### Issue #11: Optimize trimmed mean performance
**Labels**: `enhancement`, `rust`, `performance`, `p2`

**Title**: Improve trimmed_mean performance with parallelization

**Description**:
The `trimmed_mean` function processes gradients sequentially. For large participant counts, this could be parallelized for better performance.

**Task**:
1. Benchmark current performance: `cargo bench`
2. Implement parallel version using rayon
3. Add feature flag: `parallel-aggregation`
4. Compare performance (sequential vs parallel)
5. Update documentation with benchmarks
6. Ensure results are deterministic

**Acceptance Criteria**:
- [ ] Parallel implementation using rayon
- [ ] Performance improvement >30% for 1000+ participants
- [ ] Results identical to sequential version
- [ ] Feature flag works
- [ ] Benchmarks updated
- [ ] No regressions

**Resources**:
- Current implementation: `crates/praxis-agg/src/methods.rs`
- Rayon docs: https://docs.rs/rayon/latest/rayon/
- Benchmarks: `benches/aggregation_bench.rs`

**Estimated Effort**: 4-6 hours

---

### Issue #12: Add asynchronous FL support
**Labels**: `enhancement`, `rust`, `protocol`, `research`, `p3`

**Title**: Design and implement asynchronous federated learning rounds

**Description**:
Current FL rounds are synchronous (all participants submit in same round). Add support for asynchronous updates where participants can contribute at different times.

**Task**:
1. Research asynchronous FL algorithms (FedBuff, FedAsync)
2. Design protocol changes needed
3. Update FlRound state machine
4. Implement async aggregation buffer
5. Add staleness tracking
6. Write RFC document

**Acceptance Criteria**:
- [ ] RFC document in `docs/rfcs/`
- [ ] Prototype implementation
- [ ] Handles varying participant availability
- [ ] Staleness weighting implemented
- [ ] Tests demonstrate async behavior
- [ ] Performance comparable to sync

**Resources**:
- FedBuff paper: https://arxiv.org/abs/2106.06639
- Current protocol: `docs/protocol.md`
- FL zome: `zomes/fl_zome/src/lib.rs`

**Estimated Effort**: 12-16 hours (research + implementation)

---

### Issue #13: Implement DAO proposal creation UI
**Labels**: `enhancement`, `web`, `dao`, `governance`, `p2`

**Title**: Add proposal creation and voting UI

**Description**:
Create UI for DAO governance: create proposals, vote, view results.

**Task**:
1. Create `DAOPage.tsx`
2. Create `ProposalCard.tsx` component
3. Create `CreateProposalModal.tsx`
4. Implement:
   - List proposals
   - Filter by status (active, passed, failed)
   - Create new proposal form
   - Vote on proposals
   - Show vote tally
5. Use mock DAO zome client
6. Add route to App.tsx

**Acceptance Criteria**:
- [ ] DAO page fully functional
- [ ] Can create proposals
- [ ] Can vote on proposals
- [ ] Vote tallies display correctly
- [ ] Fast/normal/slow path indicators
- [ ] Responsive design
- [ ] Accessible UI

**Resources**:
- DAO types: `zomes/dao_zome/src/lib.rs`
- Mock client: `apps/web/src/services/mockHolochainClient.ts`
- Existing pages as reference

**Estimated Effort**: 6-8 hours

---

### Issue #14: Add Jupyter notebook for FL simulation
**Labels**: `enhancement`, `python`, `simulation`, `research`, `p3`

**Title**: Create Jupyter notebook demonstrating FL simulation

**Description**:
Educational notebook showing how federated learning works with Praxis's protocol.

**Task**:
Create `notebooks/fl_simulation.ipynb` that:
1. Simulates distributed learners
2. Implements gradient clipping
3. Shows different aggregation methods
4. Visualizes convergence
5. Demonstrates privacy attacks (model inversion)
6. Shows defense effectiveness

**Acceptance Criteria**:
- [ ] Notebook runs without errors
- [ ] Clear explanations and markdown
- [ ] 5+ visualizations
- [ ] Demonstrates key concepts
- [ ] Requirements.txt included
- [ ] README in notebooks/ folder

**Resources**:
- Aggregation methods: `crates/praxis-agg/src/methods.rs`
- Protocol: `docs/protocol.md`
- Python FL libraries: PySyft, Flower, TensorFlow Federated

**Estimated Effort**: 6-8 hours

---

### Issue #15: Create mobile app POC with React Native
**Labels**: `enhancement`, `mobile`, `react-native`, `p3`

**Title**: Proof-of-concept mobile app for iOS and Android

**Description**:
Create a basic React Native app that connects to Praxis (mock mode initially).

**Task**:
1. Set up React Native project
2. Implement key screens:
   - Login/onboarding
   - Course browse
   - Credential wallet
   - FL round notifications
3. Reuse web components where possible
4. Mock Holochain client integration
5. Test on iOS simulator and Android emulator

**Acceptance Criteria**:
- [ ] React Native project set up
- [ ] 4+ screens implemented
- [ ] Navigation working
- [ ] Runs on iOS and Android
- [ ] Mock data displays correctly
- [ ] Basic README in `apps/mobile/`

**Resources**:
- React Native docs: https://reactnative.dev/
- Existing web components: `apps/web/src/`
- React Navigation for routing

**Estimated Effort**: 10-12 hours

---

## Issue Creation Instructions

To create these issues on GitHub:

**Via GitHub UI**:
1. Go to https://github.com/Luminous-Dynamics/mycelix-praxis/issues
2. Click "New Issue"
3. Copy/paste title and description
4. Add appropriate labels
5. Add to "Phase 3" milestone (create if needed)
6. Submit

**Via GitHub CLI**:
```bash
# Example for issue #1
gh issue create \
  --title "Add geometric_mean aggregation method to praxis-agg" \
  --body "$(cat issue-templates/001-geometric-mean.md)" \
  --label "good first issue,rust,enhancement,p2" \
  --milestone "Phase 3"
```

**Labels to create**:
- `good first issue` (green)
- `help wanted` (green)
- `enhancement` (blue)
- `rust` (orange)
- `web` (orange)
- `docs` (purple)
- `zomes` (orange)
- `holochain` (orange)
- `p1`, `p2`, `p3` (priority levels)

---

## Notes

- **Assign yourself**: When you start working on an issue, assign it to yourself
- **Ask questions**: Use issue comments to ask for clarification
- **Link PRs**: Reference the issue number in your PR description
- **Celebrate**: We'll celebrate first-time contributors! 🎉

---

*Document created: 2025-11-15*
*Issues should be created after Phase 3 is merged*
