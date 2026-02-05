# Documentation Reorganization Plan

**Created**: January 12, 2026
**Current State**: 75 root docs + 16 subdirs in docs/ + new sympoietic/ folder
**Goal**: Clear structure where every doc has a home

---

## Current Inventory

### Root Level (75 files)
Too many files at root. Most should move to appropriate subdirectories.

### Existing docs/ Subdirectories (16)
```
docs/
├── api/           # API documentation
├── architecture/  # Architecture docs
├── archive/       # Old/deprecated docs
├── developer/     # Developer guides
├── improvements/  # Improvement tracking
├── integration/   # Integration docs
├── milestones/    # Milestone completions
├── planning/      # Planning docs
├── research/      # Research docs
├── sessions/      # Session summaries
├── status/        # Status tracking
├── sympoietic/    # NEW: Sympoietic docs (organized)
├── theory/        # Theoretical docs
├── tutorials/     # User tutorials
├── user/          # User documentation
├── versions/      # Version docs
└── weekly/        # Weekly progress
```

---

## Proposed Reorganization

### Keep at Root (Essential Project Files)
```
README.md              # Project overview
CLAUDE.md              # Claude context
CHANGELOG.md           # Version history
CONTRIBUTING.md        # Contribution guide
START_HERE.md          # Quick orientation
```

### Move to docs/architecture/
```
ARCHITECTURE_DEEP_DIVE.md
REVOLUTIONARY_ARCHITECTURE.md
UNIFIED_ENHANCEMENT_ARCHITECTURE.md
BRAIN_AND_MIND_MODELS_REVIEW.md
COMPONENT_INVENTORY.md
GENERALIZATION_REFACTORING_PLAN.md
COGNITIVE_INTEGRATION_ANALYSIS.md
```

### Move to docs/research/ (Φ Research)
```
DIMENSIONAL_SWEEP_RESULTS.md
EXTENDED_SWEEP_8D_12D_COMPLETE.md
EXTENDED_SWEEP_PRELIMINARY_RESULTS.md
PHI_IMPLEMENTATION_AUDIT.md
NORMALIZED_LAPLACIAN_FIX_COMPLETE.md
VALIDATION_EXPERIMENTS_PLAN.md
LITERATURE_REVIEW_PHI_BOUNDS.md
APPENDIX_P_CONSCIOUSNESS_RIGHTS.md
```

### Move to docs/paper/ (NEW - Academic Submission)
```
COMPLETE_MANUSCRIPT_FOR_PDF.md
MASTER_MANUSCRIPT.md
MANUSCRIPT_README.md
MANUSCRIPT_REVISIONS_FOR_VALIDATION.md
PAPER_CONCLUSIONS_SECTION.md
PAPER_DISCUSSION_SECTION.md
PAPER_METHODS_SECTION.md
PAPER_REFERENCES.md
PAPER_RESULTS_SECTION.md
PAPER_SUPPLEMENTARY_MATERIALS.md
COVER_LETTER.md
SUBMISSION_CHECKLIST.md
SUBMISSION_DAY_CHECKLIST.md
SUBMISSION_READINESS_SUMMARY.md
SUGGESTED_REVIEWERS.md
FINAL_SUBMISSION_CHECKLIST.md
PDF_CREATION_GUIDE.md
QUICK_START_SUBMISSION.md
ZENODO_ARCHIVAL_GUIDE.md
```

### Move to docs/status/
```
CURRENT_STATUS.md
FINAL_STATUS_REPORT.md
IMPROVEMENT_PROGRESS_REPORT.md
PROJECT_STATUS_AND_IMPROVEMENT_PLAN.md
PERFORMANCE_BASELINE_2026-01-04.md
ENVIRONMENT_CLEANUP_REQUIRED.md
```

### Move to docs/planning/
```
ROADMAP.md
CRITICAL_ROADMAP.md
LONG_TERM_ROADMAP.md
IMPROVEMENT_PLAN_2025.md
COMPREHENSIVE_IMPROVEMENT_PLAN.md
SYMTHAEA_IMPROVEMENT_ROADMAP.md
AWAKENING_ROADMAP_2025.md
SYMBIOTIC_AGI_ROADMAP.md
MAKING_SYMTHAEA_ALIVE_PLAN.md
IMMEDIATE_NEXT_STEPS.md
BENCHMARKING_STRATEGY.md
```

### Move to docs/tutorials/ (Quick Starts)
```
QUICK_START_CONSCIOUSNESS_SYSTEM.md
QUICK_START_INTEGRATION.md
README_FOR_TRISTAN.md
```

### Move to docs/sessions/ (Work Logs)
```
CINCINNATI_ADVANCED_COMPLETE.md
CINCINNATI_LTC_IMPROVEMENTS_SESSION.md
CINCINNATI_LTC_TEMPORAL_RESULTS.md
AUTONOMOUS_PREP_COMPLETE.md
QUICK_WINS_COMPLETE.md
QUICK_WINS_EXECUTION_SUMMARY.md
```

### Move to docs/reviews/
```
COMPREHENSIVE_PROJECT_REVIEW_2025.md
SYMTHAEA_COMPREHENSIVE_REVIEW.md
TECHNICAL_REVIEW.md
AWAKENING_INTEGRATION_ASSESSMENT.md
RESEARCH_PORTFOLIO_OVERVIEW.md
EXECUTIVE_SUMMARY.md
```

### Move to docs/developer/
```
DEVELOPER_GUIDE.md
NALGEBRA_MIGRATION_ANALYSIS.md
```

### Special Cases
```
PARADIGM_SHIFT.md          → docs/sympoietic/vision/
REVOLUTIONARY_ENHANCEMENTS.md → docs/sympoietic/implementation/
```

---

## Execution Plan

### Phase 1: Create Missing Directories
```bash
mkdir -p docs/paper
mkdir -p docs/reviews
```

### Phase 2: Move Files (by category)

#### Paper docs (19 files)
```bash
mv COMPLETE_MANUSCRIPT_FOR_PDF.md docs/paper/
mv MASTER_MANUSCRIPT.md docs/paper/
# ... etc
```

#### Architecture docs (7 files)
```bash
mv ARCHITECTURE_DEEP_DIVE.md docs/architecture/
# ... etc
```

### Phase 3: Verify No Broken Links
- Check all relative links in moved files
- Update any cross-references

### Phase 4: Create Index Files
- Create README.md in each major subdirectory
- Update root README.md with navigation

---

## Final Structure

```
symthaea-hlb/
├── README.md                    # Project overview
├── CLAUDE.md                    # Claude context
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # How to contribute
├── START_HERE.md                # Quick orientation
│
├── docs/
│   ├── README.md                # Documentation index
│   │
│   ├── sympoietic/              # ✅ ORGANIZED (14 docs)
│   │   ├── README.md
│   │   ├── vision/
│   │   ├── implementation/
│   │   ├── measurement/
│   │   └── frameworks/
│   │
│   ├── paper/                   # Academic submission (19 docs)
│   │   └── README.md
│   │
│   ├── architecture/            # System architecture (7+ docs)
│   │   └── README.md
│   │
│   ├── research/                # Φ research (8+ docs)
│   │   └── README.md
│   │
│   ├── planning/                # Roadmaps & plans (11 docs)
│   │   └── README.md
│   │
│   ├── status/                  # Status tracking (6 docs)
│   │   └── README.md
│   │
│   ├── reviews/                 # Project reviews (6 docs)
│   │   └── README.md
│   │
│   ├── tutorials/               # Quick starts (3 docs)
│   │   └── README.md
│   │
│   ├── sessions/                # Work logs (6+ docs)
│   │   └── README.md
│   │
│   ├── developer/               # Developer docs (2+ docs)
│   │   └── README.md
│   │
│   └── [existing dirs...]       # api, archive, etc.
│
└── src/                         # Source code
```

---

## Benefits

1. **Discoverability**: Every doc has a logical home
2. **Onboarding**: Clear paths for different audiences
3. **Maintenance**: Easier to keep organized
4. **Focus**: Root level is clean and essential

---

## Ready to Execute?

This plan will:
- Move ~70 files from root to appropriate subdirs
- Create 2 new directories (paper/, reviews/)
- Add README.md indexes to major sections
- Keep 5 essential files at root

Estimated time: 15-20 minutes

