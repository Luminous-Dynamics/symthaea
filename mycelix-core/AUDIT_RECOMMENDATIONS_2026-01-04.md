# Mycelix-Core Audit: Integration & Restructuring Recommendations

**Date**: 2026-01-04
**Status**: REVIEW BEFORE IMPLEMENTATION

---

## Executive Summary

This audit examined `production-fl-system` (7.3GB → ~1GB after cleanup) and `mycelix-desktop` (6.6GB → ~14MB after cleanup) to determine if they should be improved, integrated, reorganized, or restructured relative to the canonical `0TML/` implementation and `mycelix-workspace/`.

### Key Findings

| Component | Original Size | After Cleanup | Recommendation | Action |
|-----------|--------------|---------------|----------------|--------|
| production-fl-system | 7.3GB | 410MB | **ARCHIVE** | Move to `.archive-legacy/` |
| mycelix-desktop | 6.6GB | 15MB | **INTEGRATE** | Move to `mycelix-workspace/` |

**Cleanup Performed During Audit**:
- Removed `.venv` (6.9GB) and `.postgres` (39MB) from production-fl-system
- Removed target directories (519MB) from mycelix-desktop/dnas/mycelix-test/zomes/

---

## 1. production-fl-system Analysis

### Current State
- **Size**: ~1GB (after venv removal earlier)
- **Purpose**: Historical end-to-end deployment assets for production federation demos
- **Status**: LEGACY - README explicitly states migration to 0TML

### Structure Breakdown
| Directory | Size | Contents |
|-----------|------|----------|
| `data/` | 404MB | MNIST, CIFAR-10 (duplicated in 0TML/data/) |
| `archive/` | 2.3MB | 115+ legacy Python scripts |
| `docs/` | 748KB | 60+ markdown files |
| `scripts/` | ~100KB | 30+ shell scripts |
| `holochain-*/` | Various | Legacy Holochain configs |

### Overlap with 0TML

| Capability | production-fl-system | 0TML | Canon |
|------------|---------------------|------|-------|
| Byzantine FL | Legacy Python demos | 45% BFT, 105 tests | **0TML** |
| Holochain Zomes | Obsolete references | HDK 0.6.0, validated | **0TML** |
| Training Data | MNIST/CIFAR-10 | Same + more | **0TML** |
| Deployment Scripts | Ad-hoc shell | Structured, documented | **0TML** |

### Recommendation: **ARCHIVE**

**Rationale**: The README itself says "migrate to 0TML". The directory contains:
- Duplicated training data (404MB)
- Legacy demos superseded by 0TML implementations
- Historical deployment configs no longer relevant
- No unique value not already in 0TML

**Action Plan**:
1. Extract any unique experiment results to `0TML/experiments/legacy/`
2. Move entire directory to `.archive-legacy-production-fl/`
3. Create symlink for backward compatibility if needed
4. Delete after 60-day verification period

---

## 2. mycelix-desktop Analysis

### Current State
- **Size**: ~14MB (after target cleanup - was 6.6GB with targets, reduced to 500MB, then 14MB)
- **Purpose**: Tauri + SolidJS desktop application for Mycelix network
- **Status**: FUNCTIONAL but standalone

### Architecture
```
mycelix-desktop/
├── src/                    # SolidJS frontend (60KB)
│   └── App.tsx            # 895 lines - Full Holochain integration
├── src-tauri/             # Tauri backend (840KB)
│   └── src/main.rs        # 431 lines - Conductor management
├── dnas/mycelix-test/     # Test DNA (888KB .dna file)
│   └── zomes/             # hello + messages zomes
├── happs/                 # hApp definitions (444KB)
└── node_modules/          # Dependencies (12MB)
```

### Key Features (from App.tsx)
- ✅ Holochain conductor process management
- ✅ Network status display (WebSocket/P2P)
- ✅ App installation and management
- ✅ Zome function testing (hello world, messages)
- ✅ Message broadcasting and retrieval
- ✅ Real-time status updates

### Technical Observations
1. **Using Holochain 0.5.6** (noted in App.tsx comment)
2. **Self-contained** - manages its own conductor process
3. **Test-focused** - uses simple hello/messages zomes
4. **No 0TML integration** - doesn't use Byzantine FL or PoGQ

### Overlap with Workspace

| Aspect | mycelix-desktop | mycelix-workspace | Alignment |
|--------|----------------|-------------------|-----------|
| Location | Mycelix-Core/ | /srv/luminous-dynamics/ | **Should be hApp** |
| Holochain version | 0.5.6 | 0.6.0 | **Needs update** |
| Conductor | Self-managed | Shared conductor | **Could consolidate** |
| Zomes | Simple test | Full SDK zomes | **Should use SDK** |

### Recommendation: **INTEGRATE INTO WORKSPACE**

**Rationale**: mycelix-desktop is a functional Tauri desktop app that should:
1. Live in `mycelix-workspace/` as the official desktop client
2. Use workspace SDK zomes instead of test zomes
3. Share conductor configuration with other hApps
4. Integrate with 0TML for Byzantine FL visualization

**Action Plan**:
1. Move to `mycelix-workspace/desktop/` or create new symlink `mycelix-workspace/happs/desktop`
2. Update Holochain dependencies to 0.6.0
3. Replace test zomes with SDK zomes (`@mycelix/sdk`)
4. Connect to workspace's shared conductor configuration
5. Add 0TML visualization panels (FL training status, Byzantine detection)

---

## 3. Overlap Summary

### Data Duplication
| Data | Location 1 | Location 2 | Recommendation |
|------|-----------|-----------|----------------|
| MNIST | production-fl-system/data/ | 0TML/data/ | Keep only 0TML |
| CIFAR-10 | production-fl-system/data/ | 0TML/data/ | Keep only 0TML |
| Test DNAs | mycelix-desktop/dnas/ | 0TML/holochain/dnas/ | Use 0TML builds |

### Code Duplication
| Component | Locations | Canonical |
|-----------|----------|-----------|
| Byzantine FL | production-fl-system/archive/*.py | 0TML/src/ |
| Holochain configs | 6+ directories | 0TML/holochain/conductor-config.yaml |
| Zome implementations | zomes/, 0TML/holochain/zomes/, mycelix-desktop/dnas/ | 0TML/holochain/zomes/ |

### Architectural Clarity

**Current State**:
```
Mycelix-Core/
├── 0TML/                      # ✅ Canonical FL implementation
├── production-fl-system/      # ❌ Legacy, should archive
├── mycelix-desktop/           # ⚠️ Should be in workspace
├── zomes/                     # ⚠️ May be duplicate of 0TML
└── dna/                       # ⚠️ May be duplicate of 0TML
```

**Recommended State**:
```
Mycelix-Core/
├── 0TML/                      # ✅ Canonical FL implementation
├── zomes/                     # ✅ If unique, otherwise archive
├── dna/                       # ✅ If unique, otherwise archive
└── .archive-legacy/           # Archived production-fl-system

mycelix-workspace/
├── happs/core -> Mycelix-Core
├── happs/desktop -> NEW       # Migrated mycelix-desktop
├── sdk/                       # Shared SDK
└── ...
```

---

## 4. Recommended Action Plan

### Phase 1: Immediate Cleanup (Today)
- [x] Remove hidden target directories (done: saved 519MB more)
- [ ] Verify 0TML/data/ has all needed datasets
- [ ] Create `.archive-legacy-production-fl/` directory

### Phase 2: production-fl-system Archive (This Week)
1. **Extract unique value**:
   ```bash
   # Copy any unique experiment results
   cp -r production-fl-system/results/* 0TML/experiments/legacy/ 2>/dev/null

   # Copy any actively-used scripts
   # Review each script in production-fl-system/scripts/
   ```

2. **Archive the directory**:
   ```bash
   mv production-fl-system .archive-legacy-production-fl-2026-01-04/
   ```

3. **Update references**:
   - Check for symlinks pointing to production-fl-system
   - Update any CI/CD scripts
   - Update documentation

### Phase 3: mycelix-desktop Integration (Next Week)
1. **Create workspace integration**:
   ```bash
   # Move to workspace
   mv mycelix-desktop ../mycelix-workspace/desktop

   # Or create symlink in happs
   ln -s ../../../Mycelix-Core/mycelix-desktop ../mycelix-workspace/happs/desktop
   ```

2. **Update dependencies**:
   - Upgrade Holochain to 0.6.0
   - Update Tauri to latest
   - Replace test zomes with SDK zomes

3. **Integrate with 0TML**:
   - Add status panels for FL training
   - Add Byzantine detection visualization
   - Connect to shared conductor

### Phase 4: Root Directory Cleanup (Next Week)
1. **Audit root zomes/ and dna/**:
   - Compare with 0TML/holochain/zomes/ and 0TML/holochain/dnas/
   - Archive if duplicated
   - Keep if unique functionality

2. **Final structure verification**:
   - Run all tests
   - Verify workspace symlinks
   - Document final architecture

---

## 5. Space Recovery Summary

| Action | Space Saved |
|--------|-------------|
| Earlier cleanup (target dirs) | ~92GB |
| production-fl-system .venv/.postgres removal | 6.9GB |
| mycelix-desktop zome target cleanup | 519MB |
| **Total Recovered This Session** | **~100GB** |
| **Additional If Archived** | |
| production-fl-system archive | 410MB |
| Data deduplication | 404MB |
| **Grand Total Potential** | ~101GB |

---

## 6. Risk Assessment

### Low Risk
- Archiving production-fl-system (README says to migrate)
- Removing duplicate target directories (regenerable)
- Removing duplicate training data (exists in 0TML)

### Medium Risk
- Moving mycelix-desktop (may have users/scripts depending on path)
- Updating Holochain version in mycelix-desktop (breaking changes)

### Mitigation
- Create symlinks for backward compatibility
- Run all tests before and after changes
- Keep archives for 60 days before deletion

---

## Approval Checklist

Before executing this plan:
- [ ] Review this document
- [ ] Confirm production-fl-system has no active users
- [ ] Verify 0TML has all needed experiment data
- [ ] Test mycelix-desktop before and after move
- [ ] Create rollback plan

---

*Audit completed. Ready for review and execution.*
