# 📁 Mycelix Protocol Framework - Folder Reorganization Plan

**Date**: 2025-10-13
**Purpose**: Eliminate duplication, clarify active vs. historical code

---

## 🎯 Recommended Structure

```
Mycelix-Protocal-Framework/
├── flake.nix                    # DELETE (use 0TML's instead)
├── *.py (80+ files)             # ARCHIVE to archive/experiments-historical/
│
├── 0TML/              # ← MAKE THIS THE ROOT
│   ├── flake.nix               # ← PRIMARY development environment
│   ├── experiments/
│   │   ├── runner.py           # ✅ Active (used by Grand Slam)
│   │   ├── run_grand_slam.py   # ✅ Active
│   │   ├── configs/
│   │   └── results/
│   ├── src/                    # ✅ Core implementations
│   ├── baselines/              # ✅ FedAvg, Multi-Krum, PoGQ
│   ├── holochain/              # ✅ Phase 10 integration
│   └── contracts/              # ✅ Smart contracts
│
├── archive/                     # NEW: Historical experiments
│   ├── experiments-historical/  # Move all root *.py files here
│   ├── holochain-attempts/      # Old Holochain integration attempts
│   ├── performance-tests/       # Old benchmark scripts
│   └── README.md               # Document what's archived and why
│
├── docs/                        # Keep documentation
└── README.md                    # Update to point to 0TML

```

---

## 📦 Proposed Reorganization Steps

### Step 1: Archive Historical Experiments (Root Directory)
**Move these to `archive/experiments-historical/`:**
```
byzantine-fl-*.py (multiple variants)
performance_benchmarks*.py
test_cifar10_datasets*.py
byzantine_krum_defense*.py
test_hfl_*.py
fl_holochain_*.py
holochain_client*.py
phase1_smoke_test.py, phase2_*.py, phase3_*.py
run_real_*.py (multiple old training scripts)
deploy-holochain-*.py
install_*.py
verify_byzantine_*.py
implement_*.py
generate_*.py
```

**Total**: ~80 Python files

### Step 2: Archive Old Holochain Attempts
**Move to `archive/holochain-attempts/`:**
```
holochain-src/ (if no longer used)
holonix/ (if no longer used)
demo-without-conductor.py
test-admin-api.py
test-conductor-directly.py
mock_holochain_conductor.py
```

### Step 3: Consolidate to Single Flake
**Action**: Delete root `flake.nix`, use only `0TML/flake.nix`

**Rationale**:
- 0TML flake is more modern (Python 3.13, torch-bin with CUDA)
- Has all Phase 10 dependencies (asyncpg, websockets, web3, cosmpy)
- Eliminates duplicate dependency downloads

### Step 4: Update Directory Entry Point
**Make 0TML the primary workspace:**

Option A: Rename `0TML/` → `src/` or `active/`
Option B: Move everything from 0TML/ up to root
Option C: Keep 0TML name but update all docs to reference it

**Recommended: Option B (Move up to root)**
```bash
# After backing up to archive/
cd /srv/luminous-dynamics/Mycelix-Protocal-Framework
mv 0TML/* .
mv 0TML/.* . 2>/dev/null || true
rmdir 0TML
```

### Step 5: Create Archive Documentation
**File**: `archive/README.md`

Document:
- What was archived and when
- Why it was archived (superseded by Phase 10 implementation)
- How to access if needed for reference
- Historical context for research continuity

---

## 🔧 Benefits of This Reorganization

### 1. **Eliminate Duplication** ✅
- One runner.py (not scattered implementations)
- One set of baseline algorithms
- One Holochain integration (Phase 10)
- One Nix environment

### 2. **Faster Development** ⚡
- Single `nix develop` command
- No confusion about which files are active
- Clear entry point for new contributors
- Instant subsequent nix shell (cached packages)

### 3. **Clearer Architecture** 🏗️
```
experiments/    → Run experiments (Grand Slam, etc.)
src/           → Core implementations
baselines/     → Reference algorithms
holochain/     → Blockchain integration
contracts/     → Smart contracts
archive/       → Historical reference
```

### 4. **Preserve History** 📚
- Nothing deleted, just organized
- Easy to reference old experiments
- Research continuity maintained
- Can diff against historical implementations

---

## 🚀 Implementation Timeline

### Phase 1: Archive (1 hour)
1. Create `archive/` structure
2. Move historical files
3. Test that Grand Slam still runs

### Phase 2: Consolidate Flake (30 min)
1. Delete root flake.nix
2. Update docs to reference 0TML flake
3. Test `nix develop` from root and subdirs

### Phase 3: Optional Flatten (1 hour)
1. Move 0TML contents to root
2. Update all relative imports
3. Test all active scripts

**Total Time**: 2-3 hours for clean, maintainable structure

---

## ⚠️ Safety Checks Before Reorganizing

### Before moving ANY files:
```bash
# 1. Ensure Grand Slam is not running
ps aux | grep "python run_grand_slam.py"

# 2. Create full backup
cd /srv/luminous-dynamics
tar -czf Mycelix-backup-$(date +%Y%m%d-%H%M%S).tar.gz Mycelix-Protocal-Framework/

# 3. Commit current state to git
cd Mycelix-Protocal-Framework
git add -A
git commit -m "Checkpoint before reorganization"

# 4. Create reorganization branch
git checkout -b reorganize-folders
```

### After reorganization:
```bash
# Test that experiments still work
cd 0TML/experiments  # or new location
nix develop
python -c "import runner; print('✅ runner imports successfully')"

# Test Grand Slam (dry run)
nix develop --command python -c "
import yaml
import runner
print('✅ All imports successful')
"
```

---

## 📝 Questions to Decide

1. **Keep 0TML name or rename?**
   - Keep: Preserves git history
   - Rename: More intuitive (src/, active/, experiments/)

2. **Move everything to root or keep nested?**
   - Flatten: Simpler paths, easier navigation
   - Nested: Clear separation of concerns

3. **When to archive?**
   - Now: Clean slate for future work
   - After Grand Slam: Don't disrupt running experiments

---

## 💡 Nix Dependency Best Practices

### Why Dependencies Re-download
**Current Issue**: Two different flakes with different inputs

**Solution**: One canonical flake + direnv

**Setup direnv (one-time):**
```bash
# Install direnv
nix-env -iA nixpkgs.direnv

# Add to ~/.bashrc or ~/.zshrc
eval "$(direnv hook bash)"  # or zsh

# Create .envrc in project root
echo "use flake" > .envrc
direnv allow

# Now 'cd' into directory auto-enters nix shell!
```

**Result**:
- First `nix develop`: Downloads packages (one-time)
- Subsequent enters: Instant (<0.1s)
- Auto-enters shell on `cd`
- Packages cached in `/nix/store`

---

## 📊 Expected Outcomes

### Before Reorganization
- ❌ 80+ Python files in root
- ❌ Unclear which code is active
- ❌ Two flakes downloading different deps
- ❌ Duplicate implementations

### After Reorganization
- ✅ Clear active codebase
- ✅ Single source of truth
- ✅ One flake, instant subsequent runs
- ✅ History preserved in archive/
- ✅ Easy onboarding for new contributors

---

**Next Steps**:
1. Review this plan
2. Decide on naming/flattening preferences
3. Wait for Grand Slam to complete
4. Execute reorganization
5. Update documentation

---

*This reorganization preserves all research work while creating a clean, maintainable structure for future development.*
