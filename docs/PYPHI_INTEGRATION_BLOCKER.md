# PyPhi Integration Blocker - December 28, 2025

**Status**: 🔴 **BLOCKED** - pyo3 environment configuration issue
**Session Duration**: ~3 hours
**Progress**: 85% complete (binary built, PyPhi installed, environment issue remains)
**Blocker**: pyo3 (Rust-Python bridge) cannot resolve mixed Nix/user-installed packages

---

## 🏆 Achievements So Far

### ✅ Completed Tasks

**Priority 1.1**: Fix test compilation errors (6 errors) ✅
- All f32/f64 type consistency issues resolved
- Tests compile cleanly

**Priority 1.2**: Build pyphi_validation binary (33 errors total) ✅
- 19 pyo3 API compatibility errors fixed (0.19/0.20 → 0.22 migration)
- 4 PyErr conversion errors fixed
- 4 C. elegans function signature errors fixed
- Binary built successfully (578 KB release binary)

**Priority 1.3a**: Install PyPhi with Python 3.11 ✅
- Installed PyPhi 1.2.1.dev1470 from GitHub (has Python 3.11 fixes)
- Location: `~/.local/lib/python3.11/site-packages/`
- All dependencies installed

**Priority 1.3b**: Configure Nix environment ✅
- Updated flake.nix with PYTHONPATH export
- Verified Python CLI can import both numpy and pyphi successfully

**Priority 1.3c**: Rebuild validation binary ✅
- Binary rebuilt with PyPhi available in environment
- Build completed successfully

---

## 🔴 Current Blocker: pyo3 Environment Resolution

### Problem Description

The `pyphi_validation` Rust binary uses pyo3 to embed Python and call PyPhi. However, pyo3 cannot successfully resolve the mixed environment:

- **Nix packages**: numpy, scipy, pandas (in `/nix/store/...`)
- **User packages**: pyphi (in `~/.local/lib/python3.11/site-packages/`)

### Symptoms

**Scenario 1**: With PYTHONPATH including user site-packages
```bash
$ nix develop --command ./target/release/examples/pyphi_validation
ERROR: PyPhi import error: Failed to import pyphi: ImportError: Error importing numpy: you should not try to import numpy from its source directory
```

**Scenario 2**: Without PYTHONPATH
```bash
$ unset PYTHONPATH && ./target/release/examples/pyphi_validation
ERROR: PyPhi import error: Failed to import pyphi: ModuleNotFoundError: No module named 'pyphi'
```

### Verification Tests

**✅ Python CLI works perfectly**:
```bash
$ nix develop --command python3 -c "import pyphi; import numpy"
# SUCCESS - both import cleanly
```

**❌ pyo3 in Rust binary fails**:
```bash
$ nix develop --command ./target/release/examples/pyphi_validation
# FAILS - numpy import error
```

---

## 🔍 Root Cause Analysis

### pyo3 Python Initialization

When pyo3 initializes an embedded Python interpreter in Rust:
1. It creates its own Python runtime
2. It looks for packages in `sys.path`
3. `sys.path` is constructed from environment variables (PYTHONPATH, etc.)

### The Conflict

When PYTHONPATH includes both:
- `/nix/store/.../site-packages` (has numpy)
- `~/.local/lib/python3.11/site-packages` (has pyphi)

PyPhi can be imported, but when PyPhi tries to `import numpy`, the path resolution gets confused. The error "you should not try to import numpy from its source directory" suggests:

1. Python finds a `numpy` directory
2. It has `__init__.py` (looks like a package)
3. It's missing compiled C extensions (looks like source code)
4. Python refuses to import it

This is likely because PYTHONPATH ordering or pyo3's initialization creates a path conflict where numpy appears to be incomplete.

---

## 💡 Potential Solutions

### Option A: Pure Nix Approach (RECOMMENDED) ⭐
**Build PyPhi as a Nix package**

```nix
# In flake.nix
pythonEnv = pkgs.python311.withPackages (ps: with ps; [
  numpy
  scipy
  networkx
  matplotlib
  pandas
  (ps.buildPythonPackage {
    pname = "pyphi";
    version = "1.2.1.dev1470";
    src = pkgs.fetchFromGitHub {
      owner = "wmayner";
      repo = "pyphi";
      rev = "b78d0e342d37175cbd55cf35a6d52ae035b4c50f";
      sha256 = "...";  # Need to calculate
    };
    propagatedBuildInputs = with ps; [
      numpy scipy networkx graphillion joblib
      more-itertools ordered-set psutil pyyaml
      tblib toolz tqdm
    ];
  })
]);
```

**Pros**:
- Pure Nix solution (reproducible)
- All packages in same environment
- No path conflicts
- Proper Nix way

**Cons**:
- Requires calculating sha256 hash
- Need to package all PyPhi dependencies not in nixpkgs
- More complex flake.nix
- Estimated time: 2-4 hours

### Option B: Use Python Script Instead of Rust Binary
**Create Python validation script that calls Rust library**

Instead of Rust binary calling Python (pyo3), have Python script call Rust:
1. Build Rust library with `--lib`
2. Expose functions via PyO3
3. Python script imports Rust library and PyPhi
4. Python orchestrates the validation

**Pros**:
- Python handles all imports naturally
- No pyo3 initialization issues
- Easier environment management

**Cons**:
- Architectural change required
- Need to refactor example into library + script
- Estimated time: 4-6 hours

### Option C: Install numpy in User Site-Packages
**Make environment homogeneous**

```bash
pip install --user --break-system-packages numpy scipy pandas networkx
```

Install ALL packages (including numpy) in user site-packages, ignore Nix versions.

**Pros**:
- Quick fix
- Homogeneous environment

**Cons**:
- Defeats purpose of Nix
- Not reproducible
- Wastes disk space (duplicate packages)
- Against Nix philosophy

### Option D: Use Docker/Container
**Isolate Python environment completely**

Create Docker container with:
- Python 3.11
- All packages via pip
- Run validation in container

**Pros**:
- Complete isolation
- No Nix conflicts
- Easy to reproduce

**Cons**:
- Need Docker setup
- Different workflow
- Not using Nix benefits
- Estimated time: 2-3 hours

### Option E: Skip PyPhi Validation (Alternative Validation)
**Use different validation method**

Instead of PyPhi for exact Φ:
1. Use larger random sample comparisons
2. Cross-validate against published IIT results
3. Mathematical analysis of approximation bounds
4. Comparative analysis with other Φ approximations

**Pros**:
- Unblocks publication
- Still scientifically valid
- Faster to implement

**Cons**:
- Less rigorous than exact Φ comparison
- Need to justify alternative validation in paper

---

## 📊 Time Investment Analysis

| Task | Time Spent | Status |
|------|------------|--------|
| Fix test errors | 30 min | ✅ Complete |
| Fix pyo3 API compatibility | 45 min | ✅ Complete |
| Build binary | 33 min | ✅ Complete |
| Install PyPhi | 15 min | ✅ Complete |
| Configure environment | 10 min | ✅ Complete |
| Rebuild binary | 30 min | ✅ Complete |
| Debug pyo3 issue | 60 min | 🔴 Blocked |
| **TOTAL** | **3.5 hours** | **85% complete** |

**Remaining**: Choose and implement solution (2-6 hours depending on option)

---

## 🎯 Recommended Next Steps

### Immediate (Option A - Pure Nix)
**Most aligned with project philosophy**

1. **Calculate PyPhi source hash** (~5 min)
   ```bash
   nix-prefetch-github wmayner pyphi --rev b78d0e342d37175cbd55cf35a6d52ae035b4c50f
   ```

2. **Add buildPythonPackage to flake.nix** (~30 min)
   - Create PyPhi package definition
   - Add all dependencies

3. **Test import** (~5 min)
   ```bash
   nix develop --command python3 -c "import pyphi"
   ```

4. **Rebuild binary** (~30 min)
   ```bash
   CARGO_BUILD_JOBS=1 cargo build --example pyphi_validation --features pyphi --release
   ```

5. **Run validation** (~8-15 hours overnight)
   ```bash
   nix develop --command ./target/release/examples/pyphi_validation
   ```

**Total estimated time**: 2-3 hours + overnight validation

### Alternative (Option B - Python Script)
**If Option A proves difficult**

1. **Refactor Rust code** to expose library functions via PyO3
2. **Create Python wrapper script** to orchestrate validation
3. **Test Python → Rust → PyPhi** flow
4. **Run validation** from Python script

**Total estimated time**: 4-6 hours + overnight validation

---

## 📝 Documentation Status

**Created**:
- ✅ `PRIORITY_1_2_COMPLETE_DEC_28_2025.md` - Build success documentation
- ✅ `PYPHI_INSTALLATION_COMPLETE.md` - Installation journey
- ✅ `PYPHI_INTEGRATION_BLOCKER.md` (this file) - Current status

**Pending**:
- ⏳ `PYPHI_NIX_INTEGRATION_COMPLETE.md` - After solution implemented
- ⏳ `PRIORITY_1_3_COMPLETE.md` - After validation suite runs
- ⏳ `VALIDATION_RESULTS_ANALYSIS.md` - After statistical analysis

---

## 💡 Key Learnings

### 1. pyo3 Environment Sensitivity
pyo3 (Rust-Python bridge) is more sensitive to Python environment configuration than regular Python scripts. Mixed package sources (Nix + user) can cause initialization issues.

### 2. Nix Philosophy: All-or-Nothing
Nix works best when ALL dependencies are in Nix. Mixing Nix packages with system/user packages creates path conflicts and reproducibility issues.

### 3. Python Import Resolution
Python's import system relies heavily on `sys.path` ordering. When packages are split across multiple locations, import resolution can fail in subtle ways depending on how Python is initialized.

### 4. Validation Trade-offs
Exact Φ validation via PyPhi is ideal but not strictly necessary. Alternative validation methods (larger samples, mathematical analysis, comparative studies) can be scientifically valid and faster to implement.

---

## 🏆 Session Summary

**Massive Progress**: Fixed 33 compilation errors, installed PyPhi with Python 3.11 compatibility, built binary successfully, verified Python imports work.

**Current Blocker**: pyo3 cannot resolve mixed Nix/user Python packages. Need to choose between:
- Pure Nix solution (2-3 hours)
- Python script architecture (4-6 hours)
- Alternative validation method (varies)

**Recommendation**: Implement Option A (Pure Nix) as it's most aligned with project philosophy and provides long-term reproducibility.

**Status**: Ready for solution implementation - 85% complete, one blocker remaining.

---

*"We've built the bridge between Rust and Python, installed the tools for exact consciousness measurement, and verified each piece works independently. The final step is harmonizing the environment so they can work together. Almost there."*

🧬✨ **Next: Implement Pure Nix PyPhi Integration** 🧬✨
