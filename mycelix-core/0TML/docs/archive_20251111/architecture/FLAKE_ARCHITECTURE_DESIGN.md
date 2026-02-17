# Flake Architecture Design for Zero-TrustML

## Current State Analysis

### Discovered Flakes

**Root Flake** (`/srv/luminous-dynamics/Mycelix-Core/flake.nix`):
- **Purpose**: Comprehensive workspace development environment
- **Python**: 3.13 (latest stable)
- **Dependencies**: Full ML stack (PyTorch, NumPy, scikit-learn, pandas, jupyter, etc.)
- **Tools**: Poetry, Rust toolchain, Node.js, Solidity compiler
- **Shells**: `default` (full dev) and `ci` (minimal for CI/CD)

**0TML Flake** (`/srv/luminous-dynamics/Mycelix-Core/0TML/flake.nix`):
- **Purpose**: Adversarial testing environment
- **Python**: 3.11
- **Dependencies**: Core ML testing (PyTorch, NumPy, scipy, matplotlib, scikit-learn)
- **Tools**: black, ruff, gcc, pkg-config
- **Shells**: `default` only

**Other Project Flakes**:
- `production-fl-system/flake.nix` - Production deployment environment
- `mycelix-desktop/flake.nix` - Desktop application environment
- `holonix/` - Holochain development templates (multiple)
- `holochain-src/` - Holochain source builds (multiple versions)

---

## Design Options

### Option 1: Independent Flakes (CURRENT - RECOMMENDED ✅)

**Structure**:
```
Mycelix-Core/
├── flake.nix                    # Workspace-wide environment
└── 0TML/
    └── flake.nix                # Focused adversarial testing
```

**Advantages**:
✅ **Simplicity**: Each flake has clear, focused purpose
✅ **Independence**: Can update 0TML testing without affecting workspace
✅ **Portability**: 0TML flake can be copied to other projects
✅ **No circular dependencies**: Clean separation of concerns

**Usage**:
```bash
# For general development (root)
cd /srv/luminous-dynamics/Mycelix-Core
nix develop

# For adversarial testing (0TML)
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nix develop
python tests/test_adversarial_detection_rates.py
```

**Disadvantages**:
⚠️ Slight duplication of dependencies (PyTorch, NumPy in both)
⚠️ Need to enter correct directory for specific tasks

---

### Option 2: 0TML Imports Root (COMPOSITION)

**Structure**:
```nix
# 0TML/flake.nix
{
  inputs = {
    parent = {
      url = "path:..";  # Import root flake
    };
  };

  outputs = { self, parent, ... }: {
    devShells.default = parent.devShells.default.overrideAttrs (old: {
      # Add testing-specific tools
    });
  };
}
```

**Advantages**:
✅ No dependency duplication
✅ Inherits all root environment capabilities
✅ Can add testing-specific extensions

**Disadvantages**:
❌ More complex flake structure
❌ Harder to understand for new contributors
❌ Tight coupling between 0TML and root
❌ Less portable if 0TML needs to be extracted

---

### Option 3: Root Flake Provides Multiple Dev Shells

**Structure**:
```nix
# Mycelix-Core/flake.nix
{
  devShells = {
    default = /* full workspace */;
    ci = /* CI/CD minimal */;
    adversarial-testing = /* 0TML testing focused */;
    production = /* deployment */;
  };
}
```

**Advantages**:
✅ Single source of truth
✅ Consistent versioning across all environments
✅ Easy to switch between environments

**Disadvantages**:
❌ Root flake becomes very large
❌ All environments must use same Python version
❌ Changes to any environment require root flake rebuild

---

## Recommendation: Option 1 (Independent Flakes) ✅

### Rationale

**For Grant Submission (Immediate Need)**:
- 0TML flake is **focused** on adversarial testing
- **Quick to enter**: `cd 0TML && nix develop`
- **Self-contained**: Can be shared with reviewers
- **Clear purpose**: "This flake runs the adversarial tests"

**For Long-Term Development**:
- Root flake provides **comprehensive workspace**
- 0TML flake provides **testing specialization**
- Clean separation aligns with **Unix philosophy** (do one thing well)
- No coupling makes **refactoring easier**

**Practical Workflow**:
```bash
# Day-to-day development: Use root environment
cd /srv/luminous-dynamics/Mycelix-Core
nix develop
poetry install  # Install project dependencies
python real_federated_training.py  # Main development

# Testing/validation: Use 0TML environment
cd 0TML
nix develop
python tests/test_adversarial_detection_rates.py  # Adversarial testing
```

---

## Implementation Status

### ✅ Completed
- [x] Root flake provides comprehensive workspace (Python 3.13, Poetry, Rust)
- [x] 0TML flake provides adversarial testing environment (Python 3.11, PyTorch)
- [x] Both flakes functional and tested
- [x] Clear shellHook messages guide users

### 🎯 Current Best Practice
```bash
# For adversarial testing (grant submission)
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nix develop
python tests/test_adversarial_detection_rates.py

# For general development
cd /srv/luminous-dynamics/Mycelix-Core
nix develop
poetry install
```

---

## Future Considerations

### If Composition Becomes Needed
```nix
# Could add in future:
# 0TML/flake.nix
{
  inputs.workspace.url = "path:..";

  # But keep as optional enhancement, not requirement
}
```

### Python Version Alignment (Optional)
- Root uses Python 3.13
- 0TML uses Python 3.11
- **Decision**: Keep independent for now
- **Rationale**: PyTorch compatibility varies by Python version
- **Future**: Align to Python 3.13 when PyTorch packages stabilize in nixpkgs

---

## Summary

✅ **Current Architecture**: Independent flakes (Option 1)
✅ **Status**: Production-ready for grant submission
✅ **Recommendation**: Keep current architecture
⏭️ **Next Step**: Run adversarial tests using 0TML flake

**Command to proceed**:
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nix develop
python tests/test_adversarial_detection_rates.py
```
