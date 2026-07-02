# Mycelix-Core Cleanup Summary

**Date**: 2026-01-04
**Status**: COMPLETED

---

## Space Recovery Summary

| Location | Before | After | Saved |
|----------|--------|-------|-------|
| Mycelix-Core | ~125GB | 56GB | **~69GB** |
| mycelix-workspace | ~25GB | 1.9GB | **~23GB** |
| **Subtotal (Initial Cleanup)** | ~150GB | 58GB | **~92GB** |
| **Additional Cleanup (Audit)** | | | |
| production-fl-system .venv/.postgres | 7.3GB | 410MB | **~6.9GB** |
| mycelix-desktop zome targets | 520MB | 1MB | **~519MB** |
| **Grand Total** | ~158GB | ~51GB | **~100GB** |
| **Final Cleanup (Restructuring)** | | | |
| zomes/federated_learning/target | 1.5GB | 0 | **1.5GB** |
| 0TML/holochain/zomes/*/target | 1.4GB | 0 | **1.4GB** |
| **FINAL TOTAL** | **~161GB** | **~39GB** | **~122GB** |
| mycelix-mail/cli/target | 2.6GB | 0 | **2.6GB** |
| Archives consolidated | - | - | (organization) |
| **GRAND TOTAL** | **~164GB** | **~37GB** | **~127GB** |

---

## What Was Cleaned

### 1. Conductor Configurations (Archived)
- **30 conductor config files** moved to `.archive-2026-01-04/conductor-configs/`
- **13 conductor data directories** archived
- Kept: `conductor-config.yaml` as canonical

### 2. Nix File Variants (Archived)
- **12 nix files** moved to `.archive-2026-01-04/nix-variants/`
  - `flake-broken.nix`, `flake-minimal.nix`, `flake-original.nix`, etc.
  - `shell-build.nix`, `shell-holochain.nix`, `shell-python.nix`, etc.
- Kept: `flake.nix`, `flake.lock`, `shell.nix`

### 3. Docker Variants (Archived)
- **4 Dockerfile variants** moved to `.archive-2026-01-04/docker-variants/`
- Kept: `Dockerfile` (main)

### 4. Holochain Duplicates (Archived)
- `holochain/` (root level, sparse) - archived
- `holochain-src/` (2.1GB source) - archived
- `holonix/` (716KB) - archived
- Empty archives removed: `holochain.tar.gz`, `lair.tar.gz`

### 5. Zome/DNA Duplicates (Archived)
- `simple_zome/` - test zome, archived
- `reputation_zome/` (1.8GB with target) - archived
- `civitas_dna/` (1.1GB) - archived
- `happ-fl/`, `test-happ/` - archived

### 6. Build Artifacts (Removed - Regenerable)

| Directory | Size Removed |
|-----------|-------------|
| `target/` (root) | 2.0GB |
| `production-fl-system/venv/` | 6.6GB |
| `production-fl-system/node_modules/` | 6MB |
| `0TML/gen7-zkstark/target/` | 1.5GB |
| `0TML/rust-bridge/target/` | 2.6GB |
| `0TML/holochain/target/` | 11GB |
| `0TML/cosmos/target/` | 1.3GB |
| `0TML/mycelix_fl/holochain/target/` | 14GB |
| `0TML/mycelix_fl/sdk/rust/target/` | 9.7GB |
| `0TML/zerotrustml-core/target/` | 7.8GB |
| `0TML/vsv-stark/target/` | 7.5GB |
| `mycelix-workspace/sdk/target/` | 1.4GB |
| `mycelix-workspace/sdk/zk-tax/target/` | 20GB |
| **Total Build Artifacts** | **~86GB** |

### 7. Empty/Placeholder Files (Removed)
- `_0TML_docs_testing_README.md` (0 bytes)
- `_README.md` (0 bytes)
- `mock_server.pid`, `pid_file`
- `federated_learning.wasm` (8 bytes placeholder)
- `h-fl.dna` (13 bytes placeholder)
- Nix result symlinks

---

## Current Structure

```
Mycelix-Core/                    # 56GB total
├── 0TML/                        # 30GB - Main Zero-TrustML implementation
│   ├── src/                     # Python source
│   ├── tests/                   # Test suite
│   ├── holochain/               # Holochain components (no target/)
│   ├── experiments/             # Experiment results
│   ├── datasets/                # Training data (6.6GB)
│   └── docs/                    # Documentation
│
├── .archive-2026-01-04/         # 5.2GB - Today's archived files
│   ├── conductor-configs/       # All conductor variants
│   ├── nix-variants/            # All nix variants
│   ├── docker-variants/         # All docker variants
│   ├── holochain-old/           # Old holochain dirs
│   └── zomes-old/               # Old zome implementations
│
├── production-fl-system/        # 7.3GB - Production system
├── mycelix-desktop/             # 6.6GB - Tauri desktop app
├── mycelix-mail/                # 2.6GB - Mail hApp
├── zomes/                       # 2.1GB - Main zomes
├── dna/                         # 276MB - Compiled DNAs
│
├── conductor-config.yaml        # Canonical conductor config
├── flake.nix                    # Canonical flake
├── shell.nix                    # Canonical shell
├── Dockerfile                   # Canonical Dockerfile
├── README.md
└── CLAUDE.md
```

---

## Verification

The following should still work after cleanup:

```bash
# Enter Nix development environment
nix develop

# Run Python tests (in 0TML)
cd 0TML && poetry install && pytest

# Build Rust components (will regenerate target/)
cargo build

# Start Holochain conductor
hc sandbox up -c conductor-config.yaml
```

---

## Archive Contents

### `.archive-2026-01-04/conductor-configs/`
Contains 43 conductor-related files and directories that were variants/duplicates.
Can be safely deleted after 30 days if no issues arise.

### `.archive-2026-01-04/zomes-old/`
Contains old zome implementations. Source code preserved; target directories were removed.

### `.archive-legacy-fl-2025-12-31/`
Reduced from 2.4GB to 53MB by removing target directories.
Contains historical FL implementation for reference.

---

## Recommendations

### Short-term (Next 30 Days)
1. Verify all builds work correctly
2. Run full test suite
3. If no issues, delete `.archive-2026-01-04/` to recover 5.2GB

### Medium-term
1. Consider moving `production-fl-system/` to separate repo or archive
2. Review `mycelix-desktop/` - is it actively developed?
3. Document canonical build process in README

### Long-term
1. Add `.gitignore` entries for `target/`, `venv/`, `node_modules/`
2. Consider git submodules for large components
3. Set up CI to catch target directory commits

---

## Notes

- All archived files can be restored from `.archive-2026-01-04/`
- Build artifacts were removed, not archived (they're regenerable)
- The `0TML/` directory is the canonical implementation
- `mycelix-workspace/happs/core` symlinks to this repo

---

---

## Restructuring Actions (Phase 2)

### 1. production-fl-system Archived
- **Action**: Moved to `.archive-legacy-production-fl-2026-01-04/`
- **Reason**: README explicitly marked as legacy, recommends migration to 0TML
- **Size**: 410MB (after .venv removal)

### 2. mycelix-desktop Integrated into Workspace
- **Action**: Created symlink `mycelix-workspace/happs/desktop -> ../../Mycelix-Core/mycelix-desktop`
- **Reason**: Functional Tauri desktop app should be part of unified workspace
- **Next Steps**: Upgrade Holochain 0.5.6 → 0.6.0, integrate with SDK zomes

### 3. Root zomes/ and dna/ Reviewed
- **Finding**: NOT duplicates of 0TML/holochain/zomes/
  - Root zomes/: agents, bridge, federated_learning (general purpose)
  - 0TML zomes/: pogq_zome, reputation_tracker, defense_* (Byzantine FL specific)
- **Action**: Keep both, removed target directories (2.9GB saved)

---

## Final Structure

```
Mycelix-Core/                              37GB (was 125GB+)
├── 0TML/                                  28GB - Canonical FL implementation
├── mycelix-desktop/                       15MB - Desktop app (symlinked to workspace)
├── mycelix-mail/                          23MB - Mail hApp (cleaned)
├── zomes/                                 586MB - General purpose zomes
├── dna/                                   276MB - DNA definitions
└── .archive/                              5.7GB - Consolidated archives
    ├── 2025-09-30/                        4KB
    ├── 2026-01-02/                        108KB
    ├── 2026-01-04/                        5.2GB - Main cleanup
    ├── legacy-fl-2025-12-31/              53MB
    └── legacy-production-fl-2026-01-04/   410MB

mycelix-workspace/happs/
├── core -> ../../Mycelix-Core
├── desktop -> ../../Mycelix-Core/mycelix-desktop  # NEW
└── [15 other hApps...]
```

---

---

## Phase 3: Root Directory Optimization

### Directories Consolidated/Archived
Reduced from **47 directories** to **12 directories** at root level.

**Moved to `.archive/root-consolidation-2026-01-04/`:**
- Test/temp: `test-conductor-data/`, `test-wasm/`, `local-conductor-data/`, `vvW4f-8TzzK1Da9peqIM3/`
- Research: `artifacts/`, `baselines/`, `experiments/`, `figures/`, `notebooks/`, `paper/`, `results/`, `visualizations/`
- Legacy: `production_20250926_*/`, `benches/`, `bindings/`, `byzantine_fl_holochain/`, etc.
- Tools: `holochain-bin/`, `tools/`, `spikes/`, `workdir/`, `site/`

### Markdown Files Organized
- Session summaries → `.archive/root-consolidation-2026-01-04/session-docs/`
- Strategy docs → `docs/` (CI_CD, HOLOCHAIN_MIGRATION, ROADMAP, etc.)

### Justfile Updated
- Removed duplicate `justfile` (kept `Justfile`)
- Fixed broken references to archived `tools/` directory
- Added comprehensive commands for dev, test, build, clean

---

## Final Clean Structure

```
Mycelix-Core/                    37GB (was 125GB+)
├── 0TML/                        28GB  # Canonical implementation
├── docs/                        5MB   # All documentation
├── mycelix-desktop/             15MB  # Desktop app
├── mycelix-mail/                23MB  # Mail hApp
├── mycelix-fl-pure-p2p/         39MB  # Pure P2P FL
├── zomes/                       586MB # General zomes
├── dna/                         276MB # DNA definitions
├── data/                        404MB # Training data
├── real_data/                   352MB # Real experiment data
├── scripts/                     -     # Utility scripts
├── tests/                       -     # Test suite
├── _websites/                   -     # Web properties
└── .archive/                    6GB   # All archives consolidated
    ├── 2026-01-04/              5.2GB
    ├── root-consolidation-*/    300MB
    └── [other dated archives]

Root files: 6 markdown, Justfile, flake.nix, configs
```

---

*Cleanup and restructuring completed successfully. Total space recovered: ~127GB*
*Root directories reduced from 47 to 12. Project structure significantly cleaner.*
