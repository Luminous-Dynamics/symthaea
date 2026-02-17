# Mycelix Zome and hApp Audit Report

**Generated**: December 31, 2025
**Purpose**: Identify all zomes built for Mycelix FL and identify duplicates

---

## Summary

| Category | Count |
|----------|-------|
| **Total hApp Files** | 17 |
| **Total DNA Manifests** | 25+ |
| **Unique FL Implementations** | 5-6 |
| **Recommended Canonical FL** | 2 (0TML/mycelix_fl + 0TML/holochain/byzantine_defense) |

---

## Current Production-Ready hApps

### 1. Byzantine Defense hApp (RECOMMENDED - HDK 0.6.0)
**Location**: `0TML/holochain/`
**Status**: Production Ready (105 tests passing)
**Zomes**:
- `gradient_storage` - FL gradient storage
- `reputation_tracker` - Node reputation management
- `defense_coordinator` - RBFT v2.4 Byzantine defense
- `cbd_zome` - Causal Byzantine Detection

**Built Copies** (should consolidate to dist/):
- `0TML/holochain/dist/byzantine_defense.happ` (canonical)
- `0TML/holochain/byzantine_defense.happ`
- `0TML/holochain/happ/byzantine_defense.happ`
- `0TML/holochain/happ/byzantine_defense/byzantine_defense.happ`
- `0TML/holochain/conductor/byzantine_defense.happ`

### 2. Mycelix FL hApp (RECOMMENDED - HDK 0.6.0)
**Location**: `0TML/mycelix_fl/holochain/`
**Status**: Production Ready (18/18 tests passing, 45% BFT validated)
**Zomes**:
- `identity_integrity` + `identity_coordinator` - Agent identity
- `clustering_integrity` + `clustering_coordinator` - Byzantine clustering
- `core_integrity` + `core_coordinator` - FL round management

**Built Copies**:
- `0TML/mycelix_fl/holochain/workdir/mycelix_fl.happ` (canonical)
- `0TML/mycelix_fl/holochain/tests/fixtures/mycelix_fl.happ`
- `0TML/mycelix_fl/holochain/tests-tryorama/mycelix_fl.happ`

---

## Deprecated/Legacy FL Implementations

### 3. dnas/federated_learning
**Status**: LEGACY - Superseded by 0TML implementations
**DNA**: `mycelix_federated_learning`
**Zomes**: `fl_integrity`, `fl_coordinator`
**Note**: Uses MATL algorithm but older implementation

### 4. byzantine_fl_dna
**Status**: LEGACY
**DNA**: `byzantine_fl`
**Zomes**: `gradients`, `gradients_coordinator`
**Note**: Early Byzantine FL prototype

### 5. federated-learning/
**Status**: LEGACY
**DNA**: `fl_coordinator`
**Note**: Earlier Python-Holochain bridge experiments

### 6. h-fl-app/
**Status**: LEGACY
**Note**: Minimal hApp.yaml only

### 7. production-fl-system/
**Status**: PARTIAL - Has production infrastructure but uses older zomes
**Note**: Contains deployment scripts, API server, but zomes are outdated

---

## Other Specialized hApps

### 8. Civitas DNA
**Location**: `civitas_dna/`
**Purpose**: Causal contribution tracking and reputation
**Zomes**: `causal_contribution_zome`, `civitas_reputation_zome`
**Status**: Active - Separate purpose from FL

### 9. ZeroTrustML Credits
**Location**: `0TML/zerotrustml-dna/`
**Purpose**: Credit/token system for ZeroTrustML
**Zomes**: `credits`
**Status**: Active - Economy layer

### 10. ZeroTrustML Identity
**Location**: `0TML/zerotrustml-identity-dna/`
**Purpose**: Identity management
**Status**: Active

### 11. Mycelix Mail
**Location**: `mycelix-mail/dna/`
**Purpose**: Email/messaging integration
**Status**: Development

### 12. Mycelix Desktop Test
**Location**: `mycelix-desktop/happs/mycelix-test-app/`
**Purpose**: Desktop application testing
**Status**: Development

---

## All Zome Directories by Location

### `0TML/holochain/zomes/` (ACTIVE - Byzantine Defense)
| Zome | Purpose | HDK Version | Status |
|------|---------|-------------|--------|
| cbd_zome | Causal Byzantine Detection | 0.6.0 | Active |
| defense_coordinator | RBFT defense | 0.6.0 | Active |
| defense_common | Shared defense types | 0.6.0 | Active |
| gradient_storage | FL gradient storage | 0.6.0 | Active |
| reputation_tracker | Node reputation | 0.6.0 | Active |
| reputation_tracker_v2 | Enhanced reputation | 0.6.0 | Active |
| pogq_zome | PoGQ validation | 0.6.0 | Active |
| pogq_zome_dilithium | Post-quantum PoGQ | 0.6.0 | Active |
| zerotrustml_credits | Token system | 0.6.0 | Active |
| hyperfeel_zome | Hyperfeel integration | 0.6.0 | Active |

### `0TML/mycelix_fl/holochain/zomes/` (ACTIVE - Mycelix FL)
| Zome | Purpose | HDK Version | Status |
|------|---------|-------------|--------|
| identity/integrity | Agent identity validation | 0.7.0 (HDI) | Active |
| identity/coordinator | Agent identity management | 0.6.0 | Active |
| clustering/integrity | Byzantine clustering validation | 0.7.0 (HDI) | Active |
| clustering/coordinator | Byzantine clustering | 0.6.0 | Active |
| core/integrity | FL core validation | 0.7.0 (HDI) | Active |
| core/coordinator | FL round management | 0.6.0 | Active |

### `zomes/` (Root - Older)
| Zome | Purpose | Status |
|------|---------|--------|
| agents | Agent management | Legacy |
| bridge | Cross-DNA bridge | Legacy |
| federated_learning | Basic FL | Legacy |

### `civitas_dna/zomes/` (Separate Purpose)
| Zome | Purpose | Status |
|------|---------|--------|
| causal_contribution | Causal tracking | Active |
| causal_contribution_zome | Duplicate of above | Duplicate |
| civitas_reputation | Reputation scoring | Active |

### `dnas/` (Root DNAs - Legacy)
| Location | Purpose | Status |
|----------|---------|--------|
| bridge/ | DNA bridging | Legacy |
| federated_learning/ | FL DNA | Legacy |
| hfl/ | Holochain FL | Legacy |

---

## Duplicate Identification

### Critical Duplicates (FL Implementations)

1. **5+ FL hApp implementations** - Should consolidate to:
   - `0TML/holochain/byzantine_defense.happ` for Byzantine defense layer
   - `0TML/mycelix_fl/holochain/workdir/mycelix_fl.happ` for FL coordination

2. **Byzantine Defense hApp copies** (5 copies):
   - Keep only `dist/byzantine_defense.happ`
   - Remove others after verification

3. **reputation_tracker vs reputation_tracker_v2**:
   - v2 should replace original
   - Update references and remove v1

4. **causal_contribution vs causal_contribution_zome**:
   - Likely same code with different names
   - Consolidate

---

## Recommendations

### Immediate Actions (Status: Partially Complete)

1. **Consolidate Byzantine Defense hApp**
   - ✅ Canonical hApp: `0TML/holochain/dist/byzantine_defense.happ`
   - Other historical copies have been moved into `.archive/legacy-fl-2025-12-31/`

2. **Archive Legacy FL Implementations**
   - ✅ Completed in `.archive/legacy-fl-2025-12-31/` (see `ARCHIVE_LOG.md` there)
     - `byzantine_fl_dna/`
     - `federated-learning/`
     - `federated-learning-happ/`
     - `h-fl-app/`
     - `dnas/hfl/`
     - `dnas/` root copies of `bridge/` and `federated_learning/`

3. **Update production-fl-system**
   - ⚠️ `production-fl-system` still exists in the tree; any future production wiring should:
     - Use `0TML/mycelix_fl` zomes and the canonical hApps listed above
     - Treat all archived FL implementations strictly as historical reference

### Canonical Architecture

```
0TML/
├── holochain/                    # Byzantine Defense Layer
│   ├── zomes/                    # Active zomes (HDK 0.6.0)
│   ├── dist/                     # Built artifacts
│   │   └── byzantine_defense.happ
│   └── client/                   # TypeScript client
│
├── mycelix_fl/                   # FL Coordination Layer
│   └── holochain/
│       ├── zomes/                # FL zomes (HDK 0.6.0)
│       └── workdir/
│           └── mycelix_fl.happ
│
├── zerotrustml-dna/              # Credits/Economy
└── zerotrustml-identity-dna/     # Identity
```

### Python Backend Integration
Both hApps are already integrated via:
- `0TML/src/zerotrustml/backends/holochain_backend.py`

---

## Total Unique Zomes Count

| Category | Count |
|----------|-------|
| Byzantine Defense Zomes | 10 |
| Mycelix FL Zomes | 6 (3 integrity + 3 coordinator) |
| Civitas Zomes | 2-3 |
| ZeroTrustML Economy | 1 |
| Legacy/Deprecated | 5+ |
| **Total Active Zomes** | ~19-20 |

---

## Conclusion

**Yes, there are duplicate hApps for Mycelix FL.** The project evolved through multiple iterations:

1. **Early prototypes**: `byzantine_fl_dna/`, `federated-learning/`, `h-fl-app/`
2. **Mid-stage**: `dnas/federated_learning/`, `production-fl-system/`
3. **Current production**: `0TML/holochain/` + `0TML/mycelix_fl/`

The **canonical implementations** should be:
- `0TML/holochain/` for Byzantine defense (105 tests, HDK 0.6.0)
- `0TML/mycelix_fl/holochain/` for FL coordination (18 tests, 45% BFT validated)

Legacy implementations should be archived.
