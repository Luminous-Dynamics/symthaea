# Repository Registry & Visibility Plan

Last updated: 2026-03-06

## Current State

- **GitHub Org**: Luminous-Dynamics
- **Total Repos**: 44 (ALL PUBLIC)
- **Monorepo**: `luminous-dynamics` (514 MB on GitHub, ~337 GB locally with build artifacts)
- **License**: NONE (legally ambiguous — defaults to all-rights-reserved but GitHub ToS allows forking)
- **Already Archived on GitHub**: 17 repos

---

## I. Monorepo Top-Level Directories

### Core Projects (KEEP — these are active)

| Directory | What | Files | Sensitivity | Notes |
|-----------|------|-------|-------------|-------|
| `symthaea/` | Holographic Liquid Brain — main crate + 46 sub-crates | 20,274 | MIXED (see §III) | 985K LOC Rust, 3,958 tests |
| `symthaea/symthaea-core/` | Core: HDC, Phi, LTC, consciousness math | ~274K LOC | SAFE | Foundation crate |
| `mycelix-commons/` | Holochain cluster: property, housing, care, mutualaid, water, food, transport | 134 | SAFE | 35 zomes, 5,276 tests |
| `mycelix-civic/` | Holochain cluster: justice, emergency, media | 74 | SAFE | 16 zomes, 2,273 tests |
| `mycelix-hearth/` | Holochain cluster: kinship, gratitude, care, autonomy, decisions | 234 | SAFE | 12 zomes, 1,023 tests |
| `mycelix-identity/` | DID registry, MFA, trust credentials, recovery | 906 | SAFE | 9 zomes, fully implemented |
| `mycelix-governance/` | Proposals, voting, DKG threshold-signing, councils | 697 | SAFE | 7 zomes, fully implemented |
| `mycelix-personal/` | Identity vault, health vault, credential wallet | 18 | SAFE | 4 zomes |
| `mycelix-attribution/` | Dependency registry, usage receipts, reciprocity | 89 | SAFE | 3 zomes |
| `mycelix-workspace/` | Unified hApp workspace, deploy configs, justfile | 1,253 | SAFE | Build orchestration |
| `terra-atlas-mvp/` | Energy infrastructure intelligence platform | 2,022 | SAFE | Live at atlas.luminousdynamics.io |
| `kosmic-lab/` | K-index consciousness simulation framework | 58,759 | SAFE | Python research code |
| `phi-lab/` | Phi/IIT experimental code | 1,488 | SAFE | Research |

### Mycelix Satellites (mixed activity)

| Directory | Files | Status | Notes |
|-----------|-------|--------|-------|
| `mycelix-core/` | 24,288 | Stale — superseded by cluster architecture | Old monolithic approach |
| `Mycelix-Core/` | 0 | Empty/submodule | Duplicate name |
| `mycelix-health/` | 1,976 | Active (submodule) | Differential privacy |
| `mycelix-energy/` | 865 | Stale | Energy domain |
| `mycelix-finance/` | 876 | Stale | Finance domain |
| `mycelix-knowledge/` | 1,000 | Stale | Knowledge graph |
| `mycelix-climate/` | 13 | Stub | Barely started |
| `mycelix-desci/` | 113 | Stale | Decentralized science |
| `mycelix-edunet/` | 4,776 | Active | Education network |
| `mycelix-mail/` | 511 | Archived | Decentralized email |
| `mycelix-marketplace/` | 99 | Archived | P2P marketplace |
| `mycelix-music/` | 823 | Archived | Music streaming |
| `mycelix-space/` | 272 | Stub | Space awareness |
| `mycelix-supplychain/` | 1,551 | Archived | Supply chain |
| `mycelix.net/` | 0 | Website only | |

### Philosophy / Eight Harmonies (low-sensitivity, low-value for portfolio)

| Directory | What |
|-----------|------|
| `00-sacred-foundation/` | Philosophy docs |
| `01-resonant-coherence/` | Philosophy + some code |
| `02-pan-sentient-flourishing/` | Philosophy docs |
| `03-integral-wisdom/` | Codex ceremonies, genesis JS |
| `05-universal-interconnectedness/` | Philosophy docs |
| `06-sacred-reciprocity/` | Sacred economics sandbox, digital vivarium |
| `07-evolutionary-progression/` | Philosophy docs |

### Infrastructure / Tooling

| Directory | What | Keep? |
|-----------|------|-------|
| `08-infrastructure/` | Infra configs | Yes |
| `09-archives/` | Old archives | No — move out |
| `11-meta-consciousness/` | luminous-nix submodule + consciousness research | Yes |
| `_infrastructure/` | NixOS configs, scripts | Yes |
| `_development/` | Testing, web-automation | Maybe |
| `_websites/` | 10 website source dirs | Yes |
| `configs/` | Config files | Yes |
| `scripts/` | Shell scripts | Yes |
| `services/` | Service definitions | Yes |
| `shared/` | Shared middleware | Yes |
| `nix/` | Nix overlays | Yes |

### Should NOT be in repo

| Directory | Why |
|-----------|-----|
| `patents/` | IP filings (NOW GITIGNORED) |
| `venv/` | Python virtualenv — should be gitignored |
| `__pyphi_cache__/` | PyPhi cache — should be gitignored |
| `logs/` | Runtime logs — should be gitignored |
| `archive/` | Old archives — review then gitignore or delete |
| `nvim/` | Editor config — shouldn't be in project repo |
| `symthaea-core.stale-v0.1/` | Dead code — delete |
| `terra-lumina/` | Empty — delete or merge into terra-atlas |
| `spark-engine/` | 20 files, unclear purpose — review |

---

## II. GitHub Org Repos (44 total)

### Tier 1: Keep Public (active, portfolio-worthy)

| Repo | Why |
|------|-----|
| `luminous-dynamics` | Main monorepo (BUT see §IV for visibility change) |
| `terra-atlas` | Live product, energy platform |
| `mycelix-v6-living` | Active Holochain development |
| `mycelix-observatory` | Ecosystem monitoring |
| `mycelix-health` | Active, differential privacy |
| `.github` | Org profile |

### Tier 2: Keep Public (reference / niche value)

| Repo | Why |
|------|-----|
| `luminous-nix` | NixOS natural language interface — good portfolio piece |
| `kosmic-lab-research` | Academic research code |
| `historical-k-index` | Published paper data |
| `nsfw` | Nix Subsystem for Windows — niche but clever name, decent tool |
| `nixite` | NixOS visual package discovery |
| `webpilot` | Web automation tool |
| `luminousdynamics-io` | Developer portal |
| `terra-lumina` | Business docs |
| `mycelix.net` | Website |
| `nixforhumanity-org` | Website |
| `infin-love` | Philosophical |
| `evolving-resonant-cocreationism` | Philosophical library |

### Tier 3: Already Archived (17 repos — leave as-is)

| Repo | Status |
|------|--------|
| `kosmic-lab` | Archived — superseded by kosmic-lab-research |
| `Mycelix-Core` | Archived — superseded by monorepo |
| `mycelix-desci` | Archived |
| `Mycelix-Mail` | Archived |
| `mycelix-edunet` | Archived |
| `mycelix-supplychain` | Archived |
| `Mycelix-Marketplace` | Archived |
| `Mycelix-Music` | Archived |
| `mycelix-prototype` | Archived |
| `codex-of-relational-harmonics` | Archived |
| `luminous-os` | Archived |
| `sacred-infrastructure` | Archived |
| `sacred-core` | Archived |
| `the-weave` | Archived |
| `relational-harmonics-website` | Archived |
| `luminous-dynamics-website` | Archived |
| `Mycelix-Mail-client` | Archived |
| `Mycelix-Mail-hApp` | Archived |

### Tier 4: Should Archive or Make Private

| Repo | Action | Why |
|------|--------|-----|
| `Mycelix-Core-archived` | Archive | 346 MB, superseded, name says it all |
| `mycelix-protocol-core` | Archive or delete | Empty (0 KB) |
| `symthaea` | Archive | Mirror only — active dev in monorepo |
| `mycelix-space` | Archive | Stub, 162 KB |
| `docs` | Make private or merge | Stale docs hub |
| `luminousdynamics-org` | Keep public | Main site (tiny, 12 KB) |
| `relationalharmonics` | Archive | Already archived on GH |
| `erc` | Archive | Already archived on GH |

---

## III. Symthaea Sub-Crate Sensitivity Classification

### SENSITIVE (2 crates) — should be private or removed from public

| Crate | LOC | Why |
|-------|-----|-----|
| `symthaea-nuclear-forensics` | 1,375 | Nuclear material attribution, isotope modeling, IAEA safeguards. Academic/forensic but optically bad in public repo for dual-citizen in SA |
| `symthaea-physics` | 8,421 | Tokamak plasma control, Alcator C-Mod adapter. Fusion research (non-weapons) but "nuclear" + "control" = bad optics |

### MODERATE (6 crates) — fine in public, but review before sharing

| Crate | LOC | Why |
|-------|-----|-----|
| `symthaea-flight` | 10,107 | Quadrotor flight control (MuJoCo simulation only) |
| `symthaea-humanoid` | 8,208 | Bipedal humanoid control (DMC simulation only) |
| `symthaea-vehicle` | 7,190 | Autonomous vehicle control (bicycle model sim only) |
| `symthaea-hal` | 5,704 | Hardware abstraction layer (PCA9685 servo, I2C) |
| `symthaea-fabrication-kernel` | 4,571 | 3D geometry / CAD primitives |
| `symthaea-nix` | 26,841 | NixOS system management |

### SAFE (38 crates) — no concerns

All remaining crates: pure math, consciousness theory, cognitive science, ML algorithms, benchmarks.

---

## IV. Recommended Visibility Plan

### CRITICAL: Patent Filing vs Public Disclosure

5 Tier-1 IDDs are complete (2,602 lines, 72 claims) but UNFILED.
The monorepo is PUBLIC. This means:
- Anyone can see the implementations right now
- Under US law, you have a 1-year grace period from your own public disclosure
- Under international law (PCT), there is NO grace period — public disclosure = prior art
- **The code has been public since at least Feb 2026** — the international clock may already be ticking
- Making the repo private NOW does not undo prior public disclosure, but it stops the clock from getting worse

**Priority order: (1) Make repo private, (2) File provisionals, (3) Then worry about license/showcase**

### Step 1: License (TODAY)

Adopt AGPL-3.0-or-later at the repo root (with documented commercial licensing exceptions):
- `LICENSE` (AGPL-3.0-or-later)
- `COMMERCIAL_LICENSE.md` (commercial exception policy / dual-licensing posture)

### Step 2: Make monorepo private (TODAY)

```bash
gh repo edit Luminous-Dynamics/luminous-dynamics --visibility private
gh repo edit Luminous-Dynamics/symthaea --visibility private
```

### Step 3: Clean up .gitignore (TODAY)

Add to .gitignore:
- `venv/`  (if not already)
- `__pyphi_cache__/`
- `logs/`
- `nvim/`

### Step 4: Archive stale repos (TODAY)

```bash
for repo in Mycelix-Core-archived mycelix-protocol-core symthaea mycelix-space; do
  gh repo archive Luminous-Dynamics/$repo --yes
done
```

### Step 5: Create public showcase (THIS WEEK)

New repo: `Luminous-Dynamics/overview` (public)
Contents:
- Whitepaper-style README with architecture diagrams
- Links to public repos (Mycelix clusters, Terra Atlas)
- "Private repos available for review on request" for Symthaea
- Test/LOC/crate metrics as proof of scale

### Step 6: Decide on sensitive crates (THIS WEEK)

Options:
a) Move `symthaea-nuclear-forensics` and `symthaea-physics` to separate private repo
b) Just make monorepo private (Step 2 handles this)
c) Delete them if not actively needed

If monorepo goes private, option (b) is sufficient.

---

## V. What Stays Public

### Step 7: Patent Provisionals (ASAP — before papers)

5 Tier-1 IDDs ready to file ($1,600 for all 5 provisionals, self-file):

| ID | Invention | Claims | File Before |
|----|-----------|--------|-------------|
| P-001 | HDC-LTC Unified Neuron | 15 | Liquid AI publishes similar |
| P-002 | Moral Algebra | 15 | HAI paper submission |
| P-003 | LTC Vocal Tract Synthesis | 15 | Any TTS demo |
| P-004 | Consciousness Equation V2 | 13 | PLoS paper submission |
| P-005 | Consciousness-Aware FL | 14 | Mycelix production deploy |

See `patents/PATENT_REGISTRY.md` for full details (Tier 2: 7 more, Tier 3: 6 more).

---

After executing the plan:

| Public Repo | Purpose |
|-------------|---------|
| `Luminous-Dynamics/overview` | Landing page / whitepaper |
| `terra-atlas` | Live energy platform |
| `mycelix-v6-living` | Active Holochain dev |
| `mycelix-observatory` | Monitoring dashboard |
| `mycelix-health` | Healthcare hApp |
| `luminous-nix` | NixOS natural language |
| `kosmic-lab-research` | Research framework |
| `historical-k-index` | Published paper |
| `.github` | Org profile |
| Various websites | luminousdynamics-io, nixforhumanity-org, etc. |
| 17 archived repos | Read-only historical reference |
