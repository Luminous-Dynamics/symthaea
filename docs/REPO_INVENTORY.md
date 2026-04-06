# Luminous Dynamics Repository Inventory

*Generated March 2, 2026*

## Critical Issues

### 1. Monorepo Remote Misnamed
The working directory at `/srv/luminous-dynamics/` (1,113 commits, the org's monorepo) has its git remote pointed at `Luminous-Dynamics/kosmic-lab.git`. This is wrong — kosmic-lab is a separate Python research project. All monorepo pushes have been overwriting the kosmic-lab GitHub repo since ~Nov 2025.

### 2. Standalone Symthaea Repo Stale
`phi-lab/` (local, gitignored) is the actual Symthaea development repo (126 commits, Dec 2025 – Feb 2026). Its remote points to `Luminous-Dynamics/symthaea.git`. The GitHub `symthaea` repo was last pushed Feb 3 and is missing moral_algebra, moral_topology, ethics_engine, and most sub-crates that were developed in the monorepo since then.

### 3. Kosmic-Lab Has 33 Unpushed Commits
The original kosmic-lab Python project at `kosmic-lab/` (gitignored) has 33 local commits that have never reached GitHub because the monorepo hijacked its remote.

---

## Local Git Repos (46 found)

### Tier 1: The Monorepo

| Path | Remote | Commits | First | Last | Branch | Unpushed | Notes |
|------|--------|---------|-------|------|--------|----------|-------|
| `.` (root) | `kosmic-lab.git` | 1,113 | 2025-11-12 | 2026-03-02 | main | 7 | **WRONG REMOTE** — this is the org monorepo, not kosmic-lab |

**Tracked directories** (part of the monorepo): `symthaea/`, `mycelix-commons/`, `mycelix-civic/`, `mycelix-core/`, `mycelix-workspace/`, `crates/`, `docs/`, `_websites/`, `_infrastructure/`, `services/`, all `00-` through `12-` harmony dirs, etc.

**Gitignored directories** (separate repos living inside): `kosmic-lab/`, `phi-lab/`, `terra-atlas-mvp/`

---

### Tier 2: Core Project Repos (Gitignored, Independent)

| Path | Remote | Commits | First | Last | Unpushed | Size | Status |
|------|--------|---------|-------|------|----------|------|--------|
| `kosmic-lab/` | `kosmic-lab.git` | 93 | 2025-11-12 | 2026-02-03 | **33** | 25 GB | Python K-index research. **33 unpushed commits.** Remote hijacked by monorepo. |
| `phi-lab/` | `symthaea.git` | 126 | 2025-12-16 | 2026-02-05 | 1 | 204 MB | This IS the standalone Symthaea repo (renamed from sophia-hlb). 1 unpushed. |
| `terra-atlas-mvp/` | `terra-atlas.git` | 56 | 2025-09-17 | 2026-02-20 | 2 | 859 MB | Terra Atlas energy platform. 2 unpushed. |

---

### Tier 3: Submodules (Properly Declared)

| Path | Remote | Commits | First | Last | Size |
|------|--------|---------|-------|------|------|
| `11-meta-consciousness/luminous-nix/` | `luminous-nix.git` | 362 | 2025-07-25 | 2026-02-16 | 7.6 GB |
| `mycelix-health/` (via `repos/mycelix-health/`) | `mycelix-health.git` | 22 | 2026-01-22 | 2026-01-23 | — |

---

### Tier 4: Earliest Projects (Jun-Jul 2025 origins, nested with own .git)

| Path | Remote | Commits | First | Last | Unpushed | Status |
|------|--------|---------|-------|------|----------|--------|
| `03-integral-wisdom/core/codex` | `codex-of-relational-harmonics.git` | 82 | **2025-06-28** | 2025-08-01 | 2 | Oldest LD project. GH repo archived. |
| `07-evolutionary-progression/core/luminous-os` | `luminous-os.git` | 20 | **2025-07-05** | 2025-08-01 | 4 | Rust OS. GH repo archived. **4 unpushed.** |
| `01-resonant-coherence/core/the-weave` | `the-weave.git` | 7 | **2025-07-05** | 2025-07-19 | 2 | Multi-agent coordination. GH archived. |
| `services/the-weave` | `the-weave.git` | 7 | 2025-07-05 | 2025-07-19 | 2 | **DUPLICATE** of above |
| `01-resonant-coherence/core/sacred-core` | `sacred-core.git` | 5 | **2025-07-07** | 2025-07-13 | 1 | Sacred Core. GH archived. |
| `services/sacred-core` | `sacred-core.git` | 5 | 2025-07-07 | 2025-07-13 | 1 | **DUPLICATE** of above |

---

### Tier 5: Mycelix Nested Repos

| Path | Remote | Commits | First | Last | Status |
|------|--------|---------|-------|------|--------|
| `mycelix-core/mycelix-mail/` | `Mycleix-Mail.git` (typo!) | 3 | 2025-11-13 | 2025-11-13 | Dead. Typo in remote name. |
| `mycelix-core/mycelix-fl-pure-p2p/` | `mycelix-fl-pure-p2p.git` | 1 | 2025-09-29 | 2025-09-29 | Dead. 1 commit. |
| `mycelix-core/_websites/mycelix.net-pogq/` | `mycelix.net.git` | 1 | 2025-10-14 | 2025-10-14 | Dead duplicate of mycelix.net. |
| `mycelix-workspace/mycelix-v6-living/` | NO REMOTE | 19 | 2026-02-04 | 2026-02-06 | Local only. No remote. |
| `mycelix-workspace/observatory/build/` | `mycelix-observatory.git` | 1 | 2026-02-16 | 2026-02-16 | Build artifact. |
| `mycelix.net/` | `mycelix.net.git` | 2 | 2025-10-14 | 2025-12-18 | Landing page. |

---

### Tier 6: Website Repos

| Path | Remote | Commits | First | Last | Unpushed | Status |
|------|--------|---------|-------|------|----------|--------|
| `_websites/evolvingresonantcocreationism.com/` | `evolving-resonant-cocreationism.git` | 5 | 2025-09-07 | 2025-09-21 | 0 | ERC website |
| `_websites/evolving-resonant/evolving-resonant-cocreationism/` | same as above | 13 | 2025-09-07 | 2025-09-12 | 0 | **DUPLICATE** — older copy |
| `_websites/infin.love/` | `Tristan-Stoltz-ERC/infin-love.git` | 5 | 2025-09-11 | 2025-09-12 | 0 | Gift circles site. **Wrong org.** |
| `_websites/luminousdynamics-io/` | `luminousdynamics-io.git` | 1 | 2025-09-17 | 2025-09-17 | 0 | Dev portal |
| `_websites/luminousdynamics.org/` | `Tristan-Stoltz-ERC/luminousdynamics-org.git` | 15 | 2025-09-11 | 2025-09-12 | 0 | Main org site. **Wrong org.** |
| `_websites/mycelix.net/` | `mycelix.git` (not mycelix.net!) | 6 | 2025-09-13 | 2025-09-22 | **6** | All 6 commits unpushed. |
| `_websites/mycelix.net/mycelix-github-repo/` | `mycelix.git` | 1 | 2025-09-21 | 2025-09-21 | 0 | Dead nested duplicate |
| `_websites/nixforhumanity.org/` | `Tristan-Stoltz-ERC/nixforhumanity-org.git` | 5 | 2025-09-11 | 2025-09-12 | 0 | Nix site. **Wrong org.** |
| `_websites/relationalharmonics.org/` | `Tristan-Stoltz-ERC/relationalharmonics-org.git` | 10 | 2025-09-11 | 2025-09-12 | 0 | RH site. **Wrong org.** |
| `_websites/terra-lumina/terra-lumina/` | `terra-lumina.git` | 15 | 2025-08-28 | 2025-09-12 | 0 | Terra Lumina site |
| `_websites/terra-lumina/terra-lumina-standalone/` | NO REMOTE | 1 | 2025-08-28 | 2025-08-28 | — | Dead. No remote, 1 commit. |
| `_websites/terra-lumina/.../website-nextjs/` (x2) | `Tristan-Stoltz-ERC/resonantia-earth.git` | 5 | 2025-08-22 | 2025-08-22 | 0 | Dead NextJS artifacts. **Wrong org.** Both duplicates. |
| `_websites/luminous-dynamics-website-old/` | `Tristan-Stoltz-ERC/luminous-dynamics-website.git` | 0 | — | — | 0 | **Empty repo.** |

---

### Tier 7: Third-Party Clones (Data/Benchmarks)

| Path | Remote | Notes |
|------|--------|-------|
| `symthaea/data/meditation-eeg/` | OpenNeuroDatasets/ds001787 | EEG dataset |
| `symthaea/data/ds003751/` | OpenNeuroDatasets/ds003751 | Neuro dataset |
| `symthaea/data/benchmarks/arc/repo/` | fchollet/ARC-AGI | ARC benchmark |
| `symthaea/data/benchmarks/ethics/repo/` | hendrycks/ethics | Ethics benchmark |
| `symthaea/data/benchmarks/pyphi/repo/` | wmayner/pyphi | PyPhi |
| `symthaea/datasets/ethics/raw/bbq/` | nyu-mll/BBQ | Bias benchmark |
| `symthaea/datasets/ethics/raw/ethics/` | hendrycks/ethics | Duplicate of above |
| `symthaea/benchmarks/external/helm/helm-repo/` | stanford-crfm/helm | HELM benchmark |
| `kosmic-lab/.../cleanrl/` | vwxyzjn/cleanrl | RL framework |
| `mycelix-core/.../holochain-src/` | holochain/holochain | Holochain source |
| `mycelix-core/.../holonix/` | holochain/holonix | Holochain nix |
| `11-meta-consciousness/luminous-nix/sapient-hrm/` | sapientinc/HRM | HRM model |

---

## GitHub Repos (42 public)

### Active Development (pushed in 2026)

| Repo | Last Push | What It Actually Contains |
|------|-----------|--------------------------|
| `kosmic-lab` | 2026-03-02 | **THE ORG MONOREPO** (misnamed). Contains everything. |
| `symthaea` | 2026-02-03 | Stale Symthaea snapshot. Missing key files. |
| `luminous-dynamics` | 2026-02-02 | Older monorepo predecessor. |
| `luminous-nix` | 2026-02-16 | Active. Submodule of monorepo. |
| `Mycelix-Core` | 2026-02-16 | Legacy pre-cluster Mycelix. |
| `mycelix-observatory` | 2026-02-16 | Dashboard. |
| `mycelix-health` | 2026-02-15 | Active. Submodule of monorepo. |
| `mycelix-desci` | 2026-02-15 | DeSci platform. |
| `Mycelix-Mail` | 2026-02-15 | Decentralized email. |
| `terra-atlas` | 2026-02-09 | Energy platform. |
| `mycelix-praxis` | 2026-02-09 | Education network. |
| `Mycelix-Marketplace` | 2026-02-06 | P2P marketplace. |
| `.github` | 2026-02-01 | Org profile. |
| `mycelix-supplychain` | 2026-02-01 | Supply chain. |
| `mycelix-space` | 2026-01-27 | Space situational awareness. |

### Dormant (not pushed since 2025)

| Repo | Last Push | Notes |
|------|-----------|-------|
| `Mycelix-Music` | 2025-12-30 | |
| `mycelix.net` | 2025-12-18 | |
| `historical-k-index` | 2025-12-16 | Published paper data |
| `nsfw` | 2025-11-18 | Nix Subsystem for Windows |
| `mycelix-prototype` | 2025-09-21 | C++ prototype |
| `webpilot` | 2025-10-03 | |
| `terra-lumina` | 2025-11-18 | |
| `luminousdynamics-io` | 2025-11-18 | |
| `infin-love` | 2025-11-18 | |
| `evolving-resonant-cocreationism` | 2025-11-18 | |
| `nixforhumanity-org` | 2025-11-18 | |
| `docs` | 2025-11-18 | |
| `nixite` | 2025-11-17 | |
| `erc` | 2025-09-17 | |
| `luminousdynamics-org` | 2025-09-17 | |
| `relationalharmonics` | 2025-09-17 | |
| `mycelix-protocol-core` | 2025-11-18 | **EMPTY** |

### Archived on GitHub

| Repo | Notes |
|------|-------|
| `Mycelix-Core-archived` | Has `target/` committed |
| `relational-harmonics-website` | Single file |
| `luminous-dynamics-website` | Single file |
| `sacred-infrastructure` | Deployment configs |
| `luminous-os` | Rust OS |
| `the-weave` | Multi-agent |
| `sacred-core` | Sacred practice |
| `codex-of-relational-harmonics` | Has `node_modules/` committed |

---

## Summary of Problems

### Critical
1. **Monorepo remote points to `kosmic-lab.git`** — needs rename or new repo
2. **Standalone `symthaea` repo is stale** — missing key files claimed in the blog post
3. **33 unpushed kosmic-lab commits** — original Python research at risk (local only)

### Moderate
4. **phi-lab has 1 unpushed commit** and is the real symthaea dev repo
5. **terra-atlas-mvp has 2 unpushed commits**
6. **luminous-os has 4 unpushed commits** (archived on GH but local has more)
7. **6 website repos point to `Tristan-Stoltz-ERC` org** instead of `Luminous-Dynamics`
8. **mycelix-core/mycelix-mail remote has typo** (`Mycleix-Mail`)
9. **`_websites/mycelix.net/` has 6 unpushed commits** and wrong remote name (`mycelix.git` vs `mycelix.net.git`)
10. **`mycelix-protocol-core`** on GitHub is completely empty
11. **`mycelix-v6-living`** has 19 commits with no remote (local only, at risk)

### Cleanup
12. **Duplicate local repos**: sacred-core (2x), the-weave (2x), evolving-resonant-cocreationism (2x), mycelix.net (3x), resonantia-earth website (2x)
13. **Dead nested repos**: mycelix-fl-pure-p2p (1 commit), mycelix.net-pogq (1 commit), terra-lumina-standalone (1 commit, no remote), luminous-dynamics-website-old (0 commits)
14. **Committed artifacts on GitHub**: `node_modules/` in codex-of-relational-harmonics, `target/` in Mycelix-Core-archived
15. **Duplicate third-party clones**: hendrycks/ethics exists at 2 paths

### Disk Space (excluding .git)
| Directory | Size |
|-----------|------|
| `symthaea/` | 295 GB |
| `mycelix-workspace/` | 42 GB |
| `mycelix-core/` | 29 GB |
| `kosmic-lab/` | 25 GB |
| `mycelix-commons/` | 20 GB |
| `_websites/` | 11 GB |
| `mycelix-civic/` | 9.9 GB |
| `luminous-nix/` | 7.6 GB |
| `terra-atlas-mvp/` | 859 MB |
| **Total workspace** | **~440+ GB** |
