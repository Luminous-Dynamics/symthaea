# 📦 Documentation Versioning Guide

**Managing documentation versions with Mike**

---

## 🎯 Overview

The Mycelix Protocol documentation uses [mike](https://github.com/jimporter/mike) for version management. This allows users to view documentation for specific releases.

**Current Versions:**
- **v5.3** - Current stable release
- **latest** - Bleeding edge (main branch)

---

## 🚀 Quick Commands

### Deploy a New Version

```bash
# Deploy v5.3 as the latest stable version
nix develop .#docs --command mike deploy 5.3 latest stable --push

# Set as default (redirects docs.mycelix.net → v5.3)
nix develop .#docs --command mike set-default latest --push
```

### List Deployed Versions

```bash
nix develop .#docs --command mike list
```

### Delete a Version

```bash
nix develop .#docs --command mike delete 5.2 --push
```

### Preview Locally

```bash
# Serve all versions locally
nix develop .#docs --command mike serve
```

---

## 📋 Version Naming Convention

| Version Type | Format | Example | When to Use |
|--------------|--------|---------|-------------|
| **Major Release** | `X.0` | `6.0` | Breaking changes, major features |
| **Minor Release** | `X.Y` | `5.3` | New features, backwards compatible |
| **Patch Release** | `X.Y.Z` | `5.3.1` | Bug fixes only |
| **Pre-release** | `X.Y-alpha` | `6.0-alpha` | Testing new features |

**Aliases:**
- `latest` - Always points to newest stable
- `stable` - Current production version
- `dev` - Development/unreleased features

---

## 🔄 Release Workflow

### 1. Prepare Release

```bash
# Update VERSION_HISTORY.md
echo "## v5.4 ($(date +%Y-%m-%d))" >> docs/VERSION_HISTORY.md
echo "- Feature 1" >> docs/VERSION_HISTORY.md
echo "- Feature 2" >> docs/VERSION_HISTORY.md

# Commit changes
git add docs/
git commit -m "docs: Prepare v5.4 release"
git push
```

### 2. Deploy Documentation

```bash
# Deploy new version
nix develop .#docs --command mike deploy 5.4 latest --push

# Optional: Add 'stable' alias
nix develop .#docs --command mike deploy 5.4 stable --push --update-aliases

# Set as default
nix develop .#docs --command mike set-default latest --push
```

### 3. Tag Release

```bash
# Create git tag
git tag -a v5.4 -m "Release v5.4: Byzantine tolerance improvements"
git push origin v5.4
```

### 4. Verify Deployment

Visit: https://luminous-dynamics.github.io/mycelix/

Check:
- [ ] Version selector shows v5.4
- [ ] v5.4 is marked as "latest"
- [ ] Content is correct
- [ ] All links work

---

## 🏗️ GitHub Actions Integration

### Automated Deployment

The `.github/workflows/docs.yml` workflow automatically:
1. Builds documentation on every push to `main`
2. Deploys to `gh-pages` branch
3. Uses mike for versioning

### Manual Deployment

Trigger manual deployment:
```bash
# Via GitHub CLI
gh workflow run docs.yml

# Via GitHub web interface
# Actions → Documentation → Run workflow
```

---

## 🔧 Advanced Usage

### Deploy Multiple Aliases

```bash
# Deploy v6.0 as both 'latest' and '6.x'
nix develop .#docs --command mike deploy 6.0 latest 6.x --push
```

### Retitle a Version

```bash
# Change displayed title
nix develop .#docs --command mike retitle 5.3 "v5.3 (Stable)" --push
```

### Deploy Without Pushing

```bash
# Test locally before pushing
nix develop .#docs --command mike deploy 5.4 latest
mike serve  # Preview at http://localhost:8000
```

---

## 📊 Version Strategy

### When to Create New Version

**Create new version:**
- ✅ Major/minor releases (5.3 → 5.4)
- ✅ Breaking API changes
- ✅ Significant documentation restructuring
- ✅ New major features

**Don't create new version:**
- ❌ Typo fixes
- ❌ Small clarifications
- ❌ Link fixes
- ❌ CSS/styling updates

**Just update latest** for minor fixes.

---

## 🐛 Troubleshooting

### Version Not Showing

```bash
# Check deployed versions
nix develop .#docs --command mike list

# Force refresh
git fetch origin gh-pages
nix develop .#docs --command mike deploy 5.3 latest --push --force
```

### Wrong Default Version

```bash
# Reset default
nix develop .#docs --command mike set-default latest --push
```

### Clean Rebuild

```bash
# Delete all versions and start fresh
git branch -D gh-pages
git push origin --delete gh-pages

# Redeploy
nix develop .#docs --command mike deploy 5.3 latest stable --push
nix develop .#docs --command mike set-default latest --push
```

---

## 📚 Related Documentation

- **[Mike Documentation](https://github.com/jimporter/mike)** - Official mike docs
- **[MkDocs Material Versioning](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)** - Material theme setup
- **[Version History](../VERSION_HISTORY.md)** - Release notes

---

## 🎯 Quick Reference Card

```bash
# Deploy new version as latest
mike deploy <version> latest --push

# Set default version
mike set-default latest --push

# List all versions
mike list

# Delete a version
mike delete <version> --push

# Serve locally
mike serve

# Deploy with multiple aliases
mike deploy <version> alias1 alias2 --push
```

---

**Last Updated:** November 11, 2025
**Maintainer:** Luminous Dynamics Team
