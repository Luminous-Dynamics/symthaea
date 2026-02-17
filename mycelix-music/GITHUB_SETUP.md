# 🚀 GitHub Repository Setup Guide

This guide will walk you through creating a new GitHub repository for Mycelix Music.

## ✅ Prerequisites Completed

- ✅ Git repository initialized
- ✅ Initial commit created (72 files, 36K+ lines)
- ✅ Documentation organized in `docs/` folder
- ✅ `.gitignore` configured
- ✅ `README.md` written
- ✅ `CONTRIBUTING.md` created
- ✅ `LICENSE` added (MIT)

---

## 📝 Step 1: Create GitHub Repository

### Option A: Via GitHub Website

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `mycelix-music`
   - **Description**: `Decentralized music streaming platform where artists earn 10x-50x more`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Option B: Via GitHub CLI (if installed)

```bash
gh repo create mycelix-music \
  --public \
  --description "Decentralized music streaming platform where artists earn 10x-50x more" \
  --source=. \
  --remote=origin \
  --push
```

---

## 🔗 Step 2: Connect Local Repository to GitHub

After creating the repo on GitHub, you'll see a page with setup instructions. Use the "push an existing repository" section:

```bash
# Add GitHub as remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/mycelix-music.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/mycelix-music.git

# Verify remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

---

## 🎨 Step 3: Configure Repository Settings

### Repository Settings

1. Go to your repo on GitHub
2. Click "Settings" tab

### General Settings
- ✅ **Features**: Enable Issues, Wikis, Discussions
- ✅ **Pull Requests**: Enable "Automatically delete head branches"

### Branch Protection (Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Include administrators

### Topics/Tags
Add these topics for discoverability:
```
blockchain, music, decentralized, holochain, web3,
streaming, gnosis-chain, smart-contracts, nextjs, typescript
```

### About Section
- **Description**: Decentralized music streaming platform where artists earn 10x-50x more
- **Website**: https://mycelix.net (or your demo URL)
- **Topics**: (added above)

---

## 📋 Step 4: Create Initial Issues (Optional)

Create issues for Phase 7 milestones:

### Issue Template

```markdown
**Title**: [Phase 7] Backend API Implementation

**Description**:
Implement RESTful API backend with PostgreSQL database.

**Tasks**:
- [ ] Set up Express.js server
- [ ] Design database schema
- [ ] Implement authentication middleware
- [ ] Create CRUD endpoints for songs
- [ ] Add user profile endpoints
- [ ] Set up Supabase integration

**Labels**: enhancement, phase-7, backend
**Assignees**: (assign yourself)
**Projects**: Phase 7 - Backend Integration
```

Create similar issues for:
- Holochain 0.6.0 Integration
- CDN Setup (Bunny CDN)
- File Upload Implementation
- Dashboard API Integration

---

## 🏷️ Step 5: Set Up Labels

Create these custom labels:

| Label | Color | Description |
|-------|-------|-------------|
| `phase-1` | `#0E8A16` | Foundation phase |
| `phase-2` | `#1D76DB` | Smart contracts |
| `phase-3` | `#5319E7` | Frontend prototype |
| `phase-7` | `#D4C5F9` | Backend integration |
| `frontend` | `#C5DEF5` | Frontend code |
| `backend` | `#BFD4F2` | Backend/API code |
| `smart-contracts` | `#FBCA04` | Solidity contracts |
| `documentation` | `#0075CA` | Documentation |
| `holochain` | `#D93F0B` | Holochain integration |
| `enhancement` | `#84B6EB` | New feature |
| `bug` | `#D73A4A` | Something isn't working |
| `good first issue` | `#7057FF` | Good for newcomers |

---

## 📊 Step 6: Create GitHub Project Board (Optional)

1. Go to Projects tab → New project
2. Choose "Board" template
3. Name it "Mycelix Music Roadmap"
4. Create columns:
   - 📋 Backlog
   - 🚧 Phase 7: In Progress
   - ✅ Completed
   - 🔮 Future Phases

5. Add issues to appropriate columns

---

## 🔐 Step 7: Set Up Secrets (for CI/CD later)

When ready for automated deployments, add these secrets:

1. Go to Settings → Secrets and variables → Actions
2. Add repository secrets:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `BUNNY_CDN_API_KEY`
   - `PRIVATE_KEY` (for contract deployments)

**Note**: Don't add these yet, we'll set them up during Phase 7 implementation.

---

## 🤝 Step 8: Invite Collaborators (Optional)

If working with a team:

1. Go to Settings → Collaborators
2. Click "Add people"
3. Enter GitHub usernames or emails
4. Set appropriate permissions:
   - **Admin**: Full access
   - **Write**: Can push to repo
   - **Read**: Read-only access

---

## 📄 Step 9: Add Additional GitHub Files

### Create `.github/` Folder

```bash
mkdir -p .github/{ISSUE_TEMPLATE,workflows}
```

### Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. macOS]
 - Browser: [e.g. Chrome]
 - Version: [e.g. v1.0.0]
```

Create `.github/ISSUE_TEMPLATE/feature_request.md`:

```markdown
---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution**
What you want to happen.

**Additional context**
Any other context or screenshots.
```

### Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

---

## 🚀 Step 10: Push GitHub Config Files

```bash
# Create and push GitHub config
git add .github/
git commit -m "chore: add GitHub issue templates and PR template"
git push origin main
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Repository is visible on GitHub
- [ ] README.md displays correctly
- [ ] License is recognized (MIT)
- [ ] Topics/tags are added
- [ ] Branch protection is enabled (if desired)
- [ ] Issues can be created
- [ ] Pull requests can be opened
- [ ] GitHub Actions (if configured) are working

---

## 🎉 You're All Set!

Your Mycelix Music repository is now on GitHub!

### Next Steps

1. **Share the repo**: Post on social media, Discord, etc.
2. **Add collaborators**: Invite team members
3. **Start Phase 7**: Begin backend implementation
4. **Set up CI/CD**: Add GitHub Actions for automated testing

### Useful GitHub URLs

- **Repository**: https://github.com/YOUR_USERNAME/mycelix-music
- **Issues**: https://github.com/YOUR_USERNAME/mycelix-music/issues
- **Projects**: https://github.com/YOUR_USERNAME/mycelix-music/projects
- **Wiki**: https://github.com/YOUR_USERNAME/mycelix-music/wiki

---

## 🆘 Troubleshooting

### "Remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/mycelix-music.git
```

### Permission denied (SSH)
Make sure you've added your SSH key to GitHub:
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys

### Push rejected
```bash
# If GitHub has commits you don't have locally
git pull origin main --rebase
git push origin main
```

---

**Need help?** Join our Discord or email dev@mycelix.net

🎵 **Let's build in public!** 🎵
