# Quick Start Guide

Get up and running with Mycelix Praxis in under 10 minutes!

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Git** (2.x or later)
- **Rust** (1.75 or later) - [Install Rust](https://rustup.rs/)
- **Node.js** (20.x or later) - [Install Node.js](https://nodejs.org/)
- **Make** (optional, but recommended)

**Check your installation:**
```bash
git --version
rustc --version
node --version
npm --version
make --version
```

---

## Step 1: Clone the Repository (30 seconds)

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git

# Navigate into the directory
cd mycelix-praxis
```

---

## Step 2: Build the Project (2-3 minutes)

### Option A: Using Makefile (Recommended)

```bash
# Build everything (Rust workspace + web app)
make build
```

### Option B: Manual Build

```bash
# Build Rust workspace
cargo build --workspace

# Build web application
cd apps/web
npm install
npm run build
cd ../..
```

**Expected output:**
- ✅ Rust workspace compiles without errors
- ✅ 49 tests passing
- ✅ Web app builds successfully

---

## Step 3: Run Tests (1 minute)

```bash
# Run all tests
make test

# Or individually:
cargo test --workspace      # Rust tests
cd apps/web && npm test     # Web tests (when added)
```

**You should see:**
```
running 49 tests
test result: ok. 49 passed; 0 failed; 0 ignored
```

---

## Step 4: Start the Development Environment (30 seconds)

### Start Web App

```bash
cd apps/web
npm run dev
```

The web app will start at **http://localhost:5173**

### Using Docker (Alternative)

```bash
# Build and start all services
docker-compose up --build

# App will be available at http://localhost:3000
```

---

## Step 5: Explore the Application (3 minutes)

Open your browser to **http://localhost:5173** (or **:3000** if using Docker)

### 🏠 Home Page
- Overview of Praxis features
- Navigation to different sections

### 📚 Courses Page (`/courses`)
- Browse 6 example courses
- Filter by difficulty (Beginner/Intermediate/Advanced)
- Search courses by name or tags
- Click any course to see details

**Try this:**
1. Search for "Rust"
2. Filter by "Advanced" difficulty
3. Click on "Applied Cryptography" to see the modal

### 🤝 FL Rounds Page (`/rounds`)
- View federated learning rounds
- See 6-phase timeline (DISCOVER → RELEASE)
- Check round status and participants

**Try this:**
1. Click on an ACTIVE round
2. Observe the timeline visualization
3. Check provenance metrics (if completed)

### 🎓 Credentials Page (`/credentials`)
- View W3C Verifiable Credentials
- Verify credentials
- Download/copy JSON

**Try this:**
1. Click on any credential card
2. Click "Verify" to check authenticity
3. Click "Copy JSON" to get the raw VC

---

## Step 6: Run Benchmarks (1 minute)

Test aggregation performance:

```bash
make bench

# Or directly:
cargo bench --workspace
```

**Output:**
- Performance metrics for `trimmed_mean`, `median`, `weighted_mean`
- HTML report in `target/criterion/report/index.html`

---

## Step 7: Explore the Code (3 minutes)

### Key Directories

```
mycelix-praxis/
├── crates/              # Rust libraries
│   ├── praxis-core/    # Core types, crypto, provenance
│   └── praxis-agg/     # Robust aggregation algorithms
├── zomes/              # Holochain zome scaffolds
│   ├── learning_zome/  # Course management
│   ├── fl_zome/        # Federated learning
│   ├── credential_zome/# W3C VCs
│   └── dao_zome/       # Governance
├── apps/web/           # React web client
├── tests/fixtures/     # Test data generators
├── docs/               # Documentation
└── examples/           # Example data (courses, rounds, credentials)
```

### Important Files

- **`docs/protocol.md`** - FL protocol specification
- **`docs/architecture.md`** - System architecture (with diagrams!)
- **`docs/faq.md`** - Frequently asked questions
- **`ROADMAP.md`** - Project roadmap and milestones

---

## Step 8: Make Your First Change (3 minutes)

Let's add a new course to the web app!

### Create a New Course File

Create `examples/courses/your-course.json`:

```json
{
  "course_id": "your-course-id",
  "title": "Your Amazing Course",
  "description": "Learn amazing things!",
  "instructor": "Your Name",
  "syllabus": {
    "modules": [
      {
        "module_id": "1",
        "title": "Introduction",
        "description": "Getting started",
        "duration_hours": 2
      }
    ]
  },
  "tags": ["beginner", "tutorial"],
  "difficulty": "beginner",
  "enrollment_count": 0
}
```

### Add to Mock Data

Edit `apps/web/src/data/mockCourses.ts`:

```typescript
import yourCourse from '../../../examples/courses/your-course.json';

export const mockCourses: Course[] = [
  courseExample1,
  courseExample2,
  yourCourse,  // Add this line
  // ... other courses
];
```

### See Your Changes

```bash
# Restart the dev server (Ctrl+C, then npm run dev)
# Navigate to /courses
# Your course should appear!
```

---

## Step 9: Run Quality Checks (1 minute)

Before committing changes:

```bash
# Quick sanity check
make check

# Or individually:
cargo fmt --all --check  # Check formatting
cargo clippy --all       # Lint Rust code
cargo test --all         # Run tests
```

**All checks should pass** ✅

---

## Step 10: Next Steps

Congratulations! You're now set up with Mycelix Praxis! 🎉

### 🚀 What's Next?

**Want to contribute?**
1. Read [CONTRIBUTING.md](../../CONTRIBUTING.md)
2. Check [Good First Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/labels/good%20first%20issue)
3. Pick an issue and comment that you're working on it

**Want to learn more?**
- 📖 [FL Protocol Deep Dive](fl-protocol-deep-dive.md) (coming soon)
- 📖 [Architecture Guide](../architecture.md)
- 📖 [FAQ](../faq.md)

**Want to deploy?**
- 🚢 [Deployment Guide](../deployment.md)
- 🐳 [Docker Setup](../../docker-compose.yml)

**Want to chat?**
- 💬 [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- 🐛 [Report Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)

---

## Troubleshooting

### Rust Build Fails

```bash
# Update Rust toolchain
rustup update stable

# Clean and rebuild
cargo clean
cargo build --workspace
```

### Web App Won't Start

```bash
# Clear node_modules and reinstall
cd apps/web
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Tests Fail

```bash
# Make sure you're on the latest commit
git pull origin main

# Rebuild everything
make clean
make build
make test
```

### Docker Issues

```bash
# Clean up Docker resources
docker-compose down -v
docker system prune -f

# Rebuild from scratch
docker-compose up --build
```

---

## Summary

In this guide, you:
- ✅ Cloned and built the project
- ✅ Ran 49 passing tests
- ✅ Explored the web UI
- ✅ Ran performance benchmarks
- ✅ Made your first code change
- ✅ Verified code quality

**Total Time**: ~10 minutes

---

## Feedback

Found an issue with this guide? Have suggestions?
[Open an issue](https://github.com/Luminous-Dynamics/mycelix-praxis/issues) or submit a PR!

---

*Last updated: 2025-11-15*
