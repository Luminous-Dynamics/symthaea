# 🎬 Zero-TrustML Grant Demo System

**Professional, reproducible demonstrations for Ethereum Foundation and Holochain grants**

**Status**: ✅ Ready for testing (flake errors fixed)
**Testing**: See [TESTING_LOG.md](TESTING_LOG.md) for verification details

---

## 🚀 Quick Start

**⚠️ First Time**: Downloads ~7-8GB of packages, takes 10-30 minutes

```bash
# 1. Enter Nix environment (installs everything)
cd /srv/luminous-dynamics/Mycelix-Core/0TML/demos
nix develop  # First run: 10-30 min download

# 2. Generate visualizations
python visualizations/network_topology.py
python benchmarks/run_benchmarks.py

# 3. View outputs
ls -lh outputs/
```

**You now have**:
- ✅ Interactive network topology (outputs/interactive/)
- ✅ Benchmark results (benchmarks/results/)
- ✅ Ready for grant proposals!

## 📚 Documentation

- **[GRANT_DEMO_ARCHITECTURE.md](GRANT_DEMO_ARCHITECTURE.md)** - Complete system design
- **[DEMO_SYSTEM_COMPLETE.md](DEMO_SYSTEM_COMPLETE.md)** - User guide and next steps
- **[TESTING_LOG.md](TESTING_LOG.md)** - Testing verification and fixes
