# Manual DNA Installation Guide

**Date**: 2025-09-30
**Purpose**: Pragmatic path to complete Phase 7 testing
**Time Required**: 15-30 minutes

---

## ✅ Prerequisites

All infrastructure is now working:
- ✅ Conductor running with IPv4-only config
- ✅ WebSocket accepting connections on port 8888
- ✅ DNA built and ready: `zerotrustml-dna/workdir/zerotrustml.happ`

---

## 🎯 Goal

Install the Zero-TrustML DNA so we can test the 4 core zome functions:
1. `create_credit()` - Write to DHT
2. `get_credit()` - Read from DHT
3. `get_credits_for_holder()` - Query DHT
4. `get_balance()` - Calculate totals

---

## 📋 Installation Methods (Choose One)

### Method 1: Using `hc` CLI (Recommended)

```bash
# Navigate to project directory
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Start conductor with IPv4 config (in separate terminal)
holochain --config-path conductor-ipv4-only.yaml \
  --piped \
  -p /tmp/holochain_zerotrustml

# Wait for "Conductor ready." message

# Install app using hc CLI
hc app install \
  --app-id zerotrustml-app \
  --path zerotrustml-dna/workdir/zerotrustml.happ \
  --admin-port 8888

# Expected output:
# App installed successfully!
# App ID: zerotrustml-app
# Installed Cell IDs: [...]
```

**Note the Cell ID** - you'll need this for testing zome calls.

### Method 2: Using Holochain Launcher (GUI)

If `hc` CLI doesn't work:

1. Install Holochain Launcher from: https://github.com/holochain/launcher/releases
2. Launch the GUI application
3. Click "Install hApp"
4. Browse to: `zerotrustml-dna/workdir/zerotrustml.happ`
5. Set App ID: `zerotrustml-app`
6. Click "Install"
7. Note the Cell ID shown in the UI

### Method 3: Direct Admin API (If Above Fail)

```bash
# Using Python with websockets
python3 test_install_dna.py

# This will:
# 1. Connect to ws://localhost:8888
# 2. Send InstallApp admin request
# 3. Wait for response
# 4. Display installed Cell ID
```

**Note**: If this times out (only Pings received), it means admin API automation needs `holochain_client` crate (future work). Use Method 1 or 2 instead.

---

## 🔍 Verification

After installation, verify the app is installed:

```bash
# List installed apps
hc app list --admin-port 8888

# Expected output:
# zerotrustml-app (installed)
#   - Cell: [cell-id]
```

---

## 📝 Record Installation Details

Create a file with your installation details:

```bash
cat > INSTALLED_APP_INFO.txt << EOF
Installation Date: $(date)
App ID: zerotrustml-app
Cell ID: [paste-cell-id-here]
Admin Port: 8888
Conductor Config: conductor-ipv4-only.yaml
EOF
```

---

## 🧪 Next: Test Zome Functions

Once DNA is installed, proceed to test all zome functions:

```bash
# Test the bridge with installed app
python3 test_zome_calls.py
```

This will test:
1. ✅ Connection to installed app
2. ✅ create_credit() - DHT write
3. ✅ get_credit() - DHT read
4. ✅ get_credits_for_holder() - DHT query
5. ✅ get_balance() - Calculation

---

## ⚠️ Troubleshooting

### "App already installed" error
```bash
# Uninstall first
hc app uninstall --app-id zerotrustml-app --admin-port 8888

# Then install again
hc app install --app-id zerotrustml-app --path zerotrustml-dna/workdir/zerotrustml.happ --admin-port 8888
```

### "Connection refused" error
Ensure conductor is running:
```bash
# Check if conductor is running
ps aux | grep holochain

# If not, start it:
holochain --config-path conductor-ipv4-only.yaml --piped -p /tmp/holochain_zerotrustml
```

### "Command not found: hc"
Install holochain CLI tools:
```bash
nix-env -iA nixpkgs.holochain
```

### Admin API timeouts
If `hc` CLI fails and admin API times out, this is expected (requires `holochain_client` crate for automation). Use Holochain Launcher GUI (Method 2) instead.

---

## 📊 Success Criteria

Phase 7 is COMPLETE when:
- ✅ DNA installed on conductor (via any method)
- ✅ `create_credit()` tested - writes to DHT
- ✅ `get_credit()` tested - retrieves from DHT
- ✅ `get_credits_for_holder()` tested - queries work
- ✅ `get_balance()` tested - calculations correct
- ✅ Action hashes verified as real (not mock)

**Admin automation is NOT required for Phase 7 completion!**

---

## 🔮 Future: Production Automation

When production automation is needed:

1. Add `holochain_client` crate to Cargo.toml
2. Use `AdminWebsocket` from official library
3. Refactor installation to use official API
4. Test automated workflow
5. Document for production deployment

See `PHASE_7_FINAL_RECOMMENDATION.md` for complete future work plan.

---

**Status**: Manual installation documented
**Next**: Install DNA and test zome functions
**Timeline**: 15-30 minutes to completion

🌊 Pragmatic shipping wins over perfect automation!
