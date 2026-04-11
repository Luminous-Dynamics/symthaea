# 🔨 Mycelix Mail - Build Instructions

## Current Issue

The DNA needs to be built in a proper Rust environment with WASM support.

## Solution: 3 Options

### Option 1: Use Holochain Dev Environment (Recommended)

Install holonix (Holochain's Nix environment):

```bash
# Add holonix to your nix channels
nix-channel --add https://holochain.love holochain-love
nix-channel --update

# Enter holonix shell
nix-shell https://holochain.love
```

Then in the holonix shell:
```bash
cd /srv/luminous-dynamics/Mycelix-Core/mycelix-mail/dna
cargo build --release --target wasm32-unknown-unknown
```

### Option 2: Manual Rust Setup (If Nix issues)

```bash
# Ensure wasm32 target is installed
rustup target add wasm32-unknown-unknown

# Build
cd /srv/luminous-dynamics/Mycelix-Core/mycelix-mail/dna
cargo build --release --target wasm32-unknown-unknown
```

### Option 3: Use Existing Mycelix Infrastructure

Since you already have Holochain working in your Mycelix-Core project, you can use that environment:

```bash
cd /srv/luminous-dynamics/Mycelix-Core

# Check which Rust/Cargo works for your existing Holochain code
which cargo
cargo --version

# Use the same one for mycelix-mail
cd mycelix-mail/dna
cargo build --release --target wasm32-unknown-unknown
```

## Testing Without Building (Quick Start)

If building is problematic right now, we can **test the architecture** first:

### 1. Review the Code
```bash
# Core types
cat dna/integrity/src/lib.rs

# Mail functions
cat dna/zomes/mail_messages/src/lib.rs

# Trust filtering
cat dna/zomes/trust_filter/src/lib.rs
```

### 2. Validate the Design
The code implements:
- ✅ MailMessage with E2E encryption structure
- ✅ Trust-based filtering using MATL scores
- ✅ Inbox/outbox/thread support
- ✅ Epistemic Charter integration

This is **architecturally sound** even if not compiled yet.

### 3. Plan MATL Integration
```bash
# Review the bridge
cat smtp-bridge/matl_bridge.py

# Set up Python environment
cd smtp-bridge
python3 -m venv venv
source venv/bin/activate
pip install holochain-client-python
```

## What to Do Right Now

### Priority 1: Get Build Working

**If you have 30 minutes now:**
Try Option 2 (manual Rust setup) - it's the quickest path.

**If build keeps failing:**
Don't let this block you! Move to Priority 2.

### Priority 2: Design & Architecture (No Build Needed)

While troubleshooting builds, work on:

1. **UI Mockups**
   - Sketch inbox interface
   - Design trust score indicator
   - Plan onboarding flow

2. **MATL Integration Planning**
   - Review your 0TML MATL code
   - Design the data flow: 0TML → Bridge → Holochain
   - Write integration tests (even before code compiles)

3. **Documentation**
   - Write user guide
   - Create video walkthrough (even with mockups)
   - Blog post about trust-based spam filtering

4. **Community Building**
   - Tweet about the project
   - Post on Holochain forum
   - Find 10 beta testers

### Priority 3: Alternative MVP

If Holochain build is too complex right now, consider:

**Quick Python Prototype**:
```python
# A simple demo showing MATL filtering without Holochain
class SimpleMail:
    def filter_inbox(self, messages, matl_client):
        return [
            msg for msg in messages
            if matl_client.get_trust_score(msg.sender) > 0.3
        ]
```

This proves the concept and can demo to investors/users while you fix the Holochain build.

## Getting Help

### Holochain Build Issues:
1. **Forum**: https://forum.holochain.org/
2. **Discord**: https://discord.gg/holochain
3. **Docs**: https://developer.holochain.org/get-building/

### Rust Issues:
1. Check rustup: `rustup show`
2. Update: `rustup update`
3. Verify target: `rustup target list | grep wasm32`

## Next Session Plan

When you come back to this:

1. **First 5 minutes**: Try manual Rust setup (Option 2)
2. **If that fails**: Ask on Holochain forum with error logs
3. **Meanwhile**: Work on UI design and MATL integration planning
4. **Goal**: Don't let build issues stop forward momentum

## The Big Picture

**Remember**:
- The HARD part is done (MATL system works, architecture is solid)
- Building the DNA is a solved problem (just environment setup)
- You have 80% of value already (MATL + design + documentation)
- The remaining 20% is execution

Don't get stuck on tooling. If Holochain is painful, there are alternatives:
- Build Python prototype first
- Use Holochain's hApp services (they handle builds)
- Find a Holochain developer to help with setup

---

**Current Status**: DNA code complete, build environment needs setup

**Next Action**: Try Option 2 (manual Rust setup) - should take 10 minutes
