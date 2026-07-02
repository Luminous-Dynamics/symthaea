# 🚀 Mycelix Desktop - Quick Start Guide

**Last Updated**: September 30, 2025
**Status**: Day 1 Complete ✅

---

## 🎯 What You Have Now

✅ **Working Tauri v2 Desktop Application**
- Frontend: http://localhost:1420 (SolidJS + Vite)
- Backend: Rust binary running with 5 commands
- Environment: NixOS with all dependencies
- Holochain: v0.5.6 available and ready

---

## ⚡ Quick Commands

### View the Application
```bash
# Open in browser (frontend only)
firefox http://localhost:1420

# OR Chrome/Chromium
chromium http://localhost:1420
```

### Check Status
```bash
# See what's running
ps aux | grep -E "(mycelix|vite)"

# Check frontend server
curl http://localhost:1420
```

### Development Commands
```bash
# Enter Nix environment (for all dev work)
nix develop

# Inside nix develop:
npm run dev              # Frontend only (Vite)
npm run tauri dev        # Full app (requires GDK_BACKEND=x11)

# Recommended: Force X11 backend
GDK_BACKEND=x11 npm run tauri dev
```

### Test Holochain
```bash
# Must be inside nix develop
nix develop

# Then test Holochain
holochain --version      # Should show 0.5.6
which holochain          # Should show path
```

---

## 📖 Documentation Quick Links

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **TESTING.md** | Test all features | Before testing IPC |
| **SESSION_SUMMARY.md** | Full session recap | To understand what was built |
| **CURRENT_STATUS.md** | Real-time status | For current state |
| **ADR-001-technology-stack.md** | Tech decisions | To understand architecture |
| **MYCELIX_DESKTOP_ROADMAP.md** | 6-month plan | For long-term vision |
| **QUICKSTART.md** | This guide | Quick reference |

---

## 🧪 Testing the Application

### Step 1: Open Browser
Navigate to: http://localhost:1420

### Step 2: Test Greet Command
1. Enter your name in the input field
2. Click "Greet" button
3. **Expected**: "Hello, [name]! You've been greeted from Rust!"

### Step 3: Test Holochain Button
1. Click "Start Holochain" button
2. **Expected**: "Holochain starting... (placeholder)"
3. **Note**: This is a placeholder - will be implemented Day 2

### Step 4: Test Network Button
1. Click "Connect to Network" button
2. **Expected**: "Connecting to Mycelix network... (placeholder)"
3. **Note**: This is a placeholder - will be implemented Day 2-3

---

## 🐛 Common Issues

### "Connection refused" on port 1420
**Solution**: Start the frontend server
```bash
npm run dev
# OR if Tauri is running, it includes frontend
```

### "Command not found: holochain"
**Solution**: Enter nix develop first
```bash
nix develop
holochain --version  # Now works
```

### "GBM buffer creation failed"
**Expected**: This is normal in remote/server environments
**Impact**: Window may not be visible
**Workaround**: Use browser to access http://localhost:1420

### Window doesn't appear
**Solution 1**: Force X11 backend
```bash
GDK_BACKEND=x11 npm run tauri dev
```

**Solution 2**: Access frontend directly
```bash
# Frontend works even if window doesn't render
firefox http://localhost:1420
```

---

## 📁 Important Files

### Application Code
- `src/App.tsx` - Main UI component (109 lines)
- `src-tauri/src/main.rs` - Backend with 5 commands (64 lines)
- `src-tauri/tauri.conf.json` - Tauri configuration

### Configuration
- `flake.nix` - Nix environment (78 lines)
- `package.json` - npm dependencies
- `Cargo.toml` - Rust workspace

### Build Outputs
- `target/debug/mycelix-desktop` - Compiled binary (171 MB)
- `src-tauri/icons/` - RGBA format icons (5 files)

---

## 🚀 Next Steps (Day 2)

### Morning (2-3 hours)
1. **Create Holochain config** (~30 min)
   ```bash
   nix develop
   # Create conductor-config.yaml
   ```

2. **Implement real conductor start** (~1 hour)
   - Edit `src-tauri/src/main.rs`
   - Modify `start_holochain` command
   - Use `std::process::Command` to launch conductor

3. **Create test DNA** (~1 hour)
   ```bash
   hc scaffold web-app mycelix-test
   hc dna pack mycelix-test/dnas/test
   ```

### Afternoon (2-3 hours)
4. **Test integration** (~30 min)
   - Start conductor from Tauri
   - Verify it runs successfully
   - Check logs for errors

5. **Implement P2P connection** (~1-2 hours)
   - Modify `connect_to_network` command
   - Test two instances connecting
   - Verify data sync works

---

## 💡 Pro Tips

### Faster Development
```bash
# Keep Vite running in one terminal
npm run dev

# Run Tauri commands in another
GDK_BACKEND=x11 npm run tauri dev
```

### Check Logs
```bash
# Frontend logs (browser console)
# Backend logs
tail -f /tmp/tauri-direct.log

# Or check specific build logs
tail -f /tmp/tauri-x11.log
```

### Clean Rebuild
```bash
# If things get weird, clean and rebuild
cargo clean
npm run tauri dev
```

---

## 📊 What's Working

### ✅ Fully Functional
- Nix development environment
- Frontend server (Vite)
- Backend binary (Rust + Tauri)
- Icon generation
- Security hardening (libsoup v3)
- Documentation (6 comprehensive docs)

### ⏳ Partially Implemented
- IPC commands (work but return placeholders)
- Holochain integration (available but not connected)

### 📅 Not Yet Started
- Real Holochain conductor start
- P2P network connection
- Data synchronization

---

## 🎓 Learning Resources

### Tauri
- Official Docs: https://tauri.app/
- v2 Guide: https://v2.tauri.app/start/

### SolidJS
- Official Docs: https://www.solidjs.com/
- Tutorial: https://www.solidjs.com/tutorial/

### Holochain
- Official Docs: https://developer.holochain.org/
- Getting Started: https://developer.holochain.org/get-started/

### NixOS
- Official Manual: https://nixos.org/manual/nixos/stable/
- Flakes: https://nixos.wiki/wiki/Flakes

---

## 🆘 Getting Help

### Check Documentation First
1. Read `TESTING.md` for test procedures
2. Check `CURRENT_STATUS.md` for known issues
3. Review `SESSION_SUMMARY.md` for details

### Debug Commands
```bash
# Check what's running
ps aux | grep mycelix

# Check frontend server
curl http://localhost:1420

# Check Holochain
nix develop --command holochain --version

# View logs
tail -f /tmp/*.log
```

### Common Solutions
- **Port already in use**: Kill existing process
- **Missing dependencies**: Run `nix develop`
- **Build fails**: Clean and rebuild with `cargo clean`
- **Window doesn't appear**: Use browser at localhost:1420

---

## ✅ Success Checklist

### Before Starting Day 2
- [ ] Frontend accessible at http://localhost:1420
- [ ] Backend binary running (check with `ps aux | grep mycelix`)
- [ ] Holochain available (`nix develop` → `holochain --version`)
- [ ] All documentation read
- [ ] Testing guide reviewed

### Day 2 Goals
- [ ] Create Holochain conductor config
- [ ] Implement real conductor start
- [ ] Create and install test DNA
- [ ] Test conductor runs successfully
- [ ] Document integration patterns

---

## 🎉 You've Built Something Amazing!

In just 2 hours, you now have:
- ✅ Modern desktop application (Tauri v2)
- ✅ Reactive frontend (SolidJS)
- ✅ Robust backend (Rust)
- ✅ P2P foundation ready (Holochain)
- ✅ Reproducible environment (NixOS)
- ✅ Professional documentation

**Next**: Transform placeholder commands into real Holochain integration!

---

*Quick Reference Version 1.0*
*Last Updated: 2025-09-30 9:25 AM Central*

🍄 **Happy Hacking!** 🍄