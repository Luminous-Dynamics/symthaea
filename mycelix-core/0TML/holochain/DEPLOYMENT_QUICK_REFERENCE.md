# 🚀 Holochain Server Deployment - Quick Reference Card

**Package**: `holochain-zerotrustml-deploy-20251003-025836.tar.gz` (3.1M)
**Time**: 15-30 minutes
**Status**: Config validated ✅ Ready to deploy

---

## 📤 Step 1: Transfer to Server (2 minutes)

```bash
# From local machine:
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain
scp holochain-zerotrustml-deploy-*.tar.gz user@your-server:/tmp/
```

---

## 📥 Step 2: Extract on Server (1 minute)

```bash
# SSH to server:
ssh user@your-server

# Extract:
cd /tmp
tar xzf holochain-zerotrustml-deploy-*.tar.gz
cd deployment-package
```

---

## 🔧 Step 3: Quick Setup (2 minutes)

```bash
# Run automated setup:
./quick-setup.sh

# This creates:
# - User: holochain
# - Dirs: /opt/zerotrustml/holochain, /var/lib/holochain/zerotrustml
# - Copies config and bundles
```

---

## 🛠️ Step 4: Install Holochain (5-10 minutes)

### Option A: Via Nix (Recommended)
```bash
# Install Nix:
curl -L https://nixos.org/nix/install | sh
source ~/.nix-profile/etc/profile.d/nix.sh

# Enter Holonix:
nix develop github:holochain/holonix

# Verify:
holochain --version  # Should show: 0.5.6
```

### Option B: Via Cargo
```bash
# Install Rust:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Holochain:
cargo install holochain --version 0.5.6

# Verify:
holochain --version
```

---

## 📝 Step 5: Update Config (2 minutes)

```bash
# Edit config:
sudo nano /opt/zerotrustml/holochain/conductor-config-minimal.yaml

# Change these lines:
data_root_path: /var/lib/holochain/zerotrustml  # Was: /tmp/holochain-zerotrustml
keystore:
  lair_root: /var/lib/holochain/zerotrustml/lair  # Was: /tmp/.../lair
```

**Save**: Ctrl+X, Y, Enter

---

## 🧪 Step 6: Test Manual Start (2 minutes)

```bash
# Switch to holochain user:
sudo su - holochain
cd /opt/zerotrustml/holochain

# Test start:
./start-conductor.sh

# Should see:
# ✅ Holochain: holochain 0.5.6
# 🔧 Starting conductor...
# Conductor initialized successfully

# Stop with Ctrl+C
exit  # Back to your user
```

---

## ⚙️ Step 7: Create Systemd Service (3 minutes)

```bash
# Copy service file:
sudo cp holochain-zerotrustml.service /etc/systemd/system/

# Find holochain path:
which holochain

# Edit service file:
sudo nano /etc/systemd/system/holochain-zerotrustml.service

# Update ExecStart line with path from 'which holochain'
# Example: ExecStart=/home/user/.cargo/bin/holochain --config-path ...
```

**Save**: Ctrl+X, Y, Enter

---

## 🚀 Step 8: Start Service (1 minute)

```bash
# Reload systemd:
sudo systemctl daemon-reload

# Enable & start:
sudo systemctl enable holochain-zerotrustml
sudo systemctl start holochain-zerotrustml

# Check status:
sudo systemctl status holochain-zerotrustml
```

**Should show**: `Active: active (running)`

---

## 🔥 Step 9: Configure Firewall (1 minute)

```bash
# Allow ports:
sudo ufw allow from 10.0.0.0/8 to any port 8888  # Admin (internal)
sudo ufw allow 44444/tcp                          # App interface

# Reload:
sudo ufw reload
```

---

## ✅ Step 10: Verify Deployment (2 minutes)

```bash
# Run verification:
./verify-deployment.sh

# Should show:
# ✅ 7/7 checks passed
# 🎉 Deployment verified successfully!
```

---

## 🔍 Quick Troubleshooting

### Service won't start?
```bash
# Check logs:
sudo journalctl -u holochain-zerotrustml -n 50
```

### Port in use?
```bash
# Check what's using port 8888:
sudo lsof -i :8888
```

### Permission errors?
```bash
# Fix ownership:
sudo chown -R holochain:holochain /opt/zerotrustml/holochain
sudo chown -R holochain:holochain /var/lib/holochain/zerotrustml
```

### Can't connect?
```bash
# Check service:
sudo systemctl status holochain-zerotrustml

# Check firewall:
sudo ufw status

# Check port:
sudo ss -tlnp | grep 8888
```

---

## 📊 Health Check Commands

```bash
# Service status:
sudo systemctl status holochain-zerotrustml

# Live logs:
sudo journalctl -u holochain-zerotrustml -f

# Port listening:
sudo ss -tlnp | grep 8888

# Memory usage:
sudo systemctl show holochain-zerotrustml -p MemoryCurrent
```

---

## 🎯 Next Steps After Deployment

1. **Install hApp**: Use admin WebSocket to install Zero-TrustML hApp
2. **Connect Backend**: Update Python backend config
3. **Run Tests**: Execute integration tests
4. **Monitor**: Watch logs for errors
5. **Backup**: Backup `/var/lib/holochain/zerotrustml/`

---

## 📞 Need Help?

- **Logs**: `sudo journalctl -u holochain-zerotrustml -f`
- **Full Guide**: See `README.md` in deployment package
- **Discord**: https://discord.com/invite/bzPBzbDvWC
- **Config Solution**: See `CONFIG_SOLUTION.md`

---

## ✅ Success Criteria

- [ ] Service shows "active (running)"
- [ ] No errors in logs
- [ ] Port 8888 listening
- [ ] Can connect via websocat
- [ ] Verification script passes (7/7)

---

**When all checks pass**: Holochain is LIVE! 🎉

**Total Time**: 15-30 minutes
**Status**: Production ready
**Next**: Connect Zero-TrustML Python backend
