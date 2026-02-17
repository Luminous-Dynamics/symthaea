# 🚀 Holochain Production Server Deployment - Step-by-Step Guide

**Date**: October 3, 2025
**Status**: Ready to deploy - Config validated ✅
**Estimated Time**: 15-30 minutes

---

## 📋 Pre-Deployment Checklist

### Server Requirements ✅
- [ ] Linux server with SSH access
- [ ] Sudo/root privileges
- [ ] At least 2GB RAM
- [ ] 10GB free disk space
- [ ] Ports available: 8888 (admin), 44444 (app)
- [ ] Internet connectivity

### Files to Transfer
- [ ] `conductor-config-minimal.yaml` - Validated config
- [ ] `start-conductor.sh` - Startup script
- [ ] `dna/zerotrustml.dna` - DNA bundle (1.6M)
- [ ] `happ/zerotrustml.happ` - hApp bundle (1.6M)
- [ ] `zomes/` - Compiled WASM zomes (optional, already in bundles)

---

## 🔧 Step 1: Copy Files to Server

### Option A: Using SCP (Secure Copy)

```bash
# From your local machine (this directory):
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Create deployment archive
tar czf holochain-deploy.tar.gz holochain/

# Copy to server (replace with your server details)
scp holochain-deploy.tar.gz user@your-server:/tmp/

# SSH into server
ssh user@your-server

# Extract on server
cd /tmp
tar xzf holochain-deploy.tar.gz
sudo mv holochain /opt/zerotrustml/
```

### Option B: Using rsync (Recommended for Updates)

```bash
# From local machine:
rsync -avz --progress holochain/ user@your-server:/opt/zerotrustml/holochain/
```

### Option C: Git Clone (If repo is pushed)

```bash
# On server:
cd /opt
sudo git clone https://github.com/your-org/zerotrustml.git
cd zerotrustml/holochain
```

---

## 🔧 Step 2: Install Holochain on Server

### Option A: Via Nix (Recommended)

```bash
# On server - Install Nix if not present
curl -L https://nixos.org/nix/install | sh
source ~/.nix-profile/etc/profile.d/nix.sh

# Enter Holonix development environment
nix develop github:holochain/holonix

# Verify installation
holochain --version  # Should show: holochain 0.5.6
hc --version         # Should show: hc 0.5.6
lair-keystore --version  # Should show: lair-keystore 0.6.2
```

### Option B: Via Cargo (Alternative)

```bash
# On server - Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Holochain tools
cargo install holochain --version 0.5.6
cargo install holochain_cli --version 0.5.6
cargo install lair_keystore --version 0.6.2

# Verify installation
holochain --version
```

---

## 🔧 Step 3: Prepare Server Environment

```bash
# Create system user for Holochain (recommended for production)
sudo useradd -r -s /bin/bash -d /var/lib/holochain holochain

# Create directories
sudo mkdir -p /opt/zerotrustml/holochain
sudo mkdir -p /var/lib/holochain/zerotrustml
sudo mkdir -p /var/log/holochain

# Set permissions
sudo chown -R holochain:holochain /opt/zerotrustml/holochain
sudo chown -R holochain:holochain /var/lib/holochain
sudo chown -R holochain:holochain /var/log/holochain

# Copy files to production location
sudo cp /tmp/holochain/* /opt/zerotrustml/holochain/ -r
sudo chown -R holochain:holochain /opt/zerotrustml/holochain
```

---

## 🔧 Step 4: Configure for Production

### Update Config for Production Paths

```bash
# Edit the config file
sudo -u holochain nano /opt/zerotrustml/holochain/conductor-config-minimal.yaml
```

Change these lines:
```yaml
# Before:
data_root_path: /tmp/holochain-zerotrustml
keystore:
  type: lair_server_in_proc
  lair_root: /tmp/holochain-zerotrustml/lair

# After (production):
data_root_path: /var/lib/holochain/zerotrustml
keystore:
  type: lair_server_in_proc
  lair_root: /var/lib/holochain/zerotrustml/lair
```

For production security, also change:
```yaml
# Before:
allowed_origins: '*'

# After (restrict to your backend):
allowed_origins: 'zerotrustml-backend'
```

Save and exit (Ctrl+X, Y, Enter)

---

## 🔧 Step 5: Test Manual Startup

```bash
# Switch to holochain user
sudo su - holochain

# Navigate to holochain directory
cd /opt/zerotrustml/holochain

# Make startup script executable
chmod +x start-conductor.sh

# Test startup
./start-conductor.sh
```

**Expected Output**:
```
🚀 Starting Holochain Conductor
================================

✅ Environment: /var/lib/holochain/zerotrustml
✅ Config: conductor-config-minimal.yaml
✅ Holochain: holochain 0.5.6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 Starting conductor...
   Config: conductor-config-minimal.yaml
   Admin Port: 8888 (WebSocket)
   Data Path: /var/lib/holochain/zerotrustml

   Press Ctrl+C to stop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conductor initialized successfully
Listening on 0.0.0.0:8888
```

**If successful**: Press Ctrl+C to stop, then proceed to systemd setup.

**If errors**: See Troubleshooting section below.

---

## 🔧 Step 6: Create Systemd Service (Production)

```bash
# Exit holochain user back to your admin user
exit

# Create systemd service file
sudo nano /etc/systemd/system/holochain-zerotrustml.service
```

**Paste this content**:
```ini
[Unit]
Description=Holochain Zero-TrustML Conductor
Documentation=https://developer.holochain.org/
After=network.target

[Service]
Type=simple
User=holochain
Group=holochain
WorkingDirectory=/opt/zerotrustml/holochain

# Start command
ExecStart=/home/holochain/.cargo/bin/holochain --config-path /opt/zerotrustml/holochain/conductor-config-minimal.yaml

# Restart policy
Restart=always
RestartSec=10
StartLimitBurst=5
StartLimitInterval=300

# Logging
StandardOutput=append:/var/log/holochain/conductor.log
StandardError=append:/var/log/holochain/conductor.error.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/holochain/zerotrustml
ReadWritePaths=/var/log/holochain

# Resource limits
LimitNOFILE=65535
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

**Save and exit** (Ctrl+X, Y, Enter)

**Note**: If you used Nix to install Holochain, change the ExecStart path:
```ini
ExecStart=/nix/store/.../bin/holochain --config-path /opt/zerotrustml/holochain/conductor-config-minimal.yaml
```

Find the correct path with: `which holochain`

---

## 🔧 Step 7: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable holochain-zerotrustml

# Start service
sudo systemctl start holochain-zerotrustml

# Check status
sudo systemctl status holochain-zerotrustml
```

**Expected output**:
```
● holochain-zerotrustml.service - Holochain Zero-TrustML Conductor
     Loaded: loaded (/etc/systemd/system/holochain-zerotrustml.service; enabled)
     Active: active (running) since Thu 2025-10-03 10:00:00 UTC
   Main PID: 12345 (holochain)
      Tasks: 12 (limit: 4915)
     Memory: 256M
     CGroup: /system.slice/holochain-zerotrustml.service
             └─12345 /usr/bin/holochain --config-path ...

Oct 03 10:00:00 server systemd[1]: Started Holochain Zero-TrustML Conductor.
Oct 03 10:00:01 server holochain[12345]: Conductor initialized successfully
Oct 03 10:00:01 server holochain[12345]: Listening on 0.0.0.0:8888
```

---

## 🔧 Step 8: Configure Firewall

```bash
# Allow admin interface (internal only!)
sudo ufw allow from 10.0.0.0/8 to any port 8888 comment 'Holochain Admin'

# OR if you need external access (less secure):
# sudo ufw allow 8888/tcp

# Allow app interface
sudo ufw allow 44444/tcp comment 'Holochain App Interface'

# Reload firewall
sudo ufw reload

# Verify rules
sudo ufw status numbered
```

---

## ✅ Step 9: Verify Deployment

### Check Conductor is Running

```bash
# Check systemd status
sudo systemctl status holochain-zerotrustml

# Check logs
sudo journalctl -u holochain-zerotrustml -f

# Check listening ports
sudo ss -tlnp | grep 8888
```

### Test WebSocket Connection

```bash
# Install websocat for testing
cargo install websocat

# Connect to admin interface
websocat ws://localhost:8888
```

Then send (paste and press Enter):
```json
{"type":"list_apps","data":{}}
```

**Expected response**:
```json
{"type":"response","data":[]}
```

Press Ctrl+C to exit websocat.

### Install hApp Bundle

```bash
# Using websocat
websocat ws://localhost:8888
```

Send (all on one line):
```json
{"type":"install_app","data":{"agent_key":"<your-agent-key>","installed_app_id":"zerotrustml","membrane_proofs":{},"network_seed":null,"path":"/opt/zerotrustml/holochain/happ/zerotrustml.happ"}}
```

---

## 🔍 Troubleshooting

### Service Won't Start

```bash
# Check detailed logs
sudo journalctl -u holochain-zerotrustml -n 100 --no-pager

# Check file permissions
ls -la /opt/zerotrustml/holochain/
ls -la /var/lib/holochain/zerotrustml/

# Verify config syntax
holochain --config-path /opt/zerotrustml/holochain/conductor-config-minimal.yaml --dry-run
```

### Port Already in Use

```bash
# Check what's using port 8888
sudo lsof -i :8888

# Kill the process if needed
sudo kill -9 <PID>

# Or change port in config file
```

### Permission Errors

```bash
# Fix ownership
sudo chown -R holochain:holochain /opt/zerotrustml/holochain
sudo chown -R holochain:holochain /var/lib/holochain/zerotrustml

# Fix permissions
sudo chmod 755 /opt/zerotrustml/holochain
sudo chmod 644 /opt/zerotrustml/holochain/*.yaml
sudo chmod 755 /opt/zerotrustml/holochain/*.sh
```

### Connection Refused

```bash
# Check firewall
sudo ufw status

# Check service is running
sudo systemctl status holochain-zerotrustml

# Check logs for binding errors
sudo journalctl -u holochain-zerotrustml | grep -i "bind\|error"
```

---

## 🔧 Useful Management Commands

### Service Management
```bash
# Start
sudo systemctl start holochain-zerotrustml

# Stop
sudo systemctl stop holochain-zerotrustml

# Restart
sudo systemctl restart holochain-zerotrustml

# Status
sudo systemctl status holochain-zerotrustml

# Enable auto-start
sudo systemctl enable holochain-zerotrustml

# Disable auto-start
sudo systemctl disable holochain-zerotrustml
```

### Logs
```bash
# Follow live logs
sudo journalctl -u holochain-zerotrustml -f

# Last 100 lines
sudo journalctl -u holochain-zerotrustml -n 100

# Logs since last boot
sudo journalctl -u holochain-zerotrustml -b

# Logs for specific time
sudo journalctl -u holochain-zerotrustml --since "2025-10-03 10:00:00"
```

### Monitoring
```bash
# Resource usage
sudo systemctl status holochain-zerotrustml

# Detailed process info
ps aux | grep holochain

# Memory usage
sudo systemctl show holochain-zerotrustml -p MemoryCurrent

# Network connections
sudo ss -tlnp | grep holochain
```

---

## 📊 Health Check Checklist

- [ ] Service status shows "active (running)"
- [ ] No errors in `journalctl -u holochain-zerotrustml`
- [ ] Port 8888 is listening (`ss -tlnp | grep 8888`)
- [ ] Can connect via websocat to ws://localhost:8888
- [ ] hApp bundles in `/opt/zerotrustml/holochain/happ/`
- [ ] Firewall rules configured
- [ ] Auto-start enabled (`systemctl is-enabled holochain-zerotrustml`)

---

## 🎯 Next Steps After Deployment

1. **Install hApp**: Load the Zero-TrustML hApp bundle
2. **Configure Backend**: Update Python backend to connect to conductor
3. **Run Integration Tests**: Verify gradient storage/retrieval
4. **Monitor Performance**: Watch logs and resource usage
5. **Set Up Backups**: Backup `/var/lib/holochain/zerotrustml/`

---

## 📞 Support Resources

- **Logs**: `/var/log/holochain/conductor.log`
- **Config**: `/opt/zerotrustml/holochain/conductor-config-minimal.yaml`
- **Data**: `/var/lib/holochain/zerotrustml/`
- **Holochain Discord**: https://discord.com/invite/bzPBzbDvWC
- **Forum**: https://forum.holochain.org/

---

## ✅ Deployment Complete!

Once the health check passes, your Holochain conductor is **LIVE** and ready for Zero-TrustML integration! 🎉

**Status**: ✅ Holochain Backend OPERATIONAL
**Next**: Connect Python backend and run integration tests
