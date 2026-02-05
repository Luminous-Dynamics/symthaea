# Systemd

Systemd service files for running Symthaea as a system service.

## Services

- `symthaea.service` - Main consciousness service
- `symthaea-api.service` - API server
- `symthaea-dashboard.service` - Web dashboard

## Installation

```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable --now symthaea.service
```

## Management

```bash
# Status
sudo systemctl status symthaea

# Logs
journalctl -u symthaea -f

# Restart
sudo systemctl restart symthaea
```

## Configuration

Edit service files to customize:
- Working directory
- Environment variables
- Resource limits
