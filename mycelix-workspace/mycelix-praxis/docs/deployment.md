# Deployment Guide

Complete guide for deploying Mycelix Praxis to production environments.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [VPS Deployment](#vps-deployment)
- [Cloud Platforms](#cloud-platforms)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying to production:

- ✅ All tests passing (`make test`)
- ✅ Security audit completed (`cargo audit`)
- ✅ Environment variables configured (`.env`)
- ✅ SSL/TLS certificates ready (for HTTPS)
- ✅ Domain name configured (DNS)

---

## Deployment Options

| Option | Difficulty | Cost | Best For |
|--------|-----------|------|----------|
| [Docker Compose](#docker-deployment) | Easy | Low | Small deployments, testing |
| [VPS (Manual)](#vps-deployment) | Medium | Low-Medium | Full control, custom setups |
| [Cloud Platforms](#cloud-platforms) | Easy-Medium | Medium-High | Scalability, managed services |
| Kubernetes | Hard | High | Large scale, high availability |

---

## Docker Deployment

### Quick Deploy (Recommended for Testing)

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git
cd mycelix-praxis

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Build and start
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f web
```

**Access**:
- Web App: `http://localhost:3000`
- Health Check: `http://localhost:3000/health`

### Production Deploy

```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Enable auto-restart
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate
```

### Behind Reverse Proxy (nginx/Traefik)

**nginx example** (`/etc/nginx/sites-available/praxis`):

```nginx
server {
    listen 80;
    server_name praxis.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name praxis.example.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/praxis.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/praxis.example.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Proxy to Docker container
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**Enable and restart nginx**:
```bash
sudo ln -s /etc/nginx/sites-available/praxis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## VPS Deployment

### Step 1: Provision VPS

**Recommended specs** (for web app only):
- CPU: 2 cores
- RAM: 2 GB
- Storage: 20 GB SSD
- OS: Ubuntu 22.04 LTS

**Providers**:
- DigitalOcean ($12/month)
- Linode ($12/month)
- Vultr ($12/month)
- Hetzner ($5-10/month, EU)

### Step 2: Initial Server Setup

```bash
# SSH into server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Create non-root user
adduser praxis
usermod -aG sudo praxis

# Configure firewall
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# Switch to praxis user
su - praxis
```

### Step 3: Install Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

### Step 4: Deploy Application

```bash
# Clone repository
cd ~
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git
cd mycelix-praxis

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Start services
docker compose up -d

# Check logs
docker compose logs -f
```

### Step 5: Configure nginx (Optional but Recommended)

```bash
# Install nginx
sudo apt install nginx -y

# Install certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d praxis.example.com

# Configure nginx (see nginx example above)
sudo nano /etc/nginx/sites-available/praxis

# Enable site
sudo ln -s /etc/nginx/sites-available/praxis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 6: Setup Systemd Service (Auto-start)

Create `/etc/systemd/system/praxis.service`:

```ini
[Unit]
Description=Mycelix Praxis
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/praxis/mycelix-praxis
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=praxis

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable praxis
sudo systemctl start praxis
sudo systemctl status praxis
```

---

## Cloud Platforms

### AWS (Elastic Beanstalk)

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p docker praxis-app

# Create environment
eb create praxis-prod

# Deploy
eb deploy

# Open in browser
eb open
```

### Google Cloud Platform (Cloud Run)

```bash
# Build and push Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/praxis

# Deploy to Cloud Run
gcloud run deploy praxis \
  --image gcr.io/PROJECT_ID/praxis \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create praxis-app

# Set buildpack
heroku buildpacks:set heroku/nodejs

# Deploy
git push heroku main

# Open app
heroku open
```

### Render

1. Connect your GitHub repository
2. Create new "Web Service"
3. Select Docker environment
4. Deploy automatically on push

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch app
fly launch

# Deploy
fly deploy

# Open app
fly open
```

---

## Configuration

### Environment Variables

**Required**:
- `NODE_ENV` - Set to `production`
- `WEB_PORT` - Port for web server (default: 3000)

**Optional**:
- `PUBLIC_URL` - Public URL of deployment
- `RUST_LOG` - Log level (info, warn, error)
- `CORS_ALLOWED_ORIGINS` - Allowed CORS origins

**Example `.env`**:
```bash
NODE_ENV=production
WEB_PORT=3000
PUBLIC_URL=https://praxis.example.com
RUST_LOG=warn
VITE_MOCK_MODE=true
```

### Security Checklist

- [ ] Change default passwords
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set secure headers (CSP, HSTS)
- [ ] Enable rate limiting
- [ ] Keep dependencies updated
- [ ] Regular security audits (`cargo audit`)
- [ ] Monitor error logs
- [ ] Backup data regularly

---

## Monitoring

### Health Checks

**Endpoint**: `GET /health`

**Expected response**:
```
HTTP/1.1 200 OK
Content-Type: text/plain

healthy
```

**Monitoring tools**:
- UptimeRobot (free)
- Pingdom
- StatusCake
- New Relic
- Datadog

### Logs

**Docker logs**:
```bash
docker compose logs -f web
docker compose logs --tail=100 web
```

**System logs** (if using systemd):
```bash
sudo journalctl -u praxis -f
```

### Metrics

**Resource usage**:
```bash
docker stats
```

**Disk usage**:
```bash
df -h
docker system df
```

---

## Maintenance

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker compose down
docker compose up -d --build

# Clean up old images
docker system prune -a
```

### Backups

```bash
# Backup volumes (when using Holochain conductor)
docker run --rm -v conductor_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/conductor_data_$(date +%Y%m%d).tar.gz /data

# Backup configuration
tar czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml
```

### Restore

```bash
# Restore data
docker run --rm -v conductor_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/conductor_data_20250101.tar.gz -C /
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs web

# Check container status
docker compose ps

# Rebuild from scratch
docker compose down
docker compose up --build -d
```

### High Memory Usage

```bash
# Check resource limits in docker-compose.prod.yml
# Increase limits if needed:
services:
  web:
    deploy:
      resources:
        limits:
          memory: 1G
```

### SSL/TLS Issues

```bash
# Test certificate
sudo certbot certificates

# Renew manually
sudo certbot renew

# Check nginx config
sudo nginx -t
```

### Port Already in Use

```bash
# Find process using port 3000
sudo lsof -i :3000

# Kill process
sudo kill -9 PID

# Or change port in docker-compose.yml
```

---

## Performance Optimization

### nginx Caching

```nginx
# Add to nginx config
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m max_size=1g inactive=60m;

location / {
    proxy_cache my_cache;
    proxy_cache_valid 200 60m;
    proxy_cache_bypass $http_cache_control;
    add_header X-Cache-Status $upstream_cache_status;
}
```

### CDN Integration

Use a CDN for static assets:
- Cloudflare (free)
- AWS CloudFront
- Fastly
- StackPath

### Database Optimization (Future)

When Holochain conductor is integrated:
- Configure appropriate memory limits
- Enable connection pooling
- Regular data compaction

---

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  web:
    deploy:
      replicas: 3
```

Use a load balancer (nginx, HAProxy, AWS ALB).

### Vertical Scaling

Increase resource limits in `docker-compose.prod.yml`.

---

## Support

**Need help?**
- 📖 [Documentation](README.md)
- 💬 [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- 🐛 [Report Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)

---

*Last updated: 2025-11-15*
