# Production Operations Runbook: Zero-TrustML Credits DNA

**Version**: 1.0
**Last Updated**: October 1, 2025
**Status**: Production Ready

---

## Overview

This runbook provides operational procedures for running Zero-TrustML Credits DNA in production. It covers monitoring, alerting, incident response, and recovery procedures.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Monitoring Setup](#monitoring-setup)
3. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
4. [Alerting Configuration](#alerting-configuration)
5. [Incident Response](#incident-response)
6. [Recovery Procedures](#recovery-procedures)
7. [Maintenance Operations](#maintenance-operations)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                  Zero-TrustML Credits System                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Zero-TrustML    │  │  Credits     │  │  Holochain   │ │
│  │  Reputation │─→│  Integration │─→│  Credits     │ │
│  │  System     │  │              │  │  Bridge      │ │
│  └─────────────┘  └──────────────┘  └──────────────┘ │
│         ↓                ↓                   ↓          │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Production Monitor                     │  │
│  │  • Metrics Collector                            │  │
│  │  • Alert Manager                                │  │
│  │  • Health Checks                                │  │
│  └─────────────────────────────────────────────────┘  │
│                        ↓                                │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Monitoring Dashboard                   │  │
│  │  • Grafana (metrics visualization)              │  │
│  │  • Prometheus (time-series storage)             │  │
│  │  • AlertManager (notifications)                 │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Dependencies

- **Python 3.13.7+**: Runtime environment
- **Holochain Conductor**: WebSocket API for credit storage
- **PostgreSQL** (optional): Persistent storage backend
- **Prometheus** (recommended): Metrics storage
- **Grafana** (recommended): Metrics visualization

---

## Monitoring Setup

### 1. Start Production Monitor

```python
from src.monitoring.production_monitor import ProductionMonitor
import asyncio

# Initialize monitor with 60-second check interval
monitor = ProductionMonitor(check_interval_seconds=60)

# Start monitoring in background
monitor_task = asyncio.create_task(monitor.start())

# Monitor will run continuously, logging metrics and alerts
```

### 2. Integration with Zero-TrustML Credits

```python
from zerotrustml_credits_integration import Zero-TrustMLCreditsIntegration
from src.monitoring.production_monitor import ProductionMonitor

# Initialize both systems
monitor = ProductionMonitor()
integration = Zero-TrustMLCreditsIntegration(bridge)

# Hook monitoring into credit events
async def on_credit_issued(node_id, credits, event_type, response_time):
    # Issue credit
    credit_id = await integration.on_quality_gradient(...)

    # Record metrics
    monitor.metrics_collector.record_credit_issued(
        node_id=node_id,
        credits=credits,
        event_type=event_type,
        response_time_ms=response_time
    )

async def on_byzantine_detected(detector_id, detected_id, evidence):
    # Handle detection
    await integration.on_byzantine_detection(...)

    # Record metrics
    monitor.metrics_collector.record_byzantine_detection(
        detector_id=detector_id,
        detected_id=detected_id,
        evidence=evidence
    )
```

### 3. Export Metrics to Prometheus (Optional)

```python
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# Define metrics
credits_issued = Counter('zerotrustml_credits_issued_total', 'Total credits issued')
response_time = Histogram('zerotrustml_response_time_seconds', 'Response time')
byzantine_detections = Counter('zerotrustml_byzantine_detections_total', 'Byzantine detections')
active_nodes = Gauge('zerotrustml_active_nodes', 'Active nodes')

# Start Prometheus exporter on port 8000
start_http_server(8000)

# Update metrics from monitor
def export_metrics():
    metrics = monitor.metrics_collector.get_current_metrics()
    active_nodes.set(metrics.active_nodes)
    # ... update other metrics
```

---

## Key Performance Indicators (KPIs)

### Throughput KPIs

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| **Events per minute** | 10-50 | <5 | <2 |
| **Credits per hour** | 1,000-10,000 | <500 | <100 |
| **Average response time** | <1000ms | >3000ms | >5000ms |
| **P99 response time** | <3000ms | >5000ms | >10000ms |

### Security KPIs

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| **Byzantine detection rate** | >95% | <90% | <85% |
| **False positive rate** | <1% | >2% | >5% |
| **Detection latency** | <5s | >10s | >30s |

### Availability KPIs

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| **System uptime** | >99.9% | <99.5% | <99% |
| **Healthy node %** | >95% | <90% | <85% |
| **Error rate** | <0.1% | >1% | >5% |
| **Rate limit violations** | <10/hour | >50/hour | >100/hour |

### Capacity KPIs

| Metric | Target | Warning Threshold | Critical Threshold |
|--------|--------|-------------------|-------------------|
| **Active nodes** | 50-100 | >150 | >200 |
| **Memory usage** | <2GB | >4GB | >6GB |
| **Storage growth** | <100MB/day | >500MB/day | >1GB/day |

---

## Alerting Configuration

### Alert Severity Levels

1. **INFO**: Informational, no action required
2. **WARNING**: Action required within 24 hours
3. **CRITICAL**: Immediate action required

### Alert Categories

#### 1. Performance Alerts

**WARNING: High Average Response Time**
- **Trigger**: Avg response time > 3000ms
- **Impact**: Degraded user experience
- **Resolution**:
  1. Check system load: `top`, `htop`
  2. Check Holochain conductor status
  3. Review recent code changes
  4. Scale horizontally if sustained

**CRITICAL: Very High P99 Response Time**
- **Trigger**: P99 response time > 10000ms
- **Impact**: Some requests timing out
- **Resolution**:
  1. Immediate investigation required
  2. Check for database bottlenecks
  3. Review slow query logs
  4. Consider circuit breaker activation

#### 2. Security Alerts

**WARNING: Low Detection Rate**
- **Trigger**: Byzantine detection rate < 90%
- **Impact**: Potential undetected malicious activity
- **Resolution**:
  1. Verify PoGQ validation system status
  2. Check validator configuration
  3. Review recent Byzantine attack patterns
  4. Adjust detection thresholds if needed

**INFO: High Detection Activity**
- **Trigger**: >10 Byzantine detections in 1 hour
- **Impact**: Possible coordinated attack
- **Resolution**:
  1. Monitor pattern of detections
  2. Verify detections are valid (not false positives)
  3. Consider temporarily increasing reputation penalties

#### 3. Availability Alerts

**CRITICAL: High Error Rate**
- **Trigger**: Error rate > 5%
- **Impact**: System unreliable, users affected
- **Resolution**:
  1. Check application logs: `tail -f /var/log/zerotrustml/app.log`
  2. Verify Holochain connectivity: `hc sandbox call`
  3. Check database connection pool
  4. Activate backup systems if available

**CRITICAL: Low Healthy Node Percentage**
- **Trigger**: Healthy nodes < 85% of active nodes
- **Impact**: Network degradation, reduced redundancy
- **Resolution**:
  1. Identify unhealthy nodes: `monitor.get_status_report()`
  2. Check network connectivity
  3. Review node logs for failures
  4. Restart unhealthy nodes if needed

#### 4. Capacity Alerts

**WARNING: High Rate Limit Violations**
- **Trigger**: >50 rate limit violations per hour
- **Impact**: Potential spam attack or legitimate growth
- **Resolution**:
  1. Identify nodes hitting limits
  2. Analyze patterns (spam vs legitimate)
  3. Adjust rate limits if legitimate growth
  4. Ban nodes if confirmed spam attack

**WARNING: High Memory Usage**
- **Trigger**: Memory usage > 4GB
- **Impact**: Risk of OOM crashes
- **Resolution**:
  1. Check for memory leaks: `ps aux --sort=-%mem`
  2. Review metrics collector window size
  3. Clear old metrics if safe
  4. Scale up instance if sustained

---

## Incident Response

### Incident Response Process

```
┌──────────────┐
│   Detect     │ ← Alert triggered or user report
└──────┬───────┘
       ↓
┌──────────────┐
│   Assess     │ ← Determine severity and impact
└──────┬───────┘
       ↓
┌──────────────┐
│   Respond    │ ← Execute recovery procedures
└──────┬───────┘
       ↓
┌──────────────┐
│   Resolve    │ ← Verify system restored
└──────┬───────┘
       ↓
┌──────────────┐
│  Post-Mortem │ ← Document and improve
└──────────────┘
```

### Common Incidents

#### Incident 1: System Unresponsive

**Symptoms**:
- Response times > 10 seconds
- Timeouts on API calls
- No credits being issued

**Diagnosis**:
```bash
# Check system status
python -c "from src.monitoring.production_monitor import *; monitor = ProductionMonitor(); print(monitor.get_status_report())"

# Check Holochain conductor
hc sandbox list
hc sandbox call --running

# Check resource usage
top
df -h
```

**Resolution**:
1. **If Holochain down**: Restart conductor
   ```bash
   hc sandbox clean
   hc sandbox run
   ```

2. **If database connection issue**: Restart database
   ```bash
   sudo systemctl restart postgresql
   ```

3. **If application hung**: Restart application
   ```bash
   sudo systemctl restart zerotrustml-credits
   ```

4. **If resource exhaustion**: Scale up or free resources
   ```bash
   # Free memory
   sudo sync; echo 3 > /proc/sys/vm/drop_caches

   # Or scale horizontally (add nodes)
   ```

#### Incident 2: Byzantine Detection Failure

**Symptoms**:
- Detection rate suddenly drops below 85%
- Known malicious nodes not being caught
- Reputation system not updating

**Diagnosis**:
```python
# Check recent detections
from zerotrustml_credits_integration import Zero-TrustMLCreditsIntegration

integration = Zero-TrustMLCreditsIntegration(bridge)
audit = await integration.get_audit_trail("all")

# Filter Byzantine detection events
detections = [e for e in audit if e['event_type'] == 'byzantine_detection']
print(f"Recent detections: {len(detections)}")
```

**Resolution**:
1. **Verify PoGQ validation**: Check quality scores are being calculated
2. **Check validator configuration**: Ensure validators are active
3. **Review detection thresholds**: May need adjustment
4. **Check for validator Byzantine nodes**: Validators themselves may be compromised

#### Incident 3: Data Corruption

**Symptoms**:
- Inconsistent credit balances
- Audit trail gaps
- Failed validations

**Diagnosis**:
```bash
# Check Holochain DHT consistency
hc sandbox call zerotrustml_credits get_all_credits

# Check database integrity (if using PostgreSQL)
psql -U zerotrustml -d credits -c "SELECT COUNT(*) FROM credits WHERE created_at > NOW() - INTERVAL '1 hour';"
```

**Resolution**:
1. **If DHT inconsistency**: Run DHT repair
   ```bash
   # Trigger gossip protocol to re-sync
   hc sandbox call zerotrustml_credits trigger_gossip
   ```

2. **If database corruption**: Restore from backup
   ```bash
   # Stop application
   sudo systemctl stop zerotrustml-credits

   # Restore from latest backup
   pg_restore -U zerotrustml -d credits /backups/credits_latest.dump

   # Restart application
   sudo systemctl start zerotrustml-credits
   ```

3. **If irrecoverable**: Manual reconciliation required

---

## Recovery Procedures

### Backup and Restore

#### 1. Database Backup (PostgreSQL)

**Daily Backup**:
```bash
#!/bin/bash
# /opt/zerotrustml/scripts/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/zerotrustml"
DB_NAME="credits"

# Create backup
pg_dump -U zerotrustml $DB_NAME > $BACKUP_DIR/credits_$DATE.sql

# Compress
gzip $BACKUP_DIR/credits_$DATE.sql

# Keep last 30 days
find $BACKUP_DIR -name "credits_*.sql.gz" -mtime +30 -delete

echo "Backup completed: credits_$DATE.sql.gz"
```

**Schedule via cron**:
```bash
# Run daily at 2 AM
0 2 * * * /opt/zerotrustml/scripts/backup.sh
```

**Restore from Backup**:
```bash
# Stop application
sudo systemctl stop zerotrustml-credits

# Restore database
gunzip -c /backups/zerotrustml/credits_20251001_020000.sql.gz | psql -U zerotrustml credits

# Restart application
sudo systemctl start zerotrustml-credits

# Verify
python -c "from src.monitoring.production_monitor import *; monitor = ProductionMonitor(); print(monitor.get_status_report())"
```

#### 2. Holochain DHT Backup

**Export DHT State**:
```bash
# Export all credits from DHT
hc sandbox call zerotrustml_credits export_state > /backups/holochain/dht_export_$(date +%Y%m%d).json
```

**Import DHT State**:
```bash
# Re-import state (requires custom zome function)
cat /backups/holochain/dht_export_20251001.json | hc sandbox call zerotrustml_credits import_state
```

### Disaster Recovery

#### Full System Failure

**Prerequisites**:
- Recent database backup
- Recent DHT export
- System configuration backup

**Recovery Steps**:

1. **Provision new infrastructure**
   ```bash
   # Launch new server
   # Install dependencies: Python 3.13, Holochain, PostgreSQL
   ```

2. **Restore database**
   ```bash
   # Create database
   psql -U postgres -c "CREATE DATABASE credits;"

   # Restore from backup
   gunzip -c /backups/credits_latest.sql.gz | psql -U zerotrustml credits
   ```

3. **Restore Holochain DHT**
   ```bash
   # Start conductor
   hc sandbox run

   # Install DNA
   hc sandbox call zerotrustml_credits install_dna

   # Import state
   cat /backups/dht_export_latest.json | hc sandbox call zerotrustml_credits import_state
   ```

4. **Deploy application**
   ```bash
   # Clone repository
   git clone https://github.com/luminous-dynamics/0TML
   cd 0TML

   # Enter Nix environment
   nix develop

   # Start application
   python src/main.py
   ```

5. **Verify recovery**
   ```python
   from src.monitoring.production_monitor import ProductionMonitor

   monitor = ProductionMonitor()
   report = monitor.get_status_report()

   print(f"System status: {report['system_status']}")
   print(f"Active nodes: {report['metrics']['health']['active_nodes']}")
   ```

**Expected Recovery Time**: 1-2 hours

---

## Maintenance Operations

### Routine Maintenance

#### Daily Tasks (Automated)

- ✅ Database backup (2 AM)
- ✅ DHT export (2:30 AM)
- ✅ Log rotation (3 AM)
- ✅ Metrics cleanup (4 AM)

#### Weekly Tasks (Manual)

**Monday**: Review system health
```bash
# Generate weekly report
python scripts/generate_weekly_report.py

# Review alerts from past week
python -c "from src.monitoring.production_monitor import *; monitor = ProductionMonitor(); print(monitor.alert_manager.alert_history[-100:])"

# Check for security updates
nix flake update
```

**Wednesday**: Performance review
```bash
# Analyze response times
python scripts/analyze_performance.py --days 7

# Review Byzantine detection patterns
python scripts/analyze_detections.py --days 7
```

**Friday**: Capacity planning
```bash
# Check growth trends
python scripts/capacity_report.py

# Plan for scaling if needed
```

#### Monthly Tasks

- Review and update alert thresholds
- Test disaster recovery procedures
- Update documentation
- Conduct security audit
- Review rate limits

### Scaling Operations

#### Horizontal Scaling (Add Nodes)

```bash
# 1. Provision new node
# 2. Install dependencies
# 3. Configure to join network
# 4. Start monitoring

# Verify node joined
python -c "from src.monitoring.production_monitor import *; monitor = ProductionMonitor(); print(monitor.metrics_collector.active_nodes)"
```

#### Vertical Scaling (Upgrade Resources)

```bash
# 1. Schedule maintenance window
# 2. Backup current state
# 3. Stop application
sudo systemctl stop zerotrustml-credits

# 4. Upgrade instance (more CPU/RAM/storage)
# 5. Restart application
sudo systemctl start zerotrustml-credits

# 6. Verify performance improvement
```

---

## Troubleshooting Guide

### Problem: High Latency

**Symptoms**: Response times > 3 seconds

**Checks**:
1. System resources: `top`, `htop`, `iostat`
2. Database performance: Check slow query log
3. Network latency: `ping` Holochain conductor
4. Cache hit rate: Review metrics

**Solutions**:
- Optimize database queries
- Increase cache size
- Add indexes
- Scale horizontally

### Problem: Memory Leak

**Symptoms**: Memory usage continuously growing

**Checks**:
1. Monitor memory over time: `ps aux --sort=-%mem`
2. Review metrics collector window size
3. Check for lingering connections

**Solutions**:
```python
# Reduce metrics window size
monitor = ProductionMonitor()
monitor.metrics_collector.window_size = 30  # Reduce from 60 minutes

# Clear old metrics
monitor.metrics_collector.credit_events.clear()
```

### Problem: Connection Failures

**Symptoms**: "Failed to connect to Holochain"

**Checks**:
1. Holochain conductor running: `hc sandbox list`
2. WebSocket port accessible: `telnet localhost 8888`
3. Firewall rules: `sudo iptables -L`

**Solutions**:
```bash
# Restart conductor
hc sandbox clean
hc sandbox run

# Check logs
tail -f ~/.holochain/logs/conductor.log
```

### Problem: Unexpected Byzantine Detections

**Symptoms**: Legitimate nodes flagged as Byzantine

**Checks**:
1. Review PoGQ scores: Check if legitimately low quality
2. Check detection evidence: Review anomaly patterns
3. Verify validator accuracy: Cross-check multiple validators

**Solutions**:
- Adjust PoGQ thresholds if too strict
- Review detection algorithm
- Manually restore reputation if false positive

---

## Contact Information

### On-Call Rotation

| Role | Primary | Backup |
|------|---------|--------|
| **Production Engineer** | ops@luminousdynamics.org | dev@luminousdynamics.org |
| **Security Engineer** | security@luminousdynamics.org | - |
| **Database Admin** | dba@luminousdynamics.org | ops@luminousdynamics.org |

### Escalation Path

1. **Level 1**: On-call engineer (initial response within 15 minutes)
2. **Level 2**: Senior engineer (escalate if not resolved in 1 hour)
3. **Level 3**: System architect (escalate if critical and not resolved in 4 hours)

---

## Appendix

### Useful Commands

```bash
# Check system status
systemctl status zerotrustml-credits

# View application logs
tail -f /var/log/zerotrustml/app.log

# Monitor in real-time
watch -n 5 'python -c "from src.monitoring.production_monitor import *; monitor = ProductionMonitor(); print(monitor.get_status_report())"'

# Export metrics
curl http://localhost:8000/metrics

# Check Holochain DHT
hc sandbox call zerotrustml_credits get_network_stats
```

### Log Locations

- **Application logs**: `/var/log/zerotrustml/app.log`
- **Monitoring logs**: `/var/log/zerotrustml/monitor.log`
- **Holochain conductor**: `~/.holochain/logs/conductor.log`
- **PostgreSQL logs**: `/var/log/postgresql/postgresql-15-main.log`

### Configuration Files

- **Application config**: `/etc/zerotrustml/config.yaml`
- **Monitoring config**: `/etc/zerotrustml/monitoring.yaml`
- **Holochain conductor**: `~/.holochain/conductor-config.yaml`

---

*Last reviewed: October 1, 2025*
*Next review: November 1, 2025*
