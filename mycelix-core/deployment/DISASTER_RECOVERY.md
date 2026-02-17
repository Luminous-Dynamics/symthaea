# Mycelix Disaster Recovery & Backup Procedures

## Overview

This document defines backup and disaster recovery procedures for the Mycelix infrastructure. It covers data protection, recovery procedures, and defines Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO) for each component.

---

## 1. Recovery Objectives

### Service Level Definitions

| Tier | RTO | RPO | Components |
|------|-----|-----|------------|
| **Critical** | 1 hour | 15 minutes | Validator nodes, consensus layer |
| **High** | 4 hours | 1 hour | FL aggregator, PostgreSQL, Redis |
| **Standard** | 24 hours | 24 hours | Monitoring, logging, dashboards |

### Definitions

- **RTO (Recovery Time Objective)**: Maximum acceptable time to restore service after an outage
- **RPO (Recovery Point Objective)**: Maximum acceptable data loss measured in time

---

## 2. Backup Strategy

### 2.1 PostgreSQL Database

**Backup Type**: Continuous WAL archiving + Daily full snapshots

**Configuration** (add to postgres-deployment.yaml):
```yaml
env:
  - name: POSTGRES_ARCHIVE_MODE
    value: "on"
  - name: POSTGRES_ARCHIVE_COMMAND
    value: "aws s3 cp %p s3://mycelix-backups/wal/%f"
```

**Backup Schedule**:
- **WAL Archiving**: Continuous (every 5 minutes or 16MB)
- **Full Backup**: Daily at 02:00 UTC
- **Retention**: 30 days for full backups, 7 days for WAL

**Manual Backup Command**:
```bash
# Create on-demand backup
kubectl exec -n mycelix-testnet postgres-0 -- \
  pg_dump -Fc -U mycelix mycelix_testnet > backup_$(date +%Y%m%d_%H%M%S).dump

# Upload to S3
aws s3 cp backup_*.dump s3://mycelix-backups/postgres/manual/
```

**Automated CronJob** (add to deployment):
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: mycelix-testnet
spec:
  schedule: "0 2 * * *"  # Daily at 02:00 UTC
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:16-alpine
              command:
                - /bin/sh
                - -c
                - |
                  BACKUP_FILE="pg_backup_$(date +%Y%m%d_%H%M%S).dump"
                  pg_dump -h postgres -U mycelix -Fc mycelix_testnet > /backup/$BACKUP_FILE
                  aws s3 cp /backup/$BACKUP_FILE s3://mycelix-backups/postgres/daily/
              envFrom:
                - secretRef:
                    name: postgres-secrets
                - secretRef:
                    name: aws-credentials
              volumeMounts:
                - name: backup-volume
                  mountPath: /backup
          restartPolicy: OnFailure
          volumes:
            - name: backup-volume
              emptyDir: {}
```

### 2.2 Redis Cache

**Backup Type**: RDB snapshots + AOF persistence

**Configuration** (already enabled in redis-deployment.yaml):
```yaml
command:
  - redis-server
  - --appendonly
  - "yes"
  - --save
  - "900 1"    # Save if 1 key changed in 15 min
  - --save
  - "300 10"   # Save if 10 keys changed in 5 min
  - --save
  - "60 10000" # Save if 10000 keys changed in 1 min
```

**Backup Schedule**:
- **RDB Snapshots**: Every 15 minutes (automatic)
- **External Backup**: Daily sync to S3
- **Retention**: 7 days

**Manual Backup Command**:
```bash
# Trigger snapshot
kubectl exec -n mycelix-testnet redis-0 -- redis-cli BGSAVE

# Copy dump file
kubectl cp mycelix-testnet/redis-0:/data/dump.rdb ./redis_backup_$(date +%Y%m%d).rdb
```

### 2.3 Persistent Volumes (PVCs)

**Backup Method**: Volume snapshots (cloud provider specific)

**AWS EBS Snapshots**:
```bash
# List volumes
kubectl get pvc -n mycelix-testnet

# Create snapshot (AWS)
aws ec2 create-snapshot \
  --volume-id vol-XXXXX \
  --description "Mycelix PVC backup $(date +%Y%m%d)" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=mycelix-backup}]'
```

**Automated Volume Snapshots** (using Velero):
```yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-pvc-backup
  namespace: velero
spec:
  schedule: "0 3 * * *"  # Daily at 03:00 UTC
  template:
    includedNamespaces:
      - mycelix-testnet
    includedResources:
      - persistentvolumeclaims
      - persistentvolumes
    storageLocation: aws-backups
    volumeSnapshotLocations:
      - aws-snapshots
    ttl: 720h  # 30 days retention
```

### 2.4 Secrets & Configuration

**Backup Method**: Encrypted export to secure storage

**IMPORTANT**: Never store unencrypted secrets in backups!

**Using Sealed Secrets**:
```bash
# Export sealed secrets (safe to store)
kubectl get sealedsecret -n mycelix-testnet -o yaml > sealed-secrets-backup.yaml

# Store in version control or S3
aws s3 cp sealed-secrets-backup.yaml s3://mycelix-backups/secrets/
```

**Using External Secrets (AWS Secrets Manager)**:
- Secrets are stored in AWS Secrets Manager
- Enable versioning on secrets for point-in-time recovery
- Cross-region replication for DR

**ConfigMaps Backup**:
```bash
kubectl get configmap -n mycelix-testnet -o yaml > configmaps-backup.yaml
```

### 2.5 Validator Keys

**CRITICAL**: Validator private keys require special handling

**Backup Procedure**:
1. Keys should be generated offline using air-gapped machine
2. Store encrypted backup in multiple secure locations:
   - Hardware security module (HSM)
   - Cold storage (encrypted USB in secure vault)
   - Split key shares using Shamir's Secret Sharing (3-of-5)
3. Never store validator keys in S3 or cloud storage
4. Document key custodians and recovery process

---

## 3. Recovery Procedures

### 3.1 PostgreSQL Recovery

**Point-in-Time Recovery (PITR)**:
```bash
# 1. Stop application pods
kubectl scale deployment mycelix-bootstrap --replicas=0 -n mycelix-testnet
kubectl scale statefulset mycelix-fl-node --replicas=0 -n mycelix-testnet

# 2. Restore base backup
aws s3 cp s3://mycelix-backups/postgres/daily/pg_backup_YYYYMMDD.dump .
kubectl exec -i postgres-0 -n mycelix-testnet -- \
  pg_restore -U mycelix -d mycelix_testnet -c < pg_backup_YYYYMMDD.dump

# 3. Replay WAL logs to target time
kubectl exec postgres-0 -n mycelix-testnet -- \
  psql -U mycelix -c "SELECT pg_wal_replay_resume();"

# 4. Verify data integrity
kubectl exec postgres-0 -n mycelix-testnet -- \
  psql -U mycelix -d mycelix_testnet -c "SELECT COUNT(*) FROM validators;"

# 5. Restart applications
kubectl scale deployment mycelix-bootstrap --replicas=1 -n mycelix-testnet
kubectl scale statefulset mycelix-fl-node --replicas=5 -n mycelix-testnet
```

### 3.2 Full Cluster Recovery

**Scenario**: Complete cluster loss requiring rebuild

**Prerequisites**:
- Access to backup storage (S3)
- Sealed secrets controller key (stored securely offline)
- Terraform state file

**Recovery Steps**:

```bash
# 1. Provision new infrastructure
cd deployment/terraform
terraform init
terraform apply

# 2. Install base components
kubectl apply -f deployment/testnet/kubernetes/namespace.yaml
kubectl apply -f deployment/testnet/kubernetes/rbac.yaml

# 3. Restore sealed secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Import controller private key (from secure storage)
kubectl apply -f sealed-secrets-private-key.yaml

# 4. Restore application secrets
aws s3 cp s3://mycelix-backups/secrets/sealed-secrets-backup.yaml .
kubectl apply -f sealed-secrets-backup.yaml

# 5. Restore PVCs from snapshots
# (Use Velero or cloud provider tools)
velero restore create --from-backup daily-pvc-backup-YYYYMMDD

# 6. Deploy applications
kubectl apply -k deployment/testnet/kubernetes/

# 7. Restore database
# (Follow PostgreSQL recovery procedure above)

# 8. Verify services
kubectl get pods -n mycelix-testnet
kubectl logs -n mycelix-testnet deployment/mycelix-bootstrap
```

### 3.3 Redis Recovery

```bash
# 1. Scale down dependent services
kubectl scale deployment mycelix-bootstrap --replicas=0 -n mycelix-testnet

# 2. Copy backup to pod
kubectl cp redis_backup_YYYYMMDD.rdb mycelix-testnet/redis-0:/data/dump.rdb

# 3. Restart Redis
kubectl delete pod redis-0 -n mycelix-testnet
# StatefulSet will recreate, loading dump.rdb on startup

# 4. Verify data
kubectl exec redis-0 -n mycelix-testnet -- redis-cli DBSIZE

# 5. Restart applications
kubectl scale deployment mycelix-bootstrap --replicas=1 -n mycelix-testnet
```

---

## 4. Backup Verification

### 4.1 Monthly Backup Testing

**Schedule**: First Monday of each month

**Procedure**:
1. Create isolated test namespace
2. Restore latest backups to test namespace
3. Verify data integrity with checksums
4. Run application smoke tests
5. Document results in backup test log

**Verification Script**:
```bash
#!/bin/bash
# backup-verification.sh

NAMESPACE="backup-test-$(date +%Y%m%d)"
BACKUP_DATE=$(date -d "yesterday" +%Y%m%d)

# Create test namespace
kubectl create namespace $NAMESPACE

# Restore PostgreSQL
echo "Restoring PostgreSQL backup..."
kubectl run pg-restore --image=postgres:16-alpine -n $NAMESPACE --restart=Never -- sleep 3600
kubectl wait --for=condition=Ready pod/pg-restore -n $NAMESPACE
aws s3 cp s3://mycelix-backups/postgres/daily/pg_backup_${BACKUP_DATE}.dump .
kubectl cp pg_backup_${BACKUP_DATE}.dump $NAMESPACE/pg-restore:/tmp/
kubectl exec -n $NAMESPACE pg-restore -- pg_restore -l /tmp/pg_backup_${BACKUP_DATE}.dump

# Verify record counts
VALIDATOR_COUNT=$(kubectl exec -n $NAMESPACE pg-restore -- \
  psql -U mycelix -d mycelix_testnet -t -c "SELECT COUNT(*) FROM validators")
echo "Validators recovered: $VALIDATOR_COUNT"

# Cleanup
kubectl delete namespace $NAMESPACE

echo "Backup verification complete"
```

### 4.2 Backup Monitoring

**Prometheus Alerts** (add to alerting rules):
```yaml
groups:
  - name: backup-alerts
    rules:
      - alert: BackupMissing
        expr: time() - backup_last_success_timestamp > 86400
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Backup has not completed in 24 hours"

      - alert: BackupFailed
        expr: backup_last_status == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Last backup job failed"
```

---

## 5. Disaster Scenarios & Runbooks

### 5.1 Single Node Failure

**Impact**: Minimal - handled by Kubernetes self-healing
**Recovery**: Automatic pod rescheduling
**Action Required**: Monitor for repeated failures

### 5.2 Availability Zone Failure

**Impact**: Potential service degradation
**Recovery Time**: 15-30 minutes
**Procedure**:
1. Kubernetes reschedules pods to remaining AZs
2. Verify PDB (Pod Disruption Budget) maintains quorum
3. Monitor for data consistency

### 5.3 Region Failure

**Impact**: Full service outage
**Recovery Time**: 2-4 hours
**Procedure**:
1. Activate DR region (if multi-region setup)
2. Update DNS to point to DR region
3. Restore from cross-region backups
4. Verify data consistency
5. Notify stakeholders

### 5.4 Data Corruption

**Impact**: Potential data loss
**Recovery Time**: 1-4 hours (depending on RPO)
**Procedure**:
1. Identify corruption scope
2. Stop affected services
3. Perform PITR to before corruption
4. Verify data integrity
5. Resume services
6. Post-incident analysis

### 5.5 Security Breach / Compromised Keys

**Impact**: Critical - potential fund loss
**Recovery Time**: Variable
**Procedure**:
1. **IMMEDIATE**: Revoke compromised keys
2. Rotate all secrets (database, API keys, etc.)
3. Audit access logs
4. Notify security team
5. Generate new validator keys (if affected)
6. Re-establish consensus with new keys

---

## 6. Contact & Escalation

### Escalation Matrix

| Severity | Response Time | First Contact | Escalation |
|----------|---------------|---------------|------------|
| P1 Critical | 15 min | On-call SRE | VP Engineering |
| P2 High | 1 hour | Platform team | Engineering Lead |
| P3 Medium | 4 hours | Support team | Platform team |
| P4 Low | Next business day | Ticket queue | Support team |

### Emergency Contacts

| Role | Contact Method |
|------|----------------|
| On-call SRE | PagerDuty escalation |
| Platform Lead | [REDACTED] |
| Security Team | security@mycelix.network |
| Cloud Provider | AWS Support Case |

---

## 7. Appendix

### A. Backup Storage Locations

| Type | Primary | Secondary |
|------|---------|-----------|
| Database | s3://mycelix-backups/postgres/ | Cross-region replica |
| Redis | s3://mycelix-backups/redis/ | - |
| PVC Snapshots | AWS EBS Snapshots | Cross-region copy |
| Secrets | AWS Secrets Manager | Sealed Secrets in Git |
| Validator Keys | HSM + Cold Storage | Split key shares |

### B. Backup Retention Policy

| Data Type | Retention | Archive |
|-----------|-----------|---------|
| Daily DB backups | 30 days | 1 year (monthly) |
| WAL archives | 7 days | - |
| PVC snapshots | 30 days | Quarterly |
| Audit logs | 90 days | 7 years |

### C. Recovery Checklist

- [ ] Infrastructure provisioned
- [ ] Network connectivity verified
- [ ] Secrets restored
- [ ] Database restored and verified
- [ ] Redis restored and verified
- [ ] Application pods running
- [ ] Health checks passing
- [ ] Monitoring operational
- [ ] External connectivity verified
- [ ] Stakeholders notified

---

*Last Updated: 2026-01-17*
*Document Owner: Platform Team*
*Review Cycle: Quarterly*
