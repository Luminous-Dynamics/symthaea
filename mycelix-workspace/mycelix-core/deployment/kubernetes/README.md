# Mycelix Kubernetes Deployment

Production-grade Kubernetes manifests for deploying the Mycelix testnet.

## Directory Structure

```
kubernetes/
├── README.md                    # This file
├── kustomization.yaml           # Kustomize configuration
├── namespaces.yaml              # Namespace definitions
├── validator-deployment.yaml    # Validator StatefulSet
├── fl-node-deployment.yaml      # FL Aggregator & Coordinator
├── configmaps.yaml              # Application configuration
├── external-secrets.yaml        # AWS Secrets Manager integration
├── ingress.yaml                 # Ingress & TLS configuration
├── autoscaling.yaml             # HPA, PDB, Resource Quotas
└── testnet/
    └── kubernetes/
        ├── rbac.yaml            # Service accounts & RBAC
        └── network-policies.yaml # Network isolation
```

## Prerequisites

1. **Kubernetes cluster** - EKS 1.29+ (deployed via Terraform)
2. **kubectl** - Configured with cluster access
3. **Kustomize** - v5.0+ (included with kubectl)
4. **AWS Secrets Manager** - Secrets created for database, redis, keys

## Quick Start

### 1. Verify Cluster Access

```bash
kubectl cluster-info
kubectl get nodes
```

### 2. Create Secrets in AWS Secrets Manager

```bash
# Database credentials
aws secretsmanager create-secret \
  --name mycelix/testnet/database \
  --secret-string '{"connection_string":"postgresql://...", "username":"mycelix_admin", "password":"..."}'

# Redis credentials
aws secretsmanager create-secret \
  --name mycelix/testnet/redis \
  --secret-string '{"connection_string":"redis://..."}'

# Validator keys (generate with mycelix-keygen)
aws secretsmanager create-secret \
  --name mycelix/testnet/validator-keys \
  --secret-string '{"signing_key":"...", "p2p_key":"..."}'

# Ethereum credentials
aws secretsmanager create-secret \
  --name mycelix/testnet/ethereum \
  --secret-string '{"infura_project_id":"...", "wallet_private_key":"..."}'
```

### 3. Deploy with Kustomize

```bash
# Preview the manifests
kubectl kustomize . | less

# Apply to cluster
kubectl apply -k .

# Verify deployment
kubectl get pods -n mycelix-validators
kubectl get pods -n mycelix-fl
```

## Components

### Validators (StatefulSet)

- **Replicas**: 5 (configurable)
- **Resources**: 2-4 CPU, 8-14GB RAM per pod
- **Storage**: 250GB persistent volume per validator
- **Ports**:
  - 9000: P2P (exposed via LoadBalancer)
  - 9090: Metrics (internal)
  - 8080: Health checks (internal)
  - 50051: gRPC (internal)

### FL Aggregator (Deployment)

- **Replicas**: 3-10 (autoscaled)
- **Resources**: 4-8 CPU, 12-28GB RAM per pod
- **Autoscaling**: Based on CPU, memory, and queue length
- **Ports**:
  - 8081: HTTP API
  - 9091: Metrics
  - 8082: Health checks

### FL Coordinator (Deployment)

- **Replicas**: 1 (singleton)
- **Resources**: 1-2 CPU, 4-8GB RAM
- **Ports**:
  - 8083: HTTP API
  - 9092: Metrics
  - 8084: Health checks

## Configuration

### Environment Variables

Key configuration via ConfigMaps:

| Variable | Description | Default |
|----------|-------------|---------|
| `network` | Network identifier | `testnet` |
| `consensus_algorithm` | Consensus type | `rb-bft` |
| `byzantine_threshold` | Max Byzantine tolerance | `0.45` |
| `fl_round_duration_secs` | FL round duration | `300` |
| `dp_enabled` | Differential privacy | `true` |
| `dp_epsilon` | DP epsilon value | `1.0` |

### Secrets (via External Secrets)

| Secret | Namespace | Contents |
|--------|-----------|----------|
| `database-credentials` | validators, fl | PostgreSQL connection |
| `redis-credentials` | validators, fl | Redis connection |
| `validator-keys` | validators | Signing and P2P keys |
| `ethereum-credentials` | mycelix | Infura and wallet keys |

## Monitoring

### Prometheus Metrics

All components expose metrics at their metrics ports:

```bash
# Port-forward to view metrics
kubectl port-forward -n mycelix-validators svc/validator-metrics 9090:9090

# View metrics
curl http://localhost:9090/metrics
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `mycelix_validator_blocks_produced` | Blocks produced by validator |
| `mycelix_consensus_round_duration` | Consensus round time |
| `mycelix_fl_rounds_completed` | FL rounds completed |
| `mycelix_byzantine_detections` | Byzantine nodes detected |
| `mycelix_trust_score_distribution` | Trust score histogram |

### Grafana Dashboards

Access Grafana at `https://grafana.testnet.mycelix.io`:
- Validator Overview
- FL Pipeline Metrics
- Byzantine Detection
- Network Health

## Networking

### Ingress

| Host | Path | Service |
|------|------|---------|
| api.testnet.mycelix.io | /fl | fl-aggregator |
| api.testnet.mycelix.io | /coordinator | fl-coordinator |
| grafana.testnet.mycelix.io | / | grafana |
| prometheus.testnet.mycelix.io | / | prometheus-server |

### Network Policies

Default deny with explicit allow rules:
- Validators can communicate with each other
- FL nodes can access validators
- All pods can access database and redis
- Monitoring can scrape all namespaces

## Scaling

### Manual Scaling

```bash
# Scale validators (requires careful coordination)
kubectl scale statefulset validator -n mycelix-validators --replicas=7

# Scale FL aggregators
kubectl scale deployment fl-aggregator -n mycelix-fl --replicas=5
```

### Autoscaling

FL Aggregator automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Queue length (target: 100 submissions)

```bash
# View HPA status
kubectl get hpa -n mycelix-fl
```

## Upgrades

### Rolling Update (FL Nodes)

```bash
# Update image tag in kustomization.yaml, then:
kubectl apply -k .
```

### Validator Upgrades (Coordinated)

Validators require coordinated upgrades:

```bash
# 1. Announce upgrade window to validators
# 2. Update one validator at a time
kubectl rollout restart statefulset/validator -n mycelix-validators

# 3. Monitor rollout
kubectl rollout status statefulset/validator -n mycelix-validators
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n <namespace>

# Check logs
kubectl logs <pod-name> -n <namespace>

# Check resource availability
kubectl top nodes
```

### Database Connection Issues

```bash
# Verify secret exists
kubectl get secret database-credentials -n mycelix-validators -o yaml

# Test connection from pod
kubectl exec -it <pod-name> -n mycelix-validators -- \
  psql "$DATABASE_URL" -c "SELECT 1"
```

### Network Issues

```bash
# Check network policies
kubectl get networkpolicies -n mycelix-validators

# Test connectivity
kubectl exec -it <pod-name> -n mycelix-validators -- \
  curl -v http://fl-aggregator.mycelix-fl:8081/health
```

## Backup & Recovery

### PVC Snapshots

```bash
# Create snapshot (EBS CSI driver)
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: validator-0-snapshot
  namespace: mycelix-validators
spec:
  volumeSnapshotClassName: ebs-csi-snapclass
  source:
    persistentVolumeClaimName: data-validator-0
EOF
```

### Restore from Snapshot

See [DISASTER_RECOVERY.md](../../docs/operations/DISASTER_RECOVERY.md)

## Security

### Pod Security Standards

All namespaces enforce `restricted` PSS:
- Non-root containers
- Read-only root filesystem
- No privilege escalation
- Dropped capabilities

### RBAC

Service accounts have minimal required permissions:
- `validator-sa`: Pod metadata, secrets read
- `fl-node-sa`: Pod metadata, secrets read, S3 access
- `monitoring-sa`: Metrics scraping

## Support

- **Documentation**: https://docs.mycelix.io/kubernetes
- **Issues**: https://github.com/mycelix/core/issues
- **Slack**: #platform-team

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-16 | Platform Team | Initial version |
