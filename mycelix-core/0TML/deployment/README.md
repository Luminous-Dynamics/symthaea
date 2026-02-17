# Zero-TrustML Kubernetes Deployment

Complete Kubernetes and Helm deployment infrastructure for Zero-TrustML Phase 5.

## 📂 Directory Structure

```
deployment/
├── kubernetes/
│   ├── base/                    # Base Kubernetes manifests
│   │   ├── deployment.yaml      # Node + Redis deployments
│   │   ├── service.yaml         # ClusterIP + Headless services
│   │   ├── configmap.yaml       # Configuration
│   │   ├── hpa.yaml             # Horizontal Pod Autoscaler
│   │   └── servicemonitor.yaml  # Prometheus monitoring
│   └── overlays/                # Environment-specific overlays
│       ├── dev/                 # Development environment
│       ├── staging/             # Staging environment
│       └── production/          # Production (multi-region)
│           ├── values-us-east.yaml
│           ├── values-eu-west.yaml
│           └── values-ap-southeast.yaml
└── helm/
    └── zerotrustml/                 # Helm chart
        ├── Chart.yaml           # Chart metadata
        ├── values.yaml          # Default values
        └── templates/           # Kubernetes templates
            ├── deployment.yaml
            ├── service.yaml
            ├── hpa.yaml
            ├── configmap.yaml
            ├── redis.yaml
            ├── servicemonitor.yaml
            └── _helpers.tpl
```

## 🚀 Quick Start

### Option 1: Helm (Recommended)

```bash
# Add Helm repository (if published)
helm repo add zerotrustml https://charts.luminousdynamics.org
helm repo update

# Install with default values
helm install zerotrustml zerotrustml/zerotrustml

# Install with custom values
helm install zerotrustml zerotrustml/zerotrustml \
  --set replicaCount=5 \
  --set autoscaling.maxReplicas=150 \
  --set config.networking.numShards=20

# Install from local chart
cd deployment/helm
helm install zerotrustml ./zerotrustml -f values.yaml

# Upgrade existing deployment
helm upgrade zerotrustml ./zerotrustml
```

### Option 2: kubectl (Raw Manifests)

```bash
# Apply base manifests
kubectl apply -f deployment/kubernetes/base/

# Verify deployment
kubectl get pods -l app=zerotrustml
kubectl get svc -l app=zerotrustml
kubectl get hpa
```

## 🌍 Multi-Region Deployment

Deploy Zero-TrustML across multiple geographic regions:

```bash
# US East (Primary)
helm install zerotrustml-us-east ./helm/zerotrustml \
  -f deployment/kubernetes/overlays/production/values-us-east.yaml \
  --namespace zerotrustml-us-east \
  --create-namespace

# EU West
helm install zerotrustml-eu-west ./helm/zerotrustml \
  -f deployment/kubernetes/overlays/production/values-eu-west.yaml \
  --namespace zerotrustml-eu-west \
  --create-namespace

# AP Southeast
helm install zerotrustml-ap-southeast ./helm/zerotrustml \
  -f deployment/kubernetes/overlays/production/values-ap-southeast.yaml \
  --namespace zerotrustml-ap-southeast \
  --create-namespace
```

## ⚙️ Configuration

### Key Helm Values

```yaml
# Scaling
replicaCount: 3
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 100

# Resources
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# Zero-TrustML Configuration
config:
  node:
    enableTLS: true
    enableSigning: true

  performance:
    compressionAlgorithm: "zstd"
    enableRedisCache: true

  networking:
    enableGossip: true
    enableSharding: true
    numShards: 10
```

### Environment Variables

The following environment variables are automatically set:

- `NODE_ID`: Pod name (unique identifier)
- `ENABLE_TLS`: TLS encryption enabled
- `ENABLE_COMPRESSION`: Gradient compression
- `ENABLE_PROMETHEUS`: Metrics collection
- `ENABLE_GOSSIP`: Gossip protocol
- `ENABLE_SHARDING`: Network sharding
- `REDIS_HOST`: Redis service hostname
- `REDIS_PORT`: Redis service port (6379)
- `LOG_LEVEL`: Logging level (INFO)

## 📊 Monitoring

### Prometheus Integration

The chart automatically creates a ServiceMonitor for Prometheus:

```yaml
prometheus:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 15s
```

Access metrics:

```bash
# Port-forward Prometheus
kubectl port-forward svc/prometheus-operated 9090:9090

# View metrics
curl http://localhost:9090/api/v1/query?query=zerotrustml_gradients_validated_total
```

### Health Checks

```bash
# Check liveness
kubectl exec -it <pod-name> -- curl http://localhost:9090/health

# Check readiness
kubectl exec -it <pod-name> -- curl http://localhost:9090/ready

# View logs
kubectl logs -f <pod-name>
```

## 🔧 Advanced Configuration

### Custom Storage Backend

Use PostgreSQL or Holochain instead of in-memory:

```yaml
config:
  storage:
    backend: "postgresql"  # or "holochain"
    enableCheckpointing: true
```

### GPU Acceleration (Phase 5)

Enable GPU nodes:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1

nodeSelector:
  accelerator: nvidia-tesla-t4
```

### Network Policies

Enable network isolation:

```yaml
networkPolicy:
  enabled: true
  policyTypes:
    - Ingress
    - Egress
```

## 🔄 Rolling Updates

Perform zero-downtime updates:

```bash
# Update image
helm upgrade zerotrustml ./helm/zerotrustml --set image.tag=v0.5.1

# Rollback if needed
helm rollback zerotrustml

# View revision history
helm history zerotrustml
```

## 🐛 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -l app=zerotrustml

# View pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl get hpa
kubectl describe hpa zerotrustml-hpa

# Verify metrics-server is installed
kubectl get deployment metrics-server -n kube-system
```

### Prometheus Not Scraping

```bash
# Verify ServiceMonitor created
kubectl get servicemonitor

# Check Prometheus targets
# Access Prometheus UI and check Status -> Targets
```

## 📈 Performance Tuning

### High-Throughput Configuration

```yaml
replicaCount: 10
resources:
  requests:
    memory: "2Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "8000m"

config:
  performance:
    batchSize: 64
    compressionLevel: 1  # Faster, less compression

  networking:
    numShards: 50
    gossipFanout: 5
```

### Low-Latency Configuration

```yaml
config:
  performance:
    compressionAlgorithm: "lz4"  # Faster than zstd
    batchSize: 16                # Smaller batches

  networking:
    gossipInterval: 0.5  # More frequent gossip
```

## 🔒 Security Best Practices

1. **Enable TLS**: Always use `enableTLS: true` in production
2. **Network Policies**: Restrict pod-to-pod communication
3. **Resource Limits**: Prevent resource exhaustion
4. **Pod Security**: Use `runAsNonRoot` and `readOnlyRootFilesystem`
5. **Secrets Management**: Use Kubernetes Secrets or external secret managers

## 📚 Additional Resources

- [Phase 4 Deployment Guide](../docs/PHASE_4_DEPLOYMENT_GUIDE.md)
- [Phase 5 Design](../docs/PHASE_5_DESIGN.md)
- [System Architecture](../docs/SYSTEM_ARCHITECTURE.md)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)

## 🆘 Support

For issues or questions:
- GitHub Issues: https://github.com/luminous-dynamics/0TML/issues
- Documentation: https://docs.luminousdynamics.org
- Email: support@luminousdynamics.org

---

**Version**: 0.5.0
**Last Updated**: 2025-09-30
**Status**: Production Ready ✅