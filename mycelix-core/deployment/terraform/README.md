# Mycelix Testnet Infrastructure

Terraform configuration for deploying the Mycelix testnet infrastructure on AWS.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Region (us-east-1)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                              VPC (10.0.0.0/16)                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                     Public Subnets                             │  │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │    │
│  │  │  │   AZ-1a     │  │   AZ-1b     │  │   AZ-1c     │            │  │    │
│  │  │  │ NAT Gateway │  │ NAT Gateway │  │ NAT Gateway │            │  │    │
│  │  │  │     ALB     │  │             │  │             │            │  │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                     Private Subnets                            │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │                    EKS Cluster                          │  │  │    │
│  │  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │  │    │
│  │  │  │  │  Validators  │  │   FL Nodes   │  │  Monitoring  │  │  │  │    │
│  │  │  │  │  Node Group  │  │  Node Group  │  │  Node Group  │  │  │  │    │
│  │  │  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                     Database Subnets                           │  │    │
│  │  │  ┌─────────────────────────┐  ┌─────────────────────────┐    │  │    │
│  │  │  │     RDS PostgreSQL      │  │    ElastiCache Redis    │    │  │    │
│  │  │  │       (Multi-AZ)        │  │                         │    │  │    │
│  │  │  └─────────────────────────┘  └─────────────────────────┘    │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  S3: Models  │  │ S3: Backups  │  │  S3: Logs    │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.5.0
3. **kubectl** for Kubernetes management
4. **helm** for chart deployments

## Quick Start

### 1. Initialize Terraform Backend

First, create the S3 bucket and DynamoDB table for state management:

```bash
# Create S3 bucket for state
aws s3 mb s3://mycelix-terraform-state --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket mycelix-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table for locking
aws dynamodb create-table \
  --table-name mycelix-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1
```

### 2. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

Required variables:
- `grafana_admin_password` - Generate with `openssl rand -base64 32`
- `letsencrypt_email` - Email for SSL certificate notifications

### 3. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan -out=tfplan

# Apply the configuration
terraform apply tfplan
```

### 4. Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig --name $(terraform output -raw cluster_name) --region us-east-1

# Verify connection
kubectl get nodes
```

## Module Structure

```
terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Input variables
├── terraform.tfvars.example # Example variable values
├── modules/
│   ├── vpc/               # VPC, subnets, NAT gateways
│   ├── eks/               # EKS cluster and node groups
│   ├── rds/               # PostgreSQL database
│   ├── elasticache/       # Redis cache
│   ├── s3/                # S3 buckets
│   ├── security-groups/   # Security groups
│   ├── iam-role/          # IAM roles
│   └── helm-releases/     # Helm chart deployments
```

## Environments

### Testnet (Default)
- Single NAT gateway
- Smaller instance sizes
- 7-day backup retention
- Spot instances enabled

```bash
terraform apply -var="environment=testnet"
```

### Staging
- Similar to production but smaller
- Multi-AZ database

```bash
terraform apply -var="environment=staging"
```

### Production
- Multiple NAT gateways
- Larger instance sizes
- 30-day backup retention
- Multi-AZ everything
- Deletion protection enabled

```bash
terraform apply -var="environment=production"
```

## Node Groups

### Validators
- Instance type: m6i.xlarge (4 vCPU, 16GB RAM)
- Purpose: Consensus and block production
- Scaling: 3-10 nodes

### FL Nodes
- Instance type: c6i.2xlarge (8 vCPU, 16GB RAM)
- Purpose: Federated learning aggregation
- Scaling: 2-20 nodes

### Monitoring
- Instance type: t3.large (2 vCPU, 8GB RAM)
- Purpose: Prometheus, Grafana, Loki
- Scaling: 1-3 nodes

## Deployed Services

After deployment, the following services are available:

| Service | Namespace | Description |
|---------|-----------|-------------|
| ingress-nginx | ingress-nginx | Ingress controller |
| cert-manager | cert-manager | TLS certificate management |
| prometheus | monitoring | Metrics collection |
| grafana | monitoring | Dashboards |
| loki | monitoring | Log aggregation |
| external-secrets | external-secrets | AWS Secrets Manager integration |

## Accessing Services

### Grafana
```bash
# Get the load balancer URL
kubectl get svc -n monitoring grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Default credentials
# Username: admin
# Password: (from terraform.tfvars grafana_admin_password)
```

### Prometheus
```bash
# Port forward for local access
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
# Access at http://localhost:9090
```

## Outputs

After deployment, important outputs are available:

```bash
# Get all outputs
terraform output

# Get specific output
terraform output -raw cluster_endpoint
terraform output -raw rds_endpoint
terraform output -raw kubeconfig_command
```

## Cost Estimation

Monthly costs (testnet configuration):

| Resource | Estimated Cost |
|----------|----------------|
| EKS Cluster | $72 |
| EC2 (Validators: 5 × m6i.xlarge) | $345 |
| EC2 (FL Nodes: 5 × c6i.2xlarge) | $612 |
| EC2 (Monitoring: 2 × t3.large) | $60 |
| RDS (db.t3.medium) | $50 |
| ElastiCache (cache.t3.medium) | $50 |
| NAT Gateway | $45 |
| S3 + Data Transfer | ~$50 |
| **Total** | **~$1,300/month** |

*Costs can be reduced with reserved instances or spot instances*

## Disaster Recovery

### Backup Strategy

- **RDS**: Automated daily snapshots, 7-day retention (testnet) / 30-day (production)
- **Redis**: Daily snapshots
- **S3**: Versioning enabled, cross-region replication (production)
- **EKS**: etcd encrypted, can restore from snapshot

### Recovery Procedures

See [DISASTER_RECOVERY.md](../../docs/operations/DISASTER_RECOVERY.md)

## Security

### Network Security
- All resources in private subnets
- Network policies restrict pod-to-pod communication
- Security groups follow least-privilege

### Encryption
- EKS secrets encrypted with KMS
- RDS storage encrypted
- S3 buckets encrypted at rest
- TLS for all external traffic

### IAM
- IRSA enabled for pod-level permissions
- Node groups have minimal required permissions

## Troubleshooting

### Terraform State Lock

If terraform fails with a state lock error:

```bash
terraform force-unlock <LOCK_ID>
```

### EKS Node Issues

```bash
# Check node status
kubectl describe node <node-name>

# Check system pods
kubectl get pods -n kube-system
```

### Database Connectivity

```bash
# Test from a pod
kubectl run -it --rm debug --image=postgres:15 -- psql -h <rds-endpoint> -U mycelix_admin -d mycelix
```

## Cleanup

To destroy all infrastructure:

```bash
# WARNING: This will delete all data!
terraform destroy
```

For production, deletion protection is enabled on RDS. Disable it first:

```bash
aws rds modify-db-instance \
  --db-instance-identifier <instance-id> \
  --no-deletion-protection
```

## Support

- **Documentation**: https://docs.mycelix.io/infrastructure
- **Issues**: https://github.com/mycelix/core/issues
- **Slack**: #platform-team

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-16 | Platform Team | Initial version |
