# Mycelix Testnet Infrastructure
#
# This Terraform configuration deploys the complete Mycelix testnet infrastructure
# on AWS (primary) with multi-region support.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket         = "mycelix-terraform-state"
    key            = "testnet/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "mycelix-terraform-locks"
  }
}

# Primary region provider
provider "aws" {
  region = var.primary_region

  default_tags {
    tags = {
      Project     = "Mycelix"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Secondary region provider (for DR)
provider "aws" {
  alias  = "secondary"
  region = var.secondary_region

  default_tags {
    tags = {
      Project     = "Mycelix"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Random suffix for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# Local values
locals {
  cluster_name = "mycelix-${var.environment}-${random_id.suffix.hex}"
  common_tags = {
    Cluster = local.cluster_name
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  name                 = local.cluster_name
  cidr                 = var.vpc_cidr
  azs                  = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets      = var.private_subnets
  public_subnets       = var.public_subnets
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Kubernetes tags for ELB discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb"                      = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"             = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  tags = local.common_tags
}

# EKS Cluster Module
module "eks" {
  source = "./modules/eks"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Node groups
  node_groups = {
    validators = {
      name           = "validators"
      instance_types = var.validator_instance_types
      min_size       = var.validator_min_nodes
      max_size       = var.validator_max_nodes
      desired_size   = var.validator_desired_nodes
      disk_size      = 250

      labels = {
        role = "validator"
      }

      taints = []
    }

    fl_nodes = {
      name           = "fl-nodes"
      instance_types = var.fl_instance_types
      min_size       = var.fl_min_nodes
      max_size       = var.fl_max_nodes
      desired_size   = var.fl_desired_nodes
      disk_size      = 500

      labels = {
        role = "fl-node"
      }

      taints = []
    }

    monitoring = {
      name           = "monitoring"
      instance_types = ["t3.large"]
      min_size       = 1
      max_size       = 3
      desired_size   = 2
      disk_size      = 100

      labels = {
        role = "monitoring"
      }

      taints = []
    }
  }

  # OIDC for IRSA
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  tags = local.common_tags
}

# RDS PostgreSQL for metadata
module "rds" {
  source = "./modules/rds"

  identifier = "${local.cluster_name}-db"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage

  db_name  = "mycelix"
  username = "mycelix_admin"
  port     = 5432

  multi_az               = var.environment == "production"
  db_subnet_group_name   = module.vpc.database_subnet_group
  vpc_security_group_ids = [module.security_groups.rds_sg_id]

  maintenance_window      = "Mon:00:00-Mon:03:00"
  backup_window           = "03:00-06:00"
  backup_retention_period = var.environment == "production" ? 30 : 7

  deletion_protection = var.environment == "production"

  performance_insights_enabled = true
  monitoring_interval          = 60
  monitoring_role_arn          = module.rds_monitoring_role.iam_role_arn

  parameters = [
    {
      name  = "log_connections"
      value = "1"
    },
    {
      name  = "log_disconnections"
      value = "1"
    }
  ]

  tags = local.common_tags
}

# ElastiCache Redis for caching and queues
module "elasticache" {
  source = "./modules/elasticache"

  cluster_id           = "${local.cluster_name}-redis"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  num_cache_nodes      = var.environment == "production" ? 3 : 1
  parameter_group_name = "default.redis7"

  subnet_group_name  = module.vpc.elasticache_subnet_group
  security_group_ids = [module.security_groups.redis_sg_id]

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window          = "05:00-09:00"

  tags = local.common_tags
}

# S3 Buckets for model storage and backups
module "s3" {
  source = "./modules/s3"

  buckets = {
    models = {
      name       = "mycelix-${var.environment}-models-${random_id.suffix.hex}"
      versioning = true
      lifecycle_rules = [
        {
          id      = "archive-old-models"
          enabled = true
          transition = [
            {
              days          = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }

    backups = {
      name       = "mycelix-${var.environment}-backups-${random_id.suffix.hex}"
      versioning = true
      lifecycle_rules = [
        {
          id      = "delete-old-backups"
          enabled = true
          expiration = {
            days = var.environment == "production" ? 365 : 30
          }
        }
      ]
    }

    logs = {
      name       = "mycelix-${var.environment}-logs-${random_id.suffix.hex}"
      versioning = false
      lifecycle_rules = [
        {
          id      = "delete-old-logs"
          enabled = true
          expiration = {
            days = 90
          }
        }
      ]
    }
  }

  tags = local.common_tags
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security-groups"

  name_prefix = local.cluster_name
  vpc_id      = module.vpc.vpc_id
  vpc_cidr    = var.vpc_cidr

  tags = local.common_tags
}

# IAM Roles
module "rds_monitoring_role" {
  source = "./modules/iam-role"

  name = "${local.cluster_name}-rds-monitoring"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
  ]

  tags = local.common_tags
}

# Kubernetes provider configuration
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# Helm provider configuration
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Helm releases for core infrastructure
module "helm_releases" {
  source = "./modules/helm-releases"

  depends_on = [module.eks]

  cluster_name = local.cluster_name
  environment  = var.environment

  # Ingress
  enable_ingress_nginx = true
  ingress_nginx_values = {
    "controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-type" = "nlb"
  }

  # Cert Manager
  enable_cert_manager = true
  cert_manager_email  = var.letsencrypt_email

  # Monitoring
  enable_prometheus = true
  prometheus_values = {
    "alertmanager.enabled" = "true"
    "server.retention"     = "15d"
  }

  enable_grafana = true
  grafana_values = {
    "adminPassword" = var.grafana_admin_password
  }

  # Logging
  enable_loki = true

  # External Secrets for AWS Secrets Manager integration
  enable_external_secrets = true

  tags = local.common_tags
}

# Outputs
output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache.cache_nodes[0].address
}

output "s3_bucket_models" {
  description = "S3 bucket for model storage"
  value       = module.s3.bucket_ids["models"]
}

output "s3_bucket_backups" {
  description = "S3 bucket for backups"
  value       = module.s3.bucket_ids["backups"]
}

output "kubeconfig_command" {
  description = "Command to update kubeconfig"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.primary_region}"
}
