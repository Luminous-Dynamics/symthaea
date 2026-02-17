# Mycelix Testnet - Amazon Web Services Configuration

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ============================================
# VARIABLES
# ============================================

variable "region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "testnet"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "node_instance_type" {
  description = "EC2 instance type for nodes"
  type        = string
  default     = "m5.xlarge"
}

# ============================================
# PROVIDER
# ============================================

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "mycelix"
      ManagedBy   = "terraform"
    }
  }
}

# ============================================
# DATA SOURCES
# ============================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ============================================
# VPC
# ============================================

resource "aws_vpc" "mycelix_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "mycelix-${var.environment}-vpc"
  }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.mycelix_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name                                           = "mycelix-${var.environment}-private-${count.index + 1}"
    "kubernetes.io/cluster/mycelix-${var.environment}" = "shared"
    "kubernetes.io/role/internal-elb"              = "1"
  }
}

resource "aws_subnet" "public" {
  count                   = 3
  vpc_id                  = aws_vpc.mycelix_vpc.id
  cidr_block              = "10.0.${count.index + 101}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                           = "mycelix-${var.environment}-public-${count.index + 1}"
    "kubernetes.io/cluster/mycelix-${var.environment}" = "shared"
    "kubernetes.io/role/elb"                       = "1"
  }
}

resource "aws_internet_gateway" "mycelix_igw" {
  vpc_id = aws_vpc.mycelix_vpc.id

  tags = {
    Name = "mycelix-${var.environment}-igw"
  }
}

resource "aws_eip" "nat" {
  count  = 1
  domain = "vpc"

  tags = {
    Name = "mycelix-${var.environment}-nat-eip"
  }

  depends_on = [aws_internet_gateway.mycelix_igw]
}

resource "aws_nat_gateway" "mycelix_nat" {
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "mycelix-${var.environment}-nat"
  }

  depends_on = [aws_internet_gateway.mycelix_igw]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.mycelix_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mycelix_igw.id
  }

  tags = {
    Name = "mycelix-${var.environment}-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.mycelix_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.mycelix_nat.id
  }

  tags = {
    Name = "mycelix-${var.environment}-private-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 3
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 3
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ============================================
# SECURITY GROUPS
# ============================================

resource "aws_security_group" "eks_cluster" {
  name_prefix = "mycelix-${var.environment}-eks-cluster-"
  vpc_id      = aws_vpc.mycelix_vpc.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mycelix-${var.environment}-eks-cluster-sg"
  }
}

resource "aws_security_group" "mycelix_nodes" {
  name_prefix = "mycelix-${var.environment}-nodes-"
  vpc_id      = aws_vpc.mycelix_vpc.id

  # P2P port
  ingress {
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # API port
  ingress {
    from_port   = 9001
    to_port     = 9001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Internal communication
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mycelix-${var.environment}-nodes-sg"
  }
}

# ============================================
# EKS CLUSTER
# ============================================

resource "aws_iam_role" "eks_cluster" {
  name = "mycelix-${var.environment}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_eks_cluster" "mycelix_cluster" {
  name     = "mycelix-${var.environment}"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
    security_group_ids      = [aws_security_group.eks_cluster.id]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]

  tags = {
    Name = "mycelix-${var.environment}-cluster"
  }
}

# ============================================
# EKS NODE GROUP
# ============================================

resource "aws_iam_role" "eks_nodes" {
  name = "mycelix-${var.environment}-eks-nodes-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_eks_node_group" "mycelix_nodes" {
  cluster_name    = aws_eks_cluster.mycelix_cluster.name
  node_group_name = "mycelix-${var.environment}-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = var.node_count
    max_size     = var.node_count * 3
    min_size     = var.node_count
  }

  instance_types = [var.node_instance_type]
  disk_size      = 100

  update_config {
    max_unavailable = 1
  }

  labels = {
    environment = var.environment
    node_type   = "general"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry,
  ]

  tags = {
    Name = "mycelix-${var.environment}-node-group"
  }
}

# ============================================
# RDS (PostgreSQL)
# ============================================

resource "aws_db_subnet_group" "mycelix_db" {
  name       = "mycelix-${var.environment}-db-subnet"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "mycelix-${var.environment}-db-subnet-group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "mycelix-${var.environment}-rds-"
  vpc_id      = aws_vpc.mycelix_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.mycelix_nodes.id]
  }

  tags = {
    Name = "mycelix-${var.environment}-rds-sg"
  }
}

resource "random_password" "postgres_password" {
  length  = 32
  special = false
}

resource "aws_db_instance" "mycelix_postgres" {
  identifier     = "mycelix-${var.environment}-postgres"
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = "db.t3.medium"

  allocated_storage     = 50
  max_allocated_storage = 200
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "mycelix_${var.environment}"
  username = "mycelix"
  password = random_password.postgres_password.result

  multi_az               = true
  db_subnet_group_name   = aws_db_subnet_group.mycelix_db.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "mycelix-${var.environment}-final-snapshot"

  performance_insights_enabled = true

  tags = {
    Name = "mycelix-${var.environment}-postgres"
  }
}

# ============================================
# ELASTICACHE (Redis)
# ============================================

resource "aws_elasticache_subnet_group" "mycelix_redis" {
  name       = "mycelix-${var.environment}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name_prefix = "mycelix-${var.environment}-redis-"
  vpc_id      = aws_vpc.mycelix_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.mycelix_nodes.id]
  }

  tags = {
    Name = "mycelix-${var.environment}-redis-sg"
  }
}

resource "aws_elasticache_replication_group" "mycelix_redis" {
  replication_group_id = "mycelix-${var.environment}-redis"
  description          = "Mycelix ${var.environment} Redis cluster"

  node_type            = "cache.t3.medium"
  num_cache_clusters   = 2
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.mycelix_redis.name
  security_group_ids = [aws_security_group.redis.id]

  automatic_failover_enabled = true
  multi_az_enabled           = true

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  maintenance_window       = "sun:05:00-sun:06:00"
  snapshot_retention_limit = 7
  snapshot_window          = "03:00-04:00"

  tags = {
    Name = "mycelix-${var.environment}-redis"
  }
}

# ============================================
# OUTPUTS
# ============================================

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.mycelix_cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "EKS cluster CA certificate"
  value       = aws_eks_cluster.mycelix_cluster.certificate_authority[0].data
  sensitive   = true
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.mycelix_cluster.name
}

output "postgres_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.mycelix_postgres.endpoint
}

output "postgres_password" {
  description = "RDS PostgreSQL password"
  value       = random_password.postgres_password.result
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.mycelix_redis.primary_endpoint_address
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.mycelix_vpc.id
}
