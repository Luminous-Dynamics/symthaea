# Mycelix Testnet Infrastructure Variables

# General
variable "environment" {
  description = "Environment name (testnet, staging, production)"
  type        = string
  default     = "testnet"

  validation {
    condition     = contains(["testnet", "staging", "production"], var.environment)
    error_message = "Environment must be testnet, staging, or production."
  }
}

variable "primary_region" {
  description = "Primary AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "secondary_region" {
  description = "Secondary AWS region for DR"
  type        = string
  default     = "us-west-2"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.29"
}

# Validator Node Group
variable "validator_instance_types" {
  description = "Instance types for validator nodes"
  type        = list(string)
  default     = ["m6i.xlarge"]
}

variable "validator_min_nodes" {
  description = "Minimum number of validator nodes"
  type        = number
  default     = 3
}

variable "validator_max_nodes" {
  description = "Maximum number of validator nodes"
  type        = number
  default     = 10
}

variable "validator_desired_nodes" {
  description = "Desired number of validator nodes"
  type        = number
  default     = 5
}

# FL Node Group
variable "fl_instance_types" {
  description = "Instance types for FL nodes"
  type        = list(string)
  default     = ["c6i.2xlarge"]
}

variable "fl_min_nodes" {
  description = "Minimum number of FL nodes"
  type        = number
  default     = 2
}

variable "fl_max_nodes" {
  description = "Maximum number of FL nodes"
  type        = number
  default     = 20
}

variable "fl_desired_nodes" {
  description = "Desired number of FL nodes"
  type        = number
  default     = 5
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS (GB)"
  type        = number
  default     = 50
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS autoscaling (GB)"
  type        = number
  default     = 200
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

# Monitoring Configuration
variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "letsencrypt_email" {
  description = "Email for Let's Encrypt certificates"
  type        = string
}

# Domain Configuration
variable "domain_name" {
  description = "Base domain name for the testnet"
  type        = string
  default     = "testnet.mycelix.io"
}

# Alerting
variable "pagerduty_integration_key" {
  description = "PagerDuty integration key for alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

# Cost Management
variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = true
}

variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 5000
}
