# Mycelix Testnet - Terraform Variables

# ============================================
# GENERAL CONFIGURATION
# ============================================

variable "environment" {
  description = "Environment name (e.g., testnet, mainnet)"
  type        = string
  default     = "testnet"
}

variable "cloud_provider" {
  description = "Cloud provider to use"
  type        = string
  default     = "gcp"

  validation {
    condition     = contains(["gcp", "aws", "azure"], var.cloud_provider)
    error_message = "cloud_provider must be one of: gcp, aws, azure"
  }
}

# ============================================
# GCP CONFIGURATION
# ============================================

variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP Region"
  type        = string
  default     = "us-east1"
}

# ============================================
# AWS CONFIGURATION
# ============================================

variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

# ============================================
# AZURE CONFIGURATION
# ============================================

variable "azure_location" {
  description = "Azure Location/Region"
  type        = string
  default     = "eastus"
}

variable "azure_subscription_id" {
  description = "Azure Subscription ID"
  type        = string
  default     = ""
}

# ============================================
# KUBERNETES CONFIGURATION
# ============================================

variable "kubernetes_version" {
  description = "Kubernetes version for managed cluster"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of Kubernetes worker nodes"
  type        = number
  default     = 3

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 100
    error_message = "node_count must be between 1 and 100"
  }
}

variable "node_machine_type" {
  description = "Machine type/size for Kubernetes nodes"
  type        = string
  default     = "n2-standard-4"
}

variable "node_disk_size_gb" {
  description = "Disk size for nodes in GB"
  type        = number
  default     = 100
}

# ============================================
# MYCELIX CONFIGURATION
# ============================================

variable "fl_node_count" {
  description = "Number of FL seed nodes to deploy"
  type        = number
  default     = 5

  validation {
    condition     = var.fl_node_count >= 3 && var.fl_node_count <= 50
    error_message = "fl_node_count must be between 3 and 50 for proper federated learning"
  }
}

variable "enable_faucet" {
  description = "Enable the testnet faucet service"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus + Grafana)"
  type        = bool
  default     = true
}

# ============================================
# NETWORKING
# ============================================

variable "domain_name" {
  description = "Domain name for the testnet"
  type        = string
  default     = "testnet.mycelix.network"
}

variable "enable_ssl" {
  description = "Enable SSL/TLS certificates"
  type        = bool
  default     = true
}

variable "ssl_email" {
  description = "Email for Let's Encrypt certificates"
  type        = string
  default     = "ops@mycelix.network"
}

# ============================================
# DATABASE CONFIGURATION
# ============================================

variable "postgres_tier" {
  description = "PostgreSQL instance tier/size"
  type        = string
  default     = "db-custom-2-4096"
}

variable "postgres_disk_size_gb" {
  description = "PostgreSQL disk size in GB"
  type        = number
  default     = 50
}

variable "redis_memory_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

# ============================================
# TAGS
# ============================================

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "mycelix"
    ManagedBy   = "terraform"
  }
}
