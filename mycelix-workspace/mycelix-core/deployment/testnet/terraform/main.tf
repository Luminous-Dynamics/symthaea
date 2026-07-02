# Mycelix Testnet - Terraform Main Configuration
# This module orchestrates the deployment across multiple cloud providers

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Backend configuration - uncomment and configure for your environment
  # backend "gcs" {
  #   bucket = "mycelix-terraform-state"
  #   prefix = "testnet"
  # }
  # backend "s3" {
  #   bucket = "mycelix-terraform-state"
  #   key    = "testnet/terraform.tfstate"
  #   region = "us-east-1"
  # }
  # backend "azurerm" {
  #   resource_group_name  = "mycelix-terraform"
  #   storage_account_name = "mycelixtfstate"
  #   container_name       = "tfstate"
  #   key                  = "testnet.terraform.tfstate"
  # }
}

# ============================================
# VARIABLES
# ============================================

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "testnet"
}

variable "cloud_provider" {
  description = "Cloud provider to use (gcp, aws, azure)"
  type        = string
  default     = "gcp"

  validation {
    condition     = contains(["gcp", "aws", "azure"], var.cloud_provider)
    error_message = "cloud_provider must be one of: gcp, aws, azure"
  }
}

variable "region" {
  description = "Primary region for deployment"
  type        = string
  default     = "us-east1"
}

variable "project_id" {
  description = "GCP Project ID (required for GCP)"
  type        = string
  default     = ""
}

variable "kubernetes_version" {
  description = "Kubernetes version for managed cluster"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of Kubernetes nodes"
  type        = number
  default     = 3
}

variable "node_machine_type" {
  description = "Machine type for Kubernetes nodes"
  type        = string
  default     = "n2-standard-4"
}

variable "fl_node_count" {
  description = "Number of FL seed nodes"
  type        = number
  default     = 5
}

variable "enable_monitoring" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Domain name for the testnet"
  type        = string
  default     = "testnet.mycelix.network"
}

# ============================================
# RANDOM SECRETS
# ============================================

resource "random_password" "postgres_password" {
  length  = 32
  special = false
}

resource "random_password" "grafana_password" {
  length  = 24
  special = false
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = false
}

# ============================================
# CONDITIONAL MODULE LOADING
# ============================================

module "gcp" {
  source = "./gcp"
  count  = var.cloud_provider == "gcp" ? 1 : 0

  project_id         = var.project_id
  region             = var.region
  environment        = var.environment
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  node_machine_type  = var.node_machine_type
}

module "aws" {
  source = "./aws"
  count  = var.cloud_provider == "aws" ? 1 : 0

  region             = var.region
  environment        = var.environment
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  node_instance_type = var.node_machine_type
}

module "azure" {
  source = "./azure"
  count  = var.cloud_provider == "azure" ? 1 : 0

  location           = var.region
  environment        = var.environment
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  node_vm_size       = var.node_machine_type
}

# ============================================
# OUTPUTS
# ============================================

output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value = var.cloud_provider == "gcp" ? (
    length(module.gcp) > 0 ? module.gcp[0].cluster_endpoint : null
  ) : var.cloud_provider == "aws" ? (
    length(module.aws) > 0 ? module.aws[0].cluster_endpoint : null
  ) : (
    length(module.azure) > 0 ? module.azure[0].cluster_endpoint : null
  )
  sensitive = true
}

output "postgres_password" {
  description = "PostgreSQL password"
  value       = random_password.postgres_password.result
  sensitive   = true
}

output "grafana_password" {
  description = "Grafana admin password"
  value       = random_password.grafana_password.result
  sensitive   = true
}

output "jwt_secret" {
  description = "JWT secret for API authentication"
  value       = random_password.jwt_secret.result
  sensitive   = true
}

output "deployment_instructions" {
  description = "Next steps for deployment"
  value       = <<-EOT
    Mycelix Testnet Infrastructure Provisioned!

    Next steps:
    1. Configure kubectl:
       ${var.cloud_provider == "gcp" ? "gcloud container clusters get-credentials mycelix-testnet --region ${var.region}" : ""}
       ${var.cloud_provider == "aws" ? "aws eks update-kubeconfig --name mycelix-testnet --region ${var.region}" : ""}
       ${var.cloud_provider == "azure" ? "az aks get-credentials --resource-group mycelix-testnet --name mycelix-testnet" : ""}

    2. Deploy Kubernetes manifests:
       kubectl apply -k ../kubernetes/

    3. Access Grafana:
       kubectl port-forward svc/grafana 3000:3000 -n mycelix-testnet

    4. Get the external IP:
       kubectl get svc bootstrap-external -n mycelix-testnet
  EOT
}
