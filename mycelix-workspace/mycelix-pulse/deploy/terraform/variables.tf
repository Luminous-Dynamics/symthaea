# Terraform Variables for Mycelix Mail

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "mycelix-cluster"
}

variable "region" {
  description = "Cloud region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "node_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 3
}

variable "node_size" {
  description = "Size of worker nodes"
  type        = string
  default     = "t3.large"
}

variable "postgres_storage_size" {
  description = "Storage size for PostgreSQL in GB"
  type        = number
  default     = 50
}

variable "redis_memory_limit" {
  description = "Memory limit for Redis in MB"
  type        = number
  default     = 256
}

variable "api_replicas_min" {
  description = "Minimum API replicas"
  type        = number
  default     = 3
}

variable "api_replicas_max" {
  description = "Maximum API replicas"
  type        = number
  default     = 10
}

variable "enable_holochain" {
  description = "Deploy Holochain conductor"
  type        = bool
  default     = true
}

variable "enable_ml_service" {
  description = "Deploy ML/LLM service"
  type        = bool
  default     = true
}

variable "ml_model" {
  description = "LLM model to use"
  type        = string
  default     = "llama3.2"
}

variable "backup_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "mycelix-mail"
    ManagedBy   = "terraform"
    Environment = "production"
  }
}
