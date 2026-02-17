# Mycelix Mail Infrastructure
# Terraform configuration for cloud deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
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
    key            = "mail/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "domain" {
  description = "Primary domain for the application"
  type        = string
  default     = "mycelix.mail"
}

variable "replica_count" {
  description = "Number of API replicas"
  type        = number
  default     = 3
}

variable "enable_monitoring" {
  description = "Enable Prometheus/Grafana monitoring"
  type        = bool
  default     = true
}

# Random secret generation
resource "random_password" "jwt_secret" {
  length  = 32
  special = false
}

resource "random_password" "postgres_password" {
  length  = 24
  special = true
}

# Kubernetes namespace
resource "kubernetes_namespace" "mycelix_mail" {
  metadata {
    name = "mycelix-mail"
    labels = {
      environment = var.environment
      app         = "mycelix-mail"
    }
  }
}

# Secrets
resource "kubernetes_secret" "mycelix_secrets" {
  metadata {
    name      = "mycelix-secrets"
    namespace = kubernetes_namespace.mycelix_mail.metadata[0].name
  }

  data = {
    jwt-secret   = random_password.jwt_secret.result
    database-url = "postgres://mycelix:${random_password.postgres_password.result}@postgres:5432/mycelix_mail"
    redis-url    = "redis://redis:6379"
  }
}

resource "kubernetes_secret" "postgres_secret" {
  metadata {
    name      = "postgres-secret"
    namespace = kubernetes_namespace.mycelix_mail.metadata[0].name
  }

  data = {
    password = random_password.postgres_password.result
  }
}

# ConfigMap
resource "kubernetes_config_map" "mycelix_config" {
  metadata {
    name      = "mycelix-config"
    namespace = kubernetes_namespace.mycelix_mail.metadata[0].name
  }

  data = {
    "config.toml" = templatefile("${path.module}/templates/config.toml.tpl", {
      environment = var.environment
      domain      = var.domain
    })
  }
}

# Helm releases
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = "1.14.0"
  namespace  = "cert-manager"

  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }
}

resource "helm_release" "ingress_nginx" {
  name       = "ingress-nginx"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.9.0"
  namespace  = "ingress-nginx"

  create_namespace = true

  set {
    name  = "controller.metrics.enabled"
    value = "true"
  }

  set {
    name  = "controller.metrics.serviceMonitor.enabled"
    value = var.enable_monitoring
  }
}

resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0

  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "55.0.0"
  namespace  = "monitoring"

  create_namespace = true

  values = [
    file("${path.module}/values/prometheus.yaml")
  ]
}

# Outputs
output "namespace" {
  value = kubernetes_namespace.mycelix_mail.metadata[0].name
}

output "jwt_secret_name" {
  value     = kubernetes_secret.mycelix_secrets.metadata[0].name
  sensitive = true
}

output "ingress_class" {
  value = "nginx"
}
