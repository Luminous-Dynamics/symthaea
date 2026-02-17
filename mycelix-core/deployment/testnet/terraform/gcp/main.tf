# Mycelix Testnet - Google Cloud Platform Configuration

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ============================================
# VARIABLES
# ============================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-east1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "testnet"
}

variable "kubernetes_version" {
  description = "GKE Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of nodes per zone"
  type        = number
  default     = 1
}

variable "node_machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "n2-standard-4"
}

# ============================================
# PROVIDER
# ============================================

provider "google" {
  project = var.project_id
  region  = var.region
}

# ============================================
# NETWORKING
# ============================================

resource "google_compute_network" "mycelix_vpc" {
  name                    = "mycelix-${var.environment}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

resource "google_compute_subnetwork" "mycelix_subnet" {
  name          = "mycelix-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.mycelix_vpc.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }

  private_ip_google_access = true
}

resource "google_compute_router" "mycelix_router" {
  name    = "mycelix-${var.environment}-router"
  region  = var.region
  network = google_compute_network.mycelix_vpc.id
  project = var.project_id
}

resource "google_compute_router_nat" "mycelix_nat" {
  name                               = "mycelix-${var.environment}-nat"
  router                             = google_compute_router.mycelix_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  project                            = var.project_id

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ============================================
# FIREWALL RULES
# ============================================

resource "google_compute_firewall" "mycelix_internal" {
  name    = "mycelix-${var.environment}-internal"
  network = google_compute_network.mycelix_vpc.id
  project = var.project_id

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_firewall" "mycelix_external" {
  name    = "mycelix-${var.environment}-external"
  network = google_compute_network.mycelix_vpc.id
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["9000", "9001", "443", "80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mycelix-node"]
}

# ============================================
# GKE CLUSTER
# ============================================

resource "google_container_cluster" "mycelix_cluster" {
  name     = "mycelix-${var.environment}"
  location = var.region
  project  = var.project_id

  # Use regional cluster for HA
  node_locations = [
    "${var.region}-b",
    "${var.region}-c",
    "${var.region}-d"
  ]

  network    = google_compute_network.mycelix_vpc.id
  subnetwork = google_compute_subnetwork.mycelix_subnet.id

  # Enable Autopilot for simplified management (optional)
  # enable_autopilot = true

  # Standard cluster configuration
  initial_node_count       = 1
  remove_default_node_pool = true

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }

  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  release_channel {
    channel = "REGULAR"
  }

  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T04:00:00Z"
      end_time   = "2024-01-01T08:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }
}

# ============================================
# NODE POOLS
# ============================================

resource "google_container_node_pool" "mycelix_nodes" {
  name       = "mycelix-${var.environment}-nodes"
  location   = var.region
  cluster    = google_container_cluster.mycelix_cluster.name
  project    = var.project_id
  node_count = var.node_count

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node_type   = "general"
    }

    tags = ["mycelix-node"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = var.node_count
    max_node_count = var.node_count * 3
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# ============================================
# CLOUD SQL (PostgreSQL)
# ============================================

resource "google_sql_database_instance" "mycelix_postgres" {
  name             = "mycelix-${var.environment}-postgres"
  database_version = "POSTGRES_16"
  region           = var.region
  project          = var.project_id

  settings {
    tier              = "db-custom-2-4096"
    availability_type = "REGIONAL"
    disk_size         = 50
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.mycelix_vpc.id
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }

    maintenance_window {
      day          = 7
      hour         = 4
      update_track = "stable"
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "mycelix_db" {
  name     = "mycelix_${var.environment}"
  instance = google_sql_database_instance.mycelix_postgres.name
  project  = var.project_id
}

resource "google_sql_user" "mycelix_user" {
  name     = "mycelix"
  instance = google_sql_database_instance.mycelix_postgres.name
  project  = var.project_id
  password = random_password.postgres_password.result
}

resource "random_password" "postgres_password" {
  length  = 32
  special = false
}

# ============================================
# MEMORYSTORE (Redis)
# ============================================

resource "google_redis_instance" "mycelix_redis" {
  name           = "mycelix-${var.environment}-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 1
  region         = var.region
  project        = var.project_id

  authorized_network = google_compute_network.mycelix_vpc.id

  redis_version = "REDIS_7_0"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 4
        minutes = 0
      }
    }
  }
}

# ============================================
# STATIC IP FOR LOAD BALANCER
# ============================================

resource "google_compute_global_address" "mycelix_ip" {
  name    = "mycelix-${var.environment}-ip"
  project = var.project_id
}

# ============================================
# OUTPUTS
# ============================================

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.mycelix_cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.mycelix_cluster.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "postgres_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.mycelix_postgres.connection_name
}

output "postgres_private_ip" {
  description = "Cloud SQL private IP"
  value       = google_sql_database_instance.mycelix_postgres.private_ip_address
}

output "redis_host" {
  description = "Redis host"
  value       = google_redis_instance.mycelix_redis.host
}

output "static_ip" {
  description = "Static IP for load balancer"
  value       = google_compute_global_address.mycelix_ip.address
}

output "vpc_id" {
  description = "VPC ID"
  value       = google_compute_network.mycelix_vpc.id
}
