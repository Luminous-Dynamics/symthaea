# Mycelix Testnet - Microsoft Azure Configuration

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# ============================================
# VARIABLES
# ============================================

variable "location" {
  description = "Azure Region"
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "testnet"
}

variable "kubernetes_version" {
  description = "AKS Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of nodes"
  type        = number
  default     = 3
}

variable "node_vm_size" {
  description = "VM size for nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

# ============================================
# PROVIDER
# ============================================

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

# ============================================
# DATA SOURCES
# ============================================

data "azurerm_client_config" "current" {}

# ============================================
# RESOURCE GROUP
# ============================================

resource "azurerm_resource_group" "mycelix" {
  name     = "mycelix-${var.environment}-rg"
  location = var.location

  tags = {
    Environment = var.environment
    Project     = "mycelix"
    ManagedBy   = "terraform"
  }
}

# ============================================
# NETWORKING
# ============================================

resource "azurerm_virtual_network" "mycelix_vnet" {
  name                = "mycelix-${var.environment}-vnet"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  address_space       = ["10.0.0.0/16"]

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.mycelix.name
  virtual_network_name = azurerm_virtual_network.mycelix_vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_subnet" "database" {
  name                 = "database-subnet"
  resource_group_name  = azurerm_resource_group.mycelix.name
  virtual_network_name = azurerm_virtual_network.mycelix_vnet.name
  address_prefixes     = ["10.0.2.0/24"]

  delegation {
    name = "postgresql-delegation"

    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_subnet" "redis" {
  name                 = "redis-subnet"
  resource_group_name  = azurerm_resource_group.mycelix.name
  virtual_network_name = azurerm_virtual_network.mycelix_vnet.name
  address_prefixes     = ["10.0.3.0/24"]
}

resource "azurerm_network_security_group" "mycelix_nsg" {
  name                = "mycelix-${var.environment}-nsg"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name

  security_rule {
    name                       = "AllowP2P"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowAPI"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "9001"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 120
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.mycelix_nsg.id
}

# ============================================
# AKS CLUSTER
# ============================================

resource "azurerm_kubernetes_cluster" "mycelix_aks" {
  name                = "mycelix-${var.environment}-aks"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  dns_prefix          = "mycelix-${var.environment}"
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name                = "default"
    node_count          = var.node_count
    vm_size             = var.node_vm_size
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = var.node_count
    max_count           = var.node_count * 3
    os_disk_size_gb     = 100
    os_disk_type        = "Managed"

    node_labels = {
      environment = var.environment
      node_type   = "general"
    }

    upgrade_settings {
      max_surge = "1"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
    service_cidr      = "10.1.0.0/16"
    dns_service_ip    = "10.1.0.10"
  }

  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.mycelix.id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  maintenance_window {
    allowed {
      day   = "Sunday"
      hours = [4, 5, 6]
    }
  }

  tags = {
    Environment = var.environment
  }
}

# ============================================
# LOG ANALYTICS
# ============================================

resource "azurerm_log_analytics_workspace" "mycelix" {
  name                = "mycelix-${var.environment}-logs"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
  }
}

# ============================================
# POSTGRESQL FLEXIBLE SERVER
# ============================================

resource "azurerm_private_dns_zone" "postgres" {
  name                = "mycelix-${var.environment}.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.mycelix.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.mycelix_vnet.id
  resource_group_name   = azurerm_resource_group.mycelix.name
}

resource "random_password" "postgres_password" {
  length  = 32
  special = false
}

resource "azurerm_postgresql_flexible_server" "mycelix_postgres" {
  name                   = "mycelix-${var.environment}-postgres"
  resource_group_name    = azurerm_resource_group.mycelix.name
  location               = azurerm_resource_group.mycelix.location
  version                = "16"
  delegated_subnet_id    = azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = "mycelix"
  administrator_password = random_password.postgres_password.result
  zone                   = "1"

  storage_mb = 51200

  sku_name = "GP_Standard_D2s_v3"

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false

  high_availability {
    mode                      = "ZoneRedundant"
    standby_availability_zone = "2"
  }

  maintenance_window {
    day_of_week  = 0
    start_hour   = 4
    start_minute = 0
  }

  tags = {
    Environment = var.environment
  }

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "mycelix_db" {
  name      = "mycelix_${var.environment}"
  server_id = azurerm_postgresql_flexible_server.mycelix_postgres.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# ============================================
# REDIS CACHE
# ============================================

resource "azurerm_redis_cache" "mycelix_redis" {
  name                = "mycelix-${var.environment}-redis"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  capacity            = 1
  family              = "P"
  sku_name            = "Premium"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  redis_configuration {
    maxmemory_reserved = 50
    maxmemory_delta    = 50
    maxmemory_policy   = "allkeys-lru"
  }

  patch_schedule {
    day_of_week    = "Sunday"
    start_hour_utc = 4
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_private_endpoint" "redis" {
  name                = "mycelix-${var.environment}-redis-pe"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  subnet_id           = azurerm_subnet.redis.id

  private_service_connection {
    name                           = "redis-privateserviceconnection"
    private_connection_resource_id = azurerm_redis_cache.mycelix_redis.id
    subresource_names              = ["redisCache"]
    is_manual_connection           = false
  }

  tags = {
    Environment = var.environment
  }
}

# ============================================
# KEY VAULT
# ============================================

resource "azurerm_key_vault" "mycelix" {
  name                        = "mycelix-${var.environment}-kv"
  location                    = azurerm_resource_group.mycelix.location
  resource_group_name         = azurerm_resource_group.mycelix.name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false

  sku_name = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Create", "Delete", "Update",
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete",
    ]
  }

  # AKS access
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_kubernetes_cluster.mycelix_aks.key_vault_secrets_provider[0].secret_identity[0].object_id

    secret_permissions = [
      "Get", "List",
    ]
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-password"
  value        = random_password.postgres_password.result
  key_vault_id = azurerm_key_vault.mycelix.id
}

# ============================================
# PUBLIC IP FOR LOAD BALANCER
# ============================================

resource "azurerm_public_ip" "mycelix_ip" {
  name                = "mycelix-${var.environment}-ip"
  location            = azurerm_resource_group.mycelix.location
  resource_group_name = azurerm_resource_group.mycelix.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = {
    Environment = var.environment
  }
}

# ============================================
# OUTPUTS
# ============================================

output "cluster_endpoint" {
  description = "AKS cluster endpoint"
  value       = azurerm_kubernetes_cluster.mycelix_aks.fqdn
  sensitive   = true
}

output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.mycelix_aks.name
}

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.mycelix.name
}

output "postgres_fqdn" {
  description = "PostgreSQL FQDN"
  value       = azurerm_postgresql_flexible_server.mycelix_postgres.fqdn
}

output "postgres_password" {
  description = "PostgreSQL password"
  value       = random_password.postgres_password.result
  sensitive   = true
}

output "redis_hostname" {
  description = "Redis hostname"
  value       = azurerm_redis_cache.mycelix_redis.hostname
}

output "redis_primary_access_key" {
  description = "Redis primary access key"
  value       = azurerm_redis_cache.mycelix_redis.primary_access_key
  sensitive   = true
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.mycelix.vault_uri
}

output "public_ip" {
  description = "Public IP address"
  value       = azurerm_public_ip.mycelix_ip.ip_address
}

output "kube_config" {
  description = "Kubernetes config for kubectl"
  value       = azurerm_kubernetes_cluster.mycelix_aks.kube_config_raw
  sensitive   = true
}
