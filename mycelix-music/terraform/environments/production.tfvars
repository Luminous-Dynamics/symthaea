# Production Environment Configuration

project_id  = "mycelix-production"
region      = "us-central1"
environment = "production"
domain      = "mycelix.io"

# Production-sized instances
db_tier              = "db-custom-2-7680"
redis_memory_size_gb = 4
gke_node_count       = 3
gke_machine_type     = "e2-standard-4"
