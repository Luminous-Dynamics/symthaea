# Staging Environment Configuration

project_id  = "mycelix-staging"
region      = "us-central1"
environment = "staging"
domain      = "staging.mycelix.io"

# Smaller instances for staging
db_tier              = "db-custom-1-3840"
redis_memory_size_gb = 1
gke_node_count       = 2
gke_machine_type     = "e2-standard-2"
