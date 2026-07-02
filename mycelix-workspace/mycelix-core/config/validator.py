# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Configuration Validator
================================

Unified configuration loading and validation for Mycelix Byzantine-resistant
Federated Learning system.

This module:
- Loads configuration from environment variables, YAML files, and JSON
- Validates against the unified JSON Schema
- Warns about suboptimal settings
- Fails fast on invalid or dangerous configurations

Usage:
    from config.validator import load_and_validate_config, MycelixConfig

    # Load and validate (will exit on critical errors)
    config = load_and_validate_config()

    # Or with custom paths
    config = load_and_validate_config(
        env_file=".env.optimal",
        config_file="production_config.yaml"
    )

Critical Parameters:
    LABEL_SKEW_COS_MIN must be -0.5, NOT -0.3
    Using -0.3 causes 16x performance degradation (57-92% FP vs 3.55-7.1% FP)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    import jsonschema
    from jsonschema import Draft202012Validator, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None
    Draft202012Validator = None
    ValidationError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mycelix.config")


# =============================================================================
# Critical Parameter Definitions
# =============================================================================

CRITICAL_PARAMS = {
    "LABEL_SKEW_COS_MIN": {
        "optimal": -0.5,
        "dangerous": [-0.3, -0.4],
        "impact": "16x performance degradation (57-92% FP vs 3.55-7.1% FP)",
        "severity": "critical",
    },
    "BEHAVIOR_RECOVERY_THRESHOLD": {
        "optimal": 2,
        "dangerous": [3, 4, 5],
        "impact": "Slower honest node recovery, reduced system throughput",
        "severity": "high",
    },
    "BEHAVIOR_RECOVERY_BONUS": {
        "optimal": 0.12,
        "dangerous": [0.08, 0.10],
        "impact": "Permanent honest node exclusion with slow recovery",
        "severity": "high",
    },
}

# Dataset-specific optimal parameters
DATASET_OPTIMAL_PARAMS = {
    "cifar10": {
        "behavior_recovery_threshold": 2,
        "behavior_recovery_bonus": 0.12,
        "label_skew_cos_min": -0.5,
        "label_skew_cos_max": 0.95,
        "committee_reject_floor": 0.25,
        "reputation_floor": 0.01,
    },
    "emnist_balanced": {
        "behavior_recovery_threshold": 3,
        "behavior_recovery_bonus": 0.10,
        "label_skew_cos_min": -0.4,
        "label_skew_cos_max": 0.96,
        "committee_reject_floor": 0.30,
        "reputation_floor": 0.01,
    },
    "breast_cancer": {
        "behavior_recovery_threshold": 2,
        "behavior_recovery_bonus": 0.12,
        "label_skew_cos_min": -0.5,
        "label_skew_cos_max": 0.95,
        "committee_reject_floor": 0.25,
        "reputation_floor": 0.01,
    },
    "default": {
        "behavior_recovery_threshold": 3,
        "behavior_recovery_bonus": 0.10,
        "label_skew_cos_min": -0.4,
        "label_skew_cos_max": 0.96,
        "committee_reject_floor": 0.28,
        "reputation_floor": 0.01,
    },
}


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ByzantineDetectionConfig:
    """Byzantine detection parameters."""
    label_skew_cos_min: float = -0.5
    label_skew_cos_max: float = 0.95
    behavior_recovery_threshold: int = 2
    behavior_recovery_bonus: float = 0.12
    reputation_floor: float = 0.01
    reputation_decay: float = 0.95
    reputation_initial: float = 0.5
    pogq_threshold: float = 0.7
    pogq_test_size: float = 0.1
    committee_reject_floor: float = 0.25
    anomaly_sensitivity: float = 2.5
    anomaly_window_size: int = 10
    byzantine_penalty: float = -0.2
    false_positive_threshold: float = 0.01
    zero_norm_threshold: float = 1e-6
    backdoor_spike_threshold: float = 3.0
    hybrid_suspicion_threshold: float = 0.35
    hybrid_pogq_weight: float = 0.35
    hybrid_cos_weight: float = 0.30
    hybrid_committee_weight: float = 0.15
    hybrid_norm_weight: float = 0.10
    hybrid_cluster_weight: float = 0.10


@dataclass
class FederatedLearningConfig:
    """Federated learning parameters."""
    aggregation: str = "krum"
    num_rounds: int = 100
    num_clients: int = 10
    clients_per_round: int = 5
    min_clients: int = 5
    local_epochs: int = 10
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0001
    gradient_clipping: float = 10.0
    data_distribution_type: str = "dirichlet"
    data_distribution_alpha: float = 1.0


@dataclass
class HolochainConfig:
    """Holochain DHT configuration."""
    enabled: bool = False
    conductor_url: str = "ws://localhost:8888"
    admin_port: int = 8888
    app_port: int = 8889
    dna_path: str = "./holochain/dna/zerotrustml.dna"
    bootstrap_url: str = "https://bootstrap-staging.holo.host"
    signal_url: str = "wss://signal.holo.host"
    gossip_loop_iteration_delay_ms: int = 1000


@dataclass
class SecurityConfig:
    """Security settings."""
    require_signatures: bool = True
    signature_algorithm: str = "ed25519"
    encrypt_gradients: bool = True
    encryption_algorithm: str = "chacha20-poly1305"
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    tls_enabled: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and logging."""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    tensorboard: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_retention_days: int = 30
    checkpoint_interval: int = 10


@dataclass
class MycelixConfig:
    """Complete Mycelix configuration."""
    byzantine_detection: ByzantineDetectionConfig = field(default_factory=ByzantineDetectionConfig)
    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    holochain: HolochainConfig = field(default_factory=HolochainConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # System
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    # Dataset
    dataset_name: str = "CIFAR10"
    data_dir: str = "./data"
    batch_size: int = 32

    # Node
    node_id: str = "fl-node-001"
    node_region: str = "us-west-2"

    # Deployment
    deployment_mode: str = "development"
    dev_mode: bool = False


# =============================================================================
# Validation Functions
# =============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigValidationWarning(UserWarning):
    """Warning for suboptimal configuration."""
    pass


def load_env_file(path: Union[str, Path]) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    env_vars = {}
    path = Path(path)

    if not path.exists():
        logger.warning(f"Environment file not found: {path}")
        return env_vars

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                # Handle export statements
                if line.startswith("export "):
                    line = line[7:]
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Remove inline comments (# comment after value)
                if "#" in value:
                    value = value.split("#")[0].strip()
                env_vars[key] = value

    return env_vars


def load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not available, skipping YAML config")
        return {}

    path = Path(path)
    if not path.exists():
        logger.warning(f"YAML config file not found: {path}")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_json_schema(schema_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the JSON schema for validation."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.json"
    else:
        schema_path = Path(schema_path)

    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}")
        return {}

    with open(schema_path) as f:
        return json.load(f)


def validate_critical_parameters(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate critical parameters and return errors and warnings.

    Returns:
        Tuple of (critical_errors, warnings)
    """
    errors = []
    warnings_list = []

    byzantine = config.get("byzantine_detection", {})

    # Check LABEL_SKEW_COS_MIN
    cos_min = byzantine.get("label_skew_cos_min")
    if cos_min is not None:
        critical = CRITICAL_PARAMS["LABEL_SKEW_COS_MIN"]
        if cos_min in critical["dangerous"]:
            errors.append(
                f"CRITICAL: label_skew_cos_min={cos_min} causes {critical['impact']}! "
                f"Must be {critical['optimal']}. Source .env.optimal before running."
            )
        elif cos_min != critical["optimal"]:
            warnings_list.append(
                f"label_skew_cos_min={cos_min} differs from optimal {critical['optimal']}"
            )

    # Check BEHAVIOR_RECOVERY_THRESHOLD
    threshold = byzantine.get("behavior_recovery_threshold")
    if threshold is not None:
        critical = CRITICAL_PARAMS["BEHAVIOR_RECOVERY_THRESHOLD"]
        if threshold in critical["dangerous"]:
            warnings_list.append(
                f"behavior_recovery_threshold={threshold} is suboptimal. "
                f"Impact: {critical['impact']}. Recommended: {critical['optimal']}"
            )

    # Check BEHAVIOR_RECOVERY_BONUS
    bonus = byzantine.get("behavior_recovery_bonus")
    if bonus is not None:
        critical = CRITICAL_PARAMS["BEHAVIOR_RECOVERY_BONUS"]
        if bonus in critical["dangerous"]:
            warnings_list.append(
                f"behavior_recovery_bonus={bonus} is suboptimal. "
                f"Impact: {critical['impact']}. Recommended: {critical['optimal']}"
            )

    # Check hybrid weights sum to ~1.0
    weights = [
        byzantine.get("hybrid_pogq_weight", 0.35),
        byzantine.get("hybrid_cos_weight", 0.30),
        byzantine.get("hybrid_committee_weight", 0.15),
        byzantine.get("hybrid_norm_weight", 0.10),
        byzantine.get("hybrid_cluster_weight", 0.10),
    ]
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        warnings_list.append(
            f"Hybrid detection weights sum to {weight_sum:.2f}, should be 1.0"
        )

    # Check security settings in production
    deployment = config.get("deployment", {})
    security = config.get("security", {})
    if deployment.get("mode") == "production":
        if not security.get("require_signatures", True):
            errors.append(
                "CRITICAL: require_signatures is disabled in production mode! "
                "This allows message spoofing attacks."
            )
        if not security.get("encrypt_gradients", True):
            warnings_list.append(
                "encrypt_gradients is disabled in production. "
                "Model updates are exposed to eavesdropping."
            )

    return errors, warnings_list


def validate_against_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate configuration against JSON Schema."""
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema not available, skipping schema validation")
        return []

    errors = []
    validator = Draft202012Validator(schema)

    for error in validator.iter_errors(config):
        path = ".".join(str(p) for p in error.absolute_path) or "root"
        errors.append(f"Schema validation error at '{path}': {error.message}")

    return errors


def get_dataset_optimal_params(dataset: str) -> Dict[str, Any]:
    """Get optimal parameters for a specific dataset."""
    dataset_lower = dataset.lower().replace("-", "_")

    # Map common variations
    dataset_map = {
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
        "emnist": "emnist_balanced",
        "emnist_balanced": "emnist_balanced",
        "femnist": "emnist_balanced",
        "breast_cancer": "breast_cancer",
        "breastcancer": "breast_cancer",
        "mnist": "cifar10",  # Use CIFAR-10 params for MNIST
    }

    mapped = dataset_map.get(dataset_lower, "default")
    return DATASET_OPTIMAL_PARAMS.get(mapped, DATASET_OPTIMAL_PARAMS["default"])


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    Later configs override earlier ones.
    """
    result = {}

    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result


def env_to_config(env_vars: Dict[str, str]) -> Dict[str, Any]:
    """Convert environment variables to configuration dictionary."""
    config = {"byzantine_detection": {}}

    # Map environment variables to config structure
    env_mapping = {
        # Byzantine detection
        "LABEL_SKEW_COS_MIN": ("byzantine_detection", "label_skew_cos_min", float),
        "LABEL_SKEW_COS_MAX": ("byzantine_detection", "label_skew_cos_max", float),
        "BEHAVIOR_RECOVERY_THRESHOLD": ("byzantine_detection", "behavior_recovery_threshold", int),
        "BEHAVIOR_RECOVERY_BONUS": ("byzantine_detection", "behavior_recovery_bonus", float),
        "REPUTATION_FLOOR": ("byzantine_detection", "reputation_floor", float),
        "COMMITTEE_REJECT_FLOOR": ("byzantine_detection", "committee_reject_floor", float),
        "POGQ_THRESHOLD": ("byzantine_detection", "pogq_threshold", float),
        "HYBRID_SUSPICION_THRESHOLD": ("byzantine_detection", "hybrid_suspicion_threshold", float),
        "ZERO_NORM_THRESHOLD": ("byzantine_detection", "zero_norm_threshold", float),
        "BACKDOOR_SPIKE_THRESHOLD": ("byzantine_detection", "backdoor_spike_threshold", float),
        "ANOMALY_SENSITIVITY": ("byzantine_detection", "anomaly_sensitivity", float),

        # Hybrid weights
        "HYBRID_POGQ_WEIGHT": ("byzantine_detection", "hybrid_pogq_weight", float),
        "HYBRID_COS_WEIGHT": ("byzantine_detection", "hybrid_cos_weight", float),
        "HYBRID_COMMITTEE_WEIGHT": ("byzantine_detection", "hybrid_committee_weight", float),
        "HYBRID_NORM_WEIGHT": ("byzantine_detection", "hybrid_norm_weight", float),
        "HYBRID_CLUSTER_WEIGHT": ("byzantine_detection", "hybrid_cluster_weight", float),

        # System
        "SEED": ("system", "seed", int),
        "DEVICE": ("system", "device", str),
        "NUM_WORKERS": ("system", "num_workers", int),

        # Dataset
        "DATASET_NAME": ("dataset", "name", str),
        "DATA_DIR": ("dataset", "data_dir", str),
        "BATCH_SIZE": ("dataset", "batch_size", int),

        # Monitoring
        "LOG_LEVEL": ("monitoring", "log_level", str),
        "PROMETHEUS_PORT": ("monitoring", "prometheus_port", int),

        # Node
        "NODE_ID": ("node", "id", str),
        "NODE_REGION": ("node", "region", str),

        # Deployment
        "DEPLOYMENT_MODE": ("deployment", "mode", str),
        "DEV_MODE": ("deployment", "dev_mode", lambda x: x.lower() in ("true", "1", "yes")),

        # Database (ZeroTrustML naming)
        "ZEROTRUSTML_DB_HOST": ("credits", "postgresql", "host", str),
        "ZEROTRUSTML_DB_NAME": ("credits", "postgresql", "database", str),
        "ZEROTRUSTML_DB_USER": ("credits", "postgresql", "user", str),
        "ZEROTRUSTML_LOG_LEVEL": ("monitoring", "log_level", str),

        # TrustML naming (alternative)
        "TRUSTML_DB_PASSWORD": ("credits", "postgresql", "password", str),
        "TRUSTML_NODE_ID": ("node", "id", str),
        "TRUSTML_LOG_LEVEL": ("monitoring", "log_level", str),
        "TRUSTML_POGQ_THRESHOLD": ("byzantine_detection", "pogq_threshold", float),
        "TRUSTML_REPUTATION_DECAY": ("byzantine_detection", "reputation_decay", float),
        "TRUSTML_ANOMALY_SENSITIVITY": ("byzantine_detection", "anomaly_sensitivity", float),
    }

    for env_key, mapping in env_mapping.items():
        if env_key in env_vars:
            try:
                *path, type_fn = mapping
                value = type_fn(env_vars[env_key])

                # Navigate to the correct nested dict
                current = config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = value

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse {env_key}={env_vars[env_key]}: {e}")

    return config


# =============================================================================
# Main Loading Functions
# =============================================================================

def load_and_validate_config(
    env_file: Optional[Union[str, Path]] = None,
    config_file: Optional[Union[str, Path]] = None,
    schema_file: Optional[Union[str, Path]] = None,
    dataset: Optional[str] = None,
    strict: bool = True,
) -> MycelixConfig:
    """
    Load configuration from multiple sources and validate.

    Sources are merged in order (later sources override earlier):
    1. Default values
    2. Dataset-specific optimal parameters
    3. Environment file (.env.optimal)
    4. YAML/JSON config file
    5. Current environment variables

    Args:
        env_file: Path to .env file (default: .env.optimal)
        config_file: Path to YAML/JSON config file
        schema_file: Path to JSON schema file
        dataset: Dataset name for optimal parameter selection
        strict: If True, exit on critical errors; if False, only warn

    Returns:
        Validated MycelixConfig instance

    Raises:
        ConfigValidationError: If strict=True and critical errors found
    """

    # Determine base path
    base_path = Path(__file__).parent.parent

    # Default file paths
    if env_file is None:
        env_file = base_path / ".env.optimal"
    if config_file is None:
        config_file = base_path / "production_config.yaml"
    if schema_file is None:
        schema_file = Path(__file__).parent / "schema.json"

    # Start with empty config
    config: Dict[str, Any] = {}

    # 1. Load dataset-specific optimal parameters
    if dataset:
        optimal = get_dataset_optimal_params(dataset)
        config = merge_configs(config, {"byzantine_detection": optimal})
        logger.info(f"Loaded optimal parameters for dataset: {dataset}")

    # 2. Load environment file
    if Path(env_file).exists():
        env_vars = load_env_file(env_file)
        env_config = env_to_config(env_vars)
        config = merge_configs(config, env_config)
        logger.info(f"Loaded environment file: {env_file}")

    # 3. Load YAML/JSON config file
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            if config_path.suffix in (".yaml", ".yml"):
                file_config = load_yaml_file(config_path)
            else:
                with open(config_path) as f:
                    file_config = json.load(f)
            config = merge_configs(config, file_config)
            logger.info(f"Loaded config file: {config_file}")

    # 4. Load current environment variables (highest priority)
    current_env = {k: v for k, v in os.environ.items()}
    env_config = env_to_config(current_env)
    config = merge_configs(config, env_config)

    # Validate
    all_errors = []
    all_warnings = []

    # Validate against schema
    schema = load_json_schema(schema_file)
    if schema:
        schema_errors = validate_against_schema(config, schema)
        all_errors.extend(schema_errors)

    # Validate critical parameters
    critical_errors, param_warnings = validate_critical_parameters(config)
    all_errors.extend(critical_errors)
    all_warnings.extend(param_warnings)

    # Report warnings
    for warning in all_warnings:
        logger.warning(f"Config warning: {warning}")
        warnings.warn(warning, ConfigValidationWarning)

    # Handle errors
    if all_errors:
        error_msg = "\n".join(f"  - {e}" for e in all_errors)
        full_msg = f"Configuration validation failed:\n{error_msg}"

        if strict:
            logger.critical(full_msg)
            print(f"\n{'='*60}", file=sys.stderr)
            print("CONFIGURATION ERROR - CANNOT PROCEED", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            print(full_msg, file=sys.stderr)
            print(f"\n{'='*60}", file=sys.stderr)
            print("Ensure you have sourced .env.optimal:", file=sys.stderr)
            print("  source .env.optimal", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            raise ConfigValidationError(full_msg)
        else:
            logger.error(full_msg)

    # Convert to dataclass
    return dict_to_config(config)


def dict_to_config(config: Dict[str, Any]) -> MycelixConfig:
    """Convert dictionary to MycelixConfig dataclass."""

    byzantine = config.get("byzantine_detection", {})
    byzantine_config = ByzantineDetectionConfig(
        label_skew_cos_min=byzantine.get("label_skew_cos_min", -0.5),
        label_skew_cos_max=byzantine.get("label_skew_cos_max", 0.95),
        behavior_recovery_threshold=byzantine.get("behavior_recovery_threshold", 2),
        behavior_recovery_bonus=byzantine.get("behavior_recovery_bonus", 0.12),
        reputation_floor=byzantine.get("reputation_floor", 0.01),
        reputation_decay=byzantine.get("reputation_decay", 0.95),
        reputation_initial=byzantine.get("reputation_initial", 0.5),
        pogq_threshold=byzantine.get("pogq_threshold", 0.7),
        pogq_test_size=byzantine.get("pogq_test_size", 0.1),
        committee_reject_floor=byzantine.get("committee_reject_floor", 0.25),
        anomaly_sensitivity=byzantine.get("anomaly_sensitivity", 2.5),
        anomaly_window_size=byzantine.get("anomaly_window_size", 10),
        byzantine_penalty=byzantine.get("byzantine_penalty", -0.2),
        false_positive_threshold=byzantine.get("false_positive_threshold", 0.01),
        zero_norm_threshold=byzantine.get("zero_norm_threshold", 1e-6),
        backdoor_spike_threshold=byzantine.get("backdoor_spike_threshold", 3.0),
        hybrid_suspicion_threshold=byzantine.get("hybrid_suspicion_threshold", 0.35),
        hybrid_pogq_weight=byzantine.get("hybrid_pogq_weight", 0.35),
        hybrid_cos_weight=byzantine.get("hybrid_cos_weight", 0.30),
        hybrid_committee_weight=byzantine.get("hybrid_committee_weight", 0.15),
        hybrid_norm_weight=byzantine.get("hybrid_norm_weight", 0.10),
        hybrid_cluster_weight=byzantine.get("hybrid_cluster_weight", 0.10),
    )

    fl = config.get("federated_learning", config.get("federated", {}))
    data_dist = fl.get("data_distribution", {})
    fl_config = FederatedLearningConfig(
        aggregation=fl.get("aggregation", "krum"),
        num_rounds=fl.get("num_rounds", fl.get("rounds", 100)),
        num_clients=fl.get("num_clients", 10),
        clients_per_round=fl.get("clients_per_round", 5),
        min_clients=fl.get("min_clients", 5),
        local_epochs=config.get("training", {}).get("local_epochs", 10),
        learning_rate=config.get("training", {}).get("learning_rate", 0.05),
        momentum=config.get("training", {}).get("momentum", 0.9),
        weight_decay=config.get("training", {}).get("weight_decay", 0.0001),
        gradient_clipping=config.get("training", {}).get("gradient_clipping", 10.0),
        data_distribution_type=data_dist.get("type", "dirichlet"),
        data_distribution_alpha=data_dist.get("alpha", 1.0),
    )

    holo = config.get("holochain", config.get("credits", {}).get("holochain", {}))
    network = holo.get("network", {})
    holochain_config = HolochainConfig(
        enabled=holo.get("enabled", False),
        conductor_url=holo.get("conductor_url", "ws://localhost:8888"),
        admin_port=holo.get("admin_port", 8888),
        app_port=holo.get("app_port", 8889),
        dna_path=holo.get("dna_path", "./holochain/dna/zerotrustml.dna"),
        bootstrap_url=network.get("bootstrap_url", "https://bootstrap-staging.holo.host"),
        signal_url=network.get("signal_url", "wss://signal.holo.host"),
        gossip_loop_iteration_delay_ms=network.get(
            "tuning_params", {}
        ).get("gossip_loop_iteration_delay_ms", 1000),
    )

    sec = config.get("security", {})
    rate_limit = sec.get("rate_limiting", {})
    security_config = SecurityConfig(
        require_signatures=sec.get("require_signatures", True),
        signature_algorithm=sec.get("signature_algorithm", "ed25519"),
        encrypt_gradients=sec.get("encrypt_gradients", True),
        encryption_algorithm=sec.get("encryption_algorithm", "chacha20-poly1305"),
        max_requests_per_minute=rate_limit.get("max_requests_per_minute", 60),
        max_requests_per_hour=rate_limit.get("max_requests_per_hour", 1000),
        tls_enabled=sec.get("tls_enabled", False),
    )

    mon = config.get("monitoring", {})
    monitoring_config = MonitoringConfig(
        prometheus_enabled=mon.get("prometheus_enabled", True),
        prometheus_port=mon.get("prometheus_port", 9090),
        tensorboard=mon.get("tensorboard", True),
        log_level=mon.get("log_level", "INFO"),
        log_format=mon.get("log_format", "json"),
        log_retention_days=mon.get("log_retention_days", 30),
        checkpoint_interval=mon.get("checkpoint_interval", mon.get("save_checkpoint_every_n_rounds", 10)),
    )

    system = config.get("system", {})
    dataset = config.get("dataset", {})
    node = config.get("node", {})
    deployment = config.get("deployment", {})

    return MycelixConfig(
        byzantine_detection=byzantine_config,
        federated_learning=fl_config,
        holochain=holochain_config,
        security=security_config,
        monitoring=monitoring_config,
        seed=system.get("seed", 42),
        device=system.get("device", "cuda"),
        num_workers=system.get("num_workers", 4),
        dataset_name=dataset.get("name", "CIFAR10"),
        data_dir=dataset.get("data_dir", "./data"),
        batch_size=dataset.get("batch_size", 32),
        node_id=node.get("id", "fl-node-001"),
        node_region=node.get("region", "us-west-2"),
        deployment_mode=deployment.get("mode", "development"),
        dev_mode=deployment.get("dev_mode", False),
    )


def print_config_summary(config: MycelixConfig) -> None:
    """Print a summary of the current configuration."""
    print("\n" + "=" * 60)
    print("MYCELIX CONFIGURATION SUMMARY")
    print("=" * 60)

    print("\n[Byzantine Detection - CRITICAL]")
    print(f"  label_skew_cos_min:         {config.byzantine_detection.label_skew_cos_min}")
    print(f"  label_skew_cos_max:         {config.byzantine_detection.label_skew_cos_max}")
    print(f"  behavior_recovery_threshold: {config.byzantine_detection.behavior_recovery_threshold}")
    print(f"  behavior_recovery_bonus:    {config.byzantine_detection.behavior_recovery_bonus}")
    print(f"  pogq_threshold:             {config.byzantine_detection.pogq_threshold}")
    print(f"  hybrid_suspicion_threshold: {config.byzantine_detection.hybrid_suspicion_threshold}")

    print("\n[Federated Learning]")
    print(f"  aggregation:     {config.federated_learning.aggregation}")
    print(f"  num_rounds:      {config.federated_learning.num_rounds}")
    print(f"  num_clients:     {config.federated_learning.num_clients}")
    print(f"  learning_rate:   {config.federated_learning.learning_rate}")

    print("\n[System]")
    print(f"  device:          {config.device}")
    print(f"  dataset:         {config.dataset_name}")
    print(f"  deployment_mode: {config.deployment_mode}")

    print("\n[Security]")
    print(f"  require_signatures: {config.security.require_signatures}")
    print(f"  encrypt_gradients:  {config.security.encrypt_gradients}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for config validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mycelix Configuration Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validator.py                          # Validate with defaults
  python validator.py --env .env.optimal       # Use specific env file
  python validator.py --config production.yaml # Use specific config
  python validator.py --dataset cifar10        # Use dataset-specific params
  python validator.py --summary                # Print config summary
        """
    )

    parser.add_argument("--env", help="Path to .env file")
    parser.add_argument("--config", help="Path to YAML/JSON config file")
    parser.add_argument("--schema", help="Path to JSON schema file")
    parser.add_argument("--dataset", help="Dataset name for optimal params")
    parser.add_argument("--summary", action="store_true", help="Print config summary")
    parser.add_argument("--no-strict", action="store_true", help="Don't exit on errors")
    parser.add_argument("--export-env", action="store_true", help="Export as env vars")

    args = parser.parse_args()

    try:
        config = load_and_validate_config(
            env_file=args.env,
            config_file=args.config,
            schema_file=args.schema,
            dataset=args.dataset,
            strict=not args.no_strict,
        )

        if args.summary:
            print_config_summary(config)

        if args.export_env:
            print("# Export these environment variables:")
            print(f"export LABEL_SKEW_COS_MIN={config.byzantine_detection.label_skew_cos_min}")
            print(f"export LABEL_SKEW_COS_MAX={config.byzantine_detection.label_skew_cos_max}")
            print(f"export BEHAVIOR_RECOVERY_THRESHOLD={config.byzantine_detection.behavior_recovery_threshold}")
            print(f"export BEHAVIOR_RECOVERY_BONUS={config.byzantine_detection.behavior_recovery_bonus}")
            print(f"export POGQ_THRESHOLD={config.byzantine_detection.pogq_threshold}")

        print("Configuration validation PASSED")
        return 0

    except ConfigValidationError as e:
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
