# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unified Configuration for Mycelix/ZeroTrustML

Provides centralized configuration with:
- Logging configuration schema
- Backend configuration
- Security settings
- Environment variable overrides
- YAML/JSON configuration file support

Usage:
    from zerotrustml.config import (
        MycelixConfig,
        LoggingConfig,
        load_config,
        get_config
    )

    # Load from file
    config = load_config("/path/to/config.yaml")

    # Or create programmatically
    config = MycelixConfig(
        logging=LoggingConfig(level="INFO", json_format=True),
        service_name="mycelix-coordinator"
    )

    # Apply logging configuration
    config.apply_logging()
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum


# ============================================================
# Enums
# ============================================================

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StorageStrategy(Enum):
    """Storage backend strategy."""
    ALL = "all"
    PRIMARY = "primary"
    QUORUM = "quorum"


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ============================================================
# Logging Configuration
# ============================================================

@dataclass
class LoggingConfig:
    """
    Logging configuration schema.

    Attributes:
        level: Default log level
        json_format: Use structured JSON logging
        include_timestamp: Include ISO timestamp in logs
        include_location: Include file/line info
        colored_output: Use ANSI colors in console (dev only)
        log_file: Path to log file (optional)
        log_file_max_bytes: Max size before rotation
        log_file_backup_count: Number of backup files
        service_name: Service name for structured logs
        environment: Environment name (dev, staging, prod)
        version: Application version
        module_levels: Per-module log level overrides
    """

    level: str = "INFO"
    json_format: bool = False
    include_timestamp: bool = True
    include_location: bool = False
    colored_output: bool = True

    # File logging
    log_file: Optional[str] = None
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5

    # Structured logging fields
    service_name: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None

    # Per-module overrides
    module_levels: Dict[str, str] = field(default_factory=dict)

    def apply(self) -> None:
        """Apply this logging configuration."""
        from zerotrustml.logging import configure_logging
        configure_logging(
            level=self.level,
            json_format=self.json_format,
            include_timestamp=self.include_timestamp,
            include_location=self.include_location,
            colored_output=self.colored_output,
            log_file=self.log_file,
            log_file_max_bytes=self.log_file_max_bytes,
            log_file_backup_count=self.log_file_backup_count,
            service_name=self.service_name,
            environment=self.environment,
            version=self.version,
            module_levels=self.module_levels
        )

    @classmethod
    def for_production(cls, service_name: str) -> 'LoggingConfig':
        """Create production logging configuration."""
        return cls(
            level="INFO",
            json_format=True,
            include_timestamp=True,
            include_location=True,
            colored_output=False,
            service_name=service_name,
            environment="production"
        )

    @classmethod
    def for_development(cls) -> 'LoggingConfig':
        """Create development logging configuration."""
        return cls(
            level="DEBUG",
            json_format=False,
            include_timestamp=True,
            include_location=False,
            colored_output=True,
            environment="development"
        )


# ============================================================
# Backend Configurations
# ============================================================

@dataclass
class PostgresConfig:
    """PostgreSQL backend configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 5432
    database: str = "zerotrustml"
    user: str = "zerotrustml"
    password: str = ""
    pool_size: int = 10
    ssl_mode: str = "prefer"

    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}"
            f"/{self.database}?sslmode={self.ssl_mode}"
        )


@dataclass
class HolochainConfig:
    """Holochain backend configuration."""
    enabled: bool = False
    admin_url: str = "ws://localhost:8888"
    app_url: str = "ws://localhost:8889"
    app_id: str = "zerotrustml"
    cell_id: Optional[str] = None
    timeout: int = 30


@dataclass
class LocalFileConfig:
    """Local file backend configuration (for testing)."""
    enabled: bool = False
    data_dir: str = "/tmp/zerotrustml_data"


# ============================================================
# Security Configuration
# ============================================================

@dataclass
class EncryptionConfig:
    """Encryption configuration for HIPAA/GDPR compliance."""
    enabled: bool = True
    algorithm: str = "AES-256-GCM"
    key_env_var: str = "ZEROTRUSTML_ENCRYPTION_KEY"
    key_rotation_days: int = 90


@dataclass
class ZKPoCConfig:
    """Zero-Knowledge Proof of Computation configuration."""
    enabled: bool = True
    pogq_threshold: float = 0.7
    require_encryption: bool = True
    hipaa_mode: bool = False
    gdpr_mode: bool = False


# ============================================================
# Error Handling Configuration
# ============================================================

@dataclass
class RetryConfig:
    """Retry configuration for resilient operations."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for external services."""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


# ============================================================
# Observability Configuration
# ============================================================

@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    prefix: str = "mycelix"
    export_interval: float = 60.0
    include_histograms: bool = True


@dataclass
class TracingConfig:
    """Distributed tracing configuration."""
    enabled: bool = True
    sample_rate: float = 1.0
    propagation_format: str = "w3c"


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)


# ============================================================
# Main Configuration
# ============================================================

@dataclass
class MycelixConfig:
    """
    Main Mycelix/ZeroTrustML configuration.

    Combines all configuration sections into a unified config object.

    Example:
        config = MycelixConfig(
            logging=LoggingConfig.for_production("coordinator"),
            postgres=PostgresConfig(host="db.example.com"),
            encryption=EncryptionConfig(enabled=True)
        )
        config.apply_logging()
    """

    # Service identification
    service_name: str = "mycelix"
    environment: str = "development"
    version: str = "1.0.0"

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Storage backends
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    holochain: HolochainConfig = field(default_factory=HolochainConfig)
    localfile: LocalFileConfig = field(default_factory=LocalFileConfig)
    storage_strategy: str = "all"

    # Security
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    zkpoc: ZKPoCConfig = field(default_factory=ZKPoCConfig)

    # Error handling
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)

    # Observability
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def apply_logging(self) -> None:
        """Apply logging configuration."""
        # Set service info in logging config
        self.logging.service_name = self.service_name
        self.logging.environment = self.environment
        self.logging.version = self.version
        self.logging.apply()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MycelixConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        if 'postgres' in data and isinstance(data['postgres'], dict):
            data['postgres'] = PostgresConfig(**data['postgres'])
        if 'holochain' in data and isinstance(data['holochain'], dict):
            data['holochain'] = HolochainConfig(**data['holochain'])
        if 'localfile' in data and isinstance(data['localfile'], dict):
            data['localfile'] = LocalFileConfig(**data['localfile'])
        if 'encryption' in data and isinstance(data['encryption'], dict):
            data['encryption'] = EncryptionConfig(**data['encryption'])
        if 'zkpoc' in data and isinstance(data['zkpoc'], dict):
            data['zkpoc'] = ZKPoCConfig(**data['zkpoc'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'MycelixConfig':
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================
# Configuration Loading
# ============================================================

_global_config: Optional[MycelixConfig] = None


def load_config(
    path: Optional[Union[str, Path]] = None,
    env_prefix: str = "MYCELIX"
) -> MycelixConfig:
    """
    Load configuration from file and/or environment variables.

    Args:
        path: Path to config file (YAML or JSON)
        env_prefix: Prefix for environment variable overrides

    Returns:
        MycelixConfig instance

    Environment variable overrides:
        MYCELIX_LOG_LEVEL=DEBUG
        MYCELIX_LOG_JSON=true
        MYCELIX_POSTGRES_HOST=db.example.com
        MYCELIX_HOLOCHAIN_ENABLED=true
    """
    global _global_config

    config_data: Dict[str, Any] = {}

    # Load from file if provided
    if path:
        path = Path(path)
        if path.exists():
            if path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    with open(path) as f:
                        config_data = yaml.safe_load(f) or {}
                except ImportError:
                    raise ImportError("PyYAML required for YAML config: pip install pyyaml")
            elif path.suffix == '.json':
                with open(path) as f:
                    config_data = json.load(f)

    # Apply environment variable overrides
    config_data = _apply_env_overrides(config_data, env_prefix)

    # Create config
    _global_config = MycelixConfig.from_dict(config_data)
    return _global_config


def get_config() -> MycelixConfig:
    """Get the global configuration (creates default if not set)."""
    global _global_config
    if _global_config is None:
        _global_config = MycelixConfig()
    return _global_config


def _apply_env_overrides(config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    # Logging overrides
    if f"{prefix}_LOG_LEVEL" in os.environ:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = os.environ[f"{prefix}_LOG_LEVEL"]

    if f"{prefix}_LOG_JSON" in os.environ:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['json_format'] = os.environ[f"{prefix}_LOG_JSON"].lower() in ('true', '1', 'yes')

    if f"{prefix}_LOG_FILE" in os.environ:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['log_file'] = os.environ[f"{prefix}_LOG_FILE"]

    # Service overrides
    if f"{prefix}_SERVICE_NAME" in os.environ:
        config['service_name'] = os.environ[f"{prefix}_SERVICE_NAME"]

    if f"{prefix}_ENVIRONMENT" in os.environ:
        config['environment'] = os.environ[f"{prefix}_ENVIRONMENT"]

    # PostgreSQL overrides
    if f"{prefix}_POSTGRES_HOST" in os.environ:
        if 'postgres' not in config:
            config['postgres'] = {}
        config['postgres']['host'] = os.environ[f"{prefix}_POSTGRES_HOST"]

    if f"{prefix}_POSTGRES_PORT" in os.environ:
        if 'postgres' not in config:
            config['postgres'] = {}
        config['postgres']['port'] = int(os.environ[f"{prefix}_POSTGRES_PORT"])

    if f"{prefix}_POSTGRES_DB" in os.environ:
        if 'postgres' not in config:
            config['postgres'] = {}
        config['postgres']['database'] = os.environ[f"{prefix}_POSTGRES_DB"]

    if f"{prefix}_POSTGRES_USER" in os.environ:
        if 'postgres' not in config:
            config['postgres'] = {}
        config['postgres']['user'] = os.environ[f"{prefix}_POSTGRES_USER"]

    if f"{prefix}_POSTGRES_PASSWORD" in os.environ:
        if 'postgres' not in config:
            config['postgres'] = {}
        config['postgres']['password'] = os.environ[f"{prefix}_POSTGRES_PASSWORD"]

    # Holochain overrides
    if f"{prefix}_HOLOCHAIN_ENABLED" in os.environ:
        if 'holochain' not in config:
            config['holochain'] = {}
        config['holochain']['enabled'] = os.environ[f"{prefix}_HOLOCHAIN_ENABLED"].lower() in ('true', '1', 'yes')

    if f"{prefix}_HOLOCHAIN_ADMIN_URL" in os.environ:
        if 'holochain' not in config:
            config['holochain'] = {}
        config['holochain']['admin_url'] = os.environ[f"{prefix}_HOLOCHAIN_ADMIN_URL"]

    if f"{prefix}_HOLOCHAIN_APP_URL" in os.environ:
        if 'holochain' not in config:
            config['holochain'] = {}
        config['holochain']['app_url'] = os.environ[f"{prefix}_HOLOCHAIN_APP_URL"]

    # Encryption overrides
    if f"{prefix}_ENCRYPTION_ENABLED" in os.environ:
        if 'encryption' not in config:
            config['encryption'] = {}
        config['encryption']['enabled'] = os.environ[f"{prefix}_ENCRYPTION_ENABLED"].lower() in ('true', '1', 'yes')

    return config


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Enums
    'LogLevel',
    'StorageStrategy',
    'Environment',

    # Logging config
    'LoggingConfig',

    # Backend configs
    'PostgresConfig',
    'HolochainConfig',
    'LocalFileConfig',

    # Security configs
    'EncryptionConfig',
    'ZKPoCConfig',

    # Error handling configs
    'RetryConfig',
    'CircuitBreakerConfig',
    'ErrorHandlingConfig',

    # Observability configs
    'MetricsConfig',
    'TracingConfig',
    'ObservabilityConfig',

    # Main config
    'MycelixConfig',

    # Functions
    'load_config',
    'get_config',
]
