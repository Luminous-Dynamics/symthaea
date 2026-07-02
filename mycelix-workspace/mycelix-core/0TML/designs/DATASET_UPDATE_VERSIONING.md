# Dataset Update and Versioning System

**Date**: 2025-10-22
**Purpose**: Enable seamless addition of new datasets and handle dataset version updates without breaking existing deployments
**Status**: Design Phase
**Priority**: Phase 1.5 or Phase 2

---

## Problem Statement

Current system has **hardcoded dataset profiles**:
- Adding new datasets requires code changes
- No versioning (CIFAR-10 v1 vs v2)
- No data distribution drift handling
- No hot-swapping of datasets in production

**Goal**: Plug-and-play dataset system with versioning, migration, and drift detection.

---

## Design: Dataset Plugin System

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Dataset Plugin Registry                       │
│  - Dynamically load dataset definitions                     │
│  - Version management                                        │
│  - Dependency tracking                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Dataset Definition (YAML/JSON)                    │
│  name: "cifar10"                                            │
│  version: "2.0"                                             │
│  source: "torchvision.datasets.CIFAR10"                     │
│  model_factory: "models.SimpleCNN"                          │
│  pogq_config: {...}                                         │
│  migration_from: ["1.0", "1.5"]                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Dataset Loader & Validator                          │
│  - Load dataset from source                                  │
│  - Validate schema and integrity                             │
│  - Apply transformations                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│        Version Migration Engine                              │
│  - Detect dataset version changes                            │
│  - Apply migration scripts                                   │
│  - Preserve learned parameters                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│       Distribution Drift Detection                           │
│  - Monitor label distribution over time                      │
│  - Detect concept drift                                      │
│  - Trigger parameter re-tuning                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### 1. Dataset Definition Format

**File**: `datasets/cifar10_v2.yaml`

```yaml
dataset:
  name: "cifar10"
  version: "2.0"
  description: "CIFAR-10 with updated preprocessing"

  # Dataset source
  source:
    type: "torchvision"
    class: "CIFAR10"
    root: "./data"
    download: true

  # Preprocessing
  transforms:
    train:
      - type: "RandomCrop"
        size: 32
        padding: 4
      - type: "RandomHorizontalFlip"
      - type: "ToTensor"
      - type: "Normalize"
        mean: [0.4914, 0.4822, 0.4465]
        std: [0.2470, 0.2435, 0.2616]

    test:
      - type: "ToTensor"
      - type: "Normalize"
        mean: [0.4914, 0.4822, 0.4465]
        std: [0.2470, 0.2435, 0.2616]

  # Model configuration
  model:
    type: "SimpleCNN"
    input_shape: [3, 32, 32]
    num_classes: 10
    params:
      channels: [32, 64, 128]
      dropout: 0.2

  # PoGQ configuration
  pogq:
    test_set_size: 1000
    loss_fn: "CrossEntropyLoss"
    metrics:
      - "accuracy"
      - "loss"

  # BFT parameters (learned via adaptive tuning)
  bft_parameters:
    iid:
      pogq_threshold: 0.5
      reputation_threshold: 0.3
      aggregator: "coordinate_median"
      committee_size: 5

    label_skew:
      pogq_threshold: 0.35
      reputation_threshold: 0.05
      aggregator: "coordinate_median"
      committee_size: 5

  # Version migration
  migration:
    from_versions:
      - "1.0"
      - "1.5"
    migration_script: "migrations/cifar10_1_to_2.py"
    backward_compatible: true

  # Drift detection
  drift_detection:
    enabled: true
    check_interval: 100  # rounds
    metrics:
      - "label_distribution"
      - "gradient_variance"
    thresholds:
      label_distribution_divergence: 0.1
      gradient_variance_ratio: 2.0
```

---

### 2. Dataset Registry

```python
from pathlib import Path
from typing import Dict, Optional, List
import yaml

class DatasetRegistry:
    """Central registry for all dataset plugins."""

    def __init__(self, datasets_dir: Path = Path("datasets")):
        self.datasets_dir = datasets_dir
        self.datasets: Dict[str, Dict[str, DatasetDefinition]] = {}
        self._load_all_datasets()

    def _load_all_datasets(self):
        """Load all dataset definitions from datasets directory."""
        for yaml_file in self.datasets_dir.glob("**/*.yaml"):
            definition = self._load_dataset_definition(yaml_file)
            dataset_name = definition.name
            version = definition.version

            if dataset_name not in self.datasets:
                self.datasets[dataset_name] = {}

            self.datasets[dataset_name][version] = definition
            print(f"✅ Loaded {dataset_name} v{version}")

    def _load_dataset_definition(self, yaml_file: Path) -> DatasetDefinition:
        """Parse YAML dataset definition."""
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        return DatasetDefinition(**config['dataset'])

    def get_dataset(
        self,
        name: str,
        version: Optional[str] = None
    ) -> DatasetDefinition:
        """Get dataset definition by name and version."""

        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")

        versions = self.datasets[name]

        if version is None:
            # Return latest version
            version = max(versions.keys(), key=lambda v: tuple(map(int, v.split('.'))))

        if version not in versions:
            raise ValueError(f"Version {version} not found for {name}. Available: {list(versions.keys())}")

        return versions[version]

    def list_datasets(self) -> List[Tuple[str, str]]:
        """List all available datasets with versions."""
        result = []
        for name, versions in self.datasets.items():
            for version in versions:
                result.append((name, version))
        return sorted(result)

    def add_dataset(
        self,
        definition: DatasetDefinition,
        save: bool = True
    ):
        """Dynamically add a new dataset (with optional persistence)."""

        name = definition.name
        version = definition.version

        if name not in self.datasets:
            self.datasets[name] = {}

        self.datasets[name][version] = definition

        if save:
            # Persist to YAML file
            yaml_path = self.datasets_dir / f"{name}_v{version.replace('.', '_')}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump({"dataset": definition.to_dict()}, f)

        print(f"✅ Added {name} v{version}")
```

---

### 3. Dataset Loader

```python
import importlib
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DatasetLoader:
    """Load datasets from definitions."""

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        split: str = "train",
    ) -> Dataset:
        """Load dataset from definition."""

        # Get definition
        definition = self.registry.get_dataset(name, version)

        # Load dataset source
        dataset = self._load_source(definition, split)

        # Apply transforms
        dataset = self._apply_transforms(dataset, definition, split)

        return dataset

    def _load_source(
        self,
        definition: DatasetDefinition,
        split: str
    ) -> Dataset:
        """Load dataset from source specification."""

        source = definition.source

        if source.type == "torchvision":
            # Load from torchvision
            module = importlib.import_module("torchvision.datasets")
            dataset_class = getattr(module, source.class_name)

            return dataset_class(
                root=source.root,
                train=(split == "train"),
                download=source.download,
                transform=None,  # Applied later
            )

        elif source.type == "huggingface":
            # Load from HuggingFace datasets
            from datasets import load_dataset
            return load_dataset(source.name, split=split)

        elif source.type == "custom":
            # Load custom dataset
            module = importlib.import_module(source.module)
            dataset_class = getattr(module, source.class_name)
            return dataset_class(**source.kwargs)

        else:
            raise ValueError(f"Unknown source type: {source.type}")

    def _apply_transforms(
        self,
        dataset: Dataset,
        definition: DatasetDefinition,
        split: str
    ) -> Dataset:
        """Apply transformations from definition."""

        transform_configs = definition.transforms.get(split, [])

        # Build transform pipeline
        transform_list = []
        for config in transform_configs:
            transform_type = config['type']
            transform_class = getattr(transforms, transform_type)

            # Get arguments (excluding 'type')
            kwargs = {k: v for k, v in config.items() if k != 'type'}

            transform_list.append(transform_class(**kwargs))

        # Compose transforms
        composed = transforms.Compose(transform_list)

        # Wrap dataset with transform
        dataset.transform = composed

        return dataset
```

---

### 4. Version Migration Engine

```python
class DatasetMigrationEngine:
    """Handle version migrations for datasets."""

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def migrate(
        self,
        name: str,
        from_version: str,
        to_version: str,
        preserve_params: bool = True
    ) -> MigrationResult:
        """Migrate from one dataset version to another."""

        print(f"🔄 Migrating {name} from v{from_version} to v{to_version}")

        # Get definitions
        old_def = self.registry.get_dataset(name, from_version)
        new_def = self.registry.get_dataset(name, to_version)

        # Check if migration is supported
        if from_version not in new_def.migration.from_versions:
            raise ValueError(
                f"Migration from {from_version} to {to_version} not supported. "
                f"Supported: {new_def.migration.from_versions}"
            )

        # Run migration script
        migration_result = self._run_migration_script(
            old_def=old_def,
            new_def=new_def,
            migration_script=new_def.migration.migration_script,
        )

        # Preserve learned parameters if requested
        if preserve_params:
            self._transfer_learned_parameters(
                from_def=old_def,
                to_def=new_def,
                migration_result=migration_result,
            )

        print(f"✅ Migration complete")
        return migration_result

    def _run_migration_script(
        self,
        old_def: DatasetDefinition,
        new_def: DatasetDefinition,
        migration_script: str,
    ) -> MigrationResult:
        """Execute migration script."""

        # Load migration module
        module_path, function_name = migration_script.rsplit(':', 1) if ':' in migration_script else (migration_script, 'migrate')
        module = importlib.import_module(module_path)
        migrate_fn = getattr(module, function_name)

        # Run migration
        return migrate_fn(old_definition=old_def, new_definition=new_def)

    def _transfer_learned_parameters(
        self,
        from_def: DatasetDefinition,
        to_def: DatasetDefinition,
        migration_result: MigrationResult,
    ):
        """Transfer learned BFT parameters to new version."""

        # Copy learned parameters
        to_def.bft_parameters = from_def.bft_parameters

        # Mark as migrated (may need re-tuning)
        to_def.metadata['migrated_from'] = from_def.version
        to_def.metadata['needs_retuning'] = migration_result.parameters_changed

        if migration_result.parameters_changed:
            print("⚠️  Parameters may need re-tuning after migration")
```

---

### 5. Distribution Drift Detection

```python
class DriftDetector:
    """Detect data distribution drift over time."""

    def __init__(self, dataset_definition: DatasetDefinition):
        self.definition = dataset_definition
        self.baseline_stats: Optional[DistributionStats] = None
        self.history: List[DistributionStats] = []

    def establish_baseline(self, dataset: Dataset):
        """Establish baseline distribution statistics."""

        self.baseline_stats = self._compute_stats(dataset)
        print(f"📊 Baseline established: {self.baseline_stats}")

    def check_drift(
        self,
        current_dataset: Dataset,
        round_num: int
    ) -> DriftReport:
        """Check for distribution drift."""

        if self.baseline_stats is None:
            raise ValueError("Baseline not established")

        # Compute current stats
        current_stats = self._compute_stats(current_dataset)
        self.history.append(current_stats)

        # Compute divergence metrics
        label_divergence = self._kl_divergence(
            self.baseline_stats.label_distribution,
            current_stats.label_distribution,
        )

        gradient_variance_ratio = (
            current_stats.gradient_variance / self.baseline_stats.gradient_variance
        )

        # Check thresholds
        drift_config = self.definition.drift_detection
        drift_detected = (
            label_divergence > drift_config.thresholds.label_distribution_divergence or
            gradient_variance_ratio > drift_config.thresholds.gradient_variance_ratio
        )

        report = DriftReport(
            round_num=round_num,
            label_divergence=label_divergence,
            gradient_variance_ratio=gradient_variance_ratio,
            drift_detected=drift_detected,
        )

        if drift_detected:
            print(f"⚠️  DRIFT DETECTED at round {round_num}:")
            print(f"   Label divergence: {label_divergence:.4f}")
            print(f"   Gradient variance ratio: {gradient_variance_ratio:.2f}")

        return report

    def _compute_stats(self, dataset: Dataset) -> DistributionStats:
        """Compute distribution statistics."""

        # Label distribution
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        label_counts = np.bincount(labels)
        label_distribution = label_counts / len(labels)

        # Gradient variance (sample)
        # ... compute gradient variance ...

        return DistributionStats(
            label_distribution=label_distribution,
            gradient_variance=gradient_variance,
        )

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between distributions."""
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
```

---

## Usage Examples

### Adding a New Dataset

```python
# Option 1: Via YAML file (recommended)
# Create datasets/mnist_v1.yaml, then:
registry = DatasetRegistry()  # Auto-loads all YAML files

# Option 2: Programmatically
new_dataset = DatasetDefinition(
    name="mnist",
    version="1.0",
    source=SourceConfig(
        type="torchvision",
        class_name="MNIST",
        root="./data",
        download=True,
    ),
    model=ModelConfig(
        type="SimpleCNN",
        input_shape=[1, 28, 28],
        num_classes=10,
    ),
    # ... rest of config
)

registry.add_dataset(new_dataset, save=True)
```

### Loading a Dataset

```python
# Get latest version
loader = DatasetLoader(registry)
dataset = loader.load("cifar10", split="train")

# Get specific version
dataset = loader.load("cifar10", version="1.5", split="test")

# List available datasets
for name, version in registry.list_datasets():
    print(f"{name} v{version}")
```

### Migrating Dataset Version

```python
# Migrate from v1.0 to v2.0
migration_engine = DatasetMigrationEngine(registry)
result = migration_engine.migrate(
    name="cifar10",
    from_version="1.0",
    to_version="2.0",
    preserve_params=True,  # Keep learned BFT parameters
)

if result.parameters_changed:
    # Re-tune BFT parameters
    optimizer = BayesianParameterOptimizer()
    new_params = optimizer.optimize(...)
```

### Monitoring for Drift

```python
# Setup drift detector
drift_detector = DriftDetector(dataset_definition)
drift_detector.establish_baseline(train_dataset)

# During training
for round_num in range(1000):
    # ... training ...

    # Check for drift every 100 rounds
    if round_num % 100 == 0:
        report = drift_detector.check_drift(current_dataset, round_num)

        if report.drift_detected:
            # Trigger parameter re-tuning
            print("🔄 Re-tuning parameters due to drift...")
            optimizer = BayesianParameterOptimizer()
            new_params = optimizer.optimize(...)
```

---

## Benefits

1. **Plug-and-Play**: Add new datasets without code changes
2. **Version Management**: Handle dataset updates gracefully
3. **Migration Support**: Preserve learned parameters across versions
4. **Drift Detection**: Automatically detect distribution changes
5. **Hot-Swapping**: Update datasets in production without downtime

---

## Integration with Adaptive Tuning

The dataset versioning system integrates seamlessly with adaptive parameter tuning:

```python
# When new dataset version is added
new_dataset_def = registry.get_dataset("cifar10", version="2.0")

# Characterize new version
characterizer = DatasetCharacterizer()
characteristics = characterizer.characterize(new_dataset_def)

# Find if similar to old version
similar_datasets = registry.find_similar(characteristics)

if similar_datasets[0][0] == "cifar10_v1.0":
    # Very similar → transfer parameters with minor tuning
    warm_start = similar_datasets[0][2]  # Old params
    optimizer = BayesianParameterOptimizer(warm_start)
    optimal_params = optimizer.optimize(max_trials=5)  # Quick refinement

else:
    # Different → full re-tuning
    optimizer = BayesianParameterOptimizer()
    optimal_params = optimizer.optimize(max_trials=15)
```

---

## Timeline

- **Week 1**: Dataset definition format + registry
- **Week 2**: Dataset loader + validator
- **Week 3**: Migration engine + drift detection
- **Week 4**: Integration + testing

**Total**: 4 weeks for complete dataset management system

---

## Next Steps

1. Implement `DatasetRegistry` with YAML loader
2. Create dataset definitions for existing datasets (CIFAR-10, EMNIST, Breast Cancer)
3. Build `DatasetLoader` with torchvision/HuggingFace support
4. Add migration engine for version updates
5. Deploy drift detection in production

---

**Status**: Design complete, ready for implementation in Phase 1.5 or Phase 2.
