# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Experiment Runner for Federated Learning Benchmarks

Automated execution of FL experiments with YAML configuration.

Features:
- Multiple baseline comparisons
- IID and non-IID data splits
- Byzantine attack simulation
- Result logging and visualization
- Checkpoint support
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Import datasets
from torchvision import datasets, transforms

# Import models
from experiments.models.cnn_models import create_model

# Import data splitting
from experiments.utils.data_splits import (
    create_iid_split,
    create_dirichlet_split,
    create_pathological_split,
    create_dataloaders,
    analyze_split
)

# Import defense adapter (bridges old runner architecture with new defense registry)
from experiments.defense_adapter import (
    create_fedavg_experiment,
    create_coord_median_experiment,
    create_rfa_experiment,
    create_fltrust_experiment,
    create_cbf_experiment,
    create_pogq_v4_1_experiment,
    create_boba_experiment,
    create_coord_median_safe_experiment,
    evaluate_global_model,
    BaseConfig
)

try:
    # Optional Rust-backed HyperFeel integration
    from zerotrustml.hyperfeel_bridge import (  # type: ignore[import-error]
        encode_gradient as hyperfeel_encode,
        HyperFeelUnavailable,
    )
except Exception:
    hyperfeel_encode = None  # type: ignore[assignment]

    class HyperFeelUnavailable(RuntimeError):  # type: ignore[no-redef]
        """Placeholder used when HyperFeel bridge is unavailable."""
        pass

try:
    # Optional Symthaea + MATL trust bridge (Rust-backed)
    from zerotrustml.experimental.symthaea_trust_bridge import (  # type: ignore[import-error]
        RUST_TRUST_BRIDGE_AVAILABLE as symthaea_trust_available,
        log_symthaea_and_matl_metrics,
    )
except Exception:
    symthaea_trust_available = False  # type: ignore[assignment]
    log_symthaea_and_matl_metrics = None  # type: ignore[assignment]


class ExperimentRunner:
    """
    Automated FL experiment runner.

    Handles:
    - Dataset loading
    - Data splitting (IID/non-IID)
    - Baseline initialization
    - Training execution
    - Result tracking
    """

    def __init__(self, config_path: str):
        """
        Initialize runner with config file.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize results storage
        self.results = {
            'config': self.config,
            'baselines': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'device': str(self.device),
            }
        }

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self):
        """
        Load dataset based on config.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        dataset_name = self.config['dataset']['name']
        # FIX: Use absolute path relative to project root instead of relative 'datasets'
        default_data_dir = project_root / 'datasets'
        data_dir = Path(self.config['dataset'].get('data_dir', default_data_dir))

        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST(
                root=data_dir / 'mnist',
                train=True,
                transform=transform,
                download=True  # Auto-download if not present
            )

            test_dataset = datasets.MNIST(
                root=data_dir / 'mnist',
                train=False,
                transform=transform,
                download=True  # Auto-download if not present
            )

        elif dataset_name == 'fmnist':
            # Fashion-MNIST - same format as MNIST, different domain (clothing)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST statistics
            ])

            train_dataset = datasets.FashionMNIST(
                root=data_dir / 'fmnist',
                train=True,
                transform=transform,
                download=True
            )

            test_dataset = datasets.FashionMNIST(
                root=data_dir / 'fmnist',
                train=False,
                transform=transform,
                download=True
            )

        elif dataset_name == 'emnist':
            # EMNIST - Extended MNIST with 47 classes (letters + digits)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))  # EMNIST statistics
            ])

            train_dataset = datasets.EMNIST(
                root=data_dir / 'emnist',
                split='balanced',  # 47 balanced classes
                train=True,
                transform=transform,
                download=True
            )

            test_dataset = datasets.EMNIST(
                root=data_dir / 'emnist',
                split='balanced',
                train=False,
                transform=transform,
                download=True
            )

        elif dataset_name == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010)),
            ])

            # FIX: Changed download=False to download=True
            train_dataset = datasets.CIFAR10(
                root=data_dir / 'cifar10',
                train=True,
                transform=transform_train,
                download=True  # Auto-download if missing
            )

            test_dataset = datasets.CIFAR10(
                root=data_dir / 'cifar10',
                train=False,
                transform=transform_test,
                download=True  # Auto-download if missing
            )

        elif dataset_name == 'svhn':
            # SVHN - Street View House Numbers (domain shift from MNIST)
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728),
                                   (0.1980, 0.2010, 0.1970)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728),
                                   (0.1980, 0.2010, 0.1970)),
            ])

            train_dataset = datasets.SVHN(
                root=data_dir / 'svhn',
                split='train',
                transform=transform_train,
                download=True
            )

            test_dataset = datasets.SVHN(
                root=data_dir / 'svhn',
                split='test',
                transform=transform_test,
                download=True
            )

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported: mnist, fmnist, emnist, cifar10, svhn")

        print(f"Loaded {dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_dataset, test_dataset

    def create_data_splits(self, train_dataset):
        """
        Create client data splits based on config.

        Args:
            train_dataset: Training dataset

        Returns:
            List of DataLoaders for each client
        """
        split_config = self.config['data_split']
        split_type = split_config['type']
        num_clients = self.config['federated']['num_clients']
        batch_size = self.config['federated']['batch_size']

        print(f"\nCreating {split_type} split for {num_clients} clients...")

        # Create splits
        if split_type == 'iid':
            client_indices = create_iid_split(
                train_dataset,
                num_clients=num_clients,
                seed=split_config.get('seed', 42)
            )

        elif split_type == 'dirichlet':
            alpha = split_config.get('alpha', 0.5)
            client_indices = create_dirichlet_split(
                train_dataset,
                num_clients=num_clients,
                alpha=alpha,
                seed=split_config.get('seed', 42)
            )
            print(f"Dirichlet alpha: {alpha}")

        elif split_type == 'pathological':
            shards_per_client = split_config.get('shards_per_client', 2)
            client_indices = create_pathological_split(
                train_dataset,
                num_clients=num_clients,
                shards_per_client=shards_per_client,
                seed=split_config.get('seed', 42)
            )
            print(f"Shards per client: {shards_per_client}")

        else:
            raise ValueError(f"Unknown split type: {split_type}")

        # Analyze split
        stats = analyze_split(train_dataset, client_indices)
        self.results['split_stats'] = stats
        print(f"Split created: {stats['mean_samples']:.0f} samples/client (avg)")

        # Create DataLoaders
        client_loaders = create_dataloaders(
            train_dataset,
            client_indices,
            batch_size=batch_size,
            shuffle=True
        )

        return client_loaders

    def create_baseline(
        self,
        baseline_name: str,
        client_loaders: List,
        test_dataset
    ):
        """
        Create baseline server and clients using defense registry.

        Args:
            baseline_name: Name of defense from registry
            client_loaders: List of client DataLoaders
            test_dataset: Test dataset

        Returns:
            Tuple of (server, clients, test_loader)
        """
        # Create model
        model_config = self.config['model']
        model_fn = lambda: create_model(
            model_config['type'],
            **model_config.get('params', {})
        )

        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['federated']['batch_size'],
            shuffle=False
        )

        # Get federated config
        fed_config = self.config['federated']

        # Create base config (used by all defenses)
        config = BaseConfig(
            learning_rate=fed_config['learning_rate'],
            local_epochs=fed_config['local_epochs'],
            batch_size=fed_config['batch_size'],
            num_clients=fed_config['num_clients'],
            fraction_clients=fed_config.get('fraction_clients', 1.0)
        )

        # Add defense-specific config attributes
        config.conformal_alpha = fed_config.get('conformal_alpha', 0.10)
        config.ema_beta = fed_config.get('ema_beta', 0.85)
        config.warmup_rounds = fed_config.get('warmup_rounds', 3)
        config.hysteresis_k = fed_config.get('hysteresis_k', 2)
        config.hysteresis_m = fed_config.get('hysteresis_m', 3)

        # Dispatch to appropriate experiment creator
        dispatch = {
            'fedavg': create_fedavg_experiment,
            'coord_median': create_coord_median_experiment,
            'coord_median_safe': create_coord_median_safe_experiment,
            'rfa': create_rfa_experiment,
            'fltrust': create_fltrust_experiment,
            'cbf': create_cbf_experiment,
            'pogq_v4_1': create_pogq_v4_1_experiment,
            'pogq_v4.1': create_pogq_v4_1_experiment,  # Allow both naming styles
            'boba': create_boba_experiment,
        }

        if baseline_name not in dispatch:
            available = ", ".join(dispatch.keys())
            raise ValueError(f"Unknown defense: {baseline_name}. Available: {available}")

        creator_fn = dispatch[baseline_name]

        # FLTrust needs test_dataset instead of test_loader
        if baseline_name == 'fltrust':
            server, clients, test_loader = creator_fn(
                model_fn, client_loaders, test_dataset, config, self.device
            )
        else:
            server, clients, test_loader = creator_fn(
                model_fn, client_loaders, test_loader, config, self.device
            )

        return server, clients, test_loader

    def train_baseline(
        self,
        baseline_name: str,
        server,
        clients,
        test_loader
    ) -> Dict:
        """
        Train a baseline for specified number of rounds.

        Args:
            baseline_name: Name of baseline
            server: Server instance
            clients: List of client instances
            test_loader: Test DataLoader

        Returns:
            Training history dictionary
        """
        num_rounds = self.config['training']['num_rounds']
        eval_every = self.config['training'].get('eval_every', 10)

        print(f"\nTraining {baseline_name} for {num_rounds} rounds...")

        history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'trust_scores': [],
        }

        for round_num in range(num_rounds):
            # Training round
            stats = server.train_round(clients)

            # Optional: emit Symthaea + MATL trust metrics for this round.
            # This is purely observational and does not affect training.
            if symthaea_trust_available and log_symthaea_and_matl_metrics is not None:
                try:
                    # Many server implementations expose per-node details for the last
                    # round; if present, use them for richer logging.
                    local_results = getattr(server, "last_round_local_results", None)
                    if isinstance(local_results, dict):
                        for node_id_str, node_info in local_results.items():
                            grad = node_info.get("gradient")
                            rep = node_info.get("reputation", 0.5)
                            if grad is None:
                                continue
                            grad_arr = np.array(grad, dtype=np.float32)
                            log_symthaea_and_matl_metrics(
                                node_id=str(node_id_str),
                                round_num=round_num,
                                gradient=grad_arr,
                                reputation=float(rep),
                                logger_extra={"baseline": baseline_name},
                            )
                except Exception as exc:
                    print(f"⚠️ Symthaea trust logging failed for baseline {baseline_name}: {exc}")

            # Record training stats
            history['rounds'].append(round_num)
            history['train_loss'].append(stats['train_loss'])
            history['train_accuracy'].append(stats['train_accuracy'])
            if hasattr(server, 'last_trust_scores'):
                history['trust_scores'].append(server.last_trust_scores)
            else:
                history['trust_scores'].append(None)

            # Evaluate on test set
            if round_num % eval_every == 0 or round_num == num_rounds - 1:
                test_stats = evaluate_global_model(server.model, test_loader, self.device)
                history['test_loss'].append(test_stats['test_loss'])
                history['test_accuracy'].append(test_stats['test_accuracy'])

                print(f"Round {round_num:3d}: "
                      f"Train Loss={stats['train_loss']:.4f}, "
                      f"Train Acc={stats['train_accuracy']:.4f}, "
                      f"Test Loss={test_stats['test_loss']:.4f}, "
                      f"Test Acc={test_stats['test_accuracy']:.4f}")
            else:
                # Pad with None for non-evaluation rounds
                history['test_loss'].append(None)
                history['test_accuracy'].append(None)

        # Optional: encode final model state with Rust HyperFeel for diagnostics.
        if hyperfeel_encode is not None:
            try:
                flat = self._flatten_model(server.model)
                if flat.size > 0:
                    hg = hyperfeel_encode(flat)
                    vector = hg.get("vector", {})
                    orig_bytes = int(flat.size) * 4  # float32 parameters
                    hv_bytes = len(vector.get("data", []))  # i8 components
                    compression = float(orig_bytes) / float(hv_bytes) if hv_bytes > 0 else None

                    history["hyperfeel"] = {
                        "vector_dimension": len(vector.get("data", [])),
                        "timestamp": hg.get("timestamp"),
                        "original_bytes": orig_bytes,
                        "hypervector_bytes": hv_bytes,
                        "compression_ratio": compression,
                    }

                    if compression is not None:
                        print(
                            f"  [HyperFeel] Baseline {baseline_name}: "
                            f"{orig_bytes}B → {hv_bytes}B "
                            f"({compression:.1f}× compression)"
                        )
            except HyperFeelUnavailable:
                # HyperFeel CLI not available; ignore silently.
                pass
            except Exception as exc:
                print(f"⚠️ HyperFeel encoding failed for baseline {baseline_name}: {exc}")

        return history

    def run(self):
        """Execute experiment based on config."""
        print("\n" + "=" * 70)
        print(f"Starting Experiment: {self.config['experiment_name']}")
        print("=" * 70)

        # Load datasets
        train_dataset, test_dataset = self.load_dataset()

        # Create data splits
        client_loaders = self.create_data_splits(train_dataset)

        # Run each baseline
        baselines = self.config['baselines']
        for baseline_config in baselines:
            # Extract baseline name from config dict
            if isinstance(baseline_config, dict):
                baseline_name = baseline_config['name']
            else:
                # Fallback for legacy string-based configs
                baseline_name = baseline_config

            print(f"\n{'=' * 70}")
            print(f"Baseline: {baseline_name.upper()}")
            print(f"{'=' * 70}")

            # Create baseline
            server, clients, test_loader = self.create_baseline(
                baseline_name,
                client_loaders,
                test_dataset
            )

            # Train
            history = self.train_baseline(
                baseline_name,
                server,
                clients,
                test_loader
            )

            # Store results
            self.results['baselines'][baseline_name] = history

        # Save results
        self.save_results()

        print("\n" + "=" * 70)
        print("Experiment Complete!")
        print("=" * 70)

    def save_results(self):
        """Save experiment results to JSON file."""
        output_dir = Path(self.config.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['experiment_name']}_{timestamp}.json"
        filepath = output_dir / filename

        # Add end time
        self.results['metadata']['end_time'] = datetime.now().isoformat()

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to: {filepath}")

        # Generate statistical artifacts (Phase 2 Step 2)
        self._generate_artifacts(output_dir, timestamp)

    def _generate_artifacts(self, output_dir: Path, timestamp: str):
        """
        Generate statistical verification artifacts for publication.

        Creates all 6 artifact files per experiment run:
        - per_bucket_fpr.json: Mondrian conformal FPR verification
        - bootstrap_ci.json: AUROC confidence intervals
        - wilcoxon.json: Paired statistical tests
        - detection_metrics.json: TPR, FPR, precision, F1
        - ema_vs_raw.json: Phase 2 EMA smoothing metrics
        - coord_median_diagnostics.json: Phase 5 guard activation counts
        """
        from src.evaluation.statistics import generate_statistical_artifacts

        # Create artifacts subdirectory
        artifacts_dir = output_dir / f"artifacts_{timestamp}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to experiment_results format expected by statistics.py
        experiment_results = self._prepare_experiment_results()

        if not experiment_results:
            print("\n⚠️  No detection data available for artifact generation")
            print("   (Set up detection tracking in baselines to enable artifacts)")
            return

        # Generate all artifacts
        print(f"\n📊 Generating statistical artifacts...")
        try:
            artifact_paths = generate_statistical_artifacts(
                experiment_results=experiment_results,
                output_dir=str(artifacts_dir)
            )

            print(f"\n✅ Statistical artifacts generated:")
            for artifact_name, artifact_path in artifact_paths.items():
                print(f"   - {artifact_name}: {artifact_path}")

        except Exception as e:
            print(f"\n❌ Artifact generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _prepare_experiment_results(self) -> List[Dict[str, Any]]:
        """
        Convert runner results to experiment_results format for statistics.py.

        Expected format:
        [
            {
                "experiment_id": str,
                "detector": str,
                "attack": str,
                "y_true": List[bool],  # Ground truth Byzantine labels
                "y_pred": List[bool],  # Predicted Byzantine labels
                "y_scores": List[float],  # Anomaly scores
                "mondrian_profiles": List[str],  # Class profiles for Mondrian CP
                "detection_results": List[Dict],  # Per-client detection results
                "phase2_data": Dict,  # Phase 2 EMA tracking (optional)
                "phase5_data": Dict,  # Phase 5 guard tracking (optional)
            },
            ...
        ]

        Returns:
            List of experiment result dicts
        """
        experiment_results = []

        # Extract attack configuration
        attack_name = self.config.get('attack', {}).get('type', 'none')

        for baseline_name, history in self.results.get('baselines', {}).items():
            # Check if detection tracking data exists
            if 'detection_tracking' not in history:
                continue  # Skip baselines without detection tracking

            tracking = history['detection_tracking']

            experiment_results.append({
                "experiment_id": f"{self.config['experiment_name']}_{baseline_name}",
                "detector": baseline_name,
                "attack": attack_name,
                "y_true": tracking.get('y_true', []),
                "y_pred": tracking.get('y_pred', []),
                "y_scores": tracking.get('y_scores', []),
                "mondrian_profiles": tracking.get('mondrian_profiles', []),
                "detection_results": tracking.get('detection_results', []),
                "phase2_data": tracking.get('phase2_data', {}),
                "phase5_data": tracking.get('phase5_data', {}),
                "conformal_alpha": tracking.get('conformal_alpha', 0.10)
            })

        return experiment_results

    def _flatten_model(self, model) -> np.ndarray:
        """
        Flatten model parameters to a 1D numpy array.

        Used for optional HyperFeel encoding of the final model state
        without affecting training behavior.
        """
        params = []
        with torch.no_grad():
            for p in model.parameters():
                params.append(p.detach().view(-1).cpu().numpy())
        if not params:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(params, axis=0).astype(np.float32)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run FL experiments')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    args = parser.parse_args()

    # Create and run experiment
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
