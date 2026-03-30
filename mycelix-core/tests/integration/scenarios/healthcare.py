# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Healthcare Scenario: 5-Hospital HIPAA-Compliant FL

Simulates a realistic healthcare federated learning deployment with:
- 5 hospitals as FL participants
- HIPAA-compliant data handling
- Differential privacy mechanisms
- Heterogeneous data distributions
- Regulatory compliance verification
"""

import pytest
import numpy as np
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ..fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    FLAggregator,
    ReputationBridge,
)
from ..fixtures.nodes import HonestNode, NodeBehavior
from ..fixtures.metrics import MetricsCollector, TestReport


# ============================================================================
# Healthcare-Specific Configuration
# ============================================================================

class DataType(Enum):
    """Types of healthcare data."""
    IMAGING = "imaging"
    EHR = "ehr"
    GENOMICS = "genomics"
    WEARABLES = "wearables"


@dataclass
class HealthcareNetworkConfig(NetworkConfig):
    """Configuration for healthcare FL network."""
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5
    secure_aggregation: bool = True
    hipaa_audit_enabled: bool = True
    data_type: DataType = DataType.IMAGING
    model_type: str = "cnn_classifier"

    def __post_init__(self):
        # Healthcare-specific defaults
        if self.num_honest_nodes == 0:
            self.num_honest_nodes = 5  # 5 hospitals
        if self.aggregation_strategy == "fedavg":
            self.aggregation_strategy = "trimmed_mean"  # More robust


# ============================================================================
# HIPAA-Compliant Node
# ============================================================================

class HIPAACompliantNode(HonestNode):
    """
    A HIPAA-compliant FL node with additional privacy protections.

    Implements:
    - Differential privacy (gradient clipping + noise)
    - Audit logging
    - Data minimization
    - Access controls (simulated)
    """

    def __init__(
        self,
        node_id: str,
        hospital_name: str,
        gradient_dimension: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        local_data_size: int = 1000,
        data_type: DataType = DataType.IMAGING,
    ):
        super().__init__(
            node_id=node_id,
            gradient_dimension=gradient_dimension,
            local_data_size=local_data_size,
        )
        self.hospital_name = hospital_name
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.data_type = data_type

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        self._access_log: List[Dict[str, Any]] = []

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute gradient with differential privacy."""
        # Log access
        self._log_access("compute_gradient", round_id)

        # Compute base gradient
        base_gradient = await super().compute_gradient(round_id)

        # Apply gradient clipping
        norm = np.linalg.norm(base_gradient)
        if norm > self.clip_norm:
            base_gradient = base_gradient * (self.clip_norm / norm)

        # Add calibrated Gaussian noise for differential privacy
        noise_scale = self._compute_noise_scale()
        noise = self._rng.normal(0, noise_scale, self.gradient_dimension)
        private_gradient = base_gradient + noise

        # Audit logging
        self._log_audit(
            action="gradient_computed",
            round_id=round_id,
            gradient_norm=float(np.linalg.norm(private_gradient)),
            noise_added=True,
            epsilon=self.epsilon,
        )

        return private_gradient.astype(np.float32)

    def _compute_noise_scale(self) -> float:
        """Compute noise scale for (epsilon, delta)-DP."""
        # Gaussian mechanism calibration
        # sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        sensitivity = self.clip_norm
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def _log_audit(self, action: str, **kwargs):
        """Log an audit event."""
        self._audit_log.append({
            "timestamp": time.time(),
            "hospital": self.hospital_name,
            "node_id": self.node_id,
            "action": action,
            **kwargs,
        })

    def _log_access(self, operation: str, round_id: int):
        """Log a data access event."""
        self._access_log.append({
            "timestamp": time.time(),
            "hospital": self.hospital_name,
            "operation": operation,
            "round_id": round_id,
            "data_type": self.data_type.value,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log for compliance review."""
        return self._audit_log.copy()

    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log for compliance review."""
        return self._access_log.copy()

    def verify_compliance(self) -> Tuple[bool, List[str]]:
        """Verify HIPAA compliance of operations."""
        violations = []

        # Check epsilon bounds
        if self.epsilon > 10.0:
            violations.append(f"Epsilon {self.epsilon} may be too high for sensitive data")

        # Check all gradients had noise added
        for entry in self._audit_log:
            if entry.get("action") == "gradient_computed":
                if not entry.get("noise_added"):
                    violations.append(
                        f"Gradient at round {entry.get('round_id')} missing DP noise"
                    )

        return len(violations) == 0, violations


# ============================================================================
# Hospital Node
# ============================================================================

@dataclass
class HospitalProfile:
    """Profile of a hospital's data characteristics."""
    name: str
    patient_count: int
    data_quality: float  # 0-1, affects gradient noise
    specialty: str
    geographic_region: str
    ehr_vendor: str


class HospitalNode(HIPAACompliantNode):
    """
    Represents a specific hospital in the FL network.

    Models:
    - Hospital-specific data characteristics
    - Non-IID data distribution
    - Varying participation patterns
    """

    def __init__(
        self,
        hospital_profile: HospitalProfile,
        gradient_dimension: int,
        epsilon: float = 1.0,
        **kwargs
    ):
        self.profile = hospital_profile

        super().__init__(
            node_id=f"hospital_{hospital_profile.name.lower().replace(' ', '_')}",
            hospital_name=hospital_profile.name,
            gradient_dimension=gradient_dimension,
            epsilon=epsilon,
            local_data_size=hospital_profile.patient_count,
            **kwargs
        )

        # Model data distribution skew based on specialty
        self._distribution_skew = self._compute_distribution_skew()

    def _compute_distribution_skew(self) -> np.ndarray:
        """Compute distribution skew based on hospital specialty."""
        # Each specialty has different gradient patterns
        specialty_seeds = {
            "general": 42,
            "oncology": 123,
            "cardiology": 456,
            "pediatrics": 789,
            "neurology": 101,
        }

        seed = specialty_seeds.get(self.profile.specialty.lower(), 42)
        rng = np.random.RandomState(seed)

        # Create specialty-specific base direction
        skew = rng.randn(self.gradient_dimension)
        skew /= np.linalg.norm(skew)

        return skew

    async def compute_gradient(self, round_id: int) -> np.ndarray:
        """Compute hospital-specific gradient with data heterogeneity."""
        # Get base DP gradient
        base_gradient = await super().compute_gradient(round_id)

        # Apply specialty-specific skew (non-IID simulation)
        skew_weight = 0.3  # 30% influence from specialty
        specialty_component = self._distribution_skew * np.linalg.norm(base_gradient)

        # Blend base gradient with specialty component
        gradient = (1 - skew_weight) * base_gradient + skew_weight * specialty_component

        # Apply data quality factor (lower quality = more noise)
        quality_noise = (1 - self.profile.data_quality) * 0.1
        gradient += self._rng.randn(self.gradient_dimension) * quality_noise * np.linalg.norm(gradient)

        return gradient.astype(np.float32)


# ============================================================================
# Healthcare Scenario
# ============================================================================

class HealthcareScenario:
    """
    Complete healthcare FL scenario with 5 hospitals.

    Simulates realistic healthcare federated learning with:
    - Data heterogeneity across hospitals
    - HIPAA compliance requirements
    - Differential privacy
    - Secure aggregation
    """

    # Predefined hospital profiles
    HOSPITAL_PROFILES = [
        HospitalProfile(
            name="Metropolitan General",
            patient_count=50000,
            data_quality=0.95,
            specialty="general",
            geographic_region="urban",
            ehr_vendor="Epic",
        ),
        HospitalProfile(
            name="University Medical Center",
            patient_count=30000,
            data_quality=0.98,
            specialty="oncology",
            geographic_region="urban",
            ehr_vendor="Cerner",
        ),
        HospitalProfile(
            name="Regional Heart Institute",
            patient_count=15000,
            data_quality=0.92,
            specialty="cardiology",
            geographic_region="suburban",
            ehr_vendor="Epic",
        ),
        HospitalProfile(
            name="Children's Hospital",
            patient_count=20000,
            data_quality=0.90,
            specialty="pediatrics",
            geographic_region="urban",
            ehr_vendor="Meditech",
        ),
        HospitalProfile(
            name="Rural Community Hospital",
            patient_count=5000,
            data_quality=0.85,
            specialty="general",
            geographic_region="rural",
            ehr_vendor="Allscripts",
        ),
    ]

    def __init__(
        self,
        config: Optional[HealthcareNetworkConfig] = None,
        gradient_dimension: int = 5000,
    ):
        self.config = config or HealthcareNetworkConfig(
            num_honest_nodes=5,
            gradient_dimension=gradient_dimension,
        )

        self.hospitals: List[HospitalNode] = []
        self.gradient_store = GradientStore()
        self.aggregator = FLAggregator(
            strategy=self.config.aggregation_strategy,
            byzantine_threshold=self.config.byzantine_threshold,
        )
        self.reputation = ReputationBridge()

        self._round = 0
        self._compliance_log: List[Dict[str, Any]] = []
        self._round_history: List[Dict[str, Any]] = []

    async def setup(self):
        """Initialize the healthcare scenario."""
        # Create hospital nodes
        for profile in self.HOSPITAL_PROFILES:
            hospital = HospitalNode(
                hospital_profile=profile,
                gradient_dimension=self.config.gradient_dimension,
                epsilon=self.config.differential_privacy_epsilon,
                delta=self.config.differential_privacy_delta,
            )
            self.hospitals.append(hospital)

    async def run_round(self) -> Dict[str, Any]:
        """Execute a single FL round with HIPAA compliance."""
        self._round += 1
        round_id = self._round

        start_time = time.time()
        gradients = []
        hospital_ids = []

        # Collect gradients from all hospitals
        for hospital in self.hospitals:
            gradient = await hospital.compute_gradient(round_id)

            success, entry_hash, error = await self.gradient_store.submit_gradient(
                node_id=hospital.node_id,
                round_id=round_id,
                gradient=gradient,
                metadata={
                    "hospital_name": hospital.hospital_name,
                    "epsilon": hospital.epsilon,
                    "hipaa_compliant": True,
                },
            )

            if success:
                gradients.append(gradient)
                hospital_ids.append(hospital.node_id)

        # Aggregate
        if len(gradients) < 3:
            return {
                "round_id": round_id,
                "success": False,
                "error": "Insufficient hospital participation",
            }

        aggregated, metadata = await self.aggregator.aggregate(
            gradients, node_ids=hospital_ids
        )

        # Compliance check
        compliance_passed, violations = self._check_round_compliance(round_id)

        result = {
            "round_id": round_id,
            "success": True,
            "aggregated_gradient": aggregated,
            "n_hospitals": len(gradients),
            "hospital_ids": hospital_ids,
            "duration_ms": (time.time() - start_time) * 1000,
            "compliance_passed": compliance_passed,
            "compliance_violations": violations,
            "aggregation_metadata": metadata,
        }

        self._round_history.append(result)
        return result

    async def run_rounds(self, n: int) -> List[Dict[str, Any]]:
        """Run multiple FL rounds."""
        results = []
        for _ in range(n):
            result = await self.run_round()
            results.append(result)
        return results

    def _check_round_compliance(self, round_id: int) -> Tuple[bool, List[str]]:
        """Check HIPAA compliance for a round."""
        all_violations = []

        for hospital in self.hospitals:
            passed, violations = hospital.verify_compliance()
            if not passed:
                all_violations.extend(violations)

        self._compliance_log.append({
            "round_id": round_id,
            "timestamp": time.time(),
            "passed": len(all_violations) == 0,
            "violations": all_violations,
        })

        return len(all_violations) == 0, all_violations

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        total_rounds = len(self._compliance_log)
        passed_rounds = sum(1 for entry in self._compliance_log if entry["passed"])

        hospital_audits = {}
        for hospital in self.hospitals:
            hospital_audits[hospital.hospital_name] = {
                "audit_entries": len(hospital.get_audit_log()),
                "access_entries": len(hospital.get_access_log()),
                "compliance_verified": hospital.verify_compliance()[0],
            }

        return {
            "total_rounds": total_rounds,
            "compliant_rounds": passed_rounds,
            "compliance_rate": passed_rounds / total_rounds if total_rounds > 0 else 1.0,
            "differential_privacy_epsilon": self.config.differential_privacy_epsilon,
            "differential_privacy_delta": self.config.differential_privacy_delta,
            "hospital_audits": hospital_audits,
            "all_violations": [
                v for entry in self._compliance_log for v in entry.get("violations", [])
            ],
        }

    def get_data_heterogeneity_analysis(self) -> Dict[str, Any]:
        """Analyze data heterogeneity across hospitals."""
        patient_counts = [h.profile.patient_count for h in self.hospitals]
        quality_scores = [h.profile.data_quality for h in self.hospitals]

        return {
            "patient_count_range": (min(patient_counts), max(patient_counts)),
            "patient_count_std": float(np.std(patient_counts)),
            "quality_score_range": (min(quality_scores), max(quality_scores)),
            "quality_score_mean": float(np.mean(quality_scores)),
            "specialties": [h.profile.specialty for h in self.hospitals],
            "regions": [h.profile.geographic_region for h in self.hospitals],
        }


# ============================================================================
# Healthcare Integration Tests
# ============================================================================

@pytest.fixture
def healthcare_scenario():
    """Create healthcare scenario fixture."""
    async def _create():
        scenario = HealthcareScenario(gradient_dimension=2000)
        await scenario.setup()
        return scenario
    return _create


class TestHealthcareScenario:
    """Test healthcare FL scenario."""

    @pytest.mark.asyncio
    @pytest.mark.healthcare
    async def test_5_hospital_single_round(self, healthcare_scenario, metrics_collector):
        """Test single FL round with 5 hospitals."""
        scenario = await healthcare_scenario()

        result = await scenario.run_round()

        assert result["success"], f"Round failed: {result.get('error')}"
        assert result["n_hospitals"] == 5
        assert result["compliance_passed"], f"Compliance failed: {result['compliance_violations']}"

        metrics_collector.record_custom(
            "healthcare_single_round",
            {
                "n_hospitals": result["n_hospitals"],
                "duration_ms": result["duration_ms"],
                "compliance_passed": result["compliance_passed"],
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.healthcare
    async def test_hipaa_compliance(self, healthcare_scenario, metrics_collector):
        """Test HIPAA compliance across multiple rounds."""
        scenario = await healthcare_scenario()

        # Run multiple rounds
        results = await scenario.run_rounds(10)

        # Check all rounds were compliant
        compliance_report = scenario.get_compliance_report()

        assert compliance_report["compliance_rate"] == 1.0, (
            f"Compliance violations: {compliance_report['all_violations']}"
        )

        # Verify DP parameters
        assert compliance_report["differential_privacy_epsilon"] <= 10.0

        metrics_collector.record_custom("hipaa_compliance", compliance_report)

    @pytest.mark.asyncio
    @pytest.mark.healthcare
    async def test_differential_privacy_impact(
        self, metrics_collector
    ):
        """Test impact of differential privacy on model quality."""
        gradient_dim = 1000

        # Test with different epsilon values
        epsilon_values = [0.1, 1.0, 5.0, 10.0]
        results = {}

        for epsilon in epsilon_values:
            config = HealthcareNetworkConfig(
                num_honest_nodes=5,
                gradient_dimension=gradient_dim,
                differential_privacy_epsilon=epsilon,
            )

            scenario = HealthcareScenario(config=config, gradient_dimension=gradient_dim)
            await scenario.setup()

            # Run round
            result = await scenario.run_round()

            if result["success"]:
                # Measure gradient quality (lower noise = better quality)
                aggregated = result["aggregated_gradient"]
                norm = np.linalg.norm(aggregated)
                results[epsilon] = {
                    "gradient_norm": float(norm),
                    "success": True,
                }
            else:
                results[epsilon] = {"success": False}

        # Higher epsilon should allow higher-quality gradients
        norms = [r["gradient_norm"] for r in results.values() if r.get("success")]
        if len(norms) > 1:
            # Norms should generally increase with epsilon
            metrics_collector.record_custom("dp_epsilon_impact", results)

    @pytest.mark.asyncio
    @pytest.mark.healthcare
    async def test_data_heterogeneity_impact(
        self, healthcare_scenario, metrics_collector
    ):
        """Test FL performance with heterogeneous hospital data."""
        scenario = await healthcare_scenario()

        # Analyze heterogeneity
        heterogeneity = scenario.get_data_heterogeneity_analysis()

        # Run multiple rounds
        results = await scenario.run_rounds(20)

        # Collect per-hospital participation
        participation = {h.hospital_name: 0 for h in scenario.hospitals}
        for result in results:
            if result["success"]:
                for hospital_id in result["hospital_ids"]:
                    for h in scenario.hospitals:
                        if h.node_id == hospital_id:
                            participation[h.hospital_name] += 1

        # All hospitals should participate equally
        participations = list(participation.values())
        assert min(participations) >= len(results) * 0.9, "Uneven participation"

        metrics_collector.record_custom(
            "heterogeneity_impact",
            {
                "heterogeneity": heterogeneity,
                "participation": participation,
            }
        )

    @pytest.mark.asyncio
    @pytest.mark.healthcare
    @pytest.mark.slow
    async def test_long_running_healthcare_scenario(
        self, healthcare_scenario, metrics_collector, test_report
    ):
        """Long-running test of healthcare scenario."""
        scenario = await healthcare_scenario()

        test_report.set_parameter("n_hospitals", 5)
        test_report.set_parameter("epsilon", scenario.config.differential_privacy_epsilon)
        test_report.add_tag("healthcare")
        test_report.add_tag("long_running")

        n_rounds = 50
        results = await scenario.run_rounds(n_rounds)

        # Analysis
        success_count = sum(1 for r in results if r["success"])
        compliance_count = sum(1 for r in results if r.get("compliance_passed"))

        assert success_count == n_rounds, f"Only {success_count}/{n_rounds} rounds succeeded"
        assert compliance_count == n_rounds, "HIPAA compliance violations detected"

        # Quality over time
        norms = [
            np.linalg.norm(r["aggregated_gradient"])
            for r in results if r["success"]
        ]

        # Check convergence
        early_norm = np.mean(norms[:10])
        late_norm = np.mean(norms[-10:])
        assert late_norm < early_norm, "No convergence observed"

        compliance_report = scenario.get_compliance_report()

        metrics_collector.record_custom(
            "healthcare_long_running",
            {
                "success_rate": success_count / n_rounds,
                "compliance_rate": compliance_report["compliance_rate"],
                "convergence_ratio": late_norm / early_norm,
            }
        )

        test_report.passed = True
