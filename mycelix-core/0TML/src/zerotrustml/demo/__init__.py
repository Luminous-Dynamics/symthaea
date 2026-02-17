"""
ZeroTrustML Demo - Multi-Hospital Federated Learning

Demonstrates privacy-preserving federated learning with:
- Real Bulletproofs (608-byte zero-knowledge proofs)
- Byzantine-resistant aggregation
- Fair credit distribution
- PostgreSQL persistence
"""

from .hospital_node import HospitalNode
from .byzantine_node import ByzantineNode

__all__ = ['HospitalNode', 'ByzantineNode']
