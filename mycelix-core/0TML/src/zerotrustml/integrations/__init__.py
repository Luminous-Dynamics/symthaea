"""
ZeroTrustML Integrations

External system integrations for the ZeroTrustML federated learning framework.

Hybrid Architecture:
    - Holochain: FREE, high-frequency FL operations (99.9% of transactions)
    - Ethereum/Polygon: PAID, periodic anchoring and disputes

Usage:
    # Holochain FL operations (FREE)
    from zerotrustml.holochain import FLCoordinatorClient
    async with FLCoordinatorClient() as client:
        await client.submit_gradient(...)

    # Ethereum anchoring (PAID, infrequent)
    from zerotrustml.integrations import EthereumBridge
    bridge = EthereumBridge(network="polygon")
    await bridge.anchor_merkle_root(root, epoch)
"""

from .ethereum_bridge import (
    EthereumBridge,
    ReputationScore,
    TransactionResult,
    create_bridge,
    SEPOLIA_CONTRACTS,
)

# Re-export holochain for convenience
from zerotrustml.holochain import (
    FLCoordinatorClient,
    create_fl_coordinator,
    HolochainClient,
)

__all__ = [
    # Ethereum
    "EthereumBridge",
    "ReputationScore",
    "TransactionResult",
    "create_bridge",
    "SEPOLIA_CONTRACTS",
    # Holochain
    "FLCoordinatorClient",
    "create_fl_coordinator",
    "HolochainClient",
]
