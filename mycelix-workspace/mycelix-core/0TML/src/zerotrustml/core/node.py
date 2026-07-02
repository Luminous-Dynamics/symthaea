# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Federated learning node with Holochain DHT integration.

Provides P2P coordination, gradient aggregation, and DHT storage.
"""

import asyncio
import numpy as np
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from zerotrustml.aggregation import aggregate_gradients

try:
    from zerotrustml.hyperfeel_bridge import (
        encode_gradient as encode_with_hyperfeel,
        HyperFeelUnavailable,
    )
except ImportError:  # If bridge is missing or not on PYTHONPATH
    encode_with_hyperfeel = None  # type: ignore[assignment]

    class HyperFeelUnavailable(RuntimeError):  # type: ignore[no-redef]
        pass


@dataclass
class NodeConfig:
    """Configuration for a ZeroTrustML node"""
    node_id: str
    data_path: str
    model_type: str = "resnet18"
    holochain_url: str = "ws://localhost:8888"
    aggregation: str = "krum"
    batch_size: int = 32
    learning_rate: float = 0.01


@dataclass
class GradientCheckpoint:
    """Immutable gradient checkpoint for DHT storage"""
    round_id: int
    gradient_hash: str
    aggregation_method: str
    contributors: List[int]
    timestamp: float
    model_accuracy: float
    byzantine_detected: int

    def to_dht_entry(self) -> Dict:
        """Convert to DHT-compatible JSON entry"""
        return {
            "entry_type": "gradient_checkpoint",
            "round_id": self.round_id,
            "gradient_hash": self.gradient_hash,
            "aggregation_method": self.aggregation_method,
            "contributors": self.contributors,
            "timestamp": self.timestamp,
            "model_accuracy": self.model_accuracy,
            "byzantine_detected": self.byzantine_detected
        }


class Node:
    """
    ZeroTrustML federated learning node

    High-level interface for:
    - Local training
    - P2P gradient exchange
    - Byzantine-resistant aggregation
    - DHT checkpoint storage
    - Credit tracking
    """

    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_id = config.node_id
        self.model = None  # Will be initialized based on model_type
        self.accuracy = 0.0
        self.rounds_completed = 0
        self.credits = 0

        # To be initialized
        self.gradient_buffer = []
        self.checkpoints = {}
        self.peers = []
        self.local_node = None
        self.running = False
        self._fallback_gradient: Optional[np.ndarray] = None

    async def start(self):
        """
        Start the node and join the network

        This is the main entry point for users:
        >>> node = Node(config)
        >>> await node.start()
        """
        # Initialize model
        from zerotrustml.core.training import RealMLNode

        node_index = self._parse_node_index(self.node_id)
        self.local_node = RealMLNode(node_id=node_index)
        self.model = self.local_node.model

        # Bootstrap from DHT
        await self.bootstrap_from_dht()

        # Start background tasks for gradient exchange and credit tracking
        self.running = True
        self._background_tasks = []

        # Start gradient exchange task
        gradient_exchange_task = asyncio.create_task(
            self._background_gradient_exchange()
        )
        self._background_tasks.append(gradient_exchange_task)

        # Start credit tracking task
        credit_tracking_task = asyncio.create_task(
            self._background_credit_tracking()
        )
        self._background_tasks.append(credit_tracking_task)

        # Start peer health monitoring task
        peer_health_task = asyncio.create_task(
            self._background_peer_health_monitor()
        )
        self._background_tasks.append(peer_health_task)

    async def _background_gradient_exchange(self):
        """
        Background task for P2P gradient exchange with peers.

        Periodically:
        1. Broadcasts local gradients to peers
        2. Collects gradients from peers
        3. Stores received gradients in the buffer for aggregation
        """
        import websockets
        import json

        exchange_interval = 30  # seconds between exchanges

        while self.running:
            try:
                await asyncio.sleep(exchange_interval)

                if not self.peers or not self.gradient_buffer:
                    continue

                # Get latest local gradient
                if self._fallback_gradient is None:
                    continue

                gradient_data = {
                    "node_id": self.node_id,
                    "round": self.rounds_completed,
                    "gradient": self._fallback_gradient.tolist(),
                    "timestamp": time.time(),
                }

                # Broadcast to peers via Holochain
                try:
                    async with websockets.connect(self.config.holochain_url) as ws:
                        # Submit gradient to DHT for peer discovery
                        request = {
                            "jsonrpc": "2.0",
                            "id": f"gradient-{self.node_id}-{self.rounds_completed}",
                            "method": "call_zome",
                            "params": {
                                "cell_id": None,
                                "zome_name": "fl_training",
                                "fn_name": "submit_gradient",
                                "payload": gradient_data,
                            }
                        }
                        await ws.send(json.dumps(request))

                        # Wait for acknowledgment
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        response_data = json.loads(response)

                        if "result" in response_data:
                            print(f"Node {self.node_id}: Gradient broadcasted to DHT")

                        # Fetch gradients from peers
                        fetch_request = {
                            "jsonrpc": "2.0",
                            "id": f"fetch-gradients-{self.node_id}",
                            "method": "call_zome",
                            "params": {
                                "cell_id": None,
                                "zome_name": "fl_training",
                                "fn_name": "get_peer_gradients",
                                "payload": {"round": self.rounds_completed},
                            }
                        }
                        await ws.send(json.dumps(fetch_request))

                        fetch_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        fetch_data = json.loads(fetch_response)

                        if "result" in fetch_data:
                            peer_gradients = fetch_data["result"].get("gradients", [])
                            for pg in peer_gradients:
                                if pg.get("node_id") != self.node_id:
                                    # Add peer gradient to buffer
                                    gradient_array = np.array(pg["gradient"])
                                    self.gradient_buffer.append(gradient_array)
                            print(f"Node {self.node_id}: Received {len(peer_gradients)} peer gradients")

                except asyncio.TimeoutError:
                    print(f"Node {self.node_id}: Gradient exchange timed out")
                except ConnectionRefusedError:
                    print(f"Node {self.node_id}: Could not connect for gradient exchange")
                except Exception as e:
                    print(f"Node {self.node_id}: Gradient exchange error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Node {self.node_id}: Background gradient exchange error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _background_credit_tracking(self):
        """
        Background task for tracking and updating contribution credits.

        Periodically:
        1. Queries DHT for credit balance
        2. Updates local credit count
        3. Logs credit earning events
        """
        import websockets
        import json

        credit_check_interval = 60  # seconds between credit checks

        while self.running:
            try:
                await asyncio.sleep(credit_check_interval)

                try:
                    async with websockets.connect(self.config.holochain_url) as ws:
                        # Query credit balance from DHT
                        request = {
                            "jsonrpc": "2.0",
                            "id": f"credits-{self.node_id}",
                            "method": "call_zome",
                            "params": {
                                "cell_id": None,
                                "zome_name": "credits",
                                "fn_name": "get_balance",
                                "payload": {"holder": self.node_id},
                            }
                        }
                        await ws.send(json.dumps(request))

                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        response_data = json.loads(response)

                        if "result" in response_data:
                            new_credits = response_data["result"].get("balance", 0)
                            if new_credits != self.credits:
                                credit_delta = new_credits - self.credits
                                print(f"Node {self.node_id}: Credits updated: {self.credits} -> {new_credits} (delta: {credit_delta:+d})")
                                self.credits = new_credits

                        # Query credit history for recent earnings
                        history_request = {
                            "jsonrpc": "2.0",
                            "id": f"credit-history-{self.node_id}",
                            "method": "call_zome",
                            "params": {
                                "cell_id": None,
                                "zome_name": "credits",
                                "fn_name": "get_recent_transactions",
                                "payload": {
                                    "holder": self.node_id,
                                    "limit": 10,
                                },
                            }
                        }
                        await ws.send(json.dumps(history_request))

                        history_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        history_data = json.loads(history_response)

                        if "result" in history_data:
                            transactions = history_data["result"].get("transactions", [])
                            # Log any unprocessed transactions
                            for tx in transactions:
                                if tx.get("earned_from") == "gradient_quality":
                                    print(f"Node {self.node_id}: Earned {tx.get('amount', 0)} credits for quality gradient")

                except asyncio.TimeoutError:
                    pass  # Silent timeout - credit tracking is non-critical
                except ConnectionRefusedError:
                    pass  # Silent - conductor may not be running
                except Exception as e:
                    print(f"Node {self.node_id}: Credit tracking error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Node {self.node_id}: Background credit tracking error: {e}")
                await asyncio.sleep(5)

    async def _background_peer_health_monitor(self):
        """
        Background task for monitoring peer health and connectivity.

        Periodically:
        1. Pings known peers
        2. Removes unresponsive peers
        3. Discovers new peers via DHT
        """
        import websockets
        import json

        health_check_interval = 45  # seconds between health checks

        while self.running:
            try:
                await asyncio.sleep(health_check_interval)

                try:
                    async with websockets.connect(self.config.holochain_url) as ws:
                        # Get current peer count from network
                        request = {
                            "jsonrpc": "2.0",
                            "id": f"peer-health-{self.node_id}",
                            "method": "dump_network_stats",
                            "params": {}
                        }
                        await ws.send(json.dumps(request))

                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        response_data = json.loads(response)

                        if "result" in response_data:
                            network_peers = response_data["result"].get("peers", [])
                            active_count = len([p for p in network_peers if p.get("connected", False)])

                            # Update peer list
                            current_peer_keys = {p.get("agent_pub_key") for p in self.peers}
                            new_peers = []

                            for peer in network_peers:
                                peer_key = peer.get("agent", "")
                                if peer_key and peer_key != self.node_id:
                                    if peer_key not in current_peer_keys:
                                        new_peers.append({
                                            "agent_pub_key": peer_key,
                                            "urls": peer.get("urls", []),
                                            "signed_at": peer.get("signed_at_ms", 0)
                                        })

                            if new_peers:
                                self.peers.extend(new_peers)
                                print(f"Node {self.node_id}: Discovered {len(new_peers)} new peers (total: {len(self.peers)})")

                            # Remove stale peers (not seen in network stats)
                            network_peer_keys = {p.get("agent", "") for p in network_peers}
                            self.peers = [p for p in self.peers if p.get("agent_pub_key") in network_peer_keys]

                except asyncio.TimeoutError:
                    pass  # Silent timeout
                except ConnectionRefusedError:
                    pass  # Silent - conductor may not be running
                except Exception as e:
                    print(f"Node {self.node_id}: Peer health monitor error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Node {self.node_id}: Background peer health error: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        """Stop the node and cancel all background tasks."""
        self.running = False

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()
        print(f"Node {self.node_id}: Stopped")

    async def bootstrap_from_dht(self):
        """Connect to network via DHT peer discovery using Holochain conductor"""
        import websockets
        import json

        print(f"Node {self.node_id}: Bootstrapping from DHT via Holochain...")

        try:
            # Connect to Holochain admin interface
            async with websockets.connect(self.config.holochain_url) as ws:
                # Request agent info to discover peers
                request = {
                    "jsonrpc": "2.0",
                    "id": f"bootstrap-{self.node_id}",
                    "method": "agent_info",
                    "params": {"cell_id": None}
                }
                await ws.send(json.dumps(request))
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                agent_data = json.loads(response)

                if "result" in agent_data:
                    agent_infos = agent_data["result"].get("agent_infos", [])
                    self.peers = []

                    for agent in agent_infos:
                        peer_key = agent.get("agent", "")
                        if peer_key and peer_key != self.node_id:
                            self.peers.append({
                                "agent_pub_key": peer_key,
                                "urls": agent.get("urls", []),
                                "signed_at": agent.get("signed_at_ms", 0)
                            })

                    print(f"Node {self.node_id}: Discovered {len(self.peers)} peers via DHT")

                    # Request network stats for additional peer info
                    stats_request = {
                        "jsonrpc": "2.0",
                        "id": f"stats-{self.node_id}",
                        "method": "dump_network_stats",
                        "params": {}
                    }
                    await ws.send(json.dumps(stats_request))
                    stats_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    stats_data = json.loads(stats_response)

                    if "result" in stats_data:
                        network_peers = stats_data["result"].get("peers", [])
                        print(f"Node {self.node_id}: Network reports {len(network_peers)} active connections")

                elif "error" in agent_data:
                    print(f"Node {self.node_id}: DHT bootstrap error: {agent_data['error']}")

        except asyncio.TimeoutError:
            print(f"Node {self.node_id}: DHT bootstrap timed out - conductor may be starting up")
        except ConnectionRefusedError:
            print(f"Node {self.node_id}: Could not connect to Holochain at {self.config.holochain_url}")
            print(f"Node {self.node_id}: Ensure conductor is running (holochain --piped -c conductor-config.yaml)")
        except Exception as e:
            print(f"Node {self.node_id}: DHT bootstrap failed: {e}")
            # Continue without peers - can operate in standalone mode

    async def train(self, rounds: int = 10) -> List[Dict[str, Any]]:
        """
        Run federated training for specified rounds

        Args:
            rounds: Number of training rounds

        Example:
            >>> await node.train(rounds=50)
        """
        if not self.running or self.local_node is None:
            raise RuntimeError("Node must be started before calling train()")

        results: List[Dict[str, Any]] = []

        for _ in range(rounds):
            local_gradient = self.local_node.compute_gradient()
            self._fallback_gradient = local_gradient
            self.gradient_buffer.append(local_gradient)

            aggregated = self._aggregate_gradients()
            self.local_node.apply_gradient_update(
                aggregated,
                learning_rate=self.config.learning_rate
            )

            self.accuracy = self.local_node.evaluate()
            self.rounds_completed += 1

            results.append({
                "round": self.rounds_completed,
                "contributors": 1,
                "accuracy": self.accuracy
            })

            # Yield control to event loop to keep async contract
            await asyncio.sleep(0)

        return results

    def aggregate_gradients(self, gradients: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        """
        Byzantine-resistant aggregation

        Args:
            gradients: List of gradients from peers

        Returns:
            (aggregated_gradient, byzantine_count)
        """
        if not gradients:
            return np.zeros(100), 0  # Placeholder

        # Detect Byzantine gradients using median absolute deviation
        norms = [np.linalg.norm(g) for g in gradients]
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))
        threshold = median + 2.5 * mad

        # Filter out Byzantine gradients
        clean_gradients = []
        byzantine_count = 0

        for g, norm in zip(gradients, norms):
            if norm <= threshold:
                clean_gradients.append(g)
            else:
                byzantine_count += 1

        # Aggregate clean gradients
        if clean_gradients:
            aggregated = np.median(clean_gradients, axis=0)
        else:
            aggregated = np.zeros_like(gradients[0])

        return aggregated, byzantine_count

    def _aggregate_gradients(self) -> np.ndarray:
        """Aggregate gradients from local buffer using configured algorithm."""
        gradients = list(self.gradient_buffer)
        if not gradients:
            if self._fallback_gradient is not None:
                return np.zeros_like(self._fallback_gradient)
            return np.zeros(100)
        reputations = [1.0] * len(gradients)

        aggregated = aggregate_gradients(
            gradients,
            reputations,
            algorithm=self.config.aggregation,
            num_byzantine=max(0, len(gradients) - 2)
        )

        # Reset buffer after aggregation to avoid unbounded growth
        self.gradient_buffer.clear()
        return aggregated

    def encode_aggregated_gradient_hyperfeel(self, aggregated: np.ndarray):
        """
        Optionally encode an aggregated gradient using Rust HyperFeel.

        This is a non-critical helper:
        - If the HyperFeel bridge or CLI is unavailable, it returns None.
        - Failures are logged but do not affect training.
        """
        if encode_with_hyperfeel is None:
            return None

        try:
            # Flatten to 1D for the encoder
            flat = aggregated.reshape(-1).astype(float).tolist()
            return encode_with_hyperfeel(flat)
        except HyperFeelUnavailable:
            return None
        except Exception as exc:
            print(f"[Node] HyperFeel encode failed: {exc}")
            return None

    def _parse_node_index(self, node_identifier: str) -> int:
        """Extract numeric suffix from node ID for deterministic RealMLNode IDs."""
        try:
            suffix = node_identifier.split("-")[-1]
            return int(suffix)
        except (ValueError, IndexError):
            return 0

    async def checkpoint_to_dht(
        self,
        round_id: int,
        aggregated_gradient: np.ndarray,
        contributors: List[int],
        byzantine_count: int
    ) -> Optional[str]:
        """
        Store aggregation checkpoint in DHT for auditability

        Args:
            round_id: Training round number
            aggregated_gradient: Aggregated gradient
            contributors: Node IDs that contributed
            byzantine_count: Number of Byzantine nodes detected

        Returns:
            DHT hash address if successful
        """
        checkpoint = GradientCheckpoint(
            round_id=round_id,
            gradient_hash=hashlib.sha256(aggregated_gradient.tobytes()).hexdigest(),
            aggregation_method=self.config.aggregation,
            contributors=contributors,
            timestamp=time.time(),
            model_accuracy=self.accuracy,
            byzantine_detected=byzantine_count
        )

        # Store in DHT (mock for now)
        self.checkpoints[round_id] = checkpoint
        return checkpoint.gradient_hash[:16]
