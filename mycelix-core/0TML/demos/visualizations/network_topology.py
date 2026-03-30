#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Network Topology Visualization - Interactive P2P Network Graph

Creates beautiful interactive visualization of:
- 3 Holochain conductors (Boston, London, Tokyo)
- 3 ZeroTrustML nodes (hospitals)
- P2P connections
- DHT data flow
- Byzantine node isolation

Output: outputs/interactive/network_topology.html
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pyvis.network import Network
    import networkx as nx
except ImportError:
    print("Error: Missing dependencies")
    print("Run: nix develop")
    exit(1)


class NetworkTopologyVisualizer:
    """Interactive network topology visualization"""

    def __init__(self):
        self.net = Network(
            height="800px",
            width="100%",
            bgcolor="#1e1e1e",
            font_color="#ffffff",
            notebook=False,
            cdn_resources="remote"
        )

        # Configure physics for nice layout
        self.net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

    def add_holochain_nodes(self):
        """Add 3 Holochain conductor nodes"""

        nodes = [
            {
                "id": "holochain-boston",
                "label": "Holochain\nBoston",
                "title": "Conductor 1 (Port 8881)\nDHT Node\nStatus: HEALTHY",
                "color": "#3498db",  # Blue
                "size": 40,
                "shape": "hexagon"
            },
            {
                "id": "holochain-london",
                "label": "Holochain\nLondon",
                "title": "Conductor 2 (Port 8882)\nDHT Node\nStatus: HEALTHY",
                "color": "#3498db",  # Blue
                "size": 40,
                "shape": "hexagon"
            },
            {
                "id": "holochain-tokyo",
                "label": "Holochain\nTokyo",
                "title": "Conductor 3 (Port 8883)\nDHT Node\nStatus: HEALTHY",
                "color": "#3498db",  # Blue
                "size": 40,
                "shape": "hexagon"
            }
        ]

        for node in nodes:
            self.net.add_node(
                node["id"],
                label=node["label"],
                title=node["title"],
                color=node["color"],
                size=node["size"],
                shape=node["shape"]
            )

    def add_zerotrustml_nodes(self, include_byzantine: bool = False):
        """Add ZeroTrustML application nodes"""

        nodes = [
            {
                "id": "zerotrustml-hospital-a",
                "label": "Hospital A\n(ZeroTrustML)",
                "title": "Honest Node\nCredits: 450\nRounds: 10",
                "color": "#2ecc71",  # Green (honest)
                "size": 35,
                "shape": "dot"
            },
            {
                "id": "zerotrustml-hospital-b",
                "label": "Hospital B\n(ZeroTrustML)",
                "title": "Honest Node\nCredits: 430\nRounds: 10",
                "color": "#2ecc71",  # Green (honest)
                "size": 35,
                "shape": "dot"
            },
            {
                "id": "zerotrustml-hospital-c",
                "label": "Hospital C\n(ZeroTrustML)",
                "title": "Honest Node\nCredits: 460\nRounds: 10",
                "color": "#2ecc71",  # Green (honest)
                "size": 35,
                "shape": "dot"
            }
        ]

        if include_byzantine:
            nodes.append({
                "id": "zerotrustml-byzantine",
                "label": "Malicious\n(Byzantine)",
                "title": "Byzantine Node\nCredits: 0\nIsolated: YES",
                "color": "#e74c3c",  # Red (malicious)
                "size": 35,
                "shape": "dot"
            })

        for node in nodes:
            self.net.add_node(
                node["id"],
                label=node["label"],
                title=node["title"],
                color=node["color"],
                size=node["size"],
                shape=node["shape"]
            )

    def add_p2p_connections(self):
        """Add P2P connections between Holochain conductors"""

        connections = [
            ("holochain-boston", "holochain-london", "P2P DHT Sync"),
            ("holochain-london", "holochain-tokyo", "P2P DHT Sync"),
            ("holochain-tokyo", "holochain-boston", "P2P DHT Sync")
        ]

        for source, target, label in connections:
            self.net.add_edge(
                source,
                target,
                title=label,
                color="#9b59b6",  # Purple
                width=2,
                arrows="to"
            )

    def add_application_connections(self, include_byzantine: bool = False):
        """Add connections from ZeroTrustML nodes to Holochain conductors"""

        connections = [
            ("zerotrustml-hospital-a", "holochain-boston", "WebSocket"),
            ("zerotrustml-hospital-b", "holochain-london", "WebSocket"),
            ("zerotrustml-hospital-c", "holochain-tokyo", "WebSocket")
        ]

        if include_byzantine:
            # Byzantine node tries to connect but gets rejected
            connections.append(
                ("zerotrustml-byzantine", "holochain-boston", "REJECTED")
            )

        for source, target, label in connections:
            color = "#e74c3c" if label == "REJECTED" else "#95a5a6"  # Red if rejected, gray otherwise
            width = 1 if label == "REJECTED" else 2

            self.net.add_edge(
                source,
                target,
                title=label,
                color=color,
                width=width,
                dashes=label == "REJECTED"  # Dashed if rejected
            )

    def add_database(self):
        """Add PostgreSQL database (optional, for demo)"""

        self.net.add_node(
            "postgresql",
            label="PostgreSQL\n(Shared)",
            title="PostgreSQL Database\nStatus: HEALTHY\nConnections: 3",
            color="#f39c12",  # Orange
            size=30,
            shape="database"
        )

        # Connect ZeroTrustML nodes to database
        for node_id in ["zerotrustml-hospital-a", "zerotrustml-hospital-b", "zerotrustml-hospital-c"]:
            self.net.add_edge(
                node_id,
                "postgresql",
                title="Local storage",
                color="#95a5a6",
                width=1,
                dashes=True  # Dashed to show it's optional
            )

    def generate(
        self,
        output_path: str = "outputs/interactive/network_topology.html",
        include_byzantine: bool = False,
        include_database: bool = True
    ):
        """Generate the interactive HTML visualization"""

        # Add all components
        self.add_holochain_nodes()
        self.add_zerotrustml_nodes(include_byzantine=include_byzantine)
        self.add_p2p_connections()
        self.add_application_connections(include_byzantine=include_byzantine)

        if include_database:
            self.add_database()

        # Add title and legend
        title_html = """
        <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 5px; color: white; font-family: monospace;">
            <h2 style="margin: 0 0 10px 0;">🌐 ZeroTrustML P2P Network Topology</h2>
            <div style="font-size: 12px;">
                <div><span style="color: #3498db;">●</span> Holochain Conductors (DHT)</div>
                <div><span style="color: #2ecc71;">●</span> ZeroTrustML Nodes (Honest)</div>
                <div><span style="color: #e74c3c;">●</span> Byzantine Nodes (Malicious)</div>
                <div><span style="color: #f39c12;">●</span> PostgreSQL (Optional)</div>
            </div>
        </div>
        """

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML
        self.net.show(output_path)

        # Add custom title
        with open(output_path, 'r') as f:
            html_content = f.read()

        # Insert title after body tag
        html_content = html_content.replace(
            '<body>',
            f'<body>{title_html}'
        )

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Network topology saved to: {output_path}")
        print(f"   Open in browser: file://{Path(output_path).absolute()}")


def main():
    """Generate network topology visualizations"""

    print("🌐 Generating Network Topology Visualizations...")
    print()

    viz = NetworkTopologyVisualizer()

    # Generate normal network (no Byzantine)
    print("1. Normal P2P network (3 honest nodes)")
    viz.generate(
        output_path="outputs/interactive/network_topology_normal.html",
        include_byzantine=False,
        include_database=True
    )

    # Generate network with Byzantine node
    print("\n2. Network with Byzantine attack")
    viz2 = NetworkTopologyVisualizer()
    viz2.generate(
        output_path="outputs/interactive/network_topology_byzantine.html",
        include_byzantine=True,
        include_database=True
    )

    print()
    print("✅ Done! Generated 2 interactive visualizations:")
    print("   - network_topology_normal.html (happy path)")
    print("   - network_topology_byzantine.html (attack scenario)")
    print()
    print("💡 Open these in your browser to interact with the network!")


if __name__ == "__main__":
    main()
