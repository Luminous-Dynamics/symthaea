#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real-Time Federated Learning Dashboard

Beautiful web dashboard showing:
- Training progress (loss curves)
- Byzantine detection timeline
- Credit accumulation (honest vs malicious)
- Model accuracy convergence
- Network health metrics

Runs at: http://localhost:8050
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
except ImportError:
    print("Error: Missing dependencies")
    print("Run: nix develop")
    exit(1)


# Mock data generator (replace with real data in production)
class DataGenerator:
    """Generate mock FL training data"""

    def __init__(self):
        self.round_num = 0
        self.honest_loss = 2.3
        self.byzantine_detected = []
        self.credits = {
            "Hospital A": 0,
            "Hospital B": 0,
            "Hospital C": 0,
            "Byzantine": 0
        }

    def next_round(self):
        """Simulate one training round"""
        self.round_num += 1

        # Simulate loss decrease (with noise)
        self.honest_loss *= 0.92 + random.uniform(-0.02, 0.02)

        # Simulate Byzantine detection (random events)
        if random.random() < 0.3:  # 30% chance per round
            self.byzantine_detected.append({
                "round": self.round_num,
                "node": "Byzantine Node",
                "reason": "Low PoGQ score",
                "score": random.uniform(0.1, 0.3)
            })

        # Update credits (honest nodes earn, Byzantine doesn't)
        quality_score = max(0, 1.0 - (self.honest_loss / 2.3))  # Decreases as we improve
        credits_per_round = int(quality_score * 50)

        self.credits["Hospital A"] += credits_per_round + random.randint(-5, 5)
        self.credits["Hospital B"] += credits_per_round + random.randint(-5, 5)
        self.credits["Hospital C"] += credits_per_round + random.randint(-5, 5)
        # Byzantine stays at 0

        return {
            "round": self.round_num,
            "honest_loss": self.honest_loss,
            "byzantine_loss": self.honest_loss * 1.5,  # Byzantine doesn't improve
            "credits": self.credits.copy(),
            "byzantine_events": len(self.byzantine_detected)
        }


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "ZeroTrustML Federated Learning Dashboard"

# Data generator
data_gen = DataGenerator()
training_data = []

# Dashboard layout
app.layout = html.Div(
    style={"backgroundColor": "#1e1e1e", "color": "#ffffff", "fontFamily": "monospace"},
    children=[
        # Header
        html.Div([
            html.H1(
                "🔬 ZeroTrustML Federated Learning Dashboard",
                style={"textAlign": "center", "marginTop": "20px", "marginBottom": "10px"}
            ),
            html.H3(
                "Real-time Byzantine-Resistant Federated Learning",
                style={"textAlign": "center", "color": "#3498db", "marginBottom": "30px"}
            )
        ]),

        # Status row
        html.Div([
            html.Div([
                html.H4("Round", style={"textAlign": "center"}),
                html.H2(id="round-counter", children="0", style={"textAlign": "center", "color": "#2ecc71"})
            ], style={"width": "20%", "display": "inline-block"}),

            html.Div([
                html.H4("Byzantine Detected", style={"textAlign": "center"}),
                html.H2(id="byzantine-counter", children="0", style={"textAlign": "center", "color": "#e74c3c"})
            ], style={"width": "20%", "display": "inline-block"}),

            html.Div([
                html.H4("Model Loss", style={"textAlign": "center"}),
                html.H2(id="loss-value", children="2.30", style={"textAlign": "center", "color": "#f39c12"})
            ], style={"width": "20%", "display": "inline-block"}),

            html.Div([
                html.H4("Avg Credits", style={"textAlign": "center"}),
                html.H2(id="credit-value", children="0", style={"textAlign": "center", "color": "#9b59b6"})
            ], style={"width": "20%", "display": "inline-block"}),

            html.Div([
                html.H4("Network", style={"textAlign": "center"}),
                html.H2("HEALTHY", style={"textAlign": "center", "color": "#2ecc71"})
            ], style={"width": "20%", "display": "inline-block"}),
        ], style={"textAlign": "center", "marginBottom": "30px"}),

        # Main charts (2 rows x 2 cols)
        html.Div([
            # Row 1
            html.Div([
                dcc.Graph(id="loss-chart", style={"width": "48%", "display": "inline-block"}),
                dcc.Graph(id="credit-chart", style={"width": "48%", "display": "inline-block", "marginLeft": "2%"})
            ]),

            # Row 2
            html.Div([
                dcc.Graph(id="detection-timeline", style={"width": "48%", "display": "inline-block"}),
                dcc.Graph(id="accuracy-chart", style={"width": "48%", "display": "inline-block", "marginLeft": "2%"})
            ])
        ]),

        # Auto-refresh interval (1 second)
        dcc.Interval(
            id="interval-component",
            interval=1000,  # milliseconds
            n_intervals=0
        ),

        # Footer
        html.Div([
            html.P(
                "🌐 ZeroTrustML - Byzantine-Resistant Federated Learning on Holochain",
                style={"textAlign": "center", "marginTop": "20px", "color": "#7f8c8d"}
            )
        ])
    ]
)


# Callbacks for real-time updates
@app.callback(
    [Output("round-counter", "children"),
     Output("byzantine-counter", "children"),
     Output("loss-value", "children"),
     Output("credit-value", "children"),
     Output("loss-chart", "figure"),
     Output("credit-chart", "figure"),
     Output("detection-timeline", "figure"),
     Output("accuracy-chart", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    """Update all dashboard components"""

    # Generate new data
    new_data = data_gen.next_round()
    training_data.append(new_data)

    # Limit history to last 50 rounds
    if len(training_data) > 50:
        training_data.pop(0)

    # Extract data for charts
    rounds = [d["round"] for d in training_data]
    honest_losses = [d["honest_loss"] for d in training_data]
    byzantine_losses = [d["byzantine_loss"] for d in training_data]

    # Calculate average honest credits
    avg_credits = int((
        new_data["credits"]["Hospital A"] +
        new_data["credits"]["Hospital B"] +
        new_data["credits"]["Hospital C"]
    ) / 3)

    # Loss chart
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=rounds,
        y=honest_losses,
        mode='lines+markers',
        name='Honest Nodes',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=6)
    ))
    loss_fig.add_trace(go.Scatter(
        x=rounds,
        y=byzantine_losses,
        mode='lines+markers',
        name='Byzantine (no improvement)',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    loss_fig.update_layout(
        title="Training Loss Over Time",
        xaxis_title="Round",
        yaxis_title="Loss",
        template="plotly_dark",
        hovermode="x unified"
    )

    # Credit accumulation chart
    credit_fig = go.Figure()
    credit_fig.add_trace(go.Bar(
        x=list(new_data["credits"].keys()),
        y=list(new_data["credits"].values()),
        marker=dict(color=['#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c'])
    ))
    credit_fig.update_layout(
        title="Credit Accumulation (Reputation)",
        xaxis_title="Node",
        yaxis_title="Total Credits",
        template="plotly_dark"
    )

    # Byzantine detection timeline
    detection_rounds = [e["round"] for e in data_gen.byzantine_detected]
    detection_fig = go.Figure()
    detection_fig.add_trace(go.Scatter(
        x=detection_rounds,
        y=[1] * len(detection_rounds),
        mode='markers',
        name='Byzantine Attack Detected',
        marker=dict(
            color='#e74c3c',
            size=15,
            symbol='x',
            line=dict(width=2)
        )
    ))
    detection_fig.update_layout(
        title="Byzantine Detection Events",
        xaxis_title="Round",
        yaxis=dict(showticklabels=False),
        template="plotly_dark",
        height=250
    )

    # Accuracy chart (derived from loss)
    accuracies = [max(0, min(100, 100 - (loss * 40))) for loss in honest_losses]
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(
        x=rounds,
        y=accuracies,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    accuracy_fig.update_layout(
        title="Model Accuracy Convergence",
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark"
    )

    return (
        str(new_data["round"]),
        str(new_data["byzantine_events"]),
        f"{new_data['honest_loss']:.2f}",
        str(avg_credits),
        loss_fig,
        credit_fig,
        detection_fig,
        accuracy_fig
    )


def main():
    """Run the dashboard server"""

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║      🔬 ZeroTrustML Federated Learning Dashboard              ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print("📊 Starting real-time dashboard...")
    print()
    print("   URL: http://localhost:8050")
    print()
    print("   Features:")
    print("     - Live training progress")
    print("     - Byzantine detection events")
    print("     - Credit accumulation")
    print("     - Model accuracy tracking")
    print()
    print("   Press Ctrl+C to stop")
    print()

    # Run Dash server
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=8050
    )


if __name__ == "__main__":
    main()
