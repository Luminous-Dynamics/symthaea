#!/usr/bin/env python3
"""
Mycelix Core REST API Server

Minimal REST API for ecosystem interaction.
Run with: python -m zerotrustml.api.server

Author: Luminous Dynamics
"""

from aiohttp import web
import json
import time
from datetime import datetime
from typing import Any

# Constants (aligned with shared spec)
DEFAULT_QUALITY_WEIGHT = 0.4
DEFAULT_CONSISTENCY_WEIGHT = 0.3
DEFAULT_REPUTATION_WEIGHT = 0.3
MAX_BYZANTINE_TOLERANCE = 0.45
DEFAULT_BYZANTINE_THRESHOLD = 0.5

# In-memory state (would connect to Holochain in production)
_state = {
    "start_time": time.time(),
    "request_count": 0,
    "agents": {},  # agent_id -> trust data
}


def _json_response(data: Any, status: int = 200) -> web.Response:
    """Create a JSON response with CORS headers."""
    return web.Response(
        text=json.dumps(data, indent=2),
        status=status,
        content_type="application/json",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


async def health_handler(request: web.Request) -> web.Response:
    """
    GET /health

    Returns service health status.
    """
    _state["request_count"] += 1
    uptime = time.time() - _state["start_time"]

    return _json_response({
        "status": "healthy",
        "service": "mycelix-core-api",
        "version": "0.6.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


async def status_handler(request: web.Request) -> web.Response:
    """
    GET /status

    Returns ecosystem status summary.
    """
    _state["request_count"] += 1

    # Ecosystem metrics (would be live data in production)
    return _json_response({
        "ecosystem": {
            "name": "Mycelix",
            "version": "0.6.0",
            "holochain_version": "0.6.0",
            "hdk_version": "0.6.0",
        },
        "happs": {
            "production": ["core", "mail", "desci"],
            "beta": ["marketplace", "supplychain", "observatory", "fabrication"],
            "scaffold": 14,
        },
        "sdks": {
            "rust": {"version": "0.6.0", "tests_passing": 996},
            "typescript": {"version": "0.6.0", "tests_passing": 6316},
            "python": {"version": "0.1.0", "tests_passing": 45},
        },
        "matl": {
            "byzantine_tolerance": MAX_BYZANTINE_TOLERANCE,
            "quality_weight": DEFAULT_QUALITY_WEIGHT,
            "consistency_weight": DEFAULT_CONSISTENCY_WEIGHT,
            "reputation_weight": DEFAULT_REPUTATION_WEIGHT,
        },
        "network": {
            "agents_tracked": len(_state["agents"]),
            "requests_served": _state["request_count"],
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


async def trust_handler(request: web.Request) -> web.Response:
    """
    GET /trust/{agent_id}

    Returns trust score for an agent.
    """
    _state["request_count"] += 1
    agent_id = request.match_info.get("agent_id", "")

    if not agent_id:
        return _json_response({"error": "agent_id required"}, status=400)

    # Check if we have cached trust data
    if agent_id in _state["agents"]:
        agent_data = _state["agents"][agent_id]
    else:
        # New agent - return initial trust (Bayesian prior)
        agent_data = {
            "positive_count": 0,
            "negative_count": 0,
            "last_update": datetime.utcnow().isoformat() + "Z",
        }
        _state["agents"][agent_id] = agent_data

    # Calculate Bayesian-smoothed reputation
    alpha = 1 + agent_data["positive_count"]
    beta = 1 + agent_data["negative_count"]
    reputation = alpha / (alpha + beta)

    # Composite trust (simplified - would include PoGQ in production)
    composite = (
        0.5 * reputation +  # Base reputation
        0.3 * 0.5 +  # Default quality (neutral)
        0.2 * 0.5   # Default consistency (neutral)
    )

    return _json_response({
        "agent_id": agent_id,
        "trust": {
            "reputation": round(reputation, 4),
            "composite_score": round(composite, 4),
            "is_trustworthy": composite >= DEFAULT_BYZANTINE_THRESHOLD,
        },
        "interactions": {
            "positive": agent_data["positive_count"],
            "negative": agent_data["negative_count"],
            "total": agent_data["positive_count"] + agent_data["negative_count"],
        },
        "thresholds": {
            "byzantine_threshold": DEFAULT_BYZANTINE_THRESHOLD,
            "max_byzantine_tolerance": MAX_BYZANTINE_TOLERANCE,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


async def pogq_validate_handler(request: web.Request) -> web.Response:
    """
    POST /pogq/validate

    Validates a Proof of Gradient Quality submission.

    Body: {
        "quality": 0.95,
        "consistency": 0.88,
        "entropy": 0.12
    }
    """
    _state["request_count"] += 1

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "Invalid JSON body"}, status=400)

    quality = body.get("quality")
    consistency = body.get("consistency")
    entropy = body.get("entropy")

    # Validate inputs
    errors = []
    for name, value in [("quality", quality), ("consistency", consistency), ("entropy", entropy)]:
        if value is None:
            errors.append(f"{name} is required")
        elif not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
        elif not (0 <= value <= 1):
            errors.append(f"{name} must be between 0 and 1")

    if errors:
        return _json_response({"error": "Validation failed", "details": errors}, status=400)

    # Calculate composite score
    pogq_score = (
        quality * DEFAULT_QUALITY_WEIGHT +
        consistency * DEFAULT_CONSISTENCY_WEIGHT
    )

    # Honesty metric (high quality + high consistency + low entropy)
    honesty_score = (quality + consistency + (1 - entropy)) / 3
    is_byzantine = honesty_score < DEFAULT_BYZANTINE_THRESHOLD

    return _json_response({
        "valid": True,
        "pogq": {
            "quality": quality,
            "consistency": consistency,
            "entropy": entropy,
        },
        "scores": {
            "pogq_score": round(pogq_score, 4),
            "honesty_score": round(honesty_score, 4),
        },
        "assessment": {
            "is_byzantine": is_byzantine,
            "confidence": round(1 - abs(honesty_score - DEFAULT_BYZANTINE_THRESHOLD) * 2, 4),
        },
        "weights_used": {
            "quality": DEFAULT_QUALITY_WEIGHT,
            "consistency": DEFAULT_CONSISTENCY_WEIGHT,
            "reputation": DEFAULT_REPUTATION_WEIGHT,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


async def options_handler(request: web.Request) -> web.Response:
    """Handle CORS preflight requests."""
    return web.Response(
        status=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()

    # Routes
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_get("/trust/{agent_id}", trust_handler)
    app.router.add_post("/pogq/validate", pogq_validate_handler)

    # CORS preflight
    app.router.add_options("/health", options_handler)
    app.router.add_options("/status", options_handler)
    app.router.add_options("/trust/{agent_id}", options_handler)
    app.router.add_options("/pogq/validate", options_handler)

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the API server."""
    print(f"Starting Mycelix Core API on http://{host}:{port}")
    print("Endpoints:")
    print("  GET  /health           - Service health check")
    print("  GET  /status           - Ecosystem status summary")
    print("  GET  /trust/{agent_id} - Agent trust score")
    print("  POST /pogq/validate    - Validate PoGQ submission")
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
