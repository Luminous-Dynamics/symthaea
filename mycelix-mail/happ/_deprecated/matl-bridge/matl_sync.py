#!/usr/bin/env python3
"""
Mycelix Mail MATL Bridge Service
=================================

Syncs trust scores from the MATL (0TML) database to Holochain DNA for spam filtering.

This service implements Layer 6 (MATL) integration of the Mycelix Protocol,
enabling trust-based spam filtering in Mycelix Mail.

Components:
    1. Trust Score Sync - 0TML → Holochain (every 5 minutes)
    2. Spam Report Sync - Holochain → 0TML (real-time)
    3. Health Monitoring - Service status and metrics

Architecture:
    ┌──────────────┐      ┌────────────────┐      ┌──────────────┐
    │  0TML        │─────▶│  MATL Bridge   │─────▶│  Holochain   │
    │  Database    │      │  Sync Service  │      │     DNA      │
    │ (PostgreSQL) │◀─────│   (Python)     │◀─────│              │
    └──────────────┘      └────────────────┘      └──────────────┘
         Trust               Bi-directional            Mail
         Scores              Synchronization           Messages

Environment Variables:
    MATL_DATABASE_URL      - 0TML PostgreSQL connection string
    HOLOCHAIN_URL          - Holochain conductor WebSocket URL
    HOLOCHAIN_APP_ID       - Installed app ID (default: mycelix-mail)
    SYNC_INTERVAL          - Sync interval in seconds (default: 300)
    PORT                   - Health API port (default: 8400)
    LOG_LEVEL              - Logging level (default: INFO)
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import websockets

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app for health monitoring
app = FastAPI(
    title="Mycelix MATL Bridge",
    description="Layer 6 Trust Score Sync Service",
    version="1.0.0"
)

# Global state
matl_db_pool: Optional[asyncpg.Pool] = None
holochain_ws: Optional[websockets.WebSocketClientProtocol] = None
sync_stats = {
    "last_sync": None,
    "total_syncs": 0,
    "trust_scores_synced": 0,
    "spam_reports_synced": 0,
    "errors": 0,
    "uptime_start": datetime.utcnow()
}


# Pydantic models
class TrustScore(BaseModel):
    """Trust score from MATL database"""
    did: str
    composite_score: float  # 0.0 - 1.0
    pogq_score: float
    tcdm_score: float
    entropy_score: float
    updated_at: datetime


class SpamReport(BaseModel):
    """Spam report from Holochain"""
    reporter_did: str
    spammer_did: str
    message_hash: str
    reason: str
    reported_at: datetime


# Database functions
async def get_matl_db_pool() -> asyncpg.Pool:
    """Get MATL database connection pool"""
    global matl_db_pool
    if matl_db_pool is None:
        database_url = os.getenv("MATL_DATABASE_URL")
        if not database_url:
            raise RuntimeError("MATL_DATABASE_URL environment variable not set")

        matl_db_pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("MATL database connection pool created")

    return matl_db_pool


async def fetch_updated_trust_scores(since: datetime) -> List[TrustScore]:
    """
    Fetch trust scores updated since last sync.

    Args:
        since: Only fetch scores updated after this time

    Returns:
        List of TrustScore objects
    """
    pool = await get_matl_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                agent_did as did,
                composite_score,
                pogq_score,
                tcdm_score,
                entropy_score,
                updated_at
            FROM agent_reputations
            WHERE updated_at > $1
            AND is_active = true
            ORDER BY updated_at DESC
            LIMIT 10000
        """, since)

        return [TrustScore(**dict(row)) for row in rows]


async def store_spam_report(report: SpamReport) -> bool:
    """
    Store spam report in MATL database.

    Args:
        report: SpamReport object

    Returns:
        True if successful
    """
    pool = await get_matl_db_pool()

    async with pool.acquire() as conn:
        try:
            await conn.execute("""
                INSERT INTO spam_reports (
                    reporter_did,
                    spammer_did,
                    message_hash,
                    reason,
                    reported_at
                )
                VALUES ($1, $2, $3, $4, $5)
            """,
                report.reporter_did,
                report.spammer_did,
                report.message_hash,
                report.reason,
                report.reported_at
            )

            logger.info(f"Stored spam report: {report.spammer_did} by {report.reporter_did}")
            return True

        except Exception as e:
            logger.error(f"Failed to store spam report: {e}")
            return False


# Holochain integration
async def connect_to_holochain() -> websockets.WebSocketClientProtocol:
    """
    Connect to Holochain conductor WebSocket.

    Returns:
        WebSocket connection
    """
    holochain_url = os.getenv("HOLOCHAIN_URL", "ws://localhost:8888")

    logger.info(f"Connecting to Holochain at {holochain_url}")
    ws = await websockets.connect(holochain_url)
    logger.info("Connected to Holochain conductor")

    return ws


async def call_holochain_zome(
    zome: str,
    function: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call a Holochain zome function.

    Args:
        zome: Zome name (e.g., "trust_filter")
        function: Function name (e.g., "update_trust_score")
        payload: Function arguments

    Returns:
        Response from Holochain
    """
    global holochain_ws

    if holochain_ws is None or holochain_ws.closed:
        holochain_ws = await connect_to_holochain()

    app_id = os.getenv("HOLOCHAIN_APP_ID", "mycelix-mail")

    # Holochain conductor API format
    request = {
        "type": "zome_call",
        "data": {
            "cell_id": [app_id, "mycelix_mail"],  # [app_id, dna_name]
            "zome": zome,
            "fn_name": function,
            "payload": payload,
            "provenance": None  # Will use conductor's agent
        }
    }

    await holochain_ws.send(json.dumps(request))
    response = await holochain_ws.recv()

    return json.loads(response)


async def sync_trust_score_to_holochain(score: TrustScore) -> bool:
    """
    Sync a single trust score to Holochain.

    Args:
        score: TrustScore to sync

    Returns:
        True if successful
    """
    try:
        # Call update_trust_score function in trust_filter zome
        response = await call_holochain_zome(
            zome="trust_filter",
            function="update_trust_score",
            payload={
                "did": score.did,
                "score": score.composite_score,
                "timestamp": int(score.updated_at.timestamp())
            }
        )

        if response.get("type") == "success":
            logger.debug(f"Synced trust score for {score.did}: {score.composite_score:.3f}")
            return True
        else:
            logger.error(f"Failed to sync {score.did}: {response.get('data')}")
            return False

    except Exception as e:
        logger.error(f"Error syncing trust score for {score.did}: {e}")
        return False


async def fetch_spam_reports_from_holochain(since: datetime) -> List[SpamReport]:
    """
    Fetch spam reports from Holochain DNA.

    Args:
        since: Only fetch reports after this time

    Returns:
        List of SpamReport objects
    """
    try:
        # Call get_spam_reports function (would need to be added to trust_filter zome)
        response = await call_holochain_zome(
            zome="trust_filter",
            function="get_spam_reports",
            payload={
                "since": int(since.timestamp())
            }
        )

        if response.get("type") == "success":
            reports_data = response.get("data", [])
            return [SpamReport(**report) for report in reports_data]
        else:
            logger.error(f"Failed to fetch spam reports: {response.get('data')}")
            return []

    except Exception as e:
        logger.error(f"Error fetching spam reports: {e}")
        return []


# Main sync loop
async def sync_trust_scores():
    """
    Main sync loop: MATL → Holochain.

    Runs periodically to sync updated trust scores.
    """
    logger.info("Starting trust score sync loop")

    sync_interval = int(os.getenv("SYNC_INTERVAL", "300"))  # 5 minutes default
    last_sync = datetime.utcnow() - timedelta(days=1)  # Sync last 24 hours on start

    while True:
        try:
            logger.info("Fetching updated trust scores from MATL...")

            # Fetch updated scores
            scores = await fetch_updated_trust_scores(last_sync)
            logger.info(f"Found {len(scores)} updated trust scores")

            # Sync each score to Holochain
            success_count = 0
            for score in scores:
                if await sync_trust_score_to_holochain(score):
                    success_count += 1

            # Update stats
            sync_stats["last_sync"] = datetime.utcnow()
            sync_stats["total_syncs"] += 1
            sync_stats["trust_scores_synced"] += success_count

            logger.info(f"Synced {success_count}/{len(scores)} trust scores successfully")

            # Update last sync time
            last_sync = datetime.utcnow()

            # Wait for next sync
            await asyncio.sleep(sync_interval)

        except Exception as e:
            logger.error(f"Error in sync loop: {e}")
            sync_stats["errors"] += 1
            await asyncio.sleep(60)  # Wait 1 minute on error


async def sync_spam_reports():
    """
    Spam report sync loop: Holochain → MATL.

    Runs periodically to fetch spam reports from Holochain.
    """
    logger.info("Starting spam report sync loop")

    sync_interval = int(os.getenv("SYNC_INTERVAL", "300"))
    last_sync = datetime.utcnow() - timedelta(hours=1)  # Sync last hour on start

    while True:
        try:
            logger.info("Fetching spam reports from Holochain...")

            # Fetch reports
            reports = await fetch_spam_reports_from_holochain(last_sync)
            logger.info(f"Found {len(reports)} spam reports")

            # Store each report in MATL
            success_count = 0
            for report in reports:
                if await store_spam_report(report):
                    success_count += 1

            sync_stats["spam_reports_synced"] += success_count
            logger.info(f"Stored {success_count}/{len(reports)} spam reports")

            # Update last sync time
            last_sync = datetime.utcnow()

            # Wait for next sync
            await asyncio.sleep(sync_interval)

        except Exception as e:
            logger.error(f"Error in spam report sync: {e}")
            sync_stats["errors"] += 1
            await asyncio.sleep(60)


# Health API
@app.on_event("startup")
async def startup():
    """Start sync tasks on startup"""
    logger.info("Starting MATL Bridge Service")

    # Start background sync tasks
    asyncio.create_task(sync_trust_scores())
    asyncio.create_task(sync_spam_reports())

    logger.info("MATL Bridge Service started successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global matl_db_pool, holochain_ws

    if matl_db_pool:
        await matl_db_pool.close()
        logger.info("MATL database connection pool closed")

    if holochain_ws and not holochain_ws.closed:
        await holochain_ws.close()
        logger.info("Holochain WebSocket closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        pool = await get_matl_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        uptime = (datetime.utcnow() - sync_stats["uptime_start"]).total_seconds()

        return {
            "status": "healthy",
            "service": "matl-bridge",
            "uptime_seconds": uptime,
            "last_sync": sync_stats["last_sync"].isoformat() if sync_stats["last_sync"] else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/stats")
async def get_stats():
    """Get sync statistics"""
    uptime = (datetime.utcnow() - sync_stats["uptime_start"]).total_seconds()

    return {
        "uptime_seconds": uptime,
        "total_syncs": sync_stats["total_syncs"],
        "trust_scores_synced": sync_stats["trust_scores_synced"],
        "spam_reports_synced": sync_stats["spam_reports_synced"],
        "errors": sync_stats["errors"],
        "last_sync": sync_stats["last_sync"].isoformat() if sync_stats["last_sync"] else None,
        "sync_interval_seconds": int(os.getenv("SYNC_INTERVAL", "300"))
    }


# CLI entry point
def main():
    """Run the MATL bridge service"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8400"))

    logger.info(f"Starting MATL Bridge Service on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()
