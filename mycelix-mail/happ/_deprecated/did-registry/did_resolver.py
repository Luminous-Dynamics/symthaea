#!/usr/bin/env python3
"""
Mycelix Mail DID Registry Service
==================================

REST API for resolving Decentralized Identifiers (DIDs) to Holochain AgentPubKeys.

This service implements Layer 5 (Identity) of the Mycelix Protocol architecture,
providing DID resolution for the Mycelix Mail DNA.

API Endpoints:
    GET  /resolve/{did}          - Resolve DID to AgentPubKey
    POST /register               - Register new DID
    PUT  /update/{did}           - Update DID mapping
    GET  /health                 - Health check
    GET  /stats                  - Registry statistics

Environment Variables:
    DATABASE_URL   - PostgreSQL connection string
    PORT           - Server port (default: 8300)
    HOST           - Server host (default: 0.0.0.0)
    LOG_LEVEL      - Logging level (default: INFO)
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio

import asyncpg
from asyncpg import exceptions as pg_exceptions
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Mycelix DID Registry",
    description="Layer 5 Identity Service for Mycelix Mail",
    version="1.0.0"
)

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None


# Pydantic models
class DIDResolutionRequest(BaseModel):
    """Request model for DID resolution"""
    did: str

    @validator('did')
    def validate_did_format(cls, v):
        if not v.startswith('did:'):
            raise ValueError('DID must start with "did:"')
        parts = v.split(':')
        if len(parts) < 3:
            raise ValueError('DID must have format did:method:identifier')
        return v


class DIDRegistrationRequest(BaseModel):
    """Request model for DID registration"""
    did: str
    agent_pubkey: str
    display_name: Optional[str] = None
    email_alias: Optional[str] = None

    @validator('did')
    def validate_did_format(cls, v):
        if not v.startswith('did:'):
            raise ValueError('DID must start with "did:"')
        parts = v.split(':')
        if len(parts) < 3:
            raise ValueError('DID must have format did:method:identifier')
        return v

    @validator('agent_pubkey')
    def validate_pubkey_format(cls, v):
        if not v.startswith('uhCAk'):
            raise ValueError('AgentPubKey must start with "uhCAk"')
        return v


class DIDUpdateRequest(BaseModel):
    """Request model for DID update"""
    new_agent_pubkey: str
    reason: Optional[str] = "Manual update"

    @validator('new_agent_pubkey')
    def validate_pubkey_format(cls, v):
        if not v.startswith('uhCAk'):
            raise ValueError('AgentPubKey must start with "uhCAk"')
        return v


class DIDResolutionResponse(BaseModel):
    """Response model for successful DID resolution"""
    did: str
    agent_pubkey: str
    display_name: Optional[str]
    last_seen: Optional[datetime]
    resolved_at: datetime


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    details: Optional[str] = None


# Database functions
async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool"""
    global db_pool
    if db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable not set")

        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool created")

    return db_pool


async def resolve_did_from_db(did: str, request_source: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Resolve DID to AgentPubKey from database.

    Args:
        did: The DID to resolve
        request_source: Optional source identifier for logging

    Returns:
        Dict with DID resolution data, or None if not found
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Resolve DID
        row = await conn.fetchrow("""
            SELECT did, agent_pubkey, display_name, last_seen
            FROM did_registry
            WHERE did = $1 AND is_active = true
        """, did)

        # Log resolution attempt
        await conn.execute("""
            INSERT INTO did_resolution_log (did, resolved_pubkey, success, request_source)
            VALUES ($1, $2, $3, $4)
        """, did, row['agent_pubkey'] if row else None, row is not None, request_source)

        # Update last_seen if found
        if row:
            await conn.execute("""
                UPDATE did_registry
                SET last_seen = NOW()
                WHERE did = $1
            """, did)

            return dict(row)

        return None


async def register_did_in_db(did: str, agent_pubkey: str, display_name: Optional[str] = None, email_alias: Optional[str] = None) -> bool:
    """
    Register a new DID in the database.

    Args:
        did: The DID to register
        agent_pubkey: The Holochain AgentPubKey
        display_name: Optional display name
        email_alias: Optional email alias

    Returns:
        True if successful, False if DID already exists
    """
    pool = await get_db_pool()

    # Extract method and identifier
    parts = did.split(':')
    method = parts[1]
    identifier = ':'.join(parts[2:])

    async with pool.acquire() as conn:
        try:
            await conn.execute("""
                INSERT INTO did_registry (did, agent_pubkey, did_method, did_method_specific_id, display_name, email_alias)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, did, agent_pubkey, method, identifier, display_name, email_alias)

            logger.info(f"Registered new DID: {did}")
            return True

        except pg_exceptions.UniqueViolationError:
            logger.warning(f"DID already exists: {did}")
            return False


async def update_did_in_db(did: str, new_agent_pubkey: str, reason: str) -> bool:
    """
    Update existing DID mapping.

    Args:
        did: The DID to update
        new_agent_pubkey: The new AgentPubKey
        reason: Reason for update

    Returns:
        True if successful, False if DID not found
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE did_registry
            SET agent_pubkey = $2
            WHERE did = $1 AND is_active = true
        """, did, new_agent_pubkey)

        if result == "UPDATE 0":
            return False

        logger.info(f"Updated DID: {did} (reason: {reason})")
        return True


# API endpoints
@app.on_event("startup")
async def startup():
    """Initialize database connection on startup"""
    try:
        await get_db_pool()
        logger.info("DID Registry Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Close database connections on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "did-registry",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/resolve/{did}")
async def resolve_did(did: str, request: Request):
    """
    Resolve a DID to its AgentPubKey.

    Args:
        did: The DID to resolve (URL-encoded)

    Returns:
        DIDResolutionResponse with the AgentPubKey and metadata

    Raises:
        404: DID not found
        500: Database error
    """
    try:
        # Log request source
        source = request.headers.get("X-Request-Source", request.client.host if request.client else "unknown")

        result = await resolve_did_from_db(did, source)

        if not result:
            logger.warning(f"DID not found: {did}")
            raise HTTPException(
                status_code=404,
                detail=f"DID not found: {did}"
            )

        return DIDResolutionResponse(
            did=result['did'],
            agent_pubkey=result['agent_pubkey'],
            display_name=result.get('display_name'),
            last_seen=result.get('last_seen'),
            resolved_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving DID {did}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register_did(registration: DIDRegistrationRequest):
    """
    Register a new DID.

    Args:
        registration: DID registration request

    Returns:
        Success message

    Raises:
        409: DID already exists
        500: Database error
    """
    try:
        success = await register_did_in_db(
            registration.did,
            registration.agent_pubkey,
            registration.display_name,
            registration.email_alias
        )

        if not success:
            raise HTTPException(
                status_code=409,
                detail=f"DID already exists: {registration.did}"
            )

        return {
            "status": "registered",
            "did": registration.did,
            "agent_pubkey": registration.agent_pubkey
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering DID {registration.did}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update/{did}")
async def update_did(did: str, update: DIDUpdateRequest):
    """
    Update an existing DID mapping.

    Args:
        did: The DID to update
        update: Update request with new AgentPubKey

    Returns:
        Success message

    Raises:
        404: DID not found
        500: Database error
    """
    try:
        success = await update_did_in_db(did, update.new_agent_pubkey, update.reason)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"DID not found: {did}"
            )

        return {
            "status": "updated",
            "did": did,
            "new_agent_pubkey": update.new_agent_pubkey
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating DID {did}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get registry statistics.

    Returns:
        Statistics about DID registrations and resolutions
    """
    try:
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            stats = await conn.fetchrow("SELECT * FROM did_statistics")

            recent_resolutions = await conn.fetch("""
                SELECT success, COUNT(*) as count
                FROM did_resolution_log
                WHERE resolution_time > NOW() - INTERVAL '1 hour'
                GROUP BY success
            """)

            return {
                "total_dids": stats['total_dids'],
                "active_dids": stats['active_dids'],
                "active_last_week": stats['active_last_week'],
                "active_last_day": stats['active_last_day'],
                "resolutions_last_hour": {
                    "successful": next((r['count'] for r in recent_resolutions if r['success']), 0),
                    "failed": next((r['count'] for r in recent_resolutions if not r['success']), 0)
                }
            }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CLI entry point
def main():
    """Run the DID registry service"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8300"))

    logger.info(f"Starting DID Registry Service on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()
