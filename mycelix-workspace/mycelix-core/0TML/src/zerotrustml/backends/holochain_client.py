# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root"""
Holochain Admin WebSocket Client for Zero-TrustML

Working implementation based on successful connectivity testing.
Uses sync websocket-client library which is compatible with Holochain 0.6.0-rc.1.

Proven Working:
- WebSocket connection
- WireMessageRequest protocol
- generate_agent_pub_key command

Known Issues:
- list_apps may have deserialization bug in Holochain 0.6.0-rc.1
- Async websockets library gets HTTP 400 from conductor
"""

import logging
import os

import msgpack
import websocket
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HolochainResponse:
    """Parsed response from Holochain conductor."""
    request_id: int
    success: bool
    data: Any
    error: Optional[Dict] = None


class HolochainAdminClient:
    """
    WebSocket client for Holochain Admin API.

    Uses sync websocket-client library which successfully connects
    to Holochain 0.6.0-rc.1 conductor.
    """

    def __init__(self, url: str = "wss://127.0.0.1:8888"):
        """
        Initialize client.

        Args:
            url: WebSocket URL for conductor admin interface.
                 Default: wss://127.0.0.1:8888 (requires TLS certs on conductor).
                 Override with HOLOCHAIN_ADMIN_URL env var for local dev
                 (e.g. ws://127.0.0.1:8888).
        """
        self.url = os.environ.get("HOLOCHAIN_ADMIN_URL", url)
        if self.url.startswith("ws://"):
            logger.warning(
                "Holochain admin connection using insecure ws:// — "
                "use wss:// with TLS certs in production"
            )
        self.ws: Optional[websocket.WebSocket] = None
        self.request_id = 0

    def connect(self):
        """Establish WebSocket connection to conductor."""
        self.ws = websocket.WebSocket()
        # Origin header is required
        self.ws.connect(self.url, origin="http://localhost")

    def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.ws = None

    def _create_request(self, request_type: str, data: Any) -> bytes:
        """
        Create properly formatted WireMessageRequest.

        Holochain requires double MessagePack serialization:
        1. Inner AdminRequest: {'type': <tag>, 'data': <data>}
        2. Wire envelope: {'type': 'request', 'id': <id>, 'data': [bytes as list]}

        Args:
            request_type: Admin command type (e.g., 'generate_agent_pub_key')
            data: Request data (can be None)

        Returns:
            MessagePack-encoded WireMessageRequest bytes
        """
        # Step 1: Create AdminRequest
        admin_request = {
            'type': request_type,
            'data': data
        }

        # Step 2: Serialize AdminRequest
        inner_bytes = msgpack.packb(admin_request)

        # Step 3: Wrap in WireMessageRequest
        # CRITICAL: 'data' must be List[int], not bytes!
        wire_message = {
            'type': 'request',
            'id': self.request_id,
            'data': [b for b in inner_bytes]  # Convert to list of ints
        }

        # Step 4: Serialize WireMessageRequest
        final_bytes = msgpack.packb(wire_message)

        # Increment request ID for next request
        self.request_id += 1

        return final_bytes

    def _send_request(self, request_type: str, data: Any = None) -> HolochainResponse:
        """
        Send request and receive response.

        Args:
            request_type: Admin command type
            data: Request data

        Returns:
            Parsed HolochainResponse

        Raises:
            ConnectionError: If not connected
            RuntimeError: If response indicates error
        """
        if not self.ws:
            raise ConnectionError("Not connected. Call connect() first.")

        # Create and send request
        request = self._create_request(request_type, data)
        self.ws.send(request, opcode=websocket.ABNF.OPCODE_BINARY)

        # Receive response
        response_bytes = self.ws.recv()

        # Parse response
        response = msgpack.unpackb(response_bytes, raw=False)

        # Extract data field
        data_field = response.get('data')

        # Check if data is bytes (double-encoded)
        if isinstance(data_field, bytes):
            inner_data = msgpack.unpackb(data_field, raw=False)
        elif isinstance(data_field, list):
            # Convert list of ints back to bytes
            inner_bytes = bytes(data_field)
            inner_data = msgpack.unpackb(inner_bytes, raw=False)
        else:
            inner_data = data_field

        # Check for error response
        if isinstance(inner_data, dict) and inner_data.get('type') == 'error':
            return HolochainResponse(
                request_id=response.get('id', -1),
                success=False,
                data=None,
                error=inner_data.get('value')
            )

        # Success response
        return HolochainResponse(
            request_id=response.get('id', -1),
            success=True,
            data=inner_data,
            error=None
        )

    # ===== Admin API Methods =====

    def generate_agent_pub_key(self) -> bytes:
        """
        Generate a new agent public key.

        This is a proven working command on Holochain 0.6.0-rc.1.

        Returns:
            Agent public key bytes (39 bytes)

        Raises:
            RuntimeError: If request fails
        """
        response = self._send_request('generate_agent_pub_key', None)

        if not response.success:
            raise RuntimeError(f"Failed to generate agent pub key: {response.error}")

        # Extract the public key value
        if isinstance(response.data, dict):
            return response.data.get('value')

        return response.data

    def list_apps(self, status_filter: Optional[str] = None) -> list:
        """
        List installed apps.

        NOTE: This command has deserialization issues with Holochain 0.6.0-rc.1.
        May work with different versions or after installing apps.

        Args:
            status_filter: Optional filter ('Enabled', 'Disabled', etc.)

        Returns:
            List of installed apps

        Raises:
            RuntimeError: If request fails
        """
        data = None
        if status_filter:
            data = {'status_filter': status_filter}

        response = self._send_request('list_apps', data)

        if not response.success:
            raise RuntimeError(f"Failed to list apps: {response.error}")

        return response.data if isinstance(response.data, list) else []

    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# Convenience function for quick testing
def test_connection(url: str = "wss://127.0.0.1:8888"):
    """
    Test Holochain conductor connection.

    Args:
        url: WebSocket URL (wss:// recommended; requires TLS certs on conductor)

    Returns:
        True if connection successful and can generate agent key
    """
    try:
        with HolochainAdminClient(url) as client:
            # Test with working command
            pub_key = client.generate_agent_pub_key()
            print(f"✅ Connection successful! Generated agent key ({len(pub_key)} bytes)")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        return False


if __name__ == '__main__':
    # Run quick test
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "wss://127.0.0.1:8888"
    success = test_connection(url)
    sys.exit(0 if success else 1)
