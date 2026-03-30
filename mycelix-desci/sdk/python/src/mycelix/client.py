# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix-DeSci API Client
"""

from typing import Optional
import requests
import httpx

from .api.claims import ClaimsAPI, AsyncClaimsAPI
from .api.query import QueryAPI, AsyncQueryAPI
from .api.trust import TrustAPI, AsyncTrustAPI
from .exceptions import MycelixError


class MycelixClient:
    """Synchronous Mycelix-DeSci API client"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Mycelix client.

        Args:
            base_url: Base URL of the Mycelix API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        # Create session
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # Initialize API modules
        self.claims = ClaimsAPI(self)
        self.query = QueryAPI(self)
        self.trust = TrustAPI(self)

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Make HTTP request to API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            json: JSON body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            MycelixError: On API error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            raise MycelixError(f"HTTP {e.response.status_code}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise MycelixError(f"Request failed: {str(e)}")

    def health_check(self) -> dict:
        """Check API health status"""
        return self._request("GET", "/health")

    def close(self):
        """Close the client session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncMycelixClient:
    """Asynchronous Mycelix-DeSci API client"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize async Mycelix client.

        Args:
            base_url: Base URL of the Mycelix API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        # Headers
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Create async client
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
        )

        # Initialize API modules
        self.claims = AsyncClaimsAPI(self)
        self.query = AsyncQueryAPI(self)
        self.trust = AsyncTrustAPI(self)

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Make async HTTP request to API.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            json: JSON body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            MycelixError: On API error
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = await self.client.request(
                method=method,
                url=url,
                json=json,
                params=params,
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            raise MycelixError(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise MycelixError(f"Request failed: {str(e)}")

    async def health_check(self) -> dict:
        """Check API health status"""
        return await self._request("GET", "/health")

    async def close(self):
        """Close the client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
