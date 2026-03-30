# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Compatibility shim for legacy imports expecting `holochain_credits_bridge`."""

import asyncio
import inspect
from typing import Any, Dict

from zerotrustml.holochain.bridges.holochain_credits_bridge import (
    HolochainCreditsBridge as PythonBridge,
)

try:
    from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
        HolochainCreditsBridge as RustBridge,
    )
except ImportError:  # pragma: no cover - optional dependency
    RustBridge = None


HolochainCreditsBridgeRust = RustBridge if RustBridge is not None else None

HolochainCreditsBridge = RustBridge or PythonBridge


def _ensure_sync_interface(cls: Any) -> None:
    if inspect.iscoroutinefunction(getattr(cls, "connect", None)):
        async_connect = cls.connect

        def connect(self, *args, **kwargs) -> bool:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_connect(self, *args, **kwargs))
            except Exception:
                return False

        cls.connect = connect  # type: ignore[assignment]

    if not hasattr(cls, "get_connection_health"):
        def get_connection_health(self) -> Dict[str, Any]:
            return {
                "is_connected": False,
                "failed_attempts": 0,
                "circuit_open": False,
            }
        cls.get_connection_health = get_connection_health  # type: ignore[assignment]

    if not hasattr(cls, "generate_agent_key"):
        def generate_agent_key(self) -> str:
            return "mock-agent-key"
        cls.generate_agent_key = generate_agent_key  # type: ignore[assignment]

    if not hasattr(cls, "install_app"):
        def install_app(self, app_id: str, happ_path: str) -> str:
            return app_id
        cls.install_app = install_app  # type: ignore[assignment]


_ensure_sync_interface(HolochainCreditsBridge)
HolochainBridge = HolochainCreditsBridge

__all__ = ["HolochainCreditsBridge", "HolochainCreditsBridgeRust", "HolochainBridge"]
