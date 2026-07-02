# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Context

Provides structured context for errors including:
- Source location (file, line, function)
- Operation being performed
- Component that raised the error
- Additional metadata
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import traceback
import os


@dataclass
class ErrorContext:
    """
    Structured error context for debugging and logging.

    Captures:
    - Source location (file, line, function)
    - Operation and component names
    - Timestamp
    - Additional metadata

    Usage:
        context = ErrorContext(
            file=__file__,
            line=42,
            operation="connect_holochain",
            component="HolochainBridge",
            metadata={"host": "localhost", "port": 8888}
        )
    """
    # Source location
    file: Optional[str] = None
    line: Optional[int] = None
    function: Optional[str] = None

    # Operation info
    operation: Optional[str] = None
    component: Optional[str] = None

    # Timestamp (UTC)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Correlation ID for distributed tracing
    correlation_id: Optional[str] = None

    # Stack trace (captured on creation if requested)
    stack_trace: Optional[str] = None

    def __post_init__(self):
        """Clean up file path for readability."""
        if self.file:
            # Make path relative to project root if possible
            self.file = self._relative_path(self.file)

    @staticmethod
    def _relative_path(path: str) -> str:
        """Convert absolute path to relative for cleaner logs."""
        try:
            cwd = os.getcwd()
            if path.startswith(cwd):
                return path[len(cwd):].lstrip(os.sep)
        except Exception:
            pass
        return path

    @classmethod
    def capture(
        cls,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        include_stack: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        skip_frames: int = 1
    ) -> "ErrorContext":
        """
        Capture context from current stack frame.

        Args:
            operation: Name of the operation being performed
            component: Component or module name
            include_stack: Include full stack trace
            metadata: Additional context data
            correlation_id: Correlation ID for tracing
            skip_frames: Number of frames to skip (default 1 for this method)

        Returns:
            ErrorContext with captured location

        Usage:
            context = ErrorContext.capture(
                operation="validate_gradient",
                component="ByzantineDetector"
            )
        """
        # Get caller frame
        stack = traceback.extract_stack()
        if len(stack) > skip_frames:
            frame = stack[-(skip_frames + 1)]
            file = frame.filename
            line = frame.lineno
            function = frame.name
        else:
            file = None
            line = None
            function = None

        # Capture stack trace if requested
        stack_trace = None
        if include_stack:
            stack_trace = "".join(traceback.format_stack()[:-skip_frames])

        return cls(
            file=file,
            line=line,
            function=function,
            operation=operation,
            component=component,
            metadata=metadata or {},
            correlation_id=correlation_id,
            stack_trace=stack_trace,
        )

    def with_metadata(self, **kwargs) -> "ErrorContext":
        """Return new context with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return ErrorContext(
            file=self.file,
            line=self.line,
            function=self.function,
            operation=self.operation,
            component=self.component,
            timestamp=self.timestamp,
            metadata=new_metadata,
            correlation_id=self.correlation_id,
            stack_trace=self.stack_trace,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp.isoformat(),
        }

        if self.file:
            result["location"] = {
                "file": self.file,
                "line": self.line,
                "function": self.function,
            }

        if self.operation:
            result["operation"] = self.operation

        if self.component:
            result["component"] = self.component

        if self.correlation_id:
            result["correlation_id"] = self.correlation_id

        if self.metadata:
            result["metadata"] = self.metadata

        if self.stack_trace:
            result["stack_trace"] = self.stack_trace

        return result

    def format_location(self) -> str:
        """Format location as string for logs."""
        parts = []
        if self.file:
            parts.append(self.file)
        if self.line:
            parts.append(f":{self.line}")
        if self.function:
            parts.append(f" in {self.function}()")
        return "".join(parts) if parts else "<unknown location>"

    def __str__(self) -> str:
        """Human-readable representation."""
        location = self.format_location()
        parts = [f"at {location}"]

        if self.operation:
            parts.append(f"during {self.operation}")

        if self.component:
            parts.append(f"in {self.component}")

        return " ".join(parts)
