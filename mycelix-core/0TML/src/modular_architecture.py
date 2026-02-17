"""Compatibility shim for legacy imports.

Phase 10 relocated the implementation to `zerotrustml.modular_architecture`.
Older modules/tests still import `modular_architecture` at the top level, so this
module re-exports the new package to preserve backwards compatibility.
"""

from zerotrustml.modular_architecture import *  # noqa: F401,F403
