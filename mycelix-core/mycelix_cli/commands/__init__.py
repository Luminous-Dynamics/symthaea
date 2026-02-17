"""
Mycelix CLI Commands

This package contains all subcommands for the Mycelix CLI:
- demo: Run interactive FL demonstrations
- init: Initialize new Mycelix projects
- benchmark: Run performance benchmarks
- validate: Validate system and run tests
- status: Show system status
"""

from . import demo, init, benchmark, validate, status

__all__ = ["demo", "init", "benchmark", "validate", "status"]
