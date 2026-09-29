#!/usr/bin/env python3
from pathlib import Path
import importlib.util

path = Path(__file__).with_name("check-privileged-workflow-inventory.py")
spec = importlib.util.spec_from_file_location("privileged_inventory", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.self_test()
print("test_check_privileged_workflow_inventory: PASS")
