import pytest

from bridge import RustAgentBridge


def test_assert_update_schema_detects_missing_fields():
    with pytest.raises(RuntimeError):
        RustAgentBridge._assert_update_schema({"schema_version": 2})


def test_assert_update_schema_allows_valid_payload():
    RustAgentBridge._assert_update_schema({"schema_version": 2, "public_key": "pk"})
