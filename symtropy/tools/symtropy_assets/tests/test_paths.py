import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import paths

def test_find_project_root(tmp_path):
    # Setup dummy project structure
    asset_root = tmp_path / "symtropy-assets"
    asset_root.mkdir()
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)

    # Discovery from nested dir
    assert paths.find_project_root(str(nested)) == str(tmp_path)

def test_asset_root_override():
    paths.set_asset_root("/override/assets")
    assert paths.get_asset_root() == "/override/assets"
    paths.set_asset_root(None) # Reset

def test_env_asset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMTROPY_ASSET_ROOT", "/env/assets")
    assert paths.get_asset_root() == "/env/assets"

def test_registry_path_generation(tmp_path):
    # Setup dummy project root
    asset_root = tmp_path / "symtropy-assets"
    asset_root.mkdir()

    reg_path = paths.get_registry_path(str(tmp_path))
    assert reg_path == os.path.join(str(asset_root), "registry", "assets.sqlite")

def test_missing_root_fails():
    # Run in a temp dir with no symtropy-assets
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            paths.find_project_root(tmp)
