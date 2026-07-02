import os

_OVERRIDE_ASSET_ROOT = None
_OVERRIDE_REGISTRY_PATH = None

def set_asset_root(path):
    global _OVERRIDE_ASSET_ROOT
    _OVERRIDE_ASSET_ROOT = path

def set_registry_path(path):
    global _OVERRIDE_REGISTRY_PATH
    _OVERRIDE_REGISTRY_PATH = path

def find_project_root(start_path=None):
    if start_path is None:
        start_path = os.getcwd()

    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, 'symtropy-assets')) or os.path.exists(os.path.join(current, 'assets', 'symtropy-foundry-data')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Could not find Symtropy project root.")
        current = parent

def get_asset_root(project_root=None):
    if _OVERRIDE_ASSET_ROOT:
        return _OVERRIDE_ASSET_ROOT
    if project_root is None:
        # Walk up from cwd to find the real project root, so commands work
        # the same regardless of which subdirectory they're run from
        # (previously resolved relative to cwd directly, silently pointing
        # at the wrong/nonexistent registry when run from e.g.
        # tools/symtropy_assets/ instead of the symtropy root).
        try:
            project_root = find_project_root()
        except FileNotFoundError:
            project_root = os.getcwd()
    root = project_root
    legacy_root = os.path.join(root, 'symtropy-assets')
    if os.path.exists(legacy_root):
        return legacy_root
    return os.getenv('SYMTROPY_ASSET_ROOT', os.path.join(root, 'assets', 'symtropy-foundry-data'))

def get_registry_path(project_root=None):
    if _OVERRIDE_REGISTRY_PATH:
        return _OVERRIDE_REGISTRY_PATH
    return os.path.join(get_asset_root(project_root), 'registry', 'assets.sqlite')

def get_raw_vault_path(project_root=None):
    return os.path.join(get_asset_root(project_root), 'raw_vault')

def get_export_root(project_root=None):
    return os.path.join(get_asset_root(project_root), 'bevy_export')
