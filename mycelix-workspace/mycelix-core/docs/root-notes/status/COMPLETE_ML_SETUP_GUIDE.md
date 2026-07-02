# 🎯 Complete ML Setup Guide for Holochain FL

## The Problem
Installing ML packages (PyTorch, TensorFlow, etc.) on NixOS is challenging because:
1. Binary wheels expect system libraries in standard locations (/usr/lib)
2. NixOS puts everything in /nix/store with unique paths
3. GUI dependencies (tkinter) often break the build
4. Different Python versions have different compatibility

## Solutions (In Order of Preference)

### Solution 1: Use Nix Shell with Minimal Dependencies ✅
```nix
# flake.nix - Working minimal version
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # System dependencies
          gcc
          zlib
          stdenv.cc.cc.lib
          
          # Python with core packages only
          (python311.withPackages (ps: with ps; [
            numpy
            scipy
            scikit-learn
            pandas
            # PyTorch from nixpkgs (pre-built)
            torch
            torchvision
          ]))
        ];
        
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
          ]}:$LD_LIBRARY_PATH"
        '';
      };
    };
}
```

### Solution 2: System-Wide Installation 🌐
Add to `/etc/nixos/configuration.nix`:
```nix
environment.systemPackages = with pkgs; [
  (python311.withPackages (ps: with ps; [
    torch
    torchvision
    numpy
    pandas
    scikit-learn
  ]))
];
```
Then: `sudo nixos-rebuild switch`

### Solution 3: Use Docker 🐳
```dockerfile
FROM python:3.11-slim
RUN pip install torch torchvision numpy pandas scikit-learn
```

### Solution 4: Use Poetry2nix (Advanced) 📦
Create `pyproject.toml` with dependencies, then use poetry2nix flake.

## Quick Test Commands

### Test if PyTorch works:
```bash
nix-shell -p python311Packages.torch --run "python -c 'import torch; print(torch.__version__)'"
```

### Test in current project:
```bash
cd /srv/luminous-dynamics/Mycelix-Core
nix develop --command python -c "import torch; print('PyTorch works!')"
```

## Working Flake for ML (Tested)

Save as `flake-ml-working.nix`:
```nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      
      pythonEnv = pkgs.python311.withPackages (ps: [
        ps.torch
        ps.numpy
        ps.pandas
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pythonEnv pkgs.gcc ];
      };
    };
}
```

## Emergency Fallback: Use System Python

If all else fails:
```bash
# Install pip packages globally (not recommended but works)
pip install --user torch torchvision numpy

# Or use conda/mamba
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

## For This Project Specifically

The best approach for Holochain FL is to:

1. **Keep ML separate from Holochain** - Run ML clients in Docker/separate process
2. **Use JSON for communication** - Serialize models to JSON, not binary
3. **Minimize dependencies** - Use only NumPy + basic Python for the bridge

Example architecture:
```
[ML Client (Docker)] <--JSON--> [Bridge] <--WebSocket--> [Holochain]
```

This avoids dependency hell while keeping the system functional.