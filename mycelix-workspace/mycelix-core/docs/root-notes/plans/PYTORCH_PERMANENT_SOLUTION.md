# 🔥 PyTorch Permanent Solution for NixOS

## The Problem You Faced
You're right - we've been installing PyTorch over 50 times! This is frustrating and inefficient. Here's a permanent solution.

## Solution 1: System-Wide Installation (RECOMMENDED)

### Step 1: Add to your system configuration
Edit `/etc/nixos/configuration.nix` and add:

```nix
{ config, pkgs, ... }:

{
  # ... your other config ...

  environment.systemPackages = with pkgs; [
    # ... your other packages ...
    
    # Python with ML packages - ALWAYS AVAILABLE
    (python311.withPackages (ps: with ps; [
      # Core ML
      torch
      torchvision
      numpy
      pandas
      scikit-learn
      
      # Web/networking
      websockets
      requests
      
      # Utilities
      pip
      virtualenv
    ]))
  ];
}
```

### Step 2: Rebuild your system
```bash
sudo nixos-rebuild switch
```

### Step 3: Test it works
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed system-wide!')"
```

Now PyTorch is ALWAYS available, no more installing!

## Solution 2: Create a Global ML Shell

### Step 1: Create `/etc/nixos/ml-packages.nix`
```nix
{ pkgs ? import <nixpkgs> {} }:

let
  pythonML = pkgs.python311.withPackages (ps: with ps; [
    torch
    torchvision
    tensorflow
    numpy
    pandas
    scikit-learn
    matplotlib
    jupyter
    websockets
  ]);
in
pkgs.mkShell {
  packages = [ pythonML ];
  shellHook = ''
    echo "🔥 ML Environment Ready!"
    echo "PyTorch, TensorFlow, and all ML tools available"
  '';
}
```

### Step 2: Create an alias in your shell
Add to `~/.zshrc` or `~/.bashrc`:
```bash
alias ml-shell="nix-shell /etc/nixos/ml-packages.nix"
```

### Step 3: Use it anywhere
```bash
ml-shell
python3 your-ml-script.py
```

## Solution 3: Docker Container (Works Everywhere)

### Create once, use forever:
```dockerfile
# Dockerfile.ml
FROM python:3.11
RUN pip install torch torchvision numpy pandas scikit-learn websockets
WORKDIR /app
CMD ["python3"]
```

Build it:
```bash
docker build -f Dockerfile.ml -t ml-env .
```

Use it:
```bash
docker run -it -v $(pwd):/app ml-env python3 your-script.py
```

## Solution 4: For This Project Specifically

### Create `holochain-ml.nix` in project root:
```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    # Holochain tools
    holochain
    hc
    
    # Python with ML
    (python311.withPackages (ps: with ps; [
      torch
      torchvision
      numpy
      pandas
      websockets
    ]))
  ];
  
  shellHook = ''
    echo "🍄 Holochain + PyTorch ready!"
  '';
}
```

Then always enter with:
```bash
nix-shell holochain-ml.nix
```

## Why This Keeps Happening

1. **Nix is immutable** - packages aren't "installed" traditionally
2. **Each project has its own environment** - isolation is a feature
3. **Flakes can have dependency issues** - tkinter, GUI packages break builds

## The Best Long-Term Solution

**Use Solution 1 (system-wide)** for packages you use frequently like PyTorch. This way:
- Always available
- No repeated downloads
- No dependency conflicts
- Works in any project

## Quick Test: Is PyTorch Available?

Run this to check:
```bash
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} is available!')
except ImportError:
    print('❌ PyTorch not found - follow Solution 1 above')
"
```

## For Federated Learning Specifically

Since FL doesn't actually NEED PyTorch for the core algorithm, we can:
1. Use the simple demo (no ML deps) for testing
2. Run ML clients in Docker when needed
3. Keep Holochain and ML separate

This is actually better architecture - separation of concerns!

---

**Remember**: The frustration you feel is valid. Nix is powerful but can be frustrating when you just want to `pip install`. Using system-wide packages for common tools like PyTorch is the pragmatic solution.