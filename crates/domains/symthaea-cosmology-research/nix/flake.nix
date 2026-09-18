{
  description = "DE-001A independent DESI DR2 BAO cosmology verification environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/9ae611a455b90cf061d8f332b977e387bda8e1ca";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      py = pkgs.python311Packages;

      pyBobyqa = py.buildPythonPackage rec {
        pname = "Py-BOBYQA";
        version = "1.4.1";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-9piEjjcvoGJfuf06eotPVXgE14WLI6RdgYf/rqY0HjM=";
        };
        build-system = [ py.setuptools ];
        dependencies = with py; [ setuptools numpy scipy pandas ];
        doCheck = false;
      };

      getdist = py.buildPythonPackage rec {
        pname = "getdist";
        version = "1.7.4";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-G9acl0iJH6HcUW4kdLMLOvoIPSl1UxwJxDd64w6GNqw=";
        };
        build-system = with py; [ setuptools wheel ];
        dependencies = with py; [ numpy matplotlib scipy pyyaml packaging ];
        doCheck = false;
      };

      camb = py.buildPythonPackage rec {
        pname = "camb";
        version = "1.6.6";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-mFYgKlwFVwJW5SN3sgQxiRx7CLLpwzThQf0I0qCFUW8=";
        };
        build-system = with py; [ setuptools wheel ];
        dependencies = with py; [ numpy scipy sympy packaging ];
        nativeBuildInputs = [ pkgs.gfortran pkgs.gnumake ];
        preBuild = ''
          export FORUTILSPATH="$PWD/forutils"
          test -f "$FORUTILSPATH/Makefile"
        '';
        doCheck = false;
      };

      cobaya = py.buildPythonPackage rec {
        pname = "cobaya";
        version = "3.6.2";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-jxBh1jR0J/CDgOHgwLdm1pXTl4tUOfsLHMGnACFS2cg=";
        };
        build-system = with py; [ setuptools wheel ];
        dependencies = with py; [
          numpy
          scipy
          pandas
          pyyaml
          requests
          rapidfuzz
          packaging
          tqdm
          portalocker
          dill
          typing-extensions
          pyBobyqa
          getdist
        ];
        doCheck = false;
      };

      cosmologyPython = pkgs.python311.withPackages (_: [
        cobaya
        camb
        getdist
        pyBobyqa
        py.iminuit
      ]);

      environmentCheck = pkgs.runCommand "de001a-cosmology-environment-check" {
        nativeBuildInputs = [ cosmologyPython ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR/home"
        mkdir -p "$HOME"
        export PYTHONNOUSERSITE=1
        export PIP_NO_INDEX=1
        export UV_OFFLINE=1

        mkdir -p "$out"
        python - <<'PY' | tee "$out/versions.json"
import importlib.metadata as md
import json
import sys

expected = {
    "cobaya": "3.6.2",
    "camb": "1.6.6",
    "getdist": "1.7.4",
    "iminuit": "2.32.0",
    "Py-BOBYQA": "1.4.1",
}
actual = {name: md.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"version mismatch: expected={expected!r}, actual={actual!r}")

import cobaya  # noqa: F401
import camb  # noqa: F401
import getdist  # noqa: F401
import iminuit  # noqa: F401
import pybobyqa  # noqa: F401

print(json.dumps({
    "python": sys.version.split()[0],
    "packages": actual,
    "network_policy": "nix-build-sandbox",
    "scientific_claim": "NONE",
}, sort_keys=True))
PY
      '';
    in {
      packages.${system} = {
        inherit cobaya camb getdist;
        py-bobyqa = pyBobyqa;
        environment = cosmologyPython;
        default = cosmologyPython;
      };

      checks.${system}.environment = environmentCheck;

      devShells.${system}.cosmology-verify = pkgs.mkShellNoCC {
        packages = [ cosmologyPython ];
        shellHook = ''
          export PYTHONNOUSERSITE=1
          export PIP_NO_INDEX=1
          export PIP_DISABLE_PIP_VERSION_CHECK=1
          export UV_OFFLINE=1
          echo "DE-001A cosmology verification shell"
          echo "  inspection only; authoritative qualification runs through nix build/check"
          python - <<'PY'
import importlib.metadata as md
for name in ["cobaya", "camb", "getdist", "iminuit", "Py-BOBYQA"]:
    print(f"  {name}={md.version(name)}")
PY
        '';
      };

      devShells.${system}.default = self.devShells.${system}.cosmology-verify;
    };
}
