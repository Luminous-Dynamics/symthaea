{
  description = "DE-001A independent DESI DR2 BAO cosmology verification environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/9ae611a455b90cf061d8f332b977e387bda8e1ca";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      py = pkgs.python311Packages;

      # DE-001A0 scientific inputs are fixed-output derivations. Retrieval
      # location is not authority: Nix accepts each source only when its bytes
      # match the preregistered SHA-256.
      a0Mean = pkgs.fetchurl {
        url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt";
        hash = "sha256-msFUq1g851nA9+7zyXjHxwpurS0Yd0ys6t8aNQpkBYU=";
      };
      a0Covariance = pkgs.fetchurl {
        url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt";
        hash = "sha256-JSoUMnTIoHx4aUwRlhfTZZT215ZdADGcphHG/7iG5Qk=";
      };
      a0LikelihoodDefinition = pkgs.fetchurl {
        url = "https://raw.githubusercontent.com/CobayaSampler/cobaya/899f30a49f85de610dac321e91a1af50018e56aa/cobaya/likelihoods/bao/desi_dr2/desi_bao_all.yaml";
        hash = "sha256-/X6b8tz1/+6QqaMLGCJ/QzfW1cGXh4LGNRPL4NgoDao=";
      };
      a0ReferenceInput = pkgs.fetchurl {
        url = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/iminuit/base/desi-bao-all/bestfit.minimize.input.yaml";
        hash = "sha256-NEmct47K7HjbRNqfBvYc2cnuSX3FxUG3K0jNpUCRxu8=";
      };
      a0ReferenceUpdated = pkgs.fetchurl {
        url = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/iminuit/base/desi-bao-all/bestfit.minimize.updated.yaml";
        hash = "sha256-xMIwMtFpVjWq6m60f6vZCQBv8y3won6tv2SzbInzG6E=";
      };
      a0ReferenceMinimizer = pkgs.fetchurl {
        url = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/iminuit/base/desi-bao-all/minimizer.yaml";
        hash = "sha256-a1EEizWeS51kbeCTeconOo1qG8G1qI9YWgnY8uYfKQw=";
      };
      a0ReferenceBestfitText = pkgs.fetchurl {
        url = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/iminuit/base/desi-bao-all/bestfit.minimum.txt";
        hash = "sha256-v4414jgO81sTendkXcs1GvLtKpPKjaFsH9cctd16E1g=";
      };
      a0ReferenceBestfitGetdist = pkgs.fetchurl {
        url = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/iminuit/base/desi-bao-all/bestfit.minimum";
        hash = "sha256-T2Q3Zh+suSVnMVXvrQ8nR4zgjwv8n1jgvwO938lsnyE=";
      };

      a0Artifacts = pkgs.runCommand "de001a-a0-artifacts" { } ''
        set -euo pipefail
        mkdir -p "$out"
        cp ${a0Mean} "$out/dataset-mean"
        cp ${a0Covariance} "$out/dataset-covariance"
        cp ${a0LikelihoodDefinition} "$out/likelihood-definition"
        cp ${a0ReferenceInput} "$out/reference-input-configuration"
        cp ${a0ReferenceUpdated} "$out/reference-expanded-configuration"
        cp ${a0ReferenceMinimizer} "$out/reference-minimizer-configuration"
        cp ${a0ReferenceBestfitText} "$out/reference-bestfit-text"
        cp ${a0ReferenceBestfitGetdist} "$out/reference-bestfit-getdist"
        chmod 0444 "$out"/*
      '';

      a0ArtifactCheck = pkgs.runCommand "de001a-a0-artifact-check" {
        nativeBuildInputs = [ pkgs.coreutils ];
      } ''
        set -euo pipefail
        test "$(wc -c < ${a0Artifacts}/dataset-mean)" -eq 472
        test "$(wc -c < ${a0Artifacts}/dataset-covariance)" -eq 2547
        test "$(wc -c < ${a0Artifacts}/likelihood-definition)" -eq 368
        test "$(wc -c < ${a0Artifacts}/reference-input-configuration)" -eq 2381
        test "$(wc -c < ${a0Artifacts}/reference-expanded-configuration)" -eq 3969
        test "$(wc -c < ${a0Artifacts}/reference-minimizer-configuration)" -eq 2484
        test "$(wc -c < ${a0Artifacts}/reference-bestfit-text)" -eq 902
        test "$(wc -c < ${a0Artifacts}/reference-bestfit-getdist)" -eq 3940

        cat > expected.sha256 <<'EOF'
9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585  dataset-mean
252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509  dataset-covariance
fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa  likelihood-definition
34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef  reference-input-configuration
c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1  reference-expanded-configuration
6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c  reference-minimizer-configuration
bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358  reference-bestfit-text
4f6437661facb925673155efad0f27478ce08f0bfc9f58e0bf03bddfc96c9f21  reference-bestfit-getdist
EOF
        expected_file="$PWD/expected.sha256"
        (cd ${a0Artifacts} && sha256sum -c "$expected_file")
        mkdir -p "$out"
        cp expected.sha256 "$out/expected.sha256"
        echo "DE-001A0 fixed-output artifacts PASS; scientific_claim=NONE" > "$out/result"
      '';

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
        a0-artifacts = a0Artifacts;
        environment = cosmologyPython;
        default = cosmologyPython;
      };

      checks.${system} = {
        a0-artifacts = a0ArtifactCheck;
        environment = environmentCheck;
      };

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
