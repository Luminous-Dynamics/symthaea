{
  description = "Symtropy: Holographic Physics & Consciousness Development Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Standard Bevy dependencies for Linux/NixOS
      bevyDeps = with pkgs; [
        pkg-config
        alsa-lib
        udev
        vulkan-loader
        libxkbcommon
        wayland
        xorg.libX11
        xorg.libXcursor
        xorg.libXi
        xorg.libXrandr
      ];

      # Extra userspace pieces for WGPU compute tests and diagnostics.
      wgpuDeps = with pkgs; [
        libGL
        mesa
        vulkan-tools
        vulkan-validation-layers
      ];

      mycelix-asset-bake = pkgs.rustPlatform.buildRustPackage {
        pname = "mycelix-asset-bake";
        version = "0.1.0";
        src = ./.;

        cargoLock = {
          lockFile = ./Cargo.lock;
        };

        cargoBuildFlags = "-p symtropy-mesh --bin mycelix-asset-bake";

        postUnpack = ''
          # Replace the workspace Cargo.toml with a minimal version containing only the 
          # self-contained crates required to build mycelix-asset-bake, bypassing local 
          # symthaea path dependencies.
          echo '
          [workspace]
          resolver = "2"
          members = [
              "crates/symtropy-mesh",
              "crates/core/symtropy-math",
              "crates/core/symtropy-physics",
          ]
          ' > $sourceRoot/Cargo.toml

          # Delete target-specific config.toml that enforces mold linker, since mold 
          # is not available inside the Nix sandbox.
          rm -f $sourceRoot/.cargo/config.toml
        '';
      };
    in {
      packages.${system} = {
        inherit mycelix-asset-bake;
        default = mycelix-asset-bake;
      };

      devShells.${system}.default = pkgs.mkShell {
        name = "symtropy-dev";

        buildInputs = with pkgs; [
          # Rust/Build tools
          just

          # Assets
          blender
          python3
          python3Packages.pyyaml
          python3Packages.jsonschema
        ] ++ bevyDeps ++ wgpuDeps;

        # Ensure dynamic libraries are found at runtime (standard for Bevy on NixOS)
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (bevyDeps ++ wgpuDeps);
        VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

        shellHook = ''
          echo "💠 Symtropy Development Shell"
          echo "Run 'just check', 'cargo check', or 'just verify-lifesim-wgpu'."
        '';
      };
    };
}
