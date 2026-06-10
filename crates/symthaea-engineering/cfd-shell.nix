# In flake.nix, alongside devShells.${system}.default:
  devShells.${system}.cfd = pkgs.mkShell {
    buildInputs = with pkgs; [
      openfoam         # The actual CFD solver for when dry_run = false
      paraview         # The 3D viewer for OpenFOAM thermodynamic output
      freecad          # To inspect the .step/.stl files GeodesicSynthesizer makes
      meshlab          # To audit the topological integrity of the generated meshes
    ];
    shellHook = ''
      echo "Luminous Thermodynamic & CFD Auditing Shell Active."
      echo "Sourcing OpenFOAM environment..."
      # OpenFOAM requires its environment variables to be sourced
      source ${pkgs.openfoam}/etc/bashrc
    '';
  };
