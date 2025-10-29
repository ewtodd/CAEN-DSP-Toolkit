{
  description = "ROOT Waveform Analysis Framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    toolkit.url = "github:ewtodd/CAEN-DSP-Toolkit";
  };

  outputs = { self, nixpkgs, flake-utils, toolkit }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        dsp-toolkit = toolkit.packages.${system}.default;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            dsp-toolkit
            root
            gnumake
            pkg-config
            clang-tools
          ];

          shellHook = ''
            echo "ROOT Waveform Analysis Framework"
            echo "ROOT version: $(root-config --version)"
            echo "DSP Toolkit: ${dsp-toolkit}"
            echo ""

            # Make pkg-config aware of the toolkit
            export PKG_CONFIG_PATH="${dsp-toolkit}/lib/pkgconfig:$PKG_CONFIG_PATH"

            # Add toolkit to ROOT's search paths
            export ROOT_INCLUDE_PATH="${dsp-toolkit}/include:$ROOT_INCLUDE_PATH"
            export LD_LIBRARY_PATH="${dsp-toolkit}/lib:$LD_LIBRARY_PATH"

            # Verify toolkit is available
            if pkg-config --exists dsp-toolkit; then
              echo "DSP Toolkit pkg-config found"
              echo "Includes: $(pkg-config --cflags dsp-toolkit)"
              echo "Libs: $(pkg-config --libs dsp-toolkit)"
            fi
          '';
        };
      });
}
