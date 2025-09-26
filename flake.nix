{
  description = "ROOT Waveform Analysis Framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        waveformAnalysis = pkgs.stdenv.mkDerivation {
          pname = "waveform-analysis";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = with pkgs; [ gnumake pkg-config ];

          buildInputs = with pkgs; [ root gsl ];

          # Use make instead of cmake
          buildPhase = ''
            make
          '';

          installPhase = ''
                        mkdir -p $out/{lib,include,bin,share/macros}
                        
                        # Install libraries
                        cp build/lib*.so $out/lib/ 2>/dev/null || true
                        cp build/lib*.a $out/lib/ 2>/dev/null || true
                        
                        # Install headers
                        cp include/*.h $out/include/
                        
                        # Install macros
                        cp macros/*.C $out/share/macros/
                        
                        # Create wrapper script for analysis
                        cat > $out/bin/run-yap-analysis << 'EOF'
            #!/bin/bash
            export LD_LIBRARY_PATH="$out/lib:$LD_LIBRARY_PATH"
            cd "$out/share/macros"
            root -l -b -q 'RunYAPAnalysis.C++'
            EOF
                        chmod +x $out/bin/run-yap-analysis
          '';
        };

      in {
        packages.default = waveformAnalysis;
        packages.waveform-analysis = waveformAnalysis;

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            root
            gsl
            gnumake
            pkg-config
            clang-tools
            gdb
            (python3.withPackages (python-pkgs:
              with python-pkgs; [
                matplotlib
                numpy
                pandas
                tables
                seaborn
                scikit-learn
                mplhep
                awkward
                pynvim
                ipykernel
                cairosvg
                plotly
                kaleido
                pyarrow
                uproot
                h5py
              ]))
          ];

          shellHook = ''
            echo "ROOT Waveform Analysis Development Environment"
            echo "ROOT version: $(root-config --version)"
            echo ""
            echo "Available commands:"
            echo "  make                 - Build the project"
            echo "  make initial     - Initial waveform processing"
            echo "  make calibrate   - Calibrate and initial plots"
            echo "  make background  - Background subtraction"
            echo "  make plots       - Plot calibrated histograms and average waveforms"
            echo "  make PSD         - CC and SI PSD histograms and plotting"
            echo "  make optimize    - CC gate optimization"

            # Set up environment
            export ROOT_INCLUDE_PATH="$PWD/include:$ROOT_INCLUDE_PATH"
            export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
          '';
        };

        # Apps for easy execution
        apps.default = {
          type = "app";
          program = "${waveformAnalysis}/bin/run-yap-analysis";
        };
      });
}
