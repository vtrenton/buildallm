{
  description = "Dev shell with Jupyter, PyTorch (torch), and tiktoken";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          # If you later switch to torch-bin / CUDA, you'll likely need this:
          config.allowUnfree = true;
        };

        python = pkgs.python312;

        pythonEnv = python.withPackages (ps: with ps; [
          # Jupyter bits
          jupyterlab
          ipykernel

          # Requested packages
          tiktoken

          # PyTorch:
          #
          # 1) Try the source-built torch first (more "pure", slower builds sometimes)
          torch
          # torchvision
          # torchaudio
          #
          # 2) If you want the prebuilt wheel version (often easier/faster, CUDA-oriented),
          #    comment out `torch` above and uncomment this:
          # torch-bin
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
          ];

          # keeps your shell from accidentally picking up user-site pip installs
          env = {
            PYTHONNOUSERSITE = "1";
          };

          shellHook = ''
            echo
            echo "Python: $(python --version)"
            echo "Try:"
            echo "  python -c \"import torch, tiktoken; print(torch.__version__); print(tiktoken.__version__)\""
            echo "  jupyter lab"
            echo
          '';
        };
      }
    );
}

