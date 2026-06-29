{
    description = "Pytorch ML research env";
    
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    };
    
    outputs = { self, nixpkgs }: let
        systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in {
        devShells = nixpkgs.lib.genAttrs systems (system:
            let
                pkgs = import nixpkgs { inherit system; };
            in {
                default = pkgs.mkShell {
                    packages = [
                        (pkgs.python3.withPackages (py_pkgs: with py_pkgs; [
                            torch
                            polars
                            marimo
                            numpy
                            matplotlib
                            transformers
                            datasets
                            tokenizers
                        ]))
                    ];
                };
            }
        );
    };
}
