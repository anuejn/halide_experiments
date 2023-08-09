{ pkgs ? import <nixpkgs> {} }:
let
  halideWithPython = (pkgs.callPackage ./halide.nix { });
in
pkgs.mkShell {
  buildInputs = with pkgs; [
      halideWithPython
      python3
      python3.pkgs.jupyter
      python3.pkgs.numpy
      python3.pkgs.imageio
  ];

  nativeBuildInputs = [ ];

  shellHook = ''
    export PYTHONPATH="${halideWithPython}/lib/python3/site-packages:$PYTHONPATH"
  '';
}