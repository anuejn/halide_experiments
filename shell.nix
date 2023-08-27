{ pkgs ? import <nixpkgs> { } }:
let
  halideWithPython = (pkgs.callPackage ./halide.nix { });
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    halideWithPython
    (python3.withPackages (ps: with ps; [
      jupyter
      numpy
      imageio
    ]))
  ];

  nativeBuildInputs = [ ];

  shellHook = ''
    export PYTHONPATH="${halideWithPython}/lib/python3/site-packages:$PYTHONPATH"
  '';
}
