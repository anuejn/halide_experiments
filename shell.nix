{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      jupyter
      numpy
      imageio
      (callPackage ./nix/dearpygui.nix {})
      (callPackage ./nix/halide-python.nix {})
    ]))
  ];
}
