{ lib
, callPackage
, toPythonModule
, python
}:
toPythonModule ((callPackage ./halide.nix { buildPythonBindings = true; }).overrideAttrs ({
  pythonImportsCheck = [ "halide" ];
    postInstall = ''
    mkdir -p ${placeholder "out"}/lib/${python.libPrefix}/
    cp -r ${placeholder "out"}/lib/python3/site-packages ${placeholder "out"}/lib/${python.libPrefix}/site-packages
  '';
}))
