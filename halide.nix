{ llvmPackages_15
, lib
, fetchFromGitHub
, cmake
, libpng
, libjpeg
, mesa
, eigen
, openblas
, blas
, lapack
, git
, buildPythonBindings ? true, python3
}:

assert blas.implementation == "openblas" && lapack.implementation == "openblas";

llvmPackages_15.stdenv.mkDerivation rec {
  pname = "halide";
  version = "16.0.0";

  src = fetchFromGitHub {
    owner = "halide";
    repo = "Halide";
    rev = "v${version}";
    sha256 = "sha256-mnZ6QMqDr48bH2W+andGZj2EhajXKApjuW6B50xtzx0=";
  };

  cmakeFlags = [ "-DWARNINGS_AS_ERRORS=OFF" "-DWITH_PYTHON_BINDINGS=ON" "-DPYBIND11_USE_FETCHCONTENT=OFF" "-DTARGET_WEBASSEMBLY=OFF" ];

  # Note: only openblas and not atlas part of this Nix expression
  # see pkgs/development/libraries/science/math/liblapack/3.5.0.nix
  # to get a hint howto setup atlas instead of openblas
  buildInputs = [
    llvmPackages_15.llvm
    llvmPackages_15.lld
    llvmPackages_15.openmp
    llvmPackages_15.libclang
    libpng
    libjpeg
    eigen
    openblas
  ] ++ lib.optionals buildPythonBindings [ python3 python3.pkgs.pybind11 ];

  nativeBuildInputs = [ cmake ];

  meta = with lib; {
    description = "C++ based language for image processing and computational photography";
    homepage = "https://halide-lang.org";
    license = licenses.mit;
    platforms = platforms.all;
    maintainers = with maintainers; [ ck3d atila ];
  };
}