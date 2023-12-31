{ lib
, buildPythonPackage
, fetchFromGitHub
, stdenv
, python
, cmake
, glfw3
, libX11
, freetype
, darwin
}:

let
  # DearPyGui is not synced with last stable Imgui's version.
  imgui = fetchFromGitHub {
    owner = "ocornut";
    repo = "imgui";
    rev = "1b435ae3e07ca813eb3ef40aaabe7053f5570fae";
    sha256 = "sha256-4UqRKgwaERUz2GQdEZbGmWOj8tAaP+thSlaOBRt06Ho=";
  };
in
buildPythonPackage rec {
  pname = "DearPyGui";
  version = "v1.9.1";

  src = fetchFromGitHub {
    owner = "hoffstadt";
    repo = pname;
    rev = version;
    sha256 = "asJAHXg3bsP/6uimSIhfVeDzCEXA9F702YpYUYzsyMw=";
  };

  hardeningDisable = [ "format" ];

  nativeBuildInputs = [ cmake ];

  buildInputs = [ glfw3 libX11 freetype ] 
    ++ lib.optionals stdenv.isDarwin (with darwin.apple_sdk.frameworks; [ Cocoa MetalKit Quartz ]);
  prePatch = ''
    sed -i "/add_subdirectory(\"Dependencies\")/d" CMakeLists.txt
    cp -r ${imgui}/* Dependencies/imgui
  '';

  # CMake needs to be run by setuptools rather than by its hook
  dontUseCmakeConfigure = true;

  meta = with lib; {
    description = "Cross-platform graphical user interface toolkit(GUI) for Python";
    homepage = "https://github.com/hoffstadt/DearPyGui";
    license = licenses.mit;
  };
}