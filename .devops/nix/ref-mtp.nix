# The CPU reference MTP tool (tests/models/qwen38_flash_next/ref_mtp.cpp): the production package's toolchain, only this target built and installed.
{ lib, gufo }:

let
  sourceRoot = ../..;
  extra = [
    "tests"
    "tests/models"
    "tests/models/qwen38_flash_next"
    "tests/models/qwen38_flash_next/ref_mtp.cpp"
  ];
  production = gufo.src;
in
gufo.overrideAttrs (old: {
  pname = "gufo-ref-mtp";
  src = lib.cleanSourceWith {
    src = sourceRoot;
    filter =
      path: type:
      let
        root = toString sourceRoot;
        relativePath = lib.removePrefix "${root}/" (toString path);
      in
      builtins.elem relativePath extra || production.filter path type;
    name = "gufo-ref-mtp-source";
  };
  cmakeFlags = old.cmakeFlags ++ [ "-DGUFO_BUILD_REF_MTP=ON" ];
  ninjaFlags = [ "gufo_ref_mtp" ];
  doInstallCheck = false;
  installPhase = ''
    runHook preInstall
    install -Dm755 gufo_ref_mtp $out/bin/gufo_ref_mtp
    runHook postInstall
  '';
  postFixup = ''
    wrapProgram "$out/bin/gufo_ref_mtp" \
      --prefix LD_LIBRARY_PATH : "${old.passthru.rocmRuntime}/lib"
  '';
})
