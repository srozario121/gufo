# The MTP draft-head training-data dump tool (tests/models/qwen38_flash_next/dump_hidden.cpp), built from the production
# package's inputs and toolchain. It is a separate derivation: the production `gufo` derivation, its source filter and its
# outputs are untouched. Only the tool is built and installed.
{ lib, gufo }:

let
  sourceRoot = ../..;
  extra = [
    "tests"
    "tests/models"
    "tests/models/qwen38_flash_next"
    "tests/models/qwen38_flash_next/dump_hidden.cpp"
    "tests/models/qwen38_flash_next/dump_rows.hpp"
    "tests/models/qwen38_flash_next/dflash_rows.hpp"
  ];
  production = gufo.src;
in
gufo.overrideAttrs (old: {
  pname = "gufo-dump-hidden";
  src = lib.cleanSourceWith {
    src = sourceRoot;
    filter =
      path: type:
      let
        root = toString sourceRoot;
        relativePath = lib.removePrefix "${root}/" (toString path);
      in
      builtins.elem relativePath extra || production.filter path type;
    name = "gufo-dump-hidden-source";
  };
  cmakeFlags = old.cmakeFlags ++ [ "-DGUFO_BUILD_DUMP_HIDDEN=ON" ];
  ninjaFlags = [ "gufo_dump_hidden" ];
  doInstallCheck = false;
  installPhase = ''
    runHook preInstall
    install -Dm755 gufo_dump_hidden $out/bin/gufo_dump_hidden
    runHook postInstall
  '';
  postFixup = ''
    wrapProgram "$out/bin/gufo_dump_hidden" \
      --prefix LD_LIBRARY_PATH : "${old.passthru.rocmRuntime}/lib"
  '';
})
