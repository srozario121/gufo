{
  lib,
  newScope,
  releaseVersion ? null,
  revision,
  rocmPackages,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit releaseVersion revision rocmPackages; };
  mkServe = self.callPackage ./mk-serve.nix { };
  gufo-dump-hidden = self.callPackage ./dump-hidden.nix { };
  gufo-ref-mtp = self.callPackage ./ref-mtp.nix { };
})
