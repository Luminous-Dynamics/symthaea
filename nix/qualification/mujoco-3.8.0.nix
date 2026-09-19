{ lib, stdenvNoCC, fetchurl }:

let
  sources = {
    x86_64-linux = {
      asset = "mujoco-3.8.0-linux-x86_64.tar.gz";
      assetId = 404891937;
      size = 20812715;
      sha256Hex = "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab";
      hash = "sha256-K+iMb5KgbD6v/bR9Om0/vxWfvAV+nSctWS+xlOQf76s=";
    };
    aarch64-linux = {
      asset = "mujoco-3.8.0-linux-aarch64.tar.gz";
      assetId = 404891905;
      size = 20674425;
      sha256Hex = "adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4";
      hash = "sha256-rcSnhW0rjUK6ToibV8vOsToynIafQQzxrRELFTxHReQ=";
    };
  };

  source = sources.${stdenvNoCC.hostPlatform.system} or
    (throw "MuJoCo 3.8.0 qualification materialization supports only x86_64-linux and aarch64-linux");
in
stdenvNoCC.mkDerivation rec {
  pname = "mujoco";
  version = "3.8.0";

  src = fetchurl {
    url = "https://github.com/google-deepmind/mujoco/releases/download/${version}/${source.asset}";
    hash = source.hash;
  };

  sourceRoot = "mujoco-${version}";
  dontConfigure = true;
  dontBuild = true;

  # Preserve the official release payload. Qualification materialization should
  # bind the upstream bytes and the extracted tree separately rather than hide
  # mutations behind the normal Nix fixup phase.
  dontFixup = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp -a . "$out"/
    runHook postInstall
  '';

  passthru = {
    upstreamReleaseTag = version;
    upstreamReleaseCommit = "34d69ad4cb1a21846b8297e2bc5e68a4938276c1";
    upstreamAssetId = source.assetId;
    upstreamAssetName = source.asset;
    upstreamAssetSize = source.size;
    upstreamSha256Hex = source.sha256Hex;
    mujocoRsCompatibility = "4.0.1+mj-3.8.0";
  };

  meta = {
    description = "Pinned MuJoCo 3.8.0 runtime for hermetic qualification materialization";
    homepage = "https://github.com/google-deepmind/mujoco";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
