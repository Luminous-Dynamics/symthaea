{ lib, stdenvNoCC, python3, mujoco380 }:

let
  expectedSource = {
    x86_64-linux = {
      assetId = 404891937;
      assetSize = 20812715;
      sha256Hex = "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab";
    };
    aarch64-linux = {
      assetId = 404891905;
      assetSize = 20674425;
      sha256Hex = "adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4";
    };
  }.${stdenvNoCC.hostPlatform.system} or
    (throw "MuJoCo 3.8.0 qualification materialization supports only x86_64-linux and aarch64-linux");
in
assert lib.assertMsg (mujoco380.version == "3.8.0")
  "003D3 requires MuJoCo 3.8.0 source runtime";
assert lib.assertMsg (mujoco380.mujocoRsCompatibility == "4.0.1+mj-3.8.0")
  "003D3 requires mujoco-rs 4.0.1 / MuJoCo 3.8.0 compatibility";
assert lib.assertMsg (mujoco380.upstreamReleaseCommit == "34d69ad4cb1a21846b8297e2bc5e68a4938276c1")
  "003D3 requires the pinned MuJoCo 3.8.0 release commit";
assert lib.assertMsg (mujoco380.upstreamAssetId == expectedSource.assetId)
  "003D3 MuJoCo source asset ID mismatch";
assert lib.assertMsg (mujoco380.upstreamAssetSize == expectedSource.assetSize)
  "003D3 MuJoCo source asset size mismatch";
assert lib.assertMsg (mujoco380.upstreamSha256Hex == expectedSource.sha256Hex)
  "003D3 MuJoCo source SHA-256 mismatch";
stdenvNoCC.mkDerivation {
  pname = "mujoco-qualification-runtime";
  version = "3.8.0";

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;

  nativeBuildInputs = [ python3 ];

  installPhase = ''
    runHook preInstall

    test ! -e ${mujoco380}/runtime-manifest.json

    mkdir -p "$out"
    cp -a ${mujoco380}/. "$out"/

    ${python3}/bin/python ${./generate-mujoco-runtime-manifest.py} \
      --root "$out" \
      --platform ${stdenvNoCC.hostPlatform.system}

    test -f "$out/runtime-manifest.json"
    test -e "$out/lib/libmujoco.so"

    runHook postInstall
  '';

  passthru = {
    sourceRuntime = mujoco380;
    runtimeManifestSchema = "symthaea.qualification-mujoco-runtime.v1";
    mujocoRsCompatibility = "4.0.1+mj-3.8.0";
  };

  meta = {
    description = "MuJoCo 3.8.0 qualification runtime with canonical evidence manifest";
    homepage = "https://github.com/google-deepmind/mujoco";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
