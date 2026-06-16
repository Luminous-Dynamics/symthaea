from symthaea_research.nix import (
    detect_causal_relationships,
    detect_conflicts,
    generate_recommendations,
    parse_nix_file,
)


def test_parse_nix_file_detects_options_and_imports(tmp_path):
    config = tmp_path / "configuration.nix"
    config.write_text(
        """
        {
          imports = [ ./hardware.nix ];
          services.xserver.enable = true;
          hardware.pulseaudio.enable = false;
        }
        """
    )

    graph = parse_nix_file(str(config))

    assert graph.imports == ["hardware.nix"]
    assert graph.options["services.xserver.enable"].enabled is True
    assert graph.options["hardware.pulseaudio.enable"].enabled is False


def test_conflict_and_recommendation_detection(tmp_path):
    config = tmp_path / "configuration.nix"
    config.write_text(
        """
        {
          hardware.pulseaudio.enable = true;
          services.pipewire.enable = true;
          hardware.nvidia.modesetting.enable = true;
        }
        """
    )

    graph = parse_nix_file(str(config))
    detect_causal_relationships(graph)

    conflicts = detect_conflicts(graph)
    recommendations = generate_recommendations(graph)

    assert conflicts
    assert any("hardware.opengl.enable" in rec for rec in recommendations)
