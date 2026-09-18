#!/usr/bin/env python3
"""Publish exact Spore inoculation PPM captures as a browser gallery."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from spore_preview_gallery import read_ppm, write_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    manifest_path = args.root / "inoculation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    published: list[dict] = []

    for phase in manifest["phases"]:
        png_frames: list[str] = []
        for relative in phase["frames"]:
            ppm = args.root / relative
            width, height, pixels = read_ppm(ppm)
            png = ppm.with_suffix(".png")
            write_png(png, width, height, pixels)
            png_frames.append(png.relative_to(args.root).as_posix())
        published.append({**phase, "png_frames": png_frames})

    gallery_manifest = {**manifest, "phases": published}
    (args.root / "gallery-manifest.json").write_text(
        json.dumps(gallery_manifest, indent=2) + "\n"
    )
    (args.root / "index.html").write_text(build_html(gallery_manifest))
    print(args.root / "index.html")


def build_html(manifest: dict) -> str:
    cards: list[str] = []
    for phase in manifest["phases"]:
        frames = html.escape(json.dumps(phase["png_frames"]), quote=True)
        title = html.escape(phase["phase"].replace("-", " "))
        thumbnail = html.escape(phase["png_frames"][len(phase["png_frames"]) // 2])
        cards.append(
            f"""
            <article class="phase">
              <div class="visual-wrap">
                <img class="visual" src="{thumbnail}" alt="Exact Spore inoculation preview: {title}" data-frames="{frames}">
                <span class="orbital"></span>
              </div>
              <h2>{title}</h2>
            </article>
            """
        )

    width = manifest["width"]
    height = manifest["height"]
    samples = manifest["samples_per_phase"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spore Inoculation — exact renderer gallery</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#030907; color:#eef8f1; }}
header {{ max-width:1500px; margin:auto; padding:46px 28px 22px; }}
h1 {{ margin:0; font-size:clamp(2.2rem,5vw,4.5rem); font-weight:480; letter-spacing:-.045em; }}
header p {{ max-width:76ch; color:#96aa9d; line-height:1.55; }}
.badge {{ display:inline-block; border:1px solid #294236; border-radius:999px; padding:6px 10px; color:#bcd3c4; font:12px ui-monospace,monospace; }}
main {{ max-width:1500px; margin:auto; padding:18px 28px 60px; display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:22px; }}
.phase {{ background:linear-gradient(180deg,#08130e,#07100c); border:1px solid #1d3428; border-radius:20px; overflow:hidden; box-shadow:0 24px 70px rgba(0,0,0,.28); }}
.visual-wrap {{ position:relative; aspect-ratio:{width}/{height}; background:#000; overflow:hidden; }}
.visual {{ width:100%; height:100%; display:block; object-fit:cover; }}
.orbital {{ position:absolute; inset:8%; border:1px solid rgba(115,230,205,.08); border-radius:50%; pointer-events:none; }}
h2 {{ margin:0; padding:17px 19px 20px; font-size:16px; font-weight:540; text-transform:capitalize; letter-spacing:.01em; }}
footer {{ max-width:1500px; margin:auto; padding:0 28px 48px; color:#6e8377; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>Spore Inoculation</h1>
  <p>A distinct installation ceremony rendered by the same boot-safe CPU pipeline. The installation is visualized as substrate preparation and system weaving inside a projected incubation field, not as an ordinary boot with renamed labels.</p>
  <span class="badge">{width}×{height} · {len(manifest['phases'])} phases · {samples} exact samples per phase</span>
</header>
<main>{''.join(cards)}</main>
<footer>Exact renderer evidence. No generated concept art is used in this gallery.</footer>
<script>
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
for (const img of document.querySelectorAll('img.visual')) {{
  const frames = JSON.parse(img.dataset.frames);
  if (reduceMotion || frames.length < 2) continue;
  let index = 0;
  const tick = () => {{ img.src = frames[index]; index = (index + 1) % frames.length; }};
  tick();
  setInterval(tick, 720);
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
