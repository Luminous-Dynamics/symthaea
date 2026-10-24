#!/usr/bin/env python3
"""Build a browser-friendly gallery from Spore Boot Ecology PPM previews.

The renderer intentionally writes dependency-free PPM files. This publisher
converts those exact pixels to PNG using only Python's standard library and
creates a self-contained index.html that cycles each lifecycle case through its
rendered frames. No image-generation or approximation is involved.
"""

from __future__ import annotations

import argparse
import binascii
import html
import json
from pathlib import Path
import struct
import zlib


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def read_ppm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path}: expected binary P6 PPM")

    index = 2
    tokens: list[bytes] = []
    length = len(data)
    while len(tokens) < 3:
        while index < length and data[index] in b" \t\r\n":
            index += 1
        if index >= length:
            raise ValueError(f"{path}: truncated PPM header")
        if data[index] == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < length and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])

    width, height, max_value = (int(token) for token in tokens)
    if width <= 0 or height <= 0 or max_value != 255:
        raise ValueError(f"{path}: unsupported PPM dimensions/max value")
    while index < length and data[index] in b" \t\r\n":
        index += 1
    pixels = data[index:]
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"{path}: expected {expected} RGB bytes, found {len(pixels)}")
    return width, height, pixels


def write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    stride = width * 3
    scanlines = b"".join(
        b"\x00" + pixels[row * stride : (row + 1) * stride]
        for row in range(height)
    )
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
    png += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def publish_case(root: Path, case: dict) -> dict:
    name = case["name"]
    frames_dir = root / case["frames"]
    ppm_frames = sorted(frames_dir.glob("frame-*.ppm"))
    if not ppm_frames:
        raise ValueError(f"{name}: no PPM frames found in {frames_dir}")

    png_dir = root / name / "png"
    png_paths: list[str] = []
    for ppm in ppm_frames:
        width, height, pixels = read_ppm(ppm)
        png_path = png_dir / (ppm.stem + ".png")
        write_png(png_path, width, height, pixels)
        png_paths.append(png_path.relative_to(root).as_posix())

    midpoint = png_paths[len(png_paths) // 2]
    return {
        **case,
        "png_frames": png_paths,
        "thumbnail": midpoint,
    }


def build_html(manifest: dict, cases: list[dict]) -> str:
    cards = []
    for case in cases:
        frames_json = html.escape(json.dumps(case["png_frames"]), quote=True)
        cards.append(
            f"""
            <article class="case">
              <div class="visual-wrap">
                <img class="visual" src="{html.escape(case['thumbnail'])}"
                     alt="Exact Spore renderer preview for {html.escape(case['name'])}"
                     data-frames="{frames_json}">
                <div class="progress"><span></span></div>
              </div>
              <div class="meta">
                <h2>{html.escape(case['name'].replace('-', ' '))}</h2>
                <dl>
                  <div><dt>family</dt><dd>{html.escape(str(case['family']))}</dd></div>
                  <div><dt>cue</dt><dd>{html.escape(str(case['cue']))}</dd></div>
                  <div><dt>frames</dt><dd>{len(case['png_frames'])}</dd></div>
                  <div><dt>seed</dt><dd class="seed">{html.escape(str(case['seed'])[:16])}…</dd></div>
                </dl>
              </div>
            </article>
            """
        )

    width = manifest.get("width", "?")
    height = manifest.get("height", "?")
    fps = manifest.get("fps", "?")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spore Boot Ecology — exact renderer matrix</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #07100c; color: #e9f5ed; }}
header {{ max-width: 1500px; margin: 0 auto; padding: 42px 28px 24px; }}
h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 5vw, 4rem); font-weight: 520; letter-spacing: -0.04em; }}
header p {{ margin: 0; max-width: 72ch; color: #a9baaf; line-height: 1.55; }}
.badge {{ display: inline-block; margin-top: 18px; padding: 6px 10px; border: 1px solid #32483a; border-radius: 999px; color: #c4d8ca; font: 12px ui-monospace, monospace; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 20px 28px 60px; display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 22px; }}
.case {{ overflow: hidden; border: 1px solid #21352a; border-radius: 18px; background: #0b1710; box-shadow: 0 18px 55px rgba(0,0,0,.22); }}
.visual-wrap {{ position: relative; aspect-ratio: {width} / {height}; background: #020604; overflow: hidden; }}
.visual {{ width: 100%; height: 100%; display: block; object-fit: cover; image-rendering: auto; }}
.progress {{ position: absolute; left: 12px; right: 12px; bottom: 10px; height: 2px; background: rgba(255,255,255,.12); overflow: hidden; border-radius: 10px; }}
.progress span {{ display: block; height: 100%; width: 0; background: linear-gradient(90deg,#9fffae,#f2d57b,#f3fff6); }}
.meta {{ padding: 17px 18px 19px; }}
h2 {{ margin: 0 0 14px; font-size: 17px; font-weight: 560; text-transform: capitalize; }}
dl {{ margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; }}
dl div {{ min-width: 0; }}
dt {{ color: #758c7c; font: 10px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; }}
dd {{ margin: 3px 0 0; color: #d8e7dc; font: 12px ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; }}
.seed {{ color: #9fb4a6; }}
footer {{ max-width: 1500px; margin: 0 auto; padding: 0 28px 45px; color: #718479; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Spore Boot Ecology</h1>
  <p>This gallery is generated from the exact CPU pixel renderer used by the DRM/KMS boot path. Each card cycles through a factual lifecycle case; it is not concept art or an approximation.</p>
  <span class="badge">{width}×{height} · capture cadence {fps} fps · {len(cases)} lifecycle cases</span>
</header>
<main>{''.join(cards)}</main>
<footer>Organic language belongs to the visual layer. System-state inputs remain factual, bounded, and privacy-preserving.</footer>
<script>
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
for (const card of document.querySelectorAll('.case')) {{
  const img = card.querySelector('img.visual');
  const bar = card.querySelector('.progress span');
  const frames = JSON.parse(img.dataset.frames);
  let index = Math.floor(frames.length / 2);
  if (reduceMotion || frames.length < 2) continue;
  const tick = () => {{
    img.src = frames[index];
    bar.style.width = `${{((index + 1) / frames.length) * 100}}%`;
    index = (index + 1) % frames.length;
  }};
  tick();
  setInterval(tick, 650);
}}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="preview matrix output directory")
    args = parser.parse_args()

    manifest_path = args.root / "matrix-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = [publish_case(args.root, case) for case in manifest["cases"]]
    (args.root / "gallery-manifest.json").write_text(json.dumps({**manifest, "cases": cases}, indent=2) + "\n")
    (args.root / "index.html").write_text(build_html(manifest, cases))
    print(args.root / "index.html")


if __name__ == "__main__":
    main()
