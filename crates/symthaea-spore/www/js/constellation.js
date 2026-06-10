// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// ==================================================================
// constellation.js — Package installation visualized as a growing sky
//
// Each Nix store path that downloads becomes a star in a constellation.
// Packages cluster by category. Download speed shows as wind.
// When the install completes, the constellation freezes as the system's
// unique birth chart.
// ==================================================================

(function() {
  'use strict';

  var canvas = null;
  var ctx = null;
  var stars = [];
  var wind = 0;
  var running = false;
  var animFrame = null;
  var packageCount = 0;

  // Category colors
  var CATEGORIES = {
    system:   { color: '#7aa2f7', label: 'System' },
    lib:      { color: '#73daca', label: 'Libraries' },
    dev:      { color: '#9ece6a', label: 'Development' },
    editor:   { color: '#e0af68', label: 'Editors' },
    desktop:  { color: '#bb9af7', label: 'Desktop' },
    font:     { color: '#c0caf5', label: 'Fonts' },
    media:    { color: '#f7768e', label: 'Media' },
    network:  { color: '#2ac3de', label: 'Network' },
    unknown:  { color: '#565f89', label: 'Other' }
  };

  function categorize(packageName) {
    var name = packageName.toLowerCase();
    if (name.match(/^(glibc|gcc|binutils|coreutils|bash|systemd|linux|kernel|grub|boot)/)) return 'system';
    if (name.match(/^(lib|zlib|openssl|pcre|icu|gmp|ncurses|readline)/)) return 'lib';
    if (name.match(/^(git|cargo|rustc|python|node|npm|go|clang|cmake|make|pkg-config)/)) return 'dev';
    if (name.match(/^(vim|neovim|emacs|vscode|nano|kakoune)/)) return 'editor';
    if (name.match(/^(gnome|kde|plasma|sway|hypr|gtk|qt|wayland|xorg|mesa|nvidia|amdgpu)/)) return 'desktop';
    if (name.match(/^(font|noto|liberation|dejavu|jetbrains|fira|iosevka)/)) return 'font';
    if (name.match(/^(ffmpeg|gstreamer|pulse|pipewire|alsa|vlc|mpv)/)) return 'media';
    if (name.match(/^(curl|wget|openssh|network|wireless|iw|wpa)/)) return 'network';
    return 'unknown';
  }

  function createStar(packageName) {
    var cat = categorize(packageName);
    var catInfo = CATEGORIES[cat];
    var angle = Math.random() * Math.PI * 2;
    // Cluster by category — each category occupies a sector
    var catKeys = Object.keys(CATEGORIES);
    var catIndex = catKeys.indexOf(cat);
    var sectorAngle = (catIndex / catKeys.length) * Math.PI * 2;
    angle = sectorAngle + (Math.random() - 0.5) * (Math.PI / 4);

    var distance = 0.2 + Math.random() * 0.35;
    var cx = canvas.width / 2;
    var cy = canvas.height / 2;

    return {
      x: cx + Math.cos(angle) * distance * canvas.width,
      y: cy + Math.sin(angle) * distance * canvas.height,
      radius: 0.5 + Math.random() * 2,
      color: catInfo.color,
      alpha: 0,
      targetAlpha: 0.3 + Math.random() * 0.7,
      twinkle: Math.random() * Math.PI * 2,
      name: packageName,
      category: cat
    };
  }

  function draw() {
    if (!ctx || !running) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections between nearby stars (subtle)
    ctx.lineWidth = 0.3;
    for (var i = 0; i < stars.length; i++) {
      for (var j = i + 1; j < Math.min(stars.length, i + 5); j++) {
        if (stars[i].category === stars[j].category) {
          var dx = stars[i].x - stars[j].x;
          var dy = stars[i].y - stars[j].y;
          var dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 80) {
            ctx.strokeStyle = stars[i].color.replace(')', ', ' + (0.1 * (1 - dist / 80)) + ')').replace('rgb', 'rgba').replace('#', '');
            // Hex to rgba
            var hex = stars[i].color;
            var r = parseInt(hex.slice(1,3), 16);
            var g = parseInt(hex.slice(3,5), 16);
            var b = parseInt(hex.slice(5,7), 16);
            ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (0.15 * (1 - dist / 80)) + ')';
            ctx.beginPath();
            ctx.moveTo(stars[i].x, stars[i].y);
            ctx.lineTo(stars[j].x, stars[j].y);
            ctx.stroke();
          }
        }
      }
    }

    // Draw stars
    var time = Date.now() / 1000;
    for (var k = 0; k < stars.length; k++) {
      var s = stars[k];
      // Fade in
      if (s.alpha < s.targetAlpha) s.alpha += 0.02;
      // Twinkle
      var twinkleAlpha = s.alpha * (0.7 + 0.3 * Math.sin(time * 2 + s.twinkle));
      // Wind drift
      s.x += wind * 0.1;

      var hex = s.color;
      var r = parseInt(hex.slice(1,3), 16);
      var g = parseInt(hex.slice(3,5), 16);
      var b = parseInt(hex.slice(5,7), 16);

      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + twinkleAlpha + ')';
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      ctx.fill();

      // Glow for larger stars
      if (s.radius > 1.5) {
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (twinkleAlpha * 0.2) + ')';
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.radius * 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Package counter
    ctx.fillStyle = 'rgba(192, 202, 245, 0.6)';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(packageCount + ' paths', canvas.width / 2, canvas.height - 15);

    // Category legend (bottom)
    var legendX = 10;
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    Object.keys(CATEGORIES).forEach(function(cat) {
      var info = CATEGORIES[cat];
      var count = stars.filter(function(s) { return s.category === cat; }).length;
      if (count === 0) return;
      ctx.fillStyle = info.color;
      ctx.fillRect(legendX, canvas.height - 8, 6, 6);
      ctx.fillStyle = 'rgba(192, 202, 245, 0.5)';
      ctx.fillText(count, legendX + 9, canvas.height - 3);
      legendX += 35;
    });

    animFrame = requestAnimationFrame(draw);
  }

  window.constellation = {
    init: function(canvasEl) {
      canvas = canvasEl;
      ctx = canvas.getContext('2d');
      canvas.width = canvasEl.parentElement.clientWidth || 400;
      canvas.height = 250;
      stars = [];
      packageCount = 0;
      running = true;
      draw();
    },

    addPackage: function(packageName) {
      if (!canvas) return;
      packageCount++;
      // Only visualize a sample (every 3rd package) to avoid overwhelming
      if (packageCount % 3 === 0 || packageCount < 20) {
        stars.push(createStar(packageName));
      }
    },

    setWind: function(speed) {
      // speed in MB/s
      wind = Math.min(speed / 50, 2);
    },

    stop: function() {
      running = false;
      if (animFrame) cancelAnimationFrame(animFrame);
    },

    getSignature: function() {
      // Return the constellation as a unique system signature
      var counts = {};
      Object.keys(CATEGORIES).forEach(function(c) { counts[c] = 0; });
      stars.forEach(function(s) { counts[s.category]++; });
      return { total: packageCount, categories: counts, starCount: stars.length };
    }
  };
})();
