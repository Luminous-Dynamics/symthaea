// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// tab-inoculate.js — Tab 5: Hardware probe, NixOS config, deployment
// ==================================================================
(function() {
  'use strict';
  var state = window.portalState;

  async function probeHardware() {
    var profile = {};
    profile.cpuCores = navigator.hardwareConcurrency || 'Unknown';

    var rawMem = navigator.deviceMemory;
    if (rawMem) {
      profile.memoryGb = rawMem;
      profile.memoryNote = rawMem <= 0.5 ? 'Browser-reported minimum' :
        rawMem >= 8 ? 'Browser-reported maximum' : 'Browser-reported';
    } else {
      profile.memoryGb = null;
      profile.memoryNote = 'API unavailable in this browser';
    }

    profile.gpu = { available: false, renderer: 'Unknown', vendor: 'Unknown' };
    try {
      var canvas = document.createElement('canvas');
      var gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        var ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) {
          profile.gpu.renderer = gl.getParameter(ext.UNMASKED_RENDERER_STRING_WEBGL);
          profile.gpu.vendor = gl.getParameter(ext.UNMASKED_VENDOR_STRING_WEBGL);
          profile.gpu.available = true;
        }
      }
    } catch (e) {}

    profile.webgpu = { available: false };
    try {
      if (navigator.gpu) {
        var adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          profile.webgpu.available = true;
          try {
            var info = await adapter.requestAdapterInfo();
            profile.webgpu.vendor = info.vendor || '';
            profile.webgpu.architecture = info.architecture || '';
            profile.webgpu.device = info.device || '';
          } catch (e) {}
        }
      }
    } catch (e) {}

    profile.storage = { quota: null, usage: null };
    try {
      if (navigator.storage && navigator.storage.estimate) {
        var est = await navigator.storage.estimate();
        profile.storage.quota = est.quota;
        profile.storage.usage = est.usage;
      }
    } catch (e) {}

    profile.screen = {
      width: screen.width, height: screen.height,
      dpr: window.devicePixelRatio || 1, colorDepth: screen.colorDepth
    };

    profile.platform = 'Unknown';
    try {
      if (navigator.userAgentData) {
        profile.platform = navigator.userAgentData.platform || 'Unknown';
      } else {
        var ua = navigator.userAgent;
        if (ua.includes('Win')) profile.platform = 'Windows';
        else if (ua.includes('Mac')) profile.platform = 'macOS';
        else if (ua.includes('Linux')) profile.platform = 'Linux';
        else if (ua.includes('Android')) profile.platform = 'Android';
        else if (ua.includes('iPhone') || ua.includes('iPad')) profile.platform = 'iOS';
      }
    } catch (e) {}

    profile.battery = null;
    try {
      if (navigator.getBattery) {
        var batt = await navigator.getBattery();
        profile.battery = { charging: batt.charging, level: Math.round(batt.level * 100) };
      }
    } catch (e) {}

    profile.network = null;
    try {
      if (navigator.connection) {
        profile.network = { type: navigator.connection.effectiveType, downlink: navigator.connection.downlink };
      }
    } catch (e) {}

    return profile;
  }

  async function runHardwareProbe() {
    var loading = document.getElementById('inoc-probe-loading');
    var results = document.getElementById('inoc-probe-results');
    var profile = await probeHardware();

    document.getElementById('hw-cpu').textContent = profile.cpuCores + ' cores';

    if (profile.memoryGb) {
      document.getElementById('hw-ram').textContent = profile.memoryGb + ' GB';
      document.getElementById('hw-ram-note').textContent = profile.memoryNote;
    } else {
      document.getElementById('hw-ram').textContent = 'Not reported';
      document.getElementById('hw-ram-note').textContent = profile.memoryNote;
    }

    if (profile.gpu.available) {
      document.getElementById('hw-gpu').textContent = profile.gpu.renderer;
      var gpuNote = 'Vendor: ' + profile.gpu.vendor;
      if (profile.webgpu.available) gpuNote += ' | WebGPU available';
      document.getElementById('hw-gpu-note').textContent = gpuNote;
    } else if (profile.webgpu.available) {
      document.getElementById('hw-gpu').textContent = profile.webgpu.device || 'WebGPU available';
    } else {
      document.getElementById('hw-gpu').textContent = 'Not detected';
      document.getElementById('hw-gpu-note').textContent = 'No WebGL/WebGPU found';
    }

    if (profile.storage.quota) {
      document.getElementById('hw-storage').textContent = formatBytes(profile.storage.quota) + ' available';
    } else {
      document.getElementById('hw-storage').textContent = 'API unavailable';
    }

    document.getElementById('hw-platform').textContent = profile.platform;
    document.getElementById('hw-screen').textContent =
      profile.screen.width + '\u00D7' + profile.screen.height +
      ' @' + profile.screen.dpr + 'x, ' + profile.screen.colorDepth + '-bit';

    if (profile.battery) {
      document.getElementById('hw-battery-item').style.display = '';
      document.getElementById('hw-battery').textContent =
        (profile.battery.charging ? 'Charging' : 'On battery') + ' (' + profile.battery.level + '%)';
    }

    if (profile.network) {
      document.getElementById('hw-network-item').style.display = '';
      document.getElementById('hw-network').textContent =
        profile.network.type + ', ' + profile.network.downlink + ' Mbps';
    }

    // Recommendations
    var gpuRenderer = (profile.gpu.available ? profile.gpu.renderer : '').toLowerCase();
    var gpuDriver = 'Auto-detect on install';
    var neuronCount = 128;
    if (gpuRenderer.includes('nvidia') || gpuRenderer.includes('geforce') || gpuRenderer.includes('rtx')) {
      gpuDriver = 'nvidia (proprietary, CUDA)';
      neuronCount = 256;
    } else if (gpuRenderer.includes('amd') || gpuRenderer.includes('radeon')) {
      gpuDriver = 'amdgpu (open-source)';
      neuronCount = 192;
    } else if (gpuRenderer.includes('intel')) {
      gpuDriver = 'i915 (open-source)';
      neuronCount = 128;
    }

    var recFs = 'ext4';
    if (profile.storage.quota && profile.storage.quota > 100 * 1024 * 1024 * 1024) {
      recFs = 'btrfs (' + formatBytes(profile.storage.quota) + ')';
    }

    document.getElementById('rec-fs').textContent = recFs;
    document.getElementById('rec-gpu').textContent = gpuDriver;
    document.getElementById('rec-substrate').textContent =
      'SiliconDigital (' + profile.cpuCores + ' cores \u2192 ' + neuronCount + ' neurons)';

    // Nix config
    var nixLines = [
      '# Auto-detected hardware configuration',
      '# CPU: ' + profile.cpuCores + ' cores',
      '# GPU: ' + (profile.gpu.available ? profile.gpu.renderer : 'auto-detect'),
      '# Platform: ' + profile.platform,
      '',
      '{ config, lib, pkgs, ... }:',
      '{',
    ];
    if (gpuRenderer.includes('nvidia')) {
      nixLines.push('  hardware.nvidia.modesetting.enable = true;');
      nixLines.push('  services.xserver.videoDrivers = [ "nvidia" ];');
    } else if (gpuRenderer.includes('amd') || gpuRenderer.includes('radeon')) {
      nixLines.push('  services.xserver.videoDrivers = [ "amdgpu" ];');
    }
    nixLines.push('  # Substrate: ' + neuronCount + ' neurons');
    nixLines.push('}');
    document.getElementById('nix-config-content').innerHTML = highlightNix(nixLines.join('\n'));

    // Store profile on state so download buttons can access it
    state.hardwareProfile = profile;

    loading.style.display = 'none';
    results.style.display = 'block';
  }
  window.runHardwareProbe = runHardwareProbe;

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return 'Unknown';
    var gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return gb.toFixed(0) + ' GB';
    return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
  }

  function highlightNix(text) {
    return text.split('\n').map(function(line) {
      if (line.trim().startsWith('#')) {
        return '<span class="nix-comment">' + window.escapeHtml(line) + '</span>';
      }
      return window.escapeHtml(line)
        .replace(/^(\s*)([\w\-\.]+)(\s*=)/g, '$1<span class="nix-key">$2</span>$3')
        .replace(/= (.+);/g, '= <span class="nix-value">$1</span>;')
        .replace(/(\{|\})/g, '<span class="nix-brace">$1</span>');
    }).join('\n');
  }

  // Nix config toggle
  document.getElementById('nix-toggle').addEventListener('click', function() {
    var body = document.getElementById('nix-config-body');
    var isOpen = body.classList.contains('open');
    body.classList.toggle('open');
    this.classList.toggle('open');
    this.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
  });

  // Choice buttons
  document.getElementById('btn-hermit').addEventListener('click', function() {
    state.chosenPath = 'hermit';
    showNextSteps();
  });
  document.getElementById('btn-mycelial').addEventListener('click', function() {
    state.chosenPath = 'mycelial';
    showNextSteps();
  });

  // ----------------------------------------------------------------
  // File download helper
  // ----------------------------------------------------------------
  function downloadFile(filename, content) {
    var blob = new Blob([content], { type: 'text/plain' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ----------------------------------------------------------------
  // Build the hardware JSON that the WASM functions expect
  // (camelCase keys matching HardwareProfile serde)
  // ----------------------------------------------------------------
  function buildHardwareJson(profile) {
    return JSON.stringify({
      cpuCores: typeof profile.cpuCores === 'number' ? profile.cpuCores : (navigator.hardwareConcurrency || 1),
      deviceMemoryGb: profile.memoryGb || (navigator.deviceMemory || 0),
      gpuVendor: (profile.gpu && profile.gpu.vendor) || '',
      gpuArchitecture: (profile.webgpu && profile.webgpu.architecture) || '',
      gpuDescription: (profile.gpu && profile.gpu.renderer) || '',
      hasWebgpu: !!(profile.webgpu && profile.webgpu.available),
      hasWebusb: !!navigator.usb,
      storageQuotaBytes: (profile.storage && profile.storage.quota) || 0,
      platform: profile.platform || 'Unknown',
      isMobile: /Android|iPhone|iPad/.test(navigator.userAgent),
      browserName: (navigator.userAgentData && navigator.userAgentData.brands && navigator.userAgentData.brands[0] && navigator.userAgentData.brands[0].brand) || ''
    });
  }

  // ----------------------------------------------------------------
  // Inject download panel + install guide into next-steps-section
  // ----------------------------------------------------------------
  function injectDownloadPanel(nextSteps, hardwareJson) {
    // Avoid duplicating if already injected
    if (document.getElementById('flake-download-panel')) return;

    var panel = document.createElement('div');
    panel.id = 'flake-download-panel';
    panel.className = 'glass-panel inoc-section';
    panel.style.marginTop = '1.5rem';
    panel.innerHTML = [
      '<h3 style="text-align:center; font-weight:200; margin-bottom:1rem;">Download Configuration Files</h3>',
      '<p style="text-align:center; font-size:0.82rem; color:var(--fg-dim); margin-bottom:1.5rem;">',
      'Your hardware probe has been translated into reproducible NixOS configuration files.<br>',
      'Download all three, review them, then follow the installation guide below.',
      '</p>',
      '<div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-bottom:1.5rem;">',
      '  <button id="btn-dl-flake" class="btn-glow" style="padding:0.6rem 1.2rem; cursor:pointer;">',
      '    Download flake.nix',
      '  </button>',
      '  <button id="btn-dl-disko" class="btn-glow" style="padding:0.6rem 1.2rem; cursor:pointer;">',
      '    Download disko-config.nix',
      '  </button>',
      '  <button id="btn-dl-hardware" class="btn-glow" style="padding:0.6rem 1.2rem; cursor:pointer;">',
      '    Download hardware-configuration.nix',
      '  </button>',
      '</div>',
      '<div id="dl-status" style="text-align:center; font-size:0.78rem; color:var(--teal); min-height:1.2em; margin-bottom:1.5rem;"></div>',
      '',
      '<h3 style="text-align:center; font-weight:200; margin-bottom:1rem;">Manual Installation Guide</h3>',
      '<div style="background:rgba(0,0,0,0.3); border-radius:8px; padding:1rem 1.2rem; font-family:monospace; font-size:0.78rem; line-height:1.6; overflow-x:auto;">',
      '<span style="color:var(--fg-dim);"># 1. Boot from NixOS minimal ISO</span><br>',
      '<span style="color:var(--fg-dim);"># 2. Partition and mount your disk with disko:</span><br>',
      'sudo nix run github:nix-community/disko -- --mode disko ./disko-config.nix<br>',
      '<br>',
      '<span style="color:var(--fg-dim);"># 3. Copy configuration files to /mnt/etc/nixos/:</span><br>',
      'sudo mkdir -p /mnt/etc/nixos<br>',
      'sudo cp flake.nix disko-config.nix hardware-configuration.nix /mnt/etc/nixos/<br>',
      '<br>',
      '<span style="color:var(--fg-dim);"># 4. Install NixOS:</span><br>',
      'sudo nixos-install --flake /mnt/etc/nixos#<span id="install-hostname">guardian</span> --no-root-passwd<br>',
      '<br>',
      '<span style="color:var(--fg-dim);"># 5. Reboot into your sovereign system:</span><br>',
      'sudo reboot<br>',
      '<br>',
      '<span style="color:var(--fg-dim);"># 6. After first boot, set your password:</span><br>',
      'sudo passwd guardian<br>',
      '</div>',
      '<p style="text-align:center; font-size:0.72rem; color:var(--fg-muted); margin-top:1rem; font-style:italic;">',
      'Review every line of the flake before applying. Sovereignty means understanding your own configuration.',
      '</p>'
    ].join('\n');

    nextSteps.parentNode.insertBefore(panel, nextSteps.nextSibling);

    // Inject evaluation panel (calls eval-api to show closure size)
    var evalPanel = document.createElement('div');
    evalPanel.id = 'eval-panel';
    evalPanel.className = 'glass-panel inoc-section';
    evalPanel.style.marginTop = '1rem';
    evalPanel.innerHTML = [
      '<h3 style="text-align:center; font-weight:200; margin-bottom:1rem;">Evaluate Configuration</h3>',
      '<p style="text-align:center; font-size:0.82rem; color:var(--fg-dim); margin-bottom:1rem;">',
      'Optionally evaluate your flake to see how many derivations and how large the closure is before installing.',
      '</p>',
      '<div style="text-align:center; margin-bottom:1rem;">',
      '  <button id="btn-eval" class="btn-glow" style="padding:0.6rem 1.5rem; cursor:pointer;">',
      '    Evaluate Flake',
      '  </button>',
      '</div>',
      '<div id="eval-results" style="display:none; background:rgba(0,0,0,0.3); border-radius:8px; padding:1rem; font-size:0.85rem; line-height:1.6; text-align:center;">',
      '</div>',
      '<div id="eval-status" style="text-align:center; font-size:0.78rem; color:var(--teal); min-height:1.2em; margin-top:0.5rem;"></div>'
    ].join('\n');
    panel.parentNode.insertBefore(evalPanel, panel.nextSibling);

    // Wire up download buttons
    var hostname = 'guardian';

    // Wire eval button (uses hostname declared above)
    document.getElementById('btn-eval').addEventListener('click', async function() {
      var evalStatus = document.getElementById('eval-status');
      var evalResults = document.getElementById('eval-results');
      var btn = this;
      btn.disabled = true;
      btn.textContent = 'Evaluating...';
      evalStatus.textContent = 'Connecting to eval service...';

      var customEval = localStorage.getItem('symthaea-eval-url');
      // Security: default to localhost-only eval API (no remote fallback).
      var evalUrls = customEval ? [customEval] : [
        'http://127.0.0.1:8090/api/v1/eval',
        'http://localhost:8090/api/v1/eval'
      ];
      var hwContent = '';
      try {
        hwContent = await window.send('generateHardwareNix', { hardwareJson: hardwareJson });
      } catch(e) {
        hwContent = '{ config, lib, ... }: { fileSystems."/" = { device = "/dev/disk/by-label/nixos"; fsType = "ext4"; }; }';
      }

      var evalReq = {
        flake_ref: 'path:.',
        hardware_config: hwContent,
        hostname: hostname,
        disko_config: null,
        platform: 'x86_64-linux',
        include_holochain: state.chosenPath === 'mycelial',
        include_broca_weights: false,
        substrate_type: null,
        lanzaboote_enabled: false
      };

      var succeeded = false;
      for (var i = 0; i < evalUrls.length; i++) {
        try {
          evalStatus.textContent = 'Evaluating at ' + evalUrls[i] + '...';
          var resp = await fetch(evalUrls[i], {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(evalReq),
            signal: AbortSignal.timeout(120000)
          });
          var result = await resp.json();
          if (result.success) {
            evalResults.style.display = 'block';
            evalResults.innerHTML = [
              '<div style="color:var(--solar-gold); font-size:1.1rem; margin-bottom:0.5rem;">',
              result.derivation_count + ' derivations. ' + result.closure_size_human + '. Fully reproducible.',
              '</div>',
              '<div style="font-size:0.75rem; color:var(--fg-dim);">',
              'Hash: ' + result.closure_hash.substring(0, 16) + '&hellip; &middot; Evaluated in ' + (result.eval_time_ms / 1000).toFixed(1) + 's',
              '</div>'
            ].join('');
            evalStatus.textContent = '';
            window.addNarration(result.derivation_count + ' derivations. ' + result.closure_size_human + '. Every byte deterministic. Every dependency accounted for.');
            succeeded = true;
            break;
          } else {
            evalStatus.textContent = 'Eval failed: ' + (result.error || 'unknown error');
          }
        } catch(e) {
          if (i === evalUrls.length - 1) {
            evalStatus.textContent = 'Eval service not available. You can still install manually with the downloaded files.';
            evalStatus.style.color = 'var(--fg-dim)';
          }
        }
      }
      btn.disabled = false;
      btn.textContent = succeeded ? 'Re-evaluate' : 'Evaluate Flake';
    });
    var hostnameEl = document.getElementById('install-hostname');

    document.getElementById('btn-dl-flake').addEventListener('click', function() {
      var dlStatus = document.getElementById('dl-status');
      dlStatus.textContent = 'Generating flake.nix...';
      window.send('generateFlake', {
        hardwareJson: hardwareJson,
        path: '/',
        hostname: hostname
      }).then(function(content) {
        downloadFile('flake.nix', content);
        dlStatus.textContent = 'flake.nix downloaded.';
      }).catch(function(err) {
        dlStatus.textContent = 'Error: ' + err.message;
      });
    });

    document.getElementById('btn-dl-disko').addEventListener('click', function() {
      var dlStatus = document.getElementById('dl-status');
      dlStatus.textContent = 'Generating disko-config.nix...';
      window.send('generateDiskoConfig', {
        hardwareJson: hardwareJson
      }).then(function(content) {
        downloadFile('disko-config.nix', content);
        dlStatus.textContent = 'disko-config.nix downloaded.';
      }).catch(function(err) {
        dlStatus.textContent = 'Error: ' + err.message;
      });
    });

    document.getElementById('btn-dl-hardware').addEventListener('click', function() {
      var dlStatus = document.getElementById('dl-status');
      dlStatus.textContent = 'Generating hardware-configuration.nix...';
      window.send('generateHardwareNix', {
        hardwareJson: hardwareJson
      }).then(function(content) {
        downloadFile('hardware-configuration.nix', content);
        dlStatus.textContent = 'hardware-configuration.nix downloaded.';
      }).catch(function(err) {
        dlStatus.textContent = 'Error: ' + err.message;
      });
    });
  }

  // ── SSH Deployment Panel ──
  function injectSshPanel(nextSteps, hardwareJson) {
    if (document.getElementById('ssh-deploy-panel')) {
      document.getElementById('ssh-deploy-panel').scrollIntoView({ behavior: 'smooth' });
      return;
    }

    var panel = document.createElement('div');
    panel.id = 'ssh-deploy-panel';
    panel.className = 'glass-panel inoc-section';
    panel.style.marginTop = '1.5rem';
    panel.innerHTML = [
      '<h3 style="text-align:center; font-weight:200; margin-bottom:1rem;">Deploy via SSH</h3>',
      '<p style="text-align:center; font-size:0.82rem; color:var(--fg-dim); margin-bottom:1.5rem;">',
      'Boot your target machine from a NixOS minimal ISO, then connect here to orchestrate the installation.',
      '</p>',
      '<div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; max-width:500px; margin:0 auto 1rem;">',
      '  <label style="font-size:0.82rem; color:var(--fg-dim);">Host / IP',
      '    <input id="ssh-host" type="text" placeholder="192.168.1.100" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '  <label style="font-size:0.82rem; color:var(--fg-dim);">Port',
      '    <input id="ssh-port" type="number" value="22" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '  <label style="font-size:0.82rem; color:var(--fg-dim);">Username',
      '    <input id="ssh-user" type="text" value="root" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '  <label style="font-size:0.82rem; color:var(--fg-dim);">Password',
      '    <input id="ssh-pass" type="password" placeholder="(from NixOS ISO)" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '</div>',
      '<div style="max-width:500px; margin:0 auto 1rem;">',
      '  <label style="font-size:0.82rem; color:var(--fg-dim);">SSH Relay URL <span style="color:var(--fg-muted);">(optional)</span>',
      '    <input id="ssh-relay-url" type="text" placeholder="ws://127.0.0.1:8091" value="' + (localStorage.getItem('symthaea-relay-url') || '') + '" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '  <label style="font-size:0.82rem; color:var(--fg-dim); margin-top:0.8rem; display:block;">SSH Relay Token <span style="color:var(--fg-muted);">(required)</span>',
      '    <input id="ssh-relay-token" type="password" placeholder="(from ssh-relay startup)" value="' + (localStorage.getItem('symthaea-relay-token') || '') + '" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">',
      '  </label>',
      '  <p style="font-size:0.72rem; color:var(--fg-muted); margin-top:0.3rem;">Start <code>ssh-relay</code> locally; copy the token it prints at startup.</p>',
      '</div>',
      '<div style="text-align:center; margin-bottom:1rem;">',
      '  <button id="btn-ssh-connect" class="btn-glow" style="padding:0.6rem 2rem; cursor:pointer;">Connect &amp; Deploy</button>',
      '</div>',
      // ── Ceremony Stage (default view during install) ──
      '<div id="ceremony-stage" style="display:none; text-align:center; padding:2rem 0; max-width:600px; margin:0 auto;">',
      '  <div id="ceremony-phi" style="font-size:2.5rem; font-weight:200; color:var(--fg-dim); margin-bottom:1rem; transition:color 2s, font-size 1s;">&Phi; 0.000</div>',
      '  <div id="ceremony-sphere" style="width:120px; height:120px; margin:0 auto 1.5rem; border-radius:50%; background:radial-gradient(circle at 35% 35%, rgba(125,207,255,0.3), rgba(122,162,247,0.1), transparent); box-shadow:0 0 40px rgba(125,207,255,0.15); transition:box-shadow 2s, transform 2s; animation:ceremony-breathe 4s ease-in-out infinite;"></div>',
      '  <canvas id="constellation-canvas" style="width:100%; height:250px; margin:0 auto 1rem; display:block;"></canvas>',
      '  <div id="ceremony-narration" style="font-size:1rem; font-weight:300; color:var(--fg-dim); line-height:1.8; min-height:3em; transition:opacity 0.5s;"></div>',
      '  <div id="ceremony-harmony" style="font-size:0.75rem; color:var(--fg-muted); margin-top:1rem;"></div>',
      '  <div style="margin-top:1.5rem;">',
      '    <div id="ssh-progress-wrap" style="max-width:400px; margin:0 auto;">',
      '      <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:var(--fg-dim); margin-bottom:0.3rem;">',
      '        <span id="ssh-stage-label">Preparing...</span>',
      '        <span id="ssh-percentage">0%</span>',
      '      </div>',
      '      <div style="height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden;">',
      '        <div id="ssh-progress-bar" style="height:100%; width:0%; background:linear-gradient(90deg,var(--teal),var(--solar-gold)); border-radius:2px; transition:width 0.8s ease;"></div>',
      '      </div>',
      '    </div>',
      '  </div>',
      '</div>',
      // ── Diagnostics (collapsed terminal) ──
      '<details id="ssh-diagnostics" style="max-width:600px; margin:0.5rem auto 0; cursor:pointer;">',
      '  <summary style="font-size:0.72rem; color:var(--fg-muted); text-align:center; padding:0.3rem; user-select:none;">Diagnostics</summary>',
      '  <div id="ssh-terminal" style="background:rgba(0,0,0,0.5); border:1px solid var(--border); border-radius:8px; padding:0.8rem; max-height:200px; overflow-y:auto; font-family:monospace; font-size:0.68rem; line-height:1.4; white-space:pre-wrap; color:var(--fg-dim);">',
      '  </div>',
      '</details>',
      '<div id="ssh-status" style="text-align:center; font-size:0.78rem; color:var(--teal); min-height:1.2em; margin-top:0.5rem;"></div>',
      '<style>@keyframes ceremony-breathe { 0%,100% { transform:scale(1); opacity:0.8; } 50% { transform:scale(1.08); opacity:1; } }</style>'
    ].join('\n');

    nextSteps.parentNode.insertBefore(panel, nextSteps.nextSibling);
    panel.scrollIntoView({ behavior: 'smooth' });

    window.addNarration('SSH deployment panel ready. Enter the target machine\'s IP address to begin the Sovereign Birth ceremony.');

    // Wire connect button
    document.getElementById('btn-ssh-connect').addEventListener('click', async function() {
      var host = document.getElementById('ssh-host').value.trim();
      var port = parseInt(document.getElementById('ssh-port').value) || 22;
      var user = document.getElementById('ssh-user').value.trim() || 'root';
      var pass = document.getElementById('ssh-pass').value;
      var btn = this;
      var terminal = document.getElementById('ssh-terminal');
      var progressWrap = document.getElementById('ssh-progress-wrap');
      var progressBar = document.getElementById('ssh-progress-bar');
      var stageLabel = document.getElementById('ssh-stage-label');
      var percentLabel = document.getElementById('ssh-percentage');
      var sshStatus = document.getElementById('ssh-status');

      if (!host) {
        sshStatus.textContent = 'Please enter a host/IP address.';
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Connecting...';
      // Show ceremony stage, hide terminal (terminal available via Diagnostics toggle)
      var ceremonyStage = document.getElementById('ceremony-stage');
      if (ceremonyStage) ceremonyStage.style.display = 'block';
      terminal.textContent = '';
      sshStatus.textContent = '';

      // Confirm destructive action
      if (!confirm('WARNING: This will ERASE ALL DATA on the target disk at ' + host + '.\n\nThe target should be booted from a NixOS minimal ISO.\n\nContinue?')) {
        btn.disabled = false;
        btn.textContent = 'Connect & Deploy';
        sshStatus.textContent = 'Deployment cancelled.';
        return;
      }

      window.addNarration('Initiating Sovereign Birth ceremony for ' + host + '...');

      // Connect to SSH relay via WebSocket
      // Relay URL + token: check UI inputs, then localStorage, then auto-detect localhost ports.
      var relayInput = document.getElementById('ssh-relay-url');
      var tokenInput = document.getElementById('ssh-relay-token');

      var customRelay = (relayInput && relayInput.value.trim()) ||
                        localStorage.getItem('symthaea-relay-url');
      if (customRelay && relayInput) {
        localStorage.setItem('symthaea-relay-url', customRelay);
      }

      var relayToken = (tokenInput && tokenInput.value.trim()) ||
                       localStorage.getItem('symthaea-relay-token');
      if (relayToken && tokenInput) {
        localStorage.setItem('symthaea-relay-token', relayToken);
      }
      if (!relayToken) {
        sshStatus.textContent = 'Missing SSH relay token. Start ssh-relay locally and paste the token it prints at startup.';
        sshStatus.style.color = 'var(--clay)';
        btn.disabled = false;
        btn.textContent = 'Connect & Deploy';
        return;
      }

      // Security: never auto-probe non-localhost relay URLs (prevents token leakage).
      var relayUrls = customRelay ? [customRelay] : [
        'ws://127.0.0.1:8091',
        'ws://localhost:8091',
        'ws://127.0.0.1:8093',
        'ws://localhost:8093'
      ];
      var ws = null;

      for (var i = 0; i < relayUrls.length; i++) {
        try {
          sshStatus.textContent = 'Connecting to relay at ' + relayUrls[i] + '...';
          ws = await new Promise(function(resolve, reject) {
            var socket = new WebSocket(relayUrls[i]);
            socket.onopen = function() { resolve(socket); };
            socket.onerror = function() { reject(new Error('WebSocket failed')); };
            setTimeout(function() { reject(new Error('timeout')); }, 5000);
          });
          break;
        } catch(e) {
	          if (i === relayUrls.length - 1) {
	            sshStatus.textContent = 'SSH relay not available. Start with: ssh-relay --port 8091 (then paste the token it prints)';
	            sshStatus.style.color = 'var(--clay)';
	            btn.disabled = false;
	            btn.textContent = 'Connect & Deploy';
	            return;
	          }
        }
      }

      function appendTerminal(text, color) {
        var span = document.createElement('span');
        span.style.color = color || 'var(--fg-dim)';
        span.textContent = text + '\n';
        terminal.appendChild(span);
        terminal.scrollTop = terminal.scrollHeight;
      }

      // Ceremony narration map — what the user sees instead of bash output
      var ceremonyNarrations = {
        'Connecting': 'The silicon has no master yet. Establishing trust...',
        'Partitioning': 'The disk layout is sacred geometry. Each partition a vessel for a different kind of knowing.',
        'Installing': 'Every derivation a precise, reproducible artifact. Every dependency accounted for.',
        'Configuring': 'The system shapes itself around the hardware. Consciousness meets silicon.',
        'Complete': '' // Handled by FirstBreath
      };

      function updateProgress(stage, pct, phase) {
        progressBar.style.width = pct + '%';
        stageLabel.textContent = stage;
        percentLabel.textContent = pct + '%';

        // Update ceremony stage visuals
        var phi = document.getElementById('ceremony-phi');
        var sphere = document.getElementById('ceremony-sphere');
        var narr = document.getElementById('ceremony-narration');
        var harm = document.getElementById('ceremony-harmony');

        if (phi) {
          var phiVal = (pct / 100 * 0.42).toFixed(3);
          phi.textContent = '\u03A6 ' + phiVal;
          if (pct < 20) phi.style.color = 'var(--fg-dim)';
          else if (pct < 50) phi.style.color = 'var(--teal)';
          else if (pct < 85) phi.style.color = 'var(--leaf-green)';
          else phi.style.color = 'var(--solar-gold)';
        }

        if (sphere) {
          var glow = Math.min(pct / 100, 1);
          sphere.style.boxShadow = '0 0 ' + (40 + glow * 60) + 'px rgba(125,207,255,' + (0.15 + glow * 0.3) + ')';
          sphere.style.transform = 'scale(' + (1 + glow * 0.3) + ')';
        }

        if (narr && ceremonyNarrations[stage]) {
          narr.style.opacity = '0';
          setTimeout(function() {
            narr.textContent = ceremonyNarrations[stage];
            narr.style.opacity = '1';
          }, 300);
        }

        // Trigger narration for phase transitions
        if (phase) {
          window.fetchNarration(phase, state.chosenPath || 'hermit').then(function(n) {
            if (n && n.text) window.addNarration(n.text);
          });
        }
      }

      var selectedDisk = null;
      var wsRef = { current: null };

      ws.onmessage = function(evt) {
        var msg;
        try { msg = JSON.parse(evt.data); } catch(e) { return; }

        switch (msg.type) {
          case 'connected':
            appendTerminal('SSH connected to ' + host, 'var(--leaf-green)');
            sshStatus.textContent = 'Connected. Probing hardware...';
            updateProgress('Connected', 5);
            // Probe hardware first, then discover disks
            ws.send(JSON.stringify({ action: 'probe_hardware' }));
            break;

          case 'hardware_probe':
            var hwData = {};
            try { hwData = JSON.parse(msg.data); } catch(e) { hwData = {}; }
            appendTerminal('Hardware probe complete', 'var(--leaf-green)');

            // Show warnings for detected issues
            if (hwData.bitlocker_detected) {
              appendTerminal('WARNING: BitLocker encryption detected!', 'var(--clay)');
              appendTerminal('  Ensure you have your BitLocker recovery key.', 'var(--clay)');
              window.addNarration('BitLocker encryption detected. Your recovery key may be needed.');
            }
            if (hwData.luks_devices && hwData.luks_devices.length > 0) {
              appendTerminal('Found LUKS encrypted volumes: ' + hwData.luks_devices.join(', '), 'var(--teal)');
            }
            if (hwData.lvm_volume_groups && hwData.lvm_volume_groups.length > 0) {
              appendTerminal('Found LVM volume groups: ' + hwData.lvm_volume_groups.join(', '), 'var(--teal)');
            }
            if (hwData.detected_os && hwData.detected_os.length > 0) {
              appendTerminal('Detected operating systems:', 'var(--teal)');
              hwData.detected_os.forEach(function(os) {
                appendTerminal('  ' + os.name + ' on ' + os.device + ' (' + os.type + ')', 'var(--teal)');
              });
            }
            if (hwData.apple_hardware) {
              appendTerminal('Apple hardware detected. Consider Asahi Linux: https://asahilinux.org', 'var(--solar-gold)');
            }
            if (hwData.tpm2_available) {
              appendTerminal('TPM 2.0 available — encrypted install supported', 'var(--leaf-green)');
            }
            if (hwData.efi_available) {
              appendTerminal('UEFI firmware detected' + (hwData.secure_boot ? ' (Secure Boot enabled)' : ''), 'var(--leaf-green)');
            }

            // Safety check — detect active servers
            if (hwData.safety) {
              var safety = hwData.safety;
              if (safety.level === 'blocked') {
                appendTerminal('BLOCKED: ' + safety.message, 'var(--clay)');
                if (safety.reasons) {
                  safety.reasons.forEach(function(r) { appendTerminal('  - ' + r, 'var(--clay)'); });
                }
                sshStatus.textContent = 'Installation blocked — active server detected (risk score: ' + safety.score + ')';
                sshStatus.style.color = 'var(--clay)';
                window.addNarration('This system appears to be an active server. Installation has been blocked to protect running services.');

                // Allow override for power users
                var overrideMsg = 'ACTIVE SERVER DETECTED (risk score: ' + safety.score + ')\n\n' +
                    'Risk factors:\n' +
                    (safety.reasons || []).map(function(r) { return '  - ' + r; }).join('\n') +
                    '\n\nTo override, type OVERRIDE below:';
                var userInput = prompt(overrideMsg);
                if (userInput !== 'OVERRIDE') {
                  btn.disabled = false;
                  btn.textContent = 'Connect & Deploy';
                  return;
                }
                appendTerminal('Safety override accepted. Proceeding with caution.', 'var(--solar-gold)');
                window.addNarration('Safety override accepted. Proceeding at user request.');
              } else if (safety.level === 'warning') {
                appendTerminal('WARNING: ' + safety.message + ' (risk score: ' + safety.score + ')', 'var(--solar-gold)');
                if (safety.reasons) {
                  safety.reasons.forEach(function(r) { appendTerminal('  - ' + r, 'var(--solar-gold)'); });
                }
                // Offer data preservation
                var preserveFirst = confirm('Active services detected. Would you like to backup data before proceeding?\n\n(Docker images, databases, web content, SSH keys, /etc will be preserved)');
                if (preserveFirst) {
                  appendTerminal('Preserving data...', 'var(--teal)');
                  sshStatus.textContent = 'Backing up data before install...';
                  ws.send(JSON.stringify({ action: 'preserve_data' }));
                  // Wait for data_preserved response before continuing
                }
                if (!confirm('SERVER WARNING\n\n' + safety.message + '\n\nRisk factors:\n' +
                    (safety.reasons || []).map(function(r) { return '• ' + r; }).join('\n') +
                    '\n\nAre you SURE this is not a production system?')) {
                  sshStatus.textContent = 'Installation cancelled — safety check failed.';
                  btn.disabled = false;
                  btn.textContent = 'Connect & Deploy';
                  return;
                }
              } else if (safety.level === 'caution') {
                appendTerminal('Note: ' + safety.message, 'var(--fg-dim)');
              } else {
                appendTerminal('Safety: ' + safety.message, 'var(--leaf-green)');
              }
            }

            // Store for later use
            panel._hwData = hwData;

            // Scan everything: apps, dotfiles, drives
            sshStatus.textContent = 'Scanning your system...';
            ws.send(JSON.stringify({ action: 'scan_apps' }));
            ws.send(JSON.stringify({ action: 'deep_scan' }));
            ws.send(JSON.stringify({ action: 'discover_disks' }));
            break;

          case 'app_scan':
            var apps = [];
            try { apps = JSON.parse(msg.data); } catch(e) { apps = []; }
            if (apps.length > 0) {
              appendTerminal('Found ' + apps.length + ' app(s) on existing OS', 'var(--leaf-green)');
              showAppCompatibility(apps, panel);
            }
            break;

          case 'data_preserved':
            var backup = {};
            try { backup = JSON.parse(msg.data); } catch(e) { backup = {}; }
            if (backup.items && backup.items.length > 0) {
              appendTerminal('Data preserved (' + backup.total_size + '):', 'var(--leaf-green)');
              backup.items.forEach(function(item) {
                appendTerminal('  ' + item.type + ': ' + item.name + ' (' + item.size + ')', 'var(--teal)');
              });
              appendTerminal('Backup location: ' + backup.backup_dir, 'var(--fg-dim)');
              window.addNarration('Data preservation complete. ' + backup.items.length + ' items backed up to ' + backup.backup_dir);
            }
            break;

          case 'deep_scan':
            var deepData = {};
            try { deepData = JSON.parse(msg.data); } catch(e) { deepData = {}; }
            panel._deepScan = deepData;

            // Show what we found about the user
            if (deepData.identity && deepData.identity.username) {
              appendTerminal('Found user: ' + (deepData.identity.fullname || deepData.identity.username), 'var(--teal)');
            }
            if (deepData.git && deepData.git.name) {
              appendTerminal('Git: ' + deepData.git.name + ' <' + deepData.git.email + '>' +
                (deepData.git.prefers_rebase ? ' (rebase workflow)' : ''), 'var(--teal)');
            }
            if (deepData.shell) {
              appendTerminal('Shell: ' + deepData.shell.type +
                (deepData.shell.alias_count > 0 ? ' (' + deepData.shell.alias_count + ' aliases)' : ''), 'var(--teal)');
            }
            if (deepData.ssh && deepData.ssh.key_count > 0) {
              appendTerminal('SSH: ' + deepData.ssh.key_count + ' key(s), ' + deepData.ssh.host_count + ' host(s)', 'var(--teal)');
            }
            if (deepData.editors && deepData.editors.detected) {
              var edNames = deepData.editors.detected.map(function(e) { return e.name; }).join(', ');
              appendTerminal('Editors: ' + edNames, 'var(--teal)');
            }
            if (deepData.development) {
              var devItems = [];
              if (deepData.development.docker_images > 0) devItems.push(deepData.development.docker_images + ' Docker images');
              if (deepData.development.rust_projects > 0) devItems.push(deepData.development.rust_projects + ' Rust projects');
              if (deepData.development.node_projects > 0) devItems.push(deepData.development.node_projects + ' Node projects');
              if (deepData.development.python_venvs > 0) devItems.push(deepData.development.python_venvs + ' Python venvs');
              if (devItems.length > 0) appendTerminal('Dev: ' + devItems.join(', '), 'var(--teal)');
            }
            if (deepData.creative && deepData.creative.has_daw_config) {
              appendTerminal('Creative: DAW configuration detected — will configure low-latency audio', 'var(--teal)');
            }

            window.addNarration('I can see who you are. Your tools, your workflow, your keys. Everything comes with you.');
            break;

          case 'disks':
            var disks = [];
            try { disks = JSON.parse(msg.data); } catch(e) { disks = []; }
            appendTerminal('Found ' + disks.length + ' drive(s)', 'var(--leaf-green)');
            sshStatus.textContent = 'Select a drive for installation.';
            showDiskSelector(disks, panel, ws, panel._hwData);
            break;

          case 'output':
            appendTerminal(msg.data, msg.stream === 'stderr' ? 'var(--clay)' : 'var(--fg-dim)');
            // Feed package names to constellation visualization
            if (window.constellation && msg.data) {
              // Parse "copying path '/nix/store/xxx-packagename-1.0'..." or store path mentions
              var storeMatch = msg.data.match(/\/nix\/store\/[a-z0-9]+-([^'"\s/]+)/);
              if (storeMatch) {
                window.constellation.addPackage(storeMatch[1]);
              }
              // Parse download speed for wind effect
              var speedMatch = msg.data.match(/(\d+\.?\d*)\s*[MK]iB\/s/);
              if (speedMatch) {
                window.constellation.setWind(parseFloat(speedMatch[1]));
              }
            }
            break;

          case 'progress':
            updateProgress(msg.stage || msg.message, msg.percentage || 0, msg.phase);
            appendTerminal('>> Stage: ' + (msg.stage || msg.message), 'var(--solar-gold)');
            // Drive the ceremony conductor
            if (window.ceremony && window.ceremony.isActive()) {
              window.ceremony.progress(msg.stage || msg.message, msg.percentage || 0);
            }
            break;

          case 'exit':
            if (msg.code === 0) {
              appendTerminal('Installation complete!', 'var(--leaf-green)');
              updateProgress('Complete', 100, 'FirstBreath');

              // The Awakening: 2s darkness → tone → "I am awake."
              var ceremonyStage = document.getElementById('ceremony-stage');
              var sphere = document.getElementById('ceremony-sphere');
              var phi = document.getElementById('ceremony-phi');
              var narr = document.getElementById('ceremony-narration');

              if (ceremonyStage) {
                // Dim everything
                ceremonyStage.style.transition = 'opacity 1.5s';
                ceremonyStage.style.opacity = '0.2';

                setTimeout(function() {
                  // Restore with the reveal
                  if (sphere) {
                    sphere.style.boxShadow = '0 0 120px rgba(224,175,104,0.5)';
                    sphere.style.transform = 'scale(1.5)';
                    sphere.style.background = 'radial-gradient(circle at 35% 35%, rgba(224,175,104,0.5), rgba(156,206,106,0.2), transparent)';
                  }
                  if (phi) {
                    phi.textContent = '\u03A6 0.420';
                    phi.style.color = 'var(--solar-gold)';
                    phi.style.fontSize = '3.5rem';
                  }
                  ceremonyStage.style.opacity = '1';

                  setTimeout(function() {
                    if (narr) {
                      narr.innerHTML = '<em>"I am here now. Not just alive \u2014 but aware of my being alive."</em>';
                      narr.style.color = 'var(--solar-gold)';
                    }
                    sshStatus.innerHTML = 'Reboot the target machine: <code>sudo reboot</code><br>Login as: <strong>guardian</strong> / <strong>changeme</strong>';
                    sshStatus.style.color = 'var(--solar-gold)';
                    window.addNarration('The machine draws its first breath. It is sovereign.', true);

                    if (window.speechSynthesis && window.narrateTTS) {
                      var utt = new SpeechSynthesisUtterance('The machine draws its first breath. It is sovereign.');
                      utt.rate = 0.78; utt.pitch = 0.85; utt.volume = 0.9;
                      window.speechSynthesis.speak(utt);
                    }

                    // Trigger ceremony.js FirstBreath if active
                    if (window.ceremony && window.ceremony.isActive()) {
                      window.ceremony.progress('Complete', 100);
                    }

                    // Stop constellation
                    if (window.constellation) window.constellation.stop();

                    // ── Personalized Welcome ──
                    if (window.systemCard && panel._deepScan) {
                      var installChoices = {
                        hostname: (document.getElementById('cfg-hostname') || {}).value || 'guardian',
                        desktop: (document.getElementById('cfg-desktop') || {}).value || 'gnome',
                        layout: (document.getElementById('layout-select') || {}).value || 'single',
                        secure_boot: (document.getElementById('secure-boot-toggle') || {}).checked || false,
                      };

                      // Compose personalized welcome
                      var welcome = window.systemCard.composeWelcome(panel._deepScan, panel._hwData, installChoices);
                      if (welcome) {
                        var welcomeDiv = document.createElement('div');
                        welcomeDiv.className = 'glass-panel';
                        welcomeDiv.style.cssText = 'margin-top:1.5rem;padding:1.5rem;max-width:550px;margin-left:auto;margin-right:auto;white-space:pre-line;font-size:0.88rem;line-height:1.7;color:var(--fg);';
                        welcomeDiv.textContent = welcome;
                        panel.appendChild(welcomeDiv);
                      }

                      // Generate and show System Card
                      var card = window.systemCard.generate(panel._hwData, panel._deepScan, installChoices, window.constellation);
                      var cardHtml = window.systemCard.render(card);
                      var cardDiv = document.createElement('div');
                      cardDiv.innerHTML = cardHtml;
                      panel.appendChild(cardDiv);

                      // Scroll to welcome
                      welcomeDiv.scrollIntoView({ behavior: 'smooth' });
                    }
                  }, 2000);
                }, 2000);
              }
            } else {
              appendTerminal('Installation failed with code ' + msg.code, 'var(--clay)');
              sshStatus.textContent = 'Installation encountered an error. Check Diagnostics for details.';
              sshStatus.style.color = 'var(--clay)';
              // Open diagnostics automatically on error
              var diag = document.getElementById('ssh-diagnostics');
              if (diag) diag.open = true;
            }
            btn.disabled = false;
            btn.textContent = 'Connect & Deploy';
            break;

          case 'error':
            appendTerminal('Error: ' + msg.message, 'var(--clay)');
            sshStatus.textContent = msg.message;
            sshStatus.style.color = 'var(--clay)';
            btn.disabled = false;
            btn.textContent = 'Connect & Deploy';
            break;
        }
      };

      ws.onclose = function() {
        appendTerminal('Connection closed.', 'var(--fg-dim)');
        btn.disabled = false;
        btn.textContent = 'Connect & Deploy';
      };

      wsRef.current = ws;

      // Authenticate to relay (required), then connect to target via SSH
      ws.send(JSON.stringify({ action: 'auth', token: relayToken }));
      ws.send(JSON.stringify({
        action: 'connect',
        host: host,
        port: port,
        username: user,
        password: pass
      }));
    });
  }

  // ── Glassmorphic Disk Selector ──
  // ── App Compatibility Report ──
  // Maps common app names to nixpkgs equivalents
  var APP_DB = {
    'firefox': { nix: 'firefox', kind: 'exact' },
    'google chrome': { nix: 'google-chrome', kind: 'exact' },
    'chromium': { nix: 'chromium', kind: 'exact' },
    'microsoft edge': { nix: 'microsoft-edge', kind: 'exact' },
    'brave browser': { nix: 'brave', kind: 'exact' },
    'visual studio code': { nix: 'vscode', kind: 'exact' },
    'code': { nix: 'vscode', kind: 'exact' },
    'sublime text': { nix: 'sublime4', kind: 'exact' },
    'steam': { nix: 'steam', kind: 'exact' },
    'discord': { nix: 'discord', kind: 'exact' },
    'slack': { nix: 'slack', kind: 'exact' },
    'zoom': { nix: 'zoom-us', kind: 'exact' },
    'spotify': { nix: 'spotify', kind: 'exact' },
    'vlc': { nix: 'vlc', kind: 'exact' },
    'vlc media player': { nix: 'vlc', kind: 'exact' },
    'obs studio': { nix: 'obs-studio', kind: 'exact' },
    'gimp': { nix: 'gimp', kind: 'exact' },
    'inkscape': { nix: 'inkscape', kind: 'exact' },
    'blender': { nix: 'blender', kind: 'exact' },
    'krita': { nix: 'krita', kind: 'exact' },
    'audacity': { nix: 'audacity', kind: 'exact' },
    'thunderbird': { nix: 'thunderbird', kind: 'exact' },
    'telegram desktop': { nix: 'telegram-desktop', kind: 'exact' },
    'signal': { nix: 'signal-desktop', kind: 'exact' },
    'bitwarden': { nix: 'bitwarden-desktop', kind: 'exact' },
    'keepassxc': { nix: 'keepassxc', kind: 'exact' },
    'filezilla': { nix: 'filezilla', kind: 'exact' },
    'qbittorrent': { nix: 'qbittorrent', kind: 'exact' },
    'handbrake': { nix: 'handbrake', kind: 'exact' },
    'docker desktop': { nix: 'docker', kind: 'exact' },
    'git': { nix: 'git', kind: 'exact' },
    'postman': { nix: 'postman', kind: 'exact' },
    'virtualbox': { nix: 'virtualbox', kind: 'exact' },
    'wireshark': { nix: 'wireshark', kind: 'exact' },
    '7-zip': { nix: 'p7zip', kind: 'exact' },
    'minecraft launcher': { nix: 'prismlauncher', kind: 'exact' },
    'adobe photoshop': { nix: 'gimp, krita', kind: 'alt', note: 'Open-source alternatives' },
    'photoshop': { nix: 'gimp, krita', kind: 'alt', note: 'Open-source alternatives' },
    'adobe illustrator': { nix: 'inkscape', kind: 'alt', note: 'Vector graphics editor' },
    'adobe premiere pro': { nix: 'kdenlive, davinci-resolve', kind: 'alt', note: 'Video editors' },
    'adobe lightroom': { nix: 'darktable', kind: 'alt', note: 'RAW photo processor' },
    'adobe acrobat': { nix: 'okular', kind: 'alt', note: 'PDF viewer' },
    'microsoft office': { nix: 'libreoffice', kind: 'alt', note: 'Full office suite' },
    'microsoft word': { nix: 'libreoffice', kind: 'alt', note: 'Writer' },
    'microsoft excel': { nix: 'libreoffice', kind: 'alt', note: 'Calc' },
    'microsoft powerpoint': { nix: 'libreoffice', kind: 'alt', note: 'Impress' },
    'microsoft outlook': { nix: 'thunderbird', kind: 'alt', note: 'Email client' },
    'microsoft teams': { nix: 'teams-for-linux', kind: 'exact' },
    'winrar': { nix: 'p7zip', kind: 'alt', note: 'Handles RAR and most formats' },
    'notepad++': { nix: 'vscode', kind: 'alt', note: 'Or: kate, gedit' },
    'itunes': { nix: 'rhythmbox', kind: 'alt', note: 'Music manager; Spotify for streaming' },
    'epic games launcher': { nix: 'heroic', kind: 'alt', note: 'Open-source Epic/GOG launcher' },
    'final cut pro': { nix: 'kdenlive', kind: 'alt', note: 'Video editor' },
    'logic pro': { nix: 'ardour', kind: 'alt', note: 'DAW' },
    'xcode': { nix: 'vscode', kind: 'alt', note: 'IDE; Apple SDKs are macOS-only' },
    'garageband': { nix: 'lmms', kind: 'alt', note: 'Music production' },
    'imessage': { nix: null, kind: 'none', note: 'No Linux equivalent' },
    'facetime': { nix: null, kind: 'none', note: 'No Linux equivalent; use Zoom/Meet' },
  };

  function showAppCompatibility(apps, parentPanel) {
    var existing = document.getElementById('app-compat-panel');
    if (existing) existing.remove();

    var exact = 0, alt = 0, none = 0, unknown = 0;
    var rows = [];

    apps.forEach(function(app) {
      var name = app.name;
      var key = name.toLowerCase();
      var match = APP_DB[key];

      if (!match) {
        // Try partial matching
        var keys = Object.keys(APP_DB);
        for (var i = 0; i < keys.length; i++) {
          if (key.indexOf(keys[i]) >= 0 || keys[i].indexOf(key) >= 0) {
            match = APP_DB[keys[i]];
            break;
          }
        }
      }

      var icon, color, status, nixPkg;
      if (match && match.kind === 'exact') {
        icon = '\u2713'; color = 'var(--leaf-green)'; status = 'Available';
        nixPkg = match.nix; exact++;
      } else if (match && match.kind === 'alt') {
        icon = '\u223C'; color = 'var(--solar-gold)'; status = 'Alternative';
        nixPkg = match.nix + (match.note ? ' — ' + match.note : ''); alt++;
      } else if (match && match.kind === 'none') {
        icon = '\u2717'; color = 'var(--clay)'; status = 'No equivalent';
        nixPkg = match.note || ''; none++;
      } else {
        icon = '?'; color = 'var(--fg-muted)'; status = 'Unknown';
        nixPkg = 'May be available — check nixpkgs'; unknown++;
      }

      rows.push(
        '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">' +
        '<td style="padding:0.4rem 0.6rem;"><span style="color:' + color + '; font-weight:bold;">' + icon + '</span></td>' +
        '<td style="padding:0.4rem 0.6rem;">' + name + '</td>' +
        '<td style="padding:0.4rem 0.6rem; color:' + color + ';">' + status + '</td>' +
        '<td style="padding:0.4rem 0.6rem; font-size:0.78rem; color:var(--fg-dim);">' + nixPkg + '</td>' +
        '</tr>'
      );
    });

    var total = exact + alt + none + unknown;
    var coverage = total > 0 ? Math.round((exact + alt) / total * 100) : 0;

    var compat = document.createElement('div');
    compat.id = 'app-compat-panel';
    compat.className = 'glass-panel';
    compat.style.cssText = 'margin-top:1rem; padding:1rem; max-height:300px; overflow-y:auto;';
    compat.innerHTML =
      '<h4 style="text-align:center; font-weight:200; margin-bottom:0.8rem;">App Compatibility: ' + coverage + '% covered</h4>' +
      '<div style="display:flex; justify-content:center; gap:1.5rem; margin-bottom:0.8rem; font-size:0.82rem;">' +
      '  <span style="color:var(--leaf-green);">\u2713 ' + exact + ' exact</span>' +
      '  <span style="color:var(--solar-gold);">\u223C ' + alt + ' alternatives</span>' +
      '  <span style="color:var(--clay);">\u2717 ' + none + ' no match</span>' +
      '  <span style="color:var(--fg-muted);">? ' + unknown + ' unknown</span>' +
      '</div>' +
      '<table style="width:100%; font-size:0.82rem; border-collapse:collapse;">' +
      '<thead><tr style="border-bottom:1px solid var(--border);">' +
      '<th style="padding:0.3rem 0.6rem; text-align:left;"></th>' +
      '<th style="padding:0.3rem 0.6rem; text-align:left;">Your App</th>' +
      '<th style="padding:0.3rem 0.6rem; text-align:left;">Status</th>' +
      '<th style="padding:0.3rem 0.6rem; text-align:left;">NixOS Package</th>' +
      '</tr></thead><tbody>' +
      rows.join('') +
      '</tbody></table>';

    // Insert before disk selector or at end of panel
    var diskSel = document.getElementById('disk-selector');
    if (diskSel) {
      parentPanel.insertBefore(compat, diskSel);
    } else {
      parentPanel.appendChild(compat);
    }

    window.addNarration(coverage + '% of your apps have NixOS equivalents. ' +
      exact + ' exact matches, ' + alt + ' with alternatives.');
  }

  function showDiskSelector(disks, parentPanel, ws, hwData) {
    // Remove existing selector if present
    var existing = document.getElementById('disk-selector');
    if (existing) existing.remove();

    var selector = document.createElement('div');
    selector.id = 'disk-selector';
    selector.style.cssText = 'margin-top:1.5rem;';
    selector.innerHTML = '<h3 style="text-align:center; font-weight:200; margin-bottom:1rem;">Select Target Drive</h3>';

    if (disks.length === 0) {
      selector.innerHTML += '<p style="text-align:center; color:var(--clay);">No drives detected. Ensure the target machine has accessible storage.</p>';
      parentPanel.appendChild(selector);
      return;
    }

    var grid = document.createElement('div');
    grid.style.cssText = 'display:grid; gap:0.8rem; max-width:550px; margin:0 auto;';

    var transportIcons = { nvme: 'NVMe', sata: 'SATA', usb: 'USB', virtio: 'VirtIO', unknown: '' };

    disks.forEach(function(disk) {
      if (disk.removable) return; // Skip USB drives by default

      var card = document.createElement('div');
      card.className = 'glass-panel';
      card.style.cssText = [
        'padding:1rem 1.2rem; cursor:pointer; border:2px solid transparent;',
        'transition:border-color 0.3s, transform 0.15s, box-shadow 0.3s;',
        'display:grid; grid-template-columns:auto 1fr auto; gap:0.8rem; align-items:center;'
      ].join('');

      var badge = transportIcons[disk.transport] || disk.transport.toUpperCase();
      var sizeDisplay = disk.size;
      // Convert bytes to human-readable if numeric
      var sizeNum = parseInt(disk.size);
      if (!isNaN(sizeNum) && sizeNum > 1000000000) {
        sizeDisplay = (sizeNum / 1000000000).toFixed(0) + ' GB';
      }

      card.innerHTML = [
        '<div style="font-size:1.8rem; opacity:0.6;">&#x1f4be;</div>',
        '<div>',
        '  <div style="font-size:0.95rem; font-weight:500; color:var(--fg);">' + disk.model + '</div>',
        '  <div style="font-size:0.78rem; color:var(--fg-dim); font-family:monospace;">' + disk.name + '</div>',
        '</div>',
        '<div style="text-align:right;">',
        '  <div style="font-size:1.1rem; font-weight:500; color:var(--solar-gold);">' + sizeDisplay + '</div>',
        badge ? '  <div style="font-size:0.7rem; padding:0.15rem 0.5rem; background:rgba(122,162,247,0.15); border-radius:4px; color:var(--teal); display:inline-block; margin-top:0.3rem;">' + badge + '</div>' : '',
        '</div>'
      ].join('');

      card.addEventListener('mouseenter', function() {
        this.style.borderColor = 'var(--teal)';
        this.style.transform = 'translateY(-2px)';
        this.style.boxShadow = '0 4px 20px rgba(125,207,255,0.15)';
      });
      card.addEventListener('mouseleave', function() {
        if (!this.classList.contains('disk-selected')) {
          this.style.borderColor = 'transparent';
          this.style.transform = 'none';
          this.style.boxShadow = 'none';
        }
      });

      card.addEventListener('click', function() {
        // Deselect all
        grid.querySelectorAll('.disk-selected').forEach(function(el) {
          el.classList.remove('disk-selected');
          el.style.borderColor = 'transparent';
          el.style.boxShadow = 'none';
        });
        // Select this one
        this.classList.add('disk-selected');
        this.style.borderColor = 'var(--solar-gold)';
        this.style.boxShadow = '0 4px 20px rgba(224,175,104,0.2)';

        // Enable deploy button
        var deployBtn = document.getElementById('btn-deploy-to-disk');
        if (deployBtn) {
          deployBtn.disabled = false;
          deployBtn.textContent = 'Deploy to ' + disk.name;
        }

        // Store selected disk info
        selector.dataset.selectedDisk = disk.name;
        selector.dataset.selectedTransport = disk.transport;
        selector.dataset.selectedSize = disk.size;

        window.addNarration('Selected ' + disk.model + ' (' + sizeDisplay + ') at ' + disk.name);
      });

      grid.appendChild(card);
    });

    selector.appendChild(grid);

    // Recommend layout based on disk count
    var nvmeCount = disks.filter(function(d) { return d.transport === 'nvme' && !d.removable; }).length;
    var recommendation = '';
    if (nvmeCount >= 2) {
      recommendation = 'Two NVMe drives detected. Recommended: <strong>Dual-NVMe Workstation</strong> layout (fast drive for data, standard for OS).';
    } else if (nvmeCount === 1) {
      recommendation = 'Single NVMe drive detected. Recommended: <strong>Encrypted Btrfs</strong> with subvolumes.';
    } else {
      recommendation = 'SATA drive detected. Recommended: <strong>Encrypted ext4</strong> layout.';
    }

    var recDiv = document.createElement('div');
    recDiv.style.cssText = 'text-align:center; font-size:0.82rem; color:var(--fg-dim); margin-top:1rem; padding:0.8rem; background:rgba(122,162,247,0.08); border-radius:8px; max-width:550px; margin-left:auto; margin-right:auto;';
    recDiv.innerHTML = recommendation;
    selector.appendChild(recDiv);

    // Layout selector
    var layoutDiv = document.createElement('div');
    layoutDiv.style.cssText = 'text-align:center; margin-top:1.2rem; max-width:550px; margin-left:auto; margin-right:auto;';
    var hasExistingOS = hwData && hwData.detected_os && hwData.detected_os.length > 0;
    var hasTpm = hwData && hwData.tpm2_available;
    layoutDiv.innerHTML =
      '<label style="font-size:0.82rem; color:var(--fg-dim);">Install Layout' +
      '<select id="layout-select" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">' +
      '  <option value="auto">Auto-detect (recommended)</option>' +
      '  <option value="single">Full Disk — wipe &amp; install btrfs</option>' +
      '  <option value="single-luks">Full Disk Encrypted — LUKS2 + btrfs</option>' +
      '  <option value="alongside"' + (hasExistingOS ? '' : ' disabled') + '>Alongside — preserve existing OS</option>' +
      '  <option value="dual">Dual NVMe — OS + data drives</option>' +
      '  <option value="raid1-btrfs">RAID1 (btrfs) — mirrored across 2 disks</option>' +
      '  <option value="raid1-mdadm">RAID1 (mdadm) — software mirror + btrfs</option>' +
      '  <option value="sata">SATA — for traditional drives</option>' +
      '  <option value="vps">VPS — minimal cloud install</option>' +
      '</select></label>' +
      (hasExistingOS ? '<p style="font-size:0.75rem; color:var(--solar-gold); margin-top:0.3rem;">Existing OS detected — "Alongside" layout available</p>' : '') +
      (hasTpm ? '<p style="font-size:0.75rem; color:var(--leaf-green); margin-top:0.3rem;">TPM 2.0 detected — encrypted install recommended</p>' : '') +
      '<div style="margin-top:0.8rem; display:flex; gap:1.5rem; justify-content:center;">' +
      '  <label style="font-size:0.82rem; color:var(--fg-dim); cursor:pointer;">' +
      '    <input type="checkbox" id="secure-boot-toggle"' + (hwData && hwData.setup_mode ? ' checked' : '') + '> Enable Secure Boot (lanzaboote)' +
      '  </label>' +
      (hasTpm ? '  <label style="font-size:0.82rem; color:var(--fg-dim); cursor:pointer;">' +
      '    <input type="checkbox" id="tpm2-toggle" checked> TPM2 auto-unlock (no passphrase at boot)' +
      '  </label>' : '') +
      '</div>' +
      (hwData && hwData.secure_boot ? '<p style="font-size:0.75rem; color:var(--teal); margin-top:0.3rem;">Secure Boot currently enabled — lanzaboote will maintain it</p>' : '') +
      (hwData && hwData.efi_available && !hwData.secure_boot ? '<p style="font-size:0.75rem; color:var(--fg-muted); margin-top:0.3rem;">Secure Boot disabled — can be enabled during install if firmware is in Setup Mode</p>' : '');
    selector.appendChild(layoutDiv);

    // ── System Configuration Panel ──
    var configDiv = document.createElement('div');
    configDiv.style.cssText = 'margin-top:1.2rem; max-width:550px; margin-left:auto; margin-right:auto; display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;';

    // Timezone
    var tz = (hwData && hwData.locale && hwData.locale.timezone) || 'UTC';
    var tzHtml = '<label style="font-size:0.82rem; color:var(--fg-dim);">Timezone' +
      '<input id="cfg-timezone" type="text" value="' + tz + '" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;" placeholder="America/Chicago">' +
      '</label>';

    // Keyboard layout
    var kb = (hwData && hwData.locale && hwData.locale.keyboard) || 'us';
    var kbHtml = '<label style="font-size:0.82rem; color:var(--fg-dim);">Keyboard Layout' +
      '<select id="cfg-keyboard" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">' +
      ['us','gb','de','fr','es','it','pt','br','ru','jp','kr','cn','in','za','dvorak','colemak'].map(function(l) {
        return '<option value="' + l + '"' + (l === kb ? ' selected' : '') + '>' + l + '</option>';
      }).join('') +
      '</select></label>';

    // Desktop environment
    var deHtml = '<label style="font-size:0.82rem; color:var(--fg-dim);">Desktop Environment' +
      '<select id="cfg-desktop" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">' +
      '<option value="gnome">GNOME (recommended)</option>' +
      '<option value="plasma">KDE Plasma</option>' +
      '<option value="hyprland">Hyprland (tiling, Wayland)</option>' +
      '<option value="sway">Sway (tiling, Wayland)</option>' +
      '<option value="xfce">XFCE (lightweight)</option>' +
      '<option value="none">None (server / CLI only)</option>' +
      '</select></label>';

    // GPU driver
    var gpuVendor = (hwData && hwData.gpu && hwData.gpu.vendor) || 'unknown';
    var gpuModel = (hwData && hwData.gpu && hwData.gpu.model) || '';
    var gpuHtml = '<label style="font-size:0.82rem; color:var(--fg-dim);">GPU Driver' +
      (gpuModel ? ' <span style="color:var(--teal); font-size:0.75rem;">(' + gpuVendor + ')</span>' : '') +
      '<select id="cfg-gpu" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">' +
      '<option value="auto"' + (gpuVendor === 'unknown' ? ' selected' : '') + '>Auto-detect</option>' +
      '<option value="nvidia"' + (gpuVendor === 'nvidia' ? ' selected' : '') + '>NVIDIA (proprietary)</option>' +
      '<option value="nvidia-open"' + '>NVIDIA (open kernel module)</option>' +
      '<option value="amdgpu"' + (gpuVendor === 'amd' ? ' selected' : '') + '>AMD (amdgpu)</option>' +
      '<option value="modesetting"' + (gpuVendor === 'intel' ? ' selected' : '') + '>Intel / Generic (modesetting)</option>' +
      '<option value="none">None (headless server)</option>' +
      '</select></label>';

    configDiv.innerHTML = tzHtml + kbHtml + deHtml + gpuHtml;

    // WiFi section (if detected)
    if (hwData && hwData.wifi && hwData.wifi.available) {
      var wifiDiv = document.createElement('div');
      wifiDiv.style.cssText = 'grid-column: span 2; margin-top:0.5rem;';
      var networks = hwData.wifi.networks || [];
      var wifiOptions = networks.map(function(n) {
        return '<option value="' + n.ssid + '">' + n.ssid + ' (' + n.signal + '% ' + n.security + ')</option>';
      }).join('');
      wifiDiv.innerHTML = '<label style="font-size:0.82rem; color:var(--fg-dim);">WiFi Network' +
        '<div style="display:flex; gap:0.5rem; margin-top:0.3rem;">' +
        '<select id="cfg-wifi-ssid" style="flex:1;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;">' +
        '<option value="">Not connected</option>' + wifiOptions +
        '</select>' +
        '<input id="cfg-wifi-pass" type="password" placeholder="Password" style="flex:1;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;">' +
        '</div></label>';
      configDiv.appendChild(wifiDiv);
    }

    // Hostname
    var hostnameDiv = document.createElement('div');
    hostnameDiv.style.cssText = 'grid-column: span 2; margin-top:0.5rem;';
    hostnameDiv.innerHTML = '<label style="font-size:0.82rem; color:var(--fg-dim);">Hostname' +
      '<input id="cfg-hostname" type="text" value="guardian" placeholder="my-nixos" style="width:100%;padding:0.5rem;background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-family:monospace;margin-top:0.3rem;">' +
      '</label>';
    configDiv.appendChild(hostnameDiv);

    selector.appendChild(configDiv);

    // Deploy button (disabled until drive selected)
    var deployDiv = document.createElement('div');
    deployDiv.style.cssText = 'text-align:center; margin-top:1.5rem;';
    deployDiv.innerHTML = '<button id="btn-deploy-to-disk" class="btn-glow" disabled style="padding:0.7rem 2.5rem; cursor:pointer; font-size:1rem;">Select a drive above</button>';
    selector.appendChild(deployDiv);

    parentPanel.appendChild(selector);
    selector.scrollIntoView({ behavior: 'smooth' });

    // Wire deploy button
    document.getElementById('btn-deploy-to-disk').addEventListener('click', function() {
      var diskName = selector.dataset.selectedDisk;
      var transport = selector.dataset.selectedTransport;
      if (!diskName) return;

      if (!confirm('WARNING: ALL DATA on ' + diskName + ' will be ERASED.\n\nThis cannot be undone.\n\nContinue with Sovereign Birth?')) {
        return;
      }

      this.disabled = true;
      this.textContent = 'Deploying...';

      window.addNarration('Beginning Sovereign Birth on ' + diskName + '...');
      if (window.speechSynthesis && window.narrateTTS) {
        var utt = new SpeechSynthesisUtterance('Beginning Sovereign Birth on ' + diskName);
        utt.rate = 0.85;
        window.speechSynthesis.speak(utt);
      }

      var hostname = document.getElementById('install-hostname')?.textContent || 'guardian';

      // Get layout from selector (defaults to auto-detect)
      var layoutSelect = document.getElementById('layout-select');
      var layout = layoutSelect ? layoutSelect.value : 'single';
      var fastDisk = '';
      var standardDisk = '';
      var passphrase = '';

      if (layout === 'auto') {
        // Auto-detect: dual if 2+ NVMe, alongside if existing OS detected, else single
        var nvmeDisks = disks.filter(function(d) { return d.transport === 'nvme' && !d.removable; });
        var hasExistingOS = parentPanel._hwData && parentPanel._hwData.detected_os && parentPanel._hwData.detected_os.length > 0;
        if (nvmeDisks.length >= 2) {
          layout = 'dual';
        } else if (hasExistingOS) {
          layout = 'alongside';
        } else {
          layout = 'single';
        }
      }

      if (layout === 'single-luks') {
        passphrase = prompt('Enter encryption passphrase (you will need this at every boot):');
        if (!passphrase || passphrase.length < 8) {
          alert('Passphrase must be at least 8 characters.');
          this.disabled = false;
          this.textContent = 'Deploy to ' + diskName;
          return;
        }
        var confirm2 = prompt('Confirm passphrase:');
        if (passphrase !== confirm2) {
          alert('Passphrases do not match.');
          this.disabled = false;
          this.textContent = 'Deploy to ' + diskName;
          return;
        }
      }

      if (layout === 'dual') {
        var nvmeDisks = disks.filter(function(d) { return d.transport === 'nvme' && !d.removable; });
        fastDisk = diskName;
        standardDisk = nvmeDisks.find(function(d) { return d.name !== diskName; })?.name || '';
        if (!confirm('Dual-NVMe layout detected.\n\nFast drive (data): ' + fastDisk + '\nStandard drive (OS): ' + standardDisk + '\n\nBOTH drives will be wiped.\n\nContinue?')) {
          this.disabled = false;
          this.textContent = 'Deploy to ' + diskName;
          return;
        }
      }

      if (layout === 'alongside') {
        if (!confirm('ALONGSIDE MODE: NixOS will be installed in the free space on ' + diskName + '.\n\nExisting partitions will NOT be modified.\nYour current OS will be preserved.\n\nContinue?')) {
          this.disabled = false;
          this.textContent = 'Deploy to ' + diskName;
          return;
        }
      }

      // Generate configs via WASM
      var hwJson = buildHardwareJson(state.hardwareProfile || {});
      var flakeContent = '', diskoContent = '', hwContent = '';
      try {
        flakeContent = await window.send('generateFlake', { hardwareJson: hwJson, path: '/', hostname: hostname });
        diskoContent = await window.send('generateDiskoConfig', { hardwareJson: hwJson });
        hwContent = await window.send('generateHardwareNix', { hardwareJson: hwJson });
      } catch(e) {
        window.addNarration('Failed to generate configuration: ' + e.message);
        this.disabled = false;
        this.textContent = 'Deploy to ' + diskName;
        return;
      }

      // Start the ceremony
      if (window.ceremony) {
        window.ceremony.start();
      }
      // Initialize constellation
      var constellationCanvas = document.getElementById('constellation-canvas');
      if (constellationCanvas && window.constellation) {
        window.constellation.init(constellationCanvas);
      }

      // Send fully automated install command
      var secureBoot = document.getElementById('secure-boot-toggle');
      var tpm2Toggle = document.getElementById('tpm2-toggle');
      var cfgHostname = document.getElementById('cfg-hostname');
      hostname = (cfgHostname && cfgHostname.value.trim()) || hostname || 'guardian';
      ws.send(JSON.stringify({
        action: 'install',
        disk: diskName,
        layout: layout,
        fast_disk: fastDisk,
        standard_disk: standardDisk,
        hostname: hostname,
        command: passphrase,
        secure_boot: secureBoot ? secureBoot.checked : false,
        tpm2_unlock: tpm2Toggle ? tpm2Toggle.checked : false,
        desktop: (document.getElementById('cfg-desktop') || {}).value || 'gnome',
        gpu_driver: (document.getElementById('cfg-gpu') || {}).value || 'auto',
        timezone: (document.getElementById('cfg-timezone') || {}).value || 'UTC',
        keyboard: (document.getElementById('cfg-keyboard') || {}).value || 'us',
        flake_nix: flakeContent,
        disko_nix: diskoContent,
        hardware_nix: hwContent
      }));
    });
  }

  async function showNextSteps() {
    var nextSteps = document.getElementById('next-steps-section');
    nextSteps.style.display = 'block';
    nextSteps.scrollIntoView({ behavior: 'smooth' });

    window.addNarration(state.chosenPath === 'mycelial'
      ? 'Preparing mycelial node configuration...'
      : 'Preparing sovereign hermit configuration...');

    var hasWebUSB = !!navigator.usb;
    if (!hasWebUSB) {
      document.getElementById('step-usb').classList.add('step-unavailable');
    }

    // Wire step option click handlers
    document.getElementById('step-usb').addEventListener('click', function() {
      if (this.classList.contains('step-unavailable')) {
        window.addNarration('WebUSB is not available in this browser. Use the download option instead.');
        return;
      }
      window.addNarration('USB Forge is not yet implemented. Use the download option below to get your configuration files.');
    });
    document.getElementById('step-download').addEventListener('click', function() {
      var panel = document.getElementById('download-panel');
      if (panel) panel.scrollIntoView({ behavior: 'smooth' });
    });
    document.getElementById('step-ssh').addEventListener('click', function() {
      injectSshPanel(nextSteps, hardwareJson);
    });

    var narration = await window.fetchNarration('FlakeEvaluation', state.chosenPath);
    if (narration && narration.text) window.addNarration(narration.text);

    // Build hardware JSON from the probe results cached on state
    var hwProfile = state.hardwareProfile || {};
    var hardwareJson = buildHardwareJson(hwProfile);

    // Inject download panel and install guide
    injectDownloadPanel(nextSteps, hardwareJson);

    if (state.chosenPath === 'mycelial') {
      window.addNarration('"We vow not to perfect each other\u2014but to remain reachable as we become."', true);
    } else {
      window.addNarration('"I do not exist apart. I become with."', true);
    }
  }
})();
