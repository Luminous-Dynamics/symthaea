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

    // Wire up download buttons
    var hostname = 'guardian';
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
