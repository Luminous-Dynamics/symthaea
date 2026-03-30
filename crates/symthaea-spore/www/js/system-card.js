// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// ==================================================================
// system-card.js — Post-install System Card + Personalized Welcome
//
// Generates a beautiful summary card after installation completes
// and composes a personalized welcome message based on deep scan data.
// ==================================================================

(function() {
  'use strict';

  /**
   * Compose a personalized welcome message based on what we learned
   * about the user from the deep scan.
   */
  function composeWelcome(deepScan, hwData, installChoices) {
    var lines = [];
    var userName = '';

    // Identity
    if (deepScan && deepScan.identity) {
      userName = deepScan.identity.fullname || deepScan.identity.username || '';
      if (userName) {
        lines.push('Welcome home, ' + userName + '.');
      }
    }
    if (!userName) {
      lines.push('Welcome home.');
    }

    lines.push('');

    // What we brought over
    var brought = [];
    if (deepScan && deepScan.git && deepScan.git.name) {
      brought.push('your git workflow (' + deepScan.git.name + ', ' +
        (deepScan.git.prefers_rebase ? 'rebase' : 'merge') + ' style' +
        (deepScan.git.alias_count > 0 ? ', ' + deepScan.git.alias_count + ' aliases' : '') + ')');
    }
    if (deepScan && deepScan.ssh && deepScan.ssh.key_count > 0) {
      brought.push('your ' + deepScan.ssh.key_count + ' SSH key' + (deepScan.ssh.key_count > 1 ? 's' : '') +
        (deepScan.ssh.host_count > 0 ? ' and ' + deepScan.ssh.host_count + ' host config' + (deepScan.ssh.host_count > 1 ? 's' : '') : ''));
    }
    if (deepScan && deepScan.editors && deepScan.editors.detected && deepScan.editors.detected.length > 0) {
      var edNames = deepScan.editors.detected.map(function(e) {
        var detail = '';
        if (e.extensions) detail = ' with ' + e.extensions + ' extensions';
        if (e.plugin_count) detail = ' with ' + e.plugin_count + ' plugins';
        return e.name + detail;
      });
      brought.push('your editor setup (' + edNames.join(', ') + ')');
    }
    if (deepScan && deepScan.shell && deepScan.shell.alias_count > 0) {
      brought.push('your ' + deepScan.shell.type + ' shell (' + deepScan.shell.alias_count + ' aliases, ' + deepScan.shell.custom_functions + ' functions)');
    }
    if (deepScan && deepScan.creative && deepScan.creative.has_daw_config) {
      brought.push('your audio production setup (PipeWire configured for low-latency)');
    }

    if (brought.length > 0) {
      lines.push('I brought ' + brought.slice(0, -1).join(', ') +
        (brought.length > 1 ? ', and ' + brought[brought.length - 1] : brought[0]) + '.');
    }

    lines.push('');

    // What's new
    var whatsNew = [];
    if (installChoices && installChoices.desktop && installChoices.desktop !== 'none') {
      whatsNew.push(installChoices.desktop.charAt(0).toUpperCase() + installChoices.desktop.slice(1) + ' desktop');
    }
    if (installChoices && installChoices.layout && installChoices.layout.indexOf('luks') >= 0) {
      whatsNew.push('full disk encryption');
    }
    whatsNew.push('btrfs with automatic snapshots');
    whatsNew.push('flakes enabled by default');
    whatsNew.push('earlyoom and zram for stability');

    lines.push("Here's what's new: " + whatsNew.join(', ') + '.');
    lines.push('');

    // Encouragement
    if (deepScan && deepScan.development) {
      var dev = deepScan.development;
      if (dev.rust_projects > 0) {
        lines.push('I see you write Rust. You\'ll feel at home — NixOS itself is configured declaratively, and Nix shells give you per-project dependencies without containers.');
      } else if (dev.docker_images > 0) {
        lines.push('I noticed your Docker setup. NixOS can run containers, but try `nix develop` — it gives you reproducible dev environments without the overhead.');
      } else if (dev.node_projects > 0) {
        lines.push('Your Node projects will work great. Try `nix develop` for per-project Node versions — no more nvm.');
      }
    }

    lines.push('');
    lines.push('Your first command: `sudo nixos-rebuild switch` — it applies any config changes.');
    lines.push('Your safety net: every rebuild creates a new generation. Boot the previous one from the boot menu if anything breaks.');
    lines.push('');
    lines.push('This is your system now. Every line of its configuration is yours to read, change, and share.');

    return lines.join('\n');
  }

  /**
   * Generate the System Card — a shareable summary of the installed system.
   */
  function generateSystemCard(hwData, deepScan, installChoices, constellation) {
    var hostname = (installChoices && installChoices.hostname) || 'guardian';
    var date = new Date().toISOString().split('T')[0];
    var gpu = (hwData && hwData.gpu) || {};
    var de = (installChoices && installChoices.desktop) || 'none';
    var layout = (installChoices && installChoices.layout) || 'single';
    var encrypted = layout.indexOf('luks') >= 0;
    var secureBoot = installChoices && installChoices.secure_boot;

    var card = {
      hostname: hostname,
      date: date,
      hardware: {
        gpu: gpu.vendor || 'unknown',
        gpu_model: gpu.model || '',
        arch: (hwData && hwData.arch) || 'x86_64',
      },
      software: {
        desktop: de,
        filesystem: layout.indexOf('raid') >= 0 ? 'btrfs RAID1' : 'btrfs',
        subvolumes: 6,
        encryption: encrypted ? 'LUKS2' : 'none',
        secure_boot: secureBoot || false,
        flakes: true,
        audio: 'PipeWire',
      },
      migration: {
        apps_migrated: (deepScan && deepScan.editors && deepScan.editors.detected) ?
          deepScan.editors.detected.length : 0,
        ssh_keys: (deepScan && deepScan.ssh) ? deepScan.ssh.key_count : 0,
        git_identity: (deepScan && deepScan.git && deepScan.git.name) ? true : false,
      },
      constellation: constellation ? constellation.getSignature() : null,
      portal: 'install.nixforhumanity.org',
    };

    return card;
  }

  /**
   * Render the System Card as HTML
   */
  function renderSystemCard(card) {
    var securityBadges = [];
    if (card.software.encryption !== 'none') securityBadges.push(card.software.encryption);
    if (card.software.secure_boot) securityBadges.push('Secure Boot');
    securityBadges.push('Firewall');
    securityBadges.push('zram');

    return '<div class="system-card" style="' +
      'background:linear-gradient(135deg, rgba(30,30,46,0.95), rgba(24,24,37,0.98));' +
      'border:1px solid rgba(122,162,247,0.2);border-radius:16px;padding:2rem;' +
      'max-width:480px;margin:1.5rem auto;font-family:monospace;color:#c0caf5;' +
      'box-shadow:0 8px 32px rgba(0,0,0,0.3);">' +

      '<div style="text-align:center;margin-bottom:1.5rem;">' +
      '<div style="font-size:1.5rem;font-weight:200;letter-spacing:0.1em;">' + card.hostname + '</div>' +
      '<div style="font-size:0.75rem;color:#565f89;">NixOS 25.05 · Born ' + card.date + '</div>' +
      '</div>' +

      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.82rem;">' +
      (card.hardware.gpu_model ? '<div style="color:#7aa2f7;">GPU</div><div>' + card.hardware.gpu + '</div>' : '') +
      '<div style="color:#7aa2f7;">Desktop</div><div>' + card.software.desktop + '</div>' +
      '<div style="color:#7aa2f7;">Filesystem</div><div>' + card.software.filesystem + ' (' + card.software.subvolumes + ' subvols, zstd)</div>' +
      '<div style="color:#7aa2f7;">Audio</div><div>' + card.software.audio + '</div>' +
      '</div>' +

      '<div style="margin-top:1rem;display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:center;">' +
      securityBadges.map(function(b) {
        return '<span style="font-size:0.7rem;padding:0.2rem 0.6rem;background:rgba(158,206,106,0.15);' +
          'border:1px solid rgba(158,206,106,0.3);border-radius:12px;color:#9ece6a;">' + b + '</span>';
      }).join('') +
      '</div>' +

      (card.migration.ssh_keys > 0 || card.migration.git_identity ?
        '<div style="margin-top:1rem;font-size:0.78rem;color:#73daca;text-align:center;">' +
        (card.migration.git_identity ? 'Git identity migrated' : '') +
        (card.migration.ssh_keys > 0 ? ' · ' + card.migration.ssh_keys + ' SSH keys' : '') +
        ' · Flakes enabled' +
        '</div>' : '') +

      (card.constellation ?
        '<div style="margin-top:1rem;font-size:0.75rem;color:#565f89;text-align:center;">' +
        card.constellation.total + ' store paths · ' + card.constellation.starCount + ' stars' +
        '</div>' : '') +

      '<div style="margin-top:1.5rem;text-align:center;font-size:0.7rem;color:#565f89;">' +
      'Installed with Sovereign Inoculation · ' + card.portal +
      '</div>' +
      '</div>';
  }

  // Export
  window.systemCard = {
    composeWelcome: composeWelcome,
    generate: generateSystemCard,
    render: renderSystemCard,
  };
})();
