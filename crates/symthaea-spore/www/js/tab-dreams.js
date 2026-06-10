// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// tab-dreams.js — Tab 4: Dream controls, wisdom journal, FEP
// ==================================================================
(function() {
  'use strict';
  var state = window.portalState;

  async function updateDreamStats() {
    if (!state.workerReady) return;
    try {
      var stats = await window.send('dreamStats', {});
      if (stats) {
        document.getElementById('ds-events').textContent = stats.events_recorded || stats.total_events || 0;
        document.getElementById('ds-cycles').textContent = stats.cycles_completed || stats.total_cycles || state.dreamCyclesDone;
        document.getElementById('ds-insights').textContent = stats.total_insights || stats.insights || 0;
      }
    } catch (e) {}
    try {
      var wc = await window.send('dreamWisdomCount', {});
      document.getElementById('ds-wisdom').textContent = wc || 0;
    } catch (e) {}
  }

  function addWisdomEntry(entry) {
    var journal = document.getElementById('wisdom-journal');
    // Clear empty message on first entry
    if (state.dreamWisdomEntries.length === 0) {
      journal.innerHTML = '';
    }
    state.dreamWisdomEntries.push(entry);

    var div = document.createElement('div');
    div.className = 'wisdom-entry';
    div.innerHTML =
      '<div class="w-context">' + window.escapeHtml(entry.context || entry.description || 'Dream insight') + '</div>' +
      '<div class="w-action">' + window.escapeHtml(entry.better_action || entry.insight || entry.action || '') + '</div>' +
      '<div class="w-meta">\u0394\u03A6 ' + (entry.phi_improvement || 0).toFixed(3) +
      ' | confidence: ' + (entry.confidence || 0).toFixed(2) + '</div>';
    journal.appendChild(div);
    journal.scrollTop = journal.scrollHeight;
  }

  document.getElementById('btn-dream-cycle').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Dreaming...';
    try {
      var result = await window.send('dreamCycle', {});
      state.dreamCyclesDone++;
      if (result && result.wisdom) {
        addWisdomEntry(result.wisdom);
      } else if (result && result.insight) {
        addWisdomEntry({ context: 'Dream cycle', insight: result.insight, phi_improvement: result.phi_delta || 0, confidence: result.confidence || 0 });
      }
      await updateDreamStats();
    } catch (e) {
      console.warn('Dream cycle failed:', e);
    }
    this.disabled = false;
    this.textContent = 'Dream Cycle';
  });

  document.getElementById('btn-dream-session').addEventListener('click', async function() {
    if (!state.workerReady) return;
    var count = parseInt(document.getElementById('dream-cycle-count').value) || 5;
    this.disabled = true;
    this.textContent = 'Dreaming (' + count + ')...';
    try {
      var result = await window.send('dreamSession', { cycles: count });
      state.dreamCyclesDone += count;
      if (result && result.wisdoms) {
        result.wisdoms.forEach(function(w) { addWisdomEntry(w); });
      } else if (result && result.wisdom_entries) {
        result.wisdom_entries.forEach(function(w) { addWisdomEntry(w); });
      }
      await updateDreamStats();
    } catch (e) {
      console.warn('Dream session failed:', e);
    }
    this.disabled = false;
    this.textContent = 'Dream Session';
  });

  // FEP
  document.getElementById('btn-fep-cycle').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Computing...';
    try {
      var result = await window.send('fepCycle', {});
      if (result) {
        // Free energy
        var feVal = result.free_energy || result.total_free_energy || 0;
        document.getElementById('fep-val').textContent = feVal.toFixed(4);

        // Exploration mode
        var modeEl = document.getElementById('fep-mode');
        var exploring = result.exploring || result.exploration_mode || false;
        modeEl.textContent = exploring ? 'exploring' : 'exploiting';
        modeEl.className = 'fep-mode ' + (exploring ? 'exploring' : 'exploiting');

        // Motor commands
        if (result.motor_commands || result.actions) {
          var commands = result.motor_commands || result.actions || [];
          var logEl = document.getElementById('motor-log');
          if (state.motorEntries.length === 0) logEl.innerHTML = '';

          commands.forEach(function(cmd) {
            var entry = document.createElement('div');
            entry.className = 'motor-entry';
            var cmdType = cmd.command_type || cmd.type || cmd.action || 'unknown';
            var cmdIntensity = cmd.intensity || cmd.magnitude || 0;
            var cmdConf = cmd.confidence || 0;
            entry.textContent = cmdType + ' | intensity: ' + cmdIntensity.toFixed(2) + ' | conf: ' + cmdConf.toFixed(2);
            logEl.appendChild(entry);
            state.motorEntries.push(entry);
            logEl.scrollTop = logEl.scrollHeight;
          });
        }
      }
    } catch (e) {
      console.warn('FEP cycle failed:', e);
      // Try standalone free energy
      try {
        var fe = await window.send('freeEnergy', {});
        if (fe !== undefined && fe !== null) {
          var feNum = typeof fe === 'object' ? (fe.free_energy || fe.total || 0) : fe;
          document.getElementById('fep-val').textContent = (typeof feNum === 'number' ? feNum : 0).toFixed(4);
        }
      } catch (e2) {}
    }
    this.disabled = false;
    this.textContent = 'Run FEP Cycle';
  });
})();
