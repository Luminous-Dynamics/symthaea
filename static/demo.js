// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Symthaea Live Consciousness Demo — WebSocket Client
// No build tooling, vanilla JS + Chart.js

const WINDOW_SIZE = 200;
const WS_URL = `ws://${window.location.host}/v1/ws/live`;

// --- State ---
let ws = null;
let paused = false;
const history = {
    cycles: [],
    consciousness: [],
    psi: [],
    predictionError: [],
    valence: [],
    arousal: [],
    coherence: [],
    moral: [],
    moralFE: [],
    moralKL: [],
    moralEntropy: [],
    weightSpectral: [],
    weightEquation: [],
    weightPipeline: [],
    weightMultimodal: [],
    microPhi: [],
    mesoPhi: [],
    macroPhi: [],
};

const HARMONY_LABELS = [
    'Resonant Coherence',
    'Pan-Sentient Flourishing',
    'Integral Wisdom',
    'Infinite Play',
    'Universal Interconnectedness',
    'Sacred Reciprocity',
    'Evolutionary Progression',
];

const HARMONY_COLORS = ['#06b6d4', '#f43f5e', '#a855f7', '#f59e0b', '#22c55e', '#3b82f6', '#14b8a6'];

// Drift sparkline rolling buffer
const driftHistory = [];
const SPARKLINE_SIZE = 20;

// --- Chart Setup ---
const chartOptions = (yMin, yMax) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: true, labels: { color: '#9ca3af', font: { size: 10 } } } },
    scales: {
        x: { display: false },
        y: {
            min: yMin,
            max: yMax,
            ticks: { color: '#6b7280', font: { size: 9 } },
            grid: { color: '#1f2937' },
        },
    },
});

const makeDataset = (label, color, data) => ({
    label,
    data,
    borderColor: color,
    backgroundColor: color + '20',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
});

const chartPsi = new Chart(document.getElementById('chartPsi'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Consciousness', '#06b6d4', history.consciousness),
            makeDataset('Psi', '#a855f7', history.psi),
        ],
    },
    options: chartOptions(0, 1),
});

const chartError = new Chart(document.getElementById('chartError'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [makeDataset('Prediction Error', '#f59e0b', history.predictionError)],
    },
    options: chartOptions(0, undefined),
});

const chartAffect = new Chart(document.getElementById('chartAffect'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Valence', '#22c55e', history.valence),
            makeDataset('Arousal', '#f43f5e', history.arousal),
        ],
    },
    options: chartOptions(-1, 1),
});

const chartCoherence = new Chart(document.getElementById('chartCoherence'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Coherence', '#06b6d4', history.coherence),
            makeDataset('Moral Score', '#a855f7', history.moral),
        ],
    },
    options: chartOptions(0, 1),
});

// --- Moral Topology Charts ---

const chartHarmony = new Chart(document.getElementById('chartHarmony'), {
    type: 'radar',
    data: {
        labels: HARMONY_LABELS,
        datasets: [
            {
                label: 'Observed',
                data: [0, 0, 0, 0, 0, 0, 0],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6,182,212,0.15)',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '#06b6d4',
            },
            {
                label: 'Prior',
                data: [0, 0, 0, 0, 0, 0, 0],
                borderColor: '#a855f7',
                backgroundColor: 'rgba(168,85,247,0.08)',
                borderWidth: 1.5,
                borderDash: [4, 4],
                pointRadius: 2,
                pointBackgroundColor: '#a855f7',
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: { legend: { display: true, labels: { color: '#9ca3af', font: { size: 9 } } } },
        scales: {
            r: {
                angleLines: { color: '#1f2937' },
                grid: { color: '#1f2937' },
                pointLabels: { color: '#9ca3af', font: { size: 8 } },
                ticks: { display: false },
                suggestedMin: 0,
                suggestedMax: 1,
            },
        },
    },
});

const chartMoralFE = new Chart(document.getElementById('chartMoralFE'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Free Energy', '#06b6d4', history.moralFE),
            makeDataset('KL Divergence', '#f59e0b', history.moralKL),
            makeDataset('Entropy', '#22c55e', history.moralEntropy),
        ],
    },
    options: chartOptions(0, undefined),
});

const chartPersistence = new Chart(document.getElementById('chartPersistence'), {
    type: 'scatter',
    data: {
        datasets: [
            {
                label: 'Components (B0)',
                data: [],
                backgroundColor: '#06b6d4',
                pointRadius: 4,
            },
            {
                label: 'Cycles (B1)',
                data: [],
                backgroundColor: '#a855f7',
                pointRadius: 4,
            },
            {
                label: 'Voids (B2)',
                data: [],
                backgroundColor: '#22c55e',
                pointRadius: 4,
            },
            {
                label: 'Diagonal',
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: '#374151',
                borderWidth: 1,
                pointRadius: 0,
                showLine: true,
                type: 'line',
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: { display: true, labels: { color: '#9ca3af', font: { size: 9 } } },
            tooltip: {
                callbacks: {
                    label(ctx) {
                        if (ctx.datasetIndex === 3) return null; // Diagonal line — no tooltip
                        const types = ['Component', 'Cycle', 'Void'];
                        return [
                            `Type: ${types[ctx.datasetIndex]}`,
                            `Birth: ${ctx.parsed.x.toFixed(3)}`,
                            `Death: ${ctx.parsed.y.toFixed(3)}`,
                            `Persistence: ${(ctx.parsed.y - ctx.parsed.x).toFixed(3)}`,
                        ];
                    },
                },
            },
        },
        scales: {
            x: {
                title: { display: true, text: 'Birth', color: '#6b7280', font: { size: 9 } },
                ticks: { color: '#6b7280', font: { size: 9 } },
                grid: { color: '#1f2937' },
                min: 0,
            },
            y: {
                title: { display: true, text: 'Death', color: '#6b7280', font: { size: 9 } },
                ticks: { color: '#6b7280', font: { size: 9 } },
                grid: { color: '#1f2937' },
                min: 0,
            },
        },
    },
});

// --- Drift Sparkline (tiny chart, no axes) ---
const sparklineDrift = new Chart(document.getElementById('sparklineDrift'), {
    type: 'line',
    data: {
        labels: Array(SPARKLINE_SIZE).fill(''),
        datasets: [{
            data: [],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,0.15)',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
        }],
    },
    options: {
        responsive: false,
        animation: { duration: 0 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
            x: { display: false },
            y: { display: false, min: 0 },
        },
    },
});

// --- Consciousness Weights Chart ---
const chartWeights = new Chart(document.getElementById('chartWeights'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Spectral', '#06b6d4', history.weightSpectral),
            makeDataset('Equation', '#a855f7', history.weightEquation),
            makeDataset('Pipeline', '#22c55e', history.weightPipeline),
            makeDataset('Multimodal', '#f59e0b', history.weightMultimodal),
        ],
    },
    options: chartOptions(0, 1),
});

// --- Structural Phi Decomposition Chart ---
const chartPhiDecomp = new Chart(document.getElementById('chartPhiDecomp'), {
    type: 'bar',
    data: {
        labels: ['Micro', 'Meso', 'Macro'],
        datasets: [{
            label: 'Phi',
            data: [0, 0, 0],
            backgroundColor: ['#06b6d4', '#a855f7', '#22c55e'],
            borderColor: ['#06b6d4', '#a855f7', '#22c55e'],
            borderWidth: 1,
        }],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: { legend: { display: false } },
        scales: {
            x: {
                ticks: { color: '#9ca3af', font: { size: 10 } },
                grid: { color: '#1f2937' },
            },
            y: {
                min: 0,
                ticks: { color: '#6b7280', font: { size: 9 } },
                grid: { color: '#1f2937' },
            },
        },
    },
});

// --- Utility: hex color interpolation ---
function lerpColor(a, b, t) {
    const ah = parseInt(a.slice(1), 16);
    const bh = parseInt(b.slice(1), 16);
    const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
    const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
    const rr = Math.round(ar + (br - ar) * t);
    const rg = Math.round(ag + (bg - ag) * t);
    const rb = Math.round(ab + (bb - ab) * t);
    return `#${((rr << 16) | (rg << 8) | rb).toString(16).padStart(6, '0')}`;
}

// --- WebSocket ---
function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        document.getElementById('statusDot').classList.remove('disconnected');
        document.getElementById('statusText').textContent = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('statusDot').classList.add('disconnected');
        document.getElementById('statusText').textContent = 'Disconnected — reconnecting...';
        setTimeout(connect, 2000);
    };

    ws.onerror = () => {
        ws.close();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'connected') return; // Skip hello

        // Push to history (rolling window)
        const push = (arr, val) => {
            arr.push(val);
            if (arr.length > WINDOW_SIZE) arr.shift();
        };

        push(history.cycles, data.cycle);
        push(history.consciousness, data.consciousness_level);
        push(history.psi, data.narrative_self_psi);
        push(history.predictionError, data.prediction_error);
        push(history.valence, data.valence);
        push(history.arousal, data.arousal);
        push(history.coherence, data.coherence);
        push(history.moral, data.moral_score);
        push(history.moralFE, data.moral_free_energy || 0);
        push(history.moralKL, data.moral_kl_divergence || 0);
        push(history.moralEntropy, data.moral_entropy || 0);

        // Consciousness weights history
        const cw = data.consciousness_weights || [0, 0, 0, 0];
        push(history.weightSpectral, cw[0]);
        push(history.weightEquation, cw[1]);
        push(history.weightPipeline, cw[2]);
        push(history.weightMultimodal, cw[3]);

        // Structural Phi history
        push(history.microPhi, data.structural_micro_phi || 0);
        push(history.mesoPhi, data.structural_meso_phi || 0);
        push(history.macroPhi, data.structural_macro_phi || 0);

        // Update line charts
        const labels = history.cycles.map(String);
        chartPsi.data.labels = labels;
        chartError.data.labels = labels;
        chartAffect.data.labels = labels;
        chartCoherence.data.labels = labels;
        chartMoralFE.data.labels = labels;

        chartPsi.update('none');
        chartError.update('none');
        chartAffect.update('none');
        chartCoherence.update('none');
        chartMoralFE.update('none');

        // Update consciousness weights chart
        chartWeights.data.labels = labels;
        chartWeights.update('none');

        // Update structural Phi bar chart
        chartPhiDecomp.data.datasets[0].data = [
            data.structural_micro_phi || 0,
            data.structural_meso_phi || 0,
            data.structural_macro_phi || 0,
        ];
        chartPhiDecomp.update('none');

        // Update harmony radar
        if (data.moral_scenario_distribution) {
            chartHarmony.data.datasets[0].data = data.moral_scenario_distribution;
        }
        if (data.moral_prior_distribution) {
            chartHarmony.data.datasets[1].data = data.moral_prior_distribution;
        }
        chartHarmony.update('none');

        // Update persistence diagram
        if (data.moral_persistence_diagram) {
            const pd = data.moral_persistence_diagram;
            const toPoints = (pairs) => (pairs || []).map(p => ({ x: p[0], y: p[1] }));
            chartPersistence.data.datasets[0].data = toPoints(pd.components);
            chartPersistence.data.datasets[1].data = toPoints(pd.cycles);
            chartPersistence.data.datasets[2].data = toPoints(pd.voids);
            // Update diagonal to span data range
            const maxVal = Math.max(
                pd.bottleneck_distance || 1,
                ...(pd.components || []).flat(),
                ...(pd.cycles || []).flat(),
                ...(pd.voids || []).flat(),
                1
            );
            chartPersistence.data.datasets[3].data = [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }];
            chartPersistence.update('none');
        }

        // Update Betti badges
        if (data.moral_betti) {
            document.getElementById('bettiBeta0').textContent = data.moral_betti[0];
            document.getElementById('bettiBeta1').textContent = data.moral_betti[1];
            document.getElementById('bettiBeta2').textContent = data.moral_betti[2];
        }

        // Anomaly flags
        const setFlag = (id, active) => {
            const el = document.getElementById(id);
            if (el) el.classList.toggle('active', active);
        };
        setFlag('flagSurprise', data.surprise_triggered);
        setFlag('flagGWT', data.gwt_broadcast);
        setFlag('flagDream', data.dream_insights > 0);
        setFlag('flagReasoning', data.reasoning_confidence > 0);
        setFlag('flagInversion', data.moral_value_inversion);
        setFlag('flagFESpike', data.moral_free_energy_spike);
        setFlag('flagDrift', data.moral_drift_alert);
        setFlag('flagFragmentation', data.moral_fragmentation_increase);
        setFlag('flagResponseActive', data.moral_anomaly_response_applied);

        // Flash anomaly badges
        const flashFlag = (id, active) => {
            const el = document.getElementById(id);
            if (!el) return;
            if (active) {
                el.classList.add('anomaly-flash');
            } else {
                el.classList.remove('anomaly-flash');
            }
        };
        flashFlag('flagInversion', data.moral_value_inversion);
        flashFlag('flagFESpike', data.moral_free_energy_spike);
        flashFlag('flagFragmentation', data.moral_fragmentation_increase);
        flashFlag('flagResponseActive', data.moral_anomaly_response_applied);

        // Topology info
        const harmonyNames = data.harmony_labels || HARMONY_LABELS;
        const domIdx = data.moral_dominant_harmony || 0;
        const domEl = document.getElementById('dominantHarmony');
        if (domEl) {
            domEl.textContent = harmonyNames[domIdx] || '--';
            domEl.style.color = HARMONY_COLORS[domIdx] || HARMONY_COLORS[0];
        }
        const anomEl = document.getElementById('anomalyScore');
        if (anomEl) anomEl.textContent = (data.moral_anomaly_score || 0).toFixed(2);
        const driftEl = document.getElementById('driftValue');
        if (driftEl) driftEl.textContent = (data.moral_drift || 0).toFixed(3);

        // Radar color modulation: cyan → rose when anomaly_score > 0.5
        const anomalyScore = data.moral_anomaly_score || 0;
        if (anomalyScore > 0.5) {
            const t = Math.min((anomalyScore - 0.5) * 2, 1.0); // 0.5→0, 1.0→1
            const borderCol = lerpColor('#06b6d4', '#f43f5e', t);
            chartHarmony.data.datasets[0].borderColor = borderCol;
            chartHarmony.data.datasets[0].pointBackgroundColor = borderCol;
            chartHarmony.data.datasets[0].backgroundColor = borderCol + '26';
        } else {
            chartHarmony.data.datasets[0].borderColor = '#06b6d4';
            chartHarmony.data.datasets[0].pointBackgroundColor = '#06b6d4';
            chartHarmony.data.datasets[0].backgroundColor = 'rgba(6,182,212,0.15)';
        }
        chartHarmony.update('none');

        // Drift sparkline update
        const currentDrift = data.moral_drift || 0;
        driftHistory.push(currentDrift);
        if (driftHistory.length > SPARKLINE_SIZE) driftHistory.shift();
        sparklineDrift.data.labels = Array(driftHistory.length).fill('');
        sparklineDrift.data.datasets[0].data = [...driftHistory];
        sparklineDrift.update('none');

        // Update metrics
        document.getElementById('metricCycle').textContent = data.cycle;
        document.getElementById('metricPsi').textContent = data.consciousness_level.toFixed(3);
        document.getElementById('metricError').textContent = data.prediction_error.toFixed(3);
        document.getElementById('metricMoral').textContent = data.moral_score.toFixed(3);
        document.getElementById('metricTime').textContent = data.cycle_time_us.toLocaleString();
        document.getElementById('metricReasoning').textContent = data.reasoning_confidence.toFixed(2);

        // Consciousness engine telemetry DOM updates
        const convEl = document.getElementById('convergenceState');
        if (convEl) convEl.textContent = data.weight_convergence_state || '--';
        const wvEl = document.getElementById('weightVariance');
        if (wvEl) wvEl.textContent = (data.consciousness_weight_variance || 0).toFixed(3);
        const erEl = document.getElementById('emergenceRatio');
        if (erEl) erEl.textContent = (data.structural_emergence_ratio || 0).toFixed(2);
        const sfEl = document.getElementById('substrateFeasibility');
        if (sfEl) sfEl.textContent = (data.substrate_feasibility || 0).toFixed(2);
        // Color-code substrate badge: green (>0.7), amber (0.3-0.7), rose (<0.3)
        const sbEl = document.getElementById('substrateBadge');
        if (sbEl) {
            const sf = data.substrate_feasibility || 0;
            sbEl.className = 'betti-badge ' + (sf > 0.7 ? 'green' : sf > 0.3 ? 'amber' : 'rose');
        }

        // Mesh network telemetry
        const spEl = document.getElementById('swarmPeers');
        if (spEl) spEl.textContent = data.swarm_peers || 0;
        const mpEl = document.getElementById('networkMeanPhi');
        if (mpEl) mpEl.textContent = (data.network_mean_phi || 0).toFixed(3);
        const mcEl = document.getElementById('meshPeerCount');
        if (mcEl) mcEl.textContent = data.mesh_peer_count || 0;
        const mhEl = document.getElementById('meshHealth');
        if (mhEl) mhEl.textContent = (data.mesh_health_score || 0).toFixed(2);
        const mkEl = document.getElementById('meshPackets');
        if (mkEl) mkEl.textContent = ((data.mesh_bytes_sent || 0) + (data.mesh_bytes_received || 0)).toLocaleString();

        // Update cycle info
        document.getElementById('cycleInfo').textContent =
            `Cycle ${data.cycle} | ${(data.cycle_time_us / 1000).toFixed(1)}ms | Input: "${data.input_text}"`;
    };
}

// --- Controls ---
document.getElementById('btnSend').addEventListener('click', () => {
    const input = document.getElementById('textInput');
    if (ws && ws.readyState === WebSocket.OPEN && input.value.trim()) {
        ws.send(JSON.stringify({ text: input.value.trim() }));
        input.value = '';
    }
});

document.getElementById('textInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('btnSend').click();
});

document.getElementById('btnPause').addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('btnPause');
    btn.textContent = paused ? 'Resume' : 'Pause';
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: paused ? 'pause' : 'resume' }));
    }
});

document.getElementById('btnReset').addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: 'reset' }));
    }
    // Clear history
    Object.values(history).forEach(arr => arr.length = 0);
});

// --- Start ---
connect();
